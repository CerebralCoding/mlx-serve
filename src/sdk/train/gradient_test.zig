//! Native LoRA/autograd probes over the serving decoder's blocks, with synthetic frozen weights.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const Array = ops.Array;
const Graph = ops.Graph;
const testing = std.testing;
const bases = @import("../bases/bases.zig");
const blocks = bases.blocks;
const Linear = blocks.linear.Linear;
const Decoder = bases.Decoder(.{
    .embed = blocks.embed.TableEmbed,
    .norm = blocks.norm.RmsNorm,
    .mixer = blocks.attention.Attention,
    .mlp = blocks.mlp.SwiGlu,
    .seam = blocks.seam.ResidualTail,
    .head = blocks.head.LmHead,
});

fn values(g: *Graph, sh: []const c_int, phase: f32, scale: f32) !Array {
    var n: usize = 1;
    for (sh) |d| n *= @intCast(d);
    const buf = try g.a.alloc(f32, n);
    defer g.a.free(buf);
    for (buf, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.73 + phase) * scale;
    return g.fromSlice(f32, buf, sh);
}

fn item(x: Array) !f32 {
    try ops.eval(x);
    var v: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&v, x));
    return v;
}

fn get(g: *Graph, v: mlx.mlx_vector_array, i: usize) !Array {
    var x = mlx.mlx_array_new();
    mlx.check(mlx.mlx_vector_array_get(&x, v, i)) catch |e| {
        _ = mlx.mlx_array_free(x);
        return e;
    };
    return g.adopt(x);
}

const QuantizedLoss = struct {
    g: *Graph,
    x: Array,
    w: Array,
    s: Array,
    b: Array,
    target: Array,
    failure: ?anyerror = null,

    fn loss(p: *QuantizedLoss, g: *Graph, a: Array, b: Array) !Array {
        const base = try g.qmm(p.x, p.w, p.s, p.b, 32, 4);
        const delta = try g.matmul(try g.matmul(p.x, a), b);
        const err = try g.sub(try g.add(base, delta), p.target);
        return g.call(mlx.mlx_sum, .{ try g.square(err), false });
    }

    fn call(out: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, raw: ?*anyopaque) callconv(.c) c_int {
        const p: *QuantizedLoss = @ptrCast(@alignCast(raw.?));
        p.apply(out, input) catch |e| {
            p.failure = e;
            return 1;
        };
        return 0;
    }

    fn apply(p: *QuantizedLoss, out: *mlx.mlx_vector_array, input: mlx.mlx_vector_array) !void {
        var g = p.g.scope();
        defer g.deinit();
        const y = try p.loss(&g, try get(&g, input, 0), try get(&g, input, 1));
        out.* = mlx.mlx_vector_array_new_value(y);
    }
};

test "train gradient: 4-bit LoRA matches every f32 central finite difference" {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = Graph.init(testing.allocator, s);
    defer g.deinit();
    var qv = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(qv);
    try mlx.check(mlx.mlx_quantize(&qv, try values(&g, &.{ 8, 32 }, 0.3, 0.1), mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(4), "affine", .{}, s));
    var p: QuantizedLoss = .{
        .g = &g,
        .x = try values(&g, &.{ 3, 32 }, 0.8, 0.2),
        .w = try get(&g, qv, 0),
        .s = try get(&g, qv, 1),
        .b = try get(&g, qv, 2),
        .target = try values(&g, &.{ 3, 8 }, 1.7, 0.2),
    };
    const params = [_]Array{ try values(&g, &.{ 32, 2 }, 1.1, 0.1), try values(&g, &.{ 2, 8 }, 2.4, 0.1) };
    const input = mlx.mlx_vector_array_new_data(&params, params.len);
    defer _ = mlx.mlx_vector_array_free(input);
    const fun = mlx.mlx_closure_new_func_payload(QuantizedLoss.call, &p, null);
    defer _ = mlx.mlx_closure_free(fun);
    var vg: mlx.mlx_closure_value_and_grad = .{};
    defer _ = mlx.mlx_closure_value_and_grad_free(vg);
    try mlx.check(mlx.mlx_value_and_grad(&vg, fun, &.{ 0, 1 }, 2));
    var outputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs);
    var grads = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(grads);
    try mlx.check(mlx.mlx_closure_value_and_grad_apply(&outputs, &grads, vg, input));
    try testing.expect(p.failure == null);
    try testing.expectEqual(@as(usize, 2), mlx.mlx_vector_array_size(grads));
    // This loss is quadratic in each parameter; a wide stencil avoids f32 cancellation.
    const epsilon: f32 = 0.05;
    for (params, 0..) |param, k| {
        const grad = try get(&g, grads, k);
        try ops.eval(grad);
        const actual = mlx.mlx_array_data_float32(grad).?;
        const n = mlx.mlx_array_size(param);
        const perturb = try testing.allocator.alloc(f32, n);
        defer testing.allocator.free(perturb);
        @memset(perturb, 0);
        for (0..n) |i| {
            var tmp = g.scope();
            defer tmp.deinit();
            perturb[i] = epsilon;
            const d = try tmp.fromSlice(f32, perturb, ops.shape(param));
            perturb[i] = 0;
            var plus = params;
            var minus = params;
            plus[k] = try tmp.add(param, d);
            minus[k] = try tmp.sub(param, d);
            const numerical = ((try item(try p.loss(&tmp, plus[0], plus[1]))) - (try item(try p.loss(&tmp, minus[0], minus[1])))) / (2 * epsilon);
            try testing.expectApproxEqAbs(numerical, actual[i], 0.0002);
        }
    }
}

fn quantized(g: *Graph, rows: c_int, cols: c_int, seed: f32, dt: ops.Dtype) !Linear {
    var tmp = g.scope();
    defer tmp.deinit();
    const w = try tmp.astype(try values(&tmp, &.{ rows, cols }, seed, 0.1), dt);
    var v = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(v);
    try mlx.check(mlx.mlx_quantize(&v, w, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(4), "affine", .{}, g.s));
    const l: Linear = .{ .w = try get(g, v, 0), .s = try get(g, v, 1), .b = try get(g, v, 2), .bits = 4, .group_size = 32 };
    try ops.evalMany(&.{ l.w, l.s, l.b });
    return l;
}

const LayerLoss = struct {
    g: *Graph,
    linears: [7]Linear,
    embed: Linear,
    norm: Array,
    qk_norm: Array,
    ids: Array,
    params: [14]Array,
    hidden: c_int,
    heads: c_int,
    kv_heads: c_int,
    head_dim: c_int,
    rows: c_int,
    dtype: ops.Dtype,
    failure: ?anyerror = null,
    checkpointed_forward: mlx.mlx_closure = .{},

    fn init(g: *Graph, hidden: c_int, heads: c_int, kv_heads: c_int, hd: c_int, inter: c_int, rows: c_int, rank: c_int, dt: ops.Dtype) !LayerLoss {
        const q = heads * hd;
        const kv = kv_heads * hd;
        const shapes = [_][2]c_int{ .{ q, hidden }, .{ kv, hidden }, .{ kv, hidden }, .{ hidden, q }, .{ inter, hidden }, .{ inter, hidden }, .{ hidden, inter } };
        var p: LayerLoss = undefined;
        p.g = g;
        p.hidden = hidden;
        p.heads = heads;
        p.kv_heads = kv_heads;
        p.head_dim = hd;
        p.rows = rows;
        p.dtype = dt;
        p.failure = null;
        p.checkpointed_forward = .{};
        for (shapes, 0..) |sh, k| {
            const seed: f32 = @floatFromInt(k + 1);
            p.linears[k] = try quantized(g, sh[0], sh[1], seed, dt);
            p.params[2 * k] = try values(g, &.{ sh[1], rank }, seed + 0.3, 0.05);
            p.params[2 * k + 1] = try values(g, &.{ rank, sh[0] }, seed + 0.8, 0.05);
        }
        p.embed = try quantized(g, 32, hidden, 8.4, dt);
        p.norm = try g.ones(&.{hidden}, dt);
        p.qk_norm = try g.ones(&.{hd}, dt);
        p.ids = try g.reshape(try g.call(mlx.mlx_remainder, .{ try g.arange(0, @floatFromInt(rows), 1, .int32), try g.scalar(32, .int32) }), &.{ 1, rows });
        try ops.evalMany(&p.params);
        try ops.evalMany(&.{ p.norm, p.qk_norm, p.ids });
        return p;
    }

    fn logits(p: *LayerLoss, g: *Graph, params: []const Array) !Array {
        var l = p.linears;
        if (params.len != 0) for (&l, 0..) |*w, k| {
            w.lora = .{ .a = params[2 * k], .b = params[2 * k + 1], .scale = 2 };
        };
        const layers = [_]Decoder.Layer{.{
            .input_norm = p.norm,
            .post_norm = p.norm,
            .mixer = .{
                .q = l[0],
                .k = l[1],
                .v = l[2],
                .o = l[3],
                .q_norm = p.qk_norm,
                .k_norm = p.qk_norm,
                .geo = .{ .heads = p.heads, .kv_heads = p.kv_heads, .head_dim = p.head_dim, .scale = 1 / @sqrt(@as(f32, @floatFromInt(p.head_dim))), .rope_dims = p.head_dim, .rope_base = 1000000, .rope_scale = 1 },
            },
            .mlp = .{ .gate = l[4], .up = l[5], .down = l[6] },
        }};
        const w: Decoder.Weights = .{
            .embed = .{ .table = p.embed, .hidden = p.hidden, .act = p.dtype },
            .layers = &layers,
            .final_norm = p.norm,
            .head = p.embed,
        };
        var shared = bases.Shared.init(1e-6, 1e-6);
        defer shared.deinit();
        const out = try Decoder.forward(&w, &shared, g, p.ids, .{ .mode = .prefill, .rows = p.rows, .positions = &.{0}, .logits = .all_rows });
        return out.logits;
    }

    fn loss(p: *LayerLoss, g: *Graph, params: []const Array) !Array {
        const scores = try g.astype(if (p.checkpointed_forward.ctx == null) try p.logits(g, params) else blk: {
            const input = mlx.mlx_vector_array_new_data(params.ptr, params.len);
            defer _ = mlx.mlx_vector_array_free(input);
            var output = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(output);
            try mlx.check(mlx.mlx_closure_apply(&output, p.checkpointed_forward, input));
            break :blk try get(g, output, 0);
        }, .float32);
        return g.call(mlx.mlx_mean, .{ try g.square(try g.addScalar(scores, -0.2)), false });
    }

    fn call(out: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, raw: ?*anyopaque) callconv(.c) c_int {
        const p: *LayerLoss = @ptrCast(@alignCast(raw.?));
        p.apply(out, input, false) catch |e| {
            p.failure = e;
            return 1;
        };
        return 0;
    }

    fn forwardCall(out: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, raw: ?*anyopaque) callconv(.c) c_int {
        const p: *LayerLoss = @ptrCast(@alignCast(raw.?));
        p.apply(out, input, true) catch |e| {
            p.failure = e;
            return 1;
        };
        return 0;
    }

    fn apply(p: *LayerLoss, out: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, forward_only: bool) !void {
        var g = p.g.scope();
        defer g.deinit();
        var params: [14]Array = undefined;
        for (&params, 0..) |*x, i| x.* = try get(&g, input, i);
        out.* = mlx.mlx_vector_array_new_value(if (forward_only) try p.logits(&g, &params) else try p.loss(&g, &params));
    }

    fn gradient(p: *LayerLoss, g: *Graph, checkpoint: bool) ![15]Array {
        const fun = mlx.mlx_closure_new_func_payload(call, p, null);
        defer _ = mlx.mlx_closure_free(fun);
        const forward = mlx.mlx_closure_new_func_payload(forwardCall, p, null);
        defer _ = mlx.mlx_closure_free(forward);
        var remat: mlx.mlx_closure = .{};
        defer _ = mlx.mlx_closure_free(remat);
        if (checkpoint) try mlx.check(mlx.mlx_checkpoint(&remat, forward));
        p.checkpointed_forward = remat;
        defer p.checkpointed_forward = .{};
        var vg: mlx.mlx_closure_value_and_grad = .{};
        defer _ = mlx.mlx_closure_value_and_grad_free(vg);
        var argnums: [14]c_int = undefined;
        for (&argnums, 0..) |*v, i| v.* = @intCast(i);
        try mlx.check(mlx.mlx_value_and_grad(&vg, fun, &argnums, argnums.len));
        const in = mlx.mlx_vector_array_new_data(&p.params, p.params.len);
        defer _ = mlx.mlx_vector_array_free(in);
        var out = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(out);
        var grads = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(grads);
        try mlx.check(mlx.mlx_closure_value_and_grad_apply(&out, &grads, vg, in));
        if (p.failure) |e| return e;
        try testing.expectEqual(@as(usize, 14), mlx.mlx_vector_array_size(grads));
        var result: [15]Array = undefined;
        result[0] = try get(g, out, 0);
        for (result[1..], 0..) |*v, i| v.* = try get(g, grads, i);
        try ops.evalMany(&result);
        return result;
    }
};

test "train gradient: Qwen3 CPU finite differences and Metal checkpoint gradients agree" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = Graph.init(testing.allocator, s);
    defer g.deinit();
    g.differentiable = true;
    var p = try LayerLoss.init(&g, 64, 2, 1, 32, 128, 3, 2, .float32);
    const direct = try p.gradient(&g, false);
    const metal = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(metal);
    var mg = Graph.init(testing.allocator, metal);
    defer mg.deinit();
    mg.differentiable = true;
    p.g = &mg;
    const gpu = try p.gradient(&mg, false);
    const remat = try p.gradient(&mg, true);
    p.g = &g;
    for (gpu, remat) |a, b| try testing.expect((try item(try mg.max(try mg.abs(try mg.reshape(try mg.sub(a, b), &.{-1})), 0, false))) < 1e-5);
    for (direct, gpu) |a, b| {
        const peak = try item(try g.max(try g.abs(try g.reshape(a, &.{-1})), 0, false));
        const err = try item(try g.max(try g.abs(try g.reshape(try g.sub(a, b), &.{-1})), 0, false));
        try testing.expect(err < 0.0001 + peak * 0.01);
    }
    var nonzero: usize = 0;
    for (p.params, direct[1..], 0..) |param, grad, k| {
        try testing.expectEqualSlices(c_int, ops.shape(param), ops.shape(grad));
        try testing.expectEqual(ops.dtype(param), ops.dtype(grad));
        for (0..3) |probe| {
            var tmp = g.scope();
            defer tmp.deinit();
            const direction = try values(&tmp, ops.shape(param), @as(f32, @floatFromInt(k + probe)) + 0.2, 1);
            const delta = try tmp.mulScalar(direction, 0.002);
            var plus = p.params;
            var minus = p.params;
            plus[k] = try tmp.add(param, delta);
            minus[k] = try tmp.sub(param, delta);
            const numerical = ((try item(try p.loss(&tmp, &plus))) - (try item(try p.loss(&tmp, &minus)))) / 0.004;
            const actual = try item(try tmp.call(mlx.mlx_sum, .{ try tmp.mul(direction, grad), false }));
            try testing.expectApproxEqAbs(numerical, actual, 0.0005 + @abs(actual) * 0.01);
            if (@abs(actual) > 0.0001) nonzero += 1;
        }
    }
    try testing.expect(nonzero >= 14);
}

test "train gradient: zero-B starts at the base and learns on the first step" {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = Graph.init(testing.allocator, s);
    defer g.deinit();
    g.differentiable = true;
    var p = try LayerLoss.init(&g, 64, 2, 1, 32, 128, 3, 2, .float32);
    for (0..7) |k| p.params[2 * k + 1] = try g.zeros(ops.shape(p.params[2 * k + 1]), .float32);
    try ops.expectBitEqual(try p.logits(&g, &.{}), try p.logits(&g, &p.params));
    const first = try p.gradient(&g, false);
    var b_norm: f32 = 0;
    for (0..7) |k| {
        try testing.expectEqual(@as(f32, 0), try item(try g.call(mlx.mlx_sum, .{ try g.square(first[1 + 2 * k]), false })));
        b_norm += try item(try g.call(mlx.mlx_sum, .{ try g.square(first[2 + 2 * k]), false }));
    }
    try testing.expect(b_norm > 0);
    var next = p.params;
    for (&next, p.params, first[1..]) |*v, param, grad| v.* = try g.sub(param, try g.mulScalar(grad, 0.001));
    try testing.expect((try item(try p.loss(&g, &next))) < try item(first[0]));
}

test "train gradient: seq512 Qwen3-4B layer checkpoint memory" {
    if (!@import("build_options").slow_tests) return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = Graph.init(testing.allocator, s);
    defer g.deinit();
    g.differentiable = true;
    var p = try LayerLoss.init(&g, 2560, 32, 8, 128, 9728, 512, 8, .bfloat16);
    for ([_]bool{ false, true, false, true }) |checkpoint| {
        var tmp = g.scope();
        defer tmp.deinit();
        try mlx.check(mlx.mlx_synchronize(s));
        try mlx.check(mlx.mlx_clear_cache());
        var resident: usize = 0;
        try mlx.check(mlx.mlx_get_active_memory(&resident));
        try mlx.check(mlx.mlx_reset_peak_memory());
        const result = try p.gradient(&tmp, checkpoint);
        var peak: usize = 0;
        try mlx.check(mlx.mlx_get_peak_memory(&peak));
        std.debug.print("\ntrain layer: checkpoint={} resident={d} peak={d} transient={d} loss={d:.6}\n", .{ checkpoint, resident, peak, peak -| resident, try item(result[0]) });
        try testing.expect(peak >= resident);
    }
}

test "train gradient: bf16 custom-kernel control fails while plain MLX differentiates" {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = Graph.init(testing.allocator, s);
    defer g.deinit();
    var p = try LayerLoss.init(&g, 64, 2, 1, 32, 128, 3, 2, .bfloat16);
    mlx.installErrorHandler();
    var message: [2048]u8 = undefined;
    defer _ = mlx.takeError(&message);
    try testing.expectError(error.MlxError, p.gradient(&g, false));
    const why = mlx.takeError(&message) orelse return error.MissingGradientError;
    try testing.expect(std.mem.indexOf(u8, why, "vjp") != null or std.mem.indexOf(u8, why, "gradient") != null);
    p.failure = null;
    g.differentiable = true;
    const result = try p.gradient(&g, false);
    try testing.expect(std.math.isFinite(try item(result[0])));
    for (result[1..]) |grad| {
        try testing.expectEqual(ops.Dtype.float32, ops.dtype(grad));
        try ops.eval(grad);
        for (mlx.mlx_array_data_float32(grad).?[0..mlx.mlx_array_size(grad)]) |v| try testing.expect(std.math.isFinite(v));
    }
}
