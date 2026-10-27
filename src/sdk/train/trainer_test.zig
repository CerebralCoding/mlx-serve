//! Gradient, learning, memory and failure contracts over composed decoders.
const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const model = @import("model.zig");
const adapter_mod = @import("adapter.zig");
const step = @import("step.zig");
const packing = @import("packing.zig");
const grad = @import("grad.zig");
const Linear = @import("../bases/blocks/linear.zig").Linear;

const Fixture = struct {
    layers: [1]model.Decoder.Layer,
    base: model.Base,

    fn init(g: *ops.Graph, dt: ops.Dtype, qk_norm: bool) !Fixture {
        const norm = try g.ones(&.{32}, dt);
        const qnorm = if (qk_norm) norm else ops.Array{};
        const table = try weight(g, 16, 32, dt, 20);
        return .{
            .layers = .{.{ .input_norm = norm, .post_norm = norm, .mixer = .{ .q = try weight(g, 32, 32, dt, 1), .k = try weight(g, 32, 32, dt, 2), .v = try weight(g, 32, 32, dt, 3), .o = try weight(g, 32, 32, dt, 4), .q_norm = qnorm, .k_norm = qnorm, .geo = .{ .heads = 1, .kv_heads = 1, .head_dim = 32, .scale = 1 / @sqrt(@as(f32, 32)), .rope_dims = 32, .rope_base = 10000, .rope_scale = 1 } }, .mlp = .{ .gate = try weight(g, 64, 32, dt, 5), .up = try weight(g, 64, 32, dt, 6), .down = try weight(g, 32, 64, dt, 7) } }},
            .base = .{ .weights = .{ .layers = &.{}, .embed = .{ .table = table, .hidden = 32, .act = dt }, .head = table, .final_norm = norm }, .shared = @import("../bases/block.zig").Shared.init(1e-6, 1e-6), .a = g.a, .s = g.s },
        };
    }
    fn frozen(self: *Fixture) !model.Frozen {
        self.base.weights.layers = &self.layers;
        return model.Frozen.init(&self.base);
    }
    fn deinit(self: *Fixture) void {
        self.base.shared.deinit();
    }
};

fn weight(g: *ops.Graph, rows: c_int, cols: c_int, dt: ops.Dtype, seed: u64) !Linear {
    const shape = [_]c_int{ rows, cols };
    const key = try g.call(mlx.mlx_random_key, .{seed});
    const x = try g.call(mlx.mlx_random_normal, .{ &shape, shape.len, ops.Dtype.float32, @as(f32, 0), @as(f32, 0.12), key });
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    try mlx.check(mlx.mlx_quantize(&triple, try g.astype(x, dt), mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(4), "affine", .{}, g.s));
    const result: Linear = .{ .w = try grad.get(g, triple, 0), .s = try grad.get(g, triple, 1), .b = try grad.get(g, triple, 2), .bits = 4, .group_size = 32 };
    try ops.evalMany(&.{ result.w, result.s, result.b });
    return result;
}

test "train core: real decoder overfits a completion on CPU and Metal and restores its best weights" {
    for ([_]bool{ false, true }) |gpu| {
        const s = if (gpu) mlx.mlx_default_gpu_stream_new() else mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(s);
        var g = ops.Graph.init(std.testing.allocator, s);
        defer g.deinit();
        var fixture = try Fixture.init(&g, .bfloat16, gpu);
        defer fixture.deinit();
        const frozen = try fixture.frozen();
        const targets = try frozen.targets(g.a);
        defer g.a.free(targets);
        var adapter = try adapter_mod.Adapter.init(g.a, s, targets, .{});
        defer adapter.deinit();
        var trainer = try step.Trainer.initForTest(g.a, s, frozen, &adapter, .{ .learning_rate = 0.02, .warmup_steps = 2, .total_steps = 60, .accumulation = 2, .vocab_chunk = 5 });
        defer trainer.deinit();
        const e = try packing.pack(g.a, &.{ 1, 2, 3, 7, 8 }, 3, 512);
        defer e.deinit(g.a);
        const before = try trainer.evaluate(&.{e});
        const ids = try g.fromSlice(u32, e.inputs, &.{ 1, @intCast(e.inputs.len) });
        const base = try frozen.hidden(&g, ids, &.{});
        const identity = try adapter.bindings(&g, adapter.params);
        defer g.a.free(identity);
        const initial = try frozen.hidden(&g, ids, identity);
        try ops.expectBitEqual(base, initial);
        try mlx.check(mlx.mlx_synchronize(s));
        var resident: usize = 0;
        try mlx.check(mlx.mlx_get_active_memory(&resident));
        try mlx.check(mlx.mlx_reset_peak_memory());
        for (0..120) |_| _ = try trainer.microbatch(e);
        var peak: usize = 0;
        try mlx.check(mlx.mlx_get_peak_memory(&peak));
        const billed = try @import("memory.zig").reserve(frozen, targets, 8, @intCast(e.inputs.len));
        try std.testing.expect(peak -| resident <= billed);
        const after = try trainer.evaluate(&.{e});
        try std.testing.expect(after < before * 0.2);
        std.debug.print("\ntrain tiny: gpu={} loss={d:.5}->{d:.5}\n", .{ gpu, before, after });
        try std.testing.expect(after < 0.6);
        try std.testing.expect(!try trainer.observeValidation(after));
        try std.testing.expect(!try trainer.observeValidation(after + 1));
        try std.testing.expect(!try trainer.observeValidation(after + 1));
        try std.testing.expect(try trainer.observeValidation(after + 1));
        const changed = try g.addScalar(adapter.params[1], 1);
        _ = mlx.mlx_array_free(adapter.params[1]);
        adapter.params[1] = g.escape(changed);
        try trainer.restoreBest();
        try std.testing.expectApproxEqAbs(after, try trainer.evaluate(&.{e}), 1e-6);
        try std.testing.expectError(error.TrainingComplete, trainer.microbatch(e));
    }
}

test "train core: completion loss gradients match finite differences for all fourteen adapter tensors" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = ops.Graph.init(std.testing.allocator, s);
    defer g.deinit();
    g.differentiable = true;
    var fixture = try Fixture.init(&g, .float32, true);
    defer fixture.deinit();
    const frozen = try fixture.frozen();
    const targets = try frozen.targets(g.a);
    defer g.a.free(targets);
    var adapter = try adapter_mod.Adapter.init(g.a, s, targets, .{ .rank = 2, .alpha = 4, .compute = .float32 });
    defer adapter.deinit();
    for (adapter.params, 0..) |*p, i| if (i % 2 == 1) {
        const key = try g.call(mlx.mlx_random_key, .{@as(u64, i + 50)});
        const shape = ops.shape(p.*);
        const x = try g.call(mlx.mlx_random_normal, .{ shape.ptr, shape.len, ops.Dtype.float32, @as(f32, 0), @as(f32, 0.05), key });
        _ = mlx.mlx_array_free(p.*);
        p.* = g.escape(x);
    };
    const Objective = struct {
        frozen: model.Frozen,
        adapter: *adapter_mod.Adapter,
        ids: ops.Array,
        labels: ops.Array,
        mask: ops.Array,
        fn call(self: @This(), graph: *ops.Graph, params: []const ops.Array) !ops.Array {
            const bindings = try self.adapter.bindings(graph, params);
            defer graph.a.free(bindings);
            return @import("loss.zig").projectedHead(graph, try self.frozen.hidden(graph, self.ids, bindings), self.frozen.head, self.labels, self.mask, 5);
        }
    };
    const objective: Objective = .{ .frozen = frozen, .adapter = &adapter, .ids = try g.fromSlice(u32, &.{ 1, 2, 3, 4 }, &.{ 1, 4 }), .labels = try g.fromSlice(u32, &.{ 2, 3, 7, 8 }, &.{4}), .mask = try g.fromSlice(f32, &.{ 0, 0, 1, 1 }, &.{4}) };
    const result = try grad.valueAndGrad(&g, objective, Objective.call, adapter.params);
    defer result.deinit(g.a);
    try ops.evalMany(result.gradients);
    for (adapter.params, result.gradients, 0..) |p, derivative, i| {
        var t = g.scope();
        defer t.deinit();
        const shape = ops.shape(p);
        const key = try t.call(mlx.mlx_random_key, .{@as(u64, i + 100)});
        const direction = try t.call(mlx.mlx_random_normal, .{ shape.ptr, shape.len, ops.Dtype.float32, @as(f32, 0), @as(f32, 0.1), key });
        const analytic = try grad.item(try t.call(mlx.mlx_sum, .{ try t.mul(direction, derivative), false }));
        const varied = try t.a.dupe(ops.Array, adapter.params);
        defer t.a.free(varied);
        const delta = try t.mulScalar(direction, 0.01);
        varied[i] = try t.add(p, delta);
        const plus = try grad.item(try objective.call(&t, varied));
        varied[i] = try t.sub(p, delta);
        const minus = try grad.item(try objective.call(&t, varied));
        try std.testing.expectApproxEqAbs((plus - minus) / 0.02, analytic, 0.001);
    }
}

test "train core: allocation failures leave a microbatch and optimizer unchanged and release all arrays" {
    mlx.installErrorHandler();
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = ops.Graph.init(std.testing.allocator, s);
    defer g.deinit();
    var fixture = try Fixture.init(&g, .float32, false);
    defer fixture.deinit();
    const frozen = try fixture.frozen();
    const targets = try frozen.targets(g.a);
    defer g.a.free(targets);
    const e = try packing.pack(g.a, &.{ 1, 2, 3 }, 2, 512);
    defer e.deinit(g.a);
    var failures: usize = 0;
    for (0..2048) |index| {
        var adapter = try adapter_mod.Adapter.init(g.a, s, targets, .{ .rank = 1, .compute = .float32 });
        defer adapter.deinit();
        var trainer = try step.Trainer.initForTest(g.a, s, frozen, &adapter, .{ .accumulation = 2 });
        defer trainer.deinit();
        _ = try trainer.microbatch(e);
        var failing = std.testing.FailingAllocator.init(g.a, .{ .fail_index = index });
        trainer.a = failing.allocator();
        trainer.optimizer.a = failing.allocator();
        defer {
            trainer.a = g.a;
            trainer.optimizer.a = g.a;
        }
        const before = ops.liveForTest();
        const original = adapter.params[1].ctx;
        if (trainer.microbatch(e)) |_| {
            try std.testing.expect(!failing.has_induced_failure);
            break;
        } else |err| {
            try std.testing.expectEqual(error.OutOfMemory, err);
            try std.testing.expectEqual(original, adapter.params[1].ctx);
            try std.testing.expectEqual(@as(u32, 0), trainer.optimizer.updates);
            try std.testing.expectEqual(@as(u32, 1), trainer.pending);
            failures += 1;
        }
        try std.testing.expectEqual(before, ops.liveForTest());
    } else return error.AllocationSweepDidNotFinish;
    try std.testing.expect(failures > 0);
}

test "train core: unequal completion lengths accumulate by token count and partial batches flush once" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = ops.Graph.init(std.testing.allocator, s);
    defer g.deinit();
    var fixture = try Fixture.init(&g, .float32, false);
    defer fixture.deinit();
    const frozen = try fixture.frozen();
    const targets = try frozen.targets(g.a);
    defer g.a.free(targets);
    var accumulated = try adapter_mod.Adapter.init(g.a, s, targets, .{ .compute = .float32 });
    defer accumulated.deinit();
    var reference = try adapter_mod.Adapter.init(g.a, s, targets, .{ .compute = .float32 });
    defer reference.deinit();
    var ta = try step.Trainer.initForTest(g.a, s, frozen, &accumulated, .{ .accumulation = 3 });
    defer ta.deinit();
    var tr = try step.Trainer.initForTest(g.a, s, frozen, &reference, .{ .accumulation = 1 });
    defer tr.deinit();
    try std.testing.expectError(error.TrainingRequiresMetal, step.Trainer.init(g.a, s, frozen, &reference, .{}));
    var one = try packing.pack(g.a, &.{ 1, 2, 3, 7, 8 }, 2, 512);
    defer one.deinit(g.a);
    one.mask[2] = 0;
    one.mask[3] = 0;
    one.completion_tokens = 1;
    const two = try packing.pack(g.a, &.{ 1, 2, 3, 7, 8 }, 3, 512);
    defer two.deinit(g.a);
    const combined = try packing.pack(g.a, &.{ 1, 2, 3, 7, 8 }, 2, 512);
    defer combined.deinit(g.a);
    try std.testing.expect(!(try ta.microbatch(one)).updated);
    try std.testing.expect(!(try ta.microbatch(two)).updated);
    try std.testing.expectEqual(@as(u32, 0), ta.optimizer.updates);
    try ta.flush();
    try ta.flush();
    try std.testing.expectEqual(@as(u32, 1), ta.optimizer.updates);
    _ = try tr.microbatch(combined);
    for (accumulated.params, reference.params) |actual, expected| {
        const delta = try grad.item(try g.call(mlx.mlx_max, .{ try g.abs(try g.sub(actual, expected)), false }));
        try std.testing.expect(delta < 1e-7);
    }
}
