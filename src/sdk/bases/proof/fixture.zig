//! Test fixtures for the base parity proofs: a tiny random-weight pack written to disk and loaded through the legacy
//! path (`parseConfigFromJson` → `loadModelWeights` → `Transformer.init`), and the helpers that hand the legacy
//! model's loaded arrays to a base's blocks.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const model = @import("../../../engines/mlx/model.zig");
const Transformer = @import("../../../engines/mlx/transformer.zig").Transformer;
const computeQuantParams = @import("../../quant/params.zig").computeQuantParams;
const FusedRows = @import("../../blocks/linear/fused_rows.zig").FusedRows;
const linear = @import("../blocks/linear.zig");
const abi = @import("../../../abi/plugin.zig");
const block = @import("../block.zig");
const cache_ops = @import("../../../engines/mlx/kv/cache_ops.zig");
const KVCache = @import("../../../engines/mlx/kv/cache.zig").KVCache;
const SSMCacheEntry = @import("../../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const ssmFreeSpecCapture = @import("../../../engines/mlx/ssm/entry.zig").ssmFreeSpecCapture;

const testing = std.testing;
const Array = mlx.mlx_array;

pub fn stream() mlx.mlx_stream {
    return if (mlx.noGpuBackend()) mlx.mlx_default_cpu_stream_new() else mlx.gpuStream();
}

/// A pack being written: random tensors into a safetensors map.
pub const Pack = struct {
    map: mlx.mlx_map_string_to_array,
    rnd: std.Random.DefaultPrng,
    s: mlx.mlx_stream,

    pub fn init(seed: u64) Pack {
        return .{ .map = mlx.mlx_map_string_to_array_new(), .rnd = .init(seed), .s = stream() };
    }

    pub fn deinit(p: *Pack) void {
        _ = mlx.mlx_map_string_to_array_free(p.map);
    }

    fn put(p: *Pack, name: []const u8, arr: Array) !void {
        const z = try std.fmt.allocPrintSentinel(testing.allocator, "{s}", .{name}, 0);
        defer testing.allocator.free(z);
        try mlx.check(mlx.mlx_map_string_to_array_insert(p.map, z.ptr, arr));
    }

    fn random(p: *Pack, shape: []const c_int, scale: f32, offset: f32, dt: mlx.mlx_dtype) !Array {
        var n: usize = 1;
        for (shape) |d| n *= @intCast(d);
        const host = try testing.allocator.alloc(f32, n);
        defer testing.allocator.free(host);
        const r = p.rnd.random();
        for (host) |*v| v.* = offset + scale * (r.float(f32) * 2 - 1);
        const f = mlx.mlx_array_new_data(host.ptr, shape.ptr, @intCast(shape.len), .float32);
        defer _ = mlx.mlx_array_free(f);
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&out, f, dt, p.s));
        try mlx.check(mlx.mlx_array_eval(out));
        return out;
    }

    /// A bf16 tensor of uniform values in `offset ± scale`.
    pub fn dense(p: *Pack, name: []const u8, shape: []const c_int, scale: f32, offset: f32) !void {
        return p.denseAs(name, shape, scale, offset, .bfloat16);
    }

    pub fn denseAs(p: *Pack, name: []const u8, shape: []const c_int, scale: f32, offset: f32, dt: mlx.mlx_dtype) !void {
        const a = try p.random(shape, scale, offset, dt);
        defer _ = mlx.mlx_array_free(a);
        try p.put(name, a);
    }

    /// A norm weight near one.
    pub fn norm(p: *Pack, name: []const u8, n: c_int) !void {
        try p.dense(name, &.{n}, 0.2, 1.0);
    }

    /// `<base>.weight` `[rows, cols]`: bf16, or 4-bit affine (group 64) with its scales and biases.
    pub fn weight(p: *Pack, base: []const u8, rows: c_int, cols: c_int, quantized: bool) !void {
        return p.tensor(base, &.{ rows, cols }, quantized);
    }

    /// As `weight`, any rank (an expert bank is `[experts, rows, cols]`); quantized along the last axis.
    pub fn tensor(p: *Pack, base: []const u8, shape: []const c_int, quantized: bool) !void {
        var buf: [192]u8 = undefined;
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(shape[shape.len - 1])));
        if (!quantized) return p.dense(try std.fmt.bufPrint(&buf, "{s}.weight", .{base}), shape, scale, 0);
        const d = try p.random(shape, scale, 0, .bfloat16);
        defer _ = mlx.mlx_array_free(d);
        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(&triple, d, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{}, p.s));
        for ([_][]const u8{ ".weight", ".scales", ".biases" }, 0..) |suffix, i| {
            var a = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(a);
            try mlx.check(mlx.mlx_vector_array_get(&a, triple, i));
            try mlx.check(mlx.mlx_array_eval(a));
            try p.put(try std.fmt.bufPrint(&buf, "{s}{s}", .{ base, suffix }), a);
        }
    }

    /// Writes the pack and loads it through the legacy path.
    pub fn load(p: *Pack, config_json: []const u8) !*Legacy {
        const a = testing.allocator;
        const io = std.Io.Threaded.global_single_threaded.io();
        const lg = try a.create(Legacy);
        errdefer a.destroy(lg);
        lg.tmp = testing.tmpDir(.{});
        errdefer lg.tmp.cleanup();
        var path_buf: [512]u8 = undefined;
        const n = try lg.tmp.dir.realPath(io, &path_buf);
        const dir = path_buf[0..n];
        const file = try std.fmt.allocPrintSentinel(a, "{s}/model.safetensors", .{dir}, 0);
        defer a.free(file);
        const meta = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta);
        try mlx.check(mlx.mlx_save_safetensors(file.ptr, p.map, meta));
        lg.config = try model.parseConfigFromJson(a, config_json);
        errdefer lg.config.deinit(a);
        lg.weights = try model.loadModelWeights(io, a, dir, &lg.config, false);
        errdefer lg.weights.deinit();
        model.resolveWeightPrefix(&lg.config, &lg.weights);
        lg.xfm = try Transformer.init(io, a, lg.config, &lg.weights);
        return lg;
    }
};

/// A pack loaded through the legacy path. Freed with `deinit`.
pub const Legacy = struct {
    tmp: testing.TmpDir,
    config: model.ModelConfig,
    weights: model.Weights,
    xfm: Transformer,

    pub fn deinit(lg: *Legacy) void {
        lg.xfm.deinit();
        lg.weights.deinit();
        lg.config.deinit(testing.allocator);
        lg.tmp.cleanup();
        testing.allocator.destroy(lg);
    }

    /// A projection as the legacy `qmatmul` resolves it (`in_dim` = the activation's width).
    pub fn lin(lg: *const Legacy, w: Array, s: Array, b: Array, bias: Array, in_dim: u32) linear.Linear {
        if (s.ctx == null) return .{ .w = w, .bias = bias };
        const qp = computeQuantParams(&lg.config, w, s, in_dim);
        return .{ .w = w, .s = s, .b = b, .bias = bias, .bits = qp.bits, .group_size = qp.group_size, .mode = qp.mode };
    }

    /// A row-joined group as a base's `Joined` (empty when the loader built none).
    pub fn joined(lg: *const Legacy, r: FusedRows, in_dim: u32) linear.Joined {
        if (r.w.ctx == null) return .{};
        return .{ .lin = lg.lin(r.w, r.s, r.b, .{}, in_dim), .widths = r.widths };
    }
};

/// `ids` `[1, n]` int32.
pub fn tokens(g: *ops.Graph, ids: []const i32) !Array {
    return g.fromSlice(i32, ids, &.{ 1, @intCast(ids.len) });
}

/// A legacy `[1, S, V]` result as `[S, V]`, owned by `g`.
pub fn rows2(g: *ops.Graph, x: Array) !Array {
    const sh = ops.shape(x);
    return g.reshape(x, &.{ sh[0] * sh[1], sh[2] });
}

pub const Step = struct { mode: abi.Mode, ids: []const i32 };

/// Prefill, then decode and verify windows of several widths interleaved.
pub const steps = [_]Step{
    .{ .mode = .prefill, .ids = &.{ 3, 17, 42, 99, 7, 250, 1 } },
    .{ .mode = .decode, .ids = &.{88} },
    .{ .mode = .verify, .ids = &.{ 5, 61, 200 } },
    .{ .mode = .decode, .ids = &.{12} },
    .{ .mode = .verify, .ids = &.{ 9, 33, 77, 140, 2 } },
    .{ .mode = .decode, .ids = &.{64} },
};

/// Fresh recurrent entries, one per layer.
pub fn newEntries(n: usize) ![]SSMCacheEntry {
    const es = try testing.allocator.alloc(SSMCacheEntry, n);
    for (es) |*e| e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
    return es;
}

pub fn freeEntries(es: []SSMCacheEntry) void {
    for (es) |*e| {
        _ = mlx.mlx_array_free(e.conv_state);
        _ = mlx.mlx_array_free(e.ssm_state);
        ssmFreeSpecCapture(e);
    }
    testing.allocator.free(es);
}

/// Runs `steps` through the legacy forward and base `M` on caches of their own, each step's logits bit-equal.
pub fn expectSteps(comptime M: type, lg: *Legacy, w: *const M.Weights) !void {
    return expectStepsOf(M, lg, w, &steps);
}

/// `expectSteps` over the sequence `seq`.
pub fn expectStepsOf(comptime M: type, lg: *Legacy, w: *const M.Weights, seq: []const Step) !void {
    const a = testing.allocator;
    var shared = block.Shared.init(lg.config.rms_norm_eps, lg.config.postNormEps());
    defer shared.deinit();
    const n_layers = lg.config.num_hidden_layers;
    var kvc = try KVCache.init(a, n_layers);
    defer kvc.deinit();
    const entries = try newEntries(n_layers);
    defer freeEntries(entries);
    var slot: cache_ops.Slot = .{ .kv = &kvc, .ssm = entries };
    const handles = [_]abi.Handle{slot.handle()};
    var pos: u32 = 0;
    for (seq) |st| {
        var g = ops.Graph.init(a, lg.xfm.s);
        defer g.deinit();
        const ids = try tokens(&g, st.ids);
        const want = try rows2(&g, try g.adopt(try lg.xfm.forward(ids)));
        const got = try M.forward(w, &shared, &g, ids, .{
            .mode = st.mode,
            .rows = @intCast(st.ids.len),
            .positions = &.{pos},
            .caches = &handles,
            .cache_ops = &cache_ops.ops,
            .logits = .all_rows,
        });
        try ops.expectBitEqual(got.logits, want);
        pos += @intCast(st.ids.len);
    }
}

/// The base's test cache slot behind a block's `Ctx`.
pub fn slotOf(c: *const block.Ctx) *cache_ops.Slot {
    return @ptrCast(@alignCast(c.cache().?));
}
