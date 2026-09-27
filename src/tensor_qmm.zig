//! Row-invariant affine Q4 matmul on M5 tensor units (TensorFold; see NOTICE).
const std = @import("std");
const mlx = @import("mlx.zig");

const HEADER =
    \\#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    \\using namespace mpp::tensor_ops;
;
const SUM =
    \\const uint m = thread_position_in_grid.y;
    \\const uint g = thread_position_in_grid.x;
    \\if (g >= K / 64 || int(m) >= MP) return;
    \\float acc = 0.0f;
    \\if (int(m) < M) for (int i = 0; i < 64; i++) acc += float(X[m * K + g * 64 + i]);
    \\XS[g * MP + m] = acc;
;
const Key = struct { m: c_int, n: c_int, k: c_int };
const Plan = struct {
    sum: mlx.mlx_fast_metal_kernel,
    sum_config: mlx.mlx_fast_metal_kernel_config,
    main: mlx.mlx_fast_metal_kernel,
    main_config: mlx.mlx_fast_metal_kernel_config,
};
var plans: std.AutoHashMapUnmanaged(Key, Plan) = .{};
var enabled_cache: ?bool = null;

pub fn enabled() bool {
    if (enabled_cache) |v| return v;
    // Prepared weights increase residency; enable explicitly on qualified M5s.
    const on = if (std.c.getenv("MLX_SERVE_ROW_TENSOR_QMM")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else false;
    enabled_cache = on;
    return on;
}

fn kernel(name: [*:0]const u8, inputs: []const [*:0]const u8, output: [*:0]const u8, source: [*:0]const u8, header: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    const iv = mlx.mlx_vector_string_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&[_][*:0]const u8{output}, 1);
    defer _ = mlx.mlx_vector_string_free(ov);
    const result = mlx.mlx_fast_metal_kernel_new(name, iv, ov, source, header, true, false);
    if (result.ctx == null) return error.MetalKernelCompileFailed;
    return result;
}

fn planFor(key: Key) !Plan {
    if (plans.get(key)) |p| return p;
    const a = std.heap.c_allocator;
    const tiles = @divTrunc(key.n + 31, 32);
    var split: c_int = 1;
    while (split < 8 and tiles * split < 1024 and @divTrunc(@divExact(key.k, 64), split * 2) >= 8) split *= 2;
    const constants = try std.fmt.allocPrint(a, "constexpr int M={d}, MP=16, N={d}, K={d}, TMR=1, NT=32, SK={d}; constexpr bool TILED={s};\n", .{ key.m, key.n, key.k, split, if (@rem(key.n, 32) == 0) "true" else "false" });
    defer a.free(constants);
    const sum_source = try std.mem.concatWithSentinel(a, u8, &.{ constants, SUM }, 0);
    defer a.free(sum_source);
    const main_source = try std.mem.concatWithSentinel(a, u8, &.{ constants, @embedFile("metal/tensor_qmm.metal") }, 0);
    defer a.free(main_source);
    const sum_name = try std.fmt.allocPrintSentinel(a, "msv_tensor_sum_m{d}_k{d}", .{ key.m, key.k }, 0);
    defer a.free(sum_name);
    const main_name = try std.fmt.allocPrintSentinel(a, "msv_tensor_qmm_m{d}_n{d}_k{d}_s{d}", .{ key.m, key.n, key.k, split }, 0);
    defer a.free(main_name);
    const sum = try kernel(sum_name.ptr, &.{"X"}, "XS", sum_source.ptr, "");
    errdefer _ = mlx.mlx_fast_metal_kernel_free(sum);
    const main = try kernel(main_name.ptr, &.{ "X", "XS", "Wq", "SBt" }, "Y", main_source.ptr, HEADER);
    errdefer _ = mlx.mlx_fast_metal_kernel_free(main);
    const sum_config = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(sum_config);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(sum_config, &[_]c_int{ @divExact(key.k, 64), 16 }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(sum_config, @divExact(key.k, 64), 16, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(sum_config, @min(@divExact(key.k, 64), 256), 1, 1));
    const main_config = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(main_config);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(main_config, &[_]c_int{ key.m, key.n }, 2, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(main_config, tiles * 32 * split, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(main_config, 32 * split, 1, 1));
    const p = Plan{ .sum = sum, .sum_config = sum_config, .main = main, .main_config = main_config };
    try plans.put(a, key, p);
    return p;
}

fn apply(k: mlx.mlx_fast_metal_kernel, config: mlx.mlx_fast_metal_kernel_config, inputs: []const mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const iv = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, k, iv, config, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    return out;
}

/// Caller gates M5 support; unsupported geometry declines to the existing path.
pub fn qmm(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32, s: mlx.mlx_stream) !?mlx.mlx_array {
    return qmmCached(null, x, w, sc, bi, bits, group_size, s);
}

pub fn qmmCached(cache: ?*Cache, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return null;
    const key = geometry(x, w, sc, bi, bits, group_size) orelse return null;
    const xs = mlx.getShape(x);
    const m = key.m;
    const n = key.n;
    const k = key.k;
    const p = try planFor(key);
    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ m, k }, 2, s));
    const sums = try apply(p.sum, p.sum_config, &.{x2}, s);
    defer _ = mlx.mlx_array_free(sums);
    var local: ?Packed = null;
    defer if (local) |*v| v.deinit();
    const prepared = if (cache) |c| try c.get(w, sc, bi, s) else blk: {
        local = try Packed.init(w, sc, bi, s);
        break :blk local.?;
    };
    const y = try apply(p.main, p.main_config, &.{ x2, sums, prepared.w, prepared.sb }, s);
    defer _ = mlx.mlx_array_free(y);
    var shape: [8]c_int = undefined;
    @memcpy(shape[0..xs.len], xs);
    shape[xs.len - 1] = n;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, y, &shape, xs.len, s));
    return out;
}

pub fn supports(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32, s: mlx.mlx_stream) bool {
    return mlx.streamIsGpu(s) and geometry(x, w, sc, bi, bits, group_size) != null;
}

fn geometry(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32) ?Key {
    if (bits != 4 or group_size != 64 or x.ctx == null or w.ctx == null or sc.ctx == null or bi.ctx == null) return null;
    if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(w) != .uint32 or mlx.mlx_array_dtype(sc) != .bfloat16 or mlx.mlx_array_dtype(bi) != .bfloat16) return null;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    const ss = mlx.getShape(sc);
    const bs = mlx.getShape(bi);
    if (xs.len == 0 or xs.len > 8 or ws.len != 2 or ss.len != 2 or bs.len != 2) return null;
    const n = ws[0];
    const k = xs[xs.len - 1];
    if (n <= 0 or k <= 0 or @rem(n, 8) != 0 or @rem(k, 64) != 0 or ws[1] != @divExact(k, 8)) return null;
    if (ss[0] != n or bs[0] != n or ss[1] != @divExact(k, 64) or bs[1] != ss[1]) return null;
    var m: c_int = 1;
    for (xs[0 .. xs.len - 1]) |d| {
        if (d <= 0 or d > 16 or m > @divTrunc(16, d)) return null;
        m *= d;
    }
    return .{ .m = m, .n = n, .k = k };
}

const Packed = struct {
    w: mlx.mlx_array,
    sb: mlx.mlx_array,
    inputs: [3]mlx.mlx_array,

    fn init(w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, s: mlx.mlx_stream) !Packed {
        var out = Packed{ .w = mlx.mlx_array_new(), .sb = mlx.mlx_array_new(), .inputs = .{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() } };
        errdefer out.deinit();
        for (&out.inputs, [_]mlx.mlx_array{ w, sc, bi }) |*dst, src| try mlx.check(mlx.mlx_array_set(dst, src));
        const n = mlx.getShape(w)[0];
        const kw = mlx.getShape(w)[1];
        var st = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(st);
        var bt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bt);
        try mlx.check(mlx.mlx_transpose(&st, sc, s));
        try mlx.check(mlx.mlx_transpose(&bt, bi, s));
        const pair = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{ st, bt }, 2);
        defer _ = mlx.mlx_vector_array_free(pair);
        var stacked = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(stacked);
        try mlx.check(mlx.mlx_stack_axis(&stacked, pair, -1, s));
        try mlx.check(mlx.mlx_contiguous(&out.sb, stacked, false, s));
        if (@rem(n, 32) == 0) {
            var shaped = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(shaped);
            var transposed = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(transposed);
            var flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat);
            try mlx.check(mlx.mlx_reshape(&shaped, w, &[_]c_int{ @divExact(n, 32), 32, @divExact(kw, 8), 8 }, 4, s));
            try mlx.check(mlx.mlx_transpose_axes(&transposed, shaped, &[_]c_int{ 0, 2, 1, 3 }, 4, s));
            try mlx.check(mlx.mlx_reshape(&flat, transposed, &[_]c_int{ n, kw }, 2, s));
            try mlx.check(mlx.mlx_contiguous(&out.w, flat, false, s));
        } else try mlx.check(mlx.mlx_array_set(&out.w, w));
        // Materialize once: no weight-layout work in subsequent decode graphs.
        const arrays = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{ out.w, out.sb }, 2);
        defer _ = mlx.mlx_vector_array_free(arrays);
        try mlx.check(mlx.mlx_eval(arrays));
        return out;
    }

    fn deinit(self: *Packed) void {
        _ = mlx.mlx_array_free(self.w);
        _ = mlx.mlx_array_free(self.sb);
        for (self.inputs) |v| _ = mlx.mlx_array_free(v);
    }
};

/// Model-owned immutable weight copies. Retained inputs pin descriptor identities.
pub const Cache = struct {
    entries: std.AutoHashMapUnmanaged([3]usize, Packed) = .{},

    pub fn deinit(self: *Cache) void {
        var it = self.entries.valueIterator();
        while (it.next()) |v| v.deinit();
        self.entries.deinit(std.heap.c_allocator);
        self.* = .{};
    }

    pub fn count(self: *const Cache) usize {
        return self.entries.count();
    }

    fn get(self: *Cache, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, s: mlx.mlx_stream) !Packed {
        const key = [3]usize{ @intFromPtr(mlx.mlx_array_shape(w)), @intFromPtr(mlx.mlx_array_shape(sc)), @intFromPtr(mlx.mlx_array_shape(bi)) };
        if (self.entries.get(key)) |v| return v;
        var prepared = try Packed.init(w, sc, bi, s);
        errdefer prepared.deinit();
        try self.entries.put(std.heap.c_allocator, key, prepared);
        return prepared;
    }
};

fn random(shape: []const c_int, seed: u64, s: mlx.mlx_stream) !mlx.mlx_array {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed));
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_random_normal(&f, shape.ptr, shape.len, .float32, 0, 0.1, key, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, f, .bfloat16, s));
    return out;
}

fn rows(x: mlx.mlx_array, lo: c_int, hi: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&out, x, &[_]c_int{ lo, 0 }, 2, &[_]c_int{ hi, mlx.getShape(x)[1] }, 2, &[_]c_int{ 1, 1 }, 2, s));
    return out;
}

test "tensor_qmm: row invariance across windows and agreement with independent f32 truth" {
    if (!@import("transformer.zig").verifyQmmNaxAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    for ([_][2]c_int{ .{ 40, 64 }, .{ 64, 512 }, .{ 64, 8192 }, .{ 1024, 5120 }, .{ 4104, 1024 } }, 0..) |shape, seed| {
        const n = shape[0];
        const k = shape[1];
        const wf = try random(&.{ n, k }, seed + 100, s);
        defer _ = mlx.mlx_array_free(wf);
        var quant = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(quant);
        try mlx.check(mlx.mlx_quantize(&quant, wf, .some(64), .some(4), "affine", .{}, s));
        var w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w);
        var sc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc);
        var bi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bi);
        try mlx.check(mlx.mlx_vector_array_get(&w, quant, 0));
        try mlx.check(mlx.mlx_vector_array_get(&sc, quant, 1));
        try mlx.check(mlx.mlx_vector_array_get(&bi, quant, 2));
        const x = try random(&.{ 16, k }, seed + 7, s);
        defer _ = mlx.mlx_array_free(x);
        const all = (try qmm(x, w, sc, bi, 4, 64, s)) orelse return error.Declined;
        defer _ = mlx.mlx_array_free(all);
        var cache = Cache{};
        defer cache.deinit();
        for (0..2) |_| {
            const cached = (try qmmCached(&cache, x, w, sc, bi, 4, 64, s)).?;
            defer _ = mlx.mlx_array_free(cached);
            var eq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(eq);
            try mlx.check(mlx.mlx_array_equal(&eq, cached, all, false, s));
            var same = false;
            try mlx.check(mlx.mlx_array_item_bool(&same, eq));
            try std.testing.expect(same);
            try std.testing.expectEqual(@as(usize, 1), cache.count());
        }
        var batched = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(batched);
        try mlx.check(mlx.mlx_reshape(&batched, x, &[_]c_int{ 2, 8, k }, 3, s));
        const batched_out = (try qmm(batched, w, sc, bi, 4, 64, s)).?;
        defer _ = mlx.mlx_array_free(batched_out);
        try std.testing.expectEqualSlices(c_int, &.{ 2, 8, n }, mlx.getShape(batched_out));
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_reshape(&flat, batched_out, &[_]c_int{ 16, n }, 2, s));
        var batch_equal = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(batch_equal);
        try mlx.check(mlx.mlx_array_equal(&batch_equal, flat, all, false, s));
        var same_batch = false;
        try mlx.check(mlx.mlx_array_item_bool(&same_batch, batch_equal));
        try std.testing.expect(same_batch);
        for (0..16) |i| {
            const r: c_int = @intCast(i);
            const xr = try rows(x, r, r + 1, s);
            defer _ = mlx.mlx_array_free(xr);
            const got = (try qmm(xr, w, sc, bi, 4, 64, s)).?;
            defer _ = mlx.mlx_array_free(got);
            const want = try rows(all, r, r + 1, s);
            defer _ = mlx.mlx_array_free(want);
            var eq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(eq);
            try mlx.check(mlx.mlx_array_equal(&eq, got, want, false, s));
            var same = false;
            try mlx.check(mlx.mlx_array_item_bool(&same, eq));
            try std.testing.expect(same);
        }
        for ([_][2]c_int{ .{ 2, 5 }, .{ 3, 12 }, .{ 0, 8 }, .{ 7, 15 } }) |win| {
            const xr = try rows(x, win[0], win[1], s);
            defer _ = mlx.mlx_array_free(xr);
            const got = (try qmm(xr, w, sc, bi, 4, 64, s)).?;
            defer _ = mlx.mlx_array_free(got);
            const want = try rows(all, win[0], win[1], s);
            defer _ = mlx.mlx_array_free(want);
            var eq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(eq);
            try mlx.check(mlx.mlx_array_equal(&eq, got, want, false, s));
            var same = false;
            try mlx.check(mlx.mlx_array_item_bool(&same, eq));
            try std.testing.expect(same);
        }
        var wd = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wd);
        try mlx.check(mlx.mlx_dequantize(&wd, w, sc, bi, .some(64), .some(4), "affine", .{}, .{ .value = .float32, .has_value = true }, s));
        var wt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wt);
        try mlx.check(mlx.mlx_transpose(&wt, wd, s));
        var xf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xf);
        try mlx.check(mlx.mlx_astype(&xf, x, .float32, s));
        var truth = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(truth);
        try mlx.check(mlx.mlx_matmul(&truth, xf, wt, s));
        var gotf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gotf);
        try mlx.check(mlx.mlx_astype(&gotf, all, .float32, s));
        var stock = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(stock);
        try mlx.check(mlx.mlx_quantized_matmul(&stock, x, w, sc, bi, true, .some(64), .some(4), "affine", s));
        var stockf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(stockf);
        try mlx.check(mlx.mlx_astype(&stockf, stock, .float32, s));
        try mlx.check(mlx.mlx_array_eval(truth));
        try mlx.check(mlx.mlx_array_eval(gotf));
        try mlx.check(mlx.mlx_array_eval(stockf));
        const count = mlx.mlx_array_size(truth);
        const actual = mlx.mlx_array_data_float32(gotf).?[0..count];
        const expected = mlx.mlx_array_data_float32(truth).?[0..count];
        const stock_values = mlx.mlx_array_data_float32(stockf).?[0..count];
        var squared_error: f64 = 0;
        var squared_stock_error: f64 = 0;
        var squared_truth: f64 = 0;
        for (actual, expected, stock_values) |a, e, b| {
            try std.testing.expect(std.math.isFinite(a) and std.math.isFinite(e));
            squared_error += @as(f64, a - e) * (a - e);
            squared_stock_error += @as(f64, b - e) * (b - e);
            squared_truth += @as(f64, e) * e;
        }
        try std.testing.expect(@sqrt(squared_error / squared_truth) < 0.01);
        try std.testing.expect(squared_error <= 9 * squared_stock_error);
    }
}

test "tensor_qmm: unsupported dtype, quantization and geometry decline without a launch" {
    const s = mlx.gpuStream();
    const x = try random(&.{ 1, 64 }, 1, s);
    defer _ = mlx.mlx_array_free(x);
    var w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w);
    try mlx.check(mlx.mlx_zeros(&w, &[_]c_int{ 8, 8 }, 2, .uint32, s));
    const sc = try random(&.{ 8, 1 }, 2, s);
    defer _ = mlx.mlx_array_free(sc);
    const bi = try random(&.{ 8, 1 }, 3, s);
    defer _ = mlx.mlx_array_free(bi);
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    try std.testing.expectEqual(null, try qmm(x, w, sc, bi, 4, 64, cpu));
    try std.testing.expect(!supports(x, w, sc, bi, 4, 64, cpu));
    try std.testing.expectEqual(null, try qmm(x, w, sc, bi, 8, 64, s));
    try std.testing.expectEqual(null, try qmm(x, w, sc, bi, 4, 32, s));
    try std.testing.expectEqual(null, try qmm(x, w, .{ .ctx = null }, bi, 4, 64, s));
    try std.testing.expectEqual(null, try qmm(x, w, sc, .{ .ctx = null }, 4, 64, s));
    var fp32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(fp32);
    try mlx.check(mlx.mlx_astype(&fp32, x, .float32, s));
    try std.testing.expectEqual(null, try qmm(fp32, w, sc, bi, 4, 64, s));
    for ([_][]const c_int{ &.{ 17, 64 }, &.{ 1, 128 }, &.{ 0, 64 }, &.{ 2, 9, 64 } }) |shape| {
        const bad = try random(shape, 4, s);
        defer _ = mlx.mlx_array_free(bad);
        try std.testing.expectEqual(null, try qmm(bad, w, sc, bi, 4, 64, s));
    }
    const bad_scale = try random(&.{ 8, 2 }, 5, s);
    defer _ = mlx.mlx_array_free(bad_scale);
    try std.testing.expectEqual(null, try qmm(x, w, bad_scale, bi, 4, 64, s));
    try std.testing.expectEqual(null, try qmm(x, w, sc, bad_scale, 4, 64, s));
}
