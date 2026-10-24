//! sdk/blocks/linear/qmv_rows, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns MSV_QMV_ROWS_SOURCE, msv_qmv_rows_kernel, MsvQmvRowsCfgKey, msv_qmv_rows_cfgs, msv_qmv_rows_logged, getMsvQmvRowsKernel, and 1 more.

const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../util/log.zig");
const Weights = @import("../../weights/resident.zig").Weights;
const testing = std.testing;
const KernelCfgCache = @import("../../../mlx/kernel_cache.zig").KernelCfgCache;
const firstRowExactMiss = @import("../../quant/ternary.zig").firstRowExactMiss;
const testRandWeightBf16 = @import("../../testing/parity.zig").testRandWeightBf16;
const attn256MaxDiff = @import("../attention/attn256.zig").attn256MaxDiff;
const attn256RandBf16 = @import("../attention/attn256.zig").attn256RandBf16;
const moeRowsSliceRow = @import("../moe/gather_qmv_rows.zig").moeRowsSliceRow;
const cachedScalarInt = @import("verify_qmm.zig").cachedScalarInt;

pub const MSV_QMV_ROWS_SOURCE = kernelSource("kernels/linear/qmv_rows.metal");

pub var msv_qmv_rows_kernel: ?mlx.mlx_fast_metal_kernel = null;
const MsvQmvRowsCfgKey = struct { nrows: c_int, n: c_int, k: c_int, bits: u32, gs: u32, dtype: mlx.mlx_dtype };
// One slot per head projection shape and row count a batched round cycles through.
var msv_qmv_rows_cfgs = KernelCfgCache(MsvQmvRowsCfgKey, 1, 16){};
/// One engagement line per width.
var msv_qmv_rows_logged: [9]bool = @splat(false);

fn getMsvQmvRowsKernel() !mlx.mlx_fast_metal_kernel {
    if (msv_qmv_rows_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "x", "w_q", "scales", "biases", "K_size", "N_size" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "msv_qmv_rows",
        in_vec,
        out_vec,
        MSV_QMV_ROWS_SOURCE,
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    msv_qmv_rows_kernel = kernel;
    return kernel;
}

pub fn msvQmvRows(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    bits: u32,
    group_size: u32,
) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return null;
    if (bits < 3 or bits > 8 or bits == 7) return null;
    if (sc.ctx == null or bi.ctx == null) return null;
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16) return null;
    if (mlx.mlx_array_dtype(sc) != xd or mlx.mlx_array_dtype(bi) != xd) return null;
    const xsh = mlx.getShape(x);
    if (xsh.len < 1 or xsh.len > 8) return null;
    const K = xsh[xsh.len - 1];
    var nrows: c_int = 1;
    var i: usize = 0;
    while (i + 1 < xsh.len) : (i += 1) nrows *= xsh[i];
    if (nrows < 2 or nrows > 16) return null;
    // MLX qmv_fast's packs: two per thread, 8 values per 3/5 bytes at 3/5 bits, 4 per 3 at 6.
    const vpt: c_int = switch (bits) {
        3, 5 => 16,
        6 => 8,
        else => @intCast((32 / bits) * 2),
    };
    const block: c_int = vpt * 32;
    if (@rem(K, block) != 0) return null;
    // A thread's values sit inside one group.
    if (group_size < vpt or @rem(@as(c_int, @intCast(group_size)), vpt) != 0 or @rem(block, @as(c_int, @intCast(group_size))) != 0) return null;
    const wsh = mlx.getShape(w);
    if (wsh.len != 2) return null;
    const Nout = wsh[0];
    if (@rem(Nout, 8) != 0) return null;
    if (@as(i64, wsh[1]) * 32 != @as(i64, K) * bits) return null;
    if (@rem(K, @as(c_int, @intCast(group_size))) != 0) return null;

    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    if (xsh.len == 2 and xsh[0] == nrows) {
        try mlx.check(mlx.mlx_array_set(&x2, x));
    } else {
        try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ nrows, K }, 2, s));
    }

    const key = MsvQmvRowsCfgKey{ .nrows = nrows, .n = Nout, .k = K, .bits = bits, .gs = group_size, .dtype = xd };
    const config = blk: {
        if (msv_qmv_rows_cfgs.get(key)) |hit| break :blk hit[0];
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        const y_shape = [_]c_int{ nrows, Nout };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &y_shape, 2, xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32, @divExact(Nout, 4), nrows));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 2, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "BITS", @intCast(bits)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "GS", @intCast(group_size)));
        msv_qmv_rows_cfgs.put(key, .{cfg});
        break :blk cfg;
    };

    const K_arr = cachedScalarInt(K);
    const N_arr = cachedScalarInt(Nout);
    const inputs = [_]mlx.mlx_array{ x2, w, sc, bi, K_arr, N_arr };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    const kernel = try getMsvQmvRowsKernel();
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, config, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    mtp_head_row_dispatches += 1;
    if (!msv_qmv_rows_logged[bits]) {
        msv_qmv_rows_logged[bits] = true;
        log.info("[mtp] row-axis projection engaged: rows={d} K={d} N={d} bits={d} gs={d}\n", .{ nrows, K, Nout, bits, group_size });
    }
    var y2 = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y2);
    try mlx.check(mlx.mlx_vector_array_get(&y2, outputs_vec, 0));
    if (xsh.len == 2) return y2;
    var ysh_buf: [8]c_int = undefined;
    for (xsh[0 .. xsh.len - 1], 0..) |d, di| ysh_buf[di] = d;
    ysh_buf[xsh.len - 1] = Nout;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_reshape(&y, y2, &ysh_buf, @intCast(xsh.len), s));
    _ = mlx.mlx_array_free(y2);
    return y;
}

pub var mtp_head_row_dispatches: usize = 0;

// ── tests ──

const ModelConfig = @import("../../../engines/mlx/model.zig").ModelConfig;

test "firstRowExactMiss: 4, 6 and 8-bit trunk projections are inside the row kernels, 3-bit is not; vision and mtp never count" {
    const Q = struct {
        fn put(w: *Weights, base: []const u8, bits: c_int) !void {
            const st = mlx.gpuStream();
            var zeros = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(zeros);
            try mlx.check(mlx.mlx_zeros(&zeros, &[_]c_int{ 64, 128 }, 2, .bfloat16, st));
            var triple = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(triple);
            try mlx.check(mlx.mlx_quantize(&triple, zeros, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(bits), "affine", .{}, st));
            for ([_][]const u8{ ".weight", ".scales", ".biases" }, 0..) |suf, i| {
                var a = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_vector_array_get(&a, triple, i));
                try w.map.put(try std.fmt.allocPrint(std.testing.allocator, "{s}{s}", .{ base, suf }), a);
            }
        }
    };
    const cfg: @import("../../quant/params.zig").QuantParams = .{ .bits = 4, .group_size = 64, .mode = .affine };
    var w = Weights.init(std.testing.allocator);
    defer w.deinit();
    try Q.put(&w, "m.layers.0.q", 4);
    try Q.put(&w, "m.layers.1.q", 6);
    try Q.put(&w, "m.layers.2.q", 8);
    try Q.put(&w, "vision_tower.fc", 3);
    try Q.put(&w, "mtp.layers.0.fc", 3);
    try std.testing.expectEqual(@as(?[]const u8, null), firstRowExactMiss(cfg, &w, .{ .ctx = null }));
    try Q.put(&w, "m.layers.3.q", 3);
    try std.testing.expectEqualStrings("m.layers.3.q.scales", firstRowExactMiss(cfg, &w, .{ .ctx = null }).?);
}

test "msv_qmv_rows is mlx_equal per row to stock M=1 qmv at every head shape, width and group" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    std.debug.print("[msv_qmv_rows] hermetic start\n", .{});
    var prng = std.Random.DefaultPrng.init(0x51500A01);
    const rnd = prng.random();
    const shapes = [_][2]c_int{
        .{ 2560, 2560 },
        .{ 2560, 12288 },
        .{ 2560, 512 },
        .{ 12288, 2560 },
        .{ 2560, 640 },
        .{ 2560, 248320 },
    };
    const bitss = [_]u32{ 3, 4, 5, 6, 8 };
    const ns = [_]c_int{ 2, 4, 8, 16 };
    for ([_]u32{ 64, 128 }) |gs| for (bitss) |bits| {
        for (shapes) |kn| {
            const K = kn[0];
            const Nout = kn[1];
            const w_bf = try testRandWeightBf16(rnd, &[_]c_int{ Nout, K }, s);
            defer _ = mlx.mlx_array_free(w_bf);
            var qvec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(qvec);
            try mlx.check(mlx.mlx_quantize(&qvec, w_bf, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{ .ctx = null }, s));
            var w = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(w);
            var sc = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sc);
            var bi = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(bi);
            try mlx.check(mlx.mlx_vector_array_get(&w, qvec, 0));
            try mlx.check(mlx.mlx_vector_array_get(&sc, qvec, 1));
            try mlx.check(mlx.mlx_vector_array_get(&bi, qvec, 2));
            for (ns) |nrows| {
                const x = try attn256RandBf16(rnd, &[_]c_int{ nrows, K }, s);
                defer _ = mlx.mlx_array_free(x);
                const got = (try msvQmvRows(s, x, w, sc, bi, bits, gs)) orelse {
                    std.debug.print("[msv_qmv_rows] declined bits={d} K={d} N={d} rows={d}\n", .{ bits, K, Nout, nrows });
                    return error.QmvRowsDeclined;
                };
                defer _ = mlx.mlx_array_free(got);
                var r: c_int = 0;
                while (r < nrows) : (r += 1) {
                    const xr = try moeRowsSliceRow(s, x, r);
                    defer _ = mlx.mlx_array_free(xr);
                    var xr2 = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(xr2);
                    try mlx.check(mlx.mlx_reshape(&xr2, xr, &[_]c_int{ 1, K }, 2, s));
                    var want = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(want);
                    try mlx.check(mlx.mlx_quantized_matmul(
                        &want,
                        xr2,
                        w,
                        sc,
                        bi,
                        true,
                        mlx.mlx_optional_int.some(@intCast(gs)),
                        mlx.mlx_optional_int.some(@intCast(bits)),
                        "affine",
                        s,
                    ));
                    const gr = try moeRowsSliceRow(s, got, r);
                    defer _ = mlx.mlx_array_free(gr);
                    const diff = try attn256MaxDiff(gr, want, s);
                    if (diff != 0) {
                        std.debug.print("[msv_qmv_rows] NE bits={d} K={d} N={d} rows={d} row={d} maxdiff={d}\n", .{ bits, K, Nout, nrows, r, diff });
                    }
                    try testing.expectEqual(@as(f32, 0.0), diff);
                }
                std.debug.print("[msv_qmv_rows] eq bits={d} gs={d} K={d} N={d} rows={d}\n", .{ bits, gs, K, Nout, nrows });
            }
        }
    };
}

test "row kernel declines a CPU stream" {
    const s = mlx.gpuStream();
    var fx = try @import("../../spec/mtp/mtp.zig").RerankFixture.init(s, 512, 2560, 4, 64, 11);
    defer fx.deinit();
    var prng = std.Random.DefaultPrng.init(12);
    const x = try attn256RandBf16(prng.random(), &.{ 2, 2560 }, s);
    defer _ = mlx.mlx_array_free(x);
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    const got = try msvQmvRows(cpu, x, fx.xfm.lm_head_w, fx.xfm.lm_head_s, fx.xfm.lm_head_b, 4, 64);
    defer if (got) |y| {
        _ = mlx.mlx_array_free(y);
    };
    try testing.expect(got == null);
}

test "row kernel declines scales and biases with another dtype" {
    const s = mlx.gpuStream();
    var fx = try @import("../../spec/mtp/mtp.zig").RerankFixture.init(s, 512, 2560, 4, 64, 11);
    defer fx.deinit();
    var prng = std.Random.DefaultPrng.init(12);
    const x = try attn256RandBf16(prng.random(), &.{ 2, 2560 }, s);
    defer _ = mlx.mlx_array_free(x);
    var sc32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc32);
    var bi32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bi32);
    try mlx.check(mlx.mlx_astype(&sc32, fx.xfm.lm_head_s, .float32, s));
    try mlx.check(mlx.mlx_astype(&bi32, fx.xfm.lm_head_b, .float32, s));
    for ([_][2]mlx.mlx_array{ .{ sc32, fx.xfm.lm_head_b }, .{ fx.xfm.lm_head_s, bi32 } }) |pair| {
        const got = try msvQmvRows(s, x, fx.xfm.lm_head_w, pair[0], pair[1], 4, 64);
        defer if (got) |y| {
            _ = mlx.mlx_array_free(y);
        };
        try testing.expect(got == null);
    }
}

test "row kernel cache reuses alternating full-shape configurations" {
    const s = mlx.gpuStream();
    var fx = try @import("../../spec/mtp/mtp.zig").RerankFixture.init(s, 512, 2560, 8, 64, 11);
    defer fx.deinit();
    var prng = std.Random.DefaultPrng.init(12);
    const before = msv_qmv_rows_cfgs.builds;
    for ([_]c_int{ 3, 5, 3 }) |rows| {
        const x = try attn256RandBf16(prng.random(), &.{ rows, 2560 }, s);
        defer _ = mlx.mlx_array_free(x);
        const y = (try msvQmvRows(s, x, fx.xfm.lm_head_w, fx.xfm.lm_head_s, fx.xfm.lm_head_b, 8, 64)) orelse return error.HeadRowsDeclined;
        defer _ = mlx.mlx_array_free(y);
        try mlx.check(mlx.mlx_array_eval(y));
    }
    try testing.expectEqual(@as(usize, 2), msv_qmv_rows_cfgs.builds - before);
}
