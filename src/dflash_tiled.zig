//! Single-storage tiled uint4 projections. TensorFold's Python lane_qmm.py
//! tile_weight / _COOP are the reference (MIT; see NOTICE).
//! Layout is explicit in the rank: [N/64, K/GS, 64, GS/8] uint32.
//! Never reconstruct the original matrix during a forward, even for prefill.
const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");

const HEADER = "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace mpp::tensor_ops;\n";
const SUM =
    \\const int g = thread_position_in_grid.x, m = thread_position_in_grid.y;
    \\if (g >= K/GS || m >= MP) return;
    \\float acc = 0.0f;
    \\if (m < X_shape[0]) for (int i = 0; i < GS; i++) acc += float(X[m*K+g*GS+i]);
    \\XS[g*MP+m] = acc;
;
const COOP =
    \\const ushort sg = simdgroup_index_in_threadgroup;
    \\const ushort slice = sg >> 1;
    \\const ushort tip = ushort(thread_position_in_threadgroup.x) - slice*64;
    \\const int M = X_shape[0], rb = threadgroup_position_in_grid.y*16*TMR;
    \\constexpr int KG = K/GS;
    \\const int n0 = threadgroup_position_in_grid.x*64;
    \\constexpr auto desc = matmul2d_descriptor(16*TMR, 64, GS, false, true, false, matmul2d_descriptor::mode::multiply);
    \\matmul2d<desc, execution_simdgroups<2>> op;
    \\tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA((device bfloat*)X + (int64_t)rb*K, dextents<int32_t, 2>(K, M-rb));
    \\auto a0 = tA.slice(0, 0);
    \\tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b0((device uchar*)W, dextents<int32_t, 2>(GS, 64));
    \\auto P = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
    \\constexpr int CAP = 16*TMR;
    \\short ecol[CAP], erow[CAP];
    \\float C[CAP];
    \\for (int i = 0; i < CAP; i++) { auto ids = P.get_multidimensional_index(i); ecol[i] = ids[0]; erow[i] = ids[1]; C[i] = 0.0f; }
    \\for (int g = (slice*KG)/SK; g < ((slice+1)*KG)/SK; g++) {
    \\  auto a = tA.slice(g*GS, 0);
    \\  tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b(
    \\    (device uchar*)W + (TILED ? (int64_t)(threadgroup_position_in_grid.x*KG+g)*(64*GS/2) : 0), dextents<int32_t, 2>(TILED ? GS : K, TILED ? 64 : N));
    \\  auto tile = b.slice(TILED ? 0 : g*GS, TILED ? 0 : n0);
    \\  op.run(a, tile, P);
    \\  for (int i = 0; i < CAP; i++) {
    \\    const int n = n0+ecol[i];
    \\    const float xs = !EDGE || rb+erow[i] < MP ? XS[g*MP+rb+erow[i]] : 0.0f;
    \\    C[i] = fma(float(SC[n*KG+g]), P[i], fma(float(BI[n*KG+g]), xs, C[i]));
    \\  }
    \\}
    \\threadgroup float part[(SK > 1 ? SK-1 : 1)*16*64];
    \\if (SK > 1) for (int c0 = 0; c0 < CAP; c0 += 16) {
    \\  if (slice > 0) for (int i = 0; i < 16; i++) part[((slice-1)*16+i)*64+tip] = C[c0+i];
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if (slice == 0) for (int s2 = 1; s2 < SK; s2++) for (int i = 0; i < 16; i++) C[c0+i] += part[((s2-1)*16+i)*64+tip];
    \\  if (c0+16 < CAP) threadgroup_barrier(mem_flags::mem_threadgroup);
    \\}
    \\if (slice == 0) for (int i = 0; i < CAP; i++) {
    \\  const int m = rb+erow[i];
    \\  if (m < M) Y[m*N+n0+ecol[i]] = bfloat(C[i]);
    \\}
;
var kernels: [2]?mlx.mlx_fast_metal_kernel = .{ null, null };
const Plan = struct { sum: mlx.mlx_fast_metal_kernel_config, main: mlx.mlx_fast_metal_kernel_config };
var plans: std.AutoHashMapUnmanaged([6]c_int, Plan) = .{};
var logged: [2][3]bool = .{ @splat(false), @splat(false) };
var narrow_tile: ?c_int = null;
var logged_narrow = false;

fn narrowTile() c_int {
    if (narrow_tile) |v| return v;
    const raw = std.c.getenv("MLX_SERVE_DFLASH_TILED_TILE");
    const value: c_int = if (raw) |p| (if (std.mem.eql(u8, std.mem.span(p), "16")) 16 else if (std.mem.eql(u8, std.mem.span(p), "32")) 32 else 64) else 64;
    narrow_tile = value;
    return value;
}

pub fn enabled() bool {
    const p = std.c.getenv("MLX_SERVE_DFLASH_TILED_MLP") orelse return false;
    return std.mem.eql(u8, std.mem.span(p), "1");
}

/// Same arithmetic at every row width with untouched weights: isolates layout
/// performance from the stock-prefill versus row-exact numerical difference.
pub fn referenceEnabled() bool {
    const p = std.c.getenv("MLX_SERVE_DFLASH_TILED_MLP") orelse return false;
    return std.mem.eql(u8, std.mem.span(p), "reference");
}

pub fn isTiled(w: mlx.mlx_array) bool {
    if (w.ctx == null or mlx.mlx_array_ndim(w) != 4 or mlx.mlx_array_dtype(w) != .uint32) return false;
    const sh = mlx.getShape(w);
    return sh.len == 4 and sh[0] > 0 and sh[1] > 0 and sh[2] == 64 and (sh[3] == 4 or sh[3] == 8);
}

/// Caller transfers the materialized result to every owner of the original,
/// then releases the original. Transient storage is one projection, not a model.
pub fn tile(w: mlx.mlx_array, gs: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const sh = mlx.getShape(w);
    if (mlx.mlx_array_dtype(w) != .uint32 or sh.len != 2 or sh[0] <= 0 or @mod(sh[0], 64) != 0 or
        (gs != 32 and gs != 64) or sh[1] <= 0 or @mod(sh[1], @divExact(gs, 8)) != 0) return error.InvalidTiledWeight;
    var r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r);
    var t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(t);
    const shape = [_]c_int{ @divExact(sh[0], 64), 64, @divExact(sh[1] * 8, gs), @divExact(gs, 8) };
    try mlx.check(mlx.mlx_reshape(&r, w, &shape, 4, s));
    try mlx.check(mlx.mlx_transpose_axes(&t, r, &[_]c_int{ 0, 2, 1, 3 }, 4, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_contiguous(&out, t, false, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

fn kernel(sum: bool) !mlx.mlx_fast_metal_kernel {
    const slot = &kernels[if (sum) @as(usize, 0) else 1];
    if (slot.*) |k| return k;
    const ins: []const [*:0]const u8 = if (sum) &.{"X"} else &.{ "X", "W", "SC", "BI", "XS" };
    const outs = [_][*:0]const u8{if (sum) "XS" else "Y"};
    const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outs, 1);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(if (sum) "dflash_tiled_sum" else "dflash_tiled_coop", iv, ov, if (sum) SUM else COOP, if (sum) "" else HEADER, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    slot.* = k;
    return k;
}

fn plan(m: c_int, k: c_int, n: c_int, gs: c_int, split_n: c_int, tiled: bool) !Plan {
    const key = [6]c_int{ m, k, n, gs, split_n, @intFromBool(tiled) };
    if (plans.get(key)) |p| return p;
    var sk: c_int = 1;
    while (sk < 8 and @divTrunc(split_n + 31, 32) * sk < 1024 and @divTrunc(@divExact(k, gs), sk * 2) >= 8) sk *= 2;
    const mp = @divTrunc(m + 15, 16) * 16;
    const tmr: c_int = if (m > 16) 2 else 1;
    const p = Plan{ .sum = mlx.mlx_fast_metal_kernel_config_new(), .main = mlx.mlx_fast_metal_kernel_config_new() };
    errdefer {
        _ = mlx.mlx_fast_metal_kernel_config_free(p.sum);
        _ = mlx.mlx_fast_metal_kernel_config_free(p.main);
    }
    for ([_]mlx.mlx_fast_metal_kernel_config{ p.sum, p.main }) |c| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "K", k));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "GS", gs));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "MP", mp));
    }
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(p.main, "N", n));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(p.main, "SK", sk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(p.main, "TMR", tmr));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(p.main, "EDGE", @intFromBool(@mod(mp, 16 * tmr) != 0)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(p.main, "TILED", @intFromBool(tiled)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(p.sum, &[_]c_int{ @divExact(k, gs), mp }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(p.sum, @divExact(k, gs), mp, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(p.sum, 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(p.main, &[_]c_int{ m, n }, 2, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(p.main, @divExact(n, 64) * sk * 64, @divTrunc(mp + 16 * tmr - 1, 16 * tmr), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(p.main, sk * 64, 1, 1));
    try plans.put(std.heap.c_allocator, key, p);
    return p;
}

fn launch(sum: bool, c: mlx.mlx_fast_metal_kernel_config, inputs: []const mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const iv = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, try kernel(sum), iv, c, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    return out;
}

/// Mandatory consumer of a tiled matrix. Declining to stock would interpret
/// its bytes incorrectly; unsupported activations produce an explicit error.
pub fn qmm(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, split_columns: ?c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    if (!isTiled(w)) return error.InvalidTiledProjection;
    const nt = narrowTile();
    if (nt != 64) {
        if (try @import("dflash_nax.zig").qmmTiled(x, w, sc, bi, @intCast(mlx.getShape(w)[3] * 8), split_columns, nt, s)) |y| {
            if (!logged_narrow) {
                logged_narrow = true;
                log.info("[dflash-tiled] narrow reader engaged: tile={d} storage=single layout=tiled\n", .{nt});
            }
            return y;
        }
    }
    return project(x, w, sc, bi, split_columns, true, s);
}

pub fn originalQmm(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, split_columns: ?c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    return project(x, w, sc, bi, split_columns, false, s);
}

fn project(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, split_columns: ?c_int, tiled: bool, s: mlx.mlx_stream) !mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or mlx.mlx_array_dtype(x) != .bfloat16 or sc.ctx == null) return error.InvalidTiledProjection;
    const ws = mlx.getShape(w);
    if (!tiled and (ws.len != 2 or @mod(ws[0], 64) != 0 or mlx.mlx_array_dtype(w) != .uint32)) return error.InvalidTiledProjection;
    const ss = mlx.getShape(sc);
    if (ss.len != 2 or ss[1] <= 0) return error.InvalidTiledProjection;
    const n = if (tiled) ws[0] * 64 else ws[0];
    const k = if (tiled) ws[1] * ws[3] * 8 else ws[1] * 8;
    if (@mod(k, ss[1]) != 0) return error.InvalidTiledProjection;
    const gs = @divExact(k, ss[1]);
    if (gs != 32 and gs != 64) return error.InvalidTiledProjection;
    const xs = mlx.getShape(x);
    if (xs.len < 2 or xs.len > 8 or xs[xs.len - 1] != k) return error.InvalidTiledProjection;
    for ([_]mlx.mlx_array{ sc, bi }) |a| {
        if (a.ctx == null or mlx.mlx_array_dtype(a) != .bfloat16) return error.InvalidTiledProjection;
        const sh = mlx.getShape(a);
        if (sh.len != 2 or sh[0] != n or sh[1] != @divExact(k, gs)) return error.InvalidTiledProjection;
    }
    var shape: [8]c_int = undefined;
    @memcpy(shape[0..xs.len], xs);
    var m: c_int = 1;
    for (xs[0 .. xs.len - 1]) |d| {
        if (d <= 0) return error.InvalidTiledProjection;
        m = try std.math.mul(c_int, m, d);
    }
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, x, &[_]c_int{ m, k }, 2, s));
    const p = try plan(m, k, n, gs, split_columns orelse n, tiled);
    const sums = try launch(true, p.sum, &.{flat}, s);
    defer _ = mlx.mlx_array_free(sums);
    const y = try launch(false, p.main, &.{ flat, w, sc, bi, sums }, s);
    defer _ = mlx.mlx_array_free(y);
    shape[xs.len - 1] = n;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, y, &shape, xs.len, s));
    const category: usize = if (m == 1) 0 else if (m <= 16) 1 else 2;
    const layout: usize = @intFromBool(tiled);
    if (!logged[layout][category]) {
        logged[layout][category] = true;
        log.info("[dflash-tiled] coop engaged: rows={d} K={d} N={d} tile=64 storage=single layout={s}\n", .{ m, k, n, if (tiled) "tiled" else "original-reference" });
    }
    return out;
}
