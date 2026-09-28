//! Experimental M5 DFlash row-invariant uint4 matmul. Reference: TensorFold
//! Python kernels/qwen/dense/v1/lane_qmm.py (_MAIN and _XSUM), MIT, see NOTICE.
//! Keeps the existing packed weights and affine metadata: no second layout.
const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const HEADER = "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace mpp::tensor_ops;\n";
const SUM =
    \\const int g = thread_position_in_grid.x, m = thread_position_in_grid.y;
    \\if (g >= K / GS || m >= 16) return;
    \\float acc = 0.0f;
    \\if (m < X_shape[0]) for (int i = 0; i < GS; i++) acc += float(X[m*K+g*GS+i]);
    \\XS[g*16+m] = acc;
;

const MAIN =
    \\const ushort lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    \\const short qid = lane >> 2;
    \\const short fm = (qid & 4) | ((lane >> 1) & 3);
    \\const short fn = ((qid & 2) | (lane & 1)) * 4;
    \\const int M = X_shape[0], n0 = threadgroup_position_in_grid.x * NT;
    \\constexpr int KG = K / GS;
    \\threadgroup bfloat metadata[META == 2 ? NT*KG*2 : 1];
    \\if constexpr (META == 2) {
    \\  for (int i = thread_index_in_threadgroup; i < NT*KG; i += SK*32) {
    \\    const int n = i / KG, g = i % KG;
    \\    metadata[(g*NT+n)*2] = SC[(n0+n)*KG+g];
    \\    metadata[(g*NT+n)*2+1] = BI[(n0+n)*KG+g];
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\}
    \\constexpr auto desc = matmul2d_descriptor(16, NT, GS, false, true, false, matmul2d_descriptor::mode::multiply);
    \\matmul2d<desc, execution_simdgroup> op;
    \\tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA((device bfloat*)X, dextents<int32_t, 2>(K, M));
    \\float C[NT/2];
    \\for (int i = 0; i < NT/2; i++) C[i] = 0.0f;
    \\for (int g = (sg*KG)/SK; g < ((sg+1)*KG)/SK; g++) {
    \\  float scales[NT/4], biases[NT/4], xs[2];
    \\  if constexpr (META != 0) {
    \\    xs[0] = XS[g*16+fm]; xs[1] = XS[g*16+fm+8];
    \\    for (int f = 0; f < NT/16; f++) for (int j = 0; j < 4; j++) {
    \\      const int n = f*16+fn+j;
    \\      scales[f*4+j] = float(META == 2 ? metadata[(g*NT+n)*2] : SC[(n0+n)*KG+g]);
    \\      biases[f*4+j] = float(META == 2 ? metadata[(g*NT+n)*2+1] : BI[(n0+n)*KG+g]);
    \\    }
    \\  }
    \\  auto a = tA.slice(g*GS, 0);
    \\  const int64_t offset = TILED ? (int64_t)((n0/64)*KG+g)*(64*GS/2)+(n0%64)*(GS/2) : 0;
    \\  tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> tB((device uchar*)W+offset, dextents<int32_t, 2>(TILED ? GS : K, TILED ? NT : N));
    \\  auto b = tB.slice(TILED ? 0 : g*GS, TILED ? 0 : n0);
    \\  auto P = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
    \\  op.run(a, b, P);
    \\  for (int f = 0; f < NT/16; f++) for (int j = 0; j < 4; j++) {
    \\    const int n = n0+f*16+fn+j;
    \\    const float sc = META != 0 ? scales[f*4+j] : float(SC[n*KG+g]);
    \\    const float bi = META != 0 ? biases[f*4+j] : float(BI[n*KG+g]);
    \\    for (int r = 0; r < 2; r++) {
    \\      const int i = f*8+r*4+j;
    \\      C[i] = fma(sc, P[i], fma(bi, META != 0 ? xs[r] : XS[g*16+fm+r*8], C[i]));
    \\    }
    \\  }
    \\}
    \\threadgroup float part[(SK > 1 ? SK-1 : 1)*(NT/2)*32];
    \\if (SK > 1) {
    \\  if (sg > 0) for (int i = 0; i < NT/2; i++) part[((sg-1)*(NT/2)+i)*32+lane] = C[i];
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if (sg == 0) for (int s2 = 1; s2 < SK; s2++) for (int i = 0; i < NT/2; i++) C[i] += part[((s2-1)*(NT/2)+i)*32+lane];
    \\}
    \\if (sg == 0) for (int f = 0; f < NT/16; f++) for (int r = 0; r < 2; r++) {
    \\  const int m = fm+8*r, n = n0+f*16+fn;
    \\  if (m < M) for (int j = 0; j < 4; j++) Y[m*N+n+j] = bfloat(C[f*8+r*4+j]);
    \\}
;
const Plan = struct { sum: mlx.mlx_fast_metal_kernel_config, main: mlx.mlx_fast_metal_kernel_config };
var plans: std.AutoHashMapUnmanaged([8]c_int, Plan) = .{};
var sum_kernel: ?mlx.mlx_fast_metal_kernel = null;
var main_kernel: ?mlx.mlx_fast_metal_kernel = null;
var logged: u32 = 0;
var logged_shared: bool = false;
var logged_metadata: [3]bool = @splat(false);

pub fn metadataMode() u2 {
    const p = std.c.getenv("MLX_SERVE_DFLASH_NAX_METADATA") orelse return 0;
    const value = std.mem.span(p);
    if (std.mem.eql(u8, value, "1")) return 1;
    if (std.mem.eql(u8, value, "2")) return 2;
    if (std.mem.eql(u8, value, "3")) return 3;
    return 0;
}

pub fn enabled() bool {
    const p = std.c.getenv("MLX_SERVE_DFLASH_NAX") orelse return false;
    return std.mem.eql(u8, std.mem.span(p), "1");
}

pub fn drafterEnabled() bool {
    const p = std.c.getenv("MLX_SERVE_DFLASH_NAX_DRAFTER") orelse return false;
    return std.mem.eql(u8, std.mem.span(p), "1");
}

pub fn sharedInputEnabled() bool {
    const p = std.c.getenv("MLX_SERVE_DFLASH_NAX_SHARED_INPUT") orelse return false;
    return std.mem.eql(u8, std.mem.span(p), "1");
}

pub fn tileWidth() c_int {
    const raw = std.c.getenv("MLX_SERVE_DFLASH_NAX_TILE");
    return if (raw != null and std.mem.eql(u8, std.mem.span(raw.?), "16")) 16 else 32;
}

fn kernel(sum: bool) !mlx.mlx_fast_metal_kernel {
    const slot = if (sum) &sum_kernel else &main_kernel;
    if (slot.*) |k| return k;
    const ins: []const [*:0]const u8 = if (sum) &.{"X"} else &.{ "X", "W", "SC", "BI", "XS" };
    const outs = [_][*:0]const u8{if (sum) "XS" else "Y"};
    const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outs, 1);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(if (sum) "dflash_nax_xsum" else "dflash_nax_uint4", iv, ov, if (sum) SUM else MAIN, if (sum) "" else HEADER, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    slot.* = k;
    return k;
}

fn plan(m: c_int, k: c_int, n: c_int, gs: c_int, split_n: c_int, nt: c_int, metadata: u2, tiled: bool) !Plan {
    const key = [8]c_int{ m, k, n, gs, split_n, nt, metadata, @intFromBool(tiled) };
    if (plans.get(key)) |p| return p;
    var sk: c_int = 1;
    while (sk < 8 and @divTrunc(split_n + 31, 32) * sk < 1024 and @divTrunc(@divExact(k, gs), sk * 2) >= 8) sk *= 2;
    const p = Plan{ .sum = mlx.mlx_fast_metal_kernel_config_new(), .main = mlx.mlx_fast_metal_kernel_config_new() };
    errdefer {
        _ = mlx.mlx_fast_metal_kernel_config_free(p.sum);
        _ = mlx.mlx_fast_metal_kernel_config_free(p.main);
    }
    for ([_]mlx.mlx_fast_metal_kernel_config{ p.sum, p.main }) |c| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "K", k));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "GS", gs));
    }
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(p.main, "NT", nt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(p.main, "N", n));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(p.main, "SK", sk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(p.main, "META", metadata));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(p.main, "TILED", @intFromBool(tiled)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(p.sum, &[_]c_int{ @divExact(k, gs), 16 }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(p.sum, @divExact(k, gs), 16, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(p.sum, 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(p.main, &[_]c_int{ m, n }, 2, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(p.main, @divExact(n, nt) * sk * 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(p.main, sk * 32, 1, 1));
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

/// Caller must gate on M5 NAX availability. No environment gate here so tests
/// can exercise the exact dispatch without changing process environment.
pub fn qmm(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32, s: mlx.mlx_stream) !?mlx.mlx_array {
    const ws = mlx.getShape(w);
    // Tiny standalone projections are faster on the existing SIMD path.
    if (ws.len != 2 or @mod(ws[0], 32) != 0) return null;
    return qmmWithSplitColumns(x, w, sc, bi, bits, group_size, null, s);
}

/// Like TensorFold's lane_fuse, joined projections preserve each member's
/// split count rather than recomputing it from the combined output width.
pub fn qmmWithSplitColumns(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32, split_columns: ?c_int, s: mlx.mlx_stream) !?mlx.mlx_array {
    return qmmWithTile(x, w, sc, bi, bits, group_size, split_columns, tileWidth(), s);
}

pub fn qmmWithTile(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32, split_columns: ?c_int, nt: c_int, s: mlx.mlx_stream) !?mlx.mlx_array {
    var input = (try Input.init(x, s)) orelse return null;
    defer input.deinit();
    return input.project(w, sc, bi, bits, group_size, split_columns, nt);
}

/// The same arithmetic over canonical 64-column storage, reading 16 or 32
/// columns at a time. No repack and no original-layout array is constructed.
pub fn qmmTiled(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, group_size: u32, split_columns: ?c_int, nt: c_int, s: mlx.mlx_stream) !?mlx.mlx_array {
    var input = (try Input.init(x, s)) orelse return null;
    defer input.deinit();
    return input.projectWithLayout(w, sc, bi, 4, group_size, split_columns, nt, metadataMode(), true);
}

/// Scope-owned activation and group sums, shared by projections of that input.
/// Holding the reshape avoids a pointer-key cache: MLX C handles can be reused
/// while a lazy graph still retains the underlying array. Results own their
/// graph dependencies and remain valid after this scope is destroyed.
pub const Input = struct {
    x: mlx.mlx_array,
    sums: [2]?mlx.mlx_array = .{ null, null },
    shape: [8]c_int,
    ndim: usize,
    m: c_int,
    k: c_int,
    s: mlx.mlx_stream,

    pub fn init(x: mlx.mlx_array, s: mlx.mlx_stream) !?Input {
        if (!mlx.streamIsGpu(s) or mlx.mlx_array_dtype(x) != .bfloat16) return null;
        const xs = mlx.getShape(x);
        if (xs.len < 2 or xs.len > 8) return null;
        const k = xs[xs.len - 1];
        if (k <= 0 or @mod(k, 64) != 0) return null;
        var m: c_int = 1;
        for (xs[0 .. xs.len - 1]) |d| {
            if (d < 1 or d > 16 or m > @divTrunc(16, d)) return null;
            m *= d;
        }
        var input = Input{ .x = mlx.mlx_array_new(), .shape = undefined, .ndim = xs.len, .m = m, .k = k, .s = s };
        errdefer input.deinit();
        @memcpy(input.shape[0..xs.len], xs);
        try mlx.check(mlx.mlx_reshape(&input.x, x, &[_]c_int{ m, k }, 2, s));
        return input;
    }

    pub fn deinit(self: *Input) void {
        for (self.sums) |sum| if (sum) |a| {
            _ = mlx.mlx_array_free(a);
        };
        _ = mlx.mlx_array_free(self.x);
    }

    pub fn project(self: *Input, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32, split_columns: ?c_int, nt: c_int) !?mlx.mlx_array {
        return self.projectWithMetadata(w, sc, bi, bits, group_size, split_columns, nt, metadataMode());
    }

    pub fn projectWithMetadata(self: *Input, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32, split_columns: ?c_int, nt: c_int, requested_metadata: u2) !?mlx.mlx_array {
        return self.projectWithLayout(w, sc, bi, bits, group_size, split_columns, nt, requested_metadata, false);
    }

    fn projectWithLayout(self: *Input, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32, split_columns: ?c_int, nt: c_int, requested_metadata: u2, tiled: bool) !?mlx.mlx_array {
        if (bits != 4 or (group_size != 32 and group_size != 64)) return null;
        if (sc.ctx == null or bi.ctx == null or mlx.mlx_array_dtype(w) != .uint32) return null;
        for ([_]mlx.mlx_array{ sc, bi }) |a| if (mlx.mlx_array_dtype(a) != .bfloat16) return null;
        const ws = mlx.getShape(w);
        const gs: c_int = @intCast(group_size);
        if (tiled) {
            if (ws.len != 4 or ws[2] != 64 or ws[3] * 8 != gs or ws[1] * gs != self.k) return null;
        } else if (ws.len != 2 or ws[1] * 8 != self.k) return null;
        const n = ws[0] * @as(c_int, if (tiled) 64 else 1);
        if (n <= 0 or @mod(n, 32) != 0) return null;
        for ([_]mlx.mlx_array{ sc, bi }) |a| {
            const sh = mlx.getShape(a);
            if (sh.len != 2 or sh[0] != n or sh[1] != @divExact(self.k, gs)) return null;
        }
        if (split_columns) |v| if (v <= 0 or v > n) return null;
        if (nt != 16 and nt != 32) return null;
        // Keep metadata plus worst-case split partials within 32 KiB. The
        // stage is temporary threadgroup storage, never another model layout.
        // Mode 3 isolates multi-row staging from the one-row first-token path.
        const mode: u2 = if (requested_metadata == 3) (if (self.m == 1) 0 else 2) else requested_metadata;
        const metadata: u2 = if (mode == 2 and
            (nt != 16 or 4 * @as(i64, nt) * @divExact(self.k, gs) + 7 * @divExact(nt, 2) * 32 * 4 > 32768)) 1 else mode;
        const p = try plan(self.m, self.k, n, gs, split_columns orelse n, nt, metadata, tiled);
        const slot = &self.sums[if (gs == 32) @as(usize, 0) else 1];
        if (slot.* == null) {
            slot.* = try launch(true, p.sum, &.{self.x}, self.s);
        } else if (!logged_shared) {
            logged_shared = true;
            log.info("[dflash-nax] shared input sums engaged: rows={d} K={d} gs={d}\n", .{ self.m, self.k, gs });
        }
        const y = try launch(false, p.main, &.{ self.x, w, sc, bi, slot.*.? }, self.s);
        defer _ = mlx.mlx_array_free(y);
        if (metadata != 0 and !logged_metadata[metadata]) {
            logged_metadata[metadata] = true;
            log.info("[dflash-nax] affine metadata mode={d} engaged: rows={d} K={d} N={d} tile={d} layout={s}\n", .{ metadata, self.m, self.k, n, nt, if (tiled) "tiled" else "original" });
        }
        var shape = self.shape;
        shape[self.ndim - 1] = n;
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_reshape(&out, y, &shape, self.ndim, self.s));
        const flag = @as(u32, 1) << @as(u5, @intCast(self.m));
        if (logged & flag == 0) {
            logged |= flag;
            log.info("[dflash-nax] uint4 engaged: rows={d} K={d} N={d} gs={d} tile={d} layout={s}\n", .{ self.m, self.k, n, gs, nt, if (tiled) "tiled" else "original" });
        }
        return out;
    }
};
