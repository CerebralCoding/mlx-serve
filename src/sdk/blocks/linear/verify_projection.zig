//! sdk/blocks/linear/verify_projection, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns SingleVerifyFeature, single_verify_test_features, mtp_verify_shared_rows_calls, mtp_verify_shared_gate_rows_calls, mtp_verify_gdn_rows_calls, mtp_verify_gdn_output_calls, and 36 more.

const std = @import("std");
const env = @import("../../../env.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mtp_mod = @import("../../spec/mtp/mtp.zig");
const mlx = @import("../../../mlx/ffi.zig");
const model_mod = @import("../../../engines/mlx/model.zig");
const log = @import("../../../host/log.zig");
const transformer = @import("../../../engines/mlx/transformer.zig");
const verify_qmm = @import("verify_qmm.zig");
const MAX_BATCH_ROWS = @import("../../../engines/mlx/batched/layered.zig").MAX_BATCH_ROWS;
const testing = @import("../../../engines/mlx/kv/cache.zig").testing;
const lastDim = @import("../../../engines/mlx/load/weights.zig").lastDim;
const Transformer = @import("../../../engines/mlx/transformer.zig").Transformer;
const concatAxis0 = @import("../../../mlx/array.zig").concatAxis0;
const gpuArchitecture = @import("../../../mlx/device.zig").gpuArchitecture;
const QsaCfgCache = @import("../../../mlx/kernel_cache.zig").QsaCfgCache;
const qsaArraysAllEqual = @import("../../../plugins/qwen4_exp/qsa/select.zig").qsaArraysAllEqual;
const affineParamsFromGeometry = @import("../../quant/params.zig").affineParamsFromGeometry;
const attn256RandBf16 = @import("../attention/attn256.zig").attn256RandBf16;
const qmatmulBits = @import("qmatmul.zig").qmatmulBits;
const qmvBatchLimit = @import("qmatmul.zig").qmvBatchLimit;
const msvQmvRows = @import("qmv_rows.zig").msvQmvRows;
const cachedScalarInt = @import("verify_qmm.zig").cachedScalarInt;
const mixedNaxShapeEnabled = @import("verify_qmm.zig").mixedNaxShapeEnabled;
const mixedPlainEnabled = @import("verify_qmm.zig").mixedPlainEnabled;
const verifyQmmEnabled = @import("verify_qmm.zig").verifyQmmEnabled;
const naxMinMForBits = @import("verify_qmm_nax.zig").naxMinMForBits;
const verifyQmmNaxAvailable = @import("verify_qmm_nax.zig").verifyQmmNaxAvailable;

pub const SingleVerifyFeature = enum { routing, down_reduce };
pub var single_verify_test_features: [2]bool = @splat(true);

pub var mtp_verify_shared_rows_calls: u64 = 0;
pub var mtp_verify_shared_gate_rows_calls: u64 = 0;

pub var mtp_verify_gdn_rows_calls: u64 = 0;
pub var mtp_verify_gdn_output_calls: u64 = 0;

pub var mtp_verify_indexed_input_calls: u64 = 0;

pub var mtp_verify_expert_reduce_calls: u64 = 0;

pub var mtp_verify_expert_pairs_calls: u64 = 0;

pub var mtp_verify_attn_rows_calls: u64 = 0;
pub var mtp_verify_attn_output_calls: u64 = 0;

pub var mtp_verify_lmhead_rows_calls: u64 = 0;

pub const SharedProjKind = enum { qmatmul, mtp_rows, verify_rows };

pub fn sharedGateRowsEligible(widths: []const c_int) bool {
    if (widths.len < 2 or widths.len > 8) return false;
    var total: c_int = 0;
    for (widths) |width| {
        if (width < 2 or width > 7) return false;
        total += width;
    }
    return total <= 24;
}

pub const VerifyKernel = enum { dense_tiles, hc_rows, route_pack };

pub fn mtpHeadRowsFit(widths: []const c_int, hc: u32) bool {
    if (widths.len == 0 or widths.len > 8 or hc == 0) return false;
    if (@as(u64, hc) * widths.len > 16) return false;
    for (widths) |width| if (width != 1) return false;
    return true;
}

pub fn verifyFeatureEnabled(self: *const Transformer, comptime feature: SingleVerifyFeature, grouped: bool, batch: c_int, width: c_int) bool {
    if (self.qwen4 == null) return false;
    if (grouped) return true;
    if (@import("builtin").is_test and !single_verify_test_features[@backingInt(feature)]) return false;
    // Capture marks speculative verification. A short prefill tail has the
    // same shape but must keep the ordinary forward path and shared expert.
    return self.spec_capture_ssm and batch == 1 and width >= 2 and width <= 6;
}

pub fn mtpProjectionRows(self: *Transformer, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, lm_head: bool) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    if (sh.len < 2 or sh.len > 8 or sh[0] < 1 or sh[0] > MAX_BATCH_ROWS) return error.MtpProjectionShape;
    const n: usize = @intCast(sh[0]);
    var virtual_rows: c_int = 1;
    for (sh[1 .. sh.len - 1]) |d| virtual_rows *= d;
    if (virtual_rows == 1) {
        const qp = self.quantParamsHinted(w, sc, lastDim(x));
        if (qp.mode == .affine) {
            if (try msvQmvRows(self.s, x, w, sc, bi, qp.bits, qp.group_size)) |y| return y;
        }
    }
    var parts: [MAX_BATCH_ROWS]mlx.mlx_array = @splat(.{ .ctx = null });
    defer for (parts[0..n]) |a| {
        if (a.ctx != null) _ = mlx.mlx_array_free(a);
    };
    for (0..n) |row| {
        var starts: [8]c_int = @splat(0);
        var stops: [8]c_int = @splat(0);
        const strides: [8]c_int = @splat(1);
        @memcpy(stops[0..sh.len], sh);
        starts[0] = @intCast(row);
        stops[0] = @intCast(row + 1);
        var xrow = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xrow);
        try mlx.check(mlx.mlx_slice(&xrow, x, &starts, @intCast(sh.len), &stops, @intCast(sh.len), &strides, @intCast(sh.len), self.s));
        parts[row] = if (lm_head) try self.lmHeadProject(xrow) else try self.qmatmul(xrow, w, sc, bi);
    }
    return concatAxis0(self.s, parts[0..n]);
}

var verify_arch_buf: [128]u8 = undefined;
var verify_arch: ?[]const u8 = null;
var verify_arch_read = false;
/// Streams in the batched verify running now (`forwardRowAxisVerify`); 0 outside one.
pub var verify_group_rows: usize = 0;

// MLX affine qmv-fast reduction, with one 3-bit weight read serving two rows.
pub const MTP_COARSE_PAIRS_SOURCE = kernelSource("kernels/linear/mtp_coarse_pairs.metal");
var mtp_coarse_pairs_kernel: ?mlx.mlx_fast_metal_kernel = null;
var mtp_coarse_pairs_cfgs = QsaCfgCache(c_int, 1){};

pub fn mtpCoarsePairs(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, gs: u32) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or !verifySharedHardware() or bits != 3 or gs != 64) return null;
    if (x.ctx == null or w.ctx == null or sc.ctx == null or bi.ctx == null) return null;
    if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(sc) != .bfloat16 or mlx.mlx_array_dtype(bi) != .bfloat16 or mlx.mlx_array_dtype(w) != .uint32) return null;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    const ss = mlx.getShape(sc);
    if (xs.len != 3 or xs[0] < 2 or xs[0] > 8 or xs[1] != 1 or xs[2] != 2560) return null;
    if (!std.mem.eql(c_int, ws, &.{ 248320, 240 }) or !std.mem.eql(c_int, ss, &.{ 248320, 40 }) or !std.mem.eql(c_int, ss, mlx.getShape(bi))) return null;
    const kernel = blk: {
        if (mtp_coarse_pairs_kernel) |value| break :blk value;
        const ins = [_][*:0]const u8{ "x", "w", "sc", "bi", "K_size", "N_size" };
        const outs = [_][*:0]const u8{"y"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const value = mlx.mlx_fast_metal_kernel_new("mtp_coarse_pairs", iv, ov, MTP_COARSE_PAIRS_SOURCE, "", true, false);
        if (value.ctx == null) return error.MetalKernelCompileFailed;
        mtp_coarse_pairs_kernel = value;
        break :blk value;
    };
    const cfg = blk: {
        if (mtp_coarse_pairs_cfgs.get(xs[0])) |hit| break :blk hit[0];
        const value = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(value);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(value, &.{ xs[0], 1, 248320 }, 3, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(value, 32, 248320 / 4, @divTrunc(xs[0] + 1, 2)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(value, 32, 2, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(value, "T", .bfloat16));
        mtp_coarse_pairs_cfgs.put(xs[0], .{value});
        break :blk value;
    };
    const values = [_]mlx.mlx_array{ x, w, sc, bi, cachedScalarInt(2560), cachedScalarInt(248320) };
    const iv = mlx.mlx_vector_array_new_data(&values, values.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel, iv, cfg, s));
    var result = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(result);
    try mlx.check(mlx.mlx_vector_array_get(&result, ov, 0));
    transformer.mtp_coarse_pair_dispatches +%= 1;
    const Once = struct {
        var logged = false;
    };
    if (!Once.logged) {
        Once.logged = true;
        log.info("[mtp] paired coarse head engaged (rows={d})\n", .{xs[0]});
    }
    return result;
}

/// The fused verify kernels need an M5 GPU. A dual-die one (`...d`, the Ultra) gains from them
/// only in a multi-stream group: they lift a batched verify and slow a solo one.
pub fn verifySharedFor(arch: []const u8, group_rows: usize) bool {
    if (!std.mem.startsWith(u8, arch, "applegpu_g17")) return false;
    return !dualDieArch(arch) or group_rows > 1;
}

/// A dual-die GPU (the Ultra) reports an architecture ending in `d`, e.g. `applegpu_g17d`.
pub fn dualDieArch(arch: []const u8) bool {
    return std.mem.endsWith(u8, arch, "d");
}

pub fn verifyArch() ?[]const u8 {
    if (!verify_arch_read) {
        verify_arch_read = true;
        verify_arch = gpuArchitecture(&verify_arch_buf);
    }
    return verify_arch;
}

/// Whether this GPU can run the fused verify kernels at all (load-time pack checks).
pub fn verifySharedCapable() bool {
    if (!verifyQmmNaxAvailable()) return false;
    return std.mem.startsWith(u8, verifyArch() orelse return false, "applegpu_g17");
}

/// Whether the verify running now takes the fused kernels.
pub fn verifySharedHardware() bool {
    if (!verifyQmmNaxAvailable()) return false;
    return verifySharedFor(verifyArch() orelse return false, verify_group_rows);
}

// Same subchunk and shuffle reduction as Apple's MLX qmv_wide_impl (MIT).
pub const VERIFY_WIDE_SOURCE = kernelSource("kernels/linear/verify_wide.metal");
var verify_wide_kernel: ?mlx.mlx_fast_metal_kernel = null;
const VerifyWideKey = struct { m: c_int, n: c_int, k: c_int, nv: c_int };
var verify_wide_configs = QsaCfgCache(VerifyWideKey, 1){};

pub fn verifyWideProjection(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, nv: c_int) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or !verifySharedHardware() or (nv != 4 and nv != 6 and nv != 8 and nv != 12)) return null;
    if (x.ctx == null or w.ctx == null or sc.ctx == null or bi.ctx == null) return null;
    if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(w) != .uint32 or mlx.mlx_array_dtype(sc) != .bfloat16 or mlx.mlx_array_dtype(bi) != .bfloat16) return null;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    const ss = mlx.getShape(sc);
    if (xs.len != 3 or xs[0] != 1 or xs[1] < 6 or xs[1] > 12 or ws.len != 2 or ss.len != 2) return null;
    const m = xs[1];
    const k = xs[2];
    const n = ws[0];
    if ((k != 640 and k != 2560 and k != 6144) or n < 8 or @mod(n, 8) != 0 or ws[1] != @divExact(k, 4) or ss[0] != n or ss[1] != @divExact(k, 64) or !std.mem.eql(c_int, ss, mlx.getShape(bi))) return null;
    const kernel = blk: {
        if (verify_wide_kernel) |value| break :blk value;
        const ins = [_][*:0]const u8{ "x", "w", "sc", "bi" };
        const outs = [_][*:0]const u8{"y"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const value = mlx.mlx_fast_metal_kernel_new("verify_dense_wide", iv, ov, VERIFY_WIDE_SOURCE, "", true, false);
        if (value.ctx == null) return error.MetalKernelCompileFailed;
        verify_wide_kernel = value;
        break :blk value;
    };
    const key = VerifyWideKey{ .m = m, .n = n, .k = k, .nv = nv };
    const config = blk: {
        if (verify_wide_configs.get(key)) |hit| break :blk hit[0];
        const value = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(value);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(value, &.{ 1, m, n }, 3, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(value, @intCast(@divTrunc(m + nv - 1, nv) * 32), @intCast(@divExact(n, 8) * 2), 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(value, 32, 2, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(value, "T", .bfloat16));
        inline for (.{ "M", "N", "K", "NV" }, .{ m, n, k, nv }) |name, number| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(value, name, number));
        verify_wide_configs.put(key, .{value});
        break :blk value;
    };
    const iv = mlx.mlx_vector_array_new_data(&.{ x, w, sc, bi }, 4);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel, iv, config, s));
    var result = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(result);
    try mlx.check(mlx.mlx_vector_array_get(&result, ov, 0));
    return result;
}

const VerifyProjectionKind = enum { shared, gdn, lm_head, attention };

pub fn verifySharedProjection(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, widths: []const c_int) !?mlx.mlx_array {
    return verifyJoinedProjection(s, x, w, sc, bi, widths, .shared);
}

pub fn verifyGdnProjection(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, widths: []const c_int) !?mlx.mlx_array {
    return verifyJoinedProjection(s, x, w, sc, bi, widths, .gdn);
}

pub fn verifyAttnProjection(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, widths: []const c_int) !?mlx.mlx_array {
    return verifyJoinedProjection(s, x, w, sc, bi, widths, .attention);
}

pub fn verifyLmHeadProjection(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, widths: []const c_int) !?mlx.mlx_array {
    return verifyJoinedProjection(s, x, w, sc, bi, widths, .lm_head);
}

fn verifyJoinedProjection(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, widths: []const c_int, comptime kind: VerifyProjectionKind) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or !verifySharedHardware() or widths.len < 2 or widths.len > 8) return null;
    if (x.ctx == null or w.ctx == null or sc.ctx == null or bi.ctx == null) return null;
    if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(w) != .uint32 or
        mlx.mlx_array_dtype(sc) != .bfloat16 or mlx.mlx_array_dtype(bi) != .bfloat16) return null;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    const ss = mlx.getShape(sc);
    const bs = mlx.getShape(bi);
    if (xs.len != 3 or xs[0] != 1 or ws.len != 2 or ss.len != 2 or bs.len != 2) return null;
    const k = xs[2];
    const n = ws[0];
    const wide = kind != .shared;
    const shape_ok = switch (kind) {
        .shared => (k == 2560 and n == 640) or (k == 640 and n == 2560),
        .gdn => (k == 2560 and (n == 10240 or n == 6144 or n == 48)) or (k == 6144 and n == 2560),
        .lm_head => k == 2560 and n == 248320,
        .attention => (k == 2560 and (n == 12288 or n == 512)) or (k == 6144 and n == 2560),
    };
    if (!shape_ok) return null;
    const qp = affineParamsFromGeometry(w, sc, @intCast(k)) orelse return null;
    if (ss[0] != n or !std.mem.eql(c_int, ss, bs)) return null;
    // Rows must match what each slot computes alone: a slot whose width a verify lane
    // serves does not take the reference qmv_wide K reduction.
    if (qp.bits == 4 or mixedPlainEnabled(qp.bits, qp.group_size, n)) return null;
    var total: c_int = 0;
    for (widths) |width| {
        if (width < 2 or width > (if (wide) @as(c_int, 6) else 7)) return null;
        if (wide and verifyQmmEnabled() and
            mixedNaxShapeEnabled(qp.bits, qp.group_size, n, width) and width >= naxMinMForBits(qp.bits) and @mod(k, 256) == 0) return null;
        total += width;
    }
    if (total != xs[1] or total > (if (wide) @as(c_int, 12) else 24)) return null;
    if (total == 12 and n >= 2560 and qp.bits == 8 and qp.group_size == 64) {
        if (try verifyWideProjection(s, x, w, sc, bi, 6)) |value| {
            transformer.mtp_verify_kernel_calls[0] +%= 1;
            const Once = struct {
                var logged = false;
            };
            if (!Once.logged) {
                Once.logged = true;
                log.info("[batched] wider dense verify tiles engaged\n", .{});
            }
            return value;
        }
    }
    const arch = verifyArch() orelse return null;
    if (total >= qmvBatchLimit(dualDieArch(arch), k, n)) return null;
    var result = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(result);
    try mlx.check(mlx.mlx_quantized_matmul(&result, x, w, sc, bi, true, mlx.mlx_optional_int.some(@intCast(qp.group_size)), mlx.mlx_optional_int.some(@intCast(qp.bits)), "affine", s));
    return result;
}

// ── tests ──

test "shared expert gate tail admits the joined verifier envelope only" {
    try testing.expect(sharedGateRowsEligible(&.{ 2, 2, 2, 2 }));
    try testing.expect(sharedGateRowsEligible(&.{ 5, 5, 5, 5 }));
    try testing.expect(sharedGateRowsEligible(&.{ 2, 3 }));
    try testing.expect(!sharedGateRowsEligible(&.{ 7, 7, 7, 7 }));
    try testing.expect(!sharedGateRowsEligible(&.{1}));
}

test "dualDieArch: only the Ultra's suffix" {
    try std.testing.expect(dualDieArch("applegpu_g17d"));
    try std.testing.expect(!dualDieArch("applegpu_g17s"));
    try std.testing.expect(!dualDieArch("applegpu_g17g"));
    try std.testing.expect(!dualDieArch(""));
}

test "verifySharedFor: dual-die M5 takes the fused verify kernels only for a multi-stream group" {
    try std.testing.expect(verifySharedFor("applegpu_g17s", 0));
    try std.testing.expect(verifySharedFor("applegpu_g17s", 4));
    try std.testing.expect(!verifySharedFor("applegpu_g17d", 0));
    try std.testing.expect(!verifySharedFor("applegpu_g17d", 1));
    try std.testing.expect(verifySharedFor("applegpu_g17d", 2));
    try std.testing.expect(verifySharedFor("applegpu_g17d", 4));
    try std.testing.expect(!verifySharedFor("applegpu_g16d", 4));
    try std.testing.expect(!verifySharedFor("applegpu_g16s", 0));
}

test "verify shared projection preserves solo qmv values across request widths, at g64 and g128" {
    // A two-stream group: dual-die M5 takes the joined kernels only then.
    const outer_group_rows = verify_group_rows;
    verify_group_rows = 2;
    defer verify_group_rows = outer_group_rows;
    if (mlx.noGpuBackend() or !verifySharedHardware()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x5A4E);
    for ([_]u32{ 64, 128 }) |gs| for ([_][2]c_int{ .{ 2560, 640 }, .{ 640, 2560 } }) |shape| {
        const k = shape[0];
        const n = shape[1];
        const weight = try attn256RandBf16(prng.random(), &.{ n, k }, s);
        defer _ = mlx.mlx_array_free(weight);
        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(&triple, weight, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(8), "affine", .{}, s));
        var w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w);
        var sc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc);
        var bi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bi);
        try mlx.check(mlx.mlx_vector_array_get(&w, triple, 0));
        try mlx.check(mlx.mlx_vector_array_get(&sc, triple, 1));
        try mlx.check(mlx.mlx_vector_array_get(&bi, triple, 2));
        for ([_]usize{ 2, 3, 4, 8 }) |count| for ([_][8]c_int{ @splat(2), @splat(3), @splat(4), @splat(5), @splat(6), @splat(7), .{ 5, 3, 2, 4, 2, 2, 3, 3 } }) |lengths| {
            var total: c_int = 0;
            for (lengths[0..count]) |width| total += width;
            const x = try attn256RandBf16(prng.random(), &.{ 1, total, k }, s);
            defer _ = mlx.mlx_array_free(x);
            const arch = verifyArch().?;
            if (total > 24 or total >= qmvBatchLimit(dualDieArch(arch), k, n)) {
                try testing.expect((try verifySharedProjection(s, x, w, sc, bi, lengths[0..count])) == null);
                continue;
            }
            const grouped = (try verifySharedProjection(s, x, w, sc, bi, lengths[0..count])) orelse return error.SharedProjectionDeclined;
            defer _ = mlx.mlx_array_free(grouped);
            var offset: c_int = 0;
            for (lengths[0..count]) |width| {
                var row = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(row);
                try mlx.check(mlx.mlx_slice(&row, x, &.{ 0, offset, 0 }, 3, &.{ 1, offset + width, k }, 3, &.{ 1, 1, 1 }, 3, s));
                const expected = try qmatmulBits(row, w, sc, bi, 8, gs, .affine, s);
                defer _ = mlx.mlx_array_free(expected);
                var actual = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(actual);
                try mlx.check(mlx.mlx_slice(&actual, grouped, &.{ 0, offset, 0 }, 3, &.{ 1, offset + width, n }, 3, &.{ 1, 1, 1 }, 3, s));
                try testing.expect(try qsaArraysAllEqual(expected, actual, s));
                offset += width;
            }
        };
        const x = try attn256RandBf16(prng.random(), &.{ 1, 8, k }, s);
        defer _ = mlx.mlx_array_free(x);
        for ([_][]const c_int{ &.{ 1, 7 }, &.{8}, &.{ 2, 2 }, &.{ 0, 8 } }) |widths| {
            try testing.expect((try verifySharedProjection(s, x, w, sc, bi, widths)) == null);
        }
        try testing.expect((try verifySharedProjection(s, x, w, sc, .{}, &.{ 4, 4 })) == null);
        const wide = try attn256RandBf16(prng.random(), &.{ 1, 25, k }, s);
        defer _ = mlx.mlx_array_free(wide);
        try testing.expect((try verifySharedProjection(s, wide, w, sc, bi, &.{ 5, 5, 5, 5, 5 })) == null);
        const old = verify_qmm.vqmm_mixed_plain_test_force;
        verify_qmm.vqmm_mixed_plain_test_force = true;
        defer verify_qmm.vqmm_mixed_plain_test_force = old;
        try testing.expect((try verifySharedProjection(s, x, w, sc, bi, &.{ 4, 4 })) == null);
    };
}

test "verify GDN projections preserve solo qmv values across request widths" {
    if (mlx.noGpuBackend() or !verifySharedHardware()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x5A4E);
    for ([_][2]c_int{ .{ 2560, 10240 }, .{ 2560, 6144 }, .{ 2560, 48 }, .{ 6144, 2560 } }) |shape| {
        const k = shape[0];
        const n = shape[1];
        const weight = try attn256RandBf16(prng.random(), &.{ n, k }, s);
        defer _ = mlx.mlx_array_free(weight);
        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(&triple, weight, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(8), "affine", .{}, s));
        var w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w);
        var sc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc);
        var bi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bi);
        try mlx.check(mlx.mlx_vector_array_get(&w, triple, 0));
        try mlx.check(mlx.mlx_vector_array_get(&sc, triple, 1));
        try mlx.check(mlx.mlx_vector_array_get(&bi, triple, 2));
        for ([_]usize{ 2, 3, 4, 8 }) |count| for ([_][8]c_int{ @splat(2), @splat(3), @splat(4), @splat(5), @splat(6), @splat(7), .{ 5, 3, 2, 4, 2, 2, 3, 3 } }) |lengths| {
            var total: c_int = 0;
            for (lengths[0..count]) |width| total += width;
            const x = try attn256RandBf16(prng.random(), &.{ 1, total, k }, s);
            defer _ = mlx.mlx_array_free(x);
            if (total > 12 or lengths[0] > 6) {
                try testing.expect((try verifyGdnProjection(s, x, w, sc, bi, lengths[0..count])) == null);
                continue;
            }
            const grouped = (try verifyGdnProjection(s, x, w, sc, bi, lengths[0..count])) orelse return error.GdnProjectionDeclined;
            defer _ = mlx.mlx_array_free(grouped);
            var offset: c_int = 0;
            for (lengths[0..count]) |width| {
                var row = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(row);
                try mlx.check(mlx.mlx_slice(&row, x, &.{ 0, offset, 0 }, 3, &.{ 1, offset + width, k }, 3, &.{ 1, 1, 1 }, 3, s));
                const expected = try qmatmulBits(row, w, sc, bi, 8, 64, .affine, s);
                defer _ = mlx.mlx_array_free(expected);
                var actual = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(actual);
                try mlx.check(mlx.mlx_slice(&actual, grouped, &.{ 0, offset, 0 }, 3, &.{ 1, offset + width, n }, 3, &.{ 1, 1, 1 }, 3, s));
                try testing.expect(try qsaArraysAllEqual(expected, actual, s));
                offset += width;
            }
        };
        const x = try attn256RandBf16(prng.random(), &.{ 1, 8, k }, s);
        defer _ = mlx.mlx_array_free(x);
        for ([_][]const c_int{ &.{ 1, 7 }, &.{8}, &.{ 2, 2 }, &.{ 0, 8 } }) |widths| {
            try testing.expect((try verifyGdnProjection(s, x, w, sc, bi, widths)) == null);
        }
        try testing.expect((try verifyGdnProjection(s, x, w, sc, .{}, &.{ 4, 4 })) == null);
        const wide = try attn256RandBf16(prng.random(), &.{ 1, 25, k }, s);
        defer _ = mlx.mlx_array_free(wide);
        try testing.expect((try verifyGdnProjection(s, wide, w, sc, bi, &.{ 5, 5, 5, 5, 5 })) == null);
        const old = verify_qmm.vqmm_mixed_plain_test_force;
        verify_qmm.vqmm_mixed_plain_test_force = true;
        defer verify_qmm.vqmm_mixed_plain_test_force = old;
        try testing.expect((try verifyGdnProjection(s, x, w, sc, bi, &.{ 4, 4 })) == null);
    }
}

test "verify attention projections preserve solo qmv values across request widths" {
    if (mlx.noGpuBackend() or !verifySharedHardware()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x5A4E);
    for ([_][2]c_int{ .{ 2560, 12288 }, .{ 2560, 512 }, .{ 6144, 2560 } }) |shape| {
        const k = shape[0];
        const n = shape[1];
        const weight = try attn256RandBf16(prng.random(), &.{ n, k }, s);
        defer _ = mlx.mlx_array_free(weight);
        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(&triple, weight, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(8), "affine", .{}, s));
        var w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w);
        var sc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc);
        var bi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bi);
        try mlx.check(mlx.mlx_vector_array_get(&w, triple, 0));
        try mlx.check(mlx.mlx_vector_array_get(&sc, triple, 1));
        try mlx.check(mlx.mlx_vector_array_get(&bi, triple, 2));
        for ([_]usize{ 2, 3, 4, 8 }) |count| for ([_][8]c_int{ @splat(2), @splat(3), @splat(4), @splat(5), @splat(6), @splat(7), .{ 5, 3, 2, 4, 2, 2, 3, 3 } }) |lengths| {
            var total: c_int = 0;
            for (lengths[0..count]) |width| total += width;
            const x = try attn256RandBf16(prng.random(), &.{ 1, total, k }, s);
            defer _ = mlx.mlx_array_free(x);
            if (total > 12 or lengths[0] > 6) {
                try testing.expect((try verifyAttnProjection(s, x, w, sc, bi, lengths[0..count])) == null);
                continue;
            }
            const grouped = (try verifyAttnProjection(s, x, w, sc, bi, lengths[0..count])) orelse return error.AttnProjectionDeclined;
            defer _ = mlx.mlx_array_free(grouped);
            var offset: c_int = 0;
            for (lengths[0..count]) |width| {
                var row = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(row);
                try mlx.check(mlx.mlx_slice(&row, x, &.{ 0, offset, 0 }, 3, &.{ 1, offset + width, k }, 3, &.{ 1, 1, 1 }, 3, s));
                const expected = try qmatmulBits(row, w, sc, bi, 8, 64, .affine, s);
                defer _ = mlx.mlx_array_free(expected);
                var actual = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(actual);
                try mlx.check(mlx.mlx_slice(&actual, grouped, &.{ 0, offset, 0 }, 3, &.{ 1, offset + width, n }, 3, &.{ 1, 1, 1 }, 3, s));
                try testing.expect(try qsaArraysAllEqual(expected, actual, s));
                offset += width;
            }
        };
        const x = try attn256RandBf16(prng.random(), &.{ 1, 8, k }, s);
        defer _ = mlx.mlx_array_free(x);
        for ([_][]const c_int{ &.{ 1, 7 }, &.{8}, &.{ 2, 2 }, &.{ 0, 8 } }) |widths| {
            try testing.expect((try verifyAttnProjection(s, x, w, sc, bi, widths)) == null);
        }
        try testing.expect((try verifyAttnProjection(s, x, w, sc, .{}, &.{ 4, 4 })) == null);
        const wide = try attn256RandBf16(prng.random(), &.{ 1, 25, k }, s);
        defer _ = mlx.mlx_array_free(wide);
        try testing.expect((try verifyAttnProjection(s, wide, w, sc, bi, &.{ 5, 5, 5, 5, 5 })) == null);
        const old = verify_qmm.vqmm_mixed_plain_test_force;
        verify_qmm.vqmm_mixed_plain_test_force = true;
        defer verify_qmm.vqmm_mixed_plain_test_force = old;
        try testing.expect((try verifyAttnProjection(s, x, w, sc, bi, &.{ 4, 4 })) == null);
    }
}

test "verify vocabulary projections preserve solo logits across request widths (MTP_LMHEAD_TEST_FILE)" {
    if (mlx.noGpuBackend() or !verifySharedHardware()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x5A4E);
    const path = env.get(.MTP_LMHEAD_TEST_FILE) orelse return error.SkipZigTest;
    var weights = try model_mod.loadWeightsSingleFile(testing.allocator, std.mem.span(path));
    defer weights.deinit();
    const w = weights.get("language_model.lm_head.weight") orelse return error.MissingLmHead;
    const sc = weights.get("language_model.lm_head.scales") orelse return error.MissingLmHead;
    const bi = weights.get("language_model.lm_head.biases") orelse return error.MissingLmHead;
    const k: c_int = 2560;
    const n: c_int = 248320;
    {
        for ([_]usize{ 2, 3, 4, 8 }) |count| for ([_][8]c_int{ @splat(2), @splat(3), @splat(4), @splat(5), @splat(6), @splat(7), .{ 5, 3, 2, 4, 2, 2, 3, 3 } }) |lengths| {
            var total: c_int = 0;
            for (lengths[0..count]) |width| total += width;
            const x = try attn256RandBf16(prng.random(), &.{ 1, total, k }, s);
            defer _ = mlx.mlx_array_free(x);
            if (total > 12 or lengths[0] > 6) {
                try testing.expect((try verifyLmHeadProjection(s, x, w, sc, bi, lengths[0..count])) == null);
                continue;
            }
            const grouped = (try verifyLmHeadProjection(s, x, w, sc, bi, lengths[0..count])) orelse return error.LmHeadProjectionDeclined;
            defer _ = mlx.mlx_array_free(grouped);
            var offset: c_int = 0;
            for (lengths[0..count]) |width| {
                var row = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(row);
                try mlx.check(mlx.mlx_slice(&row, x, &.{ 0, offset, 0 }, 3, &.{ 1, offset + width, k }, 3, &.{ 1, 1, 1 }, 3, s));
                const expected = try qmatmulBits(row, w, sc, bi, 8, 64, .affine, s);
                defer _ = mlx.mlx_array_free(expected);
                var actual = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(actual);
                try mlx.check(mlx.mlx_slice(&actual, grouped, &.{ 0, offset, 0 }, 3, &.{ 1, offset + width, n }, 3, &.{ 1, 1, 1 }, 3, s));
                try testing.expect(try qsaArraysAllEqual(expected, actual, s));
                offset += width;
            }
        };
        const x = try attn256RandBf16(prng.random(), &.{ 1, 8, k }, s);
        defer _ = mlx.mlx_array_free(x);
        for ([_][]const c_int{ &.{ 1, 7 }, &.{8}, &.{ 2, 2 }, &.{ 0, 8 } }) |widths| {
            try testing.expect((try verifyLmHeadProjection(s, x, w, sc, bi, widths)) == null);
        }
        try testing.expect((try verifyLmHeadProjection(s, x, w, sc, .{}, &.{ 4, 4 })) == null);
        const wide = try attn256RandBf16(prng.random(), &.{ 1, 25, k }, s);
        defer _ = mlx.mlx_array_free(wide);
        try testing.expect((try verifyLmHeadProjection(s, wide, w, sc, bi, &.{ 5, 5, 5, 5, 5 })) == null);
        const old = verify_qmm.vqmm_mixed_plain_test_force;
        verify_qmm.vqmm_mixed_plain_test_force = true;
        defer verify_qmm.vqmm_mixed_plain_test_force = old;
        try testing.expect((try verifyLmHeadProjection(s, x, w, sc, bi, &.{ 4, 4 })) == null);
    }
}

test "the wide dense verify tile preserves the stock QMV reduction" {
    if (mlx.noGpuBackend() or !verifySharedHardware()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x719E);
    for ([_][2]c_int{ .{ 2560, 10240 }, .{ 2560, 6144 }, .{ 6144, 2560 }, .{ 640, 2560 } }) |shape| {
        const k = shape[0];
        const n = shape[1];
        const weights = try attn256RandBf16(prng.random(), &.{ n, k }, s);
        defer _ = mlx.mlx_array_free(weights);
        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(&triple, weights, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(8), "affine", .{}, s));
        var w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w);
        var sc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc);
        var bi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bi);
        try mlx.check(mlx.mlx_vector_array_get(&w, triple, 0));
        try mlx.check(mlx.mlx_vector_array_get(&sc, triple, 1));
        try mlx.check(mlx.mlx_vector_array_get(&bi, triple, 2));
        for ([_]c_int{ 6, 8, 12 }) |m| {
            const x = try attn256RandBf16(prng.random(), &.{ 1, m, k }, s);
            defer _ = mlx.mlx_array_free(x);
            const cpu = mlx.mlx_default_cpu_stream_new();
            defer _ = mlx.mlx_stream_free(cpu);
            try testing.expect((try verifyWideProjection(cpu, x, w, sc, bi, 6)) == null);
            try testing.expect((try verifyWideProjection(s, .{}, w, sc, bi, 6)) == null);
            try testing.expect((try verifyWideProjection(s, x, w, sc, .{}, 6)) == null);
            try testing.expect((try verifyWideProjection(s, x, w, sc, bi, 7)) == null);
            var reference = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(reference);
            try mlx.check(mlx.mlx_quantized_matmul(&reference, x, w, sc, bi, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(8), "affine", s));
            for ([_]c_int{ 4, 6, 8, 12 }) |nv| {
                const got = (try verifyWideProjection(s, x, w, sc, bi, nv)) orelse return error.WideTileDeclined;
                defer _ = mlx.mlx_array_free(got);
                try testing.expect(try qsaArraysAllEqual(reference, got, s));
            }
        }
    }
}

test "MTP coarse pairs preserve row-kernel logits (MTP_LMHEAD_TEST_FILE)" {
    const path = env.get(.MTP_LMHEAD_TEST_FILE) orelse return error.SkipZigTest;
    if (mlx.noGpuBackend() or !verifySharedHardware()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var weights = try model_mod.loadWeightsSingleFile(testing.allocator, std.mem.span(path));
    defer weights.deinit();
    const w = weights.get("language_model.lm_head.weight") orelse return error.MissingLmHead;
    const sc = weights.get("language_model.lm_head.scales") orelse return error.MissingLmHead;
    const bi = weights.get("language_model.lm_head.biases") orelse return error.MissingLmHead;
    var coarse = try mtp_mod.requantizeRows(s, w, sc, bi, 64, 8, "affine", 64, 3, 32768);
    defer coarse.deinit();
    var prng = std.Random.DefaultPrng.init(0xC0A45E);
    for ([_]c_int{ 2, 3, 4, 8 }) |rows| {
        const x = try attn256RandBf16(prng.random(), &.{ rows, 1, 2560 }, s);
        defer _ = mlx.mlx_array_free(x);
        const reference = (try msvQmvRows(s, x, coarse.w, coarse.s, coarse.b, 3, 64)) orelse return error.HeadRowsDeclined;
        defer _ = mlx.mlx_array_free(reference);
        const actual = (try mtpCoarsePairs(s, x, coarse.w, coarse.s, coarse.b, 3, 64)) orelse return error.CoarsePairsDeclined;
        defer _ = mlx.mlx_array_free(actual);
        try testing.expect(try qsaArraysAllEqual(reference, actual, s));
        if (rows == 4) {
            try testing.expect((try mtpCoarsePairs(s, x, coarse.w, coarse.s, .{}, 3, 64)) == null);
            try testing.expect((try mtpCoarsePairs(s, x, coarse.w, coarse.s, coarse.b, 2, 64)) == null);
            const cpu = mlx.mlx_default_cpu_stream_new();
            defer _ = mlx.mlx_stream_free(cpu);
            try testing.expect((try mtpCoarsePairs(cpu, x, coarse.w, coarse.s, coarse.b, 3, 64)) == null);
            var wide = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wide);
            try mlx.check(mlx.mlx_reshape(&wide, x, &.{ 2, 2, 2560 }, 3, s));
            try testing.expect((try mtpCoarsePairs(s, wide, coarse.w, coarse.s, coarse.b, 3, 64)) == null);
            var x32 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x32);
            try mlx.check(mlx.mlx_astype(&x32, x, .float32, s));
            try testing.expect((try mtpCoarsePairs(s, x32, coarse.w, coarse.s, coarse.b, 3, 64)) == null);
        }
    }
}
