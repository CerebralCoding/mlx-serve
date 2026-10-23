//! Exact top-k block selection (`msv_qsa_select`: radix select, single, sliced and merged setups) and its 16-way split
//! at decode.

const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../host/log.zig");
const mtp = @import("../mtp.zig");
const testing = @import("../../../engines/mlx/kv/cache.zig").testing;
const Transformer = @import("../../../engines/mlx/transformer.zig").Transformer;
const QsaCfgCache = @import("../../../mlx/kernel_cache.zig").QsaCfgCache;
const attn256MaxDiff = @import("../../../sdk/blocks/attention/attn256.zig").attn256MaxDiff;
const OneShotBits = @import("arms.zig").OneShotBits;
const qsaWidthBucket = @import("arms.zig").qsaWidthBucket;
const headPlaceScores = @import("mask.zig").headPlaceScores;
const headPlaceVisibility = @import("mask.zig").headPlaceVisibility;
const qsaAllBlocksVisible = @import("pooled.zig").qsaAllBlocksVisible;
const qsaSelectComposedOps = @import("score.zig").qsaSelectComposedOps;

// ── Fused QSA top-k block select (msv_qsa_select) ──
//
// The composed arm is ~10 dependent dispatches per attention layer (~0.31 ms/layer at kv
// 62.7k); this kernel is one dispatch: one threadgroup per query row, an MSD radix select
// over the f32 score's monotone ordinal, then an index-ordered compaction. Semantics are
// `qsaChunkSelect`'s: top-K visible blocks per row, ties to the LOWER index (torch.topk),
// emitted ascending, INT_MAX past the row's count. The tie rule is exact where the composed
// arm's `b * 1e-7` bias only approximates it.
pub const QSA_SELECT_KERNEL_HEADER = kernelSource("plugins/qwen4_exp/kernels/qsa_select_header.metal");

/// One threadgroup per query row. 1. MSD radix select over the 32-bit ordinal (the DIGITS
/// template picks 11/11/10-bit digits at prefill widths or four 8-bit digits at decode widths,
/// where each level's fixed cost, not the walk, is the bill) counting only visible blocks,
/// stopping early when a bin's count is exactly what is still needed. 2. Compaction: selected
/// iff `ord > T`, or `ord == T` with rank below `need_eq`; the output slot is the count of
/// selected elements with a smaller index (a simd prefix scan per chunk), so the output is
/// born sorted. 3. Rows with `bounds[row] <= K` are `0 .. bounds-1` then INT_MAX.
/// Threadgroup memory ~8.3 KiB at 2048 bins, ~1.3 KiB at 256.
const QSA_SELECT_SHARED = kernelSource("plugins/qwen4_exp/kernels/qsa_select_shared.metal");

// One radix-select body composed into three kernels (single row, row slice, merge) so a
// fix to the digit walk or the tie rule lands once; the kernels differ only in setup.
const QSA_SELECT_RADIX_BODY = kernelSource("plugins/qwen4_exp/kernels/qsa_select_radix_body.metal");

const QSA_SELECT_SINGLE_SETUP = kernelSource("plugins/qwen4_exp/kernels/qsa_select_single_setup.metal");

const QSA_SELECT_SLICE_SETUP = kernelSource("plugins/qwen4_exp/kernels/qsa_select_slice_setup.metal");

// Candidates arrive ascending by ORIGINAL index with SENTINEL slots skipped, never sorted
// in; that is what resolves a cross-slice tie at the K-th ordinal to the lowest index. Every
// simd reduction below runs in a uniform base-stepped loop: a divergent simd_sum is undefined.
const QSA_SELECT_MERGE_SETUP = kernelSource("plugins/qwen4_exp/kernels/qsa_select_merge_setup.metal");

pub const QSA_SELECT_KERNEL_SOURCE = std.fmt.comptimePrint("{s}{s}{s}", .{ QSA_SELECT_SHARED, QSA_SELECT_SINGLE_SETUP, QSA_SELECT_RADIX_BODY });
pub const QSA_SELECT_SPLIT_LOCAL_SOURCE = std.fmt.comptimePrint("{s}{s}{s}", .{ QSA_SELECT_SHARED, QSA_SELECT_SLICE_SETUP, QSA_SELECT_RADIX_BODY });
pub const QSA_SELECT_SPLIT_MERGE_SOURCE = std.fmt.comptimePrint("{s}{s}{s}", .{ QSA_SELECT_SHARED, QSA_SELECT_MERGE_SETUP, QSA_SELECT_RADIX_BODY });

/// Threads per row-threadgroup.
pub const QSA_SELECT_TG_DEFAULT: c_int = 1024;
pub var qsa_select_tg_override: ?c_int = null;

// A row's threadgroup is a per-shape choice: the wide passes of a prefill chunk run more
// rows concurrently at 256 threads, a narrow pass wants 512, and the decode widths keep
// 1024 (the split kernels size their own slices). Outputs are identical at every width.
pub fn qsaSelectTgFor(rows: c_int) c_int {
    if (qsa_select_tg_override) |forced| return forced;
    if (rows >= 256) return 256;
    if (rows >= 64) return 512;
    return QSA_SELECT_TG_DEFAULT;
}

var qsa_select_kernel_cached: ?mlx.mlx_fast_metal_kernel = null;

fn getQsaSelectKernel() !mlx.mlx_fast_metal_kernel {
    if (qsa_select_kernel_cached) |kk| return kk;
    const input_names = [_][*:0]const u8{ "scores", "bounds" };
    const output_names = [_][*:0]const u8{"ids"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "msv_qsa_select",
        in_vec,
        out_vec,
        QSA_SELECT_KERNEL_SOURCE,
        QSA_SELECT_KERNEL_HEADER,
        true, // the row walk indexes linearly; let mlx materialize a view
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    qsa_select_kernel_cached = kernel;
    return kernel;
}

/// Keyed on what the config reads: output `[1, rows, kb]`, grid `(tg, rows, 1)`, templates
/// TGS and K. `nb` is a runtime shape (it advances every `ratio` decode tokens).
const QsaSelectCfgKey = struct { rows: c_int, k: c_int, tg: c_int, digits: c_int };
var qsa_select_cfgs = QsaCfgCache(QsaSelectCfgKey, 1){};

var qsa_select_engaged_bits: OneShotBits = .{};

pub var qsa_select_kernel_override: ?bool = null;

/// Fused QSA block select; the test seam's false runs the argpartition chain.
pub fn qsaSelectKernelEnabled() bool {
    return qsa_select_kernel_override orelse true;
}

/// Per-row first-not-fully-visible block index (`b < (p+1)/ratio`). `all_vis` is the caller's
/// own `qsaAllBlocksVisible` claim, honored verbatim so both arms answer the same question.
pub fn qsaVisibleBoundsHost(out: []i32, row0: c_int, rows: c_int, nb: c_int, ratio: c_int, all_vis: bool) void {
    const n: usize = @intCast(rows);
    var r: usize = 0;
    while (r < n) : (r += 1) {
        if (all_vis) {
            out[r] = nb;
            continue;
        }
        const p: c_int = row0 + @as(c_int, @intCast(r));
        const complete: c_int = if (p < 0) 0 else @divTrunc(p + 1, ratio);
        out[r] = @min(nb, complete);
    }
}

/// Fused replacement for the composed selection tail: f32 `[1, rows, nb]` scores + int32
/// `[rows]` bounds -> sorted int32 `[1, rows, kb]` selection. Null = declined.
pub fn qsaSelectTopBlocks(
    s: mlx.mlx_stream,
    scores: mlx.mlx_array,
    bounds: mlx.mlx_array,
    kb: c_int,
) !?mlx.mlx_array {
    if (!qsaSelectKernelEnabled()) return null;
    if (!mlx.streamIsGpu(s)) return null;
    if (scores.ctx == null or bounds.ctx == null) return null;
    if (mlx.mlx_array_dtype(scores) != .float32 or mlx.mlx_array_dtype(bounds) != .int32) return null;
    if (mlx.mlx_array_ndim(scores) != 3 or mlx.mlx_array_ndim(bounds) != 1) return null;
    const ssh = mlx.getShape(scores);
    const bsh = mlx.getShape(bounds);
    if (ssh[0] != 1) return null;
    const rows = ssh[1];
    const nb = ssh[2];
    if (rows <= 0 or nb <= 0 or bsh[0] != rows) return null;
    if (kb <= 0 or kb > nb) return null;

    qsa_select_used_split = false;
    if (qsaSelectSplitEnabled() and qsaSelectSplitServes(rows, nb, kb)) {
        if (try qsaSelectTopBlocksSplit(s, scores, bounds, kb, rows, nb)) |picks| {
            qsa_select_used_split = true;
            return picks;
        }
    }

    const kernel = getQsaSelectKernel() catch return null;
    const tg = qsaSelectTgFor(rows);
    const digits = qsaSelectDigitBits(rows);
    const key = QsaSelectCfgKey{ .rows = rows, .k = kb, .tg = tg, .digits = digits };
    const sel_cfgs: [1]mlx.mlx_fast_metal_kernel_config = qsa_select_cfgs.get(key) orelse blk: {
        const config = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
        const o_shape = [_]c_int{ 1, rows, kb };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &o_shape, 3, .int32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, tg, rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, tg, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "TGS", tg));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "K", kb));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "DIGITS", digits));
        qsa_select_cfgs.put(key, .{config});
        break :blk .{config};
    };

    const inputs_arr = [_]mlx.mlx_array{ scores, bounds };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, sel_cfgs[0], s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    mtp.qsa_index_select_graphs += 1;
    if (qsa_select_engaged_bits.take(qsaWidthBucket(rows))) {
        log.info(
            "[qsa-select] engaged (S={d} nb={d} k={d} tg={d})\n",
            .{ rows, nb, kb, tg },
        );
    }
    return out;
}

// One threadgroup per row is latency-bound at decode widths (0.72 ms at 215k blocks; ~20x
// under the bandwidth floor), so rows split over G slices, each running the exact select,
// and a second dispatch selects over the G*K candidates. Exact because the order is total
// (ordinal, then lowest original index): every global top-K element is in its slice's
// top-K. The floor is the crossover of the marginal in-graph cost (probe: the split pair
// beats the single kernel from ~24k blocks once the compaction scan and 8-bit digits landed;
// live, +4% at 40k blocks and +18% serial at 200k); 2*G*K <= nb (16k blocks at G=16) keeps
// the candidate list smaller than the row it replaces and is the design's hard minimum.
pub const QSA_SELECT_SPLIT_MIN_NB: c_int = 24576;
pub const QSA_SELECT_SPLIT_MAX_ROWS: c_int = 15;
pub const QSA_SELECT_SPLIT_G: c_int = 16;

pub var qsa_select_digits_override: ?c_int = null;

pub fn qsaSelectDigitBits(rows: c_int) c_int {
    if (qsa_select_digits_override) |v| return v;
    if (rows <= QSA_SELECT_SPLIT_MAX_ROWS) return 8;
    return 11;
}

pub var qsa_select_split_override: ?bool = null;
pub var qsa_select_used_split: bool = false;

pub fn qsaSelectSplitEnabled() bool {
    return qsa_select_split_override orelse true;
}

pub fn qsaSelectSplitServes(rows: c_int, nb: c_int, kb: c_int) bool {
    if (rows < 1 or rows > QSA_SELECT_SPLIT_MAX_ROWS) return false;
    if (nb < QSA_SELECT_SPLIT_MIN_NB) return false;
    if (kb <= 0 or kb > nb) return false;
    const span = std.math.mul(c_int, kb, QSA_SELECT_SPLIT_G) catch return false;
    const twice = std.math.mul(c_int, span, 2) catch return false;
    return twice <= nb;
}

var qsa_select_split_local_kernel_cached: ?mlx.mlx_fast_metal_kernel = null;
var qsa_select_split_merge_kernel_cached: ?mlx.mlx_fast_metal_kernel = null;

fn getQsaSelectSplitLocalKernel() !mlx.mlx_fast_metal_kernel {
    if (qsa_select_split_local_kernel_cached) |kk| return kk;
    const input_names = [_][*:0]const u8{ "scores", "bounds" };
    const output_names = [_][*:0]const u8{"ids"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "msv_qsa_select_split_local",
        in_vec,
        out_vec,
        QSA_SELECT_SPLIT_LOCAL_SOURCE,
        QSA_SELECT_KERNEL_HEADER,
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    qsa_select_split_local_kernel_cached = kernel;
    return kernel;
}

fn getQsaSelectSplitMergeKernel() !mlx.mlx_fast_metal_kernel {
    if (qsa_select_split_merge_kernel_cached) |kk| return kk;
    const input_names = [_][*:0]const u8{ "scores", "local_ids" };
    const output_names = [_][*:0]const u8{"ids"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "msv_qsa_select_split_merge",
        in_vec,
        out_vec,
        QSA_SELECT_SPLIT_MERGE_SOURCE,
        QSA_SELECT_KERNEL_HEADER,
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    qsa_select_split_merge_kernel_cached = kernel;
    return kernel;
}

const QsaSelectSplitCfgKey = struct { rows: c_int, k: c_int, slice_tg: c_int, merge_tg: c_int, digits: c_int };
var qsa_select_split_cfgs = QsaCfgCache(QsaSelectSplitCfgKey, 2){};

var qsa_select_split_engaged_bits: OneShotBits = .{};

fn qsaSelectTopBlocksSplit(
    s: mlx.mlx_stream,
    scores: mlx.mlx_array,
    bounds: mlx.mlx_array,
    kb: c_int,
    rows: c_int,
    nb: c_int,
) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return null;
    const g = QSA_SELECT_SPLIT_G;
    const loc_count = std.math.mul(c_int, kb, g) catch return null;
    const local_kernel = getQsaSelectSplitLocalKernel() catch return null;
    const merge_kernel = getQsaSelectSplitMergeKernel() catch return null;
    const slice_tg: c_int = qsa_select_tg_override orelse 512;
    const merge_tg: c_int = qsa_select_tg_override orelse QSA_SELECT_TG_DEFAULT;
    const digits = qsaSelectDigitBits(rows);
    const key = QsaSelectSplitCfgKey{ .rows = rows, .k = kb, .slice_tg = slice_tg, .merge_tg = merge_tg, .digits = digits };
    const cfgs: [2]mlx.mlx_fast_metal_kernel_config = qsa_select_split_cfgs.get(key) orelse blk: {
        const local_cfg = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(local_cfg);
        const local_shape = [_]c_int{ 1, rows, loc_count };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(local_cfg, &local_shape, 3, .int32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(local_cfg, slice_tg * g, rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(local_cfg, slice_tg, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(local_cfg, "TGS", slice_tg));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(local_cfg, "K", kb));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(local_cfg, "DIGITS", digits));
        const merge_cfg = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(merge_cfg);
        const o_shape = [_]c_int{ 1, rows, kb };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(merge_cfg, &o_shape, 3, .int32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(merge_cfg, merge_tg, rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(merge_cfg, merge_tg, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(merge_cfg, "TGS", merge_tg));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(merge_cfg, "K", kb));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(merge_cfg, "DIGITS", digits));
        qsa_select_split_cfgs.put(key, .{ local_cfg, merge_cfg });
        break :blk .{ local_cfg, merge_cfg };
    };

    const local_in = [_]mlx.mlx_array{ scores, bounds };
    const local_in_vec = mlx.mlx_vector_array_new_data(&local_in, local_in.len);
    defer _ = mlx.mlx_vector_array_free(local_in_vec);
    var local_out_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(local_out_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&local_out_vec, local_kernel, local_in_vec, cfgs[0], s));
    if (mlx.mlx_vector_array_size(local_out_vec) != 1) return error.MetalKernelBadOutputCount;
    var local_ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(local_ids);
    try mlx.check(mlx.mlx_vector_array_get(&local_ids, local_out_vec, 0));

    const merge_in = [_]mlx.mlx_array{ scores, local_ids };
    const merge_in_vec = mlx.mlx_vector_array_new_data(&merge_in, merge_in.len);
    defer _ = mlx.mlx_vector_array_free(merge_in_vec);
    var merge_out_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(merge_out_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&merge_out_vec, merge_kernel, merge_in_vec, cfgs[1], s));
    if (mlx.mlx_vector_array_size(merge_out_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&out, merge_out_vec, 0));
    if (qsa_select_split_engaged_bits.take(qsaWidthBucket(rows))) {
        log.info(
            "[qsa-select] engaged (split G={d} S={d} nb={d} k={d} tg={d})\n",
            .{ g, rows, nb, kb, slice_tg },
        );
    }
    return out;
}

pub fn qsaArraysAllEqual(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !bool {
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_equal(&eq, a, b, s));
    var all = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(all);
    try mlx.check(mlx.mlx_all(&all, eq, false, s));
    try mlx.check(mlx.mlx_array_eval(all));
    var out: bool = false;
    try mlx.check(mlx.mlx_array_item_bool(&out, all));
    return out;
}

// ── tests ──

pub fn qsaSelectCfgBuilds() usize {
    return qsa_select_cfgs.builds;
}

pub fn qsaSelectSplitCfgBuilds() usize {
    return qsa_select_split_cfgs.builds;
}

test "head place: blocks reaching a dead row never score, and only dead rows are hidden" {
    // Bar: live_end 9, dead_end 12, ratio 4: blocks 0..1 keep their scores, 2.. are -inf; rows 9..11 hidden, 8 recent.
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const t = std.testing;
    const s = mlx.gpuStream();
    var place: Transformer.HeadPlace = .{ .live_end = mlx.mlx_array_new_int(9), .dead_end = 12 };
    defer _ = mlx.mlx_array_free(place.live_end);

    const raw = [_]f32{ 1, 2, 30, 40 };
    const scores = mlx.mlx_array_new_data(&raw, &[_]c_int{ 1, 1, 4 }, 3, .float32);
    defer _ = mlx.mlx_array_free(scores);
    const got = try headPlaceScores(s, scores, &place, 4, 4);
    defer _ = mlx.mlx_array_free(got);
    try mlx.check(mlx.mlx_array_eval(got));
    const g = mlx.mlx_array_data_float32(got).?[0..4];
    try t.expectEqual(@as(f32, 1), g[0]);
    try t.expectEqual(@as(f32, 2), g[1]);
    try t.expect(std.math.isNegativeInf(g[2]) and std.math.isNegativeInf(g[3]));

    // Selection picks block 0 only; rows 8.. are the recent run, 9..11 dead, 12..13 drafts.
    var sel_raw: [14]bool = @splat(false);
    for (0..4) |i| sel_raw[i] = true;
    const sel = mlx.mlx_array_new_data(&sel_raw, &[_]c_int{ 1, 1, 1, 14 }, 4, .bool_);
    defer _ = mlx.mlx_array_free(sel);
    const want = [_]bool{ true, true, true, true, false, false, false, false, true, false, false, false, true, true };
    const vis = try headPlaceVisibility(s, sel, &place, 4, 14);
    defer _ = mlx.mlx_array_free(vis);
    try mlx.check(mlx.mlx_array_eval(vis));
    try t.expectEqualSlices(bool, &want, mlx.mlx_array_data_bool(vis).?[0..14]);

    const dense = try headPlaceVisibility(s, .{ .ctx = null }, &place, 4, 14);
    defer _ = mlx.mlx_array_free(dense);
    try mlx.check(mlx.mlx_array_eval(dense));
    const want_dense = [_]bool{ true, true, true, true, true, true, true, true, true, false, false, false, true, true };
    try t.expectEqualSlices(bool, &want_dense, mlx.mlx_array_data_bool(dense).?[0..14]);
}

// ── The QSA all-visible identity skip at the engagement boundary ──

/// Random f32 score sheet (the reference's own dtype).
fn qsaRandScoresF32(rnd: std.Random, shape: []const c_int) !mlx.mlx_array {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const data = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(data);
    // relu output: many exact zeros.
    for (data) |*x| {
        const v = rnd.float(f32) - 0.35;
        x.* = if (v > 0) v else 0.0;
    }
    return mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .float32);
}

test "qsa mask: the all-visible skip must not hand a NULL sheet to an mlx op (kv just past the budget)" {
    // `mlx_logical_and(&blk_sel, picked, vis3)` in the nb > block_topk arm passed the null
    // all-visible sheet straight to mlx-c and killed the process. At decode width all-visible
    // is always true, so every generated token past kv = (block_topk + 1) * ratio crashed.
    // Bar: where the predicate claims all-visible, the real sheet is all-true and the null-sheet
    // arm produces the same mask.
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ratio: c_int = 4;
    const batch: c_int = 1;
    var prng = std.Random.DefaultPrng.init(0x2068_B0DE);
    const rnd = prng.random();

    var checked_skip: usize = 0;
    // 2049..2051 sit just below the selection boundary; 2052 is the first kv that can crash.
    for ([_]c_int{ 2049, 2052, 2068, 2100, 4100 }) |kv| {
        // `nb` is the complete-block count, floor(kv/ratio).
        const nb = @divTrunc(kv, ratio);
        for ([_]c_int{ 1, 31, 1024 }) |seq_len| {
            if (seq_len > kv) continue;
            const offset = kv - seq_len;
            var topk_buf: [11]c_int = undefined;
            topk_buf[0] = nb;
            topk_buf[1] = nb + 4;
            for (1..9) |d| topk_buf[1 + d] = nb - @as(c_int, @intCast(d));
            topk_buf[10] = 512;

            for (topk_buf) |block_topk| {
                if (block_topk <= 0) continue;
                const scores = try qsaRandScoresF32(rnd, &[_]c_int{ batch, seq_len, nb });
                defer _ = mlx.mlx_array_free(scores);
                const tie = try qsaTieBiasForTest(s, nb);
                defer _ = mlx.mlx_array_free(tie);
                const sheet = try qsaVisSheetForTest(s, offset, seq_len, nb, ratio);
                defer _ = mlx.mlx_array_free(sheet);

                const ref = try Transformer.qsaSelectMaskOps(std.testing.allocator, s, scores, sheet, tie, offset, seq_len, kv, nb, ratio, block_topk, batch);
                defer _ = mlx.mlx_array_free(ref);
                try mlx.check(mlx.mlx_array_eval(ref));

                if (!qsaAllBlocksVisible(offset, nb, ratio)) continue;
                var ones = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(ones);
                try mlx.check(mlx.mlx_ones(&ones, &[_]c_int{ 1, seq_len, nb }, 3, .bool_, s));
                try std.testing.expect(try qsaArraysAllEqual(sheet, ones, s));
                const null_vis = mlx.mlx_array{ .ctx = null };
                const got = try Transformer.qsaSelectMaskOps(std.testing.allocator, s, scores, null_vis, tie, offset, seq_len, kv, nb, ratio, block_topk, batch);
                defer _ = mlx.mlx_array_free(got);
                try std.testing.expect(try qsaArraysAllEqual(got, ref, s));
                checked_skip += 1;
            }
        }
    }
    std.debug.print("[qsa-boundary] all-visible skip exercised in {d} cases\n", .{checked_skip});
    try std.testing.expect(checked_skip >= 8);
}

test "qsa selection: the mask arm and the gather path use ONE tie rule" {
    // Two selection paths with two different tie rules would let two slots with identical
    // scores attend different blocks. The signal: the mask arm now dispatches the select kernel.
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ta = std.testing.allocator;
    const ratio: c_int = 4;
    const batch: c_int = 1;
    // All-visible at width S needs S <= (kv % ratio) + 1, so kv must not be a multiple of ratio.
    const seq_len: c_int = 4;
    const kv: c_int = 4099;
    const nb = @divTrunc(kv, ratio);
    // 511 so this test owns a config key no other test builds.
    const block_topk: c_int = 511;
    var prng = std.Random.DefaultPrng.init(0x71E5_10FF);
    const rnd = prng.random();

    const scores = try qsaRandScoresF32(rnd, &[_]c_int{ batch, seq_len, nb });
    defer _ = mlx.mlx_array_free(scores);
    const tie = try qsaTieBiasForTest(s, nb);
    defer _ = mlx.mlx_array_free(tie);
    const offset = kv - seq_len;
    try std.testing.expect(qsaAllBlocksVisible(offset, nb, ratio));
    const null_vis = mlx.mlx_array{ .ctx = null };

    const before = qsaSelectCfgBuilds();
    const got = try Transformer.qsaSelectMaskOps(ta, s, scores, null_vis, tie, offset, seq_len, kv, nb, ratio, block_topk, batch);
    defer _ = mlx.mlx_array_free(got);
    try mlx.check(mlx.mlx_array_eval(got));
    // A build, not a hit: this key is unique to this test, so the counter moving IS the dispatch.
    try std.testing.expect(qsaSelectCfgBuilds() > before);

    try std.testing.expectEqual(@as(usize, 4), @as(usize, @intCast(mlx.mlx_array_ndim(got))));
    const gsh = mlx.getShape(got);
    try std.testing.expectEqual(batch, gsh[0]);
    try std.testing.expectEqual(seq_len, gsh[2]);
    try std.testing.expectEqual(kv, gsh[3]);

    // With the kernel off both arms fall to the chain: still one rule.
    qsa_select_kernel_override = false;
    defer qsa_select_kernel_override = null;
    const chain = try Transformer.qsaSelectMaskOps(ta, s, scores, null_vis, tie, offset, seq_len, kv, nb, ratio, block_topk, batch);
    defer _ = mlx.mlx_array_free(chain);
    try mlx.check(mlx.mlx_array_eval(chain));
    try std.testing.expectEqual(kv, mlx.getShape(chain)[3]);
}

test "qsa boundary: past (block_topk+1)*ratio EVERY decode step reaches the all-visible skip" {
    // At decode width offset = kv-1 and nb = floor(kv/ratio), so the all-visible claim holds at
    // every kv and the skip is gated by `nb > block_topk` alone.
    const ratio: c_int = 4;
    const block_topk: c_int = 512; // budget 2048 / ratio
    const first: c_int = (block_topk + 1) * ratio; // 2052
    var kv: c_int = 1;
    while (kv <= 6000) : (kv += 1) {
        const nb = @divTrunc(kv, ratio);
        try std.testing.expect(qsaAllBlocksVisible(kv - 1, nb, ratio));
        try std.testing.expectEqual(kv >= first, nb > block_topk);
    }
    var kv2: c_int = 1;
    while (kv2 < first) : (kv2 += 1) {
        try std.testing.expect(!(@divTrunc(kv2, ratio) > block_topk));
    }
    // A prefill chunk's first row can sit far below the last complete block.
    try std.testing.expect(!qsaAllBlocksVisible(1028, @divTrunc(@as(c_int, 2052), ratio), ratio));
}

test "qsa visibility: decode width is ALWAYS all-visible; a prefill chunk's FIRST row decides" {
    const ratio: c_int = 4;
    // Decode (S == 1): all-visible for every ragged tail.
    var kv: c_int = 2049;
    while (kv <= 2400) : (kv += 1) {
        const nb = @divTrunc(kv, ratio);
        try testing.expect(qsaAllBlocksVisible(kv - 1, nb, ratio));
    }
    try testing.expect(qsaAllBlocksVisible(2051, 513, ratio));
    try testing.expect(!qsaAllBlocksVisible(2050, 513, ratio));
    try testing.expect(!qsaAllBlocksVisible(0, 513, ratio));
    try testing.expect(qsaAllBlocksVisible(0, 0, ratio));
}

test "qsa visibility skip: bit-identical selection where it is an identity, and a REAL difference where it is not" {
    // Where every block is visible the two arms must agree bit for bit; where they are not,
    // the arms must differ (else a blanket "always skip" would pass).
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();

    var t: Transformer = undefined;

    t.rht = null;
    t.s = s;
    t.allocator = testing.allocator;
    t.config = .{};
    t.qsa_consts = .{};
    defer t.qsa_consts.deinit();

    const ratio: c_int = 4;
    const nb: c_int = 24;
    const block_topk: c_int = 8;
    const rows: c_int = 6;
    var prng = std.Random.DefaultPrng.init(0x5EED_9E11);
    const rnd = prng.random();
    var sc_buf: [@as(usize, @intCast(rows)) * @as(usize, @intCast(nb))]f32 = undefined;
    for (&sc_buf) |*v| v.* = if (rnd.boolean()) 0.0 else rnd.float(f32) * 4.0;
    const sc_shape = [_]c_int{ 1, rows, nb };
    const scores = mlx.mlx_array_new_data(&sc_buf, &sc_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(scores);
    const none = mlx.mlx_array{ .ctx = null };

    const row0_all: c_int = nb * ratio - 1;
    try testing.expect(qsaAllBlocksVisible(row0_all, nb, ratio));
    const a = try t.qsaChunkSelect(scores, none, row0_all, rows, nb, ratio, block_topk, block_topk, true);
    defer _ = mlx.mlx_array_free(a);
    const b = try t.qsaChunkSelect(scores, none, row0_all, rows, nb, ratio, block_topk, block_topk, false);
    defer _ = mlx.mlx_array_free(b);
    try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(a, b, s));

    // One row earlier is not all-visible: the two arms must disagree.
    const row0_partial: c_int = 4;
    try testing.expect(!qsaAllBlocksVisible(row0_partial, nb, ratio));
    const c = try t.qsaChunkSelect(scores, none, row0_partial, rows, nb, ratio, block_topk, block_topk, true);
    defer _ = mlx.mlx_array_free(c);
    const d = try t.qsaChunkSelect(scores, none, row0_partial, rows, nb, ratio, block_topk, block_topk, false);
    defer _ = mlx.mlx_array_free(d);
    try testing.expect(try attn256MaxDiff(c, d, s) > 0.0);

    // The `nb <= block_topk` arm takes the same skip.
    var ar = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ar);
    try mlx.check(mlx.mlx_arange(&ar, 0, @floatFromInt(nb), 1.0, .int32, s));
    var canonical = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(canonical);
    try mlx.check(mlx.mlx_reshape(&canonical, ar, &[_]c_int{ 1, 1, nb }, 3, s));
    const e = try t.qsaChunkSelect(scores, canonical, row0_all, rows, nb, ratio, nb, nb, true);
    defer _ = mlx.mlx_array_free(e);
    const f = try t.qsaChunkSelect(scores, canonical, row0_all, rows, nb, ratio, nb, nb, false);
    defer _ = mlx.mlx_array_free(f);
    try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(e, f, s));
}

// ── Fused QSA block select (msv_qsa_select) ──

/// The `[1, rows, nb]` bool visibility sheet the composed arm masks with, free of a Transformer.
fn qsaVisSheetForTest(s: mlx.mlx_stream, row0: c_int, rows: c_int, nb: c_int, ratio: c_int) !mlx.mlx_array {
    var blk_end = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(blk_end);
    try mlx.check(mlx.mlx_arange(&blk_end, @floatFromInt(ratio - 1), @floatFromInt(nb * ratio), @floatFromInt(ratio), .int32, s));
    var blk_end2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(blk_end2);
    try mlx.check(mlx.mlx_reshape(&blk_end2, blk_end, &[_]c_int{ 1, nb }, 2, s));
    var pos = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pos);
    try mlx.check(mlx.mlx_arange(&pos, @floatFromInt(row0), @floatFromInt(row0 + rows), 1.0, .int32, s));
    var pos2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pos2);
    try mlx.check(mlx.mlx_reshape(&pos2, pos, &[_]c_int{ rows, 1 }, 2, s));
    var vis = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vis);
    try mlx.check(mlx.mlx_less_equal(&vis, blk_end2, pos2, s));
    var vis3 = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_expand_dims(&vis3, vis, 0, s));
    return vis3;
}

/// `QsaBlockConsts.tie_bias` without the cache: `b * 1e-7`.
pub fn qsaTieBiasForTest(s: mlx.mlx_stream, nb: c_int) !mlx.mlx_array {
    var idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(idx);
    try mlx.check(mlx.mlx_arange(&idx, 0, @floatFromInt(nb), 1.0, .float32, s));
    const eps = mlx.mlx_array_new_float(1e-7);
    defer _ = mlx.mlx_array_free(eps);
    var bias = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&bias, idx, eps, s));
    return bias;
}

test "qsaVisibleBoundsHost: the first NOT-fully-visible block per row" {
    var out: [6]i32 = undefined;
    qsaVisibleBoundsHost(&out, 0, 6, 700, 4, false);
    try testing.expectEqualSlices(i32, &[_]i32{ 0, 0, 0, 1, 1, 1 }, &out);
    qsaVisibleBoundsHost(&out, 2799, 6, 700, 4, false);
    try testing.expectEqualSlices(i32, &[_]i32{ 700, 700, 700, 700, 700, 700 }, &out);
    qsaVisibleBoundsHost(&out, 2044, 6, 700, 4, false);
    try testing.expectEqualSlices(i32, &[_]i32{ 511, 511, 511, 512, 512, 512 }, &out);
    qsaVisibleBoundsHost(&out, 4, 6, 24, 4, true);
    try testing.expectEqualSlices(i32, &[_]i32{ 24, 24, 24, 24, 24, 24 }, &out);
}

/// The reference both arms are measured against: per row, the top-`k` visible blocks by
/// (score descending, index ascending), emitted ascending with INT_MAX past the row's count.
fn qsaExactTopHost(alloc: std.mem.Allocator, scores: []const f32, rows: c_int, nb: c_int, bounds: []const i32, k: c_int) ![]i32 {
    const nbu: usize = @intCast(nb);
    const ku: usize = @intCast(k);
    const out = try alloc.alloc(i32, @as(usize, @intCast(rows)) * ku);
    errdefer alloc.free(out);
    const Cand = struct { v: f32, i: i32 };
    const cand = try alloc.alloc(Cand, nbu);
    defer alloc.free(cand);
    const pick = try alloc.alloc(i32, ku);
    defer alloc.free(pick);
    const Lt = struct {
        fn lt(_: void, a: Cand, b: Cand) bool {
            if (a.v != b.v) return a.v > b.v;
            return a.i < b.i;
        }
    };
    for (0..@intCast(rows)) |r| {
        const vb: usize = @intCast(@max(bounds[r], 0));
        for (0..vb) |i| cand[i] = .{ .v = scores[r * nbu + i], .i = @intCast(i) };
        std.sort.pdq(Cand, cand[0..vb], {}, Lt.lt);
        const take = @min(vb, ku);
        for (0..take) |i| pick[i] = cand[i].i;
        std.sort.pdq(i32, pick[0..take], {}, std.sort.asc(i32));
        for (0..take) |i| out[r * ku + i] = pick[i];
        for (take..ku) |i| out[r * ku + i] = std.math.maxInt(i32);
    }
    return out;
}

fn qsaExpectMatchesHost(got: mlx.mlx_array, want: []const i32, rows: c_int, k: c_int, who: []const u8) !void {
    const gs = mlx.getShape(got);
    try testing.expectEqual(@as(usize, 3), gs.len);
    try testing.expectEqual(@as(c_int, 1), gs[0]);
    try testing.expectEqual(rows, gs[1]);
    try testing.expectEqual(k, gs[2]);
    try testing.expectEqual(mlx.mlx_dtype.int32, mlx.mlx_array_dtype(got));
    try mlx.check(mlx.mlx_array_eval(got));
    const b = mlx.mlx_array_data_int32(got).?;
    const ku: usize = @intCast(k);
    for (want, 0..) |w, i| {
        if (w != b[i]) {
            std.debug.print(
                "[qsa-select] {s} differs from the exact reference at row {d} slot {d}: reference {d} got {d}\n",
                .{ who, i / ku, i % ku, w, b[i] },
            );
            return error.QsaSelectMismatch;
        }
    }
}

test "qsa select kernel: the fused radix-select IS the exact top-k, and so is the chain where its bias can resolve" {
    // Bit equality against `qsaExactTopHost` across the nb the arch reaches (700..262144), the
    // query widths it runs at and three visibility regimes. Scores are exact binary fractions
    // k/16 in [0, 0.5] plus forced zeros: the regime where the chain's bias is also exact, so
    // both arms can be held to the same bar.
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const prev_split = qsa_select_split_override;
    defer qsa_select_split_override = prev_split;
    qsa_select_split_override = false;
    const s = mlx.gpuStream();
    const ta = testing.allocator;
    const ratio: c_int = 4;
    const k: c_int = 512;

    const cases = [_]struct { nb: c_int, rows: c_int }{
        .{ .nb = 700, .rows = 1 },
        .{ .nb = 700, .rows = 6 },
        .{ .nb = 700, .rows = 15 },
        .{ .nb = 700, .rows = 256 },
        .{ .nb = 2048, .rows = 1 },
        .{ .nb = 2048, .rows = 15 },
        .{ .nb = 2048, .rows = 256 },
        .{ .nb = 15675, .rows = 1 },
        .{ .nb = 15675, .rows = 6 },
        .{ .nb = 15675, .rows = 15 },
        .{ .nb = 15675, .rows = 256 },
        .{ .nb = 65536, .rows = 1 },
        .{ .nb = 65536, .rows = 15 },
        .{ .nb = 65536, .rows = 64 },
        .{ .nb = 262144, .rows = 1 },
        .{ .nb = 262144, .rows = 6 },
    };
    var prng = std.Random.DefaultPrng.init(0x9A50_C711);
    const rnd = prng.random();

    for (cases) |c| {
        const nbu: usize = @intCast(c.nb);
        const rowsu: usize = @intCast(c.rows);
        const buf = try ta.alloc(f32, rowsu * nbu);
        defer ta.free(buf);
        for (buf) |*v| v.* = if (rnd.uintLessThan(u8, 3) == 0)
            0.0
        else
            @as(f32, @floatFromInt(rnd.uintLessThan(u32, 9))) / 16.0;
        const sc_shape = [_]c_int{ 1, c.rows, c.nb };
        const scores = mlx.mlx_array_new_data(buf.ptr, &sc_shape, 3, .float32);
        defer _ = mlx.mlx_array_free(scores);

        const row0s = [_]c_int{ 0, 2044, 2000, c.nb * ratio };
        for (row0s) |row0| {
            const all_vis = qsaAllBlocksVisible(row0, c.nb, ratio);
            const bounds_host = try ta.alloc(i32, rowsu);
            defer ta.free(bounds_host);
            qsaVisibleBoundsHost(bounds_host, row0, c.rows, c.nb, ratio, all_vis);
            const want = try qsaExactTopHost(ta, buf, c.rows, c.nb, bounds_host, k);
            defer ta.free(want);

            const bounds = mlx.mlx_array_new_data(bounds_host.ptr, &[_]c_int{c.rows}, 1, .int32);
            defer _ = mlx.mlx_array_free(bounds);
            const got = (try qsaSelectTopBlocks(s, scores, bounds, k)) orelse return error.SelectKernelDeclined;
            defer _ = mlx.mlx_array_free(got);

            var vis3 = mlx.mlx_array{ .ctx = null };
            defer if (vis3.ctx != null) {
                _ = mlx.mlx_array_free(vis3);
            };
            if (!all_vis) vis3 = try qsaVisSheetForTest(s, row0, c.rows, c.nb, ratio);
            const bias = try qsaTieBiasForTest(s, c.nb);
            defer _ = mlx.mlx_array_free(bias);
            const chain = try qsaSelectComposedOps(s, scores, vis3, bias, c.rows, c.nb, k, k, all_vis);
            defer _ = mlx.mlx_array_free(chain);

            qsaExpectMatchesHost(got, want, c.rows, k, "the fused kernel") catch |e| {
                std.debug.print("[qsa-select] case nb={d} rows={d} row0={d} all_vis={}\n", .{ c.nb, c.rows, row0, all_vis });
                return e;
            };
            qsaExpectMatchesHost(chain, want, c.rows, k, "the composed chain") catch |e| {
                std.debug.print("[qsa-select] case nb={d} rows={d} row0={d} all_vis={}\n", .{ c.nb, c.rows, row0, all_vis });
                return e;
            };
        }
    }
}

test "qsa select kernel: past the bias's f32 resolution the chain's tie rule is UNSPECIFIED and the kernel's is not" {
    // The chain's `b * 1e-7` bias only survives rounding below |v| ~= 0.84 (one f32 ulp is
    // v * 2^-23); above it ties fall back to argpartition's arbitrary order. A relu sum over
    // four heads sits far above that, so this is the production regime.
    const eps: f32 = 1e-7;
    // Distinct biased values the chain can see across a 2048-block row: 4 at production
    // magnitude, every block at |v| <= 0.5. Pure f32 arithmetic.
    const Distinct = struct {
        fn count(v: f32, n: u32) usize {
            var seen: usize = 0;
            var prev: f32 = 0;
            var b: u32 = 0;
            while (b < n) : (b += 1) {
                const cur = v - @as(f32, @floatFromInt(b)) * eps;
                if (b == 0 or cur != prev) seen += 1;
                prev = cur;
            }
            return seen;
        }
    };
    try testing.expectEqual(@as(usize, 4), Distinct.count(625.0, 2048));
    try testing.expectEqual(@as(usize, 2048), Distinct.count(0.5, 2048));
    const big: f32 = 625.0;
    try testing.expectEqual(big - 1369 * eps, big - 1370 * eps);

    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ta = testing.allocator;
    const ratio: c_int = 4;
    const nb: c_int = 2048;
    const k: c_int = 512;
    const rows: c_int = 16;

    const nbu: usize = @intCast(nb);
    const buf = try ta.alloc(f32, @as(usize, @intCast(rows)) * nbu);
    defer ta.free(buf);
    var prng = std.Random.DefaultPrng.init(0x9A50_C711);
    const rnd = prng.random();
    for (buf) |*v| v.* = if (rnd.uintLessThan(u8, 3) == 0)
        0.0
    else
        @floatFromInt(rnd.uintLessThan(u32, 1000));

    const sc_shape = [_]c_int{ 1, rows, nb };
    const scores = mlx.mlx_array_new_data(buf.ptr, &sc_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(scores);
    const row0: c_int = nb * ratio;
    try testing.expect(qsaAllBlocksVisible(row0, nb, ratio));

    const bounds_host = try ta.alloc(i32, @intCast(rows));
    defer ta.free(bounds_host);
    @memset(bounds_host, nb);
    const want = try qsaExactTopHost(ta, buf, rows, nb, bounds_host, k);
    defer ta.free(want);
    const bounds = mlx.mlx_array_new_data(bounds_host.ptr, &[_]c_int{rows}, 1, .int32);
    defer _ = mlx.mlx_array_free(bounds);

    // The kernel is the reference at production magnitude.
    const got = (try qsaSelectTopBlocks(s, scores, bounds, k)) orelse return error.SelectKernelDeclined;
    defer _ = mlx.mlx_array_free(got);
    try qsaExpectMatchesHost(got, want, rows, k, "the fused kernel");

    // The chain may only ever be wrong about a tie: the bar is the selected score multiset per row.
    const none = mlx.mlx_array{ .ctx = null };
    const bias = try qsaTieBiasForTest(s, nb);
    defer _ = mlx.mlx_array_free(bias);
    const chain = try qsaSelectComposedOps(s, scores, none, bias, rows, nb, k, k, true);
    defer _ = mlx.mlx_array_free(chain);
    try mlx.check(mlx.mlx_array_eval(chain));
    const ch = mlx.mlx_array_data_int32(chain).?;
    const ku: usize = @intCast(k);
    const sw = try ta.alloc(f32, ku);
    defer ta.free(sw);
    const sc = try ta.alloc(f32, ku);
    defer ta.free(sc);
    var diffs: usize = 0;
    for (0..@intCast(rows)) |r| {
        for (0..ku) |i| {
            sw[i] = buf[r * nbu + @as(usize, @intCast(want[r * ku + i]))];
            sc[i] = buf[r * nbu + @as(usize, @intCast(ch[r * ku + i]))];
            if (want[r * ku + i] != ch[r * ku + i]) diffs += 1;
        }
        std.sort.pdq(f32, sw, {}, std.sort.asc(f32));
        std.sort.pdq(f32, sc, {}, std.sort.asc(f32));
        try testing.expectEqualSlices(f32, sw, sc);
    }
    if (diffs > 0) {
        std.debug.print("[qsa-select] the chain differs from the exact rule in {d}/{d} slots at production magnitude (tied blocks only)\n", .{ diffs, want.len });
    }
}

test "qsa select kernel: on a near-tie the exact rule and the bias trick DISAGREE, and the kernel takes the exact one" {
    // The bias is exact only on exact ties: at nb = 262144 it spans 0.026, so a genuinely higher
    // score loses to a lower-indexed rival. torch.topk and the kernel take the higher score.
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ta = testing.allocator;
    const nb: c_int = 262144;
    const k: c_int = 512;
    const ratio: c_int = 4;
    const winner: usize = 200000;

    const buf = try ta.alloc(f32, @intCast(nb));
    defer ta.free(buf);
    @memset(buf, 0.0);
    for (1..512) |i| buf[i] = 1000.0;
    buf[0] = 10.0; //          biased: 10.000  (bias 0)
    buf[winner] = 10.005; //   biased:  9.985  (bias 0.02) — the bias flips it
    const sc_shape = [_]c_int{ 1, 1, nb };
    const scores = mlx.mlx_array_new_data(buf.ptr, &sc_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(scores);

    const row0: c_int = nb * ratio; // all-visible
    try testing.expect(qsaAllBlocksVisible(row0, nb, ratio));
    const none = mlx.mlx_array{ .ctx = null };
    const bias = try qsaTieBiasForTest(s, nb);
    defer _ = mlx.mlx_array_free(bias);
    const chain = try qsaSelectComposedOps(s, scores, none, bias, 1, nb, k, k, true);
    defer _ = mlx.mlx_array_free(chain);

    var bounds_host = [_]i32{nb};
    const bounds = mlx.mlx_array_new_data(&bounds_host, &[_]c_int{1}, 1, .int32);
    defer _ = mlx.mlx_array_free(bounds);
    const fused = (try qsaSelectTopBlocks(s, scores, bounds, k)) orelse return error.SelectKernelDeclined;
    defer _ = mlx.mlx_array_free(fused);

    try mlx.check(mlx.mlx_array_eval(chain));
    try mlx.check(mlx.mlx_array_eval(fused));
    const ch = mlx.mlx_array_data_int32(chain).?;
    const fu = mlx.mlx_array_data_int32(fused).?;
    const H = struct {
        fn has(row: [*]const i32, n: usize, v: i32) bool {
            for (0..n) |i| if (row[i] == v) return true;
            return false;
        }
    };
    const ku: usize = @intCast(k);
    try testing.expect(H.has(fu, ku, @intCast(winner)));
    try testing.expect(!H.has(fu, ku, 0));
    try testing.expect(H.has(ch, ku, 0));
    try testing.expect(!H.has(ch, ku, @intCast(winner)));
    for (1..ku) |i| try testing.expect(fu[i] > fu[i - 1]);
}

test "qsa select kernel: the config is cached per FULL shape and rebuilt when rows or nb change" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ta = testing.allocator;
    const k: c_int = 8;

    const Run = struct {
        fn once(st: mlx.mlx_stream, alloc: std.mem.Allocator, rows: c_int, nb: c_int, kk: c_int) !void {
            const buf = try alloc.alloc(f32, @intCast(rows * nb));
            defer alloc.free(buf);
            for (buf, 0..) |*v, i| v.* = @floatFromInt(i % 97);
            const sh = [_]c_int{ 1, rows, nb };
            const sc = mlx.mlx_array_new_data(buf.ptr, &sh, 3, .float32);
            defer _ = mlx.mlx_array_free(sc);
            const bh = try alloc.alloc(i32, @intCast(rows));
            defer alloc.free(bh);
            @memset(bh, nb);
            const bd = mlx.mlx_array_new_data(bh.ptr, &[_]c_int{rows}, 1, .int32);
            defer _ = mlx.mlx_array_free(bd);
            const out = (try qsaSelectTopBlocks(st, sc, bd, kk)) orelse return error.SelectKernelDeclined;
            defer _ = mlx.mlx_array_free(out);
            try mlx.check(mlx.mlx_array_eval(out));
        }
    };

    // Normalize first: one global cache shared with every test in the file.
    try Run.once(s, ta, 4, 64, k);
    const base = qsaSelectCfgBuilds();
    for (0..3) |_| try Run.once(s, ta, 4, 64, k);
    try testing.expectEqual(base, qsaSelectCfgBuilds());
    // `nb` is a runtime shape, not a config input.
    try Run.once(s, ta, 4, 128, k);
    try Run.once(s, ta, 4, 517, k);
    try Run.once(s, ta, 4, 1027, k);
    try testing.expectEqual(base, qsaSelectCfgBuilds());
    try Run.once(s, ta, 9, 64, k);
    try testing.expectEqual(base + 1, qsaSelectCfgBuilds());
    try Run.once(s, ta, 4, 64, k);
    try testing.expectEqual(base + 1, qsaSelectCfgBuilds());
    try testing.expect(@TypeOf(qsa_select_cfgs).SLOTS == 4);
}

fn qsaIdsBitEqual(s: mlx.mlx_stream, a: mlx.mlx_array, b: mlx.mlx_array) !bool {
    var res = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(res);
    try mlx.check(mlx.mlx_array_equal(&res, a, b, false, s));
    try mlx.check(mlx.mlx_array_eval(res));
    var v: bool = false;
    try mlx.check(mlx.mlx_array_item_bool(&v, res));
    return v;
}

fn qsaSelectMakeSheet(kind: u8, buf: []f32, rnd: std.Random) void {
    switch (kind) {
        0 => {
            for (buf) |*v| v.* = rnd.float(f32) * 2.0 - 1.0;
        },
        1 => {
            for (buf) |*v| v.* = if (rnd.uintLessThan(u8, 5) == 0) rnd.float(f32) else 0.0;
        },
        2 => {
            @memset(buf, 0.0);
            const n = buf.len;
            const span = @min(n, 1024);
            for (0..span) |i| buf[i] = 5.0;
        },
        3 => {
            for (buf) |*v| v.* = rnd.float(f32);
            buf[0] = std.math.nan(f32);
            if (buf.len > 1) buf[1] = -0.0;
            if (buf.len > 2) buf[2] = 0.0;
        },
        else => {
            for (buf) |*v| v.* = if (rnd.uintLessThan(u8, 3) == 0) 0.0 else @as(f32, @floatFromInt(rnd.uintLessThan(u32, 9))) / 16.0;
        },
    }
}

test "qsa select split: ids are identical to the single-threadgroup kernel" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ta = testing.allocator;
    const k: c_int = 512;
    const prev = qsa_select_split_override;
    defer qsa_select_split_override = prev;
    qsa_select_split_override = false;

    const nbs = [_]c_int{ 2048, 15675, 65536, 215000, 262144 };
    const row_counts = [_]c_int{ 1, 2, 4, 7, 15, 16 };
    var prng = std.Random.DefaultPrng.init(0x51E7_0A11);
    const rnd = prng.random();

    for (nbs) |nb| {
        for (row_counts) |rows| {
            if (nb >= 215000 and rows >= 16) continue;
            const nbu: usize = @intCast(nb);
            const rowsu: usize = @intCast(rows);
            const buf = try ta.alloc(f32, rowsu * nbu);
            defer ta.free(buf);
            var kind: u8 = 0;
            while (kind < 5) : (kind += 1) {
                qsaSelectMakeSheet(kind, buf, rnd);
                const sc_shape = [_]c_int{ 1, rows, nb };
                const scores = mlx.mlx_array_new_data(buf.ptr, &sc_shape, 3, .float32);
                defer _ = mlx.mlx_array_free(scores);
                const bound_kinds = [_]i32{ nb, @divTrunc(nb, 2), 10, 0 };
                for (bound_kinds) |bv| {
                    const bounds_host = try ta.alloc(i32, rowsu);
                    defer ta.free(bounds_host);
                    var r: usize = 0;
                    while (r < rowsu) : (r += 1) {
                        const adj = bv + @as(i32, @intCast(r % 3)) - 1;
                        bounds_host[r] = @max(0, @min(nb, adj));
                    }
                    const bounds = mlx.mlx_array_new_data(bounds_host.ptr, &[_]c_int{rows}, 1, .int32);
                    defer _ = mlx.mlx_array_free(bounds);
                    const single = (try qsaSelectTopBlocks(s, scores, bounds, k)) orelse return error.SelectKernelDeclined;
                    defer _ = mlx.mlx_array_free(single);
                    const split = (try qsaSelectTopBlocksSplit(s, scores, bounds, k, rows, nb)) orelse return error.SelectKernelDeclined;
                    defer _ = mlx.mlx_array_free(split);
                    if (!try qsaIdsBitEqual(s, single, split)) {
                        std.debug.print("[qsa-select-split] mismatch nb={d} rows={d} kind={d} bound={d}\n", .{ nb, rows, kind, bv });
                        return error.QsaSelectSplitMismatch;
                    }
                }
            }
        }
    }

    {
        const nb: c_int = 65536;
        const rows: c_int = 1;
        const kk: c_int = 511;
        const buf = try ta.alloc(f32, @intCast(nb));
        defer ta.free(buf);
        qsaSelectMakeSheet(1, buf, rnd);
        const scores = mlx.mlx_array_new_data(buf.ptr, &[_]c_int{ 1, rows, nb }, 3, .float32);
        defer _ = mlx.mlx_array_free(scores);
        var bh = [_]i32{nb};
        const bounds = mlx.mlx_array_new_data(&bh, &[_]c_int{rows}, 1, .int32);
        defer _ = mlx.mlx_array_free(bounds);
        const single = (try qsaSelectTopBlocks(s, scores, bounds, kk)) orelse return error.SelectKernelDeclined;
        defer _ = mlx.mlx_array_free(single);
        const split = (try qsaSelectTopBlocksSplit(s, scores, bounds, kk, rows, nb)) orelse return error.SelectKernelDeclined;
        defer _ = mlx.mlx_array_free(split);
        if (!try qsaIdsBitEqual(s, single, split)) return error.QsaSelectSplitMismatch;
    }
}

test "qsa select split: declines above the width floor and below the nb floor" {
    const k: c_int = 512;
    try testing.expect(!qsaSelectSplitServes(16, 215000, k));
    try testing.expect(!qsaSelectSplitServes(32, 262144, k));
    try testing.expect(!qsaSelectSplitServes(1, 24575, k));
    try testing.expect(qsaSelectSplitServes(1, 24576, k));
    try testing.expect(qsaSelectSplitServes(1, 32768, k));
    try testing.expect(!qsaSelectSplitServes(15, 2048, k));
    try testing.expect(!qsaSelectSplitServes(0, 65536, k));
    try testing.expect(!qsaSelectSplitServes(-1, 65536, k));
    try testing.expect(!qsaSelectSplitServes(1, 65536, 0));
    try testing.expect(!qsaSelectSplitServes(1, 65536, 4096));
    try testing.expect(qsaSelectSplitServes(1, 65536, k));
    try testing.expect(qsaSelectSplitServes(15, 65536, k));
    try testing.expect(qsaSelectSplitServes(7, 215000, k));

    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ta = testing.allocator;
    const prev = qsa_select_split_override;
    defer qsa_select_split_override = prev;
    qsa_select_split_override = true;

    const Run = struct {
        fn once(st: mlx.mlx_stream, alloc: std.mem.Allocator, rows: c_int, nb: c_int) !void {
            const buf = try alloc.alloc(f32, @intCast(rows * nb));
            defer alloc.free(buf);
            for (buf, 0..) |*v, i| v.* = @floatFromInt(i % 97);
            const sh = [_]c_int{ 1, rows, nb };
            const sc = mlx.mlx_array_new_data(buf.ptr, &sh, 3, .float32);
            defer _ = mlx.mlx_array_free(sc);
            const bh = try alloc.alloc(i32, @intCast(rows));
            defer alloc.free(bh);
            @memset(bh, nb);
            const bd = mlx.mlx_array_new_data(bh.ptr, &[_]c_int{rows}, 1, .int32);
            defer _ = mlx.mlx_array_free(bd);
            const out = (try qsaSelectTopBlocks(st, sc, bd, 8)) orelse return error.SelectKernelDeclined;
            defer _ = mlx.mlx_array_free(out);
            try mlx.check(mlx.mlx_array_eval(out));
        }
    };

    try Run.once(s, ta, 16, 65536);
    try testing.expect(!qsa_select_used_split);
    try Run.once(s, ta, 1, 2048);
    try testing.expect(!qsa_select_used_split);
    try Run.once(s, ta, 1, 24576);
    try testing.expect(qsa_select_used_split);
}

test "qsa select split: the fused radix-select IS the exact top-k" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ta = testing.allocator;
    const ratio: c_int = 4;
    const k: c_int = 512;
    const prev = qsa_select_split_override;
    defer qsa_select_split_override = prev;
    qsa_select_split_override = true;

    const cases = [_]struct { nb: c_int, rows: c_int }{
        .{ .nb = 700, .rows = 1 },
        .{ .nb = 700, .rows = 6 },
        .{ .nb = 2048, .rows = 1 },
        .{ .nb = 2048, .rows = 15 },
        .{ .nb = 15675, .rows = 1 },
        .{ .nb = 15675, .rows = 6 },
        .{ .nb = 65536, .rows = 1 },
        .{ .nb = 65536, .rows = 15 },
        .{ .nb = 262144, .rows = 1 },
        .{ .nb = 262144, .rows = 6 },
    };
    var prng = std.Random.DefaultPrng.init(0x9A50_C711);
    const rnd = prng.random();

    for (cases) |c| {
        const nbu: usize = @intCast(c.nb);
        const rowsu: usize = @intCast(c.rows);
        const buf = try ta.alloc(f32, rowsu * nbu);
        defer ta.free(buf);
        for (buf) |*v| v.* = if (rnd.uintLessThan(u8, 3) == 0)
            0.0
        else
            @as(f32, @floatFromInt(rnd.uintLessThan(u32, 9))) / 16.0;
        const sc_shape = [_]c_int{ 1, c.rows, c.nb };
        const scores = mlx.mlx_array_new_data(buf.ptr, &sc_shape, 3, .float32);
        defer _ = mlx.mlx_array_free(scores);

        const row0s = [_]c_int{ 0, 2044, 2000, c.nb * ratio };
        for (row0s) |row0| {
            const all_vis = qsaAllBlocksVisible(row0, c.nb, ratio);
            const bounds_host = try ta.alloc(i32, rowsu);
            defer ta.free(bounds_host);
            qsaVisibleBoundsHost(bounds_host, row0, c.rows, c.nb, ratio, all_vis);
            const want = try qsaExactTopHost(ta, buf, c.rows, c.nb, bounds_host, k);
            defer ta.free(want);

            const bounds = mlx.mlx_array_new_data(bounds_host.ptr, &[_]c_int{c.rows}, 1, .int32);
            defer _ = mlx.mlx_array_free(bounds);
            const got = (try qsaSelectTopBlocks(s, scores, bounds, k)) orelse return error.SelectKernelDeclined;
            defer _ = mlx.mlx_array_free(got);
            qsaExpectMatchesHost(got, want, c.rows, k, "split-forced fused kernel") catch |e| {
                std.debug.print("[qsa-select] split-forced case nb={d} rows={d} row0={d} all_vis={}\n", .{ c.nb, c.rows, row0, all_vis });
                return e;
            };
        }
    }
}

test "qsa select split: kill switch off uses the single-threadgroup kernel" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ta = testing.allocator;
    const prev = qsa_select_split_override;
    defer qsa_select_split_override = prev;

    const Run = struct {
        fn once(st: mlx.mlx_stream, alloc: std.mem.Allocator, rows: c_int, nb: c_int) !void {
            const buf = try alloc.alloc(f32, @intCast(rows * nb));
            defer alloc.free(buf);
            for (buf, 0..) |*v, i| v.* = @floatFromInt(i % 97);
            const sh = [_]c_int{ 1, rows, nb };
            const sc = mlx.mlx_array_new_data(buf.ptr, &sh, 3, .float32);
            defer _ = mlx.mlx_array_free(sc);
            const bh = try alloc.alloc(i32, @intCast(rows));
            defer alloc.free(bh);
            @memset(bh, nb);
            const bd = mlx.mlx_array_new_data(bh.ptr, &[_]c_int{rows}, 1, .int32);
            defer _ = mlx.mlx_array_free(bd);
            const out = (try qsaSelectTopBlocks(st, sc, bd, 512)) orelse return error.SelectKernelDeclined;
            defer _ = mlx.mlx_array_free(out);
            try mlx.check(mlx.mlx_array_eval(out));
        }
    };

    qsa_select_split_override = false;
    try Run.once(s, ta, 1, 65536);
    try testing.expect(!qsa_select_used_split);

    qsa_select_split_override = true;
    try Run.once(s, ta, 1, 65536);
    try testing.expect(qsa_select_used_split);
}

test "qsa select split: the config is cached per FULL shape and rebuilt when rows change" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ta = testing.allocator;
    const prev = qsa_select_split_override;
    defer qsa_select_split_override = prev;
    qsa_select_split_override = true;
    const k: c_int = 512;

    const Run = struct {
        fn once(st: mlx.mlx_stream, alloc: std.mem.Allocator, rows: c_int, nb: c_int, kk: c_int) !void {
            const buf = try alloc.alloc(f32, @intCast(rows * nb));
            defer alloc.free(buf);
            for (buf, 0..) |*v, i| v.* = @floatFromInt(i % 97);
            const sh = [_]c_int{ 1, rows, nb };
            const sc = mlx.mlx_array_new_data(buf.ptr, &sh, 3, .float32);
            defer _ = mlx.mlx_array_free(sc);
            const bh = try alloc.alloc(i32, @intCast(rows));
            defer alloc.free(bh);
            @memset(bh, nb);
            const bd = mlx.mlx_array_new_data(bh.ptr, &[_]c_int{rows}, 1, .int32);
            defer _ = mlx.mlx_array_free(bd);
            const out = (try qsaSelectTopBlocks(st, sc, bd, kk)) orelse return error.SelectKernelDeclined;
            defer _ = mlx.mlx_array_free(out);
            try mlx.check(mlx.mlx_array_eval(out));
        }
    };

    try Run.once(s, ta, 1, 65536, k);
    try testing.expect(qsa_select_used_split);
    const base = qsaSelectSplitCfgBuilds();
    for (0..3) |_| try Run.once(s, ta, 1, 65536, k);
    try testing.expectEqual(base, qsaSelectSplitCfgBuilds());
    try Run.once(s, ta, 1, 131072, k);
    try testing.expectEqual(base, qsaSelectSplitCfgBuilds());
    try Run.once(s, ta, 2, 65536, k);
    try testing.expectEqual(base + 1, qsaSelectSplitCfgBuilds());
    try Run.once(s, ta, 1, 65536, k);
    try testing.expectEqual(base + 1, qsaSelectSplitCfgBuilds());
}

test "qsa select: the digit scheme is a pure function of rows" {
    const saved_digits = qsa_select_digits_override;
    defer qsa_select_digits_override = saved_digits;
    qsa_select_digits_override = null;
    try testing.expectEqual(@as(c_int, 8), qsaSelectDigitBits(1));
    try testing.expectEqual(@as(c_int, 8), qsaSelectDigitBits(7));
    try testing.expectEqual(@as(c_int, 8), qsaSelectDigitBits(15));
    try testing.expectEqual(@as(c_int, 11), qsaSelectDigitBits(16));
    try testing.expectEqual(@as(c_int, 11), qsaSelectDigitBits(256));
    try testing.expectEqual(@as(c_int, 11), qsaSelectDigitBits(4096));
    qsa_select_digits_override = 8;
    try testing.expectEqual(@as(c_int, 8), qsaSelectDigitBits(256));
    qsa_select_digits_override = 11;
    try testing.expectEqual(@as(c_int, 11), qsaSelectDigitBits(1));
    qsa_select_digits_override = null;

    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();
    const ta = testing.allocator;
    const prev_split = qsa_select_split_override;
    defer qsa_select_split_override = prev_split;
    qsa_select_split_override = false;
    const kk: c_int = 13;
    const rows: c_int = 5;
    const nb: c_int = 96;

    const Run = struct {
        fn once(st: mlx.mlx_stream, alloc: std.mem.Allocator, r: c_int, n: c_int, k: c_int) !void {
            const buf = try alloc.alloc(f32, @intCast(r * n));
            defer alloc.free(buf);
            for (buf, 0..) |*v, i| v.* = @floatFromInt(i % 97);
            const sh = [_]c_int{ 1, r, n };
            const sc = mlx.mlx_array_new_data(buf.ptr, &sh, 3, .float32);
            defer _ = mlx.mlx_array_free(sc);
            const bh = try alloc.alloc(i32, @intCast(r));
            defer alloc.free(bh);
            @memset(bh, n);
            const bd = mlx.mlx_array_new_data(bh.ptr, &[_]c_int{r}, 1, .int32);
            defer _ = mlx.mlx_array_free(bd);
            const out = (try qsaSelectTopBlocks(st, sc, bd, k)) orelse return error.SelectKernelDeclined;
            defer _ = mlx.mlx_array_free(out);
            try mlx.check(mlx.mlx_array_eval(out));
        }
    };

    qsa_select_digits_override = 11;
    try Run.once(s, ta, rows, nb, kk);
    const base = qsaSelectCfgBuilds();
    try Run.once(s, ta, rows, nb, kk);
    try testing.expectEqual(base, qsaSelectCfgBuilds());
    qsa_select_digits_override = 8;
    try Run.once(s, ta, rows, nb, kk);
    try testing.expectEqual(base + 1, qsaSelectCfgBuilds());
    try Run.once(s, ta, rows, nb, kk);
    try testing.expectEqual(base + 1, qsaSelectCfgBuilds());
}

test "qsa select: 8-bit digits are the exact top-k" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ta = testing.allocator;
    const saved_digits = qsa_select_digits_override;
    const saved_split = qsa_select_split_override;
    defer {
        qsa_select_digits_override = saved_digits;
        qsa_select_split_override = saved_split;
    }
    const k: c_int = 512;
    const nbs = [_]c_int{ 2048, 15675, 24576, 65536, 215000 };
    const row_counts = [_]c_int{ 1, 7, 15, 16, 256 };
    var prng = std.Random.DefaultPrng.init(0x8B17_D161);
    const rnd = prng.random();

    for (nbs) |nb| {
        for (row_counts) |rows| {
            if (nb >= 215000 and rows >= 16) continue;
            const nbu: usize = @intCast(nb);
            const rowsu: usize = @intCast(rows);
            const ku: usize = @intCast(k);
            const buf = try ta.alloc(f32, rowsu * nbu);
            defer ta.free(buf);
            var kind: u8 = 0;
            while (kind < 5) : (kind += 1) {
                switch (kind) {
                    0 => {
                        for (buf) |*v| v.* = rnd.float(f32) * 2.0 - 1.0;
                    },
                    1 => {
                        for (buf) |*v| v.* = if (rnd.float(f32) < 0.60) 0.0 else rnd.float(f32) * 2.0 - 1.0;
                    },
                    2 => {
                        for (buf) |*v| v.* = if (rnd.float(f32) < 0.95) 0.0 else rnd.float(f32) * 2.0 - 1.0;
                    },
                    3 => {
                        for (buf) |*v| v.* = rnd.float(f32) * 0.5;
                        var r: usize = 0;
                        while (r < rowsu) : (r += 1) {
                            const row = buf[r * nbu ..][0..nbu];
                            var t: usize = 0;
                            while (t < ku + 400) : (t += 1) {
                                row[rnd.uintLessThan(usize, nbu)] = 3.0;
                            }
                            var h: usize = 0;
                            while (h < 50) : (h += 1) {
                                row[rnd.uintLessThan(usize, nbu)] = 9.0;
                            }
                        }
                    },
                    else => {
                        for (buf) |*v| v.* = rnd.float(f32);
                        var i: usize = 0;
                        while (i < buf.len) : (i += 1) {
                            const u = rnd.uintLessThan(u8, 100);
                            if (u < 2) buf[i] = std.math.nan(f32) else if (u < 32) buf[i] = -0.0 else if (u < 52) buf[i] = 0.0;
                        }
                    },
                }
                const sc_shape = [_]c_int{ 1, rows, nb };
                const scores = mlx.mlx_array_new_data(buf.ptr, &sc_shape, 3, .float32);
                defer _ = mlx.mlx_array_free(scores);
                const bound_kinds = [_]i32{ nb, @divTrunc(k, 2), @min(nb, k + 17) };
                for (bound_kinds) |bv| {
                    const bounds_host = try ta.alloc(i32, rowsu);
                    defer ta.free(bounds_host);
                    var r: usize = 0;
                    while (r < rowsu) : (r += 1) {
                        const adj = bv + @as(i32, @intCast(r % 3)) - 1;
                        bounds_host[r] = @max(0, @min(nb, adj));
                    }
                    const bounds = mlx.mlx_array_new_data(bounds_host.ptr, &[_]c_int{rows}, 1, .int32);
                    defer _ = mlx.mlx_array_free(bounds);

                    qsa_select_split_override = false;
                    qsa_select_digits_override = 11;
                    const ref11 = (try qsaSelectTopBlocks(s, scores, bounds, k)) orelse return error.SelectKernelDeclined;
                    defer _ = mlx.mlx_array_free(ref11);

                    qsa_select_digits_override = 8;
                    const single8 = (try qsaSelectTopBlocks(s, scores, bounds, k)) orelse return error.SelectKernelDeclined;
                    defer _ = mlx.mlx_array_free(single8);
                    if (!try qsaIdsBitEqual(s, ref11, single8)) {
                        std.debug.print("[qsa-select-8bit] single mismatch nb={d} rows={d} kind={d} bound={d}\n", .{ nb, rows, kind, bv });
                        return error.QsaSelect8BitMismatch;
                    }

                    if (rows <= QSA_SELECT_SPLIT_MAX_ROWS) {
                        const split8 = (try qsaSelectTopBlocksSplit(s, scores, bounds, k, rows, nb)) orelse return error.SelectKernelDeclined;
                        defer _ = mlx.mlx_array_free(split8);
                        if (!try qsaIdsBitEqual(s, ref11, split8)) {
                            std.debug.print("[qsa-select-8bit] split mismatch nb={d} rows={d} kind={d} bound={d}\n", .{ nb, rows, kind, bv });
                            return error.QsaSelect8BitMismatch;
                        }
                    }
                }
            }
        }
    }
}

test "qsa select tg policy: a pure function of rows, forced width wins" {
    const saved = qsa_select_tg_override;
    defer qsa_select_tg_override = saved;
    qsa_select_tg_override = null;
    try testing.expectEqual(@as(c_int, 256), qsaSelectTgFor(4096));
    try testing.expectEqual(@as(c_int, 256), qsaSelectTgFor(1024));
    try testing.expectEqual(@as(c_int, 256), qsaSelectTgFor(256));
    try testing.expectEqual(@as(c_int, 512), qsaSelectTgFor(255));
    try testing.expectEqual(@as(c_int, 512), qsaSelectTgFor(64));
    try testing.expectEqual(@as(c_int, 1024), qsaSelectTgFor(63));
    try testing.expectEqual(@as(c_int, 1024), qsaSelectTgFor(16));
    try testing.expectEqual(@as(c_int, 1024), qsaSelectTgFor(1));
    qsa_select_tg_override = 512;
    try testing.expectEqual(@as(c_int, 512), qsaSelectTgFor(4096));
    try testing.expectEqual(@as(c_int, 512), qsaSelectTgFor(1));
}
