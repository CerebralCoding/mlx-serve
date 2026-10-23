//! sdk/blocks/attention/kv_quant_attn, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns QKV_DEC_NT, qkvDecBlockFor, QKV_DEC_KERNEL_SOURCE, qkv_dec_kernel_cached, getQkvDecKernel, qkv_dec_override, and 52 more.

const std = @import("std");
const env = @import("../../../env.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const kv_quant = @import("../../../engines/mlx/cache/kv_quant.zig");
const log = @import("../../../host/log.zig");
const KVCache = @import("../../../engines/mlx/kv/cache.zig").KVCache;
const testing = @import("../../../engines/mlx/kv/cache.zig").testing;
const QSA_SCORE_KERNEL_HEADER = @import("../../../plugins/qwen4_exp/qsa/score.zig").QSA_SCORE_KERNEL_HEADER;
const verifyQmmNaxAvailable = @import("../linear/verify_qmm_nax.zig").verifyQmmNaxAvailable;
const verifyQmmTile = @import("../linear/verify_qmm_nax.zig").verifyQmmTile;
const DenseKVView = @import("kv_view.zig").DenseKVView;
const splitCausalSdpa = @import("sdpa.zig").splitCausalSdpa;

// ── Decode quantized-KV attention kernel (docs/kv-quant-perf.md) ──
//
// The composed fused path (quantAttention: qmm → softmax → qmm) measured
// SLOWER than dense-mode dequant+SDPA at real GQA ratios in the Phase 0
// µbench (48/8 @ 32K: composed 1.47 ms vs dense 1.11 vs kv-off 0.51) —
// mlx_quantized_matmul at M=g rows over batched 4-D banks runs ~6x below
// bandwidth. This kernel reads the packed cache views IN PLACE at decode
// width: one threadgroup per q head, NSG simdgroups striding the kv rows
// with per-simdgroup online-softmax carry (m/l/acc), merged once through
// threadgroup memory. The dense [B,H,T,D] K/V never exist, so per layer per
// token the traffic is exactly the packed bytes — BELOW kv-quant-off's dense
// read, which is the whole point of quantized KV at long context.
//
// GQA is handled in-kernel (head h_q reads bank h_q / GQA) — no repeat, no
// grouping reshape. K/V arrive as cache VIEWS (non-contiguous along the head
// axis whenever buffer capacity > logical length), hence
// ensure_row_contiguous=false + explicit stride indexing, same contract as
// msv_attn_p256. Dequant arithmetic mirrors gatherQmv's affine form
// (q*scale+bias per GS-group, f32 accumulate). NOT bit-identical to dense
// SDPA (different reduction order + f32 dequant vs bf16-rounded dense
// views) — sanctioned qmv-vs-qmm class, pinned no-worse-than-reference.
//
// Kill switches: MLX_SERVE_KV_ATTN_KERNEL=0 drops just this kernel (fused
// requests fall back to the composed path); MLX_SERVE_KV_ATTN_FUSED=0 kills
// the whole fused route (dense reads everywhere).
const QKV_DEC_NT = 128; // threads per threadgroup

/// Threadgroup block length along the kv axis, chosen so the score buffer
/// (GQA·BLOCK f32) + the staged q rows (GQA·DK f32) fit Metal's 32 KiB
/// threadgroup budget (a kernel's block size and its memory footprint are
/// ONE decision — the gdnBlockTFor class). Returns null when no block fits.
fn qkvDecBlockFor(gqa: c_int, dk: c_int) ?c_int {
    // ≤ ~10 KiB per threadgroup, NOT the full 32 KiB budget: threadgroup
    // memory is the residency limiter — a 28 KiB footprint leaves ONE
    // threadgroup per core and the latency-hiding collapses (measured
    // 1.9 ms vs dense 1.1 at 48/8 32K with 24 KiB score buffers).
    const cands = [_]c_int{ 512, 256, 128 };
    for (cands) |b| {
        const bytes: i64 = @as(i64, gqa) * @as(i64, b) * 4 + @as(i64, gqa) * @as(i64, dk) * 4 + QKV_DEC_NT * 4 + 256;
        if (bytes <= 10_000) return b;
    }
    return null;
}

// Two-phase block-parallel layout (v1 — one threadgroup per Q HEAD with a
// per-row simd_sum online-softmax chain — measured 5x SLOWER than dense
// mode: 48 threadgroups starve the GPU and the per-row reduction+exp chain
// serializes each simdgroup; see docs/kv-quant-perf.md Phase 0 table):
//
//   grid = (NT, H_kv, ceil(Tk / BLOCK))
//   Phase A — scores: thread t computes rows t, t+NT, … of the block: one
//     full-D dequant dot per (row, q-sub-head) against the g staged q rows.
//     No cross-thread reductions on the critical path.
//   Phase M — per sub-head block max + exp + row-weight sum via a shared
//     tree reduction in threadgroup memory.
//   Phase B — output: thread e walks ALL block rows for element e, dequants
//     V[t][e] ONCE and feeds all g sub-head accumulators (the packed bank is
//     read exactly once per block regardless of GQA ratio).
//
// The kernel emits per-block PARTIALS (m, l, unnormalized O) — the standard
// flash-attention split-KV carry — merged by a handful of KB-scale mlx ops
// in the dispatch wrapper. A fully-masked block clamps its max to -1e30 so
// exp never sees (-inf) - (-inf); its l = 0 partial merges to zero weight.
pub const QKV_DEC_KERNEL_SOURCE = kernelSource("kernels/attention/qkv_dec.metal");

var qkv_dec_kernel_cached: ?mlx.mlx_fast_metal_kernel = null;

fn getQkvDecKernel() !mlx.mlx_fast_metal_kernel {
    if (qkv_dec_kernel_cached) |kk| return kk;
    const input_names = [_][*:0]const u8{ "q", "kq", "ksc", "kbi", "vq", "vsc", "vbi", "scl", "tk", "msk" };
    const output_names = [_][*:0]const u8{ "m_out", "l_out", "o_out" };
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "qkv_attn_dec",
        in_vec,
        out_vec,
        QKV_DEC_KERNEL_SOURCE,
        "",
        false, // K/V are cache VIEWS (non-contiguous along the head axis) —
        // a forced contiguous copy of the packed cache would erase the win.
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    qkv_dec_kernel_cached = kernel;
    return kernel;
}

pub var qkv_dec_override: ?bool = null; // test seam
fn qkvDecKernelEnabled() bool {
    return qkv_dec_override orelse true;
}

const QkvDecCfgKey = struct {
    hq: c_int,
    dk: c_int,
    dv: c_int,
    bits: u8,
    gs: u32,
    gqa: c_int,
    nblk: c_int,
    has_mask: bool,
    dtype: mlx.mlx_dtype,
};
var qkv_dec_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var qkv_dec_cfg_key: QkvDecCfgKey = std.mem.zeroes(QkvDecCfgKey);
var qkv_dec_engaged: bool = false;

/// Try the decode-width quantized-KV attention kernel. Returns null when a
/// precondition doesn't hold — the caller falls back to the composed
/// quantAttention. Preconditions are the kernel's OWN requirements, never a
/// model_type list.
pub fn qkvAttnDecodeKernel(
    s: mlx.mlx_stream,
    q: mlx.mlx_array,
    view: *const DenseKVView,
    scale: f32,
    mask_mode: []const u8,
    mask_arr: mlx.mlx_array,
) !?mlx.mlx_array {
    if (!qkvDecKernelEnabled()) return null;
    if (!view.has_quant_triple) return null;
    if (!qkvVerifyForced() and mlx.mlx_array_ndim(q) == 4 and mlx.mlx_array_ndim(view.k_triple_q) == 4 and
        qkvMppWins(mlx.getShape(q)[2], mlx.getShape(view.k_triple_q)[2]))
    {
        if (try qkvAttnMppKernel(s, q, view, scale, mask_mode)) |o| return o;
    }
    if (view.bits != 4 and view.bits != 8) return null;
    if (view.group_size == 0) return null;
    if (mlx.mlx_array_ndim(q) != 4) return null;
    const qs = mlx.getShape(q);
    if (qs[0] != 1 or qs[2] != 1) return null; // decode width only; batched decode keeps dense reads
    const qdt = mlx.mlx_array_dtype(q);
    if (qdt != .bfloat16 and qdt != .float16 and qdt != .float32) return null;
    if (mlx.mlx_array_ndim(view.k_triple_q) != 4 or mlx.mlx_array_ndim(view.v_triple_q) != 4) return null;
    const ks = mlx.getShape(view.k_triple_q);
    const vs = mlx.getShape(view.v_triple_q);
    const vpw: c_int = @divExact(@as(c_int, 32), @as(c_int, view.bits));
    const dk: c_int = ks[3] * vpw;
    const dv: c_int = vs[3] * vpw;
    if (qs[3] != dk) return null;
    if (@rem(dk, 32) != 0 or @rem(dv, 32) != 0) return null;
    if (@rem(dk, @as(c_int, @intCast(view.group_size))) != 0 or @rem(dv, @as(c_int, @intCast(view.group_size))) != 0) return null;
    const h_q: c_int = qs[1];
    const h_kv: c_int = ks[1];
    if (h_kv <= 0 or @rem(h_q, h_kv) != 0) return null;
    if (vs[1] != h_kv or vs[2] != ks[2] or ks[0] != 1 or vs[0] != 1) return null;
    const gqa: c_int = @divExact(h_q, h_kv);
    const t_k: c_int = ks[2];
    if (t_k <= 0) return null;
    // Scales/biases layout must match the (dk/gs) geometry.
    const ksc_sh = mlx.getShape(view.k_triple_scales);
    const vsc_sh = mlx.getShape(view.v_triple_scales);
    if (ksc_sh.len != 4 or vsc_sh.len != 4) return null;
    if (ksc_sh[3] != @divExact(dk, @as(c_int, @intCast(view.group_size)))) return null;
    if (vsc_sh[3] != @divExact(dv, @as(c_int, @intCast(view.group_size)))) return null;

    // Mask: none, trivially-causal (T_q == 1), or a [.., 1, T_k] additive row.
    var has_mask = false;
    if (std.mem.eql(u8, mask_mode, "array")) {
        if (mask_arr.ctx == null) return null;
        const msize = mlx.mlx_array_size(mask_arr);
        if (msize != @as(usize, @intCast(t_k))) return null;
        has_mask = true;
    } else if (!std.mem.eql(u8, mask_mode, "") and !std.mem.eql(u8, mask_mode, "causal")) {
        return null;
    }

    const kernel = getQkvDecKernel() catch return null;

    const block = qkvDecBlockFor(gqa, dk) orelse return null;
    const nblk: c_int = @divTrunc(t_k + block - 1, block);

    const key = QkvDecCfgKey{
        .hq = h_q,
        .dk = dk,
        .dv = dv,
        .bits = view.bits,
        .gs = view.group_size,
        .gqa = gqa,
        .nblk = nblk,
        .has_mask = has_mask,
        .dtype = qdt,
    };
    if (qkv_dec_cfg == null or !std.meta.eql(qkv_dec_cfg_key, key)) {
        if (qkv_dec_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const config = mlx.mlx_fast_metal_kernel_config_new();
        const ml_shape = [_]c_int{ h_q, nblk };
        const o_shape = [_]c_int{ h_q, nblk, dv };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &ml_shape, 2, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &ml_shape, 2, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &o_shape, 3, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, QKV_DEC_NT, h_kv, nblk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, QKV_DEC_NT, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", qdt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BITS", @intCast(view.bits)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GS", @intCast(view.group_size)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "DK", dk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "DV", dv));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GQA", gqa));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BLOCK", block));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "HAS_MASK", @intFromBool(has_mask)));
        qkv_dec_cfg = config;
        qkv_dec_cfg_key = key;
    }

    const one = [_]c_int{1};
    const scl_data = [_]f32{scale};
    const scl = mlx.mlx_array_new_data(&scl_data, &one, 1, .float32);
    defer _ = mlx.mlx_array_free(scl);
    const tk_data = [_]i32{t_k};
    const tk = mlx.mlx_array_new_data(&tk_data, &one, 1, .int32);
    defer _ = mlx.mlx_array_free(tk);
    // Absent mask: a 1-elem f32 stand-in (HAS_MASK=0 never reads it).
    const dummy_data = [_]f32{0};
    const dummy = mlx.mlx_array_new_data(&dummy_data, &one, 1, .float32);
    defer _ = mlx.mlx_array_free(dummy);

    const inputs_arr = [_]mlx.mlx_array{
        q,
        view.k_triple_q,
        view.k_triple_scales,
        view.k_triple_biases,
        view.v_triple_q,
        view.v_triple_scales,
        view.v_triple_biases,
        scl,
        tk,
        if (has_mask) mask_arr else dummy,
    };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, qkv_dec_cfg.?, s));
    const o_t = try qkvMergePartials(s, outputs_vec, qdt);
    defer _ = mlx.mlx_array_free(o_t);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    const out_shape = [_]c_int{ 1, h_q, 1, dv };
    try mlx.check(mlx.mlx_reshape(&out, o_t, &out_shape, 4, s));
    if (!qkv_dec_engaged) {
        qkv_dec_engaged = true;
        log.info("[kv-attn] decode kernel engaged: bits={d} gs={d} DK={d} DV={d} Hq={d} Hkv={d} block={d} mask={} (MLX_SERVE_KV_ATTN_KERNEL=0 restores composed)\n", .{ view.bits, view.group_size, dk, dv, h_q, h_kv, block, has_mask });
    }
    return out;
}

/// Merge per-block split-KV partials (m, l, unnormalized O) into a typed
/// [ROWS, DV] tensor — the standard flash-attention carry:
///   M  = max_b m_b            [ROWS, 1]
///   w  = exp(m_b - M)         [ROWS, NBLK]
///   L  = Σ_b l_b · w          [ROWS, 1]
///   O  = Σ_b o_b · w          [ROWS, DV]
///   out = (O / L) as T.
/// KB-scale tensors — a handful of elementwise/reduce ops per layer. Shared
/// by the decode and verify packed-attention kernels; the op sequence is
/// load-bearing (decode bit-identity across refactors is a bar).
fn qkvMergePartials(s: mlx.mlx_stream, outputs_vec: mlx.mlx_vector_array, qdt: mlx.mlx_dtype) !mlx.mlx_array {
    if (mlx.mlx_vector_array_size(outputs_vec) != 3) return error.MetalKernelBadOutputCount;
    var m_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(m_b);
    var l_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(l_b);
    var o_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(o_b);
    try mlx.check(mlx.mlx_vector_array_get(&m_b, outputs_vec, 0));
    try mlx.check(mlx.mlx_vector_array_get(&l_b, outputs_vec, 1));
    try mlx.check(mlx.mlx_vector_array_get(&o_b, outputs_vec, 2));
    return qkvMergeArrays(s, m_b, l_b, o_b, qdt);
}

pub fn qkvMergeArrays(s: mlx.mlx_stream, m_b: mlx.mlx_array, l_b: mlx.mlx_array, o_b: mlx.mlx_array, qdt: mlx.mlx_dtype) !mlx.mlx_array {
    var m_max = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(m_max);
    try mlx.check(mlx.mlx_max_axis(&m_max, m_b, 1, true, s));
    var m_shift = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(m_shift);
    try mlx.check(mlx.mlx_subtract(&m_shift, m_b, m_max, s));
    var w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w);
    try mlx.check(mlx.mlx_exp(&w, m_shift, s));
    var lw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lw);
    try mlx.check(mlx.mlx_multiply(&lw, l_b, w, s));
    var l_tot = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(l_tot);
    try mlx.check(mlx.mlx_sum_axis(&l_tot, lw, 1, true, s)); // [ROWS, 1]
    var w3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w3);
    try mlx.check(mlx.mlx_expand_dims(&w3, w, 2, s)); // [ROWS, NBLK, 1]
    var ow = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ow);
    try mlx.check(mlx.mlx_multiply(&ow, o_b, w3, s));
    var o_tot = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(o_tot);
    try mlx.check(mlx.mlx_sum_axis(&o_tot, ow, 1, false, s)); // [ROWS, DV]
    var o_norm = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(o_norm);
    try mlx.check(mlx.mlx_divide(&o_norm, o_tot, l_tot, s)); // [ROWS,DV] / [ROWS,1]
    var o_t = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(o_t);
    try mlx.check(mlx.mlx_astype(&o_t, o_norm, qdt, s));
    return o_t;
}

// ── Verify-width packed attention (kv8-fused-spec-plan.md Phase 2) ──
//
// Same two-phase block-parallel layout as QKV_DEC_KERNEL_SOURCE, generalized
// to TQ (2..8) query rows per head — the spec-verify widths that Phase 1
// routed to dense dequant + SDPA. Differences from the decode kernel:
//   - q is read from DEVICE memory, not staged: LQ = GQA·TQ rows of DK f32
//     is up to 49 KB of threadgroup memory at TQ 8. The reads are lockstep
//     across threads (same element for every row of the block) and the whole
//     q tensor is L1-scale.
//   - the causal TAIL is applied from geometry: q row tq sees keys with
//     global index < Tk - TQ + 1 + tq — the same end-aligned contract as
//     SDPA "causal", which is what the dense path uses at these widths.
//     "array" masks (sliding layers) decline to dense.
//   - scale is folded at score write, not at q read.
// Partials merge through the SAME qkvMergePartials carry; row order is
// (head-major, tq inner) so the merged [Hq·TQ, DV] reshapes straight to
// [1, Hq, TQ, DV].
pub const QKV_VERIFY_MAX_TQ: c_int = 8;

pub const QKV_VERIFY_KERNEL_SOURCE = kernelSource("kernels/attention/qkv_verify.metal");

var qkv_ver_kernel_cached: ?mlx.mlx_fast_metal_kernel = null;

fn getQkvVerifyKernel() !mlx.mlx_fast_metal_kernel {
    if (qkv_ver_kernel_cached) |kk| return kk;
    const input_names = [_][*:0]const u8{ "q", "kq", "ksc", "kbi", "vq", "vsc", "vbi", "scl", "tk" };
    const output_names = [_][*:0]const u8{ "m_out", "l_out", "o_out" };
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "qkv_attn_verify",
        in_vec,
        out_vec,
        QKV_VERIFY_KERNEL_SOURCE,
        "",
        false, // K/V are cache VIEWS — see getQkvDecKernel.
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    qkv_ver_kernel_cached = kernel;
    return kernel;
}

pub var qkv_ver_override: ?bool = null; // test seam
var qkv_ver_env: ?bool = null;
fn qkvVerifyKernelEnabledFrom(nax_available: bool, raw: ?[]const u8) bool {
    if (raw) |v| return !std.mem.eql(u8, v, "0");
    // The dk<=256 adoption set was calibrated on M4. The M5/G17 round found
    // it slower at 9K KV for q4/q6/q8 targets, so G17 keeps dense verify
    // attention by default while retaining the explicit force-on QA lever.
    return !nax_available;
}

/// The kernel keeps one accumulator per q row (gqa x t_q); past 12 they spill
/// out of registers and dequant + dense sdpa wins at every kv length.
const QKV_VERIFY_MAX_ROWS: c_int = 12;
fn qkvVerifyRowsPay(gqa: c_int, t_q: c_int) bool {
    return gqa * t_q <= QKV_VERIFY_MAX_ROWS;
}

/// `MLX_SERVE_KV_ATTN_VERIFY=0` still kills the whole verify width class.
fn qkvVerifyEnvAllows() bool {
    if (qkv_ver_override) |v| return v;
    const raw = env.get(.MLX_SERVE_KV_ATTN_VERIFY) orelse return true;
    return !std.mem.eql(u8, std.mem.sliceTo(raw, 0), "0");
}

/// MLX_SERVE_KV_ATTN_VERIFY=1 (and the test seam) keeps every width on the kernel.
fn qkvVerifyForced() bool {
    if (qkv_ver_override) |v| return v;
    const raw = env.get(.MLX_SERVE_KV_ATTN_VERIFY) orelse return false;
    return std.mem.eql(u8, std.mem.sliceTo(raw, 0), "1");
}

fn qkvVerifyKernelEnabled() bool {
    if (qkv_ver_override) |v| return v;
    if (qkv_ver_env) |v| return v;
    const raw = env.get(.MLX_SERVE_KV_ATTN_VERIFY);
    const enabled = qkvVerifyKernelEnabledFrom(
        verifyQmmNaxAvailable(),
        if (raw) |v| std.mem.sliceTo(v, 0) else null,
    );
    qkv_ver_env = enabled;
    return enabled;
}

/// Threadgroup budget for the verify kernel: scores only (q is device-read).
/// 16 KB rather than decode's 10 KB — at TQ x the arithmetic intensity the
/// occupancy loss buys more reuse per byte; the live A/B is the judge.
fn qkvVerifyBlockFor(gqa: c_int, tq: c_int) ?c_int {
    const lq: i64 = @as(i64, gqa) * @as(i64, tq);
    const cands = [_]c_int{ 256, 128, 64 };
    for (cands) |b| {
        const bytes: i64 = lq * @as(i64, b) * 4 + QKV_DEC_NT * 4 + 8 * lq + 256;
        if (bytes <= 16_000) return b;
    }
    return null;
}

const QkvVerCfgKey = struct {
    hq: c_int,
    dk: c_int,
    dv: c_int,
    bits: u8,
    gs: u32,
    gqa: c_int,
    tq: c_int,
    nblk: c_int,
    dtype: mlx.mlx_dtype,
};
// Per-TQ config slots: verify widths alternate round to round (partial MTP
// rounds, variable PLD match lengths), so a single-entry cache would rebuild
// per call — the ShapeKey lesson, one slot per width instead.
var qkv_ver_cfgs: [@intCast(QKV_VERIFY_MAX_TQ + 1)]?mlx.mlx_fast_metal_kernel_config = @splat(null);
var qkv_ver_cfg_keys: [@intCast(QKV_VERIFY_MAX_TQ + 1)]QkvVerCfgKey = @splat(std.mem.zeroes(QkvVerCfgKey));
var qkv_ver_engaged_mask: u16 = 0; // one-shot log per TQ

/// Try the verify-width packed-KV attention kernel (T_q 2..8, causal only).
/// Returns null when a precondition or machine adoption gate doesn't hold —
/// the caller falls back to dense SDPA. G17 defaults dense after its q4/q6/q8
/// A/B; MLX_SERVE_KV_ATTN_VERIFY=1 retains an explicit force-on QA arm.
pub fn qkvAttnVerifyKernel(
    s: mlx.mlx_stream,
    q: mlx.mlx_array,
    view: *const DenseKVView,
    scale: f32,
    mask_mode: []const u8,
) !?mlx.mlx_array {
    if (!qkvVerifyForced() and view.has_quant_triple and mlx.mlx_array_ndim(q) == 4 and mlx.mlx_array_ndim(view.k_triple_q) == 4) {
        if (qkvMppWins(mlx.getShape(q)[2], mlx.getShape(view.k_triple_q)[2])) {
            if (try qkvAttnMppKernel(s, q, view, scale, mask_mode)) |o| return o;
        }
    }
    if (!qkvVerifyKernelEnabled()) return null;
    if (!view.has_quant_triple) return null;
    if (view.bits != 4 and view.bits != 8) return null;
    if (view.group_size == 0) return null;
    if (mlx.mlx_array_ndim(q) != 4) return null;
    const qs = mlx.getShape(q);
    const t_q: c_int = qs[2];
    if (qs[0] != 1 or t_q < 2 or t_q > QKV_VERIFY_MAX_TQ) return null;
    // The dense path masks these widths "causal" (end-aligned); the kernel
    // applies exactly that tail from geometry. Anything else declines.
    if (!std.mem.eql(u8, mask_mode, "causal")) return null;
    const qdt = mlx.mlx_array_dtype(q);
    if (qdt != .bfloat16 and qdt != .float16 and qdt != .float32) return null;
    if (mlx.mlx_array_ndim(view.k_triple_q) != 4 or mlx.mlx_array_ndim(view.v_triple_q) != 4) return null;
    const ks = mlx.getShape(view.k_triple_q);
    const vs = mlx.getShape(view.v_triple_q);
    const vpw: c_int = @divExact(@as(c_int, 32), @as(c_int, view.bits));
    const dk: c_int = ks[3] * vpw;
    const dv: c_int = vs[3] * vpw;
    if (qs[3] != dk) return null;
    if (@rem(dk, 32) != 0 or @rem(dv, 32) != 0) return null;
    // Shape gate is MEASURED adoption (the mixedNaxShapeEnabled rule): every
    // win shape is dk <= 256 (qwen3_5 24/4 hd256: +4..+7% over dense at
    // 8k/16k/32k spec-on); gemma4's full-attn gqa16 x dk512 measured a
    // consistent −1% at Tq 2 (2026-08-15) and reads dense instead.
    if (dk > 256) return null;
    if (@rem(dk, @as(c_int, @intCast(view.group_size))) != 0 or @rem(dv, @as(c_int, @intCast(view.group_size))) != 0) return null;
    const h_q: c_int = qs[1];
    const h_kv: c_int = ks[1];
    if (h_kv <= 0 or @rem(h_q, h_kv) != 0) return null;
    if (vs[1] != h_kv or vs[2] != ks[2] or ks[0] != 1 or vs[0] != 1) return null;
    const gqa: c_int = @divExact(h_q, h_kv);
    const lq: c_int = gqa * t_q;
    if (lq > QKV_DEC_NT) return null; // the m/l writer covers one row per thread
    if (!qkvVerifyRowsPay(gqa, t_q) and !qkvVerifyForced()) return null;
    const t_k: c_int = ks[2];
    if (t_k < t_q) return null;
    const ksc_sh = mlx.getShape(view.k_triple_scales);
    const vsc_sh = mlx.getShape(view.v_triple_scales);
    if (ksc_sh.len != 4 or vsc_sh.len != 4) return null;
    if (ksc_sh[3] != @divExact(dk, @as(c_int, @intCast(view.group_size)))) return null;
    if (vsc_sh[3] != @divExact(dv, @as(c_int, @intCast(view.group_size)))) return null;

    const kernel = getQkvVerifyKernel() catch return null;
    const block = qkvVerifyBlockFor(gqa, t_q) orelse return null;
    const nblk: c_int = @divTrunc(t_k + block - 1, block);

    const slot: usize = @intCast(t_q);
    const key = QkvVerCfgKey{
        .hq = h_q,
        .dk = dk,
        .dv = dv,
        .bits = view.bits,
        .gs = view.group_size,
        .gqa = gqa,
        .tq = t_q,
        .nblk = nblk,
        .dtype = qdt,
    };
    if (qkv_ver_cfgs[slot] == null or !std.meta.eql(qkv_ver_cfg_keys[slot], key)) {
        if (qkv_ver_cfgs[slot]) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const config = mlx.mlx_fast_metal_kernel_config_new();
        const rows_total: c_int = h_q * t_q;
        const ml_shape = [_]c_int{ rows_total, nblk };
        const o_shape = [_]c_int{ rows_total, nblk, dv };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &ml_shape, 2, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &ml_shape, 2, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &o_shape, 3, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, QKV_DEC_NT, h_kv, nblk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, QKV_DEC_NT, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", qdt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BITS", @intCast(view.bits)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GS", @intCast(view.group_size)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "DK", dk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "DV", dv));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GQA", gqa));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "TQ", t_q));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BLOCK", block));
        qkv_ver_cfgs[slot] = config;
        qkv_ver_cfg_keys[slot] = key;
    }

    const one = [_]c_int{1};
    const scl_data = [_]f32{scale};
    const scl = mlx.mlx_array_new_data(&scl_data, &one, 1, .float32);
    defer _ = mlx.mlx_array_free(scl);
    const tk_data = [_]i32{t_k};
    const tk = mlx.mlx_array_new_data(&tk_data, &one, 1, .int32);
    defer _ = mlx.mlx_array_free(tk);

    const inputs_arr = [_]mlx.mlx_array{
        q,
        view.k_triple_q,
        view.k_triple_scales,
        view.k_triple_biases,
        view.v_triple_q,
        view.v_triple_scales,
        view.v_triple_biases,
        scl,
        tk,
    };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, qkv_ver_cfgs[slot].?, s));
    const o_t = try qkvMergePartials(s, outputs_vec, qdt);
    defer _ = mlx.mlx_array_free(o_t);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    const out_shape = [_]c_int{ 1, h_q, t_q, dv };
    try mlx.check(mlx.mlx_reshape(&out, o_t, &out_shape, 4, s));
    const bit: u16 = @as(u16, 1) << @intCast(t_q);
    if (qkv_ver_engaged_mask & bit == 0) {
        qkv_ver_engaged_mask |= bit;
        log.info("[kv-attn] verify kernel engaged: Tq={d} bits={d} gs={d} DK={d} DV={d} Hq={d} Hkv={d} block={d} (G17 defaults dense; MLX_SERVE_KV_ATTN_VERIFY=1 forces this arm, =0 forces dense)\n", .{ t_q, view.bits, view.group_size, dk, dv, h_q, h_kv, block });
    }
    return out;
}

// ── matmul2d packed attention (t_q 1..8) ──
//
// One threadgroup (256 threads, 8 simdgroups) per (kv head, kv split). Per 32-token page:
// dequantize K into a threadgroup tile, QK through the matmul2d tensor op, an online
// softmax per q row, dequantize V into the same tile, PV accumulated into a cooperative
// running output that is rescaled when a row's max grows. Each split emits the same
// (m, l, unnormalized O) partials as the other packed kernels. Structure after Inco
// Splash's q8 attention tile (Apache-2.0, see NOTICE).
const QKV_MPP_PAGE: c_int = 32;
const QKV_MPP_MAX_SPLITS: c_int = 128;
const QKV_MPP_PAGES_PER_SPLIT: c_int = 8;

pub const QKV_MPP_KERNEL_SOURCE = kernelSource("kernels/attention/qkv_mpp.metal");

var qkv_mpp_kernel_cached: ?mlx.mlx_fast_metal_kernel = null;
const QkvMppCfgKey = struct { hq: c_int, dk: c_int, bits: u8, gs: u32, gqa: c_int, tq: c_int, nsplit: c_int, dtype: mlx.mlx_dtype };
var qkv_mpp_cfgs: [@intCast(QKV_VERIFY_MAX_TQ + 1)]?mlx.mlx_fast_metal_kernel_config = @splat(null);
var qkv_mpp_cfg_keys: [@intCast(QKV_VERIFY_MAX_TQ + 1)]QkvMppCfgKey = @splat(std.mem.zeroes(QkvMppCfgKey));
var qkv_mpp_engaged: bool = false;

/// Measured window (M4 Max, 24/4 hd 256 kv8): the matmul2d kernel's per-page fixed cost
/// needs rows or length to amortize.
pub fn qkvMppWins(t_q: c_int, t_k: c_int) bool {
    if (t_q >= 4) return t_k >= 2048;
    if (t_q >= 2) return t_k >= 8192;
    return t_k >= 16384;
}

pub fn qkvMppSplits(t_k: c_int) c_int {
    const pages = @divTrunc(t_k + QKV_MPP_PAGE - 1, QKV_MPP_PAGE);
    return std.math.clamp(@divTrunc(pages, QKV_MPP_PAGES_PER_SPLIT), 1, QKV_MPP_MAX_SPLITS);
}

/// Packed-KV attention through matmul2d, t_q 1..8, no mask or end-aligned causal. Null = declined.
pub fn qkvAttnMppKernel(s: mlx.mlx_stream, q_in: mlx.mlx_array, view: *const DenseKVView, scale: f32, mask_mode: []const u8) !?mlx.mlx_array {
    // Same machines as the verify-qmm matmul2d tile: M5 matrix units or M4-family shader code.
    if (verifyQmmTile() == .off) return null;
    if (!view.has_quant_triple or view.group_size == 0) return null;
    if (view.bits != 4 and view.bits != 8) return null;
    if (mlx.mlx_array_ndim(q_in) != 4) return null;
    const qs = mlx.getShape(q_in);
    const t_q: c_int = qs[2];
    if (qs[0] != 1 or t_q < 1 or t_q > QKV_VERIFY_MAX_TQ) return null;
    const causal = std.mem.eql(u8, mask_mode, "causal");
    if (!causal and !(t_q == 1 and mask_mode.len == 0)) return null;
    const qdt = mlx.mlx_array_dtype(q_in);
    if (qdt != .bfloat16 and qdt != .float16) return null;
    if (mlx.mlx_array_ndim(view.k_triple_q) != 4 or mlx.mlx_array_ndim(view.v_triple_q) != 4) return null;
    const ks = mlx.getShape(view.k_triple_q);
    const vs = mlx.getShape(view.v_triple_q);
    const vpw: c_int = @divExact(@as(c_int, 32), @as(c_int, view.bits));
    const dk: c_int = ks[3] * vpw;
    if (qs[3] != dk or vs[3] != ks[3] or dk != 256) return null;
    const gs: c_int = @intCast(view.group_size);
    if (@rem(dk, gs) != 0 or @divExact(dk, gs) > 8) return null;
    const h_q: c_int = qs[1];
    const h_kv: c_int = ks[1];
    if (h_kv <= 0 or @rem(h_q, h_kv) != 0 or vs[1] != h_kv or vs[2] != ks[2] or ks[0] != 1) return null;
    const gqa: c_int = @divExact(h_q, h_kv);
    if (@divTrunc(gqa * t_q + 7, 8) * 8 > 56) return null; // past 56 padded rows the threadgroup arrays exceed Metal's 32 KiB
    const t_k: c_int = ks[2];
    if (t_k < t_q) return null;

    if (qkv_mpp_kernel_cached == null) {
        const input_names = [_][*:0]const u8{ "q", "kq", "ksc", "kbi", "vq", "vsc", "vbi", "scl", "tk" };
        const output_names = [_][*:0]const u8{ "m_out", "l_out", "o_out" };
        const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
        defer _ = mlx.mlx_vector_string_free(in_vec);
        const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
        defer _ = mlx.mlx_vector_string_free(out_vec);
        const kk = mlx.mlx_fast_metal_kernel_new("msv_qkv_mpp", in_vec, out_vec, QKV_MPP_KERNEL_SOURCE, QSA_SCORE_KERNEL_HEADER, false, false);
        if (kk.ctx == null) return null;
        qkv_mpp_kernel_cached = kk;
    }
    const nsplit = qkvMppSplits(t_k);
    const real_rows: c_int = gqa * t_q;
    const m_rows: c_int = @divTrunc(real_rows + 7, 8) * 8;
    const slot: usize = @intCast(t_q);
    const key = QkvMppCfgKey{ .hq = h_q, .dk = dk, .bits = view.bits, .gs = view.group_size, .gqa = gqa, .tq = t_q, .nsplit = nsplit, .dtype = qdt };
    if (qkv_mpp_cfgs[slot] == null or !std.meta.eql(qkv_mpp_cfg_keys[slot], key)) {
        if (qkv_mpp_cfgs[slot]) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const config = mlx.mlx_fast_metal_kernel_config_new();
        const ml_shape = [_]c_int{ h_kv, nsplit, m_rows };
        const o_shape = [_]c_int{ h_kv, nsplit, m_rows, dk };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &ml_shape, 3, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &ml_shape, 3, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &o_shape, 4, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 256, h_kv, nsplit));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", qdt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BITS", @intCast(view.bits)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GS", gs));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "DK", dk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GQA", gqa));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "TQ", t_q));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "MPAD", m_rows));
        qkv_mpp_cfgs[slot] = config;
        qkv_mpp_cfg_keys[slot] = key;
    }

    // [1, Hq, TQ, D] is already (kv head, group head, row)-major: [Hkv, real_rows, D], zero-padded.
    var q3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q3);
    const q3_shape = [_]c_int{ h_kv, real_rows, dk };
    try mlx.check(mlx.mlx_reshape(&q3, q_in, &q3_shape, 3, s));
    var qp = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(qp);
    {
        const zero = mlx.mlx_array_new_float(0.0);
        defer _ = mlx.mlx_array_free(zero);
        const axes = [_]c_int{1};
        const lo = [_]c_int{0};
        const hi = [_]c_int{m_rows - real_rows};
        try mlx.check(mlx.mlx_pad(&qp, q3, &axes, 1, &lo, 1, &hi, 1, zero, "constant", s));
    }
    var q = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q);
    try mlx.check(mlx.mlx_contiguous(&q, qp, false, s));
    const one = [_]c_int{1};
    const scl_data = [_]f32{scale};
    const scl = mlx.mlx_array_new_data(&scl_data, &one, 1, .float32);
    defer _ = mlx.mlx_array_free(scl);
    const tk_data = [_]i32{t_k};
    const tk = mlx.mlx_array_new_data(&tk_data, &one, 1, .int32);
    defer _ = mlx.mlx_array_free(tk);
    const inputs_arr = [_]mlx.mlx_array{ q, view.k_triple_q, view.k_triple_scales, view.k_triple_biases, view.v_triple_q, view.v_triple_scales, view.v_triple_biases, scl, tk };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, qkv_mpp_kernel_cached.?, inputs_vec, qkv_mpp_cfgs[slot].?, s));

    // [Hkv, NS, M(, D)] -> [Hkv*M, NS(, D)] for the shared merge.
    var parts: [3]mlx.mlx_array = undefined;
    var n_parts: usize = 0;
    defer for (parts[0..n_parts]) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (0..3) |i| {
        var raw = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(raw);
        try mlx.check(mlx.mlx_vector_array_get(&raw, outputs_vec, i));
        var sw = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sw);
        const ax4 = [_]c_int{ 0, 2, 1, 3 };
        try mlx.check(mlx.mlx_transpose_axes(&sw, raw, &ax4, if (i == 2) 4 else 3, s));
        // Drop the pad rows: [Hkv, M, NS(, D)] -> [Hkv, real_rows, ...].
        var cut = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cut);
        const start = [_]c_int{ 0, 0, 0, 0 };
        const stop = [_]c_int{ h_kv, real_rows, nsplit, dk };
        const strd = [_]c_int{ 1, 1, 1, 1 };
        const nd: usize = if (i == 2) 4 else 3;
        try mlx.check(mlx.mlx_slice(&cut, sw, &start, nd, &stop, nd, &strd, nd, s));
        var flat = mlx.mlx_array_new();
        const sh3 = [_]c_int{ h_kv * real_rows, nsplit, dk };
        try mlx.check(mlx.mlx_reshape(&flat, cut, &sh3, if (i == 2) 3 else 2, s));
        parts[n_parts] = flat;
        n_parts += 1;
    }
    const o_t = try qkvMergeArrays(s, parts[0], parts[1], parts[2], qdt);
    defer _ = mlx.mlx_array_free(o_t);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    const out_shape = [_]c_int{ 1, h_q, t_q, dk };
    try mlx.check(mlx.mlx_reshape(&out, o_t, &out_shape, 4, s));
    if (!qkv_mpp_engaged) {
        qkv_mpp_engaged = true;
        log.info("[kv-attn] matmul2d packed attention engaged: bits={d} gs={d} Hq={d} Hkv={d}\n", .{ view.bits, view.group_size, h_q, h_kv });
    }
    return out;
}

// ── Fused quant-attention gate (docs/kv-quant-perf.md Phase 1) ──


/// Per-LAYER kv-length floor: below this the packed read cannot pay for the
/// kernel's split-KV merge ops (~11 dispatches/layer) — Laguna's 512-window
/// sliding layers measured a net −2% when they fused, and gemma4-12B's
/// 1024-window sliding layers sat exactly AT the old 1024 floor and measured
/// a ~4 tok/s decode loss fused (masked kernel at ~1k KV vs near-free dense
/// SDPA, 2026-08-15). Short-KV layers keep dense reads even inside a fused
/// request. MLX_SERVE_KV_ATTN_MIN_TK overrides (correctness tests set 1 so
/// short prompts still exercise the fused paths; the default is a PERF
/// gate, not a correctness gate).
pub const KV_ATTN_FUSED_MIN_TK: c_int = 2048;
var kv_attn_min_tk_env: ?c_int = null;
fn kvAttnFusedMinTk() c_int {
    if (kv_attn_min_tk_env) |v| return v;
    const v: c_int = blk: {
        const raw = env.get(.MLX_SERVE_KV_ATTN_MIN_TK) orelse break :blk KV_ATTN_FUSED_MIN_TK;
        break :blk std.fmt.parseInt(c_int, std.mem.sliceTo(raw, 0), 10) catch KV_ATTN_FUSED_MIN_TK;
    };
    kv_attn_min_tk_env = v;
    return v;
}

/// The ONE eligibility gate every fused quant-attention call site reads
/// (`forwardStandardWith`, `gatedFullAttnWith`, `lagunaAttnWith`).
/// `forwardBatchedDecode` deliberately keeps dense-mode reads: batched decode
/// is the packed-words-to-SDPA server-kill class (guarded by
/// tests/test_batched_equivalence.sh).
///
/// Decode width (t_q == 1) ONLY: that is the width `qkvAttnDecodeKernel`
/// serves (the measured +10..+56% serial win). Verify widths (T_q 2..32
/// under MTP/spec) used to be admitted and fell to the composed qmm chain —
/// 2 packed matmuls x every attention layer per spec round at qmm's M 2..7
/// dead zone, measured a 2.1x decode LOSS at 8k+ (2026-08-15). They read
/// the dense view now, like prefill always has. A forced per-request
/// `kv_attn_mode:"fused"` gets the dense read at those widths too.
pub fn kvAttnFusedEligible(view: *const DenseKVView, t_q: c_int) bool {
    if (!(view.has_quant_triple and t_q == 1)) return false;
    const ks = mlx.getShape(view.k_triple_q);
    if (ks.len < 4) return false;
    return ks[2] >= kvAttnFusedMinTk();
}

/// Verify-width packed-attention gate (Phase 2): T_q 2..8 served by
/// `qkvAttnVerifyKernel`, same triple + per-layer kv floor as decode.
/// Wider multi-token forwards (real prefill chunks) stay on dense SDPA;
/// G17 defaults dense; MLX_SERVE_KV_ATTN_VERIFY=1 forces the kernel and =0
/// kills the whole width class on every machine.
pub fn kvAttnVerifyEligible(view: *const DenseKVView, t_q: c_int) bool {
    if (!(view.has_quant_triple and t_q >= 2 and t_q <= QKV_VERIFY_MAX_TQ and qkvVerifyEnvAllows())) return false;
    const ks = mlx.getShape(view.k_triple_q);
    if (ks.len < 4) return false;
    return ks[2] >= kvAttnFusedMinTk();
}

var kv_attn_fused_engaged: bool = false; // one-shot log guard
pub fn logKvAttnFusedEngaged(view: *const DenseKVView, q: mlx.mlx_array, t_q: c_int) void {
    if (kv_attn_fused_engaged) return;
    kv_attn_fused_engaged = true;
    const q_shape = mlx.getShape(q);
    const h_q: c_int = if (q_shape.len >= 4) q_shape[1] else 0;
    const k_shape = mlx.getShape(view.k_triple_q);
    const h_kv: c_int = if (k_shape.len >= 4) k_shape[1] else 0;
    log.info("[kv-attn] fused engaged: bits={d} gs={d} Hq={d} Hkv={d} Tq={d} (MLX_SERVE_KV_ATTN_FUSED=0 or --kv-attn-mode dense restores dense reads)\n", .{ view.bits, view.group_size, h_q, h_kv, t_q });
}

// ── tests ──

/// Shared harness for the decode quantized-KV kernel parity tests: prefill
/// `t_prefill` tokens + one decode token through a real affine KVCache (so
/// the views carry the REAL strided layout, capacity > logical length), then
/// compare the kernel against dense SDPA over the SAME dequantized views.
/// No-worse-than-reference (quantization error folds into both paths); the
/// kernel is NOT expected to bit-match (f32 in-register dequant vs the
/// bf16-rounded dense views + different reduction order — qmv class).
fn qkvDecParityCase(h_q: c_int, h_kv: c_int, d: c_int, bits: u8, t_prefill: usize, masked: bool) !void {
    const s = mlx.gpuStream();
    var cache = try KVCache.initWithConfig(testing.allocator, 1, kv_quant.KVQuantConfig.affine(bits));
    defer cache.deinit();

    // Smooth ramp values (quantization-friendly but non-trivial).
    const mkArr = struct {
        fn f(str: mlx.mlx_stream, hh: c_int, tt: usize, dd: c_int, seed: f32) !mlx.mlx_array {
            const total: usize = @intCast(@as(usize, @intCast(hh)) * tt * @as(usize, @intCast(dd)));
            const buf = try testing.allocator.alloc(f32, total);
            defer testing.allocator.free(buf);
            for (buf, 0..) |*x, i| {
                const fi: f32 = @floatFromInt(i);
                x.* = @sin(fi * 0.37 + seed) * 0.8;
            }
            const shape = [_]c_int{ 1, hh, @intCast(tt), dd };
            const f32_arr = mlx.mlx_array_new_data(buf.ptr, &shape, 4, .float32);
            defer _ = mlx.mlx_array_free(f32_arr);
            var bf = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&bf, f32_arr, .bfloat16, str));
            return bf;
        }
    }.f;

    // Prefill then one decode token — the grow path leaves capacity slack,
    // so the update's views are STRIDED along the head axis (the layout the
    // kernel must index correctly; a contiguous-only test would pass green
    // on broken stride math).
    {
        const k = try mkArr(s, h_kv, t_prefill, d, 0.1);
        defer _ = mlx.mlx_array_free(k);
        const v = try mkArr(s, h_kv, t_prefill, d, 1.7);
        defer _ = mlx.mlx_array_free(v);
        var dv0 = try cache.update(0, k, v, s, 0);
        dv0.deinit();
    }
    const k1 = try mkArr(s, h_kv, 1, d, 2.9);
    defer _ = mlx.mlx_array_free(k1);
    const v1 = try mkArr(s, h_kv, 1, d, 3.3);
    defer _ = mlx.mlx_array_free(v1);
    var dv = try cache.update(0, k1, v1, s, 0);
    defer dv.deinit();
    try testing.expect(dv.has_quant_triple);

    const t_k: c_int = @intCast(t_prefill + 1);
    const q = try mkArr(s, h_q, 1, d, 5.1);
    defer _ = mlx.mlx_array_free(q);
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));

    // Optional additive mask row: block the first 2 kv positions.
    var mask = mlx.mlx_array{ .ctx = null };
    defer if (mask.ctx != null) {
        _ = mlx.mlx_array_free(mask);
    };
    if (masked) {
        const tk_us: usize = @intCast(t_k);
        const mbuf = try testing.allocator.alloc(f32, tk_us);
        defer testing.allocator.free(mbuf);
        for (mbuf, 0..) |*x, i| x.* = if (i < 2) -std.math.inf(f32) else 0.0;
        const mshape = [_]c_int{ 1, 1, 1, t_k };
        const mf32 = mlx.mlx_array_new_data(mbuf.ptr, &mshape, 4, .float32);
        defer _ = mlx.mlx_array_free(mf32);
        var mbf = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&mbf, mf32, .bfloat16, s));
        mask = mbf;
    }

    // Reference: dense SDPA over the view's own dequantized K/V.
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    const none = mlx.mlx_array{ .ctx = null };
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(
        &ref,
        q,
        dv.k,
        dv.v,
        scale,
        if (masked) "array" else "",
        if (masked) mask else none,
        .{ .ctx = null },
        false,
        s,
    ));

    const cand = (try qkvAttnDecodeKernel(
        s,
        q,
        &dv,
        scale,
        if (masked) "array" else "",
        if (masked) mask else none,
    )) orelse return error.KernelDeclined; // engagement IS the assertion

    defer _ = mlx.mlx_array_free(cand);
    const cand_shape = mlx.getShape(cand);
    try testing.expectEqual(@as(c_int, 1), cand_shape[0]);
    try testing.expectEqual(h_q, cand_shape[1]);
    try testing.expectEqual(@as(c_int, 1), cand_shape[2]);
    try testing.expectEqual(d, cand_shape[3]);

    // Compare in f32.
    const readFlat = struct {
        fn f(str: mlx.mlx_stream, arr: mlx.mlx_array, allocator: std.mem.Allocator) ![]f32 {
            var f32v = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(f32v);
            try mlx.check(mlx.mlx_astype(&f32v, arr, .float32, str));
            const n: c_int = @intCast(mlx.mlx_array_size(f32v));
            var flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat);
            const sh = [_]c_int{n};
            try mlx.check(mlx.mlx_reshape(&flat, f32v, &sh, 1, str));
            const ev = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(ev);
            _ = mlx.mlx_vector_array_append_value(ev, flat);
            try mlx.check(mlx.mlx_eval(ev));
            const ptr = mlx.mlx_array_data_float32(flat) orelse return error.NullData;
            const out = try allocator.alloc(f32, @intCast(n));
            @memcpy(out, ptr[0..@intCast(n)]);
            return out;
        }
    }.f;
    const ref_flat = try readFlat(s, ref, testing.allocator);
    defer testing.allocator.free(ref_flat);
    const cand_flat = try readFlat(s, cand, testing.allocator);
    defer testing.allocator.free(cand_flat);
    try testing.expectEqual(ref_flat.len, cand_flat.len);
    var max_err: f32 = 0;
    for (ref_flat, cand_flat) |r, c| {
        // NaN-blind comparisons pass on an all-NaN candidate (NaN > x is
        // false) — the composed causal arm shipped NaN for months behind
        // exactly this hole. Finiteness is asserted BEFORE the diff.
        try testing.expect(!std.math.isNan(c));
        const e = @abs(r - c);
        if (e > max_err) max_err = e;
    }
    if (max_err >= 0.05) {
        std.debug.print("qkvDecParityCase Hq={d} Hkv={d} D={d} bits={d} Tk={d} masked={}: max_err={d}\n", .{ h_q, h_kv, d, bits, t_k, masked, max_err });
    }
    try testing.expect(max_err < 0.05);
}

test "qkv decode kernel parity: GQA ratios x bits x head dims (strided views)" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    // T_k = 37+1 — not a multiple of NSG(4), exercises uneven simdgroup trip
    // counts; the second update leaves capacity slack so views are strided.
    try qkvDecParityCase(8, 2, 64, 8, 37, false);
    try qkvDecParityCase(8, 2, 64, 4, 37, false);
    try qkvDecParityCase(12, 2, 128, 8, 37, false);
    try qkvDecParityCase(12, 2, 128, 4, 37, false);
    try qkvDecParityCase(4, 4, 128, 8, 21, false); // MHA (GQA=1)
    try qkvDecParityCase(6, 1, 256, 8, 21, false); // hd-256 (gemma geometry)
}

test "qkv decode kernel parity: additive mask row (sliding decode shape)" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    try qkvDecParityCase(8, 2, 128, 8, 37, true);
    try qkvDecParityCase(8, 2, 128, 4, 37, true);
}

test "qkv decode kernel parity: multi-block split-KV merge (T_k > BLOCK)" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    // g=4/DK=64 → BLOCK 512, so 700+1 rows exercises the two-block partial
    // merge (m/l/O carry through the mlx-op reduction); the masked case
    // additionally crosses a block boundary with -inf rows.
    try qkvDecParityCase(8, 2, 64, 8, 700, false);
    try qkvDecParityCase(8, 2, 128, 8, 700, true);
}

fn qkvVerParityCase(h_q: c_int, h_kv: c_int, d: c_int, bits: u8, t_prefill: usize, t_q: c_int) !void {
    return qkvVerParityCaseWith(h_q, h_kv, d, bits, t_prefill, t_q, false);
}

fn qkvVerParityCaseWith(h_q: c_int, h_kv: c_int, d: c_int, bits: u8, t_prefill: usize, t_q: c_int, mpp: bool) !void {
    // Parity validates the explicit QA arm, independent of the measured
    // per-machine default (G17 defaults this kernel off).
    const old_qkv_ver_override = qkv_ver_override;
    qkv_ver_override = true;
    defer qkv_ver_override = old_qkv_ver_override;
    const s = mlx.gpuStream();
    var cache = try KVCache.initWithConfig(testing.allocator, 1, kv_quant.KVQuantConfig.affine(bits));
    defer cache.deinit();
    const mkArr = struct {
        fn f(str: mlx.mlx_stream, hh: c_int, tt: c_int, dd: c_int, seed: f32) !mlx.mlx_array {
            const total: usize = @intCast(hh * tt * dd);
            const buf = try testing.allocator.alloc(f32, total);
            defer testing.allocator.free(buf);
            for (buf, 0..) |*x, i| x.* = @sin(@as(f32, @floatFromInt(i)) * 0.37 + seed) * 0.7;
            const shape = [_]c_int{ 1, hh, tt, dd };
            const f32_arr = mlx.mlx_array_new_data(buf.ptr, &shape, 4, .float32);
            defer _ = mlx.mlx_array_free(f32_arr);
            var bf = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&bf, f32_arr, .bfloat16, str));
            return bf;
        }
    }.f;
    if (t_prefill > 0) {
        const k0 = try mkArr(s, h_kv, @intCast(t_prefill), d, 0.4);
        defer _ = mlx.mlx_array_free(k0);
        const v0 = try mkArr(s, h_kv, @intCast(t_prefill), d, 1.3);
        defer _ = mlx.mlx_array_free(v0);
        var dv0 = try cache.update(0, k0, v0, s, 0);
        dv0.deinit();
    }
    // The verify chunk: t_q new K/V rows appended (the spec contract — the
    // draft rows are in the cache when attention runs).
    const k1 = try mkArr(s, h_kv, t_q, d, 2.9);
    defer _ = mlx.mlx_array_free(k1);
    const v1 = try mkArr(s, h_kv, t_q, d, 3.3);
    defer _ = mlx.mlx_array_free(v1);
    var dv = try cache.update(0, k1, v1, s, 0);
    defer dv.deinit();
    try testing.expect(dv.has_quant_triple);

    const q = try mkArr(s, h_q, t_q, d, 5.1);
    defer _ = mlx.mlx_array_free(q);
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));

    // Reference: dense SDPA over the view's own dequantized K/V, "causal"
    // (end-aligned) — the exact mode the dense path serves these widths.
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    const none = mlx.mlx_array{ .ctx = null };
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&ref, q, dv.k, dv.v, scale, "causal", none, .{ .ctx = null }, false, s));

    const cand = (if (mpp) try qkvAttnMppKernel(s, q, &dv, scale, "causal") else try qkvAttnVerifyKernel(s, q, &dv, scale, "causal")) orelse return error.KernelDeclined;
    defer _ = mlx.mlx_array_free(cand);
    const cand_shape = mlx.getShape(cand);
    try testing.expectEqual(@as(c_int, 1), cand_shape[0]);
    try testing.expectEqual(h_q, cand_shape[1]);
    try testing.expectEqual(t_q, cand_shape[2]);
    try testing.expectEqual(d, cand_shape[3]);

    const readFlat = struct {
        fn f(str: mlx.mlx_stream, arr: mlx.mlx_array, allocator: std.mem.Allocator) ![]f32 {
            var f32v = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(f32v);
            try mlx.check(mlx.mlx_astype(&f32v, arr, .float32, str));
            const n: c_int = @intCast(mlx.mlx_array_size(f32v));
            var flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat);
            const sh = [_]c_int{n};
            try mlx.check(mlx.mlx_reshape(&flat, f32v, &sh, 1, str));
            const ev = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(ev);
            _ = mlx.mlx_vector_array_append_value(ev, flat);
            try mlx.check(mlx.mlx_eval(ev));
            const ptr = mlx.mlx_array_data_float32(flat) orelse return error.NullData;
            const out = try allocator.alloc(f32, @intCast(n));
            @memcpy(out, ptr[0..@intCast(n)]);
            return out;
        }
    }.f;
    const ref_flat = try readFlat(s, ref, testing.allocator);
    defer testing.allocator.free(ref_flat);
    const cand_flat = try readFlat(s, cand, testing.allocator);
    defer testing.allocator.free(cand_flat);
    try testing.expectEqual(ref_flat.len, cand_flat.len);
    var max_err: f32 = 0;
    for (ref_flat, cand_flat) |r, c| {
        // Finiteness BEFORE the diff (NaN > x is false — the composed causal
        // arm shipped NaN for months behind exactly this hole).
        try testing.expect(!std.math.isNan(c));
        const e = @abs(r - c);
        if (e > max_err) max_err = e;
    }
    if (max_err >= 0.05) {
        std.debug.print("qkvVerParityCase Hq={d} Hkv={d} D={d} bits={d} pre={d} Tq={d}: max_err={d}\n", .{ h_q, h_kv, d, bits, t_prefill, t_q, max_err });
    }
    try testing.expect(max_err < 0.05);
}

test "qkv verify kernel parity: TQ 2..8 x bits x GQA (strided views, causal tail)" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    // Mutation-checked: dropping the in-kernel causal tail (gpos < lim) or
    // mis-ordering the (head, tq) row layout fails the Tq>=2 cases — the
    // reference is end-aligned SDPA over the SAME dequantized view, so the
    // tail is load-bearing, not decorative. 37+Tq rows keeps views strided
    // (capacity slack from the two-update sequence).
    try qkvVerParityCase(8, 2, 64, 8, 37, 2);
    try qkvVerParityCase(8, 2, 64, 4, 37, 3);
    try qkvVerParityCase(12, 2, 128, 8, 37, 4);
    try qkvVerParityCase(12, 2, 128, 4, 37, 5);
    try qkvVerParityCase(4, 4, 128, 8, 21, 6); // MHA (GQA=1)
    try qkvVerParityCase(6, 1, 256, 8, 21, 8); // hd-256, deepest width
    try qkvVerParityCase(24, 4, 256, 8, 21, 4); // the qwen3_5 live geometry
}

test "qkv matmul2d kernel parity: t_q 1..8 x bits, partial last page, many splits, pad rows" {
    if (verifyQmmTile() == .off) return error.SkipZigTest; // no matmul2d tile (CI's paravirtual GPU)
    for ([_]u8{ 8, 4 }) |bits| {
        for ([_]c_int{ 1, 2, 4, 5, 8 }) |tq| {
            try qkvVerParityCaseWith(24, 4, 256, bits, 700, tq, true);
            try qkvVerParityCaseWith(16, 8, 256, bits, 3000, tq, true);
        }
    }
    try qkvVerParityCaseWith(24, 4, 256, 8, 0, 4, true); // cache holds only the verify rows
}

test "qkv matmul2d kernel declines rows past its 32 KiB threadgroup budget (GQA 8 x t_q 8)" {
    if (verifyQmmTile() == .off) return error.SkipZigTest;
    try testing.expectError(error.KernelDeclined, qkvVerParityCaseWith(16, 2, 256, 4, 3000, 8, true));
    try qkvVerParityCaseWith(16, 2, 256, 4, 3000, 7, true);
}

test "qkvMppWins: the matmul2d window needs rows or length" {
    try testing.expect(qkvMppWins(4, 2048));
    try testing.expect(!qkvMppWins(2, 4096));
    try testing.expect(qkvMppWins(2, 8192));
    try testing.expect(!qkvMppWins(1, 8192));
    try testing.expect(qkvMppWins(1, 16384));
}

test "qkv verify kernel serves by default only the row counts that stay in registers" {
    try testing.expect(qkvVerifyRowsPay(6, 2));
    try testing.expect(!qkvVerifyRowsPay(6, 3));
    try testing.expect(qkvVerifyRowsPay(2, 6));
    try testing.expect(!qkvVerifyRowsPay(2, 8));
}

test "qkv verify kernel defaults off on G17 but an explicit env value wins" {
    try testing.expect(qkvVerifyKernelEnabledFrom(false, null));
    try testing.expect(!qkvVerifyKernelEnabledFrom(true, null));
    try testing.expect(qkvVerifyKernelEnabledFrom(true, "1"));
    try testing.expect(!qkvVerifyKernelEnabledFrom(false, "0"));
}

test "qkv verify kernel parity: multi-block split-KV merge (T_k > BLOCK)" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    // GQA 4 x TQ 2 → BLOCK 256; 700+2 rows exercises the multi-block carry,
    // including a final block whose tail rows are causally -inf for tq 0.
    try qkvVerParityCase(8, 2, 64, 8, 700, 2);
    try qkvVerParityCase(8, 2, 128, 8, 700, 6);
}

test "qkv verify kernel declines non-verify widths and foreign masks" {
    // Exercise the kernel's own decline predicates even where the machine
    // adoption default declines the entire arm first.
    const old_qkv_ver_override = qkv_ver_override;
    qkv_ver_override = true;
    defer qkv_ver_override = old_qkv_ver_override;
    const s = mlx.gpuStream();
    var view: DenseKVView = .{ .k = .{ .ctx = null }, .v = .{ .ctx = null }, .owned = false, .has_quant_triple = false };
    const shape = [_]c_int{ 1, 4, 2, 64 };
    var q2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q2);
    try mlx.check(mlx.mlx_ones(&q2, &shape, 4, .bfloat16, s));
    // no triple → decline regardless of width
    try testing.expectEqual(@as(?mlx.mlx_array, null), try qkvAttnVerifyKernel(s, q2, &view, 1.0, "causal"));
    // real triple, wrong widths / modes
    var cache = try KVCache.initWithConfig(testing.allocator, 1, kv_quant.KVQuantConfig.affine(8));
    defer cache.deinit();
    const kshape = [_]c_int{ 1, 2, 40, 64 };
    var kv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kv);
    try mlx.check(mlx.mlx_ones(&kv, &kshape, 4, .bfloat16, s));
    var dv = try cache.update(0, kv, kv, s, 0);
    defer dv.deinit();
    var q1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q1);
    const q1shape = [_]c_int{ 1, 4, 1, 64 };
    try mlx.check(mlx.mlx_ones(&q1, &q1shape, 4, .bfloat16, s));
    try testing.expectEqual(@as(?mlx.mlx_array, null), try qkvAttnVerifyKernel(s, q1, &dv, 1.0, "causal")); // T_q 1 belongs to the decode kernel
    var q9 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q9);
    const q9shape = [_]c_int{ 1, 4, QKV_VERIFY_MAX_TQ + 1, 64 };
    try mlx.check(mlx.mlx_ones(&q9, &q9shape, 4, .bfloat16, s));
    try testing.expectEqual(@as(?mlx.mlx_array, null), try qkvAttnVerifyKernel(s, q9, &dv, 1.0, "causal")); // past the width cap
    var q4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q4);
    const q4shape = [_]c_int{ 1, 4, 4, 64 };
    try mlx.check(mlx.mlx_ones(&q4, &q4shape, 4, .bfloat16, s));
    try testing.expectEqual(@as(?mlx.mlx_array, null), try qkvAttnVerifyKernel(s, q4, &dv, 1.0, "array")); // sliding masks read dense
    try testing.expectEqual(@as(?mlx.mlx_array, null), try qkvAttnVerifyKernel(s, q4, &dv, 1.0, "")); // unmasked multi-row is not this kernel's contract
}

fn composedStridedCase(h_q: c_int, h_kv: c_int, d: c_int, bits: u8, t_pre: usize, t_chunk: c_int, mode: [:0]const u8) !void {
    const s = mlx.gpuStream();
    var cache = try KVCache.initWithConfig(testing.allocator, 1, kv_quant.KVQuantConfig.affine(bits));
    defer cache.deinit();
    const mk = struct {
        fn f(str: mlx.mlx_stream, hh: c_int, tt: c_int, dd: c_int, seed: f32) !mlx.mlx_array {
            const total: usize = @intCast(hh * tt * dd);
            const buf = try testing.allocator.alloc(f32, total);
            defer testing.allocator.free(buf);
            for (buf, 0..) |*x, i| x.* = @sin(@as(f32, @floatFromInt(i)) * 0.31 + seed) * 0.7;
            const shape = [_]c_int{ 1, hh, tt, dd };
            const f32_arr = mlx.mlx_array_new_data(buf.ptr, &shape, 4, .float32);
            defer _ = mlx.mlx_array_free(f32_arr);
            var bf = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&bf, f32_arr, .bfloat16, str));
            return bf;
        }
    }.f;
    if (t_pre > 0) {
        const k = try mk(s, h_kv, @intCast(t_pre), d, 0.2);
        defer _ = mlx.mlx_array_free(k);
        const v = try mk(s, h_kv, @intCast(t_pre), d, 1.1);
        defer _ = mlx.mlx_array_free(v);
        var dv0 = try cache.update(0, k, v, s, 0);
        dv0.deinit();
    }
    const k1 = try mk(s, h_kv, t_chunk, d, 2.3);
    defer _ = mlx.mlx_array_free(k1);
    const v1 = try mk(s, h_kv, t_chunk, d, 3.7);
    defer _ = mlx.mlx_array_free(v1);
    var dv = try cache.update(0, k1, v1, s, 0);
    defer dv.deinit();
    try testing.expect(dv.has_quant_triple);

    const t_k: c_int = @as(c_int, @intCast(t_pre)) + t_chunk;
    const q = try mk(s, h_q, t_chunk, d, 5.5);
    defer _ = mlx.mlx_array_free(q);
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));

    // Optional additive mask (sliding-window prefill shape).
    var mask = mlx.mlx_array{ .ctx = null };
    defer if (mask.ctx != null) {
        _ = mlx.mlx_array_free(mask);
    };
    if (std.mem.eql(u8, mode, "array")) {
        const rows: usize = @intCast(t_chunk);
        const cols: usize = @intCast(t_k);
        const mbuf = try testing.allocator.alloc(f32, rows * cols);
        defer testing.allocator.free(mbuf);
        const off: usize = @intCast(t_k - t_chunk);
        for (0..rows) |r| {
            for (0..cols) |c| {
                // causal + band(3): visible iff c <= off+r and (off+r)-c < 3
                const vis = c <= off + r and (off + r) - c < 3;
                mbuf[r * cols + c] = if (vis) 0.0 else -std.math.inf(f32);
            }
        }
        const mshape = [_]c_int{ 1, 1, t_chunk, t_k };
        const mf32 = mlx.mlx_array_new_data(mbuf.ptr, &mshape, 4, .float32);
        defer _ = mlx.mlx_array_free(mf32);
        var mbf = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&mbf, mf32, .bfloat16, s));
        mask = mbf;
    }
    const none = mlx.mlx_array{ .ctx = null };
    const mask_used = if (mask.ctx != null) mask else none;

    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&ref, q, dv.k, dv.v, scale, mode, mask_used, .{ .ctx = null }, false, s));

    const cand = try kv_quant.quantAttention(q, dv.kTriple(), dv.vTriple(), dv.bits, dv.group_size, scale, mode, mask_used, s);
    defer _ = mlx.mlx_array_free(cand);

    const readFlat = struct {
        fn f(str: mlx.mlx_stream, arr: mlx.mlx_array, allocator: std.mem.Allocator) ![]f32 {
            var f32v = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(f32v);
            try mlx.check(mlx.mlx_astype(&f32v, arr, .float32, str));
            const n: c_int = @intCast(mlx.mlx_array_size(f32v));
            var flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat);
            const sh = [_]c_int{n};
            try mlx.check(mlx.mlx_reshape(&flat, f32v, &sh, 1, str));
            const ev = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(ev);
            _ = mlx.mlx_vector_array_append_value(ev, flat);
            try mlx.check(mlx.mlx_eval(ev));
            const ptr = mlx.mlx_array_data_float32(flat) orelse return error.NullData;
            const out = try allocator.alloc(f32, @intCast(n));
            @memcpy(out, ptr[0..@intCast(n)]);
            return out;
        }
    }.f;
    const ref_flat = try readFlat(s, ref, testing.allocator);
    defer testing.allocator.free(ref_flat);
    const cand_flat = try readFlat(s, cand, testing.allocator);
    defer testing.allocator.free(cand_flat);
    var max_err: f32 = 0;
    for (ref_flat, cand_flat) |r, c| {
        // NaN-blind comparisons pass on an all-NaN candidate (NaN > x is
        // false) — the composed causal arm shipped NaN for months behind
        // exactly this hole. Finiteness is asserted BEFORE the diff.
        try testing.expect(!std.math.isNan(c));
        const e = @abs(r - c);
        if (e > max_err) max_err = e;
    }
    if (max_err >= 0.05) std.debug.print("composedStridedCase Hq={d} Hkv={d} D={d} bits={d} pre={d} chunk={d} mode={s}: max_err={d}\n", .{ h_q, h_kv, d, bits, t_pre, t_chunk, mode, max_err });
    try testing.expect(max_err < 0.05);
}

test "composed quantAttention is correct on STRIDED cache views" {
    mlx.installErrorHandler(); // MLX CUDA has no quantized_matmul for some multi-row shapes: fail, never abort
    // Class guard (splitPackedGateUp rule): slice-born views into
    // mlx_quantized_matmul must compute correctly — the cache's packed views
    // are non-contiguous whenever buffer capacity > logical length. Covers
    // multi-row widths, the gemma E4B live geometry (single 15-token chunk,
    // hd 256, 4-bit — the "<pad><pad>" shape), and the sliding "array" mask.
    // (Serving only routes t_q == 1 here since 2026-08-15, but the helper's
    // contract is width-agnostic and the strided-view class is the point.)
    try composedStridedCase(8, 2, 64, 4, 20, 4, "causal");
    try composedStridedCase(8, 2, 256, 4, 0, 15, "causal");
    try composedStridedCase(8, 2, 256, 4, 0, 15, "array");
    try composedStridedCase(8, 2, 256, 4, 20, 15, "array");
}

test "fused kv-quant serving arms are kernel-or-DENSE — composed attention never serves" {
    // gemma4-12B measured (2026-08-15, kv8, 11k ctx): composed reads at
    // t_q == 1 cost −17% decode on the full-attention layers where
    // `qkvAttnDecodeKernel` declines (gqa 16 x dk 512 overflows the
    // threadgroup q-staging budget), and Phase 1 already routes every other
    // width dense. So a declined kernel falls through to the dense SDPA
    // arms; the composed helper remains ONLY as the strided-view parity
    // reference (`composedStridedCase`).
    const src = @embedFile("kv_quant_attn.zig");
    const needle = "kv_quant.quant" ++ "Attention(";
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, src, needle));
    const only = std.mem.indexOf(u8, src, needle).?;
    const helper = std.mem.indexOf(u8, src, "fn composedStridedCase(").?;
    try testing.expect(only > helper);
}

test "qkv decode kernel declines non-decode widths and missing triples" {
    const s = mlx.gpuStream();
    // T_q = 2 → decline (only decode width may reach the kernel).
    var view: DenseKVView = .{ .k = .{ .ctx = null }, .v = .{ .ctx = null }, .owned = false, .has_quant_triple = false };
    const shape = [_]c_int{ 1, 4, 2, 64 };
    var q2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q2);
    try mlx.check(mlx.mlx_ones(&q2, &shape, 4, .bfloat16, s));
    // No triple → decline regardless of shape.
    try testing.expectEqual(@as(?mlx.mlx_array, null), try qkvAttnDecodeKernel(s, q2, &view, 1.0, "", .{ .ctx = null }));
}

test "qkv packed-attention surface µbench (env-gated: MLX_SERVE_KVQ_UBENCH=1)" {
    // Per (t_q, kv): (a) packed kernel, (b) dequant + dense sdpa, (c) bf16
    // sdpa with no dequant. Arms interleave in one process and the cell order
    // reverses per pass; BATCH launches share one eval so the sync barrier
    // does not price the cell.
    if (!env.on(.MLX_SERVE_KVQ_UBENCH)) return;
    const old_ver = qkv_ver_override;
    qkv_ver_override = true;
    defer qkv_ver_override = old_ver;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    const H_Q: c_int = 24;
    const H_KV: c_int = 4;
    const D: c_int = 256;
    const BITS: u8 = 8;
    const kvs = [_]usize{ 2048, 4096, 8192, 16384, 32768, 65536 };
    const tqs = [_]c_int{ 1, 2, 3, 4, 6, 8 };
    const PASSES = 4;
    const BATCH = 8;
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
    const none = mlx.mlx_array{ .ctx = null };

    const randBf16 = struct {
        fn f(str: mlx.mlx_stream, shape: []const c_int, sd: f32) !mlx.mlx_array {
            var tmp = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(tmp);
            try mlx.check(mlx.mlx_random_normal(&tmp, shape.ptr, shape.len, .float32, 0.0, sd, .{ .ctx = null }, str));
            var out = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&out, tmp, .bfloat16, str));
            return out;
        }
    }.f;

    std.debug.print("\n[kvq-surface] Hq={d} Hkv={d} D={d} bits={d}\n[kvq-surface] {s:>6} {s:>3} {s:>9} {s:>9} {s:>9} {s:>9}\n", .{ H_Q, H_KV, D, BITS, "kv", "tq", "packed", "dq+sdpa", "bf16", "mpp" });
    for (kvs) |kv| {
        var ns: [tqs.len][4]u64 = @splat(@splat(0));
        for (tqs, 0..) |tq, ti| {
            // Own cache per width: the verify contract has the tq rows appended.
            var cache = try KVCache.initWithConfig(testing.allocator, 1, kv_quant.KVQuantConfig.affine(BITS));
            defer cache.deinit();
            {
                const k = try randBf16(s, &[_]c_int{ 1, H_KV, @intCast(kv - @as(usize, @intCast(tq))), D }, 0.2);
                defer _ = mlx.mlx_array_free(k);
                var dv0 = try cache.update(0, k, k, s, 0);
                dv0.deinit();
            }
            const k1 = try randBf16(s, &[_]c_int{ 1, H_KV, tq, D }, 0.2);
            defer _ = mlx.mlx_array_free(k1);
            var dv = try cache.update(0, k1, k1, s, 0);
            defer dv.deinit();
            var k_off = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(k_off);
            try mlx.check(mlx.mlx_contiguous(&k_off, dv.k, false, s));
            const q = try randBf16(s, &[_]c_int{ 1, H_Q, tq, D }, 0.5);
            defer _ = mlx.mlx_array_free(q);
            {
                const ev = mlx.mlx_vector_array_new();
                defer _ = mlx.mlx_vector_array_free(ev);
                _ = mlx.mlx_vector_array_append_value(ev, k_off);
                _ = mlx.mlx_vector_array_append_value(ev, q);
                try mlx.check(mlx.mlx_eval(ev));
            }
            const mode: [:0]const u8 = if (tq == 1) "" else "causal";

            for (0..PASSES + 1) |pass| { // pass 0 = warmup
                for (0..4) |a0| {
                    const arm = if (pass % 2 == 0) a0 else 3 - a0;
                    const ev = mlx.mlx_vector_array_new();
                    defer _ = mlx.mlx_vector_array_free(ev);
                    const mark = std.Io.Timestamp.now(io, .boot);
                    for (0..BATCH) |_| {
                        var out = mlx.mlx_array_new();
                        defer _ = mlx.mlx_array_free(out);
                        switch (arm) {
                            0 => out = (if (tq == 1)
                                try qkvAttnDecodeKernel(s, q, &dv, scale, "", none)
                            else
                                try qkvAttnVerifyKernel(s, q, &dv, scale, "causal")) orelse return error.KernelDeclined,
                            3 => out = (try qkvAttnMppKernel(s, q, &dv, scale, mode)) orelse return error.KernelDeclined,
                            else => {
                                var kd = k_off;
                                if (arm == 1) kd = try kv_quant.dequantizeAffine(s, dv.k_triple_q, dv.k_triple_scales, dv.k_triple_biases, dv.group_size, dv.bits);
                                defer if (arm == 1) {
                                    _ = mlx.mlx_array_free(kd);
                                };
                                var vd = k_off;
                                if (arm == 1) vd = try kv_quant.dequantizeAffine(s, dv.v_triple_q, dv.v_triple_scales, dv.v_triple_biases, dv.group_size, dv.bits);
                                defer if (arm == 1) {
                                    _ = mlx.mlx_array_free(vd);
                                };
                                if (tq > 1) {
                                    if (try splitCausalSdpa(s, q, kd, vd, scale)) |sp| out = sp;
                                }
                                if (out.ctx == null) try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&out, q, kd, vd, scale, mode.ptr, none, .{ .ctx = null }, false, s));
                            },
                        }
                        _ = mlx.mlx_vector_array_append_value(ev, out);
                    }
                    try mlx.check(mlx.mlx_eval(ev));
                    if (pass > 0) ns[ti][arm] += @intCast(mark.untilNow(io, .boot).nanoseconds);
                }
            }
            _ = mlx.mlx_clear_cache();
        }
        for (tqs, 0..) |tq, ti| {
            var ms: [4]f64 = undefined;
            for (&ms, 0..) |*m, a| m.* = @as(f64, @floatFromInt(ns[ti][a])) / @as(f64, PASSES * BATCH) / 1e6;
            std.debug.print("[kvq-surface] {d:>6} {d:>3} {d:>9.3} {d:>9.3} {d:>9.3} {d:>9.3}\n", .{ kv, tq, ms[0], ms[1], ms[2], ms[3] });
        }
    }
}

test "kvAttnFusedEligible: fused reads are decode-width (t_q == 1) only" {
    // This is the explicit verify-kernel eligibility arm. G17's production
    // default is dense, so force the QA seam before testing width/floor rules.
    const old_qkv_ver_override = qkv_ver_override;
    qkv_ver_override = true;
    defer qkv_ver_override = old_qkv_ver_override;
    // The ONE gate every fused call site reads. Verify widths (T_q 2..32
    // under MTP/spec) used to be admitted and fell to the composed qmm
    // chain — a deliberate choice that predates deep-MTP defaults and was
    // never A/B'd at spec depth. Measured (M4 Max, 2026-08-15, 8k+ ctx,
    // deep MTP rounds): 28 tok/s decode vs 60 on the dense read — the
    // composed chain runs 2 packed matmuls x every attention layer per spec
    // round, at qmm's M 2..7 dead zone. Decode width keeps the packed-read
    // kernel win; every other width reads the dense view (the reference
    // semantics). Short-KV layers (sliding windows) stay dense too: the
    // kernel's split-KV merge overhead measured a net loss there.
    const s = mlx.gpuStream();
    const long_shape = [_]c_int{ 1, 2, KV_ATTN_FUSED_MIN_TK, 16 };
    var kq_long = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kq_long);
    try mlx.check(mlx.mlx_zeros(&kq_long, &long_shape, 4, .uint32, s));
    const short_shape = [_]c_int{ 1, 2, 512, 16 };
    var kq_short = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kq_short);
    try mlx.check(mlx.mlx_zeros(&kq_short, &short_shape, 4, .uint32, s));

    var view_with: DenseKVView = .{ .k = .{ .ctx = null }, .v = .{ .ctx = null }, .owned = false, .has_quant_triple = true, .k_triple_q = kq_long };
    var view_short: DenseKVView = .{ .k = .{ .ctx = null }, .v = .{ .ctx = null }, .owned = false, .has_quant_triple = true, .k_triple_q = kq_short };
    var view_without: DenseKVView = .{ .k = .{ .ctx = null }, .v = .{ .ctx = null }, .owned = false, .has_quant_triple = false };
    try testing.expect(kvAttnFusedEligible(&view_with, 1));
    try testing.expect(!kvAttnFusedEligible(&view_with, 2)); // verify width
    try testing.expect(!kvAttnFusedEligible(&view_with, 9));
    try testing.expect(!kvAttnFusedEligible(&view_with, 32));
    try testing.expect(!kvAttnFusedEligible(&view_with, 2048)); // prefill width
    try testing.expect(!kvAttnFusedEligible(&view_short, 1)); // below the kv floor
    try testing.expect(!kvAttnFusedEligible(&view_without, 1));
    // Phase 2: verify widths (2..QKV_VERIFY_MAX_TQ) have their OWN gate,
    // serving the TQ-wide packed kernel — never the composed chain.
    try testing.expect(!kvAttnVerifyEligible(&view_with, 1)); // decode belongs to the decode kernel
    try testing.expect(kvAttnVerifyEligible(&view_with, 2));
    try testing.expect(kvAttnVerifyEligible(&view_with, QKV_VERIFY_MAX_TQ));
    try testing.expect(!kvAttnVerifyEligible(&view_with, QKV_VERIFY_MAX_TQ + 1)); // prefill chunks stay dense
    try testing.expect(!kvAttnVerifyEligible(&view_short, 4)); // below the kv floor
    try testing.expect(!kvAttnVerifyEligible(&view_without, 4));
}
