//! sdk/blocks/attention/attn256, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns ATTN256_KERNEL_HEADER, ATTN256_KERNEL_SOURCE, attn256_kernel_cached, getAttn256Kernel, fused256_env_cached, fused256_causal_env_cached, and 29 more.

const std = @import("std");
const env = @import("../../../env.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../host/log.zig");
const sdpa = @import("sdpa.zig");
const naxSdpaPreferred = @import("sdpa.zig").naxSdpaPreferred;

// ── Fused head_dim-256 prefill attention (flash-style Metal kernel) ──
//
// MLX's fused SDPA (steel_attention) covers head_dim <= 128; every Gemma-4
// and Qwen3.5/3.6 checkpoint ships head_dim 256, which used to fall back to
// the composed path that MATERIALIZES a [heads, chunk, total_kv] bf16 score
// tensor per layer (26.8 GB/layer at a 102K prompt — the long-context OOM
// class, commit 7550895 budgeted around it). This kernel started as a
// faithful self-contained port of MLX's steel attention (FA-2 online
// softmax, float32 accumulation, exp2 softmax) specialized to BD=256; the
// v2 tiles (BQ=64, BK=32, 8 simdgroups, register-resident Q, uint4 staging
// — see the geometry comment above ATTN256_KERNEL_SOURCE) are our own,
// picked by micro-bench sweep. An optional sliding-window band covers
// Gemma's local layers ("array" mask during prefill) too. The score tensor
// never exists — O(tile) working memory — so the three prefill OOM guards
// (generate.boundedPrefillChunk / prefillEvalCadence / server.prefillMemoryNeeded)
// drop their score term via `prefillHeadDimFused` when this kernel is active.
//
// Scope: seq_len > 1 (prefill), any batch, GQA (Hq % Hk == 0), bf16, causal
// bottom-right alignment (query row r sits at absolute KV position
// kL - qL + r, matching MLX "causal" and createSlidingWindowMask semantics).
// K/V may be non-contiguous cache views (strides used; innermost dim must be
// contiguous — guaranteed for slices along T). Guards and dispatch read one
// shared predicate, so they cannot drift.
pub const ATTN256_KERNEL_HEADER = kernelSource("kernels/attention/attn256_header.metal");

// v2 tile geometry (2026-07-12 micro-bench sweep on the M4 Max, Qwen 24q/4kv
// and Gemma geometries): BQ=64 with 8 simdgroups halves the K/V staging walks
// per query row vs the v1 32-row tile; BK=32 halves the per-column softmax
// rescale + barrier overhead; Q lives in REGISTERS loaded straight from
// global (no Q smem at all — that memory goes to the wider K tile); staging
// runs on uint4 (8 bf16 per instruction, was scalar). Measured vs v1 at
// (qL x kL): (2048x16384) 250 -> 73 ms, (2048x65536) 999 -> 325 ms,
// (8192x8192) 279 -> 77 ms. vs the composed path the v2 kernel wins where
// kL <= ~4*qL (+63% at ratio 1, +15% at 2, +1.8% at 4) and loses beyond
// (-2.3% at 6, -6% at 8) — the causal arm's ratio gate (fusedSdpa256Prefill)
// encodes exactly that envelope. Rejected variants (measured worse): BK=8
// ping-pong (per-block rescale doubles), V-from-global fragments (device
// loads too slow), K-only double-buffer (barrier structure unchanged),
// Q-smem BQ=32 with 2 threadgroups/core (SLC thrash at long kL).
pub const ATTN256_KERNEL_SOURCE = kernelSource("kernels/attention/attn256.metal");

var attn256_kernel_cached: ?mlx.mlx_fast_metal_kernel = null;

fn getAttn256Kernel() !mlx.mlx_fast_metal_kernel {
    if (attn256_kernel_cached) |kk| return kk;
    const input_names = [_][*:0]const u8{ "q", "k", "v", "scl", "win", "kr", "phase", "m_in", "l_in", "o_in", "mask", "skip", "sinks" };
    const output_names = [_][*:0]const u8{ "out", "m_out", "l_out", "o_out" };
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "msv_attn_p256",
        in_vec,
        out_vec,
        ATTN256_KERNEL_SOURCE,
        ATTN256_KERNEL_HEADER,
        false, // ensure_row_contiguous=false — K/V are cache VIEWS; a forced
        // contiguous copy of the full cache per layer would erase the win.
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    attn256_kernel_cached = kernel;
    return kernel;
}


pub fn fused256Enabled() bool {
    return fused256_override orelse true;
}

/// Plain-CAUSAL dispatch mode. Default `.all` since the kv-chunk dispatch
/// budget landed (2026-07-22): the historical net-LOSS of every pre-budget
/// ratio-gated variant (8K: fused-to-ratio-4 231.5, fused-to-ratio-2 232.9,
/// composed 234.7 tok/s — monotonic: more fused = slower, despite the v2
/// kernel winning the µbench at kL <= 2*qL) was the macOS IOGPU
/// interactivity-preemption class: one long dispatch scanning the whole KV
/// gets preempted, and the penalty scales with dispatch length. With the
/// key axis split into budget-sized dispatches (see
/// FUSED256_DEFAULT_DISPATCH_BUDGET) the causal arm WINS the same
/// same-session A/B on the 27B: +2.9%/+2.3%/+4.6% at 8K/16K/32K.
/// The SLIDING-BAND arm (window > 0) has no such competition — composed
/// computes full-width scores + a GB-scale mask while the kernel
/// block-skips outside the band (gemma-26B 99K: 317 -> 712 tok/s) — so it
/// rides the master switch alone and never chunks.
pub const Fused256CausalMode = enum { all, off };

// ── Dispatch work budget (kv-axis chunked dispatch) ──
// One long dispatch scanning the whole KV monopolizes the GPU long enough at
// high kv_len to trip macOS IOGPU interactivity preemption, collapsing
// long-context prefill (oMLX issue #2225: the M3 Max cliff sits near ~1.5e9
// work units ≈ 50 ms/dispatch; design ported from oMLX's
// qwen35_fa256_attention.py, Apache-2.0, oMLX by jundot). Above the budget
// (work = batch·Hq·qL·kL) the CAUSAL arm splits the key axis into
// separately-dispatched BK-aligned chunks with exact online-softmax carry
// (fp32 m/l/O buffers chained between dispatches — bit-identical to the
// single-dispatch result). The band arm never chunks: its in-kernel block
// skip already bounds per-dispatch work by the window.
// The budget mirrors oMLX's 250M fallback (~23 ms/dispatch on the
// M4 Max at the kernel's measured ~1.1e10 work-units/s).
pub const FUSED256_DEFAULT_DISPATCH_BUDGET: i64 = 250_000_000;

/// Test seam: forces the budget without the environment.
pub var fused256_budget_override: ?i64 = null;

pub fn fused256DispatchBudget() i64 {
    return fused256_budget_override orelse FUSED256_DEFAULT_DISPATCH_BUDGET;
}

/// kv-axis chunk length for the budgeted causal dispatch: the largest
/// BK(32)-multiple keeping work-per-dispatch = batch·hq·ql·chunk within
/// budget, floored at one BK block, capped at kl. budget <= 0 = one dispatch.
pub fn fused256KvChunkLen(batch: i64, hq: i64, ql: i64, kl: i64, budget: i64) c_int {
    if (budget <= 0) return @intCast(kl);
    const per_key = batch * hq * ql;
    var chunk = @divTrunc(budget, @max(per_key, 1));
    chunk = @divTrunc(chunk, 32) * 32;
    if (chunk < 32) chunk = 32;
    if (chunk > kl) chunk = kl;
    return @intCast(chunk);
}

/// Test seam: dispatches issued by the LAST fusedSdpa256Prefill call
/// (engagement is counted, never inferred from output equality).
pub var fused256_last_dispatch_count: u32 = 0;

pub fn fused256CausalMode() Fused256CausalMode {
    if (fused256_override) |v| return if (v) .all else .off;
    return if (fused256Enabled()) .all else .off;
}

/// ONE predicate consumed by the three prefill OOM guards AND the dispatch
/// sites: true when prefill attention at this head_dim will NOT materialize
/// the composed [heads, chunk, total_kv] score tensor — either MLX's own
/// fused kernel covers it (<= 128) or our hd-256 kernel does for EVERY arm.
/// Keyed on causal mode: at `.all` (the default since the budgeted-dispatch
/// flip) no hd-256 score tensor materializes; the test seam's composed causal
/// moves the guards' score billing with it. Guards and dispatch must never drift (the effectivePrefillChunk rule).
pub fn prefillHeadDimFused(head_dim: u32) bool {
    return head_dim <= 128 or (head_dim == 256 and fused256CausalMode() == .all);
}

/// Shortest query block the fused hd-256 kernel accepts. Below it MLX's
/// sdpa_vector wins (see the decline in fusedSdpa256Prefill), which also means
/// the kernel is NOT the thing band-masking a spec-verify block —
/// `slidingViewFor` reads the same constant so the trim predicate and the
/// decline can never drift.
pub const FUSED256_MIN_Q_LEN: c_int = 16;

/// Try the fused hd-256 flash prefill kernel (msv_attn_p256). Returns null
/// when a precondition doesn't hold — the caller falls back to the composed
/// path. `window` > 0 adds the sliding-band mask (Gemma local layers,
/// createSlidingWindowMask semantics: key masked when row_abs - col >=
/// window); 0 = plain bottom-right causal. Free function (stream param) so
/// the MTP head can use it too.
///
/// STRUCTURAL precondition (not checkable — lazy arrays carry no strides at
/// graph-build time): q/k/v innermost dim contiguous. True for fresh
/// rope/transpose outputs, cache views (slices along T of a contiguous
/// buffer), and dequantized dense views — i.e. every attention call site.
pub fn fusedSdpa256Prefill(
    s: mlx.mlx_stream,
    q: mlx.mlx_array,
    k: mlx.mlx_array,
    v: mlx.mlx_array,
    scale: f32,
    window: c_int,
) !?mlx.mlx_array {
    // Band arm rides the master switch; plain causal is opt-in (see
    // fused256CausalMode — composed wins live on qwen chunked prefill).
    if (window > 0) {
        if (!fused256Enabled()) return null;
    } else {
        // Opt-in int8 Q/K causal arm on NAX (lossy); null when off or ineligible.
        if (try @import("nax_attention.zig").attention256Int8(s, q, k, v, scale)) |q8| return q8;
        if (fused256CausalMode() == .off) return null;
        // The band arm stays ours (mlx's NAX kernel takes no band); the
        // causal arm yields to the stock fused kernel on NAX.
        if (naxSdpaPreferred()) return null;
    }
    return fusedSdpa256Impl(s, q, k, v, scale, window, null, null);
}

/// The same kernel for the sink-attention arm (gpt_oss / MiMo): any of its head
/// shapes (MiMo's q/k 192, v 128), the layer's learned sinks, causal or banded.
/// MLX has no fused kernel at 192/128, so this one serves on NAX too. Null when
/// it declines (gpt_oss's hd 64 keeps MLX's).
/// No kill switch: MiMo bills no prefill score tensor (`prefillScoreHeadDim` 0),
/// so a composed fallback here would be an unbilled one.
pub fn fusedSinkAttnPrefill(s: mlx.mlx_stream, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32, window: c_int, sinks: ?mlx.mlx_array) !?mlx.mlx_array {
    // On NAX the tensor-unit kernel runs 192/128; ours takes everything else.
    if (try @import("nax_attention.zig").attention(s, q, k, v, scale, window, sinks)) |out| return out;
    return fusedSdpa256Impl(s, q, k, v, scale, window, null, sinks);
}

/// True when a bf16 view's bound address is 16-byte aligned — the bar every
/// `uint4` staging load in `msv_attn_p256` / `msv_attn_qsa256` needs. A lazy
/// array has no base yet (the kernel binds its buffer at dispatch, like the
/// stride checks in `gatherQsa256` see); an available one returns the address
/// the kernel really reads. Same base test `hcUvEligible` runs for its weights.
pub fn attn256Base16Aligned(a: mlx.mlx_array) bool {
    var avail = false;
    if (mlx._mlx_array_is_available(&avail, a) != 0 or !avail) return true;
    const p = mlx.mlx_array_data_uint32(a) orelse return true;
    return @intFromPtr(p) % 16 == 0;
}

pub var qsa_fused_override: ?bool = null;

/// The test seam's false runs the stock array-mask sdpa (the unfused
/// hd-256 fallback on every silicon: `use_fallback` has no fused arm for an
/// array mask at hd 256).
pub fn qsaFusedEnabled() bool {
    return qsa_fused_override orelse true;
}

var qsa_fused_logged = false;

/// Array-mask arm of msv_attn_p256 (qwen4_exp QSA prefill): `mask` is the
/// [B,1,qL,kL] bool visibility with causal folded in. Key blocks no row of a
/// 64-row q tile sees are skipped in-kernel via a per-tile any-visible table
/// built here. Same envelope as the causal arm (q >= 16, hd 256, bf16).
pub fn fusedSdpa256Masked(
    s: mlx.mlx_stream,
    q: mlx.mlx_array,
    k: mlx.mlx_array,
    v: mlx.mlx_array,
    scale: f32,
    mask: mlx.mlx_array,
) !?mlx.mlx_array {
    if (!fused256Enabled() or !qsaFusedEnabled()) return null;
    if (mask.ctx == null or mlx.mlx_array_ndim(mask) != 4 or mlx.mlx_array_dtype(mask) != .bool_) return null;
    const ms = mlx.getShape(mask);
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    if (ms[0] != qs[0] or ms[1] != 1 or ms[2] != qs[2] or ms[3] != ks[2]) return null;
    const out = try fusedSdpa256Impl(s, q, k, v, scale, 0, mask, null);
    if (out != null and !qsa_fused_logged) {
        qsa_fused_logged = true;
        log.info("[qsa-fused] engaged: msv_attn_p256 mask arm qL={d} kL={d} Hq={d} Hkv={d}\n", .{ qs[2], ks[2], qs[1], ks[1] });
    }
    return out;
}

/// Per-(batch, q tile, key block) any-visible table for the QSA arm:
/// [B, ceil(qL/64), ceil(kL/32)] bool.
fn qsaSkipTable(s: mlx.mlx_stream, mask: mlx.mlx_array) !mlx.mlx_array {
    const ms = mlx.getShape(mask);
    const nq: c_int = @divTrunc(ms[2] + 63, 64);
    const nk: c_int = @divTrunc(ms[3] + 31, 32);
    const axes = [_]c_int{ 2, 3 };
    const low = [_]c_int{ 0, 0 };
    const high = [_]c_int{ nq * 64 - ms[2], nk * 32 - ms[3] };
    const falsev = mlx.mlx_array_new_bool(false);
    defer _ = mlx.mlx_array_free(falsev);
    var padded = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(padded);
    try mlx.check(mlx.mlx_pad(&padded, mask, &axes, 2, &low, 2, &high, 2, falsev, "constant", s));
    var r5 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r5);
    try mlx.check(mlx.mlx_reshape(&r5, padded, &[_]c_int{ ms[0], nq, 64, nk, 32 }, 5, s));
    const red = [_]c_int{ 2, 4 };
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_any_axes(&out, r5, &red, 2, false, s));
    return out;
}

pub fn fusedSdpa256Impl(
    s: mlx.mlx_stream,
    q: mlx.mlx_array,
    k: mlx.mlx_array,
    v: mlx.mlx_array,
    scale: f32,
    window: c_int,
    mask: ?mlx.mlx_array,
    /// Learned per-head softmax sinks [Hq] (gpt_oss / MiMo), or null.
    sinks: ?mlx.mlx_array,
) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return null;
    if (mlx.mlx_array_ndim(q) != 4 or mlx.mlx_array_ndim(k) != 4 or mlx.mlx_array_ndim(v) != 4) return null;
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const vs = mlx.getShape(v);
    // q/k 256 with v 256, or MiMo's q/k 192 with v 128.
    if (qs[3] != ks[3] or !((qs[3] == 256 and vs[3] == 256) or (qs[3] == 192 and vs[3] == 128))) return null;
    const bdv: c_int = vs[3];
    // Short sequences (< 16: decode AND spec-decode VERIFY forwards) belong
    // to MLX's sdpa_vector, which covers hd 256 natively and beats a 64-row
    // prefill tile walking the whole KV for a few-row query — dispatching
    // those here measured decode 48 -> 18 tok/s at 4K ctx on the 27B (MTP
    // verify is seq 1+depth, up to 9 at the NAX depth-8 cap — with causal
    // fused default-on the floor must clear it). 16 matches oMLX's
    // _MIN_ROUTE_Q_LEN; genuine prefill chunks are always far above it.
    if (qs[2] < FUSED256_MIN_Q_LEN) return null;
    if (ks[1] <= 0 or @rem(qs[1], ks[1]) != 0) return null;
    if (ks[2] < qs[2] or ks[2] != vs[2] or ks[1] != vs[1] or ks[0] != qs[0] or vs[0] != qs[0]) return null;
    const attn_dt = mlx.mlx_array_dtype(q);
    if ((attn_dt != .bfloat16 and attn_dt != .float16) or mlx.mlx_array_dtype(k) != attn_dt or mlx.mlx_array_dtype(v) != attn_dt) return null;

    // K/V stage through `uint4` loads: an unaligned materialized view base
    // (e.g. the [3,259) slice of a padded cache) reads wrong values; decline
    // and the caller's fallback runs the same shape.
    if (!attn256Base16Aligned(k) or !attn256Base16Aligned(v)) return null;

    const kernel = getAttn256Kernel() catch return null;

    const one = [_]c_int{1};
    const scl_data = [_]f32{scale};
    const scl = mlx.mlx_array_new_data(&scl_data, &one, 1, .float32);
    defer _ = mlx.mlx_array_free(scl);
    const win_data = [_]i32{@intCast(window)};
    const win = mlx.mlx_array_new_data(&win_data, &one, 1, .int32);
    defer _ = mlx.mlx_array_free(win);

    // Budgeted kv-axis chunking (CAUSAL arm only — the band arm's in-kernel
    // block skip already bounds per-dispatch work by the window, so chunking
    // it would only add carry traffic). Chunk boundaries are BK(32)-aligned;
    // the fp32 m/l/O carry chained between dispatches keeps the result
    // bit-identical to a single dispatch (see the budget comment above).
    const kL: c_int = ks[2];
    const chunk_len: c_int = if (window > 0)
        kL
    else
        fused256KvChunkLen(@intCast(qs[0]), @intCast(qs[1]), @intCast(qs[2]), @intCast(kL), fused256DispatchBudget());

    // Dummy 1-elem f32 stands in for absent carry inputs / unwritten outputs.
    const dummy_data = [_]f32{0};
    const dummy = mlx.mlx_array_new_data(&dummy_data, &one, 1, .float32);
    defer _ = mlx.mlx_array_free(dummy);
    // 4-D so `mask_strides[..3]` exists for the QSA=0 template too.
    const dummy_b_data = [_]bool{false};
    const dummy_b_shape = [_]c_int{ 1, 1, 1, 1 };
    const dummy_b = mlx.mlx_array_new_data(&dummy_b_data, &dummy_b_shape, 4, .bool_);
    defer _ = mlx.mlx_array_free(dummy_b);
    var skip = mlx.mlx_array{ .ctx = null };
    defer if (skip.ctx != null) {
        _ = mlx.mlx_array_free(skip);
    };
    if (mask) |m| skip = try qsaSkipTable(s, m);

    var m_prev = mlx.mlx_array{ .ctx = null };
    var l_prev = mlx.mlx_array{ .ctx = null };
    var o_prev = mlx.mlx_array{ .ctx = null };
    defer {
        if (m_prev.ctx != null) _ = mlx.mlx_array_free(m_prev);
        if (l_prev.ctx != null) _ = mlx.mlx_array_free(l_prev);
        if (o_prev.ctx != null) _ = mlx.mlx_array_free(o_prev);
    }

    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    var dispatches: u32 = 0;
    var k0: c_int = 0;
    while (k0 < kL) : (k0 += chunk_len) {
        const k1: c_int = @min(k0 + chunk_len, kL);
        const final = k1 == kL;
        const has_carry = k0 > 0;

        const two = [_]c_int{2};
        const kr_data = [_]i32{ @intCast(k0), @intCast(k1) };
        const kr = mlx.mlx_array_new_data(&kr_data, &two, 1, .int32);
        defer _ = mlx.mlx_array_free(kr);
        const phase_data = [_]i32{(if (has_carry) @as(i32, 1) else 0) | (if (final) @as(i32, 2) else 0)};
        const phase = mlx.mlx_array_new_data(&phase_data, &one, 1, .int32);
        defer _ = mlx.mlx_array_free(phase);

        const config = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
        if (final) {
            const o_shape = [_]c_int{ qs[0], qs[1], qs[2], bdv };
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &o_shape, 4, attn_dt));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &one, 1, .float32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &one, 1, .float32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &one, 1, .float32));
        } else {
            const ml_shape = [_]c_int{ qs[0], qs[1], qs[2] };
            const oc_shape = [_]c_int{ qs[0], qs[1], qs[2], bdv };
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &one, 1, attn_dt));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &ml_shape, 3, .float32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &ml_shape, 3, .float32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &oc_shape, 4, .float32));
        }
        // One threadgroup (32,8,1) per 64-row q tile per head per batch; grid
        // is in THREADS (dispatch_threads), padded to whole tiles so every
        // threadgroup is full (the cooperative staging loops assume NT=256).
        const nq_tiles: c_int = @divTrunc(qs[2] + 63, 64);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, nq_tiles * 32, qs[1] * 8, qs[0]));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 8, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", attn_dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "QSA", if (mask != null) 1 else 0));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BDK", qs[3]));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BDV", bdv));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "SINK", if (sinks != null) 1 else 0));

        const inputs_arr = [_]mlx.mlx_array{
            q,                                k,
            v,                                scl,
            win,                              kr,
            phase,                            if (has_carry) m_prev else dummy,
            if (has_carry) l_prev else dummy, if (has_carry) o_prev else dummy,
            if (mask) |m| m else dummy_b,     if (mask != null) skip else dummy_b,
            sinks orelse dummy,
        };
        const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
        defer _ = mlx.mlx_vector_array_free(inputs_vec);

        var outputs_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(outputs_vec);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, config, s));
        if (mlx.mlx_vector_array_size(outputs_vec) != 4) return error.MetalKernelBadOutputCount;
        dispatches += 1;

        if (final) {
            try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
        } else {
            var m_new = mlx.mlx_array_new();
            var l_new = mlx.mlx_array_new();
            var o_new = mlx.mlx_array_new();
            errdefer {
                if (m_new.ctx != null) _ = mlx.mlx_array_free(m_new);
                if (l_new.ctx != null) _ = mlx.mlx_array_free(l_new);
                if (o_new.ctx != null) _ = mlx.mlx_array_free(o_new);
            }
            try mlx.check(mlx.mlx_vector_array_get(&m_new, outputs_vec, 1));
            try mlx.check(mlx.mlx_vector_array_get(&l_new, outputs_vec, 2));
            try mlx.check(mlx.mlx_vector_array_get(&o_new, outputs_vec, 3));
            if (m_prev.ctx != null) _ = mlx.mlx_array_free(m_prev);
            if (l_prev.ctx != null) _ = mlx.mlx_array_free(l_prev);
            if (o_prev.ctx != null) _ = mlx.mlx_array_free(o_prev);
            m_prev = m_new;
            l_prev = l_new;
            o_prev = o_new;
            m_new = .{ .ctx = null };
            l_new = .{ .ctx = null };
            o_new = .{ .ctx = null };
        }
    }
    fused256_last_dispatch_count = dispatches;
    return out;
}

pub fn attn256MaxDiff(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    // Raw data-pointer reads below: force row-major first. The fused sdpa
    // kernels hand back a [B,L,H,D]-strided VIEW and astype keeps the strides.
    var a_c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a_c);
    var b_c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b_c);
    try mlx.check(mlx.mlx_contiguous(&a_c, a, false, s));
    try mlx.check(mlx.mlx_contiguous(&b_c, b, false, s));
    var a32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a32);
    var b32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b32);
    try mlx.check(mlx.mlx_astype(&a32, a_c, .float32, s));
    try mlx.check(mlx.mlx_astype(&b32, b_c, .float32, s));
    try mlx.check(mlx.mlx_array_eval(a32));
    try mlx.check(mlx.mlx_array_eval(b32));
    const ad = mlx.mlx_array_data_float32(a32) orelse return error.InvalidDtype;
    const bd = mlx.mlx_array_data_float32(b32) orelse return error.InvalidDtype;
    const n = mlx.mlx_array_size(a32);
    if (n != mlx.mlx_array_size(b32)) return error.ShapeMismatch;
    var max_diff: f32 = 0;
    for (0..n) |i| max_diff = @max(max_diff, @abs(ad[i] - bd[i]));
    return max_diff;
}

/// Test seam: forces BOTH hd-256 kernel arms on/off.
pub var fused256_override: ?bool = null;

// ── tests ──

const attn256QsaMask = @import("../../../plugins/qwen4_exp/qsa/gather.zig").attn256QsaMask;
const qsaMaskArm = @import("../../../plugins/qwen4_exp/qsa/sparse_attn.zig").qsaMaskArm;

// ── fused hd-256 prefill attention parity ──

/// Composed reference for the fused kernel tests: MLX's own SDPA (which takes
/// the composed matmul→softmax→matmul path at head_dim 256).
pub fn attn256Reference(
    q: mlx.mlx_array,
    k: mlx.mlx_array,
    v: mlx.mlx_array,
    scale: f32,
    mode: [*:0]const u8,
    mask: mlx.mlx_array,
    s: mlx.mlx_stream,
) !mlx.mlx_array {
    var ref = mlx.mlx_array_new();
    const none = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&ref, q, k, v, scale, mode, if (mask.ctx != null) mask else none, .{ .ctx = null }, false, s));
    return ref;
}

pub fn attn256RandBf16(rnd: std.Random, shape: []const c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const data = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(data);
    for (data) |*x| x.* = rnd.float(f32) - 0.5;
    const f32arr = mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .float32);
    defer _ = mlx.mlx_array_free(f32arr);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, f32arr, .bfloat16, s));
    return out;
}

pub fn attn256RandBf16Scaled(rnd: std.Random, shape: []const c_int, scale: f32, s: mlx.mlx_stream) !mlx.mlx_array {
    const unit = try attn256RandBf16(rnd, shape, s);
    defer _ = mlx.mlx_array_free(unit);
    const k = mlx.mlx_array_new_float(scale);
    defer _ = mlx.mlx_array_free(k);
    var wide = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wide);
    try mlx.check(mlx.mlx_multiply(&wide, unit, k, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, wide, .bfloat16, s));
    return out;
}

/// Largest |element| of an array: the scale a bf16 rounding bar is relative to.
fn attn256MaxAbs(a: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    var a_c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a_c);
    try mlx.check(mlx.mlx_contiguous(&a_c, a, false, s));
    var a32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a32);
    try mlx.check(mlx.mlx_astype(&a32, a_c, .float32, s));
    try mlx.check(mlx.mlx_array_eval(a32));
    const ad = mlx.mlx_array_data_float32(a32) orelse return error.InvalidDtype;
    const n = mlx.mlx_array_size(a32);
    var m: f32 = 0;
    for (0..n) |i| m = @max(m, @abs(ad[i]));
    return m;
}

/// Two bf16 ulps at the reference's own scale.
pub fn attn256UlpBar(ref: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    const m = try attn256MaxAbs(ref, s);
    return 2.0 * m * 0.00390625 + 1.0e-6;
}

/// Cosine of two attention outputs. A max-abs bar alone cannot see a uniform
/// scale error (a dropped or double-counted key shifts every row a little),
/// so the gather parity tests assert BOTH.
pub fn attn256Cosine(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    var a_c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a_c);
    var b_c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b_c);
    try mlx.check(mlx.mlx_contiguous(&a_c, a, false, s));
    try mlx.check(mlx.mlx_contiguous(&b_c, b, false, s));
    var a32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a32);
    var b32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b32);
    try mlx.check(mlx.mlx_astype(&a32, a_c, .float32, s));
    try mlx.check(mlx.mlx_astype(&b32, b_c, .float32, s));
    try mlx.check(mlx.mlx_array_eval(a32));
    try mlx.check(mlx.mlx_array_eval(b32));
    const ad = mlx.mlx_array_data_float32(a32) orelse return error.InvalidDtype;
    const bd = mlx.mlx_array_data_float32(b32) orelse return error.InvalidDtype;
    const n = mlx.mlx_array_size(a32);
    if (n != mlx.mlx_array_size(b32)) return error.ShapeMismatch;
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (0..n) |i| {
        dot += @as(f64, ad[i]) * @as(f64, bd[i]);
        na += @as(f64, ad[i]) * @as(f64, ad[i]);
        nb += @as(f64, bd[i]) * @as(f64, bd[i]);
    }
    if (na == 0 or nb == 0) return error.ZeroNorm;
    return @floatCast(dot / (@sqrt(na) * @sqrt(nb)));
}

test "fusedSdpa256Prefill: causal parity vs composed SDPA (GQA, ragged shapes, chunk offset)" {
    const s = mlx.gpuStream();
    fused256_override = true;
    defer fused256_override = null;
    // The causal arm yields to the stock fused kernel on NAX silicon, so on an
    // M5-class Mac this parity check would DECLINE instead of running — the
    // kernel under test would never be exercised. Pin the arm, not the host.
    sdpa.nax_sdpa_override = false;
    defer sdpa.nax_sdpa_override = null;
    var prng = std.Random.DefaultPrng.init(0x256256);
    const rnd = prng.random();

    // qL=70 (partial 32-tile), kL=193 (partial 16-block), offset 123 (chunked
    // prefill bottom-right alignment), Hq=6/Hk=2 (gqa 3).
    const q_shape = [_]c_int{ 1, 6, 70, 256 };
    const kv_shape = [_]c_int{ 1, 2, 193, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);
    const scale: f32 = 1.0 / 16.0; // 1/sqrt(256)

    const fused = (try fusedSdpa256Prefill(s, q, k, v, scale, 0)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(fused);
    const ref = try attn256Reference(q, k, v, scale, "causal", .{ .ctx = null }, s);
    defer _ = mlx.mlx_array_free(ref);

    // Not byte-parity: the composed path rounds softmax probs to bf16 before
    // the AV matmul, the fused kernel keeps float32 — bf16-rounding-scale
    // differences only.
    // Measured 0.00049 (one bf16 ULP at 0.5 magnitude) on this seed.
    const max_diff = try attn256MaxDiff(fused, ref, s);
    try std.testing.expect(max_diff < 0.005);
}

test "fusedSdpa256Masked: QSA bool-mask parity vs composed 'array' SDPA (GQA, ragged tiles, kv chunking)" {
    const s = mlx.gpuStream();
    fused256_override = true;
    defer fused256_override = null;
    qsa_fused_override = true;
    defer qsa_fused_override = null;
    var prng = std.Random.DefaultPrng.init(0x45a45a);
    const rnd = prng.random();

    const q_shape = [_]c_int{ 1, 6, 70, 256 };
    const kv_shape = [_]c_int{ 1, 2, 193, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);
    const mask = try attn256QsaMask(rnd, 70, 193);
    defer _ = mlx.mlx_array_free(mask);
    const scale: f32 = 1.0 / 16.0;

    const fused = (try fusedSdpa256Masked(s, q, k, v, scale, mask)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(fused);
    const ref = try attn256Reference(q, k, v, scale, "array", mask, s);
    defer _ = mlx.mlx_array_free(ref);
    try std.testing.expect(try attn256MaxDiff(fused, ref, s) < 0.005);

    // Wrong mask shape declines; a shape-valid mask under the kill switch too.
    const bad_shape = [_]c_int{ 1, 1, 70, 192 };
    const bad = try attn256RandBf16(rnd, &bad_shape, s);
    defer _ = mlx.mlx_array_free(bad);
    try std.testing.expect((try fusedSdpa256Masked(s, q, k, v, scale, bad)) == null);
    qsa_fused_override = false;
    try std.testing.expect((try fusedSdpa256Masked(s, q, k, v, scale, mask)) == null);
}

/// Materialized `[B,H,T,264]` padded cache sliced on the last axis to
/// `[lo, lo+256)`: a 256-wide view whose row stride is divisible by 8 elements
/// (16 bytes) while its base sits `lo*2` bytes into the buffer — the shape the
/// `uint4` staging loads of the two old kernels must not read.
pub fn attn256SliceLastDim(
    parent: mlx.mlx_array,
    lo: c_int,
    s: mlx.mlx_stream,
) !mlx.mlx_array {
    const ps = mlx.getShape(parent);
    const one = [_]c_int{ 1, 1, 1, 1 };
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    const stop = [_]c_int{ ps[0], ps[1], ps[2], lo + 256 };
    try mlx.check(mlx.mlx_slice(&out, parent, &[_]c_int{ 0, 0, 0, lo }, 4, &stop, 4, &one, 4, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

test "fusedSdpa256Masked: a last-axis slice with an unaligned base declines and the fallback reads exactly" {
    // `[3,259)` has a 6-byte base: the row stride (264) is divisible by 8 so the
    // existing stride checks pass, but `msv_attn_p256`'s `uint4` K/V staging
    // reads wrong values. The gate must decline it and the caller's fallback
    // (`qsaMaskArm`, the same arm production takes) must serve the shape.
    // `[0,256)` is the aligned twin: it keeps the fused kernel.
    const s = mlx.gpuStream();
    fused256_override = true;
    defer fused256_override = null;
    qsa_fused_override = true;
    defer qsa_fused_override = null;
    var prng = std.Random.DefaultPrng.init(0x5E7_5E7);
    const rnd = prng.random();

    const q_shape = [_]c_int{ 1, 6, 70, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k_pad = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, 193, 264 }, s);
    defer _ = mlx.mlx_array_free(k_pad);
    const v_pad = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, 193, 264 }, s);
    defer _ = mlx.mlx_array_free(v_pad);
    try mlx.check(mlx.mlx_array_eval(k_pad));
    try mlx.check(mlx.mlx_array_eval(v_pad));
    const mask = try attn256QsaMask(rnd, 70, 193);
    defer _ = mlx.mlx_array_free(mask);
    const scale: f32 = 1.0 / 16.0;

    const k = try attn256SliceLastDim(k_pad, 3, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256SliceLastDim(v_pad, 3, s);
    defer _ = mlx.mlx_array_free(v);
    try std.testing.expect((try fusedSdpa256Masked(s, q, k, v, scale, mask)) == null);
    const arm = try qsaMaskArm(s, q, k, v, scale, mask);
    defer _ = mlx.mlx_array_free(arm);
    const ref = try attn256Reference(q, k, v, scale, "array", mask, s);
    defer _ = mlx.mlx_array_free(ref);
    try std.testing.expect(try attn256MaxDiff(arm, ref, s) < 0.005);

    // An aligned twin keeps the fast path.
    const k_al = try attn256SliceLastDim(k_pad, 0, s);
    defer _ = mlx.mlx_array_free(k_al);
    const v_al = try attn256SliceLastDim(v_pad, 0, s);
    defer _ = mlx.mlx_array_free(v_al);
    const fast = (try fusedSdpa256Masked(s, q, k_al, v_al, scale, mask)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(fast);
    const ref_al = try attn256Reference(q, k_al, v_al, scale, "array", mask, s);
    defer _ = mlx.mlx_array_free(ref_al);
    try std.testing.expect(try attn256MaxDiff(fast, ref_al, s) < 0.005);
}

test "fusedSdpa256Prefill: sliding-band parity vs composed 'array' mask (Gemma local layers)" {
    const s = mlx.gpuStream();
    fused256_override = true;
    defer fused256_override = null;
    var prng = std.Random.DefaultPrng.init(0xBA9D);
    const rnd = prng.random();

    const qL: c_int = 70;
    const kL: c_int = 193;
    const sw: c_int = 40;
    const q_shape = [_]c_int{ 1, 4, qL, 256 };
    const kv_shape = [_]c_int{ 1, 4, kL, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);
    const scale: f32 = 1.0;

    // Reference mask: exactly createSlidingWindowMask semantics — masked when
    // col > row_abs (causal) or row_abs - col >= sw, row_abs = kL - qL + r.
    const mask_n: usize = @intCast(qL * kL);
    const mask_data = try std.testing.allocator.alloc(f32, mask_n);
    defer std.testing.allocator.free(mask_data);
    const off: c_int = kL - qL;
    var r: c_int = 0;
    while (r < qL) : (r += 1) {
        var c: c_int = 0;
        while (c < kL) : (c += 1) {
            const row_abs = off + r;
            const masked = (c > row_abs) or (row_abs - c >= sw);
            mask_data[@intCast(r * kL + c)] = if (masked) -std.math.inf(f32) else 0.0;
        }
    }
    const mask_shape = [_]c_int{ 1, 1, qL, kL };
    const mask_f32 = mlx.mlx_array_new_data(mask_data.ptr, &mask_shape, 4, .float32);
    defer _ = mlx.mlx_array_free(mask_f32);
    var mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask);
    try mlx.check(mlx.mlx_astype(&mask, mask_f32, .bfloat16, s));

    const fused = (try fusedSdpa256Prefill(s, q, k, v, scale, sw)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(fused);
    const ref = try attn256Reference(q, k, v, scale, "array", mask, s);
    defer _ = mlx.mlx_array_free(ref);

    const max_diff = try attn256MaxDiff(fused, ref, s);
    try std.testing.expect(max_diff < 0.02);
}

test "fusedSinkAttnPrefill: q/k 192, v 128 with sinks matches MLX sdpa (causal and band, with a cache prefix)" {
    // MiMo's attention: K 192 / V 128, GQA 8, a learned sink per head. The
    // reference is MLX's composed sdpa with the same sinks and an explicit mask.
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x5191);
    const rnd = prng.random();
    fused256_override = true;
    defer fused256_override = null;
    const Hq: c_int = 16;
    const Hk: c_int = 2;
    const qL: c_int = 96;
    for ([_]c_int{ 96, 200 }) |kL| for ([_]c_int{ 0, 40 }) |sw| {
        const q = try attn256RandBf16(rnd, &.{ 1, Hq, qL, 192 }, s);
        defer _ = mlx.mlx_array_free(q);
        const k = try attn256RandBf16(rnd, &.{ 1, Hk, kL, 192 }, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try attn256RandBf16(rnd, &.{ 1, Hk, kL, 128 }, s);
        defer _ = mlx.mlx_array_free(v);
        const sinks = try attn256RandBf16Scaled(rnd, &.{Hq}, 4.0, s);
        defer _ = mlx.mlx_array_free(sinks);
        const scale: f32 = 1.0 / @sqrt(192.0);

        const mask_data = try std.testing.allocator.alloc(f32, @intCast(qL * kL));
        defer std.testing.allocator.free(mask_data);
        for (0..@intCast(qL)) |r| for (0..@intCast(kL)) |c| {
            const row_abs: c_int = kL - qL + @as(c_int, @intCast(r));
            const col: c_int = @intCast(c);
            const masked = col > row_abs or (sw > 0 and row_abs - col >= sw);
            mask_data[r * @as(usize, @intCast(kL)) + c] = if (masked) -std.math.inf(f32) else 0.0;
        };
        const mask_f32 = mlx.mlx_array_new_data(mask_data.ptr, &[_]c_int{ 1, 1, qL, kL }, 4, .float32);
        defer _ = mlx.mlx_array_free(mask_f32);
        var mask = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mask);
        try mlx.check(mlx.mlx_astype(&mask, mask_f32, .bfloat16, s));

        const fused = (try fusedSinkAttnPrefill(s, q, k, v, scale, sw, sinks)) orelse return error.FusedDeclined;
        defer _ = mlx.mlx_array_free(fused);
        var ref = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ref);
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&ref, q, k, v, scale, "array", mask, sinks, false, s));
        const max_diff = try attn256MaxDiff(fused, ref, s);
        std.testing.expect(max_diff < 0.02) catch |e| {
            std.debug.print("\n[sink-attn] kL={d} sw={d} max diff {d:.4}\n", .{ kL, sw, max_diff });
            return e;
        };
    };
}

test "fusedSdpa256Prefill: declines cleanly outside its envelope" {
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0xDEC1);
    const rnd = prng.random();

    // Wrong head_dim -> null.
    const q128_shape = [_]c_int{ 1, 4, 64, 128 };
    const q128 = try attn256RandBf16(rnd, &q128_shape, s);
    defer _ = mlx.mlx_array_free(q128);
    fused256_override = true;
    defer fused256_override = null;
    try std.testing.expect((try fusedSdpa256Prefill(s, q128, q128, q128, 1.0, 0)) == null);

    // Decode/verify shapes (q_len < 16) -> null (sdpa_vector owns hd 256
    // there; stealing MTP verify forwards measured decode 48 -> 18 tok/s;
    // the 16 floor matches oMLX's _MIN_ROUTE_Q_LEN — with causal fused
    // default-on, a depth-8 MTP verify is q_len 9 and must NOT route here).
    const q1_shape = [_]c_int{ 1, 4, 1, 256 };
    const q8_shape = [_]c_int{ 1, 4, 8, 256 };
    const q15_shape = [_]c_int{ 1, 4, 15, 256 };
    const kv_shape = [_]c_int{ 1, 4, 64, 256 };
    const q1 = try attn256RandBf16(rnd, &q1_shape, s);
    defer _ = mlx.mlx_array_free(q1);
    const q8 = try attn256RandBf16(rnd, &q8_shape, s);
    defer _ = mlx.mlx_array_free(q8);
    const q15 = try attn256RandBf16(rnd, &q15_shape, s);
    defer _ = mlx.mlx_array_free(q15);
    const k1 = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k1);
    try std.testing.expect((try fusedSdpa256Prefill(s, q1, k1, k1, 1.0, 0)) == null);
    try std.testing.expect((try fusedSdpa256Prefill(s, q8, k1, k1, 1.0, 0)) == null);
    try std.testing.expect((try fusedSdpa256Prefill(s, q15, k1, k1, 1.0, 0)) == null);

    // Kill switch -> null even for a conforming call.
    fused256_override = false;
    const q_shape = [_]c_int{ 1, 4, 64, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    try std.testing.expect((try fusedSdpa256Prefill(s, q, q, q, 1.0, 0)) == null);
}

test "fusedSdpa256Prefill: both arms FUSED off NAX, causal yields to stock ON NAX" {
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x4A7E);
    const rnd = prng.random();

    // No master-switch override: the causal arm's historical net-loss (every
    // pre-budget ratio-gated variant lost same-boot on the 27B) was the IOGPU
    // preemption class — with the kv-chunk dispatch budget it wins live
    // (2026-07-22 same-session A/B: +2.9%/+2.3%/+4.6% at 8K/16K/32K on the
    // 27B), so causal is default-on now.
    std.debug.assert(fused256_override == null);
    const q_shape = [_]c_int{ 1, 6, 64, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const kv_shape = [_]c_int{ 1, 2, 64, 256 };
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);

    // Off NAX both arms engage — the budgeted-dispatch flip this test was
    // written for. The preference is PINNED rather than read off the host:
    // `naxSdpaPreferred()` defaults to whether the machine has NAX, so leaving
    // it to the silicon makes this assertion true on an M4 and false on an M5.
    sdpa.nax_sdpa_override = false;
    defer sdpa.nax_sdpa_override = null;
    {
        const causal = try fusedSdpa256Prefill(s, q, k, k, 1.0, 0);
        try std.testing.expect(causal != null);
        if (causal) |f| _ = mlx.mlx_array_free(f);
    }

    // On NAX the causal arm yields to the stock fused sdpa (mlx's own hd-256
    // kernel) while the BAND arm stays ours — mlx's NAX kernel takes no band.
    // Nothing pinned that asymmetry before; a decline is invisible from the
    // outside, since the composed fallback is output-equivalent.
    sdpa.nax_sdpa_override = true;
    {
        const causal = try fusedSdpa256Prefill(s, q, k, k, 1.0, 0);
        try std.testing.expect(causal == null);
    }

    const banded = try fusedSdpa256Prefill(s, q, k, k, 1.0, 40);
    try std.testing.expect(banded != null);
    if (banded) |f| _ = mlx.mlx_array_free(f);
}

test "fused256KvChunkLen: BK alignment, one-block floor, kl cap, budget-off" {
    const t = std.testing;
    // budget <= 0: single dispatch covering the full key axis.
    try t.expectEqual(@as(c_int, 193), fused256KvChunkLen(1, 6, 70, 193, 0));
    try t.expectEqual(@as(c_int, 65536), fused256KvChunkLen(1, 24, 2048, 65536, -1));
    // 27B live shape (24 heads, 2048-chunk q) at the default budget:
    // 250M / (24*2048) = 5086 keys -> floored to the BK(32) multiple 5056.
    try t.expectEqual(@as(c_int, 5056), fused256KvChunkLen(1, 24, 2048, 65536, FUSED256_DEFAULT_DISPATCH_BUDGET));
    // Never below one BK block even at absurdly small budgets…
    try t.expectEqual(@as(c_int, 32), fused256KvChunkLen(1, 24, 2048, 65536, 1));
    // …and never above the actual key length.
    try t.expectEqual(@as(c_int, 193), fused256KvChunkLen(1, 6, 70, 193, 1 << 40));
}

test "fusedSdpa256Prefill: budgeted kv chunking engages and is exact vs single dispatch" {
    const s = mlx.gpuStream();
    fused256_override = true;
    defer fused256_override = null;
    // Causal arm: declines on NAX silicon unless the preference is pinned.
    sdpa.nax_sdpa_override = false;
    defer sdpa.nax_sdpa_override = null;
    var prng = std.Random.DefaultPrng.init(0xC4A2);
    const rnd = prng.random();

    // Hq=6/Hk=2, qL=70 (ragged q tile), kL=193 (ragged kv block). Per-key
    // work = 6*70 = 420; budget 40000 -> chunk 64 -> dispatches over
    // [0,64) [64,128) [128,192) [192,193) — 4, incl. a 1-key ragged tail.
    const q_shape = [_]c_int{ 1, 6, 70, 256 };
    const kv_shape = [_]c_int{ 1, 2, 193, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);
    const scale: f32 = 1.0 / 16.0;

    fused256_budget_override = 0;
    const single = (try fusedSdpa256Prefill(s, q, k, v, scale, 0)) orelse {
        fused256_budget_override = null;
        return error.FusedDeclined;
    };
    defer _ = mlx.mlx_array_free(single);
    try std.testing.expectEqual(@as(u32, 1), fused256_last_dispatch_count);

    fused256_budget_override = 40_000;
    defer fused256_budget_override = null;
    const chunked = (try fusedSdpa256Prefill(s, q, k, v, scale, 0)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(chunked);
    // Engagement counted, not inferred: the budget must actually split.
    try std.testing.expectEqual(@as(u32, 4), fused256_last_dispatch_count);

    // The carry rides fp32 buffers — the exact register precision — so the
    // chunked result is BIT-IDENTICAL to the single dispatch, not just close.
    const max_diff = try attn256MaxDiff(single, chunked, s);
    try std.testing.expect(max_diff == 0.0);

    // And still correct in absolute terms vs the composed reference.
    const ref = try attn256Reference(q, k, v, scale, "causal", .{ .ctx = null }, s);
    defer _ = mlx.mlx_array_free(ref);
    const ref_diff = try attn256MaxDiff(chunked, ref, s);
    try std.testing.expect(ref_diff < 0.005);
}

test "fusedSdpa256Prefill: band arm never chunks (single dispatch under a tiny budget)" {
    const s = mlx.gpuStream();
    fused256_override = true;
    defer fused256_override = null;
    fused256_budget_override = 1; // would force max chunking on the causal arm
    defer fused256_budget_override = null;
    var prng = std.Random.DefaultPrng.init(0xBA9D2);
    const rnd = prng.random();

    const qL: c_int = 70;
    const kL: c_int = 193;
    const sw: c_int = 40;
    const q_shape = [_]c_int{ 1, 4, qL, 256 };
    const kv_shape = [_]c_int{ 1, 4, kL, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);

    const fused = (try fusedSdpa256Prefill(s, q, k, v, 1.0, sw)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(fused);
    // The band arm's block skip already bounds per-dispatch work; chunking it
    // would only add carry traffic. It must stay a single dispatch.
    try std.testing.expectEqual(@as(u32, 1), fused256_last_dispatch_count);

    // Unchanged band semantics (same reference as the band parity test).
    const mask_n: usize = @intCast(qL * kL);
    const mask_data = try std.testing.allocator.alloc(f32, mask_n);
    defer std.testing.allocator.free(mask_data);
    const off: c_int = kL - qL;
    var r: c_int = 0;
    while (r < qL) : (r += 1) {
        var c: c_int = 0;
        while (c < kL) : (c += 1) {
            const row_abs = off + r;
            const masked = (c > row_abs) or (row_abs - c >= sw);
            mask_data[@intCast(r * kL + c)] = if (masked) -std.math.inf(f32) else 0.0;
        }
    }
    const mask_shape = [_]c_int{ 1, 1, qL, kL };
    const mask_f32 = mlx.mlx_array_new_data(mask_data.ptr, &mask_shape, 4, .float32);
    defer _ = mlx.mlx_array_free(mask_f32);
    var mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask);
    try mlx.check(mlx.mlx_astype(&mask, mask_f32, .bfloat16, s));
    const ref = try attn256Reference(q, k, v, 1.0, "array", mask, s);
    defer _ = mlx.mlx_array_free(ref);
    const max_diff = try attn256MaxDiff(fused, ref, s);
    try std.testing.expect(max_diff < 0.02);
}

test "fusedSdpa256Prefill: causal parity at Qwen 24q/4kv geometry (gqa 6, ragged 64-row tile)" {
    const s = mlx.gpuStream();
    fused256_override = true;
    defer fused256_override = null;
    // Causal arm: declines on NAX silicon unless the preference is pinned.
    sdpa.nax_sdpa_override = false;
    defer sdpa.nax_sdpa_override = null;
    var prng = std.Random.DefaultPrng.init(0x27B27B);
    const rnd = prng.random();

    // The production Qwen3.6-27B full-attention geometry: 24 q heads over 4
    // kv heads (gqa 6), hd 256. qL=97 exercises a partial 64-row q tile,
    // kL=250 a partial kv block; offset 153 = chunked-prefill alignment.
    const q_shape = [_]c_int{ 1, 24, 97, 256 };
    const kv_shape = [_]c_int{ 1, 4, 250, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);
    const scale: f32 = 1.0 / 16.0;

    const fused = (try fusedSdpa256Prefill(s, q, k, v, scale, 0)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(fused);
    const ref = try attn256Reference(q, k, v, scale, "causal", .{ .ctx = null }, s);
    defer _ = mlx.mlx_array_free(ref);

    const max_diff = try attn256MaxDiff(fused, ref, s);
    try std.testing.expect(max_diff < 0.005);
}
