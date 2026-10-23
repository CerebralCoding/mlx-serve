//! sdk/blocks/attention/sinks, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns sinkAttnWith, sinkCacheUpdate, sinkAttnSlots, sinkAttnCore, sinkAttnOut, mimoDecodeCopies.

const mlx = @import("../../../mlx/ffi.zig");
const dec_attn = @import("dec_attn.zig");
const sliding_mod = @import("../../../engines/mlx/kv/sliding.zig");
const MAX_BATCH_ROWS = @import("../../../engines/mlx/batched/layered.zig").MAX_BATCH_ROWS;
const ForwardCtx = @import("../../../engines/mlx/forward_ctx.zig").ForwardCtx;
const KVCache = @import("../../../engines/mlx/kv/cache.zig").KVCache;
const slidingViewFor = @import("../../../engines/mlx/kv/sliding.zig").slidingViewFor;
const FullAttnWeights = @import("../../../engines/mlx/layer_weights.zig").FullAttnWeights;
const scalarOf = @import("../../../engines/mlx/load/weights.zig").scalarOf;
const Transformer = @import("../../../engines/mlx/transformer.zig").Transformer;
const constTableAs = @import("../../../mlx/dtype.zig").constTableAs;
const FUSED256_MIN_Q_LEN = @import("attn256.zig").FUSED256_MIN_Q_LEN;
const fusedSinkAttnPrefill = @import("attn256.zig").fusedSinkAttnPrefill;
const DenseKVView = @import("kv_view.zig").DenseKVView;

/// gpt_oss / MiMo attention. Plain GQA (64/8 at hd 64 on the 20B) with four
/// departures from `gatedFullAttnWith`, which is why it is its own arm:
///
///   1. LEARNED PER-HEAD SINKS. `fa.sinks` rides into mlx's fused SDPA as
///      an extra softmax-denominator column. It is a function constant
///      (`_has_sinks_`) in the Metal kernel, so this costs no fallback.
///   2. ALTERNATING SLIDING/FULL layers (`layer_types`), window 128.
///      `gatedFullAttnWith` has no local-mask path at all.
///   3. ADDITIVE q/k/v/o biases.
///   4. NO QK norm.
///
/// gpt_oss RoPE is ONE configuration for every layer (its sliding theta is
/// the global one, YaRN for both). MiMo reads every geometry per layer:
/// global 4 / sliding 8 KV heads, K 192 / V 128, RoPE on 64 channels, two
/// thetas, sinks on the sliding layers only, V scaled before caching.
///
/// The fused quantized-KV attention path is deliberately NOT wired: those
/// kernels are ours and take no sink argument, so engaging them would drop
/// the sinks silently. Under `--kv-quant` this arm reads the dequantized
/// `DenseKVView` that `cache.update` already returns.
pub fn sinkAttnWith(
    self: *Transformer,
    ctx: *ForwardCtx,
    x: mlx.mlx_array,
    fa: *const FullAttnWeights,
    layer: u32,
    offset: c_int,
    batch: c_int,
    seq_len: c_int,
    is_prefill: bool,
    local_prefill_mask: *mlx.mlx_array,
    local_decode_mask: mlx.mlx_array,
) !mlx.mlx_array {
    const cfg = &self.config;
    const is_full = cfg.isGlobalLayer(layer);
    const h_count: c_int = @intCast(cfg.layerNumHeads(layer));
    const kv_h: c_int = @intCast(cfg.layerKVHeads(layer));
    const hd: c_int = @intCast(cfg.layerHeadDim(layer));
    const vd: c_int = @intCast(cfg.layerVHeadDim(layer));
    const rope_dims: c_int = @intCast(cfg.layerRopeDims(layer));
    const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.query_pre_attn_scalar)));
    const q_shape = [_]c_int{ batch, seq_len, h_count, hd };
    const kv_shape = [_]c_int{ batch, seq_len, kv_h, hd };
    const v_shape = [_]c_int{ batch, seq_len, kv_h, vd };
    const flat_shape = [_]c_int{ batch, seq_len, h_count * vd };
    const perm = [_]c_int{ 0, 2, 1, 3 };
    const decode_shape = mimoDecodeCopies(cfg.isMimo(), batch, is_prefill, ctx.spec_verify, seq_len);

    const use_yarn = self.rope_freqs_yarn != null;
    const rope_base = mlx.mlx_optional_float{
        .value = if (is_full) cfg.rope_theta else cfg.rope_local_base_freq,
        .has_value = !use_yarn, // precomputed yarn freqs override the base
    };
    const rope_freqs: mlx.mlx_array = if (use_yarn) self.rope_freqs_yarn.? else .{ .ctx = null };

    const none_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none_mask);

    // Q/K/V projections, each with its additive bias.
    const q_proj = try self.attnProjBias(x, fa.q_w, fa.q_s, fa.q_b, fa.q_bias, decode_shape, layer);
    defer _ = mlx.mlx_array_free(q_proj);
    const k_proj = try self.attnProjBias(x, fa.k_w, fa.k_s, fa.k_b, fa.k_bias, decode_shape, layer);
    defer _ = mlx.mlx_array_free(k_proj);
    const v_proj = try self.attnProjBias(x, fa.v_w, fa.v_s, fa.v_b, fa.v_bias, decode_shape, layer);
    defer _ = mlx.mlx_array_free(v_proj);

    // Reshape → transpose to [B, H, S, D] → RoPE. No QK norm on this arch.
    var q_r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_r);
    try mlx.check(mlx.mlx_reshape(&q_r, q_proj, &q_shape, 4, self.s));
    var q_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_t);
    try mlx.check(mlx.mlx_transpose_axes(&q_t, q_r, &perm, 4, self.s));

    var k_r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_r);
    try mlx.check(mlx.mlx_reshape(&k_r, k_proj, &kv_shape, 4, self.s));
    var k_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_t);
    try mlx.check(mlx.mlx_transpose_axes(&k_t, k_r, &perm, 4, self.s));

    var q_rope = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_rope);
    var k_rope = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_rope);
    if (ctx.batch_rope_offsets) |off_arr| {
        // Batched decode: every slot sits at its own position.
        try mlx.check(mlx.mlx_fast_rope_dynamic(&q_rope, q_t, rope_dims, false, rope_base, 1.0, off_arr, rope_freqs, self.s));
        try mlx.check(mlx.mlx_fast_rope_dynamic(&k_rope, k_t, rope_dims, false, rope_base, 1.0, off_arr, rope_freqs, self.s));
    } else {
        try mlx.check(mlx.mlx_fast_rope(&q_rope, q_t, rope_dims, false, rope_base, 1.0, offset, rope_freqs, self.s));
        try mlx.check(mlx.mlx_fast_rope(&k_rope, k_t, rope_dims, false, rope_base, 1.0, offset, rope_freqs, self.s));
    }

    // YaRN mscale on the rotated dims (full rotary here, so the whole
    // head). Cast to q/k's dtype first — an f32 table would promote the
    // product and upcast every downstream weight read (measured 3x on
    // Laguna XS).
    if (use_yarn) {
        if (self.yarn_mscale) |ms_f32| {
            const ms = try constTableAs(ms_f32, mlx.mlx_array_dtype(q_rope), &self.yarn_mscale_cast, self.s);
            var q_scaled = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_multiply(&q_scaled, q_rope, ms, self.s));
            _ = mlx.mlx_array_free(q_rope);
            q_rope = q_scaled;
            var k_scaled = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_multiply(&k_scaled, k_rope, ms, self.s));
            _ = mlx.mlx_array_free(k_rope);
            k_rope = k_scaled;
        }
    }

    var v_r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v_r);
    try mlx.check(mlx.mlx_reshape(&v_r, v_proj, &v_shape, 4, self.s));
    var v_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v_t);
    try mlx.check(mlx.mlx_transpose_axes(&v_t, v_r, &perm, 4, self.s));
    // MiMo-V2.6 caches V scaled (`attention_value_scale`), in V's own dtype.
    if (cfg.attention_value_scale != 1.0) {
        const vs = try scalarOf(cfg.attention_value_scale, mlx.mlx_array_dtype(v_t), self.s);
        defer _ = mlx.mlx_array_free(vs);
        var scaled = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&scaled, v_t, vs, self.s));
        _ = mlx.mlx_array_free(v_t);
        v_t = scaled;
    }

    // Sliding layers read only the tail their queries can reach, and the
    // width is a property of the whole BLOCK: `window + q_len - 1`,
    // because the FIRST query of the block reaches back furthest. That is
    // `slidingViewFor`'s job, and it is the SAME call every mask below is
    // built from — hand the view one width and the mask another and SDPA
    // fails to broadcast them, which is an uncatchable MLX error rather
    // than a bad answer. `cfg.sliding_window` here reads correct at
    // q_len == 1 (span == window) and kills the process on the first
    // spec-verify block or prefill chunk past the window.
    if (ctx.batch_slots) |slots| {
        const attn_b = try self.sinkAttnSlots(slots, q_rope, k_rope, v_t, fa, layer, seq_len, attn_scale);
        defer _ = mlx.mlx_array_free(attn_b);
        return self.sinkAttnOut(attn_b, fa, &flat_shape, decode_shape, layer);
    }
    const sliding = slidingViewFor(cfg, offset + seq_len, seq_len);
    const max_kv: u32 = if (is_full) 0 else sliding.span;
    var kv_view = try self.sinkCacheUpdate(ctx.cache, layer, k_rope, v_t, max_kv);
    defer kv_view.deinit();
    const attn_out = try self.sinkAttnCore(q_rope, kv_view.k, kv_view.v, fa, layer, offset, seq_len, sliding.kv_len, attn_scale, local_prefill_mask, local_decode_mask);
    defer _ = mlx.mlx_array_free(attn_out);
    return self.sinkAttnOut(attn_out, fa, &flat_shape, decode_shape, layer);
}

/// Each batched slot's attention over its own cache: the slot's row of the
/// stacked q/k/v, appended at the slot's own length and read through its own
/// sliding view. Returns `[N, H, S, V]`.
/// MiMo's sliding layers cache a ring of their newest rows (`ModelConfig.slidingRing`).
pub fn sinkCacheUpdate(self: *Transformer, cache: *KVCache, layer: u32, k: mlx.mlx_array, v: mlx.mlx_array, max_kv: u32) !DenseKVView {
    const cfg = &self.config;
    if (max_kv == 0 or !cfg.slidingRing() or sliding_mod.sliding_ring_off_for_test) return cache.update(layer, k, v, self.s, max_kv);
    return cache.updateSliding(layer, k, v, self.s, max_kv, cfg.sliding_window, cfg.slidingKeepRows());
}

pub fn sinkAttnSlots(self: *Transformer, slots: []const *ForwardCtx, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, fa: *const FullAttnWeights, layer: u32, seq_len: c_int, attn_scale: f32) !mlx.mlx_array {
    const is_full = self.config.isGlobalLayer(layer);
    var outs: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
    var n: usize = 0;
    defer for (outs[0..n]) |o| {
        _ = mlx.mlx_array_free(o);
    };
    for (slots, 0..) |slot, i| {
        var rows: [3]mlx.mlx_array = undefined;
        for (&rows, [_]mlx.mlx_array{ q, k, v }) |*dst, src| {
            const sh = mlx.getShape(src);
            dst.* = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(dst, src, &.{ @intCast(i), 0, 0, 0 }, 4, &.{ @intCast(i + 1), sh[1], sh[2], sh[3] }, 4, &.{ 1, 1, 1, 1 }, 4, self.s));
        }
        defer for (rows) |r| {
            _ = mlx.mlx_array_free(r);
        };
        const offset: c_int = @intCast(slot.cache.seqLen(layer));
        const sliding = slidingViewFor(&self.config, offset + seq_len, seq_len);
        var view = try self.sinkCacheUpdate(slot.cache, layer, rows[1], rows[2], if (is_full) 0 else sliding.span);
        defer view.deinit();
        var prefill_mask = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(prefill_mask);
        // A slot's decode view is exactly its window, so no decode mask is needed.
        outs[n] = try self.sinkAttnCore(rows[0], view.k, view.v, fa, layer, offset, seq_len, sliding.kv_len, attn_scale, &prefill_mask, .{ .ctx = null });
        n += 1;
    }
    const vec = mlx.mlx_vector_array_new_data(&outs, n);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 0, self.s));
    return out;
}

/// Attention of `q` [B, H, S, D] (MiMo's 192, Flint's 128) over one cache view at its own `offset`
/// (keys before this block): fused kernels where they serve, MLX's SDPA with
/// the sinks, causal at S > 1 and banded to the window on sliding layers.
/// `prefill_mask` is built on first use and kept by the caller.
pub fn sinkAttnCore(self: *Transformer, q_rope: mlx.mlx_array, full_k: mlx.mlx_array, full_v: mlx.mlx_array, fa: *const FullAttnWeights, layer: u32, offset: c_int, seq_len: c_int, view_kv_len: c_int, attn_scale: f32, local_prefill_mask: *mlx.mlx_array, local_decode_mask: mlx.mlx_array) !mlx.mlx_array {
    const cfg = &self.config;
    const is_full = cfg.isGlobalLayer(layer);
    const is_prefill = seq_len > 1;
    const none_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none_mask);
    var attn_out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(attn_out);
    const sinks: ?mlx.mlx_array = if (fa.sinks.ctx != null) fa.sinks else null;
    const fused: ?mlx.mlx_array = if (is_prefill and seq_len >= FUSED256_MIN_Q_LEN)
        try fusedSinkAttnPrefill(self.s, q_rope, full_k, full_v, attn_scale, if (is_full) 0 else @intCast(cfg.sliding_window), sinks)
    else if (is_full and sinks == null)
        try dec_attn.attention(self.s, q_rope, full_k, full_v, attn_scale)
    else
        null;
    if (fused) |f| {
        _ = mlx.mlx_array_free(attn_out);
        attn_out = f;
    } else if (is_full) {
        const mode: [*:0]const u8 = if (is_prefill) "causal" else "";
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, mode, none_mask, fa.sinks, false, self.s));
    } else {
        const sw: c_int = @intCast(cfg.sliding_window);
        const total_kv: c_int = offset + seq_len;
        if (is_prefill and total_kv <= sw) {
            // Window degenerates to plain causal.
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "causal", none_mask, fa.sinks, false, self.s));
        } else if (is_prefill) {
            if (local_prefill_mask.ctx == null) {
                // Built at the TRIMMED width, matching the view.
                local_prefill_mask.* = try self.createSlidingWindowMask(seq_len, view_kv_len, sw);
            }
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "array", local_prefill_mask.*, fa.sinks, false, self.s));
        } else if (total_kv <= sw or local_decode_mask.ctx == null) {
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "", none_mask, fa.sinks, false, self.s));
        } else {
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "array", local_decode_mask, fa.sinks, false, self.s));
        }
    }
    return attn_out;
}

/// [B,H,S,D] → [B,S,H*D] → o_proj (+ additive bias).
pub fn sinkAttnOut(self: *Transformer, attn_out: mlx.mlx_array, fa: *const FullAttnWeights, flat_shape: *const [3]c_int, decode_shape: bool, layer: u32) !mlx.mlx_array {
    var attn_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn_t);
    try mlx.check(mlx.mlx_transpose_axes(&attn_t, attn_out, &[_]c_int{ 0, 2, 1, 3 }, 4, self.s));
    var attn_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn_flat);
    try mlx.check(mlx.mlx_reshape(&attn_flat, attn_t, flat_shape, 3, self.s));
    return self.attnProjBias(attn_flat, fa.o_w, fa.o_s, fa.o_b, fa.o_bias, decode_shape, layer);
}

/// MiMo's attention reads its LOSSY decode-attn-quant copies at decode width and on a
/// speculative verify window (drafts behind one committed token, the rows its serial
/// steps would read); a prompt chunk prefills dense whatever its length.
pub fn mimoDecodeCopies(is_mimo: bool, batch: c_int, is_prefill: bool, spec_verify: bool, seq_len: c_int) bool {
    if (batch != 1) return false;
    if (!is_prefill) return true;
    return is_mimo and spec_verify and seq_len <= dec_attn.MAX_ROWS;
}
