//! plugins/gemma4/attention, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns gemma4MoeAttnWith.

const mlx = @import("../../mlx/ffi.zig");
const ForwardCtx = @import("../../engines/mlx/forward_ctx.zig").ForwardCtx;
const slidingViewFor = @import("../../engines/mlx/kv/sliding.zig").slidingViewFor;
const FullAttnWeights = @import("../../engines/mlx/layer_weights.zig").FullAttnWeights;
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;
const fusedSdpa256Prefill = @import("../../sdk/blocks/attention/attn256.zig").fusedSdpa256Prefill;
const sdpaForceFused = @import("../../sdk/blocks/attention/sdpa.zig").sdpaForceFused;
const FUSED_ROWS_MAX_M = @import("../../sdk/blocks/linear/fused_rows.zig").FUSED_ROWS_MAX_M;
const fusedQkNormRope256 = @import("../../sdk/blocks/rope/qk_norm_rope.zig").fusedQkNormRope256;
const qkNormRopeFusedEnabled = @import("../../sdk/blocks/rope/qk_norm_rope.zig").qkNormRopeFusedEnabled;

// ── Gemma 4 Full Attention for MoE layers ──
// Handles dual head dims, v_norm, sliding window, per-layer RoPE.

pub fn gemma4MoeAttnWith(
    self: *Transformer,
    ctx: *ForwardCtx,
    x: mlx.mlx_array,
    fa: *const FullAttnWeights,
    layer: u32,
    offset: c_int,
    batch: c_int,
    seq_len: c_int,
    is_prefill: bool,
    // Pointer: the caller may have SKIPPED the eager sliding-mask build
    // (fused hd-256 kernel expected to band-mask in-kernel); if a per-call
    // check declines, the mask is built here once and cached back.
    local_prefill_mask: *mlx.mlx_array,
    local_decode_mask: mlx.mlx_array,
) !mlx.mlx_array {
    const cfg = &self.config;
    const is_global = cfg.isGlobalLayer(layer);
    const h_count: c_int = @intCast(cfg.num_attention_heads);

    // Per-layer dimensions
    const cur_hd: u32 = cfg.layerHeadDim(layer);
    const cur_kv_h: u32 = cfg.layerKVHeads(layer);
    const q_shape = [_]c_int{ batch, seq_len, h_count, @intCast(cur_hd) };
    const kv_shape = [_]c_int{ batch, seq_len, @intCast(cur_kv_h), @intCast(cur_hd) };
    const flat_shape = [_]c_int{ batch, seq_len, @intCast(@as(u32, @intCast(h_count)) * cur_hd) };

    // Per-layer RoPE
    // RoPE: proportional for global layers (custom freqs), standard for local
    const use_prop_rope = is_global and self.rope_freqs_global != null;
    const rope_dims: c_int = @intCast(cur_hd); // full head dim for proportional (freqs handle partial)
    const rope_base = mlx.mlx_optional_float{ .value = if (is_global) cfg.rope_theta else cfg.rope_local_base_freq, .has_value = !use_prop_rope };
    const rope_scale: f32 = if (use_prop_rope) 1.0 else if (is_global) (1.0 / cfg.rope_scaling_factor) else 1.0;
    const rope_freqs: mlx.mlx_array = if (use_prop_rope) self.rope_freqs_global.? else .{ .ctx = null };

    // Gemma 4: scale = 1.0 (QK-norm handles normalization)
    const attn_scale: f32 = 1.0;

    const perm = [_]c_int{ 0, 2, 1, 3 };
    const none_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none_mask);

    // Q/K/V projections: one row-fused matmul at decode width.
    const fused_rows = fa.qkv.w.ctx != null and batch * seq_len <= FUSED_ROWS_MAX_M;
    var qkv_all = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(qkv_all);
    if (fused_rows) qkv_all = try self.qmatmul(x, fa.qkv.w, fa.qkv.s, fa.qkv.b);
    var q_proj = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_proj);
    var k_proj = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_proj);
    var v_proj = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v_proj);
    if (fused_rows) {
        try fa.qkv.part(&q_proj, qkv_all, 0, self.s);
        try fa.qkv.part(&k_proj, qkv_all, 1, self.s);
        try fa.qkv.part(&v_proj, qkv_all, 2, self.s);
    } else {
        q_proj = try self.qmatmul(x, fa.q_w, fa.q_s, fa.q_b);
        k_proj = try self.qmatmul(x, fa.k_w, fa.k_s, fa.k_b);
        v_proj = try self.qmatmul(x, fa.v_w, fa.v_s, fa.v_b);
    }
    var q_r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_r);
    try mlx.check(mlx.mlx_reshape(&q_r, q_proj, &q_shape, 4, self.s));
    var k_r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_r);
    var v_r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v_r);
    try mlx.check(mlx.mlx_reshape(&k_r, k_proj, &kv_shape, 4, self.s));
    try mlx.check(mlx.mlx_reshape(&v_r, v_proj, &kv_shape, 4, self.s));
    const has_dual_hd = cfg.global_head_dim > 0 and cfg.global_head_dim != cfg.head_dim;
    const vnw_opt: ?mlx.mlx_array = if (!cfg.has_v_norm) null else if (has_dual_hd and is_global)
        (self.v_norm_weight_global orelse self.v_norm_weight.?)
    else
        self.v_norm_weight.?;

    // Decode at hd 256: the per-head norms, transposes and both RoPEs in
    // one dispatch (fusedQkNormRope256); else the composed chain.
    var qkv_fused: ?[3]mlx.mlx_array = null;
    defer if (qkv_fused) |f| {
        for (f) |a| if (a.ctx != null) {
            _ = mlx.mlx_array_free(a);
        };
    };
    if (batch == 1 and seq_len == 1 and cur_hd == 256 and vnw_opt != null and self.rms_eps_arr.ctx != null and qkNormRopeFusedEnabled()) blk: {
        const family: usize = if (is_global) 1 else 0;
        const angles = self.qkAngleRowsFor(family, rope_dims, rope_base, rope_freqs, offset, 1, 1.0, rope_scale) catch break :blk;
        qkv_fused = (fusedQkNormRope256(self.s, q_r, k_r, v_r, fa.q_norm, fa.k_norm, vnw_opt.?, angles, self.rms_eps_arr, h_count, @intCast(cur_kv_h), 1, rope_dims) catch null) orelse break :blk;
    }

    var q_rope = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_rope);
    var k_rope = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_rope);
    var v_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v_t);
    if (qkv_fused) |f| {
        _ = mlx.mlx_array_set(&q_rope, f[0]);
        _ = mlx.mlx_array_set(&k_rope, f[1]);
        _ = mlx.mlx_array_set(&v_t, f[2]);
    } else {
        const q_normed = try self.rmsNorm(q_r, fa.q_norm);
        defer _ = mlx.mlx_array_free(q_normed);
        var q_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_t);
        try mlx.check(mlx.mlx_transpose_axes(&q_t, q_normed, &perm, 4, self.s));
        try mlx.check(mlx.mlx_fast_rope(&q_rope, q_t, rope_dims, false, rope_base, rope_scale, offset, rope_freqs, self.s));

        // K norm
        const k_normed = try self.rmsNorm(k_r, fa.k_norm);
        defer _ = mlx.mlx_array_free(k_normed);

        // V norm (parameter-free RMS norm)
        var v_after_norm = v_r;
        var v_normed_arr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v_normed_arr);
        if (vnw_opt) |vnw| {
            v_normed_arr = try self.rmsNorm(v_r, vnw);
            v_after_norm = v_normed_arr;
        }

        // Transpose K, V to [B, H, S, D]
        var k_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_t);
        try mlx.check(mlx.mlx_transpose_axes(&k_t, k_normed, &perm, 4, self.s));
        try mlx.check(mlx.mlx_transpose_axes(&v_t, v_after_norm, &perm, 4, self.s));

        // RoPE on K
        try mlx.check(mlx.mlx_fast_rope(&k_rope, k_t, rope_dims, false, rope_base, rope_scale, offset, rope_freqs, self.s));
    }

    // Update KV cache: local layers read only the tail their queries
    // reach (`window + q_len - 1`), so a spec-verify block stops paying
    // full attention on every sliding layer. The masks below are built at
    // `sliding.kv_len` to match; when the fused hd-256 kernel owns the
    // band it declines the trim and `sliding.kv_len == total_kv`.
    const sliding = slidingViewFor(cfg, offset + seq_len, seq_len);
    const max_kv: u32 = if (is_global) 0 else sliding.span;
    var kv_view = try ctx.cache.update(layer, k_rope, v_t, self.s, max_kv);
    defer kv_view.deinit();
    const full_k = kv_view.k;
    const full_v = kv_view.v;

    // Scaled dot-product attention with sliding window masking
    var attn_out = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn_out);

    // Causal prefill arms first try the fused hd-256 flash kernel (null =
    // precondition miss -> composed fallback, identical to before).
    if (!cfg.has_sliding_window) {
        if (is_prefill) {
            if (try fusedSdpa256Prefill(self.s, q_rope, full_k, full_v, attn_scale, 0)) |fused| {
                _ = mlx.mlx_array_free(attn_out);
                attn_out = fused;
            } else {
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "causal", none_mask, .{ .ctx = null }, sdpaForceFused(q_rope, full_k), self.s));
            }
        } else {
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "", none_mask, .{ .ctx = null }, false, self.s));
        }
    } else {
        const sw: c_int = @intCast(cfg.sliding_window);
        const total_kv: c_int = offset + seq_len;
        if (is_prefill and (is_global or total_kv <= sw)) {
            // Global layers, or everything fits in the window (the sliding
            // mask degenerates to plain causal — same kernel and reduction
            // order the mlx-lm/mlx-vlm reference picks for short prompts,
            // which keeps near-tie MoE router decisions from flipping vs
            // the reference; DiffusionGemma parity).
            if (try fusedSdpa256Prefill(self.s, q_rope, full_k, full_v, attn_scale, 0)) |fused| {
                _ = mlx.mlx_array_free(attn_out);
                attn_out = fused;
            } else {
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "causal", none_mask, .{ .ctx = null }, sdpaForceFused(q_rope, full_k), self.s));
            }
        } else if (is_prefill) {
            // Sliding-window prefill: band mask runs in-kernel when fused.
            if (try fusedSdpa256Prefill(self.s, q_rope, full_k, full_v, attn_scale, sw)) |fused| {
                _ = mlx.mlx_array_free(attn_out);
                attn_out = fused;
            } else {
                if (local_prefill_mask.ctx == null) {
                    // Eager build was skipped for the fused path but this
                    // call declined — build once, cache for later layers.
                    local_prefill_mask.* = try self.createSlidingWindowMask(seq_len, sliding.kv_len, sw);
                }
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "array", local_prefill_mask.*, .{ .ctx = null }, false, self.s));
            }
        } else if (is_global) {
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "", none_mask, .{ .ctx = null }, false, self.s));
        } else if (@as(c_int, @intCast(ctx.cache.seqLen(layer))) <= sw) {
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "", none_mask, .{ .ctx = null }, false, self.s));
        } else {
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "array", local_decode_mask, .{ .ctx = null }, false, self.s));
        }
    }

    // Reshape: [B,H,S,D] → [B,S,H*D]
    const perm_back = [_]c_int{ 0, 2, 1, 3 };
    var attn_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn_t);
    try mlx.check(mlx.mlx_transpose_axes(&attn_t, attn_out, &perm_back, 4, self.s));
    var attn_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn_flat);
    try mlx.check(mlx.mlx_reshape(&attn_flat, attn_t, &flat_shape, 3, self.s));

    return self.qmatmul(attn_flat, fa.o_w, fa.o_s, fa.o_b);
}
