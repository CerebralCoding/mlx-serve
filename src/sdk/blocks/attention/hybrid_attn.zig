//! sdk/blocks/attention/hybrid_attn, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns hybridAttnWith.

const mlx = @import("../../../mlx/ffi.zig");
const rowqmv = @import("../qmm/rowqmv.zig");
const ForwardCtx = @import("../../../engines/mlx/forward_ctx.zig").ForwardCtx;
const FullAttnWeights = @import("../../../engines/mlx/layer_weights.zig").FullAttnWeights;
const Transformer = @import("../../../engines/mlx/transformer.zig").Transformer;
const FUSED_ROWS_MAX_M = @import("../linear/fused_rows.zig").FUSED_ROWS_MAX_M;
const fusedSdpa256Prefill = @import("attn256.zig").fusedSdpa256Prefill;
const rowByRowSdpa = @import("sdpa.zig").rowByRowSdpa;
const sdpaForceFused = @import("sdpa.zig").sdpaForceFused;

// ── Hybrid attention (LFM2, Nemotron-H) ──

pub fn hybridAttnWith(
    self: *Transformer,
    ctx: *ForwardCtx,
    x: mlx.mlx_array,
    fa: *const FullAttnWeights,
    layer_idx: u32,
    offset: c_int,
    batch: c_int,
    seq_len: c_int,
    is_prefill: bool,
) !mlx.mlx_array {
    const cfg = &self.config;
    const n_heads: c_int = @intCast(cfg.num_attention_heads);
    const n_kv_heads: c_int = @intCast(cfg.num_key_value_heads);
    const hd: c_int = @intCast(cfg.head_dim);
    const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));

    // Q/K/V projections: one row-fused matmul at decode width.
    var q_raw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_raw);
    var k_raw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_raw);
    var v_raw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v_raw);
    if (fa.qkv.w.ctx != null and batch * seq_len <= FUSED_ROWS_MAX_M) {
        const qkv = try self.qmatmul(x, fa.qkv.w, fa.qkv.s, fa.qkv.b);
        defer _ = mlx.mlx_array_free(qkv);
        try fa.qkv.part(&q_raw, qkv, 0, self.s);
        try fa.qkv.part(&k_raw, qkv, 1, self.s);
        try fa.qkv.part(&v_raw, qkv, 2, self.s);
    } else {
        q_raw = try self.qmatmul(x, fa.q_w, fa.q_s, fa.q_b);
        k_raw = try self.qmatmul(x, fa.k_w, fa.k_s, fa.k_b);
        v_raw = try self.qmatmul(x, fa.v_w, fa.v_s, fa.v_b);
    }

    // Reshape to heads: [B, S, n*hd] → [B, S, n, hd]
    const q_shape = [_]c_int{ batch, seq_len, n_heads, hd };
    const kv_shape = [_]c_int{ batch, seq_len, n_kv_heads, hd };
    var q = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q);
    var k = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k);
    var v = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v);
    try mlx.check(mlx.mlx_reshape(&q, q_raw, &q_shape, 4, self.s));
    try mlx.check(mlx.mlx_reshape(&k, k_raw, &kv_shape, 4, self.s));
    try mlx.check(mlx.mlx_reshape(&v, v_raw, &kv_shape, 4, self.s));

    // QK LayerNorm (LFM2 has it, Nemotron-H does not — checked by weight presence)
    if (cfg.has_qk_norm) {
        var q_normed = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_fast_rms_norm(&q_normed, q, fa.q_norm, cfg.rms_norm_eps, self.s));
        _ = mlx.mlx_array_free(q);
        q = q_normed;
        var k_normed = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_fast_rms_norm(&k_normed, k, fa.k_norm, cfg.rms_norm_eps, self.s));
        _ = mlx.mlx_array_free(k);
        k = k_normed;
    }

    // Transpose to [B, n, S, hd] for attention and RoPE
    const perm = [_]c_int{ 0, 2, 1, 3 };
    var q_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_t);
    var k_t = mlx.mlx_array_new();
    var v_t = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&q_t, q, &perm, 4, self.s));
    try mlx.check(mlx.mlx_transpose_axes(&k_t, k, &perm, 4, self.s));
    try mlx.check(mlx.mlx_transpose_axes(&v_t, v, &perm, 4, self.s));

    // RoPE (applied after transpose to [B, n, S, hd])
    const rope_base = mlx.mlx_optional_float.some(cfg.rope_theta);
    const no_freqs = mlx.mlx_array{ .ctx = null };
    if (!cfg.layerSkipsRope(layer_idx)) {
        try mlx.check(mlx.mlx_fast_rope(&q_t, q_t, hd, false, rope_base, cfg.rope_scaling_factor, offset, no_freqs, self.s));
        try mlx.check(mlx.mlx_fast_rope(&k_t, k_t, hd, false, rope_base, cfg.rope_scaling_factor, offset, no_freqs, self.s));
    }

    // KV cache: update and get full K/V (DenseKVView owns its arrays only
    // in quant mode; in dense mode it aliases the cache view, so the defer
    // below is a no-op there).
    var kv_view = try ctx.cache.update(layer_idx, k_t, v_t, self.s, cfg.max_position_embeddings);
    defer kv_view.deinit();
    _ = mlx.mlx_array_free(k_t);
    _ = mlx.mlx_array_free(v_t);
    k_t = kv_view.k;
    v_t = kv_view.v;

    // Scaled dot-product attention (causal)
    const none_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none_mask);

    var attn_out = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn_out);
    if (cfg.rowExactDecode() and seq_len > 1 and seq_len <= rowqmv.MAX_ROWS and batch == 1) {
        // A verify window attends row by row, each row the one-query call
        // serial decoding makes over its own keys: the multi-query kernel
        // sums in another order.
        try rowByRowSdpa(&attn_out, q_t, k_t, v_t, attn_scale, seq_len, self.s);
    } else if (is_prefill) {
        if (try fusedSdpa256Prefill(self.s, q_t, k_t, v_t, attn_scale, 0)) |fused| {
            _ = mlx.mlx_array_free(attn_out);
            attn_out = fused;
        } else {
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_t, k_t, v_t, attn_scale, "causal", none_mask, .{ .ctx = null }, sdpaForceFused(q_t, k_t), self.s));
        }
    } else {
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_t, k_t, v_t, attn_scale, "", none_mask, .{ .ctx = null }, false, self.s));
    }

    // Transpose back: [B, n, S, hd] → [B, S, n, hd] → [B, S, n*hd]
    var attn_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn_t);
    try mlx.check(mlx.mlx_transpose_axes(&attn_t, attn_out, &perm, 4, self.s));
    const flat_shape = [_]c_int{ batch, seq_len, n_heads * hd };
    var attn_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn_flat);
    try mlx.check(mlx.mlx_reshape(&attn_flat, attn_t, &flat_shape, 3, self.s));

    return self.qmatmul(attn_flat, fa.o_w, fa.o_s, fa.o_b);
}
