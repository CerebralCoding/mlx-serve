//! plugins/gemma3/encoder, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns forwardGemma3EncoderWith.

const mlx = @import("../../mlx/ffi.zig");
const ForwardCtx = @import("../../engines/mlx/forward_ctx.zig").ForwardCtx;
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;
const encoderBandMask = @import("../../sdk/blocks/attention/masks.zig").encoderBandMask;

/// Bidirectional batched encoder forward for embedding models built on a
/// decoder arch (EmbeddingGemma: gemma3_text + use_bidirectional_attention).
/// Self-contained, mirroring the BERT-arm precedent: no KV cache, no
/// causality — full-attention layers attend over the whole (unpadded)
/// sequence, sliding layers within a symmetric |i-j| < window band;
/// `ctx.key_pad_mask` folds into both. Returns final-normed hidden
/// [B, L, H] (this path only ever serves embeddings).
pub fn forwardGemma3EncoderWith(self: *Transformer, ctx: *ForwardCtx, token_ids: mlx.mlx_array) !mlx.mlx_array {
    const cfg = &self.config;
    const h_count = cfg.num_attention_heads;
    const kv_h = cfg.num_key_value_heads;
    const hd = cfg.head_dim;
    const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.query_pre_attn_scalar)));

    var h = try self.embedding(token_ids);
    errdefer _ = mlx.mlx_array_free(h);
    const x_shape = mlx.getShape(h);
    const batch: c_int = x_shape[0];
    const seq_len: c_int = x_shape[1];

    const q_shape = [_]c_int{ batch, seq_len, @intCast(h_count), @intCast(hd) };
    const kv_shape = [_]c_int{ batch, seq_len, @intCast(kv_h), @intCast(hd) };
    const out_shape = [_]c_int{ batch, seq_len, @intCast(h_count * hd) };
    const perm = [_]c_int{ 0, 2, 1, 3 };

    const none_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none_mask);

    // Additive masks, built once for the whole stack. Sequences within
    // the window need no band (equivalent to full attention).
    var band = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(band);
    if (cfg.has_sliding_window and seq_len > @as(c_int, @intCast(cfg.sliding_window))) {
        band = try encoderBandMask(self.allocator, @intCast(seq_len), cfg.sliding_window, self.s);
    }
    var band_pad = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(band_pad);
    if (band.ctx != null) {
        if (ctx.key_pad_mask) |kp| try mlx.check(mlx.mlx_add(&band_pad, band, kp, self.s));
    }
    const full_mask: mlx.mlx_array = ctx.key_pad_mask orelse none_mask;
    const sliding_mask: mlx.mlx_array = if (band_pad.ctx != null) band_pad else if (band.ctx != null) band else full_mask;

    var dt = mlx.DtypeTrace.begin("gemma3-encoder", h, if (self.layers.len > 0) self.layers[0].q_w else null);

    for (0..cfg.num_hidden_layers) |layer_idx| {
        const li: u32 = @intCast(layer_idx);
        const lw = &self.layers[layer_idx];
        // No cache exists here to share KV through; gemma3 never sets it.
        if (lw.kv_source != null) return error.UnsupportedEncoderArch;
        if (!cfg.has_pre_ff_norm) return error.UnsupportedEncoderArch;
        const is_global = cfg.isGlobalLayer(li);

        const normed = try self.rmsNorm(h, lw.input_norm);
        defer _ = mlx.mlx_array_free(normed);

        // Q — projection, per-head norm, transpose, RoPE at 0..L-1.
        const q = try self.qmatmulMaybeBias(normed, lw.q_w, lw.q_s, lw.q_b, lw.q_bias);
        defer _ = mlx.mlx_array_free(q);
        var q_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_r);
        try mlx.check(mlx.mlx_reshape(&q_r, q, &q_shape, 4, self.s));
        const q_normed: ?mlx.mlx_array = if (lw.q_norm) |qn| try self.rmsNorm(q_r, qn) else null;
        defer if (q_normed) |qn| {
            _ = mlx.mlx_array_free(qn);
        };
        var q_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_t);
        try mlx.check(mlx.mlx_transpose_axes(&q_t, q_normed orelse q_r, &perm, 4, self.s));

        // Same per-layer-type theta selection as the decoder forward.
        const use_prop_rope = is_global and self.rope_freqs_global != null;
        const rope_base_opt = mlx.mlx_optional_float{
            .value = if (is_global) cfg.rope_theta else cfg.rope_local_base_freq,
            .has_value = !use_prop_rope,
        };
        const rope_scale: f32 = if (use_prop_rope) 1.0 else if (is_global) (1.0 / cfg.rope_scaling_factor) else 1.0;
        const rope_freqs: mlx.mlx_array = if (use_prop_rope) self.rope_freqs_global.? else .{ .ctx = null };
        var q_rope = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_rope);
        try mlx.check(mlx.mlx_fast_rope(&q_rope, q_t, @intCast(hd), false, rope_base_opt, rope_scale, 0, rope_freqs, self.s));

        // K, V
        const k = try self.qmatmulMaybeBias(normed, lw.k_w, lw.k_s, lw.k_b, lw.k_bias);
        defer _ = mlx.mlx_array_free(k);
        const v = if (lw.k_eq_v) k else try self.qmatmulMaybeBias(normed, lw.v_w, lw.v_s, lw.v_b, lw.v_bias);
        defer if (!lw.k_eq_v) {
            _ = mlx.mlx_array_free(v);
        };
        var k_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_r);
        var v_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v_r);
        try mlx.check(mlx.mlx_reshape(&k_r, k, &kv_shape, 4, self.s));
        try mlx.check(mlx.mlx_reshape(&v_r, v, &kv_shape, 4, self.s));
        const k_normed: ?mlx.mlx_array = if (lw.k_norm) |kn| try self.rmsNorm(k_r, kn) else null;
        defer if (k_normed) |kn| {
            _ = mlx.mlx_array_free(kn);
        };
        var k_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_t);
        var v_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v_t);
        try mlx.check(mlx.mlx_transpose_axes(&k_t, k_normed orelse k_r, &perm, 4, self.s));
        try mlx.check(mlx.mlx_transpose_axes(&v_t, v_r, &perm, 4, self.s));
        var k_rope = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_rope);
        try mlx.check(mlx.mlx_fast_rope(&k_rope, k_t, @intCast(hd), false, rope_base_opt, rope_scale, 0, rope_freqs, self.s));

        // Bidirectional SDPA (mlx composes internally for head_dim 256).
        const mask = if (is_global) full_mask else sliding_mask;
        var attn_out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_out);
        if (mask.ctx != null) {
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, k_rope, v_t, attn_scale, "array", mask, .{ .ctx = null }, false, self.s));
        } else {
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, k_rope, v_t, attn_scale, "", none_mask, .{ .ctx = null }, false, self.s));
        }

        var attn_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_t);
        try mlx.check(mlx.mlx_transpose_axes(&attn_t, attn_out, &perm, 4, self.s));
        var attn_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_flat);
        try mlx.check(mlx.mlx_reshape(&attn_flat, attn_t, &out_shape, 3, self.s));
        const o_out = try self.qmatmul(attn_flat, lw.o_w, lw.o_s, lw.o_b);
        defer _ = mlx.mlx_array_free(o_out);

        // Gemma sandwich norms + GeGLU MLP, exactly as the decoder.
        const attn_normed = try self.rmsNorm(o_out, lw.post_attn_norm);
        defer _ = mlx.mlx_array_free(attn_normed);
        var h_new = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&h_new, h, attn_normed, self.s));
        _ = mlx.mlx_array_free(h);
        h = h_new;

        const ff_normed = try self.rmsNorm(h, lw.pre_ff_norm.?);
        defer _ = mlx.mlx_array_free(ff_normed);
        const gate_raw = try self.qmatmul(ff_normed, lw.gate_w, lw.gate_s, lw.gate_b);
        defer _ = mlx.mlx_array_free(gate_raw);
        const up = try self.qmatmul(ff_normed, lw.up_w, lw.up_s, lw.up_b);
        defer _ = mlx.mlx_array_free(up);
        const gate_up = try self.computeGeglu(gate_raw, up);
        defer _ = mlx.mlx_array_free(gate_up);
        const down = try self.qmatmul(gate_up, lw.down_w, lw.down_s, lw.down_b);
        defer _ = mlx.mlx_array_free(down);
        const mlp_normed = try self.rmsNorm(down, lw.post_ff_norm.?);
        defer _ = mlx.mlx_array_free(mlp_normed);
        var h_next = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&h_next, h, mlp_normed, self.s));
        _ = mlx.mlx_array_free(h);
        h = h_next;
        dt.layer(h, layer_idx);
    }
    dt.end(h);

    const final = try self.rmsNorm(h, self.final_norm);
    _ = mlx.mlx_array_free(h);
    return final;
}
