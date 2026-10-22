//! plugins/embedding_gemma2/forward, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns projectionOnlyPle, forwardEmbeddingGemma2With.

const mlx = @import("../../mlx/ffi.zig");
const ForwardCtx = @import("../../engines/mlx/forward_ctx.zig").ForwardCtx;
const scalarOf = @import("../../engines/mlx/load/weights.zig").scalarOf;
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;
const encoderBandMask = @import("../../sdk/blocks/attention/masks.zig").encoderBandMask;

/// EmbeddingGemma 2's per-layer inputs, `rms_norm(project(h) * hidden^-0.5)` as [B, S, n_layers, ple_dim]:
/// derived from the embeddings alone, unlike Gemma 4's (`computePLEInput`), which adds a per-token table.
pub fn projectionOnlyPle(self: *Transformer, h: mlx.mlx_array, batch: c_int, seq_len: c_int) !mlx.mlx_array {
    const cfg = &self.config;
    const proj_raw = try self.denseHead(h, self.ple_proj_w, self.ple_proj_s, self.ple_proj_b);
    defer _ = mlx.mlx_array_free(proj_raw);
    const scale = try scalarOf(1.0 / @sqrt(@as(f32, @floatFromInt(cfg.hidden_size))), mlx.mlx_array_dtype(proj_raw), self.s);
    defer _ = mlx.mlx_array_free(scale);
    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    try mlx.check(mlx.mlx_multiply(&scaled, proj_raw, scale, self.s));
    const shape = [_]c_int{ batch, seq_len, @intCast(cfg.num_hidden_layers), @intCast(cfg.hidden_size_per_layer_input) };
    var per_layer = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(per_layer);
    try mlx.check(mlx.mlx_reshape(&per_layer, scaled, &shape, 4, self.s));
    var normed = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(normed);
    try mlx.check(mlx.mlx_fast_rms_norm(&normed, per_layer, self.ple_proj_norm, cfg.rms_norm_eps, self.s));
    return normed;
}

/// EmbeddingGemma 2: the Gemma 4 text trunk trained as a bidirectional encoder — no cache, no causality.
/// Each layer is sandwich-normed attention (q/k/v norms, unit scale) and a GeGLU MLP, then a per-layer
/// input block and the layer scalar. Full-attention layers (head width 512, one KV head) attend over the
/// whole unpadded sequence, the rest within the INCLUSIVE band |i-j| <= sliding_window; `ctx.key_pad_mask`
/// folds into both. Returns the final-normed hidden [B, L, H]; the projection head runs after pooling
/// (a bias-free projection commutes with the mean).
pub fn forwardEmbeddingGemma2With(self: *Transformer, ctx: *ForwardCtx, token_ids: mlx.mlx_array) !mlx.mlx_array {
    const cfg = &self.config;
    const h_count: c_int = @intCast(cfg.num_attention_heads);

    var h = try self.embedding(token_ids);
    errdefer _ = mlx.mlx_array_free(h);
    // Soft tokens enter before the per-layer inputs are derived: those read the merged embeddings.
    h = try self.applyVisionEmbeddingsWith(ctx, h, token_ids);
    const x_shape = mlx.getShape(h);
    const batch: c_int = x_shape[0];
    const seq_len: c_int = x_shape[1];

    const ple = try self.projectionOnlyPle(h, batch, seq_len);
    defer _ = mlx.mlx_array_free(ple);

    const none_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none_mask);
    // `encoderBandMask` is exclusive (|i-j| < window): radius + 1 makes the band inclusive. A sequence
    // inside the band needs none.
    var band = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(band);
    if (seq_len > @as(c_int, @intCast(cfg.sliding_window)) + 1) {
        band = try encoderBandMask(self.allocator, @intCast(seq_len), cfg.sliding_window + 1, self.s);
    }
    var band_pad = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(band_pad);
    if (band.ctx != null) {
        if (ctx.key_pad_mask) |kp| {
            // A padded query whose band holds only padding would softmax over -inf, and its NaN row would
            // reach the real rows through V (0 x NaN) in every later layer: floor the sum at a finite value.
            var sum = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sum);
            try mlx.check(mlx.mlx_add(&sum, band, kp, self.s));
            const floor = try scalarOf(-1.0e9, mlx.mlx_array_dtype(sum), self.s);
            defer _ = mlx.mlx_array_free(floor);
            try mlx.check(mlx.mlx_maximum(&band_pad, sum, floor, self.s));
        }
    }
    const full_mask: mlx.mlx_array = ctx.key_pad_mask orelse none_mask;
    const sliding_mask: mlx.mlx_array = if (band_pad.ctx != null) band_pad else if (band.ctx != null) band else full_mask;

    const perm = [_]c_int{ 0, 2, 1, 3 };
    var dt = mlx.DtypeTrace.begin("embedding-gemma2", h, if (self.layers.len > 0) self.layers[0].q_w else null);

    for (0..cfg.num_hidden_layers) |layer_idx| {
        const li: u32 = @intCast(layer_idx);
        const lw = &self.layers[layer_idx];
        const is_global = cfg.isGlobalLayer(li);
        const hd: c_int = @intCast(cfg.layerHeadDim(li));
        const kv_h: c_int = @intCast(cfg.layerKVHeads(li));
        const q_shape = [_]c_int{ batch, seq_len, h_count, hd };
        const kv_shape = [_]c_int{ batch, seq_len, kv_h, hd };
        const out_shape = [_]c_int{ batch, seq_len, h_count * hd };
        const rope_base = mlx.mlx_optional_float{ .value = if (is_global) cfg.rope_theta else cfg.rope_local_base_freq, .has_value = true };
        const no_freqs = mlx.mlx_array{ .ctx = null };

        const normed = try self.rmsNorm(h, lw.input_norm);
        defer _ = mlx.mlx_array_free(normed);

        // Q, K, V: projection, per-head norm (V's carries no weight), transpose, RoPE at 0..L-1 on Q and K.
        const q = try self.qmatmul(normed, lw.q_w, lw.q_s, lw.q_b);
        defer _ = mlx.mlx_array_free(q);
        var q_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_r);
        try mlx.check(mlx.mlx_reshape(&q_r, q, &q_shape, 4, self.s));
        const q_normed = try self.rmsNorm(q_r, lw.q_norm.?);
        defer _ = mlx.mlx_array_free(q_normed);
        var q_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_t);
        try mlx.check(mlx.mlx_transpose_axes(&q_t, q_normed, &perm, 4, self.s));
        var q_rope = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_rope);
        try mlx.check(mlx.mlx_fast_rope(&q_rope, q_t, hd, false, rope_base, 1.0, 0, no_freqs, self.s));

        const k = try self.qmatmul(normed, lw.k_w, lw.k_s, lw.k_b);
        defer _ = mlx.mlx_array_free(k);
        var k_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_r);
        try mlx.check(mlx.mlx_reshape(&k_r, k, &kv_shape, 4, self.s));
        const k_normed = try self.rmsNorm(k_r, lw.k_norm.?);
        defer _ = mlx.mlx_array_free(k_normed);
        var k_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_t);
        try mlx.check(mlx.mlx_transpose_axes(&k_t, k_normed, &perm, 4, self.s));
        var k_rope = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_rope);
        try mlx.check(mlx.mlx_fast_rope(&k_rope, k_t, hd, false, rope_base, 1.0, 0, no_freqs, self.s));

        const v = try self.qmatmul(normed, lw.v_w, lw.v_s, lw.v_b);
        defer _ = mlx.mlx_array_free(v);
        var v_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v_r);
        try mlx.check(mlx.mlx_reshape(&v_r, v, &kv_shape, 4, self.s));
        const v_normed = try self.rmsNorm(v_r, if (is_global) self.v_norm_weight_global orelse self.v_norm_weight.? else self.v_norm_weight.?);
        defer _ = mlx.mlx_array_free(v_normed);
        var v_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v_t);
        try mlx.check(mlx.mlx_transpose_axes(&v_t, v_normed, &perm, 4, self.s));

        const mask = if (is_global) full_mask else sliding_mask;
        var attn_out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_out);
        if (mask.ctx != null) {
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, k_rope, v_t, cfg.attnScale(), "array", mask, .{ .ctx = null }, false, self.s));
        } else {
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, k_rope, v_t, cfg.attnScale(), "", none_mask, .{ .ctx = null }, false, self.s));
        }
        var attn_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_t);
        try mlx.check(mlx.mlx_transpose_axes(&attn_t, attn_out, &perm, 4, self.s));
        var attn_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_flat);
        try mlx.check(mlx.mlx_reshape(&attn_flat, attn_t, &out_shape, 3, self.s));
        const o_out = try self.qmatmul(attn_flat, lw.o_w, lw.o_s, lw.o_b);
        defer _ = mlx.mlx_array_free(o_out);

        const attn_normed = try self.rmsNorm(o_out, lw.post_attn_norm);
        defer _ = mlx.mlx_array_free(attn_normed);
        var h_attn = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&h_attn, h, attn_normed, self.s));
        _ = mlx.mlx_array_free(h);
        h = h_attn;

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
        var h_mlp = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&h_mlp, h, mlp_normed, self.s));
        _ = mlx.mlx_array_free(h);
        h = h_mlp;

        // Per-layer input block, then the layer scalar.
        const projected = try self.pleProject(h, lw, ple, li, batch, seq_len);
        defer _ = mlx.mlx_array_free(projected);
        const ple_normed = try self.rmsNorm(projected, lw.ple_norm.?);
        defer _ = mlx.mlx_array_free(ple_normed);
        var h_ple = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(h_ple);
        try mlx.check(mlx.mlx_add(&h_ple, h, ple_normed, self.s));
        _ = mlx.mlx_array_free(h);
        h = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&h, h_ple, lw.layer_scalar.?, self.s));
        dt.layer(h, layer_idx);
    }
    dt.end(h);

    const final = try self.rmsNorm(h, self.final_norm);
    _ = mlx.mlx_array_free(h);
    return final;
}
