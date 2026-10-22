//! plugins/bert/forward, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns bertEmbedding, forwardBertWith.

const mlx = @import("../../mlx/ffi.zig");
const ForwardCtx = @import("../../engines/mlx/forward_ctx.zig").ForwardCtx;
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;

// ── BERT encoder-only forward pass ──

pub fn bertEmbedding(self: *const Transformer, token_ids: mlx.mlx_array) !mlx.mlx_array {
    const id_shape = mlx.getShape(token_ids);
    const batch = id_shape[0];
    const seq_len = id_shape[1];

    const flat_shape = [_]c_int{batch * seq_len};
    var flat_ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat_ids);
    try mlx.check(mlx.mlx_reshape(&flat_ids, token_ids, &flat_shape, 1, self.s));

    // Word embeddings
    const word_emb = try self.dequantTake(self.emb_w, self.emb_s, self.emb_b, flat_ids);
    defer _ = mlx.mlx_array_free(word_emb);

    // Reshape word embeddings to [B, S, H] up front: position / token-type
    // embeddings are computed once per POSITION ([1, S, H]) and broadcast
    // across the batch — adding them to the flat [B*S, H] form only
    // happened to work at batch == 1.
    const out_shape = [_]c_int{ batch, seq_len, @intCast(self.config.hidden_size) };
    var word_3d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(word_3d);
    try mlx.check(mlx.mlx_reshape(&word_3d, word_emb, &out_shape, 3, self.s));

    // Position IDs: [0, 1, 2, ..., seq_len-1]
    var pos_ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pos_ids);
    try mlx.check(mlx.mlx_arange(&pos_ids, 0, @as(f64, @floatFromInt(seq_len)), 1, .int32, self.s));

    const pos_emb = try self.dequantTake(self.bert_pos_w, self.bert_pos_s, self.bert_pos_b, pos_ids);
    defer _ = mlx.mlx_array_free(pos_emb);

    // Token type IDs: all zeros (one row per position, broadcast over B)
    const seq_shape = [_]c_int{seq_len};
    var toktype_ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(toktype_ids);
    try mlx.check(mlx.mlx_zeros(&toktype_ids, &seq_shape, 1, .int32, self.s));

    const toktype_emb = try self.dequantTake(self.bert_toktype_w, self.bert_toktype_s, self.bert_toktype_b, toktype_ids);
    defer _ = mlx.mlx_array_free(toktype_emb);

    const bcast_shape = [_]c_int{ 1, seq_len, @intCast(self.config.hidden_size) };
    var pos_3d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pos_3d);
    try mlx.check(mlx.mlx_reshape(&pos_3d, pos_emb, &bcast_shape, 3, self.s));
    var toktype_3d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(toktype_3d);
    try mlx.check(mlx.mlx_reshape(&toktype_3d, toktype_emb, &bcast_shape, 3, self.s));

    // Sum: word + position + token_type → [B, S, H]
    var wp = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wp);
    try mlx.check(mlx.mlx_add(&wp, word_3d, pos_3d, self.s));
    var sum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sum);
    try mlx.check(mlx.mlx_add(&sum, wp, toktype_3d, self.s));

    // LayerNorm
    return self.layerNorm(sum, self.bert_emb_norm_w, self.bert_emb_norm_b);
}

pub fn forwardBertWith(self: *Transformer, ctx: *ForwardCtx, token_ids: mlx.mlx_array) !mlx.mlx_array {
    // BERT is encoder-only: no per-request KV state; ctx only carries the
    // optional key-padding mask for padded [B, T] embedding batches.
    const pad_mask = ctx.key_pad_mask;
    const bert_layers = self.bert_layers.?;
    const h_count = self.config.num_attention_heads;
    const head_dim = self.config.head_dim;
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    const id_shape = mlx.getShape(token_ids);
    const batch = id_shape[0];
    const seq_len = id_shape[1];

    var h = try self.bertEmbedding(token_ids);
    var dt = mlx.DtypeTrace.begin("bert", h, if (bert_layers.len > 0) bert_layers[0].q_w else null);

    for (bert_layers, 0..) |lw, layer_idx| {
        // Self-attention
        const q = try self.qmatmulAddBias(h, lw.q_w, lw.q_s, lw.q_b, lw.q_bias);
        defer _ = mlx.mlx_array_free(q);
        const k = try self.qmatmulAddBias(h, lw.k_w, lw.k_s, lw.k_b, lw.k_bias);
        defer _ = mlx.mlx_array_free(k);
        const v = try self.qmatmulAddBias(h, lw.v_w, lw.v_s, lw.v_b, lw.v_bias);
        defer _ = mlx.mlx_array_free(v);

        // Reshape [B, S, H] -> [B, S, heads, head_dim] -> [B, heads, S, head_dim]
        const qkv_shape = [_]c_int{ batch, seq_len, @intCast(h_count), @intCast(head_dim) };
        var q_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_r);
        try mlx.check(mlx.mlx_reshape(&q_r, q, &qkv_shape, 4, self.s));
        var k_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_r);
        try mlx.check(mlx.mlx_reshape(&k_r, k, &qkv_shape, 4, self.s));
        var v_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v_r);
        try mlx.check(mlx.mlx_reshape(&v_r, v, &qkv_shape, 4, self.s));

        const perm = [_]c_int{ 0, 2, 1, 3 };
        var q_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_t);
        try mlx.check(mlx.mlx_transpose_axes(&q_t, q_r, &perm, 4, self.s));
        var k_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_t);
        try mlx.check(mlx.mlx_transpose_axes(&k_t, k_r, &perm, 4, self.s));
        var v_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v_t);
        try mlx.check(mlx.mlx_transpose_axes(&v_t, v_r, &perm, 4, self.s));

        // Bidirectional attention (no causal mask); padded batches mask
        // their pad KEYS so real positions never attend into padding.
        var attn = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn);
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(
            &attn,
            q_t,
            k_t,
            v_t,
            scale,
            if (pad_mask != null) "array" else "",
            if (pad_mask) |m| m else mlx.mlx_array_new(),
            mlx.mlx_array_new(),
            false,
            self.s,
        ));

        // Transpose back [B, heads, S, head_dim] -> [B, S, heads, head_dim] -> [B, S, H]
        var attn_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_t);
        try mlx.check(mlx.mlx_transpose_axes(&attn_t, attn, &perm, 4, self.s));
        const flat_shape = [_]c_int{ batch, seq_len, @intCast(self.config.hidden_size) };
        var attn_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_flat);
        try mlx.check(mlx.mlx_reshape(&attn_flat, attn_t, &flat_shape, 3, self.s));

        // Output projection
        const o = try self.qmatmulAddBias(attn_flat, lw.o_w, lw.o_s, lw.o_b, lw.o_bias);
        defer _ = mlx.mlx_array_free(o);

        // Residual + LayerNorm
        var h_plus_attn = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(h_plus_attn);
        try mlx.check(mlx.mlx_add(&h_plus_attn, h, o, self.s));
        _ = mlx.mlx_array_free(h);

        h = try self.layerNorm(h_plus_attn, lw.attn_norm_w, lw.attn_norm_b);

        // MLP: intermediate (GELU) -> output
        const inter = try self.qmatmulAddBias(h, lw.inter_w, lw.inter_s, lw.inter_b, lw.inter_bias);
        defer _ = mlx.mlx_array_free(inter);
        const activated = try self.gelu(inter);
        defer _ = mlx.mlx_array_free(activated);
        const out = try self.qmatmulAddBias(activated, lw.out_w, lw.out_s, lw.out_b, lw.out_bias);
        defer _ = mlx.mlx_array_free(out);

        // Residual + LayerNorm
        var h_plus_out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(h_plus_out);
        try mlx.check(mlx.mlx_add(&h_plus_out, h, out, self.s));
        _ = mlx.mlx_array_free(h);

        h = try self.layerNorm(h_plus_out, lw.out_norm_w, lw.out_norm_b);
        dt.layer(h, layer_idx);
    }
    dt.end(h);

    return h; // [B, S, H] — hidden states for mean pooling
}
