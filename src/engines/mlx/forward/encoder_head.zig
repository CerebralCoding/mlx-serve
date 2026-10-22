//! engines/mlx/forward/encoder_head, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns forwardEmbedding, forwardEmbeddingMasked, hasEmbedProjection, embedProjection, denseHead.

const mlx = @import("../../../mlx/ffi.zig");
const transposeBf16Weight = @import("../load/weights.zig").transposeBf16Weight;
const Transformer = @import("../transformer.zig").Transformer;

/// Forward pass that returns hidden states (after final_norm, before lm_head).
/// Output shape: [1, seq_len, hidden_size]. Caller must free.
pub fn forwardEmbedding(self: *Transformer, token_ids: mlx.mlx_array) !mlx.mlx_array {
    return self.forwardEmbeddingMasked(token_ids, null, null);
}

/// Embedding forward for a padded [B, T] batch: `key_pad_mask` (additive,
/// [B, 1, 1, T]) keeps padded positions out of encoder attention. Null
/// mask == plain forwardEmbedding. `vision` is one prompt's image/video
/// soft-token rows, spliced in at the placeholder ids (EmbeddingGemma 2).
pub fn forwardEmbeddingMasked(self: *Transformer, token_ids: mlx.mlx_array, key_pad_mask: ?mlx.mlx_array, vision: ?mlx.mlx_array) !mlx.mlx_array {
    self.embedding_mode = true;
    defer self.embedding_mode = false;
    var ctx = self.defaultCtx();
    ctx.key_pad_mask = key_pad_mask;
    ctx.vision_embeddings = vision;
    return self.forwardWith(&ctx, token_ids);
}

/// True when the checkpoint ships a sentence-transformers Dense head
/// (EmbeddingGemma dense.0/dense.1, EmbeddingGemma 2's embedding_projection
/// as the lone dense.0), applied between pool and normalize.
pub fn hasEmbedProjection(self: *const Transformer) bool {
    return self.dense0_w.ctx != null;
}

/// Sentence-transformers Dense head: pooled [B, H] → dense.0 (→ dense.1)
/// (identity activations, no layer bias — EmbeddingGemma's config).
/// Input is cast to bf16 for the quantized matmuls. Caller frees.
pub fn embedProjection(self: *const Transformer, pooled: mlx.mlx_array) !mlx.mlx_array {
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_astype(&x, pooled, .bfloat16, self.s));
    const mid = try self.denseHead(x, self.dense0_w, self.dense0_s, self.dense0_b);
    const head = if (self.dense1_w.ctx == null) mid else blk: {
        defer _ = mlx.mlx_array_free(mid);
        break :blk try self.denseHead(mid, self.dense1_w, self.dense1_s, self.dense1_b);
    };
    defer _ = mlx.mlx_array_free(head);
    // f32 out: the normalization squares and sums it, and a bf16 head output rounds that norm away from 1.
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, head, .float32, self.s));
    return out;
}

/// One Dense layer: quantized weights ride `qmatmul`; a dense checkpoint keeps its [out, in] layout
/// (nothing pre-transposes the head) and projects through a transposed view.
pub fn denseHead(self: *const Transformer, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array) !mlx.mlx_array {
    if (sc.ctx != null) return self.qmatmul(x, w, sc, bi);
    const wt = try transposeBf16Weight(w, self.s);
    defer _ = mlx.mlx_array_free(wt);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_matmul(&out, x, wt, self.s));
    return out;
}
