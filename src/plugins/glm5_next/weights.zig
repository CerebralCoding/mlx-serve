//! GLM-5.3-Flash's layer weight types: the sparse attention layer's (`DsaWeights`) and a sublayer's
//! hyper-connection (`MhcWeights`).

const mlx = @import("../../mlx/ffi.zig");
const QW = @import("../../engines/mlx/layer_weights.zig").QW;

/// GLM-5-Next DeepSeek sparse attention: absorbed NoPE MLA over a latent, plus the
/// pooled indexer that picks the tokens each query reads.
pub const DsaWeights = struct {
    q_a: QW,
    kv_a: QW,
    q_a_norm: mlx.mlx_array,
    kv_a_norm: mlx.mlx_array,
    q_b: QW,
    /// [H, latent, head_dim]: folds a query into the latent, or lifts the latent to keys.
    embed_q: QW,
    /// [H, head_dim, latent]: lifts the attended latent to values.
    unembed: QW,
    o: QW,
    idx_wq_b: QW,
    idx_wk: QW,
    idx_k_norm_w: mlx.mlx_array,
    idx_k_norm_b: mlx.mlx_array,
    idx_wproj: QW,
    /// f32 [hidden, index_head_dim] (pool gate projection, transposed).
    idx_gate_t: mlx.mlx_array,
    /// f32 [kpool, index_head_dim] (per-slot gate offset inside a pool).
    idx_ape: mlx.mlx_array,
};

/// One DeepSeek-V4 Sinkhorn hyper-connection (GLM-5-Next `attn_hc` / `ffn_hc`).
pub const MhcWeights = struct {
    /// f32 [(2+hc)*hc, hc*hidden]
    fn_rows: mlx.mlx_array,
    /// f32 [(2+hc)*hc]
    base: mlx.mlx_array,
    /// f32 [3]: pre / post / comb.
    scale: mlx.mlx_array,
};
