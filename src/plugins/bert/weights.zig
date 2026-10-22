//! plugins/bert/weights, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns BertLayerWeights.

const mlx = @import("../../mlx/ffi.zig");

// ── Standard model per-layer weights ──

// ── BERT encoder-only layer weights ──

pub const BertLayerWeights = struct {
    // Self-attention (separate Q/K/V projections with real bias)
    q_w: mlx.mlx_array,
    q_s: mlx.mlx_array,
    q_b: mlx.mlx_array,
    q_bias: mlx.mlx_array,
    k_w: mlx.mlx_array,
    k_s: mlx.mlx_array,
    k_b: mlx.mlx_array,
    k_bias: mlx.mlx_array,
    v_w: mlx.mlx_array,
    v_s: mlx.mlx_array,
    v_b: mlx.mlx_array,
    v_bias: mlx.mlx_array,
    o_w: mlx.mlx_array,
    o_s: mlx.mlx_array,
    o_b: mlx.mlx_array,
    o_bias: mlx.mlx_array,
    attn_norm_w: mlx.mlx_array,
    attn_norm_b: mlx.mlx_array,
    // MLP: intermediate -> GELU -> output
    inter_w: mlx.mlx_array,
    inter_s: mlx.mlx_array,
    inter_b: mlx.mlx_array,
    inter_bias: mlx.mlx_array,
    out_w: mlx.mlx_array,
    out_s: mlx.mlx_array,
    out_b: mlx.mlx_array,
    out_bias: mlx.mlx_array,
    out_norm_w: mlx.mlx_array,
    out_norm_b: mlx.mlx_array,
};
