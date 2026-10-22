//! plugins/diffusion_gemma/weights, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns SelfCondWeights.

const mlx = @import("../../mlx/ffi.zig");

/// DiffusionGemma self-conditioning module: a GeGLU FFN over the previous
/// denoising step's soft embeddings, added to the canvas token embeddings
/// and re-normalized (scale-free) before decoder layer 0.
///   out = rms_norm_no_scale(embeds + down(geglu(gate(pre_norm(sig)), up(pre_norm(sig)))))
pub const SelfCondWeights = struct {
    pre_norm: mlx.mlx_array,
    gate_w: mlx.mlx_array,
    gate_s: mlx.mlx_array,
    gate_b: mlx.mlx_array,
    up_w: mlx.mlx_array,
    up_s: mlx.mlx_array,
    up_b: mlx.mlx_array,
    down_w: mlx.mlx_array,
    down_s: mlx.mlx_array,
    down_b: mlx.mlx_array,
};
