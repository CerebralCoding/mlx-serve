//! plugins/laguna/load, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns lagunaRouterBase.

const Weights = @import("../../engines/mlx/model.zig").Weights;
const getLayerWeightOpt = @import("../../engines/mlx/load/weights.zig").getLayerWeightOpt;

/// Resolve the laguna MoE router's weight base. poolside's NVFP4-mlx build
/// flattens the router module to `mlp.gate.*`; mlx-community's oQ conversions
/// keep the HF module layout, where the same [E, hidden] tensor sits at
/// `mlp.gate.proj.*` (issue #169). Prefers the flat name so the poolside
/// checkpoint binds byte-identically, and falls back to it when neither exists
/// so the MISSING WEIGHT error names the canonical key.
pub fn lagunaRouterBase(weights: *const Weights, buf: *[256]u8, prefix: []const u8, layer: u32) []const u8 {
    return if (getLayerWeightOpt(weights, buf, prefix, layer, "mlp.gate.weight") == null and
        getLayerWeightOpt(weights, buf, prefix, layer, "mlp.gate.proj.weight") != null)
        "mlp.gate.proj"
    else
        "mlp.gate";
}
