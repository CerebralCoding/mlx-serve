//! Ling 3.0's load-time probe: the MoE router's base name (`mlp.gate`, or mlx-lm's quantized `mlp.gate.gate_proj`).

const Weights = @import("../../engines/mlx/model.zig").Weights;
const getWeightAtOpt = @import("../../engines/mlx/load/weights.zig").getWeightAtOpt;

/// Resolve the bailing_hybrid MoE router's weight base. The reference Ling-3.0
/// checkpoint carries a dense `mlp.gate.weight`; mlx-lm's converter quantizes
/// the router and keeps its module layout, so the same [E, hidden] tensor lands
/// at `mlp.gate.gate_proj.*` with its own scales and biases
/// (djrsystemservices/Ling-3.0-tiny-oQ8e, d1sl1ke/Ling-3.0-flash-oQ4e-mtp).
/// Same shape as lagunaRouterBase: prefer the flat name so reference
/// checkpoints bind byte-identically, and fall back to it when neither exists
/// so the MISSING WEIGHT error names the canonical key.
pub fn bailingRouterBaseAt(weights: *const Weights, buf: *[256]u8, base: []const u8) []const u8 {
    return if (getWeightAtOpt(weights, buf, base, "mlp.gate.weight") == null and
        getWeightAtOpt(weights, buf, base, "mlp.gate.gate_proj.weight") != null)
        "mlp.gate.gate_proj"
    else
        "mlp.gate";
}
