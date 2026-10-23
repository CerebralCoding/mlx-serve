//! The router base name probe the legacy MoE load reads (MiMo's arm; laguna's own load is `family.zig`).

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const testing = std.testing;
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

// ── tests ──

test "lagunaRouterBase resolves both the flat mlx-lm gate and the HF-native gate.proj naming" {
    // The poolside NVFP4-mlx build flattens the router module to
    // `mlp.gate.weight`; mlx-community's oQ conversions (Laguna-S-2.1-oQ4e-fast,
    // -oQ5e) keep the HF module layout, so the SAME [E, hidden] tensor lands at
    // `mlp.gate.proj.weight` and the load died with MISSING WEIGHT
    // (issue #169). `mlp.gate.e_score_correction_bias` is a buffer on the gate
    // module in both, so it never moves.
    const allocator = testing.allocator;
    var buf: [256]u8 = undefined;
    const put = struct {
        fn add(w: *Weights, alloc: std.mem.Allocator, key: []const u8) !void {
            const k = try alloc.dupe(u8, key);
            try w.map.put(k, mlx.mlx_array_new());
        }
    }.add;

    // oQ build: only the nested projection exists (was the crash).
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "language_model.model.layers.1.mlp.gate.proj.weight");
        try testing.expectEqualStrings(
            "mlp.gate.proj",
            lagunaRouterBase(&w, &buf, "language_model.model", 1),
        );
    }
    // poolside build: the flat name is preferred, so it binds byte-identically.
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "model.layers.1.mlp.gate.weight");
        try testing.expectEqualStrings("mlp.gate", lagunaRouterBase(&w, &buf, "model", 1));
    }
    // Neither present: keep the flat name so the error names the canonical key.
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try testing.expectEqualStrings("mlp.gate", lagunaRouterBase(&w, &buf, "model", 1));
    }
}
