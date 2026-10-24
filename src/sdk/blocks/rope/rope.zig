//! sdk/blocks/rope/rope, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns ropeAtPositions, ropeAtFreqs, computeLlama3Freqs.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const model_mod = @import("../../../engines/mlx/model.zig");
const scalarOf = @import("../../../mlx/array.zig").scalarOf;
const Transformer = @import("../../../engines/mlx/transformer.zig").Transformer;
const MropeCosSin = @import("mrope.zig").MropeCosSin;

/// HF `_compute_llama3_parameters` as mlx_fast_rope denominators (angle =
/// position / freqs[i]): low frequencies slow down by `factor`, high ones keep
/// their rate, the band between is smoothed. `out.len` = head_dim / 2.
pub fn computeLlama3Freqs(out: []f64, head_dim: u32, base: f64, l3: model_mod.Llama3Rope) void {
    const factor: f64 = l3.factor;
    const low: f64 = l3.low_freq_factor;
    const high: f64 = l3.high_freq_factor;
    const window: f64 = l3.original_max_position_embeddings;
    for (out, 0..) |*o, i| {
        const period = std.math.pow(f64, base, @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(head_dim)));
        const wavelen = 2.0 * std.math.pi * period;
        o.* = if (wavelen < window / high) period else if (wavelen > window / low) period * factor else blk: {
            const smooth = (window / wavelen - low) / (high - low);
            break :blk period / ((1.0 - smooth) / factor + smooth);
        };
    }
}

// ── tests ──

/// Partial neox RoPE of `x [B,1,N,hd]` at positions `base + step*i` with explicit cos/sin.
pub fn ropeAtPositions(self: *Transformer, x: mlx.mlx_array, rope_dims: c_int, theta: f32, base: f32, step: f32, n: c_int) !mlx.mlx_array {
    return self.ropeAtFreqs(x, rope_dims, try self.ropeInvFreq(rope_dims, theta), base, step, n); // takes ownership
}

/// The shared tail of `ropeAtPositions`: rotate `x` at positions
/// `base + step*i` using the given inv_freq array (`[half]`, f32).
pub fn ropeAtFreqs(self: *Transformer, x: mlx.mlx_array, rope_dims: c_int, inv_freq: mlx.mlx_array, base: f32, step: f32, n: c_int) !mlx.mlx_array {
    const cs = try self.ropeCosSinFromFreqs(rope_dims, inv_freq, base, step, n, mlx.mlx_array_dtype(x));
    defer _ = mlx.mlx_array_free(cs.cos);
    defer _ = mlx.mlx_array_free(cs.sin);
    return self.ropeApplyCosSin(x, rope_dims, cs.cos, cs.sin);
}
