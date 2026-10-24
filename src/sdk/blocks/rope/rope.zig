//! sdk/blocks/rope/rope, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns computeLlama3Freqs.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const scalarOf = @import("../../../mlx/array.zig").scalarOf;
const MropeCosSin = @import("mrope.zig").MropeCosSin;

/// HF `_compute_llama3_parameters` as mlx_fast_rope denominators (angle =
/// position / freqs[i]): low frequencies slow down by `factor`, high ones keep
/// their rate, the band between is smoothed. `out.len` = head_dim / 2.
/// `l3` is a `Llama3Rope` or any struct with its fields (f32 or f64).
pub fn computeLlama3Freqs(out: []f64, head_dim: u32, base: f64, l3: anytype) void {
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

/// Llama 3.x `rope_type: "llama3"` (HF `_compute_llama3_parameters`).
pub const Llama3Rope = struct {
    factor: f32,
    low_freq_factor: f32 = 1.0,
    high_freq_factor: f32 = 4.0,
    original_max_position_embeddings: f32 = 8192.0,
};
