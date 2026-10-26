//! Incremental training reservation; the loaded frozen weights are billed separately.
const std = @import("std");
const Frozen = @import("model.zig").Frozen;
const Target = @import("adapter.zig").Target;
const ops = @import("../../mlx/ops.zig");

pub fn reserve(model: Frozen, targets: []const Target, rank: u32, rows: u32) !u64 {
    if (rank == 0 or rows == 0 or rows > 512) return error.InvalidTrainingShape;
    var parameters: u128 = 0;
    for (targets) |target| parameters += @as(u128, rank) * (@as(u128, target.input) + target.output);
    // Masters, moments, accumulated and current gradients, proposals, checkpoint and bf16 compute copies.
    var bytes: u128 = parameters * 64;
    for (model.base.weights.layers) |layer| {
        const hidden: u128 = @intCast(model.base.weights.embed.hidden);
        const intermediate: u128 = @intCast(if (layer.mlp.gate.s.ctx != null) ops.shape(layer.mlp.gate.w)[0] else ops.shape(layer.mlp.gate.w)[1]);
        const heads: u128 = @intCast(layer.mixer.geo.heads);
        const width: u128 = @intCast(layer.mixer.geo.head_dim);
        const kv: u128 = @intCast(layer.mixer.geo.kv_heads);
        bytes += @as(u128, rows) * (hidden * 16 + intermediate * 6 + (heads + kv) * width * 4) * 2;
        // Metal's SDPA VJP expands to the native-op fallback and retains dense attention intermediates.
        bytes += heads * rows * rows * 8;
    }
    // Chunking avoids a single full logits allocation; autograd still retains per-chunk intermediates.
    bytes += @as(u128, rows) * @as(u128, @intCast(ops.shape(model.base.weights.head.w)[0])) * 16;
    bytes += 64 << 20;
    return std.math.cast(u64, bytes) orelse error.TrainingMemoryOverflow;
}
