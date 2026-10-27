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
    bytes += model.activationBytes(rows);
    // Chunking avoids a single full logits allocation; autograd still retains per-chunk intermediates.
    bytes += @as(u128, rows) * @as(u128, @intCast(ops.shape(model.head.head.w)[0])) * 16;
    bytes += 64 << 20;
    return std.math.cast(u64, bytes) orelse error.TrainingMemoryOverflow;
}
