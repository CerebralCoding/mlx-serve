//! Stable per-token distribution drift from the frozen base.
const std = @import("std");
const ops = @import("../../mlx/ops.zig");
const mlx = @import("../../mlx/ffi.zig");

/// KL(base || adapted), in nats, for one vocabulary row.
pub fn divergence(g: *ops.Graph, base: ops.Array, adapted: ops.Array) !f32 {
    const p = try logProb(g, base);
    const q = try logProb(g, adapted);
    const value = try @import("grad.zig").item(try g.sum(try g.mul(try g.exp(p), try g.sub(p, q)), 0, false));
    if (!std.math.isFinite(value)) return error.NonFiniteVerification;
    return @max(0, value);
}

fn logProb(g: *ops.Graph, logits: ops.Array) !ops.Array {
    const f = try g.reshape(try g.astype(logits, .float32), &.{-1});
    const z = try g.call(mlx.mlx_logsumexp_axis, .{ f, @as(c_int, 0), false });
    return g.sub(f, z);
}

test "finetune: retain KL agrees with a probability oracle and identity is zero" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = ops.Graph.init(std.testing.allocator, s);
    defer g.deinit();
    const p = try g.fromSlice(f32, &.{ @log(@as(f32, 0.75)), @log(@as(f32, 0.25)) }, &.{2});
    const q = try g.fromSlice(f32, &.{ 0, 0 }, &.{2});
    try std.testing.expectApproxEqAbs(@as(f32, @floatCast(0.75 * @log(@as(f64, 1.5)) + 0.25 * @log(@as(f64, 0.5)))), try divergence(&g, p, q), 1e-6);
    try std.testing.expectEqual(@as(f32, 0), try divergence(&g, p, p));
}
