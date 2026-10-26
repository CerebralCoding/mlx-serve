//! Completion-only cross entropy with a vocabulary-chunked frozen output projection.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const grad = @import("grad.zig");
const Linear = @import("../bases/blocks/linear.zig").Linear;
const Array = ops.Array;
const Graph = ops.Graph;

pub fn crossEntropy(g: *Graph, logits: Array, targets: Array, mask: Array) !Array {
    if (ops.shape(logits).len != 2) return error.InvalidTrainingShape;
    try validate(logits, targets, mask);
    const f = try g.astype(logits, .float32);
    const normalizer = try g.call(mlx.mlx_logsumexp_axis, .{ f, @as(c_int, -1), false });
    const selected = try g.reshape(try g.takeAlong(f, try g.reshape(targets, &.{ -1, 1 }), -1), &.{-1});
    return maskedMean(g, try g.sub(normalizer, selected), mask);
}

pub fn projected(g: *Graph, hidden: Array, head: Linear, targets: Array, mask: Array, chunk: u32) !Array {
    if (chunk == 0 or ops.shape(head.w).len != 2) return error.InvalidTrainingShape;
    const rows = ops.shape(targets)[0];
    const h = try g.reshape(hidden, &.{ rows, -1 });
    try validate(h, targets, mask);
    const vocab: u32 = @intCast(ops.shape(head.w)[0]);
    if (vocab == 0) return error.InvalidTrainingShape;
    const ids = try g.astype(targets, .int32);
    var z: ?Array = null;
    var chosen = try g.zeros(&.{rows}, .float32);
    var start: u32 = 0;
    while (start < vocab) {
        const end = start + @min(chunk, vocab - start);
        var part = head;
        part.w = try g.sliceAxis(head.w, 0, @intCast(start), @intCast(end));
        if (head.s.ctx != null) part.s = try g.sliceAxis(head.s, 0, @intCast(start), @intCast(end));
        if (head.b.ctx != null) part.b = try g.sliceAxis(head.b, 0, @intCast(start), @intCast(end));
        if (head.bias.ctx != null) part.bias = try g.sliceAxis(head.bias, 0, @intCast(start), @intCast(end));
        const logits = try g.astype(try @import("../bases/blocks/head.zig").project(g, &part, h), .float32);
        const lse = try g.call(mlx.mlx_logsumexp_axis, .{ logits, @as(c_int, -1), false });
        if (z) |previous| {
            const m = try g.maximum(previous, lse);
            z = try g.add(m, try g.log(try g.add(try g.exp(try g.sub(previous, m)), try g.exp(try g.sub(lse, m)))));
        } else z = lse;
        const relative = try g.sub(ids, try g.scalar(@floatFromInt(start), .int32));
        const index = try g.minimum(try g.maximum(relative, try g.scalar(0, .int32)), try g.scalar(@floatFromInt(end - start - 1), .int32));
        const selected = try g.reshape(try g.takeAlong(logits, try g.reshape(index, &.{ -1, 1 }), -1), &.{rows});
        const valid = try g.equal(relative, index);
        chosen = try g.add(chosen, try g.where(valid, selected, try g.scalar(0, .float32)));
        start = end;
    }
    return maskedMean(g, try g.sub(z.?, chosen), mask);
}

fn validate(x: Array, targets: Array, mask: Array) !void {
    const t = ops.shape(targets);
    const m = ops.shape(mask);
    if (t.len != 1 or m.len != 1 or t[0] == 0 or t[0] != m[0] or ops.shape(x)[0] != t[0]) return error.InvalidTrainingShape;
}

fn maskedMean(g: *Graph, values: Array, mask: Array) !Array {
    const fmask = try g.astype(mask, .float32);
    const selected = try g.where(try g.greater(fmask, try g.scalar(0, .float32)), values, try g.scalar(0, .float32));
    return g.div(try g.sum(try g.mul(selected, fmask), 0, false), try g.maximum(try g.sum(fmask, 0, false), try g.scalar(1, .float32)));
}

test "train core: masked CE ignores prompt rows and stays stable at large logits" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = Graph.init(std.testing.allocator, s);
    defer g.deinit();
    const logits = try g.fromSlice(f32, &.{ 1000, 1001, 999, -900, 900, 0, 2, 2, 2 }, &.{ 3, 3 });
    const ids = try g.fromSlice(u32, &.{ 1, 0, 2 }, &.{3});
    const mask = try g.fromSlice(f32, &.{ 1, 0, 1 }, &.{3});
    const want: f32 = (@log(1 + @exp(@as(f32, -1)) + @exp(@as(f32, -2))) + @log(@as(f32, 3))) / 2;
    try std.testing.expectApproxEqAbs(want, try grad.item(try crossEntropy(&g, logits, ids, mask)), 0.0001);
    try std.testing.expectEqual(@as(f32, 0), try grad.item(try crossEntropy(&g, logits, ids, try g.zeros(&.{3}, .float32))));
}

test "train core: chunked vocabulary loss and hidden gradients equal a dense f32 oracle" {
    const Objective = struct {
        head: Linear,
        targets: Array,
        mask: Array,
        chunk: u32,
        fn loss(p: *@This(), g: *Graph, x: []const Array) !Array {
            return if (p.chunk == 0)
                crossEntropy(g, try g.matmul(x[0], try g.transpose(p.head.w, &.{ 1, 0 })), p.targets, p.mask)
            else
                projected(g, x[0], p.head, p.targets, p.mask, p.chunk);
        }
    };
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = Graph.init(std.testing.allocator, s);
    defer g.deinit();
    var p: Objective = .{ .head = .{ .w = try g.fromSlice(f32, &.{ 1, 2, 3, -1, -2, 1, 0, 2, 3, 2 }, &.{ 5, 2 }) }, .targets = try g.fromSlice(u32, &.{ 0, 2, 4 }, &.{3}), .mask = try g.fromSlice(f32, &.{ 0, 1, 1 }, &.{3}), .chunk = 0 };
    const h = try g.fromSlice(f32, &.{ 1, 0, 0.5, 1, -0.3, 0.1 }, &.{ 3, 2 });
    const dense = try grad.valueAndGrad(&g, &p, Objective.loss, &.{h});
    defer dense.deinit(g.a);
    for ([_]u32{ 1, 2, 4, 9 }) |chunk| {
        p.chunk = chunk;
        const actual = try grad.valueAndGrad(&g, &p, Objective.loss, &.{h});
        defer actual.deinit(g.a);
        try std.testing.expectApproxEqAbs(try grad.item(dense.value), try grad.item(actual.value), 1e-5);
        const delta = try g.max(try g.abs(try g.reshape(try g.sub(dense.gradients[0], actual.gradients[0]), &.{-1})), 0, false);
        try std.testing.expect((try grad.item(delta)) < 1e-5);
    }
}
