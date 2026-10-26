//! Shifted next-token examples; a completion starts at an explicit token boundary.
const std = @import("std");

pub const Example = struct {
    inputs: []u32,
    targets: []u32,
    mask: []f32,
    completion_tokens: u32,

    pub fn deinit(e: Example, a: std.mem.Allocator) void {
        a.free(e.inputs);
        a.free(e.targets);
        a.free(e.mask);
    }
};

pub fn pack(a: std.mem.Allocator, tokens: []const u32, completion_start: usize, max_rows: usize) !Example {
    if (completion_start == 0 or completion_start > tokens.len) return error.InvalidTrainingBoundary;
    if (completion_start == tokens.len) return error.EmptyTrainingCompletion;
    if (tokens.len - 1 > max_rows or tokens.len > std.math.maxInt(u32)) return error.TrainingSequenceTooLong;
    const inputs = try a.dupe(u32, tokens[0 .. tokens.len - 1]);
    errdefer a.free(inputs);
    const targets = try a.dupe(u32, tokens[1..]);
    errdefer a.free(targets);
    const mask = try a.alloc(f32, inputs.len);
    for (mask, 0..) |*m, i| m.* = if (i + 1 >= completion_start) 1 else 0;
    return .{ .inputs = inputs, .targets = targets, .mask = mask, .completion_tokens = @intCast(tokens.len - completion_start) };
}

test "train core: packing masks only predicted completion tokens and rejects truncation" {
    const a = std.testing.allocator;
    const e = try pack(a, &.{ 4, 5, 6, 7, 8 }, 3, 512);
    defer e.deinit(a);
    try std.testing.expectEqualSlices(u32, &.{ 4, 5, 6, 7 }, e.inputs);
    try std.testing.expectEqualSlices(u32, &.{ 5, 6, 7, 8 }, e.targets);
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 1, 1 }, e.mask);
    try std.testing.expectEqual(@as(u32, 2), e.completion_tokens);
    try std.testing.expectError(error.TrainingSequenceTooLong, pack(a, &.{ 4, 5, 6 }, 1, 1));
    try std.testing.expectError(error.EmptyTrainingCompletion, pack(a, &.{ 4, 5 }, 2, 512));
    try std.testing.expectError(error.InvalidTrainingBoundary, pack(a, &.{ 4, 5 }, 0, 512));
}
