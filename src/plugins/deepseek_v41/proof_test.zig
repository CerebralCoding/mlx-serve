//! Parity proof for the V4.1 plugin: its forward on a tiny random pack equals the deleted legacy seam at every step,
//! its verify block the module's own all-rows extend, and a resumed prompt the legacy resume, as recorded in the tapes
//! in `fixtures/` (`../deepseek_v4/proof_test.zig`).

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const proof = @import("../deepseek_v4/proof_test.zig");
const fixture = @import("../deepseek_v4/fixture.zig");
const v41 = @import("plugin.zig");

const testing = std.testing;
const Twin = proof.Twin(v41);
test "deepseek_v41 plugin: prefill, GPU decode, a verify block and a later chunk equal the legacy seam bit for bit" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin.open(try fixture.v41Files(arena.allocator(), 7), @embedFile("fixtures/steps.tape"));
    defer t.close();
    try proof.expectSteps(v41, t);
}

test "deepseek_v41 plugin: a prompt extending the last one resumes as the legacy resume does, and continues bit-equal" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin.open(try fixture.v41Files(arena.allocator(), 7), @embedFile("fixtures/proof-resume.tape"));
    defer t.close();
    const first = [_]u32{ 3, 17, 42, 9, 7, 25, 1, 60 };
    const second = first ++ [_]u32{ 11, 2, 90 };
    for ([_][]const u32{ &first, &second }) |prompt| {
        const n: u32 = @intCast(v41.resumePrompt(t.m, .of(prompt), .{}));
        try t.check(mlx.mlx_array_new_data(&n, &[_]c_int{1}, 1, .uint32));
        try t.check(try t.plugin(.{ .mode = .prefill, .ids = prompt[n..] }, n));
    }
    try t.tape.expectDone();
}

test "deepseek_v41 plugin: a decode token on the device equals the legacy decode" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin.open(try fixture.v41Files(arena.allocator(), 7), @embedFile("fixtures/device-decode.tape"));
    defer t.close();
    try proof.expectDeviceDecode(v41, t);
}

test "deepseek_v41 plugin: trunk_weight hands out the loaded embedding and head" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin.open(try fixture.v41Files(arena.allocator(), 7), "");
    defer t.close();
    const m: *v41.Model = @ptrCast(@alignCast(t.m));
    var w: @import("../../abi/quant.zig").Weight = .{};
    try testing.expect(v41.table.spec.?.trunk_weight.?(t.m, .embed, &w).isOk());
    try testing.expectEqual(m.mdl.embed.w.ctx, w.w.ctx);
    try testing.expect(v41.table.spec.?.trunk_weight.?(t.m, .head, &w).isOk());
    try testing.expectEqual(m.mdl.head.w.ctx, w.w.ctx);
    try testing.expectEqual(m.mdl.head.qp.bits, w.bits);
}
