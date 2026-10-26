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
const sdk = @import("../../sdk/root.zig");

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
        const n: u32 = @intCast(v41.resumePrompt(t.m, 0, .of(prompt), .{}));
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

/// The plugin's logits `[rows, V]` (host f32, caller frees) for `toks` of `slots` at `positions`.
fn slotForward(t: *Twin, mode: abi.Mode, slots: []const u32, toks: []const u32, positions: []const u32) ![]f32 {
    var out: abi.ForwardOut = .{};
    const rows: u32 = if (mode == .batched_rows) 1 else @intCast(toks.len);
    const v = v41.table.forward(t.m, &.{ .mode = mode, .rows = rows, .tokens = .of(toks), .slots = .of(slots), .positions = .of(positions) }, &out);
    if (!v.isOk()) {
        std.debug.print("forward: {s}\n", .{v.why.slice()});
        return error.PluginForward;
    }
    const l: mlx.mlx_array = .{ .ctx = out.logits.ctx };
    defer _ = mlx.mlx_array_free(l);
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, l, .float32, t.s));
    try mlx.check(mlx.mlx_array_eval(f));
    return testing.allocator.dupe(f32, (mlx.mlx_array_data_float32(f) orelse return error.NoData)[0..mlx.mlx_array_size(f)]);
}

/// Three batched ticks of two slots (prompts `a` and `b`, decode tokens `da` and `db`), each tick's `[2, V]` logits.
fn batchTicks(files: []const sdk.fixture.File, a: []const u32, b: []const u32, da: [3]u32, db: [3]u32) ![3][]f32 {
    const t = try Twin.open(files, "");
    defer t.close();
    testing.allocator.free(try slotForward(t, .prefill, &.{1}, a, &.{0}));
    testing.allocator.free(try slotForward(t, .prefill, &.{2}, b, &.{0}));
    var out: [3][]f32 = undefined;
    for (0..3) |i| out[i] = try slotForward(t, .batched_rows, &.{ 1, 2 }, &.{ da[i], db[i] }, &.{ @intCast(a.len + i), @intCast(b.len + i) });
    return out;
}

test "deepseek_v41 plugin: a slot's row in a batch depends on its own state only, past the window, a pooling group and the indexer's budget" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    const p = [2][]const u32{ &.{ 3, 17, 42, 9, 7, 25, 1, 60, 11, 2, 90 }, &.{ 5, 8, 13, 21, 34, 55, 89, 4, 6, 10, 12, 14, 16, 18 } };
    const d = [2][3]u32{ .{ 44, 2, 71 }, .{ 9, 30, 63 } };
    // Two rows run other kernels than one (the MoE's gather arm), so a row is compared at the same width: beside
    // another slot it equals itself beside a copy of its own conversation.
    const mixed = try batchTicks(files, p[0], p[1], d[0], d[1]);
    defer for (mixed) |x| testing.allocator.free(x);
    const own0 = try batchTicks(files, p[0], p[0], d[0], d[0]);
    defer for (own0) |x| testing.allocator.free(x);
    const own1 = try batchTicks(files, p[1], p[1], d[1], d[1]);
    defer for (own1) |x| testing.allocator.free(x);
    for (0..3) |i| {
        const V = mixed[i].len / 2;
        try testing.expectEqualSlices(f32, own0[i][0..V], mixed[i][0..V]);
        try testing.expectEqualSlices(f32, own1[i][V..], mixed[i][V..]);
    }
}

test "deepseek_v41 plugin: a one-slot batch is the slot's own decode step, bit for bit" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    const prompt = [_]u32{ 3, 17, 42, 9, 7, 25, 1, 60, 11, 2, 90 };
    const steps = [_]u32{ 44, 2, 71 };
    var ts: [2]*Twin = undefined;
    for (&ts) |*t| t.* = try Twin.open(files, "");
    defer for (ts) |t| t.close();
    for (ts) |t| testing.allocator.free(try slotForward(t, .prefill, &.{1}, &prompt, &.{0}));
    for (steps, 0..) |tok, i| {
        const pos: u32 = @intCast(prompt.len + i);
        const want = try slotForward(ts[0], .decode, &.{1}, &.{tok}, &.{pos});
        defer testing.allocator.free(want);
        const got = try slotForward(ts[1], .batched_rows, &.{1}, &.{tok}, &.{pos});
        defer testing.allocator.free(got);
        try testing.expectEqualSlices(f32, want, got);
    }
}
