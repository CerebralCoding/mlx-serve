//! The DeepSeek plugins' bills and configs pinned against the engine's own (`ModelConfig`, `prefillNeededAtChunk`).

const std = @import("std");
const abi = @import("../../../abi/plugin.zig");
const config = @import("../../../plugins/deepseek_v4/config.zig");
const bill = @import("../../../plugins/deepseek_v4/bill.zig").bill;

test "deepseek_v4 bill: the plugin's terms equal the engine's on the release config" {
    const t = std.testing;
    const model = @import("../model.zig");
    const prefillNeededAtChunk = @import("../admission/prefill_chunk.zig").prefillNeededAtChunk;
    const sdk = @import("../../../sdk/root.zig");
    const legacy = try model.parseConfigFromJson(t.allocator, config.release_config);
    const mem: sdk.fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = config.release_config }} };
    const dir = mem.packDir();
    var p: abi.Parsed = .{};
    try t.expect(config.parse(&dir, .of(config.release_config), .{}, null, true, &p).isOk());
    defer config.freeParsed(&p);
    var b: abi.Bill = .{};
    bill(&p, &.{ .context = 1 << 20 }, &b);
    try t.expectEqual(legacy.kvBytesPerToken(), b.kv_per_token);
    for ([_]u64{ 1, 100, 128, 129, 4096, 65536, 1 << 20 }) |seq| {
        try t.expectEqual(legacy.prefillAttnKeys(seq), b.prefill_attn_keys.?(&p, seq));
        try t.expectEqual(prefillNeededAtChunk(&legacy, seq, 256, 16, 4096, .{}), b.prefill_bytes.?(&p, seq, 4096, 16, 0));
    }
}
