//! DeepSeek-V4.1's memory bill: its compressed caches per token, the keys a prefill query reads, and the prompt
//! pass's working set (`dsv41_mod.prefillBytes`).

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const cfg_mod = @import("config.zig");
const model_mod = @import("../../engines/mlx/model.zig");
const dsv41_mod = @import("deepseek_v41.zig");
const toModelConfig = @import("../deepseek_v4/model_config.zig").toModelConfig;

const PREFILL_RUNTIME_FLOOR_BYTES = @import("../../engines/mlx/admission/bills.zig").PREFILL_RUNTIME_FLOOR_BYTES;
const prefillNeededAtChunk = @import("../../engines/mlx/admission/prefill_chunk.zig").prefillNeededAtChunk;

pub fn dsv41PrefillMemoryNeeded(config: *const model_mod.ModelConfig, seq: u64, chunk: u64) u64 {
    return dsv41_mod.prefillBytes(config, seq, chunk, PREFILL_RUNTIME_FLOOR_BYTES);
}

/// The table's `bill`: only the KV sources' compressed latents and index keys grow per token (bf16); the windows
/// are a fixed ring per slot inside the module's state.
pub fn bill(p: *const abi.Parsed, shape: *const abi.Shape, out: *abi.Bill) callconv(.c) void {
    _ = shape;
    out.* = .{ .kv_per_token = kvPerToken(cfg_mod.of(p)), .prefill_attn_keys = &attnKeys, .prefill_bytes = &prefillBytes };
}

pub fn kvPerToken(c: *const cfg_mod.Config) u64 {
    var sum: u64 = 0;
    for (0..c.num_hidden_layers) |l| {
        const r: u64 = c.dsv4_compress_ratios[l];
        if (r == 0 or (c.dsv41_kv_sources >> @intCast(l)) & 1 == 0) continue;
        sum += c.head_dim / r;
        if ((c.dsv41_index_sources >> @intCast(l)) & 1 != 0) sum += c.dsv4_index_head_dim / r;
    }
    return sum * 2;
}

/// Every layer reads its window plus the indexer's top-k compressed entries, plus the sink.
fn attnKeys(p: *const abi.Parsed, seq: u64) callconv(.c) u64 {
    const c = cfg_mod.of(p);
    return @min(seq, @as(u64, c.sliding_window) + c.dsv4_index_topk + 1);
}

fn prefillBytes(p: *const abi.Parsed, seq: u64, chunk: u32, kv_bits: u8, warm: u64) callconv(.c) u64 {
    _ = kv_bits;
    _ = warm;
    const c = cfg_mod.of(p);
    const mc = toModelConfig(std.heap.page_allocator, c, null) catch unreachable; // no pack dir: nothing to allocate
    return dsv41PrefillMemoryNeeded(&mc, seq, chunk);
}

// ── tests ──

test "dsv41PrefillMemoryNeeded: state per compressed token, the chunk's stream, one span's transients" {
    const t = std.testing;
    const config = try model_mod.parseConfigFromJson(t.allocator, model_mod.dsv41_release_config);
    const gb: u64 = 1 << 30;
    // A short prompt is the runtime floor and one span, nowhere near a
    // dense 40-layer 512-wide KV bill.
    try t.expect(dsv41PrefillMemoryNeeded(&config, 64, 4096) < gb);
    // The chunk's stream rides every layer: a wider chunk costs its rows.
    const d4 = dsv41PrefillMemoryNeeded(&config, 32768, 4096);
    const d8 = dsv41PrefillMemoryNeeded(&config, 32768, 8192);
    try t.expect(d8 > d4 + 4096 * 4 * 5120 * 2 * 2);
    // 1M tokens of context: ~3.2 KB per token of compressed caches plus the
    // [span, entries] index scores, bounded, never [span, heads, entries].
    const far = dsv41PrefillMemoryNeeded(&config, 1 << 20, 4096);
    try t.expect(far > (1 << 20) * 3200);
    try t.expect(far < 16 * gb);
    // The one estimator the guard reads.
    try t.expectEqual(far, prefillNeededAtChunk(&config, 1 << 20, 256, 16, 4096, .{}));
}

test "deepseek_v41 bill: the plugin's terms equal the engine's on the release config" {
    const t = std.testing;
    const sdk = @import("../../sdk/root.zig");
    const json = model_mod.dsv41_release_config;
    const legacy = try model_mod.parseConfigFromJson(t.allocator, json);
    const mem: sdk.fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = json }} };
    const dir = mem.packDir();
    var p: abi.Parsed = .{};
    try t.expect(cfg_mod.parse(&dir, .of(json), .{}, null, true, &p).isOk());
    defer cfg_mod.freeParsed(&p);
    var b: abi.Bill = .{};
    bill(&p, &.{ .context = 1 << 20 }, &b);
    try t.expectEqual(legacy.kvBytesPerToken(), b.kv_per_token);
    for ([_]u64{ 1, 100, 640, 641, 4096, 1 << 20 }) |seq| {
        try t.expectEqual(legacy.prefillAttnKeys(seq), b.prefill_attn_keys.?(&p, seq));
        try t.expectEqual(prefillNeededAtChunk(&legacy, seq, 256, 16, 4096, .{}), b.prefill_bytes.?(&p, seq, 4096, 16, 0));
    }
}
