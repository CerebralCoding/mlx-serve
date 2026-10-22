//! dsv41PrefillMemoryNeeded

const std = @import("std");
const model_mod = @import("../../engines/mlx/model.zig");
const dsv41_mod = @import("deepseek_v41.zig");

const PREFILL_RUNTIME_FLOOR_BYTES = @import("../../engines/mlx/admission/bills.zig").PREFILL_RUNTIME_FLOOR_BYTES;
const prefillNeededAtChunk = @import("../../engines/mlx/admission/prefill_chunk.zig").prefillNeededAtChunk;

pub fn dsv41PrefillMemoryNeeded(config: *const model_mod.ModelConfig, seq: u64, chunk: u64) u64 {
    return dsv41_mod.prefillBytes(config, seq, chunk, PREFILL_RUNTIME_FLOOR_BYTES);
}

// ── tests ──

test "dsv41PrefillMemoryNeeded: state per compressed token, the chunk's stream, one span's transients" {
    const t = std.testing;
    const config = try model_mod.parseConfigFromJson(t.allocator, model_mod.dsv41_release_config);
    const gb: u64 = 1 << 30;
    const bill = struct {
        fn f(cfg: *const model_mod.ModelConfig, seq: u64, chunk: u64) u64 {
            return dsv41PrefillMemoryNeeded(cfg, seq, chunk);
        }
    }.f;
    // A short prompt is the runtime floor and one span, nowhere near a
    // dense 40-layer 512-wide KV bill.
    try t.expect(bill(&config, 64, 4096) < gb);
    // The chunk's stream rides every layer: a wider chunk costs its rows.
    const d4 = bill(&config, 32768, 4096);
    const d8 = bill(&config, 32768, 8192);
    try t.expect(d8 > d4 + 4096 * 4 * 5120 * 2 * 2);
    // 1M tokens of context: ~3.2 KB per token of compressed caches plus the
    // [span, entries] index scores, bounded, never [span, heads, entries].
    const far = bill(&config, 1 << 20, 4096);
    try t.expect(far > (1 << 20) * 3200);
    try t.expect(far < 16 * gb);
    // The one estimator the guard reads.
    try t.expectEqual(far, prefillNeededAtChunk(&config, 1 << 20, 256, 16, 4096, .{}));
}
