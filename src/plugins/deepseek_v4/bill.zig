//! dsv4PrefillMemoryNeeded

const std = @import("std");

const PREFILL_RUNTIME_FLOOR_BYTES = @import("../../engines/mlx/admission/bills.zig").PREFILL_RUNTIME_FLOOR_BYTES;
const prefillMemoryNeeded = @import("../../engines/mlx/admission/bills.zig").prefillMemoryNeeded;

/// deepseek_v4 sibling of `prefillMemoryNeeded`: bills what `extendState`
/// ACTUALLY allocates (third bite of the bills-what-the-arch-reads class,
/// live 2026-08-01: the generic estimator billed 4069 MB for a 7514-token pi
/// prompt against 3610 MB available and 400'd it; the honest bill is ~2.3 GB).
/// Two generic terms are wrong for this arch: `chunk` is the generic MoE
/// prefill cap (4096) but dsv4 sub-chunks internally at `prefillSub()` (512),
/// shrinking the per-chunk MLP envelope 8x; and the fp16 score-scratch term
/// misses dsv4's real attention transient, the [C, tk, latent] f32 gathered-K
/// set — the very allocation PREFILL_SUB exists to bound. State term: raw kv
/// latents are [n, latent] f32 per layer, and the compressed arms add about
/// half the raw again (ratio-4 slots at 2x width + ratio-128 + indexer), so
/// 3/2 x raw covers the lot; kv-quant never applies to this module-owned
/// state, so the bill is unconditionally f32.
pub fn dsv4PrefillMemoryNeeded(seq: u64, layers: u64, latent: u64, hidden: u64, ffn: u64, sub_chunk: u64, attn_keys: u64) u64 {
    const fwd: u64 = @min(sub_chunk, @max(seq, 1));
    const kv_bytes: u64 = layers * seq * latent * 4 * 3 / 2;
    const gather: u64 = fwd * @min(attn_keys, seq) * latent * 4;
    const mlp: u64 = 8 * fwd * @max(hidden, ffn) * 2;
    // Same chunk-independent runtime floor the generic estimator bills: it is
    // a property of the MLX runtime and the KV cache's growth, not of the arch
    // (the 2026-08-01 live case still admits — 2984 MB against 3610 free).
    return (kv_bytes + gather + 3 * mlp + PREFILL_RUNTIME_FLOOR_BYTES) * 5 / 4;
}

// ── tests ──

test "dsv4PrefillMemoryNeeded: bills the arch's own sub-chunk and f32 gather, not the generic MoE chunk" {
    const t = std.testing;
    // Live 2026-08-01 (pi + stochastic DSpark, stages resident): a 7514-token
    // prompt was 400-rejected — the generic estimator billed 4069 MB against
    // 3610 MB available. Its chunk term is the generic MoE prefill cap (4096)
    // but dsv4 sub-chunks internally at 512 (deepseek_v4.prefillSub — the
    // MLP transient envelope is 8x smaller), and its score term is fp16
    // attention scratch while dsv4's real transient is the [C, tk, latent]
    // f32 gathered-K set (the very allocation PREFILL_SUB exists to bound).
    // The honest bill for the same request is ~2.3 GB — admitted with room.
    const available: u64 = 3610 * 1024 * 1024;
    const generic = prefillMemoryNeeded(7514, 64, 1, 88064, 512, 512, 4096, 12288, 16, 4096, 641, 0, 0, .{});
    try t.expectEqual(@as(u64, 4069 + 640), generic / (1024 * 1024));
    try t.expect(generic > available);
    const honest = dsv4PrefillMemoryNeeded(7514, 43, 512, 4096, 12288, 512, 641);
    try t.expectEqual(@as(u64, 2344 + 640), honest / (1024 * 1024));
    try t.expect(honest < available);
    // Not under-billing: the full 25% margin survives over the real state +
    // transient floor (raw + compressed latents in f32, the gathered-K set).
    const floor: u64 = 43 * 7514 * 512 * 4 * 3 / 2 + 512 * 641 * 512 * 4;
    try t.expect(honest > floor * 5 / 4);
    // A prompt shorter than the sub-chunk bills its own width, not the cap
    // (the hy_v3 31-token-prompt class).
    try t.expect(dsv4PrefillMemoryNeeded(31, 43, 512, 4096, 12288, 512, 641) < 704 * 1024 * 1024);
    // The bill stays honest at the far end: a full 40960-token window costs
    // real state (~5 GB of latents) — the fix must not flatten the curve.
    try t.expect(dsv4PrefillMemoryNeeded(40960, 43, 512, 4096, 12288, 512, 641) > 6 * 1024 * 1024 * 1024);
}
