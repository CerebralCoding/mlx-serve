//! qsaMaskBytes

const std = @import("std");
const transformer_mod = @import("../../engines/mlx/transformer.zig");
const model_mod = @import("../../engines/mlx/model.zig");

const prefillTransientReserve = @import("../../engines/mlx/admission/bills.zig").prefillTransientReserve;
const resolvePrefillChunk = @import("../../engines/mlx/admission/prefill_chunk.zig").resolvePrefillChunk;

/// Bytes per (query, key) the QSA prefill holds for ONE live layer past the
/// indexer budget: the `[S, kv]` bool mask, its additive copy for the sdpa
/// arm, and the f32 indexer scores over kv/4 blocks (4 bytes/block = 1/key).
const QSA_MASK_BYTES_PER_KEY: u64 = 4;

/// qwen4_exp sparse attention materializes a `[S, kv]` mask per layer that
/// no envelope term models: 4096 x 25k keys = 410 MB, and a tight box met it
/// as a Metal OOM at 25k+. Zero for every arch without an indexer. The sizer
/// bills it at kv = chunk (no prompt yet); the admission guard at the real
/// prompt length.
pub fn qsaMaskBytes(config: *const model_mod.ModelConfig, fwd: u64, kv: u64) u64 {
    if (config.indexer_budget == 0) return 0;
    // Prefill widths gather by block index (row-chunked score sheet, no
    // [S, kv] mask); decode/verify widths still build the dense mask.
    if (fwd >= transformer_mod.FUSED256_MIN_Q_LEN and transformer_mod.qsaGatherEnabled())
        return transformer_mod.qsaPrefillTransientBytes(config.indexer_n_heads, fwd, kv, config.indexer_compress_ratio, config.indexer_head_dim);
    return QSA_MASK_BYTES_PER_KEY * fwd * kv * 5 / 4;
}

// ── tests ──

test "qsaMaskBytes: a qwen4_exp twin bills the QSA mask and steps a rung the qwen3_5 twin keeps" {
    const t = std.testing;
    var twin = model_mod.ModelConfig{};
    twin.num_hidden_layers = 40;
    twin.num_attention_heads = 24;
    twin.num_key_value_heads = 2;
    twin.head_dim = 256;
    twin.hidden_size = 2560;
    twin.intermediate_size = 9216;
    twin.intermediate_size_declared = true;
    twin.quant_bits = 4;
    twin.max_position_embeddings = 262144;
    var q4 = twin;
    q4.indexer_budget = 2048;
    q4.indexer_n_heads = 4;
    q4.indexer_head_dim = 128;
    q4.indexer_compress_ratio = 4;
    try t.expectEqual(@as(u64, 0), qsaMaskBytes(&twin, 4096, 25000));
    // Decode/verify widths build the dense mask; prefill widths gather by
    // block and bill the bounded score sheet instead of 4 B x rows x keys.
    try t.expectEqual(@as(u64, 4 * 8 * 25000 * 5 / 4), qsaMaskBytes(&q4, 8, 25000));
    transformer_mod.qsa_gather_override = true;
    defer transformer_mod.qsa_gather_override = null;
    const saved_fused = transformer_mod.qsa_score_fused_override;
    const saved_nax = transformer_mod.vqmm_nax_probe_override;
    defer {
        transformer_mod.qsa_score_fused_override = saved_fused;
        transformer_mod.vqmm_nax_probe_override = saved_nax;
    }
    transformer_mod.vqmm_nax_probe_override = false;
    transformer_mod.qsa_score_fused_override = true;
    const gathered_composed = qsaMaskBytes(&q4, 4096, 250_000);
    try t.expect(gathered_composed > 0);
    try t.expect(gathered_composed < 4 * 4096 * 250_000);
    try t.expectEqual(transformer_mod.qsaPrefillTransientBytes(4, 4096, 250_000, 4, 128), gathered_composed);
    transformer_mod.vqmm_nax_probe_override = null;
    transformer_mod.qsa_score_fused_override = true;
    const gathered_fused = qsaMaskBytes(&q4, 4096, 250_000);
    try t.expectEqual(transformer_mod.qsaPrefillTransientBytes(4, 4096, 250_000, 4, 128), gathered_fused);
    if (transformer_mod.qsaScoreFusedActive()) {
        try t.expect(gathered_fused < gathered_composed);
    }
    transformer_mod.qsa_score_fused_override = false;
    transformer_mod.vqmm_nax_probe_override = false;
    const gathered = qsaMaskBytes(&q4, 4096, 250_000);
    try t.expectEqual(gathered_composed, gathered);
    const twin_bill = prefillTransientReserve(&twin, 16, 8192);
    const q4_bill = prefillTransientReserve(&q4, 16, 8192);
    try t.expectEqual(twin_bill + qsaMaskBytes(&q4, 8192, 8192) + q4.qsaRingBytes() * 5 / 4, q4_bill);
    // A ceiling whose quarter-share sits between the two bills at 8192.
    const weights: u64 = 70_000_000_000;
    const ceiling = weights + (twin_bill + q4_bill) / 2 * 4;
    try t.expectEqual(@as(u32, 8192), resolvePrefillChunk(&twin, 16, ceiling, weights, 0, 0));
    try t.expectEqual(@as(u32, 4096), resolvePrefillChunk(&q4, 16, ceiling, weights, 0, 0));
}
