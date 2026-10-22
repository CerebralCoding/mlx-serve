//! DiffusionGemma on the scheduler: prefill and decode tick.

const transformer_mod = @import("../../engines/mlx/transformer.zig");
const generate_mod = @import("../../engines/mlx/generate.zig");
const diffusion_mod = @import("runner.zig");
const Transformer = transformer_mod.Transformer;

const finishSlot = @import("../../engines/mlx/finish.zig").finishSlot;
const Scheduler = @import("../../engines/mlx/scheduler.zig").Scheduler;
const Slot = @import("../../engines/mlx/slot.zig").Slot;

/// DiffusionGemma prefill: refresh the slot ctx, build the per-slot
/// diffusion Runner (which dequantizes the embedding table for
/// self-conditioning), and run the causal ENCODER pass over the full prompt
/// to fill the slot's KV cache. The hot prefix cache is intentionally NOT
/// consulted (v1): restored snapshots leave per-layer cache VIEWS stale, and
/// the diffusion decoder reads them via denseView before any update would
/// rebuild them.
pub fn runPrefillDiffusion(sch: *Scheduler, slot: *Slot) !void {
    _ = sch;
    slot.ctx.cache = &slot.cache;
    slot.ctx.moe_seq_offset = &slot.moe_seq_offset;
    slot.ctx.ssm_entries = slot.ssm_entries;
    slot.ctx.vision_embeddings = null; // vision tower not wired for this arch
    slot.ctx.capture_hidden = null;
    slot.ctx.kv_attn_fused = false;

    const xfm: *Transformer = slot.model.transformer.?;
    const runner = try slot.allocator.create(diffusion_mod.Runner);
    errdefer slot.allocator.destroy(runner);
    runner.* = try diffusion_mod.Runner.init(
        slot.allocator,
        xfm,
        &slot.ctx,
        slot.sampling.temperature,
        slot.max_tokens,
    );
    errdefer runner.deinit();
    runner.cancel_flag = &slot.cancelled;

    try runner.prefill(slot.full_prompt);

    slot.diffusion = runner;
    slot.prompt_tokens = @intCast(slot.full_prompt.len);
    slot.state = .decoding;
}

/// Diffusion decode tick: denoise and commit ONE canvas (≤ 48 decoder
/// forwards), then emit its tokens through the slot — block-wise streaming
/// falls out of the normal slot machinery. EOS inside the canvas finishes
/// the request without emitting the stop token (matching the AR paths); the
/// canvas remainder after EOS is discarded. The runner checks
/// `slot.cancelled` once per denoising step.
pub fn runDiffusionDecodeTick(sch: *Scheduler, slot: *Slot, runner: *diffusion_mod.Runner) !void {
    const result = runner.nextCanvas(slot.allocator) catch |err| switch (err) {
        error.Cancelled => return,
        else => return err,
    };
    if (result == null) {
        finishSlot(sch, slot, "length");
        return;
    }
    defer slot.allocator.free(result.?.tokens);
    for (result.?.tokens) |t| {
        if (slot.cancelled.load(.acquire)) return;
        if (generate_mod.isEosId(t, slot.eos_token_ids)) {
            finishSlot(sch, slot, "stop");
            return;
        }
        slot.pushToken(t);
        if (t != 0) slot.was_pad_only = false;
        slot.completion_tokens += 1;
        if (slot.completion_tokens >= slot.max_tokens) {
            finishSlot(sch, slot, "length");
            return;
        }
    }
}
