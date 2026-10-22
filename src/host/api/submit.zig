//! nonStreamingViaScheduler

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const transformer_mod = @import("../../engines/mlx/transformer.zig");
const tokenizer_mod = @import("../tokenizer/tokenizer.zig");
const generate_mod = @import("../../engines/mlx/generate.zig");
const log = @import("../log.zig");
const prefix_cache_mod = @import("../../engines/mlx/cache/prefix_cache.zig");
const scheduler_mod = @import("../../engines/mlx/scheduler.zig");
const model_registry_mod = @import("../registry/registry.zig");
const Tokenizer = tokenizer_mod.Tokenizer;
const LoadedModel = model_registry_mod.LoadedModel;
const http_server = @import("../http/server.zig");

const Conn = @import("../http/conn.zig").Conn;
const resolveKvAttnFused = @import("../../engines/mlx/request_options.zig").resolveKvAttnFused;
const mtpCapable = @import("../../engines/mlx/request_options.zig").mtpCapable;
const decodeTokens = @import("tokenize.zig").decodeTokens;
const loopTrimmedIds = @import("../chat/tools/pipeline.zig").loopTrimmedIds;
const slotFailure = @import("../http/errors.zig").slotFailure;
const nonStreamFinishReason = @import("../chat/stops.zig").nonStreamFinishReason;
const MropeData = @import("../media/encode.zig").MropeData;

/// Run a non-streaming generation through the scheduler. Returns the same
/// shape as `generate.generate` so the calling handler's response builder
/// is unchanged.
///
/// `vision_embeddings` (Phase A4/A8): when non-null, ownership transfers
/// into the slot — the slot's deinit frees the array. Each handler holds a
/// per-request `?mlx_array` local; it nulls its copy before passing here so
/// its own `defer` doesn't double-free.
pub fn nonStreamingViaScheduler(
    allocator: std.mem.Allocator,
    sch: *scheduler_mod.Scheduler,
    lm: *LoadedModel,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    full_prompt: []const u32,
    max_tokens: u32,
    sampling: generate_mod.SamplingParams,
    eos_token_ids: []const u32,
    cached_tokens: u32,
    has_tools: bool,
    enable_thinking: bool,
    enable_pld: bool,
    enable_drafter: bool,
    enable_mtp: bool,
    allow_batch_mtp: bool,
    timeout_ns: u64,
    vision_embeddings: ?mlx.mlx_array,
    media: []const prefix_cache_mod.MediaSpan,
    cache_key: u64,
    mrope: MropeData,
    logprobs_n: u32,
    /// `/v1/completions` echo: score the prompt's own tokens too (`GenerationResult.prompt_logprobs`).
    prompt_logprobs: bool,
    /// Wave 1.A: per-request KV-quant override; null = inherit scheduler default.
    kv_quant_override: ?transformer_mod.KVQuantConfig,
    kv_attn_explicit: ?bool,
    /// When non-null, the peer socket is probed on idle wakeups during the
    /// wait — a vanished client cancels the slot (aborting its prefill)
    /// instead of grinding out a ghost generation nobody will read.
    conn: ?*Conn,
) !generate_mod.GenerationResult {
    var slot = try sch.submit(.{
        .model = lm,
        .prompt_ids = prompt_ids,
        .full_prompt = full_prompt,
        .cached_tokens = cached_tokens,
        .has_tools = has_tools,
        .enable_thinking = enable_thinking,
        .sampling = sampling,
        .eos_token_ids = eos_token_ids,
        .max_tokens = max_tokens,
        .timeout_ns = timeout_ns,
        .enable_pld = enable_pld,
        .enable_drafter = enable_drafter and (lm.drafter != null or lm.dflash != null),
        .drafter = if (enable_drafter) lm.drafter else null,
        .dflash = if (enable_drafter) lm.dflash else null,
        .drafter_block_size = lm.drafter_block_size,
        .allow_batch_mtp = allow_batch_mtp,
        .enable_mtp = enable_mtp and mtpCapable(lm),
        .mtp = if (enable_mtp) lm.mtp else null,
        .mtp_depth = lm.mtp_depth,
        .pld_draft_len = http_server.server_config.default_pld_draft_len,
        .pld_key_len = http_server.server_config.default_pld_key_len,
        .kv_attn_fused = resolveKvAttnFused(lm.config.?, kv_attn_explicit, prompt_ids.len, kv_quant_override),
        .vision_embeddings = vision_embeddings,
        .media = media,
        .cache_key = cache_key,
        .client = if (conn) |c| c.client else .other,
        .mrope_pos = mrope.pos,
        .mrope_total = mrope.total,
        .mrope_delta = mrope.delta,
        .logprobs_n = logprobs_n,
        .prompt_logprobs = prompt_logprobs,
        .kv_quant_config = kv_quant_override,
    });
    defer sch.complete(slot);

    var output_ids = std.ArrayList(u32).empty;
    defer output_ids.deinit(allocator);

    var client_gone = false;
    wait: while (true) {
        const nr = slot.waitNextTimeout(Conn.STREAM_KEEPALIVE_MS) orelse {
            // Idle (long prefill). If the client is gone, cancel and serve
            // whatever accumulated — the response write will fail upstream,
            // which is fine; the win is freeing the GPU.
            if (conn) |c| {
                if (c.peerClosed()) {
                    log.info("  [cancel] client disconnected while waiting (non-stream) — cancelling slot\n", .{});
                    slot.cancel();
                    client_gone = true;
                    break :wait;
                }
            }
            continue :wait;
        };
        switch (nr) {
            .token => |t| try output_ids.append(allocator, t),
            .done => break :wait,
            .err => return slotFailure(slot),
        }
        if (conn) |c| {
            if (c.peerClosed()) {
                log.info("  [cancel] client disconnected while decoding (non-stream) — cancelling slot\n", .{});
                slot.cancel();
                client_gone = true;
                break :wait;
            }
        }
    }

    // The scheduler measures prefill_ns / decode_ns per-slot directly. Pull
    // those instead of the old single-wall-clock approximation, which
    // double-counted total time into both phases.
    const prefill_tps = generate_mod.prefillTokensPerSec(slot.prompt_tokens, slot.cached_tokens, slot.prefill_ns);
    const decode_tps = generate_mod.tokensPerSec(slot.completion_tokens, slot.decode_ns);

    const strip_leading = tok.tok_type == .sentencepiece_bpe and (sampling.constraint == null or sampling.constraint.?.proto == null);
    // A loop cut is a truncation, and the degenerate span is the part the
    // client must not get: an agent re-sends the cut turn as history, the
    // model reads its own loop back and resumes it — five loop-stops in a
    // row, each firing sooner than the last (live 2026-08-05, under pi).
    // Only reachable non-streaming: a delta cannot be retracted, so a
    // streaming client has already received the tail.
    const emit_ids = loopTrimmedIds(output_ids.items, slot.loop_trim_start);
    const text = try decodeTokens(allocator, lm, tok, emit_ids, strip_leading);
    const token_ids = try output_ids.toOwnedSlice(allocator);

    // Convert the generator's (token index, byte offset) payload boundary
    // into a byte offset inside `text`. The prefix decode uses the same
    // leading-space convention as the full text, so offsets are exact.
    var constraint_payload_byte: ?usize = null;
    if (slot.constraintPayloadStart()) |ps| {
        var pb: ?usize = ps.byte_offset;
        if (ps.token_index > 0) {
            if (ps.token_index <= emit_ids.len) {
                const prefix = decodeTokens(allocator, lm, tok, emit_ids[0..ps.token_index], strip_leading) catch null;
                if (prefix) |p| {
                    pb = p.len + ps.byte_offset;
                    allocator.free(p);
                } else pb = null;
            } else pb = null;
        }
        if (pb) |b| {
            if (b <= text.len) constraint_payload_byte = b;
        }
    }

    // Phase A5: take ownership of the slot's accumulated logprobs. After
    // `toOwnedSlice` the slot's list is empty, so `Slot.deinit` doesn't try
    // to free what we just transferred to the caller.
    const logprobs_slice: ?[]generate_mod.LogprobResult = if (slot.logprobs_buf.items.len > 0)
        slot.logprobs_buf.toOwnedSlice(slot.allocator) catch null
    else
        null;
    const prompt_logprobs_slice: ?[]generate_mod.LogprobResult = if (slot.prompt_logprobs_buf.items.len > 0)
        slot.prompt_logprobs_buf.toOwnedSlice(slot.allocator) catch null
    else
        null;

    return .{
        .text = text,
        .token_ids = token_ids,
        .prompt_tokens = slot.prompt_tokens,
        .completion_tokens = slot.completion_tokens,
        .finish_reason = nonStreamFinishReason(client_gone, slot.finish_reason),
        .prefill_tps = prefill_tps,
        .decode_tps = decode_tps,
        .prefill_ns = slot.prefill_ns,
        .decode_ns = slot.decode_ns,
        .cached_tokens = slot.cached_tokens,
        .logprobs = logprobs_slice,
        .prompt_logprobs = prompt_logprobs_slice,
        .finish_details = slot.finish_details,
        .constraint_payload_byte = constraint_payload_byte,
    };
}
