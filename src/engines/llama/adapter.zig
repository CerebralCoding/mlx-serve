//! The llama.cpp engine on the scheduler: load, prefill, sequence pick and decode ticks.

const std = @import("std");
const generate_mod = @import("../mlx/generate.zig");
const tokenize_cache_mod = @import("../../host/tokenizer/cache.zig");
const load = @import("load.zig");
const policy = @import("policy.zig");
const arch_llama = if (@import("build_options").llama_engine) @import("bridge.zig") else @import("stub.zig");
const log = @import("../../sdk/util/log.zig");

const MAX_BATCH_GROUP = @import("../mlx/batching/verdict.zig").MAX_BATCH_GROUP;
const publishHotCacheResidency = @import("../mlx/cache/budget.zig").publishHotCacheResidency;
const finishSlot = @import("../mlx/finish.zig").finishSlot;
const Scheduler = @import("../mlx/scheduler.zig").Scheduler;
const Slot = @import("../mlx/slot.zig").Slot;
const scheduler_mod = @import("../mlx/scheduler.zig");

/// llama.cpp load on the inference thread. Mirrors `doLoadDs4OnInferenceThread`:
/// open the embedded engine (its Metal kernels bind to this thread's GPU stream
/// from t0), install it on the entry, move the stub config/tok/chat_config over,
/// and mark ready. The stub config carries the effective context length so the
/// server's memory estimate and `runPrefillLlama` size the session correctly.
pub fn doLoadLlamaOnInferenceThread(sch: *Scheduler, params: anytype) !void {
    // The stub config carries the per-sequence context (--ctx-size or 8192).
    const opened = try load.open(arch_llama, sch.io, sch.allocator, params.llama_path, params.llama, params.config.max_position_embeddings);
    const engine = opened.engine;
    errdefer engine.close();
    const llama_ctx = opened.ctx;
    errdefer llama_ctx.free();

    // Make sure the stub config's EOS set includes the engine's EOS so the
    // streaming/non-streaming stop checks fire.
    const eos_id: u32 = @intCast(engine.eosToken());
    params.config.addEosToken(eos_id);

    // Adopt the GGUF's embedded chat template into the stub ChatConfig so
    // `chat.encodeChatViaLlama` can render it through mlx-serve's Jinja engine
    // (which also supplies the tool-synthesis fallback). The stub starts with an
    // empty allocator-owned template; swap it for the model's, freed by deinit.
    if (engine.chatTemplate()) |tmpl| {
        if (params.chat_config.allocator.dupe(u8, tmpl)) |dup| {
            params.chat_config.allocator.free(params.chat_config.chat_template);
            params.chat_config.chat_template = dup;
        } else |_| {}
    }

    const entry = params.entry;
    entry.llama_engine = engine;
    entry.llama_ctx = llama_ctx;
    entry.releaseRetainedCpuState();
    entry.config = params.config;
    entry.tokenizer = params.tok;
    entry.chat_config = params.chat_config;
    entry.weights = null;
    entry.transformer = null;
    entry.vision_encoder = null;
    entry.drafter = null;
    entry.dflash = null;
    entry.drafter_block_size = 0;
    entry.drafter_path = "";
    entry.prefix_cache = null;
    // Iteration 2: tokenize cache also applies on the llama path — same
    // chat-template render + tokenize round-trip per request. Wire it
    // here too so `--tokenize-cache-entries N` works for GGUFs.
    if (params.tokenize_cache_entries > 0) {
        entry.tokenize_cache = tokenize_cache_mod.TokenizeCache.init(
            sch.allocator,
            params.tokenize_cache_entries,
        );
    }
    const bytes_resident: u64 = if (entry.bytes_on_disk) |b| b else 0;

    sch.registry.mutex.lockUncancelable(sch.io);
    sch.registry.markReadyLocked(entry, bytes_resident);
    sch.registry.mutex.unlock(sch.io);

    sch.current_model = entry;
    sch.xfm = null;
    sch.weights = null;
    sch.vision_encoder = null;
    sch.drafter = null;
    sch.dflash = null;
    publishHotCacheResidency(sch);
    if (scheduler_mod.hot_cache_budget_invalidate) |f| f();
}

pub const pickLlamaSeq = policy.pickSeq;

/// llama.cpp prefill: claim a sequence of the model's context, sync it to the
/// prompt (reusing the KV prefix a previous request left there), and sample the
/// first token. `cached_tokens` reports the reused prefix; `prompt_tokens` stays
/// the full prompt so prefill tok/s reflects only the uncached suffix.
pub fn runPrefillLlama(sch: *Scheduler, slot: *Slot) !void {
    const ctx = slot.model.llama_ctx orelse return error.ModelNotLoaded;
    const i32_prompt = try slot.allocator.alloc(i32, slot.full_prompt.len);
    defer slot.allocator.free(i32_prompt);
    for (slot.full_prompt, 0..) |t, i| i32_prompt[i] = @intCast(t);

    // Under the lock `complete` releases sequences under: a request already
    // completed (cancelled mid-admission) must not take one.
    const seq = blk: {
        sch.queue_mu.lockUncancelable(sch.io);
        defer sch.queue_mu.unlock(sch.io);
        if (!slot.holds_session) return error.Cancelled;
        const seq = &ctx.seqs[pickLlamaSeq(ctx.seqs, i32_prompt) orelse return error.NoFreeLlamaSequence];
        seq.busy = true;
        seq.last_used_ns = @intCast(std.Io.Timestamp.now(sch.io, .boot).nanoseconds);
        slot.llama_seq = seq;
        break :blk seq;
    };

    const cached = try seq.syncWithFallback(i32_prompt);
    slot.llama_next = seq.sample(llamaSampling(slot), &slot.llama_rng);
    if (slot.llama_next < 0) return error.LlamaSampleFailed;
    slot.prompt_tokens = @intCast(slot.full_prompt.len);
    slot.cached_tokens = @intCast(cached);
    slot.state = .decoding;
}

fn llamaSampling(slot: *const Slot) arch_llama.Sampling {
    return .{
        .temperature = slot.sampling.temperature,
        .top_k = @intCast(slot.sampling.top_k),
        .top_p = slot.sampling.top_p,
        .min_p = slot.sampling.min_p orelse 0.0,
    };
}

/// Emit one llama.cpp token: an end-of-generation token stops the slot (never
/// emitted, like the MLX path), and so does reaching `max_tokens`. True when
/// the slot keeps decoding.
fn emitLlamaToken(sch: *Scheduler, slot: *Slot, tok: i32) bool {
    const engine = slot.model.llama_engine.?;
    const tok_u32: u32 = @intCast(tok);
    if (engine.isEog(tok) or generate_mod.isEosId(tok_u32, slot.eos_token_ids)) {
        finishSlot(sch, slot, "stop");
        return false;
    }
    slot.pushToken(tok_u32);
    if (tok_u32 != 0) slot.was_pad_only = false;
    slot.completion_tokens += 1;
    if (slot.completion_tokens >= slot.max_tokens) {
        finishSlot(sch, slot, "length");
        return false;
    }
    return true;
}

/// Run each model's llama.cpp slots as one group. A group that fails fails
/// its own slots only, never another model's.
pub fn runLlamaGroups(sch: *Scheduler, slots: []*Slot) void {
    std.sort.pdq(*Slot, slots, {}, struct {
        fn lt(_: void, a: *Slot, b: *Slot) bool {
            return @intFromPtr(a.model) < @intFromPtr(b.model);
        }
    }.lt);
    var i: usize = 0;
    while (i < slots.len) {
        var j = i + 1;
        while (j < slots.len and slots[j].model == slots[i].model) j += 1;
        runLlamaDecodeTick(sch, slots[i..j]) catch |err| {
            log.err("[llama] decode tick failed: {s}\n", .{@errorName(err)});
            for (slots[i..j]) |s| s.markError(@errorName(err));
        };
        i = j;
    }
}

/// llama.cpp decode tick for one model's slots: each emits the token it
/// sampled last round, then every slot still decoding feeds it in ONE batched
/// step and samples its next. A slot decoding alone on a context with an MTP
/// head runs an MTP round instead: with company, drafts cost more batch rows
/// than they save.
pub fn runLlamaDecodeTick(sch: *Scheduler, slots: []*Slot) !void {
    var live_buf: [MAX_BATCH_GROUP]*Slot = undefined;
    var seqs: [MAX_BATCH_GROUP]*arch_llama.LlamaSeq = undefined;
    var toks: [MAX_BATCH_GROUP]i32 = undefined;
    var n: usize = 0;
    for (slots) |slot| {
        if (!emitLlamaToken(sch, slot, slot.llama_next)) continue;
        live_buf[n] = slot;
        seqs[n] = slot.llama_seq.?;
        toks[n] = slot.llama_next;
        n += 1;
    }
    if (n == 0) return;
    const live = live_buf[0..n];
    const ctx = live[0].model.llama_ctx.?;

    const drafts = llamaDraftsThisTick(n, ctx.mtpDrafts(), live[0].max_tokens - live[0].completion_tokens);
    if (drafts > 0) {
        const slot = live[0];
        var buf: [MAX_LLAMA_DRAFTS + 1]i32 = undefined;
        const got = try seqs[0].specStep(toks[0], drafts, llamaSampling(slot), &slot.llama_rng, &buf);
        slot.llama_mtp_rounds += 1;
        slot.llama_mtp_drafted += drafts;
        slot.llama_mtp_accepted += @intCast(got.len - 1);
        for (got[0 .. got.len - 1]) |t| {
            if (!emitLlamaToken(sch, slot, t)) return;
        }
        slot.llama_next = got[got.len - 1];
        return;
    }

    try ctx.step(seqs[0..n], toks[0..n]);
    if (n >= 2 and !llama_batched_logged) {
        llama_batched_logged = true;
        log.info("[batched] llama.cpp decode engaged (seqs={d})\n", .{n});
    }
    for (live, seqs[0..n]) |slot, seq| {
        slot.llama_next = seq.sample(llamaSampling(slot), &slot.llama_rng);
        if (slot.llama_next < 0) slot.markError("llama_sample_failed");
    }
}

var llama_batched_logged = false;

pub const MAX_LLAMA_DRAFTS = policy.max_drafts;
pub const llamaDraftsThisTick = policy.draftsThisTick;

// ── tests ──

test {
    _ = policy;
}
