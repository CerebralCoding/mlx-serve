//! The ds4 engine on the scheduler: GGUF stub state, load, prefill and decode tick.

const std = @import("std");
const tokenizer_mod = @import("../../host/tokenizer/tokenizer.zig");
const generate_mod = @import("../mlx/generate.zig");
const model_mod = @import("../mlx/model.zig");
const chat_mod = @import("../../host/chat/chat.zig");
const tokenize_cache_mod = @import("../../host/tokenizer/cache.zig");
const gguf_meta = @import("../../host/registry/engine_pick.zig");
const arch_ds4 = if (@import("build_options").ds4_engine) @import("bridge.zig") else @import("stub.zig");
const log = @import("../../sdk/util/log.zig");
const load = @import("load.zig");
const round = @import("round.zig");
const ModelConfig = model_mod.ModelConfig;
const Tokenizer = tokenizer_mod.Tokenizer;
const ChatConfig = chat_mod.ChatConfig;
const testing = std.testing;

const publishHotCacheResidency = @import("../mlx/cache/budget.zig").publishHotCacheResidency;
const finishSlot = @import("../mlx/finish.zig").finishSlot;
const Scheduler = @import("../mlx/scheduler.zig").Scheduler;
const Slot = @import("../mlx/slot.zig").Slot;
const CpuState = @import("../../host/registry/cpu_state.zig").CpuState;
const freeCpuState = @import("../../host/registry/cpu_state.zig").freeCpuState;
const scheduler_mod = @import("../mlx/scheduler.zig");

/// Stub CPU state for a GGUF cold load. Mirrors the stubs main.zig's
/// `runLlamaServe`/`runDs4Serve` build for the startup path: the embedded
/// engine owns the real tokenizer + chat template (the llama arm adopts the
/// GGUF's embedded template at load), so only `model_type` (echoed in
/// /v1/models + engine-arm dispatch guards) and `max_position_embeddings`
/// (session sizing in runPrefillLlama/runPrefillDs4 + the server's context
/// guard) matter. `ctx_size` is the --ctx-size launch flag; 0 → the same
/// defaults the startup paths use (8192 for llama — the GGUF's trained
/// context isn't readable until the engine opens; ds4's own default via
/// clampSessionCtx).
pub fn buildGgufStubCpuState(allocator: std.mem.Allocator, engine: gguf_meta.Engine, ctx_size: u32) !CpuState {
    const config = try allocator.create(ModelConfig);
    errdefer allocator.destroy(config);
    config.* = switch (engine) {
        .llama => ModelConfig{
            .model_type = "gguf",
            .weight_prefix = "model",
            .head_dim = 128,
            .max_position_embeddings = if (ctx_size > 0) ctx_size else 8192,
            .is_encoder_only = false,
        },
        .ds4 => ModelConfig{
            .model_type = "deepseek_v4",
            .weight_prefix = "model",
            .num_hidden_layers = 61,
            .hidden_size = 7168,
            .head_dim = 128,
            .num_attention_heads = 56,
            .num_key_value_heads = 56,
            .max_position_embeddings = arch_ds4.clampSessionCtx(ctx_size),
            .is_encoder_only = false,
        },
    };

    const tok = try allocator.create(Tokenizer);
    errdefer allocator.destroy(tok);
    var byte_map: [256]u21 = undefined;
    for (0..256) |b| byte_map[b] = @intCast(b);
    tok.* = .{
        .vocab = std.StringHashMap(u32).init(allocator),
        .id_to_token = std.AutoHashMap(u32, []const u8).init(allocator),
        .merge_ranks = @TypeOf(tok.merge_ranks).init(allocator),
        .allocator = allocator,
        .special_tokens = std.StringHashMap(u32).init(allocator),
        .tok_type = .byte_level_bpe,
        .byte_to_unicode = byte_map,
        .unicode_to_byte = std.AutoHashMap(u21, u8).init(allocator),
        .bos_id = null,
        .eos_id = null,
        .parsed_json = null,
    };
    errdefer tok.deinit();

    const cc = try allocator.create(ChatConfig);
    errdefer allocator.destroy(cc);
    cc.* = .{
        .chat_template = try allocator.dupe(u8, ""),
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };

    return .{ .config = config, .tok = tok, .chat_config = cc };
}

/// ds4 load on the inference thread. Mirrors the MLX path's "open weights
/// → install on entry → markReady" shape but works exclusively through the
/// embedded engine. The stub `config`/`tok`/`chat_config` come in via
/// `params.config`/`params.tok`/`params.chat_config` (main.zig allocates
/// them); the entry takes ownership.
pub fn doLoadDs4OnInferenceThread(sch: *Scheduler, params: anytype) !void {
    const engine = try load.open(arch_ds4, sch.io, sch.allocator, params.ds4_path, .{
        .mtp = params.ds4_mtp,
        .dspark = params.ds4_dspark,
        .ssd_streaming = params.ds4_ssd_streaming,
        .prefill_chunk = generate_mod.explicitPrefillChunkU32(),
    });
    errdefer engine.close();
    // draft_tokens is the spec-readiness signal for BOTH support kinds
    // (legacy MTP count, or DSpark block size when the runtime is armed).
    log.info("[ds4] engine ready (EOS={d}, has_mtp={}, draft_tokens={d})\n", .{ engine.eosToken(), engine.hasMtp(), engine.mtpDraftTokens() });

    // Make sure the stub config knows about the engine's EOS token so the
    // streaming/non-streaming paths' EOS check fires correctly. addEosToken
    // is a no-op if the slot is already present.
    const eos_id: u32 = @intCast(engine.eosToken());
    params.config.addEosToken(eos_id);

    // ── Install on entry. Everything below must be infallible (mirrors
    //    the MLX path's invariant about the per-ptr errdefers above).
    const entry = params.entry;
    entry.ds4_engine = engine;
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
    // Iteration 2: tokenize cache also applies on the ds4 path. The
    // MLX-branch assignment isn't reached here because we early-return.
    if (params.tokenize_cache_entries > 0) {
        entry.tokenize_cache = tokenize_cache_mod.TokenizeCache.init(
            sch.allocator,
            params.tokenize_cache_entries,
        );
    }

    // Bytes-resident is whatever main.zig handed us (typically the GGUF
    // on-disk size). ds4's `ds4_context_memory_estimate` could give a
    // tighter number; we leave that as a TODO since the registry's
    // eviction gate doesn't currently support multi-engine residency.
    const bytes_resident: u64 = if (entry.bytes_on_disk) |b| b else 0;

    sch.registry.mutex.lockUncancelable(sch.io);
    sch.registry.markReadyLocked(entry, bytes_resident);
    sch.registry.mutex.unlock(sch.io);

    // Scheduler's borrowed views: leave the MLX fields null. `runPrefill`
    // / `runSingleDecodeTick` branch on `slot.ds4_session` and never touch
    // `sch.xfm` for ds4 slots.
    sch.current_model = entry;
    sch.xfm = null;
    sch.weights = null;
    sch.vision_encoder = null;
    sch.drafter = null;
    sch.dflash = null;
    publishHotCacheResidency(sch);
    if (scheduler_mod.hot_cache_budget_invalidate) |f| f();
}

/// ds4 prefill: sync the model's ONE persistent session to the full prompt.
/// ds4 keeps the common prefix against the session's live KV (the previous
/// request's prompt + reply) and rebuilds only the tail, so the mlx-serve hot
/// prefix cache stays out of the picture. `cached_tokens` is that prefix.
pub fn runPrefillDs4(sch: *Scheduler, slot: *Slot, engine: *arch_ds4.Ds4Engine) !void {
    _ = sch;
    // Convert the slot's u32 prompt to ds4's i32 view. Sized once per
    // prefill — ds4's session_sync owns the read of these IDs and the
    // buffer can be freed before decode.
    const i32_prompt = try slot.allocator.alloc(i32, slot.full_prompt.len);
    defer slot.allocator.free(i32_prompt);
    for (slot.full_prompt, 0..) |t, i| i32_prompt[i] = @intCast(t);

    // Session ctx from the user's --ctx-size, which runDs4Serve carries on the
    // stub config's max_position_embeddings. Floored at ds4's prefill chunk so
    // an under-sized ctx can't drop into the junk-output regime; 0/unset →
    // ds4's default. Larger ctx → larger KV scratch up front.
    const req_ctx: u32 = if (slot.model.config) |c| c.max_position_embeddings else 0;
    const ctx_size: i32 = @intCast(arch_ds4.clampSessionCtx(req_ctx));
    const sess = slot.model.ds4_session orelse blk: {
        const created = try engine.createSession(ctx_size);
        slot.model.ds4_session = created;
        break :blk created;
    };

    const cached = sess.commonPrefix(i32_prompt) catch 0;
    sess.sync(i32_prompt) catch |err| {
        // Leave nothing half-built behind for the next request.
        sess.invalidate();
        return err;
    };

    slot.ds4_session = sess;
    slot.prompt_tokens = @intCast(slot.full_prompt.len);
    slot.cached_tokens = @intCast(@max(cached, 0));
    slot.state = .decoding;
}

/// ds4 decode tick: argmax (temp ≤ 0) or sample, check EOS, push token,
/// `eval(token)` to extend the session, and stop on max_tokens. Each call
/// emits exactly one token (unlike PLD/drafter which can emit several).
pub fn runDs4DecodeTick(sch: *Scheduler, slot: *Slot, session: *arch_ds4.Ds4Session) !void {
    const engine = slot.model.ds4_engine.?;
    const d: round.Draw = .{
        .temperature = slot.sampling.temperature,
        .top_k = slot.sampling.top_k,
        .top_p = slot.sampling.top_p,
        .min_p = slot.sampling.min_p orelse 0.05,
        .rng = &slot.ds4_rng,
    };
    const next_id = round.draw(session, d);

    // ds4 returns an i32 token id; we treat any negative value as a sampler
    // failure rather than push it through the unsigned ring buffer.
    if (next_id < 0) {
        slot.markError("ds4_sample_failed");
        return;
    }
    const tok_u32: u32 = @intCast(next_id);

    // EOS handling — match the MLX path: do NOT emit the stop token.
    if (engine.isStop(next_id) or generate_mod.isEosId(tok_u32, slot.eos_token_ids)) {
        finishSlot(sch, slot, "stop");
        return;
    }

    // MTP speculative decode: ONE call commits the sampled token AND drafts +
    // verifies several more, advancing ds4's KV internally (no separate eval).
    // It emits `[sampled, accepted…]`, so this tick may push several tokens.
    // Mirrors ds4's own CLI loop; engages only under greedy sampling.
    if (round.shouldSpeculate(engine.mtpDraftTokens(), d.temperature, engine.embedded_mtp)) {
        var spec_buf: [round.max_round_tokens]i32 = undefined;
        const got = round.speculate(engine, session, next_id, d, slot.max_tokens -| slot.completion_tokens, &spec_buf) catch {
            session.invalidate();
            slot.markError("ds4_spec_failed");
            return;
        };
        for (got) |t| {
            const t_u32: u32 = @intCast(t);
            // EOS may appear mid-batch — stop, and never emit it.
            if (engine.isStop(t) or generate_mod.isEosId(t_u32, slot.eos_token_ids)) {
                finishSlot(sch, slot, "stop");
                return;
            }
            slot.pushToken(t_u32);
            if (t_u32 != 0) slot.was_pad_only = false;
            slot.completion_tokens += 1;
            if (slot.completion_tokens >= slot.max_tokens) {
                finishSlot(sch, slot, "length");
                return;
            }
        }
        return;
    }

    slot.pushToken(tok_u32);
    if (tok_u32 != 0) slot.was_pad_only = false;
    slot.completion_tokens += 1;

    // Advance ds4's KV by feeding the freshly-sampled token. After this
    // the session is in the state expected by the NEXT decode tick.
    session.eval(next_id) catch |err| {
        session.invalidate();
        return err;
    };

    if (slot.completion_tokens >= slot.max_tokens) {
        finishSlot(sch, slot, "length");
        return;
    }
}

// ── tests ──

test "buildGgufStubCpuState: llama stub carries gguf model_type + ctx sizing" {
    const a = std.testing.allocator;
    // No --ctx-size → the same 8192 default runLlamaServe uses (the GGUF's
    // trained context isn't readable until the engine opens).
    var s = try buildGgufStubCpuState(a, .llama, 0);
    defer freeCpuState(a, &s);
    try testing.expectEqualStrings("gguf", s.config.model_type);
    try testing.expectEqual(@as(u32, 8192), s.config.max_position_embeddings);
    try testing.expect(s.gguf == null); // route is attached by preloadGgufCpuState
    try testing.expectEqual(@as(usize, 0), s.chat_config.chat_template.len);

    // Explicit --ctx-size flows through to session sizing + context guard.
    var s2 = try buildGgufStubCpuState(a, .llama, 4096);
    defer freeCpuState(a, &s2);
    try testing.expectEqual(@as(u32, 4096), s2.config.max_position_embeddings);
}

test "buildGgufStubCpuState: ds4 stub carries deepseek_v4 + clamped ctx" {
    const a = std.testing.allocator;
    // Under-sized ctx floors at ds4's prefill chunk, exactly like runDs4Serve.
    var s = try buildGgufStubCpuState(a, .ds4, 512);
    defer freeCpuState(a, &s);
    try testing.expectEqualStrings("deepseek_v4", s.config.model_type);
    try testing.expectEqual(arch_ds4.clampSessionCtx(512), s.config.max_position_embeddings);
}
