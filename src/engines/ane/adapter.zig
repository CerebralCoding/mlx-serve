//! The Neural Engine LM engine on the scheduler: load, admission and per-slot runs.

const std = @import("std");
const tokenizer_mod = @import("../../host/tokenizer/tokenizer.zig");
const generate_mod = @import("../mlx/generate.zig");
const model_mod = @import("../mlx/model.zig");
const metrics_mod = @import("../../host/metrics/metrics.zig");
const tokenize_cache_mod = @import("../../host/tokenizer/cache.zig");
const arch_ane = @import("lm.zig");
const log = @import("../../sdk/util/log.zig");
const io_util = @import("../../sdk/util/io_util.zig");
const ModelConfig = model_mod.ModelConfig;
const SamplingParams = generate_mod.SamplingParams;
const dflash_mod = @import("../mlx/spec/dflash/dflash.zig");
const testing = std.testing;

const publishSlotTerminator = @import("../mlx/finish.zig").publishSlotTerminator;
const finishOutcome = @import("../mlx/finish.zig").finishOutcome;
const recordSlotEnd = @import("../mlx/finish.zig").recordSlotEnd;
const Scheduler = @import("../mlx/scheduler.zig").Scheduler;
const Slot = @import("../mlx/slot.zig").Slot;
const makeRoomInRam = @import("../../host/gpu_thread/load_requests.zig").makeRoomInRam;
const drafterFor = @import("../../host/registry/cold_load.zig").drafterFor;
const availForLoad = @import("../../host/registry/preflight.zig").availForLoad;
const memInsufficientForLoad = @import("../../host/registry/preflight.zig").memInsufficientForLoad;
const loadRequirementBytes = @import("../../host/registry/preflight.zig").loadRequirementBytes;
const scheduler_mod = @import("../mlx/scheduler.zig");

/// Whether this load serves on the Neural Engine: the model's `ane` setting, else `--ane`.
pub fn aneWanted(config: *const ModelConfig, flag: bool) bool {
    return config.ane_override orelse flag;
}

/// An MLX pack on the Neural Engine: build its programs (a model's first load on this Mac compiles
/// them, minutes; cached after) and start its worker. Installs the CPU state like the llama arm, but
/// the inference thread's current model stays: the ANE model runs beside it. False when the model
/// cannot run there (by name in the log); the caller then serves it on the GPU.
pub fn doLoadAneOnInferenceThread(sch: *Scheduler, params: anytype) !bool {
    const arch = aneAdmit(sch, params) orelse return false;
    return aneBuild(sch, params, arch);
}

/// The model's Neural Engine layout, its RAM made and its preflight passed; null serves it on the
/// GPU (by name in the log). Inference thread: making room frees MLX models.
pub fn aneAdmit(sch: *Scheduler, params: anytype) ?arch_ane.Arch {
    const arch = arch_ane.archOf(params.config) catch {
        log.warn("[ane] {s} ({s}) does not run on the Neural Engine (dense qwen3_5 and Qwen3 packs only); serving it on the GPU\n", .{ params.model_dir, params.config.model_type });
        return null;
    };
    // Its own preflight: the int8 copies outweigh a 4-bit pack's shards, and the GPU arm checks those.
    const need = arch_ane.loadBytes(arch, params.ane_slots);
    makeRoomInRam(sch, params.entry.id, need);
    if (!scheduler_mod.skip_mem_preflight) {
        const avail = availForLoad(sch.io, need);
        if (memInsufficientForLoad(need, avail)) {
            log.warn("[ane] {s} needs ~{d:.1} GB on the Neural Engine, {d:.1} GB available; serving it on the GPU\n", .{
                params.model_dir,
                @as(f64, @floatFromInt(loadRequirementBytes(need))) / 1_073_741_824.0,
                @as(f64, @floatFromInt(avail)) / 1_073_741_824.0,
            });
            return null;
        }
    }
    return arch;
}

/// Build (or restore) the programs and publish the model; false when the build failed (the GPU
/// then serves it). Never calls MLX, so a cold load runs it on its own thread (`aneBuildThread`).
pub fn aneBuild(sch: *Scheduler, params: anytype, arch: arch_ane.Arch) !bool {
    log.info("[ane] opening {s}\n", .{params.model_dir});
    // The hot prefix cache's switches: its entry cap counts checkpoints here, each a full DeltaNet state.
    const limits: ?arch_ane.CacheLimits = if (params.prefix_cache_capacity > 0 and params.prefix_cache_ram_enabled and params.ssm_checkpoint_stride > 0)
        .{ .max_bytes = params.prefix_cache_mem_bytes, .max_states = params.prefix_cache_capacity, .stride = params.ssm_checkpoint_stride }
    else
        null;
    // The drafter resolves as on the GPU: launch flags, the per-model setting, else the pack's own.
    const chosen_drafter = drafterFor(params.no_drafter, params.drafter_dir, params.config.drafter_override);
    const in_dir_drafter: ?[]u8 = if (chosen_drafter == null) dflash_mod.resolveInDirDrafter(sch.io, sch.allocator, params.model_dir) else null;
    defer if (in_dir_drafter) |d| sch.allocator.free(d);
    const drafter: ?[]const u8 = if (chosen_drafter) |d| (if (d.len > 0) d else null) else in_dir_drafter;
    // The GPU's sampling rule (`installSuppressMask`): reserved specials and padding rows are never drawn.
    const suppress_ids: []u32 = tokenizer_mod.reservedOutputIds(sch.allocator, params.tok.flagged_specials, params.chat_config.chat_template, params.config.eosTokenSlice()) catch &.{};
    defer sch.allocator.free(suppress_ids);
    const defined_vocab: u32 = if (params.config.unpadded_vocab_size > 0) params.config.unpadded_vocab_size else @intCast(params.tok.definedVocabSize());
    const worker = arch_ane.openPool(sch.allocator, sch.io, params.model_dir, arch, .{
        .slots = params.ane_slots,
        .drafter = drafter,
        .progress = &params.entry.load_progress,
        .suppress_ids = suppress_ids,
        .defined_vocab = defined_vocab,
    }, limits, sch, runAneSlot) catch |err| {
        log.err("[ane] {s} not loaded on the Neural Engine ({s}); serving it on the GPU\n", .{ params.model_dir, @errorName(err) });
        return false;
    };
    errdefer worker.close();
    const drafter_path: []const u8 = if (drafter != null and worker.engines[0].drafter != null) try sch.allocator.dupe(u8, drafter.?) else "";
    // The engine's top rung, or less where the pack was trained to less.
    const trained = params.config.max_position_embeddings;
    const max_pos = if (trained > 0) @min(trained, worker.engines[0].model.maxPositions()) else worker.engines[0].model.maxPositions();
    params.config.max_position_embeddings = max_pos;
    // What the engine holds is the context it advertises, whatever `--ctx-size` asks.
    params.config.ctx_override = if (params.config.ctx_override > 0) @min(params.config.ctx_override, max_pos) else max_pos;
    const entry = params.entry;
    entry.ane = worker;
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
    entry.drafter_path = drafter_path;
    entry.prefix_cache = null;
    if (params.tokenize_cache_entries > 0) {
        entry.tokenize_cache = tokenize_cache_mod.TokenizeCache.init(sch.allocator, params.tokenize_cache_entries);
    }
    sch.registry.mutex.lockUncancelable(sch.io);
    sch.registry.markReadyLocked(entry, worker.engines[0].model.weight_bytes + worker.stateBytes());
    sch.registry.mutex.unlock(sch.io);
    return true;
}

/// One request on a Neural Engine worker thread: sync the engine to the prompt, then sample, publish
/// and feed one token at a time. Never calls MLX, so it ends the slot itself instead of `finishSlot`.
fn runAneSlot(ctx: *anyopaque, engine: *arch_ane.Engine, job: *anyopaque) void {
    const sch: *Scheduler = @ptrCast(@alignCast(ctx));
    const slot: *Slot = @ptrCast(@alignCast(job));
    defer _ = slot.in_pass.fetchSub(1, .release);
    runAneRequest(sch, slot, engine) catch |err| {
        if (err == error.Cancelled) return endAneSlot(sch, slot, "cancelled");
        log.err("[ane] request failed: {s}\n", .{@errorName(err)});
        recordAneSlotEnd(sch, slot, .failed);
        slot.markError(@errorName(err));
        return;
    };
    // After the terminator: the next turn of this conversation starts from here.
    engine.checkpoint();
}

fn runAneRequest(sch: *Scheduler, slot: *Slot, engine: *arch_ane.Engine) !void {
    if (slot.cancelled.load(.acquire)) return error.Cancelled;
    var prefill_sw = io_util.Stopwatch.init(sch.io);
    engine.gauges = &sch.ane_prefill_gauges;
    _ = sch.requests_prefilling.fetchAdd(1, .monotonic);
    const synced = synced: {
        defer {
            engine.ungauge();
            _ = sch.requests_prefilling.fetchSub(1, .monotonic);
        }
        break :synced try engine.sync(slot.full_prompt, &slot.cancelled);
    };
    slot.prompt_tokens = @intCast(slot.full_prompt.len);
    slot.cached_tokens = @intCast(synced.cached);
    slot.prefill_ns = prefill_sw.read();
    slot.first_token_ns = @intCast(slot.request_start_ts.untilNow(sch.io, .boot).nanoseconds);
    slot.state = .decoding;
    const seed = slot.sampling.seed orelse blk: {
        var r: u64 = undefined;
        sch.io.random(std.mem.asBytes(&r));
        break :blk r;
    };
    var prng = std.Random.DefaultPrng.init(seed);
    const sampling: arch_ane.Sampling = .{
        .temperature = slot.sampling.temperature,
        .top_k = slot.sampling.top_k,
        .top_p = slot.sampling.top_p,
        .min_p = slot.sampling.min_p orelse 0,
    };
    var decode_sw = io_util.Stopwatch.init(sch.io);
    var tok = engine.draw(synced.logits, sampling, prng.random());
    var emitted = false;
    // DFlash2: each cycle commits `tok` plus the accepted drafts and yields the drafts and the next token.
    if (engine.drafter != null) {
        const drafts: usize = arch_ane.P - 1;
        var cycles: usize = 0;
        var accepted: usize = 0;
        defer if (cycles > 0) log.info("[ane] spec: {d} cycles, {d}% of drafts accepted, {d:.2} tokens per call\n", .{ cycles, 100 * accepted / (drafts * cycles), @as(f64, @floatFromInt(accepted + cycles)) / @as(f64, @floatFromInt(cycles)) });
        if (aneEmit(sch, slot, tok)) |reason| return endAneSlot(sch, slot, reason);
        emitted = true;
        while (true) {
            const c = engine.cycle(tok, sampling, prng.random(), slot.eos_token_ids) catch |err| switch (err) {
                error.ContextFull => break, // the last positions take one token per call
                else => return err,
            };
            cycles += 1;
            accepted += c.accepted;
            slot.decode_ns = decode_sw.read();
            for (c.tokens[0..c.n]) |t| if (aneEmit(sch, slot, t)) |reason| return endAneSlot(sch, slot, reason);
            tok = c.tokens[c.n - 1];
        }
    }
    while (true) {
        if (!emitted) if (aneEmit(sch, slot, tok)) |reason| return endAneSlot(sch, slot, reason);
        emitted = false;
        const logits = engine.step(tok) catch |err| switch (err) {
            error.ContextFull => return endAneSlot(sch, slot, "length"),
            else => return err,
        };
        slot.decode_ns = decode_sw.read();
        tok = engine.draw(logits, sampling, prng.random());
    }
}

/// Whether a Neural Engine model honors the request: its engine samples the raw logits, so a grammar,
/// penalties, logprobs or a forced tool call would be ignored. A thinking budget is not enforced.
pub fn aneServes(sampling: SamplingParams, logprobs_n: u32, prompt_logprobs: bool) bool {
    return !sampling.shapesLogits() and sampling.tool_force == null and logprobs_n == 0 and !prompt_logprobs;
}

/// Publish one generated token; the finish reason when the slot ends here.
fn aneEmit(sch: *Scheduler, slot: *Slot, tok: u32) ?[]const u8 {
    if (generate_mod.isEosId(tok, slot.eos_token_ids)) return "stop";
    slot.pushToken(tok);
    if (tok != 0) slot.was_pad_only = false;
    slot.completion_tokens += 1;
    _ = sch.ane_generated.fetchAdd(1, .monotonic);
    if (slot.completion_tokens >= slot.max_tokens) return "length";
    if (slot.cancelled.load(.acquire)) return "cancelled";
    return null;
}

/// Its generated tokens move from the live count to the request totals in one go.
fn recordAneSlotEnd(sch: *Scheduler, slot: *Slot, outcome: metrics_mod.Outcome) void {
    _ = sch.ane_generated.fetchSub(slot.completion_tokens, .monotonic);
    recordSlotEnd(sch.metrics, slot, outcome);
}

/// Metrics first: once the terminator is out, the connection thread may complete the slot.
fn endAneSlot(sch: *Scheduler, slot: *Slot, reason: []const u8) void {
    recordAneSlotEnd(sch, slot, finishOutcome(reason, null));
    publishSlotTerminator(slot, reason, null);
}

// ── tests ──

test "aneServes: plain sampling only; a grammar, a penalty, logprobs or a forced tool call is refused" {
    try testing.expect(aneServes(.{ .temperature = 0.7, .top_p = 0.9, .top_k = 20 }, 0, false));
    try testing.expect(!aneServes(.{ .repeat_penalty = 1.1 }, 0, false));
    try testing.expect(!aneServes(.{ .presence_penalty = 0.5 }, 0, false));
    try testing.expect(!aneServes(.{ .constraint = @ptrFromInt(8) }, 0, false));
    try testing.expect(!aneServes(.{ .tool_force = @ptrFromInt(8) }, 0, false));
    try testing.expect(!aneServes(.{}, 5, false));
    try testing.expect(!aneServes(.{}, 0, true));
    // A thinking budget is not enforced, but the answer is still the model's own.
    try testing.expect(aneServes(.{ .think_bound = @ptrFromInt(8) }, 0, false));
}
