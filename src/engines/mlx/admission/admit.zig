//! AdmissionBill, fitAfterUnloadingIdle; prefillAdmissionBill, verdict/log, checkAttentionMemory

const std = @import("std");
const builtin = @import("builtin");
const mlx = @import("../../../mlx/ffi.zig");
const transformer_mod = @import("../transformer.zig");
const generate_mod = @import("../generate.zig");
const model_mod = @import("../model.zig");
const log = @import("../../../host/log.zig");
const scheduler_mod = @import("../scheduler.zig");
const model_registry_mod = @import("../../../host/registry/registry.zig");
const metrics = @import("../../../host/status.zig");
const LoadedModel = model_registry_mod.LoadedModel;
const http_server = @import("../../../host/http/server.zig");

const Conn = @import("../../../host/http/conn.zig").Conn;
const configuredKvQuantFor = @import("../../../host/http/server.zig").configuredKvQuantFor;
const currentGpuMemoryCeiling = @import("../../../host/registry/memory_ceiling.zig").currentGpuMemoryCeiling;
const explicitPrefillChunk = @import("prefill_chunk.zig").explicitPrefillChunk;
const qsaScoreFusedOffGuard = @import("bills_test.zig").qsaScoreFusedOffGuard;
const WarmPrefix = @import("bills.zig").WarmPrefix;
const mlxMemoryGuardApplies = @import("bills.zig").mlxMemoryGuardApplies;
const memoryRefusalMessage = @import("../../../host/http/errors.zig").memoryRefusalMessage;
const prefillNeededAtChunk = @import("prefill_chunk.zig").prefillNeededAtChunk;
const perRequestPrefillChunkEnabled = @import("prefill_chunk.zig").perRequestPrefillChunkEnabled;
const chooseRequestPrefillChunk = @import("prefill_chunk.zig").chooseRequestPrefillChunk;
const countRejected = @import("../../../host/http/errors.zig").countRejected;
const sendErrorResponse = @import("../../../host/http/errors.zig").sendErrorResponse;
const sendAnthropicError = @import("../../../host/api/anthropic.zig").sendAnthropicError;
const qwen4ExpOomConfig = @import("bills_test.zig").qwen4ExpOomConfig;

/// The admission bill and the headroom it is compared against, so the connection thread's
/// guard and the inference thread's evict-or-refuse decision cannot compute them differently.
pub const AdmissionBill = struct {
    needed: u64 = 0,
    available: u64 = 0,
    /// Bytes the hot prefix cache is holding. Reported in a refusal, never credited (see `reclaimable`).
    evictable: u64 = 0,
    /// The part of `evictable` an eviction can prove it will return: residency minus the
    /// largest entry (a restore pins at most one).
    reclaimable: u64 = 0,
    /// The prefill width this bill was taken at, so a refusal can name it.
    chunk: u64 = 0,

    pub fn fits(self: AdmissionBill) bool {
        return self.needed <= self.available;
    }

    /// The one credit the decision and both sentences read: `reclaimable`, never `evictable`
    /// (a refusal once quoted the slot's own checked-out prefix as "evicted").
    pub fn evictionCredit(self: AdmissionBill) u64 {
        return self.reclaimable;
    }

    /// Fits once the cache has given back what it provably can. Credits `reclaimable`, not
    /// `evictable`: a 383k prompt was admitted on one entry it then restored from and refused
    /// after the eviction pass returned nothing.
    pub fn fitsAfterEviction(self: AdmissionBill) bool {
        return self.needed <= self.available +| self.evictionCredit();
    }
};

/// Unloads idle resident models, least recently used first, until `fits` holds; false when
/// nothing idle is left. An idle model is memory its next request reloads.
pub fn fitAfterUnloadingIdle(ctx: anytype, fits: fn (@TypeOf(ctx)) bool, unloadOne: fn (@TypeOf(ctx)) bool) bool {
    while (!fits(ctx)) if (!unloadOne(ctx)) return false;
    return true;
}

pub fn prefillAdmissionBill(config: *const model_mod.ModelConfig, prompt_len: usize, max_tokens: u32, kv_override: ?transformer_mod.KVQuantConfig, unchunked_prefill: bool, prompt_tokens: ?[]const u32, warm: WarmPrefix) AdmissionBill {
    const heads = config.num_attention_heads;
    if (heads == 0) return .{ .needed = 0, .available = std.math.maxInt(u64) };
    const seq: u64 = @intCast(prompt_len);

    const kv_cfg: transformer_mod.KVQuantConfig = kv_override orelse configuredKvQuantFor(config);
    const kv_bits: u64 = if (kv_cfg.scheme == .off) 16 else kv_cfg.bits;
    // Available = the lesser of Metal's static working-set max and what is physically reachable
    // now (#64). Read before the width is chosen: on a per-request arch the width is a function of it.
    var active_mem: usize = 0;
    _ = mlx.mlx_get_active_memory(&active_mem);
    const total_limit: u64 = currentGpuMemoryCeiling(config, active_mem);
    const available = if (total_limit > active_mem) total_limit - active_mem else 0;

    // A vision prefill chunks like text since issue #197 (the splice resumes
    // its row index across chunks), so it bills the chunk-bounded envelope —
    // UNLESS the MLX_SERVE_VISION_CHUNKED=0 kill switch restored the
    // whole-prompt forward, in which case `unchunked_prefill` bills the real
    // width. Call sites pass generate_mod.visionPrefillUnchunked(has_vision)
    // so the guard and the prefill loop cannot disagree.
    //
    // Otherwise the width comes from `chooseRequestPrefillChunk`, the same rule the scheduler
    // pins with. An ungated arch (or an explicit `--prefill-chunk`) gets the load-time pin.
    const chosen: u32 = if (unchunked_prefill) 0 else chooseRequestPrefillChunk(
        config,
        seq,
        max_tokens,
        kv_bits,
        available,
        config.pinned_prefill_chunk,
        explicitPrefillChunk(),
        warm,
    );
    const chunk: u64 = if (unchunked_prefill)
        @max(seq, 1)
    else
        @intCast(generate_mod.effectivePrefillChunk(config.prefillScoreHeadDim(), config.num_attention_heads, prompt_len, config.has_sliding_window, config.isMoe(), config.longCtxGated(), chosen));
    // RAM hot-cache restores rebind MLX array handles by refcount; they do not
    // allocate another copy of the cached buffers. `active_mem` above already
    // includes the resident entry, while `prefillMemoryNeeded` bills the full
    // destination KV capacity that may be allocated when the restored cache
    // grows. Adding the resident entry here again would invent a third copy
    // and reject long warm prompts (and even cache misses) spuriously.
    const needed: u64 = prefillNeededAtChunk(config, seq, max_tokens, kv_bits, chunk, warm);
    // Published number, never the pointer: `hot_prefix_cache` is inference-thread state, freed
    // on every model switch. Stale by at most one entry, which is all a hint has to be.
    const evictable: u64 = if (http_server.global_scheduler) |sch|
        sch.resident_hot_cache_bytes.load(.monotonic)
    else
        0;
    // Prompt-aware when the caller has the ids: an entry this prompt could not restore from is
    // genuinely evictable. Still a published snapshot, never the cache pointer.
    const reclaimable: u64 = if (http_server.global_scheduler) |sch|
        if (prompt_tokens) |toks|
            scheduler_mod.reclaimableHotCacheBytesFor(sch, toks)
        else
            sch.reclaimable_hot_cache_bytes.load(.monotonic)
    else
        0;
    return .{ .needed = needed, .available = available, .evictable = evictable, .reclaimable = reclaimable, .chunk = chunk };
}

/// Bytes of the resident hot cache this prompt pins: residency less what an eviction can prove
/// it will return. Non-zero means an entry this prompt would restore from is holding memory
/// that neither goes back nor has to be allocated again; the connection thread cannot price
/// how much, so it defers to the inference thread's `WarmPrefix` bill.
pub fn pinnedResidentBytes(bill: AdmissionBill) u64 {
    return bill.evictable -| bill.evictionCredit();
}

/// What the admission guard decided, from one bill so the log line and the arm agree: fits
/// outright, fits once the hot cache gives back what it provably can, or does not fit.
pub const AdmissionVerdict = enum {
    admit,
    evict,
    refuse,

    pub fn name(self: AdmissionVerdict) []const u8 {
        return switch (self) {
            .admit => "admit",
            .evict => "evict",
            .refuse => "refuse",
        };
    }
};

pub fn admissionVerdict(bill: AdmissionBill) AdmissionVerdict {
    if (bill.fits()) return .admit;
    if (bill.fitsAfterEviction()) return .evict;
    return .refuse;
}

/// The generation of `markReadyLocked` the last info-level admission line was emitted for.
var admission_logged_generation = std.atomic.Value(u64).init(std.math.maxInt(u64));

/// Has a model become resident since the last time this returned true? Consuming, so N racing connection threads produce one line.
fn firstRequestAfterLoad() bool {
    const gen = model_registry_mod.load_generation.load(.monotonic);
    const last = admission_logged_generation.load(.monotonic);
    if (gen == last) return false;
    return admission_logged_generation.cmpxchgStrong(last, gen, .monotonic, .monotonic) == null;
}

/// The level one admission line is emitted at: info on the first request after a load and
/// whenever the bill is within 10% of what is available; debug otherwise.
pub fn admissionLogLevel(needed: u64, available: u64, first_after_load: bool) log.Level {
    if (first_after_load) return .info;
    // `needed > 0.9 * available` without floating point or overflow.
    if (needed *| 10 > available *| 9) return .info;
    return .debug;
}

/// One line per request at the admission decision, on admitted requests too. Every field comes
/// off the same `bill` the arms act on. The level check runs first.
fn logAdmissionDecision(bill: AdmissionBill) void {
    // `info` is the loudest level this line is emitted at, and `firstRequestAfterLoad` consumes its token.
    if (!log.enabled(.info)) return;
    const level = admissionLogLevel(bill.needed, bill.available, firstRequestAfterLoad());
    if (!log.enabled(level)) return;
    const mb = 1024 * 1024;
    log.atLevel(level, "[admission] needed={d} MB available={d} MB reclaimable={d} MB width={d} verdict={s}\n", .{
        bill.needed / mb,
        bill.available / mb,
        bill.evictionCredit() / mb,
        bill.chunk,
        admissionVerdict(bill).name(),
    });
}

/// Inference-thread predicate behind the scheduler's admission hook: does this request fit
/// right now, with live memory re-read? Every input is forwarded, none defaulted (a hardcoded
/// `kv_override = null` priced a `kv_quant: 4` request at fp16 and evicted a cache for nothing).
pub fn prefillFitsNow(config: *const model_mod.ModelConfig, prompt_len: usize, max_tokens: u32, kv_cfg: transformer_mod.KVQuantConfig, unchunked_prefill: bool, warm_matched: u64, warm_capacity: u64, warm_will_donate: bool, enable_mtp: bool) bool {
    return prefillAdmissionBill(config, prompt_len, max_tokens, kv_cfg, unchunked_prefill, null, .{
        .matched_tokens = warm_matched,
        .capacity_tokens = warm_capacity,
        .will_donate = warm_will_donate,
        .mtp_on = enable_mtp,
    }).fits();
}

/// {needed, available} of the cold bill, live memory re-read, for the scheduler's sibling ledger.
pub fn prefillBillNumbersNow(config: *const model_mod.ModelConfig, prompt_len: usize, max_tokens: u32, kv_cfg: transformer_mod.KVQuantConfig, unchunked_prefill: bool, enable_mtp: bool) [2]u64 {
    const bill = prefillAdmissionBill(config, prompt_len, max_tokens, kv_cfg, unchunked_prefill, null, .{ .mtp_on = enable_mtp });
    return .{ bill.needed, bill.available };
}

/// The inference thread's refusal, quoting the numbers it compared; returns {needed, available} for the client.
pub fn logPrefillRefusal(config: *const model_mod.ModelConfig, prompt_len: usize, max_tokens: u32, kv_cfg: transformer_mod.KVQuantConfig, unchunked_prefill: bool, warm_matched: u64, warm_capacity: u64, warm_will_donate: bool, enable_mtp: bool) [2]u64 {
    // The same warm inputs the probe was refused on, the checkout decision included.
    const bill = prefillAdmissionBill(config, prompt_len, max_tokens, kv_cfg, unchunked_prefill, null, .{
        .matched_tokens = warm_matched,
        .capacity_tokens = warm_capacity,
        .will_donate = warm_will_donate,
        .mtp_on = enable_mtp,
    });
    const mb = 1024 * 1024;
    // "Narrowest width tried" is only true where the ladder was walked.
    const width_note: []const u8 = if (perRequestPrefillChunkEnabled(config)) "the narrowest width tried" else "the pinned width";
    // The comparison first (`fitsAfterEviction` spelled out), then the residency as context, with
    // the part eviction cannot take named.
    log.warn("  prompt {d} tokens needs ~{d}MB at prefill chunk {d} ({s}), ~{d}MB available + ~{d}MB the hot cache can give back (~{d}MB resident, ~{d}MB of it pinned by this request's own prefix) — refused before prefill\n", .{
        prompt_len,
        bill.needed / mb,
        bill.chunk,
        width_note,
        bill.available / mb,
        bill.evictionCredit() / mb,
        bill.evictable / mb,
        pinnedResidentBytes(bill) / mb,
    });
    return .{ bill.needed, bill.available };
}

pub fn checkAttentionMemory(allocator: std.mem.Allocator, stream: *Conn, prompt_ids: []const u32, max_tokens: u32, config: *const model_mod.ModelConfig, is_anthropic: bool, kv_override: ?transformer_mod.KVQuantConfig, lm: *const LoadedModel, unchunked_prefill: bool, enable_mtp: bool, media_bytes: u64) !bool {
    const prompt_len: usize = prompt_ids.len;
    if (!mlxMemoryGuardApplies(lm.engineBacked())) return true;
    if (config.num_attention_heads == 0) return true; // unknown architecture, skip check
    // The connection thread has no slot and no cache: it bills cold and defers a warm prompt.
    var bill = prefillAdmissionBill(config, prompt_len, max_tokens, kv_override, unchunked_prefill, prompt_ids, .{ .mtp_on = enable_mtp });
    // The request's media rows are one more tensor the prefill allocates.
    bill.needed += media_bytes;
    logAdmissionDecision(bill);
    const needed = bill.needed;
    const available = bill.available;
    // Evict to admit (#353): when the bill only fails because the cache is holding memory, admit;
    // the inference thread evicts LRU entries until it fits and refuses by name only if it still does not.
    if (needed > available and bill.fitsAfterEviction()) {
        log.info("  prompt {d} tokens needs ~{d}MB, ~{d}MB available + ~{d}MB reclaimable hot cache (of ~{d}MB resident) — admitting, the prefill will evict\n", .{
            prompt_len, needed / (1024 * 1024), available / (1024 * 1024), bill.evictionCredit() / (1024 * 1024), bill.evictable / (1024 * 1024),
        });
        return true;
    }
    // Warm deferral. The bill above is cold; a non-zero `pinnedResidentBytes` means an entry
    // this prompt would restore from is holding bytes that will neither be allocated nor
    // reclaimed. How many is knowable only on the inference thread, which bills with `WarmPrefix`.
    if (needed > available and pinnedResidentBytes(bill) > 0) {
        log.info("  prompt {d} tokens needs ~{d}MB cold, ~{d}MB available, and ~{d}MB of hot cache is a prefix this prompt would SHARE — deferring to the prefill, which bills the warm request\n", .{
            prompt_len, needed / (1024 * 1024), available / (1024 * 1024), pinnedResidentBytes(bill) / (1024 * 1024),
        });
        return true;
    }
    if (needed > available) if (http_server.global_scheduler) |sch| {
        const Rebill = struct {
            sch: *scheduler_mod.Scheduler,
            keep: []const u8,
            bill: *AdmissionBill,
            config: *const model_mod.ModelConfig,
            prompt_ids: []const u32,
            max_tokens: u32,
            kv_override: ?transformer_mod.KVQuantConfig,
            unchunked: bool,
            enable_mtp: bool,
            media_bytes: u64,
            fn fits(r: *@This()) bool {
                return r.bill.fits();
            }
            fn unloadOne(r: *@This()) bool {
                const id = r.sch.idleModelToUnload(r.keep) orelse return false;
                log.info("  prompt {d} tokens does not fit — unloading idle model {s} first\n", .{ r.prompt_ids.len, id });
                r.sch.unloadModelIfIdle(id, 0) catch return false;
                r.bill.* = prefillAdmissionBill(r.config, r.prompt_ids.len, r.max_tokens, r.kv_override, r.unchunked, r.prompt_ids, .{ .mtp_on = r.enable_mtp });
                r.bill.needed += r.media_bytes;
                return true;
            }
        };
        var rebill = Rebill{ .sch = sch, .keep = lm.id, .bill = &bill, .config = config, .prompt_ids = prompt_ids, .max_tokens = max_tokens, .kv_override = kv_override, .unchunked = unchunked_prefill, .enable_mtp = enable_mtp, .media_bytes = media_bytes };
        if (fitAfterUnloadingIdle(&rebill, Rebill.fits, Rebill.unloadOne)) return true;
    };
    if (bill.needed > bill.available) {
        const needed_mb = bill.needed / (1024 * 1024);
        const avail_mb = bill.available / (1024 * 1024);
        log.warn("  prompt {d} tokens needs ~{d}MB (KV+working+margin) at prefill chunk {d}, ~{d}MB available + ~{d}MB reclaimable of ~{d}MB resident hot cache — rejecting\n", .{ prompt_len, needed_mb, bill.chunk, avail_mb, bill.evictionCredit() / (1024 * 1024), bill.evictable / (1024 * 1024) });
        // A refusal quotes the numbers it compared, hot cache included; everything the cache holds
        // is evictable here by construction (the withheld case took the deferral arm).
        const msg = try memoryRefusalMessage(allocator, prompt_len, needed_mb, avail_mb, bill);
        defer allocator.free(msg);
        countRejected();
        if (is_anthropic) {
            try sendAnthropicError(allocator, stream, "invalid_request_error", msg, 400);
        } else {
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", msg, 400);
        }
        return false;
    }
    return true;
}

pub extern "c" fn sysctlbyname(name: [*:0]const u8, oldp: ?*anyopaque, oldlenp: ?*usize, newp: ?*const anyopaque, newlen: usize) c_int;

/// Get the max buffer allocation limit (~75% of system unified memory).
/// `hw.memsize` on Darwin; on Linux there is no unified-memory sysctl, so the
/// same 75%-of-physical-RAM heuristic runs against /proc/meminfo.
pub fn getMetalBufferLimit() u64 {
    var mem: u64 = 0;
    if (comptime builtin.os.tag.isDarwin()) {
        var len: usize = @sizeOf(u64);
        _ = sysctlbyname("hw.memsize", @ptrCast(&mem), &len, null, 0);
    } else {
        mem = metrics.getTotalMemBytes();
    }
    if (mem == 0) return 8 * 1024 * 1024 * 1024; // fallback 8GB
    return mem * 75 / 100;
}

// ── tests ──

test "a prompt that does not fit unloads idle models, oldest first, until it fits" {
    const t = std.testing;
    const Fake = struct {
        needed: u64,
        available: u64,
        idle: []const u64, // resident bytes, least recently used first
        unloaded: usize = 0,
        fn fits(f: *@This()) bool {
            return f.needed <= f.available;
        }
        fn unloadOne(f: *@This()) bool {
            if (f.unloaded == f.idle.len) return false;
            f.available += f.idle[f.unloaded];
            f.unloaded += 1;
            return true;
        }
    };
    // Already fits: nothing is unloaded.
    var a = Fake{ .needed = 5, .available = 9, .idle = &.{ 15, 3 } };
    try t.expect(fitAfterUnloadingIdle(&a, Fake.fits, Fake.unloadOne));
    try t.expectEqual(@as(usize, 0), a.unloaded);
    // The live shape: GLM's prompt needs 10.5 GB beside 9.4 GB free and an idle 14.7 GB image model.
    var b = Fake{ .needed = 10_558, .available = 9_466, .idle = &.{ 14_730, 8_000 } };
    try t.expect(fitAfterUnloadingIdle(&b, Fake.fits, Fake.unloadOne));
    try t.expectEqual(@as(usize, 1), b.unloaded);
    // Nothing idle left to give back: still refused.
    var c = Fake{ .needed = 50, .available = 9, .idle = &.{ 15, 3 } };
    try t.expect(!fitAfterUnloadingIdle(&c, Fake.fits, Fake.unloadOne));
    try t.expectEqual(@as(usize, 2), c.unloaded);
}

test "a refusal names the credit it COMPARED, never the residency it cannot take" {
    const t = std.testing;
    const mb: u64 = 1024 * 1024;

    // The live shape: the whole residency is the checked-out entry.
    const pinned_only = AdmissionBill{
        .needed = 13_664 * mb,
        .available = 13_374 * mb,
        .evictable = 7_802 * mb,
        .reclaimable = 0,
        .chunk = 512,
    };
    try t.expectEqual(@as(u64, 0), pinned_only.evictionCredit());
    try t.expectEqual(7_802 * mb, pinnedResidentBytes(pinned_only));
    try t.expect(!pinned_only.fits());
    try t.expect(!pinned_only.fitsAfterEviction());
    try t.expectEqual(
        pinned_only.needed <= pinned_only.available +| pinned_only.evictionCredit(),
        pinned_only.fitsAfterEviction(),
    );

    // The same residency, genuinely reclaimable: the credit is real and the request is admitted to evict.
    var reclaimable = pinned_only;
    reclaimable.reclaimable = 7_802 * mb;
    try t.expectEqual(7_802 * mb, reclaimable.evictionCredit());
    try t.expectEqual(@as(u64, 0), pinnedResidentBytes(reclaimable));
    try t.expect(reclaimable.fitsAfterEviction());
}

test "checkAttentionMemory routes deepseek_v4 through its own estimator with the REAL sub-chunk" {
    // The estimator existing proves nothing if the call site keeps handing
    // dsv4 to the generic bill (the attn_keys scan-test class). Needles are
    // split with `++` so this test's own source never matches them.
    const t = std.testing;
    const src = @embedFile("prefill_chunk.zig");
    const call = "dsv4PrefillMemoryNeeded(seq, layers, " ++
        "kv_heads * hdim, hidden, ffn, dsv4_mod.prefillSub(), config.prefillAttnKeys(seq))";
    try t.expect(std.mem.indexOf(u8, src, call) != null);
}

test "checkAttentionMemory does not bill resident hot-cache buffers twice" {
    // RAM restore uses refcount-sharing. The resident entry is already inside
    // `active_mem`, and prefillMemoryNeeded bills the destination KV growth.
    // Pin the chokepoint so a future cache-accounting change cannot add the
    // resident entry to `needed` again (the 138k-token false-400 regression).
    const t = std.testing;
    const src = @embedFile("admit.zig");
    // The bill and the headroom moved into `prefillAdmissionBill`.
    const start = std.mem.indexOf(u8, src, "pub fn prefillAdmission" ++ "Bill(") orelse return error.CallSiteMoved;
    const tail = src[start..];
    const end = std.mem.indexOf(u8, tail, "\npub extern \"c\" fn sysctlbyname") orelse return error.CallSiteMoved;
    const body = tail[0..end];
    try t.expect(std.mem.indexOf(u8, body, "largestEntry" ++ "Bytes") == null);
    try t.expect(std.mem.indexOf(u8, body, "hot_" ++ "restore") == null);
    try t.expect(std.mem.indexOf(u8, body, "mlx_get_active_memory(&active_mem)") != null);
    // The hot cache appears once, as `evictable`, read as the published atomic.
    try t.expect(std.mem.indexOf(u8, body, "resident_hot_cache_bytes.load(.monotonic)") != null);
    try t.expect(std.mem.indexOf(u8, body, "sch." ++ "hot_prefix_cache") == null);
    try t.expect(std.mem.indexOf(u8, body, ".hot_prefix_cache.?") == null);
    try t.expect(std.mem.indexOf(u8, body, "needed = needed +") == null);
}

test "checkAttentionMemory wires the CONFIG's key bound, not a dense seq" {
    // The estimator taking an `attn_keys` parameter proves nothing if the one
    // caller hands it `seq` — that reproduces the dense bill exactly, and every
    // unit test above still passes because they call the function directly.
    // Same class as the two hardcoded `use_drafter=false` call sites: the
    // engagement has to be pinned at the SITE.
    //
    // The KV bill is pinned at the same site and for the same reason: it must
    // come from `config.kvBytesPerToken()` (which knows how many layers cache
    // and how wide their keys and values are), never a uniform product
    // recomputed here. Same for the SCORE width: `config.prefillScoreHeadDim()`
    // is the width the score is contracted at, and handing `hdim` for both
    // declares an MLA arch's 192-wide scores "fused" and bills them at zero.
    const t = std.testing;
    const src = @embedFile("prefill_chunk.zig") ++ @embedFile("bills.zig") ++ @embedFile("context.zig") ++ @embedFile("admit.zig");
    const call = "prefillMemoryNeeded(seq, heads, kv_heads, config.kvBytesPerToken(), hdim, config.prefillScoreHeadDim(), hidden, ffn, kv_bits, chunk,";
    const at = std.mem.indexOf(u8, src, call) orelse return error.CallSiteMoved;
    const tail = src[at + call.len ..];
    const end = std.mem.indexOfScalar(u8, tail, ')') orelse return error.CallSiteMoved;
    try t.expectEqualStrings(" config.prefillAttnKeys(seq", tail[0..end]);
    // The arch's own prefill streams and its dequantized-weight working set
    // are the same class of parameter: derived from the CONFIG at the site, or
    // a GatedDeltaNet hybrid gets an attention arch's bill (measured 33% low)
    // and a quantized checkpoint gets a dense one's.
    // The per-request terms are seq-scaled and derived from the config at the site. Counted, not
    // just found: the literal below is itself part of `src`.
    const wired = "config.prefillAttnKeys(seq), prefillStreamBytesPerToken(config), prefillDequantWeightBytes(config), prefillRequestTerms(config, seq, max_tokens, kv_bits, chunk, warm))";
    var wired_seen: usize = 0;
    var wired_at: usize = 0;
    while (std.mem.indexOfPos(u8, src, wired_at, wired)) |hit| {
        wired_seen += 1;
        wired_at = hit + 1;
    }
    try t.expectEqual(@as(usize, 2), wired_seen); // the call site + this literal
    // The reserve half bills at an explicit KV length (`prefillTransientReserveAtKv`); `seq` must
    // stay `@max(kv_len, chunk)`.
    try t.expect(std.mem.indexOf(u8, src, "    const seq: u64 = @max(kv_len, chunk);\n") != null);
    try t.expect(std.mem.indexOf(u8, src, "        config.prefillAttnKeys(seq),\n        prefillStreamBytesPerToken(config),\n        prefillDequantWeightBytes(config),\n") != null);

    // Auto-context sizing reads the same helpers — the two must not drift.
    // The KV width: both bill through `kvBytesPerTokenAtBits`, so a
    // `--kv-quant 4` server cannot size its context against fp16 while the
    // guard admits against 4-bit (that mismatch reported a third of what fits).
    try t.expect(std.mem.indexOf(u8, src, "const kv_bytes: u64 = seq * kvBytesPerTokenAtBits(kv_per_tok, kv_bits);") != null);
    try t.expect(std.mem.indexOf(u8, src, "const per_tok: u64 = kvBytesPerTokenAtBits(config.kvBytesPerToken(), kv_bits);") != null);
    // The prefill transient: the sizer RESERVES it once at the chunk width
    // (`prefillTransientReserve` is the same estimator with the KV term zeroed),
    // never as a per-token multiplier on the context it is solving for.
    // The sizer reserves the constant, not the ask or the resolved budget.
    try t.expect(std.mem.indexOf(u8, src, "ctxCacheShare(ctxSizingCacheReserve(config), ceiling, active_mem, transient) +| transient") != null);
}

test "the guard credits only PROVABLY reclaimable cache bytes, never the entry a restore will pin" {
    const t = std.testing;
    // The guard admitted on `available + evictable >= needed`, but the evictable 1,564 MB was
    // the one entry this prompt then restored from; the provable credit is residency minus the largest entry.
    const MB: u64 = 1024 * 1024;
    var bill = AdmissionBill{ .needed = 37_394 * MB, .available = 37_130 * MB, .evictable = 1_564 * MB, .reclaimable = 0 };
    try t.expect(!bill.fits());
    try t.expect(!bill.fitsAfterEviction()); // it DID, before this fix
    bill.reclaimable = 1_564 * MB;
    try t.expect(bill.fitsAfterEviction());
}

test "the admission probe bills the request's OWN kv-quant and chunking, not the process defaults" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // The probe used to hardcode `kv_override = null`, pricing a `kv_quant: 4` request at fp16 on the inference thread.
    const t = std.testing;
    const cfg = qwen4ExpOomConfig();
    const seq: usize = 200_000;
    const q4 = transformer_mod.KVQuantConfig.affine(4);
    const q8 = transformer_mod.KVQuantConfig.affine(8);
    const bill4 = prefillAdmissionBill(&cfg, seq, 2048, q4, false, null, .{});
    const bill8 = prefillAdmissionBill(&cfg, seq, 2048, q8, false, null, .{});
    const bill16 = prefillAdmissionBill(&cfg, seq, 2048, transformer_mod.KVQuantConfig.dense, false, null, .{});
    try t.expect(bill4.needed < bill8.needed);
    try t.expect(bill8.needed < bill16.needed);
}

test "pinnedResidentBytes names the entry a restore would share, and only that" {
    const t = std.testing;
    // Residency the eviction pass cannot return: the connection thread's one signal that this prompt is warm.
    try t.expectEqual(@as(u64, 0), pinnedResidentBytes(.{ .evictable = 0, .reclaimable = 0 }));
    try t.expectEqual(@as(u64, 0), pinnedResidentBytes(.{ .evictable = 4096, .reclaimable = 4096 }));
    // The live 768k reading: 24,826 MB resident, 10,294 MB reclaimable.
    const mb: u64 = 1024 * 1024;
    try t.expectEqual(@as(u64, 14_532 * mb), pinnedResidentBytes(.{ .evictable = 24_826 * mb, .reclaimable = 10_294 * mb }));
    // Saturating: the two numbers are published from different sites.
    try t.expectEqual(@as(u64, 0), pinnedResidentBytes(.{ .evictable = 1, .reclaimable = 2 }));
}

test "prefillAdmissionBill: both admit arms are driven by the two credits" {
    const t = std.testing;
    const MB: u64 = 1024 * 1024;

    const evict = AdmissionBill{ .needed = 30 * MB, .available = 20 * MB, .evictable = 15 * MB, .reclaimable = 15 * MB };
    try t.expect(!evict.fits());
    try t.expect(evict.fitsAfterEviction());
    try t.expectEqual(AdmissionVerdict.evict, admissionVerdict(evict));

    const deferral = AdmissionBill{ .needed = 30 * MB, .available = 20 * MB, .evictable = 15 * MB, .reclaimable = 0 };
    try t.expect(!deferral.fitsAfterEviction());
    try t.expect(pinnedResidentBytes(deferral) > 0); // warm deferral

    // An empty cache carries neither credit, and both arms go dead.
    const empty = AdmissionBill{ .needed = 30 * MB, .available = 20 * MB };
    try t.expectEqual(empty.fits(), empty.fitsAfterEviction());
    try t.expectEqual(@as(u64, 0), pinnedResidentBytes(empty));
    try t.expectEqual(AdmissionVerdict.refuse, admissionVerdict(empty));
    // A bill that fits still admits on both arms.
    const roomy = AdmissionBill{ .needed = 10 * MB, .available = 20 * MB };
    try t.expectEqual(AdmissionVerdict.admit, admissionVerdict(roomy));
}
