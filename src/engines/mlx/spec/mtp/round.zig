//! The MTP round: begin, chain, verify, group accept.
const depth_bounds = @import("../mtp_depth_bounds.zig");
const std = @import("std");
const mlx = @import("../../../../mlx/ffi.zig");
const transformer_mod = @import("../../transformer.zig");
const tokenizer_mod = @import("../../../../host/tokenizer/tokenizer.zig");
const model_mod = @import("../../model.zig");
const log = @import("../../../../host/log.zig");
const io_util = @import("../../../../sdk/util/io_util.zig");
const mtp_mod = @import("../../../../sdk/spec/mtp/mtp.zig");
const round_cost = @import("../round_cost.zig");
const group_planner = @import("../mtp_group_planner.zig");
const Transformer = transformer_mod.Transformer;
const Tokenizer = tokenizer_mod.Tokenizer;
const ForwardCtx = transformer_mod.ForwardCtx;
const dflash_mod = @import("../dflash/dflash.zig");
const KVCache = transformer_mod.KVCache;
const testing = std.testing;
const chain_mod = @import("chain.zig");
const generate_mod = @import("../../generate.zig");
const trace_mod = @import("trace.zig");
const CommittedArgmax = @import("../verify.zig").CommittedArgmax;
const DrafterStepResult = @import("../drafter.zig").DrafterStepResult;
const Generator = @import("../../generate/generator.zig").Generator;
const MTP_EV_WARMUP_ROUNDS = @import("depth.zig").MTP_EV_WARMUP_ROUNDS;
const MtpCacheRef = @import("../mtp_head_ref.zig").MtpCacheRef;
const MtpChainTestSlot = @import("chain.zig").MtpChainTestSlot;
const MtpHeadRef = @import("../mtp_head_ref.zig").MtpHeadRef;
const MtpHistStash = @import("chain.zig").MtpHistStash;
const MtpPreDraft = @import("chain.zig").MtpPreDraft;
const SamplingParams = @import("../../sampling/params.zig").SamplingParams;
const SubLap = @import("trace.zig").SubLap;
const VERIFY_PIPELINE_LAYERS = @import("../dflash_tree.zig").VERIFY_PIPELINE_LAYERS;
const capAcceptedForTokenBudget = @import("../verify.zig").capAcceptedForTokenBudget;
const deinit = @import("../../generate/generator.zig").Generator.deinit;
const groupProbsBlock = @import("../../sampling/spec_sampling.zig").groupProbsBlock;
const groupProbsRow = @import("../../sampling/spec_sampling.zig").groupProbsRow;
const isEosId = @import("../../generate/stops.zig").isEosId;
const mtpAdaptiveEnabled = @import("trace.zig").mtpAdaptiveEnabled;
const mtpAdaptiveReentryTokens = @import("adaptive.zig").mtpAdaptiveReentryTokens;
const mtpAdaptiveSerialEnabled = @import("adaptive.zig").mtpAdaptiveSerialEnabled;
const mtpBatchCorrEnabled = @import("serial.zig").mtpBatchCorrEnabled;
const mtpChainDispatch = @import("chain_build.zig").mtpChainDispatch;
const mtpEarlyDispatchEnabled = @import("trace.zig").mtpEarlyDispatchEnabled;
const mtpEmaMs = @import("regime.zig").mtpEmaMs;
const mtpEvSeedEnabled = @import("trace.zig").mtpEvSeedEnabled;
const mtpLiveCostEnabled = @import("regime.zig").mtpLiveCostEnabled;
const mtpPinnedDepth = @import("regime.zig").mtpPinnedDepth;
const mtpTraceEnabled = @import("trace.zig").mtpTraceEnabled;
const readChainConfidence = @import("../../sampling/spec_sampling.zig").readChainConfidence;
const sampleFromProbsLazy = @import("../../sampling/lazy.zig").sampleFromProbsLazy;
const sampleRows = @import("../../sampling/lazy.zig").sampleRows;
const sampleRowsAnySeed = @import("../../sampling/lazy.zig").sampleRowsAnySeed;
const sampleRowsHomogeneous = @import("../../sampling/lazy.zig").sampleRowsHomogeneous;
const specDecodeUnsupported = @import("../verify.zig").specDecodeUnsupported;
const verifyArgmax = @import("../verify.zig").verifyArgmax;


/// A speculative round between its draft chain and its verdict. `mtpRoundBegin`
/// builds it, `mtpRoundVerify` (solo) or the scheduler's group verify fills the
/// three verify outputs, `mtpRoundFinish` consumes them.
pub const MtpRoundState = struct {
    chain: MtpPreDraft,
    tracing: bool,
    ph: io_util.Stopwatch,
    livecost: bool,
    round_watch: io_util.Stopwatch,
    kv_step_snap: usize,
    kv_snap: ?transformer_mod.KVCacheSnapshot,
    moe_seq_offset_snap: usize,
    /// `[1, 1+m]` int32.
    verify_input: mlx.mlx_array,
    /// Positions the verify forward ran; `1 + m` solo, the group's widest row batched.
    verify_len: u32,
    /// `c_n`: the positions this row committed, stamped by the finish.
    retained: u32 = 0,
    verify_rollback: ?Transformer.VerifyRollback = null,
    verify_ctx: ?*ForwardCtx = null,
    verify_argmax: CommittedArgmax = .{},
    verify_logits: mlx.mlx_array = .{ .ctx = null },
    new_hidden: mlx.mlx_array = .{ .ctx = null },
    verify_hidden_all: mlx.mlx_array = .{ .ctx = null },
    /// The group's joined `[1, Σ(1+m_i), V]` verify block, held by the FIRST
    /// row's state; the accept side filters it once for every sampled row.
    group_logits: mlx.mlx_array = .{ .ctx = null },
    /// Phase-4 terms the group prepared for this row (sampled rows only):
    /// the accept graph's own outputs, whichever acceptance mode built them.
    /// `accept_p` null-ctx = this row builds its own in the finish.
    accept_p: mlx.mlx_array = .{ .ctx = null },
    accept_q: mlx.mlx_array = .{ .ctx = null },
    accept_defer: mlx.mlx_array = .{ .ctx = null },
    corr: mlx.mlx_array = .{ .ctx = null },

    pub fn deinit(self: *MtpRoundState, allocator: std.mem.Allocator) void {
        self.chain.deinit(allocator);
        if (self.kv_snap) |*snap| snap.deinit();
        _ = mlx.mlx_array_free(self.verify_input);
        self.verify_argmax.deinit();
        if (self.verify_logits.ctx != null) _ = mlx.mlx_array_free(self.verify_logits);
        if (self.group_logits.ctx != null) _ = mlx.mlx_array_free(self.group_logits);
        self.releasePreparedAccept();
        if (self.new_hidden.ctx != null) _ = mlx.mlx_array_free(self.new_hidden);
        if (self.verify_hidden_all.ctx != null) _ = mlx.mlx_array_free(self.verify_hidden_all);
        if (self.verify_ctx) |ctx| {
            if (ctx.ssm_entries) |entries| for (entries) |*entry| transformer_mod.ssmFreeSpecCapture(entry);
        }
        if (self.verify_rollback) |*saved| {
            saved.restore();
            saved.deinit();
        }
    }

    /// Drop the group-prepared Phase-4 terms, so the row's finish builds its own.
    pub fn releasePreparedAccept(self: *MtpRoundState) void {
        inline for (.{ &self.accept_p, &self.accept_q, &self.accept_defer, &self.corr }) |slot| {
            if (slot.ctx != null) {
                _ = mlx.mlx_array_free(slot.*);
                slot.* = .{ .ctx = null };
            }
        }
    }
};

pub const MtpRowVerdict = struct { accepted: u32, next_pending: u32 };

/// One row's greedy verdict. Takes the committed-argmax wrapper, so no raw
/// argmax can reach a commit; `mtpGreedyVerdict` is the pure core.
pub fn mtpAcceptRowGreedy(am: CommittedArgmax, len: u32, drafts: []const u32, completion: u32, max_tokens: u32) !MtpRowVerdict {
    return mtpGreedyVerdict(try am.ids(len), drafts, completion, max_tokens);
}

pub fn mtpGreedyVerdict(am: []const i32, drafts: []const u32, completion: u32, max_tokens: u32) MtpRowVerdict {
    const accepted = capAcceptedForTokenBudget(mtpGreedyPrefix(am, drafts), completion, max_tokens);
    return .{ .accepted = accepted, .next_pending = @intCast(am[accepted]) };
}

/// Drafts matching the verify argmax, before any budget or stop cut.
pub fn mtpGreedyPrefix(am: []const i32, drafts: []const u32) u32 {
    var accepted: u32 = 0;
    while (accepted < drafts.len and @as(u32, @intCast(am[accepted])) == drafts[accepted]) accepted += 1;
    return accepted;
}

/// The committed increment of a verified row: its pending token plus the drafts it
/// accepted. The group's `[N]` retained-position vector is these values.
pub fn mtpRetained(accepted: u32) u32 {
    return 1 + accepted;
}

/// Full accept keeps the state the verify forward left, so it needs every draft
/// accepted AND the row's own width; a padded row always takes the rollback arm.
pub fn mtpFullAccept(accepted: u32, m: u32, verify_len: u32) bool {
    return accepted == m and verify_len == 1 + m;
}

pub const MTP_GROUP_ROWS_MAX = 32;

pub const MtpRoundOpen = struct {
    chain: MtpPreDraft,
    tracing: bool,
    ph: io_util.Stopwatch,
    livecost: bool,
    round_watch: io_util.Stopwatch,
};

pub const MtpRoundBegin = union(enum) { done: ?DrafterStepResult, verify: MtpRoundState, open: MtpRoundOpen };

pub fn nextMtp(self: *Generator, allocator: std.mem.Allocator) !?DrafterStepResult {
    var st = switch (try self.mtpRoundBegin(allocator)) {
        .done => |r| return r,
        .verify => |v| v,
        .open => |o| try self.mtpRoundContinue(allocator, o),
    };
    defer st.deinit(allocator);
    try self.mtpRoundVerify(&st);
    return self.mtpRoundFinish(allocator, &st);
}

/// Phases 0–2: serial-block dispatch, the draft chain, rollback anchors, the verify input.
pub fn mtpRoundBegin(self: *Generator, allocator: std.mem.Allocator) !MtpRoundBegin {
    if (self.done) return .{ .done = null };
    std.debug.assert(self.mtp != null);
    std.debug.assert(self.mtp_cache != null);
    std.debug.assert(self.has_last_hidden);
    if (group_planner.enabled() and self.mtp_planner_owned and self.mtp_hidden_stale) return error.MtpHiddenStale;
    // Release-enforced guard (issue #97): the MTP path cannot honor a
    // grammar constraint or logprobs (compiled-out asserts before).
    if (specDecodeUnsupported(self.sampling, self.logprobs_n)) return error.SpecDecodeUnsupported;

    // `--max-mtp-ctx`, checked at the top of the round against prompt + generated. Sticky
    // (context only grows), routed through the same `spec_disabled_runtime` fallback.
    if (!self.spec_disabled_runtime and
        !mtp_mod.mtpCtxWithinLimit(generate_mod.max_mtp_ctx, self.mtpKvLen()))
    {
        log.info(
            "  mtp=off (ctx {d} > --max-mtp-ctx {d})\n",
            .{ self.mtpKvLen(), generate_mod.max_mtp_ctx },
        );
        self.spec_disabled_runtime = true;
        self.spec_disable_reason = .max_ctx;
    }

    // Adaptive serial re-entry: sticky, except when the request crossed into a different KV
    // bucket. `--max-mtp-ctx` (`.max_ctx`) never re-enters.
    if (self.spec_disabled_runtime and self.spec_disable_reason == .adaptive and
        mtpAdaptiveSerialEnabled() and self.mtpAdaptiveHeadMayResume())
    {
        const b = self.mtpAdaptiveBucket(self.mtpKvLen());
        const prev_bucket = self.mtp_adaptive.bucket;
        const prev_arm = self.mtp_adaptive.arm;
        const action = self.mtp_adaptive.serialTick(b, mtpAdaptiveReentryTokens());
        self.mtpAdaptiveSyncWindow(prev_bucket, prev_arm);
        // Re-entry only when the head can prove it is in sync (else `qwen4MtpForward` returns
        // `error.MtpPositionGap`). A decline latches: the drift cannot shrink on its own.
        var may_reenter = action == .to_mtp;
        if (may_reenter) {
            if (self.mtpHeadDriftNow()) |drift| {
                if (drift != 0) {
                    log.info(
                        "  [mtp] adaptive: re-entry declined (head out of sync by {d}) -> serial for the rest of the request\n",
                        .{drift},
                    );
                    self.mtp_adaptive.declineReentry();
                    may_reenter = false;
                }
            }
        }
        if (may_reenter) {
            log.info(
                "  [mtp] adaptive: kv {d} crossed into bucket {s} -> mtp\n",
                .{ self.mtpKvLen(), round_cost.bucketName(self.xfm.round_cost.layout, b) },
            );
            self.spec_disabled_runtime = false;
            self.spec_disable_reason = .none;
            self.mtp_serial_exit = .drain;
        }
    }

    // Serial block: the runtime-disabled fallback, the bounded probe, or the ramp back.
    if (self.spec_disabled_runtime or self.mtp_serial_left > 0 or self.mtp_serial_exit != .none) {
        try self.mtpDetachHead(allocator, self.mtpSerialMayResume());
        self.mtpMaybeReleaseModuleHead();
        return .{ .done = try self.mtpSerialTick(allocator) };
    }

    if (try self.checkStop()) return .{ .done = null }; // t1 is this block's first emit: stop before drafting

    const head = self.mtp.?;
    // Cross-request EV seed: inherit the head's last healthy acceptance
    // surface so the controller plans from round 1 instead of re-warming
    // (~10 legacy rounds + a +1/round base climb — a third of a short
    // generation). Demotion stays instant (EMA decay + sticky disable are
    // per-request), so a workload change costs a few rounds, not the win.
    // A pinned depth is a measurement mode: every round drafts
    // exactly n and the controller never plans, so a seed must not be
    // applied (nor, at deinit, published).
    if (self.mtp_ev_rounds == 0 and self.mtp_attempted == 0 and
        mtpAdaptiveEnabled() and mtpEvSeedEnabled() and mtpPinnedDepth() == null and
        (!group_planner.enabled() or !self.mtp_planner_owned))
    {
        if (head.evSeed()) |seed| {
            self.mtp_ev_accept = seed.accept;
            self.mtp_ev_m_lo_prev = @min(@max(seed.m_lo, 1), mtp_mod.MAX_DEPTH);
            self.mtp_ev_rounds = MTP_EV_WARMUP_ROUNDS;
        }
    }
    const tracing = mtpTraceEnabled();
    var ph: io_util.Stopwatch = undefined;
    if (tracing) {
        ph = io_util.Stopwatch.init(self.timer.io);
        // Close the scheduler gap opened at the previous round's return.
        if (self.mtp_gap_watch) |*gw| self.mtp_trace.add(.gap, gw.read());
        self.mtp_gap_watch = null;
    }
    // Always-on round wall-clock for the live-cost round EMA (the
    // denominator of the sync fraction). Read at both commit exits.
    const livecost = mtpLiveCostEnabled();
    const round_watch = io_util.Stopwatch.init(self.timer.io);

    // ── Phase 0/1: acquire this round's draft chain ──
    // Round plan: fixed mode (and EV warmup) is a single chunk at the
    // windowed adaptive depth; post-warmup EV mode plans a base chunk
    // m_lo plus a confidence-gated extension to m_hi (see the EV
    // controller section below). `chain.m` is the tokens actually
    // drafted this round — it grows from m_lo to m_hi iff the gate
    // clears at the chunk boundary below.
    //
    // Either the PREVIOUS round built + dispatched chunk A at its tail
    // (cross-round pipelining, `mtpMaybePreDraft` — the drafts are
    // already materializing on the GPU) or we build it here (round 1,
    // MLX_SERVE_MTP_PREDRAFT=0, or a round with no successor state).
    //
    // No head-cache snapshot in either case: a snapshot refcount-shares
    // the head's KV buffer, which forces every draft append's
    // slice_update to copy-on-write the WHOLE history buffer (~268
    // MB/append at 64k). Rollback is truncate — offset-only — since
    // draft entries only ever append past chain.off0 (the committed
    // origin, resolved from the pending history stash when one exists).
    const chain: MtpPreDraft = if (self.mtp_pre_draft) |pd| blk: {
        self.mtp_pre_draft = null;
        // The pre-draft consumed the history stash when it was built.
        std.debug.assert(self.mtp_hist_stash == null);
        std.debug.assert(pd.t1 == self.next_token_id);
        break :blk pd;
    } else blk: {
        const plan_now = self.mtpRoundPlan();
        if (try self.mtpLookupChain(allocator, plan_now, self.next_token_id)) |lc| break :blk lc;
        var c = try self.mtpChainInit(allocator, plan_now, self.next_token_id);
        errdefer c.deinit(allocator);
        if (self.mtp_batch_head) {
            return .{ .open = .{
                .chain = c,
                .tracing = tracing,
                .ph = ph,
                .livecost = livecost,
                .round_watch = round_watch,
            } };
        }
        const chain_lap = SubLap.start(tracing, self.timer.io);
        try self.mtpChainBuild(&c, 0, plan_now.m_lo);
        if (mtpEarlyDispatchEnabled()) try mtpChainDispatch(&c, 0, plan_now.m_lo);
        chain_lap.stop(&self.mtp_trace, .chain);
        break :blk c;
    };
    return .{ .verify = try self.mtpRoundAfterChain(allocator, chain, tracing, ph, livecost, round_watch) };
}

pub fn mtpRoundContinue(self: *Generator, allocator: std.mem.Allocator, open: MtpRoundOpen) !MtpRoundState {
    var chain = open.chain;
    if (chain.n_drafted < chain.m) {
        const chain_lap = SubLap.start(open.tracing, self.timer.io);
        self.mtpChainBuild(&chain, chain.n_drafted, chain.m) catch |err| {
            chain.deinit(allocator);
            return err;
        };
        chain_lap.stop(&self.mtp_trace, .chain);
    }
    return self.mtpRoundAfterChain(allocator, chain, open.tracing, open.ph, open.livecost, open.round_watch);
}

pub fn mtpRoundAfterChain(
    self: *Generator,
    allocator: std.mem.Allocator,
    chain_in: MtpPreDraft,
    tracing: bool,
    ph_in: io_util.Stopwatch,
    livecost: bool,
    round_watch: io_util.Stopwatch,
) !MtpRoundState {
    var chain = chain_in;
    errdefer chain.deinit(allocator);
    var ph = ph_in;
    const s = self.xfm.s;
    const plan = chain.plan;
    // A chain pre-drafted before this slot joined a group may be wider than the
    // cap: use its first `cap` drafts (the head's tail past them is truncated at
    // the next consume, like any rejected draft).
    // A lookup chain was sized by its own gate.
    const cap: u32 = if (self.mtp_group_cap > 0) self.mtp_group_cap else if (chain.lookup) chain.m else mtp_mod.MAX_DEPTH;
    const m_lo: u32 = @min(plan.m_lo, cap);
    const m_max: u32 = @min(plan.m_hi, cap);
    chain.m = @min(chain.m, cap);
    if (tracing) {
        self.mtp_trace.add(.draft, ph.read());
        ph.reset();
    }

    // ── chunk-A boundary: extension decision (two-chunk EV plans) ──
    // The one bounded sync of the round; near-free when the chain was
    // pre-drafted (ids + confidences already materialized).
    if (m_max > m_lo) {
        var sync_watch = io_util.Stopwatch.init(self.timer.io);
        const chain_ln = try readChainConfidence(chain.draft_arrs[0..m_lo], chain.conf_arrs.?[0..m_lo], s);
        const sync_ns = sync_watch.read();
        // Live sync-cost EMA: measured only on considered rounds, so it
        // holds the true per-sync cost (not the suppressed-round average).
        if (livecost) self.mtp_ev_sync_ms = mtpEmaMs(self.mtp_ev_sync_ms, sync_ns);
        if (tracing) {
            self.mtp_trace.add(.sync, sync_ns);
            ph.reset();
        }
        if (chain_ln >= plan.tau_ln) {
            chain.m = m_max;
            self.mtp_ext_rounds += 1;
            self.mtp_ext_dry_streak = 0;
            try self.mtpChainBuild(&chain, m_lo, m_max);
            if (mtpEarlyDispatchEnabled()) try mtpChainDispatch(&chain, m_lo, m_max);
        } else {
            // Considered but the gate didn't clear — feeds the
            // dry-spell policy (mtpExtDryAllows).
            self.mtp_ext_dry_streak +|= 1;
        }
        if (tracing) {
            self.mtp_trace.add(.ext, ph.read());
            ph.reset();
        }
    }
    const m: u32 = chain.m;
    // Chain complete — release the chain hidden (only chunk B needed it).
    if (chain.h_chain) |h| {
        _ = mlx.mlx_array_free(h);
        chain.h_chain = null;
    }
    const draft_arrs = chain.draft_arrs;

    // ── Phase 2: record rollback anchors (NO snapshot on the GDN path) ──
    // A KVCache.snapshot() refcount-shares the KV buffers, which forces
    // verify's slice_update writes to COPY-on-write every full-attention
    // layer's WHOLE buffer — ~4.3 GB per round at 64k context, the
    // dominant round cost at long context. On a GDN trunk (every real MTP
    // target is qwen3_5-family hybrid) rollback needs only the pre-verify
    // LENGTH (KV truncate is offset-only; the stale tail past it is
    // unreachable) plus the verify pass's per-position SSM capture, so no
    // snapshot is taken at all; nor on MiMo, whose attention-only trunk
    // rolls back by the same truncate. Any other pure-attention target
    // keeps the proven snapshot + re-forward path.
    const kv_step_snap = self.ctx.cache.step;
    const truncate_only = self.xfm.config.isMimo();
    var kv_snap: ?transformer_mod.KVCacheSnapshot = if (self.ctx.ssm_entries != null or truncate_only) null else try self.ctx.cache.snapshot();
    errdefer if (kv_snap) |*snap| snap.deinit();
    const moe_seq_offset_snap = self.ctx.moe_seq_offset.*;

    // ── Phase 3: verify input [t1, drafts...] as one [1, 1+m] tensor ──
    const reshape_2d = [_]c_int{ 1, 1 };
    var t1_2d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(t1_2d);
    try mlx.check(mlx.mlx_reshape(&t1_2d, chain.t1_arr, &reshape_2d, 2, s));

    var verify_input = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(verify_input);
    {
        const drafts_2d = try allocator.alloc(mlx.mlx_array, m);
        var drafts_2d_n: usize = 0;
        defer {
            for (drafts_2d[0..drafts_2d_n]) |arr| _ = mlx.mlx_array_free(arr);
            allocator.free(drafts_2d);
        }
        for (draft_arrs[0..m], drafts_2d) |dlazy, *out| {
            out.* = mlx.mlx_array_new();
            drafts_2d_n += 1;
            try mlx.check(mlx.mlx_reshape(out, dlazy, &reshape_2d, 2, s));
        }
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        _ = mlx.mlx_vector_array_append_value(vec, t1_2d);
        for (drafts_2d) |arr| _ = mlx.mlx_vector_array_append_value(vec, arr);
        try mlx.check(mlx.mlx_concatenate_axis(&verify_input, vec, 1, s));
    }

    return .{
        .chain = chain,
        .tracing = tracing,
        .ph = ph,
        .livecost = livecost,
        .round_watch = round_watch,
        .kv_step_snap = kv_step_snap,
        .kv_snap = kv_snap,
        .moe_seq_offset_snap = moe_seq_offset_snap,
        .verify_input = verify_input,
        .verify_len = 1 + m,
    };
}

/// Phase 3 of a solo round: the trunk verify forward over `[t1, drafts…]`.
/// A batched group replaces this with one forward for every slot (`mtpGroupVerify`).
pub fn mtpRoundVerify(self: *Generator, st: *MtpRoundState) !void {
    const xfm = self.xfm;
    const tracing = st.tracing;
    var new_hidden = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(new_hidden);
    var verify_hidden_all = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(verify_hidden_all);
    // Enable per-position SSM capture for the verify pass on a GDN trunk
    // so partial accept can roll back without re-forwarding the accepted
    // prefix (mirrors nextPld — the re-forward re-runs the 48-layer
    // sequential recurrence AND a full trunk weight read, and at depth > 1
    // MOST rounds are partial, so it dominated the round cost).
    self.ctx.capture_ssm_seq = self.ctx.ssm_entries != null;
    // DIAGNOSTIC (MLX_SERVE_MTP_TRACE_SYNC=1): drain the GPU before the
    // verify build so the `sync` lap shows the pending lazy work (draft
    // chain, rollback, commit) and `verify` the forward alone.
    if (tracing and std.c.getenv("MLX_SERVE_MTP_TRACE_SYNC") != null) {
        var sync_watch = io_util.Stopwatch.init(self.timer.io);
        try mlx.check(mlx.mlx_array_eval(st.verify_input));
        self.mtp_trace.add(.sync, sync_watch.read());
        st.ph.reset();
    }
    // Captures the post-final-norm hidden at the LAST position (next
    // round's h_prev) AND all 1+m positions (history re-append).
    //
    // qwen4_exp: the n-gram PLE at trunk layer 1 is a HOST gather keyed on
    // these ids, and every id past t1 is still a lazy draft — gathering
    // eagerly parks the build of layers 2..N behind the whole draft chain
    // finishing on the GPU. Defer the leaf so the graph builds while the
    // chain runs, then sync ONCE (`flushDeferredPle`, below) before Phase
    // 4 evaluates anything. Other arches never set `ple_pending`.
    self.ctx.ple_defer = true;
    self.ctx.pipeline_build = if (self.mtp_serial_accept) VERIFY_PIPELINE_LAYERS else 0;
    defer self.ctx.pipeline_build = 0;
    self.ctx.spec_verify = true;
    defer self.ctx.spec_verify = false;
    const verify_logits = xfm.forwardWithCaptureAll(&self.ctx, st.verify_input, &new_hidden, &verify_hidden_all) catch |e| {
        self.ctx.ple_defer = false;
        self.ctx.capture_ssm_seq = false;
        xfm.discardDeferredPle(&self.ctx);
        return e;
    };
    errdefer _ = mlx.mlx_array_free(verify_logits);
    self.ctx.ple_defer = false;
    self.ctx.capture_ssm_seq = false;
    xfm.verify_laps.ple_sync_ns = 0;
    if (tracing) {
        const build_ns = st.ph.read();
        self.mtp_trace.add(.verify, build_ns);
        self.mtp_trace.addSub(.build, build_ns);
        st.ph.reset();
    }
    const ple_lap = SubLap.start(tracing, self.timer.io);
    defer if (tracing) {
        ple_lap.stop(&self.mtp_trace, .ple);
        self.mtp_trace.addSub(.ple_sync, xfm.verify_laps.ple_sync_ns);
    };
    // Fill the deferred PLE leaf: the one host read of the verify ids, and
    // the point at which the entry's n-gram history + `spec_ple_tokens`
    // advance. Must precede BOTH the first eval of anything downstream of
    // the leaf (Phase 4) and `ssmRollbackFromCapture` (Phase 5). The GPU
    // arm has no leaf: its history settles from the draft read in the
    // finish, still before Phase 5, so this round reads no ids here.
    errdefer xfm.discardDeferredPle(&self.ctx);
    if (!Transformer.deferredPleOnGpu(&self.ctx)) try xfm.flushDeferredPle(&self.ctx);
    st.verify_logits = verify_logits;
    st.new_hidden = new_hidden;
    st.verify_hidden_all = verify_hidden_all;
}

pub fn mtpGroupVerify(gens: []const *Generator, states: []const *MtpRoundState) !void {
    const n = gens.len;
    std.debug.assert(n == states.len);
    std.debug.assert(n >= 1 and n <= MTP_GROUP_ROWS_MAX);
    const xfm = gens[0].xfm;
    for (gens[1..n]) |gen| {
        if (gen.xfm != xfm) return error.MtpGroupModelMismatch;
    }
    var rows: [MTP_GROUP_ROWS_MAX]mlx.mlx_array = undefined;
    var ctxs: [MTP_GROUP_ROWS_MAX]*ForwardCtx = undefined;
    var out_logits: [MTP_GROUP_ROWS_MAX]mlx.mlx_array = undefined;
    var out_last: [MTP_GROUP_ROWS_MAX]mlx.mlx_array = undefined;
    var out_all: [MTP_GROUP_ROWS_MAX]mlx.mlx_array = undefined;
    for (gens[0..n], states[0..n], 0..) |gen, st, i| {
        rows[i] = st.verify_input;
        ctxs[i] = &gen.ctx;
    }
    var rollback: [MTP_GROUP_ROWS_MAX]Transformer.VerifyRollback = undefined;
    const retain_rollback = n > 1 and xfm.qwen4 != null;
    for (gens, states) |gen, st| st.verify_ctx = &gen.ctx;
    var row_ns: [MTP_GROUP_ROWS_MAX]u64 = undefined;
    var tracing_any = false;
    for (states[0..n]) |st| {
        if (st.tracing) tracing_any = true;
    }
    try xfm.forwardRowAxisVerify(rows[0..n], ctxs[0..n], out_logits[0..n], out_last[0..n], out_all[0..n], if (tracing_any) row_ns[0..n] else null, if (retain_rollback) rollback[0..n] else null);
    for (gens[0..n], states[0..n], 0..) |gen, st, i| {
        if (retain_rollback) st.verify_rollback = rollback[i];
        st.verify_logits = out_logits[i];
        st.new_hidden = out_last[i];
        st.verify_hidden_all = out_all[i];
        // Row 0 owns the joined block for the round when the lm_head ran once.
        if (i == 0) st.group_logits = xfm.takeJoinedVerifyLogits();
        if (st.tracing) {
            gen.mtp_trace.add(.verify, row_ns[i]);
            const laps = xfm.verify_laps;
            gen.mtp_trace.addSub(.build, laps.build_ns);
            gen.mtp_trace.addSub(.ple, laps.ple_ns);
            gen.mtp_trace.addSub(.ple_sync, laps.ple_sync_ns);
            st.ph.reset();
        }
    }
}

pub fn mtpGroupAccept(gens: []const *Generator, states: []const *MtpRoundState) !void {
    const n = gens.len;
    std.debug.assert(n == states.len);
    if (n == 0) return;
    var any_tracing = false;
    for (states[0..n]) |st| {
        if (st.tracing) any_tracing = true;
    }
    const dispatch_lap = SubLap.start(any_tracing, gens[0].timer.io);
    defer if (dispatch_lap.read()) |taken| {
        for (gens[0..n], states[0..n]) |gen, st| {
            if (st.tracing) gen.mtp_trace.addSub(.dispatch, taken);
        }
    };
    for (gens[1..n]) |gen| {
        if (gen.xfm != gens[0].xfm) return error.MtpGroupModelMismatch;
    }
    const s = gens[0].xfm.s;
    const eval_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(eval_vec);
    errdefer for (states[0..n]) |st| {
        st.verify_argmax.deinit();
        st.releasePreparedAccept();
    };
    for (gens[0..n], states[0..n]) |gen, st| {
        for (st.chain.draft_arrs[0..st.chain.m]) |arr| _ = mlx.mlx_vector_array_append_value(eval_vec, arr);
        _ = mlx.mlx_vector_array_append_value(eval_vec, st.new_hidden);
        _ = mlx.mlx_vector_array_append_value(eval_vec, st.verify_hidden_all);
        if (gen.sampling.temperature > 0.01) continue;
        st.verify_argmax = try verifyArgmax(st.verify_logits, gen.sampling.suppress_mask, s);
        _ = mlx.mlx_vector_array_append_value(eval_vec, st.verify_argmax.lazy());
    }
    try mtpGroupSampledAccept(gens[0..n], states[0..n], eval_vec, s);
    try mlx.check(mlx.mlx_async_eval(eval_vec));
}

/// Whether ONE filtered block can serve every sampled row of a group: the
/// filter is per REQUEST, and one categorical cannot carry a per-row key.
pub fn mtpGroupAcceptBatchable(params: []const SamplingParams) bool {
    if (params.len == 0) return false;
    return sampleRowsHomogeneous(params) and !sampleRowsAnySeed(params);
}

/// Phase 4 for the sampled rows of a group, built together: ONE
/// `probsAllPositions` over their joined verify logits — the block the
/// lm_head already projected in one go — and each row's accept graph on the
/// group's single async eval, instead of a build-wait-build staircase per
/// row. Declines (every row then builds its own in `mtpRoundFinish`) when
/// the rows do not share one sampler, any row is seeded, or a row was
/// padded past `1 + m`.
pub fn mtpGroupSampledAccept(
    gens: []const *Generator,
    states: []const *MtpRoundState,
    eval_vec: mlx.mlx_vector_array,
    s: mlx.mlx_stream,
) !void {
    if (!mtpBatchCorrEnabled()) return;
    var rows: [MTP_GROUP_ROWS_MAX]usize = undefined;
    var params: [MTP_GROUP_ROWS_MAX]SamplingParams = undefined;
    var k: usize = 0;
    for (gens, states, 0..) |gen, st, i| {
        if (gen.sampling.temperature <= 0.01 or gen.mtp_serial_accept) continue;
        if (st.chain.m == 0 or st.verify_len != 1 + st.chain.m) return;
        rows[k] = i;
        params[k] = gen.sampling;
        k += 1;
    }
    if (k == 0 or !mtpGroupAcceptBatchable(params[0..k])) return;

    // Offsets of each sampled row in the block, ascending: the projection's
    // own `offset += row.seq` walk, which is why the joined array can stand
    // in for the concatenation whenever every row of it is sampled.
    var offs: [MTP_GROUP_ROWS_MAX]c_int = undefined;
    var total: c_int = 0;
    for (rows[0..k], 0..) |i, j| {
        offs[j] = total;
        total += @intCast(states[i].verify_len);
    }

    const joined = states[0].group_logits;
    const use_joined = k == gens.len and joined.ctx != null and
        mlx.getShape(joined).len == 3 and mlx.getShape(joined)[1] == total;
    var block = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(block);
    if (use_joined) {
        try mlx.check(mlx.mlx_array_set(&block, joined));
    } else {
        var parts: [MTP_GROUP_ROWS_MAX]mlx.mlx_array = undefined;
        for (rows[0..k], 0..) |i, j| parts[j] = states[i].verify_logits;
        const vec = mlx.mlx_vector_array_new_data(&parts, k);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&block, vec, 1, s));
    }

    const probs = try groupProbsBlock(block, params[0], s);
    defer _ = mlx.mlx_array_free(probs);
    for (rows[0..k], 0..) |i, j| {
        const st = states[i];
        const row_probs = try groupProbsRow(probs, offs[j], st.verify_len, s);
        defer _ = mlx.mlx_array_free(row_probs);
        // The row's OWN acceptance mode, threshold and draw index: only the
        // filter is shared, so a seeded row keeps its own key.
        const gen = gens[i];
        const bg = try gen.acceptGraphFor(st.chain.lookup)(
            row_probs,
            st.chain.draft_arrs[0..st.chain.m],
            if (st.chain.q_probs) |qs| qs[0..st.chain.m] else null,
            st.chain.m,
            gen.mtp_accept_param,
            gen.mtpSamplingDraw(gen.sampling),
            s,
        );
        st.accept_p = bg.accept_p;
        st.accept_q = bg.accept_q;
        st.accept_defer = bg.accept_defer;
        st.corr = bg.corr_samples;
        _ = mlx.mlx_vector_array_append_value(eval_vec, st.accept_p);
        if (st.accept_q.ctx != null) _ = mlx.mlx_vector_array_append_value(eval_vec, st.accept_q);
        if (st.accept_defer.ctx != null) _ = mlx.mlx_vector_array_append_value(eval_vec, st.accept_defer);
        _ = mlx.mlx_vector_array_append_value(eval_vec, st.corr);
    }
}

/// The whole verify row as head history: ids `[t1, drafts]`, hiddens `last_hidden ++ verify[0..m]`.
pub fn mtpPaddedStash(self: *Generator, allocator: std.mem.Allocator, t1: u32, drafts: []const u32, verify_hidden_all: mlx.mlx_array) !@FieldType(MtpHistStash, "pad") {
    const rows = 1 + drafts.len;
    const ids_buf = try allocator.alloc(i32, rows);
    defer allocator.free(ids_buf);
    ids_buf[0] = @intCast(t1);
    for (drafts, ids_buf[1..]) |d, *v| v.* = @intCast(d);
    const ids = mlx.mlx_array_new_data(ids_buf.ptr, &[_]c_int{@intCast(rows)}, 1, .int32);
    errdefer _ = mlx.mlx_array_free(ids);
    var hidden = mlx.mlx_array_new();
    try self.mtpPaddedHidden(&hidden, drafts.len, verify_hidden_all);
    return .{ .ids = ids, .hidden = hidden, .rows = rows };
}

/// `last_hidden ++ verify_hidden_all[0..m]`: the hiddens paired with `[t1, drafts]`.
pub fn mtpPaddedHidden(self: *Generator, out: *mlx.mlx_array, m: usize, verify_hidden_all: mlx.mlx_array) !void {
    const s = self.xfm.s;
    const vh_shape = mlx.getShape(verify_hidden_all);
    var vh = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vh);
    try mlx.check(mlx.mlx_slice(&vh, verify_hidden_all, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ 1, @intCast(m), vh_shape[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    const parts = [_]mlx.mlx_array{ self.last_hidden, vh };
    const vec = mlx.mlx_vector_array_new_data(&parts, 2);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_concatenate_axis(out, vec, 1, s));
}

/// Phases 4–5 of a round: accept, stash the history, commit or roll back.
/// `st.verify_len` may exceed `1 + m` when a batched group padded the row;
/// then even a full accept takes the rollback arm (KV + SSM clamp to `1 + m`).
pub fn mtpStopPrefix(drafts: []const u32, accepted: u32, eos: []const u32) struct { accepted: u32, stop: ?u32 } {
    for (drafts[0..accepted], 0..) |token, i| {
        if (isEosId(token, eos)) return .{ .accepted = @intCast(i), .stop = token };
    }
    return .{ .accepted = accepted, .stop = null };
}

// ── tests ──


test "seeded MTP draft and correction draws replay despite unrelated MLX random draws" {
    const s = mlx.gpuStream();
    const q_data = [_]f32{ 0.5, 0.5, 0.0, 0.0 };
    const q_shape = [_]c_int{ 1, 4 };
    const q = mlx.mlx_array_new_data(&q_data, &q_shape, 2, .float32);
    defer _ = mlx.mlx_array_free(q);
    const p_data = [_]f32{ 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0 };
    const p_shape = [_]c_int{ 1, 2, 4 };
    const p = mlx.mlx_array_new_data(&p_data, &p_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(p);
    const d_data: i32 = 3;
    const d_shape = [_]c_int{1};
    const d = mlx.mlx_array_new_data(&d_data, &d_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(d);
    const drafts = [_]mlx.mlx_array{d};

    var runs: [2][16][3]i32 = undefined;
    for (&runs) |*run| {
        for (run, 0..) |*out, index| {
            const draft_params = SamplingParams{ .seed = 0xC0FFEE, .draw = @as(u64, @intCast(index)) * 2 };
            const lazy_draft = try sampleFromProbsLazy(q, draft_params, s);
            defer _ = mlx.mlx_array_free(lazy_draft);
            try mlx.check(mlx.mlx_array_eval(lazy_draft));
            try mlx.check(mlx.mlx_array_item_int32(&out[0], lazy_draft));

            const correction_params = SamplingParams{ .seed = 0xC0FFEE, .draw = draft_params.draw + 1 };
            var graph = try Generator.mtpBatchedAcceptGraph(p, &drafts, null, 1, correction_params, s);
            defer graph.deinit();
            try mlx.check(mlx.mlx_array_eval(graph.corr_samples));
            const corrected = mlx.mlx_array_data_int32(graph.corr_samples) orelse return error.MlxArrayDataNull;
            out[1] = corrected[0];
            out[2] = corrected[1];
        }

        // A seeded replay must not depend on unrelated requests consuming the
        // process-global MLX RNG between runs.
        const unrelated = try sampleFromProbsLazy(q, SamplingParams{}, s);
        defer _ = mlx.mlx_array_free(unrelated);
        try mlx.check(mlx.mlx_array_eval(unrelated));
    }
    try testing.expectEqualSlices([3]i32, &runs[0], &runs[1]);
    var all_same = true;
    for (runs[0][1..]) |draw| all_same = all_same and std.mem.eql(i32, &draw, &runs[0][0]);
    try testing.expect(!all_same);
}

test "mtpCommittedLen: speculative tails (pre-draft, stash) bound the committable history" {
    try testing.expectEqual(@as(usize, 42), Generator.mtpCommittedLen(42, null, null));
    // Pending stash: the cache past off0 is the producing round's stale draft tail.
    try testing.expectEqual(@as(usize, 40), Generator.mtpCommittedLen(46, null, 40));
    // Built pre-draft: next-round draft entries sit past ITS off0.
    try testing.expectEqual(@as(usize, 43), Generator.mtpCommittedLen(47, 43, null));
    // Both live: the min wins.
    try testing.expectEqual(@as(usize, 40), Generator.mtpCommittedLen(47, 43, 40));
}

test "mtpRoundOff0: a pending history stash overrides the stale cache length" {
    // No stash: the head cache is fully committed — its step IS the origin.
    try testing.expectEqual(@as(usize, 42), Generator.mtpRoundOff0(null, 42));
    // Pending stash: the cache still carries the previous round's draft tail
    // (step is stale/uncommitted), so the origin is where the stash's entries
    // will END once the consume-time truncate + merged forward run. A round
    // that read mc.step here would draft at the WRONG RoPE offsets.
    const st = Generator.MtpHistStash{
        .ids = .{ .ctx = null },
        .hidden = .{ .ctx = null },
        .n = 3, // t1 + 2 accepted drafts
        .off0 = 40,
    };
    // cache_step (46 = 40 committed + 6 stale draft entries) must be ignored.
    try testing.expectEqual(@as(usize, 43), Generator.mtpRoundOff0(st, 46));
}

test "mtpEvObserve: conditional EMA updates hit accepted indices, the reject index, and nothing past it" {
    var a = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
    // 3 drafted, 1 accepted: index 0 saw a success, index 1 saw the reject,
    // index 2 was never conditionally reached (no observation).
    Generator.mtpEvObserve(&a, 3, 1, 0.15);
    try testing.expectApproxEqAbs(@as(f32, 0.575), a[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.425), a[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.5), a[2], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.5), a[3], 1e-5);
    // Full accept: every drafted index saw a success, none saw a reject.
    var b = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
    Generator.mtpEvObserve(&b, 2, 2, 0.15);
    try testing.expectApproxEqAbs(@as(f32, 0.575), b[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.575), b[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.5), b[2], 1e-5);
}

test "mtpChainLogConf: sums clamped log-confidences; NaN can never pass a gate" {
    try testing.expectApproxEqAbs(@as(f32, -0.3), Generator.mtpChainLogConf(&[_]f32{ -0.1, -0.2 }), 1e-5);
    // Positive numeric noise clamps to 0 (a log-prob is never > 0).
    try testing.expectApproxEqAbs(@as(f32, -0.1), Generator.mtpChainLogConf(&[_]f32{ 0.05, -0.1 }), 1e-5);
    // NaN -> -inf: `chain >= tau_ln` is false for every finite tau.
    const nan_chain = Generator.mtpChainLogConf(&[_]f32{ -0.1, std.math.nan(f32) });
    try testing.expect(nan_chain == -std.math.inf(f32));
    try testing.expect(!(nan_chain >= @log(@as(f32, 0.05))));
}

// Hermetic Nemotron-H MTP round loop: a tiny hybrid trunk (Mamba2 +
// attention + MoE-with-shared-expert) on disk beside a bare-key head file. The
// bar is the spec-decode contract on THIS trunk family — greedy MTP emits
// exactly the serial greedy stream at whatever accepted counts the rounds
// produce, through the hybrid path's pre-norm hidden capture, the fused-window
// per-row SSM capture and the capture-based rollback.
test "mtp: nextMtp on a Nemotron-H trunk emits the serial stream, greedy and seeded" {
    if (mlx.noGpuBackend()) return;
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var dir_buf: [512]u8 = undefined;
    const dir_path = dir_buf[0..try tmp.dir.realPath(io, &dir_buf)];

    const H: usize = 64;
    const V: usize = 16;
    const MH: usize = 4; // mamba heads x hd 16 = d_inner 64
    const MHD: usize = 16;
    const NG: usize = 2;
    const DS: usize = 32; // state % 32 == 0: the fused decode window serves the verify rows, as live
    const KC: usize = 4;
    const CD = MH * MHD + 2 * NG * DS; // 192
    const AH: usize = 2; // attention heads x hd 32
    const AHD: usize = 32;
    const E: usize = 4;
    const I: usize = 64;
    const SI: usize = 128;

    const Fx = dflash_mod.TinyFix;
    const W = struct {
        // 4-bit/gs64 affine `<key>.weight/.scales/.biases` from a seeded f32 tensor of any rank.
        fn quant(map: mlx.mlx_map_string_to_array, alloc: std.mem.Allocator, key: []const u8, shape: []const c_int, seed: usize, st: mlx.mlx_stream) !void {
            var n: usize = 1;
            for (shape) |d| n *= @intCast(d);
            const host = try alloc.alloc(f32, n);
            defer alloc.free(host);
            var prng = std.Random.DefaultPrng.init(@intCast(seed));
            const r = prng.random();
            for (host) |*v| v.* = (r.float(f32) - 0.5) * 0.4;
            const f32w = mlx.mlx_array_new_data(host.ptr, shape.ptr, @intCast(shape.len), .float32);
            defer _ = mlx.mlx_array_free(f32w);
            // bf16 in, bf16 scales out: the row kernels' geometry, as a real pack.
            var f = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(f);
            try mlx.check(mlx.mlx_astype(&f, f32w, .bfloat16, st));
            var triple = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(triple);
            try mlx.check(mlx.mlx_quantize(&triple, f, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{}, st));
            for ([_][]const u8{ ".weight", ".scales", ".biases" }, 0..) |suf, i| {
                var arr = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(arr);
                try mlx.check(mlx.mlx_vector_array_get(&arr, triple, i));
                try mlx.check(mlx.mlx_array_eval(arr));
                const full = try std.fmt.allocPrintSentinel(alloc, "{s}{s}", .{ key, suf }, 0);
                defer alloc.free(full);
                _ = mlx.mlx_map_string_to_array_insert(map, full.ptr, arr);
            }
        }
        // Dense bf16 tensor of any rank (the conv1d weight is [C, K, 1]).
        fn bf16(map: mlx.mlx_map_string_to_array, alloc: std.mem.Allocator, key: []const u8, shape: []const c_int, seed: usize, st: mlx.mlx_stream) !void {
            var n: usize = 1;
            for (shape) |d| n *= @intCast(d);
            const host = try alloc.alloc(f32, n);
            defer alloc.free(host);
            var prng = std.Random.DefaultPrng.init(@intCast(seed));
            const r = prng.random();
            for (host) |*v| v.* = (r.float(f32) - 0.5) * 0.6;
            const f = mlx.mlx_array_new_data(host.ptr, shape.ptr, @intCast(shape.len), .float32);
            defer _ = mlx.mlx_array_free(f);
            var arr = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(arr);
            try mlx.check(mlx.mlx_astype(&arr, f, .bfloat16, st));
            try mlx.check(mlx.mlx_array_eval(arr));
            const full = try std.fmt.allocPrintSentinel(alloc, "{s}", .{key}, 0);
            defer alloc.free(full);
            _ = mlx.mlx_map_string_to_array_insert(map, full.ptr, arr);
        }
        fn f32Arr(map: mlx.mlx_map_string_to_array, alloc: std.mem.Allocator, key: []const u8, n: usize, seed: usize) !void {
            const host = try alloc.alloc(f32, n);
            defer alloc.free(host);
            var prng = std.Random.DefaultPrng.init(@intCast(seed));
            const r = prng.random();
            for (host) |*v| v.* = (r.float(f32) - 0.5) * 0.2;
            const shape = [_]c_int{@intCast(n)};
            const arr = mlx.mlx_array_new_data(host.ptr, &shape, 1, .float32);
            defer _ = mlx.mlx_array_free(arr);
            const full = try std.fmt.allocPrintSentinel(alloc, "{s}", .{key}, 0);
            defer alloc.free(full);
            _ = mlx.mlx_map_string_to_array_insert(map, full.ptr, arr);
        }
    };

    // Trunk: backbone.* keys, layers [mamba, attention, moe], quantized MoE banks.
    {
        const cfg =
            \\{
            \\  "model_type": "nemotron_h",
            \\  "hidden_size": 64, "num_hidden_layers": 3, "vocab_size": 16,
            \\  "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32,
            \\  "layers_block_type": ["mamba", "attention", "moe"],
            \\  "mamba_num_heads": 4, "mamba_head_dim": 16, "n_groups": 2, "ssm_state_size": 32,
            \\  "conv_kernel": 4, "expand": 1,
            \\  "n_routed_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 64,
            \\  "moe_shared_expert_intermediate_size": 128, "n_group": 1, "topk_group": 1,
            \\  "norm_topk_prob": true, "routed_scaling_factor": 2.5,
            \\  "layer_norm_epsilon": 1e-5, "rope_theta": 10000.0, "tie_word_embeddings": false,
            \\  "quantization": {"group_size": 64, "bits": 4}
            \\}
        ;
        try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = cfg });
        const st_path = try std.fmt.allocPrintSentinel(allocator, "{s}/model.safetensors", .{dir_path}, 0);
        defer allocator.free(st_path);
        const map = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(map);
        const meta = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta);
        try Fx.putW(map, "backbone.embeddings.weight", V, H, 1, s);
        try Fx.putW(map, "lm_head.weight", V, H, 2, s);
        try Fx.putNorm(map, "backbone.norm_f.weight", H, s);
        try Fx.putNorm(map, "backbone.layers.0.norm.weight", H, s);
        try Fx.putW(map, "backbone.layers.0.mixer.in_proj.weight", MH * MHD + CD + MH, H, 3, s);
        try W.bf16(map, allocator, "backbone.layers.0.mixer.conv1d.weight", &.{ CD, KC, 1 }, 4, s);
        try Fx.putNorm(map, "backbone.layers.0.mixer.conv1d.bias", CD, s);
        try Fx.putNorm(map, "backbone.layers.0.mixer.A_log", MH, s);
        try Fx.putNorm(map, "backbone.layers.0.mixer.D", MH, s);
        try Fx.putNorm(map, "backbone.layers.0.mixer.dt_bias", MH, s);
        try Fx.putNorm(map, "backbone.layers.0.mixer.norm.weight", MH * MHD, s);
        try Fx.putW(map, "backbone.layers.0.mixer.out_proj.weight", H, MH * MHD, 5, s);
        try Fx.putNorm(map, "backbone.layers.1.norm.weight", H, s);
        try Fx.putW(map, "backbone.layers.1.mixer.q_proj.weight", AH * AHD, H, 6, s);
        try Fx.putW(map, "backbone.layers.1.mixer.k_proj.weight", AHD, H, 7, s);
        try Fx.putW(map, "backbone.layers.1.mixer.v_proj.weight", AHD, H, 8, s);
        try Fx.putW(map, "backbone.layers.1.mixer.o_proj.weight", H, AH * AHD, 9, s);
        try Fx.putNorm(map, "backbone.layers.2.norm.weight", H, s);
        try Fx.putW(map, "backbone.layers.2.mixer.gate.weight", E, H, 10, s);
        try W.f32Arr(map, allocator, "backbone.layers.2.mixer.gate.e_score_correction_bias", E, 11);
        try W.quant(map, allocator, "backbone.layers.2.mixer.switch_mlp.fc1", &.{ E, I, H }, 12, s);
        try W.quant(map, allocator, "backbone.layers.2.mixer.switch_mlp.fc2", &.{ E, H, I }, 13, s);
        try W.quant(map, allocator, "backbone.layers.2.mixer.shared_experts.up_proj", &.{ SI, H }, 14, s);
        try W.quant(map, allocator, "backbone.layers.2.mixer.shared_experts.down_proj", &.{ H, SI }, 15, s);
        try mlx.check(mlx.mlx_save_safetensors(st_path.ptr, map, meta));
    }
    // Head: the sevren-ai bare-key layout in the trunk dir.
    {
        const st_path = try std.fmt.allocPrintSentinel(allocator, "{s}/mtp_head.safetensors", .{dir_path}, 0);
        defer allocator.free(st_path);
        const map = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(map);
        const meta = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta);
        try Fx.putW(map, "eh_proj", H, 2 * H, 21, s);
        try Fx.putNorm(map, "enorm", H, s);
        try Fx.putNorm(map, "hnorm", H, s);
        try Fx.putNorm(map, "final_norm", H, s);
        try Fx.putNorm(map, "attn.norm.weight", H, s);
        try Fx.putW(map, "attn.mixer.q_proj.weight", AH * AHD, H, 22, s);
        try Fx.putW(map, "attn.mixer.k_proj.weight", AHD, H, 23, s);
        try Fx.putW(map, "attn.mixer.v_proj.weight", AHD, H, 24, s);
        try Fx.putW(map, "attn.mixer.o_proj.weight", H, AH * AHD, 25, s);
        try Fx.putNorm(map, "moe.norm.weight", H, s);
        try Fx.putW(map, "moe.mixer.gate.weight", E, H, 26, s);
        try W.f32Arr(map, allocator, "moe.mixer.gate.e_score_correction_bias", E, 27);
        try W.quant(map, allocator, "moe.mixer.switch_mlp.fc1", &.{ E, I, H }, 28, s);
        try W.quant(map, allocator, "moe.mixer.switch_mlp.fc2", &.{ E, H, I }, 29, s);
        try W.quant(map, allocator, "moe.mixer.shared_experts.up_proj", &.{ SI, H }, 30, s);
        try W.quant(map, allocator, "moe.mixer.shared_experts.down_proj", &.{ H, SI }, 31, s);
        try mlx.check(mlx.mlx_save_safetensors(st_path.ptr, map, meta));
    }

    var config = try model_mod.parseConfig(io, allocator, dir_path);
    var weights = try model_mod.loadWeights(io, allocator, dir_path);
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    try testing.expect(mtp_mod.hasMtpHead(io, allocator, dir_path));
    // Exact mode is on only while a DFlash drafter is bound; this pins its MTP rounds.
    config.dflash_bound = true;

    var tok_dummy: Tokenizer = undefined; // never read by the Generator
    const prompt = [_]u32{ 3, 7, 1, 12, 5, 9, 4, 2, 11, 6, 14, 8 };
    const greedy = SamplingParams{ .temperature = 0.0 };
    const want: usize = 16;

    // Greedy, and a seeded sampler: a draft is kept only when the serial
    // sampler draws it at that position, so both streams are the serial one.
    for ([_]SamplingParams{ greedy, .{ .temperature = 1.0, .seed = 7 } }) |sp| {
        var serial: [want]u32 = undefined;
        {
            var xfm = try Transformer.init(io, allocator, config, &weights);
            defer xfm.deinit();
            var gen = try Generator.initWithOptions(io, allocator, &xfm, &tok_dummy, &prompt, @intCast(want), sp, &.{}, .{
                .skip_lazy_preforward = true,
            });
            defer gen.deinit(allocator);
            var n: usize = 0;
            while (n < want) {
                const t = (try gen.next(allocator)) orelse break;
                serial[n] = t;
                n += 1;
            }
            try testing.expectEqual(want, n);
        }

        {
            var xfm = try Transformer.init(io, allocator, config, &weights);
            defer xfm.deinit();
            var head = try mtp_mod.loadMtp(io, allocator, s, dir_path);
            defer head.deinit();
            try testing.expectEqual(mtp_mod.Layout.nemotron, head.layout);
            try head.bind(&xfm);

            var gen = try Generator.initWithOptions(io, allocator, &xfm, &tok_dummy, &prompt, @intCast(want), sp, &.{}, .{
                .mtp_enabled = true,
                .mtp = .{ .qwen = &head },
                .model_has_mtp = true,
                .mtp_depth = 2,
            });
            defer gen.deinit(allocator);

            var got = std.ArrayList(u32).empty;
            defer got.deinit(allocator);
            while (true) {
                const rounds_before = gen.mtp_attempted + gen.mtp_lookup_rounds;
                const res = (try gen.nextMtp(allocator)) orelse break;
                defer allocator.free(res.tokens);
                try got.appendSlice(allocator, res.tokens);
                // Every emitted token passes through a real spec round, head or lookup (no silent fallback).
                try testing.expect(gen.mtp_attempted + gen.mtp_lookup_rounds != rounds_before);
            }
            try testing.expect(gen.mtp_attempted > 0);
            try testing.expect(!gen.spec_disabled_runtime);
            try testing.expectEqual(want, got.items.len);
            for (serial, got.items) |a, b| try testing.expectEqual(a, b);
        }
    }

    // Bar: the head's attention is NoPE like the trunk's, so a history row's
    // cached key is the same at position 0 and position 1.
    {
        var xfm = try Transformer.init(io, allocator, config, &weights);
        defer xfm.deinit();
        var head = try mtp_mod.loadMtp(io, allocator, s, dir_path);
        defer head.deinit();
        try head.bind(&xfm);
        var rows: [2 * H]f32 = undefined;
        for (&rows, 0..) |*v, i| v.* = @sin(0.37 * @as(f32, @floatFromInt(i)) + 0.2);
        const kv_w = AHD; // one kv head
        const Keys = struct {
            // Cached keys `[2, kv_w]` after appending rows in `order`.
            fn run(hd: *const mtp_mod.MtpModel, x: *Transformer, all: *const [2 * H]f32, order: [2]usize, out: *[2 * kv_w]f32, st: mlx.mlx_stream) !void {
                var hist: [2 * H]f32 = undefined;
                for (order, 0..) |r, j| @memcpy(hist[j * H .. (j + 1) * H], all[r * H .. (r + 1) * H]);
                const ids = [2]i32{ 5, 9 };
                const hist_ids = [2]i32{ ids[order[0]], ids[order[1]] };
                var cache = try KVCache.init(std.testing.allocator, 1);
                defer cache.deinit();
                const h_f = mlx.mlx_array_new_data(&hist, &[_]c_int{ 1, 2, @intCast(H) }, 3, .float32);
                defer _ = mlx.mlx_array_free(h_f);
                var h_bf = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(h_bf);
                try mlx.check(mlx.mlx_astype(&h_bf, h_f, .bfloat16, st));
                const id_h = mlx.mlx_array_new_data(&hist_ids, &[_]c_int{2}, 1, .int32);
                defer _ = mlx.mlx_array_free(id_h);
                try mtp_mod.appendKvOnly(hd, x, &cache, id_h, h_bf, 0, null);
                var view = try cache.denseView(0, st);
                defer view.deinit();
                var kf = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(kf);
                try mlx.check(mlx.mlx_astype(&kf, view.k, .float32, st));
                var kc = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(kc);
                try mlx.check(mlx.mlx_contiguous(&kc, kf, false, st));
                try mlx.check(mlx.mlx_array_eval(kc));
                const p = mlx.mlx_array_data_float32(kc) orelse return error.NoData;
                @memcpy(out, p[0 .. 2 * kv_w]);
            }
        };
        var fwd: [2 * kv_w]f32 = undefined;
        var rev: [2 * kv_w]f32 = undefined;
        try Keys.run(&head, &xfm, &rows, .{ 0, 1 }, &fwd, s);
        try Keys.run(&head, &xfm, &rows, .{ 1, 0 }, &rev, s);
        // Row 0's key: position 0 in `fwd`, position 1 in `rev`.
        var max_ref: f32 = 0;
        var max_err: f32 = 0;
        for (fwd[0..kv_w], rev[kv_w..]) |a, b| {
            max_ref = @max(max_ref, @abs(a));
            max_err = @max(max_err, @abs(a - b));
        }
        if (!(max_ref > 0.01)) return error.RefTooSmall;
        if (max_err > 1e-2 * max_ref) {
            std.debug.print("MTP head key moved with its position: max_err={d:.6} max_ref={d:.6}\n", .{ max_err, max_ref });
            return error.HeadAttentionIsPositional;
        }
    }
}

test "qwen4: the coarse rerank head is READY before any draft has run" {
    // The head is a full `requantizeRows` of the trunk lm_head plus a
    // synchronous eval of its ~240 MB. Built lazily it landed inside the
    // FIRST request's draft chain — on the inference thread, mid-round,
    // draining the stream, and paid as first-token latency by whoever
    // happened to be first after a load. `qwen4BuildDraftRerank` is that
    // build as a standalone LOAD-time step: no Generator, no chain, no
    // request. Afterwards `canRerankDrafts()` is a pure read.
    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();

    var fx = try mtp_mod.RerankFixture.init(s, mtp_mod.TOP32_MIN_ROWS + 96, 256, 8, 64, 0xC0DE);
    defer fx.deinit();
    // Only the rerank fields are read by the build; the rest of the head is
    // the trunk's business.
    var head: transformer_mod.Qwen4Mtp = undefined;
    head.rerank = null;
    head.rerank_logged = false;
    head.rerank_tried = false;
    fx.xfm.qwen4_mtp = head;
    defer if (fx.xfm.qwen4_mtp) |*m| {
        if (m.rerank) |*rc| rc.deinit();
    };

    // Nothing is built by construction...
    try testing.expect(fx.xfm.qwen4_mtp.?.rerank == null);
    // ...the eager build stands on its own...
    try testing.expect(fx.xfm.qwen4BuildDraftRerank());
    const built = fx.xfm.qwen4_mtp.?.rerank orelse return error.NoCoarseHead;
    try testing.expectEqual(fx.vocab, built.rows);
    try testing.expect(built.bits > 0 and built.bits < 8);

    // ...and the draft path now only READS it. Same mlx handle => the ask
    // did not rebuild anything (which is what the lazy path did).
    const head_ctx = built.q.w.ctx;
    const ref = MtpHeadRef{ .qwen4 = &fx.xfm };
    try testing.expect(ref.canRerankDrafts());
    try testing.expectEqual(head_ctx, fx.xfm.qwen4_mtp.?.rerank.?.q.w.ctx);

    // One-shot: a second eager build (both load paths firing, a re-probe)
    // keeps the head it already has.
    try testing.expect(fx.xfm.qwen4BuildDraftRerank());
    try testing.expectEqual(head_ctx, fx.xfm.qwen4_mtp.?.rerank.?.q.w.ctx);
}

test "mtpPreDraftOpenAllows: an eager pre-draft needs a reader, planner-owned or not" {
    const G = Generator;
    // One token under budget: the next round reads the chain, so open.
    try testing.expect(G.mtpPreDraftOpenAllows(9, 10, false));
    // Exactly at budget: the request ends on length, no later round reads it.
    try testing.expect(!G.mtpPreDraftOpenAllows(10, 10, false));
    // Over budget: same.
    try testing.expect(!G.mtpPreDraftOpenAllows(11, 10, false));
    // A pending EOS ends the request even with budget left, so no reader.
    try testing.expect(!G.mtpPreDraftOpenAllows(5, 10, true));
}

test "group acceptance: every row's verdict comes from its OWN am, at any accept position" {
    const accept = Generator.mtpGreedyVerdict;
    {
        const am = [_]i32{ 11, 22, 33 };
        const drafts = [_]u32{ 11, 22 };
        const v = accept(&am, &drafts, 0, 400);
        try testing.expectEqual(@as(u32, 2), v.accepted);
        try testing.expectEqual(@as(u32, 33), v.next_pending);
    }
    {
        const am = [_]i32{ 11, 99, 33 };
        const drafts = [_]u32{ 11, 22 };
        const v = accept(&am, &drafts, 0, 400);
        try testing.expectEqual(@as(u32, 1), v.accepted);
        try testing.expectEqual(@as(u32, 99), v.next_pending);
    }
    {
        const am = [_]i32{ 7, 22, 33 };
        const drafts = [_]u32{ 11, 22 };
        const v = accept(&am, &drafts, 0, 400);
        try testing.expectEqual(@as(u32, 0), v.accepted);
        try testing.expectEqual(@as(u32, 7), v.next_pending);
    }
    {
        const am = [_]i32{ 11, 22, 33, 44, 55 };
        const drafts = [_]u32{ 11, 22 };
        const v = accept(&am, &drafts, 0, 400);
        try testing.expectEqual(@as(u32, 2), v.accepted);
        try testing.expectEqual(@as(u32, 33), v.next_pending);
    }
    {
        const am = [_]i32{ 11, 22, 33, 44 };
        const drafts = [_]u32{ 11, 22, 33 };
        const full = accept(&am, &drafts, 0, 400);
        try testing.expectEqual(@as(u32, 3), full.accepted);
        try testing.expectEqual(@as(u32, 44), full.next_pending);
        const clipped = accept(&am, &drafts, 397, 400);
        try testing.expectEqual(@as(u32, 2), clipped.accepted);
        try testing.expectEqual(@as(u32, 33), clipped.next_pending);
        const none = accept(&am, &drafts, 399, 400);
        try testing.expectEqual(@as(u32, 0), none.accepted);
        try testing.expectEqual(@as(u32, 11), none.next_pending);
    }
    {
        const am = [_]i32{42};
        const v = accept(&am, &.{}, 0, 400);
        try testing.expectEqual(@as(u32, 0), v.accepted);
        try testing.expectEqual(@as(u32, 42), v.next_pending);
    }
}

test "batched MTP draft rows stop at their own depth and allocation" {
    var chains: [3]Generator.MtpPreDraft = undefined;
    var wide: [5]mlx.mlx_array = undefined;
    var short: [1]mlx.mlx_array = undefined;
    var middle: [3]mlx.mlx_array = undefined;
    chains[0].m = 4;
    chains[0].draft_arrs = &wide;
    chains[1].m = 1;
    chains[1].draft_arrs = &short;
    chains[2].m = 3;
    chains[2].draft_arrs = &middle;
    var indices: [3]usize = undefined;
    const expected = [_][]const usize{ &.{ 0, 1, 2 }, &.{ 0, 2 }, &.{ 0, 2 }, &.{0}, &.{}, &.{} };
    for (expected, 0..) |want, step| {
        const n = Generator.mtpDraftActiveRows(&chains, @intCast(step), &indices);
        try testing.expectEqualSlices(usize, want, indices[0..n]);
        for (indices[0..n]) |row| try testing.expect(step < chains[row].draft_arrs.len);
    }
}

test "batched MTP draft rows stop without touching inactive generators" {
    var gens: [2]Generator = undefined;
    var ptrs = [_]*Generator{ &gens[0], &gens[1] };
    var chains: [2]Generator.MtpPreDraft = undefined;
    chains[0].m = 4;
    chains[0].n_drafted = 4;
    chains[1].m = 1;
    chains[1].n_drafted = 1;
    try Generator.mtpChainBuildBatched(&ptrs, &chains, 4, 9);
    try testing.expectEqual(@as(u32, 4), chains[0].n_drafted);
    try testing.expectEqual(@as(u32, 1), chains[1].n_drafted);
}

test "MTP continuation releases its owned chain when drafting fails" {
    const allocator = testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var xfm: Transformer = undefined;
    xfm.rht = null;
    xfm.s = s;
    xfm.qwen4_mtp = null;
    var state: Transformer.Qwen4MtpState = undefined;
    var gen: Generator = undefined;
    gen.xfm = &xfm;
    gen.mtp = .{ .qwen4 = &xfm };
    gen.mtp_cache = .{ .qwen4 = .{ .t = &xfm, .st = &state, .allocator = allocator } };
    gen.sampling = .{ .temperature = 0 };
    gen.ctx.mrope_pos = null;
    gen.mtp_hist_stash = null;
    gen.last_hidden = .{ .ctx = null };
    const chain: Generator.MtpPreDraft = .{
        .plan = .{ .m_lo = 1, .m_hi = 2, .tau_ln = 0 },
        .off0 = 0,
        .t1 = 1,
        .t1_arr = mlx.mlx_array_new(),
        .drafts = try allocator.alloc(u32, 2),
        .draft_arrs = try allocator.alloc(mlx.mlx_array, 2),
        .n_drafted = 0,
        .conf_arrs = try allocator.alloc(mlx.mlx_array, 1),
        .n_conf = 0,
        .q_probs = try allocator.alloc(mlx.mlx_array, 2),
        .n_qp = 0,
        .h_chain = null,
        .m = 1,
    };
    const open: Generator.MtpRoundOpen = .{
        .chain = chain,
        .tracing = false,
        .ph = undefined,
        .livecost = false,
        .round_watch = undefined,
    };
    try testing.expectError(error.NoMtpHead, gen.mtpRoundContinue(allocator, open));
}

test "batched MTP state includes the last row activated by round begin" {
    const allocator = testing.allocator;
    var xfm: Transformer = undefined;
    xfm.rht = null;
    xfm.qwen4_mtp_owner = null;
    var head: transformer_mod.Qwen4Mtp = undefined;
    head.cache = try KVCache.init(allocator, 1);
    head.entry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true };
    head.qsa_marks = .{};
    head.seq_offset = 0;
    head.pos_base = 0;
    xfm.qwen4_mtp = head;
    defer xfm.qwen4_mtp.?.cache.deinit();
    var a: Transformer.Qwen4MtpState = .{ .cache = try KVCache.init(allocator, 1), .entry = head.entry };
    defer a.deinit();
    var b: Transformer.Qwen4MtpState = .{ .cache = try KVCache.init(allocator, 1), .entry = head.entry };
    defer b.deinit();
    xfm.qwen4MtpActivate(&a);
    xfm.qwen4_mtp.?.seq_offset = 7;
    xfm.qwen4_mtp.?.cache.step = 7;
    xfm.qwen4_mtp.?.pos_base = 3;
    xfm.qwen4MtpActivate(&b);
    xfm.qwen4_mtp.?.seq_offset = 2;
    xfm.qwen4_mtp.?.cache.step = 2;
    xfm.qwen4_mtp.?.pos_base = 11;
    const caches = [_]MtpCacheRef{
        .{ .qwen4 = .{ .t = &xfm, .st = &a, .allocator = allocator } },
        .{ .qwen4 = .{ .t = &xfm, .st = &b, .allocator = allocator } },
    };
    const want = [_]Transformer.Qwen4MtpState.Meta{
        .{ .stash_len = 7, .origin = 3, .cache_step = 7 },
        .{ .stash_len = 2, .origin = 11, .cache_step = 2 },
    };
    for (caches, want) |cache, meta| {
        const state = cache.qwen4StateForBatch();
        try testing.expectEqual(meta, state.meta());
    }
    try testing.expect(xfm.qwen4_mtp_owner == null);
}

test "Generator rounds preserve acceptance and next-round stashes at N=2/4" {
    const model_dir = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    // Group builders keep the merged history step; the solo arm must draft the same way.
    chain_mod.mtp_padded_head_override = false;
    defer chain_mod.mtp_padded_head_override = null;
    const predraft = trace_mod.mtp_predraft_cache;
    const forced = depth_bounds.active;
    const planner = group_planner.enabled_override;
    defer {
        trace_mod.mtp_predraft_cache = predraft;
        depth_bounds.active = forced;
        group_planner.enabled_override = planner;
    }
    // Observe the completed-round stash before any successor consumes it.
    trace_mod.mtp_predraft_cache = false;
    group_planner.enabled_override = true;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const coarse_before = transformer_mod.mtp_coarse_pair_dispatches;
    const verify_kernel_before = transformer_mod.mtp_verify_kernel_calls;
    var config = try model_mod.parseConfig(io, a, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |path| a.free(path);
    var weights = try model_mod.loadWeights(io, a, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, a, config, &weights);
    defer xfm.deinit();
    xfm.compileQwen4Hc();
    xfm.compileGdnGate();
    xfm.compileMoeRouting();
    var tok = Tokenizer.initEmptyForTests(a, .byte_level_bpe);
    defer tok.deinit();
    var prompt: [96]u32 = undefined;
    for (&prompt, 0..) |*id, i| id.* = @intCast(1 + (i * 7919) % 40000);
    for ([_]usize{ 2, 4 }) |n| {
        for ([_]u32{ 1, 2, 4 }) |depth| {
            depth_bounds.active = .{ .min = depth, .max = depth };
            var solo: [4]?*MtpChainTestSlot = @splat(null);
            var batch: [4]?*MtpChainTestSlot = @splat(null);
            defer for (solo, batch) |ss, bs| {
                if (ss) |x| x.deinit(a);
                if (bs) |x| x.deinit(a);
            };
            for (0..n) |i| {
                const ids = prompt[0..if (i % 2 == 0) @as(usize, 96) else 61];
                solo[i] = try MtpChainTestSlot.init(a, io, &xfm, &tok, ids, depth);
                batch[i] = try MtpChainTestSlot.init(a, io, &xfm, &tok, ids, depth);
            }
            for (0..3) |round| {
                var ss: [4]?Generator.MtpRoundState = @splat(null);
                var results: [4]?Generator.DrafterStepResult = @splat(null);
                defer for (&ss, results) |*state, result| {
                    if (state.*) |*v| v.deinit(a);
                    if (result) |r| a.free(r.tokens);
                };
                for (0..n) |i| {
                    const g = &solo[i].?.gen.?;
                    const open = try MtpChainTestSlot.begin(g, a, depth, i % 2 == 1);
                    ss[i] = try g.mtpRoundContinue(a, open);
                    try g.mtpRoundVerify(&ss[i].?);
                }
                var opens: [4]Generator.MtpRoundOpen = undefined;
                var chains: [4]Generator.MtpPreDraft = undefined;
                var owns: [4]bool = @splat(false);
                var gens: [4]*Generator = undefined;
                defer for (&chains, owns) |*chain, owned| {
                    if (owned) chain.deinit(a);
                };
                for (0..n) |i| {
                    gens[i] = &batch[i].?.gen.?;
                    opens[i] = try MtpChainTestSlot.begin(gens[i], a, depth, i % 2 == 1);
                    chains[i] = opens[i].chain;
                    owns[i] = true;
                }
                try Generator.mtpChainBuildBatched(gens[0..n], chains[0..n], 0, depth);
                var batch_states: [4]?Generator.MtpRoundState = @splat(null);
                defer for (&batch_states) |*state| {
                    if (state.*) |*p| p.deinit(a);
                };
                var batch_ptrs: [4]*Generator.MtpRoundState = undefined;
                for (0..n) |i| {
                    opens[i].chain = chains[i];
                    owns[i] = false;
                    batch_states[i] = try gens[i].mtpRoundContinue(a, opens[i]);
                    batch_ptrs[i] = &batch_states[i].?;
                }
                const grouped_verify = @import("../mtp_group_planner.zig").enabled();
                if (grouped_verify) {
                    try Generator.mtpGroupVerify(gens[0..n], batch_ptrs[0..n]);
                    try Generator.mtpGroupAccept(gens[0..n], batch_ptrs[0..n]);
                } else {
                    for (gens[0..n], batch_ptrs[0..n]) |g, state| try g.mtpRoundVerify(state);
                }
                for (0..n) |i| {
                    const bs = batch_ptrs[i];
                    std.debug.print("[mtp compare] N={d} depth={d} round={d} row={d}\n", .{ n, depth, round, i });
                    try testing.expectEqual(depth, bs.chain.m);
                    for (ss[i].?.chain.draft_arrs[0..depth], bs.chain.draft_arrs[0..depth]) |x, y| try MtpChainTestSlot.equal(x, y, xfm.s, "x");
                    try MtpChainTestSlot.equal(ss[i].?.verify_logits, bs.verify_logits, xfm.s, "ss[i].?.verify_logits");
                    const sg = &solo[i].?.gen.?;
                    results[i] = try sg.mtpRoundFinish(a, &ss[i].?);
                    try testing.expect(results[i] != null);
                    const result = (try gens[i].mtpRoundFinish(a, bs)) orelse return error.MtpRoundDeclined;
                    defer a.free(result.tokens);
                    try testing.expectEqualSlices(u32, results[i].?.tokens, result.tokens);
                    try testing.expectEqual(ss[i].?.retained, bs.retained);
                    try testing.expectEqual(sg.next_token_id, gens[i].next_token_id);
                    try testing.expectEqual(sg.mtp_accepted_tokens, gens[i].mtp_accepted_tokens);
                    try testing.expectEqual(solo[i].?.off, batch[i].?.off);
                    try MtpChainTestSlot.equal(sg.last_hidden, gens[i].last_hidden, xfm.s, "sg.last_hidden");
                    const sh = sg.mtp_hist_stash orelse return error.MissingSoloStash;
                    const bh = gens[i].mtp_hist_stash orelse return error.MissingBatchStash;
                    try testing.expectEqual(sh.off0, bh.off0);
                    try MtpChainTestSlot.equal(sh.ids, bh.ids, xfm.s, "sh.ids");
                    try MtpChainTestSlot.equal(sh.hidden, bh.hidden, xfm.s, "sh.hidden");
                    const hs = sg.mtp_cache.?.qwen4StateForBatch();
                    const hb = gens[i].mtp_cache.?.qwen4StateForBatch();
                    try testing.expectEqual(hs.meta(), hb.meta());
                    try MtpChainTestSlot.equal(hs.entry.aux_state, hb.entry.aux_state, xfm.s, "hs.entry.aux_state");
                    try MtpChainTestSlot.equal(hs.entry.qsa_pooled, hb.entry.qsa_pooled, xfm.s, "hs.entry.qsa_pooled");
                    for (solo[i].?.entries, batch[i].?.entries) |se, be| {
                        try MtpChainTestSlot.equal(se.conv_state, be.conv_state, xfm.s, "se.conv_state");
                        try MtpChainTestSlot.equal(se.ssm_state, be.ssm_state, xfm.s, "se.ssm_state");
                        try MtpChainTestSlot.equal(se.aux_state, be.aux_state, xfm.s, "se.aux_state");
                        try MtpChainTestSlot.equal(se.qsa_pooled, be.qsa_pooled, xfm.s, "se.qsa_pooled");
                        try testing.expectEqual(se.ple_prev_valid, be.ple_prev_valid);
                        if (se.ple_prev_valid) try testing.expectEqualSlices(u32, &se.ple_prev, &be.ple_prev);
                    }
                    std.debug.print("[mtp round] N={d} depth={d} round={d} row={d} accepted={d} stash={d}\n", .{ n, depth, round, i, bs.retained - 1, mlx.mlx_array_size(bh.ids) });
                }
            }
        }
    }
    if (transformer_mod.verifySharedHardware()) {
        try testing.expect(transformer_mod.mtp_coarse_pair_dispatches > coarse_before);
        inline for (.{ .dense_tiles, .hc_rows, .route_pack }, 0..) |lever, i| {
            {
                try testing.expect(transformer_mod.mtp_verify_kernel_calls[i] > verify_kernel_before[i]);
                std.debug.print("[verify kernel] {s} calls={d}\n", .{ @tagName(lever), transformer_mod.mtp_verify_kernel_calls[i] - verify_kernel_before[i] });
            }
        }
    }
}

test "acceptance failure restores every unfinished row after verify returns" {
    const model_dir = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try model_mod.parseConfig(io, a, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |path| a.free(path);
    var weights = try model_mod.loadWeights(io, a, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, a, config, &weights);
    defer xfm.deinit();
    xfm.compileQwen4Hc();
    xfm.compileGdnGate();
    xfm.compileMoeRouting();
    var tok = Tokenizer.initEmptyForTests(a, .byte_level_bpe);
    defer tok.deinit();
    var prompt: [96]u32 = undefined;
    for (&prompt, 0..) |*id, i| id.* = @intCast(1 + (i * 7919) % 40000);
    var slots: [2]?*MtpChainTestSlot = @splat(null);
    defer for (slots) |slot| {
        if (slot) |p| p.deinit(a);
    };
    var states: [2]?Generator.MtpRoundState = @splat(null);
    defer for (&states) |*state| {
        if (state.*) |*p| p.deinit(a);
    };
    var gens: [2]*Generator = undefined;
    var ptrs: [2]*Generator.MtpRoundState = undefined;
    var before: [2]usize = undefined;
    var old_keys: [2]mlx.mlx_array = @splat(.{ .ctx = null });
    var layers: [2]usize = undefined;
    var saved_ssm: [2][]transformer_mod.SSMCacheEntrySnapshot = undefined;
    var saved_rows: usize = 0;
    defer for (saved_ssm[0..saved_rows]) |snapshots| {
        for (snapshots) |*snapshot| transformer_mod.ssmSnapshotDeinit(snapshot);
        a.free(snapshots);
    };
    const Prefix = struct {
        fn equal(left: mlx.mlx_array, right: mlx.mlx_array, len: usize, s: mlx.mlx_stream) !void {
            const lsh = mlx.getShape(left);
            const rsh = mlx.getShape(right);
            var left_slice = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(left_slice);
            var right_slice = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(right_slice);
            try mlx.check(mlx.mlx_slice(&left_slice, left, &.{ 0, 0, 0, 0 }, 4, &.{ lsh[0], lsh[1], @intCast(len), lsh[3] }, 4, &.{ 1, 1, 1, 1 }, 4, s));
            try mlx.check(mlx.mlx_slice(&right_slice, right, &.{ 0, 0, 0, 0 }, 4, &.{ rsh[0], rsh[1], @intCast(len), rsh[3] }, 4, &.{ 1, 1, 1, 1 }, 4, s));
            try MtpChainTestSlot.equal(left_slice, right_slice, s, "KV valid prefix");
        }
        fn state(saved: []const transformer_mod.SSMCacheEntrySnapshot, current: []const transformer_mod.SSMCacheEntry, s: mlx.mlx_stream) !void {
            try testing.expectEqual(saved.len, current.len);
            for (saved, current) |snapshot, entry| {
                try testing.expectEqual(snapshot.initialized, entry.initialized);
                try MtpChainTestSlot.equal(snapshot.conv_state, entry.conv_state, s, "restored conv state");
                try MtpChainTestSlot.equal(snapshot.ssm_state, entry.ssm_state, s, "restored SSM state");
                try MtpChainTestSlot.equal(snapshot.aux_state, entry.aux_state, s, "restored QSA/PLE state");
                try MtpChainTestSlot.equal(snapshot.qsa_pooled, entry.qsa_pooled, s, "restored QSA pool");
                try testing.expectEqual(snapshot.qsa_ratio, entry.qsa_ratio);
                try testing.expectEqual(snapshot.ple_prev_valid, entry.ple_prev_valid);
                try testing.expectEqualSlices(u32, &snapshot.ple_prev, &entry.ple_prev);
            }
        }
    };
    defer for (old_keys) |keys| {
        if (keys.ctx != null) _ = mlx.mlx_array_free(keys);
    };
    for (0..2) |i| {
        slots[i] = try MtpChainTestSlot.init(a, io, &xfm, &tok, prompt[0..if (i == 0) @as(usize, 96) else 61], 1);
        const slot = slots[i].?;
        gens[i] = &slot.gen.?;
        states[i] = try gens[i].mtpRoundContinue(a, try MtpChainTestSlot.begin(gens[i], a, 1, false));
        ptrs[i] = &states[i].?;
        before[i] = slot.off;
        for (slot.cache.entries, 0..) |entry, layer| {
            if (!entry.initialized) continue;
            layers[i] = layer;
            old_keys[i] = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&old_keys[i], entry.keys));
            break;
        }
        try testing.expect(old_keys[i].ctx != null);
        saved_ssm[i] = try a.alloc(transformer_mod.SSMCacheEntrySnapshot, slot.entries.len);
        for (slot.entries, saved_ssm[i]) |*entry, *snapshot| snapshot.* = transformer_mod.ssmSnapshot(entry);
        saved_rows += 1;
    }
    const syncs = transformer_mod.deferred_ple_group_syncs;
    try Generator.mtpGroupVerify(&gens, &ptrs);
    try testing.expectEqual(@as(u64, 1), transformer_mod.deferred_ple_group_syncs - syncs);
    for (states) |state| try testing.expect(state.?.verify_rollback != null);
    mlx.fault.arm(1);
    const result = Generator.mtpGroupAccept(&gens, &ptrs);
    mlx.fault.disarm();
    try testing.expect(mlx.fault.didFire());
    try testing.expectError(error.MlxError, result);
    for (&states, 0..) |*state, i| {
        state.*.?.deinit(a);
        state.* = null;
        const slot = slots[i].?;
        try testing.expectEqual(before[i], slot.off);
        try testing.expectEqual(before[i], slot.cache.entries[layers[i]].offset);
        try Prefix.equal(old_keys[i], slot.cache.entries[layers[i]].keys, before[i], xfm.s);
        try Prefix.state(saved_ssm[i], slot.entries, xfm.s);
        for (slot.entries) |entry| {
            try testing.expectEqual(@as(u32, 0), entry.spec_ple_len);
            try testing.expect(entry.spec_state_seq.ctx == null);
        }
    }
    for (0..2) |i| {
        states[i] = try gens[i].mtpRoundContinue(a, try MtpChainTestSlot.begin(gens[i], a, 1, false));
        ptrs[i] = &states[i].?;
    }
    try Generator.mtpGroupVerify(&gens, &ptrs);
    try Generator.mtpGroupAccept(&gens, &ptrs);
    try mlx.check(mlx.mlx_synchronize(xfm.s));
    for (&states, 0..) |*state, i| {
        state.*.?.deinit(a);
        state.* = null;
        const slot = slots[i].?;
        try testing.expectEqual(before[i], slot.off);
        try testing.expectEqual(before[i], slot.cache.entries[layers[i]].offset);
        try Prefix.equal(old_keys[i], slot.cache.entries[layers[i]].keys, before[i], xfm.s);
        try Prefix.state(saved_ssm[i], slot.entries, xfm.s);
        for (slot.entries) |entry| {
            try testing.expectEqual(@as(u32, 0), entry.spec_ple_len);
            try testing.expect(entry.spec_state_seq.ctx == null);
        }
    }
}

test "MTP commit stops before an accepted EOS and keeps EOS pending" {
    const drafts = [_]u32{ 11, 2, 12, 3 };
    const eos = [_]u32{ 2, 3 };
    const stop = Generator.mtpStopPrefix(&drafts, 4, &eos);
    try testing.expectEqual(@as(u32, 1), stop.accepted);
    try testing.expectEqual(@as(?u32, 2), stop.stop);
    try testing.expectEqual(@as(u32, 1), Generator.mtpStopPrefix(&drafts, 2, &eos).accepted);
    const rejected_eos = Generator.mtpStopPrefix(&drafts, 1, &eos);
    try testing.expectEqual(@as(u32, 1), rejected_eos.accepted);
    try testing.expect(rejected_eos.stop == null);
    try testing.expectEqual(@as(u32, 0), Generator.mtpStopPrefix(&drafts, 0, &eos).accepted);
    try testing.expectEqual(@as(u32, 4), Generator.mtpStopPrefix(&drafts, 4, &.{}).accepted);
}

test "selected depth bypasses legacy planning and keeps acceptance private" {
    const previous = group_planner.enabled_override;
    group_planner.enabled_override = true;
    defer group_planner.enabled_override = previous;
    var gen: Generator = undefined;
    gen.mtp_planner_width = 2;
    gen.mtp_planner_owned = true;
    gen.mtp_planner_probe = true;
    gen.mtp_planner_probes = 0;
    gen.mtp_planner_recovery = .{};
    gen.mtp_planner_recovering = false;
    gen.completion_tokens = 10;
    gen.mtp_planner_spec_rounds = 0;
    gen.mtp_planner_history = .{};
    gen.mtp_hidden_stale = true;
    const plan = gen.mtpRoundPlan();
    try testing.expectEqual(@as(u32, 2), plan.m_lo);
    try testing.expectEqual(plan.m_lo, plan.m_hi);
    try testing.expect(!plan.width_trial);
    gen.mtpPlannerObserve(2, 1);
    try testing.expectEqual(@as(u16, 1), gen.mtp_planner_history.accepted[0]);
    try testing.expectEqual(@as(u16, 0), gen.mtp_planner_history.accepted[1]);
    try testing.expectEqual(@as(u8, 1), gen.mtp_planner_probes);
    try testing.expect(!gen.mtp_hidden_stale);
    gen.mtp_planner_recovering = true;
    gen.completion_tokens = 12;
    gen.mtpPlannerObserve(2, 2);
    try testing.expectEqual(@as(u8, 1), gen.mtp_planner_probes);
    try testing.expectEqual(@as(u8, 1), gen.mtp_planner_recovery.rounds);
    try testing.expectEqual(@as(u32, 12), gen.mtp_planner_recovery.last_spec_tokens);
}

test "MTP detaching a pre-draft preserves the committed head boundary" {
    const allocator = testing.allocator;
    const stream = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    for ([_]bool{ false, true }) |owned| {
        for ([_]bool{ false, true }) |apply_stash| {
            var xfm: Transformer = undefined;
            xfm.rht = null;
            xfm.s = stream;
            var gen: Generator = undefined;
            gen.xfm = &xfm;
            gen.mtp_planner_owned = owned;
            gen.mtp_cache = .{ .qwen = try KVCache.init(allocator, 0) };
            defer gen.mtp_cache.?.deinit();
            gen.mtp_cache.?.qwen.step = 9;
            gen.mtp_hist_stash = null;
            const arrays = try allocator.alloc(mlx.mlx_array, 3);
            for (arrays) |*array| array.* = mlx.mlx_array_new();
            gen.mtp_pre_draft = .{
                .plan = .{ .m_lo = 3, .m_hi = 3, .tau_ln = 0 },
                .off0 = 6,
                .t1 = 1,
                .t1_arr = mlx.mlx_array_new(),
                .drafts = try allocator.alloc(u32, 3),
                .draft_arrs = arrays,
                .n_drafted = 3,
                .conf_arrs = null,
                .n_conf = 0,
                .q_probs = null,
                .n_qp = 0,
                .h_chain = null,
                .m = 3,
            };
            defer if (gen.mtp_pre_draft) |*draft| draft.deinit(allocator);
            const committed = gen.mtpCommittedHistoryLen();
            try gen.mtpDetachHead(allocator, apply_stash);
            try testing.expect(gen.mtp_pre_draft == null);
            try testing.expectEqual(@as(usize, 6), committed);
            try testing.expectEqual(if (owned) committed else @as(usize, 9), gen.mtpCommittedHistoryLen());
        }
    }
}

const L11PlainReplay = struct {
    fn tick(xfm: *Transformer, slots: []const *MtpChainTestSlot, capture: bool) !void {
        const a = testing.allocator;
        var tokens: [4]u32 = undefined;
        var offsets: [4]u32 = undefined;
        var contexts: [4]*ForwardCtx = undefined;
        var params: [4]SamplingParams = undefined;
        for (slots, 0..) |slot, i| {
            const gen = &slot.gen.?;
            tokens[i] = gen.next_token_id;
            offsets[i] = @intCast(slot.off);
            contexts[i] = &gen.ctx;
            params[i] = gen.sampling;
        }
        var hidden: ?[]mlx.mlx_array = null;
        defer if (hidden) |rows| {
            for (rows) |row| _ = mlx.mlx_array_free(row);
            a.free(rows);
        };
        const logits = try xfm.forwardMoeBatchedDecode(tokens[0..slots.len], contexts[0..slots.len], offsets[0..slots.len], if (capture) &hidden else null);
        defer {
            for (logits) |row| _ = mlx.mlx_array_free(row);
            a.free(logits);
        }
        var sampled: [4]i32 = undefined;
        try sampleRows(sampled[0..slots.len], logits, params[0..slots.len], xfm.s);
        for (slots, 0..) |slot, i| {
            const gen = &slot.gen.?;
            try gen.generated_ids.append(a, gen.next_token_id);
            gen.advanceStep(1);
            gen.next_token_id = @intCast(sampled[i]);
            gen.sampling.draw = params[i].draw;
            slot.off += 1;
            gen.mtp_hidden_stale = !capture;
            if (hidden) |rows| {
                if (gen.has_last_hidden) _ = mlx.mlx_array_free(gen.last_hidden);
                gen.last_hidden = rows[i];
                rows[i] = .{ .ctx = null };
                gen.has_last_hidden = true;
            }
        }
    }
};

test "plain to prime resumes the same MTP history and next draft as legacy hidden capture" {
    const path = std.c.getenv("MTP_TEST_MODEL") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try model_mod.parseConfig(io, a, std.mem.span(path));
    defer if (config.ngram_table_path) |name| a.free(name);
    var weights = try model_mod.loadWeights(io, a, std.mem.span(path));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, a, config, &weights);
    defer xfm.deinit();
    xfm.compileQwen4Hc();
    xfm.compileGdnGate();
    xfm.compileMoeRouting();
    var tok = Tokenizer.initEmptyForTests(a, .byte_level_bpe);
    defer tok.deinit();
    const previous = group_planner.enabled_override;
    group_planner.enabled_override = true;
    defer group_planner.enabled_override = previous;
    var prompt: [96]u32 = undefined;
    for (&prompt, 0..) |*id, i| id.* = @intCast(1 + (i * 7919) % 40000);
    var reference: [2]*MtpChainTestSlot = undefined;
    var planned: [2]*MtpChainTestSlot = undefined;
    var built: usize = 0;
    defer for (reference[0..built], planned[0..built]) |ref, candidate| {
        ref.deinit(a);
        candidate.deinit(a);
    };
    for (0..2) |row| {
        const ids = prompt[0..if (row == 0) @as(usize, 96) else 61];
        reference[row] = try MtpChainTestSlot.init(a, io, &xfm, &tok, ids, 2);
        planned[row] = MtpChainTestSlot.init(a, io, &xfm, &tok, ids, 2) catch |err| {
            reference[row].deinit(a);
            return err;
        };
        built += 1;
        for ([_]*MtpChainTestSlot{ reference[row], planned[row] }) |slot| {
            const gen = &slot.gen.?;
            gen.mtp_planner_owned = true;
            gen.mtp_planner_width = 2;
            gen.mtp_batch_head = true;
            const result = (try gen.nextMtp(a)) orelse return error.MissingInitialMtpRound;
            a.free(result.tokens);
            try testing.expect(gen.mtp_hist_stash != null);
            try gen.mtpDetachHead(a, true);
        }
    }
    for (0..48) |_| {
        try L11PlainReplay.tick(&xfm, &reference, true);
        try L11PlainReplay.tick(&xfm, &planned, false);
    }
    for (planned) |slot| try testing.expectError(error.MtpHiddenStale, slot.gen.?.mtpRoundBegin(a));
    try L11PlainReplay.tick(&xfm, &reference, true);
    try L11PlainReplay.tick(&xfm, &planned, true);
    for (reference, planned) |ref, candidate| {
        const rg = &ref.gen.?;
        const cg = &candidate.gen.?;
        try testing.expectEqual(rg.next_token_id, cg.next_token_id);
        try testing.expectEqual(ref.off, candidate.off);
        try MtpChainTestSlot.equal(rg.last_hidden, cg.last_hidden, xfm.s, "resumed hidden");
        const rh = rg.mtp_cache.?.qwen4StateForBatch();
        const ch = cg.mtp_cache.?.qwen4StateForBatch();
        try testing.expectEqual(rh.meta(), ch.meta());
        try MtpChainTestSlot.equal(rh.entry.aux_state, ch.entry.aux_state, xfm.s, "resumed head keys");
        try MtpChainTestSlot.equal(rh.entry.qsa_pooled, ch.entry.qsa_pooled, xfm.s, "resumed head pool");
        var rs = try rg.mtpRoundContinue(a, try MtpChainTestSlot.begin(rg, a, 2, false));
        defer rs.deinit(a);
        var cs = try cg.mtpRoundContinue(a, try MtpChainTestSlot.begin(cg, a, 2, false));
        defer cs.deinit(a);
        for (rs.chain.draft_arrs[0..2], cs.chain.draft_arrs[0..2]) |left, right| {
            try MtpChainTestSlot.equal(left, right, xfm.s, "resumed draft ids");
        }
        try rg.mtpRoundVerify(&rs);
        try cg.mtpRoundVerify(&cs);
        try MtpChainTestSlot.equal(rs.verify_logits, cs.verify_logits, xfm.s, "resumed verify logits");
        rg.mtp_batch_head = true;
        cg.mtp_batch_head = true;
        const rr = (try rg.mtpRoundFinish(a, &rs)) orelse return error.MissingReferenceRound;
        defer a.free(rr.tokens);
        const cr = (try cg.mtpRoundFinish(a, &cs)) orelse return error.MissingPlannedRound;
        defer a.free(cr.tokens);
        try testing.expectEqualSlices(u32, rr.tokens, cr.tokens);
        try testing.expectEqual(rr.accepted_tokens, cr.accepted_tokens);
        try testing.expectEqual(rg.next_token_id, cg.next_token_id);
        try MtpChainTestSlot.equal(rg.mtp_hist_stash.?.ids, cg.mtp_hist_stash.?.ids, xfm.s, "next draft stash ids");
        try MtpChainTestSlot.equal(rg.mtp_hist_stash.?.hidden, cg.mtp_hist_stash.?.hidden, xfm.s, "next draft stash hidden");
    }
}

test "cold grouped MTP head seeds the solo position and preserves valid origins" {
    try testing.expectEqual(@as(c_int, 1), Generator.mtpBatchHeadPosition(-1, 0));
    try testing.expectEqual(@as(c_int, 0), Generator.mtpBatchHeadPosition(0, 0));
    try testing.expectEqual(@as(c_int, 8), Generator.mtpBatchHeadPosition(8, 0));
    try testing.expectEqual(@as(c_int, 12), Generator.mtpBatchHeadPosition(8, 4));
    try testing.expectEqual(@as(c_int, 10), Generator.mtpBatchHeadPosition(1, 9));
}

test "cold grouped MTP rounds detach and resume exactly like solo heads" {
    const path = std.c.getenv("MTP_TEST_MODEL") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    // Group builders keep the merged history step; the solo arm must draft the same way.
    chain_mod.mtp_padded_head_override = false;
    defer chain_mod.mtp_padded_head_override = null;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try model_mod.parseConfig(io, a, std.mem.span(path));
    defer if (config.ngram_table_path) |name| a.free(name);
    var weights = try model_mod.loadWeights(io, a, std.mem.span(path));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, a, config, &weights);
    defer xfm.deinit();
    xfm.compileQwen4Hc();
    xfm.compileGdnGate();
    xfm.compileMoeRouting();
    var tok = Tokenizer.initEmptyForTests(a, .byte_level_bpe);
    defer tok.deinit();
    const prior = group_planner.enabled_override;
    group_planner.enabled_override = true;
    defer group_planner.enabled_override = prior;
    var prompt: [96]u32 = undefined;
    for (&prompt, 0..) |*id, i| id.* = @intCast(1 + (i * 7919) % 40000);
    var solo: [2]*MtpChainTestSlot = undefined;
    var grouped: [2]*MtpChainTestSlot = undefined;
    var built: usize = 0;
    defer for (solo[0..built], grouped[0..built]) |ref, row| {
        ref.deinit(a);
        row.deinit(a);
    };
    for (0..2) |row| {
        const ids = prompt[0..if (row == 0) @as(usize, 96) else 61];
        solo[row] = try MtpChainTestSlot.init(a, io, &xfm, &tok, ids, 1);
        grouped[row] = MtpChainTestSlot.init(a, io, &xfm, &tok, ids, 1) catch |err| {
            solo[row].deinit(a);
            return err;
        };
        built += 1;
        for ([_]*MtpChainTestSlot{ solo[row], grouped[row] }) |slot| {
            const gen = &slot.gen.?;
            try testing.expect(gen.mtp_hist_stash == null and gen.mtp_pre_draft == null);
            gen.mtp_cache.?.activate();
            try xfm.qwen4MtpReset();
            gen.mtp_planner_owned = true;
            gen.mtp_planner_width = 1;
            gen.mtp_batch_head = true;
        }
    }
    for (0..2) |_| {
        for (solo) |slot| {
            const result = (try slot.gen.?.nextMtp(a)) orelse return error.MissingSoloRound;
            a.free(result.tokens);
        }
        var gens: [2]*Generator = undefined;
        var opens: [2]Generator.MtpRoundOpen = undefined;
        var chains: [2]Generator.MtpPreDraft = undefined;
        var owns: [2]bool = @splat(false);
        defer for (&chains, owns) |*chain, owned| {
            if (owned) chain.deinit(a);
        };
        for (grouped, 0..) |slot, row| {
            gens[row] = &slot.gen.?;
            opens[row] = try MtpChainTestSlot.begin(gens[row], a, 1, false);
            gens[row].mtp_batch_head = true;
            chains[row] = opens[row].chain;
            owns[row] = true;
        }
        try Generator.mtpChainBuildBatched(&gens, &chains, 0, 1);
        var states: [2]?Generator.MtpRoundState = @splat(null);
        defer for (&states) |*state| {
            if (state.*) |*value| value.deinit(a);
        };
        var pointers: [2]*Generator.MtpRoundState = undefined;
        for (0..2) |row| {
            opens[row].chain = chains[row];
            owns[row] = false;
            states[row] = try gens[row].mtpRoundContinue(a, opens[row]);
            pointers[row] = &states[row].?;
        }
        try Generator.mtpGroupVerify(&gens, &pointers);
        try Generator.mtpGroupAccept(&gens, &pointers);
        for (gens, pointers) |gen, state| {
            const result = (try gen.mtpRoundFinish(a, state)) orelse return error.MissingGroupedRound;
            a.free(result.tokens);
        }
    }
    for (solo, grouped) |ref, row| {
        const rg = &ref.gen.?;
        const bg = &row.gen.?;
        try testing.expect(bg.mtp_hist_stash.?.off0 > 0);
        try bg.mtpDetachHead(a, true);
        try rg.mtpDetachHead(a, true);
        const rh = rg.mtp_cache.?.qwen4StateForBatch();
        const bh = bg.mtp_cache.?.qwen4StateForBatch();
        try testing.expectEqual(@as(c_int, 1), bh.pos_base);
        try testing.expectEqual(rh.meta(), bh.meta());
        try testing.expectEqualSlices(u32, rg.generated_ids.items, bg.generated_ids.items);
        try MtpChainTestSlot.equal(rg.last_hidden, bg.last_hidden, xfm.s, "cold head committed hidden");
        try MtpChainTestSlot.equal(rh.entry.aux_state, bh.entry.aux_state, xfm.s, "cold head QSA keys");
        try MtpChainTestSlot.equal(rh.entry.qsa_pooled, bh.entry.qsa_pooled, xfm.s, "cold head QSA pool");
        const rr = (try rg.nextMtp(a)) orelse return error.MissingSoloResume;
        defer a.free(rr.tokens);
        const br = (try bg.nextMtp(a)) orelse return error.MissingGroupedResume;
        defer a.free(br.tokens);
        try testing.expectEqualSlices(u32, rr.tokens, br.tokens);
        try testing.expectEqual(rr.accepted_tokens, br.accepted_tokens);
        try MtpChainTestSlot.equal(rg.mtp_hist_stash.?.ids, bg.mtp_hist_stash.?.ids, xfm.s, "cold resumed stash ids");
        try MtpChainTestSlot.equal(rg.mtp_hist_stash.?.hidden, bg.mtp_hist_stash.?.hidden, xfm.s, "cold resumed stash hidden");
    }
}
