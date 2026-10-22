//! MTP and DFlash traces and their env switches.
const std = @import("std");
const log = @import("../../../../host/log.zig");
const io_util = @import("../../../../sdk/util/io_util.zig");
const mtp_mod = @import("../../../../sdk/spec/mtp/mtp.zig");
const testing = std.testing;
const Generator = @import("../../generate/generator.zig").Generator;
const readEnvBool = @import("../../env_generate.zig").readEnvBool;


/// Per-phase wall-time accumulator behind MLX_SERVE_MTP_TRACE=1. Pure
/// bookkeeping; `nextMtp` stamps phases with a Stopwatch and emits one
/// summary line every LOG_EVERY rounds. Zero cost when the env is absent
/// (every stamp is guarded on the cached env check).
pub const MtpTrace = struct {
    pub const LOG_EVERY: u32 = 32;
    pub const Phase = enum(u4) { draft, sync, ext, verify, corr, eval, hist, commit, predraft, gap };
    pub const N_PHASES = @typeInfo(Phase).@"enum".field_names.len;
    // Sub-laps attribute time inside a phase and are not a partition of it: a group's build lap is
    // stamped into every row while each row's `verify` carries only its own share.
    pub const Sub = enum(u3) { build, ple, ple_sync, dispatch, wait, chain };
    pub const N_SUBS = @typeInfo(Sub).@"enum".field_names.len;

    rounds: u32 = 0,
    ns: [N_PHASES]u64 = @splat(0),
    sub_ns: [N_SUBS]u64 = @splat(0),
    drafted: u64 = 0,
    accepted: u64 = 0,
    extended: u32 = 0,
    /// Per draft index: rounds that drafted index i / accepted it.
    drafted_idx: [mtp_mod.MAX_DEPTH]u32 = @splat(0),
    accepted_idx: [mtp_mod.MAX_DEPTH]u32 = @splat(0),

    pub fn add(self: *MtpTrace, phase: Phase, dur_ns: u64) void {
        self.ns[@backingInt(phase)] += dur_ns;
    }

    pub fn addSub(self: *MtpTrace, sub: Sub, dur_ns: u64) void {
        self.sub_ns[@backingInt(sub)] += dur_ns;
    }

    /// Close one round; true when a summary line is due (caller logs,
    /// then calls reset()).
    pub fn endRound(self: *MtpTrace, drafted_n: u32, accepted_n: u32, was_extended: bool) bool {
        self.rounds += 1;
        self.drafted += drafted_n;
        self.accepted += accepted_n;
        if (was_extended) self.extended += 1;
        var i: usize = 0;
        while (i < drafted_n and i < mtp_mod.MAX_DEPTH) : (i += 1) {
            self.drafted_idx[i] += 1;
            if (i < accepted_n) self.accepted_idx[i] += 1;
        }
        return self.rounds >= LOG_EVERY;
    }

    /// `a0/a1/...` acceptance per drafted index (only indices ever drafted).
    pub fn accIdxStr(self: *const MtpTrace, buf: []u8) []const u8 {
        var w: std.Io.Writer = .fixed(buf);
        var i: usize = 0;
        while (i < mtp_mod.MAX_DEPTH and self.drafted_idx[i] > 0) : (i += 1) {
            w.print("{s}{d:.2}", .{ if (i == 0) "" else "/", @as(f64, @floatFromInt(self.accepted_idx[i])) / @as(f64, @floatFromInt(self.drafted_idx[i])) }) catch break;
        }
        return w.buffered();
    }

    pub fn avgMs(self: *const MtpTrace, phase: Phase) f64 {
        if (self.rounds == 0) return 0.0;
        return @as(f64, @floatFromInt(self.ns[@backingInt(phase)])) /
            (@as(f64, @floatFromInt(self.rounds)) * 1e6);
    }

    pub fn subAvgMs(self: *const MtpTrace, sub: Sub) f64 {
        if (self.rounds == 0) return 0.0;
        return @as(f64, @floatFromInt(self.sub_ns[@backingInt(sub)])) /
            (@as(f64, @floatFromInt(self.rounds)) * 1e6);
    }

    pub fn totalAvgMs(self: *const MtpTrace) f64 {
        if (self.rounds == 0) return 0.0;
        var total: u64 = 0;
        for (self.ns) |v| total += v;
        return @as(f64, @floatFromInt(total)) / (@as(f64, @floatFromInt(self.rounds)) * 1e6);
    }

    // Profilers parse the phase summary as one run from `rounds=` to `acc_idx=`; new fields go behind it.
    pub const LINE_FMT = "  [mtp-trace] rounds={d} avg_ms draft={d:.2} sync={d:.2} ext={d:.2} verify={d:.2} corr={d:.2} eval={d:.2} hist={d:.2} commit={d:.2} predraft={d:.2} gap={d:.2} total={d:.2} | m_avg={d:.2} acc_avg={d:.2} ext_rate={d:.2} acc_idx={s} | sub_ms build={d:.2} ple={d:.2} ple_sync={d:.2} dispatch={d:.2} wait={d:.2} chain={d:.2}\n";

    pub fn reset(self: *MtpTrace) void {
        self.* = .{};
    }
};

pub const SubLap = struct {
    watch: ?io_util.Stopwatch,

    pub fn start(on: bool, io: std.Io) SubLap {
        return .{ .watch = if (on) io_util.Stopwatch.init(io) else null };
    }

    pub fn read(self: SubLap) ?u64 {
        return if (self.watch) |w| w.read() else null;
    }

    pub fn stop(self: SubLap, trace: *MtpTrace, sub: MtpTrace.Sub) void {
        if (self.read()) |ns| trace.addSub(sub, ns);
    }
};

/// Per-phase wall-time accumulator behind MLX_SERVE_DFLASH_TRACE=1.
/// `nextDflash` stamps phases with a Stopwatch and emits one summary line
/// every LOG_EVERY rounds. Zero cost when the env is absent.
pub const DflashTrace = struct {
    pub const LOG_EVERY: u32 = 16;
    pub const Phase = enum(u4) { assist, head, verify, accept, append, gap };
    pub const N_PHASES = @typeInfo(Phase).@"enum".field_names.len;

    rounds: u32 = 0,
    ns: [N_PHASES]u64 = @splat(0),
    accepted: u64 = 0,

    pub fn add(self: *DflashTrace, phase: Phase, dur_ns: u64) void {
        self.ns[@backingInt(phase)] += dur_ns;
    }

    /// Close one round; true when a summary line is due (caller logs,
    /// then calls reset()).
    pub fn endRound(self: *DflashTrace, accepted_n: u32) bool {
        self.rounds += 1;
        self.accepted += accepted_n;
        return self.rounds >= LOG_EVERY;
    }

    pub fn avgMs(self: *const DflashTrace, phase: Phase) f64 {
        if (self.rounds == 0) return 0.0;
        return @as(f64, @floatFromInt(self.ns[@backingInt(phase)])) /
            (@as(f64, @floatFromInt(self.rounds)) * 1e6);
    }

    pub fn totalAvgMs(self: *const DflashTrace) f64 {
        if (self.rounds == 0) return 0.0;
        var total: u64 = 0;
        for (self.ns) |v| total += v;
        return @as(f64, @floatFromInt(total)) / (@as(f64, @floatFromInt(self.rounds)) * 1e6);
    }

    pub fn reset(self: *DflashTrace) void {
        self.* = .{};
    }
};

/// Sampled DFlash drafts — DEFAULT OFF, measured negative.
/// `MLX_SERVE_DFLASH_SAMPLED_DRAFTS=1` draws each draft from the
/// request's own filtered distribution and accepts through the full
/// Leviathan ratio; off keeps the argmax draft and its one-hot q.
///
/// The theory said this should help: a greedy draft's acceptance is
/// p(argmax q), which a flattened target row deflates, while a matched
/// proposal accepts at 1 − TV(p, q). Measured on Muse 4-bit at temp 0.7
/// (3 reps x 4 prompts, same-boot serial reference), it is a LOSS —
/// 52.4 tok/s / 54.5% per-draft against 55.5 / 57.7% greedy. This
/// assistant's ARGMAX tracks the trunk well while its distribution SHAPE
/// does not, so a one-hot proposal is the better proposal and sampling
/// spends the round's budget landing on tokens the trunk would not pick.
/// Greedy requests never reach this either way.
var dflash_sampled_drafts_cache: ?bool = null;
pub fn dflashSampledDraftsEnabled() bool {
    if (dflash_sampled_drafts_cache) |v| return v;
    var on = false;
    if (std.c.getenv("MLX_SERVE_DFLASH_SAMPLED_DRAFTS")) |p| {
        const val = std.mem.span(p);
        if (val.len > 0 and val[0] == '1') on = true;
    }
    dflash_sampled_drafts_cache = on;
    return on;
}

/// `MLX_SERVE_DFLASH_SELECTOR`: default ON when the sidecar ships a
/// selector (DFlash2). "0" forces the v1 draft arms (argmax / block
/// sampling) for A/Bs — the conv layers stay, they are the checkpoint.
var dflash_selector_cache: ?bool = null;
/// `MLX_SERVE_DFLASH_MARKOV=0` drafts a DSpark sidecar's block from its
/// UNCORRECTED base logits (the v1 arm) — an A/B lever for measuring what
/// the Markov chain is worth, never a default.
pub fn dflashMarkovEnabled() bool {
    const p = std.c.getenv("MLX_SERVE_DFLASH_MARKOV") orelse return true;
    return !std.mem.eql(u8, std.mem.span(p), "0");
}

pub fn dflashSelectorEnabled() bool {
    if (dflash_selector_cache) |v| return v;
    var on = true;
    if (std.c.getenv("MLX_SERVE_DFLASH_SELECTOR")) |p| {
        const val = std.mem.span(p);
        if (val.len > 0 and val[0] == '0') on = false;
    }
    dflash_selector_cache = on;
    return on;
}

var dflash_trace_cache: ?bool = null;
pub fn dflashTraceEnabled() bool {
    if (dflash_trace_cache) |v| return v;
    const on = readEnvBool("MLX_SERVE_DFLASH_TRACE");
    dflash_trace_cache = on;
    return on;
}

/// Close one traced dflash round; emits + resets at the cadence.
pub fn dflashTraceRoundEnd(self: *Generator, accepted: u32) void {
    if (!self.dflash_trace.endRound(accepted)) return;
    const t = &self.dflash_trace;
    log.info(
        "  [dflash-trace] rounds={d} avg_ms assist={d:.2} head={d:.2} verify={d:.2} accept={d:.2} append={d:.2} gap={d:.2} total={d:.2} | acc_avg={d:.2}\n",
        .{
            t.rounds,
            t.avgMs(.assist),
            t.avgMs(.head),
            t.avgMs(.verify),
            t.avgMs(.accept),
            t.avgMs(.append),
            t.avgMs(.gap),
            t.totalAvgMs(),
            @as(f64, @floatFromInt(t.accepted)) / @as(f64, @floatFromInt(t.rounds)),
        },
    );
    t.reset();
}

/// Adaptive (EV) controller gate — DEFAULT ON. MLX_SERVE_MTP_ADAPTIVE=0
/// reverts to the fixed-depth windowed controller for same-boot A/Bs.
var mtp_adaptive_cache: ?bool = null;

pub fn mtpAdaptiveEnabled() bool {
    if (mtp_adaptive_cache) |v| return v;
    var on = true;
    if (std.c.getenv("MLX_SERVE_MTP_ADAPTIVE")) |p| {
        const val = std.mem.span(p);
        if (val.len > 0 and val[0] == '0') on = false;
    }
    mtp_adaptive_cache = on;
    return on;
}

/// Cross-request EV seeding gate — default ON; set
/// MLX_SERVE_MTP_EV_SEED=0 to keep request planning independent.
var mtp_ev_seed_cache: ?bool = null;
pub fn mtpEvSeedEnabledFromEnv(raw: ?[]const u8) bool {
    const value = raw orelse return true;
    return value.len == 0 or value[0] != '0';
}

pub fn mtpEvSeedEnabled() bool {
    if (mtp_ev_seed_cache) |v| return v;
    const raw: ?[]const u8 = if (std.c.getenv("MLX_SERVE_MTP_EV_SEED")) |p| std.mem.span(p) else null;
    const on = mtpEvSeedEnabledFromEnv(raw);
    mtp_ev_seed_cache = on;
    return on;
}

/// Early draft-chain dispatch (round pipelining) — DEFAULT ON. Fires the
/// draft-chain graph at the GPU as soon as Phase 1 finishes building it,
/// so the head chain runs while the CPU builds the verify/accept graphs
/// (~2 ms the GPU used to spend idle each round). Dispatch timing only —
/// lazy sampling ops bind their PRNG key at graph BUILD time, so values
/// are identical. MLX_SERVE_MTP_EARLY_DISPATCH=0 restores the serial
/// round shape for same-boot A/Bs.
var mtp_early_dispatch_cache: ?bool = null;
pub fn mtpEarlyDispatchEnabledFromEnv(raw: ?[]const u8) bool {
    const value = raw orelse return true;
    return value.len == 0 or value[0] != '0';
}

pub fn mtpEarlyDispatchEnabled() bool {
    if (mtp_early_dispatch_cache) |v| return v;
    const raw: ?[]const u8 = if (std.c.getenv("MLX_SERVE_MTP_EARLY_DISPATCH")) |p| std.mem.span(p) else null;
    const on = mtpEarlyDispatchEnabledFromEnv(raw);
    mtp_early_dispatch_cache = on;
    return on;
}

/// Cross-round pre-draft (round pipelining) — DEFAULT ON. Builds and
/// dispatches the next round's chunk-A draft chain at the current
/// round's tail (see mtpMaybePreDraft) so the GPU drafts while the CPU
/// does emit/SSE bookkeeping. MLX_SERVE_MTP_PREDRAFT=0 reverts to
/// head-of-round drafting for same-boot A/Bs (combine with
/// MLX_SERVE_MTP_EARLY_DISPATCH=0 for the fully serial round shape).
pub var mtp_predraft_cache: ?bool = null;
pub fn mtpPredraftEnabledFromEnv(raw: ?[]const u8) bool {
    const value = raw orelse return true;
    return value.len == 0 or value[0] != '0';
}

pub fn mtpPredraftEnabled() bool {
    if (mtp_predraft_cache) |v| return v;
    const raw: ?[]const u8 = if (std.c.getenv("MLX_SERVE_MTP_PREDRAFT")) |p| std.mem.span(p) else null;
    const on = mtpPredraftEnabledFromEnv(raw);
    mtp_predraft_cache = on;
    return on;
}

pub fn mtpTraceOn() bool {
    return mtpTraceEnabled();
}

pub fn mtpTraceSub(self: *Generator, sub: MtpTrace.Sub, dur_ns: u64) void {
    if (!mtpTraceEnabled()) return;
    self.mtp_trace.addSub(sub, dur_ns);
}

var mtp_trace_cache: ?bool = null;
pub fn mtpTraceEnabled() bool {
    if (mtp_trace_cache) |v| return v;
    const on = readEnvBool("MLX_SERVE_MTP_TRACE");
    mtp_trace_cache = on;
    return on;
}

// ── tests ──


test "MtpTrace: per-phase accumulation, round averaging, log cadence, reset" {
    var t = Generator.MtpTrace{};
    // 2 rounds: draft 2ms+4ms, eval 10ms+30ms.
    t.add(.draft, 2_000_000);
    t.add(.eval, 10_000_000);
    try testing.expect(!t.endRound(3, 2, false));
    t.add(.draft, 4_000_000);
    t.add(.eval, 30_000_000);
    try testing.expect(!t.endRound(7, 6, true));
    try testing.expectApproxEqAbs(@as(f64, 3.0), t.avgMs(.draft), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 20.0), t.avgMs(.eval), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 0.0), t.avgMs(.sync), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 23.0), t.totalAvgMs(), 1e-9);
    try testing.expectEqual(@as(u64, 10), t.drafted);
    try testing.expectEqual(@as(u64, 8), t.accepted);
    try testing.expectEqual(@as(u32, 1), t.extended);
    // Log line falls due exactly at LOG_EVERY rounds.
    var i: u32 = 2;
    while (i < Generator.MtpTrace.LOG_EVERY - 1) : (i += 1) {
        try testing.expect(!t.endRound(1, 1, false));
    }
    try testing.expect(t.endRound(1, 1, false));
    t.reset();
    try testing.expectEqual(@as(u32, 0), t.rounds);
    try testing.expectApproxEqAbs(@as(f64, 0.0), t.avgMs(.draft), 1e-9);
}

test "the [mtp-trace] phase summary stays contiguous for its log consumers" {
    const line = try std.fmt.allocPrint(testing.allocator, Generator.MtpTrace.LINE_FMT, .{
        @as(u32, 32), 0.45, 0.0, 0.0,         4.09, 56.86, 0.0,  0.0,   0.03, 0.0,  0.07, 61.57,
        2.0,          1.4,  0.0, "0.91/0.80", 4.09, 7.13,  6.40, 52.28, 0.94, 0.44,
    });
    defer testing.allocator.free(line);
    try testing.expect(std.mem.indexOf(u8, line, "total=61.57 | m_avg=") != null);
    const acc_at = std.mem.indexOf(u8, line, "acc_idx=").?;
    const sub_at = std.mem.indexOf(u8, line, "| sub_ms ").?;
    try testing.expect(acc_at < sub_at);
}

test "MtpTrace sub-laps attribute inside a phase without changing the round total" {
    var t = Generator.MtpTrace{};
    t.add(.verify, 20_000_000);
    t.addSub(.build, 15_000_000);
    t.addSub(.ple, 3_000_000);
    t.addSub(.ple_sync, 2_000_000);
    t.add(.corr, 6_000_000);
    t.addSub(.dispatch, 2_000_000);
    t.add(.eval, 60_000_000);
    t.addSub(.wait, 58_000_000);
    t.add(.gap, 12_000_000);
    t.addSub(.chain, 2_000_000);
    try testing.expect(!t.endRound(2, 2, false));
    try testing.expectApproxEqAbs(@as(f64, 98.0), t.totalAvgMs(), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 20.0), t.avgMs(.verify), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 20.0), t.subAvgMs(.build) + t.subAvgMs(.ple) + t.subAvgMs(.ple_sync), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 58.0), t.subAvgMs(.wait), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 2.0), t.subAvgMs(.chain), 1e-9);
    t.reset();
    try testing.expectApproxEqAbs(@as(f64, 0.0), t.subAvgMs(.build), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 0.0), t.totalAvgMs(), 1e-9);
}

test "MtpTrace per-index acceptance: index i counts only rounds that drafted it" {
    var t: Generator.MtpTrace = .{};
    _ = t.endRound(3, 3, false); // all three accepted
    _ = t.endRound(3, 1, false); // a0 accepted, a1 rejected, a2 never judged
    _ = t.endRound(1, 0, false); // depth-1 round: only index 0 drafted
    var buf: [64]u8 = undefined;
    try std.testing.expectEqualStrings("0.67/0.50/0.50", t.accIdxStr(&buf));
    t.reset();
    try std.testing.expectEqualStrings("", t.accIdxStr(&buf));
}

test "every request-finalize seam that logs [spec-stats] also logs [qsa-arms]" {
    // Class guard for a wiring bug this test was written after: `logQsaArms`
    // was added beside the three `gen.logSpecStats()` calls in THIS file, which
    // are the legacy/CLI path only. The SERVER finalizes requests in
    // `scheduler.finishSlot` ("scheduler-driven slots finalize here instead"),
    // so `[qsa-arms]` was dead on every served request while looking wired.
    // A per-request meter is worth nothing if it misses the path that actually
    // serves requests, so the SEAM — not the symbol — is what gets pinned:
    // every call site of the request summary must carry both. Needles are
    // assembled at comptime so this test's own source cannot satisfy the scan.
    const spec_call = "logSpec" ++ "Stats();";
    const qsa_call = "logQsa" ++ "Arms();";
    inline for (.{ @import("../../generate.zig").all_source, @embedFile("../../finish.zig") }) |src| {
        var i: usize = 0;
        var seams: usize = 0;
        while (std.mem.indexOfPos(u8, src, i, spec_call)) |at| {
            seams += 1;
            // The paired call must sit in the same short block, not merely
            // somewhere else in the file.
            const win_end = @min(src.len, at + 400);
            if (std.mem.indexOfPos(u8, src, at, qsa_call)) |q| {
                if (q >= win_end) return error.SpecStatsSeamMissesQsaArms;
            } else return error.SpecStatsSeamMissesQsaArms;
            i = at + spec_call.len;
        }
        try std.testing.expect(seams > 0);
    }
}
