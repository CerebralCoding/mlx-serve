//! DeepSeek-V4.1-Flash: the module-owned native arch (`deepseek_v41.zig`) and its vision tower behind the model table.
//! Serial, single-flight; one conversation resumes (`resume`, keyed on its image rows). Streamed EXL3 packs stay on
//! lib/mlx-stream until P4 merges the two implementations.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const mlx = @import("../../mlx/ffi.zig");
const model = @import("../../engines/mlx/model.zig");
const dsv41 = @import("deepseek_v41.zig");
const vision = @import("vision.zig");
const media = @import("media.zig");
const bill_mod = @import("bill.zig");
const fixture = @import("../deepseek_v4/fixture.zig");
const v4plugin = @import("../deepseek_v4/plugin.zig");
const toModelConfig = @import("../deepseek_v4/model_config.zig").toModelConfig;
const quant_abi = @import("../../abi/quant.zig");
const intree = @import("../../engines/mlx/plugin/intree.zig");
const distributed = @import("../../engines/mlx/cluster/distributed.zig");
const Layout = @import("../../sdk/quant/native.zig").Layout;
pub const config = @import("config.zig");

pub const plugin = config.plugin;

pub const table = abi.ModelTable{
    .name = abi.Str.of(plugin.name),
    .claims = config.claims,
    .parse = &config.parse,
    .free_parsed = &config.freeParsed,
    .bill = &bill_mod.bill,
    .load = &load,
    .unload = &unload,
    .forward = &forward,
    .cache = &.{ .layout = &layout, .resume_prompt = &resumePrompt, .state_bytes = &stateBytes },
    .media = &.{ .tower = &media.tower, .prepare = &media.prepare, .placeholder_rows = &media.placeholderRows },
    .spec = &.{ .trunk_weight = &trunkWeight },
    .weights = &.{},
};

const Verdict = abi.Verdict;
const gpa = std.heap.c_allocator;

/// A KV source grows its compressed rows; a compressed layer that is not one reads the last source at or above it;
/// a ratio-0 layer keeps its window. All of it lives in the module's state, never in the engine's cache.
pub fn layout(p: *const abi.Parsed, span: ?*const abi.LayerSpan, out: [*]abi.LayerCache, cap: u32) callconv(.c) u32 {
    const c = config.of(p);
    const first: u32 = if (span) |s| s.first else 0;
    const n = @min(if (span) |s| s.count else c.num_hidden_layers, cap);
    var source: i16 = -1;
    for (0..first) |li| if ((c.dsv41_kv_sources >> @intCast(li)) & 1 != 0) {
        source = @intCast(li);
    };
    for (out[0..n], first..) |*l, li| {
        const ratio = c.dsv4_compress_ratios[li];
        const own = (c.dsv41_kv_sources >> @intCast(li)) & 1 != 0;
        if (own) source = @intCast(li);
        l.* = if (ratio == 0)
            .{ .kind = .sliding, .window = c.sliding_window, .kv_heads = 1, .k_dim = c.head_dim }
        else
            .{ .kind = .full, .kv_source = if (own) -1 else source, .kv_heads = 1, .k_dim = c.head_dim / ratio };
    }
    return n;
}

pub const Module = dsv41.Dsv41Model;

/// The module a loaded model runs on (`engines/mlx/plugin/intree.zig`).
pub fn moduleOf(mp: *anyopaque) *Module {
    const m: *Model = @ptrCast(@alignCast(mp));
    return m.mdl;
}

pub const Model = struct {
    cfg: model.ModelConfig,
    weights: model.Weights,
    mdl: *dsv41.Dsv41Model,
    tower: ?vision.Dsv41Vision,
    caps: abi.Caps,
    s: mlx.mlx_stream,
};

/// The trunk's embedding table or LM head as loaded (borrowed), for the DSpark drafter.
fn trunkWeight(mp: *anyopaque, which: abi.TrunkWeight, out: *quant_abi.Weight) callconv(.c) Verdict {
    const m: *Model = @ptrCast(@alignCast(mp));
    const l = switch (which) {
        .embed => &m.mdl.embed,
        .head => &m.mdl.head,
    };
    out.* = .{ .w = .{ .ctx = l.w.ctx }, .scales = .{ .ctx = l.s.ctx }, .biases = .{ .ctx = l.b.ctx }, .bits = @intCast(l.qp.bits), .group_size = l.qp.group_size };
    if (l.s.ctx != null) out.layout = @backingInt(switch (l.qp.mode) {
        .nvfp4 => Layout.nvfp4,
        .mxfp4 => Layout.mxfp4,
        .mxfp8 => Layout.mxfp8,
        else => Layout.affine,
    });
    return .ok;
}

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    const s = gpuOf(.{ .ctx = ctx.stream.ctx }) orelse {
        out.* = Verdict.refuse(.unsupported, "deepseek_v41: its kernels run on Metal only");
        return null;
    };
    return loadModel(ctx, s) catch |e| {
        out.* = switch (e) {
            error.OutOfMemory => Verdict.refuse(.failed, "deepseek_v41: out of memory"),
            error.MissingWeight => Verdict.refuse(.failed, "deepseek_v41: a weight is missing"),
            else => Verdict.refuse(.failed, "deepseek_v41: the pack does not load"),
        };
        return null;
    };
}

/// The host's stream when it is a GPU one, else this thread's GPU stream: the module's kernels are Metal-only.
fn gpuOf(s: mlx.mlx_stream) ?mlx.mlx_stream {
    if (mlx.streamIsGpu(s)) return s;
    if (mlx.noGpuBackend()) return null;
    return mlx.gpuStream();
}

/// The per-model `mtp` setting (`{"mtp": false}` keeps the DSpark stages off).
fn mtpSetting(settings_json: []const u8) ?bool {
    if (settings_json.len == 0) return null;
    const S = struct { mtp: ?bool = null };
    const parsed = std.json.parseFromSlice(S, gpa, settings_json, .{ .ignore_unknown_fields = true }) catch return null;
    defer parsed.deinit();
    return parsed.value.mtp;
}

fn loadModel(ctx: *const abi.LoadCtx, s: mlx.mlx_stream) !*Model {
    const io = std.Io.Threaded.global_single_threaded.io();
    const c = config.of(ctx.parsed);
    const m = try gpa.create(Model);
    errdefer gpa.destroy(m);
    m.cfg = try toModelConfig(gpa, c, ctx.model_dir.slice());
    errdefer m.cfg.deinit(gpa);
    m.cfg.mtp_override = mtpSetting(ctx.settings_json.slice());
    // The driver's map never holds the vision tower (the engine's encoder reads it and hands its rows to `forward`).
    const taken = intree.takeWeights(ctx);
    m.weights = taken orelse try model.loadModelWeights(io, gpa, ctx.model_dir.slice(), &m.cfg, c.dsv41_vision);
    errdefer m.weights.deinit();
    // The driver's load planned a Thunderbolt pair on the engine's config (`model.loadModelWeights`): the same plan.
    if (taken != null) if (try distributed.planFor(ctx.model_dir.slice(), &m.cfg)) |part| switch (part) {
        .remote => |r| m.cfg.dist_dsv41 = .{ .lo = r.start, .hi = r.end, .worker = false },
        .span, .shard => {},
    };
    m.s = s;
    m.caps = ctx.parsed.caps;
    m.mdl = try dsv41.init(gpa, &m.cfg, &m.weights, s, .{ .dspark = dsv41.dsparkWanted(&m.cfg), .pair = m.cfg.dist_dsv41, .head_8bit = dsv41.head8Wanted() });
    errdefer m.mdl.deinit();
    m.tower = if (c.dsv41_vision and taken == null) try vision.Dsv41Vision.init(gpa, m.cfg, &m.weights) else null;
    return m;
}

fn unload(mp: *anyopaque) callconv(.c) void {
    const m: *Model = @ptrCast(@alignCast(mp));
    if (m.tower) |*t| t.deinit();
    m.mdl.deinit();
    m.weights.deinit();
    m.cfg.deinit(gpa);
    gpa.destroy(m);
}

pub fn forward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
    const m: *Model = @ptrCast(@alignCast(mp));
    if (!abi.declaresMode(m.caps, f.mode)) return Verdict.refuse(.unsupported, "deepseek_v41: forward mode not declared");
    if (f.capture_layers.len > 0) return Verdict.refuse(.unsupported, "deepseek_v41: no layer capture");
    if (f.slots.len > 1 or f.rows == 0) return Verdict.refuse(.unsupported, "deepseek_v41: one slot's rows at a time");
    const in = v4plugin.Ids.of(f) catch return Verdict.refuse(.failed, "deepseek_v41: the token ids do not read back");
    defer in.deinit();
    const pos: usize = if (f.positions.len > 0) f.positions.slice()[0] else 0;
    if (pos != 0 and (m.mdl.dec_state == null or m.mdl.dec_state.?.n != pos)) return Verdict.refuse(.failed, "deepseek_v41: the position does not follow the model's state");
    for (in.host) |t| if (t >= m.cfg.vocab_size) return Verdict.refuse(.failed, "deepseek_v41: token id past the vocabulary");
    const rows: ?dsv41.VisionRows = if (f.media.ctx != null) .{ .rows = .{ .ctx = f.media.ctx }, .offset = f.media_offset } else null;
    const logits = run(m, in.host, pos, f.logits == .all_rows, rows, in.dev) catch return Verdict.refuse(.failed, "deepseek_v41: the forward failed");
    if (f.logits == .none) _ = mlx.mlx_array_free(logits) else out.logits = .{ .ctx = logits.ctx };
    return .ok;
}

/// The legacy seam's arms: position 0 starts a request (a resumed one arrives past 0); a decode token stays on the
/// GPU (`dev`, or the host id uploaded); everything else extends the state. `all` returns every row's logits.
fn run(m: *Model, host_ids: []const u32, pos: usize, all: bool, rows: ?dsv41.VisionRows, dev: mlx.mlx_array) !mlx.mlx_array {
    const mdl = m.mdl;
    var one: [1]u32 = undefined;
    const gpu_decode = !all and pos > 0 and (dev.ctx != null or host_ids.len == 1) and mdl.pair == null;
    const ids = if (dev.ctx != null and !gpu_decode) blk: {
        one[0] = try v4plugin.hostToken(dev);
        if (one[0] >= m.cfg.vocab_size) return error.TokenPastVocabulary;
        break :blk one[0..];
    } else host_ids;
    const logits = if (all) blk: {
        if (pos == 0) try dsv41.reset(mdl, gpa);
        break :blk try dsv41.extendAllResumable(mdl, gpa, ids);
    } else if (gpu_decode) blk: {
        const tok = if (dev.ctx != null) try v4plugin.deviceToken(dev, &.{ 1, 1 }, m.s) else blk2: {
            const id: i32 = @intCast(ids[0]);
            break :blk2 mlx.mlx_array_new_data(&id, &[_]c_int{ 1, 1 }, 2, .int32);
        };
        defer _ = mlx.mlx_array_free(tok);
        break :blk try dsv41.extendResumableGpu(mdl, gpa, tok);
    } else if (pos == 0)
        try dsv41.prefill(mdl, gpa, ids, rows)
    else
        try dsv41.extendResumable(mdl, gpa, ids, rows);
    defer _ = mlx.mlx_array_free(logits);
    var o = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_reshape(&o, logits, &[_]c_int{ -1, @intCast(mdl.vocab) }, 2, m.s));
    return o;
}

/// `CacheHooks.resume_prompt`: restores the longest prefix of `prompt` the model kept (images keyed by their rows)
/// and returns its length; a cold prompt, or a failed restore, resets the state and returns 0.
pub fn resumePrompt(mp: *anyopaque, prompt: abi.Span(u32), media_rows: abi.Array) callconv(.c) u64 {
    const m: *Model = @ptrCast(@alignCast(mp));
    const rows: ?mlx.mlx_array = if (media_rows.ctx != null) .{ .ctx = media_rows.ctx } else null;
    const n = dsv41.resumePrompt(m.mdl, prompt.slice(), rows) catch 0;
    // A failed reset leaves no state; the forward at position 0 builds one.
    if (n == 0) dsv41.reset(m.mdl, gpa) catch {};
    return n;
}

/// `CacheHooks.state_bytes`: the bytes the request's state holds on the GPU, for the engine's cache budget.
pub fn stateBytes(mp: *anyopaque) callconv(.c) u64 {
    const m: *Model = @ptrCast(@alignCast(mp));
    return if (m.mdl.dec_state) |*st| dsv41.stateBytes(st) else 0;
}

/// The tiny pack the conformance suite runs: the oracle's pipe layout without Engram or DSpark.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    return .{ .files = try fixture.v41Files(a, 7) };
}

// ── tests ──

test {
    _ = config;
    _ = bill_mod;
    _ = media;
    _ = @import("contract.zig");
    _ = intree;
}
