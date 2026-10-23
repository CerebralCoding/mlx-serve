//! GLM-5.3-Flash (`glm5_next`): KDA linear attention and DeepSeek sparse attention inside DeepSeek-V4's 4-stream
//! Sinkhorn hyper-connections, a 288-expert sigmoid MoE. Its own forward (`table_forward.zig`) over the layered
//! weights (`host.zig`); batched rows advance each slot's state by its own call.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const mlx = @import("../../mlx/ffi.zig");
const host = @import("host.zig");
const fwd = @import("table_forward.zig");
const head = @import("head.zig");
const fixture = @import("fixture.zig");
pub const config = @import("config.zig");

pub const plugin = sdk.Plugin{
    .name = "glm5_next",
    .kind = .model,
    .claims = &config.model_types,
    .gate_models = &.{"ddalcu/GLM-5.3-Flash-MLX-Serve-iQ-MLX-5.1bpw"},
};

pub const table = abi.ModelTable{
    .name = abi.Str.of(plugin.name),
    .claims = config.claims,
    .parse = &config.parse,
    .free_parsed = &config.freeParsed,
    .bill = &bill,
    .load = &load,
    .unload = &host.unload,
    .forward = &forward,
    .cache = &.{ .layout = &layout },
    .weights = &.{},
    .spec = &.{ .trunk_weight = &host.trunkWeight, .head = &head.hooks },
};

const Verdict = abi.Verdict;
const Array = mlx.mlx_array;

/// KDA layers keep a recurrent state; every fourth layer caches one latent row `k` and the indexer's key and pool
/// gate `v` per token, unrotated.
pub fn layout(p: *const abi.Parsed, span: ?*const abi.LayerSpan, out: [*]abi.LayerCache, cap: u32) callconv(.c) u32 {
    const c = config.of(p);
    const first: u32 = if (span) |s| s.first else 0;
    const n = @min(if (span) |s| s.count else c.num_hidden_layers, cap);
    for (out[0..n], first..) |*l, li| l.* = if (c.isLinearLayer(@intCast(li)))
        .{ .kind = .linear }
    else
        .{ .kind = .full, .nope = true, .kv_heads = 1, .k_dim = c.dsa_kv_lora_rank, .v_dim = 2 * c.dsa_index_head_dim };
    return n;
}

/// Dense bf16 bytes per token (one latent row plus the indexer's key and gate per sparse layer) and the checkpoint.
/// The prefill bill stays the engine's `glm5PrefillMemoryNeeded` until `Bill.prefill_bytes` carries its terms.
fn bill(p: *const abi.Parsed, _: *const abi.Shape, out: *abi.Bill) callconv(.c) void {
    const c = config.of(p);
    out.* = .{
        .kv_per_token = @as(u64, c.attnLayerCount()) * (c.dsa_kv_lora_rank + 2 * c.dsa_index_head_dim) * 2,
        .ssm_checkpoint = c.ssmCheckpointBytes(4),
    };
}

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    return host.load(plugin.name, ctx, out);
}

pub fn forward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
    const m = host.of(mp);
    if (!abi.declaresMode(m.caps, f.mode)) return Verdict.refuse(.unsupported, "glm5_next: forward mode not declared");
    const ops = f.cache_ops orelse return Verdict.refuse(.unsupported, "glm5_next: a forward without cache_ops");
    if (f.mode == .batched_rows) {
        if (f.rows != 1 or f.tokens.len != f.caches.len or f.positions.len != f.caches.len) return Verdict.refuse(.unsupported, "glm5_next: batched rows take one host token per slot");
        if (f.capture_layers.len > 0 or f.media.ctx != null) return Verdict.refuse(.unsupported, "glm5_next: no captures or media on batched rows");
        batched(m, ops, f, out) catch |e| {
            host.clear(out, 0);
            return Verdict.named(.failed, @errorName(e));
        };
        return .ok;
    }
    return host.forwardOne(plugin.name, fwd, m, ops, f, out);
}

/// N slots' next tokens in as few trunk forwards as the MoE row limit allows, each slot's token a row of one window:
/// logits `[N, V]` and, when asked, the final-normed hidden `[N, H]`.
fn batched(m: *host.Model, ops: *const abi.CacheOps, f: *const abi.Forward, out: *abi.ForwardOut) !void {
    const n = f.caches.len;
    const groups = (n + fwd.MOE_DECODE_BATCH_MAX - 1) / fwd.MOE_DECODE_BATCH_MAX;
    const parts = try std.heap.c_allocator.alloc([2]Array, groups);
    defer std.heap.c_allocator.free(parts);
    var made: usize = 0;
    defer for (parts[0..made]) |p| for (p) |a| {
        _ = mlx.mlx_array_free(a);
    };
    var start: usize = 0;
    while (start < n) : (start += fwd.MOE_DECODE_BATCH_MAX) {
        const end = @min(n, start + fwd.MOE_DECODE_BATCH_MAX);
        const ids = try host.hostIds(f.tokens.slice()[start..end]);
        defer _ = mlx.mlx_array_free(ids);
        const st: fwd.Step = .{
            .slots = .{ .ops = ops, .caches = f.caches.slice()[start..end], .positions = f.positions.slice()[start..end] },
            .moe = m.moe.layers,
            .batched = true,
        };
        const hidden = try fwd.hidden(&m.xfm, &st, ids);
        defer _ = mlx.mlx_array_free(hidden);
        const logits = try m.xfm.lmHeadProject(hidden);
        defer _ = mlx.mlx_array_free(logits);
        parts[made] = .{ mlx.mlx_array_new(), mlx.mlx_array_new() };
        made += 1;
        try rows2d(m, logits, &parts[made - 1][0]);
        try rows2d(m, hidden, &parts[made - 1][1]);
    }
    out.logits = .{ .ctx = (try joinRows(m, parts[0..made], 0)).ctx };
    if (f.want_hidden) out.hidden = .{ .ctx = (try joinRows(m, parts[0..made], 1)).ctx };
}

/// `[1, R, W]` as `[R, W]`.
fn rows2d(m: *host.Model, x: Array, dst: *Array) !void {
    const sh = mlx.getShape(x);
    try mlx.check(mlx.mlx_reshape(dst, x, &[_]c_int{ sh[1], sh[2] }, 2, m.xfm.s));
}

/// The groups' rows `which` (0 logits, 1 hidden) stacked in slot order, a new reference.
fn joinRows(m: *host.Model, parts: []const [2]Array, which: usize) !Array {
    var all = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(all);
    if (parts.len == 1) {
        _ = mlx.mlx_array_set(&all, parts[0][which]);
        return all;
    }
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (parts) |p| _ = mlx.mlx_vector_array_append_value(vec, p[which]);
    try mlx.check(mlx.mlx_concatenate_axis(&all, vec, 0, m.xfm.s));
    return all;
}

/// The tiny pack the conformance suite runs: every arm of the trunk (dense and MoE MLPs, KDA, sparse attention).
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    return .{ .files = try fixture.glmFiles(a, 7) };
}

// ── tests ──

test {
    _ = config;
    _ = @import("model_config.zig");
    _ = @import("bill.zig");
    _ = fixture;
    _ = @import("proof_test.zig");
    _ = @import("served_test.zig");
}
