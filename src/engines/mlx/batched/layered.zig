//! engines/mlx/batched/layered, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns gdn_batched_logged, per_slot_attn_logged, gdn_batched_verify_rows_logged, row_axis_verify_logged, verify_fail_after_flush, MAX_BATCH_ROWS, and 5 more.

const std = @import("std");
const plugin_driver = @import("../plugin/driver.zig");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../sdk/util/log.zig");
const ForwardCtx = @import("../forward_ctx.zig").ForwardCtx;
const SSMCacheEntry = @import("../ssm/entry.zig").SSMCacheEntry;
const splitSpecCaptures = @import("../ssm/entry.zig").splitSpecCaptures;
const ssmFreeSpecCapture = @import("../ssm/entry.zig").ssmFreeSpecCapture;
const SsmMergeKind = @import("../ssm/group.zig").SsmMergeKind;
const ssmBindSlotRefs = @import("../ssm/group.zig").ssmBindSlotRefs;
const ssmBindTickState = @import("../ssm/group.zig").ssmBindTickState;
const ssmFreeMergedLayer = @import("../ssm/group.zig").ssmFreeMergedLayer;
const ssmSplitAll = @import("../ssm/group.zig").ssmSplitAll;
const Transformer = @import("../transformer.zig").Transformer;

var gdn_batched_logged: bool = false;

pub var per_slot_attn_logged: bool = false;
var gdn_batched_verify_rows_logged: c_int = 0; // widest row total logged so far
pub var row_axis_verify_logged: bool = false;
pub var verify_fail_after_flush: ?usize = null;

pub const MAX_BATCH_ROWS = @import("../../../sdk/bases/run.zig").MAX_BATCH_ROWS;
pub var std_batched_logged: bool = false; // one-shot log guard

/// Batched decode for a GatedDeltaNet trunk (qwen3_5 family), the hybrid
/// twin of `forwardBatchedDecode`.
///
/// Why this exists: without it, concurrent requests on a GDN model decode
/// strictly serially, so N streams read the whole weight set N times.
/// Batching reads it ONCE — measured against oMLX/mlx-lm, which batch
/// these models and were ~2x our aggregate throughput at 4 streams.
///
/// The trick is that `forwardMoeWith` is already batch-generic: every
/// projection, norm, MLP and the GDN recurrence kernel (whose grid is
/// `B*Hv` on the z axis) take B=N unchanged. Only two things are per-slot:
///
///   - the GDN recurrent state — merged here into ONE [N, …] entry per
///     layer and split back after, so the layers never learn about slots;
///   - the KV cache of the full-attention layers — left in place and
///     handled by the `ctx.batch_slots` branch in `gatedFullAttnWith`,
///     because kv_len differs per slot and cannot be a single tensor.
///
/// Returns N logits arrays `[1,1,V]`, caller owns each and the slice.
/// `hidden_rows`: when given, receives every slot's post-norm hidden `[1,1,H]`
/// (owned slice), for a slot that keeps an MTP head resumable across plain ticks.
pub fn forwardMoeBatchedDecode(
    self: *Transformer,
    next_tokens: []const u32,
    ctxs: []const *ForwardCtx,
    rope_offsets: []const u32,
    hidden_rows: ?*?[]mlx.mlx_array,
) ![]mlx.mlx_array {
    if (self.plugin) |b| return plugin_driver.forwardBatched(self, b, next_tokens, ctxs, rope_offsets, hidden_rows);
    if (self.config.isGlm5()) return self.forwardGlmBatchedDecode(next_tokens, ctxs, hidden_rows);
    const N: c_int = @intCast(next_tokens.len);
    std.debug.assert(next_tokens.len == ctxs.len);
    if (!gdn_batched_logged) {
        gdn_batched_logged = true;
        log.info("[batched] gdn batched decode engaged (slots={d})\n", .{next_tokens.len});
    }
    // [N,1] tokens.
    var token_buf = try self.allocator.alloc(i32, next_tokens.len);
    defer self.allocator.free(token_buf);
    for (next_tokens, 0..) |t, i| token_buf[i] = @intCast(t);
    const tok_shape = [_]c_int{ N, 1 };
    const token_arr = mlx.mlx_array_new_data(token_buf.ptr, &tok_shape, 2, .int32);
    defer _ = mlx.mlx_array_free(token_arr);

    var hidden = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hidden);
    const logits = try self.forwardMoeBatchedRows(token_arr, ctxs, rope_offsets, false, if (hidden_rows != null) &hidden else null, null);
    defer _ = mlx.mlx_array_free(logits);
    if (hidden_rows) |out| out.* = try sliceBatchRows(self.allocator, self.s, hidden, next_tokens.len);
    return sliceBatchRows(self.allocator, self.s, logits, next_tokens.len);
}

/// Batched spec VERIFY: `[N, S]` rows (every slot's `[t1, drafts…]`, padded to the
/// group's widest), per-position SSM capture ON so each slot rolls back on its own,
/// post-norm hidden captured at the last row (`hidden_last` `[N,1,H]`) and at every
/// row (`hidden_all` `[N,S,H]`). Returns logits `[N, S, V]`; caller owns everything.
pub fn forwardMoeBatchedVerify(
    self: *Transformer,
    token_arr: mlx.mlx_array,
    ctxs: []const *ForwardCtx,
    rope_offsets: []const u32,
    hidden_last: *mlx.mlx_array,
    hidden_all: *mlx.mlx_array,
) !mlx.mlx_array {
    const width = mlx.getShape(token_arr)[1];
    const rows = @as(c_int, @intCast(ctxs.len)) * width;
    if (rows > gdn_batched_verify_rows_logged) {
        gdn_batched_verify_rows_logged = rows;
        log.info("[batched] gdn batched verify engaged (slots={d}, width={d})\n", .{ ctxs.len, width });
    }
    if (self.plugin) |b| return plugin_driver.forwardBatchedVerify(self, b, token_arr, ctxs, rope_offsets, hidden_last, hidden_all);
    return self.forwardMoeBatchedRows(token_arr, ctxs, rope_offsets, true, hidden_last, hidden_all);
}

/// Row `i` of `[N, S, V]` logits as an owned `[1, S, V]` array, for every row.
pub fn sliceBatchRows(allocator: std.mem.Allocator, s: mlx.mlx_stream, arr: mlx.mlx_array, n: usize) ![]mlx.mlx_array {
    const shape = mlx.getShape(arr);
    const out = try allocator.alloc(mlx.mlx_array, n);
    var built: usize = 0;
    errdefer {
        for (out[0..built]) |a| _ = mlx.mlx_array_free(a);
        allocator.free(out);
    }
    for (out, 0..) |*slot, i| {
        const i_c: c_int = @intCast(i);
        const start = [_]c_int{ i_c, 0, 0 };
        const stop = [_]c_int{ i_c + 1, shape[1], shape[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        slot.* = mlx.mlx_array_new();
        built = i + 1;
        try mlx.check(mlx.mlx_slice(slot, arr, &start, 3, &stop, 3, &strides, 3, s));
    }
    return out;
}

pub fn forwardMoeBatchedRows(
    self: *Transformer,
    token_arr: mlx.mlx_array,
    ctxs: []const *ForwardCtx,
    rope_offsets: []const u32,
    capture_ssm: bool,
    hidden_last: ?*mlx.mlx_array,
    hidden_all: ?*mlx.mlx_array,
) !mlx.mlx_array {
    self.fwd_gen +%= 1; // per-forward QSA scratch key
    const N: c_int = @intCast(ctxs.len);
    std.debug.assert(ctxs.len == rope_offsets.len);
    std.debug.assert(N >= 1);
    const ml = self.moe_layers orelse return error.BatchedGdnRequiresMoeLayers;

    // Per-slot RoPE offsets, each already carrying its own M-RoPE delta.
    var off_buf = try self.allocator.alloc(i32, rope_offsets.len);
    defer self.allocator.free(off_buf);
    for (rope_offsets, 0..) |o, i| {
        const d: i32 = if (ctxs[i].mrope_pos != null) ctxs[i].mrope_delta else 0;
        off_buf[i] = @as(i32, @intCast(o)) + d;
    }
    const off_shape = [_]c_int{N};
    const rope_offset_arr = mlx.mlx_array_new_data(off_buf.ptr, &off_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(rope_offset_arr);

    // Merge each linear layer's per-slot GDN state into one B=N entry.
    // A layer whose state is not initialized on EVERY slot cannot be
    // merged (widths would disagree) — the caller gates on
    // `batchedGdnReady`.
    const persist = self.persistent_group_state_override orelse true;
    self.ssm_group.allocator = self.allocator;
    const stable = persist and self.ssm_group.live and self.ssm_group.matches(ctxs);
    var kinds_buf: []SsmMergeKind = &.{};
    defer if (kinds_buf.len != 0) self.allocator.free(kinds_buf);
    const kinds: []const SsmMergeKind = if (stable) self.ssm_group.kinds else blk: {
        kinds_buf = try self.allocator.alloc(SsmMergeKind, ml.len);
        for (ml, 0..) |*lw, l| kinds_buf[l] = .{ .linear = lw.is_linear, .ple = lw.ple != null };
        break :blk kinds_buf;
    };
    // A trunk with no recurrent layer (MiMo) has no per-slot state to bind.
    const stateless = !self.hasRecurrentLayers();
    var merged: []SSMCacheEntry = &.{};
    var merged_owned = false;
    if (!stateless) {
        const bind = try ssmBindTickState(self.allocator, self.s, &self.ssm_group, ctxs, kinds, persist);
        merged = bind.merged;
        merged_owned = bind.owned;
    }
    defer if (merged_owned) {
        for (merged) |*m| ssmFreeMergedLayer(m);
        self.allocator.free(merged);
    };

    // Scratch ctx. `cache` is never read on this path — the batched branch
    // in `gatedFullAttnWith` goes to the slot caches — but the field is
    // non-optional, so slot 0's stands in. `moe_seq_offset` likewise only
    // feeds the scalar-RoPE path the batched branch replaces.
    var scratch_offset: usize = rope_offsets[0];
    var bctx: ForwardCtx = .{
        .cache = ctxs[0].cache,
        .moe_seq_offset = &scratch_offset,
        .ssm_entries = if (stateless) null else merged,
        .capture_hidden = hidden_last,
        .capture_hidden_all = hidden_all,
        .capture_ssm_seq = capture_ssm,
        .vision_embeddings = null,
        .batch_slots = ctxs,
        .batch_rope_offsets = rope_offset_arr,
        .ple_defer = self.qwen4 != null,
    };
    // A verify's captures live on the merged entries until split below.
    defer if (capture_ssm) {
        for (merged) |*m| ssmFreeSpecCapture(m);
    };

    // qwen4_exp rides the same driver: GDN + PLE window merged here, QSA
    // key histories reached per slot through `batch_slots` (kv differs).
    const logits = (if (self.qwen4 != null) self.forwardQwen4With(&bctx, token_arr) else self.forwardMoeWith(&bctx, token_arr)) catch |e| {
        self.discardDeferredPle(&bctx);
        return e;
    };
    errdefer {
        self.discardDeferredPle(&bctx);
        _ = mlx.mlx_array_free(logits);
    }
    try self.flushDeferredPle(&bctx);
    if (stateless) return logits;

    if (persist) {
        if (capture_ssm) {
            for (merged, 0..) |*m, li| {
                if (!kinds[li].linear and !kinds[li].ple) continue;
                for (ctxs, 0..) |c, i| try splitSpecCaptures(self.s, m, &c.ssm_entries.?[li], i);
            }
        }
        if (!stable) ssmBindSlotRefs(&self.ssm_group, ctxs);
    } else {
        try ssmSplitAll(self.s, merged, ctxs, kinds);
    }
    return logits;
}
