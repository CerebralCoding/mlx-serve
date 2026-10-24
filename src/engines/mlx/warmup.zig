//! engines/mlx/warmup, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns warmup, SpecWarmSlot, specWarmIds, specWarmEvalFree, specWarmVerify, MAX_WARM_SEQ, and 5 more.

const std = @import("std");
const plugin_settle = @import("plugin/settle.zig");
const mlx = @import("../../mlx/ffi.zig");
const KVQuantConfig = @import("cache/kv_quant.zig").KVQuantConfig;
const log = @import("../../sdk/util/log.zig");
const io_util_mod = @import("../../sdk/util/io_util.zig");
const Qwen4MtpProject = @import("../../plugins/qwen4_exp/mtp.zig").Qwen4MtpProject;
const ssmFreeQsaState = @import("../../plugins/qwen4_exp/qsa/history.zig").ssmFreeQsaState;
const verifyQmmNaxAvailable = @import("../../sdk/blocks/linear/verify_qmm_nax.zig").verifyQmmNaxAvailable;
const MAX_BATCH_ROWS = @import("batched/layered.zig").MAX_BATCH_ROWS;
const ForwardCtx = @import("forward_ctx.zig").ForwardCtx;
const KVCache = @import("kv/cache.zig").KVCache;
const testing = @import("kv/cache.zig").testing;
const init = @import("load/init.zig").init;
const SSMCacheEntry = @import("ssm/entry.zig").SSMCacheEntry;
const ssmFreeSpecCapture = @import("ssm/entry.zig").ssmFreeSpecCapture;
const nextSsmMemberGen = @import("ssm/group.zig").nextSsmMemberGen;
const Transformer = @import("transformer.zig").Transformer;
const deinit = @import("transformer.zig").Transformer.deinit;

/// Pre-fault weight pages and trigger first-touch kernel compiles before
/// the first real request so cold prefill doesn't pay 800+ms of GPU page
/// faulting (measured on Gemma 4 E4B 4-bit). Runs three forward passes:
///   1. [1, 1] decode-shape: faults embed matrix + compiles decode kernel
///   2. [1, 8] prefill-shape: compiles short-prefill kernel
/// then resets the cache so the first real request starts from clean state.
/// Idempotent — calling twice is wasted work but not incorrect.
pub fn warmup(self: *Transformer) !void {
    if (self.mlx_stream_ext != null) return; // mlx-stream warms its own kernels at construction
    const dummy_id: i32 = 0; // BOS-ish placeholder; the actual id doesn't matter for warmup
    const decode_shape = [_]c_int{ 1, 1 };
    const decode_input = mlx.mlx_array_new_data(&dummy_id, &decode_shape, 2, .int32);
    defer _ = mlx.mlx_array_free(decode_input);
    const decode_logits = try self.forward(decode_input);
    _ = mlx.mlx_array_free(decode_logits);
    // Materialize the cache update so subsequent forwards see initialized entries.
    {
        const eval_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(eval_vec);
        for (self.cache.entries) |*entry| {
            if (!entry.initialized) continue;
            _ = mlx.mlx_vector_array_append_value(eval_vec, entry.keys);
            _ = mlx.mlx_vector_array_append_value(eval_vec, entry.values);
        }
        _ = mlx.mlx_eval(eval_vec);
    }
    _ = mlx.mlx_clear_cache();

    // Reset before the prefill-shape pass so we exercise the cold-init path,
    // not the partial-cache path.
    try self.resetCache();

    // A verify-width pass, then a prompt-width one (past 16 rows the lane matmul runs 32-row blocks).
    const ids: [32]i32 = @splat(0);
    for ([_]c_int{ 8, 32 }) |width| {
        const prefill_shape = [_]c_int{ 1, width };
        const prefill_input = mlx.mlx_array_new_data(&ids, &prefill_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(prefill_input);
        const prefill_logits = try self.forward(prefill_input);
        _ = mlx.mlx_array_free(prefill_logits);
        {
            const eval_vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(eval_vec);
            for (self.cache.entries) |*entry| {
                if (!entry.initialized) continue;
                _ = mlx.mlx_vector_array_append_value(eval_vec, entry.keys);
                _ = mlx.mlx_vector_array_append_value(eval_vec, entry.values);
            }
            _ = mlx.mlx_eval(eval_vec);
        }
        _ = mlx.mlx_clear_cache();
        try self.resetCache();
    }
}

// A throw-away slot for the spec warm-up: its own KV, SSM entries and context, so nothing
// the warm-up forwards reaches the model's own state or the round-cost table.
pub const SpecWarmSlot = struct {
    cache: KVCache,
    off: usize = 0,
    entries: []SSMCacheEntry,
    ctx: ForwardCtx = undefined,

    fn init(xfm: *Transformer, kv_config: KVQuantConfig) !*SpecWarmSlot {
        const alloc = xfm.allocator;
        const n_layers = xfm.config.num_hidden_layers;
        const sl = try alloc.create(SpecWarmSlot);
        errdefer alloc.destroy(sl);
        var cache = try KVCache.initWithConfig(alloc, n_layers, kv_config);
        errdefer cache.deinit();
        sl.* = .{ .cache = cache, .entries = try alloc.alloc(SSMCacheEntry, n_layers) };
        for (sl.entries) |*e| e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
        sl.ctx = .{
            .cache = &sl.cache,
            .moe_seq_offset = &sl.off,
            .ssm_entries = sl.entries,
            .ssm_member_gen = nextSsmMemberGen(),
            .capture_hidden = null,
            .vision_embeddings = null,
        };
        return sl;
    }

    fn deinit(sl: *SpecWarmSlot, alloc: std.mem.Allocator) void {
        for (sl.entries) |*e| {
            ssmFreeSpecCapture(e);
            _ = mlx.mlx_array_free(e.conv_state);
            _ = mlx.mlx_array_free(e.ssm_state);
            ssmFreeQsaState(e);
        }
        alloc.free(sl.entries);
        sl.cache.deinit();
        alloc.destroy(sl);
    }
};

pub fn specWarmIds(buf: []i32) mlx.mlx_array {
    for (buf, 0..) |*t, i| t.* = @intCast((i * 7919 + 13) % 40000);
    const shape = [_]c_int{ 1, @intCast(buf.len) };
    return mlx.mlx_array_new_data(buf.ptr, &shape, 2, .int32);
}

pub fn specWarmEvalFree(arrays: []const mlx.mlx_array) void {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (arrays) |a| {
        if (a.ctx != null) _ = mlx.mlx_vector_array_append_value(vec, a);
    }
    _ = mlx.mlx_eval(vec);
    for (arrays) |a| {
        if (a.ctx != null) _ = mlx.mlx_array_free(a);
    }
}

pub fn specWarmVerify(self: *Transformer, slots: []const *SpecWarmSlot, seq: usize) !void {
    var ids: [MAX_BATCH_ROWS][MAX_WARM_SEQ]i32 = undefined;
    var rows: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
    var ctxs: [MAX_BATCH_ROWS]*ForwardCtx = undefined;
    defer for (rows[0..slots.len]) |r| {
        _ = mlx.mlx_array_free(r);
    };
    for (slots, 0..) |sl, i| {
        rows[i] = specWarmIds(ids[i][0..seq]);
        ctxs[i] = &sl.ctx;
    }
    var logits: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
    var last: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
    var all: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
    try self.forwardRowAxisVerify(rows[0..slots.len], ctxs[0..slots.len], logits[0..slots.len], last[0..slots.len], all[0..slots.len], null, null);
    specWarmEvalFree(logits[0..slots.len]);
    specWarmEvalFree(last[0..slots.len]);
    specWarmEvalFree(all[0..slots.len]);
}

pub const MAX_WARM_SEQ: usize = MAX_WIDTH_WARM + 1;
pub const MAX_WIDTH_WARM: usize = 8;

// The verify at each draft width, the grouped verify at each row total and the head's own step
// each JIT their Metal pipelines on first dispatch; left to the first live request that compile
// lands inside a measured round and the table folds it as that width's price.
pub fn warmupSpecVerify(self: *Transformer, max_width: u32, kv_config: KVQuantConfig) !void {
    if (self.qwen4 == null or self.qwen4_mtp == null) return;
    var lap = io_util_mod.Stopwatch.init(std.Io.Threaded.global_single_threaded.io());
    const width_cap: usize = @min(@max(max_width, 1), MAX_WIDTH_WARM);
    const group_cap: usize = if (verifyQmmNaxAvailable()) 16 else 7;
    const max_rows: usize = @min(specWarmGroupRows(group_cap), MAX_BATCH_ROWS);

    var slots: [MAX_BATCH_ROWS]*SpecWarmSlot = undefined;
    var n_slots: usize = 0;
    defer {
        for (slots[0..n_slots]) |sl| {
            self.ssmGroupRelease(&sl.ctx) catch {};
            plugin_settle.release(self, &sl.ctx);
            sl.deinit(self.allocator);
        }
        _ = mlx.mlx_clear_cache();
    }
    var head_in = mlx.mlx_array_new();
    defer if (head_in.ctx != null) {
        _ = mlx.mlx_array_free(head_in);
    };
    while (n_slots < max_rows) : (n_slots += 1) {
        slots[n_slots] = try SpecWarmSlot.init(self, kv_config);
        var prompt_ids: [8]i32 = undefined;
        const prompt = specWarmIds(&prompt_ids);
        defer _ = mlx.mlx_array_free(prompt);
        var stream = mlx.mlx_array_new();
        slots[n_slots].ctx.capture_stream_all = if (n_slots == 0) &stream else null;
        const logits = try self.forwardWith(&slots[n_slots].ctx, prompt);
        slots[n_slots].ctx.capture_stream_all = null;
        if (n_slots == 0 and stream.ctx != null) {
            const sh = mlx.getShape(stream);
            try mlx.check(mlx.mlx_slice(&head_in, stream, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ sh[0], 1, sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, self.s));
        }
        specWarmEvalFree(&[_]mlx.mlx_array{ logits, stream });
    }

    var w: usize = 1;
    while (w <= width_cap) : (w += 1) try self.specWarmVerify(slots[0..1], w + 1);

    var n: usize = 2;
    var group_seq_max: usize = 0;
    while (n <= n_slots) : (n += 1) {
        const per_row_cap = specWarmGroupSeq(group_cap, n, width_cap) - 1;
        group_seq_max = @max(group_seq_max, per_row_cap + 1);
        var seq: usize = 2;
        while (seq <= per_row_cap + 1) : (seq += 1) try self.specWarmVerify(slots[0..n], seq);
    }
    if (head_in.ctx != null) try self.specWarmHead(head_in);
    log.info("[spec-warmup] verify widths 1..{d} solo, group rows 2..{d} at up to {d} tokens each, head width 1 ({d} ms).\n", .{ width_cap, n_slots, group_seq_max, lap.read() / std.time.ns_per_ms });
}

// Mirror of `scheduler.mtpSubGroupSize`: the slots one row-axis tick can carry at this row
// budget (NAX wide lane 16, stock lane 7) and the drafts each slot gets.
pub fn specWarmGroupRows(group_cap: usize) usize {
    return if (group_cap >= 16) 4 else 3;
}

pub fn specWarmGroupSeq(group_cap: usize, rows: usize, width_cap: usize) usize {
    return @min(group_cap / rows - 1, width_cap) + 1;
}

pub fn specWarmHead(self: *Transformer, stream_row: mlx.mlx_array) !void {
    var st = try self.qwen4MtpStateNew();
    defer st.deinit();
    var id_buf: [1]i32 = undefined;
    const id_arr = specWarmIds(&id_buf);
    defer _ = mlx.mlx_array_free(id_arr);
    for ([_]Qwen4MtpProject{ .mixed_last_row, .none }, 0..) |project, i| {
        const out = try self.qwen4MtpForwardOn(&st, stream_row, id_arr, @intCast(1 + i), null, project);
        specWarmEvalFree(&[_]mlx.mlx_array{ out.logits, out.stream, out.mixed });
    }
}

// ── tests ──

test "spec warm-up covers exactly the group shapes the scheduler can build" {
    try testing.expectEqual(@as(usize, 4), specWarmGroupRows(16));
    try testing.expectEqual(@as(usize, 3), specWarmGroupRows(7));
    try testing.expectEqual(@as(usize, 7), specWarmGroupSeq(16, 2, 6));
    try testing.expectEqual(@as(usize, 5), specWarmGroupSeq(16, 3, 6));
    try testing.expectEqual(@as(usize, 4), specWarmGroupSeq(16, 4, 6));
    try testing.expectEqual(@as(usize, 3), specWarmGroupSeq(7, 2, 6));
    try testing.expectEqual(@as(usize, 2), specWarmGroupSeq(7, 3, 6));
    try testing.expectEqual(@as(usize, 3), specWarmGroupSeq(16, 2, 2));
}
