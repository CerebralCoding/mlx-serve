//! The engine's `CacheOps` over today's `KVCache` and SSM entries: a pure adapter that makes the same calls the
//! legacy forward paths make directly. A `Handle` points at a `Slot`.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const abi = @import("../../../abi/plugin.zig");
const KVCache = @import("cache.zig").KVCache;
const KVQuantConfig = @import("../cache/kv_quant.zig").KVQuantConfig;
const SSMCacheEntry = @import("../ssm/entry.zig").SSMCacheEntry;
const DenseKVView = @import("../../../sdk/blocks/attention/kv_view.zig").DenseKVView;
const MAX_BATCH_ROWS = @import("../batched/layered.zig").MAX_BATCH_ROWS;

/// What a `Handle` points at: one slot's KV cache and its recurrent entries (one per layer, or none).
pub const Slot = struct {
    kv: *KVCache,
    ssm: []SSMCacheEntry = &.{},
    /// `KVCache.appendFromStacked`'s one-quantize arm for a stacked append (the Transformer's
    /// `batched_kv_append_override orelse true`).
    batched_append: bool = true,
    /// The request's `ForwardCtx.kv_attn_fused`, handed to attention on every view.
    packed_reads: bool = false,

    pub fn handle(self: *Slot) abi.Handle {
        return @ptrCast(self);
    }
};

pub const ops: abi.CacheOps = .{
    .len = len,
    .append = append,
    .append_ring = appendRing,
    .view = view,
    .release = release,
    .append_rows = appendRows,
    .state = state,
    .set_state = setState,
    .state_rows = Rows.stateRows,
    .set_state_rows = Rows.setStateRows,
};

const Rows = @import("../../../sdk/util/state_rows.zig").Rows(state, setState);

fn slotOf(h: abi.Handle) *Slot {
    return @ptrCast(@alignCast(h.?));
}

fn mx(a: abi.Array) mlx.mlx_array {
    return .{ .ctx = a.ctx };
}

fn ab(a: mlx.mlx_array) abi.Array {
    return .{ .ctx = a.ctx };
}

fn failed(e: anyerror) abi.Verdict {
    return .{ .code = .failed, .why = abi.Str.of(@errorName(e)) };
}

fn toAbi(v: DenseKVView, packed_reads: bool) abi.KvView {
    return .{
        .k = ab(v.k),
        .v = ab(v.v),
        .owned = v.owned,
        .packed_reads = packed_reads,
        .bits = if (v.has_quant_triple) v.bits else 0,
        .group_size = v.group_size,
        .capacity = @intCast(v.kv_cap),
        .k_q = ab(v.k_triple_q),
        .k_scales = ab(v.k_triple_scales),
        .k_biases = ab(v.k_triple_biases),
        .v_q = ab(v.v_triple_q),
        .v_scales = ab(v.v_triple_scales),
        .v_biases = ab(v.v_triple_biases),
    };
}

fn len(h: abi.Handle, layer: u32) callconv(.c) u64 {
    return slotOf(h).kv.seqLen(layer);
}

fn append(h: abi.Handle, layer: u32, k: abi.Array, v: abi.Array, keep: u32, s: abi.Stream, out: *abi.KvView) callconv(.c) abi.Verdict {
    const sl = slotOf(h);
    const r = sl.kv.update(layer, mx(k), mx(v), .{ .ctx = s.ctx }, keep) catch |e| return failed(e);
    out.* = toAbi(r, sl.packed_reads);
    return .ok;
}

fn appendRing(h: abi.Handle, layer: u32, k: abi.Array, v: abi.Array, keep: u32, window: u32, ring: u32, s: abi.Stream, out: *abi.KvView) callconv(.c) abi.Verdict {
    const sl = slotOf(h);
    const r = sl.kv.updateSliding(layer, mx(k), mx(v), .{ .ctx = s.ctx }, keep, window, ring) catch |e| return failed(e);
    out.* = toAbi(r, sl.packed_reads);
    return .ok;
}

fn view(h: abi.Handle, layer: u32, s: abi.Stream, out: *abi.KvView) callconv(.c) abi.Verdict {
    const sl = slotOf(h);
    const r = sl.kv.denseView(layer, .{ .ctx = s.ctx }) catch |e| return failed(e);
    out.* = toAbi(r, sl.packed_reads);
    return .ok;
}

fn release(v: *abi.KvView) callconv(.c) void {
    if (v.owned) {
        _ = mlx.mlx_array_free(mx(v.k));
        _ = mlx.mlx_array_free(mx(v.v));
    }
    v.* = .{};
}

fn appendRows(caches: [*]const abi.Handle, n: u32, layer: u32, k: abi.Array, v: abi.Array, keep: u32, s: abi.Stream) callconv(.c) abi.Verdict {
    if (n == 0) return .ok;
    if (n > MAX_BATCH_ROWS) return abi.Verdict.refuse(.too_large, "more batched rows than MAX_BATCH_ROWS");
    var kvs: [MAX_BATCH_ROWS]*KVCache = undefined;
    for (caches[0..n], 0..) |h, i| kvs[i] = slotOf(h).kv;
    KVCache.appendFromStacked(mx(k), mx(v), kvs[0..n], layer, .{ .ctx = s.ctx }, keep, slotOf(caches[0]).batched_append) catch |e| return failed(e);
    return .ok;
}

fn state(h: abi.Handle, layer: u32, which: abi.StateSlot) callconv(.c) abi.Array {
    const sl = slotOf(h);
    if (layer >= sl.ssm.len) return .{};
    const e = &sl.ssm[layer];
    if (!e.initialized) return .{};
    return ab(switch (which) {
        .conv => e.conv_state,
        .ssm => e.ssm_state,
        .conv_input => e.spec_conv_input,
        .state_seq => e.spec_state_seq,
    });
}

fn setState(h: abi.Handle, layer: u32, which: abi.StateSlot, value: abi.Array) callconv(.c) abi.Verdict {
    const sl = slotOf(h);
    if (layer >= sl.ssm.len) return abi.Verdict.refuse(.unsupported, "layer has no recurrent state");
    const e = &sl.ssm[layer];
    const dst = switch (which) {
        .conv => &e.conv_state,
        .ssm => &e.ssm_state,
        .conv_input => &e.spec_conv_input,
        .state_seq => &e.spec_state_seq,
    };
    var r = mlx.mlx_array_new();
    _ = mlx.mlx_array_set(&r, mx(value));
    if (dst.ctx != null) _ = mlx.mlx_array_free(dst.*);
    dst.* = r;
    if (which == .conv or which == .ssm) e.initialized = true;
    return .ok;
}

// ── tests ──

const testing = std.testing;
const conformance = @import("../../../sdk/testing/cache_ops.zig");

fn newEntries(n: usize) ![]SSMCacheEntry {
    const es = try testing.allocator.alloc(SSMCacheEntry, n);
    for (es) |*e| e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
    return es;
}

fn freeEntries(es: []SSMCacheEntry) void {
    for (es) |*e| {
        _ = mlx.mlx_array_free(e.conv_state);
        _ = mlx.mlx_array_free(e.ssm_state);
    }
    testing.allocator.free(es);
}

test "bases cache ops: the KVCache adapter passes the CacheOps conformance checks" {
    var c1 = try KVCache.init(testing.allocator, conformance.LAYERS);
    defer c1.deinit();
    var c2 = try KVCache.init(testing.allocator, conformance.LAYERS);
    defer c2.deinit();
    const e1 = try newEntries(conformance.LAYERS);
    defer freeEntries(e1);
    const e2 = try newEntries(conformance.LAYERS);
    defer freeEntries(e2);
    var s1: Slot = .{ .kv = &c1, .ssm = e1 };
    var s2: Slot = .{ .kv = &c2, .ssm = e2 };
    try conformance.expectConforms(&ops, s1.handle(), s2.handle());
}

// Bar: a quantized cache hands the packed triples to the view and owns the dequantized arrays it returns.
test "bases cache ops: a quantized KVCache's view carries its triples" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var c = try KVCache.initWithConfig(testing.allocator, 1, KVQuantConfig.affine(8));
    defer c.deinit();
    var sl: Slot = .{ .kv = &c };
    var k = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k);
    try mlx.check(mlx.mlx_ones(&k, &[_]c_int{ 1, 2, 3, 64 }, 4, .bfloat16, s));
    var out: abi.KvView = .{};
    try testing.expect(ops.append(sl.handle(), 0, ab(k), ab(k), 0, .{ .ctx = s.ctx }, &out).isOk());
    defer ops.release(&out);
    try testing.expect(out.owned and out.bits == 8 and out.group_size == 64);
    try testing.expect(out.k_q.ctx != null and out.v_biases.ctx != null and out.capacity >= 3);
    try testing.expectEqualSlices(c_int, &.{ 1, 2, 3, 64 }, mlx.getShape(mx(out.k)));
    try testing.expectEqual(@as(u64, 3), ops.len(sl.handle(), 0));
}
