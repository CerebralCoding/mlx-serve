//! A recurrent layer's state across the forward's slots: one slot's state as it is, N slots' stacked on the batch
//! axis (and split back after). Blocks read and write recurrent state only through these.

const ops = @import("../../../mlx/ops.zig");
const abi = @import("../../../abi/plugin.zig");
const block = @import("../block.zig");

const Array = ops.Array;
const Ctx = block.Ctx;

/// Slots one forward can stack (the engine's `MAX_BATCH_ROWS`).
pub const MAX_SLOTS = 32;

/// The batch axis of `slot`'s arrays: the per-row state sequence is `[S, B, ...]`.
fn axis(slot: abi.StateSlot) usize {
    return if (slot == .state_seq) 1 else 0;
}

/// The current layer's `slot` over every slot of the forward, owned by `c.g`; null when no slot holds it yet. A
/// batch where only some slots hold it is refused (their widths would disagree).
pub fn get(c: *const Ctx, slot: abi.StateSlot) !?Array {
    const o = try c.cacheOps();
    if (c.caches.len > MAX_SLOTS) return error.TooManySlots;
    var parts: [MAX_SLOTS]Array = undefined;
    var held: usize = 0;
    for (c.caches, 0..) |h, i| {
        const a = o.state(h, c.layer, slot);
        if (a.ctx != null) held += 1;
        parts[i] = .{ .ctx = a.ctx };
    }
    if (held == 0) return null;
    if (held != c.caches.len) return error.BatchedStateUnready;
    if (c.caches.len == 1) return try c.g.keep(parts[0]);
    return try c.g.concat(parts[0..c.caches.len], @intCast(axis(slot)));
}

/// Writes `value` as the current layer's `slot`, each forward slot taking its row.
pub fn put(c: *const Ctx, slot: abi.StateSlot, value: Array) !void {
    const o = try c.cacheOps();
    if (c.caches.len == 1) return ok(o.set_state(c.caches[0], c.layer, slot, .{ .ctx = value.ctx }));
    var t = c.g.scope();
    defer t.deinit();
    for (c.caches, 0..) |h, i| {
        const row = try t.sliceAxis(value, axis(slot), @intCast(i), @intCast(i + 1));
        try ok(o.set_state(h, c.layer, slot, .{ .ctx = row.ctx }));
    }
}

fn ok(v: abi.Verdict) !void {
    if (!v.isOk()) return error.CacheOpFailed;
}
