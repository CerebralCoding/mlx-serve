//! A table forward's slots through `CacheOps`: each slot's cache and first position, a KV view the attention layers
//! append and read, and a KDA layer on the slot's recurrent state. The shared GatedDeltaNet (`gatedDeltaNet`) takes
//! an `SSMCacheEntry`; `kda` hands it the slot's state and writes what it leaves back (Ling 3.0 shares this).

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;
const SSMCacheEntry = @import("../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const LinearAttnWeights = @import("../../sdk/blocks/gdn/gated_delta_net.zig").LinearAttnWeights;

const Array = mlx.mlx_array;

fn mx(a: abi.Array) Array {
    return .{ .ctx = a.ctx };
}

fn ab(a: Array) abi.Array {
    return .{ .ctx = a.ctx };
}

fn ok(v: abi.Verdict) !void {
    if (!v.isOk()) return error.CacheOpFailed;
}

/// One slot's dense K/V as attention reads them, freed with `deinit`.
pub const View = struct {
    raw: abi.KvView,
    ops: *const abi.CacheOps,
    k: Array,
    v: Array,

    pub fn deinit(self: *View) void {
        self.ops.release(&self.raw);
    }
};

pub const Slots = struct {
    ops: *const abi.CacheOps,
    caches: []const abi.Handle,
    positions: []const u32,

    pub fn len(self: Slots) usize {
        return self.caches.len;
    }

    /// Appends `k`/`v` `[B, H, T, D]` at `layer` of slot `i`; the view holds every row.
    pub fn append(self: Slots, i: usize, layer: u32, k: Array, v: Array, s: mlx.mlx_stream) !View {
        var raw: abi.KvView = .{};
        try ok(self.ops.append(self.caches[i], layer, ab(k), ab(v), 0, .{ .ctx = s.ctx }, &raw));
        return .{ .raw = raw, .ops = self.ops, .k = mx(raw.k), .v = mx(raw.v) };
    }

    /// The rows `layer` of slot `i` already holds.
    pub fn view(self: Slots, i: usize, layer: u32, s: mlx.mlx_stream) !View {
        var raw: abi.KvView = .{};
        try ok(self.ops.view(self.caches[i], layer, .{ .ctx = s.ctx }, &raw));
        return .{ .raw = raw, .ops = self.ops, .k = mx(raw.k), .v = mx(raw.v) };
    }

    /// The state `which` of `layer` in slot `i`, borrowed; null before its first write.
    pub fn state(self: Slots, i: usize, layer: u32, which: abi.StateSlot) Array {
        return mx(self.ops.state(self.caches[i], layer, which));
    }

    /// Writes `value` as the state `which` (the cache takes its own reference).
    pub fn setState(self: Slots, i: usize, layer: u32, which: abi.StateSlot, value: Array) !void {
        try ok(self.ops.set_state(self.caches[i], layer, which, ab(value)));
    }

    /// Evaluates `h` with every recurrent state the slots hold (a prompt chunk's memory bound).
    pub fn evalWith(self: Slots, h: Array, layers: u32) !void {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        _ = mlx.mlx_vector_array_append_value(vec, h);
        for (0..self.caches.len) |i| for (0..layers) |li| {
            const c = self.state(i, @intCast(li), .conv);
            if (c.ctx != null) _ = mlx.mlx_vector_array_append_value(vec, c);
        };
        try mlx.check(mlx.mlx_eval(vec));
    }
};

fn takeRef(a: Array) Array {
    var r = mlx.mlx_array_new();
    if (a.ctx != null) _ = mlx.mlx_array_set(&r, a);
    return r;
}

fn freeIf(a: Array) void {
    if (a.ctx != null) _ = mlx.mlx_array_free(a);
}

/// Slot `i`'s recurrent state at `layer` as the entry `gatedDeltaNet` reads, holding its own references.
pub fn entryOf(sl: Slots, i: usize, layer: u32) SSMCacheEntry {
    const conv = sl.state(i, layer, .conv);
    const ssm = sl.state(i, layer, .ssm);
    return .{ .conv_state = takeRef(conv), .ssm_state = takeRef(ssm), .initialized = conv.ctx != null or ssm.ctx != null };
}

/// Writes the entry's state (and a verify's per-row capture) back to slot `i`, then frees the entry.
pub fn putEntry(sl: Slots, i: usize, layer: u32, e: *SSMCacheEntry) !void {
    defer {
        for ([_]Array{ e.conv_state, e.ssm_state, e.spec_conv_input, e.spec_state_seq }) |a| freeIf(a);
        e.* = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false };
    }
    if (e.conv_state.ctx != null) try sl.setState(i, layer, .conv, e.conv_state);
    if (e.ssm_state.ctx != null) try sl.setState(i, layer, .ssm, e.ssm_state);
    if (e.spec_conv_input.ctx != null) try sl.setState(i, layer, .conv_input, e.spec_conv_input);
    if (e.spec_state_seq.ctx != null) try sl.setState(i, layer, .state_seq, e.spec_state_seq);
}

/// A KDA (GatedDeltaNet) layer on slot `i`'s rows `x`.
pub fn kda(xfm: *Transformer, sl: Slots, i: usize, x: Array, la: *const LinearAttnWeights, li: usize, batch: c_int, seq: c_int, is_prefill: bool) !Array {
    var e = entryOf(sl, i, @intCast(li));
    errdefer for ([_]Array{ e.conv_state, e.ssm_state, e.spec_conv_input, e.spec_state_seq }) |a| freeIf(a);
    const y = try xfm.gatedDeltaNet(x, la, &e, li, batch, seq, is_prefill);
    errdefer _ = mlx.mlx_array_free(y);
    try putEntry(sl, i, @intCast(li), &e);
    return y;
}
