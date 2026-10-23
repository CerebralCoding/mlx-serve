//! The dense decoder's attention with its comptime extras (Gemma 3/4, Muse, Spark, and with none of them llama/qwen3):
//! sliding layers reading a block-sized tail of the cache under band masks, layers reading an earlier layer's KV,
//! a parameter-free value norm, K=V layers, and a sigmoid output gate per channel or per head. Batched rows too.
//! copied_from: engines/mlx/forward/dense.zig (`forwardStandardWith`'s attention, no `prefill_mask_add`),
//! engines/mlx/batched/dense.zig (`forwardBatchedDecode`'s attention), sdk/blocks/attention/gates.zig
//! (`attnOutGate`).

const std = @import("std");
const abi = @import("../../../abi/plugin.zig");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const block = @import("../block.zig");
const kv = @import("../kv.zig");
const linear = @import("linear.zig");
const attention = @import("attention.zig");
const sliding = @import("sliding.zig");
const attn256 = @import("../../blocks/attention/attn256.zig");
const sdpa = @import("../../blocks/attention/sdpa.zig");
const kvq = @import("../../blocks/attention/kv_quant_attn.zig");
const batched = @import("../../blocks/attention/batched.zig");
const gates = @import("../../blocks/attention/gates.zig");
const qk_norm_rope = @import("../../blocks/rope/qk_norm_rope.zig");
const DenseKVView = @import("../../blocks/attention/kv_view.zig").DenseKVView;
const FUSED_ROWS_MAX_M = @import("../../blocks/linear/fused_rows.zig").FUSED_ROWS_MAX_M;

const Array = ops.Array;
const Graph = ops.Graph;
const Ctx = block.Ctx;
const Linear = linear.Linear;
const Joined = linear.Joined;
pub const Geo = attention.Geo;

const perm = [_]c_int{ 0, 2, 1, 3 };
const MAX_SLOTS = attention.MAX_SLOTS;

pub const Gate = enum { none, channel, head };

pub const Options = struct {
    /// Layers with a `window` attend only their last `window` keys.
    sliding: bool = false,
    /// Layers with a `kv_source` read that layer's KV and project none of their own.
    kv_sharing: bool = false,
    /// A parameter-free RMS norm on values (`v_norm`, ones) and K=V layers (`k_eq_v`).
    value_norm: bool = false,
    /// `sigmoid(gate(x))` multiplies the attention output before o_proj.
    gate: Gate = .none,
};

pub fn DecoderAttention(comptime opts: Options) type {
    return struct {
        pub const kind: block.Kind = .mixer;
        pub const supports_batched_rows = true;
        pub const Weights = struct {
            q: Linear,
            k: Linear = .{},
            v: Linear = .{},
            o: Linear,
            q_norm: Array = .{},
            k_norm: Array = .{},
            /// q|k|v (| gate) joined for one-row steps.
            qkv: Joined = .{},
            geo: Geo,
            /// Keys a sliding layer attends; 0 = every key (a global layer).
            window: u32 = 0,
            kv_source: ?u32 = null,
            v_norm: Array = .{},
            k_eq_v: bool = false,
            out_gate: Linear = .{},
        };

        fn shared(w: *const Weights) ?u32 {
            return if (opts.kv_sharing) w.kv_source else null;
        }
        fn windowOf(w: *const Weights) u32 {
            return if (opts.sliding) w.window else 0;
        }

        /// k and v `[N, kv_heads, S, D]` before the cache: normed, roped (k) by `ropeFn`.
        fn ownKv(w: *const Weights, c: *Ctx, n: c_int, own_k: Array, own_v: Array, roped: anytype) ![2]Array {
            const g = c.g;
            const geo = &w.geo;
            const kv_shape = [_]c_int{ n, c.rows, geo.kv_heads, geo.head_dim };
            const k_r = try g.reshape(own_k, &kv_shape);
            const v_r = try g.reshape(own_v, &kv_shape);
            const k_n = if (w.k_norm.ctx != null) try g.rmsNorm(k_r, w.k_norm, c.shared.eps) else k_r;
            const v_n = if (opts.value_norm and w.v_norm.ctx != null) try g.rmsNorm(v_r, w.v_norm, c.shared.eps) else v_r;
            return .{ try roped.apply(g, try g.transpose(k_n, &perm)), try g.transpose(v_n, &perm) };
        }

        pub fn forward(w: *const Weights, c: *Ctx, x: Array) !Array {
            if (c.mode == .batched_rows) return batchedForward(w, c, x);
            const g = c.g;
            const geo = &w.geo;
            const s = c.rows;
            const offset: c_int = @intCast(c.offset());
            const src = shared(w);
            const kev = opts.value_norm and w.k_eq_v;

            const fused_rows = w.qkv.built() and s <= FUSED_ROWS_MAX_M;
            const all: Array = if (fused_rows) try w.qkv.lin.forward(g, x) else .{};
            const q = if (fused_rows) try w.qkv.part(g, all, 0) else try w.q.forward(g, x);
            const q_r = try g.reshape(q, &.{ 1, s, geo.heads, geo.head_dim });
            var own_k: Array = .{};
            var own_v: Array = .{};
            if (src == null) {
                own_k = if (fused_rows) try w.qkv.part(g, all, 1) else try w.k.forward(g, x);
                own_v = if (kev) own_k else if (fused_rows) try w.qkv.part(g, all, 2) else try w.v.forward(g, x);
            }
            // The legacy decode step at hd 256 runs one fused norm+RoPE kernel this block does not carry.
            const has_vn = opts.value_norm and w.v_norm.ctx != null;
            if (src == null and !kev and !geo.nope and s == 1 and geo.head_dim == 256 and (w.q_norm.ctx != null) == (w.k_norm.ctx != null) and
                (w.q_norm.ctx != null or !has_vn) and qk_norm_rope.qkNormRopeFusedEnabled())
                return error.FusedQkNormRope256NotPorted;

            const q_n = if (w.q_norm.ctx != null) try g.rmsNorm(q_r, w.q_norm, c.shared.eps) else q_r;
            const q_rope = try ropeAt(g, try g.transpose(q_n, &perm), geo, offset);

            const window = windowOf(w);
            const total_kv = offset + s;
            const sv = sliding.view(window, geo.head_dim, false, total_kv, s);
            var view: abi.KvView = if (src) |l|
                try kv.view(c, c.cache(), l)
            else blk: {
                const kvs = try ownKv(w, c, 1, own_k, own_v, Roper{ .geo = geo, .offset = offset });
                break :blk try kv.append(c, kvs[0], kvs[1], if (window > 0) sv.span else 0);
            };
            defer kv.release(c, &view);
            const dv = kv.dense(view);

            // The mode and mask the dense arms and the packed kernels share.
            var mode: [:0]const u8 = if (c.wide()) "causal" else "";
            var mask: Array = .{};
            if (window > 0) {
                const w_c: c_int = @intCast(window);
                if (c.wide()) {
                    mode = "array";
                    if (!sv.band_in_kernel) mask = try sliding.bandMask(g, s, sv.kv_len, w_c);
                } else {
                    const held: c_int = @intCast((try c.cacheOps()).len(c.cache(), src orelse c.layer));
                    if (held > w_c) {
                        mode = "array";
                        mask = try sliding.decodeMask(g, @min(total_kv, w_c), w_c);
                    }
                }
            }

            var out: ?Array = null;
            if (view.packed_reads and kvq.kvAttnFusedEligible(&dv, s)) {
                if (std.mem.eql(u8, mode, "array") and mask.ctx == null) mask = try sliding.bandMask(g, s, sv.kv_len, @intCast(window));
                out = try own(g, try kvq.qkvAttnDecodeKernel(g.s, q_rope, &dv, geo.scale, mode, mask));
                if (out != null) kvq.logKvAttnFusedEngaged(&dv, q_rope, s);
            } else if (view.packed_reads and kvq.kvAttnVerifyEligible(&dv, s)) {
                out = try own(g, try kvq.qkvAttnVerifyKernel(g.s, q_rope, &dv, geo.scale, mode));
            }
            if (out == null) {
                if (std.mem.eql(u8, mode, "causal")) {
                    out = try own(g, try attn256.fusedSdpa256Prefill(g.s, q_rope, dv.k, dv.v, geo.scale, 0));
                    if (out == null) out = try own(g, try sdpa.splitCausalSdpa(g.s, q_rope, dv.k, dv.v, geo.scale));
                    if (out == null) out = try sdpaRaw(g, q_rope, dv.k, dv.v, geo.scale, "causal", .{}, sdpa.sdpaForceFused(q_rope, dv.k));
                } else if (std.mem.eql(u8, mode, "array")) {
                    if (c.wide()) out = try own(g, try attn256.fusedSdpa256Prefill(g.s, q_rope, dv.k, dv.v, geo.scale, @intCast(window)));
                    if (out == null) {
                        if (mask.ctx == null) mask = try sliding.bandMask(g, s, sv.kv_len, @intCast(window));
                        out = try sdpaRaw(g, q_rope, dv.k, dv.v, geo.scale, "array", mask, false);
                    }
                } else {
                    out = try sdpaRaw(g, q_rope, dv.k, dv.v, geo.scale, "", .{}, false);
                }
            }
            const flat = try g.mergeHeads(out.?);
            const g_pre: Array = if (opts.gate != .none and fused_rows and w.qkv.widths[3] != 0) try w.qkv.part(g, all, 3) else .{};
            return w.o.forward(g, try outGate(w, g, x, flat, g_pre));
        }

        /// N one-row slots: projections at width N, each slot roped at its own position, keys appended per slot
        /// (a sliding layer's cache keeping its window), then a per-slot or a padded stacked read.
        fn batchedForward(w: *const Weights, c: *Ctx, x: Array) !Array {
            const g = c.g;
            const geo = &w.geo;
            const n = c.slots;
            const s = c.rows;
            if (n > MAX_SLOTS) return error.TooManySlots;
            var off_buf: [MAX_SLOTS]i32 = undefined;
            for (c.positions, 0..) |p, i| off_buf[i] = @intCast(p);
            const offsets = try g.fromSlice(i32, off_buf[0..c.positions.len], &.{n});
            const src = shared(w);

            const q_r = try g.reshape(try w.q.forward(g, x), &.{ n, s, geo.heads, geo.head_dim });
            const q_n = if (w.q_norm.ctx != null) try g.rmsNorm(q_r, w.q_norm, c.shared.eps) else q_r;
            const q_rope = try ropeDyn(g, try g.transpose(q_n, &perm), geo, offsets);

            if (src == null) {
                const own_k = try w.k.forward(g, x);
                const own_v = if (opts.value_norm and w.k_eq_v) own_k else try w.v.forward(g, x);
                const kvs = try ownKv(w, c, n, own_k, own_v, DynRoper{ .geo = geo, .offsets = offsets });
                try kv.appendRows(c, kvs[0], kvs[1], windowOf(w));
            }

            var views: [MAX_SLOTS]abi.KvView = undefined;
            var dense: [MAX_SLOTS]DenseKVView = undefined;
            var packed_reads: [MAX_SLOTS]bool = undefined;
            var got: usize = 0;
            defer for (views[0..got]) |*v| kv.release(c, v);
            for (c.caches, 0..) |h, i| {
                views[i] = try kv.view(c, h, src orelse c.layer);
                got = i + 1;
                dense[i] = kv.dense(views[i]);
                packed_reads[i] = views[i].packed_reads;
            }
            const attn = if (batched.batchedAttnPerSlot(dense[0..got]))
                try attention.perSlot(g, q_rope, dense[0..got], packed_reads[0..got], s, geo.scale)
            else
                try attention.stacked(g, q_rope, dense[0..got], s, geo.scale);
            return w.o.forward(g, try outGate(w, g, x, try g.mergeHeads(attn), .{}));
        }

        /// The output gate over `flat` `[N, S, heads * head_dim]`: one exact dispatch where it serves, else
        /// sigmoid then a per-channel or per-head multiply. `g_pre` is the gate projection when the joined rows made it.
        fn outGate(w: *const Weights, parent: *Graph, x: Array, flat: Array, g_pre: Array) !Array {
            if (opts.gate == .none) return flat;
            var sc = parent.scope();
            defer sc.deinit();
            const g = &sc;
            const raw = if (g_pre.ctx != null) g_pre else try w.out_gate.forward(g, x);
            const heads: c_int = if (opts.gate == .head) w.geo.heads else 0;
            if (try gates.fusedSigmoidGateMul(g.s, flat, raw, heads)) |y| return parent.keep(try g.adopt(y));
            const sig = try g.sigmoid(raw);
            if (opts.gate == .channel) return parent.keep(try g.mul(flat, sig));
            const sh = ops.shape(flat);
            const a4 = try g.reshape(flat, &.{ sh[0], sh[1], heads, @divExact(sh[2], heads) });
            const g4 = try g.reshape(sig, &.{ sh[0], sh[1], heads, 1 });
            return parent.keep(try g.reshape(try g.mul(a4, g4), sh));
        }
    };
}

const Roper = struct {
    geo: *const Geo,
    offset: c_int,
    fn apply(r: Roper, g: *Graph, x: Array) !Array {
        return ropeAt(g, x, r.geo, r.offset);
    }
};

const DynRoper = struct {
    geo: *const Geo,
    offsets: Array,
    fn apply(r: DynRoper, g: *Graph, x: Array) !Array {
        return ropeDyn(g, x, r.geo, r.offsets);
    }
};

fn base(geo: *const Geo) mlx.mlx_optional_float {
    return .{ .value = geo.rope_base, .has_value = geo.freqs.ctx == null };
}

pub fn ropeAt(g: *Graph, x: Array, geo: *const Geo, offset: c_int) !Array {
    if (geo.nope) return x;
    return g.call(mlx.mlx_fast_rope, .{ x, geo.rope_dims, false, base(geo), geo.rope_scale, offset, geo.freqs });
}

pub fn ropeDyn(g: *Graph, x: Array, geo: *const Geo, offsets: Array) !Array {
    if (geo.nope) return x;
    return g.call(mlx.mlx_fast_rope_dynamic, .{ x, geo.rope_dims, false, base(geo), geo.rope_scale, offsets, geo.freqs });
}

pub fn sdpaRaw(g: *Graph, q: Array, k: Array, v: Array, scale: f32, mode: [*:0]const u8, mask: Array, force: bool) !Array {
    return sdpaSinks(g, q, k, v, scale, mode, mask, .{}, force);
}

pub fn sdpaSinks(g: *Graph, q: Array, k: Array, v: Array, scale: f32, mode: [*:0]const u8, mask: Array, sinks: Array, force: bool) !Array {
    return g.call(mlx.mlx_fast_scaled_dot_product_attention, .{ q, k, v, scale, mode, mask, sinks, force });
}

pub fn own(g: *Graph, x: ?Array) !?Array {
    return if (x) |a| try g.adopt(a) else null;
}
