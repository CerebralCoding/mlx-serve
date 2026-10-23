//! Attention mixers in the legacy forwards' exact op order: `Attention` is the dense decoder's (llama, qwen3;
//! batched rows too), `HybridAttention` the hybrid stacks' (LFM2, Nemotron-H), both causal over the slot's KV cache;
//! `BidirAttention` the encoder's (BERT). copied_from: engines/mlx/forward/dense.zig (`forwardStandardWith`'s attention, no sliding window,
//! KV sharing, value norm, output gate or proportional RoPE), engines/mlx/batched/dense.zig (`forwardBatchedDecode`'s
//! attention), sdk/blocks/attention/batched.zig, sdk/blocks/attention/hybrid_attn.zig.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const block = @import("../block.zig");
const kv = @import("../kv.zig");
const linear = @import("linear.zig");
const attn256 = @import("../../blocks/attention/attn256.zig");
const sdpa = @import("../../blocks/attention/sdpa.zig");
const kvq = @import("../../blocks/attention/kv_quant_attn.zig");
const batched = @import("../../blocks/attention/batched.zig");
const qk_norm_rope = @import("../../blocks/rope/qk_norm_rope.zig");
const qk_rope = @import("qk_rope.zig");
const DenseKVView = @import("../../blocks/attention/kv_view.zig").DenseKVView;
const FUSED_ROWS_MAX_M = @import("../../blocks/linear/fused_rows.zig").FUSED_ROWS_MAX_M;

const Array = ops.Array;
const Graph = ops.Graph;
const Ctx = block.Ctx;
const Linear = linear.Linear;
const Joined = linear.Joined;

const perm = [_]c_int{ 0, 2, 1, 3 };

/// Slots a batched forward takes (the engine's `MAX_BATCH_ROWS`).
pub const MAX_SLOTS = 32;

/// One head geometry and its rotary embedding.
pub const Geo = struct {
    heads: c_int,
    kv_heads: c_int,
    head_dim: c_int,
    scale: f32,
    rope_dims: c_int,
    rope_base: f32,
    rope_scale: f32,
    /// No rotary position on this layer (NoPE).
    nope: bool = false,
    /// A frequency table (llama3 / proportional RoPE) in place of `rope_base`; null = none.
    freqs: Array = .{},

    fn base(geo: *const Geo) mlx.mlx_optional_float {
        return .{ .value = geo.rope_base, .has_value = geo.freqs.ctx == null };
    }
};

fn rope(g: *Graph, x: Array, geo: *const Geo, offset: c_int) !Array {
    if (geo.nope) return x;
    return g.call(mlx.mlx_fast_rope, .{ x, geo.rope_dims, false, geo.base(), geo.rope_scale, offset, geo.freqs });
}

fn ropeDynamic(g: *Graph, x: Array, geo: *const Geo, offsets: Array) !Array {
    if (geo.nope) return x;
    return g.call(mlx.mlx_fast_rope_dynamic, .{ x, geo.rope_dims, false, geo.base(), geo.rope_scale, offsets, geo.freqs });
}

fn sdpaRaw(g: *Graph, q: Array, k: Array, v: Array, scale: f32, mode: [*:0]const u8, mask: Array, force: bool) !Array {
    return g.call(mlx.mlx_fast_scaled_dot_product_attention, .{ q, k, v, scale, mode, mask, Array{}, force });
}

fn own(g: *Graph, x: ?Array) !?Array {
    return if (x) |a| try g.adopt(a) else null;
}

/// The dense decoder's attention.
pub const Attention = struct {
    pub const kind: block.Kind = .mixer;
    pub const supports_batched_rows = true;
    pub const Weights = struct {
        q: Linear,
        k: Linear,
        v: Linear,
        o: Linear,
        /// Per-head RMS norms on q and k (qwen3); null = none.
        q_norm: Array = .{},
        k_norm: Array = .{},
        /// q|k|v joined for one-row steps.
        qkv: Joined = .{},
        geo: Geo,
        /// The model's angle rows for the fused hd-256 decode step; null = built per call.
        angles: ?*qk_rope.AngleCache = null,
    };

    pub fn forward(w: *const Weights, c: *Ctx, x: Array) !Array {
        if (c.mode == .batched_rows) return batchedForward(w, c, x);
        const g = c.g;
        const geo = &w.geo;
        const b: c_int = 1;
        const s = c.rows;
        const offset: c_int = @intCast(c.offset());

        const fused_rows = w.qkv.built() and b * s <= FUSED_ROWS_MAX_M;
        const all: Array = if (fused_rows) try w.qkv.lin.forward(g, x) else .{};
        const q = if (fused_rows) try w.qkv.part(g, all, 0) else try w.q.forward(g, x);
        const q_r = try g.reshape(q, &.{ b, s, geo.heads, geo.head_dim });
        const own_k = if (fused_rows) try w.qkv.part(g, all, 1) else try w.k.forward(g, x);
        const own_v = if (fused_rows) try w.qkv.part(g, all, 2) else try w.v.forward(g, x);
        const kv_shape = [_]c_int{ b, s, geo.kv_heads, geo.head_dim };
        const k_r = try g.reshape(own_k, &kv_shape);
        const v_r = try g.reshape(own_v, &kv_shape);
        const v_t = try g.transpose(v_r, &perm);

        // A decode step at hd 256: one kernel norms, transposes and ropes q and k.
        var fused: ?[3]Array = null;
        if (!geo.nope and s == 1 and geo.head_dim == 256 and (w.q_norm.ctx != null) == (w.k_norm.ctx != null) and qk_norm_rope.qkNormRopeFusedEnabled()) blk: {
            const ang = qk_rope.angles(g, w.angles, 1, geo.rope_dims, geo.base(), geo.freqs, offset, 1, 1.0, geo.rope_scale) catch break :blk;
            fused = try qk_rope.fused256(g, q_r, k_r, .{}, w.q_norm, w.k_norm, .{}, ang, c.shared.eps_arr, geo.heads, geo.kv_heads, 1, geo.rope_dims);
        }
        const q_rope = if (fused) |f| f[0] else blk: {
            const q_n = if (w.q_norm.ctx != null) try g.rmsNorm(q_r, w.q_norm, c.shared.eps) else q_r;
            break :blk try rope(g, try g.transpose(q_n, &perm), geo, offset);
        };
        const k_rope = if (fused) |f| f[1] else blk: {
            const k_n = if (w.k_norm.ctx != null) try g.rmsNorm(k_r, w.k_norm, c.shared.eps) else k_r;
            break :blk try rope(g, try g.transpose(k_n, &perm), geo, offset);
        };

        var view = try kv.append(c, k_rope, v_t, 0);
        defer kv.release(c, &view);
        const dv = kv.dense(view);
        const k_all = dv.k;
        const v_all = dv.v;
        const none = Array{};
        const mode: []const u8 = if (c.wide()) "causal" else "";

        var out: ?Array = null;
        if (view.packed_reads and kvq.kvAttnFusedEligible(&dv, s)) {
            out = try own(g, try kvq.qkvAttnDecodeKernel(g.s, q_rope, &dv, geo.scale, mode, none));
            if (out != null) kvq.logKvAttnFusedEngaged(&dv, q_rope, s);
        } else if (view.packed_reads and kvq.kvAttnVerifyEligible(&dv, s)) {
            out = try own(g, try kvq.qkvAttnVerifyKernel(g.s, q_rope, &dv, geo.scale, mode));
        }
        if (out == null) {
            if (c.wide()) {
                out = try own(g, try attn256.fusedSdpa256Prefill(g.s, q_rope, k_all, v_all, geo.scale, 0));
                if (out == null) out = try own(g, try sdpa.splitCausalSdpa(g.s, q_rope, k_all, v_all, geo.scale));
                if (out == null) out = try sdpaRaw(g, q_rope, k_all, v_all, geo.scale, "causal", none, sdpa.sdpaForceFused(q_rope, k_all));
            } else {
                out = try sdpaRaw(g, q_rope, k_all, v_all, geo.scale, "", none, false);
            }
        }
        return w.o.forward(g, try g.mergeHeads(out.?));
    }

    /// N slots of `rows` tokens: projections at width N, each slot roped at its own position, keys appended per
    /// slot, then a per-slot or a padded stacked read.
    fn batchedForward(w: *const Weights, c: *Ctx, x: Array) !Array {
        const g = c.g;
        const geo = &w.geo;
        const n = c.slots;
        const s = c.rows;
        if (n > MAX_SLOTS) return error.TooManySlots;
        var off_buf: [MAX_SLOTS]i32 = undefined;
        for (c.positions, 0..) |p, i| off_buf[i] = @intCast(p);
        const offsets = try g.fromSlice(i32, off_buf[0..c.positions.len], &.{n});

        const q = try w.q.forward(g, x);
        const q_r = try g.reshape(q, &.{ n, s, geo.heads, geo.head_dim });
        const q_n = if (w.q_norm.ctx != null) try g.rmsNorm(q_r, w.q_norm, c.shared.eps) else q_r;
        const q_rope = try ropeDynamic(g, try g.transpose(q_n, &perm), geo, offsets);

        const own_k = try w.k.forward(g, x);
        const own_v = try w.v.forward(g, x);
        const kv_shape = [_]c_int{ n, s, geo.kv_heads, geo.head_dim };
        const k_r = try g.reshape(own_k, &kv_shape);
        const v_r = try g.reshape(own_v, &kv_shape);
        const k_n = if (w.k_norm.ctx != null) try g.rmsNorm(k_r, w.k_norm, c.shared.eps) else k_r;
        const k_t = try g.transpose(k_n, &perm);
        const v_t = try g.transpose(v_r, &perm);
        const k_rope = try ropeDynamic(g, k_t, geo, offsets);
        try kv.appendRows(c, k_rope, v_t, 0);

        var views: [MAX_SLOTS]@import("../../../abi/plugin.zig").KvView = undefined;
        var dense: [MAX_SLOTS]DenseKVView = undefined;
        var packed_reads: [MAX_SLOTS]bool = undefined;
        var got: usize = 0;
        defer for (views[0..got]) |*v| kv.release(c, v);
        for (c.caches, 0..) |h, i| {
            views[i] = try kv.view(c, h, c.layer);
            got = i + 1;
            dense[i] = kv.dense(views[i]);
            packed_reads[i] = views[i].packed_reads;
        }
        const attn = if (batched.batchedAttnPerSlot(dense[0..got]))
            try perSlot(g, q_rope, dense[0..got], packed_reads[0..got], s, geo.scale)
        else
            try stacked(g, q_rope, dense[0..got], s, geo.scale);
        return w.o.forward(g, try g.mergeHeads(attn));
    }
};

/// Each slot's query rows attend its own view: no pad, no stack, no mask (`perSlotBatchedAttn`).
pub fn perSlot(parent: *Graph, q: Array, views: []const DenseKVView, packed_reads: []const bool, seq: c_int, scale: f32) !Array {
    var t = parent.scope();
    defer t.deinit();
    const g = &t;
    var outs: [MAX_SLOTS]Array = undefined;
    const none = Array{};
    for (views, 0..) |*dv, i| {
        const q_i = try g.adopt(try @import("../../../mlx/array.zig").axisView(g.s, q, 0, i));
        const mode: [:0]const u8 = if (seq > 1) "causal" else "";
        var out: ?Array = null;
        if (packed_reads[i] and kvq.kvAttnFusedEligible(dv, seq)) {
            out = try own(g, try kvq.qkvAttnDecodeKernel(g.s, q_i, dv, scale, mode, none));
            if (out != null) kvq.logKvAttnFusedEngaged(dv, q_i, seq);
        } else if (packed_reads[i] and kvq.kvAttnVerifyEligible(dv, seq)) {
            out = try own(g, try kvq.qkvAttnVerifyKernel(g.s, q_i, dv, scale, mode));
        }
        if (out == null and seq > 1) {
            out = try own(g, try attn256.fusedSdpa256Prefill(g.s, q_i, dv.k, dv.v, scale, 0));
            if (out == null) out = try own(g, try sdpa.splitCausalSdpa(g.s, q_i, dv.k, dv.v, scale));
        }
        if (out == null) out = try sdpaRaw(g, q_i, dv.k, dv.v, scale, mode.ptr, none, seq > 1 and sdpa.sdpaForceFused(q_i, dv.k));
        outs[i] = out.?;
    }
    return parent.keep(try g.concat(outs[0..views.len], 0));
}

/// Every slot's view padded to the longest and stacked, one array-masked read (`stackedBatchedAttn`).
pub fn stacked(parent: *Graph, q: Array, views: []const DenseKVView, seq: c_int, scale: f32) !Array {
    var t = parent.scope();
    defer t.deinit();
    const g = &t;
    var lens: [MAX_SLOTS]i32 = undefined;
    var kv_max: c_int = 0;
    for (views, 0..) |dv, i| {
        lens[i] = ops.shape(dv.k)[2];
        kv_max = @max(kv_max, lens[i]);
    }
    const k = try padStack(g, views, true, kv_max);
    const v = try padStack(g, views, false, kv_max);
    const mask = try decodeMask(g, lens[0..views.len], kv_max, seq, ops.dtype(q));
    return parent.keep(try sdpaRaw(g, q, k, v, scale, "array", mask, false));
}

fn padStack(parent: *Graph, views: []const DenseKVView, keys: bool, kv_max: c_int) !Array {
    var t = parent.scope();
    defer t.deinit();
    const g = &t;
    const zero = try g.scalar(0, .bfloat16);
    var padded: [MAX_SLOTS]Array = undefined;
    for (views, 0..) |dv, i| {
        const x = if (keys) dv.k else dv.v;
        const hi = [_]c_int{kv_max - ops.shape(x)[2]};
        padded[i] = try g.call(mlx.mlx_pad, .{ x, &[_]c_int{2}, @as(usize, 1), &[_]c_int{0}, @as(usize, 1), &hi, @as(usize, 1), zero, "constant" });
    }
    return parent.keep(try g.concat(padded[0..views.len], 0));
}

/// Additive `[N, 1, S, kv_max]` mask: row j of slot n sees positions below `len_n - S + 1 + j`.
fn decodeMask(parent: *Graph, lens: []const i32, kv_max: c_int, seq: c_int, dt: mlx.mlx_dtype) !Array {
    var t = parent.scope();
    defer t.deinit();
    const g = &t;
    const n: c_int = @intCast(lens.len);
    const pos = try g.reshape(try g.arange(0, @floatFromInt(kv_max), 1, .int32), &.{ 1, 1, 1, kv_max });
    var lim: [MAX_SLOTS * 16]i32 = undefined;
    const rows: usize = @intCast(seq);
    if (lens.len * rows > lim.len) return error.BatchedMaskTooWide;
    for (lens, 0..) |l, i| for (0..rows) |j| {
        lim[i * rows + j] = l - seq + 1 + @as(i32, @intCast(j));
    };
    const limits = try g.fromSlice(i32, lim[0 .. lens.len * rows], &.{ n, 1, seq, 1 });
    const valid = try g.less(pos, limits);
    return parent.keep(try g.where(valid, try g.scalar(0, dt), try g.scalar(-std.math.inf(f32), dt)));
}

/// The hybrid stacks' attention (LFM2 with q/k norms, Nemotron-H NoPE): unit-free causal reads, the cache keeping
/// `keep` rows of view.
pub const HybridAttention = struct {
    pub const kind: block.Kind = .mixer;
    pub const supports_batched_rows = false;
    pub const Weights = struct {
        q: Linear,
        k: Linear,
        v: Linear,
        o: Linear,
        q_norm: Array = .{},
        k_norm: Array = .{},
        qkv: Joined = .{},
        geo: Geo,
        /// The view's tail span on every append (the legacy passes `max_position_embeddings`).
        keep: u32 = 0,
    };

    pub fn forward(w: *const Weights, c: *Ctx, x: Array) !Array {
        if (c.mode == .batched_rows) return error.BatchedRowsUnsupported;
        const g = c.g;
        const geo = &w.geo;
        const b: c_int = 1;
        const s = c.rows;
        const offset: c_int = @intCast(c.offset());
        const fused_rows = w.qkv.built() and b * s <= FUSED_ROWS_MAX_M;
        const all: Array = if (fused_rows) try w.qkv.lin.forward(g, x) else .{};
        const q_raw = if (fused_rows) try w.qkv.part(g, all, 0) else try w.q.forward(g, x);
        const k_raw = if (fused_rows) try w.qkv.part(g, all, 1) else try w.k.forward(g, x);
        const v_raw = if (fused_rows) try w.qkv.part(g, all, 2) else try w.v.forward(g, x);
        var q = try g.reshape(q_raw, &.{ b, s, geo.heads, geo.head_dim });
        var k = try g.reshape(k_raw, &.{ b, s, geo.kv_heads, geo.head_dim });
        const v = try g.reshape(v_raw, &.{ b, s, geo.kv_heads, geo.head_dim });
        if (w.q_norm.ctx != null) {
            q = try g.rmsNorm(q, w.q_norm, c.shared.eps);
            k = try g.rmsNorm(k, w.k_norm, c.shared.eps);
        }
        const q_t = try rope(g, try g.transpose(q, &perm), geo, offset);
        const k_t = try rope(g, try g.transpose(k, &perm), geo, offset);
        const v_t = try g.transpose(v, &perm);

        var view = try kv.append(c, k_t, v_t, w.keep);
        defer kv.release(c, &view);
        const none = Array{};
        const out = if (c.wide())
            (try own(g, try attn256.fusedSdpa256Prefill(g.s, q_t, mxa(view.k), mxa(view.v), geo.scale, 0))) orelse
                try sdpaRaw(g, q_t, mxa(view.k), mxa(view.v), geo.scale, "causal", none, sdpa.sdpaForceFused(q_t, mxa(view.k)))
        else
            try sdpaRaw(g, q_t, mxa(view.k), mxa(view.v), geo.scale, "", none, false);
        return w.o.forward(g, try g.mergeHeads(out));
    }
};

/// Bidirectional attention with additive projection biases and no cache (BERT); `Ctx.mask` masks padded keys.
/// copied_from: plugins/bert/forward.zig (`forwardBertWith`'s attention).
pub const BidirAttention = struct {
    pub const kind: block.Kind = .mixer;
    pub const supports_batched_rows = false;
    pub const Weights = struct {
        q: Linear,
        k: Linear,
        v: Linear,
        o: Linear,
        heads: c_int,
        head_dim: c_int,
    };

    pub fn forward(w: *const Weights, c: *Ctx, x: Array) !Array {
        const g = c.g;
        const q = try g.splitHeads(try w.q.forward(g, x), w.heads, w.head_dim);
        const k = try g.splitHeads(try w.k.forward(g, x), w.heads, w.head_dim);
        const v = try g.splitHeads(try w.v.forward(g, x), w.heads, w.head_dim);
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(w.head_dim)));
        const mode: [*:0]const u8 = if (c.mask.ctx != null) "array" else "";
        return w.o.forward(g, try g.mergeHeads(try sdpaRaw(g, q, k, v, scale, mode, c.mask, false)));
    }
};

fn mxa(a: @import("../../../abi/plugin.zig").Array) Array {
    return .{ .ctx = a.ctx };
}
