//! The qwen3_5 family's full-attention layer: q|gate from one projection, per-head q/k RMS norms, partial RoPE
//! (YaRN-scaled where the pack asks, M-RoPE rows on an image chunk), causal attention over the slot's KV cache and a
//! sigmoid output gate. Decode rows take the fused norm+RoPE kernels (hd 128 and 256), prompt chunks the hd-256
//! prefill kernel; batched rows rope each slot at its own position and read per slot or padded.
//! copied_from: sdk/blocks/attention/gated_full.zig (`gatedFullAttnProjected` without the qwen4 QSA, spec-tree
//! and grouped-verify arms), sdk/blocks/attention/proj.zig (`attnProj`), sdk/blocks/rope/mrope.zig
//! (`applyMrope`), sdk/blocks/rope/yarn.zig (`yarnScaleQK`).

const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const log = @import("../../util/log.zig");
const abi = @import("../../../abi/plugin.zig");
const block = @import("../block.zig");
const kv = @import("../kv.zig");
const linear = @import("linear.zig");
const row_exact = @import("row_exact.zig");
const slot_reads = @import("slot_reads.zig");
const qk_rope = @import("qk_rope.zig");
const attn256 = @import("../../blocks/attention/attn256.zig");
const sdpa = @import("../../blocks/attention/sdpa.zig");
const kvq = @import("../../blocks/attention/kv_quant_attn.zig");
const batched = @import("../../blocks/attention/batched.zig");
const gates = @import("../../blocks/attention/gates.zig");
const dq = @import("../../blocks/attention/decode_attn_quant.zig");
const qk_norm_rope = @import("../../blocks/rope/qk_norm_rope.zig");
const residual_norm = @import("../../blocks/norms/residual_norm.zig");
const decode_fused = @import("../../blocks/gdn/decode_fused.zig");
const DenseKVView = @import("../../blocks/attention/kv_view.zig").DenseKVView;

const Array = ops.Array;
const Graph = ops.Graph;
const Ctx = block.Ctx;
const Linear = linear.Linear;
const Joined = linear.Joined;

const perm = [_]c_int{ 0, 2, 1, 3 };

/// A projection with its decode-only requantized side copy (`--decode-attn-quant`), read at one-row solo steps.
pub const Proj = struct {
    lin: Linear,
    /// Null `w` = none.
    dq: Linear = .{},

    pub fn forward(p: *const Proj, g: *Graph, x: Array, decode: bool) !Array {
        return p.onLane(.off, g, x, decode);
    }

    /// `forward`, the projection on the row-exact `lane` (the side copy keeps its own kernel, as the engine's).
    pub fn onLane(p: *const Proj, lane: row_exact.Lane, g: *Graph, x: Array, decode: bool) !Array {
        if (decode and p.dq.w.ctx != null and (p.lin.s.ctx == null or dq.attnDqRequantEnabled()) and dq.decodeAttnQuantEnabled())
            return p.dq.forward(g, x);
        return row_exact.linear(lane, g, &p.lin, x);
    }
};

/// The layer geometry and its rotary embedding.
pub const Geo = struct {
    heads: c_int,
    kv_heads: c_int,
    head_dim: c_int,
    scale: f32,
    rope_dims: c_int,
    rope_base: f32,
    /// YaRN: the scaled spectrum (in place of `rope_base`), the `[head_dim]` mscale table in the activation dtype and
    /// its attention factor; null `yarn_freqs` = plain RoPE.
    yarn_freqs: Array = .{ .ctx = null },
    yarn_mscale: Array = .{ .ctx = null },
    yarn_factor: f32 = 1.0,
    /// The q projection carries a per-head sigmoid output gate beside each head (`attn_output_gate`).
    output_gate: bool = true,

    fn yarn(geo: *const Geo) bool {
        return geo.yarn_freqs.ctx != null;
    }
    fn base(geo: *const Geo) mlx.mlx_optional_float {
        return .{ .value = geo.rope_base, .has_value = !geo.yarn() };
    }
};

pub const GatedAttention = struct {
    pub const kind: block.Kind = .mixer;
    pub const supports_batched_rows = true;
    pub const Weights = struct {
        q: Proj,
        k: Proj,
        v: Proj,
        o: Proj,
        q_norm: Array,
        k_norm: Array,
        /// q|k|v joined for decode widths.
        qkv: Joined = .{},
        geo: Geo,
        /// The model's angle rows (one per forward); null = built per call.
        angles: ?*qk_rope.AngleCache = null,
    };

    pub fn forward(w: *const Weights, c: *Ctx, x: Array) !Array {
        return mix(w, c, x);
    }
};

/// Splits `[.., 2 * D]` into its two halves along the last axis (one op, both owned by `g`).
fn split2(g: *Graph, x: Array) ![2]Array {
    var vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_split(&vec, x, 2, -1, g.s));
    if (mlx.mlx_vector_array_size(vec) != 2) return error.UnexpectedSplitCount;
    var out: [2]Array = undefined;
    for (&out, 0..) |*o, i| {
        var a = mlx.mlx_array_new();
        mlx.check(mlx.mlx_vector_array_get(&a, vec, i)) catch |e| {
            _ = mlx.mlx_array_free(a);
            return e;
        };
        o.* = try g.adopt(a);
    }
    return out;
}

fn own(g: *Graph, x: ?Array) !?Array {
    return if (x) |a| try g.adopt(a) else null;
}

fn mix(w: *const GatedAttention.Weights, c: *Ctx, x: Array) !Array {
    const g = c.g;
    const geo = &w.geo;
    const is_batched = c.mode == .batched_rows;
    const b: c_int = if (is_batched) c.slots else 1;
    const s = c.rows;
    const h = geo.heads;
    const kvh = geo.kv_heads;
    const hd = geo.head_dim;
    const rd = geo.rope_dims;
    const decode_shape = b == 1 and s == 1;

    const joined = w.qkv.built() and row_exact.groupServes(c.row_exact, b * s);
    const all: Array = if (joined) try row_exact.linear(c.row_exact, g, &w.qkv.lin, x) else .{};
    if (joined and !decode_fused.attn_qkv_joined_engaged) {
        decode_fused.attn_qkv_joined_engaged = true;
        log.info("[attn] joined q|k|v engaged: rows={d}\n", .{b * s});
    }
    const q_proj = if (joined) try w.qkv.part(g, all, 0) else try w.q.onLane(c.row_exact, g, x, decode_shape);
    var queries: Array = undefined;
    var gate: Array = .{ .ctx = null };
    if (geo.output_gate) {
        // The gate stays 4-D: flattening the split view would copy across the q/gate interleave.
        const halves = try split2(g, try g.reshape(q_proj, &.{ b, s, h, hd * 2 }));
        queries = halves[0];
        gate = halves[1];
    } else {
        queries = try g.reshape(q_proj, &.{ b, s, h, hd });
    }
    const k_proj = if (joined) try w.qkv.part(g, all, 1) else try w.k.onLane(c.row_exact, g, x, decode_shape);
    const v_proj = if (joined) try w.qkv.part(g, all, 2) else try w.v.onLane(c.row_exact, g, x, decode_shape);
    const k_r = try g.reshape(k_proj, &.{ b, s, kvh, hd });
    const v_r = try g.reshape(v_proj, &.{ b, s, kvh, hd });
    const v_t = try g.transpose(v_r, &perm);

    const family: usize = if (geo.yarn()) 1 else 0;
    const mscale: f32 = if (geo.yarn()) geo.yarn_factor else 1.0;
    const off: c_int = @intCast(c.offset());
    var q_rope: ?Array = null;
    var k_rope: Array = undefined;
    const fusable = b == 1 and c.mrope == null and w.q_norm.ctx != null and w.k_norm.ctx != null and qk_norm_rope.qkNormRopeFusedEnabled();
    if (fusable and s == 1 and hd == 128) blk: {
        const ang = qk_rope.angles(c, w.angles, family, rd, geo.base(), geo.yarn_freqs, off, 1, 1.0, 1.0) catch break :blk;
        const ms: ?Array = if (geo.yarn()) geo.yarn_mscale else null;
        const pair = (qk_norm_rope.fusedQkNormRope(g.s, queries, k_r, w.q_norm, w.k_norm, ang, ms, c.shared.eps_arr, h, kvh, rd) catch null) orelse break :blk;
        q_rope = g.adopt(pair[0]) catch |e| {
            _ = mlx.mlx_array_free(pair[1]);
            return e;
        };
        k_rope = try g.adopt(pair[1]);
    }
    // The hd-256 kernel serves decode and verify widths; YaRN's factor rides the angle rows.
    if (q_rope == null and fusable and hd == 256 and s <= 32) blk: {
        const ang = qk_rope.angles(c, w.angles, family, rd, geo.base(), geo.yarn_freqs, off, s, mscale, 1.0) catch break :blk;
        const none = Array{ .ctx = null };
        const r = (try qk_rope.fused256(g, queries, k_r, none, w.q_norm, w.k_norm, none, ang, c.shared.eps_arr, h, kvh, s, rd)) orelse break :blk;
        q_rope = r[0];
        k_rope = r[1];
    }
    if (q_rope == null) {
        const q_t = try g.transpose(try g.rmsNorm(queries, w.q_norm, c.shared.eps), &perm);
        const k_t = try g.transpose(try g.rmsNorm(k_r, w.k_norm, c.shared.eps), &perm);
        if (c.mrope) |m| {
            q_rope = try applyMrope(g, q_t, m[0], m[1], rd);
            k_rope = try applyMrope(g, k_t, m[0], m[1], rd);
        } else if (is_batched) {
            if (c.positions.len > slot_reads.MAX_SLOTS) return error.TooManySlots;
            var off_buf: [slot_reads.MAX_SLOTS]i32 = undefined;
            for (c.positions, 0..) |p, i| off_buf[i] = @intCast(p);
            const offsets = try g.fromSlice(i32, off_buf[0..c.positions.len], &.{b});
            q_rope = try g.call(mlx.mlx_fast_rope_dynamic, .{ q_t, rd, false, geo.base(), 1.0, offsets, geo.yarn_freqs });
            k_rope = try g.call(mlx.mlx_fast_rope_dynamic, .{ k_t, rd, false, geo.base(), 1.0, offsets, geo.yarn_freqs });
            if (geo.yarn()) {
                q_rope = try g.mul(q_rope.?, geo.yarn_mscale);
                k_rope = try g.mul(k_rope, geo.yarn_mscale);
            }
        } else {
            q_rope = try g.call(mlx.mlx_fast_rope, .{ q_t, rd, false, geo.base(), 1.0, off, geo.yarn_freqs });
            k_rope = try g.call(mlx.mlx_fast_rope, .{ k_t, rd, false, geo.base(), 1.0, off, geo.yarn_freqs });
            if (geo.yarn()) {
                q_rope = try g.mul(q_rope.?, geo.yarn_mscale);
                k_rope = try g.mul(k_rope, geo.yarn_mscale);
            }
        }
    }
    const q = q_rope.?;

    const out = if (is_batched) try batchedRead(c, q, k_rope, v_t, s, geo.scale) else try soloRead(c, q, k_rope, v_t, s, geo.scale);
    return tail(w, c.row_exact, g, out, gate, b, s, decode_shape);
}

fn soloRead(c: *Ctx, q: Array, k: Array, v: Array, s: c_int, scale: f32) !Array {
    const g = c.g;
    var view = try kv.append(c, k, v, 0);
    defer kv.release(c, &view);
    const dv = kv.dense(view);
    const mode: []const u8 = if (c.wide()) "causal" else "";
    const none = Array{ .ctx = null };
    var out: ?Array = null;
    if (view.packed_reads and kvq.kvAttnFusedEligible(&dv, s)) {
        out = try own(g, try kvq.qkvAttnDecodeKernel(g.s, q, &dv, scale, mode, none));
        if (out != null) kvq.logKvAttnFusedEngaged(&dv, q, s);
    } else if (view.packed_reads and kvq.kvAttnVerifyEligible(&dv, s)) {
        out = try own(g, try kvq.qkvAttnVerifyKernel(g.s, q, &dv, scale, mode));
    }
    if (out) |o| return o;
    if (try row_exact.attendFixed(c.row_exact, g, q, dv.k, dv.v, scale, s)) |o| return o;
    if (try row_exact.attend(c.row_exact, g, q, dv.k, dv.v, scale, s)) |o| return o;
    if (c.wide()) {
        if (try own(g, try attn256.fusedSdpa256Prefill(g.s, q, dv.k, dv.v, scale, 0))) |o| return o;
        if (try own(g, try sdpa.splitCausalSdpa(g.s, q, dv.k, dv.v, scale))) |o| return o;
        return g.call(mlx.mlx_fast_scaled_dot_product_attention, .{ q, dv.k, dv.v, scale, "causal", none, none, sdpa.sdpaForceFused(q, dv.k) });
    }
    return g.call(mlx.mlx_fast_scaled_dot_product_attention, .{ q, dv.k, dv.v, scale, "", none, none, false });
}

fn batchedRead(c: *Ctx, q: Array, k: Array, v: Array, s: c_int, scale: f32) !Array {
    try kv.appendRows(c, k, v, 0);
    var views: [slot_reads.MAX_SLOTS]abi.KvView = undefined;
    var dense: [slot_reads.MAX_SLOTS]DenseKVView = undefined;
    var packed_reads: [slot_reads.MAX_SLOTS]bool = undefined;
    var got: usize = 0;
    defer for (views[0..got]) |*vw| kv.release(c, vw);
    for (c.caches, 0..) |hdl, i| {
        views[i] = try kv.view(c, hdl, c.layer);
        got = i + 1;
        dense[i] = kv.dense(views[i]);
        packed_reads[i] = views[i].packed_reads;
    }
    if (batched.batchedAttnPerSlot(dense[0..got])) return slot_reads.perSlot(c.g, q, dense[0..got], packed_reads[0..got], s, scale);
    return slot_reads.stacked(c.g, q, dense[0..got], s, scale);
}

/// `[B, H, S, D]` back to `[B, S, H*D]` through the sigmoid output gate, then the out projection.
fn tail(w: *const GatedAttention.Weights, lane: row_exact.Lane, parent: *Graph, attn: Array, gate: Array, b: c_int, s: c_int, decode_shape: bool) !Array {
    var t = parent.scope();
    defer t.deinit();
    const g = &t;
    const flat = [_]c_int{ b, s, w.geo.heads * w.geo.head_dim };
    const attn_t = try g.transpose(attn, &perm);
    if (w.geo.output_gate and b * s <= residual_norm.add_norm_max_rows) {
        if (try own(g, try gates.stridedSigmoidGateMul(g.s, attn_t, gate))) |gated|
            return parent.keep(try w.o.onLane(lane, g, try g.reshape(gated, &flat), decode_shape));
    }
    if (w.geo.output_gate) return parent.keep(try w.o.onLane(lane, g, try g.reshape(try g.mul(attn_t, try g.sigmoid(gate)), &flat), decode_shape));
    return parent.keep(try w.o.onLane(lane, g, try g.reshape(attn_t, &flat), decode_shape));
}

/// Interleaved M-RoPE on `[B, H, S, D]` from per-token `cos`/`sin` rows over the first `rd` dims.
fn applyMrope(parent: *Graph, x: Array, cos: Array, sin: Array, rd: c_int) !Array {
    var t = parent.scope();
    defer t.deinit();
    const g = &t;
    const sh = ops.shape(x);
    const half = @divExact(rd, 2);
    const rot = try g.slice(x, &.{ 0, 0, 0, 0 }, &.{ sh[0], sh[1], sh[2], rd });
    const r2 = try g.slice(rot, &.{ 0, 0, 0, half }, &.{ sh[0], sh[1], sh[2], rd });
    const r1 = try g.slice(rot, &.{ 0, 0, 0, 0 }, &.{ sh[0], sh[1], sh[2], half });
    const rh = try g.concat(&.{ try g.neg(r2), r1 }, -1);
    const out_rot = try g.add(try g.mul(rot, cos), try g.mul(rh, sin));
    if (sh[3] == rd) return parent.keep(try g.astype(out_rot, ops.dtype(x)));
    const pass = try g.slice(x, &.{ 0, 0, 0, rd }, &.{ sh[0], sh[1], sh[2], sh[3] });
    return parent.keep(try g.concat(&.{ out_rot, pass }, -1));
}
