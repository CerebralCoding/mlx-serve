//! plugins/inkling/attention, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns inklingConvSlot, InklingAttnOut, inklingAttnWith, InklingSconvOut, inklingSconvCore, inklingRelBias, and 3 more.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ForwardCtx = @import("../../engines/mlx/forward_ctx.zig").ForwardCtx;
const testing = @import("../../engines/mlx/kv/cache.zig").testing;
const slidingViewFor = @import("../../engines/mlx/kv/sliding.zig").slidingViewFor;
const FullAttnWeights = @import("../../engines/mlx/layer_weights.zig").FullAttnWeights;
const SSMCacheEntry = @import("../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;
const materializedOwnedCopy = @import("../../mlx/array.zig").materializedOwnedCopy;
const jsonF64 = @import("../../sdk/testing/parity.zig").jsonF64;
const jsonToMlxArray = @import("../../sdk/testing/parity.zig").jsonToMlxArray;
const maxAbsDiffF32 = @import("../../sdk/testing/parity.zig").maxAbsDiffF32;
const inklingRouterChain = @import("moe.zig").inklingRouterChain;

// ── Inkling attention (no RoPE: rel-logits bias + k/v short conv) ──
// Reference: inkling_mlx/attention.py. Per-head q/k RMSNorm with scale
// 1/head_dim; hybrid sliding(512)/global; the additive mask carries BOTH
// the hidden-state-conditioned relative-position bias and the
// causal/sliding constraint; log-scaling on global layers past n_floor
// (exact no-op below — skipped entirely). k/v projections pass through
// depthwise causal short-convolutions whose kernel-1 f32 tails ride the
// per-slot ssm entry (slots 0 and 1 of the layer's concatenated
// conv_state; the layer tail owns slots 2/3 and the state reassembly).

/// Slice one conv slot [B, K-1, dim] out of the layer's concatenated
/// conv_state, or null when the entry is uninitialized (sequence start).
pub fn inklingConvSlot(self: *Transformer, ssm: *const SSMCacheEntry, chan_off: c_int, chan_dim: c_int) !?mlx.mlx_array {
    if (!ssm.initialized or ssm.conv_state.ctx == null) return null;
    const st_shape = mlx.getShape(ssm.conv_state);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    const start = [_]c_int{ 0, 0, chan_off };
    const stop = [_]c_int{ st_shape[0], st_shape[1], chan_off + chan_dim };
    const strides = [_]c_int{ 1, 1, 1 };
    try mlx.check(mlx.mlx_slice(&out, ssm.conv_state, &start, 3, &stop, 3, &strides, 3, self.s));
    return out;
}

pub const InklingAttnOut = struct { out: mlx.mlx_array, k_state: mlx.mlx_array, v_state: mlx.mlx_array };

pub fn inklingAttnWith(
    self: *Transformer,
    ctx: *ForwardCtx,
    x: mlx.mlx_array,
    fa: *const FullAttnWeights,
    ssm: *const SSMCacheEntry,
    layer: u32,
    offset: c_int,
    batch: c_int,
    seq_len: c_int,
    is_prefill: bool,
) !InklingAttnOut {
    const cache = ctx.cache;
    const cfg = &self.config;
    const h_count: c_int = @intCast(cfg.num_attention_heads);
    const kv_h: c_int = @intCast(cfg.num_key_value_heads);
    const hd: c_int = @intCast(cfg.head_dim);
    const kv_dim: c_int = kv_h * hd;
    const d_rel: c_int = @intCast(cfg.inkling_d_rel);
    // 1/head_dim via the shared 1/sqrt(query_pre_attn_scalar) convention.
    const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.query_pre_attn_scalar)));
    const is_global = cfg.isGlobalLayer(layer);
    const decode_hint = batch == 1 and !is_prefill;

    // Projections (no biases in any shipped config).
    const q_proj = try self.attnProj(x, fa.q_w, fa.q_s, fa.q_b, decode_hint, layer);
    defer _ = mlx.mlx_array_free(q_proj);
    const k_proj = try self.attnProj(x, fa.k_w, fa.k_s, fa.k_b, decode_hint, layer);
    defer _ = mlx.mlx_array_free(k_proj);
    const v_proj = try self.attnProj(x, fa.v_w, fa.v_s, fa.v_b, decode_hint, layer);
    defer _ = mlx.mlx_array_free(v_proj);
    const rel_out = try self.attnProj(x, fa.rel_w, fa.rel_s, fa.rel_b, decode_hint, layer);
    defer _ = mlx.mlx_array_free(rel_out);

    // k/v short convolutions (f32 compute, conv slots 0 and 1).
    const k_prev = try self.inklingConvSlot(ssm, 0, kv_dim);
    defer if (k_prev) |p| {
        _ = mlx.mlx_array_free(p);
    };
    const v_prev = try self.inklingConvSlot(ssm, kv_dim, kv_dim);
    defer if (v_prev) |p| {
        _ = mlx.mlx_array_free(p);
    };
    const ks = try inklingSconvCore(self.s, k_proj, fa.k_sconv_w, k_prev);
    defer _ = mlx.mlx_array_free(ks.out);
    errdefer _ = mlx.mlx_array_free(ks.new_state);
    const vs = try inklingSconvCore(self.s, v_proj, fa.v_sconv_w, v_prev);
    defer _ = mlx.mlx_array_free(vs.out);
    errdefer _ = mlx.mlx_array_free(vs.new_state);

    // Per-head RMSNorm on q/k, then [B,S,H,D] → [B,H,S,D].
    const q_shape = [_]c_int{ batch, seq_len, h_count, hd };
    const kv_shape = [_]c_int{ batch, seq_len, kv_h, hd };
    const perm = [_]c_int{ 0, 2, 1, 3 };
    var q_r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_r);
    try mlx.check(mlx.mlx_reshape(&q_r, q_proj, &q_shape, 4, self.s));
    const q_normed = try self.rmsNorm(q_r, fa.q_norm);
    defer _ = mlx.mlx_array_free(q_normed);
    var q_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_t);
    try mlx.check(mlx.mlx_transpose_axes(&q_t, q_normed, &perm, 4, self.s));

    var k_r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_r);
    try mlx.check(mlx.mlx_reshape(&k_r, ks.out, &kv_shape, 4, self.s));
    const k_normed = try self.rmsNorm(k_r, fa.k_norm);
    defer _ = mlx.mlx_array_free(k_normed);
    var k_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_t);
    try mlx.check(mlx.mlx_transpose_axes(&k_t, k_normed, &perm, 4, self.s));

    var v_r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v_r);
    try mlx.check(mlx.mlx_reshape(&v_r, vs.out, &kv_shape, 4, self.s));
    var v_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v_t);
    try mlx.check(mlx.mlx_transpose_axes(&v_t, v_r, &perm, 4, self.s));

    // KV cache: sliding layers read only the tail their queries reach.
    // Nothing else here changes — inklingRelBias and inklingCombineMask
    // already take `kv_len` off the view and `kv_start = total_kv - kv_len`,
    // so they are correct for any trim; they simply never got one. Inkling
    // is serial, so the width that matters is the prefill chunk.
    const total_kv: c_int = offset + seq_len;
    const sliding = slidingViewFor(cfg, total_kv, seq_len);
    const max_kv: u32 = if (is_global) 0 else sliding.span;
    var kv_view = try cache.update(layer, k_t, v_t, self.s, max_kv);
    defer kv_view.deinit();
    const kv_len: c_int = mlx.getShape(kv_view.k)[2];
    const kv_start: c_int = total_kv - kv_len;

    // Relative-logits bias + causal/sliding constraint as ONE additive mask.
    const rel_shape = [_]c_int{ batch, seq_len, h_count, d_rel };
    var rel_r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rel_r);
    try mlx.check(mlx.mlx_reshape(&rel_r, rel_out, &rel_shape, 4, self.s));
    const bias = try inklingRelBias(self.s, rel_r, fa.rel_proj, offset, kv_len, kv_start);
    defer _ = mlx.mlx_array_free(bias);
    const window: ?c_int = if (!is_global and cfg.has_sliding_window) @intCast(cfg.sliding_window) else null;
    var mask = try inklingCombineMask(self.s, bias, offset, kv_len, kv_start, window);
    defer _ = mlx.mlx_array_free(mask);

    // Log-scaling (global layers only): tau = 1 + alpha·ln(max((pos+1)/n_floor, 1))
    // multiplies q AND the mask per query position. Exact no-op while the
    // whole chunk sits at or below the floor — skip the ops entirely.
    var q_final = q_t;
    var q_scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_scaled);
    if (is_global and cfg.inkling_log_n_floor > 0 and total_kv > @as(c_int, @intCast(cfg.inkling_log_n_floor))) {
        const tau = try inklingLogScaleTau(self.s, offset, seq_len, cfg.inkling_log_n_floor, cfg.inkling_log_alpha);
        defer _ = mlx.mlx_array_free(tau);
        // q: [B,H,S,D] × tau [1,1,S,1] (f32; cast back to q dtype)
        var q32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q32);
        try mlx.check(mlx.mlx_astype(&q32, q_t, .float32, self.s));
        var qs32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(qs32);
        try mlx.check(mlx.mlx_multiply(&qs32, q32, tau, self.s));
        try mlx.check(mlx.mlx_astype(&q_scaled, qs32, mlx.mlx_array_dtype(q_t), self.s));
        q_final = q_scaled;
        var m_scaled = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&m_scaled, mask, tau, self.s));
        _ = mlx.mlx_array_free(mask);
        mask = m_scaled;
    }

    // SDPA with the additive mask, in the query's dtype (reference casts).
    var mask_cast = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask_cast);
    try mlx.check(mlx.mlx_astype(&mask_cast, mask, mlx.mlx_array_dtype(q_final), self.s));
    var attn_out = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn_out);
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_final, kv_view.k, kv_view.v, attn_scale, "", mask_cast, .{ .ctx = null }, false, self.s));

    // [B,H,S,D] → [B,S,H*D] → o_proj
    var attn_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn_t);
    try mlx.check(mlx.mlx_transpose_axes(&attn_t, attn_out, &perm, 4, self.s));
    const flat_shape = [_]c_int{ batch, seq_len, h_count * hd };
    var attn_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn_flat);
    try mlx.check(mlx.mlx_reshape(&attn_flat, attn_t, &flat_shape, 3, self.s));

    const out = try self.attnProj(attn_flat, fa.o_w, fa.o_s, fa.o_b, decode_hint, layer);
    return .{ .out = out, .k_state = ks.new_state, .v_state = vs.new_state };
}

// ── Inkling (inkling_mm_model) forward helpers ──
// Reference: the checkpoint's bundled inkling_mlx/ package (Apache-2.0,
// parity-validated vs transformers PR #47347). Dtype discipline mirrors the
// reference EXACTLY: the residual stream runs in the checkpoint's dtype (bf16
// on the shipped builds — the dense mlp.global_scale tensors are STORED bf16,
// so no promotion), while the short convs, the router chain and the rel-bias
// build compute in f32 and the shared-expert sum is f32-then-cast.

const InklingSconvOut = struct { out: mlx.mlx_array, new_state: mlx.mlx_array };

/// Depthwise causal short-convolution with residual add, computed in fp32
/// (`InklingShortConvolution`). `prev_state` = the last kernel-1 f32 inputs
/// (null ⇒ zeros — sequence start). Returns the output in x's dtype and the
/// new f32 state; caller owns both.
pub fn inklingSconvCore(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, prev_state: ?mlx.mlx_array) !InklingSconvOut {
    const in_dtype = mlx.mlx_array_dtype(x);
    const x_shape = mlx.getShape(x); // [B, S, C]
    const w_shape = mlx.getShape(w); // [C, K, 1]
    const kernel = w_shape[1];
    const channels = x_shape[2];

    var x32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x32);
    try mlx.check(mlx.mlx_astype(&x32, x, .float32, s));
    var w32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w32);
    try mlx.check(mlx.mlx_astype(&w32, w, .float32, s));

    var left = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(left);
    if (prev_state) |st| {
        try mlx.check(mlx.mlx_astype(&left, st, .float32, s));
    } else {
        const zshape = [_]c_int{ x_shape[0], kernel - 1, channels };
        try mlx.check(mlx.mlx_zeros(&left, &zshape, 3, .float32, s));
    }

    var conv_in = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(conv_in);
    {
        const arr = [_]mlx.mlx_array{ left, x32 };
        const vec = mlx.mlx_vector_array_new_data(&arr, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&conv_in, vec, 1, s));
    }

    // "valid" conv over [left, x] yields exactly S causal outputs.
    var conv_out = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(conv_out);
    try mlx.check(mlx.mlx_conv1d(&conv_out, conv_in, w32, 1, 0, 1, channels, s));

    var out32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out32);
    try mlx.check(mlx.mlx_add(&out32, conv_out, x32, s));

    var new_state = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(new_state);
    {
        const start = [_]c_int{ 0, x_shape[1], 0 }; // last kernel-1 rows of [K-1+S]
        const stop = [_]c_int{ x_shape[0], x_shape[1] + kernel - 1, channels };
        const strides = [_]c_int{ 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&new_state, conv_in, &start, 3, &stop, 3, &strides, 3, s));
    }

    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, out32, in_dtype, s));
    return .{ .out = out, .new_state = new_state };
}

/// Hidden-state-conditioned relative position bias (`InklingRelativeLogits`):
/// rel_states [B, Lq, H, d_rel] × proj [d_rel, extent] → per-distance bias,
/// gathered per (q, kv) pair as bias[b,h,i,j] = profile[clip(qpos_i - kvpos_j)]
/// and zeroed outside 0 <= distance < extent. `kv_start` = the absolute
/// position of the view's first key (non-zero when a sliding-layer decode
/// reads the last-window KV view). Returns [B, H, Lq, kv_len] f32.
fn inklingRelBias(s: mlx.mlx_stream, rel_states: mlx.mlx_array, proj: mlx.mlx_array, q_start: i64, kv_len: c_int, kv_start: c_int) !mlx.mlx_array {
    const rs_shape = mlx.getShape(rel_states); // [B, Lq, H, d_rel]
    const batch = rs_shape[0];
    const lq = rs_shape[1];
    const heads = rs_shape[2];
    const extent = mlx.getShape(proj)[1];

    var rs32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rs32);
    try mlx.check(mlx.mlx_astype(&rs32, rel_states, .float32, s));
    var proj32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(proj32);
    try mlx.check(mlx.mlx_astype(&proj32, proj, .float32, s));

    // [B, Lq, H, extent] → [B, H, Lq, extent]
    var rel_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rel_logits);
    try mlx.check(mlx.mlx_matmul(&rel_logits, rs32, proj32, s));
    var rel_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rel_t);
    try mlx.check(mlx.mlx_transpose_axes(&rel_t, rel_logits, &[_]c_int{ 0, 2, 1, 3 }, 4, s));

    // distance[i, j] = (q_start + i) - j, [Lq, kv_len] int32
    var q_pos = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_pos);
    try mlx.check(mlx.mlx_arange(&q_pos, @floatFromInt(q_start), @floatFromInt(q_start + lq), 1, .int32, s));
    var q_col = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_col);
    try mlx.check(mlx.mlx_reshape(&q_col, q_pos, &[_]c_int{ lq, 1 }, 2, s));
    var kv_pos = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kv_pos);
    try mlx.check(mlx.mlx_arange(&kv_pos, @floatFromInt(kv_start), @floatFromInt(kv_start + kv_len), 1, .int32, s));
    var distance = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(distance);
    try mlx.check(mlx.mlx_subtract(&distance, q_col, kv_pos, s));

    // gather = clip(distance, 0, extent-1), broadcast [B, H, Lq, kv_len]
    const zero_i = mlx.mlx_array_new_int(0);
    defer _ = mlx.mlx_array_free(zero_i);
    const max_i = mlx.mlx_array_new_int(extent - 1);
    defer _ = mlx.mlx_array_free(max_i);
    var lo = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lo);
    try mlx.check(mlx.mlx_maximum(&lo, distance, zero_i, s));
    var gather = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gather);
    try mlx.check(mlx.mlx_minimum(&gather, lo, max_i, s));
    var gather4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gather4);
    try mlx.check(mlx.mlx_reshape(&gather4, gather, &[_]c_int{ 1, 1, lq, kv_len }, 4, s));
    var gather_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gather_b);
    try mlx.check(mlx.mlx_broadcast_to(&gather_b, gather4, &[_]c_int{ batch, heads, lq, kv_len }, 4, s));

    var bias = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bias);
    try mlx.check(mlx.mlx_take_along_axis(&bias, rel_t, gather_b, 3, s));

    // valid = 0 <= distance < extent (per the reference, additive-zero outside)
    var ge0 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ge0);
    try mlx.check(mlx.mlx_greater_equal(&ge0, distance, zero_i, s));
    const ext_i = mlx.mlx_array_new_int(extent);
    defer _ = mlx.mlx_array_free(ext_i);
    var lt_ext = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lt_ext);
    try mlx.check(mlx.mlx_less(&lt_ext, distance, ext_i, s));
    var valid = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(valid);
    try mlx.check(mlx.mlx_logical_and(&valid, ge0, lt_ext, s));
    var valid4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(valid4);
    try mlx.check(mlx.mlx_reshape(&valid4, valid, &[_]c_int{ 1, 1, lq, kv_len }, 4, s));

    const zero_f = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(zero_f);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_where(&out, valid4, bias, zero_f, s));
    return out;
}

/// Fold the causal (or sliding-window) constraint into the rel bias as ONE
/// additive mask: where allowed, the bias; elsewhere -1e30 (the reference's
/// NEG_INF — built with where, never indicator × -inf). Returns f32
/// [.., Lq, kv_len] broadcastable against the bias shape.
fn inklingCombineMask(s: mlx.mlx_stream, bias: mlx.mlx_array, q_start: c_int, kv_len: c_int, kv_start: c_int, window: ?c_int) !mlx.mlx_array {
    const b_shape = mlx.getShape(bias); // [B, H, Lq, kv_len]
    const lq = b_shape[2];
    var q_pos = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_pos);
    try mlx.check(mlx.mlx_arange(&q_pos, @floatFromInt(q_start), @floatFromInt(q_start + lq), 1, .int32, s));
    var q_col = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_col);
    try mlx.check(mlx.mlx_reshape(&q_col, q_pos, &[_]c_int{ lq, 1 }, 2, s));
    var kv_pos = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kv_pos);
    try mlx.check(mlx.mlx_arange(&kv_pos, @floatFromInt(kv_start), @floatFromInt(kv_start + kv_len), 1, .int32, s));
    var distance = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(distance);
    try mlx.check(mlx.mlx_subtract(&distance, q_col, kv_pos, s));

    const zero_i = mlx.mlx_array_new_int(0);
    defer _ = mlx.mlx_array_free(zero_i);
    var allowed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(allowed);
    try mlx.check(mlx.mlx_greater_equal(&allowed, distance, zero_i, s));
    if (window) |w| {
        const w_i = mlx.mlx_array_new_int(w);
        defer _ = mlx.mlx_array_free(w_i);
        var in_win = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(in_win);
        try mlx.check(mlx.mlx_less(&in_win, distance, w_i, s));
        var both = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_logical_and(&both, allowed, in_win, s));
        _ = mlx.mlx_array_free(allowed);
        allowed = both;
    }
    var allowed4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(allowed4);
    try mlx.check(mlx.mlx_reshape(&allowed4, allowed, &[_]c_int{ 1, 1, lq, kv_len }, 4, s));

    const neg_inf = mlx.mlx_array_new_float(-1e30);
    defer _ = mlx.mlx_array_free(neg_inf);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_where(&out, allowed4, bias, neg_inf, s));
    return out;
}

/// Per-query log-scaling factor for global layers past the floor:
/// tau_i = 1 + alpha·ln(max((q_start + i + 1)/n_floor, 1)), f32 [1,1,Lq,1].
fn inklingLogScaleTau(s: mlx.mlx_stream, q_start: c_int, lq: c_int, n_floor: u32, alpha: f32) !mlx.mlx_array {
    var eff_n = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eff_n);
    try mlx.check(mlx.mlx_arange(&eff_n, @floatFromInt(q_start + 1), @floatFromInt(q_start + lq + 1), 1, .float32, s));
    const floor_f = mlx.mlx_array_new_float(@floatFromInt(n_floor));
    defer _ = mlx.mlx_array_free(floor_f);
    var ratio = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ratio);
    try mlx.check(mlx.mlx_divide(&ratio, eff_n, floor_f, s));
    const one_f = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one_f);
    var clamped = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(clamped);
    try mlx.check(mlx.mlx_maximum(&clamped, ratio, one_f, s));
    var logv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(logv);
    try mlx.check(mlx.mlx_log(&logv, clamped, s));
    const alpha_f = mlx.mlx_array_new_float(alpha);
    defer _ = mlx.mlx_array_free(alpha_f);
    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    try mlx.check(mlx.mlx_multiply(&scaled, logv, alpha_f, s));
    var tau_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tau_flat);
    try mlx.check(mlx.mlx_add(&tau_flat, scaled, one_f, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, tau_flat, &[_]c_int{ 1, 1, lq, 1 }, 4, s));
    return out;
}

// ── tests ──

fn loadInklingFixtures(allocator: std.mem.Allocator) !?std.json.Parsed(std.json.Value) {
    const path_z = std.c.getenv("INKLING_FIXTURES") orelse return null;
    const path = std.mem.span(path_z);
    if (path.len == 0) return null;
    const io = std.Io.Threaded.global_single_threaded.io();
    const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);
    var read_buf: [4096]u8 = undefined;
    var reader_state = file.reader(io, &read_buf);
    const data = try reader_state.interface.allocRemaining(allocator, .limited(1 << 26));
    defer allocator.free(data);
    return try std.json.parseFromSlice(std.json.Value, allocator, data, .{});
}

test "inkling sconv parity vs reference ShortConvolution (INKLING_FIXTURES)" {
    const allocator = testing.allocator;
    var parsed = (try loadInklingFixtures(allocator)) orelse return error.SkipZigTest;
    defer parsed.deinit();
    const sc = parsed.value.object.get("sconv").?.object;
    const s = mlx.gpuStream();

    const w = try jsonToMlxArray(allocator, sc.get("weight").?);
    defer _ = mlx.mlx_array_free(w);
    const x = try jsonToMlxArray(allocator, sc.get("x").?);
    defer _ = mlx.mlx_array_free(x);
    const y_ref = try jsonToMlxArray(allocator, sc.get("y_full").?);
    defer _ = mlx.mlx_array_free(y_ref);
    const state_ref = try jsonToMlxArray(allocator, sc.get("final_cache_state").?);
    defer _ = mlx.mlx_array_free(state_ref);

    // Full-sequence (no prior state).
    const full = try inklingSconvCore(s, x, w, null);
    defer _ = mlx.mlx_array_free(full.out);
    defer _ = mlx.mlx_array_free(full.new_state);
    try testing.expect(try maxAbsDiffF32(full.out, y_ref, s) < 1e-5);

    // Incremental: one position at a time through the carried state must
    // reproduce the full-sequence output AND the reference's final state.
    const x_shape = mlx.getShape(x);
    const seq: usize = @intCast(x_shape[1]);
    var state: ?mlx.mlx_array = null;
    defer if (state) |st| {
        _ = mlx.mlx_array_free(st);
    };
    var worst: f32 = 0;
    for (0..seq) |t| {
        var xt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xt);
        const start = [_]c_int{ 0, @intCast(t), 0 };
        const stop = [_]c_int{ x_shape[0], @intCast(t + 1), x_shape[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&xt, x, &start, 3, &stop, 3, &strides, 3, s));
        const step = try inklingSconvCore(s, xt, w, state);
        if (state) |st| _ = mlx.mlx_array_free(st);
        state = step.new_state;
        defer _ = mlx.mlx_array_free(step.out);
        var yt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(yt);
        try mlx.check(mlx.mlx_slice(&yt, y_ref, &start, 3, &stop, 3, &strides, 3, s));
        const e = try maxAbsDiffF32(step.out, yt, s);
        if (e > worst) worst = e;
    }
    try testing.expect(worst < 1e-5);
    try testing.expect(try maxAbsDiffF32(state.?, state_ref, s) < 1e-5);
    std.debug.print("[inkling-sconv] full+incremental parity ok (worst step err {e:.3})\n", .{worst});
}

test "inkling relative-logits bias parity vs reference (INKLING_FIXTURES)" {
    const allocator = testing.allocator;
    var parsed = (try loadInklingFixtures(allocator)) orelse return error.SkipZigTest;
    defer parsed.deinit();
    const rb = parsed.value.object.get("rel_bias").?.object;
    const s = mlx.gpuStream();

    const proj = try jsonToMlxArray(allocator, rb.get("proj").?);
    defer _ = mlx.mlx_array_free(proj);

    // Prefill: Lq == Lkv, start 0.
    {
        const pf = rb.get("prefill").?.object;
        const rel_states = try jsonToMlxArray(allocator, pf.get("rel_states").?);
        defer _ = mlx.mlx_array_free(rel_states);
        const bias_ref = try jsonToMlxArray(allocator, pf.get("bias").?);
        defer _ = mlx.mlx_array_free(bias_ref);
        const shape = mlx.getShape(rel_states);
        const bias = try inklingRelBias(s, rel_states, proj, 0, shape[1], 0);
        defer _ = mlx.mlx_array_free(bias);
        try testing.expect(try maxAbsDiffF32(bias, bias_ref, s) < 1e-5);
    }
    // Decode: Lq = 1 at a position past the extent (zero-beyond-extent region).
    {
        const dc = rb.get("decode").?.object;
        const rel_states = try jsonToMlxArray(allocator, dc.get("rel_states").?);
        defer _ = mlx.mlx_array_free(rel_states);
        const bias_ref = try jsonToMlxArray(allocator, dc.get("bias").?);
        defer _ = mlx.mlx_array_free(bias_ref);
        const kv_len: c_int = @intCast(dc.get("kv_len").?.integer);
        const q_pos: i64 = dc.get("q_pos").?.integer;
        const bias = try inklingRelBias(s, rel_states, proj, q_pos, kv_len, 0);
        defer _ = mlx.mlx_array_free(bias);
        try testing.expect(try maxAbsDiffF32(bias, bias_ref, s) < 1e-5);
    }
    std.debug.print("[inkling-relbias] prefill+decode parity ok\n", .{});
}

test "inkling router chain parity vs reference (INKLING_FIXTURES)" {
    const allocator = testing.allocator;
    var parsed = (try loadInklingFixtures(allocator)) orelse return error.SkipZigTest;
    defer parsed.deinit();
    const rt = parsed.value.object.get("router").?.object;
    const cfgj = parsed.value.object.get("config").?.object;
    const s = mlx.gpuStream();

    const w = try jsonToMlxArray(allocator, rt.get("weight").?); // [E+ns, H]
    defer _ = mlx.mlx_array_free(w);
    const bias = try jsonToMlxArray(allocator, rt.get("bias").?); // [E]
    defer _ = mlx.mlx_array_free(bias);
    const x = try jsonToMlxArray(allocator, rt.get("x").?); // [1, T, H]
    defer _ = mlx.mlx_array_free(x);
    const tw_ref = try jsonToMlxArray(allocator, rt.get("topk_weights").?);
    defer _ = mlx.mlx_array_free(tw_ref);
    const sg_ref = try jsonToMlxArray(allocator, rt.get("shared_gammas").?);
    defer _ = mlx.mlx_array_free(sg_ref);

    const top_k: c_int = @intCast(cfgj.get("num_experts_per_tok").?.integer);
    const n_shared: c_int = @intCast(cfgj.get("n_shared_experts").?.integer);
    const route_scale: f32 = @floatCast(jsonF64(cfgj.get("route_scale").?));
    const global_scale: f32 = @floatCast(jsonF64(rt.get("global_scale").?));

    // logits = flatten(x) @ w.T in f32 (the caller's job in the forward).
    const x_shape = mlx.getShape(x);
    var xf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xf);
    const flat_shape = [_]c_int{ x_shape[0] * x_shape[1], x_shape[2] };
    try mlx.check(mlx.mlx_reshape(&xf, x, &flat_shape, 2, s));
    var wt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wt);
    try mlx.check(mlx.mlx_transpose_axes(&wt, w, &[_]c_int{ 1, 0 }, 2, s));
    var logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(logits);
    try mlx.check(mlx.mlx_matmul(&logits, xf, wt, s));

    const r = try inklingRouterChain(s, logits, bias, top_k, n_shared, route_scale, global_scale);
    defer _ = mlx.mlx_array_free(r.topk_weights);
    defer _ = mlx.mlx_array_free(r.topk_idx);
    defer _ = mlx.mlx_array_free(r.shared_gammas);

    // Indices: order within the top-k is irrelevant downstream, but the
    // reference dumps argpartition order — compare as SETS per token, and
    // weights via a value-sorted comparison per token.
    const ti_ref = rt.get("topk_idx").?.array;
    const idx_c = try materializedOwnedCopy(s, r.topk_idx);
    defer _ = mlx.mlx_array_free(idx_c);
    try mlx.check(mlx.mlx_array_eval(idx_c));
    const ip = mlx.mlx_array_data_uint32(idx_c) orelse return error.NoData;
    const T: usize = @intCast(flat_shape[0]);
    const K: usize = @intCast(top_k);
    for (0..T) |t| {
        const row_ref = ti_ref.items[t].array;
        for (0..K) |k| {
            const want: u32 = @intCast(row_ref.items[k].integer);
            var found = false;
            for (0..K) |k2| {
                if (ip[t * K + k2] == want) found = true;
            }
            try testing.expect(found);
        }
    }
    // Weights: match by expert id (reference order vs ours may differ).
    const tw_c = try materializedOwnedCopy(s, r.topk_weights);
    defer _ = mlx.mlx_array_free(tw_c);
    try mlx.check(mlx.mlx_array_eval(tw_c));
    const wp = mlx.mlx_array_data_float32(tw_c) orelse return error.NoData;
    try mlx.check(mlx.mlx_array_eval(tw_ref));
    const wr = mlx.mlx_array_data_float32(tw_ref) orelse return error.NoData;
    for (0..T) |t| {
        const row_ref = ti_ref.items[t].array;
        for (0..K) |k| {
            const expert: u32 = @intCast(row_ref.items[k].integer);
            var ours: f32 = std.math.nan(f32);
            for (0..K) |k2| {
                if (ip[t * K + k2] == expert) ours = wp[t * K + k2];
            }
            try testing.expect(std.math.isFinite(ours));
            try testing.expect(@abs(ours - wr[t * K + k]) < 1e-4);
        }
    }
    try testing.expect(try maxAbsDiffF32(r.shared_gammas, sg_ref, s) < 1e-4);
    std.debug.print("[inkling-router] top-k select + sink weights parity ok\n", .{});
}
