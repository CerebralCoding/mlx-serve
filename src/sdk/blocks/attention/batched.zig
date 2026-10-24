//! sdk/blocks/attention/batched, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns BATCHED_PER_SLOT_ATTN_MIN_KV, batchedAttnPerSlot, stackedBatchedAttn, perSlotBatchedAttn, padAndStackBatchedKV, buildBatchedDecodeMask.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../host/log.zig");
const layered = @import("../../../engines/mlx/batched/layered.zig");
const MAX_BATCH_ROWS = @import("../../../engines/mlx/batched/layered.zig").MAX_BATCH_ROWS;
const bf16Scalar = @import("../../../mlx/array.zig").bf16Scalar;
const scalarOf = @import("../../../mlx/array.zig").scalarOf;
const Transformer = @import("../../../engines/mlx/transformer.zig").Transformer;
const axisView = @import("../../../mlx/array.zig").axisView;
const testRandWeightBf16 = @import("../../testing/parity.zig").testRandWeightBf16;
const fusedSdpa256Prefill = @import("attn256.zig").fusedSdpa256Prefill;
const kvAttnFusedEligible = @import("kv_quant_attn.zig").kvAttnFusedEligible;
const kvAttnVerifyEligible = @import("kv_quant_attn.zig").kvAttnVerifyEligible;
const logKvAttnFusedEngaged = @import("kv_quant_attn.zig").logKvAttnFusedEngaged;
const qkvAttnDecodeKernel = @import("kv_quant_attn.zig").qkvAttnDecodeKernel;
const qkvAttnVerifyKernel = @import("kv_quant_attn.zig").qkvAttnVerifyKernel;
const DenseKVView = @import("kv_view.zig").DenseKVView;
const sdpaForceFused = @import("sdpa.zig").sdpaForceFused;
const splitCausalSdpa = @import("sdpa.zig").splitCausalSdpa;

/// Longest slot view at or past which a batched group attends per slot: below it one
/// stacked dispatch is cheaper than N, above it the pad + concat copy dominates.
pub const BATCHED_PER_SLOT_ATTN_MIN_KV: c_int = 1024;

pub fn batchedAttnPerSlot(views: []const DenseKVView) bool {
    for (views) |dv| if (mlx.getShape(dv.k)[2] >= BATCHED_PER_SLOT_ATTN_MIN_KV) return true;
    return false;
}

/// The stacked batched read: every slot's view padded to the longest and concatenated,
/// one array-mask SDPA for the group. `narrow` (bool, null-ctx for none) removes more
/// positions from the pad mask.
pub fn stackedBatchedAttn(
    self: *const Transformer,
    q: mlx.mlx_array,
    views: []const DenseKVView,
    seq_len: c_int,
    attn_scale: f32,
    narrow: mlx.mlx_array,
) !mlx.mlx_array {
    var kv_len_buf: [MAX_BATCH_ROWS]i32 = undefined;
    var kv_max: c_int = 0;
    for (views, 0..) |dv, i| {
        kv_len_buf[i] = mlx.getShape(dv.k)[2];
        kv_max = @max(kv_max, kv_len_buf[i]);
    }
    const stacked_k = try self.padAndStackBatchedKV(views, true, kv_max);
    defer _ = mlx.mlx_array_free(stacked_k);
    const stacked_v = try self.padAndStackBatchedKV(views, false, kv_max);
    defer _ = mlx.mlx_array_free(stacked_v);
    var mask = try self.buildBatchedDecodeMask(kv_len_buf[0..views.len], kv_max, seq_len, mlx.mlx_array_dtype(q));
    defer _ = mlx.mlx_array_free(mask);
    if (narrow.ctx != null) {
        const neg_inf = try scalarOf(-std.math.inf(f32), mlx.mlx_array_dtype(mask), self.s);
        defer _ = mlx.mlx_array_free(neg_inf);
        var narrowed = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_where(&narrowed, narrow, mask, neg_inf, self.s));
        _ = mlx.mlx_array_free(mask);
        mask = narrowed;
    }
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&out, q, stacked_k, stacked_v, attn_scale, "array", mask, .{ .ctx = null }, false, self.s));
    return out;
}

/// The per-slot batched read: each slot's query rows attend its OWN view, so nothing is
/// padded, copied or masked, and the causal and quantized-KV kernels serve it as they
/// serve a solo slot. `fused[i]` is slot i's `ForwardCtx.kv_attn_fused`.
pub fn perSlotBatchedAttn(
    self: *const Transformer,
    q: mlx.mlx_array,
    views: []const DenseKVView,
    fused: []const bool,
    seq_len: c_int,
    attn_scale: f32,
) !mlx.mlx_array {
    const none_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none_mask);
    var outs: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
    var n: usize = 0;
    defer for (outs[0..n]) |o| {
        _ = mlx.mlx_array_free(o);
    };
    if (!layered.per_slot_attn_logged) {
        layered.per_slot_attn_logged = true;
        log.info("[batched] per-slot attention engaged (slots={d}, rows={d}): no KV pad or stack past {d} tokens\n", .{ views.len, seq_len, BATCHED_PER_SLOT_ATTN_MIN_KV });
    }
    for (views, 0..) |*dv, i| {
        const q_i = try axisView(self.s, q, 0, i);
        defer _ = mlx.mlx_array_free(q_i);
        const mode: [:0]const u8 = if (seq_len > 1) "causal" else "";
        var out: ?mlx.mlx_array = null;
        if (fused[i] and kvAttnFusedEligible(dv, seq_len)) {
            out = try qkvAttnDecodeKernel(self.s, q_i, dv, attn_scale, mode, none_mask);
            if (out != null) logKvAttnFusedEngaged(dv, q_i, seq_len);
        } else if (fused[i] and kvAttnVerifyEligible(dv, seq_len)) {
            out = try qkvAttnVerifyKernel(self.s, q_i, dv, attn_scale, mode);
        }
        if (out == null and seq_len > 1) {
            out = try fusedSdpa256Prefill(self.s, q_i, dv.k, dv.v, attn_scale, 0);
            if (out == null) out = try splitCausalSdpa(self.s, q_i, dv.k, dv.v, attn_scale);
        }
        if (out == null) {
            var o = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(o);
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&o, q_i, dv.k, dv.v, attn_scale, mode.ptr, none_mask, .{ .ctx = null }, seq_len > 1 and sdpaForceFused(q_i, dv.k), self.s));
            out = o;
        }
        outs[n] = out.?;
        n += 1;
    }
    const vec = mlx.mlx_vector_array_new_data(&outs, n);
    defer _ = mlx.mlx_vector_array_free(vec);
    var joined = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(joined);
    try mlx.check(mlx.mlx_concatenate_axis(&joined, vec, 0, self.s));
    return joined;
}

// Pads each slot's dense KV view (shape [1, kv_h, kv_len_i, head_dim]) to a
// common kv_max along axis 2 with bf16 zeros and concatenates axis 0 →
// [N, kv_h, kv_max, head_dim]. Views come from KVCache.denseView so quant
// schemes are already dequantized; `key_not_value` selects k or v.
pub fn padAndStackBatchedKV(
    self: *const Transformer,
    views: []const DenseKVView,
    key_not_value: bool,
    kv_max: c_int,
) !mlx.mlx_array {
    const pad_axes = [_]c_int{2};
    const pad_value = bf16Scalar(0.0, self.s);
    defer _ = mlx.mlx_array_free(pad_value);

    const padded_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(padded_vec);
    // Track padded arrays so we can free them after the concat.
    var padded_arrs = try self.allocator.alloc(mlx.mlx_array, views.len);
    defer {
        for (padded_arrs) |a| _ = mlx.mlx_array_free(a);
        self.allocator.free(padded_arrs);
    }

    for (views, 0..) |dv, i| {
        const view = if (key_not_value) dv.k else dv.v;
        const vshape = mlx.getShape(view);
        const klen: c_int = vshape[2];
        const high_pad: c_int = kv_max - klen;
        const low_pad_arr = [_]c_int{0};
        const high_pad_arr = [_]c_int{high_pad};
        padded_arrs[i] = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_pad(
            &padded_arrs[i],
            view,
            &pad_axes,
            1,
            &low_pad_arr,
            1,
            &high_pad_arr,
            1,
            pad_value,
            "constant",
            self.s,
        ));
        _ = mlx.mlx_vector_array_append_value(padded_vec, padded_arrs[i]);
    }

    var stacked = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&stacked, padded_vec, 0, self.s));
    return stacked;
}

// Builds the additive per-slot decode mask [N,1,1,kv_max] in bf16 where
// valid columns are 0 and out-of-range columns are -inf. Computed via
// broadcasting: positions[1,1,1,kv_max] < kv_lens[N,1,1,1].
/// `[N, 1, S, kv_max]` additive mask: row j of slot n sees positions below
/// `kv_len_n - S + 1 + j` (its own causal prefix; the pad past `kv_len_n` never).
/// Additive [N, 1, S, kv_max] mask in `dt`, the queries' dtype: sdpa
/// rejects a mask that does not promote to its output type.
pub fn buildBatchedDecodeMask(
    self: *const Transformer,
    kv_lens: []const i32,
    kv_max: c_int,
    seq_len: c_int,
    dt: mlx.mlx_dtype,
) !mlx.mlx_array {
    const N: c_int = @intCast(kv_lens.len);
    var positions = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(positions);
    try mlx.check(mlx.mlx_arange(&positions, 0, @floatFromInt(kv_max), 1, .int32, self.s));
    const pos_shape = [_]c_int{ 1, 1, 1, kv_max };
    var pos_4d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pos_4d);
    try mlx.check(mlx.mlx_reshape(&pos_4d, positions, &pos_shape, 4, self.s));

    // limit[n, 0, j, 0] = kv_len_n - S + 1 + j
    var lim_buf: [MAX_BATCH_ROWS * 16]i32 = undefined;
    if (kv_lens.len * @as(usize, @intCast(seq_len)) > lim_buf.len) return error.BatchedMaskTooWide;
    for (kv_lens, 0..) |kl, n| {
        for (0..@intCast(seq_len)) |j| lim_buf[n * @as(usize, @intCast(seq_len)) + j] = kl - seq_len + 1 + @as(i32, @intCast(j));
    }
    const lim_shape = [_]c_int{ N, 1, seq_len, 1 };
    const lens_4d = mlx.mlx_array_new_data(lim_buf[0 .. kv_lens.len * @as(usize, @intCast(seq_len))].ptr, &lim_shape, 4, .int32);
    defer _ = mlx.mlx_array_free(lens_4d);

    var valid = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(valid);
    try mlx.check(mlx.mlx_less(&valid, pos_4d, lens_4d, self.s));

    const zero = try scalarOf(0.0, dt, self.s);
    defer _ = mlx.mlx_array_free(zero);
    const neg_inf = try scalarOf(-std.math.inf(f32), dt, self.s);
    defer _ = mlx.mlx_array_free(neg_inf);
    var mask = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_where(&mask, valid, zero, neg_inf, self.s));
    return mask;
}

// ── tests ──

test "batched decode mask at verify width: row j of slot n sees its own causal prefix only" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const t = std.testing;
    var xfm: Transformer = undefined;
    xfm.rht = null;
    xfm.s = mlx.gpuStream();
    xfm.allocator = t.allocator;
    // Two slots, three verify rows each; kv_len is the length AFTER the rows were appended.
    const kv_lens = [_]i32{ 5, 8 };
    const mask = try xfm.buildBatchedDecodeMask(&kv_lens, 8, 3, .bfloat16);
    defer _ = mlx.mlx_array_free(mask);
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, mask, .float32, xfm.s));
    try mlx.check(mlx.mlx_array_eval(f));
    const d = mlx.mlx_array_data_float32(f) orelse return error.MlxArrayDataNull;
    const sh = mlx.getShape(f);
    try t.expectEqual(@as(c_int, 2), sh[0]);
    try t.expectEqual(@as(c_int, 3), sh[2]);
    try t.expectEqual(@as(c_int, 8), sh[3]);
    for (0..2) |n| for (0..3) |j| for (0..8) |pos| {
        const visible = @as(i32, @intCast(pos)) < kv_lens[n] - 3 + 1 + @as(i32, @intCast(j));
        const v = d[(n * 3 + j) * 8 + pos];
        if (visible) try t.expectEqual(@as(f32, 0.0), v) else try t.expect(v == -std.math.inf(f32));
    };
}

test "per-slot batched attention equals the stacked arm on skewed slot lengths" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const t = std.testing;
    var xfm: Transformer = undefined;
    xfm.rht = null;
    xfm.s = mlx.gpuStream();
    xfm.allocator = t.allocator;
    var prng = std.Random.DefaultPrng.init(0x51075);
    const rnd = prng.random();
    const lens = [_]c_int{ 37, 1500 };
    var views: [2]DenseKVView = undefined;
    var made: usize = 0;
    defer for (views[0..made]) |dv| {
        _ = mlx.mlx_array_free(dv.k);
        _ = mlx.mlx_array_free(dv.v);
    };
    for (lens, 0..) |len, i| {
        views[i] = .{
            .k = try testRandWeightBf16(rnd, &[_]c_int{ 1, 4, len, 256 }, xfm.s),
            .v = try testRandWeightBf16(rnd, &[_]c_int{ 1, 4, len, 256 }, xfm.s),
            .owned = false,
        };
        made += 1;
    }
    for ([_]c_int{ 1, 4 }) |seq| {
        const q = try testRandWeightBf16(rnd, &[_]c_int{ 2, 24, seq, 256 }, xfm.s);
        defer _ = mlx.mlx_array_free(q);
        const stacked = try xfm.stackedBatchedAttn(q, &views, seq, 0.0625, .{ .ctx = null });
        defer _ = mlx.mlx_array_free(stacked);
        const fused = [_]bool{ false, false };
        const per_slot = try xfm.perSlotBatchedAttn(q, &views, &fused, seq, 0.0625);
        defer _ = mlx.mlx_array_free(per_slot);
        try t.expectEqualSlices(c_int, mlx.getShape(stacked), mlx.getShape(per_slot));
        var diff = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(diff);
        try mlx.check(mlx.mlx_subtract(&diff, stacked, per_slot, xfm.s));
        var mag = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mag);
        try mlx.check(mlx.mlx_abs(&mag, diff, xfm.s));
        var mx = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mx);
        try mlx.check(mlx.mlx_max(&mx, mag, false, xfm.s));
        var mx32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mx32);
        try mlx.check(mlx.mlx_astype(&mx32, mx, .float32, xfm.s));
        try mlx.check(mlx.mlx_array_eval(mx32));
        var worst: f32 = 0;
        try mlx.check(mlx.mlx_array_item_float32(&worst, mx32));
        // Values are U(-0.5, 0.5) averages; two kernels agree to bf16 rounding.
        try t.expect(worst < 5e-3);
    }
}

test "batched decode mask is built in the query dtype: an f16 batch attends through it" {
    // A bf16 mask against f16 queries is an MLX error mid-generation (sdpa
    // requires the mask to promote to the output dtype).
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var xfm: Transformer = undefined;
    xfm.rht = null;
    xfm.s = mlx.gpuStream();
    const kv_lens = [_]i32{ 5, 8 };
    const mask = try xfm.buildBatchedDecodeMask(&kv_lens, 8, 3, .float16);
    defer _ = mlx.mlx_array_free(mask);
    try std.testing.expectEqual(mlx.mlx_dtype.float16, mlx.mlx_array_dtype(mask));
    var q = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q);
    var kv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kv);
    try mlx.check(mlx.mlx_ones(&q, &[_]c_int{ 2, 4, 3, 64 }, 4, .float16, xfm.s));
    try mlx.check(mlx.mlx_ones(&kv, &[_]c_int{ 2, 2, 8, 64 }, 4, .float16, xfm.s));
    var out = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&out, q, kv, kv, 0.125, "array", mask, .{ .ctx = null }, false, xfm.s));
    try mlx.check(mlx.mlx_array_eval(out));
    try std.testing.expectEqual(mlx.mlx_dtype.float16, mlx.mlx_array_dtype(out));
}
