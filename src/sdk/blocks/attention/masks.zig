//! sdk/blocks/attention/masks, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns createCausalMask, createSlidingWindowDecodeMask, createSlidingWindowMask, encoderBandMask.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const testing = std.testing;
const bf16Scalar = @import("../../../mlx/array.zig").bf16Scalar;
const Transformer = @import("../../../engines/mlx/transformer.zig").Transformer;

// ── Mask helpers ──

pub fn createCausalMask(self: *const Transformer, q_len: c_int, kv_len: c_int) !mlx.mlx_array {
    const offset_val = kv_len - q_len;
    const shape = [_]c_int{ q_len, kv_len };
    var ones = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ones);
    try mlx.check(mlx.mlx_full(&ones, &shape, 2, self.one, .bfloat16, self.s));
    var upper = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(upper);
    try mlx.check(mlx.mlx_triu(&upper, ones, offset_val + 1, self.s));
    var bool_upper = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bool_upper);
    try mlx.check(mlx.mlx_astype(&bool_upper, upper, .bool_, self.s));
    const zero = bf16Scalar(0.0, self.s);
    defer _ = mlx.mlx_array_free(zero);
    const neg_inf = bf16Scalar(-std.math.inf(f32), self.s);
    defer _ = mlx.mlx_array_free(neg_inf);
    var mask = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_where(&mask, bool_upper, neg_inf, zero, self.s));
    const mask_shape = [_]c_int{ 1, 1, q_len, kv_len };
    var mask_4d = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&mask_4d, mask, &mask_shape, 4, self.s));
    _ = mlx.mlx_array_free(mask);
    return mask_4d;
}

pub fn createSlidingWindowDecodeMask(self: *const Transformer, kv_len: c_int, window: c_int) !mlx.mlx_array {
    var positions = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(positions);
    try mlx.check(mlx.mlx_arange(&positions, 0, @floatFromInt(kv_len), 1, .int32, self.s));
    const window_start = mlx.mlx_array_new_int(kv_len - window);
    defer _ = mlx.mlx_array_free(window_start);
    var too_old = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(too_old);
    try mlx.check(mlx.mlx_less(&too_old, positions, window_start, self.s));
    const zero = bf16Scalar(0.0, self.s);
    defer _ = mlx.mlx_array_free(zero);
    const neg_inf = bf16Scalar(-std.math.inf(f32), self.s);
    defer _ = mlx.mlx_array_free(neg_inf);
    var sw_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sw_mask);
    try mlx.check(mlx.mlx_where(&sw_mask, too_old, neg_inf, zero, self.s));
    const mask_shape = [_]c_int{ 1, 1, 1, kv_len };
    var mask_4d = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&mask_4d, sw_mask, &mask_shape, 4, self.s));
    return mask_4d;
}

pub fn createSlidingWindowMask(self: *const Transformer, q_len: c_int, kv_len: c_int, window: c_int) !mlx.mlx_array {
    const causal = try self.createCausalMask(q_len, kv_len);
    defer _ = mlx.mlx_array_free(causal);
    const offset_val = kv_len - q_len;
    var row_idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(row_idx);
    try mlx.check(mlx.mlx_arange(&row_idx, @floatFromInt(offset_val), @floatFromInt(offset_val + q_len), 1, .int32, self.s));
    var col_idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(col_idx);
    try mlx.check(mlx.mlx_arange(&col_idx, 0, @floatFromInt(kv_len), 1, .int32, self.s));
    const row_shape = [_]c_int{ q_len, 1 };
    const col_shape = [_]c_int{ 1, kv_len };
    var row_r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(row_r);
    var col_r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(col_r);
    try mlx.check(mlx.mlx_reshape(&row_r, row_idx, &row_shape, 2, self.s));
    try mlx.check(mlx.mlx_reshape(&col_r, col_idx, &col_shape, 2, self.s));
    var dist = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dist);
    try mlx.check(mlx.mlx_subtract(&dist, row_r, col_r, self.s));
    const window_arr = mlx.mlx_array_new_int(window);
    defer _ = mlx.mlx_array_free(window_arr);
    var too_far = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(too_far);
    try mlx.check(mlx.mlx_greater_equal(&too_far, dist, window_arr, self.s));
    const neg_inf = bf16Scalar(-std.math.inf(f32), self.s);
    defer _ = mlx.mlx_array_free(neg_inf);
    const zero = bf16Scalar(0.0, self.s);
    defer _ = mlx.mlx_array_free(zero);
    var sw_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sw_mask);
    try mlx.check(mlx.mlx_where(&sw_mask, too_far, neg_inf, zero, self.s));
    const mask_shape = [_]c_int{ 1, 1, q_len, kv_len };
    var sw_4d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sw_4d);
    try mlx.check(mlx.mlx_reshape(&sw_4d, sw_mask, &mask_shape, 4, self.s));
    var combined = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&combined, causal, sw_4d, self.s));
    return combined;
}

/// Additive symmetric band mask for bidirectional sliding-window layers:
/// [1, 1, L, L] bf16, 0 where |i-j| < window, -inf outside — the causal
/// sliding semantic (lookback window-1 + self) mirrored in both directions.
/// Broadcasts additively against a [B, 1, 1, L] key-pad mask.
pub fn encoderBandMask(allocator: std.mem.Allocator, seq_len: usize, window: u32, s: mlx.mlx_stream) !mlx.mlx_array {
    const buf = try allocator.alloc(f32, seq_len * seq_len);
    defer allocator.free(buf);
    for (0..seq_len) |i| {
        for (0..seq_len) |j| {
            const d = if (i > j) i - j else j - i;
            buf[i * seq_len + j] = if (d < window) 0 else -std.math.inf(f32);
        }
    }
    const shape = [_]c_int{ 1, 1, @intCast(seq_len), @intCast(seq_len) };
    const f32_mask = mlx.mlx_array_new_data(buf.ptr, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(f32_mask);
    var mask = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&mask, f32_mask, .bfloat16, s));
    return mask;
}

// ── tests ──

test "encoderBandMask: symmetric |i-j| < window band, -inf outside" {
    const s = mlx.gpuStream();
    const mask = try encoderBandMask(testing.allocator, 4, 2, s);
    defer _ = mlx.mlx_array_free(mask);
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, mask, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f));
    const vals = mlx.mlx_array_data_float32(f).?;
    // Row 0: [0, 0, -inf, -inf]; row 2: [-inf, 0, 0, 0].
    try testing.expectEqual(@as(f32, 0), vals[0]);
    try testing.expectEqual(@as(f32, 0), vals[1]);
    try testing.expect(std.math.isNegativeInf(vals[2]));
    try testing.expect(std.math.isNegativeInf(vals[3]));
    try testing.expect(std.math.isNegativeInf(vals[2 * 4 + 0]));
    try testing.expectEqual(@as(f32, 0), vals[2 * 4 + 1]);
    try testing.expectEqual(@as(f32, 0), vals[2 * 4 + 2]);
    try testing.expectEqual(@as(f32, 0), vals[2 * 4 + 3]);
    const shape = mlx.getShape(mask);
    try testing.expectEqual(@as(c_int, 1), shape[0]);
    try testing.expectEqual(@as(c_int, 1), shape[1]);
    try testing.expectEqual(@as(c_int, 4), shape[2]);
    try testing.expectEqual(@as(c_int, 4), shape[3]);
}
