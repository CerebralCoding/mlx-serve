const std = @import("std");
const mlx = @import("mlx.zig");
const mrope = @import("mrope.zig");
const A = mlx.mlx_array;

// mlx-vlm's rotation keeps frequencies, angles and pair arithmetic in float32.
const source =
    \\uint elem = thread_position_in_grid.x;
    \\int half_dim = RD / 2, heads = x_shape[1], seq = x_shape[2], dim = x_shape[3];
    \\int slots = half_dim + dim - RD;
    \\if (elem >= uint(heads * seq * slots)) return;
    \\int slot = elem % slots, t = (elem / slots) % seq;
    \\int base = int(elem / slots) * dim;
    \\if (slot >= half_dim) {
    \\    int d = RD + slot - half_dim;
    \\    out[base + d] = x[base + d];
    \\    return;
    \\}
    \\int axis = selector[slot];
    \\float angle = float(positions[axis * seq + t]) * inv_freq[slot];
    \\float c = metal::cos(angle), s = metal::sin(angle);
    \\float xv = float(x[base + slot]), xp = float(x[base + slot + half_dim]);
    \\out[base + slot] = T(xv * c - xp * s);
    \\out[base + slot + half_dim] = T(xp * c + xv * s);
;

pub const Rotation = struct {
    positions: A,
    inv_freq: A,
    selector: A,
    kernel: mlx.mlx_fast_metal_kernel,
    dims: c_int,
    length: c_int,

    pub fn init(a: std.mem.Allocator, s: mlx.mlx_stream, positions: []const i32, dims: c_int, theta: f32, sections: [3]u32) !Rotation {
        if (dims <= 0 or @rem(dims, 2) != 0 or positions.len == 0 or positions.len % 3 != 0) return error.InvalidMropeShape;
        const half: usize = @intCast(@divExact(dims, 2));
        const selectors = try a.alloc(u8, half);
        defer a.free(selectors);
        mrope.interleavedSelector(selectors, sections);
        const selector = mlx.mlx_array_new_data(selectors.ptr, &[_]c_int{@intCast(half)}, 1, .uint8);
        errdefer _ = mlx.mlx_array_free(selector);
        const pos = mlx.mlx_array_new_data(positions.ptr, &[_]c_int{ 3, @intCast(positions.len / 3) }, 2, .int32);
        errdefer _ = mlx.mlx_array_free(pos);
        var index = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(index);
        try mlx.check(mlx.mlx_arange(&index, 0, @floatFromInt(dims), 2, .float32, s));
        const dim = mlx.mlx_array_new_float(@floatFromInt(dims));
        defer _ = mlx.mlx_array_free(dim);
        const base = mlx.mlx_array_new_float(theta);
        defer _ = mlx.mlx_array_free(base);
        const one = mlx.mlx_array_new_float(1);
        defer _ = mlx.mlx_array_free(one);
        var exponent = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(exponent);
        try mlx.check(mlx.mlx_divide(&exponent, index, dim, s));
        var power = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(power);
        try mlx.check(mlx.mlx_power(&power, base, exponent, s));
        var inv = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(inv);
        try mlx.check(mlx.mlx_divide(&inv, one, power, s));
        const inputs = mlx.mlx_vector_string_new_data(&.{ "x", "positions", "inv_freq", "selector" }, 4);
        defer _ = mlx.mlx_vector_string_free(inputs);
        const outputs = mlx.mlx_vector_string_new_data(&.{"out"}, 1);
        defer _ = mlx.mlx_vector_string_free(outputs);
        const kernel = mlx.mlx_fast_metal_kernel_new("mrope_float32", inputs, outputs, source, "", true, false);
        return .{ .positions = pos, .inv_freq = inv, .selector = selector, .kernel = kernel, .dims = dims, .length = @intCast(positions.len / 3) };
    }

    pub fn deinit(self: *Rotation) void {
        _ = mlx.mlx_fast_metal_kernel_free(self.kernel);
        inline for (.{ self.positions, self.inv_freq, self.selector }) |x| _ = mlx.mlx_array_free(x);
    }

    pub fn apply(self: *const Rotation, s: mlx.mlx_stream, x: A) !A {
        const shape = mlx.getShape(x);
        if (shape.len != 4 or shape[0] != 1 or shape[2] != self.length or shape[3] < self.dims) return error.InvalidMropeShape;
        const config = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "RD", self.dims));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", mlx.mlx_array_dtype(x)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, shape.ptr, shape.len, mlx.mlx_array_dtype(x)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, shape[1] * shape[2] * (shape[3] - @divExact(self.dims, 2)), 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1));
        const inputs = mlx.mlx_vector_array_new_data(&.{ x, self.positions, self.inv_freq, self.selector }, 4);
        defer _ = mlx.mlx_vector_array_free(inputs);
        var outputs = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(outputs);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, self.kernel, inputs, config, s));
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_vector_array_get(&out, outputs, 0));
        return out;
    }
};

test "clef: float32 M-RoPE selects each image axis and preserves the unrotated tail" {
    const s = mlx.gpuStream();
    var rotation = try Rotation.init(std.testing.allocator, s, &.{ 0, 2, 0, 3, 0, 5 }, 6, 64, .{ 1, 1, 1 });
    defer rotation.deinit();
    const values = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
    const x = mlx.mlx_array_new_data(&values, &[_]c_int{ 1, 1, 2, 8 }, 4, .float32);
    defer _ = mlx.mlx_array_free(x);
    const output = try rotation.apply(s, x);
    defer _ = mlx.mlx_array_free(output);
    try mlx.check(mlx.mlx_array_eval(output));
    const actual = mlx.mlx_array_data_float32(output).?[0..values.len];
    try std.testing.expectEqualSlices(f32, values[0..8], actual[0..8]);
    const angles = [_]f32{ 2, 3.0 / 4.0, 5.0 / 16.0 };
    for (angles, 0..) |angle, i| {
        try std.testing.expectApproxEqAbs(values[8 + i] * @cos(angle) - values[11 + i] * @sin(angle), actual[8 + i], 0.000002);
        try std.testing.expectApproxEqAbs(values[11 + i] * @cos(angle) + values[8 + i] * @sin(angle), actual[11 + i], 0.000002);
    }
    try std.testing.expectEqualSlices(f32, values[14..], actual[14..]);
    try std.testing.expectError(error.InvalidMropeShape, Rotation.init(std.testing.allocator, s, &.{ 0, 0, 0 }, 3, 64, .{ 1, 1, 1 }));
    try std.testing.expectError(error.InvalidMropeShape, rotation.apply(s, rotation.positions));
}
