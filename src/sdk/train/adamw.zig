//! Transactional AdamW updates over f32 adapter masters.
const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const grad = @import("grad.zig");
const Array = ops.Array;

pub const Options = struct { beta1: f32 = 0.9, beta2: f32 = 0.999, epsilon: f32 = 1e-8, weight_decay: f32 = 0, clip_norm: f32 = 1 };

pub const AdamW = struct {
    a: std.mem.Allocator,
    s: ops.Stream,
    moments: []Array,
    variances: []Array,
    updates: u32 = 0,
    options: Options,

    pub fn init(a: std.mem.Allocator, s: ops.Stream, params: []const Array, options: Options) !AdamW {
        if (params.len == 0 or !std.math.isFinite(options.beta1) or options.beta1 < 0 or options.beta1 >= 1 or !std.math.isFinite(options.beta2) or options.beta2 < 0 or options.beta2 >= 1 or !std.math.isFinite(options.epsilon) or options.epsilon <= 0 or !std.math.isFinite(options.weight_decay) or options.weight_decay < 0 or !std.math.isFinite(options.clip_norm) or options.clip_norm <= 0) return error.InvalidOptimizerOptions;
        const m = try empty(a, params.len);
        errdefer release(a, m);
        const v = try empty(a, params.len);
        errdefer release(a, v);
        var g = ops.Graph.init(a, s);
        defer g.deinit();
        for (params, 0..) |p, i| {
            if (ops.dtype(p) != .float32) return error.TrainingMasterMustBeFloat32;
            const zero = try g.zeros(ops.shape(p), .float32);
            m[i] = g.escape(zero);
            v[i] = g.escape(zero);
        }
        try ops.evalMany(m);
        return .{ .a = a, .s = s, .moments = m, .variances = v, .options = options };
    }
    pub fn deinit(self: *AdamW) void {
        release(self.a, self.moments);
        release(self.a, self.variances);
    }
    pub fn update(self: *AdamW, params: []Array, gradients: []const Array, lr: f32) !f32 {
        if (params.len != gradients.len or params.len != self.moments.len or !std.math.isFinite(lr) or lr < 0 or self.updates == std.math.maxInt(u32)) return error.InvalidOptimizerStep;
        var g = ops.Graph.init(self.a, self.s);
        defer g.deinit();
        const next = try empty(self.a, params.len * 3);
        defer release(self.a, next);
        var squared_norm = try g.scalar(0, .float32);
        for (params, gradients) |p, d| {
            if (!std.mem.eql(c_int, ops.shape(p), ops.shape(d))) return error.InvalidTrainingShape;
            var t = g.scope();
            defer t.deinit();
            squared_norm = try g.add(squared_norm, try g.keep(try t.call(mlx.mlx_sum, .{ try t.square(try t.astype(d, .float32)), false })));
        }
        const norm = @sqrt(try grad.item(squared_norm));
        if (!std.math.isFinite(norm)) return error.NonFiniteTrainingGradient;
        const o = self.options;
        const clip = @min(@as(f32, 1), o.clip_norm / @max(norm, 1e-30));
        const t: f32 = @floatFromInt(self.updates + 1);
        const correction1 = 1 - std.math.pow(f32, o.beta1, t);
        const correction2 = 1 - std.math.pow(f32, o.beta2, t);
        for (params, gradients, 0..) |p, raw, i| {
            var part = g.scope();
            defer part.deinit();
            const d = try part.mulScalar(try part.astype(raw, .float32), clip);
            const m = try part.add(try part.mulScalar(self.moments[i], o.beta1), try part.mulScalar(d, 1 - o.beta1));
            const v = try part.add(try part.mulScalar(self.variances[i], o.beta2), try part.mulScalar(try part.square(d), 1 - o.beta2));
            const direction = try part.div(try part.mulScalar(m, 1 / correction1), try part.addScalar(try part.sqrt(try part.mulScalar(v, 1 / correction2)), o.epsilon));
            const updated = try part.sub(try part.mulScalar(p, 1 - lr * o.weight_decay), try part.mulScalar(direction, lr));
            next[3 * i] = part.escape(updated);
            next[3 * i + 1] = part.escape(m);
            next[3 * i + 2] = part.escape(v);
        }
        try ops.evalMany(next);
        try finite(&g, next);
        for (params, 0..) |*p, i| {
            std.mem.swap(Array, p, &next[3 * i]);
            std.mem.swap(Array, &self.moments[i], &next[3 * i + 1]);
            std.mem.swap(Array, &self.variances[i], &next[3 * i + 2]);
        }
        self.updates += 1;
        return norm;
    }
};

pub fn empty(a: std.mem.Allocator, count: usize) ![]Array {
    const arrays = try a.alloc(Array, count);
    @memset(arrays, .{});
    return arrays;
}

pub fn finite(g: *ops.Graph, arrays: []const Array) !void {
    var maximum = try g.scalar(0, .float32);
    for (arrays) |x| {
        var t = g.scope();
        defer t.deinit();
        maximum = try g.maximum(maximum, try g.keep(try t.call(mlx.mlx_max, .{ try t.abs(x), false })));
    }
    if (!std.math.isFinite(try grad.item(maximum))) return error.NonFiniteTrainingGradient;
}

pub fn release(a: std.mem.Allocator, arrays: []Array) void {
    for (arrays) |x| if (x.ctx != null) {
        _ = mlx.mlx_array_free(x);
    };
    a.free(arrays);
}

test "train core: AdamW bias correction, clipping and zero decay match a scalar oracle" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = ops.Graph.init(std.testing.allocator, s);
    defer g.deinit();
    var params = [_]Array{g.escape(try g.scalar(2, .float32))};
    defer _ = mlx.mlx_array_free(params[0]);
    var optimizer = try AdamW.init(g.a, s, &params, .{ .beta1 = 0.5, .beta2 = 0.5, .epsilon = 0.1, .clip_norm = 1 });
    defer optimizer.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 3), try optimizer.update(&params, &.{try g.scalar(3, .float32)}, 0.1), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2 - 0.1 / 1.1), try grad.item(params[0]), 1e-6);
    const want: f32 = 2 - 0.1 / 1.1 - 0.1 * (-1.0 / 3.0) / 1.1;
    _ = try optimizer.update(&params, &.{try g.scalar(-2, .float32)}, 0.1);
    try std.testing.expectApproxEqAbs(want, try grad.item(params[0]), 1e-6);
    try std.testing.expectEqual(@as(u32, 2), optimizer.updates);
    try std.testing.expectError(error.NonFiniteTrainingGradient, optimizer.update(&params, &.{try g.scalar(std.math.nan(f32), .float32)}, 0.1));
    try std.testing.expectApproxEqAbs(want, try grad.item(params[0]), 1e-6);
    try std.testing.expectEqual(@as(u32, 2), optimizer.updates);
}
