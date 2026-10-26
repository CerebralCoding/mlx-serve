//! Differentiable closures whose returned arrays belong to the caller's Graph.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const Array = ops.Array;
const Graph = ops.Graph;

pub const Result = struct {
    value: Array,
    gradients: []Array,

    pub fn deinit(r: Result, a: std.mem.Allocator) void {
        a.free(r.gradients);
    }
};

pub fn valueAndGrad(g: *Graph, context: anytype, comptime loss: anytype, params: []const Array) !Result {
    if (params.len == 0 or params.len > std.math.maxInt(c_int)) return error.InvalidTrainableParameters;
    const Payload = struct {
        graph: *Graph,
        context: @TypeOf(context),
        failure: ?anyerror = null,

        fn call(out: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, raw: ?*anyopaque) callconv(.c) c_int {
            const p: *@This() = @ptrCast(@alignCast(raw.?));
            p.run(out, input) catch |err| {
                p.failure = err;
                return 1;
            };
            return 0;
        }

        fn run(p: *@This(), out: *mlx.mlx_vector_array, input: mlx.mlx_vector_array) !void {
            var t = p.graph.scope();
            defer t.deinit();
            t.differentiable = true;
            const args = try t.a.alloc(Array, mlx.mlx_vector_array_size(input));
            defer t.a.free(args);
            for (args, 0..) |*x, i| x.* = try get(&t, input, i);
            const value = try loss(p.context, &t, args);
            if (mlx.mlx_array_size(value) != 1) return error.TrainingLossMustBeScalar;
            _ = mlx.mlx_vector_array_free(out.*);
            out.* = mlx.mlx_vector_array_new_value(value);
        }
    };
    var payload: Payload = .{ .graph = g, .context = context };
    const fun = mlx.mlx_closure_new_func_payload(Payload.call, &payload, null);
    defer _ = mlx.mlx_closure_free(fun);
    var vg: mlx.mlx_closure_value_and_grad = .{};
    defer _ = mlx.mlx_closure_value_and_grad_free(vg);
    const argnums = try g.a.alloc(c_int, params.len);
    defer g.a.free(argnums);
    for (argnums, 0..) |*n, i| n.* = @intCast(i);
    try mlx.check(mlx.mlx_value_and_grad(&vg, fun, argnums.ptr, argnums.len));
    const input = mlx.mlx_vector_array_new_data(params.ptr, params.len);
    defer _ = mlx.mlx_vector_array_free(input);
    var output = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(output);
    var grads = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(grads);
    const had_error = mlx.errorPending();
    const status = mlx.mlx_closure_value_and_grad_apply(&output, &grads, vg, input);
    if (payload.failure) |err| {
        mlx.dropLatchedErrorUnless(had_error);
        return err;
    }
    try mlx.check(status);
    if (mlx.mlx_vector_array_size(output) != 1 or mlx.mlx_vector_array_size(grads) != params.len) return error.InvalidTrainingGradient;
    const result = try g.a.alloc(Array, params.len);
    errdefer g.a.free(result);
    for (result, 0..) |*x, i| x.* = try get(g, grads, i);
    return .{ .value = try get(g, output, 0), .gradients = result };
}

pub fn get(g: *Graph, vector: mlx.mlx_vector_array, index: usize) !Array {
    var x = mlx.mlx_array_new();
    mlx.check(mlx.mlx_vector_array_get(&x, vector, index)) catch |err| {
        _ = mlx.mlx_array_free(x);
        return err;
    };
    return g.adopt(x);
}

pub fn item(x: Array) !f32 {
    try ops.eval(x);
    var v: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&v, x));
    return v;
}

test "train core: value and grad owns every output and differentiates all parameters" {
    const Objective = struct {
        fn loss(_: void, g: *Graph, p: []const Array) !Array {
            return g.add(try g.call(mlx.mlx_sum, .{ try g.square(p[0]), false }), try g.call(mlx.mlx_sum, .{ try g.mulScalar(p[1], 3), false }));
        }
    };
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const before = ops.liveForTest();
    {
        var g = Graph.init(std.testing.allocator, s);
        defer g.deinit();
        const p = [_]Array{ try g.fromSlice(f32, &.{ 1, -2, 3 }, &.{3}), try g.scalar(4, .float32) };
        const result = try valueAndGrad(&g, {}, Objective.loss, &p);
        defer result.deinit(g.a);
        try std.testing.expectApproxEqAbs(@as(f32, 26), try item(result.value), 1e-6);
        try std.testing.expectEqual(@as(usize, 2), result.gradients.len);
        try ops.expectBitEqual(try g.fromSlice(f32, &.{ 2, -4, 6 }, &.{3}), result.gradients[0]);
        try std.testing.expectEqual(@as(f32, 3), try item(result.gradients[1]));
    }
    try std.testing.expectEqual(before, ops.liveForTest());
}

test "train core: a closure failure preserves its Zig error and frees its graph" {
    mlx.installErrorHandler();
    const Objective = struct {
        fn loss(_: void, g: *Graph, p: []const Array) !Array {
            _ = try g.square(p[0]);
            return error.InjectedTrainingFailure;
        }
    };
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = Graph.init(std.testing.allocator, s);
    defer g.deinit();
    const p = try g.scalar(2, .float32);
    const before = ops.liveForTest();
    try std.testing.expectError(error.InjectedTrainingFailure, valueAndGrad(&g, {}, Objective.loss, &.{p}));
    try std.testing.expectEqual(before, ops.liveForTest());
}
