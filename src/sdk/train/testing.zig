//! Shared gradient and activation-reservation checks for native training forwards.
const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const model = @import("model.zig");
const adapter_mod = @import("adapter.zig");
const grad = @import("grad.zig");
const Target = @import("adapter.zig").Target;

pub fn expectGradients(g: *ops.Graph, frozen: model.Frozen, adapter: *adapter_mod.Adapter) !void {
    const previous = g.differentiable;
    g.differentiable = true;
    defer g.differentiable = previous;
    for (adapter.params, 0..) |*param, i| if (i % 2 == 1) {
        const changed = try g.addScalar(param.*, 0.01);
        _ = mlx.mlx_array_free(param.*);
        param.* = g.escape(changed);
    };
    const Objective = struct {
        frozen: model.Frozen,
        adapter: *adapter_mod.Adapter,
        ids: ops.Array,
        labels: ops.Array,
        mask: ops.Array,
        fn call(self: @This(), graph: *ops.Graph, params: []const ops.Array) !ops.Array {
            const bindings = try self.adapter.bindings(graph, params);
            defer graph.a.free(bindings);
            return @import("loss.zig").projectedHead(graph, try self.frozen.hidden(graph, self.ids, bindings), self.frozen.head, self.labels, self.mask, 61);
        }
    };
    const objective: Objective = .{ .frozen = frozen, .adapter = adapter, .ids = try g.fromSlice(u32, &.{ 3, 17, 42, 99, 7, 250 }, &.{ 1, 6 }), .labels = try g.fromSlice(u32, &.{ 17, 42, 99, 7, 250, 1 }, &.{6}), .mask = try g.fromSlice(f32, &.{ 0, 0, 0, 1, 1, 1 }, &.{6}) };
    const result = try grad.valueAndGrad(g, objective, Objective.call, adapter.params);
    defer result.deinit(g.a);
    for (adapter.params, result.gradients, 0..) |param, derivative, i| {
        var scope = g.scope();
        defer scope.deinit();
        const shape = ops.shape(param);
        const key = try scope.call(mlx.mlx_random_key, .{@as(u64, i + 50)});
        const direction = try scope.call(mlx.mlx_random_normal, .{ shape.ptr, shape.len, ops.Dtype.float32, @as(f32, 0), @as(f32, 0.1), key });
        const analytic = try grad.item(try scope.call(mlx.mlx_sum, .{ try scope.mul(direction, derivative), false }));
        const varied = try g.a.dupe(ops.Array, adapter.params);
        defer g.a.free(varied);
        const delta = try scope.mulScalar(direction, 0.001);
        varied[i] = try scope.add(param, delta);
        const plus = try grad.item(try objective.call(&scope, varied));
        varied[i] = try scope.sub(param, delta);
        const minus = try grad.item(try objective.call(&scope, varied));
        try std.testing.expectApproxEqAbs((plus - minus) / 0.002, analytic, 0.002);
    }
}

pub fn expectReservation(frozen: model.Frozen, targets: []const Target) !void {
    const a = std.testing.allocator;
    var adapter = try adapter_mod.Adapter.init(a, frozen.s, targets, .{});
    defer adapter.deinit();
    var trainer = try @import("step.zig").Trainer.initForTest(a, frozen.s, frozen, &adapter, .{ .accumulation = 1 });
    defer trainer.deinit();
    var tokens: [513]u32 = undefined;
    for (&tokens, 0..) |*token, i| token.* = @intCast(i % 256);
    const example = try @import("packing.zig").pack(a, &tokens, 500, 512);
    defer example.deinit(a);
    try mlx.check(mlx.mlx_synchronize(frozen.s));
    try mlx.check(mlx.mlx_clear_cache());
    var resident: usize = 0;
    try mlx.check(mlx.mlx_get_active_memory(&resident));
    try mlx.check(mlx.mlx_reset_peak_memory());
    _ = try trainer.microbatch(example);
    var peak: usize = 0;
    try mlx.check(mlx.mlx_get_peak_memory(&peak));
    const bill = try @import("memory.zig").reserve(frozen, targets, 8, 512);
    std.debug.print("\nLoRA seq512: extra={d} reserved={d}\n", .{ peak -| resident, bill });
    try std.testing.expect(peak -| resident <= bill);
}
