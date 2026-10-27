//! Dense Qwen training with frozen GDN projections and differentiable recurrent state.
const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const ops = @import("../../mlx/ops.zig");
const bases = @import("../../sdk/bases/bases.zig");
const model = @import("../../sdk/train/model.zig");
const plugin = @import("plugin.zig");
const Target = abi.LoraTarget;
const Base = bases.Model(plugin.Dense);
const Linear = bases.blocks.linear.Linear;

pub fn frozen(base: *const Base, prefix: []const u8) !model.Frozen {
    const w = &base.weights;
    if (w.layers.len == 0) return error.TrainingUnsupportedModel;
    try model.validateLinear(w.embed.table);
    try model.validateLinear(w.head);
    for (w.layers) |layer| {
        for (linears(layer)) |linear| if (linear.w.ctx != null) try model.validateLinear(linear);
        if (layer.mixer == .first) {
            const gdn = layer.mixer.first;
            if (gdn.combined) return error.TrainingUnsupportedModel;
            for ([_]Linear{ gdn.qkv, gdn.z, gdn.a, gdn.b, gdn.out } ++ gdn.qkv_parts) |linear| if (linear.w.ctx != null) try model.validateLinear(linear);
        }
    }
    return .{ .context = base, .s = base.s, .head = .{ .head = w.head }, .prefix = prefix, .targets_fn = model.wrapTargets(targets), .hidden_fn = model.wrapHidden(hidden), .activations_fn = activations };
}

fn linears(layer: plugin.Dense.Layer) [7]Linear {
    var result: [7]Linear = .{ .{}, .{}, .{}, .{}, layer.mlp.gate, layer.mlp.up, layer.mlp.down };
    if (layer.mixer == .second) {
        const attention = layer.mixer.second;
        result[0..4].* = .{ attention.q.lin, attention.k.lin, attention.v.lin, attention.o.lin };
    }
    return result;
}

pub fn targets(ctx: *const anyopaque, a: std.mem.Allocator) ![]Target {
    const base: *const Base = @ptrCast(@alignCast(ctx));
    var result: std.ArrayList(Target) = .empty;
    errdefer result.deinit(a);
    for (base.weights.layers, 0..) |layer, i| for (linears(layer), 0..) |linear, projection| {
        if (linear.w.ctx != null) try result.append(a, model.target(linear, @intCast(i), @fromBackingInt(@intCast(projection))));
    };
    return result.toOwnedSlice(a);
}

fn hidden(ctx: *const anyopaque, g: *ops.Graph, ids: ops.Array, bindings: []const abi.LoraBinding) !ops.Array {
    const base: *const Base = @ptrCast(@alignCast(ctx));
    return (try plugin.Dense.forward(&base.weights, &base.shared, g, ids, .{ .mode = .prefill, .rows = ops.shape(ids)[1], .positions = &.{0}, .logits = .none, .want_hidden = true, .lora = bindings })).hidden;
}

fn activations(ctx: *const anyopaque, rows: u32) u128 {
    const base: *const Base = @ptrCast(@alignCast(ctx));
    var bytes: u128 = 0;
    for (base.weights.layers) |layer| {
        const h: u32 = @intCast(base.weights.embed.hidden);
        const intermediate = model.target(layer.mlp.gate, 0, .gate).output;
        switch (layer.mixer) {
            .first => |gdn| {
                const geo = gdn.geo;
                bytes += model.layerBytes(h, intermediate, @intCast(geo.hv), @intCast(geo.hk), @intCast(geo.dk + geo.dv), rows);
                // Autograd retains the f32 recurrent states and their decay/update intermediates.
                bytes += @as(u128, rows) * @as(u128, @intCast(geo.hv)) * @as(u128, @intCast(geo.dk)) * @as(u128, @intCast(geo.dv)) * 24;
            },
            .second => |attention| {
                const geo = attention.geo;
                bytes += model.layerBytes(h, intermediate, @intCast(geo.heads * 2), @intCast(geo.kv_heads), @intCast(geo.head_dim), rows);
            },
        }
    }
    return bytes;
}
