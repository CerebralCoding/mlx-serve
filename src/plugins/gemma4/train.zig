//! Dense Gemma training through the serving decoder, retaining PLE and shared KV gradients.
const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const ops = @import("../../mlx/ops.zig");
const bases = @import("../../sdk/bases/bases.zig");
const model = @import("../../sdk/train/model.zig");
const Target = @import("../../sdk/train/adapter.zig").Target;
const decoder = @import("decoder.zig");
const Base = bases.Model(decoder);
const Linear = bases.blocks.linear.Linear;

pub fn frozen(base: *const Base, prefix: []const u8) !model.Frozen {
    const w = &base.weights;
    if (w.layers.len == 0) return error.TrainingUnsupportedModel;
    try model.validateLinear(w.embed.table);
    try model.validateLinear(w.head.head);
    if (w.ple_dim > 0) {
        try model.validateLinear(w.ple_table);
        try model.validateLinear(w.ple_proj);
    }
    for (w.layers) |layer| {
        if (layer.moe != null) return error.TrainingUnsupportedModel;
        for (linears(layer)) |linear| if (linear.w.ctx != null) try model.validateLinear(linear);
        if (layer.ple) |ple| {
            try model.validateLinear(ple.gate);
            try model.validateLinear(ple.proj);
        }
    }
    return .{ .context = base, .s = base.s, .head = w.head, .prefix = prefix, .targets_fn = model.wrapTargets(targets), .hidden_fn = model.wrapHidden(hidden), .activations_fn = activations };
}

fn linears(layer: decoder.Layer) [7]Linear {
    return .{ layer.attn.q, layer.attn.k, if (layer.attn.k_eq_v) .{} else layer.attn.v, layer.attn.o, layer.mlp.gate, layer.mlp.up, layer.mlp.down };
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
    return (try decoder.forward(&base.weights, &base.shared, g, ids, .{ .mode = .prefill, .rows = ops.shape(ids)[1], .positions = &.{0}, .logits = .none, .want_hidden = true, .lora = bindings })).hidden;
}

fn activations(ctx: *const anyopaque, rows: u32) u128 {
    const base: *const Base = @ptrCast(@alignCast(ctx));
    var bytes: u128 = 0;
    for (base.weights.layers) |layer| {
        const geo = layer.attn.geo;
        bytes += model.layerBytes(@intCast(base.weights.embed.hidden), model.target(layer.mlp.gate, 0, .gate).output, @intCast(geo.heads), @intCast(geo.kv_heads), @intCast(geo.head_dim), rows);
        bytes += @as(u128, rows) * @as(u128, @intCast(base.weights.ple_dim)) * 32;
    }
    return bytes;
}
