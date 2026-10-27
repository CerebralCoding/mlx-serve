//! Borrowed architecture-specific training forwards; all arrays stay owned by the loaded model.
const std = @import("std");
const ops = @import("../../mlx/ops.zig");
const bases = @import("../bases/bases.zig");
const b = bases.blocks;
const adapter = @import("adapter.zig");
const abi = @import("../../abi/plugin.zig");
pub const Decoder = bases.Decoder(.{ .embed = b.embed.TableEmbed, .norm = b.norm.RmsNorm, .mixer = b.attention.Attention, .mlp = b.mlp.SwiGlu, .seam = b.seam.ResidualTail, .head = b.head.LmHead });
pub const Base = bases.Model(Decoder);

pub const Frozen = struct {
    context: *const anyopaque,
    s: ops.Stream,
    head: b.extras.SoftcapHead.Weights,
    prefix: []const u8 = "model",
    targets_fn: *const fn (*const anyopaque, std.mem.Allocator, *[]adapter.Target) ?[]const u8,
    hidden_fn: *const fn (*const anyopaque, *ops.Graph, ops.Array, []const abi.LoraBinding, *ops.Array) ?[]const u8,
    activations_fn: *const fn (*const anyopaque, u32) u128,

    pub fn init(base: *const Base) !Frozen {
        if (base.weights.layers.len == 0) return error.TrainingUnsupportedModel;
        try validateLinear(base.weights.embed.table);
        try validateLinear(base.weights.head);
        for (base.weights.layers) |layer| for (linears(layer)) |linear| try validateLinear(linear);
        return .{ .context = base, .s = base.s, .head = .{ .head = base.weights.head }, .targets_fn = wrapTargets(plainTargets), .hidden_fn = wrapHidden(plainHidden), .activations_fn = plainActivations };
    }
    pub fn targets(self: Frozen, a: std.mem.Allocator) ![]adapter.Target {
        var out: []adapter.Target = undefined;
        try @import("../util/err_name.zig").check(self.targets_fn(self.context, a, &out));
        return out;
    }
    pub fn project(self: Frozen, g: *ops.Graph, hidden_rows: ops.Array) !ops.Array {
        return b.extras.SoftcapHead.project(g, &self.head, hidden_rows);
    }
    pub fn activationBytes(self: Frozen, rows: u32) u128 {
        return self.activations_fn(self.context, rows);
    }
    pub fn hidden(self: Frozen, g: *ops.Graph, ids: ops.Array, bindings: []const abi.LoraBinding) !ops.Array {
        const shape = ops.shape(ids);
        if (shape.len != 2 or shape[0] != 1 or shape[1] < 1 or shape[1] > 512) return error.InvalidTrainingShape;
        const previous = g.differentiable;
        g.differentiable = true;
        defer g.differentiable = previous;
        var out: ops.Array = undefined;
        try @import("../util/err_name.zig").check(self.hidden_fn(self.context, g, ids, bindings, &out));
        return out;
    }
};

pub fn wrapTargets(comptime f: anytype) *const fn (*const anyopaque, std.mem.Allocator, *[]adapter.Target) ?[]const u8 {
    return struct {
        fn call(ctx: *const anyopaque, a: std.mem.Allocator, out: *[]adapter.Target) ?[]const u8 {
            out.* = f(ctx, a) catch |e| return @errorName(e);
            return null;
        }
    }.call;
}

pub fn wrapHidden(comptime f: anytype) *const fn (*const anyopaque, *ops.Graph, ops.Array, []const abi.LoraBinding, *ops.Array) ?[]const u8 {
    return struct {
        fn call(ctx: *const anyopaque, g: *ops.Graph, ids: ops.Array, bindings: []const abi.LoraBinding, out: *ops.Array) ?[]const u8 {
            out.* = f(ctx, g, ids, bindings) catch |e| return @errorName(e);
            return null;
        }
    }.call;
}

fn plainTargets(ctx: *const anyopaque, a: std.mem.Allocator) ![]adapter.Target {
    const base: *const Base = @ptrCast(@alignCast(ctx));
    const result = try a.alloc(adapter.Target, base.weights.layers.len * 7);
    for (base.weights.layers, 0..) |layer, index| for (linears(layer), 0..) |linear, projection| {
        result[index * 7 + projection] = target(linear, @intCast(index), @fromBackingInt(@intCast(projection)));
    };
    return result;
}

pub fn target(linear: b.linear.Linear, layer: u32, projection: abi.LoraProjection) adapter.Target {
    const sh = ops.shape(linear.w);
    return .{ .layer = layer, .projection = projection, .input = @intCast(if (linear.s.ctx != null) @divExact(sh[1] * 32, @as(c_int, @intCast(linear.bits))) else sh[0]), .output = @intCast(if (linear.s.ctx != null) sh[0] else sh[1]) };
}

fn plainHidden(ctx: *const anyopaque, g: *ops.Graph, ids: ops.Array, bindings: []const abi.LoraBinding) !ops.Array {
    const base: *const Base = @ptrCast(@alignCast(ctx));
    return (try Decoder.forward(&base.weights, &base.shared, g, ids, .{ .mode = .prefill, .rows = ops.shape(ids)[1], .positions = &.{0}, .logits = .none, .want_hidden = true, .lora = bindings })).hidden;
}

pub fn layerBytes(hidden: u32, intermediate: u32, heads: u32, kv_heads: u32, width: u32, rows: u32) u128 {
    return @as(u128, rows) * (@as(u128, hidden) * 16 + @as(u128, intermediate) * 6 + (@as(u128, heads) + kv_heads) * width * 4) * 2 + @as(u128, heads) * rows * rows * 8;
}

fn plainActivations(ctx: *const anyopaque, rows: u32) u128 {
    const base: *const Base = @ptrCast(@alignCast(ctx));
    var bytes: u128 = 0;
    for (base.weights.layers) |layer| bytes += layerBytes(@intCast(base.weights.embed.hidden), target(layer.mlp.gate, 0, .gate).output, @intCast(layer.mixer.geo.heads), @intCast(layer.mixer.geo.kv_heads), @intCast(layer.mixer.geo.head_dim), rows);
    return bytes;
}

fn linears(layer: Decoder.Layer) [7]b.linear.Linear {
    return .{ layer.mixer.q, layer.mixer.k, layer.mixer.v, layer.mixer.o, layer.mlp.gate, layer.mlp.up, layer.mlp.down };
}

pub fn validateLinear(linear: b.linear.Linear) !void {
    if (linear.w.ctx == null or ops.shape(linear.w).len != 2 or linear.rht != null or linear.mode != .affine or linear.ternary or linear.hasLora()) return error.TrainingUnsupportedModel;
    if (linear.s.ctx != null) {
        if (linear.bits != 4 or linear.b.ctx == null) return error.TrainingRequiresFourBitOrBfloat16;
    } else if (ops.dtype(linear.w) != .bfloat16 and !(@import("builtin").is_test and ops.dtype(linear.w) == .float32)) return error.TrainingRequiresFourBitOrBfloat16;
}
