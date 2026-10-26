//! A frozen dense decoder borrowed from a loaded llama or qwen3 model.
const std = @import("std");
const ops = @import("../../mlx/ops.zig");
const bases = @import("../bases/bases.zig");
const b = bases.blocks;
const adapter = @import("adapter.zig");
const abi = @import("../../abi/plugin.zig");
pub const Decoder = bases.Decoder(.{ .embed = b.embed.TableEmbed, .norm = b.norm.RmsNorm, .mixer = b.attention.Attention, .mlp = b.mlp.SwiGlu, .seam = b.seam.ResidualTail, .head = b.head.LmHead });
pub const Base = bases.Model(Decoder);

pub const Frozen = struct {
    base: *const Base,

    pub fn init(base: *const Base) !Frozen {
        if (base.weights.layers.len == 0) return error.TrainingUnsupportedModel;
        try validateLinear(base.weights.embed.table);
        try validateLinear(base.weights.head);
        for (base.weights.layers) |layer| for (linears(layer)) |linear| try validateLinear(linear);
        return .{ .base = base };
    }
    pub fn targets(self: Frozen, a: std.mem.Allocator) ![]adapter.Target {
        const result = try a.alloc(adapter.Target, self.base.weights.layers.len * 7);
        for (self.base.weights.layers, 0..) |layer, index| {
            for (linears(layer), 0..) |linear, projection| {
                const sh = ops.shape(linear.w);
                const quantized = linear.s.ctx != null;
                result[index * 7 + projection] = .{ .layer = @intCast(index), .projection = @fromBackingInt(@intCast(projection)), .input = @intCast(if (quantized) sh[1] * 8 else sh[0]), .output = @intCast(if (quantized) sh[0] else sh[1]) };
            }
        }
        return result;
    }
    pub fn hidden(self: Frozen, g: *ops.Graph, ids: ops.Array, bindings: []const abi.LoraBinding) !ops.Array {
        const shape = ops.shape(ids);
        if (shape.len != 2 or shape[0] != 1 or shape[1] < 1 or shape[1] > 512) return error.InvalidTrainingShape;
        const previous = g.differentiable;
        g.differentiable = true;
        defer g.differentiable = previous;
        const result = try Decoder.forward(&self.base.weights, &self.base.shared, g, ids, .{ .mode = .prefill, .rows = shape[1], .positions = &.{0}, .logits = .none, .want_hidden = true, .lora = bindings });
        return result.hidden;
    }
};

fn linears(layer: Decoder.Layer) [7]b.linear.Linear {
    return .{ layer.mixer.q, layer.mixer.k, layer.mixer.v, layer.mixer.o, layer.mlp.gate, layer.mlp.up, layer.mlp.down };
}

fn validateLinear(linear: b.linear.Linear) !void {
    if (linear.w.ctx == null or ops.shape(linear.w).len != 2 or linear.rht != null or linear.mode != .affine or linear.ternary or linear.hasLora()) return error.TrainingUnsupportedModel;
    if (linear.s.ctx != null) {
        if (linear.bits != 4 or linear.b.ctx == null) return error.TrainingRequiresFourBitOrBfloat16;
    } else if (ops.dtype(linear.w) != .bfloat16 and !(@import("builtin").is_test and ops.dtype(linear.w) == .float32)) return error.TrainingRequiresFourBitOrBfloat16;
}
