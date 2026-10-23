//! The resident weight source: a pack's safetensors loaded whole into one map by the engine's loader (the pack
//! adaptations ModelOpt and Sushi packs need included), read by the quant formats through `abi.Source`.

const std = @import("std");
const abi = @import("../../abi/quant.zig");
const mlx = @import("../../mlx/ffi.zig");
const model = @import("../../engines/mlx/model.zig");

pub const Weights = model.Weights;

pub const Resident = struct {
    weights: Weights,

    /// Loads `model_dir` as the engine does (`loadModelWeights`).
    pub fn open(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, config: *model.ModelConfig, load_vision: bool) !Resident {
        return .{ .weights = try model.loadModelWeights(io, allocator, model_dir, config, load_vision) };
    }

    /// A map already loaded (a sidecar file, a test); the source owns it from here.
    pub fn adopt(weights: Weights) Resident {
        return .{ .weights = weights };
    }

    pub fn deinit(self: *Resident) void {
        self.weights.deinit();
    }

    /// The formats' view; borrows `self`, which must not move while it is in use.
    pub fn source(self: *const Resident) abi.Source {
        return .{ .ctx = @constCast(self), .get = get };
    }

    fn get(ctx: ?*anyopaque, name: abi.Str) callconv(.c) abi.Array {
        const self: *const Resident = @ptrCast(@alignCast(ctx.?));
        const a = self.weights.get(name.slice()) orelse return .{};
        return .{ .ctx = a.ctx };
    }
};

// ── tests ──

const testing = std.testing;

test "resident: the source hands out the map's own arrays and an empty one for a missing name" {
    var w = Weights.init(testing.allocator);
    const a = mlx.mlx_array_new_float(2.0);
    try w.put("x.weight", a);
    var r = Resident.adopt(w);
    defer r.deinit();
    const src = r.source();
    try testing.expectEqual(a.ctx, src.tensor("x.weight").ctx);
    try testing.expectEqual(@as(?*anyopaque, null), src.tensor("x.scales").ctx);
}
