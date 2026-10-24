//! The streamed half of MiMo's expert parity (`plugins/mimo_v2/streamed_test.zig`, main test module): the stream's
//! served decode MoE (`moe.zig`'s k1 texts over the slot banks, then its f32 combine) on the slots one fetch names.

const std = @import("std");
const builtin = @import("builtin");
const mlx = @import("sdk").mlx;
const moe = @import("moe.zig");
const streamed = @import("../../../sdk/weights/streamed/bank_source.zig");

comptime {
    if (!builtin.is_test) @compileError("mimo_v2 dupes.zig is a parity harness; served code never imports it");
}

const T = mlx.mlx_array;

/// The decode MoE's routed sum `[n, H]` bf16 for `x` bf16 `[n, H]`, the routes `ids` (`n * top_k`, router order) and
/// their f32 `weights` `[n, top_k]`, model layer `layer`'s slots fetched from `src`; owned by `src.g` (read it before
/// `src.g.reset()`).
pub fn decodeSum(src: *streamed.Mxfp4, layer: u32, x: T, ids: []const u16, weights: T) !T {
    const g = &src.g;
    const p = ids.len;
    const n = p / moe.top_k;
    const f = try src.fetch(layer, ids);
    var first: ?streamed.Banks = null;
    for (f.banks) |b| if (b) |x_| {
        first = x_;
        break;
    };
    var banks: [moe.n_banks]moe.Arrays = undefined;
    var widest: u32 = 0;
    for (f.banks, &banks) |b, *o| {
        const bk = b orelse first.?;
        o.* = .{ .gate = .{ .w = bk.gate.w, .scales = bk.gate.scales }, .up = .{ .w = bk.up.w, .scales = bk.up.scales }, .down = .{ .w = bk.down.w, .scales = bk.down.scales } };
        widest = @max(widest, @as(u32, @intCast(mlx.getShape(bk.gate.w)[0])));
    }
    var packed_ids: [moe.max_pairs]u32 = undefined;
    var tok: [moe.max_pairs]u32 = undefined;
    for (0..p) |i| {
        packed_ids[i] = @as(u32, f.kind[i]) << moe.row_bits | f.row[i];
        tok[i] = @intCast(i / moe.top_k);
    }
    const id_arr = try g.adopt(mlx.mlx_array_new_data(&packed_ids, &[_]c_int{@intCast(p)}, 1, .uint32));
    const tok_arr = try g.adopt(mlx.mlx_array_new_data(&tok, &[_]c_int{@intCast(p)}, 1, .uint32));
    var xp = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_take_axis(&xp, x, tok_arr, 0, g.s));
    const x_pairs = try g.adopt(xp);
    var fused = try moe.Fused.initFamily(src.bank.geometry, widest, .k1);
    defer fused.deinit();
    const h = try fused.gateUp(g, x_pairs, id_arr, &banks);
    const d = try fused.down(g, h, id_arr, &banks);
    return fused.combineRows(g, try g.reshape(d, &.{ @intCast(n), @intCast(moe.top_k), @intCast(moe.hidden) }), weights);
}
