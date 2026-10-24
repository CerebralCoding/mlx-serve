//! sdk/blocks/rope/mrope, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns MropeCosSin, mropeContext, endMropeChunk, and 1 more.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const mrope = @import("../../vision/mrope.zig");
const ForwardCtx = @import("../../../engines/mlx/forward_ctx.zig").ForwardCtx;
const yarnMscale = @import("yarn.zig").yarnMscale;

// ── Full Attention for MoE models (with optional output gate) ──

pub const MropeCosSin = struct { cos: mlx.mlx_array, sin: mlx.mlx_array };

pub fn mropeContext(ctx: *const ForwardCtx) mrope.PositionContext {
    return .{ .pos = ctx.mrope_pos.?, .total = ctx.mrope_total, .delta = ctx.mrope_delta };
}

pub fn endMropeChunk(ctx: *ForwardCtx) void {
    if (ctx.mrope_cos_cur) |c| _ = mlx.mlx_array_free(c);
    if (ctx.mrope_sin_cur) |sn| _ = mlx.mlx_array_free(sn);
    ctx.mrope_cos_cur = null;
    ctx.mrope_sin_cur = null;
}

