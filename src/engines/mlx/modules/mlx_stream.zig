//! engines/mlx/modules/mlx_stream, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns forwardMlxStream, initMlxStream.

const std = @import("std");
const mlx_stream_mod = if (@import("build_options").mlx_stream) @import("../../../sdk/weights/streamed/glue.zig") else @import("../../../sdk/weights/streamed/stub.zig");
const mlx = @import("../../../mlx/ffi.zig");
const ModelConfig = @import("../model.zig").ModelConfig;
const hostIds = @import("../../../mlx/array.zig").hostIds;
const ForwardCtx = @import("../forward_ctx.zig").ForwardCtx;
const Transformer = @import("../transformer.zig").Transformer;
const moduleShell = @import("shell.zig").moduleShell;

/// An mlx-stream pack: the plugin runs the request's prompt pass, then decode (`mlx_stream.forward`).
pub fn forwardMlxStream(self: *Transformer, ctx: *ForwardCtx, token_ids: mlx.mlx_array, m: *mlx_stream_mod.Model) !mlx.mlx_array {
    const ids = try hostIds(self, token_ids);
    defer self.allocator.free(ids);
    const logits = try mlx_stream_mod.forward(m, ids, self.s);
    ctx.cache.step += ids.len;
    return logits;
}

/// A pack mlx-stream serves (DeepSeek-V4.1's EXL3 repack, a MiMo stream pack or a GLM-5.3 pack): the plugin's module.
pub fn initMlxStream(io: std.Io, allocator: std.mem.Allocator, config: ModelConfig, s: mlx.mlx_stream) !Transformer {
    const m = try mlx_stream_mod.open(allocator, io, s, &config);
    errdefer mlx_stream_mod.close(m);
    var t = try moduleShell(allocator, config, s);
    t.mlx_stream_ext = m;
    return t;
}
