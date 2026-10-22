//! plugins/deepseek_v41/bridge, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns forwardDsv41WithImpl, initDsv41.

const std = @import("std");
const dsv41_mod = @import("deepseek_v41.zig");
const mlx = @import("../../mlx/ffi.zig");
const ModelConfig = @import("../../engines/mlx/model.zig").ModelConfig;
const Weights = @import("../../engines/mlx/model.zig").Weights;
const ForwardCtx = @import("../../engines/mlx/forward_ctx.zig").ForwardCtx;
const initMlxStream = @import("../../engines/mlx/modules/mlx_stream.zig").initMlxStream;
const moduleShell = @import("../../engines/mlx/modules/shell.zig").moduleShell;
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;
const hostIds = @import("../../mlx/array.zig").hostIds;

/// DeepSeek-V4.1: a fresh request (step 0) rebuilds the module's state; a
/// resumed one (`resumePrompt` set step) and every later call, prefill chunk
/// or decode token alike, extends it. Returns the last position's logits `[1, 1, V]`.
pub fn forwardDsv41WithImpl(self: *Transformer, ctx: *ForwardCtx, token_ids: mlx.mlx_array, mdl: *dsv41_mod.Dsv41Model) !mlx.mlx_array {
    const n = mlx.mlx_array_size(token_ids);
    // A decode token stays on the GPU: the module reads it back once the layers before Engram are queued.
    const logits = if (ctx.cache.step > 0 and mdl.dec_state != null and n == 1 and mdl.pair == null)
        try dsv41_mod.extendResumableGpu(mdl, self.allocator, token_ids)
    else blk: {
        const ids = try hostIds(self, token_ids);
        defer self.allocator.free(ids);
        // The module splices the image rows itself (its embedding never leaves it).
        const vision: ?dsv41_mod.VisionRows = if (ctx.vision_embeddings) |ve| .{ .rows = ve, .offset = ctx.vision_splice_offset } else null;
        break :blk if (ctx.cache.step == 0 or mdl.dec_state == null)
            try dsv41_mod.prefill(mdl, self.allocator, ids, vision)
        else
            try dsv41_mod.extendResumable(mdl, self.allocator, ids, vision);
    };
    defer _ = mlx.mlx_array_free(logits);
    ctx.cache.step += n;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, logits, &[_]c_int{ 1, 1, @intCast(mdl.vocab) }, 3, self.s));
    return out;
}

/// DeepSeek-V4.1: its pack's Engram tables sit beside the weights; the
/// DSpark stages load as `deepseek_v41.dsparkWanted` decides.
pub fn initDsv41(io: std.Io, allocator: std.mem.Allocator, config: ModelConfig, weights: *Weights, s: mlx.mlx_stream) !Transformer {
    if (config.dsv41_stream) return initMlxStream(io, allocator, config, s);
    const mdl = try dsv41_mod.init(allocator, &config, weights, s, .{ .dspark = dsv41_mod.dsparkWanted(&config), .pair = config.dist_dsv41, .head_8bit = dsv41_mod.head8Wanted() });
    errdefer mdl.deinit();
    var t = try moduleShell(allocator, config, s);
    t.dsv41 = mdl;
    return t;
}
