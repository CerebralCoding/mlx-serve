//! Loads the GLM-5.3-Flash MTP layer at model load.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const transformer_mod = @import("../../engines/mlx/transformer.zig");
const glm_mtp = @import("mtp.zig");
const model_mod = @import("../../engines/mlx/model.zig");
const log = @import("../../sdk/util/log.zig");
const Transformer = transformer_mod.Transformer;

/// The GLM-5.3-Flash MTP layer in `weights`, with its coarse draft readout bound; null when the pack ships none.
pub fn loadGlmHead(allocator: std.mem.Allocator, config: *const model_mod.ModelConfig, weights: *const model_mod.Weights, xfm: *Transformer) !?*glm_mtp.Head {
    var loaded = (try glm_mtp.Head.load(allocator, mlx.gpuStream(), config, weights)) orelse return null;
    errdefer loaded.deinit();
    const h = try allocator.create(glm_mtp.Head);
    h.* = loaded;
    h.bindRerank(xfm);
    log.info("GLM MTP head ready (draft rerank {s}).\n", .{if (h.canRerankDrafts()) "on" else "off"});
    return h;
}
