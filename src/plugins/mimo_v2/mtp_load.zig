//! Loads the MiMo-V2 MTP heads at model load.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const transformer_mod = @import("../../engines/mlx/transformer.zig");
const mimo_mtp = @import("mtp.zig");
const model_mod = @import("../../engines/mlx/model.zig");
const log = @import("../../host/log.zig");
const Transformer = transformer_mod.Transformer;

/// The MiMo-V2 MTP heads in `weights`, with their coarse draft readout bound; null when the pack ships none.
pub fn loadMimoHead(allocator: std.mem.Allocator, config: *const model_mod.ModelConfig, weights: *const model_mod.Weights, xfm: *Transformer) !?*mimo_mtp.Head {
    var loaded = (try mimo_mtp.Head.load(allocator, mlx.gpuStream(), config, weights)) orelse return null;
    errdefer loaded.deinit();
    const h = try allocator.create(mimo_mtp.Head);
    h.* = loaded;
    h.bindRerank(xfm);
    log.info("MiMo MTP heads ready ({d} heads, draft rerank {s}).\n", .{ h.heads, if (h.canRerankDrafts()) "on" else "off" });
    return h;
}
