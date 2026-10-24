//! The legacy `Transformer`'s methods over `sdk/quant/hadamard.zig` (what reads the engine's model state).

const mlx = @import("../../../mlx/ffi.zig");
const rht = @import("../../../sdk/quant/hadamard/rht.zig");
const Transformer = @import("../transformer.zig").Transformer;

/// Hadamard packs: gathered embedding rows back to the residual basis.
/// Takes ownership of `rows`.
pub fn unrotateEmbedding(self: *const Transformer, rows: mlx.mlx_array) !mlx.mlx_array {
    const reg = self.rht orelse return rows;
    const signs = reg.get(self.emb_w) orelse return rows;
    defer _ = mlx.mlx_array_free(rows);
    return rht.transform(rows, signs, reg.block, true, self.s);
}
