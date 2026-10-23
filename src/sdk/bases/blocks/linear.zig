//! copied_from: sdk/blocks/linear/qmatmul.zig (`qmatmul`, `qmatmulMaybeBias` without the rotated, ternary and
//! row-exact arms). A projection over its own quant geometry, and a row-joined group of them.

const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const QuantMode = @import("../../../engines/mlx/model.zig").QuantMode;
const qmatmulBits = @import("../../blocks/linear/qmatmul.zig").qmatmulBits;
const sliceQkvPart = @import("../../blocks/linear/fused_rows.zig").sliceQkvPart;

const Array = ops.Array;
const Graph = ops.Graph;

/// `x @ w^T` (+ an additive `bias`). A dense `w` is stored pre-transposed `[in, out]` with null scales.
pub const Linear = struct {
    w: Array = .{},
    s: Array = .{},
    b: Array = .{},
    /// The additive projection bias (Qwen2's q/k/v, BERT); null = none.
    bias: Array = .{},
    bits: u32 = 0,
    group_size: u32 = 0,
    mode: QuantMode = .affine,

    pub fn forward(l: *const Linear, g: *Graph, x: Array) !Array {
        if (l.bias.ctx == null) return g.adopt(try qmatmulBits(x, l.w, l.s, l.b, l.bits, l.group_size, l.mode, g.s));
        var t = g.scope();
        defer t.deinit();
        return g.keep(try t.add(try t.adopt(try qmatmulBits(x, l.w, l.s, l.b, l.bits, l.group_size, l.mode, g.s)), l.bias));
    }
};

/// Projections that read one input joined along their output rows (one matmul at decode width); null `lin.w` =
/// not built.
pub const Joined = struct {
    lin: Linear = .{},
    widths: [4]c_int = @splat(0),

    pub fn built(j: *const Joined) bool {
        return j.lin.w.ctx != null;
    }

    /// Columns of part `i` of a joined matmul result `y`.
    pub fn part(j: *const Joined, g: *Graph, y: Array, i: usize) !Array {
        var start: c_int = 0;
        for (j.widths[0..i]) |w| start += w;
        var out = mlx.mlx_array_new();
        sliceQkvPart(&out, y, start, j.widths[i], g.s) catch |e| {
            _ = mlx.mlx_array_free(out);
            return e;
        };
        return g.adopt(out);
    }
};
