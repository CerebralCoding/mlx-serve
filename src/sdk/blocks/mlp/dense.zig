//! sdk/blocks/mlp/dense, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns DenseMlpWeights, SimpleMlpWeights, simpleMLP, residualTail, standardMlp, rowGroupServes, and 1 more.

const mlx = @import("../../../mlx/ffi.zig");
const lane_qmm = @import("../qmm/lane_qmm.zig");
const simd_qmm = @import("../qmm/simd_qmm.zig");
const LayerWeights = @import("../../../engines/mlx/layer_weights.zig").LayerWeights;
const Transformer = @import("../../../engines/mlx/transformer.zig").Transformer;
const naxAvailable = @import("../../../mlx/device.zig").naxAvailable;
const FUSED_ROWS_MAX_M = @import("../linear/fused_rows.zig").FUSED_ROWS_MAX_M;
const FusedRows = @import("../linear/fused_rows.zig").FusedRows;
const fusedResidualNorm = @import("../norms/residual_norm.zig").fusedResidualNorm;
const swigluJoined = @import("fused_activations.zig").swigluJoined;

pub const DenseMlpWeights = struct {
    gate_w: mlx.mlx_array,
    gate_s: mlx.mlx_array,
    gate_b: mlx.mlx_array,
    up_w: mlx.mlx_array,
    up_s: mlx.mlx_array,
    up_b: mlx.mlx_array,
    down_w: mlx.mlx_array,
    down_s: mlx.mlx_array,
    down_b: mlx.mlx_array,
    /// Row-joined [gate | up] for decode width; gate/up are views of it once
    /// built (fuseRowGroup).
    gu: FusedRows = .{},
};

pub const SimpleMlpWeights = struct {
    up_w: mlx.mlx_array,
    up_s: mlx.mlx_array,
    up_b: mlx.mlx_array,
    down_w: mlx.mlx_array,
    down_s: mlx.mlx_array,
    down_b: mlx.mlx_array,
};

// ── Simple (ungated) MLP with ReLU^2 (Nemotron-H) ──

pub fn simpleMLP(self: *Transformer, x: mlx.mlx_array, sw: *const SimpleMlpWeights) !mlx.mlx_array {
    const up = try self.qmatmul(x, sw.up_w, sw.up_s, sw.up_b);
    defer _ = mlx.mlx_array_free(up);
    const activated = try self.reluSquared(up);
    defer _ = mlx.mlx_array_free(activated);
    return self.qmatmul(activated, sw.down_w, sw.down_s, sw.down_b);
}

// ── Dense MLP (SwiGLU: SiLU(gate(x)) * up(x) -> down) ──

/// The residual seam: `h = (h + [rms(b, w_pre)]) [* scale]`, then
/// `rms(h, w_post)` if asked (null-ctx otherwise). One dispatch when
/// `fused` and the kernel takes the shape (fusedResidualNorm), else the
/// composed ops. `pre_is_post_eps` picks the post-norm eps for w_pre.
pub fn residualTail(self: *const Transformer, h: *mlx.mlx_array, b: mlx.mlx_array, w_pre: mlx.mlx_array, pre_is_post_eps: bool, scale: mlx.mlx_array, w_post: mlx.mlx_array, fused: bool) !mlx.mlx_array {
    const eps_pre = if (pre_is_post_eps) self.post_eps_arr else self.rms_eps_arr;
    if (fused and self.config.norm_groups <= 1) {
        if (try fusedResidualNorm(self.s, h.*, b, .{ .w1 = w_pre, .eps1 = eps_pre, .scale = scale, .w_post = w_post, .eps = self.rms_eps_arr })) |r| {
            _ = mlx.mlx_array_free(h.*);
            h.* = r.sum;
            return r.normed;
        }
    }
    var nb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(nb);
    if (w_pre.ctx != null) {
        nb = try self.rmsNormEps(b, w_pre, if (pre_is_post_eps) self.config.postNormEps() else self.config.rms_norm_eps);
    } else {
        _ = mlx.mlx_array_set(&nb, b);
    }
    var sum = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&sum, h.*, nb, self.s));
    _ = mlx.mlx_array_free(h.*);
    h.* = sum;
    if (scale.ctx != null) {
        var scaled = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&scaled, h.*, scale, self.s));
        _ = mlx.mlx_array_free(h.*);
        h.* = scaled;
    }
    if (w_post.ctx == null) return .{ .ctx = null };
    return self.rmsNorm(h.*, w_post);
}

/// gate/up (one row-fused matmul when `fused_rows`) -> activation -> down.
pub fn standardMlp(self: *const Transformer, x: mlx.mlx_array, lw: *const LayerWeights, fused_rows: bool) !mlx.mlx_array {
    var gate_raw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gate_raw);
    var up = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(up);
    if (fused_rows and lw.gu.w.ctx != null) {
        const both = try self.qmatmul(x, lw.gu.w, lw.gu.s, lw.gu.b);
        defer _ = mlx.mlx_array_free(both);
        try lw.gu.part(&gate_raw, both, 0, self.s);
        try lw.gu.part(&up, both, 1, self.s);
    } else {
        gate_raw = try self.qmatmul(x, lw.gate_w, lw.gate_s, lw.gate_b);
        up = try self.qmatmul(x, lw.up_w, lw.up_s, lw.up_b);
    }
    const gate_up = try self.computeGeglu(gate_raw, up);
    defer _ = mlx.mlx_array_free(gate_up);
    return self.qmatmul(gate_up, lw.down_w, lw.down_s, lw.down_b);
}

/// A joined row group serves one-row steps, and every width while the
/// exact path runs: serial steps and verify rows then share its shapes.
pub fn rowGroupServes(self: *const Transformer, rows: c_int) bool {
    const exact_max: c_int = if (naxAvailable()) lane_qmm.MAX_ROWS else simd_qmm.MAX_ROWS;
    return rows <= FUSED_ROWS_MAX_M or (self.config.rowExactDecode() and rows <= exact_max);
}

pub fn denseMLP(self: *Transformer, x: mlx.mlx_array, dw: *const DenseMlpWeights) !mlx.mlx_array {
    const xsh = mlx.getShape(x);
    var rows: c_int = 1;
    for (xsh[0 .. xsh.len - 1]) |d| rows *= d;
    var gate = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gate);
    var up = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(up);
    if (dw.gu.w.ctx != null and self.rowGroupServes(rows)) {
        const both = try self.qmatmul(x, dw.gu.w, dw.gu.s, dw.gu.b);
        defer _ = mlx.mlx_array_free(both);
        if (self.config.hidden_act == .silu and dw.gu.widths[0] == dw.gu.widths[1]) {
            if (try swigluJoined(self.s, both)) |act| {
                defer _ = mlx.mlx_array_free(act);
                return self.qmatmul(act, dw.down_w, dw.down_s, dw.down_b);
            }
        }
        try dw.gu.part(&gate, both, 0, self.s);
        try dw.gu.part(&up, both, 1, self.s);
    } else {
        gate = try self.qmatmul(x, dw.gate_w, dw.gate_s, dw.gate_b);
        up = try self.qmatmul(x, dw.up_w, dw.up_s, dw.up_b);
    }
    const activated = try self.computeGeglu(gate, up);
    defer _ = mlx.mlx_array_free(activated);
    return self.qmatmul(activated, dw.down_w, dw.down_s, dw.down_b);
}
