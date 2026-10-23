//! plugins/gemma4/moe_tail, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns gemma4MoeLayerTail, trace_tail_once.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const debugTraceHead = @import("../../engines/mlx/diag/profile.zig").debugTraceHead;
const MoeLayerWeights = @import("../../engines/mlx/forward/layered.zig").MoeLayerWeights;
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;
const fusedResidualNorm = @import("../../sdk/blocks/norms/residual_norm.zig").fusedResidualNorm;

/// Gemma 4 / DiffusionGemma layer tail (from the HF reference):
///   h = residual + post_attn_norm(attn_out)          [attn_out borrowed]
///   residual = h
///   shared  = post_ff_norm_1(mlp(pre_ff_norm(h)))
///   experts = post_ff_norm_2(moe(pre_ff_norm_2(residual)))  [router sees RAW residual]
///   h = residual + post_ff_norm(shared + experts)
///   h *= layer_scalar (encoder variant when requested and bound)
/// Consumes `h_in`; returns the new layer output (caller owns).
///
/// With `fused` (decode width) each residual seam is one dispatch
/// (fusedResidualNorm): the attention seam also yields both feed-forward
/// input norms, and the feed-forward seam folds the two post-norms, the
/// combine norm, the scalar and `next_norm` — handed back through
/// `carried` so the next layer skips its input norm.
pub fn gemma4MoeLayerTail(self: *Transformer, h_in: mlx.mlx_array, attn_out: mlx.mlx_array, lw: *const MoeLayerWeights, next_norm: mlx.mlx_array, carried: *?mlx.mlx_array, fused: bool) !mlx.mlx_array {
    var h = h_in;
    const none = mlx.mlx_array{ .ctx = null };
    const eps = self.rms_eps_arr;

    // Attention residual, then the two feed-forward input norms.
    var shared_in = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(shared_in);
    var expert_in = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(expert_in);
    const seam1 = if (fused) try fusedResidualNorm(self.s, h, attn_out, .{ .w1 = lw.post_attn_norm, .eps1 = eps, .w_post = lw.pre_ff_norm.?, .w_post2 = lw.pre_ff_norm_2.?, .eps = eps }) else null;
    if (seam1) |r| {
        _ = mlx.mlx_array_free(h);
        h = r.sum;
        shared_in = r.normed;
        expert_in = r.normed2;
    } else {
        const attn_normed = try self.rmsNorm(attn_out, lw.post_attn_norm);
        defer _ = mlx.mlx_array_free(attn_normed);
        var h_new = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&h_new, h, attn_normed, self.s));
        _ = mlx.mlx_array_free(h);
        h = h_new;
        shared_in = try self.rmsNorm(h, lw.pre_ff_norm.?);
        expert_in = try self.rmsNorm(h, lw.pre_ff_norm_2.?);
    }

    // Shared expert: pre_ff_norm → mlp → post_ff_norm_1
    const shared_out = if (lw.shared_mlp) |smlp|
        try self.denseMLP(shared_in, &smlp)
    else
        try self.denseMLP(shared_in, &(switch (lw.mlp) {
            .dense => |dw| dw,
            .moe => unreachable,
        }));
    defer _ = mlx.mlx_array_free(shared_out);

    // Routed experts: router gets raw residual, experts get pre_ff_norm_2(residual)
    const expert_out = switch (lw.mlp) {
        .moe => |*mw| try self.moeMLP2(h, expert_in, mw),
        .dense => |*dw| try self.denseMLP(expert_in, dw),
    };
    defer _ = mlx.mlx_array_free(expert_out);

    const scalar = lw.layer_scalar;

    const seam2 = if (fused) try fusedResidualNorm(self.s, h, shared_out, .{
        .b2 = expert_out,
        .w1 = lw.post_ff_norm_1.?,
        .eps1 = eps,
        .w2 = lw.post_ff_norm_2.?,
        .eps2 = eps,
        .wc = lw.post_ff_norm.?,
        .epsc = eps,
        .scale = scalar orelse none,
        .w_post = next_norm,
        .eps = eps,
    }) else null;
    if (seam2) |r| {
        _ = mlx.mlx_array_free(h);
        if (r.normed.ctx != null) carried.* = r.normed;
        return r.sum;
    }

    const shared_normed = try self.rmsNorm(shared_out, lw.post_ff_norm_1.?);
    defer _ = mlx.mlx_array_free(shared_normed);
    const expert_normed = try self.rmsNorm(expert_out, lw.post_ff_norm_2.?);
    defer _ = mlx.mlx_array_free(expert_normed);

    if (std.c.getenv("MLX_SERVE_DIFFUSION_TRACE") != null and trace_tail_once) {
        trace_tail_once = false;
        debugTraceHead("tail residual", h, self.s);
        debugTraceHead("tail shared_normed", shared_normed, self.s);
        debugTraceHead("tail expert_normed", expert_normed, self.s);
    }

    // Combine: shared + experts → post_ff_norm → residual add
    var combined = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(combined);
    try mlx.check(mlx.mlx_add(&combined, shared_normed, expert_normed, self.s));
    const combined_normed = try self.rmsNorm(combined, lw.post_ff_norm.?);
    defer _ = mlx.mlx_array_free(combined_normed);
    var h_ff = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&h_ff, h, combined_normed, self.s));
    _ = mlx.mlx_array_free(h);
    h = h_ff;

    if (scalar) |ls| {
        var h_scaled = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&h_scaled, h, ls, self.s));
        _ = mlx.mlx_array_free(h);
        h = h_scaled;
    }
    return h;
}

// ── DiffusionGemma bidirectional canvas decoder ──

/// MLX_SERVE_DIFFUSION_TRACE=1 debugging aid: print the first 8 values
/// of position 0 of a hidden state.
pub var trace_tail_once: bool = true;
