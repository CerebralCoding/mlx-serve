//! sdk/blocks/moe/decode, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns glmMoeDecode, batchedExpertMm, moeDecodeGatherQmv, moeFp4Decode, moeAffine4Decode, moeCudaDecode, and 9 more.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const model_mod = @import("../../../engines/mlx/model.zig");
const log = @import("../../../host/log.zig");
const ModelConfig = @import("../../../engines/mlx/model.zig").ModelConfig;
const QuantMode = @import("../../../engines/mlx/model.zig").QuantMode;
const moe_fp4 = @import("moe_fp4.zig");
const moe_affine4 = @import("../../../plugins/qwen4_exp/moe_affine4.zig");
const glm5 = @import("../../../plugins/glm5_next/glm5_next.zig");
const down_reduce = @import("down_reduce.zig");
const gather_qmv = @import("gather_qmv.zig");
const gather_qmv_rows = @import("gather_qmv_rows.zig");
const envFlagCached = @import("../../../engines/mlx/env_transformer.zig").envFlagCached;
const testing = @import("../../../engines/mlx/kv/cache.zig").testing;
const lastDim = @import("../../../engines/mlx/load/weights.zig").lastDim;
const Transformer = @import("../../../engines/mlx/transformer.zig").Transformer;
const qwen4Standin = @import("../../diag/standin.zig").qwen4Standin;
const standinOnes = @import("../../diag/standin.zig").standinOnes;
const QuantParams = @import("../../quant/params.zig").QuantParams;
const swigluClampLimit = @import("../mlp/activations.zig").swigluClampLimit;
const swigluSigTable = @import("../mlp/fused_activations.zig").swigluSigTable;
const gatherQmvDownReduce = @import("down_reduce.zig").gatherQmvDownReduce;
const gatherQmvDownReduceRows = @import("down_reduce.zig").gatherQmvDownReduceRows;
const gatherQmvDownReduceRowsEligible = @import("down_reduce.zig").gatherQmvDownReduceRowsEligible;
const gatherQmv = @import("gather_qmv.zig").gatherQmv;
const gatherQmvGateUp = @import("gather_qmv.zig").gatherQmvGateUp;
const gatherQmvGateUpEnabled = @import("gather_qmv.zig").gatherQmvGateUpEnabled;
const gatherQmvGateUpRows = @import("gather_qmv_rows.zig").gatherQmvGateUpRows;
const MoeMlpWeights = @import("weights.zig").MoeMlpWeights;

/// GLM-5-Next one-token MoE (`glm5.moeDecode`): routed and shared experts in two dispatches.
pub fn glmMoeDecode(self: *Transformer, x: mlx.mlx_array, inds: mlx.mlx_array, scores: mlx.mlx_array, mw: *const MoeMlpWeights, gate_qp: QuantParams, up_qp: QuantParams, down_qp: QuantParams) !?mlx.mlx_array {
    const inter: ?u32 = if (self.config.shared_expert_intermediate_size > 0) self.config.shared_expert_intermediate_size else null;
    const sg = self.quantParamsHinted(mw.shared_gate_w, mw.shared_gate_s, lastDim(x));
    const su = self.quantParamsHinted(mw.shared_up_w, mw.shared_up_s, lastDim(x));
    const sd = self.quantParamsHinted(mw.shared_down_w, mw.shared_down_s, inter);
    inline for (.{ gate_qp, up_qp, down_qp, sg, su, sd }) |qp| if (qp.mode != .affine) return null;
    const k: c_int = @intCast(mlx.mlx_array_size(inds));
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, inds, &[_]c_int{k}, 1, self.s));
    var ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids);
    try mlx.check(mlx.mlx_astype(&ids, flat, .uint32, self.s));
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_reshape(&sc, scores, &[_]c_int{k}, 1, self.s));
    const y = (try glm5.moeDecode(self.s, x, ids, sc, //
        .{ .w = mw.switch_gate_w, .s = mw.switch_gate_s, .b = mw.switch_gate_b, .bits = gate_qp.bits, .gs = gate_qp.group_size }, //
        .{ .w = mw.switch_up_w, .s = mw.switch_up_s, .b = mw.switch_up_b, .bits = up_qp.bits, .gs = up_qp.group_size }, //
        .{ .w = mw.switch_down_w, .s = mw.switch_down_s, .b = mw.switch_down_b, .bits = down_qp.bits, .gs = down_qp.group_size }, //
        .{ .w = mw.shared_gate_w, .s = mw.shared_gate_s, .b = mw.shared_gate_b, .bits = sg.bits, .gs = sg.group_size }, //
        .{ .w = mw.shared_up_w, .s = mw.shared_up_s, .b = mw.shared_up_b, .bits = su.bits, .gs = su.group_size }, //
        .{ .w = mw.shared_down_w, .s = mw.shared_down_s, .b = mw.shared_down_b, .bits = sd.bits, .gs = sd.group_size }, //
        self.config.swiglu_limit)) orelse return null;
    defer _ = mlx.mlx_array_free(y);
    const xs = mlx.getShape(x);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, y, xs.ptr, xs.len, self.s));
    return out;
}

/// Decode-width expert projection via take + batched quantized_matmul,
/// dodging our libmlx's serialized gather_qmm. `x_bc` is the [K,1,in] input
/// (per-expert broadcast); `inds_flat` is the [K] selected-expert index
/// vector. Returns [K,1,out]. Caller owns the result.
pub fn batchedExpertMm(self: *Transformer, x_bc: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, inds_flat: mlx.mlx_array, bits: u32, group_size: u32, mode: QuantMode) !mlx.mlx_array {
    var w_k = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_k);
    try mlx.check(mlx.mlx_take_axis(&w_k, w, inds_flat, 0, self.s));
    var sc_k = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc_k);
    try mlx.check(mlx.mlx_take_axis(&sc_k, sc, inds_flat, 0, self.s));
    var bi_k = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bi_k);
    var bi_arg = mlx.mlx_array{ .ctx = null };
    if (bi.ctx != null) {
        try mlx.check(mlx.mlx_take_axis(&bi_k, bi, inds_flat, 0, self.s));
        bi_arg = bi_k;
    }
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_quantized_matmul(&out, x_bc, w_k, sc_k, bi_arg, true, mlx.mlx_optional_int.some(@intCast(group_size)), mlx.mlx_optional_int.some(@intCast(bits)), mode.cstr(), self.s));
    return out; // [K, 1, out]
}

/// Decode-width MoE expert compute through the in-place gather-qmv kernel
/// (`gatherQmv`), which indexes the expert bank directly instead of
/// materializing the top-K experts. On success writes [B,S,K,hidden] into
/// `out` and returns true; returns false when any projection is outside the
/// kernel's supported set (mxfp4/mxfp8 bank, 3/5/6-bit affine width, odd
/// geometry), leaving `out` untouched so the caller runs the batched take path.
pub fn moeDecodeGatherQmv(
    self: *Transformer,
    out: *mlx.mlx_array,
    expert_x: mlx.mlx_array,
    inds: mlx.mlx_array,
    norm_scores: mlx.mlx_array,
    reduced: *bool,
    mw: *const MoeMlpWeights,
    gate_qp: anytype,
    up_qp: anytype,
    down_qp: anytype,
    D: c_int,
    K: c_int,
    B: c_int,
    S: c_int,
) !bool {
    // Routing indices arrive as int32/uint32 [.., K]; the kernel reads a
    // flat uint32 [K].
    var inds_u32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(inds_u32);
    {
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        const kshape = [_]c_int{K};
        try mlx.check(mlx.mlx_reshape(&flat, inds, &kshape, 1, self.s));
        try mlx.check(mlx.mlx_astype(&inds_u32, flat, .uint32, self.s));
    }
    var x_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x_flat);
    const dshape = [_]c_int{D};
    try mlx.check(mlx.mlx_reshape(&x_flat, expert_x, &dshape, 1, self.s));

    // gate / up share the single token's hidden state, so one kernel can
    // do both dot products and the activation — three dispatches off the
    // chain into down_proj. Falls back to the split form on any geometry,
    // quant mode or dtype the fused kernel declines.
    var act_2d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(act_2d);
    var inter: c_int = 0;
    if (qwen4Standin().moe_gateup and self.config.moe_intermediate_size > 0) {
        // Diagnostic: unit activations, the gate+up kernel drops out.
        const sh = [_]c_int{ K, @intCast(self.config.moe_intermediate_size) };
        _ = mlx.mlx_array_free(act_2d);
        act_2d = try standinOnes(&sh, self.s);
        inter = sh[1];
    } else if (self.config.hidden_act == .silu and
        gate_qp.bits == up_qp.bits and gate_qp.group_size == up_qp.group_size and gate_qp.mode == up_qp.mode)
    {
        if (try gatherQmvGateUp(
            self.s,
            x_flat,
            mw.switch_gate_w,
            mw.switch_gate_s,
            mw.switch_gate_b,
            mw.switch_up_w,
            mw.switch_up_s,
            mw.switch_up_b,
            inds_u32,
            gate_qp.bits,
            gate_qp.group_size,
            gate_qp.mode,
            swigluClampLimit(&self.config).?,
        )) |fused| {
            _ = mlx.mlx_array_free(act_2d);
            act_2d = fused; // [K, inter], activation already applied
            inter = mlx.getShape(act_2d)[1];
        }
    }
    if (inter == 0) {
        const gate_2d = try gatherQmv(self.s, x_flat, mw.switch_gate_w, mw.switch_gate_s, mw.switch_gate_b, inds_u32, gate_qp.bits, gate_qp.group_size, gate_qp.mode, false) orelse return false;
        defer _ = mlx.mlx_array_free(gate_2d);
        const up_2d = try gatherQmv(self.s, x_flat, mw.switch_up_w, mw.switch_up_s, mw.switch_up_b, inds_u32, up_qp.bits, up_qp.group_size, up_qp.mode, false) orelse return false;
        defer _ = mlx.mlx_array_free(up_2d);

        inter = mlx.getShape(gate_2d)[1];
        const k1i = [_]c_int{ K, 1, inter };
        var gate_out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gate_out);
        try mlx.check(mlx.mlx_reshape(&gate_out, gate_2d, &k1i, 3, self.s));
        var up_out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(up_out);
        try mlx.check(mlx.mlx_reshape(&up_out, up_2d, &k1i, 3, self.s));

        const expert_act = try self.computeGeglu(gate_out, up_out); // [K,1,inter]
        defer _ = mlx.mlx_array_free(expert_act);

        // down: each expert consumes ITS OWN activation → per-expert x layout.
        const ki = [_]c_int{ K, inter };
        _ = mlx.mlx_array_free(act_2d);
        act_2d = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&act_2d, expert_act, &ki, 2, self.s));
    }
    // Fused down + router weighting + K-reduction: one dispatch replaces
    // the down gather, the score multiply and the sum over K — and the
    // [K, hidden] intermediate never materializes. Declines (dtype
    // mismatch, geometry, kill switch) fall through to the split form
    // with `reduced` untouched.
    if (qwen4Standin().moe_down and @rem(D, inter) == 0) {
        // Diagnostic: the down kernel drops out; the activations are still
        // consumed (sum over K, tiled to hidden) so gate+up stays live.
        var summed = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(summed);
        try mlx.check(mlx.mlx_sum_axis(&summed, act_2d, 0, false, self.s));
        const reps = [_]c_int{@divExact(D, inter)};
        var tiled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(tiled);
        try mlx.check(mlx.mlx_tile(&tiled, summed, &reps, 1, self.s));
        const bsh_shape = [_]c_int{ B, S, D };
        try mlx.check(mlx.mlx_reshape(out, tiled, &bsh_shape, 3, self.s));
        reduced.* = true;
        return true;
    }
    {
        var scores_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(scores_flat);
        const kshape2 = [_]c_int{K};
        try mlx.check(mlx.mlx_reshape(&scores_flat, norm_scores, &kshape2, 1, self.s));
        if (try gatherQmvDownReduce(self.s, act_2d, mw.switch_down_w, mw.switch_down_s, mw.switch_down_b, inds_u32, scores_flat, down_qp.bits, down_qp.group_size, down_qp.mode)) |sum_flat| {
            defer _ = mlx.mlx_array_free(sum_flat);
            const hidden = mlx.getShape(sum_flat)[0];
            const bsh_shape = [_]c_int{ B, S, hidden };
            try mlx.check(mlx.mlx_reshape(out, sum_flat, &bsh_shape, 3, self.s));
            reduced.* = true;
            if (!down_reduce.downred_engaged) {
                down_reduce.downred_engaged = true;
                log.info("[moe] fused down+reduce kernel engaged: E_topk={d} inter={d} hidden={d} bits={d} gs={d}\n", .{ K, inter, hidden, down_qp.bits, down_qp.group_size });
            }
            if (!gather_qmv.gqmv_engaged) {
                gather_qmv.gqmv_engaged = true;
                log.info("[moe] gather-qmv kernel engaged: E_topk={d} inter={d} hidden={d} bits={d} gs={d}\n", .{ K, inter, hidden, gate_qp.bits, gate_qp.group_size });
            }
            return true;
        }
    }
    const down_2d = try gatherQmv(self.s, act_2d, mw.switch_down_w, mw.switch_down_s, mw.switch_down_b, inds_u32, down_qp.bits, down_qp.group_size, down_qp.mode, true) orelse return false;
    defer _ = mlx.mlx_array_free(down_2d);

    const hidden = mlx.getShape(down_2d)[1];
    const bskh_shape = [_]c_int{ B, S, K, hidden };
    try mlx.check(mlx.mlx_reshape(out, down_2d, &bskh_shape, 4, self.s));
    // Engagement is COUNTED, never inferred: a silent fallback to the
    // batched take path is output-identical and would look like a null win.
    if (!gather_qmv.gqmv_engaged) {
        gather_qmv.gqmv_engaged = true;
        log.info("[moe] gather-qmv kernel engaged: E_topk={d} inter={d} hidden={d} bits={d} gs={d}\n", .{ K, inter, hidden, gate_qp.bits, gate_qp.group_size });
    }
    return true;
}

/// fp4 banks at decode and verify widths: `moe_fp4`'s two dispatches over
/// every token's own experts. Writes the weighted sum [B,S,D] into `out`;
/// false when the kernels decline.
pub fn moeFp4Decode(self: *Transformer, out: *mlx.mlx_array, expert_x: mlx.mlx_array, inds: mlx.mlx_array, norm_scores: mlx.mlx_array, mw: *const MoeMlpWeights, mode: QuantMode, group_size: u32, shared: ?moe_affine4.Shared) !bool {
    const xs = mlx.getShape(expert_x);
    const rows = xs[0] * xs[1];
    const ks = mlx.getShape(inds);
    const k = ks[ks.len - 1];
    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    try mlx.check(mlx.mlx_reshape(&x2, expert_x, &.{ rows, xs[2] }, 2, self.s));
    var ind2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ind2);
    try mlx.check(mlx.mlx_reshape(&ind2, inds, &.{ rows, k }, 2, self.s));
    var ind_u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ind_u);
    try mlx.check(mlx.mlx_astype(&ind_u, ind2, .uint32, self.s));
    var sc2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc2);
    try mlx.check(mlx.mlx_reshape(&sc2, norm_scores, &.{ rows, k }, 2, self.s));
    const sigtab = try swigluSigTable(self.s, mlx.mlx_array_dtype(expert_x), std.heap.c_allocator);
    const Bank = moe_fp4.Bank;
    const scales: moe_fp4.Scales = .{
        .gate = if (mw.gate_expert_scale.ctx != null) mw.gate_expert_scale else null,
        .up = if (mw.up_expert_scale.ctx != null) mw.up_expert_scale else null,
        .clamp = swigluClampLimit(&self.config) orelse return false,
    };
    const y = (try moe_fp4.decode(self.s, x2, Bank{ .w = mw.switch_gate_w, .s = mw.switch_gate_s }, Bank{ .w = mw.switch_up_w, .s = mw.switch_up_s }, Bank{ .w = mw.switch_down_w, .s = mw.switch_down_s }, mw.bank_views, scales, shared, ind_u, sc2, sigtab, mode, group_size)) orelse return false;
    defer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_reshape(out, y, &.{ xs[0], xs[1], xs[2] }, 3, self.s));
    return true;
}

/// Affine-4 banks at ONE decode token: `moe_affine4`'s two dispatches over the token's
/// experts. Writes the weighted sum [1,1,D] into `out` (with the gated shared expert added when
/// `shared` is given); false when the kernels decline.
pub fn moeAffine4Decode(self: *Transformer, out: *mlx.mlx_array, expert_x: mlx.mlx_array, inds: mlx.mlx_array, norm_scores: mlx.mlx_array, mw: *const MoeMlpWeights, group_size: u32, shared: ?moe_affine4.Shared, ungated: ?moe_affine4.Ungated) !bool {
    // The sigmoid table covers 16-bit patterns only; an f32 stream (the fixture oracle) declines.
    const xd = mlx.mlx_array_dtype(expert_x);
    if (xd != .bfloat16 and xd != .float16) return false;
    const xs = mlx.getShape(expert_x);
    const ks = mlx.getShape(inds);
    const k = ks[ks.len - 1];
    var x1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x1);
    try mlx.check(mlx.mlx_reshape(&x1, expert_x, &.{xs[2]}, 1, self.s));
    var ind1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ind1);
    try mlx.check(mlx.mlx_reshape(&ind1, inds, &.{k}, 1, self.s));
    var ind_u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ind_u);
    try mlx.check(mlx.mlx_astype(&ind_u, ind1, .uint32, self.s));
    var sc1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc1);
    try mlx.check(mlx.mlx_reshape(&sc1, norm_scores, &.{k}, 1, self.s));
    const sigtab = try swigluSigTable(self.s, mlx.mlx_array_dtype(expert_x), std.heap.c_allocator);
    const Bank = moe_affine4.Bank;
    const y = (try moe_affine4.decode(self.s, x1, Bank{ .w = mw.switch_gate_w, .s = mw.switch_gate_s, .b = mw.switch_gate_b }, Bank{ .w = mw.switch_up_w, .s = mw.switch_up_s, .b = mw.switch_up_b }, Bank{ .w = mw.switch_down_w, .s = mw.switch_down_s, .b = mw.switch_down_b }, mw.bank_views, shared, ungated, ind_u, sc1, sigtab, group_size)) orelse return false;
    defer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_reshape(out, y, &.{ xs[0], xs[1], xs[2] }, 3, self.s));
    return true;
}

/// fp banks at decode on CUDA: `moe_cuda`'s two launches, the weighted sum [B,S,D] into `out`;
/// false when they decline (the composed chain serves the call).
pub fn moeCudaDecode(self: *Transformer, out: *mlx.mlx_array, expert_x: mlx.mlx_array, inds: mlx.mlx_array, norm_scores: mlx.mlx_array, mw: *const MoeMlpWeights, fmt: @import("moe_cuda.zig").Format) !bool {
    const moe_cuda = @import("moe_cuda.zig");
    const xs = mlx.getShape(expert_x);
    const ks = mlx.getShape(inds);
    const rows = xs[0] * xs[1];
    const k = ks[ks.len - 1];
    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    try mlx.check(mlx.mlx_reshape(&x2, expert_x, &.{ rows, xs[2] }, 2, self.s));
    var ind2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ind2);
    try mlx.check(mlx.mlx_reshape(&ind2, inds, &.{ rows, k }, 2, self.s));
    var ind_u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ind_u);
    try mlx.check(mlx.mlx_astype(&ind_u, ind2, .uint32, self.s));
    var sc2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc2);
    try mlx.check(mlx.mlx_reshape(&sc2, norm_scores, &.{ rows, k }, 2, self.s));
    const gate_scale: ?mlx.mlx_array = if (mw.gate_expert_scale.ctx != null) mw.gate_expert_scale else null;
    const y = (try moe_cuda.decode(self.s, x2, .{ .w = mw.switch_gate_w, .s = mw.switch_gate_s }, .{ .w = mw.switch_up_w, .s = mw.switch_up_s }, .{ .w = mw.switch_down_w, .s = mw.switch_down_s }, gate_scale, ind_u, sc2, fmt)) orelse return false;
    defer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_reshape(out, y, &.{ xs[0], xs[1], xs[2] }, 3, self.s));
    return true;
}

/// The fused-rows MoE arm at decode (B >= 2). It runs ahead of the sorted
/// chain, which stays the fallback whenever this declines: a decline is never
/// a wrong answer, only the sorted chain's extra dispatches.
pub fn moeDecodeGatherQmvRows(
    self: *Transformer,
    out: *mlx.mlx_array,
    expert_x: mlx.mlx_array,
    inds: mlx.mlx_array,
    norm_scores: mlx.mlx_array,
    reduced: *bool,
    mw: *const MoeMlpWeights,
    gate_qp: anytype,
    up_qp: anytype,
    down_qp: anytype,
    D: c_int,
    K: c_int,
    batch: c_int,
    seq: c_int,
) !bool {
    const B = batch * seq; // one kernel row per token
    if (!gatherQmvDownReduceRowsEligible(self.config.hidden_act, gate_qp, up_qp, down_qp, D)) return false;
    var inds_u32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(inds_u32);
    {
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        const kshape = [_]c_int{ B, K };
        try mlx.check(mlx.mlx_reshape(&flat, inds, &kshape, 2, self.s));
        try mlx.check(mlx.mlx_astype(&inds_u32, flat, .uint32, self.s));
    }
    var x_2d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x_2d);
    const dshape = [_]c_int{ B, D };
    try mlx.check(mlx.mlx_reshape(&x_2d, expert_x, &dshape, 2, self.s));

    const act_3d = (try gatherQmvGateUpRows(
        self.s,
        x_2d,
        mw.switch_gate_w,
        mw.switch_gate_s,
        mw.switch_gate_b,
        mw.switch_up_w,
        mw.switch_up_s,
        mw.switch_up_b,
        inds_u32,
        gate_qp.bits,
        gate_qp.group_size,
        gate_qp.mode,
        if (mw.gate_expert_scale.ctx != null) mw.gate_expert_scale else null,
    )) orelse return false;
    defer _ = mlx.mlx_array_free(act_3d);
    var scores_2d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores_2d);
    const kshape2 = [_]c_int{ B, K };
    try mlx.check(mlx.mlx_reshape(&scores_2d, norm_scores, &kshape2, 2, self.s));
    const sum_2d = (try gatherQmvDownReduceRows(
        self.s,
        act_3d,
        mw.switch_down_w,
        mw.switch_down_s,
        mw.switch_down_b,
        inds_u32,
        scores_2d,
        down_qp.bits,
        down_qp.group_size,
        down_qp.mode,
    )) orelse return false;
    defer _ = mlx.mlx_array_free(sum_2d);
    const hidden = mlx.getShape(sum_2d)[1];
    const bsh_shape = [_]c_int{ batch, seq, hidden };
    try mlx.check(mlx.mlx_reshape(out, sum_2d, &bsh_shape, 3, self.s));
    if (seq > 1 and !gather_qmv_rows.moe_verify_rows_logged) {
        gather_qmv_rows.moe_verify_rows_logged = true;
        log.info("[moe-rows] verify rows engaged: S={d} K={d}\n", .{ seq, K });
    }
    reduced.* = true;
    return true;
}

/// MoE MLP with separate router and expert inputs.
/// router_x: input for routing (raw hidden states).
/// expert_x: input for expert computation (possibly normalized).
pub fn moeMLP2(self: *Transformer, router_x: mlx.mlx_array, expert_x_in: mlx.mlx_array, mw: *const MoeMlpWeights) !mlx.mlx_array {
    return self.moeMLP2WithRouter(router_x, expert_x_in, mw, null, false);
}

// MoE decode expert compute: our self-built libmlx serializes `gather_qmm` at
// decode width (~10× vs a batched `quantized_matmul`; plain qmv overlaps fine —
// verified by the MoE gather µbench). So at S==1 we CAN compute the top-K experts
// as `take(experts) + broadcast(x) + batched quantized_matmul` (batchedExpertMm)
// instead of one gather_qmm. Numerically the same per-expert matmul; not bit-
// identical to the gather kernel (different reduction order, qmv-vs-qmm class).
//
// But this is a per-arch VALIDATED opt-in, NOT default-on-for-all-MoE. The
// take-materialization (extracting the top-K experts from the bank every token)
// only pays off where the serialized gather dominates — Laguna 2-bit large
// experts: 17→48 tok/s. On small-expert MoEs the materialization overhead is a
// NET LOSS: gemma4-26B-A4B measured 114→85 and Qwen3.6-MoE regressed too when
// this path was briefly default-on (the 26.7.10 bench). Same class as the
// "eligibility predicate silently adopts every matching shape" kernel rule.
//
// Policy (pure, unit-tested by `batchedExpertDecodePolicy` test):
//   - MLX_SERVE_MOE_GATHER_DECODE=1  → hard force gather (beats everything).
//   - MLX_SERVE_MOE_BATCHED_DECODE=1 → hard force batched (for A/B on any arch).
//   - otherwise                      → stock gather, on every arch.
//
// The laguna default opted INTO the batched path on a 17→48 tok/s measurement
// taken while the YaRN mscale f32 promotion inflated the whole forward ~3x.
// Re-measured on the fixed tree it loses on both Laguna models (see the
// `batchedExpertDecodePolicy` test for the numbers), so no arch opts in by
// default any more. Both alternatives stay one env var away.
fn batchedExpertDecodePolicy(model_type: []const u8, gather_force: bool, batched_force: bool) bool {
    _ = model_type;
    if (gather_force) return false;
    return batched_force;
}

var moe_gather_force_env: ?bool = null;
var moe_batched_force_env: ?bool = null;

pub fn useBatchedExpertDecode(self: *const Transformer) bool {
    return batchedExpertDecodePolicy(
        self.config.model_type,
        envFlagCached(&moe_gather_force_env, "MLX_SERVE_MOE_GATHER_DECODE"),
        envFlagCached(&moe_batched_force_env, "MLX_SERVE_MOE_BATCHED_DECODE"),
    );
}

/// `MLX_SERVE_MOE_GATHER_DECODE=1` — force stock `gather_qmm` everywhere,
/// the A/B control arm for any arch wiring the gather-qmv kernels in.
pub fn gatherDecodeForcedStock() bool {
    return envFlagCached(&moe_gather_force_env, "MLX_SERVE_MOE_GATHER_DECODE");
}

pub fn useGatherQmvDecode(self: *const Transformer, gate_qp: QuantParams, up_qp: QuantParams) bool {
    if (envFlagCached(&moe_gather_force_env, "MLX_SERVE_MOE_GATHER_DECODE")) return false;
    if (useBatchedExpertDecode(self)) return true; // explicit batched opt-in still tries it first
    if (gate_qp.mode == .gguf) return false; // raw ggml blocks: gatherExpertMm has their kernel
    if (self.config.hidden_act != .silu) return false;
    // The fused kernel bakes in silu(gate)*up, clamped at a whole-number limit; others stay composed.
    if (swigluClampLimit(&self.config) == null) return false;
    if (!gatherQmvGateUpEnabled()) return false;
    return gate_qp.bits == up_qp.bits and
        gate_qp.group_size == up_qp.group_size and
        gate_qp.mode == up_qp.mode;
}

/// The fp4 MoE decode kernels read no expert biases, and apply a SwiGLU clamp only as a whole-number
/// limit: a pack with a bias, or `clamp` null (`swigluClampLimit`: a clamp no kernel serves), keeps the
/// composed chain, as does an activation that is not 16-bit (the SwiGLU table covers 16-bit patterns).
pub fn moeFp4DecodeEligible(rows: c_int, has_expert_bias: bool, clamp: ?u32, act: mlx.mlx_dtype, gate: model_mod.QuantMode, up: model_mod.QuantMode, down: model_mod.QuantMode) bool {
    return rows <= moe_fp4.MAX_ROWS and !has_expert_bias and clamp != null and (act == .bfloat16 or act == .float16) and (gate == .mxfp4 or gate == .nvfp4) and up == gate and down == gate;
}

// ── tests ──

test "batchedExpertDecodePolicy: stock gather is the default on EVERY arch" {
    // History, and why the laguna opt-in was retired.
    //
    // The batchedExpertMm decode path (take + batched quantized_matmul) and
    // then `gatherQmv` both exist to dodge our self-built libmlx's decode
    // `gather_qmm`, whose cost is O(EXPERT BANK SIZE) rather than O(work).
    // Laguna opted in by default on a measured 17→48 tok/s.
    //
    // That measurement was taken on a forward pass inflated ~3x by the YaRN
    // mscale f32 promotion. With the promotion fixed the expert math is a much
    // larger share of a much cheaper token, the O(bank) addressing term is a
    // correspondingly smaller one, and MLX's better-tuned `gather_qmv_fast`
    // tile wins — the SAME reasoning that already explained why gemma4-26B-A4B
    // loses with gatherQmv. Re-measured on the fixed tree, alternating boots,
    // 512-token window, serial:
    //
    //   Laguna XS  gatherQmv 13.296 ms/tok   stock 13.113   stock +1.4%
    //   Laguna S   gatherQmv 17.02  ms/tok   stock 15.45    stock +9.2%
    //   (batched take+qmm on XS: 14.839 — worst of the three)
    //
    // Both models, every round, no overlap. So there is no arch left whose
    // default should be anything but stock gather. Both kernels stay reachable
    // by env for A/B — the finding is about the DEFAULT, and the next arch
    // with a genuinely bank-dominated decode can be measured back in.
    try testing.expect(!batchedExpertDecodePolicy("laguna", false, false));
    try testing.expect(!batchedExpertDecodePolicy("gemma4_text", false, false));
    try testing.expect(!batchedExpertDecodePolicy("qwen3_5_moe", false, false));
    try testing.expect(!batchedExpertDecodePolicy("hy_v3", false, false));
    // MLX_SERVE_MOE_BATCHED_DECODE forces it on for any arch (experimentation A/B).
    try testing.expect(batchedExpertDecodePolicy("gemma4_text", false, true));
    try testing.expect(batchedExpertDecodePolicy("laguna", false, true));
    // MLX_SERVE_MOE_GATHER_DECODE is the hard override: still beats the force.
    try testing.expect(!batchedExpertDecodePolicy("laguna", true, false));
    try testing.expect(!batchedExpertDecodePolicy("gemma4_text", true, true));
}

test "moe fp4 decode: expert biases or a clamp no kernel serves keep the composed chain" {
    var cfg: ModelConfig = undefined;
    cfg.swiglu_limit = 0;
    cfg.swiglu_clamp = false;
    try testing.expectEqual(@as(?u32, 0), swigluClampLimit(&cfg));
    cfg.swiglu_limit = 10;
    cfg.swiglu_clamp = true;
    try testing.expectEqual(@as(?u32, 10), swigluClampLimit(&cfg));
    try testing.expect(moeFp4DecodeEligible(1, false, 0, .bfloat16, .mxfp4, .mxfp4, .mxfp4));
    try testing.expect(!moeFp4DecodeEligible(1, true, 0, .bfloat16, .mxfp4, .mxfp4, .mxfp4));
    try testing.expect(moeFp4DecodeEligible(1, false, 10, .bfloat16, .nvfp4, .nvfp4, .nvfp4));
    // An f32 stream (an f32 norm weight upstream) has no sigmoid table: decline, never throw.
    try testing.expect(!moeFp4DecodeEligible(1, false, 0, .float32, .mxfp4, .mxfp4, .mxfp4));
    try testing.expect(moeFp4DecodeEligible(1, false, 0, .float16, .mxfp4, .mxfp4, .mxfp4));
    try testing.expect(!moeFp4DecodeEligible(1, false, null, .bfloat16, .mxfp4, .mxfp4, .mxfp4));
    try testing.expect(!moeFp4DecodeEligible(1, false, 0, .bfloat16, .affine, .mxfp4, .mxfp4));
    try testing.expect(moeFp4DecodeEligible(1, false, 0, .bfloat16, .nvfp4, .nvfp4, .nvfp4));
    try testing.expect(!moeFp4DecodeEligible(1, false, 0, .bfloat16, .nvfp4, .mxfp4, .nvfp4));
    try testing.expect(!moeFp4DecodeEligible(moe_fp4.MAX_ROWS + 1, false, 0, .bfloat16, .mxfp4, .mxfp4, .mxfp4));
}
