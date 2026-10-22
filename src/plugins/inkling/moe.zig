//! plugins/inkling/moe, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns inklingExpertsApply, inklingMoeMlp, InkgroupLimitedRouting, inklingRouterChain.

const mlx = @import("../../mlx/ffi.zig");
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;
const gatherExpertMm = @import("../../sdk/blocks/moe/experts.zig").gatherExpertMm;
const MoeMlpWeights = @import("../../sdk/blocks/moe/weights.zig").MoeMlpWeights;

// ── Sparse MoE MLP ──

/// Inkling expert application: x_flat [T, D] rows through the selected
/// experts of a stacked quantized bank via the sorted gather_qmm path
/// (moeMLP2's global-sort pattern). inds [T, K] (uint32). Returns
/// [T, K, hidden]; caller owns. Serves both the routed bank (K = top_k)
/// and the shared bank (K = n_shared, fixed indices).
pub fn inklingExpertsApply(
    self: *Transformer,
    x_flat: mlx.mlx_array,
    inds: mlx.mlx_array,
    gate_w: mlx.mlx_array,
    gate_s: mlx.mlx_array,
    gate_b: mlx.mlx_array,
    up_w: mlx.mlx_array,
    up_s: mlx.mlx_array,
    up_b: mlx.mlx_array,
    down_w: mlx.mlx_array,
    down_s: mlx.mlx_array,
    down_b: mlx.mlx_array,
) !mlx.mlx_array {
    const cfg = &self.config;
    const x_shape = mlx.getShape(x_flat); // [T, D]
    const t_count = x_shape[0];
    const d_dim = x_shape[1];
    const k_count = mlx.getShape(inds)[1];
    const total_inds: c_int = t_count * k_count;
    const no_idx = mlx.mlx_array{ .ctx = null };

    const gate_qp = self.quantParamsHinted(gate_w, gate_s, @intCast(d_dim));
    const up_qp = self.quantParamsHinted(up_w, up_s, @intCast(d_dim));
    const down_qp = self.quantParamsHinted(down_w, down_s, if (cfg.moe_intermediate_size > 0) cfg.moe_intermediate_size else null);

    // Flatten + global sort so gather_qmm streams each expert's rows once.
    const flat_shape = [_]c_int{total_inds};
    var flat_inds = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat_inds);
    try mlx.check(mlx.mlx_reshape(&flat_inds, inds, &flat_shape, 1, self.s));
    var order = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(order);
    try mlx.check(mlx.mlx_argsort_axis(&order, flat_inds, 0, self.s));
    var inv_order = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(inv_order);
    try mlx.check(mlx.mlx_argsort_axis(&inv_order, order, 0, self.s));
    var sorted_inds = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sorted_inds);
    try mlx.check(mlx.mlx_take_axis(&sorted_inds, flat_inds, order, 0, self.s));

    const k_arr = mlx.mlx_array_new_int(k_count);
    defer _ = mlx.mlx_array_free(k_arr);
    var lhs_idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lhs_idx);
    try mlx.check(mlx.mlx_floor_divide(&lhs_idx, order, k_arr, self.s));

    var x_gathered = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x_gathered);
    try mlx.check(mlx.mlx_take_axis(&x_gathered, x_flat, lhs_idx, 0, self.s));
    const n1d_shape = [_]c_int{ total_inds, 1, d_dim };
    var x_rep = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x_rep);
    try mlx.check(mlx.mlx_reshape(&x_rep, x_gathered, &n1d_shape, 3, self.s));

    var gate_out_3d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gate_out_3d);
    try gatherExpertMm(&gate_out_3d, x_rep, gate_w, gate_s, gate_b, no_idx, sorted_inds, gate_qp.bits, gate_qp.group_size, gate_qp.mode, true, self.s);
    var gate_out = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gate_out);
    try mlx.check(mlx.mlx_squeeze(&gate_out, gate_out_3d, self.s));

    var up_out_3d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(up_out_3d);
    try gatherExpertMm(&up_out_3d, x_rep, up_w, up_s, up_b, no_idx, sorted_inds, up_qp.bits, up_qp.group_size, up_qp.mode, true, self.s);
    var up_out = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(up_out);
    try mlx.check(mlx.mlx_squeeze(&up_out, up_out_3d, self.s));

    const expert_act = try self.computeGeglu(gate_out, up_out);
    defer _ = mlx.mlx_array_free(expert_act);

    var act_exp = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(act_exp);
    try mlx.check(mlx.mlx_expand_dims(&act_exp, expert_act, -2, self.s));
    var down_3d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(down_3d);
    try gatherExpertMm(&down_3d, act_exp, down_w, down_s, down_b, no_idx, sorted_inds, down_qp.bits, down_qp.group_size, down_qp.mode, true, self.s);
    var down_squeezed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(down_squeezed);
    try mlx.check(mlx.mlx_squeeze(&down_squeezed, down_3d, self.s));

    var down_unsorted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(down_unsorted);
    try mlx.check(mlx.mlx_take_axis(&down_unsorted, down_squeezed, inv_order, 0, self.s));
    const hidden = mlx.getShape(down_unsorted)[1];
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    const tkh_shape = [_]c_int{ t_count, k_count, hidden };
    try mlx.check(mlx.mlx_reshape(&out, down_unsorted, &tkh_shape, 3, self.s));
    return out;
}

/// Inkling MoE block (`InklingMoE`): fp32 router (mlp.gate, [E+ns, H] —
/// shared rows included; gate.global_scale pre-folded into mw.route_scale
/// at load), routed top-k + the router-gated shared experts (the sink).
/// Expert math runs in the stream's dtype; the routing weights are cast to
/// it before weighting (reference semantics — bf16 on shipped builds).
pub fn inklingMoeMlp(self: *Transformer, x: mlx.mlx_array, mw: *const MoeMlpWeights) !mlx.mlx_array {
    const cfg = &self.config;
    const x_shape = mlx.getShape(x); // [B, S, D]
    const batch = x_shape[0];
    const seq = x_shape[1];
    const d_dim = x_shape[2];
    const t_count: c_int = batch * seq;
    const top_k: c_int = @intCast(cfg.num_experts_per_tok);
    const n_shared: c_int = @intCast(cfg.inkling_n_shared_experts);

    var x_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x_flat);
    const flat_shape = [_]c_int{ t_count, d_dim };
    try mlx.check(mlx.mlx_reshape(&x_flat, x, &flat_shape, 2, self.s));

    // Router in f32: router_w is bf16 pre-transposed [D, E+ns].
    var x32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x32);
    try mlx.check(mlx.mlx_astype(&x32, x_flat, .float32, self.s));
    var rw32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rw32);
    try mlx.check(mlx.mlx_astype(&rw32, mw.router_w, .float32, self.s));
    var logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(logits);
    try mlx.check(mlx.mlx_matmul(&logits, x32, rw32, self.s));

    const routing = try inklingRouterChain(self.s, logits, mw.expert_bias.?, top_k, n_shared, mw.route_scale, 1.0);
    defer _ = mlx.mlx_array_free(routing.topk_weights);
    defer _ = mlx.mlx_array_free(routing.topk_idx);
    defer _ = mlx.mlx_array_free(routing.shared_gammas);

    // Routed experts, weighted sum over K. The reference casts the f32
    // routing weights to x's dtype BEFORE the multiply (bf16 stream on the
    // shipped checkpoints — the sum then accumulates in bf16 like mx.sum).
    const x_dtype = mlx.mlx_array_dtype(x_flat);
    const routed = try self.inklingExpertsApply(x_flat, routing.topk_idx, mw.switch_gate_w, mw.switch_gate_s, mw.switch_gate_b, mw.switch_up_w, mw.switch_up_s, mw.switch_up_b, mw.switch_down_w, mw.switch_down_s, mw.switch_down_b);
    defer _ = mlx.mlx_array_free(routed);
    var tw_x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tw_x);
    try mlx.check(mlx.mlx_astype(&tw_x, routing.topk_weights, x_dtype, self.s));
    var tw_exp = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tw_exp);
    try mlx.check(mlx.mlx_expand_dims(&tw_exp, tw_x, -1, self.s));
    var routed_w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(routed_w);
    try mlx.check(mlx.mlx_multiply(&routed_w, routed, tw_exp, self.s));
    var routed_sum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(routed_sum);
    try mlx.check(mlx.mlx_sum_axis(&routed_sum, routed_w, 1, false, self.s));

    // Shared experts: fixed indices [T, ns], gamma-weighted f32 sum.
    var sh_arange = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sh_arange);
    try mlx.check(mlx.mlx_arange(&sh_arange, 0, @floatFromInt(n_shared), 1, .uint32, self.s));
    var sh_row = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sh_row);
    try mlx.check(mlx.mlx_reshape(&sh_row, sh_arange, &[_]c_int{ 1, n_shared }, 2, self.s));
    var sh_idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sh_idx);
    try mlx.check(mlx.mlx_broadcast_to(&sh_idx, sh_row, &[_]c_int{ t_count, n_shared }, 2, self.s));

    const shared = try self.inklingExpertsApply(x_flat, sh_idx, mw.shared_gate_w, mw.shared_gate_s, mw.shared_gate_b, mw.shared_up_w, mw.shared_up_s, mw.shared_up_b, mw.shared_down_w, mw.shared_down_s, mw.shared_down_b);
    defer _ = mlx.mlx_array_free(shared);
    // Reference: gammas are ROUNDED to x's dtype first (astype(x.dtype)),
    // then the shared sum is computed in f32 and cast to the routed dtype.
    var sh32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sh32);
    try mlx.check(mlx.mlx_astype(&sh32, shared, .float32, self.s));
    var sg_x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sg_x);
    try mlx.check(mlx.mlx_astype(&sg_x, routing.shared_gammas, x_dtype, self.s));
    var sg32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sg32);
    try mlx.check(mlx.mlx_astype(&sg32, sg_x, .float32, self.s));
    var sg_exp = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sg_exp);
    try mlx.check(mlx.mlx_expand_dims(&sg_exp, sg32, -1, self.s));
    var shared_w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(shared_w);
    try mlx.check(mlx.mlx_multiply(&shared_w, sh32, sg_exp, self.s));
    var shared_sum32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(shared_sum32);
    try mlx.check(mlx.mlx_sum_axis(&shared_sum32, shared_w, 1, false, self.s));
    var shared_sum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(shared_sum);
    try mlx.check(mlx.mlx_astype(&shared_sum, shared_sum32, mlx.mlx_array_dtype(routed_sum), self.s));

    var combined = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(combined);
    try mlx.check(mlx.mlx_add(&combined, routed_sum, shared_sum, self.s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    const out_shape = [_]c_int{ batch, seq, d_dim };
    try mlx.check(mlx.mlx_reshape(&out, combined, &out_shape, 3, self.s));
    return out;
}

const InkgroupLimitedRouting = struct { topk_weights: mlx.mlx_array, topk_idx: mlx.mlx_array, shared_gammas: mlx.mlx_array };

/// The Inkling router chain (`InklingTopkRouter`), all in f32: sigmoid scores
/// + selection bias pick the top-k routed experts, then ONE softmax over the
/// logsigmoid of [selected routed logits, shared logits] produces the routed
/// weights AND the shared-expert gammas (the "sink"), scaled by
/// route_scale × global_scale. `logits` = x_f32 @ gate_weight.T [T, E+ns].
pub fn inklingRouterChain(s: mlx.mlx_stream, logits: mlx.mlx_array, bias: mlx.mlx_array, top_k: c_int, n_shared: c_int, route_scale: f32, global_scale: f32) !InkgroupLimitedRouting {
    const l_shape = mlx.getShape(logits); // [T, E+ns]
    const t_count = l_shape[0];
    const n_routed = l_shape[1] - n_shared;
    const strides2 = [_]c_int{ 1, 1 };

    var routed_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(routed_logits);
    try mlx.check(mlx.mlx_slice(&routed_logits, logits, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ t_count, n_routed }, 2, &strides2, 2, s));
    var shared_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(shared_logits);
    try mlx.check(mlx.mlx_slice(&shared_logits, logits, &[_]c_int{ 0, n_routed }, 2, &[_]c_int{ t_count, l_shape[1] }, 2, &strides2, 2, s));

    // top-k on sigmoid(routed) + bias (selection only; weights use the logits)
    var scores = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores);
    try mlx.check(mlx.mlx_sigmoid(&scores, routed_logits, s));
    var choice = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(choice);
    try mlx.check(mlx.mlx_add(&choice, scores, bias, s));
    var neg_choice = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(neg_choice);
    try mlx.check(mlx.mlx_negative(&neg_choice, choice, s));
    var part = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(part);
    try mlx.check(mlx.mlx_argpartition_axis(&part, neg_choice, top_k - 1, -1, s));
    var topk_idx = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(topk_idx);
    {
        var sliced = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sliced);
        try mlx.check(mlx.mlx_slice(&sliced, part, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ t_count, top_k }, 2, &strides2, 2, s));
        try mlx.check(mlx.mlx_astype(&topk_idx, sliced, .uint32, s));
    }

    var gathered = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gathered);
    try mlx.check(mlx.mlx_take_along_axis(&gathered, routed_logits, topk_idx, 1, s));
    var topk_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(topk_logits);
    {
        const arr = [_]mlx.mlx_array{ gathered, shared_logits };
        const vec = mlx.mlx_vector_array_new_data(&arr, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&topk_logits, vec, 1, s));
    }

    // logsigmoid(x) = -logaddexp(0, -x), then softmax over [K + ns]
    var neg_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(neg_logits);
    try mlx.check(mlx.mlx_negative(&neg_logits, topk_logits, s));
    const zero_f = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(zero_f);
    var lae = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lae);
    try mlx.check(mlx.mlx_logaddexp(&lae, zero_f, neg_logits, s));
    var log_probs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(log_probs);
    try mlx.check(mlx.mlx_negative(&log_probs, lae, s));
    var soft = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(soft);
    try mlx.check(mlx.mlx_softmax_axis(&soft, log_probs, -1, true, s));
    const scale_f = mlx.mlx_array_new_float(route_scale * global_scale);
    defer _ = mlx.mlx_array_free(scale_f);
    var weights = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weights);
    try mlx.check(mlx.mlx_multiply(&weights, soft, scale_f, s));

    var topk_weights = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(topk_weights);
    try mlx.check(mlx.mlx_slice(&topk_weights, weights, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ t_count, top_k }, 2, &strides2, 2, s));
    var shared_gammas = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(shared_gammas);
    try mlx.check(mlx.mlx_slice(&shared_gammas, weights, &[_]c_int{ 0, top_k }, 2, &[_]c_int{ t_count, top_k + n_shared }, 2, &strides2, 2, s));

    return .{ .topk_weights = topk_weights, .topk_idx = topk_idx, .shared_gammas = shared_gammas };
}
