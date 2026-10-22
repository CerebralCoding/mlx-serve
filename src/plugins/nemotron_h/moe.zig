//! plugins/nemotron_h/moe, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns NemotronMoeWeights, NemotronMoeParts, nemotronMoeExperts, nemotronMoeExpertsRouted, nemotron_latent_logged, nemotronMoe, and 9 more.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const rowqmv = @import("../../sdk/blocks/qmm/rowqmv.zig");
const model_mod = @import("../../engines/mlx/model.zig");
const log = @import("../../host/log.zig");
const ModelConfig = @import("../../engines/mlx/model.zig").ModelConfig;
const lastDim = @import("../../engines/mlx/load/weights.zig").lastDim;
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;
const projectWithConfig = @import("../../sdk/blocks/linear/qmatmul.zig").projectWithConfig;
const verifyQmmNaxAvailable = @import("../../sdk/blocks/linear/verify_qmm_nax.zig").verifyQmmNaxAvailable;
const reluSquaredOp = @import("../../sdk/blocks/mlp/activations.zig").reluSquaredOp;
const SimpleMlpWeights = @import("../../sdk/blocks/mlp/dense.zig").SimpleMlpWeights;
const gatherExpertMm = @import("../../sdk/blocks/moe/experts.zig").gatherExpertMm;
const gatherQmv = @import("../../sdk/blocks/moe/gather_qmv.zig").gatherQmv;
const gatherQmvAct = @import("../../sdk/blocks/moe/gather_qmv.zig").gatherQmvAct;
const groupLimitedRouting = @import("../../sdk/blocks/moe/routing.zig").groupLimitedRouting;
const quantParamsOrDense = @import("../../sdk/quant/params.zig").quantParamsOrDense;

/// Nemotron-H MoE block (mlx-lm `NemotronHMoE`).
/// Puzzle trunks set the latent pair: experts run in `moe_latent_size`, and
/// the router plus the shared expert stay on the full hidden width.
pub const NemotronMoeWeights = struct {
    router_w: mlx.mlx_array, // mixer.gate.weight — dense, pre-transposed [hidden, E]
    router_s: mlx.mlx_array,
    router_b: mlx.mlx_array,
    expert_bias: mlx.mlx_array, // e_score_correction_bias f32 [E], selection only
    fc1_w: mlx.mlx_array, // switch_mlp.fc1 [E, inter, expert_in]
    fc1_s: mlx.mlx_array,
    fc1_b: mlx.mlx_array,
    fc2_w: mlx.mlx_array, // switch_mlp.fc2 [E, expert_in, inter]
    fc2_s: mlx.mlx_array,
    fc2_b: mlx.mlx_array,
    shared: ?SimpleMlpWeights = null, // shared_experts (ReLU^2, ungated, always added)
    router_rows: mlx.mlx_array = .{ .ctx = null }, // mixer.gate.weight as stored [E, hidden], for `rowqmv.router`
    fc1_latent_w: mlx.mlx_array = .{ .ctx = null }, // [latent, hidden] after the dense transpose
    fc1_latent_s: mlx.mlx_array = .{ .ctx = null },
    fc1_latent_b: mlx.mlx_array = .{ .ctx = null },
    fc2_latent_w: mlx.mlx_array = .{ .ctx = null }, // [hidden, latent] after the dense transpose
    fc2_latent_s: mlx.mlx_array = .{ .ctx = null },
    fc2_latent_b: mlx.mlx_array = .{ .ctx = null },
};

/// Nemotron-H MoE block (mlx-lm `NemotronHMoE`): `groupLimitedRouting` over
/// the sigmoid router, ReLU^2 routed experts (one token: in-place `gatherQmv`
/// reads, `nemotronMoeDecodeExperts`; else the sorted gather_qmm path), the
/// f32 routing weights applied before the K-sum, then the shared expert
/// added. x [B, S, hidden] → [B, S, hidden]; caller owns.
/// The routed experts' outputs before the K-sum: `y_tkh` [T, KE, D] per slot
/// (KE = top-k plus the folded shared rows), `scores` [T, K] f32 routing
/// weights for the first K slots.
const NemotronMoeParts = struct {
    y_tkh: mlx.mlx_array,
    scores: mlx.mlx_array,
    k_count: c_int,
    pub fn deinit(self: *const NemotronMoeParts) void {
        _ = mlx.mlx_array_free(self.y_tkh);
        _ = mlx.mlx_array_free(self.scores);
    }
};

/// `stream_pad`: pad a short sorted window up to the expert-streaming kernel (`moeStreamPadRows`).
pub fn nemotronMoeExperts(x: mlx.mlx_array, nm: *const NemotronMoeWeights, cfg: *const ModelConfig, stream_pad: bool, s: mlx.mlx_stream) !NemotronMoeParts {
    return nemotronMoeExpertsRouted(x, x, nm, cfg, stream_pad, s);
}

/// Experts consume `x`. The router reads `route_x` (the full hidden state when
/// `x` has already been projected into the latent).
fn nemotronMoeExpertsRouted(x: mlx.mlx_array, route_x: mlx.mlx_array, nm: *const NemotronMoeWeights, cfg: *const ModelConfig, stream_pad: bool, s: mlx.mlx_stream) !NemotronMoeParts {
    const x_shape = mlx.getShape(x);
    const batch = x_shape[0];
    const seq = x_shape[1];
    const d_dim = x_shape[2];
    const t_count: c_int = batch * seq;
    const k_count: c_int = @intCast(cfg.num_experts_per_tok);
    const total_inds: c_int = t_count * k_count;
    const no_idx = mlx.mlx_array{ .ctx = null };

    var x_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x_flat);
    try mlx.check(mlx.mlx_reshape(&x_flat, x, &[_]c_int{ t_count, d_dim }, 2, s));

    var route_owned = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(route_owned);
    const route_flat = if (route_x.ctx == x.ctx) x_flat else blk: {
        try mlx.check(mlx.mlx_reshape(&route_owned, route_x, &[_]c_int{ t_count, mlx.getShape(route_x)[2] }, 2, s));
        break :blk route_owned;
    };

    const row_logits = if (cfg.rowExactDecode() and nm.router_rows.ctx != null) try rowqmv.router(route_flat, nm.router_rows, s) else null;
    const logits = row_logits orelse try projectWithConfig(cfg, route_flat, nm.router_w, nm.router_s, nm.router_b, s);
    defer _ = mlx.mlx_array_free(logits);
    const routing = try groupLimitedRouting(logits, nm.expert_bias, k_count, @intCast(cfg.moe_n_group), @intCast(cfg.moe_topk_group), cfg.moe_route_norm, cfg.router_scaling_factor, s);
    errdefer _ = mlx.mlx_array_free(routing.norm_scores);
    defer _ = mlx.mlx_array_free(routing.inds);

    var y_tkh = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y_tkh);
    if (t_count <= nemotron_gather_max_rows or (cfg.rowExactDecode() and t_count <= rowqmv.MAX_ROWS)) {
        try nemotronMoeDecodeExperts(&y_tkh, x_flat, nm, routing.inds, t_count, k_count, d_dim, cfg, s);
    } else {
        // Global sort so gather_qmm streams each expert's rows once (the
        // `inklingExpertsApply` pattern, ungated).
        var flat_inds = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat_inds);
        try mlx.check(mlx.mlx_reshape(&flat_inds, routing.inds, &[_]c_int{total_inds}, 1, s));
        const n_experts = mlx.getShape(nm.fc1_w)[0];
        const pad = if (stream_pad) moeStreamPadRows(total_inds, n_experts) else 0;
        if (pad > 0) {
            // Pad rows sort to order index >= total_inds: the inv_order slice drops them,
            // so the x row they read (clamped to the last token) is irrelevant.
            var pad_i = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(pad_i);
            try mlx.check(mlx.mlx_arange(&pad_i, 0, @floatFromInt(pad), 1, .int32, s));
            const e_arr = mlx.mlx_array_new_int(n_experts);
            defer _ = mlx.mlx_array_free(e_arr);
            var pad_e = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(pad_e);
            try mlx.check(mlx.mlx_remainder(&pad_e, pad_i, e_arr, s));
            var pad_t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(pad_t);
            try mlx.check(mlx.mlx_astype(&pad_t, pad_e, mlx.mlx_array_dtype(flat_inds), s));
            const pair = [_]mlx.mlx_array{ flat_inds, pad_t };
            const vec = mlx.mlx_vector_array_new_data(&pair, 2);
            defer _ = mlx.mlx_vector_array_free(vec);
            var joined = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_concatenate_axis(&joined, vec, 0, s));
            _ = mlx.mlx_array_free(flat_inds);
            flat_inds = joined;
        }
        const rows_all: c_int = total_inds + pad;
        var order = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(order);
        try mlx.check(mlx.mlx_argsort_axis(&order, flat_inds, 0, s));
        var inv_order = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(inv_order);
        try mlx.check(mlx.mlx_argsort_axis(&inv_order, order, 0, s));
        var sorted_inds = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sorted_inds);
        try mlx.check(mlx.mlx_take_axis(&sorted_inds, flat_inds, order, 0, s));
        const k_arr = mlx.mlx_array_new_int(k_count);
        defer _ = mlx.mlx_array_free(k_arr);
        var lhs_idx = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(lhs_idx);
        try mlx.check(mlx.mlx_floor_divide(&lhs_idx, order, k_arr, s));
        if (pad > 0) {
            const last = mlx.mlx_array_new_int(t_count - 1);
            defer _ = mlx.mlx_array_free(last);
            var clamped = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_minimum(&clamped, lhs_idx, last, s));
            _ = mlx.mlx_array_free(lhs_idx);
            lhs_idx = clamped;
        }
        var x_gathered = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_gathered);
        try mlx.check(mlx.mlx_take_axis(&x_gathered, x_flat, lhs_idx, 0, s));
        var x_rep = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_rep);
        try mlx.check(mlx.mlx_reshape(&x_rep, x_gathered, &[_]c_int{ rows_all, 1, d_dim }, 3, s));

        const fc1_qp = quantParamsOrDense(cfg, nm.fc1_w, nm.fc1_s, @intCast(d_dim));
        var h_3d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(h_3d);
        try gatherExpertMm(&h_3d, x_rep, nm.fc1_w, nm.fc1_s, nm.fc1_b, no_idx, sorted_inds, fc1_qp.bits, fc1_qp.group_size, fc1_qp.mode, true, s);
        const act = try reluSquaredOp(h_3d, s);
        defer _ = mlx.mlx_array_free(act);
        const fc2_qp = quantParamsOrDense(cfg, nm.fc2_w, nm.fc2_s, lastDim(act));
        var y_3d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(y_3d);
        try gatherExpertMm(&y_3d, act, nm.fc2_w, nm.fc2_s, nm.fc2_b, no_idx, sorted_inds, fc2_qp.bits, fc2_qp.group_size, fc2_qp.mode, true, s);
        var y_unsorted = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(y_unsorted);
        if (pad > 0) {
            var head = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&head, inv_order, &[_]c_int{0}, 1, &[_]c_int{total_inds}, 1, &[_]c_int{1}, 1, s));
            _ = mlx.mlx_array_free(inv_order);
            inv_order = head;
        }
        try mlx.check(mlx.mlx_take_axis(&y_unsorted, y_3d, inv_order, 0, s));
        try mlx.check(mlx.mlx_reshape(&y_tkh, y_unsorted, &[_]c_int{ t_count, k_count, d_dim }, 3, s));
    }
    return .{ .y_tkh = y_tkh, .scores = routing.norm_scores, .k_count = k_count };
}

/// Nemotron-H MoE block (mlx-lm `NemotronHMoE`): `groupLimitedRouting` over
/// the sigmoid router, ReLU^2 routed experts (one token: in-place `gatherQmv`
/// reads, `nemotronMoeDecodeExperts`; else the sorted gather_qmm path), the
/// f32 routing weights applied before the K-sum, then the shared expert
/// added. x [B, S, hidden] → [B, S, hidden]; caller owns.
var nemotron_latent_logged = false;

pub fn nemotronMoe(x: mlx.mlx_array, nm: *const NemotronMoeWeights, cfg: *const ModelConfig, s: mlx.mlx_stream) !mlx.mlx_array {
    if (nm.fc1_latent_w.ctx != null) {
        if (!nemotron_latent_logged) {
            nemotron_latent_logged = true;
            log.info("[moe] nemotron latent projection engaged: latent={d}\n", .{cfg.moe_latent_size});
        }
        return nemotronMoeLatent(x, nm, cfg, s);
    }
    const parts = try nemotronMoeExperts(x, nm, cfg, moeStreamPadPays(nm), s);
    defer parts.deinit();
    return nemotronMoeCombine(x, &parts, nm, cfg, s, x);
}

/// Puzzle MoE: route on the full hidden state, run experts in the latent,
/// project back, then add the shared expert on the original residual.
fn nemotronMoeLatent(x: mlx.mlx_array, nm: *const NemotronMoeWeights, cfg: *const ModelConfig, s: mlx.mlx_stream) !mlx.mlx_array {
    const x_shape = mlx.getShape(x);
    const batch = x_shape[0];
    const seq = x_shape[1];
    const hidden = x_shape[2];
    const t_count: c_int = batch * seq;

    var x_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x_flat);
    try mlx.check(mlx.mlx_reshape(&x_flat, x, &[_]c_int{ t_count, hidden }, 2, s));
    const lat = try projectWithConfig(cfg, x_flat, nm.fc1_latent_w, nm.fc1_latent_s, nm.fc1_latent_b, s);
    defer _ = mlx.mlx_array_free(lat);
    const lat_dim = mlx.getShape(lat)[1];
    var x_lat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x_lat);
    try mlx.check(mlx.mlx_reshape(&x_lat, lat, &[_]c_int{ batch, seq, lat_dim }, 3, s));

    const parts = try nemotronMoeExpertsRouted(x_lat, x, nm, cfg, moeStreamPadPays(nm), s);
    defer parts.deinit();
    return nemotronMoeCombine(x_lat, &parts, nm, cfg, s, x);
}

/// The shared expert (ReLU^2, ungated) over x [B, S, hidden] → [B*S, hidden];
/// null when the block has none. Caller owns.
pub fn nemotronSharedExpert(x: mlx.mlx_array, nm: *const NemotronMoeWeights, cfg: *const ModelConfig, s: mlx.mlx_stream) !?mlx.mlx_array {
    const sm = nm.shared orelse return null;
    const x_shape = mlx.getShape(x);
    var x_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x_flat);
    try mlx.check(mlx.mlx_reshape(&x_flat, x, &[_]c_int{ x_shape[0] * x_shape[1], x_shape[2] }, 2, s));
    const up_qp = quantParamsOrDense(cfg, sm.up_w, sm.up_s, lastDim(x_flat));
    const up_act = (if (cfg.rowExactDecode()) try rowqmv.qmvAct(x_flat, sm.up_w, sm.up_s, sm.up_b, up_qp.bits, up_qp.group_size, true, s) else null) orelse blk: {
        const up = try projectWithConfig(cfg, x_flat, sm.up_w, sm.up_s, sm.up_b, s);
        defer _ = mlx.mlx_array_free(up);
        break :blk try reluSquaredOp(up, s);
    };
    defer _ = mlx.mlx_array_free(up_act);
    return try projectWithConfig(cfg, up_act, sm.down_w, sm.down_s, sm.down_b, s);
}

/// Reference: (y * scores[..., None]).sum(-2).astype(y.dtype) with f32
/// scores (product and sum in f32), plus the shared expert.
pub fn nemotronMoeCombine(x: mlx.mlx_array, parts: *const NemotronMoeParts, nm: *const NemotronMoeWeights, cfg: *const ModelConfig, s: mlx.mlx_stream, shared_x: ?mlx.mlx_array) !mlx.mlx_array {
    const x_shape = mlx.getShape(x);
    const batch = x_shape[0];
    const seq = x_shape[1];

    var sc_exp = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc_exp);
    try mlx.check(mlx.mlx_expand_dims(&sc_exp, parts.scores, -1, s));
    var weighted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weighted);
    try mlx.check(mlx.mlx_multiply(&weighted, parts.y_tkh, sc_exp, s));
    var summed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(summed);
    try mlx.check(mlx.mlx_sum_axis(&summed, weighted, 1, false, s));
    var routed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(routed);
    try mlx.check(mlx.mlx_astype(&routed, summed, mlx.mlx_array_dtype(parts.y_tkh), s));
    if (nm.fc2_latent_w.ctx != null) {
        const projected = try projectWithConfig(cfg, routed, nm.fc2_latent_w, nm.fc2_latent_s, nm.fc2_latent_b, s);
        _ = mlx.mlx_array_free(routed);
        routed = projected;
    }

    var combined = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(combined);
    const shared = if (shared_x) |input| try nemotronSharedExpert(input, nm, cfg, s) else null;
    if (shared) |value| {
        defer _ = mlx.mlx_array_free(value);
        try mlx.check(mlx.mlx_add(&combined, routed, value, s));
    } else {
        try mlx.check(mlx.mlx_array_set(&combined, routed));
    }
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, combined, &[_]c_int{ batch, seq, mlx.getShape(routed)[1] }, 3, s));
    return out;
}

var nemotron_gqmv_engaged = false;
/// Widest token window that reads its experts in place (decode steps and
/// spec-decode verify windows); wider windows take the sorted gather_qmm
/// path, which streams each expert once. Test seam.
pub var nemotron_gather_max_rows: c_int = 8;

/// Rows to append so a sorted quantized gather reaches the expert-streaming kernel, which
/// needs 4+ rows per expert (`B / E >= 4`, quantized.cpp); below it every row re-reads its
/// expert. Below 2 rows per expert the re-reads stay cheaper than the pad.
fn moeStreamPadRows(rows: c_int, experts: c_int) c_int {
    return if (rows >= 2 * experts and rows < 4 * experts) 4 * experts - rows else 0;
}

/// The pad was measured only with the NAX sorted gather; dense banks run `mlx_gather_mm`
/// unsorted, where it is pure extra work.
pub fn moeStreamPadPays(nm: *const NemotronMoeWeights) bool {
    return nm.fc1_s.ctx != null and verifyQmmNaxAvailable();
}

/// A short window of T tokens: every (token, expert) pair is one `gatherQmv`
/// row (fc1 with ReLU^2 fused, then fc2), falling back to UNSORTED
/// gather_qmm when the bank is outside its envelope. Either way none of the
/// sort machinery runs. `res` = [T, K, hidden].
fn nemotronMoeDecodeExperts(res: *mlx.mlx_array, x_flat: mlx.mlx_array, nm: *const NemotronMoeWeights, inds: mlx.mlx_array, t_count: c_int, k_count: c_int, d_dim: c_int, cfg: *const ModelConfig, s: mlx.mlx_stream) !void {
    const no_idx = mlx.mlx_array{ .ctx = null };
    const out_shape = [_]c_int{ t_count, k_count, d_dim };
    const pairs = t_count * k_count;
    var inds_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(inds_flat);
    try mlx.check(mlx.mlx_reshape(&inds_flat, inds, &[_]c_int{pairs}, 1, s));
    const fc1_qp = quantParamsOrDense(cfg, nm.fc1_w, nm.fc1_s, @intCast(d_dim));

    if (cfg.rowExactDecode() and fc1_qp.mode == .affine and nm.fc1_s.ctx != null and nm.fc2_s.ctx != null) {
        var inds_u32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(inds_u32);
        try mlx.check(mlx.mlx_astype(&inds_u32, inds_flat, .uint32, s));
        const fc2_qp = quantParamsOrDense(cfg, nm.fc2_w, nm.fc2_s, @intCast(mlx.getShape(nm.fc1_w)[1]));
        if (try rowqmv.experts(x_flat, inds_u32, k_count, .{ nm.fc1_w, nm.fc1_s, nm.fc1_b }, .{ nm.fc2_w, nm.fc2_s, nm.fc2_b }, .{ fc1_qp.bits, fc1_qp.group_size }, .{ fc2_qp.bits, fc2_qp.group_size }, s)) |y| {
            defer _ = mlx.mlx_array_free(y);
            try mlx.check(mlx.mlx_reshape(res, y, &out_shape, 3, s));
            return;
        }
    }

    if (nm.fc1_s.ctx != null and nm.fc2_s.ctx != null) fused: {
        var inds_u32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(inds_u32);
        try mlx.check(mlx.mlx_astype(&inds_u32, inds_flat, .uint32, s));
        // One token: x shared by its K experts. T tokens: x repeated per pair.
        var x_in = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_in);
        if (t_count == 1) {
            try mlx.check(mlx.mlx_reshape(&x_in, x_flat, &[_]c_int{d_dim}, 1, s));
        } else {
            var x_3d = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x_3d);
            try mlx.check(mlx.mlx_reshape(&x_3d, x_flat, &[_]c_int{ t_count, 1, d_dim }, 3, s));
            var x_b = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x_b);
            try mlx.check(mlx.mlx_broadcast_to(&x_b, x_3d, &[_]c_int{ t_count, k_count, d_dim }, 3, s));
            try mlx.check(mlx.mlx_reshape(&x_in, x_b, &[_]c_int{ pairs, d_dim }, 2, s));
        }
        const act = (try gatherQmvAct(s, x_in, nm.fc1_w, nm.fc1_s, nm.fc1_b, inds_u32, fc1_qp.bits, fc1_qp.group_size, fc1_qp.mode, t_count > 1, true)) orelse break :fused;
        defer _ = mlx.mlx_array_free(act);
        const fc2_qp = quantParamsOrDense(cfg, nm.fc2_w, nm.fc2_s, lastDim(act));
        const y = (try gatherQmv(s, act, nm.fc2_w, nm.fc2_s, nm.fc2_b, inds_u32, fc2_qp.bits, fc2_qp.group_size, fc2_qp.mode, true)) orelse break :fused;
        defer _ = mlx.mlx_array_free(y);
        try mlx.check(mlx.mlx_reshape(res, y, &out_shape, 3, s));
        if (!nemotron_gqmv_engaged) {
            nemotron_gqmv_engaged = true;
            log.info("[moe] nemotron decode gather-qmv engaged: K={d} inter={d} hidden={d} bits={d} gs={d}\n", .{ k_count, mlx.getShape(act)[1], d_dim, fc2_qp.bits, fc2_qp.group_size });
        }
        return;
    }

    // Batch dims (T, 1) broadcast against the (T, K) indices; (T) alone only does at T == 1.
    var x_rep = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x_rep);
    try mlx.check(mlx.mlx_reshape(&x_rep, x_flat, &[_]c_int{ t_count, 1, 1, d_dim }, 4, s));
    var inds_tk = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(inds_tk);
    try mlx.check(mlx.mlx_reshape(&inds_tk, inds, &[_]c_int{ t_count, k_count }, 2, s));
    var h_3d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(h_3d);
    try gatherExpertMm(&h_3d, x_rep, nm.fc1_w, nm.fc1_s, nm.fc1_b, no_idx, inds_tk, fc1_qp.bits, fc1_qp.group_size, fc1_qp.mode, false, s);
    const act = try reluSquaredOp(h_3d, s);
    defer _ = mlx.mlx_array_free(act);
    const fc2_qp = quantParamsOrDense(cfg, nm.fc2_w, nm.fc2_s, lastDim(act));
    var y_3d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y_3d);
    try gatherExpertMm(&y_3d, act, nm.fc2_w, nm.fc2_s, nm.fc2_b, no_idx, inds_tk, fc2_qp.bits, fc2_qp.group_size, fc2_qp.mode, false, s);
    try mlx.check(mlx.mlx_reshape(res, y_3d, &out_shape, 3, s));
}

// ── tests ──

// Bar: the Nemotron-H MoE block matches a host-side f32 reference of mlx-lm's
// `NemotronHMoE` — sigmoid router, top-k SELECTED on scores + bias but
// WEIGHTED by the unbiased scores, renormalized, x routed_scaling_factor,
// ReLU^2 experts, plus the ungated shared expert.
fn testNemotronMoeHostReference(comptime latent: bool) !void {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const t = std.testing;
    const allocator = t.allocator;
    const s = mlx.gpuStream();
    const T: usize = 6;
    const D: usize = 128;
    const L: usize = if (latent) 64 else D;
    const I: usize = 64;
    const SI: usize = 128;
    const E: usize = 8;
    const K: usize = 3;
    const scale: f32 = 2.5;

    var prng = std.Random.DefaultPrng.init(0x4e454d4f);
    const rnd = prng.random();
    const Q = struct {
        // Quantize host values to 4-bit/gs64 and hand back the dequantized
        // copy, so the reference sees the exact weights the kernel reads.
        fn make(alloc: std.mem.Allocator, r: std.Random, shape: []const c_int, st: mlx.mlx_stream, deq_out: []f32) ![3]mlx.mlx_array {
            const buf = try alloc.alloc(f32, deq_out.len);
            defer alloc.free(buf);
            for (buf) |*v| v.* = (r.float(f32) - 0.5) * 0.5;
            const a = mlx.mlx_array_new_data(buf.ptr, shape.ptr, @intCast(shape.len), .float32);
            defer _ = mlx.mlx_array_free(a);
            var triple = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(triple);
            try mlx.check(mlx.mlx_quantize(&triple, a, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{}, st));
            var out: [3]mlx.mlx_array = undefined;
            for (0..3) |i| {
                out[i] = mlx.mlx_array_new();
                errdefer for (out[0 .. i + 1]) |made| {
                    _ = mlx.mlx_array_free(made);
                };
                try mlx.check(mlx.mlx_vector_array_get(&out[i], triple, i));
            }
            var dq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(dq);
            try mlx.check(mlx.mlx_dequantize(&dq, out[0], out[1], out[2], mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{ .ctx = null }, .{ .value = .float32, .has_value = true }, st));
            try mlx.check(mlx.mlx_array_eval(dq));
            const p = mlx.mlx_array_data_float32(dq) orelse return error.MlxArrayDataNull;
            @memcpy(deq_out, p[0..deq_out.len]);
            return out;
        }
    };

    const w1 = try allocator.alloc(f32, E * I * L);
    defer allocator.free(w1);
    const w2 = try allocator.alloc(f32, E * L * I);
    defer allocator.free(w2);
    const su = try allocator.alloc(f32, SI * D);
    defer allocator.free(su);
    const sd = try allocator.alloc(f32, D * SI);
    defer allocator.free(sd);
    const fc1 = try Q.make(allocator, rnd, &.{ @intCast(E), @intCast(I), @intCast(L) }, s, w1);
    const fc2 = try Q.make(allocator, rnd, &.{ @intCast(E), @intCast(L), @intCast(I) }, s, w2);
    const up = try Q.make(allocator, rnd, &.{ @intCast(SI), @intCast(D) }, s, su);
    const down = try Q.make(allocator, rnd, &.{ @intCast(D), @intCast(SI) }, s, sd);
    defer for ([_][3]mlx.mlx_array{ fc1, fc2, up, down }) |tr| {
        for (tr) |a| _ = mlx.mlx_array_free(a);
    };

    var lat1_host: [L * D]f32 = @splat(0);
    var lat2_host: [D * L]f32 = @splat(0);
    const lat1 = try Q.make(allocator, rnd, &.{ @intCast(L), @intCast(D) }, s, &lat1_host);
    const lat2 = try Q.make(allocator, rnd, &.{ @intCast(D), @intCast(L) }, s, &lat2_host);
    defer for ([_][3]mlx.mlx_array{ lat1, lat2 }) |tr| {
        for (tr) |array| _ = mlx.mlx_array_free(array);
    };

    var router: [E * D]f32 = @splat(0);
    for (&router) |*v| v.* = (rnd.float(f32) - 0.5) * 0.5;
    var bias: [E]f32 = @splat(0);
    for (&bias) |*v| v.* = (rnd.float(f32) - 0.5) * 0.2;
    var xh: [T * D]f32 = @splat(0);
    for (&xh) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;

    // Router is dense and stored pre-transposed [D, E], as the loader leaves it.
    var router_t: [D * E]f32 = @splat(0);
    for (0..E) |e| for (0..D) |d| {
        router_t[d * E + e] = router[e * D + d];
    };
    const router_arr = mlx.mlx_array_new_data(&router_t, &[_]c_int{ @intCast(D), @intCast(E) }, 2, .float32);
    defer _ = mlx.mlx_array_free(router_arr);
    const bias_arr = mlx.mlx_array_new_data(&bias, &[_]c_int{@intCast(E)}, 1, .float32);
    defer _ = mlx.mlx_array_free(bias_arr);
    const x = mlx.mlx_array_new_data(&xh, &[_]c_int{ 1, @intCast(T), @intCast(D) }, 3, .float32);
    defer _ = mlx.mlx_array_free(x);

    const nm = NemotronMoeWeights{
        .router_w = router_arr,
        .router_s = mlx.mlx_array_new(),
        .router_b = mlx.mlx_array_new(),
        .expert_bias = bias_arr,
        .fc1_w = fc1[0],
        .fc1_s = fc1[1],
        .fc1_b = fc1[2],
        .fc2_w = fc2[0],
        .fc2_s = fc2[1],
        .fc2_b = fc2[2],
        .fc1_latent_w = if (latent) lat1[0] else .{ .ctx = null },
        .fc1_latent_s = if (latent) lat1[1] else .{ .ctx = null },
        .fc1_latent_b = if (latent) lat1[2] else .{ .ctx = null },
        .fc2_latent_w = if (latent) lat2[0] else .{ .ctx = null },
        .fc2_latent_s = if (latent) lat2[1] else .{ .ctx = null },
        .fc2_latent_b = if (latent) lat2[2] else .{ .ctx = null },
        .shared = .{ .up_w = up[0], .up_s = up[1], .up_b = up[2], .down_w = down[0], .down_s = down[1], .down_b = down[2] },
    };
    var cfg = ModelConfig{};
    cfg.moe_latent_size = if (latent) L else 0;
    cfg.quant_bits = 4;
    cfg.quant_group_size = 64;
    cfg.num_experts = E;
    cfg.num_experts_per_tok = K;
    cfg.moe_intermediate_size = I;
    cfg.router_scaling_factor = scale;

    const got = try nemotronMoe(x, &nm, &cfg, s);
    defer _ = mlx.mlx_array_free(got);
    try t.expectEqualSlices(c_int, &.{ 1, @intCast(T), @intCast(D) }, mlx.getShape(got));
    try mlx.check(mlx.mlx_array_eval(got));
    const gp = mlx.mlx_array_data_float32(got) orelse return error.MlxArrayDataNull;

    const relu2 = struct {
        fn f(v: f32) f32 {
            const r = @max(v, 0);
            return r * r;
        }
    }.f;
    var max_err: f32 = 0;
    var max_err_one: f32 = 0;
    var max_ref: f32 = 0;
    nemotron_gqmv_engaged = false;
    for (0..T) |ti| {
        const xt = xh[ti * D ..][0..D];
        var sig: [E]f32 = undefined;
        var sel: [E]f32 = undefined;
        for (0..E) |e| {
            var l: f32 = 0;
            for (0..D) |d| l += xt[d] * router[e * D + d];
            sig[e] = 1.0 / (1.0 + @exp(-l));
            sel[e] = sig[e] + bias[e];
        }
        var chosen: [K]usize = undefined;
        var taken: [E]bool = @splat(false);
        for (0..K) |k| {
            var best: usize = E;
            for (0..E) |e| if (!taken[e] and (best == E or sel[e] > sel[best])) {
                best = e;
            };
            taken[best] = true;
            chosen[k] = best;
        }
        var wsum: f32 = 0;
        for (chosen) |e| wsum += sig[e];
        var expert_x: [L]f32 = @splat(0);
        if (latent) {
            for (0..L) |l| for (0..D) |d| {
                expert_x[l] += xt[d] * lat1_host[l * D + d];
            };
        } else @memcpy(&expert_x, xt);
        var expert_out: [L]f32 = @splat(0);
        for (chosen) |e| {
            const wt = sig[e] / wsum * scale;
            var h: [I]f32 = undefined;
            for (0..I) |i| {
                var a: f32 = 0;
                for (0..L) |d| a += expert_x[d] * w1[(e * I + i) * L + d];
                h[i] = relu2(a);
            }
            for (0..L) |d| {
                var a: f32 = 0;
                for (0..I) |i| a += h[i] * w2[(e * L + d) * I + i];
                expert_out[d] += wt * a;
            }
        }
        var out: [D]f32 = @splat(0);
        if (latent) {
            for (0..D) |d| for (0..L) |l| {
                out[d] += expert_out[l] * lat2_host[d * L + l];
            };
        } else @memcpy(&out, &expert_out);
        var hs: [SI]f32 = undefined;
        for (0..SI) |i| {
            var a: f32 = 0;
            for (0..D) |d| a += xt[d] * su[i * D + d];
            hs[i] = relu2(a);
        }
        for (0..D) |d| {
            var a: f32 = 0;
            for (0..SI) |i| a += hs[i] * sd[d * SI + i];
            out[d] += a;
            max_err = @max(max_err, @abs(out[d] - gp[ti * D + d]));
            max_ref = @max(max_ref, @abs(out[d]));
        }
        // The same token alone takes the one-token (gather-qmv) path.
        var x_one = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_one);
        const tc: c_int = @intCast(ti);
        try mlx.check(mlx.mlx_slice(&x_one, x, &[_]c_int{ 0, tc, 0 }, 3, &[_]c_int{ 1, tc + 1, @intCast(D) }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        const got_one = try nemotronMoe(x_one, &nm, &cfg, s);
        defer _ = mlx.mlx_array_free(got_one);
        try mlx.check(mlx.mlx_array_eval(got_one));
        const g1 = mlx.mlx_array_data_float32(got_one) orelse return error.MlxArrayDataNull;
        for (0..D) |d| max_err_one = @max(max_err_one, @abs(out[d] - g1[d]));
    }
    // Off Metal the kernel declines and the composed gather_qmm path must meet the same bar.
    try t.expectEqual(mlx.metalKernelsAvailable(), nemotron_gqmv_engaged);
    try t.expect(std.math.isFinite(max_err) and std.math.isFinite(max_err_one));
    try t.expect(max_ref > 0.1);
    try t.expect(max_err < 1e-3 * max_ref);
    try t.expect(max_err_one < 1e-3 * max_ref);

    // Sorted gather_qmm path (windows wider than the in-place seam), same reference.
    {
        nemotron_gather_max_rows = 0;
        defer nemotron_gather_max_rows = 8;
        const got_sorted = try nemotronMoe(x, &nm, &cfg, s);
        defer _ = mlx.mlx_array_free(got_sorted);
        try mlx.check(mlx.mlx_array_eval(got_sorted));
        const gs_p = mlx.mlx_array_data_float32(got_sorted) orelse return error.MlxArrayDataNull;
        var max_err_sorted: f32 = 0;
        for (0..T * D) |i| max_err_sorted = @max(max_err_sorted, @abs(gp[i] - gs_p[i]));
        try t.expect(max_err_sorted < 2e-3 * max_ref);

        if (!latent) {
            // Padded up to 4 rows per expert for the streaming kernel; the pad rows are dropped.
            try t.expectEqual(@as(c_int, 0), moeStreamPadRows(2 * E - 1, E));
            try t.expectEqual(@as(c_int, 0), moeStreamPadRows(4 * E, E));
            try t.expectEqual(@as(c_int, 4 * E - T * K), moeStreamPadRows(T * K, E));
            const parts = try nemotronMoeExperts(x, &nm, &cfg, true, s);
            defer parts.deinit();
            const got_pad = try nemotronMoeCombine(x, &parts, &nm, &cfg, s, x);
            defer _ = mlx.mlx_array_free(got_pad);
            try mlx.check(mlx.mlx_array_eval(got_pad));
            const gpad = mlx.mlx_array_data_float32(got_pad) orelse return error.MlxArrayDataNull;
            var max_err_pad: f32 = 0;
            for (0..T * D) |i| max_err_pad = @max(max_err_pad, @abs(gp[i] - gpad[i]));
            try t.expect(max_err_pad < 2e-3 * max_ref);
        }
    }
}

test "nemotronMoe matches a host reference of NemotronHMoE" {
    try testNemotronMoeHostReference(false);
    try testNemotronMoeHostReference(true);
}

test "nemotron_h: Super checkpoint dense and affine language layouts load and forward identically" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const t = std.testing;
    const allocator = t.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.gpuStream();
    const Fix = @import("../../engines/mlx/spec/dflash/dflash.zig").TinyFix;
    const native =
        \\{"model_type":"nemotron_h_omni","llm_config":{"model_type":"nemotron_h",
        \\"hidden_size":64,"intermediate_size":128,"vocab_size":128,
        \\"num_attention_heads":4,"num_key_value_heads":2,"head_dim":16,
        \\"layers_block_type":["attention"],"tie_word_embeddings":false}}
    ;
    const P = struct {
        fn weight(map: mlx.mlx_map_string_to_array, name: []const u8, rows: usize, cols: usize, seed: usize, quantized: bool, st: mlx.mlx_stream) !void {
            if (!quantized) return Fix.putW(map, name, rows, cols, seed, st);
            const dense = try Fix.bf16Arr(rows, cols, seed, st);
            defer _ = mlx.mlx_array_free(dense);
            var triple = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(triple);
            try mlx.check(mlx.mlx_quantize(&triple, dense, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{}, st));
            const base = name[0 .. name.len - ".weight".len];
            for ([_][]const u8{ ".weight", ".scales", ".biases" }, 0..) |suffix, i| {
                var a = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(a);
                try mlx.check(mlx.mlx_vector_array_get(&a, triple, i));
                var buf: [160]u8 = undefined;
                try Fix.put(map, try std.fmt.bufPrint(&buf, "{s}{s}", .{ base, suffix }), a);
            }
        }
    };
    for ([_]bool{ false, true }) |quantized| {
        var flat_logits: ?[]f32 = null;
        defer if (flat_logits) |values| allocator.free(values);
        for ([_][]const u8{ "backbone", "language_model.backbone" }) |prefix| {
            var tmp = t.tmpDir(.{});
            defer tmp.cleanup();
            var path_buf: [512]u8 = undefined;
            const path_len = try tmp.dir.realPath(io, &path_buf);
            const dir = path_buf[0..path_len];
            const map = mlx.mlx_map_string_to_array_new();
            defer _ = mlx.mlx_map_string_to_array_free(map);
            const meta = mlx.mlx_map_string_to_string_new();
            defer _ = mlx.mlx_map_string_to_string_free(meta);
            var name: [160]u8 = undefined;
            try P.weight(map, try std.fmt.bufPrint(&name, "{s}.embeddings.weight", .{prefix}), 128, 64, 1, quantized, s);
            try Fix.putNorm(map, try std.fmt.bufPrint(&name, "{s}.norm_f.weight", .{prefix}), 64, s);
            try Fix.putNorm(map, try std.fmt.bufPrint(&name, "{s}.layers.0.norm.weight", .{prefix}), 64, s);
            const heads = [_]struct { name: []const u8, rows: usize, seed: usize }{
                .{ .name = "q_proj", .rows = 64, .seed = 10 },
                .{ .name = "k_proj", .rows = 32, .seed = 20 },
                .{ .name = "v_proj", .rows = 32, .seed = 30 },
                .{ .name = "o_proj", .rows = 64, .seed = 40 },
            };
            for (heads) |head| try P.weight(map, try std.fmt.bufPrint(&name, "{s}.layers.0.mixer.{s}.weight", .{ prefix, head.name }), head.rows, 64, head.seed, quantized, s);
            const head_name = if (std.mem.eql(u8, prefix, "backbone")) "lm_head.weight" else "language_model.lm_head.weight";
            try P.weight(map, head_name, 128, 64, 77, quantized, s);
            const file = try std.fmt.allocPrintSentinel(allocator, "{s}/model.safetensors", .{dir}, 0);
            defer allocator.free(file);
            try mlx.check(mlx.mlx_save_safetensors(file.ptr, map, meta));
            var config = try model_mod.parseConfigFromJson(allocator, native);
            defer config.deinit(allocator);
            if (quantized) {
                config.quant_bits = 4;
                config.quant_group_size = 64;
            }
            var weights = try model_mod.loadModelWeights(io, allocator, dir, &config, false);
            defer weights.deinit();
            model_mod.resolveWeightPrefix(&config, &weights);
            try t.expectEqualStrings(prefix, config.weight_prefix);
            var xfm = try Transformer.init(io, allocator, config, &weights);
            defer xfm.deinit();
            try t.expectEqual(weights.get(head_name).?.ctx, xfm.lm_head_w.ctx);
            try t.expectEqual(quantized, xfm.lm_head_s.ctx != null);
            if (quantized) {
                const base = head_name[0 .. head_name.len - ".weight".len];
                try t.expectEqual(weights.get(try std.fmt.bufPrint(&name, "{s}.scales", .{base})).?.ctx, xfm.lm_head_s.ctx);
                try t.expectEqual(weights.get(try std.fmt.bufPrint(&name, "{s}.biases", .{base})).?.ctx, xfm.lm_head_b.ctx);
            }
            const ids = mlx.mlx_array_new_data(&[_]i32{ 1, 4, 7 }, &[_]c_int{ 1, 3 }, 2, .int32);
            defer _ = mlx.mlx_array_free(ids);
            const logits = try xfm.forward(ids);
            defer _ = mlx.mlx_array_free(logits);
            const values = try Fix.readF32(logits, allocator, s);
            defer allocator.free(values);
            var magnitude: f32 = 0;
            for (values) |v| {
                try t.expect(std.math.isFinite(v));
                magnitude = @max(magnitude, @abs(v));
            }
            try t.expect(magnitude > 0.01);
            if (flat_logits) |flat| try t.expectEqualSlices(f32, flat, values) else flat_logits = try allocator.dupe(f32, values);
        }
    }
}
