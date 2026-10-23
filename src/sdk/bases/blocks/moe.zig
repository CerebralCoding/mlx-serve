//! The generic routed-expert MLP: a router (softmax top-k renormed, sigmoid + selection-only `expert_bias`,
//! group-limited, or top-k on biased fp32 logits), the experts (one-token fused kernels, the batched-rows kernels,
//! the sorted `gather_qmm` prefill, the stock gather) and an optional shared expert, gated or always added.
//! Serves qwen3_5_moe, laguna, flint, kolibri1, hy_v3 and lfm2_moe as the legacy MoE forward does.
//! copied_from: sdk/blocks/moe/moe.zig (`moeMLP2WithRouter`, `moeRouterLogits`, the shared-expert helpers, without
//! the exl3, GLM, tensor-parallel, qwen4 verify-feature, Gemma 4 sigma-MoE and gpt_oss bias arms),
//! sdk/blocks/moe/decode.zig (the decode arms), sdk/blocks/moe/routing.zig (`computeMoeRouting`,
//! `computeHy3Routing`), sdk/blocks/mlp/activations.zig (`computeGeglu`'s silu arms).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const log = @import("../../../host/log.zig");
const block = @import("../block.zig");
const linear = @import("linear.zig");
const compiled = @import("compiled.zig");
const QuantParams = @import("../../quant/params.zig").QuantParams;
const quant_abi = @import("../../../abi/quant.zig");
const model = @import("../../../engines/mlx/model.zig");
const envFlagCached = @import("../../../engines/mlx/env_transformer.zig").envFlagCached;
const routing = @import("../../blocks/moe/routing.zig");
const router_kernel = @import("../../blocks/moe/router_kernel.zig");
const moe_decode = @import("../../blocks/moe/decode.zig");
const moe_fp4 = @import("../../blocks/moe/moe_fp4.zig");
const moe_cuda = @import("../../blocks/moe/moe_cuda.zig");
const moe_prefill_cuda = @import("../../blocks/moe/moe_prefill_cuda.zig");
const moe_affine4 = @import("../../../plugins/qwen4_exp/moe_affine4.zig");
const gather_qmv = @import("../../blocks/moe/gather_qmv.zig");
const gather_qmv_rows = @import("../../blocks/moe/gather_qmv_rows.zig");
const gather_qmm_nax = @import("../../blocks/moe/gather_qmm_nax.zig");
const down_reduce = @import("../../blocks/moe/down_reduce.zig");
const verify_kernels = @import("../../blocks/moe/verify_kernels.zig");
const gatherExpertMm = @import("../../blocks/moe/experts.zig").gatherExpertMm;
const verifyQmmNaxAvailable = @import("../../blocks/linear/verify_qmm_nax.zig").verifyQmmNaxAvailable;
const fused_act = @import("../../blocks/mlp/fused_activations.zig");
const clampedSwiGLU = @import("../../blocks/mlp/activations.zig").clampedSwiGLU;
const sigmoidTable = @import("../../blocks/hc/hc_prefill.zig").sigmoidTable;

const Array = ops.Array;
const Graph = ops.Graph;
const Ctx = block.Ctx;
const Linear = linear.Linear;

/// An expert bank `[E, out, in]` and its quant geometry.
pub const Bank = struct {
    w: Array,
    s: Array = .{ .ctx = null },
    b: Array = .{ .ctx = null },
    qp: QuantParams,
    /// ModelOpt NVFP4: each expert's global scale `[E]` on this projection's output; null = none.
    expert_scale: Array = .{ .ctx = null },

    fn a4(bank: *const Bank) moe_affine4.Bank {
        return .{ .w = bank.w, .s = bank.s, .b = bank.b };
    }
};

/// How the router picks and weights the top-k.
pub const Router = union(enum) {
    /// Softmax over the experts, top-k renormalized (qwen3_5_moe).
    softmax,
    /// Sigmoid scores; top-k selected on scores + `expert_bias` `[E]` f32, weighted by the unbiased scores
    /// (laguna, flint, hy_v3, lfm2_moe), group-limited when `n_group > 1`.
    sigmoid_bias: struct { expert_bias: Array, norm: bool, scale: f32, n_group: c_int = 1, topk_group: c_int = 1 },
    /// Top-k on the f32 logits + `expert_bias`, weights sigmoid(logits) (kolibri1; the router weight is f32 `[in, E]`).
    logit_bias: Array,
};

/// Routed experts a quant format runs as one SwiGLU dispatch (`QuantTable.swiglu`, EXL3): its three banks as the
/// format loaded them; the `Bank`s stay unbuilt.
pub const FormatBanks = struct {
    table: *const quant_abi.QuantTable,
    gate: quant_abi.Weight,
    up: quant_abi.Weight,
    down: quant_abi.Weight,
};

/// The shared expert; `gate_logit` unset = always added.
pub const SharedExpert = struct {
    gate: Linear,
    up: Linear,
    down: Linear,
    /// One logit per token whose sigmoid scales the shared expert (`shared_expert_gate`).
    gate_logit: Linear = .{},
    /// `gate_logit`'s weight as one contiguous `[hidden]` vector: the softmax router kernel dots it in at one row.
    gate_vec: Array = .{ .ctx = null },

    fn gated(se: *const SharedExpert) bool {
        return se.gate_logit.w.ctx != null;
    }
};

pub const Moe = struct {
    pub const kind: block.Kind = .mlp;
    pub const supports_batched_rows = true;
    pub const Weights = struct {
        router: Linear,
        route: Router,
        gate: Bank,
        up: Bank,
        down: Bank,
        shared: ?SharedExpert = null,
        /// One-expert views of the affine-4 banks the one-token kernels bind.
        bank_views: ?moe_affine4.Views = null,
        top_k: c_int,
        experts: c_int,
        /// A whole-number SwiGLU clamp (0 = none); null = a clamp no kernel serves.
        clamp: ?u32 = 0,
        /// The clamp as the composed chain applies it (`swiglu_limit` with `swiglu_clamp`); 0 = none.
        clamp_limit: f32 = 0,
        /// The routing chain compiled (`compileRouting`); null = op by op.
        route_fn: compiled.Closure = .{ .ctx = null },
        /// Set: the experts are the format's (`gate`/`up`/`down` unread).
        format_banks: ?FormatBanks = null,
    };

    pub fn forward(w: *const Weights, c: *Ctx, x: Array) !Array {
        return mlp(w, c.g, x);
    }
};

// ── routing ──

const RoutePayload = struct { k: c_int, norm: bool, scale: f32, sigmoid: bool, s: ops.Stream };

fn routeBody(p: *const RoutePayload, in: []const Array, out: *[2]Array) anyerror!void {
    const r = if (p.sigmoid)
        try routing.hy3RoutingChain(in[0], in[1], p.k, p.norm, p.scale, p.s)
    else
        try routing.moeRoutingChain(in[0], p.k, p.s);
    out.* = .{ r.inds, r.norm_scores };
}

/// The routing chain compiled per input shape, as the legacy load compiles it (the sigmoid form for a
/// `sigmoid_bias` router).
pub fn compileRouting(w: *const Moe.Weights, s: ops.Stream) compiled.Closure {
    const p: RoutePayload = switch (w.route) {
        .softmax => .{ .k = w.top_k, .norm = true, .scale = 1.0, .sigmoid = false, .s = s },
        .sigmoid_bias => |r| .{ .k = w.top_k, .norm = r.norm, .scale = r.scale, .sigmoid = true, .s = s },
        .logit_bias => return .{ .ctx = null },
    };
    return compiled.compile(RoutePayload, p, 2, routeBody, false);
}

const Routed = struct { inds: Array, scores: Array, gate_logit: ?Array = null };

fn adoptRouting(g: *Graph, r: anytype) !Routed {
    const inds = g.adopt(r.inds) catch |e| {
        _ = mlx.mlx_array_free(r.norm_scores);
        if (r.gate_logit.ctx != null) _ = mlx.mlx_array_free(r.gate_logit);
        return e;
    };
    const scores = g.adopt(r.norm_scores) catch |e| {
        if (r.gate_logit.ctx != null) _ = mlx.mlx_array_free(r.gate_logit);
        return e;
    };
    const gl: ?Array = if (r.gate_logit.ctx != null) try g.adopt(r.gate_logit) else null;
    return .{ .inds = inds, .scores = scores, .gate_logit = gl };
}

fn routerLogits(w: *const Moe.Weights, parent: *Graph, x: Array) !Array {
    if (w.route != .logit_bias) return w.router.forward(parent, x);
    var t = parent.scope();
    defer t.deinit();
    return parent.keep(try t.matmul(try t.astype(x, .float32), w.router.w));
}

fn route(w: *const Moe.Weights, g: *Graph, logits: Array, fold: ?router_kernel.RouterGate) !Routed {
    switch (w.route) {
        .logit_bias => |bias| {
            if (try router_kernel.moeRouterTopK(g.s, logits, bias, w.top_k, .logit_bias, false, 1.0, .float32, 0, 0, null)) |r| return adoptRouting(g, r);
            return adoptRouting(g, try routing.kolibriRoutingChain(logits, bias, w.top_k, g.s));
        },
        .sigmoid_bias => |r| {
            if (r.n_group > 1) return adoptRouting(g, try routing.groupLimitedRouting(logits, r.expert_bias, w.top_k, r.n_group, r.topk_group, r.norm, r.scale, g.s));
            if (try router_kernel.moeRouterTopK(g.s, logits, r.expert_bias, w.top_k, .sigmoid_bias, r.norm, r.scale, .bfloat16, 0, 0, null)) |f| return adoptRouting(g, f);
            if (w.route_fn.ctx != null) if (try compiled.apply(g, w.route_fn, &.{ logits, r.expert_bias }, 2)) |o| return .{ .inds = o[0], .scores = o[1] };
            return adoptRouting(g, try routing.hy3RoutingChain(logits, r.expert_bias, w.top_k, r.norm, r.scale, g.s));
        },
        .softmax => {
            if (try router_kernel.moeRouterTopK(g.s, logits, .{ .ctx = null }, w.top_k, .softmax, true, 1.0, ops.dtype(logits), 0, 0, fold)) |r| return adoptRouting(g, r);
            if (w.route_fn.ctx != null) if (try compiled.apply(g, w.route_fn, &.{logits}, 2)) |o| return .{ .inds = o[0], .scores = o[1] };
            return adoptRouting(g, try routing.moeRoutingChain(logits, w.top_k, g.s));
        },
    }
}

// ── the MLP ──

/// `act(gate) * up`: SiLU, clamped where the pack clamps.
fn geglu(w: *const Moe.Weights, parent: *Graph, gate: Array, up: Array) !Array {
    const s = parent.s;
    if (w.clamp_limit > 0) {
        if (try fused_act.fusedClampedSwiGLU(s, gate, up, w.clamp_limit)) |y| return parent.adopt(y);
        return parent.adopt(try clampedSwiGLU(s, gate, up, w.clamp_limit));
    }
    if (try fused_act.fusedSwiGLU(s, gate, up)) |y| return parent.adopt(y);
    var t = parent.scope();
    defer t.deinit();
    return parent.keep(try t.mul(try t.mul(gate, try t.sigmoid(gate)), up));
}

pub fn mlp(w: *const Moe.Weights, parent: *Graph, x: Array) !Array {
    var t = parent.scope();
    defer t.deinit();
    const g = &t;
    const logits = try routerLogits(w, g, x);

    // At one row the shared expert's gate logit rides the softmax router kernel.
    const fold: ?router_kernel.RouterGate = blk: {
        const se = w.shared orelse break :blk null;
        if (w.route != .softmax or !se.gated() or se.gate_vec.ctx == null) break :blk null;
        const lsh = ops.shape(logits);
        var rows: usize = 1;
        for (lsh[0 .. lsh.len - 1]) |d| rows *= @intCast(d);
        if (rows != 1 or mlx.mlx_array_size(x) != mlx.mlx_array_size(se.gate_vec)) break :blk null;
        break :blk .{ .x = x, .w = se.gate_vec };
    };
    var r = try route(w, g, logits, fold);
    const shared_down: ?Array = if (r.gate_logit != null) try sharedDown(w, g, &w.shared.?, x) else null;
    // The group-limited chain hands its weights back in f32; they land in the activation dtype.
    if (ops.dtype(r.scores) != ops.dtype(x)) r.scores = try g.astype(r.scores, ops.dtype(x));

    const xs = ops.shape(x);
    const B = xs[0];
    const S = xs[1];
    const D = xs[xs.len - 1];
    const K = ops.shape(r.inds)[ops.shape(r.inds).len - 1];
    const total: c_int = B * S * K;
    const gq = w.gate.qp;
    const uq = w.up.qp;
    const dq = w.down.qp;
    const has_gate_scale = w.gate.expert_scale.ctx != null;
    const do_sort = B * S > 1 or total >= 64 or has_gate_scale;

    var out: Array = undefined;
    var reduced = false;
    var shared_fused = false;
    const ungated_a4: ?moe_affine4.Ungated = if (B * S != 1) null else ungatedA4(w, gq.group_size);
    const fp4_shared: ?moe_affine4.Shared = if (B * S == 1) if (shared_down) |sd| .{ .down = sd, .logit = r.gate_logit.? } else null else null;
    const cuda_fmt = if (mlx.cudaAvailable()) gather_qmv_rows.moeCudaFormat(B * S, gq.mode, uq.mode, dq.mode, true, false, w.up.expert_scale.ctx != null, w.clamp_limit > 0) else null;
    if (w.format_banks) |fb| {
        out = try formatSwiglu(w, &fb, g, x, r);
        reduced = true;
    } else if (moe_decode.moeFp4DecodeEligible(B * S, false, w.clamp, ops.dtype(x), gq.mode, uq.mode, dq.mode) and
        try fp4Decode(w, g, &out, x, r, fp4_shared))
    {
        reduced = true;
        shared_fused = fp4_shared != null;
    } else if (B * S == 1 and (w.clamp orelse 1) == 0 and gq.mode == .affine and uq.mode == .affine and dq.mode == .affine and
        gq.bits == 4 and uq.bits == 4 and dq.bits == 4 and gq.group_size == uq.group_size and gq.group_size == dq.group_size and
        try affine4Decode(w, g, &out, x, r, if (shared_down) |sd| .{ .down = sd, .logit = r.gate_logit.? } else null, ungated_a4))
    {
        reduced = true;
        shared_fused = shared_down != null or ungated_a4 != null;
    } else if (gather_qmv_rows.moeDecodeDispatchArm(B, S, K, false, (w.clamp orelse 1) != 0) == .rows and
        useGatherQmv(w) and w.gate.s.ctx != null and w.up.s.ctx != null and w.down.s.ctx != null and
        try gatherQmvRows(w, g, &out, x, r, D, K, B, S))
    {
        gather_qmv_rows.moe_rows_fused_layers +%= 1;
        reduced = true;
    } else if (cuda_fmt != null and try cudaDecode(w, g, &out, x, r, cuda_fmt.?)) {
        reduced = true;
    } else if (do_sort) {
        gather_qmv_rows.moe_sorted_layers +%= 1;
        const s = try sorted(w, g, x, r, B, S, D, K);
        out = s.out;
        reduced = s.reduced;
    } else if (B * S == 1 and useGatherQmv(w) and w.gate.s.ctx != null and w.up.s.ctx != null and w.down.s.ctx != null and
        try gatherQmvDecode(w, g, &out, &reduced, x, r, D, K, B, S))
    {} else if (B * S == 1 and batchedForced() and w.gate.s.ctx != null and w.up.s.ctx != null and w.down.s.ctx != null) {
        out = try batchedTake(w, g, x, r, B, S, D, K);
    } else {
        const x5 = try g.reshape(x, &.{ B, S, 1, 1, D });
        const gate = try g.call(mlx.mlx_squeeze, .{try expertMm(g, x5, &w.gate, .{ .ctx = null }, r.inds, false)});
        const up = try g.call(mlx.mlx_squeeze, .{try expertMm(g, x5, &w.up, .{ .ctx = null }, r.inds, false)});
        const act = try g.expandDims(try geglu(w, g, gate, up), -2);
        out = try g.call(mlx.mlx_squeeze, .{try expertMm(g, act, &w.down, .{ .ctx = null }, r.inds, false)});
    }

    const sum = if (reduced) out else try g.sum(try g.mul(out, try g.expandDims(r.scores, -1)), -2, false);
    if (shared_fused) return parent.keep(sum);
    const se = w.shared orelse return parent.keep(sum);
    if (!se.gated()) return parent.keep(try g.add(sum, try sharedDown(w, g, &se, x)));
    const down = shared_down orelse try sharedDown(w, g, &se, x);
    const logit = r.gate_logit orelse try se.gate_logit.forward(g, x);
    return parent.keep(try g.add(sum, try g.mul(try g.sigmoid(logit), down)));
}

/// The format's routed SwiGLU over the top-k, weighted and summed (the engine's exl3 arm).
fn formatSwiglu(w: *const Moe.Weights, fb: *const FormatBanks, g: *Graph, x: Array, r: Routed) !Array {
    const swiglu = fb.table.swiglu orelse return error.QuantFormatHasNoSwiglu;
    const limit = w.clamp orelse return error.QuantFormatClampUnsupported;
    var y: quant_abi.Array = .{};
    const v = swiglu(&fb.gate, &fb.up, &fb.down, .{ .ctx = x.ctx }, .{ .ctx = r.inds.ctx }, .{ .ctx = r.scores.ctx }, &.{ .limit = @intCast(limit) }, .{ .ctx = g.s.ctx }, &y);
    if (!v.isOk()) return error.QuantFormatSwigluFailed;
    return g.adopt(.{ .ctx = y.ctx });
}

fn sharedDown(w: *const Moe.Weights, parent: *Graph, se: *const SharedExpert, x: Array) !Array {
    var t = parent.scope();
    defer t.deinit();
    const g = &t;
    return parent.keep(try se.down.forward(g, try geglu(w, g, try se.gate.forward(g, x), try se.up.forward(g, x))));
}

/// An ungated shared expert the affine-4 kernels carry: 4-bit affine at the routed group size.
fn ungatedA4(w: *const Moe.Weights, group_size: u32) ?moe_affine4.Ungated {
    const se = w.shared orelse return null;
    if (se.gated() or se.gate.s.ctx == null) return null;
    for ([_]*const Linear{ &se.gate, &se.up, &se.down }) |l| {
        if (l.mode != .affine or l.bits != 4 or l.group_size != group_size) return null;
    }
    return .{
        .gate = .{ .w = se.gate.w, .s = se.gate.s, .b = se.gate.b },
        .up = .{ .w = se.up.w, .s = se.up.s, .b = se.up.b },
        .down = .{ .w = se.down.w, .s = se.down.s, .b = se.down.b },
    };
}

var batched_force_env: ?bool = null;

/// `MLX_SERVE_MOE_BATCHED_DECODE=1` (an A/B lever no arch defaults to), beaten by `MLX_SERVE_MOE_GATHER_DECODE=1`.
fn batchedForced() bool {
    if (moe_decode.gatherDecodeForcedStock()) return false;
    return envFlagCached(&batched_force_env, "MLX_SERVE_MOE_BATCHED_DECODE");
}

fn useGatherQmv(w: *const Moe.Weights) bool {
    if (moe_decode.gatherDecodeForcedStock()) return false;
    if (batchedForced()) return true;
    if (w.gate.qp.mode == .gguf) return false;
    if (w.clamp == null) return false;
    if (!gather_qmv.gatherQmvGateUpEnabled()) return false;
    return w.gate.qp.bits == w.up.qp.bits and w.gate.qp.group_size == w.up.qp.group_size and w.gate.qp.mode == w.up.qp.mode;
}

fn expertMm(g: *Graph, x: Array, bank: *const Bank, lhs: Array, rhs: Array, is_sorted: bool) !Array {
    var y = mlx.mlx_array_new();
    gatherExpertMm(&y, x, bank.w, bank.s, bank.b, lhs, rhs, bank.qp.bits, bank.qp.group_size, bank.qp.mode, is_sorted, g.s) catch |e| {
        _ = mlx.mlx_array_free(y);
        return e;
    };
    return g.adopt(y);
}

/// Owned by `g`, or false (`out` untouched) when the kernels decline.
fn put(g: *Graph, out: *Array, y: ?Array, shape: []const c_int) !bool {
    const v = y orelse return false;
    out.* = try g.reshape(try g.adopt(v), shape);
    return true;
}

fn fp4Decode(w: *const Moe.Weights, g: *Graph, out: *Array, x: Array, r: Routed, shared: ?moe_affine4.Shared) !bool {
    const xs = ops.shape(x);
    const rows = xs[0] * xs[1];
    const k = ops.shape(r.inds)[ops.shape(r.inds).len - 1];
    const x2 = try g.reshape(x, &.{ rows, xs[2] });
    const ind = try g.astype(try g.reshape(r.inds, &.{ rows, k }), .uint32);
    const sc = try g.reshape(r.scores, &.{ rows, k });
    const sigtab = try fused_act.swigluSigTable(g.s, ops.dtype(x), std.heap.c_allocator);
    const scales: moe_fp4.Scales = .{
        .gate = if (w.gate.expert_scale.ctx != null) w.gate.expert_scale else null,
        .up = if (w.up.expert_scale.ctx != null) w.up.expert_scale else null,
        .clamp = w.clamp orelse return false,
    };
    const y = try moe_fp4.decode(g.s, x2, .{ .w = w.gate.w, .s = w.gate.s }, .{ .w = w.up.w, .s = w.up.s }, .{ .w = w.down.w, .s = w.down.s }, w.bank_views, scales, shared, ind, sc, sigtab, w.gate.qp.mode, w.gate.qp.group_size);
    return put(g, out, y, &.{ xs[0], xs[1], xs[2] });
}

fn affine4Decode(w: *const Moe.Weights, g: *Graph, out: *Array, x: Array, r: Routed, shared: ?moe_affine4.Shared, ungated: ?moe_affine4.Ungated) !bool {
    // The sigmoid table covers 16-bit patterns only.
    const xd = ops.dtype(x);
    if (xd != .bfloat16 and xd != .float16) return false;
    const xs = ops.shape(x);
    const k = ops.shape(r.inds)[ops.shape(r.inds).len - 1];
    const x1 = try g.reshape(x, &.{xs[2]});
    const ind = try g.astype(try g.reshape(r.inds, &.{k}), .uint32);
    const sc = try g.reshape(r.scores, &.{k});
    const sigtab = try fused_act.swigluSigTable(g.s, xd, std.heap.c_allocator);
    const y = try moe_affine4.decode(g.s, x1, w.gate.a4(), w.up.a4(), w.down.a4(), w.bank_views, shared, ungated, ind, sc, sigtab, w.gate.qp.group_size);
    return put(g, out, y, &.{ xs[0], xs[1], xs[2] });
}

fn cudaDecode(w: *const Moe.Weights, g: *Graph, out: *Array, x: Array, r: Routed, fmt: moe_cuda.Format) !bool {
    const xs = ops.shape(x);
    const rows = xs[0] * xs[1];
    const k = ops.shape(r.inds)[ops.shape(r.inds).len - 1];
    const x2 = try g.reshape(x, &.{ rows, xs[2] });
    const ind = try g.astype(try g.reshape(r.inds, &.{ rows, k }), .uint32);
    const sc = try g.reshape(r.scores, &.{ rows, k });
    const gscale: ?Array = if (w.gate.expert_scale.ctx != null) w.gate.expert_scale else null;
    const y = try moe_cuda.decode(g.s, x2, .{ .w = w.gate.w, .s = w.gate.s }, .{ .w = w.up.w, .s = w.up.s }, .{ .w = w.down.w, .s = w.down.s }, gscale, ind, sc, fmt);
    return put(g, out, y, &.{ xs[0], xs[1], xs[2] });
}

fn gatherQmvRows(w: *const Moe.Weights, g: *Graph, out: *Array, x: Array, r: Routed, D: c_int, K: c_int, batch: c_int, seq: c_int) !bool {
    const B = batch * seq;
    if (!down_reduce.gatherQmvDownReduceRowsEligible(model.HiddenAct.silu, w.gate.qp, w.up.qp, w.down.qp, D)) return false;
    const ind = try g.astype(try g.reshape(r.inds, &.{ B, K }), .uint32);
    const x2 = try g.reshape(x, &.{ B, D });
    const gscale: ?Array = if (w.gate.expert_scale.ctx != null) w.gate.expert_scale else null;
    const act = try g.adopt((try gather_qmv_rows.gatherQmvGateUpRows(g.s, x2, w.gate.w, w.gate.s, w.gate.b, w.up.w, w.up.s, w.up.b, ind, w.gate.qp.bits, w.gate.qp.group_size, w.gate.qp.mode, gscale)) orelse return false);
    const sc = try g.reshape(r.scores, &.{ B, K });
    const sum = (try down_reduce.gatherQmvDownReduceRows(g.s, act, w.down.w, w.down.s, w.down.b, ind, sc, w.down.qp.bits, w.down.qp.group_size, w.down.qp.mode)) orelse return false;
    const hidden = ops.shape(sum)[1];
    _ = try put(g, out, sum, &.{ batch, seq, hidden });
    if (seq > 1 and !gather_qmv_rows.moe_verify_rows_logged) {
        gather_qmv_rows.moe_verify_rows_logged = true;
        log.info("[moe-rows] verify rows engaged: S={d} K={d}\n", .{ seq, K });
    }
    return true;
}

fn gatherQmvDecode(w: *const Moe.Weights, g: *Graph, out: *Array, reduced: *bool, x: Array, r: Routed, D: c_int, K: c_int, B: c_int, S: c_int) !bool {
    const ind = try g.astype(try g.reshape(r.inds, &.{K}), .uint32);
    const x1 = try g.reshape(x, &.{D});
    const gq = w.gate.qp;
    const uq = w.up.qp;
    const dq = w.down.qp;
    var act: ?Array = null;
    if (gq.bits == uq.bits and gq.group_size == uq.group_size and gq.mode == uq.mode) {
        if (try gather_qmv.gatherQmvGateUp(g.s, x1, w.gate.w, w.gate.s, w.gate.b, w.up.w, w.up.s, w.up.b, ind, gq.bits, gq.group_size, gq.mode, w.clamp.?)) |y|
            act = try g.adopt(y);
    }
    if (act == null) {
        const gate2 = try g.adopt((try gather_qmv.gatherQmv(g.s, x1, w.gate.w, w.gate.s, w.gate.b, ind, gq.bits, gq.group_size, gq.mode, false)) orelse return false);
        const up2 = try g.adopt((try gather_qmv.gatherQmv(g.s, x1, w.up.w, w.up.s, w.up.b, ind, uq.bits, uq.group_size, uq.mode, false)) orelse return false);
        const inter = ops.shape(gate2)[1];
        const e = try geglu(w, g, try g.reshape(gate2, &.{ K, 1, inter }), try g.reshape(up2, &.{ K, 1, inter }));
        act = try g.reshape(e, &.{ K, inter });
    }
    const inter = ops.shape(act.?)[1];
    const sc = try g.reshape(r.scores, &.{K});
    if (try down_reduce.gatherQmvDownReduce(g.s, act.?, w.down.w, w.down.s, w.down.b, ind, sc, dq.bits, dq.group_size, dq.mode)) |sum| {
        const hidden = ops.shape(sum)[0];
        _ = try put(g, out, sum, &.{ B, S, hidden });
        reduced.* = true;
        if (!down_reduce.downred_engaged) {
            down_reduce.downred_engaged = true;
            log.info("[moe] fused down+reduce kernel engaged: E_topk={d} inter={d} hidden={d} bits={d} gs={d}\n", .{ K, inter, hidden, dq.bits, dq.group_size });
        }
        if (!gather_qmv.gqmv_engaged) {
            gather_qmv.gqmv_engaged = true;
            log.info("[moe] gather-qmv kernel engaged: E_topk={d} inter={d} hidden={d} bits={d} gs={d}\n", .{ K, inter, hidden, gq.bits, gq.group_size });
        }
        return true;
    }
    const down2 = try g.adopt((try gather_qmv.gatherQmv(g.s, act.?, w.down.w, w.down.s, w.down.b, ind, dq.bits, dq.group_size, dq.mode, true)) orelse return false);
    const hidden = ops.shape(down2)[1];
    out.* = try g.reshape(down2, &.{ B, S, K, hidden });
    if (!gather_qmv.gqmv_engaged) {
        gather_qmv.gqmv_engaged = true;
        log.info("[moe] gather-qmv kernel engaged: E_topk={d} inter={d} hidden={d} bits={d} gs={d}\n", .{ K, inter, hidden, gq.bits, gq.group_size });
    }
    return true;
}

/// The env-forced decode arm (`MLX_SERVE_MOE_BATCHED_DECODE`): the top-k experts taken, one batched matmul each.
fn batchedTake(w: *const Moe.Weights, parent: *Graph, x: Array, r: Routed, B: c_int, S: c_int, D: c_int, K: c_int) !Array {
    var t = parent.scope();
    defer t.deinit();
    const g = &t;
    const flat = try g.reshape(r.inds, &.{K});
    const x_bc = try g.broadcast(try g.reshape(x, &.{ 1, 1, D }), &.{ K, 1, D });
    const gate = try takeMm(g, x_bc, &w.gate, flat);
    const up = try takeMm(g, x_bc, &w.up, flat);
    const down = try takeMm(g, try geglu(w, g, gate, up), &w.down, flat);
    const sh = ops.shape(down);
    return parent.keep(try g.reshape(down, &.{ B, S, K, sh[sh.len - 1] }));
}

fn takeMm(parent: *Graph, x: Array, bank: *const Bank, flat: Array) !Array {
    var t = parent.scope();
    defer t.deinit();
    const g = &t;
    const wk = try g.take(bank.w, flat, 0);
    const sk = try g.take(bank.s, flat, 0);
    const bk: Array = if (bank.b.ctx != null) try g.take(bank.b, flat, 0) else .{ .ctx = null };
    return parent.keep(try g.call(mlx.mlx_quantized_matmul, .{ x, wk, sk, bk, true, mlx.mlx_optional_int.some(@intCast(bank.qp.group_size)), mlx.mlx_optional_int.some(@intCast(bank.qp.bits)), bank.qp.mode.cstr() }));
}

/// `w`'s per-expert scale on `y`'s rows (by `sel`), as `scaleExperts`.
fn scaleExperts(parent: *Graph, y: Array, scale: Array, sel: Array) !Array {
    if (scale.ctx == null) return y;
    var t = parent.scope();
    defer t.deinit();
    return parent.keep(try t.mul(y, try t.expandDims(try t.take(scale, sel, 0), -1)));
}

/// mlx-lm's `_gather_sort`: every (token, expert) assignment sorted by expert so consecutive rows read one bank,
/// the down output weighted and summed back in token order.
fn sorted(w: *const Moe.Weights, g: *Graph, x: Array, r: Routed, B: c_int, S: c_int, D: c_int, K: c_int) !struct { out: Array, reduced: bool } {
    const total = B * S * K;
    const gq = w.gate.qp;
    const uq = w.up.qp;
    const flat = try g.reshape(r.inds, &.{total});
    var inv: Array = undefined;
    var sorted_inds: Array = undefined;
    var lhs: Array = undefined;
    if (try verify_kernels.moePrefillRoutePack(g.s, flat, K, w.experts)) |p| {
        inv = try g.adopt(p.inverse);
        sorted_inds = try g.adopt(p.sorted);
        lhs = try g.adopt(p.lhs);
        if (!route_pack_logged) {
            route_pack_logged = true;
            log.info("[moe] prefill counting-sort route packing engaged (assignments={d})\n", .{total});
        }
    } else {
        const order = try g.argsort(flat, 0);
        inv = try g.argsort(order, 0);
        sorted_inds = try g.take(flat, order, 0);
        lhs = try g.call(mlx.mlx_floor_divide, .{ order, try g.adopt(mlx.mlx_array_new_int(K)) });
    }
    const x_flat = try g.reshape(x, &.{ B * S, D });

    var act: ?Array = null;
    var mapped_gate: ?Array = null;
    var mapped_up: ?Array = null;
    const nax = verifyQmmNaxAvailable();
    const experts = ops.shape(w.gate.w)[0];
    const row_block = total >= 16 and experts > 0 and @divTrunc(total, experts) >= 4;
    if (row_block and nax and gq.mode == uq.mode and (gq.mode == .affine or gq.mode.isFp4())) {
        const x_tok = try g.reshape(x_flat, &.{ B * S, 1, D });
        const row_map = try g.astype(lhs, .uint32);
        if (fused_act.swigluFusedEnabled() and w.clamp != null and w.gate.expert_scale.ctx == null and w.up.expert_scale.ctx == null and
            gq.bits == uq.bits and gq.group_size == uq.group_size)
        {
            const sigtab = try sigmoidTable(g.s);
            if (try gather_qmm_nax.sortedGateUp(x_tok, row_map, w.gate.w, w.gate.s, w.gate.b, w.up.w, w.up.s, w.up.b, sorted_inds, sigtab, w.clamp.?, gq.bits, gq.group_size, gq.mode.isFp4(), nax, g.s)) |a3| {
                const sh = ops.shape(a3);
                act = try g.reshape(try g.adopt(a3), &.{ sh[0], sh[2] });
            }
        }
        if (act == null) {
            mapped_gate = if (try gather_qmm_nax.sortedGatherMapped(x_tok, row_map, w.gate.w, w.gate.s, w.gate.b, sorted_inds, gq.bits, gq.group_size, gq.mode.isFp4(), nax, g.s)) |y| try g.adopt(y) else null;
            mapped_up = if (try gather_qmm_nax.sortedGatherMapped(x_tok, row_map, w.up.w, w.up.s, w.up.b, sorted_inds, uq.bits, uq.group_size, uq.mode.isFp4(), nax, g.s)) |y| try g.adopt(y) else null;
            if (mapped_gate == null or mapped_up == null) {
                mapped_gate = null;
                mapped_up = null;
            }
        }
    }
    if (act == null and mapped_gate == null and mlx.cudaAvailable() and w.clamp_limit == 0 and
        w.up.expert_scale.ctx == null and gq.mode == uq.mode and (gq.mode == .nvfp4 or gq.mode == .mxfp8))
    {
        const row_map = try g.astype(lhs, .uint32);
        const fmt: moe_prefill_cuda.Format = if (gq.mode == .nvfp4) .nvfp4 else .mxfp8;
        const gscale: ?Array = if (w.gate.expert_scale.ctx != null) w.gate.expert_scale else null;
        if (try moe_prefill_cuda.gateUp(g.s, x_flat, row_map, sorted_inds, w.gate.w, w.gate.s, w.up.w, w.up.s, gscale, fmt)) |y| act = try g.adopt(y);
    }
    const expert_act = act orelse blk: {
        const x_rep: ?Array = if (mapped_gate == null) try g.reshape(try g.take(x_flat, lhs, 0), &.{ total, 1, D }) else null;
        const none = Array{ .ctx = null };
        const gate3 = mapped_gate orelse try expertMm(g, x_rep.?, &w.gate, none, sorted_inds, true);
        const gate = try scaleExperts(g, try g.call(mlx.mlx_squeeze, .{gate3}), w.gate.expert_scale, sorted_inds);
        const up3 = mapped_up orelse try expertMm(g, x_rep.?, &w.up, none, sorted_inds, true);
        const up = try scaleExperts(g, try g.call(mlx.mlx_squeeze, .{up3}), w.up.expert_scale, sorted_inds);
        break :blk try geglu(w, g, gate, up);
    };
    const down3 = try expertMm(g, try g.expandDims(expert_act, -2), &w.down, .{ .ctx = null }, sorted_inds, true);
    const down = try g.call(mlx.mlx_squeeze, .{down3});
    if (try verify_kernels.moePrefillExpertReduce(g.s, down, inv, r.scores)) |sum| {
        if (!prefill_reduce_logged) {
            prefill_reduce_logged = true;
            log.info("[moe] fused prefill weighted sum engaged (tokens={d})\n", .{B * S});
        }
        return .{ .out = try g.adopt(sum), .reduced = true };
    }
    const unsorted = try g.take(down, inv, 0);
    return .{ .out = try g.reshape(unsorted, &.{ B, S, K, ops.shape(unsorted)[1] }), .reduced = false };
}

var route_pack_logged = false;
var prefill_reduce_logged = false;
