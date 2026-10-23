//! GLM-5.3-Flash's and Ling's MoE layers as the shared `Moe` block's weights, borrowed from the layered load: the
//! sigmoid router with its selection bias (group-limited where the config groups), the ungated shared expert,
//! ModelOpt's per-expert scales, Sushi's EXL3 banks through the format table, and GLM's one-kernel decode
//! (`glm5.moeDecode`, affine banks, up to `MOE_DECODE_MAX_ROWS` rows) as the block's `fused` arm.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const moe = @import("../../sdk/bases/blocks/moe.zig");
const compiled = @import("../../sdk/bases/blocks/compiled.zig");
const Linear = @import("../../sdk/bases/blocks/linear.zig").Linear;
const quant_abi = @import("../../abi/quant.zig");
const exl3 = @import("../../sdk/quant/exl3/format.zig");
const QuantParams = @import("../../sdk/quant/params.zig").QuantParams;
const MoeMlpWeights = @import("../../sdk/blocks/moe/weights.zig").MoeMlpWeights;
const swigluClampLimit = @import("../../sdk/blocks/mlp/activations.zig").swigluClampLimit;
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;
const glm5 = @import("glm5_next.zig");

const Array = mlx.mlx_array;
pub const MoeWeights = moe.Moe.Weights;

/// Frees what `weightsOf` made (its compiled routing chain); the arrays are the load's.
pub fn free(w: *MoeWeights) void {
    compiled.free(&w.route_fn);
}

/// Every layer's MoE (null for a dense layer); the arrays are the load's, the compiled routing chains this table's.
pub const Table = struct {
    layers: []?moe.Moe.Weights,

    pub fn deinit(t: *Table, a: std.mem.Allocator) void {
        for (t.layers) |*w| if (w.*) |*m| free(m);
        a.free(t.layers);
    }
};

pub fn build(a: std.mem.Allocator, xfm: *Transformer) !Table {
    const ml = xfm.moe_layers orelse return .{ .layers = &.{} };
    var t: Table = .{ .layers = try a.alloc(?moe.Moe.Weights, ml.len) };
    @memset(t.layers, null);
    errdefer t.deinit(a);
    for (t.layers, ml) |*w, *lw| if (lw.mlp == .moe) {
        w.* = try weightsOf(xfm, &lw.mlp.moe);
    };
    return t;
}

/// One MoE layer's weights (the MTP head's included); free its `route_fn` with the layer.
pub fn weightsOf(xfm: *Transformer, mw: *const MoeMlpWeights) !moe.Moe.Weights {
    const cfg = &xfm.config;
    const hidden = cfg.hidden_size;
    // The in-dim hints the legacy MoE resolves with (none where the config gives no width).
    const inter: ?u32 = if (cfg.moe_intermediate_size > 0) cfg.moe_intermediate_size else null;
    const shared_inter: ?u32 = if (cfg.shared_expert_intermediate_size > 0) cfg.shared_expert_intermediate_size else null;
    var w: moe.Moe.Weights = .{
        .router = lin(xfm, mw.router_w, mw.router_s, mw.router_b, hidden),
        .route = .{ .sigmoid_bias = .{
            .expert_bias = mw.expert_bias orelse return error.MissingWeight,
            .norm = cfg.moe_route_norm,
            .scale = cfg.router_scaling_factor,
            .n_group = @intCast(@max(cfg.moe_n_group, 1)),
            .topk_group = @intCast(@max(cfg.moe_topk_group, 1)),
        } },
        .gate = bank(xfm, mw.switch_gate_w, mw.switch_gate_s, mw.switch_gate_b, hidden, mw.gate_expert_scale),
        .up = bank(xfm, mw.switch_up_w, mw.switch_up_s, mw.switch_up_b, hidden, mw.up_expert_scale),
        .down = bank(xfm, mw.switch_down_w, mw.switch_down_s, mw.switch_down_b, inter, .{ .ctx = null }),
        .shared = .{
            .gate = lin(xfm, mw.shared_gate_w, mw.shared_gate_s, mw.shared_gate_b, hidden),
            .up = lin(xfm, mw.shared_up_w, mw.shared_up_s, mw.shared_up_b, hidden),
            .down = lin(xfm, mw.shared_down_w, mw.shared_down_s, mw.shared_down_b, shared_inter),
        },
        .bank_views = mw.bank_views,
        .top_k = @intCast(cfg.num_experts_per_tok),
        .experts = @intCast(cfg.num_experts),
        .clamp = swigluClampLimit(cfg),
        .clamp_limit = if (cfg.swiglu_clamp) cfg.swiglu_limit else 0,
        .route_scale = mw.per_expert_scale orelse .{ .ctx = null },
    };
    if (cfg.exl3) |spec| {
        w.format_banks = .{
            .table = &exl3.table,
            .gate = exl3Weight(mw.switch_gate_w, mw.switch_gate_s, mw.switch_gate_b, spec),
            .up = exl3Weight(mw.switch_up_w, mw.switch_up_s, mw.switch_up_b, spec),
            .down = exl3Weight(mw.switch_down_w, mw.switch_down_s, mw.switch_down_b, spec),
        };
    } else if (cfg.isGlm5()) w.fused = .{ .rows = glm5.MOE_DECODE_MAX_ROWS, .limit = cfg.swiglu_limit, .run = &fusedDecode };
    w.route_fn = moe.compileRouting(&w, xfm.s);
    return w;
}

/// A projection with the quant geometry the legacy MoE resolves for it.
fn lin(xfm: *const Transformer, w: Array, s: Array, b: Array, in_dim: ?u32) Linear {
    if (s.ctx == null) return .{ .w = w };
    const qp = xfm.quantParamsHinted(w, s, in_dim);
    return .{ .w = w, .s = s, .b = b, .bits = qp.bits, .group_size = qp.group_size, .mode = qp.mode };
}

fn bank(xfm: *const Transformer, w: Array, s: Array, b: Array, in_dim: ?u32, scale: Array) moe.Bank {
    return .{ .w = w, .s = s, .b = b, .qp = xfm.quantParamsHinted(w, s, in_dim), .expert_scale = scale };
}

/// A Sushi bank as the EXL3 format table loads one: trellis, suh, svh and the pack's decode.
fn exl3Weight(trellis: Array, suh: Array, svh: Array, spec: exl3.Spec) quant_abi.Weight {
    const d = exl3.decode(spec);
    return .{
        .w = .{ .ctx = trellis.ctx },
        .scales = .{ .ctx = suh.ctx },
        .biases = .{ .ctx = svh.ctx },
        .layout = @backingInt(d.codebook),
        .flags = @intCast(d.window.bits()),
        .block = @intCast(mlx.getShape(trellis)[3]),
        .state = if (d.table) |i| @ptrFromInt(@as(usize, i) + 1) else null,
    };
}

/// GLM's routed + shared experts in two dispatches; null (another quant mode, geometry or device) takes the block's
/// arms.
fn fusedDecode(f: *const moe.Fused, w: *const moe.Moe.Weights, s: mlx.mlx_stream, x: Array, inds: Array, scores: Array) !?Array {
    const se = &(w.shared orelse return null);
    inline for (.{ w.gate.qp, w.up.qp, w.down.qp }) |qp| if (qp.mode != .affine) return null;
    inline for (.{ se.gate, se.up, se.down }) |l| if (l.mode != .affine or l.s.ctx == null) return null;
    const k: c_int = @intCast(mlx.mlx_array_size(inds));
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, inds, &[_]c_int{k}, 1, s));
    var ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids);
    try mlx.check(mlx.mlx_astype(&ids, flat, .uint32, s));
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_reshape(&sc, scores, &[_]c_int{k}, 1, s));
    const y = (try glm5.moeDecode(s, x, ids, sc, qb(w.gate), qb(w.up), qb(w.down), ql(se.gate), ql(se.up), ql(se.down), f.limit)) orelse return null;
    defer _ = mlx.mlx_array_free(y);
    const xs = mlx.getShape(x);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, y, xs.ptr, xs.len, s));
    return out;
}

fn qb(b: moe.Bank) glm5.QBank {
    return .{ .w = b.w, .s = b.s, .b = b.b, .bits = b.qp.bits, .gs = b.qp.group_size };
}

fn ql(l: Linear) glm5.QBank {
    return .{ .w = l.w, .s = l.s, .b = l.b, .bits = l.bits, .gs = l.group_size };
}

/// The block over `x` (owned by the caller).
pub fn forward(w: *const moe.Moe.Weights, s: mlx.mlx_stream, x: Array) !Array {
    var g = ops.Graph.init(std.heap.c_allocator, s);
    defer g.deinit();
    return g.escape(try moe.mlp(w, &g, x));
}
