//! Parity proof for the `Moe` block's router and shared-expert variants against the legacy `moeMLP` on one loaded
//! layer: sigmoid + selection bias with an ungated shared expert (laguna, flint, hy_v3), the group-limited form,
//! no shared expert (lfm2_moe), top-k on biased f32 logits (kolibri1) and a clamped SwiGLU; each at one token, verify
//! rows, a prompt chunk and two batched rows.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const bases = @import("../bases.zig");
const fixture = @import("fixture.zig");
const q35 = @import("qwen35_test.zig");
const moe = @import("../blocks/moe.zig");
const compiled = @import("../blocks/compiled.zig");
const MoeMlpWeights = @import("../../blocks/moe/weights.zig").MoeMlpWeights;
const swigluClampLimit = @import("../../blocks/mlp/activations.zig").swigluClampLimit;

const testing = std.testing;
const Array = mlx.mlx_array;

const Variant = enum { sigmoid_ungated, grouped, no_shared, logit_bias, clamped };

/// The token shapes every variant runs: one token, verify rows, a prompt chunk, two batched rows.
const shapes = [_][2]c_int{ .{ 1, 1 }, .{ 1, 3 }, .{ 1, 40 }, .{ 2, 1 } };

fn random(g: *ops.Graph, shape: []const c_int, scale: f32, dt: mlx.mlx_dtype, seed: u64) !Array {
    const key = try g.call(mlx.mlx_random_key, .{seed});
    const n = try g.call(mlx.mlx_random_normal, .{ shape.ptr, shape.len, mlx.mlx_dtype.float32, @as(f32, 0), scale, key });
    const out = try g.astype(n, dt);
    try ops.eval(out);
    return out;
}

fn case(v: Variant, seed: u64) !void {
    const a = testing.allocator;
    var p = fixture.Pack.init(seed);
    defer p.deinit();
    try q35.writePack(&p, true, true, 8);
    const lg = try p.load(q35.config(true, true, 8));
    defer lg.deinit();
    const cfg = &lg.xfm.config;
    var keep = ops.Graph.init(a, lg.xfm.s);
    defer keep.deinit();
    const experts: c_int = @intCast(cfg.num_experts);
    const hidden: c_int = @intCast(cfg.hidden_size);

    var mw: MoeMlpWeights = lg.xfm.moe_layers.?[0].mlp.moe;
    var route_fn: compiled.Closure = .{ .ctx = null };
    defer compiled.free(&route_fn);
    var w = q35.moeWeights(lg, &mw, &route_fn);
    compiled.free(&route_fn);
    w.route_fn = .{ .ctx = null };

    const bias = try random(&keep, &.{experts}, 0.1, .float32, seed + 1);
    switch (v) {
        .sigmoid_ungated, .grouped, .no_shared => {
            cfg.moe_sigmoid_router = true;
            cfg.moe_route_norm = true;
            cfg.router_scaling_factor = 2.5;
            if (v == .grouped) {
                cfg.moe_n_group = 2;
                cfg.moe_topk_group = 1;
            }
            mw.expert_bias = bias;
            mw.shared_ungated = true;
            mw.shared_expert_gate_w = null;
            if (v == .no_shared) mw.shared_gate_w = .{ .ctx = null };
            w.route = .{ .sigmoid_bias = .{ .expert_bias = bias, .norm = true, .scale = 2.5, .n_group = @intCast(cfg.moe_n_group), .topk_group = @intCast(cfg.moe_topk_group) } };
            if (v == .no_shared) w.shared = null else {
                w.shared.?.gate_logit = .{};
                w.shared.?.gate_vec = .{ .ctx = null };
            }
        },
        .logit_bias => {
            cfg.moe_sigmoid_router = true;
            cfg.moe_logit_select = true;
            const router = try random(&keep, &.{ hidden, experts }, 0.05, .float32, seed + 2);
            mw.router_w = router;
            mw.router_s = .{ .ctx = null };
            mw.router_b = .{ .ctx = null };
            mw.expert_bias = bias;
            mw.shared_ungated = true;
            mw.shared_expert_gate_w = null;
            w.router = .{ .w = router };
            w.route = .{ .logit_bias = bias };
            w.shared.?.gate_logit = .{};
            w.shared.?.gate_vec = .{ .ctx = null };
        },
        .clamped => {
            cfg.swiglu_clamp = true;
            cfg.swiglu_limit = 1.0;
            w.clamp = swigluClampLimit(cfg.swiglu_limit, cfg.swiglu_clamp);
            w.clamp_limit = 1.0;
        },
    }
    lg.xfm.compileMoeRouting();
    route_fn = moe.compileRouting(&w, lg.xfm.s);
    w.route_fn = route_fn;

    var shared = bases.Shared.init(cfg.rms_norm_eps, cfg.postNormEps());
    defer shared.deinit();
    for (shapes, 0..) |sh, i| {
        var g = ops.Graph.init(a, lg.xfm.s);
        defer g.deinit();
        const x = try random(&g, &.{ sh[0], sh[1], hidden }, 1.0, .bfloat16, seed + 10 + i);
        const want = try g.adopt(try lg.xfm.moeMLP(x, &mw));
        var c: bases.Ctx = .{ .g = &g, .mode = if (sh[0] > 1) .batched_rows else .prefill, .rows = sh[1], .slots = sh[0], .positions = &.{0}, .shared = &shared };
        const got = try moe.Moe.forward(&w, &c, x);
        try ops.expectBitEqual(got, want);
    }
}

// ── tests ──

test "bases mixers: MoE sigmoid router + selection bias, ungated shared expert, is bit-equal to moeMLP" {
    try case(.sigmoid_ungated, 0x40e1);
}

test "bases mixers: MoE group-limited sigmoid router is bit-equal to moeMLP" {
    try case(.grouped, 0x40e2);
}

test "bases mixers: MoE without a shared expert is bit-equal to moeMLP" {
    try case(.no_shared, 0x40e3);
}

test "bases mixers: MoE top-k on biased f32 logits (kolibri1) is bit-equal to moeMLP" {
    try case(.logit_bias, 0x40e4);
}

test "bases mixers: MoE with a clamped SwiGLU is bit-equal to moeMLP" {
    try case(.clamped, 0x40e5);
}
