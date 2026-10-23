//! Parity proof for the hybrid base (`forwardHybridWith`): an LFM2 trunk (gated conv + attention, each followed by
//! its SwiGLU, `residualTail` seams) and Nemotron-H trunks (Mamba2, NoPE attention, MoE folded into the closing seam,
//! a ReLU² MLP, `add_norm` seams; a Puzzle latent MoE), prefill, decode and verify bit-equal to the legacy forward.
//! Every block is the base's own; the Mamba2 op chain and the sorted MoE gather get steps of their own. Neither
//! family batches decode on the legacy path.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const bases = @import("../bases.zig");
const fixture = @import("fixture.zig");
const FullAttnWeights = @import("../../../engines/mlx/layer_weights.zig").FullAttnWeights;
const GatedConvWeights = @import("../../../plugins/lfm2/gated_conv.zig").GatedConvWeights;
const legacy_mamba2 = @import("../../blocks/ssm/mamba2.zig");
const Mamba2Weights = legacy_mamba2.Mamba2Weights;
const NemotronMoeWeights = @import("../../../plugins/nemotron_h/moe.zig").NemotronMoeWeights;
const quantParamsOrDense = @import("../../quant/params.zig").quantParamsOrDense;

const testing = std.testing;
const Array = mlx.mlx_array;
const b = bases.blocks;
const GatedConv = b.gated_conv.GatedConv;
const Mamba2 = b.mamba2.Mamba2;
const NemotronMoe = b.nemotron_moe.NemotronMoe;

pub const Lfm2 = bases.Hybrid(.{
    .embed = b.embed.TableEmbed,
    .norm = b.norm.RmsNorm,
    .op = bases.Either(GatedConv, b.attention.HybridAttention),
    .mlp = b.mlp.GatedMlp,
    .seam = b.seam.ResidualTail,
    .head = b.head.LmHead,
});

pub const Nemotron = bases.Hybrid(.{
    .embed = b.embed.TableEmbed,
    .norm = b.norm.RmsNorm,
    .op = bases.Either(bases.Either(Mamba2, b.attention.HybridAttention), bases.Either(NemotronMoe, b.mlp.ReluSq)),
    .mlp = b.mlp.GatedMlp,
    .seam = b.seam.AddNorm,
    .head = b.head.LmHead,
});

const H: c_int = 128;
const HEADS: c_int = 4;
const KV_HEADS: c_int = 2;
const HD: c_int = 32;
const INTER: c_int = 256;
const VOCAB: c_int = 256;

fn at(buf: []u8, prefix: []const u8, layer: usize, name: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}.layers.{d}.{s}", .{ prefix, layer, name });
}

/// The base's attention over a legacy hybrid layer's (LFM2 and Nemotron-H read the same geometry).
fn attention(lg: *const fixture.Legacy, fa: *const FullAttnWeights, li: u32) b.attention.HybridAttention.Weights {
    const cfg = &lg.config;
    const hidden = cfg.hidden_size;
    const none = Array{};
    return .{
        .q = lg.lin(fa.q_w, fa.q_s, fa.q_b, none, hidden),
        .k = lg.lin(fa.k_w, fa.k_s, fa.k_b, none, hidden),
        .v = lg.lin(fa.v_w, fa.v_s, fa.v_b, none, hidden),
        .o = lg.lin(fa.o_w, fa.o_s, fa.o_b, none, cfg.num_attention_heads * cfg.head_dim),
        .q_norm = if (cfg.has_qk_norm) fa.q_norm else none,
        .k_norm = if (cfg.has_qk_norm) fa.k_norm else none,
        .qkv = lg.joined(fa.qkv, hidden),
        .geo = .{
            .heads = @intCast(cfg.num_attention_heads),
            .kv_heads = @intCast(cfg.num_key_value_heads),
            .head_dim = @intCast(cfg.head_dim),
            .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim))),
            .rope_dims = @intCast(cfg.head_dim),
            .rope_base = cfg.rope_theta,
            .rope_scale = cfg.rope_scaling_factor,
            .nope = cfg.layerSkipsRope(li),
        },
        .keep = cfg.max_position_embeddings,
    };
}

fn trunk(comptime M: type, lg: *fixture.Legacy, layers: []const M.Layer) M.Weights {
    const x = &lg.xfm;
    const hidden = lg.config.hidden_size;
    const none = Array{};
    return .{
        .embed = .{ .table = lg.lin(x.emb_w, x.emb_s, x.emb_b, none, hidden), .hidden = @intCast(hidden), .act = lg.config.actDtype() },
        .layers = layers,
        .final_norm = if (lg.config.has_final_norm) x.final_norm else null,
        .head = lg.lin(x.lm_head_w, x.lm_head_s, x.lm_head_b, none, hidden),
    };
}

// ── LFM2 ──

const lfm2_config =
    \\{"model_type":"lfm2","hidden_size":128,"intermediate_size":256,"num_hidden_layers":4,"num_attention_heads":4,
    \\"num_key_value_heads":2,"vocab_size":256,"norm_eps":1e-5,"rope_theta":1000000.0,"max_position_embeddings":4096,
    \\"conv_L_cache":3,"layer_types":["conv","full_attention","conv","full_attention"],
    \\"quantization":{"group_size":64,"bits":4}}
;

fn writeLfm2(p: *fixture.Pack) !void {
    var buf: [160]u8 = undefined;
    const m = "model";
    try p.weight("model.embed_tokens", VOCAB, H, true);
    try p.norm("model.embedding_norm.weight", H);
    for (0..4) |i| {
        try p.norm(try at(&buf, m, i, "operator_norm.weight"), H);
        try p.norm(try at(&buf, m, i, "ffn_norm.weight"), H);
        try p.weight(try at(&buf, m, i, "feed_forward.w1"), INTER, H, true);
        try p.weight(try at(&buf, m, i, "feed_forward.w3"), INTER, H, true);
        try p.weight(try at(&buf, m, i, "feed_forward.w2"), H, INTER, true);
        if (i % 2 == 0) {
            try p.weight(try at(&buf, m, i, "conv.in_proj"), 3 * H, H, true);
            try p.dense(try at(&buf, m, i, "conv.conv.weight"), &.{ H, 3, 1 }, 0.5, 0);
            try p.weight(try at(&buf, m, i, "conv.out_proj"), H, H, true);
        } else {
            try p.weight(try at(&buf, m, i, "self_attn.q_proj"), HEADS * HD, H, true);
            try p.weight(try at(&buf, m, i, "self_attn.k_proj"), KV_HEADS * HD, H, true);
            try p.weight(try at(&buf, m, i, "self_attn.v_proj"), KV_HEADS * HD, H, true);
            try p.weight(try at(&buf, m, i, "self_attn.out_proj"), H, HEADS * HD, true);
            try p.norm(try at(&buf, m, i, "self_attn.q_layernorm.weight"), HD);
            try p.norm(try at(&buf, m, i, "self_attn.k_layernorm.weight"), HD);
        }
    }
}

// ── Nemotron-H ──

/// The pack's free shape: the Mamba2 head width (64 makes the grouped norm's group 128 wide, the kernel's floor) and
/// the experts' latent width (0 = experts read the full hidden).
const NemoShape = struct { mamba_hd: c_int = 32, latent: c_int = 0 };

fn nemotronConfig(buf: []u8, shape: NemoShape) ![]const u8 {
    return std.fmt.bufPrint(buf,
        \\{{"model_type":"nemotron_h","hidden_size":128,"intermediate_size":256,"num_hidden_layers":5,"num_attention_heads":4,
        \\"num_key_value_heads":2,"head_dim":32,"vocab_size":256,"layer_norm_epsilon":1e-5,"rope_theta":10000.0,
        \\"max_position_embeddings":4096,"hybrid_override_pattern":"M*E-E","mamba_num_heads":4,"mamba_head_dim":{d},
        \\"n_groups":2,"ssm_state_size":32,"conv_kernel":4,"time_step_limit":[0.001,0.1],"n_routed_experts":8,
        \\"num_experts_per_tok":3,"moe_intermediate_size":64,"moe_shared_expert_intermediate_size":128,
        \\"routed_scaling_factor":2.5,"n_group":1,"topk_group":1,"norm_topk_prob":true,"tie_word_embeddings":false,
        \\"moe_latent_size":{d},"quantization":{{"group_size":64,"bits":4}}}}
    , .{ shape.mamba_hd, shape.latent });
}

fn writeNemotron(p: *fixture.Pack, shape: NemoShape) !void {
    var buf: [160]u8 = undefined;
    const m = "backbone";
    const d_inner = 4 * shape.mamba_hd;
    const conv_dim = d_inner + 2 * 2 * 32;
    const ein: c_int = if (shape.latent > 0) shape.latent else H;
    try p.weight("backbone.embeddings", VOCAB, H, true);
    try p.norm("backbone.norm_f.weight", H);
    try p.weight("lm_head", VOCAB, H, true);
    for ("M*E-E", 0..) |kind, i| {
        try p.norm(try at(&buf, m, i, "norm.weight"), H);
        switch (kind) {
            'M' => {
                try p.weight(try at(&buf, m, i, "mixer.in_proj"), d_inner + conv_dim + 4, H, true);
                try p.dense(try at(&buf, m, i, "mixer.conv1d.weight"), &.{ conv_dim, 4, 1 }, 0.6, 0.1);
                try p.dense(try at(&buf, m, i, "mixer.conv1d.bias"), &.{conv_dim}, 0.2, 0);
                try p.dense(try at(&buf, m, i, "mixer.A_log"), &.{4}, 1.0, 0.3);
                try p.dense(try at(&buf, m, i, "mixer.D"), &.{4}, 1.0, 1.0);
                try p.dense(try at(&buf, m, i, "mixer.dt_bias"), &.{4}, 1.0, 0);
                try p.norm(try at(&buf, m, i, "mixer.norm.weight"), d_inner);
                try p.weight(try at(&buf, m, i, "mixer.out_proj"), H, d_inner, true);
            },
            '*' => {
                try p.weight(try at(&buf, m, i, "mixer.q_proj"), HEADS * HD, H, true);
                try p.weight(try at(&buf, m, i, "mixer.k_proj"), KV_HEADS * HD, H, true);
                try p.weight(try at(&buf, m, i, "mixer.v_proj"), KV_HEADS * HD, H, true);
                try p.weight(try at(&buf, m, i, "mixer.o_proj"), H, HEADS * HD, true);
            },
            'E' => {
                try p.dense(try at(&buf, m, i, "mixer.gate.weight"), &.{ 8, H }, 0.5, 0);
                try p.denseAs(try at(&buf, m, i, "mixer.gate.e_score_correction_bias"), &.{8}, 0.2, 0, .float32);
                try p.tensor(try at(&buf, m, i, "mixer.switch_mlp.fc1"), &.{ 8, 64, ein }, true);
                try p.tensor(try at(&buf, m, i, "mixer.switch_mlp.fc2"), &.{ 8, ein, 64 }, true);
                try p.weight(try at(&buf, m, i, "mixer.shared_experts.up_proj"), 128, H, true);
                try p.weight(try at(&buf, m, i, "mixer.shared_experts.down_proj"), H, 128, true);
                if (shape.latent > 0) {
                    try p.weight(try at(&buf, m, i, "mixer.fc1_latent_proj"), shape.latent, H, true);
                    try p.weight(try at(&buf, m, i, "mixer.fc2_latent_proj"), H, shape.latent, true);
                }
            },
            else => {
                try p.weight(try at(&buf, m, i, "mixer.up_proj"), INTER, H, true);
                try p.weight(try at(&buf, m, i, "mixer.down_proj"), H, INTER, true);
            },
        }
    }
}

/// An expert bank as the legacy resolves it (`quantParamsOrDense`).
fn bank(lg: *const fixture.Legacy, w: Array, sc: Array, bi: Array, in_dim: u32) b.linear.Linear {
    const qp = quantParamsOrDense(&lg.config, w, sc, in_dim);
    return .{ .w = w, .s = sc, .b = bi, .bits = qp.bits, .group_size = qp.group_size, .mode = qp.mode };
}

fn mamba2(lg: *const fixture.Legacy, mw: *const Mamba2Weights) Mamba2.Weights {
    const cfg = &lg.config;
    const inner = cfg.mamba_num_heads * cfg.mamba_head_dim;
    return .{
        .in_proj = lg.lin(mw.in_proj_w, mw.in_proj_s, mw.in_proj_b, .{}, cfg.hidden_size),
        .conv_w = mw.conv1d_w,
        .conv_b = mw.conv1d_b orelse .{},
        .A_log = mw.A_log,
        .D = mw.D,
        .dt_bias = mw.dt_bias,
        .norm = mw.norm_w,
        .out_proj = lg.lin(mw.out_proj_w, mw.out_proj_s, mw.out_proj_b, .{}, inner),
        .geo = .{
            .heads = @intCast(cfg.mamba_num_heads),
            .head_dim = @intCast(cfg.mamba_head_dim),
            .groups = @intCast(cfg.mamba_n_groups),
            .state = @intCast(cfg.ssm_state_size),
            .kernel = @intCast(cfg.mamba_conv_kernel),
            .dt_min = cfg.time_step_min,
            .dt_max = cfg.time_step_max,
        },
    };
}

fn nemotronMoe(lg: *const fixture.Legacy, nm: *const NemotronMoeWeights) NemotronMoe.Weights {
    const cfg = &lg.config;
    const hidden = cfg.hidden_size;
    const ein: u32 = if (nm.fc1_latent_w.ctx != null) cfg.moe_latent_size else hidden;
    const inter: u32 = @intCast(mlx.getShape(nm.fc1_w)[1]);
    var w: NemotronMoe.Weights = .{
        .router = lg.lin(nm.router_w, nm.router_s, nm.router_b, .{}, hidden),
        .expert_bias = nm.expert_bias,
        .fc1 = bank(lg, nm.fc1_w, nm.fc1_s, nm.fc1_b, ein),
        .fc2 = bank(lg, nm.fc2_w, nm.fc2_s, nm.fc2_b, inter),
        .routing = .{
            .k = @intCast(cfg.num_experts_per_tok),
            .n_group = @intCast(cfg.moe_n_group),
            .topk_group = @intCast(cfg.moe_topk_group),
            .renorm = cfg.moe_route_norm,
            .scale = cfg.router_scaling_factor,
        },
    };
    if (nm.shared) |sm| {
        w.shared_up = bank(lg, sm.up_w, sm.up_s, sm.up_b, hidden);
        const up_sh = mlx.getShape(sm.up_w);
        const shared_inter: u32 = @intCast(if (sm.up_s.ctx != null) up_sh[0] else up_sh[1]);
        w.shared_down = bank(lg, sm.down_w, sm.down_s, sm.down_b, shared_inter);
    }
    if (nm.fc1_latent_w.ctx != null) {
        w.latent_in = lg.lin(nm.fc1_latent_w, nm.fc1_latent_s, nm.fc1_latent_b, .{}, hidden);
        w.latent_out = lg.lin(nm.fc2_latent_w, nm.fc2_latent_s, nm.fc2_latent_b, .{}, cfg.moe_latent_size);
    }
    return w;
}

fn nemotronLayers(lg: *fixture.Legacy, layers: []Nemotron.Layer) !void {
    const hidden = lg.config.hidden_size;
    for (layers, lg.xfm.hybrid_layers.?, 0..) |*L, *lw, i| {
        L.* = .{
            .input_norm = lw.input_norm,
            .op = switch (lw.op) {
                .mamba2 => |*mw| .{ .first = .{ .first = mamba2(lg, mw) } },
                .full_attn => |*fa| .{ .first = .{ .second = attention(lg, fa, @intCast(i)) } },
                .nemotron_moe => |*nm| .{ .second = .{ .first = nemotronMoe(lg, nm) } },
                .simple_mlp => |*sm| .{ .second = .{ .second = .{
                    .up = lg.lin(sm.up_w, sm.up_s, sm.up_b, .{}, hidden),
                    .down = lg.lin(sm.down_w, sm.down_s, sm.down_b, .{}, lg.config.intermediate_size),
                } } },
                else => return error.UnexpectedLayer,
            },
        };
    }
}

/// A Nemotron-H pack of `shape` stepped through `seq`, the base bit-equal to the legacy forward at every step.
fn nemotronCase(shape: NemoShape, seed: u64, seq: []const fixture.Step) !void {
    var p = fixture.Pack.init(seed);
    defer p.deinit();
    try writeNemotron(&p, shape);
    var cbuf: [1024]u8 = undefined;
    const lg = try p.load(try nemotronConfig(&cbuf, shape));
    defer lg.deinit();
    var layers: [5]Nemotron.Layer = undefined;
    try nemotronLayers(lg, &layers);
    const w = trunk(Nemotron, lg, &layers);
    try fixture.expectStepsOf(Nemotron, lg, &w, seq);
}

/// A prompt wider than the in-place expert window (`GATHER_MAX_ROWS`), so the MoE takes its sorted gather.
const wide_steps = [_]fixture.Step{
    .{ .mode = .prefill, .ids = &.{ 3, 17, 42, 99, 7, 250, 1, 8, 66, 31, 120, 5, 77, 200 } },
    .{ .mode = .decode, .ids = &.{88} },
    .{ .mode = .verify, .ids = &.{ 5, 61, 200 } },
    .{ .mode = .prefill, .ids = &.{ 9, 33, 77, 140, 2, 18, 19, 20, 21, 22, 23 } },
    .{ .mode = .decode, .ids = &.{64} },
};

// ── tests ──

test "bases hybrid: LFM2 gated conv + attention prefill, decode and verify are bit-equal to forwardHybridWith" {
    var p = fixture.Pack.init(0x1f2);
    defer p.deinit();
    try writeLfm2(&p);
    const lg = try p.load(lfm2_config);
    defer lg.deinit();
    const x = &lg.xfm;
    const cfg = &lg.config;
    const hidden = cfg.hidden_size;
    const none = Array{};
    var layers: [4]Lfm2.Layer = undefined;
    for (&layers, x.hybrid_layers.?, 0..) |*L, *lw, i| {
        const m = &lw.mlp.?;
        L.* = .{
            .input_norm = lw.input_norm,
            .post_norm = lw.post_norm.?,
            .op = switch (lw.op) {
                .gated_conv => |*cw| .{ .first = .{
                    .in_proj = lg.lin(cw.in_proj_w, cw.in_proj_s, cw.in_proj_b, none, hidden),
                    .conv = cw.conv_w,
                    .out_proj = lg.lin(cw.out_proj_w, cw.out_proj_s, cw.out_proj_b, none, hidden),
                    .hidden = @intCast(hidden),
                    .kernel = @intCast(cfg.lfm_conv_kernel),
                } },
                .full_attn => |*fa| .{ .second = attention(lg, fa, @intCast(i)) },
                else => return error.UnexpectedLayer,
            },
            .mlp = .{
                .gate = lg.lin(m.gate_w, m.gate_s, m.gate_b, none, hidden),
                .up = lg.lin(m.up_w, m.up_s, m.up_b, none, hidden),
                .down = lg.lin(m.down_w, m.down_s, m.down_b, none, cfg.intermediate_size),
                .gu = lg.joined(m.gu, hidden),
            },
        };
    }
    const w = trunk(Lfm2, lg, &layers);
    try fixture.expectSteps(Lfm2, lg, &w);
}

test "bases hybrid: Nemotron-H Mamba2 + attention + folded MoE prefill, decode and verify are bit-equal to forwardHybridWith" {
    try nemotronCase(.{}, 0x6e6d, &fixture.steps);
}

test "bases hybrid: Nemotron-H with the grouped-norm kernel and a sorted MoE window is bit-equal" {
    try nemotronCase(.{ .mamba_hd = 64 }, 0x6e6e, &wide_steps);
}

test "bases hybrid: Nemotron-H Mamba2 op chain (fused step off) is bit-equal" {
    const saved = legacy_mamba2.mamba2_fused_max_rows;
    legacy_mamba2.mamba2_fused_max_rows = 0;
    defer legacy_mamba2.mamba2_fused_max_rows = saved;
    try nemotronCase(.{ .mamba_hd = 64 }, 0x6e6f, &fixture.steps);
    try nemotronCase(.{}, 0x6e70, &fixture.steps);
}

test "bases hybrid: Nemotron-H latent MoE (routes on the full hidden, experts in the latent) is bit-equal" {
    try nemotronCase(.{ .latent = 64 }, 0x6e71, &fixture.steps);
    try nemotronCase(.{ .latent = 64 }, 0x6e72, &wide_steps);
}
