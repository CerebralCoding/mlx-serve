//! Parity proof for the qwen3_5 family's mixer blocks (`Gdn`, `GatedAttention`) and the generic `Moe` MLP through
//! the decoder base, against `forwardMoeWith` and the batched GDN trunk (`forwardMoeBatchedDecode` /
//! `forwardMoeBatchedVerify`): hd-128 linear-attention heads (the fused decode kernels' geometry), hd-256 attention,
//! prompt chunks cold and warm, decode, captured verify rows, and two slots of different lengths batched. Logits,
//! every recurrent state and every verify capture must match bit for bit.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const abi = @import("../../../abi/plugin.zig");
const bases = @import("../bases.zig");
const fixture = @import("fixture.zig");
const gdn = @import("../blocks/gdn.zig");
const gated = @import("../blocks/gated_attention.zig");
const moe = @import("../blocks/moe.zig");
const qk_rope = @import("../blocks/qk_rope.zig");
const cache_ops = @import("../../../engines/mlx/kv/cache_ops.zig");
const KVCache = @import("../../../engines/mlx/kv/cache.zig").KVCache;
const ForwardCtx = @import("../../../engines/mlx/forward_ctx.zig").ForwardCtx;
const SSMCacheEntry = @import("../../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const LinearAttnWeights = @import("../../blocks/gdn/gated_delta_net.zig").LinearAttnWeights;
const FullAttnWeights = @import("../../../engines/mlx/layer_weights.zig").FullAttnWeights;
const MoeMlpWeights = @import("../../blocks/moe/weights.zig").MoeMlpWeights;
const computeQuantParams = @import("../../quant/params.zig").computeQuantParams;
const swigluClampLimit = @import("../../blocks/mlp/activations.zig").swigluClampLimit;

const testing = std.testing;
const Array = mlx.mlx_array;
const b = bases.blocks;

const Mixer = bases.Either(gdn.Gdn, gated.GatedAttention);

pub const Dense35 = bases.Decoder(.{
    .embed = b.embed.TableEmbed,
    .norm = b.norm.RmsNorm,
    .mixer = Mixer,
    .mlp = b.mlp.GatedMlp,
    .seam = b.seam.AddNorm,
    .head = b.head.LmHead,
});

pub const Moe35 = bases.Decoder(.{
    .embed = b.embed.TableEmbed,
    .norm = b.norm.RmsNorm,
    .mixer = Mixer,
    .mlp = moe.Moe,
    .seam = b.seam.AddNorm,
    .head = b.head.LmHead,
});

const H: c_int = 256;
const VOCAB: c_int = 256;
const LAYERS = 4;
const HEADS: c_int = 4;
const KV_HEADS: c_int = 2;
const HD: c_int = 256;
const INTER: c_int = 512;
const HK: c_int = 2;
const HV: c_int = 8;
const DK: c_int = 128;
const DV: c_int = 128;
const KERNEL: c_int = 4;
const MOE_INTER: c_int = 128;

pub fn config(comptime moe_pack: bool, comptime quantized: bool, comptime experts: c_int) []const u8 {
    const mt = if (moe_pack) "qwen3_5_moe" else "qwen3_5";
    const moe_keys = if (moe_pack) "\"num_experts\":" ++ std.fmt.comptimePrint("{d}", .{experts}) ++ ",\"num_experts_per_tok\":2,\"moe_intermediate_size\":128,\"shared_expert_intermediate_size\":128," else "";
    return "{\"model_type\":\"" ++ mt ++ "\",\"text_config\":{\"model_type\":\"" ++ mt ++ "\",\"hidden_size\":256,\"intermediate_size\":512," ++
        "\"num_hidden_layers\":4,\"num_attention_heads\":4,\"num_key_value_heads\":2,\"head_dim\":256,\"vocab_size\":256," ++ moe_keys ++
        "\"rms_norm_eps\":1e-6,\"full_attention_interval\":2,\"linear_num_key_heads\":2,\"linear_num_value_heads\":8," ++
        "\"linear_key_head_dim\":128,\"linear_value_head_dim\":128,\"linear_conv_kernel_dim\":4,\"max_position_embeddings\":4096," ++
        "\"rope_parameters\":{\"rope_theta\":100000.0,\"partial_rotary_factor\":0.25},\"tie_word_embeddings\":false}," ++
        "\"tie_word_embeddings\":false" ++ (if (quantized) ",\"quantization\":{\"group_size\":64,\"bits\":4}}" else "}");
}

fn at(buf: []u8, layer: usize, name: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "language_model.model.layers.{d}.{s}", .{ layer, name });
}

/// Linear-attention layers 0 and 2, gated attention 1 and 3; 4-bit or bf16 (dense attention: decode reads the
/// requantized side copies). The shared-expert gate stays bf16.
pub fn writePack(p: *fixture.Pack, comptime moe_pack: bool, q: bool, experts: c_int) !void {
    var buf: [160]u8 = undefined;
    try p.weight("language_model.model.embed_tokens", VOCAB, H, q);
    try p.norm("language_model.model.norm.weight", H);
    try p.weight("language_model.lm_head", VOCAB, H, q);
    for (0..LAYERS) |i| {
        try p.norm(try at(&buf, i, "input_layernorm.weight"), H);
        try p.norm(try at(&buf, i, "post_attention_layernorm.weight"), H);
        if (moe_pack) {
            try p.weight(try at(&buf, i, "mlp.gate"), experts, H, q);
            try p.tensor(try at(&buf, i, "mlp.switch_mlp.gate_proj"), &.{ experts, MOE_INTER, H }, q);
            try p.tensor(try at(&buf, i, "mlp.switch_mlp.up_proj"), &.{ experts, MOE_INTER, H }, q);
            try p.tensor(try at(&buf, i, "mlp.switch_mlp.down_proj"), &.{ experts, H, MOE_INTER }, q);
            try p.weight(try at(&buf, i, "mlp.shared_expert.gate_proj"), MOE_INTER, H, q);
            try p.weight(try at(&buf, i, "mlp.shared_expert.up_proj"), MOE_INTER, H, q);
            try p.weight(try at(&buf, i, "mlp.shared_expert.down_proj"), H, MOE_INTER, q);
            try p.weight(try at(&buf, i, "mlp.shared_expert_gate"), 1, H, false);
        } else {
            try p.weight(try at(&buf, i, "mlp.gate_proj"), INTER, H, q);
            try p.weight(try at(&buf, i, "mlp.up_proj"), INTER, H, q);
            try p.weight(try at(&buf, i, "mlp.down_proj"), H, INTER, q);
        }
        if (i % 2 == 0) {
            try p.weight(try at(&buf, i, "linear_attn.in_proj_qkv"), 2 * HK * DK + HV * DV, H, q);
            try p.weight(try at(&buf, i, "linear_attn.in_proj_z"), HV * DV, H, q);
            try p.weight(try at(&buf, i, "linear_attn.in_proj_a"), HV, H, q);
            try p.weight(try at(&buf, i, "linear_attn.in_proj_b"), HV, H, q);
            try p.dense(try at(&buf, i, "linear_attn.conv1d.weight"), &.{ 2 * HK * DK + HV * DV, KERNEL, 1 }, 0.5, 0);
            try p.dense(try at(&buf, i, "linear_attn.A_log"), &.{HV}, 0.5, 0.2);
            try p.dense(try at(&buf, i, "linear_attn.dt_bias"), &.{HV}, 0.5, 0);
            try p.norm(try at(&buf, i, "linear_attn.norm.weight"), DV);
            try p.weight(try at(&buf, i, "linear_attn.out_proj"), H, HV * DV, q);
        } else {
            try p.weight(try at(&buf, i, "self_attn.q_proj"), 2 * HEADS * HD, H, q);
            try p.weight(try at(&buf, i, "self_attn.k_proj"), KV_HEADS * HD, H, q);
            try p.weight(try at(&buf, i, "self_attn.v_proj"), KV_HEADS * HD, H, q);
            try p.weight(try at(&buf, i, "self_attn.o_proj"), H, HEADS * HD, q);
            try p.norm(try at(&buf, i, "self_attn.q_norm.weight"), HD);
            try p.norm(try at(&buf, i, "self_attn.k_norm.weight"), HD);
        }
    }
}

// ── the base's weights over the legacy model's loaded arrays ──

/// What a model of these blocks holds beside its layers: the shared GDN constants, the angle rows, the closures.
const Model = struct {
    geo: gdn.Geo,
    angles: qk_rope.AngleCache = .{},
    route_fns: [LAYERS]@import("../blocks/compiled.zig").Closure = @splat(.{ .ctx = null }),

    fn deinit(m: *Model) void {
        gdn.freeConsts(&m.geo);
        m.angles.deinit();
        for (&m.route_fns) |*c| @import("../blocks/compiled.zig").free(c);
    }
};

fn gdnGeo(lg: *fixture.Legacy) !gdn.Geo {
    const cfg = &lg.xfm.config;
    var geo: gdn.Geo = .{
        .hk = @intCast(cfg.linear_num_key_heads),
        .hv = @intCast(cfg.linear_num_value_heads),
        .dk = @intCast(cfg.linear_key_head_dim),
        .dv = @intCast(cfg.linear_value_head_dim),
        .kernel = @intCast(cfg.linear_conv_kernel_dim),
        .state = cfg.ssmStateDtype(),
        .qk_eps = cfg.gdnQkNormEps(),
        .arch = cfg.model_type,
        .q_scale = .{ .ctx = null },
        .k_scale = .{ .ctx = null },
        .ones = .{ .ctx = null },
    };
    try gdn.consts(&geo, cfg.actDtype(), lg.xfm.s);
    geo.gate_fn = gdn.compileGate(geo.state, lg.xfm.s);
    return geo;
}

fn gdnWeights(lg: *fixture.Legacy, la: *const LinearAttnWeights, geo: gdn.Geo) gdn.Gdn.Weights {
    const hidden = lg.config.hidden_size;
    const none = Array{};
    var w: gdn.Gdn.Weights = .{
        .qkv = lg.lin(la.qkv_w, la.qkv_s, la.qkv_b, none, hidden),
        .z = lg.lin(la.z_w, la.z_s, la.z_b, none, hidden),
        .a = lg.lin(la.a_w, la.a_s, la.a_b, none, hidden),
        .b = lg.lin(la.b_w, la.b_s, la.b_b, none, hidden),
        .out = lg.lin(la.out_w, la.out_s, la.out_b, none, @intCast(geo.hv * geo.dv)),
        .in = lg.joined(la.in, hidden),
        .combined = la.combined_proj,
        .conv_w = la.conv1d_w,
        .A_log = la.A_log,
        .dt_bias = la.dt_bias,
        .norm_w = la.norm_w,
        .geo = geo,
    };
    for (&w.qkv_parts, la.qkv_parts) |*l, q| if (q.w.ctx != null) {
        l.* = lg.lin(q.w, q.s, q.b, none, hidden);
    };
    return w;
}

fn proj(lg: *fixture.Legacy, w: Array, s: Array, bi: Array, in_dim: u32, layer: u32) !gated.Proj {
    var p: gated.Proj = .{ .lin = lg.lin(w, s, bi, .{}, in_dim) };
    if (s.ctx == null) if (try lg.xfm.attnDqFor(w, s, bi, in_dim, layer)) |c| {
        p.dq = .{ .w = c.w, .s = c.s, .b = c.b, .bits = c.bits, .group_size = c.group_size, .mode = c.mode };
    };
    return p;
}

fn gatedWeights(lg: *fixture.Legacy, fa: *const FullAttnWeights, layer: u32, angles: *qk_rope.AngleCache) !gated.GatedAttention.Weights {
    const cfg = &lg.xfm.config;
    const hidden = cfg.hidden_size;
    return .{
        .q = try proj(lg, fa.q_w, fa.q_s, fa.q_b, hidden, layer),
        .k = try proj(lg, fa.k_w, fa.k_s, fa.k_b, hidden, layer),
        .v = try proj(lg, fa.v_w, fa.v_s, fa.v_b, hidden, layer),
        .o = try proj(lg, fa.o_w, fa.o_s, fa.o_b, cfg.num_attention_heads * cfg.head_dim, layer),
        .q_norm = fa.q_norm,
        .k_norm = fa.k_norm,
        .qkv = lg.joined(fa.qkv, hidden),
        .geo = .{
            .heads = @intCast(cfg.num_attention_heads),
            .kv_heads = @intCast(cfg.num_key_value_heads),
            .head_dim = @intCast(cfg.head_dim),
            .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.query_pre_attn_scalar))),
            .rope_dims = @intFromFloat(@as(f32, @floatFromInt(cfg.head_dim)) * cfg.partial_rotary_factor),
            .rope_base = cfg.rope_theta,
            .output_gate = cfg.attn_output_gate,
        },
        .angles = angles,
    };
}

fn bank(lg: *fixture.Legacy, w: Array, s: Array, bi: Array, in_dim: ?u32, scale: Array) moe.Bank {
    // A dense bank resolves to the model-wide scheme, as `quantParamsHinted` does.
    const cfg = &lg.config;
    const qp = if (s.ctx == null) @import("../../quant/params.zig").QuantParams{ .bits = cfg.quant_bits, .group_size = cfg.quant_group_size, .mode = cfg.quant_mode } else computeQuantParams(cfg, w, s, in_dim);
    return .{ .w = w, .s = s, .b = bi, .qp = qp, .expert_scale = scale };
}

pub fn moeWeights(lg: *fixture.Legacy, mw: *const MoeMlpWeights, route_fn: *@import("../blocks/compiled.zig").Closure) moe.Moe.Weights {
    const cfg = &lg.xfm.config;
    const hidden = cfg.hidden_size;
    const inter = cfg.moe_intermediate_size;
    const none = Array{};
    var w: moe.Moe.Weights = .{
        .router = lg.lin(mw.router_w, mw.router_s, mw.router_b, none, hidden),
        .route = .softmax,
        .gate = bank(lg, mw.switch_gate_w, mw.switch_gate_s, mw.switch_gate_b, hidden, mw.gate_expert_scale),
        .up = bank(lg, mw.switch_up_w, mw.switch_up_s, mw.switch_up_b, hidden, mw.up_expert_scale),
        .down = bank(lg, mw.switch_down_w, mw.switch_down_s, mw.switch_down_b, inter, none),
        .bank_views = mw.bank_views,
        .top_k = @intCast(cfg.num_experts_per_tok),
        .experts = @intCast(cfg.num_experts),
        .clamp = swigluClampLimit(cfg),
        .clamp_limit = if (cfg.swiglu_clamp) cfg.swiglu_limit else 0,
    };
    if (mw.shared_expert_gate_w) |seg| w.shared = .{
        .gate = lg.lin(mw.shared_gate_w, mw.shared_gate_s, mw.shared_gate_b, none, hidden),
        .up = lg.lin(mw.shared_up_w, mw.shared_up_s, mw.shared_up_b, none, hidden),
        .down = lg.lin(mw.shared_down_w, mw.shared_down_s, mw.shared_down_b, none, cfg.shared_expert_intermediate_size),
        .gate_logit = lg.lin(seg, mw.shared_expert_gate_s.?, mw.shared_expert_gate_b.?, none, hidden),
        .gate_vec = mw.shared_gate_vec,
    };
    route_fn.* = moe.compileRouting(&w, lg.xfm.s);
    w.route_fn = route_fn.*;
    return w;
}

fn baseWeights(comptime D: type, lg: *fixture.Legacy, m: *Model, layers: []D.Layer) !D.Weights {
    const x = &lg.xfm;
    const hidden = lg.config.hidden_size;
    const none = Array{};
    for (layers, x.moe_layers.?, 0..) |*L, *lw, i| {
        L.input_norm = lw.input_norm;
        L.post_norm = lw.post_attn_norm;
        L.mixer = switch (lw.attn) {
            .linear => |*la| .{ .first = gdnWeights(lg, la, m.geo) },
            .full => |*fa| .{ .second = try gatedWeights(lg, fa, @intCast(i), &m.angles) },
        };
        L.mlp = if (D == Moe35) moeWeights(lg, &lw.mlp.moe, &m.route_fns[i]) else blk: {
            const dw = &lw.mlp.dense;
            break :blk .{
                .gate = lg.lin(dw.gate_w, dw.gate_s, dw.gate_b, none, hidden),
                .up = lg.lin(dw.up_w, dw.up_s, dw.up_b, none, hidden),
                .down = lg.lin(dw.down_w, dw.down_s, dw.down_b, none, lg.config.intermediate_size),
                .gu = lg.joined(dw.gu, hidden),
            };
        };
    }
    return .{
        .embed = .{ .table = lg.lin(x.emb_w, x.emb_s, x.emb_b, none, hidden), .hidden = @intCast(hidden), .act = lg.config.actDtype() },
        .layers = layers,
        .final_norm = x.final_norm,
        .head = lg.lin(x.lm_head_w, x.lm_head_s, x.lm_head_b, none, hidden),
    };
}

// ── one slot on each side ──

/// A request on both sides: the legacy forward's own cache and recurrent entries, and the base's.
const Lane = struct {
    legacy_kv: KVCache,
    legacy_ssm: []SSMCacheEntry,
    offset: usize = 0,
    ctx: ForwardCtx = undefined,
    base_kv: KVCache,
    base_ssm: []SSMCacheEntry,
    slot: cache_ops.Slot = undefined,

    fn init(lane: *Lane, lg: *fixture.Legacy) !void {
        const n = lg.config.num_hidden_layers;
        lane.* = .{ .legacy_kv = try KVCache.init(testing.allocator, n), .legacy_ssm = try fixture.newEntries(n), .base_kv = try KVCache.init(testing.allocator, n), .base_ssm = try fixture.newEntries(n) };
        lane.ctx = lg.xfm.defaultCtx();
        lane.ctx.cache = &lane.legacy_kv;
        lane.ctx.moe_seq_offset = &lane.offset;
        lane.ctx.ssm_entries = lane.legacy_ssm;
        lane.slot = .{ .kv = &lane.base_kv, .ssm = lane.base_ssm };
    }

    fn deinit(lane: *Lane) void {
        lane.legacy_kv.deinit();
        lane.base_kv.deinit();
        fixture.freeEntries(lane.legacy_ssm);
        fixture.freeEntries(lane.base_ssm);
    }

    /// Every recurrent state and verify capture equal to the legacy forward's.
    fn expectStates(lane: *const Lane) !void {
        for (lane.legacy_ssm, lane.base_ssm) |*want, *got| {
            try testing.expectEqual(want.initialized, got.initialized);
            inline for (.{ "conv_state", "ssm_state", "spec_conv_input", "spec_state_seq" }) |f| {
                const wa = @field(want, f);
                const ga = @field(got, f);
                try testing.expectEqual(wa.ctx == null, ga.ctx == null);
                if (wa.ctx != null) try expectBitsEqual(ga, wa);
            }
        }
    }
};

/// Bit-equal, except a state sequence's last row: the verify kernels never write it (a rollback never reads it).
fn expectBitsEqual(got: Array, want: Array) !void {
    var g = ops.Graph.init(testing.allocator, fixture.stream());
    defer g.deinit();
    const sh = ops.shape(want);
    if (sh.len == 5 and sh[0] > 1 and ops.shape(got).len == 5)
        return ops.expectBitEqual(try g.sliceAxis(got, 0, 0, sh[0] - 1), try g.sliceAxis(want, 0, 0, sh[0] - 1));
    return ops.expectBitEqual(got, want);
}

const Step = struct { mode: abi.Mode, ids: []const i32 };

fn ids(comptime n: usize, seed: u32) [n]i32 {
    var out: [n]i32 = undefined;
    for (&out, 0..) |*v, i| v.* = @intCast((seed + 37 * @as(u32, @intCast(i))) % 256);
    return out;
}

const chunk20 = ids(20, 3);
const chunk70 = ids(70, 11);
const chunk6 = ids(6, 5);

/// A cold 20-token chunk (composed GDN chain, the hd-256 prefill kernel), a warm 70-token chunk (blocked recurrence),
/// decode, captured verify rows of 3 and 5, a warm 6-token chunk (fused prework), decode.
const solo_steps = [_]Step{
    .{ .mode = .prefill, .ids = &chunk20 },
    .{ .mode = .prefill, .ids = &chunk70 },
    .{ .mode = .decode, .ids = &.{88} },
    .{ .mode = .verify, .ids = &.{ 5, 61, 200 } },
    .{ .mode = .decode, .ids = &.{12} },
    .{ .mode = .verify, .ids = &.{ 9, 33, 77, 140, 2 } },
    .{ .mode = .decode, .ids = &.{64} },
    .{ .mode = .prefill, .ids = &chunk6 },
    .{ .mode = .decode, .ids = &.{17} },
};

fn soloStep(comptime D: type, lg: *fixture.Legacy, w: *const D.Weights, shared: *const bases.Shared, lane: *Lane, st: Step) !void {
    var g = ops.Graph.init(testing.allocator, lg.xfm.s);
    defer g.deinit();
    const t = try fixture.tokens(&g, st.ids);
    const pos: u32 = @intCast(lane.offset);
    lane.ctx.capture_ssm_seq = st.mode == .verify;
    const want = try fixture.rows2(&g, try g.adopt(try lg.xfm.forwardWith(&lane.ctx, t)));
    const handles = [_]abi.Handle{lane.slot.handle()};
    const got = try D.forward(w, shared, &g, t, .{ .mode = st.mode, .rows = @intCast(st.ids.len), .positions = &.{pos}, .caches = &handles, .cache_ops = &cache_ops.ops, .logits = .all_rows });
    try ops.expectBitEqual(got.logits, want);
    try lane.expectStates();
}

const Opts = struct { f32_state: bool = false, quantized: bool = true, experts: c_int = 8 };

fn soloCase(comptime D: type, comptime moe_pack: bool, comptime opts: Opts, seed: u64) !void {
    var p = fixture.Pack.init(seed);
    defer p.deinit();
    try writePack(&p, moe_pack, opts.quantized, opts.experts);
    const lg = try p.load(config(moe_pack, opts.quantized, opts.experts));
    defer lg.deinit();
    if (opts.f32_state) lg.xfm.config.ssm_state_f32 = true;
    lg.xfm.compileGdnGate();
    if (moe_pack) lg.xfm.compileMoeRouting();
    var m: Model = .{ .geo = try gdnGeo(lg) };
    defer m.deinit();
    var layers: [LAYERS]D.Layer = undefined;
    const w = try baseWeights(D, lg, &m, &layers);
    var shared = bases.Shared.init(lg.config.rms_norm_eps, lg.config.postNormEps());
    defer shared.deinit();
    var lane: Lane = undefined;
    try lane.init(lg);
    defer lane.deinit();
    for (solo_steps) |st| try soloStep(D, lg, &w, &shared, &lane, st);
}

/// Two slots prefilled apart (20 and 7 tokens), then decoded, verified (3 rows each) and decoded together: every
/// batched row and every slot's states equal to the legacy GDN batched driver's.
fn batchedCase(comptime D: type, comptime moe_pack: bool, seed: u64) !void {
    const a = testing.allocator;
    var p = fixture.Pack.init(seed);
    defer p.deinit();
    try writePack(&p, moe_pack, true, 8);
    const lg = try p.load(config(moe_pack, true, 8));
    defer lg.deinit();
    lg.xfm.compileGdnGate();
    if (moe_pack) lg.xfm.compileMoeRouting();
    // Split the merged states back after every tick, so each slot's states can be compared.
    lg.xfm.persistent_group_state_override = false;
    var m: Model = .{ .geo = try gdnGeo(lg) };
    defer m.deinit();
    var layers: [LAYERS]D.Layer = undefined;
    const w = try baseWeights(D, lg, &m, &layers);
    var shared = bases.Shared.init(lg.config.rms_norm_eps, lg.config.postNormEps());
    defer shared.deinit();
    var lanes: [2]Lane = undefined;
    try lanes[0].init(lg);
    defer lanes[0].deinit();
    try lanes[1].init(lg);
    defer lanes[1].deinit();
    try soloStep(D, lg, &w, &shared, &lanes[0], .{ .mode = .prefill, .ids = &chunk20 });
    try soloStep(D, lg, &w, &shared, &lanes[1], .{ .mode = .prefill, .ids = &.{ 42, 101, 7, 9, 250, 3, 77 } });

    const ctxs = [_]*ForwardCtx{ &lanes[0].ctx, &lanes[1].ctx };
    const handles = [_]abi.Handle{ lanes[0].slot.handle(), lanes[1].slot.handle() };
    const ticks = [_][2][]const i32{
        .{ &.{77}, &.{3} },
        .{ &.{ 5, 61, 200 }, &.{ 9, 33, 140 } },
        .{ &.{190}, &.{64} },
    };
    for (ticks) |tk| {
        var g = ops.Graph.init(a, lg.xfm.s);
        defer g.deinit();
        const rows = tk[0].len;
        var flat: [6]i32 = undefined;
        @memcpy(flat[0..rows], tk[0]);
        @memcpy(flat[rows .. 2 * rows], tk[1]);
        const t = try g.fromSlice(i32, flat[0 .. 2 * rows], &.{ 2, @intCast(rows) });
        const positions = [_]u32{ @intCast(lanes[0].offset), @intCast(lanes[1].offset) };
        var want: Array = undefined;
        if (rows == 1) {
            const next = [_]u32{ @intCast(tk[0][0]), @intCast(tk[1][0]) };
            const legacy = try lg.xfm.forwardMoeBatchedDecode(&next, &ctxs, &positions, null);
            defer a.free(legacy);
            for (legacy) |x| _ = try g.adopt(x);
            want = try g.concat(&.{ try fixture.rows2(&g, legacy[0]), try fixture.rows2(&g, legacy[1]) }, 0);
        } else {
            var hl = mlx.mlx_array_new();
            var ha = mlx.mlx_array_new();
            const logits = try lg.xfm.forwardMoeBatchedVerify(t, &ctxs, &positions, &hl, &ha);
            _ = try g.adopt(hl);
            _ = try g.adopt(ha);
            want = try fixture.rows2(&g, try g.reshape(try g.adopt(logits), &.{ 1, 2 * @as(c_int, @intCast(rows)), VOCAB }));
        }
        const got = try D.forward(&w, &shared, &g, t, .{ .mode = .batched_rows, .rows = @intCast(rows), .slots = 2, .positions = &positions, .caches = &handles, .cache_ops = &cache_ops.ops, .logits = .all_rows });
        try ops.expectBitEqual(got.logits, want);
        for (&lanes) |*l| {
            l.offset += rows;
            try l.expectStates();
        }
    }
}

// ── tests ──

test "bases mixers: qwen3_5 GDN + gated attention blocks are bit-equal to forwardMoeWith (chunks, decode, captured verify)" {
    try soloCase(Dense35, false, .{}, 0x35b1);
}

test "bases mixers: qwen3_5 with an f32 recurrent state is bit-equal to forwardMoeWith" {
    try soloCase(Dense35, false, .{ .f32_state = true }, 0x35b2);
}

test "bases mixers: qwen3_5_moe 4-bit (MoE MLP: one-token, verify-rows and sorted expert arms) is bit-equal to forwardMoeWith" {
    try soloCase(Moe35, true, .{}, 0x35b3);
}

test "bases mixers: qwen3_5_moe 4-bit, 64 experts (the router kernel carries the shared-expert gate) is bit-equal to forwardMoeWith" {
    try soloCase(Moe35, true, .{ .experts = 64 }, 0x35b7);
}

test "bases mixers: qwen3_5_moe bf16 (dense experts, decode attention side copies) is bit-equal to forwardMoeWith" {
    try soloCase(Moe35, true, .{ .quantized = false }, 0x35b6);
}

test "bases mixers: qwen3_5 batched rows (decode, verify, decode; two lengths) are bit-equal to the GDN batched driver" {
    try testing.expect(Dense35.batches_decode);
    try batchedCase(Dense35, false, 0x35b4);
}

test "bases mixers: qwen3_5_moe batched rows are bit-equal to the GDN batched driver" {
    try testing.expect(Moe35.batches_decode);
    try batchedCase(Moe35, true, 0x35b5);
}
