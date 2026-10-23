//! Parity proofs for the attention extras: Gemma 3 (sliding layers under band masks, sandwich norms, GeGLU, logit
//! softcap) and Gemma 4 dense (KV-sharing layers, value norm, K=V full layers, dual head dims, proportional RoPE)
//! against `forwardStandardWith` and `forwardBatchedDecode`; a Laguna-shaped stack (per-layer heads, YaRN full layers,
//! sliding layers, softplus gate, the fused hd-128 q/k norm+RoPE) and Flint (sinks on a sliding/NoPE mix, plus the
//! same pack through a ring cache) against `forwardMoeWith` and its batched rows. Prefill, decode and verify widths
//! cross every window.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const abi = @import("../../../abi/plugin.zig");
const bases = @import("../bases.zig");
const fixture = @import("fixture.zig");
const cache_ops = @import("../../../engines/mlx/kv/cache_ops.zig");
const KVCache = @import("../../../engines/mlx/kv/cache.zig").KVCache;
const ForwardCtx = @import("../../../engines/mlx/forward_ctx.zig").ForwardCtx;
const LayerWeights = @import("../../../engines/mlx/layer_weights.zig").LayerWeights;
const FullAttnWeights = @import("../../../engines/mlx/layer_weights.zig").FullAttnWeights;

const testing = std.testing;
const Array = mlx.mlx_array;
const b = bases.blocks;
const extras = b.extras;

// ── Gemma ──

const GemmaAttn = b.decoder_attention.DecoderAttention(.{ .sliding = true, .kv_sharing = true, .value_norm = true });

pub const Gemma = bases.Decoder(.{
    .embed = b.embed.TableEmbed,
    .norm = b.norm.RmsNorm,
    .mixer = extras.PostNormed(GemmaAttn),
    .mlp = extras.PostNormed(extras.GeGlu),
    .seam = b.seam.ResidualTail,
    .head = extras.SoftcapHead,
});

const H: c_int = 128;
const VOCAB: c_int = 256;
const INTER: c_int = 256;

const gemma3_config =
    \\{"model_type":"gemma3_text","hidden_size":128,"intermediate_size":256,"num_hidden_layers":4,
    \\"num_attention_heads":4,"num_key_value_heads":2,"head_dim":32,"vocab_size":256,"rms_norm_eps":1e-6,
    \\"rope_theta":1000000.0,"rope_local_base_freq":10000.0,"sliding_window":4,"sliding_window_pattern":2,
    \\"query_pre_attn_scalar":32,"final_logit_softcapping":30.0,"max_position_embeddings":4096,
    \\"quantization":{"group_size":64,"bits":4}}
;

/// Six layers: sliding, sliding, full (K=V), sliding, then two that read layers 3 and 2's KV.
const gemma4_config =
    \\{"model_type":"gemma4_text","hidden_size":128,"intermediate_size":256,"num_hidden_layers":6,
    \\"num_attention_heads":4,"num_key_value_heads":2,"head_dim":32,"global_head_dim":64,
    \\"num_global_key_value_heads":1,"num_kv_shared_layers":2,"attention_k_eq_v":true,"vocab_size":256,
    \\"rms_norm_eps":1e-6,"sliding_window":4,"tie_word_embeddings":true,"hidden_size_per_layer_input":0,
    \\"layer_types":["sliding_attention","sliding_attention","full_attention","sliding_attention","sliding_attention","full_attention"],
    \\"rope_parameters":{"full_attention":{"rope_theta":1000000.0,"rope_type":"proportional","partial_rotary_factor":0.25,"factor":1.0},
    \\"sliding_attention":{"rope_theta":10000.0}},"final_logit_softcapping":30.0,"max_position_embeddings":4096,
    \\"quantization":{"group_size":64,"bits":4}}
;

fn layerName(buf: []u8, prefix: []const u8, layer: usize, name: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}.layers.{d}.{s}", .{ prefix, layer, name });
}

/// A Gemma pack: `layers` of `(global, kv_shared)`; global layers project `ghd`-wide heads over `gkv` KV heads and
/// share V with K when `k_eq_v`.
fn writeGemma(p: *fixture.Pack, prefix: []const u8, kinds: []const [2]bool, ghd: c_int, gkv: c_int, k_eq_v: bool) !void {
    var buf: [192]u8 = undefined;
    var nb: [96]u8 = undefined;
    try p.weight(try std.fmt.bufPrint(&nb, "{s}.embed_tokens", .{prefix}), VOCAB, H, true);
    try p.norm(try std.fmt.bufPrint(&nb, "{s}.norm.weight", .{prefix}), H);
    for (kinds, 0..) |kind, i| {
        const global = kind[0];
        const hd: c_int = if (global) ghd else 32;
        const kvh: c_int = if (global) gkv else 2;
        for ([_][]const u8{ "input_layernorm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight" }) |n|
            try p.norm(try layerName(&buf, prefix, i, n), H);
        try p.weight(try layerName(&buf, prefix, i, "self_attn.q_proj"), 4 * hd, H, true);
        try p.weight(try layerName(&buf, prefix, i, "self_attn.o_proj"), H, 4 * hd, true);
        try p.norm(try layerName(&buf, prefix, i, "self_attn.q_norm.weight"), hd);
        if (!kind[1]) {
            try p.weight(try layerName(&buf, prefix, i, "self_attn.k_proj"), kvh * hd, H, true);
            if (!(global and k_eq_v)) try p.weight(try layerName(&buf, prefix, i, "self_attn.v_proj"), kvh * hd, H, true);
            try p.norm(try layerName(&buf, prefix, i, "self_attn.k_norm.weight"), hd);
        }
        try p.weight(try layerName(&buf, prefix, i, "mlp.gate_proj"), INTER, H, true);
        try p.weight(try layerName(&buf, prefix, i, "mlp.up_proj"), INTER, H, true);
        try p.weight(try layerName(&buf, prefix, i, "mlp.down_proj"), H, INTER, true);
    }
}

/// The dense decoder's geometry of layer `li`, as `forwardStandardWith` resolves it.
fn denseGeo(lg: *const fixture.Legacy, li: u32) b.attention.Geo {
    const cfg = &lg.config;
    const global = cfg.isGlobalLayer(li);
    const dual = cfg.global_head_dim > 0 and cfg.global_head_dim != cfg.head_dim;
    const hd: u32 = if (dual and global) cfg.global_head_dim else cfg.head_dim;
    const gkv: u32 = if (cfg.num_global_key_value_heads > 0) cfg.num_global_key_value_heads else cfg.num_key_value_heads;
    const prop = global and lg.xfm.rope_freqs_global != null;
    const partial: c_int = if (global and cfg.partial_rotary_factor_global < 1.0)
        @intCast(@as(u32, @intFromFloat(@as(f32, @floatFromInt(hd)) * cfg.partial_rotary_factor_global)))
    else
        @intCast(hd);
    return .{
        .heads = @intCast(cfg.num_attention_heads),
        .kv_heads = @intCast(if (dual and global) gkv else cfg.num_key_value_heads),
        .head_dim = @intCast(hd),
        .scale = cfg.attnScale(),
        .rope_dims = if (prop) @intCast(hd) else partial,
        .rope_base = if (global) cfg.rope_theta else cfg.rope_local_base_freq,
        .rope_scale = if (prop) 1.0 else if (global) 1.0 / cfg.rope_scaling_factor else 1.0,
        .nope = cfg.layerSkipsRope(li),
        .freqs = if (prop) lg.xfm.rope_freqs_global.? else .{},
    };
}

fn gemmaLayer(lg: *const fixture.Legacy, lw: *const LayerWeights, li: u32, table: Array) Gemma.Layer {
    const cfg = &lg.config;
    const x = &lg.xfm;
    const hidden = cfg.hidden_size;
    const none = Array{};
    const geo = denseGeo(lg, li);
    const global = cfg.isGlobalLayer(li);
    const dual = cfg.global_head_dim > 0 and cfg.global_head_dim != cfg.head_dim;
    const vn: Array = if (!cfg.has_v_norm) none else if (dual and global) (x.v_norm_weight_global orelse x.v_norm_weight.?) else x.v_norm_weight.?;
    return .{
        .input_norm = lw.input_norm,
        .mixer = .{ .norm = lw.post_attn_norm, .inner = .{
            .q = lg.lin(lw.q_w, lw.q_s, lw.q_b, lw.q_bias, hidden),
            .k = if (lw.kv_source == null) lg.lin(lw.k_w, lw.k_s, lw.k_b, lw.k_bias, hidden) else .{},
            .v = if (lw.kv_source == null) lg.lin(lw.v_w, lw.v_s, lw.v_b, lw.v_bias, hidden) else .{},
            .o = lg.lin(lw.o_w, lw.o_s, lw.o_b, none, @intCast(geo.heads * geo.head_dim)),
            .q_norm = lw.q_norm orelse none,
            .k_norm = lw.k_norm orelse none,
            .qkv = lg.joined(lw.qkv, hidden),
            .geo = geo,
            .window = if (cfg.has_sliding_window and !global) cfg.sliding_window else 0,
            .kv_source = lw.kv_source,
            .v_norm = vn,
            .k_eq_v = lw.k_eq_v,
        } },
        .post_norm = lw.pre_ff_norm.?,
        .mlp = .{ .norm = lw.post_ff_norm.?, .inner = .{
            .gate = lg.lin(lw.gate_w, lw.gate_s, lw.gate_b, none, hidden),
            .up = lg.lin(lw.up_w, lw.up_s, lw.up_b, none, hidden),
            .down = lg.lin(lw.down_w, lw.down_s, lw.down_b, none, cfg.intermediate_size),
            .gu = lg.joined(lw.gu, hidden),
            .rows_joined = lw.qkv.w.ctx != null,
            .table = table,
        } },
    };
}

fn gemmaWeights(lg: *fixture.Legacy, layers: []Gemma.Layer) !Gemma.Weights {
    const x = &lg.xfm;
    const hidden = lg.config.hidden_size;
    const none = Array{};
    const table = (try x.geluTanhTable(.bfloat16)) orelse none;
    for (layers, x.layers, 0..) |*L, *lw, i| L.* = gemmaLayer(lg, lw, @intCast(i), table);
    return .{
        .embed = .{ .table = lg.lin(x.emb_w, x.emb_s, x.emb_b, none, hidden), .hidden = @intCast(hidden), .act = lg.config.actDtype(), .scale = x.emb_scale orelse none },
        .layers = layers,
        .final_norm = x.final_norm,
        .head = .{ .head = lg.lin(x.lm_head_w, x.lm_head_s, x.lm_head_b, none, hidden), .cap = x.softcap_scalar orelse none, .mult = x.output_mult orelse none },
    };
}

const gemma3_kinds = [_][2]bool{ .{ false, false }, .{ true, false }, .{ false, false }, .{ true, false } };
const gemma4_kinds = [_][2]bool{ .{ false, false }, .{ false, false }, .{ true, false }, .{ false, false }, .{ false, true }, .{ true, true } };

fn loadGemma(seed: u64, gemma4: bool) !*fixture.Legacy {
    var p = fixture.Pack.init(seed);
    defer p.deinit();
    if (gemma4)
        try writeGemma(&p, "language_model.model", &gemma4_kinds, 64, 1, true)
    else
        try writeGemma(&p, "model", &gemma3_kinds, 32, 2, false);
    return p.load(if (gemma4) gemma4_config else gemma3_config);
}

// ── Spark ──

pub const Spark = bases.Decoder(.{
    .embed = b.embed.TableEmbed,
    .norm = b.norm.RmsNorm,
    .mixer = b.decoder_attention.DecoderAttention(.{ .sliding = true, .gate = .head }),
    .mlp = extras.GatedGelu(.erf),
    .seam = b.seam.ResidualTail,
    .head = b.head.LmHead,
});

/// Three sliding layers and a full one with 25% rotary; one joined q|k|v projection split at load.
const spark_config =
    \\{"model_type":"spark2_5","hidden_size":128,"intermediate_size":256,"num_hidden_layers":4,"num_attention_heads":4,
    \\"num_key_value_heads":2,"head_dim":32,"vocab_size":256,"rms_norm_eps":1e-6,"sliding_window":4,
    \\"layer_types":["sliding_attention","sliding_attention","sliding_attention","full_attention"],
    \\"rope_parameters":{"full_attention":{"rope_theta":5000000.0,"partial_rotary_factor":0.25},
    \\"sliding_attention":{"rope_theta":10000.0}},"tie_word_embeddings":true,"max_position_embeddings":4096,
    \\"quantization":{"group_size":64,"bits":4}}
;

fn writeSpark(p: *fixture.Pack) !void {
    var buf: [160]u8 = undefined;
    const m = "model";
    try p.weight("model.embedding", VOCAB, H, true);
    try p.norm("model.norm.weight", H);
    for (0..4) |i| {
        try p.norm(try layerName(&buf, m, i, "input_layernorm.weight"), H);
        try p.norm(try layerName(&buf, m, i, "post_attention_layernorm.weight"), H);
        try p.weight(try layerName(&buf, m, i, "self_attn.q_k_v_proj"), 4 * 32 + 2 * 2 * 32, H, true);
        try p.weight(try layerName(&buf, m, i, "self_attn.out_proj"), H, 4 * 32, true);
        try p.weight(try layerName(&buf, m, i, "self_attn.g_proj"), 4, H, true);
        try p.weight(try layerName(&buf, m, i, "mlp.gate_proj"), INTER, H, true);
        try p.weight(try layerName(&buf, m, i, "mlp.up_proj"), INTER, H, true);
        try p.weight(try layerName(&buf, m, i, "mlp.down_proj"), H, INTER, true);
    }
}

fn sparkWeights(lg: *fixture.Legacy, layers: []Spark.Layer) Spark.Weights {
    const x = &lg.xfm;
    const cfg = &lg.config;
    const hidden = cfg.hidden_size;
    const none = Array{};
    for (layers, x.layers, 0..) |*L, *lw, i| {
        const li: u32 = @intCast(i);
        const geo = denseGeo(lg, li);
        L.* = .{
            .input_norm = lw.input_norm,
            .mixer = .{
                .q = lg.lin(lw.q_w, lw.q_s, lw.q_b, lw.q_bias, hidden),
                .k = lg.lin(lw.k_w, lw.k_s, lw.k_b, lw.k_bias, hidden),
                .v = lg.lin(lw.v_w, lw.v_s, lw.v_b, lw.v_bias, hidden),
                .o = lg.lin(lw.o_w, lw.o_s, lw.o_b, none, @intCast(geo.heads * geo.head_dim)),
                .qkv = lg.joined(lw.qkv, hidden),
                .geo = geo,
                .window = if (cfg.has_sliding_window and !cfg.isGlobalLayer(li)) cfg.sliding_window else 0,
                .out_gate = lg.lin(lw.ag_w.?, lw.ag_s, lw.ag_b, none, hidden),
            },
            .post_norm = lw.post_attn_norm,
            .mlp = .{
                .gate = lg.lin(lw.gate_w, lw.gate_s, lw.gate_b, none, hidden),
                .up = lg.lin(lw.up_w, lw.up_s, lw.up_b, none, hidden),
                .down = lg.lin(lw.down_w, lw.down_s, lw.down_b, none, cfg.intermediate_size),
                .gu = lg.joined(lw.gu, hidden),
                .rows_joined = lw.qkv.w.ctx != null,
            },
        };
    }
    return .{
        .embed = .{ .table = lg.lin(x.emb_w, x.emb_s, x.emb_b, none, hidden), .hidden = @intCast(hidden), .act = cfg.actDtype() },
        .layers = layers,
        .final_norm = x.final_norm,
        .head = lg.lin(x.lm_head_w, x.lm_head_s, x.lm_head_b, none, hidden),
    };
}

fn loadSpark(seed: u64) !*fixture.Legacy {
    var p = fixture.Pack.init(seed);
    defer p.deinit();
    try writeSpark(&p);
    return p.load(spark_config);
}

// ── Laguna and Flint ──

fn LagunaStack(comptime opts: b.laguna_attention.Options) type {
    return bases.Decoder(.{
        .embed = b.embed.TableEmbed,
        .norm = b.norm.RmsNorm,
        .mixer = b.laguna_attention.LagunaAttention(opts),
        .mlp = b.mlp.GatedMlp,
        .seam = b.seam.AddNorm,
        .head = b.head.LmHead,
    });
}

pub const Laguna = LagunaStack(.{ .softplus_gate = true });
pub const Flint = LagunaStack(.{ .sinks = true });

const laguna_config =
    \\{"model_type":"laguna","hidden_size":128,"intermediate_size":256,"num_hidden_layers":4,"num_attention_heads":4,
    \\"num_key_value_heads":1,"head_dim":128,"rms_norm_eps":1e-6,"vocab_size":256,"max_position_embeddings":4096,
    \\"tie_word_embeddings":false,"gating":"per-head","sliding_window":4,"num_experts":8,"num_experts_per_tok":2,
    \\"moe_intermediate_size":64,"mlp_only_layers":[0,1,2,3],"num_attention_heads_per_layer":[2,4,4,2],
    \\"layer_types":["full_attention","sliding_attention","sliding_attention","full_attention"],
    \\"rope_parameters":{"full_attention":{"rope_theta":500000.0,"rope_type":"yarn","factor":32.0,
    \\"original_max_position_embeddings":8192,"beta_slow":1.0,"beta_fast":32.0,"partial_rotary_factor":0.5},
    \\"sliding_attention":{"rope_type":"default","rope_theta":10000.0,"partial_rotary_factor":1.0}},
    \\"quantization":{"group_size":64,"bits":4}}
;

const flint_config =
    \\{"model_type":"flint","hidden_size":128,"intermediate_size":256,"num_hidden_layers":4,"num_attention_heads":2,
    \\"num_key_value_heads":1,"head_dim":128,"num_experts":8,"num_experts_per_tok":2,"moe_intermediate_size":64,
    \\"norm_topk_prob":true,"vocab_size":256,"rms_norm_eps":1e-6,"rope_theta":10000.0,"sliding_window":4,
    \\"use_sliding_window":true,"layer_types":["full_attention","sliding_attention","sliding_attention","full_attention"],
    \\"no_rope_layer_indices":[0,3],"mlp_only_layers":[0,1,2,3],"tie_word_embeddings":false,
    \\"quantization":{"group_size":64,"bits":4}}
;

fn writeLaguna(p: *fixture.Pack, heads: []const c_int, gate: bool, sinks: bool) !void {
    var buf: [160]u8 = undefined;
    const m = "model";
    try p.weight("model.embed_tokens", VOCAB, H, true);
    try p.norm("model.norm.weight", H);
    try p.weight("lm_head", VOCAB, H, true);
    for (heads, 0..) |nh, i| {
        try p.norm(try layerName(&buf, m, i, "input_layernorm.weight"), H);
        try p.norm(try layerName(&buf, m, i, "post_attention_layernorm.weight"), H);
        try p.weight(try layerName(&buf, m, i, "self_attn.q_proj"), nh * 128, H, true);
        try p.weight(try layerName(&buf, m, i, "self_attn.k_proj"), 128, H, true);
        try p.weight(try layerName(&buf, m, i, "self_attn.v_proj"), 128, H, true);
        try p.weight(try layerName(&buf, m, i, "self_attn.o_proj"), H, nh * 128, true);
        try p.norm(try layerName(&buf, m, i, "self_attn.q_norm.weight"), 128);
        try p.norm(try layerName(&buf, m, i, "self_attn.k_norm.weight"), 128);
        if (gate) try p.weight(try layerName(&buf, m, i, "self_attn.g_proj"), nh, H, true);
        if (sinks) try p.denseAs(try layerName(&buf, m, i, "self_attn.softmax_offset"), &.{nh}, 1.0, 0, .float32);
        try p.weight(try layerName(&buf, m, i, "mlp.gate_proj"), INTER, H, true);
        try p.weight(try layerName(&buf, m, i, "mlp.up_proj"), INTER, H, true);
        try p.weight(try layerName(&buf, m, i, "mlp.down_proj"), H, INTER, true);
    }
}

fn lagunaLayer(comptime M: type, lg: *const fixture.Legacy, lw: anytype, li: u32, ring: u32) M.Layer {
    const cfg = &lg.config;
    const x = &lg.xfm;
    const hidden = cfg.hidden_size;
    const none = Array{};
    const fa: *const FullAttnWeights = &lw.attn.full;
    const full = cfg.isGlobalLayer(li);
    const yarn = full and x.rope_freqs_yarn != null;
    const heads = cfg.layerNumHeads(li);
    const pf = if (yarn) cfg.partial_rotary_factor_global else cfg.partial_rotary_factor;
    const dw = &lw.mlp.dense;
    return .{
        .input_norm = lw.input_norm,
        .mixer = .{
            .q = lg.lin(fa.q_w, fa.q_s, fa.q_b, none, hidden),
            .k = lg.lin(fa.k_w, fa.k_s, fa.k_b, none, hidden),
            .v = lg.lin(fa.v_w, fa.v_s, fa.v_b, none, hidden),
            .o = lg.lin(fa.o_w, fa.o_s, fa.o_b, none, heads * cfg.head_dim),
            .q_norm = fa.q_norm,
            .k_norm = fa.k_norm,
            .gate = if (cfg.laguna_attn_gate) lg.lin(fa.g_w, fa.g_s, fa.g_b, none, hidden) else .{},
            .sinks = fa.sinks,
            .geo = .{
                .heads = @intCast(heads),
                .kv_heads = @intCast(cfg.num_key_value_heads),
                .head_dim = @intCast(cfg.head_dim),
                .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.query_pre_attn_scalar))),
                .rope_dims = @intFromFloat(@as(f32, @floatFromInt(cfg.head_dim)) * pf),
                .rope_base = if (full) cfg.rope_theta else cfg.rope_local_base_freq,
                .rope_scale = 1.0,
                .nope = cfg.layerSkipsRope(li),
                .freqs = if (yarn) x.rope_freqs_yarn.? else .{},
            },
            .mscale = if (yarn) x.yarn_mscale orelse none else none,
            .window = if (full) 0 else cfg.sliding_window,
            .ring = if (full) 0 else ring,
        },
        .post_norm = lw.post_attn_norm,
        .mlp = .{
            .gate = lg.lin(dw.gate_w, dw.gate_s, dw.gate_b, none, hidden),
            .up = lg.lin(dw.up_w, dw.up_s, dw.up_b, none, hidden),
            .down = lg.lin(dw.down_w, dw.down_s, dw.down_b, none, cfg.intermediate_size),
            .gu = lg.joined(dw.gu, hidden),
        },
    };
}

fn lagunaWeights(comptime M: type, lg: *fixture.Legacy, layers: []M.Layer, ring: u32) M.Weights {
    const x = &lg.xfm;
    const hidden = lg.config.hidden_size;
    const none = Array{};
    for (layers, x.moe_layers.?, 0..) |*L, *lw, i| L.* = lagunaLayer(M, lg, lw, @intCast(i), ring);
    return .{
        .embed = .{ .table = lg.lin(x.emb_w, x.emb_s, x.emb_b, none, hidden), .hidden = @intCast(hidden), .act = lg.config.actDtype() },
        .layers = layers,
        .final_norm = x.final_norm,
        .head = lg.lin(x.lm_head_w, x.lm_head_s, x.lm_head_b, none, hidden),
    };
}

fn loadLaguna(seed: u64, flint: bool) !*fixture.Legacy {
    var p = fixture.Pack.init(seed);
    defer p.deinit();
    if (flint)
        try writeLaguna(&p, &.{ 2, 2, 2, 2 }, false, true)
    else
        try writeLaguna(&p, &.{ 2, 4, 4, 2 }, true, false);
    const lg = try p.load(if (flint) flint_config else laguna_config);
    errdefer lg.deinit();
    if (lg.xfm.moe_layers == null) return error.NotLayered;
    return lg;
}

/// A longer run: a prompt past the window, a second chunk, verify widths and decodes all across it.
const long_steps = [_]fixture.Step{
    .{ .mode = .prefill, .ids = &.{ 3, 17, 42, 99, 7, 250, 1, 8, 66, 31 } },
    .{ .mode = .decode, .ids = &.{88} },
    .{ .mode = .verify, .ids = &.{ 5, 61, 200 } },
    .{ .mode = .prefill, .ids = &.{ 9, 33, 77, 140, 2, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30 } },
    .{ .mode = .decode, .ids = &.{64} },
    .{ .mode = .verify, .ids = &.{ 9, 33, 77, 140, 2 } },
    .{ .mode = .decode, .ids = &.{12} },
};

// ── batched rows ──

/// Two slots prefilled apart through both paths, then decoded together: each batched row bit-equal to the legacy
/// batched forward (`forwardBatchedDecode`, or `forwardMoeBatchedDecode` for a layered trunk).
fn batchedCase(comptime M: type, lg: *fixture.Legacy, w: *const M.Weights, comptime layered: bool) !void {
    try testing.expect(M.batches_decode);
    const a = testing.allocator;
    var shared = bases.Shared.init(lg.config.rms_norm_eps, lg.config.postNormEps());
    defer shared.deinit();
    const s = lg.xfm.s;
    const n_layers = lg.config.num_hidden_layers;
    const prompts = [_][]const i32{ &.{ 4, 8, 15, 16, 23, 42, 9 }, &.{ 42, 101, 7 } };
    var legacy_kv: [2]KVCache = .{ try KVCache.init(a, n_layers), try KVCache.init(a, n_layers) };
    defer for (&legacy_kv) |*c| c.deinit();
    var base_kv: [2]KVCache = .{ try KVCache.init(a, n_layers), try KVCache.init(a, n_layers) };
    defer for (&base_kv) |*c| c.deinit();
    var slots: [2]cache_ops.Slot = .{ .{ .kv = &base_kv[0] }, .{ .kv = &base_kv[1] } };
    const handles = [_]abi.Handle{ slots[0].handle(), slots[1].handle() };
    var ctxs: [2]ForwardCtx = undefined;
    var offsets: [2]usize = .{ 0, 0 };
    var positions: [2]u32 = undefined;
    for (prompts, 0..) |pr, i| {
        var g = ops.Graph.init(a, s);
        defer g.deinit();
        ctxs[i] = lg.xfm.defaultCtx();
        ctxs[i].cache = &legacy_kv[i];
        ctxs[i].moe_seq_offset = &offsets[i];
        const ids = try fixture.tokens(&g, pr);
        const want = try fixture.rows2(&g, try g.adopt(try lg.xfm.forwardWith(&ctxs[i], ids)));
        const got = try M.forward(w, &shared, &g, ids, .{ .mode = .prefill, .rows = @intCast(pr.len), .positions = &.{0}, .caches = handles[i .. i + 1], .cache_ops = &cache_ops.ops, .logits = .all_rows });
        try ops.expectBitEqual(got.logits, want);
        positions[i] = @intCast(pr.len);
    }
    const ctx_ptrs = [_]*ForwardCtx{ &ctxs[0], &ctxs[1] };
    for ([_][2]u32{ .{ 77, 3 }, .{ 190, 64 }, .{ 11, 12 }, .{ 13, 250 } }) |next| {
        var g = ops.Graph.init(a, s);
        defer g.deinit();
        const legacy = if (layered)
            try lg.xfm.forwardMoeBatchedDecode(&next, &ctx_ptrs, &positions, null)
        else
            try lg.xfm.forwardBatchedDecode(&next, &ctx_ptrs, &positions);
        defer a.free(legacy);
        for (legacy) |x| _ = try g.adopt(x);
        const ids = try g.fromSlice(i32, &.{ @intCast(next[0]), @intCast(next[1]) }, &.{ 2, 1 });
        const got = try M.forward(w, &shared, &g, ids, .{ .mode = .batched_rows, .rows = 1, .slots = 2, .positions = &positions, .caches = &handles, .cache_ops = &cache_ops.ops, .logits = .last_row });
        for (legacy, 0..) |x, i| try ops.expectBitEqual(try g.sliceAxis(got.logits, 0, @intCast(i), @intCast(i + 1)), try fixture.rows2(&g, x));
        for (&positions) |*pp| pp.* += 1;
    }
}

// ── tests ──

test "bases attention: Gemma 3 sliding layers, sandwich norms and softcap are bit-equal to forwardStandardWith" {
    const lg = try loadGemma(0x9e33, false);
    defer lg.deinit();
    try testing.expect(lg.config.has_sliding_window and lg.xfm.softcap_scalar != null);
    var layers: [4]Gemma.Layer = undefined;
    const w = try gemmaWeights(lg, &layers);
    try fixture.expectStepsOf(Gemma, lg, &w, &long_steps);
}

test "bases attention: Gemma 3 batched rows across the window are bit-equal to forwardBatchedDecode" {
    const lg = try loadGemma(0x9e34, false);
    defer lg.deinit();
    var layers: [4]Gemma.Layer = undefined;
    const w = try gemmaWeights(lg, &layers);
    try batchedCase(Gemma, lg, &w, false);
}

test "bases attention: Gemma 4 KV sharing, value norm, K=V and proportional RoPE are bit-equal to forwardStandardWith" {
    const lg = try loadGemma(0x9e43, true);
    defer lg.deinit();
    try testing.expect(lg.xfm.layers[4].kv_source != null and lg.xfm.layers[2].k_eq_v and lg.xfm.rope_freqs_global != null);
    var layers: [6]Gemma.Layer = undefined;
    const w = try gemmaWeights(lg, &layers);
    try fixture.expectStepsOf(Gemma, lg, &w, &long_steps);
}

test "bases attention: Gemma 4 batched rows (shared KV, value norm) are bit-equal to forwardBatchedDecode" {
    const lg = try loadGemma(0x9e44, true);
    defer lg.deinit();
    var layers: [6]Gemma.Layer = undefined;
    const w = try gemmaWeights(lg, &layers);
    try batchedCase(Gemma, lg, &w, false);
}

test "bases attention: Spark per-head sigmoid gate (joined at decode) and sliding layers are bit-equal to forwardStandardWith" {
    const lg = try loadSpark(0x5a2c);
    defer lg.deinit();
    try testing.expect(lg.config.attn_gate_headwise and lg.xfm.layers[0].qkv.count == 4);
    var layers: [4]Spark.Layer = undefined;
    const w = sparkWeights(lg, &layers);
    try fixture.expectStepsOf(Spark, lg, &w, &long_steps);
}

test "bases attention: Spark batched rows are bit-equal to forwardBatchedDecode" {
    const lg = try loadSpark(0x5a2d);
    defer lg.deinit();
    var layers: [4]Spark.Layer = undefined;
    const w = sparkWeights(lg, &layers);
    try batchedCase(Spark, lg, &w, false);
}

test "bases attention: a Laguna stack (YaRN full layers, sliding layers, softplus gate) is bit-equal to forwardMoeWith" {
    const lg = try loadLaguna(0x1a90, false);
    defer lg.deinit();
    try testing.expect(lg.config.laguna_attn_gate and lg.xfm.rope_freqs_yarn != null);
    var layers: [4]Laguna.Layer = undefined;
    const w = lagunaWeights(Laguna, lg, &layers, 0);
    try fixture.expectStepsOf(Laguna, lg, &w, &long_steps);
}

test "bases attention: Flint (sinks, sliding and NoPE layers) is bit-equal to forwardMoeWith" {
    const lg = try loadLaguna(0xf117, true);
    defer lg.deinit();
    try testing.expect(lg.config.has_attn_sinks and lg.config.layerSkipsRope(0));
    var layers: [4]Flint.Layer = undefined;
    const w = lagunaWeights(Flint, lg, &layers, 0);
    try fixture.expectStepsOf(Flint, lg, &w, &long_steps);
}

test "bases attention: Flint's sliding layers through a ring cache read the same rows as the legacy full buffer" {
    const lg = try loadLaguna(0xf118, true);
    defer lg.deinit();
    var layers: [4]Flint.Layer = undefined;
    // Two rows past the window: the ring drops rows on most grows over these steps.
    const w = lagunaWeights(Flint, lg, &layers, lg.config.sliding_window + 2);
    try fixture.expectStepsOf(Flint, lg, &w, &long_steps);
}

test "bases attention: Flint batched rows are bit-equal to forwardMoeBatchedDecode" {
    const lg = try loadLaguna(0xf119, true);
    defer lg.deinit();
    var layers: [4]Flint.Layer = undefined;
    const w = lagunaWeights(Flint, lg, &layers, 0);
    try batchedCase(Flint, lg, &w, true);
}
