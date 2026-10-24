//! Parity proofs for the attention extras: Gemma 3 (sliding layers under band masks, sandwich norms, GeGLU, logit
//! softcap) and Gemma 4 dense (KV-sharing layers, value norm, K=V full layers, dual head dims, proportional RoPE)
//! against `forwardStandardWith` and `forwardBatchedDecode`. Prefill, decode and verify widths cross every window.
//! `LagunaAttention` is proven by the laguna and flint plugins, `DecoderAttention(.gate = .head)` by spark2_5's
//! `bases_test.zig` (their legacy forwards are deleted; tapes recorded from them).

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

/// Six layers: sliding, sliding, full (K=V), sliding, then two that read layers 3 and 2's KV; sliding heads `hd`
/// wide, global heads twice that.
fn gemma4Config(comptime hd: c_int) []const u8 {
    return std.fmt.comptimePrint(gemma4_config, .{ hd, 2 * hd });
}

const gemma4_config =
    \\{{"model_type":"gemma4_text","hidden_size":128,"intermediate_size":256,"num_hidden_layers":6,
    \\"num_attention_heads":4,"num_key_value_heads":2,"head_dim":{d},"global_head_dim":{d},
    \\"num_global_key_value_heads":1,"num_kv_shared_layers":2,"attention_k_eq_v":true,"vocab_size":256,
    \\"rms_norm_eps":1e-6,"sliding_window":4,"tie_word_embeddings":true,"hidden_size_per_layer_input":0,
    \\"layer_types":["sliding_attention","sliding_attention","full_attention","sliding_attention","sliding_attention","full_attention"],
    \\"rope_parameters":{{"full_attention":{{"rope_theta":1000000.0,"rope_type":"proportional","partial_rotary_factor":0.25,"factor":1.0}},
    \\"sliding_attention":{{"rope_theta":10000.0}}}},"final_logit_softcapping":30.0,"max_position_embeddings":4096,
    \\"quantization":{{"group_size":64,"bits":4}}}}
;

fn layerName(buf: []u8, prefix: []const u8, layer: usize, name: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}.layers.{d}.{s}", .{ prefix, layer, name });
}

/// A Gemma pack: `layers` of `(global, kv_shared)`; sliding layers project `hd`-wide heads, global layers `ghd`-wide
/// ones over `gkv` KV heads, sharing V with K when `k_eq_v`.
fn writeGemma(p: *fixture.Pack, prefix: []const u8, kinds: []const [2]bool, hd_local: c_int, ghd: c_int, gkv: c_int, k_eq_v: bool) !void {
    var buf: [192]u8 = undefined;
    var nb: [96]u8 = undefined;
    try p.weight(try std.fmt.bufPrint(&nb, "{s}.embed_tokens", .{prefix}), VOCAB, H, true);
    try p.norm(try std.fmt.bufPrint(&nb, "{s}.norm.weight", .{prefix}), H);
    for (kinds, 0..) |kind, i| {
        const global = kind[0];
        const hd: c_int = if (global) ghd else hd_local;
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
        .head = .{ .head = lg.lin(x.lm_head_w, x.lm_head_s, x.lm_head_b, none, hidden), .cap = x.softcap_scalar orelse none, .mult = none },
    };
}

const gemma3_kinds = [_][2]bool{ .{ false, false }, .{ true, false }, .{ false, false }, .{ true, false } };
const gemma4_kinds = [_][2]bool{ .{ false, false }, .{ false, false }, .{ true, false }, .{ false, false }, .{ false, true }, .{ true, true } };

fn loadGemma(seed: u64, gemma4: bool) !*fixture.Legacy {
    return loadGemmaHd(seed, gemma4, 32);
}

/// Gemma 4's sliding heads `hd` wide (256: the fused q/k/v norm+RoPE decode step).
fn loadGemmaHd(seed: u64, gemma4: bool, comptime hd: c_int) !*fixture.Legacy {
    var p = fixture.Pack.init(seed);
    defer p.deinit();
    if (gemma4)
        try writeGemma(&p, "language_model.model", &gemma4_kinds, hd, 2 * hd, 1, true)
    else
        try writeGemma(&p, "model", &gemma3_kinds, 32, 32, 2, false);
    return p.load(if (gemma4) gemma4Config(hd) else gemma3_config);
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

test "bases attention: Gemma 4 at hd 256 (the fused q/k/v norm+RoPE decode step) is bit-equal to forwardStandardWith" {
    const lg = try loadGemmaHd(0x9e45, true, 256);
    defer lg.deinit();
    try testing.expect(lg.config.has_v_norm and lg.config.head_dim == 256);
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

extern "c" fn open_memstream(buf: *?[*]u8, len: *usize) ?*std.c.FILE;

/// `RMSNorm` nodes in the lazy graph of `x`, read off mlx's own printout of it.
fn rmsNormNodes(x: Array) !usize {
    var buf: ?[*]u8 = null;
    var len: usize = 0;
    const f = open_memstream(&buf, &len) orelse return error.NoStream;
    const outs = mlx.mlx_vector_array_new_value(x);
    defer _ = mlx.mlx_vector_array_free(outs);
    const namer = mlx.mlx_node_namer_new();
    defer _ = mlx.mlx_node_namer_free(namer);
    _ = mlx.mlx_print_graph(f, namer, outs);
    _ = std.c.fclose(f);
    defer std.c.free(buf);
    return std.mem.count(u8, buf.?[0..len], "RMSNorm");
}

test "bases attention: a Gemma 3 decode step folds all four sandwich norms of a layer into its seam dispatches" {
    const a = testing.allocator;
    const lg = try loadGemma(0x9e33, false);
    defer lg.deinit();
    var layers: [4]Gemma.Layer = undefined;
    const w = try gemmaWeights(lg, &layers);
    var shared = bases.Shared.init(lg.config.rms_norm_eps, lg.config.postNormEps());
    defer shared.deinit();
    var nodes: [2]usize = undefined;
    for ([_]abi.Mode{ .prefill, .decode }, &nodes) |mode, *n| {
        var kv = try KVCache.init(a, lg.config.num_hidden_layers);
        defer kv.deinit();
        var slot: cache_ops.Slot = .{ .kv = &kv };
        const h = [_]abi.Handle{slot.handle()};
        var g = ops.Graph.init(a, lg.xfm.s);
        defer g.deinit();
        const out = try Gemma.forward(&w, &shared, &g, try fixture.tokens(&g, &.{42}), .{ .mode = mode, .rows = 1, .positions = &.{0}, .caches = &h, .cache_ops = &cache_ops.ops, .logits = .last_row });
        n.* = try rmsNormNodes(out.logits);
    }
    try testing.expectEqual(nodes[0] - 4 * layers.len, nodes[1]);
}
