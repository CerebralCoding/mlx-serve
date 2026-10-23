//! Parity proof for the decoder base on the dense path (`forwardStandardWith`, `forwardBatchedDecode`): a llama
//! (4-bit) and a qwen3 (bf16, q/k norms) of tiny random weights, the base's prefill, decode, verify and batched rows
//! bit-equal to the legacy forward's.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const abi = @import("../../../abi/plugin.zig");
const bases = @import("../bases.zig");
const fixture = @import("fixture.zig");
const cache_ops = @import("../../../engines/mlx/kv/cache_ops.zig");
const KVCache = @import("../../../engines/mlx/kv/cache.zig").KVCache;
const ForwardCtx = @import("../../../engines/mlx/forward_ctx.zig").ForwardCtx;

const testing = std.testing;
const Array = mlx.mlx_array;
const b = bases.blocks;

pub const Dense = bases.Decoder(.{
    .embed = b.embed.TableEmbed,
    .norm = b.norm.RmsNorm,
    .mixer = b.attention.Attention,
    .mlp = b.mlp.SwiGlu,
    .seam = b.seam.ResidualTail,
    .head = b.head.LmHead,
});

const H: c_int = 128;
const HEADS: c_int = 4;
const KV_HEADS: c_int = 2;
const HD: c_int = 32;
const INTER: c_int = 256;
const VOCAB: c_int = 256;
const LAYERS = 2;

fn config(comptime model_type: []const u8, comptime quantized: bool) []const u8 {
    return "{\"model_type\":\"" ++ model_type ++ "\",\"hidden_size\":128,\"intermediate_size\":256,\"num_hidden_layers\":2," ++
        "\"num_attention_heads\":4,\"num_key_value_heads\":2,\"head_dim\":32,\"vocab_size\":256,\"rms_norm_eps\":1e-5," ++
        "\"rope_theta\":10000.0,\"max_position_embeddings\":4096,\"tie_word_embeddings\":false,\"hidden_act\":\"silu\"" ++
        (if (quantized) ",\"quantization\":{\"group_size\":64,\"bits\":4}" else "") ++ "}";
}

fn at(buf: []u8, layer: usize, name: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "model.layers.{d}.{s}", .{ layer, name });
}

fn writePack(p: *fixture.Pack, quantized: bool, qk_norm: bool) !void {
    var buf: [128]u8 = undefined;
    try p.weight("model.embed_tokens", VOCAB, H, quantized);
    try p.norm("model.norm.weight", H);
    try p.weight("lm_head", VOCAB, H, quantized);
    for (0..LAYERS) |i| {
        try p.norm(try at(&buf, i, "input_layernorm.weight"), H);
        try p.norm(try at(&buf, i, "post_attention_layernorm.weight"), H);
        try p.weight(try at(&buf, i, "self_attn.q_proj"), HEADS * HD, H, quantized);
        try p.weight(try at(&buf, i, "self_attn.k_proj"), KV_HEADS * HD, H, quantized);
        try p.weight(try at(&buf, i, "self_attn.v_proj"), KV_HEADS * HD, H, quantized);
        try p.weight(try at(&buf, i, "self_attn.o_proj"), H, HEADS * HD, quantized);
        try p.weight(try at(&buf, i, "mlp.gate_proj"), INTER, H, quantized);
        try p.weight(try at(&buf, i, "mlp.up_proj"), INTER, H, quantized);
        try p.weight(try at(&buf, i, "mlp.down_proj"), H, INTER, quantized);
        if (qk_norm) {
            try p.norm(try at(&buf, i, "self_attn.q_norm.weight"), HD);
            try p.norm(try at(&buf, i, "self_attn.k_norm.weight"), HD);
        }
    }
}

/// The base's weights over the legacy model's loaded arrays (the same handles, so the same bytes).
pub fn denseWeights(lg: *const fixture.Legacy, layers: []Dense.Layer) Dense.Weights {
    const x = &lg.xfm;
    const cfg = &lg.config;
    const hidden = cfg.hidden_size;
    const none = Array{};
    for (layers, x.layers, 0..) |*L, *lw, i| {
        const li: u32 = @intCast(i);
        const qd = cfg.num_attention_heads * cfg.head_dim;
        const rope_dims: c_int = if (cfg.isGlobalLayer(li) and cfg.partial_rotary_factor_global < 1.0)
            @intCast(@as(u32, @intFromFloat(@as(f32, @floatFromInt(cfg.head_dim)) * cfg.partial_rotary_factor_global)))
        else
            @intCast(cfg.head_dim);
        L.* = .{
            .input_norm = lw.input_norm,
            .post_norm = lw.post_attn_norm,
            .mixer = .{
                .q = lg.lin(lw.q_w, lw.q_s, lw.q_b, lw.q_bias, hidden),
                .k = lg.lin(lw.k_w, lw.k_s, lw.k_b, lw.k_bias, hidden),
                .v = lg.lin(lw.v_w, lw.v_s, lw.v_b, lw.v_bias, hidden),
                .o = lg.lin(lw.o_w, lw.o_s, lw.o_b, none, qd),
                .q_norm = lw.q_norm orelse none,
                .k_norm = lw.k_norm orelse none,
                .qkv = lg.joined(lw.qkv, hidden),
                .geo = .{
                    .heads = @intCast(cfg.num_attention_heads),
                    .kv_heads = @intCast(cfg.num_key_value_heads),
                    .head_dim = @intCast(cfg.head_dim),
                    .scale = cfg.attnScale(),
                    .rope_dims = rope_dims,
                    .rope_base = cfg.rope_theta,
                    .rope_scale = 1.0 / cfg.rope_scaling_factor,
                    .nope = cfg.layerSkipsRope(li),
                },
            },
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
        .embed = .{ .table = lg.lin(x.emb_w, x.emb_s, x.emb_b, none, hidden), .hidden = @intCast(hidden), .act = cfg.actDtype(), .scale = x.emb_scale orelse none },
        .layers = layers,
        .final_norm = x.final_norm,
        .head = lg.lin(x.lm_head_w, x.lm_head_s, x.lm_head_b, none, hidden),
    };
}

fn soloCase(comptime model_type: []const u8, comptime quantized: bool, qk_norm: bool, seed: u64) !void {
    var p = fixture.Pack.init(seed);
    defer p.deinit();
    try writePack(&p, quantized, qk_norm);
    const lg = try p.load(config(model_type, quantized));
    defer lg.deinit();
    var layers: [LAYERS]Dense.Layer = undefined;
    const w = denseWeights(lg, &layers);
    try fixture.expectSteps(Dense, lg, &w);
}

/// Two slots prefilled apart, then decoded together: each batched row bit-equal to `forwardBatchedDecode`'s.
fn batchedCase(comptime model_type: []const u8, comptime quantized: bool, qk_norm: bool, seed: u64) !void {
    const a = testing.allocator;
    var p = fixture.Pack.init(seed);
    defer p.deinit();
    try writePack(&p, quantized, qk_norm);
    const lg = try p.load(config(model_type, quantized));
    defer lg.deinit();
    var layers: [LAYERS]Dense.Layer = undefined;
    const w = denseWeights(lg, &layers);
    var shared = bases.Shared.init(lg.config.rms_norm_eps, lg.config.postNormEps());
    defer shared.deinit();
    const s = lg.xfm.s;
    const n_layers = lg.config.num_hidden_layers;

    const prompts = [_][]const i32{ &.{ 4, 8, 15, 16, 23 }, &.{ 42, 101, 7 } };
    var legacy_kv: [2]KVCache = .{ try KVCache.init(a, n_layers), try KVCache.init(a, n_layers) };
    defer for (&legacy_kv) |*c| c.deinit();
    var base_kv: [2]KVCache = .{ try KVCache.init(a, n_layers), try KVCache.init(a, n_layers) };
    defer for (&base_kv) |*c| c.deinit();
    var slots: [2]cache_ops.Slot = .{ .{ .kv = &base_kv[0] }, .{ .kv = &base_kv[1] } };
    const handles = [_]abi.Handle{ slots[0].handle(), slots[1].handle() };
    var ctxs: [2]ForwardCtx = undefined;
    var positions: [2]u32 = undefined;
    for (prompts, 0..) |pr, i| {
        var g = ops.Graph.init(a, s);
        defer g.deinit();
        ctxs[i] = lg.xfm.defaultCtx();
        ctxs[i].cache = &legacy_kv[i];
        const ids = try fixture.tokens(&g, pr);
        const want = try fixture.rows2(&g, try g.adopt(try lg.xfm.forwardWith(&ctxs[i], ids)));
        const got = try Dense.forward(&w, &shared, &g, ids, .{ .mode = .prefill, .rows = @intCast(pr.len), .positions = &.{0}, .caches = handles[i .. i + 1], .cache_ops = &cache_ops.ops, .logits = .all_rows });
        try ops.expectBitEqual(got.logits, want);
        positions[i] = @intCast(pr.len);
    }
    const ctx_ptrs = [_]*ForwardCtx{ &ctxs[0], &ctxs[1] };
    for ([_][2]u32{ .{ 77, 3 }, .{ 190, 64 } }) |next| {
        var g = ops.Graph.init(a, s);
        defer g.deinit();
        const legacy = try lg.xfm.forwardBatchedDecode(&next, &ctx_ptrs, &positions);
        defer a.free(legacy);
        for (legacy) |x| _ = try g.adopt(x);
        const ids = try g.fromSlice(i32, &.{ @intCast(next[0]), @intCast(next[1]) }, &.{ 2, 1 });
        const got = try Dense.forward(&w, &shared, &g, ids, .{ .mode = .batched_rows, .rows = 1, .slots = 2, .positions = &positions, .caches = &handles, .cache_ops = &cache_ops.ops, .logits = .last_row });
        for (legacy, 0..) |x, i| try ops.expectBitEqual(try g.sliceAxis(got.logits, 0, @intCast(i), @intCast(i + 1)), try fixture.rows2(&g, x));
        for (&positions) |*pp| pp.* += 1;
    }
}

// ── tests ──

test "bases decoder: llama 4-bit prefill, decode and verify are bit-equal to forwardStandardWith" {
    try soloCase("llama", true, false, 0x11a3a);
}

test "bases decoder: qwen3 bf16 (q/k norms) prefill, decode and verify are bit-equal to forwardStandardWith" {
    try soloCase("qwen3", false, true, 0x9e3);
}

test "bases decoder: llama 4-bit batched rows are bit-equal to forwardBatchedDecode" {
    try testing.expect(Dense.batches_decode);
    try batchedCase("llama", true, false, 0xba7c4);
}

test "bases decoder: qwen3 bf16 batched rows are bit-equal to forwardBatchedDecode" {
    try batchedCase("qwen3", false, true, 0xba7c5);
}
