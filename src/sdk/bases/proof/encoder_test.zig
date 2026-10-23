//! Parity proof for the encoder base (`forwardBertWith`): a 4-bit BERT of tiny random weights, the base's hidden
//! rows bit-equal to the legacy forward's for one sequence and for a padded batch under its key mask; and the
//! projection head bit-equal to `embedProjection` (a quantized dense.0, a bf16 dense.1). BERT's embedding (word +
//! position + token type, LayerNorm) is family code and lives here as a proof block until P3 moves it.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const bases = @import("../bases.zig");
const fixture = @import("fixture.zig");
const buildKeyPadMask = @import("../../../engines/mlx/embed/pooling.zig").buildKeyPadMask;

const testing = std.testing;
const Array = mlx.mlx_array;
const b = bases.blocks;
const Ctx = bases.Ctx;
const Linear = b.linear.Linear;

/// copied_from: plugins/bert/forward.zig (`bertEmbedding`). Word + position + token-type rows (type 0), LayerNorm.
const BertEmbed = struct {
    pub const kind: bases.block.Kind = .embed;
    pub const supports_batched_rows = false;
    pub const Weights = struct { word: Linear, pos: Linear, toktype: Linear, norm: b.norm.LayerNorm.Weights, hidden: c_int };

    pub fn forward(w: *const Weights, c: *Ctx, ids: Array) !Array {
        const g = c.g;
        const sh = ops.shape(ids);
        const flat = try g.reshape(ids, &.{sh[0] * sh[1]});
        const word = try g.reshape(try b.embed.gatherRows(g, &w.word, flat, .bfloat16), &.{ sh[0], sh[1], w.hidden });
        const row = [_]c_int{ 1, sh[1], w.hidden };
        const pos = try g.reshape(try b.embed.gatherRows(g, &w.pos, try g.arange(0, @floatFromInt(sh[1]), 1, .int32), .bfloat16), &row);
        const tt = try g.reshape(try b.embed.gatherRows(g, &w.toktype, try g.zeros(&.{sh[1]}, .int32), .bfloat16), &row);
        return b.norm.LayerNorm.forward(&w.norm, c, try g.add(try g.add(word, pos), tt));
    }
};

pub const Bert = bases.Encoder(.{
    .embed = BertEmbed,
    .attention = b.attention.BidirAttention,
    .mlp = b.mlp.GeluMlp,
    .norm = b.norm.LayerNorm,
});

const H: c_int = 128;
const INTER: c_int = 256;
const VOCAB: c_int = 256;
const LAYERS = 2;

const config_json =
    \\{"model_type":"bert","hidden_size":128,"intermediate_size":256,"num_hidden_layers":2,"num_attention_heads":4,
    \\"vocab_size":256,"max_position_embeddings":64,"type_vocab_size":2,"layer_norm_eps":1e-12,
    \\"quantization":{"group_size":64,"bits":4}}
;

fn ln(p: *fixture.Pack, buf: []u8, base: []const u8) !void {
    try p.norm(try std.fmt.bufPrint(buf, "{s}.weight", .{base}), H);
    try p.dense(try std.fmt.bufPrint(buf, "{s}.bias", .{base}), &.{H}, 0.1, 0);
}

fn lin(p: *fixture.Pack, buf: []u8, base: []const u8, rows: c_int, cols: c_int) !void {
    try p.weight(base, rows, cols, true);
    try p.dense(try std.fmt.bufPrint(buf, "{s}.bias", .{base}), &.{rows}, 0.1, 0);
}

fn writePack(p: *fixture.Pack) !void {
    var buf: [160]u8 = undefined;
    var nb: [160]u8 = undefined;
    try p.weight("embeddings.word_embeddings", VOCAB, H, true);
    try p.weight("embeddings.position_embeddings", 64, H, true);
    try p.weight("embeddings.token_type_embeddings", 2, H, true);
    try ln(p, &buf, "embeddings.LayerNorm");
    for (0..LAYERS) |i| {
        const at = struct {
            fn f(bb: []u8, layer: usize, name: []const u8) ![]const u8 {
                return std.fmt.bufPrint(bb, "encoder.layer.{d}.{s}", .{ layer, name });
            }
        }.f;
        try lin(p, &buf, try at(&nb, i, "attention.self.query"), H, H);
        try lin(p, &buf, try at(&nb, i, "attention.self.key"), H, H);
        try lin(p, &buf, try at(&nb, i, "attention.self.value"), H, H);
        try lin(p, &buf, try at(&nb, i, "attention.output.dense"), H, H);
        try ln(p, &buf, try at(&nb, i, "attention.output.LayerNorm"));
        try lin(p, &buf, try at(&nb, i, "intermediate.dense"), INTER, H);
        try lin(p, &buf, try at(&nb, i, "output.dense"), H, INTER);
        try ln(p, &buf, try at(&nb, i, "output.LayerNorm"));
    }
    // A projection head beside the trunk (EmbeddingGemma's dense.0 / dense.1 shape), wired by hand in the test.
    try p.weight("dense.0", 2 * H, H, true);
    try p.weight("dense.1", H, 2 * H, false);
}

/// `Transformer.deinit` never frees the BERT layer array (a leak on the legacy path, one per load).
fn freeBertLayers(lg: *fixture.Legacy) void {
    if (lg.xfm.bert_layers) |bl| testing.allocator.free(bl);
}

fn bertWeights(lg: *const fixture.Legacy, layers: []Bert.Layer) Bert.Weights {
    const x = &lg.xfm;
    const hidden = lg.config.hidden_size;
    const none = Array{};
    for (layers, x.bert_layers.?) |*L, *lw| {
        L.* = .{
            .attention = .{
                .q = lg.lin(lw.q_w, lw.q_s, lw.q_b, lw.q_bias, hidden),
                .k = lg.lin(lw.k_w, lw.k_s, lw.k_b, lw.k_bias, hidden),
                .v = lg.lin(lw.v_w, lw.v_s, lw.v_b, lw.v_bias, hidden),
                .o = lg.lin(lw.o_w, lw.o_s, lw.o_b, lw.o_bias, hidden),
                .heads = @intCast(lg.config.num_attention_heads),
                .head_dim = @intCast(lg.config.head_dim),
            },
            .attn_norm = .{ .w = lw.attn_norm_w, .b = lw.attn_norm_b },
            .mlp = .{
                .inter = lg.lin(lw.inter_w, lw.inter_s, lw.inter_b, lw.inter_bias, hidden),
                .out = lg.lin(lw.out_w, lw.out_s, lw.out_b, lw.out_bias, lg.config.intermediate_size),
            },
            .mlp_norm = .{ .w = lw.out_norm_w, .b = lw.out_norm_b },
        };
    }
    return .{
        .embed = .{
            .word = lg.lin(x.emb_w, x.emb_s, x.emb_b, none, hidden),
            .pos = lg.lin(x.bert_pos_w, x.bert_pos_s, x.bert_pos_b, none, hidden),
            .toktype = lg.lin(x.bert_toktype_w, x.bert_toktype_s, x.bert_toktype_b, none, hidden),
            .norm = .{ .w = x.bert_emb_norm_w, .b = x.bert_emb_norm_b },
            .hidden = @intCast(hidden),
        },
        .layers = layers,
    };
}

// ── tests ──

test "bases encoder: BERT hidden rows (one sequence and a padded batch) are bit-equal to forwardBertWith" {
    const a = testing.allocator;
    var p = fixture.Pack.init(0xbe27);
    defer p.deinit();
    try writePack(&p);
    const lg = try p.load(config_json);
    defer lg.deinit();
    defer freeBertLayers(lg);
    var layers: [LAYERS]Bert.Layer = undefined;
    const w = bertWeights(lg, &layers);
    var shared = bases.Shared.init(lg.config.layer_norm_eps, lg.config.layer_norm_eps);
    defer shared.deinit();

    {
        var g = ops.Graph.init(a, lg.xfm.s);
        defer g.deinit();
        const ids = try fixture.tokens(&g, &.{ 101, 7, 42, 250, 3, 102 });
        const want = try g.adopt(try lg.xfm.forwardEmbeddingMasked(ids, null, null));
        const got = try Bert.forward(&w, &shared, &g, ids, .{ .mode = .prefill, .rows = 6, .positions = &.{0} });
        try ops.expectBitEqual(got.hidden, want);
    }
    {
        var g = ops.Graph.init(a, lg.xfm.s);
        defer g.deinit();
        const ids = try g.fromSlice(i32, &.{ 101, 7, 42, 250, 102, 101, 9, 102, 0, 0 }, &.{ 2, 5 });
        const mask = try g.adopt(try buildKeyPadMask(a, &.{ 5, 3 }, 5, lg.xfm.s));
        const want = try g.adopt(try lg.xfm.forwardEmbeddingMasked(ids, mask, null));
        const got = try Bert.forward(&w, &shared, &g, ids, .{ .mode = .prefill, .rows = 5, .slots = 2, .positions = &.{ 0, 0 }, .mask = mask });
        try ops.expectBitEqual(got.hidden, want);
    }
}

test "bases encoder: the projection head is bit-equal to embedProjection" {
    const a = testing.allocator;
    var p = fixture.Pack.init(0xde05);
    defer p.deinit();
    try writePack(&p);
    const lg = try p.load(config_json);
    defer lg.deinit();
    defer freeBertLayers(lg);
    const get = struct {
        fn f(l: *const fixture.Legacy, name: []const u8) Array {
            return l.weights.get(name) orelse .{};
        }
    }.f;
    const x = &lg.xfm;
    x.dense0_w = get(lg, "dense.0.weight");
    x.dense0_s = get(lg, "dense.0.scales");
    x.dense0_b = get(lg, "dense.0.biases");
    x.dense1_w = get(lg, "dense.1.weight");
    var layers: [LAYERS]Bert.Layer = undefined;
    var w = bertWeights(lg, &layers);
    w.projection = .{
        .first = lg.lin(x.dense0_w, x.dense0_s, x.dense0_b, .{}, lg.config.hidden_size),
        .second = .{ .w = x.dense1_w },
    };

    var g = ops.Graph.init(a, x.s);
    defer g.deinit();
    var host: [2 * 128]f32 = undefined;
    for (&host, 0..) |*v, i| v.* = @sin(0.37 * @as(f32, @floatFromInt(i)));
    const pooled = try g.fromSlice(f32, &host, &.{ 2, H });
    const want = try g.adopt(try x.embedProjection(pooled));
    try ops.expectBitEqual(try Bert.project(&w, &g, pooled), want);
}
