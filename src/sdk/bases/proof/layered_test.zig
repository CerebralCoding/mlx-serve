//! Parity proof for the decoder base on the layered path (`forwardMoeWith`, qwen3_5): GatedDeltaNet and gated full
//! attention layers interleaved, a dense SwiGLU, the add+norm seams. The two mixers are still `*Transformer` blocks,
//! reached here through test adapters; the skeleton, seams, MLP, embedding and head are the base's own.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const bases = @import("../bases.zig");
const fixture = @import("fixture.zig");
const ForwardCtx = @import("../../../engines/mlx/forward_ctx.zig").ForwardCtx;
const Transformer = @import("../../../engines/mlx/transformer.zig").Transformer;
const LinearAttnWeights = @import("../../blocks/gdn/gated_delta_net.zig").LinearAttnWeights;
const FullAttnWeights = @import("../../../engines/mlx/layer_weights.zig").FullAttnWeights;

const testing = std.testing;
const Array = mlx.mlx_array;
const b = bases.blocks;
const Ctx = bases.Ctx;

/// Test adapter: the legacy GatedDeltaNet over the slot's recurrent entry.
const Gdn = struct {
    pub const kind: bases.block.Kind = .mixer;
    pub const supports_batched_rows = false;
    pub const Weights = struct { xfm: *Transformer, la: *const LinearAttnWeights };

    pub fn forward(w: *const Weights, c: *Ctx, x: Array) !Array {
        const sl = fixture.slotOf(c);
        return c.g.adopt(try w.xfm.gatedDeltaNet(x, w.la, &sl.ssm[c.layer], c.layer, 1, c.rows, c.wide()));
    }
};

/// Test adapter: the legacy gated full attention over the slot's KV cache.
const GatedFull = struct {
    pub const kind: bases.block.Kind = .mixer;
    pub const supports_batched_rows = false;
    pub const Weights = struct { xfm: *Transformer, fa: *const FullAttnWeights };

    pub fn forward(w: *const Weights, c: *Ctx, x: Array) !Array {
        const sl = fixture.slotOf(c);
        var off: usize = c.offset();
        var fctx: ForwardCtx = .{ .cache = sl.kv, .moe_seq_offset = &off, .ssm_entries = sl.ssm, .capture_hidden = null, .vision_embeddings = null };
        return c.g.adopt(try w.xfm.gatedFullAttnWith(&fctx, x, w.fa, c.layer, @intCast(c.offset()), 1, c.rows, c.wide()));
    }
};

pub const Layered = bases.Decoder(.{
    .embed = b.embed.TableEmbed,
    .norm = b.norm.RmsNorm,
    .mixer = bases.Either(Gdn, GatedFull),
    .mlp = b.mlp.GatedMlp,
    .seam = b.seam.AddNorm,
    .head = b.head.LmHead,
});

const H: c_int = 128;
const HEADS: c_int = 4;
const KV_HEADS: c_int = 2;
const HD: c_int = 64;
const INTER: c_int = 256;
const VOCAB: c_int = 256;
const LAYERS = 4;
const HK: c_int = 2;
const HV: c_int = 4;
const DK: c_int = 32;
const DV: c_int = 32;
const KERNEL: c_int = 4;

const config_json =
    \\{"model_type":"qwen3_5","text_config":{"model_type":"qwen3_5","hidden_size":128,"intermediate_size":256,
    \\"num_hidden_layers":4,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":64,"vocab_size":256,
    \\"rms_norm_eps":1e-6,"full_attention_interval":2,"linear_num_key_heads":2,"linear_num_value_heads":4,
    \\"linear_key_head_dim":32,"linear_value_head_dim":32,"linear_conv_kernel_dim":4,"max_position_embeddings":4096,
    \\"rope_parameters":{"rope_theta":100000.0,"partial_rotary_factor":0.25},"tie_word_embeddings":false},
    \\"tie_word_embeddings":false,"quantization":{"group_size":64,"bits":4}}
;

fn at(buf: []u8, layer: usize, name: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "language_model.model.layers.{d}.{s}", .{ layer, name });
}

fn writePack(p: *fixture.Pack) !void {
    var buf: [160]u8 = undefined;
    try p.weight("language_model.model.embed_tokens", VOCAB, H, true);
    try p.norm("language_model.model.norm.weight", H);
    try p.weight("language_model.lm_head", VOCAB, H, true);
    for (0..LAYERS) |i| {
        try p.norm(try at(&buf, i, "input_layernorm.weight"), H);
        try p.norm(try at(&buf, i, "post_attention_layernorm.weight"), H);
        try p.weight(try at(&buf, i, "mlp.gate_proj"), INTER, H, true);
        try p.weight(try at(&buf, i, "mlp.up_proj"), INTER, H, true);
        try p.weight(try at(&buf, i, "mlp.down_proj"), H, INTER, true);
        if ((i + 1) % 2 != 0) {
            try p.weight(try at(&buf, i, "linear_attn.in_proj_qkv"), 2 * HK * DK + HV * DV, H, true);
            try p.weight(try at(&buf, i, "linear_attn.in_proj_z"), HV * DV, H, true);
            try p.weight(try at(&buf, i, "linear_attn.in_proj_a"), HV, H, true);
            try p.weight(try at(&buf, i, "linear_attn.in_proj_b"), HV, H, true);
            try p.dense(try at(&buf, i, "linear_attn.conv1d.weight"), &.{ 2 * HK * DK + HV * DV, KERNEL, 1 }, 0.5, 0);
            try p.dense(try at(&buf, i, "linear_attn.A_log"), &.{HV}, 0.5, 0.2);
            try p.dense(try at(&buf, i, "linear_attn.dt_bias"), &.{HV}, 0.5, 0);
            try p.norm(try at(&buf, i, "linear_attn.norm.weight"), DV);
            try p.weight(try at(&buf, i, "linear_attn.out_proj"), H, HV * DV, true);
        } else {
            try p.weight(try at(&buf, i, "self_attn.q_proj"), 2 * HEADS * HD, H, true);
            try p.weight(try at(&buf, i, "self_attn.k_proj"), KV_HEADS * HD, H, true);
            try p.weight(try at(&buf, i, "self_attn.v_proj"), KV_HEADS * HD, H, true);
            try p.weight(try at(&buf, i, "self_attn.o_proj"), H, HEADS * HD, true);
            try p.norm(try at(&buf, i, "self_attn.q_norm.weight"), HD);
            try p.norm(try at(&buf, i, "self_attn.k_norm.weight"), HD);
        }
    }
}

fn layeredWeights(lg: *fixture.Legacy, layers: []Layered.Layer) Layered.Weights {
    const x = &lg.xfm;
    const hidden = lg.config.hidden_size;
    const none = Array{};
    for (layers, x.moe_layers.?) |*L, *lw| {
        const dw = &lw.mlp.dense;
        L.* = .{
            .input_norm = lw.input_norm,
            .post_norm = lw.post_attn_norm,
            .mixer = switch (lw.attn) {
                .linear => |*la| .{ .first = .{ .xfm = x, .la = la } },
                .full => |*fa| .{ .second = .{ .xfm = x, .fa = fa } },
            },
            .mlp = .{
                .gate = lg.lin(dw.gate_w, dw.gate_s, dw.gate_b, none, hidden),
                .up = lg.lin(dw.up_w, dw.up_s, dw.up_b, none, hidden),
                .down = lg.lin(dw.down_w, dw.down_s, dw.down_b, none, lg.config.intermediate_size),
                .gu = lg.joined(dw.gu, hidden),
            },
        };
    }
    return .{
        .embed = .{ .table = lg.lin(x.emb_w, x.emb_s, x.emb_b, none, hidden), .hidden = @intCast(hidden), .act = lg.config.actDtype() },
        .layers = layers,
        .final_norm = x.final_norm,
        .head = lg.lin(x.lm_head_w, x.lm_head_s, x.lm_head_b, none, hidden),
    };
}

// ── tests ──

test "bases decoder: qwen3_5 GatedDeltaNet + gated attention prefill, decode and verify are bit-equal to forwardMoeWith" {
    var p = fixture.Pack.init(0x35a5);
    defer p.deinit();
    try writePack(&p);
    const lg = try p.load(config_json);
    defer lg.deinit();
    try testing.expect(lg.xfm.moe_layers != null and lg.xfm.ssm_entries != null);
    var layers: [LAYERS]Layered.Layer = undefined;
    const w = layeredWeights(lg, &layers);
    try fixture.expectSteps(Layered, lg, &w);
}
