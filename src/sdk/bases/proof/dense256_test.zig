//! Parity proof for the dense decoder's attention at head_dim 256, whose decode step runs the fused q/k norm+RoPE
//! kernel (`fusedQkNormRope256`): a qwen3 (q/k norms, bf16) and a llama (no norms, 4-bit) through the base vs
//! `forwardStandardWith`.

const std = @import("std");
const bases = @import("../bases.zig");
const fixture = @import("fixture.zig");
const dense = @import("dense_test.zig");

const testing = std.testing;

const H: c_int = 256;
const HEADS: c_int = 2;
const KV_HEADS: c_int = 1;
const HD: c_int = 256;
const INTER: c_int = 512;
const VOCAB: c_int = 256;
const LAYERS = 2;

fn config(comptime model_type: []const u8, comptime quantized: bool) []const u8 {
    return "{\"model_type\":\"" ++ model_type ++ "\",\"hidden_size\":256,\"intermediate_size\":512,\"num_hidden_layers\":2," ++
        "\"num_attention_heads\":2,\"num_key_value_heads\":1,\"head_dim\":256,\"vocab_size\":256,\"rms_norm_eps\":1e-6," ++
        "\"rope_theta\":1000000.0,\"max_position_embeddings\":4096,\"tie_word_embeddings\":false,\"hidden_act\":\"silu\"" ++
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

fn case(comptime model_type: []const u8, comptime quantized: bool, qk_norm: bool, cached: bool, seed: u64) !void {
    var p = fixture.Pack.init(seed);
    defer p.deinit();
    try writePack(&p, quantized, qk_norm);
    const lg = try p.load(config(model_type, quantized));
    defer lg.deinit();
    var layers: [LAYERS]dense.Dense.Layer = undefined;
    const w = dense.denseWeights(lg, &layers);
    var angles: bases.blocks.qk_rope.AngleCache = .{};
    defer angles.deinit();
    if (cached) for (&layers) |*L| {
        L.mixer.angles = &angles;
    };
    try fixture.expectSteps(dense.Dense, lg, &w);
}

// ── tests ──

test "bases decoder: qwen3 bf16 at head_dim 256 (fused q/k norm+RoPE decode) is bit-equal to forwardStandardWith" {
    try case("qwen3", false, true, true, 0x256a);
}

test "bases decoder: llama 4-bit at head_dim 256, angle rows built per call, is bit-equal to forwardStandardWith" {
    try case("llama", true, false, false, 0x256b);
}
