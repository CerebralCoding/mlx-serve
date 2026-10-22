//! plugins/glm5_next/load, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns loadMhcWeights, loadDsaWeightsAt.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const Weights = @import("../../engines/mlx/model.zig").Weights;
const QW = @import("../../engines/mlx/layer_weights.zig").QW;
const f32Table = @import("../../engines/mlx/load/weights.zig").f32Table;
const getLayerWeightAt = @import("../../engines/mlx/load/weights.zig").getLayerWeightAt;
const getWeightAt = @import("../../engines/mlx/load/weights.zig").getWeightAt;
const loadQWAt = @import("../../engines/mlx/load/weights.zig").loadQWAt;
const maybeTransposeForBf16 = @import("../../engines/mlx/load/weights.zig").maybeTransposeForBf16;
const DsaWeights = @import("weights.zig").DsaWeights;
const MhcWeights = @import("weights.zig").MhcWeights;

pub fn loadMhcWeights(weights: *const Weights, name_buf: *[256]u8, prefix: []const u8, li: u32, base: []const u8, owned: *std.ArrayList(mlx.mlx_array), allocator: std.mem.Allocator, s: mlx.mlx_stream) !MhcWeights {
    return .{
        .fn_rows = try f32Table(try getLayerWeightAt(weights, name_buf, prefix, li, base, "fn"), false, owned, allocator, s),
        .base = try f32Table(try getLayerWeightAt(weights, name_buf, prefix, li, base, "base"), false, owned, allocator, s),
        .scale = try f32Table(try getLayerWeightAt(weights, name_buf, prefix, li, base, "scale"), false, owned, allocator, s),
    };
}

/// GLM-5-Next sparse latent attention weights under `<base>.self_attn.*` (a trunk layer or the
/// MTP block); dense bf16 projections are transposed for `qmatmul`.
pub fn loadDsaWeightsAt(weights: *const Weights, name_buf: *[256]u8, base: []const u8, owned_bf16: *std.ArrayList(mlx.mlx_array), allocator: std.mem.Allocator, s: mlx.mlx_stream) !DsaWeights {
    var dsa: DsaWeights = .{
        .q_a = try loadQWAt(weights, name_buf, base, "self_attn.q_a_proj"),
        .kv_a = try loadQWAt(weights, name_buf, base, "self_attn.kv_a_proj_with_mqa"),
        .q_a_norm = try getWeightAt(weights, name_buf, base, "self_attn.q_a_layernorm.weight"),
        .kv_a_norm = try getWeightAt(weights, name_buf, base, "self_attn.kv_a_layernorm.weight"),
        .q_b = try loadQWAt(weights, name_buf, base, "self_attn.q_b_proj"),
        .embed_q = try loadQWAt(weights, name_buf, base, "self_attn.embed_q"),
        .unembed = try loadQWAt(weights, name_buf, base, "self_attn.unembed_out"),
        .o = try loadQWAt(weights, name_buf, base, "self_attn.o_proj"),
        .idx_wq_b = try loadQWAt(weights, name_buf, base, "self_attn.indexer.wq_b"),
        .idx_wk = try loadQWAt(weights, name_buf, base, "self_attn.indexer.wk"),
        .idx_k_norm_w = try getWeightAt(weights, name_buf, base, "self_attn.indexer.k_norm.weight"),
        .idx_k_norm_b = try getWeightAt(weights, name_buf, base, "self_attn.indexer.k_norm.bias"),
        .idx_wproj = try loadQWAt(weights, name_buf, base, "self_attn.indexer.weights_proj"),
        .idx_gate_t = try f32Table(try getWeightAt(weights, name_buf, base, "self_attn.indexer.index_kpool_compress_gate"), true, owned_bf16, allocator, s),
        .idx_ape = try f32Table(try getWeightAt(weights, name_buf, base, "self_attn.indexer.index_kpool_compress_ape"), false, owned_bf16, allocator, s),
    };
    for ([_]*QW{ &dsa.q_a, &dsa.kv_a, &dsa.q_b, &dsa.o, &dsa.idx_wq_b, &dsa.idx_wk, &dsa.idx_wproj }) |q| {
        try maybeTransposeForBf16(&q.w, q.s, owned_bf16, allocator, s);
    }
    return dsa;
}
