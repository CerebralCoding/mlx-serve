//! plugins/bert/load, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns getBertWeight, getBertLayerWeight, initBertLayers, initBert.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../../host/log.zig");
const ModelConfig = @import("../../engines/mlx/model.zig").ModelConfig;
const Weights = @import("../../engines/mlx/model.zig").Weights;
const KVCache = @import("../../engines/mlx/kv/cache.zig").KVCache;
const bf16Scalar = @import("../../engines/mlx/load/weights.zig").bf16Scalar;
const structFields = @import("../../engines/mlx/load/weights.zig").structFields;
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;
const BertLayerWeights = @import("weights.zig").BertLayerWeights;

fn getBertWeight(weights: *const Weights, buf: *[256]u8, name: []const u8) error{MissingWeight}!mlx.mlx_array {
    const n = std.fmt.bufPrint(buf, "{s}", .{name}) catch unreachable;
    return weights.get(n) orelse {
        log.err("MISSING WEIGHT: {s}\n", .{n});
        return error.MissingWeight;
    };
}

fn getBertLayerWeight(weights: *const Weights, buf: *[256]u8, layer: u32, suffix: []const u8) error{MissingWeight}!mlx.mlx_array {
    const name = std.fmt.bufPrint(buf, "encoder.layer.{d}.{s}", .{ layer, suffix }) catch unreachable;
    return weights.get(name) orelse {
        log.err("MISSING WEIGHT: {s}\n", .{name});
        return error.MissingWeight;
    };
}

fn initBertLayers(allocator: std.mem.Allocator, config: ModelConfig, weights: *const Weights, name_buf: *[256]u8) ![]BertLayerWeights {
    log.info("Precomputing BERT layer weights...\n", .{});
    const layers = try allocator.alloc(BertLayerWeights, config.num_hidden_layers);

    for (0..config.num_hidden_layers) |i| {
        const li: u32 = @intCast(i);
        const lw = &layers[i];

        lw.q_w = try getBertLayerWeight(weights, name_buf, li, "attention.self.query.weight");
        lw.q_s = try getBertLayerWeight(weights, name_buf, li, "attention.self.query.scales");
        lw.q_b = try getBertLayerWeight(weights, name_buf, li, "attention.self.query.biases");
        lw.q_bias = try getBertLayerWeight(weights, name_buf, li, "attention.self.query.bias");
        lw.k_w = try getBertLayerWeight(weights, name_buf, li, "attention.self.key.weight");
        lw.k_s = try getBertLayerWeight(weights, name_buf, li, "attention.self.key.scales");
        lw.k_b = try getBertLayerWeight(weights, name_buf, li, "attention.self.key.biases");
        lw.k_bias = try getBertLayerWeight(weights, name_buf, li, "attention.self.key.bias");
        lw.v_w = try getBertLayerWeight(weights, name_buf, li, "attention.self.value.weight");
        lw.v_s = try getBertLayerWeight(weights, name_buf, li, "attention.self.value.scales");
        lw.v_b = try getBertLayerWeight(weights, name_buf, li, "attention.self.value.biases");
        lw.v_bias = try getBertLayerWeight(weights, name_buf, li, "attention.self.value.bias");
        lw.o_w = try getBertLayerWeight(weights, name_buf, li, "attention.output.dense.weight");
        lw.o_s = try getBertLayerWeight(weights, name_buf, li, "attention.output.dense.scales");
        lw.o_b = try getBertLayerWeight(weights, name_buf, li, "attention.output.dense.biases");
        lw.o_bias = try getBertLayerWeight(weights, name_buf, li, "attention.output.dense.bias");
        lw.attn_norm_w = try getBertLayerWeight(weights, name_buf, li, "attention.output.LayerNorm.weight");
        lw.attn_norm_b = try getBertLayerWeight(weights, name_buf, li, "attention.output.LayerNorm.bias");
        lw.inter_w = try getBertLayerWeight(weights, name_buf, li, "intermediate.dense.weight");
        lw.inter_s = try getBertLayerWeight(weights, name_buf, li, "intermediate.dense.scales");
        lw.inter_b = try getBertLayerWeight(weights, name_buf, li, "intermediate.dense.biases");
        lw.inter_bias = try getBertLayerWeight(weights, name_buf, li, "intermediate.dense.bias");
        lw.out_w = try getBertLayerWeight(weights, name_buf, li, "output.dense.weight");
        lw.out_s = try getBertLayerWeight(weights, name_buf, li, "output.dense.scales");
        lw.out_b = try getBertLayerWeight(weights, name_buf, li, "output.dense.biases");
        lw.out_bias = try getBertLayerWeight(weights, name_buf, li, "output.dense.bias");
        lw.out_norm_w = try getBertLayerWeight(weights, name_buf, li, "output.LayerNorm.weight");
        lw.out_norm_b = try getBertLayerWeight(weights, name_buf, li, "output.LayerNorm.bias");
    }
    return layers;
}

pub fn initBert(io: std.Io, allocator: std.mem.Allocator, config: ModelConfig, weights: *const Weights, name_buf: *[256]u8, s: mlx.mlx_stream) !Transformer {
    // Word embeddings (reuse standard emb_w/s/b fields)
    const emb_w = try getBertWeight(weights, name_buf, "embeddings.word_embeddings.weight");
    const emb_s = try getBertWeight(weights, name_buf, "embeddings.word_embeddings.scales");
    const emb_b = try getBertWeight(weights, name_buf, "embeddings.word_embeddings.biases");

    // Position embeddings
    const pos_w = try getBertWeight(weights, name_buf, "embeddings.position_embeddings.weight");
    const pos_s = try getBertWeight(weights, name_buf, "embeddings.position_embeddings.scales");
    const pos_b = try getBertWeight(weights, name_buf, "embeddings.position_embeddings.biases");

    // Token type embeddings
    const toktype_w = try getBertWeight(weights, name_buf, "embeddings.token_type_embeddings.weight");
    const toktype_s = try getBertWeight(weights, name_buf, "embeddings.token_type_embeddings.scales");
    const toktype_b = try getBertWeight(weights, name_buf, "embeddings.token_type_embeddings.biases");

    // Embedding LayerNorm
    const emb_norm_w = try getBertWeight(weights, name_buf, "embeddings.LayerNorm.weight");
    const emb_norm_b = try getBertWeight(weights, name_buf, "embeddings.LayerNorm.bias");

    const bert_layers = try initBertLayers(allocator, config, weights, name_buf);

    // Batch eval all BERT weights
    {
        const eval_start = std.Io.Timestamp.now(io, .awake);
        const all_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(all_vec);

        _ = mlx.mlx_vector_array_append_value(all_vec, emb_w);
        _ = mlx.mlx_vector_array_append_value(all_vec, emb_s);
        _ = mlx.mlx_vector_array_append_value(all_vec, emb_b);
        _ = mlx.mlx_vector_array_append_value(all_vec, pos_w);
        _ = mlx.mlx_vector_array_append_value(all_vec, pos_s);
        _ = mlx.mlx_vector_array_append_value(all_vec, pos_b);
        _ = mlx.mlx_vector_array_append_value(all_vec, toktype_w);
        _ = mlx.mlx_vector_array_append_value(all_vec, emb_norm_w);
        _ = mlx.mlx_vector_array_append_value(all_vec, emb_norm_b);

        for (bert_layers) |lw| {
            inline for (comptime structFields(BertLayerWeights)) |field| {
                _ = mlx.mlx_vector_array_append_value(all_vec, @field(lw, field.name));
            }
        }

        try mlx.check(mlx.mlx_eval(all_vec));
        const eval_ms: i64 = @intCast(@divTrunc(eval_start.untilNow(io, .awake).nanoseconds, std.time.ns_per_ms));
        log.info("Batch eval all weights: {d}ms\n", .{eval_ms});
    }

    const cache = try KVCache.init(allocator, 0);

    return .{
        .config = config,
        .cache = cache,
        .s = s,
        .allocator = allocator,
        .emb_w = emb_w,
        .emb_s = emb_s,
        .emb_b = emb_b,
        .emb_scale = null,
        .final_norm = mlx.mlx_array_new(),
        .lm_head_w = mlx.mlx_array_new(),
        .lm_head_s = mlx.mlx_array_new(),
        .lm_head_b = mlx.mlx_array_new(),
        .layers = &.{},
        .owns_lm_head = false,
        .owns_norms = false,
        .gelu_coeff = bf16Scalar(0.7978845608028654, s),
        .gelu_inner = bf16Scalar(0.044715, s),
        .half = bf16Scalar(0.5, s),
        .one = bf16Scalar(1.0, s),
        .rms_eps_arr = mlx.mlx_array_new_float(config.rms_norm_eps),
        .post_eps_arr = mlx.mlx_array_new_float(config.postNormEps()),
        .three = bf16Scalar(3.0, s),
        .neg_one = null,
        .ple_emb_w = mlx.mlx_array_new(),
        .ple_emb_s = mlx.mlx_array_new(),
        .ple_emb_b = mlx.mlx_array_new(),
        .ple_proj_w = mlx.mlx_array_new(),
        .ple_proj_s = mlx.mlx_array_new(),
        .ple_proj_b = mlx.mlx_array_new(),
        .ple_proj_norm = mlx.mlx_array_new(),
        .ple_proj_quantized = false,
        .softcap_scalar = null,
        .v_norm_weight = null,
        .v_norm_weight_global = null,
        .rope_freqs_global = null,
        .bert_layers = bert_layers,
        .bert_pos_w = pos_w,
        .bert_pos_s = pos_s,
        .bert_pos_b = pos_b,
        .bert_toktype_w = toktype_w,
        .bert_toktype_s = toktype_s,
        .bert_toktype_b = toktype_b,
        .bert_emb_norm_w = emb_norm_w,
        .bert_emb_norm_b = emb_norm_b,
        .moe_layers = null,
        .ssm_entries = null,
        .moe_seq_offset = 0,
        .hybrid_layers = null,
        .embedding_norm = null,
        .prompt_cache = null,
    };
}
