//! engines/mlx/load/layers_dense, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns initStandardLayers.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../host/log.zig");
const ModelConfig = @import("../model.zig").ModelConfig;
const Weights = @import("../model.zig").Weights;
const fuseRowsInPlace = @import("../../../sdk/blocks/linear/fused_rows.zig").fuseRowsInPlace;
const LayerWeights = @import("../layer_weights.zig").LayerWeights;
const addOne = @import("weights.zig").addOne;
const bf16Scalar = @import("weights.zig").bf16Scalar;
const getLayerBias = @import("weights.zig").getLayerBias;
const getLayerScaleOrEmpty = @import("weights.zig").getLayerScaleOrEmpty;
const getLayerScaleOrEmptyOpt = @import("weights.zig").getLayerScaleOrEmptyOpt;
const getLayerWeight = @import("weights.zig").getLayerWeight;
const getLayerWeightOpt = @import("weights.zig").getLayerWeightOpt;
const maybeTransposeForBf16 = @import("weights.zig").maybeTransposeForBf16;
const moeExpertSuffix = @import("weights.zig").moeExpertSuffix;

// ── Init helpers ──

pub fn initStandardLayers(allocator: std.mem.Allocator, config: ModelConfig, weights: *Weights, name_buf: *[256]u8, s: mlx.mlx_stream) !struct { layers: []LayerWeights, owned_bf16: []mlx.mlx_array } {
    log.info("Precomputing layer weights...\n", .{});
    const prefix = config.weight_prefix;
    const layers = try allocator.alloc(LayerWeights, config.num_hidden_layers);
    // Dense bf16 weights are pre-transposed at load; track the new arrays so
    // Transformer.deinit frees them. Empty (no allocations) for quantized models.
    var owned_bf16: std.ArrayList(mlx.mlx_array) = .empty;
    errdefer {
        for (owned_bf16.items) |a| _ = mlx.mlx_array_free(a);
        owned_bf16.deinit(allocator);
    }

    for (0..config.num_hidden_layers) |i| {
        const li: u32 = @intCast(i);
        const lw = &layers[i];

        // Gemma 4 KV-layer sharing: layers in the shared tail reuse an earlier
        // layer's K/V (the forward reads `kv_source`'s cache), so they carry no
        // k_proj/k_norm/v_proj of their own. Some exports physically drop those
        // tensors — load them only for non-shared layers.
        lw.kv_source = config.getKVSourceLayer(li);
        const kv_shared = lw.kv_source != null;

        const input_norm_raw = try getLayerWeight(weights, name_buf, prefix, li, "input_layernorm.weight");
        lw.input_norm = if (config.norm_has_offset) try addOne(input_norm_raw, s) else input_norm_raw;
        const post_attn_raw = try getLayerWeight(weights, name_buf, prefix, li, "post_attention_layernorm.weight");
        lw.post_attn_norm = if (config.norm_has_offset) try addOne(post_attn_raw, s) else post_attn_raw;

        if (config.has_pre_ff_norm) {
            const pre_ff_raw = try getLayerWeight(weights, name_buf, prefix, li, "pre_feedforward_layernorm.weight");
            lw.pre_ff_norm = if (config.norm_has_offset) try addOne(pre_ff_raw, s) else pre_ff_raw;
            const post_ff_raw = try getLayerWeight(weights, name_buf, prefix, li, "post_feedforward_layernorm.weight");
            lw.post_ff_norm = if (config.norm_has_offset) try addOne(post_ff_raw, s) else post_ff_raw;
        } else {
            lw.pre_ff_norm = null;
            lw.post_ff_norm = null;
        }

        if (config.has_qk_norm) {
            const q_norm_raw = try getLayerWeight(weights, name_buf, prefix, li, "self_attn.q_norm.weight");
            lw.q_norm = if (config.norm_has_offset) try addOne(q_norm_raw, s) else q_norm_raw;
            // KV-shared layers compute no K, so they carry no k_norm.
            if (kv_shared) {
                lw.k_norm = null;
            } else {
                const k_norm_raw = try getLayerWeight(weights, name_buf, prefix, li, "self_attn.k_norm.weight");
                lw.k_norm = if (config.norm_has_offset) try addOne(k_norm_raw, s) else k_norm_raw;
            }
        } else {
            lw.q_norm = null;
            lw.k_norm = null;
        }

        // Additive q-proj bias (Qwen2). Optional — empty for archs without it.
        lw.q_bias = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.q_proj.bias") orelse mlx.mlx_array_new();
        lw.q_w = try getLayerWeight(weights, name_buf, prefix, li, "self_attn.q_proj.weight");
        lw.q_s = try getLayerScaleOrEmpty(weights, name_buf, prefix, li, "self_attn.q_proj.scales", config.quant_bits);
        lw.q_b = try getLayerBias(weights, name_buf, prefix, li, "self_attn.q_proj.biases", &config);
        if (kv_shared) {
            // No own K/V — the forward reads kv_source's cache. Leave empty.
            lw.k_eq_v = false;
            lw.k_w = mlx.mlx_array_new();
            lw.k_s = mlx.mlx_array_new();
            lw.k_b = mlx.mlx_array_new();
            lw.k_bias = mlx.mlx_array_new();
            lw.v_w = mlx.mlx_array_new();
            lw.v_s = mlx.mlx_array_new();
            lw.v_b = mlx.mlx_array_new();
            lw.v_bias = mlx.mlx_array_new();
        } else {
            lw.k_w = try getLayerWeight(weights, name_buf, prefix, li, "self_attn.k_proj.weight");
            lw.k_s = try getLayerScaleOrEmpty(weights, name_buf, prefix, li, "self_attn.k_proj.scales", config.quant_bits);
            lw.k_b = try getLayerBias(weights, name_buf, prefix, li, "self_attn.k_proj.biases", &config);
            lw.k_bias = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.k_proj.bias") orelse mlx.mlx_array_new();
            // Gemma 4 (31B): full_attention layers share V with K (no v_proj weight stored).
            // Sliding_attention layers still have separate V.
            lw.k_eq_v = config.attention_k_eq_v and config.isGlobalLayer(li);
            if (lw.k_eq_v) {
                lw.v_w = lw.k_w;
                lw.v_s = lw.k_s;
                lw.v_b = lw.k_b;
                lw.v_bias = lw.k_bias;
            } else {
                lw.v_w = try getLayerWeight(weights, name_buf, prefix, li, "self_attn.v_proj.weight");
                lw.v_s = try getLayerScaleOrEmpty(weights, name_buf, prefix, li, "self_attn.v_proj.scales", config.quant_bits);
                lw.v_b = try getLayerBias(weights, name_buf, prefix, li, "self_attn.v_proj.biases", &config);
                lw.v_bias = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.v_proj.bias") orelse mlx.mlx_array_new();
            }
        }
        var sfx: [64]u8 = undefined;
        const o_name = "self_attn.o_proj";
        lw.o_w = try getLayerWeight(weights, name_buf, prefix, li, moeExpertSuffix(&sfx, o_name, "weight"));
        lw.o_s = try getLayerScaleOrEmpty(weights, name_buf, prefix, li, moeExpertSuffix(&sfx, o_name, "scales"), config.quant_bits);
        lw.o_b = try getLayerBias(weights, name_buf, prefix, li, moeExpertSuffix(&sfx, o_name, "biases"), &config);

        lw.gate_w = try getLayerWeight(weights, name_buf, prefix, li, "mlp.gate_proj.weight");
        lw.gate_s = try getLayerScaleOrEmpty(weights, name_buf, prefix, li, "mlp.gate_proj.scales", config.quant_bits);
        lw.gate_b = try getLayerBias(weights, name_buf, prefix, li, "mlp.gate_proj.biases", &config);
        lw.up_w = try getLayerWeight(weights, name_buf, prefix, li, "mlp.up_proj.weight");
        lw.up_s = try getLayerScaleOrEmpty(weights, name_buf, prefix, li, "mlp.up_proj.scales", config.quant_bits);
        lw.up_b = try getLayerBias(weights, name_buf, prefix, li, "mlp.up_proj.biases", &config);
        lw.down_w = try getLayerWeight(weights, name_buf, prefix, li, "mlp.down_proj.weight");
        lw.down_s = try getLayerScaleOrEmpty(weights, name_buf, prefix, li, "mlp.down_proj.scales", config.quant_bits);
        lw.down_b = try getLayerBias(weights, name_buf, prefix, li, "mlp.down_proj.biases", &config);

        // Dense bf16: pre-transpose [out,in]→[in,out] so qmatmulBits dispatches to
        // a plain matmul. No-ops on quantized weights (scales non-null).
        if (!config.attn_fused_qkv) {
            try maybeTransposeForBf16(&lw.q_w, lw.q_s, &owned_bf16, allocator, s);
            try maybeTransposeForBf16(&lw.k_w, lw.k_s, &owned_bf16, allocator, s);
            if (lw.k_eq_v) {
                lw.v_w = lw.k_w; // re-alias V to the transposed K (no second copy)
            } else {
                try maybeTransposeForBf16(&lw.v_w, lw.v_s, &owned_bf16, allocator, s);
            }
        }
        try maybeTransposeForBf16(&lw.o_w, lw.o_s, &owned_bf16, allocator, s);
        try maybeTransposeForBf16(&lw.gate_w, lw.gate_s, &owned_bf16, allocator, s);
        try maybeTransposeForBf16(&lw.up_w, lw.up_s, &owned_bf16, allocator, s);
        try maybeTransposeForBf16(&lw.down_w, lw.down_s, &owned_bf16, allocator, s);

        // Decode-width row fusion: one matmul for q|k|v and one for
        // gate|up. The originals are released (map entries) and the fields
        // become views of the joined buffers.
        lw.qkv = .{};
        lw.gu = .{};
        if (lw.q_s.ctx != null and !kv_shared and !lw.k_eq_v and lw.q_bias.ctx == null and lw.k_bias.ctx == null and lw.v_bias.ctx == null) {
            var parts = [_][3]*mlx.mlx_array{ .{ &lw.q_w, &lw.q_s, &lw.q_b }, .{ &lw.k_w, &lw.k_s, &lw.k_b }, .{ &lw.v_w, &lw.v_s, &lw.v_b } };
            const names = [_][]const u8{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj" };
            lw.qkv = try fuseRowsInPlace(&parts, 0, &names, weights, name_buf, prefix, li, &owned_bf16, allocator, s);
        }
        if (lw.gate_s.ctx != null) {
            var parts = [_][3]*mlx.mlx_array{ .{ &lw.gate_w, &lw.gate_s, &lw.gate_b }, .{ &lw.up_w, &lw.up_s, &lw.up_b } };
            const names = [_][]const u8{ "mlp.gate_proj", "mlp.up_proj" };
            lw.gu = try fuseRowsInPlace(&parts, 0, &names, weights, name_buf, prefix, li, &owned_bf16, allocator, s);
        }

        // Gemma 4: per-layer scalar
        lw.layer_scalar = getLayerWeightOpt(weights, name_buf, prefix, li, "layer_scalar");

        // Gemma 4: PLE per-layer weights. Must initialize even in the no-PLE case so the
        // optional tags are not read as uninitialized memory later in the eval loop
        // (the layers slice comes from `allocator.alloc` which skips struct defaults).
        lw.ple_gate_w = null;
        lw.ple_gate_s = null;
        lw.ple_gate_b = null;
        lw.ple_proj_w = null;
        lw.ple_proj_s = null;
        lw.ple_proj_b = null;
        lw.ple_norm = null;
        if (config.hidden_size_per_layer_input > 0) {
            lw.ple_gate_w = getLayerWeightOpt(weights, name_buf, prefix, li, "per_layer_input_gate.weight");
            // Dense bf16: scales/biases don't exist. The forward unwraps these with
            // `.?` then feeds qmatmul, so supply a null-ctx array (not Zig-null) →
            // qmatmulBits sees the bf16 path.
            lw.ple_gate_s = getLayerScaleOrEmptyOpt(weights, name_buf, prefix, li, "per_layer_input_gate.scales", config.quant_bits);
            lw.ple_gate_b = try getLayerBias(weights, name_buf, prefix, li, "per_layer_input_gate.biases", &config);
            lw.ple_proj_w = getLayerWeightOpt(weights, name_buf, prefix, li, "per_layer_projection.weight");
            lw.ple_proj_s = getLayerScaleOrEmptyOpt(weights, name_buf, prefix, li, "per_layer_projection.scales", config.quant_bits);
            lw.ple_proj_b = try getLayerBias(weights, name_buf, prefix, li, "per_layer_projection.biases", &config);
            lw.ple_norm = getLayerWeightOpt(weights, name_buf, prefix, li, "post_per_layer_input_norm.weight");
            // bf16: pre-transpose the two PLE projections (used via qmatmul).
            if (lw.ple_gate_w) |*w| try maybeTransposeForBf16(w, lw.ple_gate_s.?, &owned_bf16, allocator, s);
            if (lw.ple_proj_w) |*w| try maybeTransposeForBf16(w, lw.ple_proj_s.?, &owned_bf16, allocator, s);
        }
    }
    return .{ .layers = layers, .owned_bf16 = try owned_bf16.toOwnedSlice(allocator) };
}
