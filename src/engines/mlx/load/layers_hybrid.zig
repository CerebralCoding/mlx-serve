//! engines/mlx/load/layers_hybrid, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns initHybridLayers.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../host/log.zig");
const ModelConfig = @import("../model.zig").ModelConfig;
const Weights = @import("../model.zig").Weights;
const fuseRowsInPlace = @import("../../../sdk/blocks/linear/fused_rows.zig").fuseRowsInPlace;
const HybridLayerWeights = @import("../forward/hybrid.zig").HybridLayerWeights;
const SSMCacheEntry = @import("../ssm/entry.zig").SSMCacheEntry;
const depthwiseConvNeedsTranspose = @import("weights.zig").depthwiseConvNeedsTranspose;
const ffKey = @import("weights.zig").ffKey;
const getLayerBias = @import("weights.zig").getLayerBias;
const getLayerScaleOpt = @import("weights.zig").getLayerScaleOpt;
const getLayerWeight = @import("weights.zig").getLayerWeight;
const getLayerWeightOpt = @import("weights.zig").getLayerWeightOpt;
const maybeTransposeDepthwiseConv = @import("weights.zig").maybeTransposeDepthwiseConv;
const maybeTransposeForBf16 = @import("weights.zig").maybeTransposeForBf16;

pub fn initHybridLayers(allocator: std.mem.Allocator, config: ModelConfig, weights: *Weights, name_buf: *[256]u8, s: mlx.mlx_stream) !struct { hybrid_layers: []HybridLayerWeights, ssm_entries: []SSMCacheEntry, owned_bf16: []mlx.mlx_array } {
    log.info("Precomputing hybrid layer weights...\n", .{});
    const prefix = config.weight_prefix;
    const hybrid_layers = try allocator.alloc(HybridLayerWeights, config.num_hidden_layers);
    const ssm_entries = try allocator.alloc(SSMCacheEntry, config.num_hidden_layers);
    // Dense bf16 weights are pre-transposed at load; track the new arrays so
    // Transformer.deinit can free them (same list the standard/MoE paths use).
    var owned_bf16: std.ArrayList(mlx.mlx_array) = .empty;
    errdefer {
        for (owned_bf16.items) |a| _ = mlx.mlx_array_free(a);
        owned_bf16.deinit(allocator);
    }
    const is_lfm2 = std.mem.eql(u8, config.model_type, "lfm2");
    const is_nemotron = std.mem.eql(u8, config.model_type, "nemotron_h");

    for (0..config.num_hidden_layers) |i| {
        const li: u32 = @intCast(i);
        const lw = &hybrid_layers[i];
        const block_type = config.layer_block_types[i];

        // Input norm: LFM2 uses "operator_norm", Nemotron-H uses "norm"
        if (is_lfm2) {
            lw.input_norm = try getLayerWeight(weights, name_buf, prefix, li, "operator_norm.weight");
        } else {
            lw.input_norm = try getLayerWeight(weights, name_buf, prefix, li, "norm.weight");
        }

        // Post norm (before MLP): LFM2 uses "ffn_norm", Nemotron-H single-op blocks have none
        if (is_lfm2) {
            lw.post_norm = getLayerWeightOpt(weights, name_buf, prefix, li, "ffn_norm.weight");
        } else {
            lw.post_norm = null;
        }

        // Initialize SSM/conv cache entry
        ssm_entries[i] = .{
            .conv_state = mlx.mlx_array_new(),
            .ssm_state = mlx.mlx_array_new(),
            .initialized = false,
        };

        switch (block_type) {
            .gated_conv => {
                lw.op = .{ .gated_conv = .{
                    .in_proj_w = try getLayerWeight(weights, name_buf, prefix, li, "conv.in_proj.weight"),
                    .in_proj_s = getLayerScaleOpt(weights, name_buf, prefix, li, "conv.in_proj.scales"),
                    .in_proj_b = try getLayerBias(weights, name_buf, prefix, li, "conv.in_proj.biases", &config),
                    .conv_w = try getLayerWeight(weights, name_buf, prefix, li, "conv.conv.weight"),
                    .out_proj_w = try getLayerWeight(weights, name_buf, prefix, li, "conv.out_proj.weight"),
                    .out_proj_s = getLayerScaleOpt(weights, name_buf, prefix, li, "conv.out_proj.scales"),
                    .out_proj_b = try getLayerBias(weights, name_buf, prefix, li, "conv.out_proj.biases", &config),
                } };
                try maybeTransposeDepthwiseConv(&lw.op.gated_conv.conv_w, &owned_bf16, allocator, s);
            },
            .attention => {
                if (is_nemotron) {
                    // Nemotron-H: mixer.{q,k,v,o}_proj, no QK norms
                    // Use Opt for biases — mxfp8 quantized layers may lack them
                    lw.op = .{ .full_attn = .{
                        .q_w = try getLayerWeight(weights, name_buf, prefix, li, "mixer.q_proj.weight"),
                        .q_s = getLayerScaleOpt(weights, name_buf, prefix, li, "mixer.q_proj.scales"),
                        .q_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mixer.q_proj.biases") orelse mlx.mlx_array_new(),
                        .k_w = try getLayerWeight(weights, name_buf, prefix, li, "mixer.k_proj.weight"),
                        .k_s = getLayerScaleOpt(weights, name_buf, prefix, li, "mixer.k_proj.scales"),
                        .k_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mixer.k_proj.biases") orelse mlx.mlx_array_new(),
                        .v_w = try getLayerWeight(weights, name_buf, prefix, li, "mixer.v_proj.weight"),
                        .v_s = getLayerScaleOpt(weights, name_buf, prefix, li, "mixer.v_proj.scales"),
                        .v_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mixer.v_proj.biases") orelse mlx.mlx_array_new(),
                        .o_w = try getLayerWeight(weights, name_buf, prefix, li, "mixer.o_proj.weight"),
                        .o_s = getLayerScaleOpt(weights, name_buf, prefix, li, "mixer.o_proj.scales"),
                        .o_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mixer.o_proj.biases") orelse mlx.mlx_array_new(),
                        .q_norm = mlx.mlx_array_new(),
                        .k_norm = mlx.mlx_array_new(),
                    } };
                } else {
                    // LFM2: self_attn.{q,k,v}_proj + out_proj, QK layernorms
                    lw.op = .{ .full_attn = .{
                        .q_w = try getLayerWeight(weights, name_buf, prefix, li, "self_attn.q_proj.weight"),
                        .q_s = getLayerScaleOpt(weights, name_buf, prefix, li, "self_attn.q_proj.scales"),
                        .q_b = try getLayerBias(weights, name_buf, prefix, li, "self_attn.q_proj.biases", &config),
                        .k_w = try getLayerWeight(weights, name_buf, prefix, li, "self_attn.k_proj.weight"),
                        .k_s = getLayerScaleOpt(weights, name_buf, prefix, li, "self_attn.k_proj.scales"),
                        .k_b = try getLayerBias(weights, name_buf, prefix, li, "self_attn.k_proj.biases", &config),
                        .v_w = try getLayerWeight(weights, name_buf, prefix, li, "self_attn.v_proj.weight"),
                        .v_s = getLayerScaleOpt(weights, name_buf, prefix, li, "self_attn.v_proj.scales"),
                        .v_b = try getLayerBias(weights, name_buf, prefix, li, "self_attn.v_proj.biases", &config),
                        .o_w = try getLayerWeight(weights, name_buf, prefix, li, "self_attn.out_proj.weight"),
                        .o_s = getLayerScaleOpt(weights, name_buf, prefix, li, "self_attn.out_proj.scales"),
                        .o_b = try getLayerBias(weights, name_buf, prefix, li, "self_attn.out_proj.biases", &config),
                        .q_norm = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.q_layernorm.weight") orelse mlx.mlx_array_new(),
                        .k_norm = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.k_layernorm.weight") orelse mlx.mlx_array_new(),
                    } };
                }
            },
            .mamba2 => {
                lw.op = .{ .mamba2 = .{
                    .in_proj_w = try getLayerWeight(weights, name_buf, prefix, li, "mixer.in_proj.weight"),
                    .in_proj_s = getLayerScaleOpt(weights, name_buf, prefix, li, "mixer.in_proj.scales"),
                    .in_proj_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mixer.in_proj.biases") orelse mlx.mlx_array_new(),
                    .conv1d_w = try getLayerWeight(weights, name_buf, prefix, li, "mixer.conv1d.weight"),
                    .conv1d_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mixer.conv1d.bias"),
                    .A_log = try getLayerWeight(weights, name_buf, prefix, li, "mixer.A_log"),
                    .D = try getLayerWeight(weights, name_buf, prefix, li, "mixer.D"),
                    .dt_bias = try getLayerWeight(weights, name_buf, prefix, li, "mixer.dt_bias"),
                    .norm_w = try getLayerWeight(weights, name_buf, prefix, li, "mixer.norm.weight"),
                    .out_proj_w = try getLayerWeight(weights, name_buf, prefix, li, "mixer.out_proj.weight"),
                    .out_proj_s = getLayerScaleOpt(weights, name_buf, prefix, li, "mixer.out_proj.scales"),
                    .out_proj_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mixer.out_proj.biases") orelse mlx.mlx_array_new(),
                } };
            },
            .mlp => {
                // Nemotron-H standalone MLP (ReLU^2, ungated: up + down only)
                lw.op = .{ .simple_mlp = .{
                    .up_w = try getLayerWeight(weights, name_buf, prefix, li, "mixer.up_proj.weight"),
                    .up_s = getLayerScaleOpt(weights, name_buf, prefix, li, "mixer.up_proj.scales"),
                    .up_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mixer.up_proj.biases") orelse mlx.mlx_array_new(),
                    .down_w = try getLayerWeight(weights, name_buf, prefix, li, "mixer.down_proj.weight"),
                    .down_s = getLayerScaleOpt(weights, name_buf, prefix, li, "mixer.down_proj.scales"),
                    .down_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mixer.down_proj.biases") orelse mlx.mlx_array_new(),
                } };
            },
            .moe => {
                if (!is_nemotron) return error.UnsupportedHybridBlock;
                const has_shared = getLayerWeightOpt(weights, name_buf, prefix, li, "mixer.shared_experts.up_proj.weight") != null;
                const lat1 = getLayerWeightOpt(weights, name_buf, prefix, li, "mixer.fc1_latent_proj.weight");
                const lat2 = getLayerWeightOpt(weights, name_buf, prefix, li, "mixer.fc2_latent_proj.weight");
                if (config.moe_latent_size > 0 and (lat1 == null or lat2 == null)) {
                    log.err("MISSING WEIGHT: layer {d} latent proj (moe_latent_size={d})\n", .{ li, config.moe_latent_size });
                    return error.MissingWeight;
                }
                if ((lat1 == null) != (lat2 == null)) return error.MissingWeight;
                lw.op = .{ .nemotron_moe = .{
                    .router_w = try getLayerWeight(weights, name_buf, prefix, li, "mixer.gate.weight"),
                    .router_s = getLayerScaleOpt(weights, name_buf, prefix, li, "mixer.gate.scales"),
                    .router_b = getLayerScaleOpt(weights, name_buf, prefix, li, "mixer.gate.biases"),
                    .expert_bias = try getLayerWeight(weights, name_buf, prefix, li, "mixer.gate.e_score_correction_bias"),
                    .fc1_w = try getLayerWeight(weights, name_buf, prefix, li, "mixer.switch_mlp.fc1.weight"),
                    .fc1_s = getLayerScaleOpt(weights, name_buf, prefix, li, "mixer.switch_mlp.fc1.scales"),
                    .fc1_b = getLayerScaleOpt(weights, name_buf, prefix, li, "mixer.switch_mlp.fc1.biases"),
                    .fc2_w = try getLayerWeight(weights, name_buf, prefix, li, "mixer.switch_mlp.fc2.weight"),
                    .fc2_s = getLayerScaleOpt(weights, name_buf, prefix, li, "mixer.switch_mlp.fc2.scales"),
                    .fc2_b = getLayerScaleOpt(weights, name_buf, prefix, li, "mixer.switch_mlp.fc2.biases"),
                    .fc1_latent_w = lat1 orelse mlx.mlx_array_new(),
                    .fc1_latent_s = getLayerScaleOpt(weights, name_buf, prefix, li, "mixer.fc1_latent_proj.scales"),
                    .fc1_latent_b = getLayerScaleOpt(weights, name_buf, prefix, li, "mixer.fc1_latent_proj.biases"),
                    .fc2_latent_w = lat2 orelse mlx.mlx_array_new(),
                    .fc2_latent_s = getLayerScaleOpt(weights, name_buf, prefix, li, "mixer.fc2_latent_proj.scales"),
                    .fc2_latent_b = getLayerScaleOpt(weights, name_buf, prefix, li, "mixer.fc2_latent_proj.biases"),
                    .shared = if (has_shared) .{
                        .up_w = try getLayerWeight(weights, name_buf, prefix, li, "mixer.shared_experts.up_proj.weight"),
                        .up_s = getLayerScaleOpt(weights, name_buf, prefix, li, "mixer.shared_experts.up_proj.scales"),
                        .up_b = getLayerScaleOpt(weights, name_buf, prefix, li, "mixer.shared_experts.up_proj.biases"),
                        .down_w = try getLayerWeight(weights, name_buf, prefix, li, "mixer.shared_experts.down_proj.weight"),
                        .down_s = getLayerScaleOpt(weights, name_buf, prefix, li, "mixer.shared_experts.down_proj.scales"),
                        .down_b = getLayerScaleOpt(weights, name_buf, prefix, li, "mixer.shared_experts.down_proj.biases"),
                    } else null,
                } };
            },
        }

        // MLP: present for all LFM2 layers, absent for Nemotron-H single-op blocks
        lw.moe_mlp = null;
        if (is_lfm2 and config.lfm2_moe and li >= config.num_dense_layers) {
            // lfm2_moe sparse block: sigmoid routing with a selection-only
            // `expert_bias` (the hy3 chain), no shared expert. The switch
            // weights are mlx-lm's SwitchGLU stack: [experts, out, in].
            lw.mlp = null;
            const none = mlx.mlx_array_new();
            lw.moe_mlp = .{
                .router_w = try getLayerWeight(weights, name_buf, prefix, li, "feed_forward.gate.weight"),
                .router_s = getLayerScaleOpt(weights, name_buf, prefix, li, "feed_forward.gate.scales"),
                .router_b = getLayerScaleOpt(weights, name_buf, prefix, li, "feed_forward.gate.biases"),
                .switch_gate_w = try getLayerWeight(weights, name_buf, prefix, li, "feed_forward.switch_mlp.gate_proj.weight"),
                .switch_gate_s = getLayerScaleOpt(weights, name_buf, prefix, li, "feed_forward.switch_mlp.gate_proj.scales"),
                .switch_gate_b = getLayerScaleOpt(weights, name_buf, prefix, li, "feed_forward.switch_mlp.gate_proj.biases"),
                .switch_up_w = try getLayerWeight(weights, name_buf, prefix, li, "feed_forward.switch_mlp.up_proj.weight"),
                .switch_up_s = getLayerScaleOpt(weights, name_buf, prefix, li, "feed_forward.switch_mlp.up_proj.scales"),
                .switch_up_b = getLayerScaleOpt(weights, name_buf, prefix, li, "feed_forward.switch_mlp.up_proj.biases"),
                .switch_down_w = try getLayerWeight(weights, name_buf, prefix, li, "feed_forward.switch_mlp.down_proj.weight"),
                .switch_down_s = getLayerScaleOpt(weights, name_buf, prefix, li, "feed_forward.switch_mlp.down_proj.scales"),
                .switch_down_b = getLayerScaleOpt(weights, name_buf, prefix, li, "feed_forward.switch_mlp.down_proj.biases"),
                // No shared expert on this family — null handles keep
                // `moeMLP`'s shared arms unreached.
                .shared_gate_w = none,
                .shared_gate_s = none,
                .shared_gate_b = none,
                .shared_up_w = none,
                .shared_up_s = none,
                .shared_up_b = none,
                .shared_down_w = none,
                .shared_down_s = none,
                .shared_down_b = none,
                .expert_bias = getLayerWeightOpt(weights, name_buf, prefix, li, "feed_forward.expert_bias") orelse blk: {
                    // use_expert_bias=false ships none; the reference
                    // zero-inits it (selection = plain sigmoid top-k).
                    var zeros = mlx.mlx_array_new();
                    const zshape = [_]c_int{@intCast(config.num_experts)};
                    try mlx.check(mlx.mlx_zeros(&zeros, &zshape, 1, .float32, s));
                    try owned_bf16.append(allocator, zeros);
                    break :blk zeros;
                },
                .route_norm = config.moe_route_norm,
                .route_scale = config.router_scaling_factor,
            };
            try maybeTransposeForBf16(&lw.moe_mlp.?.router_w, lw.moe_mlp.?.router_s, &owned_bf16, allocator, s);
        } else if (is_lfm2) {
            // Dense SwiGLU. transformers' Lfm2 spells it w1/w3/w2; the
            // mlx-lm lfm2_moe converter spells the same three gate/up/down —
            // probe, because a hardcoded spelling loses one of the two packs
            // at its first missing weight.
            const gate_key = if (getLayerWeightOpt(weights, name_buf, prefix, li, "feed_forward.w1.weight") != null)
                [3][]const u8{ "w1", "w3", "w2" }
            else
                [3][]const u8{ "gate_proj", "up_proj", "down_proj" };
            var kb: [64]u8 = undefined;
            lw.mlp = .{
                .gate_w = try getLayerWeight(weights, name_buf, prefix, li, ffKey(&kb, gate_key[0], "weight")),
                .gate_s = getLayerScaleOpt(weights, name_buf, prefix, li, ffKey(&kb, gate_key[0], "scales")),
                .gate_b = try getLayerBias(weights, name_buf, prefix, li, ffKey(&kb, gate_key[0], "biases"), &config),
                .up_w = try getLayerWeight(weights, name_buf, prefix, li, ffKey(&kb, gate_key[1], "weight")),
                .up_s = getLayerScaleOpt(weights, name_buf, prefix, li, ffKey(&kb, gate_key[1], "scales")),
                .up_b = try getLayerBias(weights, name_buf, prefix, li, ffKey(&kb, gate_key[1], "biases"), &config),
                .down_w = try getLayerWeight(weights, name_buf, prefix, li, ffKey(&kb, gate_key[2], "weight")),
                .down_s = getLayerScaleOpt(weights, name_buf, prefix, li, ffKey(&kb, gate_key[2], "scales")),
                .down_b = try getLayerBias(weights, name_buf, prefix, li, ffKey(&kb, gate_key[2], "biases"), &config),
            };
        } else {
            lw.mlp = null;
        }

        // Dense bf16 (null-ctx scales): pre-transpose [out,in]→[in,out] so
        // qmatmulBits' plain-matmul arm contracts correctly. Matmul operands
        // only — depthwise conv kernels and SSM state tensors are read as-is.
        switch (lw.op) {
            .gated_conv => |*cw| {
                try maybeTransposeForBf16(&cw.in_proj_w, cw.in_proj_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&cw.out_proj_w, cw.out_proj_s, &owned_bf16, allocator, s);
            },
            .full_attn => |*fa| {
                try maybeTransposeForBf16(&fa.q_w, fa.q_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&fa.k_w, fa.k_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&fa.v_w, fa.v_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&fa.o_w, fa.o_s, &owned_bf16, allocator, s);
                if (is_lfm2 and fa.q_s.ctx != null) {
                    var parts = [_][3]*mlx.mlx_array{ .{ &fa.q_w, &fa.q_s, &fa.q_b }, .{ &fa.k_w, &fa.k_s, &fa.k_b }, .{ &fa.v_w, &fa.v_s, &fa.v_b } };
                    const names = [_][]const u8{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj" };
                    fa.qkv = try fuseRowsInPlace(&parts, 0, &names, weights, name_buf, prefix, li, &owned_bf16, allocator, s);
                }
            },
            .mamba2 => |*mw| {
                try maybeTransposeForBf16(&mw.in_proj_w, mw.in_proj_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.out_proj_w, mw.out_proj_s, &owned_bf16, allocator, s);
            },
            .simple_mlp => |*sm| {
                try maybeTransposeForBf16(&sm.up_w, sm.up_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&sm.down_w, sm.down_s, &owned_bf16, allocator, s);
            },
            .nemotron_moe => |*nm| {
                if (nm.router_s.ctx == null) nm.router_rows = nm.router_w;
                try maybeTransposeForBf16(&nm.router_w, nm.router_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&nm.fc1_w, nm.fc1_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&nm.fc2_w, nm.fc2_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&nm.fc1_latent_w, nm.fc1_latent_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&nm.fc2_latent_w, nm.fc2_latent_s, &owned_bf16, allocator, s);
                if (nm.shared) |*sm| {
                    try maybeTransposeForBf16(&sm.up_w, sm.up_s, &owned_bf16, allocator, s);
                    try maybeTransposeForBf16(&sm.down_w, sm.down_s, &owned_bf16, allocator, s);
                }
            },
            .dense_mlp => |*dm| {
                try maybeTransposeForBf16(&dm.gate_w, dm.gate_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&dm.up_w, dm.up_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&dm.down_w, dm.down_s, &owned_bf16, allocator, s);
            },
        }
        if (lw.mlp) |*m| {
            try maybeTransposeForBf16(&m.gate_w, m.gate_s, &owned_bf16, allocator, s);
            try maybeTransposeForBf16(&m.up_w, m.up_s, &owned_bf16, allocator, s);
            try maybeTransposeForBf16(&m.down_w, m.down_s, &owned_bf16, allocator, s);
            if (is_lfm2 and m.gate_s.ctx != null) {
                var parts = [_][3]*mlx.mlx_array{ .{ &m.gate_w, &m.gate_s, &m.gate_b }, .{ &m.up_w, &m.up_s, &m.up_b } };
                const names = if (getLayerWeightOpt(weights, name_buf, prefix, li, "feed_forward.w1.weight") != null)
                    [_][]const u8{ "feed_forward.w1", "feed_forward.w3" }
                else
                    [_][]const u8{ "feed_forward.gate_proj", "feed_forward.up_proj" };
                m.gu = try fuseRowsInPlace(&parts, 0, &names, weights, name_buf, prefix, li, &owned_bf16, allocator, s);
            }
        }
    }

    return .{
        .hybrid_layers = hybrid_layers,
        .ssm_entries = ssm_entries,
        .owned_bf16 = try owned_bf16.toOwnedSlice(allocator),
    };
}

// ── tests ──

test "hybrid init loads a DENSE bf16 checkpoint (LFM2.5-2.6B-bf16)" {
    // mlx-community/LFM2.5-2.6B-bf16 carries no `.scales` for any linear, and
    // `initHybridLayers` fetched them with the MANDATORY getter — the load died
    // on "MISSING WEIGHT: language_model.model.layers.0.conv.in_proj.scales".
    // Both other layer-init paths (standard, MoE) already read absent scales as
    // "plain bf16" and pre-transpose the weight so qmatmulBits contracts with a
    // single mlx_matmul; the hybrid path was the one that never learned it.
    const allocator = std.testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);

    var w = Weights.init(allocator);
    defer w.deinit();
    const put = struct {
        fn add(weights: *Weights, alloc: std.mem.Allocator, name: []const u8, shape: []const c_int, st: mlx.mlx_stream) !void {
            var n: usize = 1;
            for (shape) |d| n *= @intCast(d);
            const vals = try alloc.alloc(f32, n);
            defer alloc.free(vals);
            @memset(vals, 0.5);
            const f32_arr = mlx.mlx_array_new_data(vals.ptr, shape.ptr, @intCast(shape.len), .float32);
            defer _ = mlx.mlx_array_free(f32_arr);
            var bf = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&bf, f32_arr, .bfloat16, st));
            try weights.map.put(try alloc.dupe(u8, name), bf);
        }
    }.add;

    const H: c_int = 8; // hidden
    const FF: c_int = 16; // feed-forward inner (asymmetric on purpose)
    const P = "language_model.model";
    // Layer 0: gated conv. Layer 1: full attention. Both carry an LFM2 SwiGLU MLP.
    inline for (.{ 0, 1 }) |li| {
        const pre = std.fmt.comptimePrint("{s}.layers.{d}.", .{ P, li });
        try put(&w, allocator, pre ++ "operator_norm.weight", &.{H}, s);
        try put(&w, allocator, pre ++ "ffn_norm.weight", &.{H}, s);
        try put(&w, allocator, pre ++ "feed_forward.w1.weight", &.{ FF, H }, s);
        try put(&w, allocator, pre ++ "feed_forward.w3.weight", &.{ FF, H }, s);
        try put(&w, allocator, pre ++ "feed_forward.w2.weight", &.{ H, FF }, s);
        if (li == 0) {
            try put(&w, allocator, pre ++ "conv.in_proj.weight", &.{ 3 * H, H }, s);
            try put(&w, allocator, pre ++ "conv.conv.weight", &.{ H, 3, 1 }, s);
            try put(&w, allocator, pre ++ "conv.out_proj.weight", &.{ H, H }, s);
        } else {
            try put(&w, allocator, pre ++ "self_attn.q_proj.weight", &.{ FF, H }, s);
            try put(&w, allocator, pre ++ "self_attn.k_proj.weight", &.{ H, H }, s);
            try put(&w, allocator, pre ++ "self_attn.v_proj.weight", &.{ H, H }, s);
            try put(&w, allocator, pre ++ "self_attn.out_proj.weight", &.{ H, FF }, s);
            try put(&w, allocator, pre ++ "self_attn.q_layernorm.weight", &.{H}, s);
            try put(&w, allocator, pre ++ "self_attn.k_layernorm.weight", &.{H}, s);
        }
    }

    var config = ModelConfig{ .model_type = "lfm2", .weight_prefix = P };
    config.has_hybrid_layers = true;
    config.num_hidden_layers = 2;
    config.hidden_size = @intCast(H);
    config.quant_bits = 0;
    config.layer_block_types[0] = .gated_conv;
    config.layer_block_types[1] = .attention;

    var name_buf: [256]u8 = undefined;
    const hl = try initHybridLayers(allocator, config, &w, &name_buf, s);
    defer {
        allocator.free(hl.hybrid_layers);
        for (hl.ssm_entries) |*e| {
            _ = mlx.mlx_array_free(e.conv_state);
            _ = mlx.mlx_array_free(e.ssm_state);
        }
        allocator.free(hl.ssm_entries);
        for (hl.owned_bf16) |a| _ = mlx.mlx_array_free(a);
        allocator.free(hl.owned_bf16);
    }

    // Absent scales stay null-ctx (qmatmulBits' plain-matmul arm) and every
    // weight a matmul contracts is stored [in, out].
    const cw = hl.hybrid_layers[0].op.gated_conv;
    try std.testing.expect(cw.in_proj_s.ctx == null);
    try std.testing.expect(cw.out_proj_s.ctx == null);
    try std.testing.expectEqualSlices(c_int, &.{ H, 3 * H }, mlx.getShape(cw.in_proj_w));
    // The depthwise conv weight is never a matmul operand — it must NOT be transposed.
    try std.testing.expectEqualSlices(c_int, &.{ H, 3, 1 }, mlx.getShape(cw.conv_w));

    const fa = hl.hybrid_layers[1].op.full_attn;
    try std.testing.expect(fa.q_s.ctx == null);
    try std.testing.expectEqualSlices(c_int, &.{ H, FF }, mlx.getShape(fa.q_w));
    try std.testing.expectEqualSlices(c_int, &.{ FF, H }, mlx.getShape(fa.o_w));

    const mlp = hl.hybrid_layers[0].mlp.?;
    try std.testing.expect(mlp.gate_s.ctx == null);
    try std.testing.expectEqualSlices(c_int, &.{ H, FF }, mlx.getShape(mlp.gate_w));
    try std.testing.expectEqualSlices(c_int, &.{ FF, H }, mlx.getShape(mlp.down_w));
}

test "depthwiseConvNeedsTranspose reads the LAYOUT, not the arch" {
    const t = std.testing;
    // MLX layout [C, K, 1] — what mlx_lm's LFM2 conversion writes. Leave it.
    try t.expect(!depthwiseConvNeedsTranspose(&.{ 2048, 3, 1 }));
    // Torch layout [C, 1, K] — what LiquidAI's own bf16 repos ship.
    try t.expect(depthwiseConvNeedsTranspose(&.{ 2048, 1, 3 }));
    try t.expect(depthwiseConvNeedsTranspose(&.{ 8, 1, 4 }));
    // K == 1 is the one ambiguous shape, and it is its own transpose.
    try t.expect(!depthwiseConvNeedsTranspose(&.{ 2048, 1, 1 }));
    // Not a depthwise conv weight at all.
    try t.expect(!depthwiseConvNeedsTranspose(&.{ 2048, 3 }));
}

test "hybrid init accepts a TORCH-layout depthwise conv (LiquidAI LFM2.5 bf16)" {
    // Live 2026-08-22: LFM2.5-1.2B-Instruct and LFM2.5-2.6B, downloaded
    // straight from LiquidAI, both died at warmup with
    //   "Given groups=2048 and weights of shape (2048,1,3), expected to have
    //    6144 input channels but got 2048"
    // because the loader assumed every checkpoint had been through an mlx_lm
    // conversion. Same weights, same arch, different converter.
    const allocator = std.testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);

    var w = Weights.init(allocator);
    defer w.deinit();
    const put = struct {
        fn add(weights: *Weights, alloc: std.mem.Allocator, name: []const u8, shape: []const c_int, st: mlx.mlx_stream) !void {
            var n: usize = 1;
            for (shape) |d| n *= @intCast(d);
            const vals = try alloc.alloc(f32, n);
            defer alloc.free(vals);
            @memset(vals, 0.5);
            const f32_arr = mlx.mlx_array_new_data(vals.ptr, shape.ptr, @intCast(shape.len), .float32);
            defer _ = mlx.mlx_array_free(f32_arr);
            var bf = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&bf, f32_arr, .bfloat16, st));
            try weights.map.put(try alloc.dupe(u8, name), bf);
        }
    }.add;

    const H: c_int = 8;
    const FF: c_int = 16;
    const P = "language_model.model";
    const pre = P ++ ".layers.0.";
    try put(&w, allocator, pre ++ "operator_norm.weight", &.{H}, s);
    try put(&w, allocator, pre ++ "ffn_norm.weight", &.{H}, s);
    try put(&w, allocator, pre ++ "feed_forward.w1.weight", &.{ FF, H }, s);
    try put(&w, allocator, pre ++ "feed_forward.w3.weight", &.{ FF, H }, s);
    try put(&w, allocator, pre ++ "feed_forward.w2.weight", &.{ H, FF }, s);
    try put(&w, allocator, pre ++ "conv.in_proj.weight", &.{ 3 * H, H }, s);
    // The whole point: torch's [C, 1, K], not MLX's [C, K, 1].
    try put(&w, allocator, pre ++ "conv.conv.weight", &.{ H, 1, 3 }, s);
    try put(&w, allocator, pre ++ "conv.out_proj.weight", &.{ H, H }, s);

    var config = ModelConfig{ .model_type = "lfm2", .weight_prefix = P };
    config.has_hybrid_layers = true;
    config.num_hidden_layers = 1;
    config.hidden_size = @intCast(H);
    config.quant_bits = 0;
    config.layer_block_types[0] = .gated_conv;

    var name_buf: [256]u8 = undefined;
    const hl = try initHybridLayers(allocator, config, &w, &name_buf, s);
    defer {
        allocator.free(hl.hybrid_layers);
        for (hl.ssm_entries) |*e| {
            _ = mlx.mlx_array_free(e.conv_state);
            _ = mlx.mlx_array_free(e.ssm_state);
        }
        allocator.free(hl.ssm_entries);
        for (hl.owned_bf16) |a| _ = mlx.mlx_array_free(a);
        allocator.free(hl.owned_bf16);
    }

    // It must arrive at the forward in the layout mlx_conv1d solves against.
    const cw = hl.hybrid_layers[0].op.gated_conv;
    try std.testing.expectEqualSlices(c_int, &.{ H, 3, 1 }, mlx.getShape(cw.conv_w));
}
