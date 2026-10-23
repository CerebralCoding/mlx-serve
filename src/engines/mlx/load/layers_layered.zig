//! engines/mlx/load/layers_layered, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns initMoeLayers.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const model_mod = @import("../model.zig");
const log = @import("../../../host/log.zig");
const ModelConfig = @import("../model.zig").ModelConfig;
const moe_affine4 = @import("../../../plugins/qwen4_exp/moe_affine4.zig");
const Weights = @import("../model.zig").Weights;
const concatRows = @import("../../../mlx/array.zig").concatRows;
const materializedOwnedCopy = @import("../../../mlx/array.zig").materializedOwnedCopy;
const naxAvailable = @import("../../../mlx/device.zig").naxAvailable;
const loadDsaWeightsAt = @import("../../../plugins/glm5_next/load.zig").loadDsaWeightsAt;
const loadMhcWeights = @import("../../../plugins/glm5_next/load.zig").loadMhcWeights;
const hy3ExpertContainer = @import("../../../plugins/hy_v3/load.zig").hy3ExpertContainer;
const lagunaRouterBase = @import("../../../plugins/laguna/load.zig").lagunaRouterBase;
const loadHcWeights = @import("../../../plugins/qwen4_exp/load.zig").loadHcWeights;
const loadPleWeights = @import("../../../plugins/qwen4_exp/load.zig").loadPleWeights;
const fuseRowsInPlace = @import("../../../sdk/blocks/linear/fused_rows.zig").fuseRowsInPlace;
const joinableCols = @import("../../../sdk/blocks/linear/fused_rows.zig").joinableCols;
const checkExl3Bank = @import("../../../sdk/quant/exl3.zig").checkExl3Bank;
const MoeLayerWeights = @import("../forward/layered.zig").MoeLayerWeights;
const QW = @import("../layer_weights.zig").QW;
const SSMCacheEntry = @import("../ssm/entry.zig").SSMCacheEntry;
const bf16Scalar = @import("weights.zig").bf16Scalar;
const dequantizeSmall = @import("weights.zig").dequantizeSmall;
const getLayerBias = @import("weights.zig").getLayerBias;
const getLayerScaleOrEmpty = @import("weights.zig").getLayerScaleOrEmpty;
const getLayerWeight = @import("weights.zig").getLayerWeight;
const getLayerWeightAt = @import("weights.zig").getLayerWeightAt;
const getLayerWeightAtOpt = @import("weights.zig").getLayerWeightAtOpt;
const getLayerWeightOpt = @import("weights.zig").getLayerWeightOpt;
const lastDim = @import("weights.zig").lastDim;
const layerBase = @import("weights.zig").layerBase;
const loadQW = @import("weights.zig").loadQW;
const loadRoutedMoeAt = @import("weights.zig").loadRoutedMoeAt;
const maybeTransposeForBf16 = @import("weights.zig").maybeTransposeForBf16;
const moeExpertSuffix = @import("weights.zig").moeExpertSuffix;
const splitPackedGateUp = @import("weights.zig").splitPackedGateUp;

pub fn initMoeLayers(allocator: std.mem.Allocator, config: ModelConfig, weights: *Weights, name_buf: *[256]u8, s: mlx.mlx_stream) !struct { moe_layers: []MoeLayerWeights, ssm_entries: []SSMCacheEntry, owned_bf16: []mlx.mlx_array } {
    log.info("Precomputing MoE layer weights...\n", .{});
    const prefix = config.weight_prefix;
    const moe_layers = try allocator.alloc(MoeLayerWeights, config.num_hidden_layers);
    const ssm_entries = try allocator.alloc(SSMCacheEntry, config.num_hidden_layers);
    var owned_bf16: std.ArrayList(mlx.mlx_array) = .empty;
    errdefer {
        for (owned_bf16.items) |a| _ = mlx.mlx_array_free(a);
        owned_bf16.deinit(allocator);
    }
    // DiffusionGemma reuses the Gemma 4 26B-A4B layer structure verbatim —
    // both take the gemma4 binding/forward arms here.
    const is_gemma4 = config.isGemma4Layers();
    const is_hy3 = std.mem.eql(u8, config.model_type, "hy_v3");
    const is_bailing = std.mem.eql(u8, config.model_type, "bailing_hybrid");
    const is_qwen4 = config.isQwen4();
    const is_mimo = config.isMimo();
    const is_glm = config.isGlm5();
    // Attention projections that may ship dense bf16 beside quantized experts.
    const probe_attn_scales = is_mimo or config.modelopt;

    for (0..config.num_hidden_layers) |i| {
        const li: u32 = @intCast(i);
        const lw = &moe_layers[i];
        const is_linear = config.isLinearLayer(li);

        // qwen4_exp has none: the hyper-connection read normalizes.
        lw.input_norm = if (is_qwen4)
            mlx.mlx_array_new()
        else
            try getLayerWeight(weights, name_buf, prefix, li, "input_layernorm.weight");
        lw.post_attn_norm = if (is_qwen4)
            mlx.mlx_array_new()
        else
            try getLayerWeight(weights, name_buf, prefix, li, "post_attention_layernorm.weight");
        lw.is_linear = is_linear;

        // `moe_layers` comes from `allocator.alloc` which skips struct defaults, so every
        // optional must be initialized before the conditional Gemma-4-only assignments;
        // otherwise the eval loop reads uninitialized memory as valid handles (segfaults
        // with 0xaa...aa on Qwen3-Next and similar non-Gemma MoE models).
        lw.pre_ff_norm = null;
        lw.post_ff_norm = null;
        lw.pre_ff_norm_2 = null;
        lw.post_ff_norm_1 = null;
        lw.post_ff_norm_2 = null;
        lw.layer_scalar = null;
        lw.shared_mlp = null;
        lw.hc_attn = null;
        lw.hc_mlp = null;
        lw.ple = null;
        lw.mhc_attn = null;
        lw.mhc_ffn = null;
        if (is_glm) {
            lw.mhc_attn = try loadMhcWeights(weights, name_buf, prefix, li, "attn_hc", &owned_bf16, allocator, s);
            lw.mhc_ffn = try loadMhcWeights(weights, name_buf, prefix, li, "ffn_hc", &owned_bf16, allocator, s);
        }
        if (is_qwen4) {
            var hb: [96]u8 = undefined;
            lw.hc_attn = try loadHcWeights(weights, std.fmt.bufPrint(&hb, "{s}.layers.{d}.attn_hyper_connection", .{ prefix, li }) catch unreachable, true, config.hc_count, config.hidden_size, &owned_bf16, allocator, s);
            lw.hc_mlp = try loadHcWeights(weights, std.fmt.bufPrint(&hb, "{s}.layers.{d}.mlp_hyper_connection", .{ prefix, li }) catch unreachable, true, config.hc_count, config.hidden_size, &owned_bf16, allocator, s);
            if (config.pleLocalLayer() == li) {
                lw.ple = try loadPleWeights(weights, name_buf, prefix, li, config.hc_count, config.hidden_size, &owned_bf16, allocator, s);
            }
        }

        // Gemma 4 MoE: extra feedforward norms, layer scalar, shared expert MLP
        if (is_gemma4) {
            lw.pre_ff_norm = getLayerWeightOpt(weights, name_buf, prefix, li, "pre_feedforward_layernorm.weight");
            lw.post_ff_norm = getLayerWeightOpt(weights, name_buf, prefix, li, "post_feedforward_layernorm.weight");
            lw.pre_ff_norm_2 = getLayerWeightOpt(weights, name_buf, prefix, li, "pre_feedforward_layernorm_2.weight");
            lw.post_ff_norm_1 = getLayerWeightOpt(weights, name_buf, prefix, li, "post_feedforward_layernorm_1.weight");
            lw.post_ff_norm_2 = getLayerWeightOpt(weights, name_buf, prefix, li, "post_feedforward_layernorm_2.weight");
            lw.layer_scalar = getLayerWeightOpt(weights, name_buf, prefix, li, "layer_scalar");
            lw.shared_mlp = .{
                .gate_w = try getLayerWeight(weights, name_buf, prefix, li, "mlp.gate_proj.weight"),
                .gate_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.gate_proj.scales") orelse mlx.mlx_array_new(),
                .gate_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.gate_proj.biases") orelse mlx.mlx_array_new(),
                .up_w = try getLayerWeight(weights, name_buf, prefix, li, "mlp.up_proj.weight"),
                .up_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.up_proj.scales") orelse mlx.mlx_array_new(),
                .up_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.up_proj.biases") orelse mlx.mlx_array_new(),
                .down_w = try getLayerWeight(weights, name_buf, prefix, li, "mlp.down_proj.weight"),
                .down_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.down_proj.scales") orelse mlx.mlx_array_new(),
                .down_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.down_proj.biases") orelse mlx.mlx_array_new(),
            };
            const sm = &lw.shared_mlp.?;
            try maybeTransposeForBf16(&sm.gate_w, sm.gate_s, &owned_bf16, allocator, s);
            try maybeTransposeForBf16(&sm.up_w, sm.up_s, &owned_bf16, allocator, s);
            try maybeTransposeForBf16(&sm.down_w, sm.down_s, &owned_bf16, allocator, s);
            if (sm.gate_s.ctx != null) {
                var parts = [_][3]*mlx.mlx_array{ .{ &sm.gate_w, &sm.gate_s, &sm.gate_b }, .{ &sm.up_w, &sm.up_s, &sm.up_b } };
                const names = [_][]const u8{ "mlp.gate_proj", "mlp.up_proj" };
                sm.gu = try fuseRowsInPlace(&parts, 0, &names, weights, name_buf, prefix, li, &owned_bf16, allocator, s);
            }
        }

        if (is_linear and (is_bailing or is_glm)) {
            // Ling names the block `attention.*` with full-rank gates; GLM-5-Next
            // `self_attn.*` with low-rank ones (f_a→f_b, g_a→g_b).
            const ab: []const u8 = if (is_glm) "self_attn" else "attention";
            // KDA. The checkpoint ships q/k/v as three separate projections and
            // three separate depthwise convs; joining them here is EXACT (the
            // convs are depthwise, so channel blocks never mix, and affine
            // groups run along the input axis) and leaves the recurrence's hot
            // path byte-for-byte the qwen3.5 GatedDeltaNet one.
            //   f_proj → the per-channel gate (`a`)
            //   b_proj → beta,  g_proj → the output gate (`z`),  o_norm → norm
            const qkv_names = [_][]const u8{ "q_proj", "k_proj", "v_proj" };
            var qkv_w: [3]mlx.mlx_array = undefined;
            var qkv_s: [3]mlx.mlx_array = undefined;
            var qkv_b: [3]mlx.mlx_array = undefined;
            var conv_w: [3]mlx.mlx_array = undefined;
            var any_scales = false;
            // oMLX 0.7.1 packs ship the three convs already joined ([C, K, 1]) and
            // the gate (A_log, dt_bias, f_a/f_b_proj) under `forget_gate.`.
            const fused_conv = getLayerWeightAtOpt(weights, name_buf, prefix, li, ab, "conv1d.weight");
            var fg_buf: [64]u8 = undefined;
            const fg: []const u8 = if (getLayerWeightAtOpt(weights, name_buf, prefix, li, ab, "forget_gate.A_log") != null)
                std.fmt.bufPrint(&fg_buf, "{s}.forget_gate", .{ab}) catch unreachable
            else
                ab;
            for (qkv_names, 0..) |base, n| {
                var nb: [256]u8 = undefined;
                qkv_w[n] = try getLayerWeightAt(weights, name_buf, prefix, li, ab, std.fmt.bufPrint(&nb, "{s}.weight", .{base}) catch unreachable);
                qkv_s[n] = getLayerWeightAtOpt(weights, name_buf, prefix, li, ab, std.fmt.bufPrint(&nb, "{s}.scales", .{base}) catch unreachable) orelse mlx.mlx_array_new();
                qkv_b[n] = getLayerWeightAtOpt(weights, name_buf, prefix, li, ab, std.fmt.bufPrint(&nb, "{s}.biases", .{base}) catch unreachable) orelse mlx.mlx_array_new();
                if (qkv_s[n].ctx != null) any_scales = true;
                // The depthwise conv is `<x>_conv1d.conv.weight` in the reference
                // HF checkpoint, where it sits inside a ShortConvolution module.
                // mlx-lm's converter flattens that module away and emits
                // `<x>_conv1d.weight` instead — same [C, 1, K] tensor, one name
                // segment shorter. Both public MLX Ling-3.0 quants ship the flat
                // form (djrsystemservices/Ling-3.0-tiny-oQ8e,
                // d1sl1ke/Ling-3.0-flash-oQ4e-mtp), so a nested-only lookup
                // aborts the load on every one of them. Prefer the nested name
                // so reference checkpoints bind exactly as before, then fall
                // back to the flat one.
                const conv_nested = switch (n) {
                    0 => "q_conv1d.conv.weight",
                    1 => "k_conv1d.conv.weight",
                    else => "v_conv1d.conv.weight",
                };
                const conv_flat = switch (n) {
                    0 => "q_conv1d.weight",
                    1 => "k_conv1d.weight",
                    else => "v_conv1d.weight",
                };
                if (fused_conv == null) conv_w[n] = getLayerWeightAtOpt(weights, name_buf, prefix, li, ab, conv_nested) orelse
                    try getLayerWeightAt(weights, name_buf, prefix, li, ab, conv_flat);
            }
            // The three projections join only when quantized IDENTICALLY: a q at
            // 5-bit beside an 8-bit k (TensorFold's GLM pack, layer 6) has other
            // packed widths, and a dense q beside a quantized k has scales for
            // part of the rows. Such a layer projects q, k and v one by one.
            const joinable = blk: {
                var w_cols: [3]c_int = undefined;
                var s_cols: [3]c_int = undefined;
                for (qkv_w, 0..) |w, n| {
                    if (any_scales and qkv_s[n].ctx == null) break :blk false;
                    w_cols[n] = @intCast(lastDim(w) orelse 0);
                    s_cols[n] = @intCast(lastDim(qkv_s[n]) orelse 0);
                }
                break :blk joinableCols(&w_cols) and joinableCols(&s_cols);
            };
            var qkv_parts: [3]QW = .{ .{}, .{}, .{} };
            var joined_w = mlx.mlx_array_new();
            if (joinable) {
                joined_w = try concatRows(&qkv_w, s);
                try owned_bf16.append(allocator, joined_w);
            } else {
                for (&qkv_parts, 0..) |*q, n| q.* = .{ .w = qkv_w[n], .s = qkv_s[n], .b = qkv_b[n] };
            }
            var joined_conv = fused_conv orelse blk: {
                const j = try concatRows(&conv_w, s);
                try owned_bf16.append(allocator, j);
                break :blk j;
            };
            // HF-layout packs store the depthwise conv [C, 1, K]; MLX's conv1d reads [C, K, 1].
            const csh = mlx.getShape(joined_conv);
            if (csh.len == 3 and csh[1] == 1 and csh[2] > 1) {
                var t = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(t);
                try mlx.check(mlx.mlx_swapaxes(&t, joined_conv, 1, 2, s));
                joined_conv = try materializedOwnedCopy(s, t);
                try mlx.check(mlx.mlx_array_eval(joined_conv));
                try owned_bf16.append(allocator, joined_conv);
            }
            var joined_s = mlx.mlx_array_new();
            var joined_b = mlx.mlx_array_new();
            if (any_scales and joinable) {
                joined_s = try concatRows(&qkv_s, s);
                try owned_bf16.append(allocator, joined_s);
                if (qkv_b[0].ctx != null) {
                    joined_b = try concatRows(&qkv_b, s);
                    try owned_bf16.append(allocator, joined_b);
                }
            }

            // A_log is [Hv] but the gate is per KEY CHANNEL — expand once here
            // so the hot path stays elementwise over flat [B,S,Hv*Dk] arrays
            // (no reshape, no per-head broadcast, per layer per token).
            const a_log_raw = try getLayerWeightAt(weights, name_buf, prefix, li, fg, "A_log");
            const dk_c: c_int = @intCast(config.linear_key_head_dim);
            const hv_c: c_int = @intCast(config.linear_num_value_heads);
            var a_log_exp = mlx.mlx_array_new();
            {
                var a_col = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(a_col);
                try mlx.check(mlx.mlx_reshape(&a_col, a_log_raw, &[_]c_int{ hv_c, 1 }, 2, s));
                var a_bc = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(a_bc);
                try mlx.check(mlx.mlx_broadcast_to(&a_bc, a_col, &[_]c_int{ hv_c, dk_c }, 2, s));
                try mlx.check(mlx.mlx_reshape(&a_log_exp, a_bc, &[_]c_int{hv_c * dk_c}, 1, s));
                try mlx.check(mlx.mlx_array_eval(a_log_exp));
            }
            try owned_bf16.append(allocator, a_log_exp);

            var gb: [64]u8 = undefined;
            const z = try loadQW(weights, name_buf, prefix, li, std.fmt.bufPrint(&gb, "{s}.{s}", .{ ab, if (is_glm) "g_a_proj" else "g_proj" }) catch unreachable);
            const f = try loadQW(weights, name_buf, prefix, li, std.fmt.bufPrint(&gb, "{s}.{s}", .{ fg, if (is_glm) "f_a_proj" else "f_proj" }) catch unreachable);
            const bq = try loadQW(weights, name_buf, prefix, li, std.fmt.bufPrint(&gb, "{s}.b_proj", .{ab}) catch unreachable);
            const oq = try loadQW(weights, name_buf, prefix, li, std.fmt.bufPrint(&gb, "{s}.o_proj", .{ab}) catch unreachable);
            lw.attn = .{ .linear = .{
                .qkv_w = joined_w,
                .qkv_s = joined_s,
                .qkv_b = joined_b,
                .z_w = z.w,
                .z_s = z.s,
                .z_b = z.b,
                .a_w = f.w,
                .a_s = f.s,
                .a_b = f.b,
                .b_w = bq.w,
                .b_s = bq.s,
                .b_b = bq.b,
                .conv1d_w = joined_conv,
                .A_log = a_log_exp,
                .dt_bias = try getLayerWeightAt(weights, name_buf, prefix, li, fg, "dt_bias"),
                .norm_w = try getLayerWeightAt(weights, name_buf, prefix, li, ab, "o_norm.weight"),
                .out_w = oq.w,
                .out_s = oq.s,
                .out_b = oq.b,
                .a2 = if (is_glm) try loadQW(weights, name_buf, prefix, li, std.fmt.bufPrint(&gb, "{s}.f_b_proj", .{fg}) catch unreachable) else .{},
                .z2 = if (is_glm) try loadQW(weights, name_buf, prefix, li, std.fmt.bufPrint(&gb, "{s}.g_b_proj", .{ab}) catch unreachable) else .{},
                .qkv_parts = qkv_parts,
            } };
            const la = &lw.attn.linear;
            if (is_glm) {
                inline for (.{ .{ &la.a_log_h, a_log_raw }, .{ &la.dt_f32, la.dt_bias } }) |pair| {
                    pair[0].* = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_astype(pair[0], pair[1], .float32, s));
                    try mlx.check(mlx.mlx_array_eval(pair[0].*));
                    try owned_bf16.append(allocator, pair[0].*);
                }
            }
            try maybeTransposeForBf16(&la.qkv_w, la.qkv_s, &owned_bf16, allocator, s);
            for (&la.qkv_parts) |*q| try maybeTransposeForBf16(&q.w, q.s, &owned_bf16, allocator, s);
            try maybeTransposeForBf16(&la.z_w, la.z_s, &owned_bf16, allocator, s);
            try maybeTransposeForBf16(&la.a_w, la.a_s, &owned_bf16, allocator, s);
            try maybeTransposeForBf16(&la.b_w, la.b_s, &owned_bf16, allocator, s);
            try maybeTransposeForBf16(&la.out_w, la.out_s, &owned_bf16, allocator, s);
            if (la.a2.w.ctx != null) try maybeTransposeForBf16(&la.a2.w, la.a2.s, &owned_bf16, allocator, s);
            if (la.z2.w.ctx != null) try maybeTransposeForBf16(&la.z2.w, la.z2.s, &owned_bf16, allocator, s);
            // Decode reads q|k|v and the first gate stages as one row-joined matmul.
            if (is_glm and la.qkv_s.ctx != null) {
                var parts = [_][3]*mlx.mlx_array{ .{ &la.qkv_w, &la.qkv_s, &la.qkv_b }, .{ &la.z_w, &la.z_s, &la.z_b }, .{ &la.a_w, &la.a_s, &la.a_b }, .{ &la.b_w, &la.b_s, &la.b_b } };
                var fa_buf: [64]u8 = undefined;
                const names = [_][]const u8{ "self_attn.g_a_proj", std.fmt.bufPrint(&fa_buf, "{s}.f_a_proj", .{fg}) catch unreachable, "self_attn.b_proj" };
                la.in = try fuseRowsInPlace(&parts, 1, &names, weights, name_buf, prefix, li, &owned_bf16, allocator, s);
            }
        } else if (is_linear) {
            // Detect combined (qkvz+ba) vs separate (qkv+z+a+b) projections.
            // Each projection's `*_s`/`*_b` are loaded optionally — Unsloth Dynamic
            // checkpoints (e.g. Qwen3.6 UD) leave linear_attn projections as plain
            // bf16 with no scales/biases tensors, even though the rest of the model
            // is quantized. Null-ctx scales triggers a transpose-on-load so that
            // `qmatmulBits` can dispatch to plain `mlx_matmul`.
            const combined = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_qkvz.weight") != null;
            if (combined) {
                lw.attn = .{ .linear = .{
                    .combined_proj = true,
                    .qkv_w = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.in_proj_qkvz.weight"),
                    .qkv_s = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_qkvz.scales") orelse mlx.mlx_array_new(),
                    .qkv_b = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_qkvz.biases") orelse mlx.mlx_array_new(),
                    .z_w = mlx.mlx_array_new(),
                    .z_s = mlx.mlx_array_new(),
                    .z_b = mlx.mlx_array_new(),
                    .a_w = mlx.mlx_array_new(),
                    .a_s = mlx.mlx_array_new(),
                    .a_b = mlx.mlx_array_new(),
                    .b_w = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.in_proj_ba.weight"),
                    .b_s = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_ba.scales") orelse mlx.mlx_array_new(),
                    .b_b = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_ba.biases") orelse mlx.mlx_array_new(),
                    .conv1d_w = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.conv1d.weight"),
                    .A_log = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.A_log"),
                    .dt_bias = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.dt_bias"),
                    .norm_w = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.norm.weight"),
                    .out_w = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.out_proj.weight"),
                    .out_s = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.out_proj.scales") orelse mlx.mlx_array_new(),
                    .out_b = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.out_proj.biases") orelse mlx.mlx_array_new(),
                } };
                const la = &lw.attn.linear;
                try maybeTransposeForBf16(&la.qkv_w, la.qkv_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&la.b_w, la.b_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&la.out_w, la.out_s, &owned_bf16, allocator, s);
            } else {
                lw.attn = .{ .linear = .{
                    .qkv_w = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.in_proj_qkv.weight"),
                    .qkv_s = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_qkv.scales") orelse mlx.mlx_array_new(),
                    .qkv_b = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_qkv.biases") orelse mlx.mlx_array_new(),
                    .z_w = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.in_proj_z.weight"),
                    .z_s = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_z.scales") orelse mlx.mlx_array_new(),
                    .z_b = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_z.biases") orelse mlx.mlx_array_new(),
                    .a_w = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.in_proj_a.weight"),
                    .a_s = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_a.scales") orelse mlx.mlx_array_new(),
                    .a_b = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_a.biases") orelse mlx.mlx_array_new(),
                    .b_w = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.in_proj_b.weight"),
                    .b_s = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_b.scales") orelse mlx.mlx_array_new(),
                    .b_b = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_b.biases") orelse mlx.mlx_array_new(),
                    .conv1d_w = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.conv1d.weight"),
                    .A_log = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.A_log"),
                    .dt_bias = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.dt_bias"),
                    .norm_w = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.norm.weight"),
                    .out_w = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.out_proj.weight"),
                    .out_s = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.out_proj.scales") orelse mlx.mlx_array_new(),
                    .out_b = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.out_proj.biases") orelse mlx.mlx_array_new(),
                } };
                const la = &lw.attn.linear;
                try maybeTransposeForBf16(&la.qkv_w, la.qkv_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&la.z_w, la.z_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&la.a_w, la.a_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&la.b_w, la.b_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&la.out_w, la.out_s, &owned_bf16, allocator, s);
                if (la.qkv_s.ctx != null) {
                    var parts = [_][3]*mlx.mlx_array{ .{ &la.qkv_w, &la.qkv_s, &la.qkv_b }, .{ &la.z_w, &la.z_s, &la.z_b }, .{ &la.a_w, &la.a_s, &la.a_b }, .{ &la.b_w, &la.b_s, &la.b_b } };
                    const names = [_][]const u8{ "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_a", "linear_attn.in_proj_b" };
                    la.in = try fuseRowsInPlace(&parts, 0, &names, weights, name_buf, prefix, li, &owned_bf16, allocator, s);
                }
            }
        } else if (is_glm) {
            // Sparse latent attention: no q/k/v projections; `o_proj` is the output.
            var base_buf: [96]u8 = undefined;
            const dsa = try loadDsaWeightsAt(weights, name_buf, layerBase(&base_buf, prefix, li), &owned_bf16, allocator, s);
            lw.attn = .{ .full = .{
                .q_w = mlx.mlx_array_new(),
                .q_s = mlx.mlx_array_new(),
                .q_b = mlx.mlx_array_new(),
                .k_w = mlx.mlx_array_new(),
                .k_s = mlx.mlx_array_new(),
                .k_b = mlx.mlx_array_new(),
                .v_w = mlx.mlx_array_new(),
                .v_s = mlx.mlx_array_new(),
                .v_b = mlx.mlx_array_new(),
                .o_w = dsa.o.w,
                .o_s = dsa.o.s,
                .o_b = dsa.o.b,
                .q_norm = mlx.mlx_array_new(),
                .k_norm = mlx.mlx_array_new(),
                .dsa = dsa,
            } };
        } else if (is_bailing) {
            // MLA. There are no q/k/v projections at all — the low-rank Q and
            // the compressed KV latent replace them, and `dense` is the output
            // projection. q_w/k_w/v_w stay null-ctx (skipped by the eval batch
            // and by every fused-QKV probe, which all key on q_w).
            lw.attn = .{
                .full = .{
                    .q_w = mlx.mlx_array_new(),
                    .q_s = mlx.mlx_array_new(),
                    .q_b = mlx.mlx_array_new(),
                    .k_w = mlx.mlx_array_new(),
                    .k_s = mlx.mlx_array_new(),
                    .k_b = mlx.mlx_array_new(),
                    .v_w = mlx.mlx_array_new(),
                    .v_s = mlx.mlx_array_new(),
                    .v_b = mlx.mlx_array_new(),
                    .o_w = try getLayerWeight(weights, name_buf, prefix, li, "attention.dense.weight"),
                    .o_s = getLayerWeightOpt(weights, name_buf, prefix, li, "attention.dense.scales") orelse mlx.mlx_array_new(),
                    .o_b = getLayerWeightOpt(weights, name_buf, prefix, li, "attention.dense.biases") orelse mlx.mlx_array_new(),
                    // MLA norms the LATENTS, not the per-head q/k.
                    .q_norm = mlx.mlx_array_new(),
                    .k_norm = mlx.mlx_array_new(),
                    .g_w = try getLayerWeight(weights, name_buf, prefix, li, "attention.g_proj.weight"),
                    .g_s = getLayerWeightOpt(weights, name_buf, prefix, li, "attention.g_proj.scales") orelse mlx.mlx_array_new(),
                    .g_b = getLayerWeightOpt(weights, name_buf, prefix, li, "attention.g_proj.biases") orelse mlx.mlx_array_new(),
                    // Exactly ONE Q arm per checkpoint: the low-rank pair
                    // (tiny) or a direct q_proj (flash, `q_lora_rank: null`).
                    // Asking for the absent one would abort the load.
                    .mla_q_direct_w = if (config.mlaHasQLora()) mlx.mlx_array_new() else try getLayerWeight(weights, name_buf, prefix, li, "attention.q_proj.weight"),
                    .mla_q_direct_s = if (config.mlaHasQLora()) mlx.mlx_array_new() else (getLayerWeightOpt(weights, name_buf, prefix, li, "attention.q_proj.scales") orelse mlx.mlx_array_new()),
                    .mla_q_direct_b = if (config.mlaHasQLora()) mlx.mlx_array_new() else (getLayerWeightOpt(weights, name_buf, prefix, li, "attention.q_proj.biases") orelse mlx.mlx_array_new()),
                    .mla_q_a_w = if (config.mlaHasQLora()) try getLayerWeight(weights, name_buf, prefix, li, "attention.q_a_proj.weight") else mlx.mlx_array_new(),
                    .mla_q_a_s = if (config.mlaHasQLora()) (getLayerWeightOpt(weights, name_buf, prefix, li, "attention.q_a_proj.scales") orelse mlx.mlx_array_new()) else mlx.mlx_array_new(),
                    .mla_q_a_b = if (config.mlaHasQLora()) (getLayerWeightOpt(weights, name_buf, prefix, li, "attention.q_a_proj.biases") orelse mlx.mlx_array_new()) else mlx.mlx_array_new(),
                    .mla_q_a_norm = if (config.mlaHasQLora()) try getLayerWeight(weights, name_buf, prefix, li, "attention.q_a_layernorm.weight") else mlx.mlx_array_new(),
                    .mla_q_b_w = if (config.mlaHasQLora()) try getLayerWeight(weights, name_buf, prefix, li, "attention.q_b_proj.weight") else mlx.mlx_array_new(),
                    .mla_q_b_s = if (config.mlaHasQLora()) (getLayerWeightOpt(weights, name_buf, prefix, li, "attention.q_b_proj.scales") orelse mlx.mlx_array_new()) else mlx.mlx_array_new(),
                    .mla_q_b_b = if (config.mlaHasQLora()) (getLayerWeightOpt(weights, name_buf, prefix, li, "attention.q_b_proj.biases") orelse mlx.mlx_array_new()) else mlx.mlx_array_new(),
                    .mla_kv_a_w = try getLayerWeight(weights, name_buf, prefix, li, "attention.kv_a_proj_with_mqa.weight"),
                    .mla_kv_a_s = getLayerWeightOpt(weights, name_buf, prefix, li, "attention.kv_a_proj_with_mqa.scales") orelse mlx.mlx_array_new(),
                    .mla_kv_a_b = getLayerWeightOpt(weights, name_buf, prefix, li, "attention.kv_a_proj_with_mqa.biases") orelse mlx.mlx_array_new(),
                    .mla_kv_a_norm = try getLayerWeight(weights, name_buf, prefix, li, "attention.kv_a_layernorm.weight"),
                    .mla_kv_b_w = try getLayerWeight(weights, name_buf, prefix, li, "attention.kv_b_proj.weight"),
                    .mla_kv_b_s = getLayerWeightOpt(weights, name_buf, prefix, li, "attention.kv_b_proj.scales") orelse mlx.mlx_array_new(),
                    .mla_kv_b_b = getLayerWeightOpt(weights, name_buf, prefix, li, "attention.kv_b_proj.biases") orelse mlx.mlx_array_new(),
                },
            };
            {
                const fa = &lw.attn.full;
                try maybeTransposeForBf16(&fa.o_w, fa.o_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&fa.g_w, fa.g_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&fa.mla_q_direct_w, fa.mla_q_direct_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&fa.mla_q_a_w, fa.mla_q_a_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&fa.mla_q_b_w, fa.mla_q_b_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&fa.mla_kv_a_w, fa.mla_kv_a_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&fa.mla_kv_b_w, fa.mla_kv_b_s, &owned_bf16, allocator, s);
            }
        } else {
            // MiMo and ModelOpt packs may ship attention DENSE bf16 (no scales)
            // beside quantized experts. Probe per-tensor: scales
            // present → the quant path (quantParamsHinted resolves bits from
            // geometry); absent → bf16 (maybeTransposeForBf16). Other archs keep
            // mandating scales (config.quant_bits) so a genuinely-missing scale
            // still errors loudly. getLayerBias already tolerates absence.
            const k_s = if (probe_attn_scales)
                (getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.k_proj.scales") orelse mlx.mlx_array_new())
            else
                try getLayerScaleOrEmpty(weights, name_buf, prefix, li, "self_attn.k_proj.scales", config.quant_bits);
            const k_w = try getLayerWeight(weights, name_buf, prefix, li, "self_attn.k_proj.weight");
            const k_b = try getLayerBias(weights, name_buf, prefix, li, "self_attn.k_proj.biases", &config);
            // Gemma 4 MoE: global layers use K=V (no separate v_proj)
            const v_w = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.v_proj.weight") orelse k_w;
            const v_s = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.v_proj.scales") orelse k_s;
            const v_b = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.v_proj.biases") orelse k_b;
            const v_aliases_k = v_w.ctx == k_w.ctx;
            lw.attn = .{
                .full = .{
                    .q_w = try getLayerWeight(weights, name_buf, prefix, li, "self_attn.q_proj.weight"),
                    .q_s = if (probe_attn_scales)
                        (getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.q_proj.scales") orelse mlx.mlx_array_new())
                    else
                        try getLayerScaleOrEmpty(weights, name_buf, prefix, li, "self_attn.q_proj.scales", config.quant_bits),
                    .q_b = try getLayerBias(weights, name_buf, prefix, li, "self_attn.q_proj.biases", &config),
                    .k_w = k_w,
                    .k_s = k_s,
                    .k_b = k_b,
                    .v_w = v_w,
                    .v_s = v_s,
                    .v_b = v_b,
                    .o_w = try getLayerWeight(weights, name_buf, prefix, li, "self_attn.o_proj.weight"),
                    .o_s = if (probe_attn_scales)
                        (getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.o_proj.scales") orelse mlx.mlx_array_new())
                    else
                        try getLayerScaleOrEmpty(weights, name_buf, prefix, li, "self_attn.o_proj.scales", config.quant_bits),
                    .o_b = try getLayerBias(weights, name_buf, prefix, li, "self_attn.o_proj.biases", &config),
                    // An arch that DECLARES QK norm must ship it (a missing weight
                    // there is a broken checkpoint, not a variant). One that
                    // declares it off (MiMo) has no q_norm/k_norm tensors at
                    // all, and the mandatory getter's `unreachable` would kill the
                    // process on load rather than report anything.
                    .q_norm = if (config.has_qk_norm)
                        try getLayerWeight(weights, name_buf, prefix, li, "self_attn.q_norm.weight")
                    else
                        (getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.q_norm.weight") orelse mlx.mlx_array_new()),
                    .k_norm = if (config.has_qk_norm)
                        try getLayerWeight(weights, name_buf, prefix, li, "self_attn.k_norm.weight")
                    else
                        (getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.k_norm.weight") orelse mlx.mlx_array_new()),
                },
            };
            {
                // Dense bf16 (null-ctx scales): pre-transpose [out,in]→[in,out] so
                // qmatmulBits dispatches to a plain matmul. No-op on quantized weights.
                const fa = &lw.attn.full;
                try maybeTransposeForBf16(&fa.q_w, fa.q_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&fa.k_w, fa.k_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&fa.o_w, fa.o_s, &owned_bf16, allocator, s);
                if (v_aliases_k) {
                    // K=V share one weight — re-alias V to the transposed K (don't
                    // create a second copy or double-free).
                    fa.v_w = fa.k_w;
                } else {
                    try maybeTransposeForBf16(&fa.v_w, fa.v_s, &owned_bf16, allocator, s);
                }
                if (is_qwen4 and config.indexer_n_heads > 0) {
                    fa.idx_qk_w = try getLayerWeight(weights, name_buf, prefix, li, "self_attn.indexer.index_qk_proj.weight");
                    fa.idx_qk_s = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.indexer.index_qk_proj.scales") orelse mlx.mlx_array_new();
                    fa.idx_qk_b = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.indexer.index_qk_proj.biases") orelse mlx.mlx_array_new();
                    fa.idx_q_norm = try getLayerWeight(weights, name_buf, prefix, li, "self_attn.indexer.q_layernorm.weight");
                    fa.idx_k_norm = try getLayerWeight(weights, name_buf, prefix, li, "self_attn.indexer.k_layernorm.weight");
                    try maybeTransposeForBf16(&fa.idx_qk_w, fa.idx_qk_s, &owned_bf16, allocator, s);
                }
                // MiMo: ADDITIVE projection biases + per-head sinks.
                //
                // `.bias` and `.biases` are DIFFERENT TENSORS living side by
                // side on the same projection in this checkpoint:
                //   self_attn.q_proj.bias    [n_heads*head_dim]  additive
                //   self_attn.q_proj.biases  [out, groups]       affine zero-points
                // The affine ones are already bound above as `q_b`/`k_b`/…
                // via getLayerBias. Reading either name for the other is a
                // shape error at best and silently wrong output at worst, so
                // the two suffixes stay spelled out in full at every site.
                if (is_mimo) {
                    fa.q_bias = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.q_proj.bias") orelse .{ .ctx = null };
                    fa.k_bias = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.k_proj.bias") orelse .{ .ctx = null };
                    fa.v_bias = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.v_proj.bias") orelse .{ .ctx = null };
                    fa.o_bias = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.o_proj.bias") orelse .{ .ctx = null };
                    fa.sinks = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.sinks") orelse
                        getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.attention_sink_bias") orelse .{ .ctx = null };
                }
                // Row-exact archs on NAX: one lane matmul serves q|k|v at every
                // width (a column's bits follow the joined shape, the same at 1 row and 16).
                const lane_join = naxAvailable() and config.rowExactArch();
                if ((is_gemma4 or lane_join) and !v_aliases_k and fa.q_s.ctx != null) {
                    var parts = [_][3]*mlx.mlx_array{ .{ &fa.q_w, &fa.q_s, &fa.q_b }, .{ &fa.k_w, &fa.k_s, &fa.k_b }, .{ &fa.v_w, &fa.v_s, &fa.v_b } };
                    const names = [_][]const u8{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj" };
                    fa.qkv = try fuseRowsInPlace(&parts, 0, &names, weights, name_buf, prefix, li, &owned_bf16, allocator, s);
                }
            }
        }

        // hy_v3: layers [0, first_k_dense_replace) are DENSE (layer 0 on the
        // 295B) — they take the dense binding arm below. Every established
        // arch has first_k_dense_replace == 0, so layer_is_moe == isMoe() there.
        const layer_is_moe = config.isMoe() and li >= config.first_k_dense_replace;
        if (layer_is_moe and (is_bailing or is_glm)) {
            var base_buf: [96]u8 = undefined;
            lw.mlp = .{ .moe = try loadRoutedMoeAt(weights, name_buf, layerBase(&base_buf, prefix, li), is_glm, &config, &owned_bf16, allocator, s) };
        } else if (layer_is_moe and is_gemma4) {
            // Gemma 4 MoE: different weight naming, Sigma-MoE routing, no shared expert gate.
            // Each `*_s`/`*_b` is loaded optionally for Unsloth Dynamic compatibility —
            // bf16 layers carry only the weight, no scales/biases. The post-construction
            // `maybeTransposeForBf16` calls are no-ops for already-quantized weights.
            //
            // Expert naming variants:
            //   Gemma 4 26B-A4B:  experts.switch_glu.{gate,up,down}_proj.*
            //   DiffusionGemma:   experts.gate_up_proj.* (PACKED [E, 2M, X],
            //                     gate rows first) + experts.down_proj.*
            // The packed variant is sliced into gate/up halves at load
            // (splitPackedGateUp) so both feed the same MoeMlpWeights fields.
            const packed_gu_w = getLayerWeightOpt(weights, name_buf, prefix, li, "experts.gate_up_proj.weight");
            var sw_gate_w: mlx.mlx_array = undefined;
            var sw_gate_s: mlx.mlx_array = undefined;
            var sw_gate_b: mlx.mlx_array = undefined;
            var sw_up_w: mlx.mlx_array = undefined;
            var sw_up_s: mlx.mlx_array = undefined;
            var sw_up_b: mlx.mlx_array = undefined;
            var sw_down_w: mlx.mlx_array = undefined;
            var sw_down_s: mlx.mlx_array = undefined;
            var sw_down_b: mlx.mlx_array = undefined;
            if (packed_gu_w) |gu_w| {
                const w_pair = try splitPackedGateUp(gu_w, s);
                try owned_bf16.append(allocator, w_pair.gate);
                try owned_bf16.append(allocator, w_pair.up);
                sw_gate_w = w_pair.gate;
                sw_up_w = w_pair.up;
                if (getLayerWeightOpt(weights, name_buf, prefix, li, "experts.gate_up_proj.scales")) |gu_s| {
                    const s_pair = try splitPackedGateUp(gu_s, s);
                    try owned_bf16.append(allocator, s_pair.gate);
                    try owned_bf16.append(allocator, s_pair.up);
                    sw_gate_s = s_pair.gate;
                    sw_up_s = s_pair.up;
                } else {
                    sw_gate_s = mlx.mlx_array_new();
                    sw_up_s = mlx.mlx_array_new();
                }
                if (getLayerWeightOpt(weights, name_buf, prefix, li, "experts.gate_up_proj.biases")) |gu_b| {
                    const b_pair = try splitPackedGateUp(gu_b, s);
                    try owned_bf16.append(allocator, b_pair.gate);
                    try owned_bf16.append(allocator, b_pair.up);
                    sw_gate_b = b_pair.gate;
                    sw_up_b = b_pair.up;
                } else {
                    sw_gate_b = mlx.mlx_array_new();
                    sw_up_b = mlx.mlx_array_new();
                }
                sw_down_w = try getLayerWeight(weights, name_buf, prefix, li, "experts.down_proj.weight");
                sw_down_s = getLayerWeightOpt(weights, name_buf, prefix, li, "experts.down_proj.scales") orelse mlx.mlx_array_new();
                sw_down_b = getLayerWeightOpt(weights, name_buf, prefix, li, "experts.down_proj.biases") orelse mlx.mlx_array_new();
            } else {
                sw_gate_w = try getLayerWeight(weights, name_buf, prefix, li, "experts.switch_glu.gate_proj.weight");
                sw_gate_s = getLayerWeightOpt(weights, name_buf, prefix, li, "experts.switch_glu.gate_proj.scales") orelse mlx.mlx_array_new();
                sw_gate_b = getLayerWeightOpt(weights, name_buf, prefix, li, "experts.switch_glu.gate_proj.biases") orelse mlx.mlx_array_new();
                sw_up_w = try getLayerWeight(weights, name_buf, prefix, li, "experts.switch_glu.up_proj.weight");
                sw_up_s = getLayerWeightOpt(weights, name_buf, prefix, li, "experts.switch_glu.up_proj.scales") orelse mlx.mlx_array_new();
                sw_up_b = getLayerWeightOpt(weights, name_buf, prefix, li, "experts.switch_glu.up_proj.biases") orelse mlx.mlx_array_new();
                sw_down_w = try getLayerWeight(weights, name_buf, prefix, li, "experts.switch_glu.down_proj.weight");
                sw_down_s = getLayerWeightOpt(weights, name_buf, prefix, li, "experts.switch_glu.down_proj.scales") orelse mlx.mlx_array_new();
                sw_down_b = getLayerWeightOpt(weights, name_buf, prefix, li, "experts.switch_glu.down_proj.biases") orelse mlx.mlx_array_new();
            }
            lw.mlp = .{
                .moe = .{
                    .router_w = try getLayerWeight(weights, name_buf, prefix, li, "router.proj.weight"),
                    .router_s = getLayerWeightOpt(weights, name_buf, prefix, li, "router.proj.scales") orelse mlx.mlx_array_new(),
                    .router_b = getLayerWeightOpt(weights, name_buf, prefix, li, "router.proj.biases") orelse mlx.mlx_array_new(),
                    .router_scale = getLayerWeightOpt(weights, name_buf, prefix, li, "router.scale"),
                    .per_expert_scale = getLayerWeightOpt(weights, name_buf, prefix, li, "router.per_expert_scale"),
                    .switch_gate_w = sw_gate_w,
                    .switch_gate_s = sw_gate_s,
                    .switch_gate_b = sw_gate_b,
                    .switch_up_w = sw_up_w,
                    .switch_up_s = sw_up_s,
                    .switch_up_b = sw_up_b,
                    .switch_down_w = sw_down_w,
                    .switch_down_s = sw_down_s,
                    .switch_down_b = sw_down_b,
                    // Shared expert handled via lw.shared_mlp for Gemma 4 (separate branch in forward)
                    .shared_gate_w = try getLayerWeight(weights, name_buf, prefix, li, "mlp.gate_proj.weight"),
                    .shared_gate_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.gate_proj.scales") orelse mlx.mlx_array_new(),
                    .shared_gate_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.gate_proj.biases") orelse mlx.mlx_array_new(),
                    .shared_up_w = try getLayerWeight(weights, name_buf, prefix, li, "mlp.up_proj.weight"),
                    .shared_up_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.up_proj.scales") orelse mlx.mlx_array_new(),
                    .shared_up_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.up_proj.biases") orelse mlx.mlx_array_new(),
                    .shared_down_w = try getLayerWeight(weights, name_buf, prefix, li, "mlp.down_proj.weight"),
                    .shared_down_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.down_proj.scales") orelse mlx.mlx_array_new(),
                    .shared_down_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.down_proj.biases") orelse mlx.mlx_array_new(),
                },
            };
            {
                const mw = &lw.mlp.moe;
                try maybeTransposeForBf16(&mw.router_w, mw.router_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.switch_gate_w, mw.switch_gate_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.switch_up_w, mw.switch_up_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.switch_down_w, mw.switch_down_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.shared_gate_w, mw.shared_gate_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.shared_up_w, mw.shared_up_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.shared_down_w, mw.shared_down_s, &owned_bf16, allocator, s);
            }
            // Pre-fold the sigma-MoE router norm scale: at runtime the router does
            // `rms_norm(x, router_scale * hidden_size^-0.5, eps)`. Multiplying once
            // at load time saves the per-layer multiply (3 ops × num_layers).
            if (lw.mlp.moe.router_scale) |rs| {
                const root_size: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(config.hidden_size)));
                const root_scalar = bf16Scalar(root_size, s);
                defer _ = mlx.mlx_array_free(root_scalar);
                var folded = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_multiply(&folded, rs, root_scalar, s));
                try owned_bf16.append(allocator, folded);
                lw.mlp.moe.router_scale = folded;
            }
        } else if (layer_is_moe and is_mimo) {
            // MiMo binds laguna's layout: qwen3_moe WEIGHT NAMING (mlp.gate router bf16 — or the
            // HF-native mlp.gate.proj on oQ builds, resolved by probe,
            // mlp.switch_mlp.* nvfp4 experts, mlp.shared_expert.* bf16 shared)
            // with hy_v3 ROUTING SEMANTICS — sigmoid + f32 selection bias
            // (mlp.gate.e_score_correction_bias) + top-k renorm + route_scale,
            // and the shared expert ALWAYS added, no gate (shared_ungated). The
            // nvfp4 experts carry scales (u8 fp8) but no biases (bias-less mode);
            // the router/shared weights are bf16 (null-ctx scales → pre-transposed
            // to plain matmul by maybeTransposeForBf16).
            const rt = lagunaRouterBase(weights, name_buf, prefix, li);
            var rtbuf: [64]u8 = undefined;
            var exbuf: [64]u8 = undefined;
            const lf: [3][]const u8 = if (is_mimo and config.exl3 != null) .{ "trellis", "suh", "svh" } else .{ "weight", "scales", "biases" };
            lw.mlp = .{
                .moe = .{
                    .router_w = try getLayerWeight(weights, name_buf, prefix, li, moeExpertSuffix(&rtbuf, rt, "weight")),
                    .router_s = getLayerWeightOpt(weights, name_buf, prefix, li, moeExpertSuffix(&rtbuf, rt, "scales")) orelse mlx.mlx_array_new(),
                    .router_b = getLayerWeightOpt(weights, name_buf, prefix, li, moeExpertSuffix(&rtbuf, rt, "biases")) orelse mlx.mlx_array_new(),
                    .switch_gate_w = try getLayerWeight(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, "mlp.switch_mlp.gate_proj", lf[0])),
                    .switch_gate_s = getLayerWeightOpt(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, "mlp.switch_mlp.gate_proj", lf[1])) orelse mlx.mlx_array_new(),
                    .switch_gate_b = getLayerWeightOpt(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, "mlp.switch_mlp.gate_proj", lf[2])) orelse mlx.mlx_array_new(),
                    .switch_up_w = try getLayerWeight(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, "mlp.switch_mlp.up_proj", lf[0])),
                    .switch_up_s = getLayerWeightOpt(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, "mlp.switch_mlp.up_proj", lf[1])) orelse mlx.mlx_array_new(),
                    .switch_up_b = getLayerWeightOpt(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, "mlp.switch_mlp.up_proj", lf[2])) orelse mlx.mlx_array_new(),
                    .switch_down_w = try getLayerWeight(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, "mlp.switch_mlp.down_proj", lf[0])),
                    .switch_down_s = getLayerWeightOpt(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, "mlp.switch_mlp.down_proj", lf[1])) orelse mlx.mlx_array_new(),
                    .switch_down_b = getLayerWeightOpt(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, "mlp.switch_mlp.down_proj", lf[2])) orelse mlx.mlx_array_new(),
                    .shared_gate_w = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.gate_proj.weight") orelse mlx.mlx_array_new(),
                    .shared_gate_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.gate_proj.scales") orelse mlx.mlx_array_new(),
                    .shared_gate_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.gate_proj.biases") orelse mlx.mlx_array_new(),
                    .shared_up_w = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.up_proj.weight") orelse mlx.mlx_array_new(),
                    .shared_up_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.up_proj.scales") orelse mlx.mlx_array_new(),
                    .shared_up_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.up_proj.biases") orelse mlx.mlx_array_new(),
                    .shared_down_w = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.down_proj.weight") orelse mlx.mlx_array_new(),
                    .shared_down_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.down_proj.scales") orelse mlx.mlx_array_new(),
                    .shared_down_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.down_proj.biases") orelse mlx.mlx_array_new(),
                    .expert_bias = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.gate.e_score_correction_bias") orelse blk: {
                        // Some community conversions (mlx-lm quantizer) DROP the
                        // aux-loss-free bias — the reference zero-inits it, so
                        // selection reduces to plain sigmoid top-k. Non-null keeps
                        // moeMLP2 on the sigmoid chain (null falls back to softmax).
                        var zeros = mlx.mlx_array_new();
                        const zshape = [_]c_int{@intCast(config.num_experts)};
                        try mlx.check(mlx.mlx_zeros(&zeros, &zshape, 1, .float32, s));
                        try owned_bf16.append(allocator, zeros);
                        break :blk zeros;
                    },
                    .route_norm = config.moe_route_norm,
                    .route_scale = config.router_scaling_factor,
                    .shared_ungated = true,
                },
            };
            if (is_mimo) if (config.exl3) |spec| try checkExl3Bank(&lw.mlp.moe, config, spec);
            {
                const mw = &lw.mlp.moe;
                try maybeTransposeForBf16(&mw.router_w, mw.router_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.switch_gate_w, mw.switch_gate_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.switch_up_w, mw.switch_up_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.switch_down_w, mw.switch_down_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.shared_gate_w, mw.shared_gate_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.shared_up_w, mw.shared_up_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.shared_down_w, mw.shared_down_s, &owned_bf16, allocator, s);
            }
            // MiMo routes in f32 (the reference's `F.linear` on f32 inputs and weights);
            // a dense bf16 router widens exactly, once.
            if (is_mimo and lw.mlp.moe.router_s.ctx == null and mlx.mlx_array_dtype(lw.mlp.moe.router_w) != .float32) {
                var f = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_astype(&f, lw.mlp.moe.router_w, .float32, s));
                try mlx.check(mlx.mlx_array_eval(f));
                try owned_bf16.append(allocator, f);
                lw.mlp.moe.router_w = f;
            }
        } else if (layer_is_moe and is_hy3) {
            // Hy3 (hy_v3): stacked experts (already [E, out, packed] in MLX
            // conversions — never per-expert), the QUANTIZED router under
            // mlp.router.gate.* (8-bit on shipped checkpoints; scales optional
            // for a bf16 build — mlx-community oQ2e ships a bf16 router), the
            // f32 selection bias mlp.expert_bias, and an UNGATED shared expert
            // under mlp.shared_mlp.* (absent on 0-shared configs — binds empty
            // and moeMLP skips the branch). route_norm and router_scaling_factor
            // ride on the weights struct so moeMLP2 needs no config re-derivation
            // per call. The expert container name varies by converter
            // (`mlp.experts.*` in ox-ox builds, `mlp.switch_mlp.*` in mlx-lm
            // builds) — probe once and thread the resolved name through.
            const ex = hy3ExpertContainer(weights, name_buf, prefix, li);
            var exbuf: [64]u8 = undefined;
            lw.mlp = .{
                .moe = .{
                    .router_w = try getLayerWeight(weights, name_buf, prefix, li, "mlp.router.gate.weight"),
                    .router_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.router.gate.scales") orelse mlx.mlx_array_new(),
                    .router_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.router.gate.biases") orelse mlx.mlx_array_new(),
                    .switch_gate_w = try getLayerWeight(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, ex, "gate_proj.weight")),
                    .switch_gate_s = getLayerWeightOpt(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, ex, "gate_proj.scales")) orelse mlx.mlx_array_new(),
                    .switch_gate_b = getLayerWeightOpt(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, ex, "gate_proj.biases")) orelse mlx.mlx_array_new(),
                    .switch_up_w = try getLayerWeight(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, ex, "up_proj.weight")),
                    .switch_up_s = getLayerWeightOpt(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, ex, "up_proj.scales")) orelse mlx.mlx_array_new(),
                    .switch_up_b = getLayerWeightOpt(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, ex, "up_proj.biases")) orelse mlx.mlx_array_new(),
                    .switch_down_w = try getLayerWeight(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, ex, "down_proj.weight")),
                    .switch_down_s = getLayerWeightOpt(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, ex, "down_proj.scales")) orelse mlx.mlx_array_new(),
                    .switch_down_b = getLayerWeightOpt(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, ex, "down_proj.biases")) orelse mlx.mlx_array_new(),
                    .shared_gate_w = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_mlp.gate_proj.weight") orelse mlx.mlx_array_new(),
                    .shared_gate_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_mlp.gate_proj.scales") orelse mlx.mlx_array_new(),
                    .shared_gate_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_mlp.gate_proj.biases") orelse mlx.mlx_array_new(),
                    .shared_up_w = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_mlp.up_proj.weight") orelse mlx.mlx_array_new(),
                    .shared_up_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_mlp.up_proj.scales") orelse mlx.mlx_array_new(),
                    .shared_up_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_mlp.up_proj.biases") orelse mlx.mlx_array_new(),
                    .shared_down_w = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_mlp.down_proj.weight") orelse mlx.mlx_array_new(),
                    .shared_down_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_mlp.down_proj.scales") orelse mlx.mlx_array_new(),
                    .shared_down_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_mlp.down_proj.biases") orelse mlx.mlx_array_new(),
                    .expert_bias = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.expert_bias") orelse blk: {
                        // moe_router_enable_expert_bias=false checkpoints ship no
                        // bias tensor; the reference keeps zeros — selection
                        // reduces to plain sigmoid top-k. Non-null keeps moeMLP2
                        // on the sigmoid chain (null would fall back to softmax).
                        var zeros = mlx.mlx_array_new();
                        const zshape = [_]c_int{@intCast(config.num_experts)};
                        try mlx.check(mlx.mlx_zeros(&zeros, &zshape, 1, .float32, s));
                        try owned_bf16.append(allocator, zeros);
                        break :blk zeros;
                    },
                    .route_norm = config.moe_route_norm,
                    .route_scale = config.router_scaling_factor,
                    .shared_ungated = true,
                },
            };
            {
                const mw = &lw.mlp.moe;
                try maybeTransposeForBf16(&mw.router_w, mw.router_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.switch_gate_w, mw.switch_gate_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.switch_up_w, mw.switch_up_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.switch_down_w, mw.switch_down_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.shared_gate_w, mw.shared_gate_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.shared_up_w, mw.shared_up_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.shared_down_w, mw.shared_down_s, &owned_bf16, allocator, s);
            }
        } else if (layer_is_moe) {
            // Qwen3.5 MoE — also serves Qwen3-30B-A3B (`qwen3_moe`), which shares
            // this exact router/switch_mlp layout. Each `*_s`/`*_b` is loaded
            // optionally — Unsloth Dynamic checkpoints (e.g. Qwen3.6-A3B UD) leave
            // the router (`mlp.gate`) and the shared-expert gate
            // (`mlp.shared_expert_gate`) as plain bf16, with no scales/biases.
            // The shared-expert WEIGHTS themselves are also optional: qwen3_moe
            // (Qwen3-30B-A3B / Coder) dropped the shared expert entirely
            // (shared_expert_intermediate_size: 0, no mlp.shared_expert.*). When
            // absent they bind to empty handles and `shared_expert_gate_w` stays
            // null, which makes moeMLP early-return the routed-expert sum without
            // ever reading them — so no MISSING WEIGHT crash. The
            // `maybeTransposeForBf16` calls below pre-transpose bf16 weights from
            // `[out, in]` → `[in, out]` so `qmatmulBits` can dispatch to plain
            // `mlx_matmul`; they no-op on already-quantized AND on empty handles.
            // An EXL3 pack (src/sdk/quant/exl3) binds trellis/suh/svh in the same slots.
            const leaf: [3][]const u8 = if (config.exl3 != null) .{ "trellis", "suh", "svh" } else .{ "weight", "scales", "biases" };
            var exbuf: [64]u8 = undefined;
            lw.mlp = .{ .moe = .{
                .router_w = try getLayerWeight(weights, name_buf, prefix, li, "mlp.gate.weight"),
                .router_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.gate.scales") orelse mlx.mlx_array_new(),
                .router_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.gate.biases") orelse mlx.mlx_array_new(),
                .switch_gate_w = try getLayerWeight(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, "mlp.switch_mlp.gate_proj", leaf[0])),
                .switch_gate_s = getLayerWeightOpt(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, "mlp.switch_mlp.gate_proj", leaf[1])) orelse mlx.mlx_array_new(),
                .switch_gate_b = getLayerWeightOpt(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, "mlp.switch_mlp.gate_proj", leaf[2])) orelse mlx.mlx_array_new(),
                .switch_up_w = try getLayerWeight(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, "mlp.switch_mlp.up_proj", leaf[0])),
                .switch_up_s = getLayerWeightOpt(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, "mlp.switch_mlp.up_proj", leaf[1])) orelse mlx.mlx_array_new(),
                .switch_up_b = getLayerWeightOpt(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, "mlp.switch_mlp.up_proj", leaf[2])) orelse mlx.mlx_array_new(),
                .switch_down_w = try getLayerWeight(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, "mlp.switch_mlp.down_proj", leaf[0])),
                .switch_down_s = getLayerWeightOpt(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, "mlp.switch_mlp.down_proj", leaf[1])) orelse mlx.mlx_array_new(),
                .switch_down_b = getLayerWeightOpt(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, "mlp.switch_mlp.down_proj", leaf[2])) orelse mlx.mlx_array_new(),
                .shared_gate_w = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.gate_proj.weight") orelse mlx.mlx_array_new(),
                .shared_gate_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.gate_proj.scales") orelse mlx.mlx_array_new(),
                .shared_gate_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.gate_proj.biases") orelse mlx.mlx_array_new(),
                .shared_up_w = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.up_proj.weight") orelse mlx.mlx_array_new(),
                .shared_up_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.up_proj.scales") orelse mlx.mlx_array_new(),
                .shared_up_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.up_proj.biases") orelse mlx.mlx_array_new(),
                .shared_down_w = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.down_proj.weight") orelse mlx.mlx_array_new(),
                .shared_down_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.down_proj.scales") orelse mlx.mlx_array_new(),
                .shared_down_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.down_proj.biases") orelse mlx.mlx_array_new(),
                .shared_expert_gate_w = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert_gate.weight"),
                .shared_expert_gate_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert_gate.scales") orelse mlx.mlx_array_new(),
                .shared_expert_gate_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert_gate.biases") orelse mlx.mlx_array_new(),
                .per_expert_scale = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.switch_mlp.expert_scale"),
                .gate_expert_scale = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.switch_mlp.gate_proj.expert_scale") orelse .{ .ctx = null },
                .up_expert_scale = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.switch_mlp.up_proj.expert_scale") orelse .{ .ctx = null },
            } };
            if (config.exl3) |spec| try checkExl3Bank(&lw.mlp.moe, config, spec);
            {
                const mw = &lw.mlp.moe;
                try maybeTransposeForBf16(&mw.router_w, mw.router_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.switch_gate_w, mw.switch_gate_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.switch_up_w, mw.switch_up_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.switch_down_w, mw.switch_down_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.shared_gate_w, mw.shared_gate_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.shared_up_w, mw.shared_up_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.shared_down_w, mw.shared_down_s, &owned_bf16, allocator, s);
                if (mw.shared_expert_gate_w) |*seg_w_ptr| {
                    try dequantizeSmall(seg_w_ptr, &mw.shared_expert_gate_s.?, &mw.shared_expert_gate_b.?, config.hidden_size, &owned_bf16, allocator, s);
                    try maybeTransposeForBf16(seg_w_ptr, mw.shared_expert_gate_s.?, &owned_bf16, allocator, s);
                    if (mw.shared_expert_gate_s.?.ctx == null and mlx.mlx_array_size(seg_w_ptr.*) == config.hidden_size) {
                        var flat = mlx.mlx_array_new();
                        defer _ = mlx.mlx_array_free(flat);
                        try mlx.check(mlx.mlx_reshape(&flat, seg_w_ptr.*, &[_]c_int{@intCast(config.hidden_size)}, 1, s));
                        var vec = mlx.mlx_array_new();
                        try mlx.check(mlx.mlx_contiguous(&vec, flat, false, s));
                        try owned_bf16.append(allocator, vec);
                        mw.shared_gate_vec = vec;
                    }
                }
                const Bk = moe_affine4.Bank;
                mw.bank_views = try moe_affine4.bindViews(Bk{ .w = mw.switch_gate_w, .s = mw.switch_gate_s, .b = mw.switch_gate_b }, Bk{ .w = mw.switch_up_w, .s = mw.switch_up_s, .b = mw.switch_up_b }, Bk{ .w = mw.switch_down_w, .s = mw.switch_down_s, .b = mw.switch_down_b }, &owned_bf16, allocator, s);
            }
        } else {
            lw.mlp = .{ .dense = .{
                .gate_w = try getLayerWeight(weights, name_buf, prefix, li, "mlp.gate_proj.weight"),
                .gate_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.gate_proj.scales") orelse mlx.mlx_array_new(),
                .gate_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.gate_proj.biases") orelse mlx.mlx_array_new(),
                .up_w = try getLayerWeight(weights, name_buf, prefix, li, "mlp.up_proj.weight"),
                .up_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.up_proj.scales") orelse mlx.mlx_array_new(),
                .up_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.up_proj.biases") orelse mlx.mlx_array_new(),
                .down_w = try getLayerWeight(weights, name_buf, prefix, li, "mlp.down_proj.weight"),
                .down_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.down_proj.scales") orelse mlx.mlx_array_new(),
                .down_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.down_proj.biases") orelse mlx.mlx_array_new(),
            } };
            {
                const dw = &lw.mlp.dense;
                try maybeTransposeForBf16(&dw.gate_w, dw.gate_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&dw.up_w, dw.up_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&dw.down_w, dw.down_s, &owned_bf16, allocator, s);
                // Hadamard packs: `aliasFusedRotations` binds the joined buffer's signs.
                if (dw.gate_s.ctx != null) {
                    var parts = [_][3]*mlx.mlx_array{ .{ &dw.gate_w, &dw.gate_s, &dw.gate_b }, .{ &dw.up_w, &dw.up_s, &dw.up_b } };
                    const names = [_][]const u8{ "mlp.gate_proj", "mlp.up_proj" };
                    dw.gu = try fuseRowsInPlace(&parts, 0, &names, weights, name_buf, prefix, li, &owned_bf16, allocator, s);
                }
            }
        }

        ssm_entries[i] = .{
            .conv_state = mlx.mlx_array_new(),
            .ssm_state = mlx.mlx_array_new(),
            .initialized = false,
        };
    }

    // The PLE is placed by exact equality inside the loop above and nothing downstream notices
    // its absence; a silently PLE-less qwen4 emits plausible text.
    if (config.isQwen4()) {
        const has_ple = try allocator.alloc(bool, moe_layers.len);
        defer allocator.free(has_ple);
        for (moe_layers, 0..) |*lw, i| has_ple[i] = lw.ple != null;
        if (!model_mod.qwen4PleInstalledAt(has_ple, if (config.pleLocalLayer()) |l| @intCast(l) else -1)) {
            log.err("[qwen4] the PLE was not installed at layer {d} ({d} layers carry PLE weights)\n", .{ config.ple_layer_idx, std.mem.count(bool, has_ple, &.{true}) });
            return error.Qwen4PleNotInstalled;
        }
    }

    return .{
        .moe_layers = moe_layers,
        .ssm_entries = ssm_entries,
        .owned_bf16 = try owned_bf16.toOwnedSlice(allocator),
    };
}
