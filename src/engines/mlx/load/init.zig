//! engines/mlx/load/init, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns init.

const std = @import("std");
const env = @import("../../../env.zig");
const qwen4_mod = @import("../../../plugins/qwen4_exp/qwen4_exp.zig");
const distributed_mod = @import("../cluster/distributed.zig");
const ple_gpu = @import("../../../plugins/qwen4_exp/ple_gpu.zig");
const mlx = @import("../../../mlx/ffi.zig");
const rht = @import("../../../sdk/quant/hadamard/rht.zig");
const qmm_int8 = @import("../../../sdk/blocks/qmm/qmm_int8.zig");
const model_mod = @import("../model.zig");
const log = @import("../../../host/log.zig");
const ModelConfig = @import("../model.zig").ModelConfig;
const Weights = @import("../model.zig").Weights;
const fused_rows = @import("../../../sdk/blocks/linear/fused_rows.zig");
const transformer = @import("../transformer.zig");
const QWEN4_DECODE_ASYNC_LADDER_DEFAULT = @import("../../../plugins/qwen4_exp/forward.zig").QWEN4_DECODE_ASYNC_LADDER_DEFAULT;
const loadHcWeights = @import("../../../plugins/qwen4_exp/load.zig").loadHcWeights;
const Qwen4Mtp = @import("../../../plugins/qwen4_exp/mtp.zig").Qwen4Mtp;
const loadQwen4Mtp = @import("../../../plugins/qwen4_exp/mtp.zig").loadQwen4Mtp;
const qsaNaxArm = @import("../../../plugins/qwen4_exp/qsa/gather.zig").qsaNaxArm;
const checkQsaCompressRatio = @import("../../../plugins/qwen4_exp/qsa/history.zig").checkQsaCompressRatio;
const ssmFreeQsaState = @import("../../../plugins/qwen4_exp/qsa/history.zig").ssmFreeQsaState;
const qsaScoreFusedArm = @import("../../../plugins/qwen4_exp/qsa/score.zig").qsaScoreFusedArm;
const HcWeights = @import("../../../plugins/qwen4_exp/weights.zig").HcWeights;
const attnProjRouted = @import("../../../sdk/blocks/attention/decode_attn_quant.zig").attnProjRouted;
const weightsHaveDenseAttnProj = @import("../../../sdk/blocks/attention/decode_attn_quant.zig").weightsHaveDenseAttnProj;
const DenseMlpWeights = @import("../../../sdk/blocks/mlp/dense.zig").DenseMlpWeights;
const computeLlama3Freqs = @import("../../../sdk/blocks/rope/rope.zig").computeLlama3Freqs;
const computeYarnFreqs = @import("../../../sdk/blocks/rope/yarn.zig").computeYarnFreqs;
const yarnSpec = @import("../../../sdk/blocks/rope/yarn.zig").yarnSpec;
const aliasFusedRotations = @import("../../../sdk/quant/hadamard.zig").aliasFusedRotations;
const registerRhtSigns = @import("../../../sdk/quant/hadamard.zig").registerRhtSigns;
const lmHeadMxfp8On = @import("../../../sdk/quant/mxfp8.zig").lmHeadMxfp8On;
const quantizeHeadMxfp8 = @import("../../../sdk/quant/mxfp8.zig").quantizeHeadMxfp8;
const BitsCache = @import("../../../sdk/quant/params.zig").BitsCache;
const firstNonTernaryBias = @import("../../../sdk/quant/ternary.zig").firstNonTernaryBias;
const firstRowExactMiss = @import("../../../sdk/quant/ternary.zig").firstRowExactMiss;
const ternaryKernelCandidate = @import("../../../sdk/quant/ternary.zig").ternaryKernelCandidate;
const MoeLayerWeights = @import("../forward/layered.zig").MoeLayerWeights;
const KVCache = @import("../kv/cache.zig").KVCache;
const LayerWeights = @import("../layer_weights.zig").LayerWeights;
const initMlxStream = @import("../modules/mlx_stream.zig").initMlxStream;
const mtpNaxCalibratedModelFrom = @import("../spec/mtp_nax_profile.zig").mtpNaxCalibratedModelFrom;
const mtpNaxOqeAffineTrunkFrom = @import("../spec/mtp_nax_profile.zig").mtpNaxOqeAffineTrunkFrom;
const mtpNaxUniformAffineTrunkFrom = @import("../spec/mtp_nax_profile.zig").mtpNaxUniformAffineTrunkFrom;
const SSMCacheEntry = @import("../ssm/entry.zig").SSMCacheEntry;
const Transformer = @import("../transformer.zig").Transformer;
const initStandardLayers = @import("layers_dense.zig").initStandardLayers;
const initMoeLayers = @import("layers_layered.zig").initMoeLayers;
const addOne = @import("weights.zig").addOne;
const appendFullAttnWeights = @import("weights.zig").appendFullAttnWeights;
const appendHybridMlpWeights = @import("weights.zig").appendHybridMlpWeights;
const appendLinearAttnWeights = @import("weights.zig").appendLinearAttnWeights;
const appendStructArrays = @import("weights.zig").appendStructArrays;
const bf16Scalar = @import("weights.zig").bf16Scalar;
const floatDtypeTable = @import("weights.zig").floatDtypeTable;
const getNamedWeight = @import("weights.zig").getNamedWeight;
const getWeightFmt = @import("weights.zig").getWeightFmt;
const getWeightFmtOpt = @import("weights.zig").getWeightFmtOpt;
const structFields = @import("weights.zig").structFields;

pub fn init(io: std.Io, allocator: std.mem.Allocator, config: ModelConfig, weights: *Weights) !Transformer {
    // Use the current thread's default GPU stream rather than a dedicated stream.
    // mlx 0.31.2 made streams thread-local — a stream created on one thread isn't
    // visible to other threads, so a long-lived dedicated stream stored on Transformer
    // would break as soon as a different thread (e.g. an HTTP connection handler)
    // tried to use it. We re-bind `self.s` to the connection thread's default stream
    // via `useCurrentThreadStream` before each request.
    const s = mlx.gpuStream();
    if (config.indexer_budget != 0) {
        qsaScoreFusedArm();
        qsaNaxArm();
    }
    const prefix = config.weight_prefix;

    var name_buf: [256]u8 = undefined;

    // Bidirectional decoder archs (EmbeddingGemma) load through the standard arm and dispatch to
    // forwardGemma3EncoderWith. LFM2, Nemotron-H and DeepSeek-V4/V4.1's MLX packs run on their plugins only.
    if (config.has_hybrid_layers or std.mem.eql(u8, config.model_type, "deepseek_v4") or (config.isDsv41() and !config.dsv41_stream)) {
        log.err("[load] {s}: its plugin does not serve this pack and the legacy forward is gone\n", .{config.model_type});
        return error.PluginDeclinedNoLegacy;
    }
    if (config.mimo_stream) return initMlxStream(io, allocator, config, s);
    if (config.glm_stream) return initMlxStream(io, allocator, config, s);
    if (config.glm5_stream) return initMlxStream(io, allocator, config, s);
    if (config.dsv41_stream) return initMlxStream(io, allocator, config, s);

    const emb_base: []const u8 = "embed_tokens";
    const emb_w = getNamedWeight(weights, &name_buf, prefix, emb_base, "weight") orelse {
        log.err("MISSING WEIGHT: {s}.{s}.weight\n", .{ prefix, emb_base });
        return error.MissingWeight;
    };
    // Dense bf16 (quant_bits==0): no scales/biases exist → null-ctx arrays
    // signal "plain bf16" to embedding()/dequantTake().
    // Bias-less quant modes (nvfp4/mxfp4/mxfp8): scales exist, biases are
    // OPTIONAL — absent on fp8 tensors, present on affine-override tensors
    // in mixed QAT checkpoints. A mandatory fetch would be a spurious
    // MISSING WEIGHT crash (issue #24).
    // Mixed checkpoints can also leave the WHOLE embedding table dense
    // beside quantized layers (hy_v3 2-bit ships a bf16 embed_tokens) —
    // scales-presence is decided by the TABLE's dtype (float ⇒ dense),
    // never by the config's global bits. A packed (uint32) table missing
    // its scales still crashes honestly.
    const bias_mandatory = config.quant_bits > 0 and config.quant_mode.hasBiases();
    const emb_dense = floatDtypeTable(mlx.mlx_array_dtype(emb_w));
    const emb_s_arr = if (emb_dense)
        mlx.mlx_array_new()
    else
        getNamedWeight(weights, &name_buf, prefix, emb_base, "scales") orelse {
            log.err("MISSING WEIGHT: {s}.{s}.scales\n", .{ prefix, emb_base });
            return error.MissingWeight;
        };
    const emb_b_arr = if (emb_dense)
        mlx.mlx_array_new()
    else
        getNamedWeight(weights, &name_buf, prefix, emb_base, "biases") orelse blk: {
            if (bias_mandatory) {
                log.err("MISSING WEIGHT: {s}.{s}.biases\n", .{ prefix, emb_base });
                return error.MissingWeight;
            }
            break :blk mlx.mlx_array_new();
        };

    const emb_scale: ?mlx.mlx_array = if (config.scale_embeddings)
        bf16Scalar(@sqrt(@as(f32, @floatFromInt(config.hidden_size))), s)
    else
        null;

    var final_norm: mlx.mlx_array = undefined;
    if (!config.has_final_norm) {
        final_norm = mlx.mlx_array_new(); // placeholder, unused
    } else {
        const final_norm_raw = try getWeightFmt(weights, &name_buf, "{s}.norm.weight", prefix);
        if (config.norm_has_offset) {
            final_norm = try addOne(final_norm_raw, s);
            try mlx.check(mlx.mlx_array_eval(final_norm));
        } else {
            final_norm = final_norm_raw;
        }
    }

    var lm_head_w: mlx.mlx_array = undefined;
    var lm_head_s: mlx.mlx_array = undefined;
    var lm_head_b: mlx.mlx_array = undefined;
    var owns_lm_head = false;
    // The head's own map name, when it has one (not the embedding's).
    var head_key_buf: [160]u8 = undefined;
    var head_key: ?[]const u8 = null;

    {
        // A nested decoder's head is a sibling of model/backbone.
        const lm_prefix = if (std.mem.eql(u8, prefix, "language_model.model"))
            "language_model"
        else
            prefix;
        const maybe_lm_w = getWeightFmtOpt(weights, &name_buf, "{s}.lm_head.weight", lm_prefix);
        if (maybe_lm_w) |w| {
            lm_head_w = w;
            // Dense bf16: no scales/biases → null-ctx; lmHeadProject() then
            // projects via a transposed view of the [vocab, hidden] weight.
            // Per-TENSOR dense detection (float dtype), same as embed_tokens
            // above — mixed checkpoints may quantize layers but not the head.
            const head_dense = floatDtypeTable(mlx.mlx_array_dtype(w));
            lm_head_s = if (head_dense)
                mlx.mlx_array_new()
            else
                try getWeightFmt(weights, &name_buf, "{s}.lm_head.scales", lm_prefix);
            lm_head_b = if (head_dense)
                mlx.mlx_array_new()
            else if (bias_mandatory)
                try getWeightFmt(weights, &name_buf, "{s}.lm_head.biases", lm_prefix)
            else
                getWeightFmtOpt(weights, &name_buf, "{s}.lm_head.biases", lm_prefix) orelse mlx.mlx_array_new();
            owns_lm_head = !config.tie_word_embeddings;
            head_key = std.fmt.bufPrint(&head_key_buf, "{s}.lm_head.weight", .{lm_prefix}) catch null;
        } else if (weights.get("lm_head.weight")) |w| {
            lm_head_w = w;
            head_key = "lm_head.weight";
            lm_head_s = weights.get("lm_head.scales") orelse emb_s_arr;
            lm_head_b = weights.get("lm_head.biases") orelse emb_b_arr;
            owns_lm_head = !config.tie_word_embeddings;
        } else if (config.tie_word_embeddings) {
            lm_head_w = emb_w;
            lm_head_s = emb_s_arr;
            lm_head_b = emb_b_arr;
        } else {
            log.err("MISSING WEIGHT: lm_head.weight\n", .{});
            return error.MissingWeight;
        }
    }
    if (lmHeadMxfp8On(config.lm_head_mxfp8_override, transformer.cli().lm_head_mxfp8_cli, mlx.cudaIntegrated(), config.isQwen4(), config.quant_bits)) {
        if (head_key != null and lm_head_s.ctx == null) {
            lm_head_s = try quantizeHeadMxfp8(weights, head_key.?, lm_head_w, s);
            lm_head_w = weights.get(head_key.?).?;
            const shape = mlx.getShape(lm_head_s);
            log.info("[lm-head] mxfp8 (lossy): the {d}x{d} bf16 head quantized at load\n", .{ shape[0], shape[1] * 32 });
        } else if ((config.lm_head_mxfp8_override orelse transformer.cli().lm_head_mxfp8_cli) != null)
            log.info("[lm-head] mxfp8 declined: the head is {s}\n", .{if (lm_head_s.ctx != null) "already quantized" else "the embedding"});
    }

    // EmbeddingGemma sentence-transformers projection head: folded into
    // the main safetensors as root-level dense.0/dense.1 by the
    // mlx-community conversion. Optional — absent on every other model.
    const dense0_w: mlx.mlx_array = weights.get("dense.0.weight") orelse .{};
    const dense0_s: mlx.mlx_array = weights.get("dense.0.scales") orelse .{};
    const dense0_b: mlx.mlx_array = weights.get("dense.0.biases") orelse .{};
    const dense1_w: mlx.mlx_array = weights.get("dense.1.weight") orelse .{};
    const dense1_s: mlx.mlx_array = weights.get("dense.1.scales") orelse .{};
    const dense1_b: mlx.mlx_array = weights.get("dense.1.biases") orelse .{};

    // Cache for KV (standard models use all entries, MoE only uses full-attn layers)
    const cache = try KVCache.init(allocator, config.num_hidden_layers);

    const need_gelu = config.hidden_act == .gelu_approx;
    const need_silu = config.hidden_act == .silu;

    // Load architecture-specific layer weights
    var layers: []LayerWeights = &.{};
    var moe_layers: ?[]MoeLayerWeights = null;
    var ssm_entries: ?[]SSMCacheEntry = null;
    var moe_owned_bf16: ?[]mlx.mlx_array = null;

    if (config.isMoe() or config.full_attention_interval > 0) {
        const ml = try initMoeLayers(allocator, config, weights, &name_buf, s);
        moe_layers = ml.moe_layers;
        moe_owned_bf16 = ml.owned_bf16;
        // ssm_entries are only meaningful when the family actually has
        // linear-attention (GDN) layers — full_attention_interval > 0.
        // A pure-attention MoE (qwen3_moe, Gemma 4 MoE) carrying non-null
        // ssm_entries makes the hot prefix cache classify the model as
        // hybrid and force a cold prefill on EVERY request: no checkpoint
        // can ever exist, so every lookup is a "hybrid miss" (caught live
        // by llmprobe cache-hit-reported on Qwen3-Coder, 2026-06-10).
        // Predicate shared with the scheduler's per-slot allocation.
        if (config.needsSsmEntries()) {
            ssm_entries = ml.ssm_entries;
        } else {
            for (ml.ssm_entries) |*e| {
                _ = mlx.mlx_array_free(e.conv_state);
                _ = mlx.mlx_array_free(e.ssm_state);
                // Was written twice: the `.ctx != null` guard stays true after a free.
                ssmFreeQsaState(e);
            }
            allocator.free(ml.ssm_entries);
        }
    } else {
        const sl = try initStandardLayers(allocator, config, weights, &name_buf, s);
        layers = sl.layers;
        moe_owned_bf16 = sl.owned_bf16; // reuse the same deinit-tracked owned list
    }
    if (fused_rows.fused_row_groups > 0) log.info("[load] row-joined projection groups: {d}\n", .{fused_rows.fused_row_groups});
    fused_rows.fused_row_groups = 0;

    const bits_cache: BitsCache = .{};

    // LFM2: load embedding norm
    var embedding_norm_w: ?mlx.mlx_array = null;
    if (config.has_embedding_norm) {
        embedding_norm_w = getWeightFmtOpt(weights, &name_buf, "{s}.embedding_norm.weight", prefix);
    }

    // Gemma 4: load PLE global weights
    var ple_emb_w = mlx.mlx_array_new();
    var ple_emb_s = mlx.mlx_array_new();
    var ple_emb_b = mlx.mlx_array_new();
    var ple_proj_w_g = mlx.mlx_array_new();
    var ple_proj_s_g = mlx.mlx_array_new();
    var ple_proj_b_g = mlx.mlx_array_new();
    var ple_proj_norm = mlx.mlx_array_new();
    var ple_proj_quantized = false;
    if (config.hidden_size_per_layer_input > 0) {
        ple_emb_w = try getWeightFmt(weights, &name_buf, "{s}.embed_tokens_per_layer.weight", prefix);
        // Dense bf16: no scales/biases → null-ctx; dequantTake takes its dense path.
        ple_emb_s = if (config.quant_bits == 0) mlx.mlx_array_new() else try getWeightFmt(weights, &name_buf, "{s}.embed_tokens_per_layer.scales", prefix);
        ple_emb_b = if (config.quant_bits == 0)
            mlx.mlx_array_new()
        else if (bias_mandatory)
            try getWeightFmt(weights, &name_buf, "{s}.embed_tokens_per_layer.biases", prefix)
        else
            getWeightFmtOpt(weights, &name_buf, "{s}.embed_tokens_per_layer.biases", prefix) orelse mlx.mlx_array_new();
        ple_proj_w_g = try getWeightFmt(weights, &name_buf, "{s}.per_layer_model_projection.weight", prefix);
        // per_layer_model_projection may be unquantized (no scales/biases)
        if (getWeightFmtOpt(weights, &name_buf, "{s}.per_layer_model_projection.scales", prefix)) |sc| {
            ple_proj_s_g = sc;
            ple_proj_b_g = if (bias_mandatory)
                try getWeightFmt(weights, &name_buf, "{s}.per_layer_model_projection.biases", prefix)
            else
                getWeightFmtOpt(weights, &name_buf, "{s}.per_layer_model_projection.biases", prefix) orelse mlx.mlx_array_new();
            ple_proj_quantized = true;
        }
        ple_proj_norm = try getWeightFmt(weights, &name_buf, "{s}.per_layer_projection_norm.weight", prefix);
    }

    // Gemma 4: logit softcapping scalar
    var softcap_scalar: ?mlx.mlx_array = null;
    if (config.final_logit_softcapping > 0) {
        softcap_scalar = bf16Scalar(config.final_logit_softcapping, s);
    }

    // Gemma 4: v_norm weights (parameter-free: ones vectors)
    var v_norm_weight: ?mlx.mlx_array = null;
    var v_norm_weight_global: ?mlx.mlx_array = null;
    if (config.has_v_norm) {
        const one_val = bf16Scalar(1.0, s);
        defer _ = mlx.mlx_array_free(one_val);
        const hd_shape = [_]c_int{@intCast(config.head_dim)};
        v_norm_weight = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_full(&v_norm_weight.?, &hd_shape, 1, one_val, .bfloat16, s));
        if (config.global_head_dim > 0 and config.global_head_dim != config.head_dim) {
            const ghd_shape = [_]c_int{@intCast(config.global_head_dim)};
            v_norm_weight_global = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_full(&v_norm_weight_global.?, &ghd_shape, 1, one_val, .bfloat16, s));
        }
    }

    var ones_hidden: ?mlx.mlx_array = null;
    if (config.isQwen4()) {
        const one_val = bf16Scalar(1.0, s);
        defer _ = mlx.mlx_array_free(one_val);
        const h_shape = [_]c_int{@intCast(config.hidden_size)};
        ones_hidden = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_full(&ones_hidden.?, &h_shape, 1, one_val, .bfloat16, s));
    }

    // Gemma 4: proportional RoPE frequencies for global/full attention layers
    // freqs = factor * base^(arange(0, rotated_dims, 2) / full_dims)
    // padded with inf for non-rotated dimensions
    var rope_freqs_global: ?mlx.mlx_array = null;
    if (config.rope_proportional) {
        const ghd: u32 = if (config.global_head_dim > 0) config.global_head_dim else config.head_dim;
        const rotated_dims: u32 = @intFromFloat(@as(f32, @floatFromInt(ghd)) * config.partial_rotary_factor_global);
        const n_rotated: u32 = rotated_dims / 2;
        const n_pad: u32 = (ghd - rotated_dims) / 2;
        const total: u32 = n_rotated + n_pad;

        const freq_shape = [_]c_int{@intCast(total)};
        var freqs_arr = mlx.mlx_array_new();

        // Compute rotated part: factor * base^(arange(0, rotated_dims, 2) / ghd)
        var arange_arr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(arange_arr);
        try mlx.check(mlx.mlx_arange(&arange_arr, 0, @floatFromInt(rotated_dims), 2, .float32, s));

        const ghd_scalar = mlx.mlx_array_new_float(@floatFromInt(ghd));
        defer _ = mlx.mlx_array_free(ghd_scalar);
        var exponents = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(exponents);
        try mlx.check(mlx.mlx_divide(&exponents, arange_arr, ghd_scalar, s));

        const base_scalar = mlx.mlx_array_new_float(config.rope_theta);
        defer _ = mlx.mlx_array_free(base_scalar);
        var base_pow = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(base_pow);
        try mlx.check(mlx.mlx_power(&base_pow, base_scalar, exponents, s));

        if (config.rope_proportional_factor != 1.0) {
            const factor_scalar = mlx.mlx_array_new_float(config.rope_proportional_factor);
            defer _ = mlx.mlx_array_free(factor_scalar);
            var scaled = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_multiply(&scaled, base_pow, factor_scalar, s));
            _ = mlx.mlx_array_free(base_pow);
            base_pow = scaled;
        }

        if (n_pad > 0) {
            // Pad with inf for non-rotated dims
            const pad_shape = [_]c_int{@intCast(n_pad)};
            const inf_val = mlx.mlx_array_new_float(std.math.inf(f32));
            defer _ = mlx.mlx_array_free(inf_val);
            var inf_arr = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(inf_arr);
            try mlx.check(mlx.mlx_full(&inf_arr, &pad_shape, 1, inf_val, .float32, s));
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            _ = mlx.mlx_vector_array_append_value(vec, base_pow);
            _ = mlx.mlx_vector_array_append_value(vec, inf_arr);
            try mlx.check(mlx.mlx_concatenate_axis(&freqs_arr, vec, 0, s));
        } else {
            try mlx.check(mlx.mlx_reshape(&freqs_arr, base_pow, &freq_shape, 1, s));
        }
        rope_freqs_global = freqs_arr;
    } else if (config.rope_llama3) |l3| {
        const half: usize = config.head_dim / 2;
        const freqs_f64 = try allocator.alloc(f64, half);
        defer allocator.free(freqs_f64);
        computeLlama3Freqs(freqs_f64, config.head_dim, config.rope_theta, l3);
        const freqs_f32 = try allocator.alloc(f32, half);
        defer allocator.free(freqs_f32);
        for (freqs_f64, freqs_f32) |v, *o| o.* = @floatCast(v);
        const fshape = [_]c_int{@intCast(half)};
        rope_freqs_global = mlx.mlx_array_new_data(freqs_f32.ptr, &fshape, 1, .float32);
    }

    // YaRN (laguna: full-attention layers only; qwen4_exp: the trunk's one
    // rope). Precompute the mlx_fast_rope denominator array, the mscale
    // vector (attention_factor on the rotated dims, 1.0 on the pass-through
    // tail) and the HOST inverse frequencies the M-RoPE tables and the QSA
    // pooled-key ladder are filled from. Sliding layers use default RoPE.
    var rope_freqs_yarn: ?mlx.mlx_array = null;
    var yarn_mscale: ?mlx.mlx_array = null;
    var yarn_inv_freq: ?[]const f64 = null;
    if (config.rope_yarn) {
        const spec = yarnSpec(&config);
        const rotary_dim: u32 = config.yarnRotaryDim();
        const half: usize = rotary_dim / 2;
        const freqs_f64 = try allocator.alloc(f64, half);
        defer allocator.free(freqs_f64);
        computeYarnFreqs(
            freqs_f64,
            config.head_dim,
            config.yarnPartial(),
            config.rope_theta,
            config.yarn_factor,
            config.yarn_beta_fast,
            config.yarn_beta_slow,
            config.yarn_orig_max_pos,
            config.yarn_truncate,
        );
        // The same table without the reciprocal: `fillCosSin` and
        // `ropeAtPositions` multiply positions by inv_freq (host f64), while
        // mlx_fast_rope divides by freqs — two spellings, one spectrum.
        const inv_f64 = try allocator.alloc(f64, half);
        errdefer allocator.free(inv_f64);
        spec.invFreq(inv_f64);
        yarn_inv_freq = inv_f64;
        const freqs_f32 = try allocator.alloc(f32, half);
        defer allocator.free(freqs_f32);
        for (freqs_f64, 0..) |v, i| freqs_f32[i] = @floatCast(v);
        const fshape = [_]c_int{@intCast(half)};
        rope_freqs_yarn = mlx.mlx_array_new_data(freqs_f32.ptr, &fshape, 1, .float32);
        // mscale vector [head_dim]: af on rotary dims, 1.0 on the tail.
        // Multiplying the post-rope q/k by this is exactly the reference's
        // cos/sin *= attention_factor (rotated dims only, pass dims untouched).
        const mscale_f32 = try allocator.alloc(f32, config.head_dim);
        defer allocator.free(mscale_f32);
        for (mscale_f32, 0..) |*m, i| m.* = if (i < rotary_dim) config.yarn_attention_factor else 1.0;
        const mshape = [_]c_int{@intCast(config.head_dim)};
        yarn_mscale = mlx.mlx_array_new_data(mscale_f32.ptr, &mshape, 1, .float32);
        const eval_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(eval_vec);
        _ = mlx.mlx_vector_array_append_value(eval_vec, rope_freqs_yarn.?);
        _ = mlx.mlx_vector_array_append_value(eval_vec, yarn_mscale.?);
        try mlx.check(mlx.mlx_eval(eval_vec));
        const cr = spec.correctionRange();
        log.info("[rope] YaRN: factor {d:.4} (window {d} -> {d}), rotary {d} dims, ramp [{d:.0}, {d:.0}], mscale {d:.6}\n", .{
            config.yarn_factor,
            config.yarn_orig_max_pos,
            config.contextCap(),
            rotary_dim,
            cr.low,
            cr.high,
            config.yarn_attention_factor,
        });
    }

    // Batch eval all weights
    {
        const eval_start = std.Io.Timestamp.now(io, .awake);
        const all_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(all_vec);

        _ = mlx.mlx_vector_array_append_value(all_vec, emb_w);
        // Dense bf16 models have null-ctx embedding/lm_head scales/biases —
        // appending a null array aborts in mlx (vector.cpp "non-empty" guard).
        if (emb_s_arr.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, emb_s_arr);
        if (emb_b_arr.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, emb_b_arr);
        _ = mlx.mlx_vector_array_append_value(all_vec, lm_head_w);
        if (lm_head_s.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lm_head_s);
        if (lm_head_b.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lm_head_b);
        if (dense0_w.ctx != null) {
            _ = mlx.mlx_vector_array_append_value(all_vec, dense0_w);
            if (dense1_w.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, dense1_w);
            if (dense0_s.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, dense0_s);
            if (dense0_b.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, dense0_b);
            if (dense1_s.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, dense1_s);
            if (dense1_b.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, dense1_b);
        }

        if (moe_layers) |ml| {
            for (ml) |lw| {
                // qwen4_exp has no per-layer norms (null-ctx); its
                // hyper-connection + PLE arrays are appended below.
                if (lw.input_norm.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.input_norm);
                if (lw.post_attn_norm.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.post_attn_norm);
                if (lw.hc_attn) |*hw| appendStructArrays(all_vec, hw);
                if (lw.hc_mlp) |*hw| appendStructArrays(all_vec, hw);
                if (lw.ple) |*pw| appendStructArrays(all_vec, pw);
                if (lw.pre_ff_norm) |n| _ = mlx.mlx_vector_array_append_value(all_vec, n);
                if (lw.post_ff_norm) |n| _ = mlx.mlx_vector_array_append_value(all_vec, n);
                if (lw.pre_ff_norm_2) |n| _ = mlx.mlx_vector_array_append_value(all_vec, n);
                if (lw.post_ff_norm_1) |n| _ = mlx.mlx_vector_array_append_value(all_vec, n);
                if (lw.post_ff_norm_2) |n| _ = mlx.mlx_vector_array_append_value(all_vec, n);
                if (lw.layer_scalar) |n| _ = mlx.mlx_vector_array_append_value(all_vec, n);
                appendHybridMlpWeights(all_vec, &lw.mlp);
                if (lw.shared_mlp) |smlp| {
                    inline for (comptime structFields(DenseMlpWeights)) |field| {
                        if (field.type == mlx.mlx_array) {
                            const arr = @field(smlp, field.name);
                            if (arr.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, arr);
                        }
                    }
                }
                switch (lw.attn) {
                    .full => |fa| appendFullAttnWeights(all_vec, &fa),
                    .linear => |la| appendLinearAttnWeights(all_vec, &la),
                }
            }
        } else {
            for (layers) |lw| {
                _ = mlx.mlx_vector_array_append_value(all_vec, lw.input_norm);
                _ = mlx.mlx_vector_array_append_value(all_vec, lw.post_attn_norm);
                if (lw.pre_ff_norm) |n| _ = mlx.mlx_vector_array_append_value(all_vec, n);
                if (lw.post_ff_norm) |n| _ = mlx.mlx_vector_array_append_value(all_vec, n);
                if (lw.q_norm) |n| _ = mlx.mlx_vector_array_append_value(all_vec, n);
                if (lw.k_norm) |n| _ = mlx.mlx_vector_array_append_value(all_vec, n);
                // Dense bf16 layers carry null-ctx scales/biases — skip them so
                // the eval batch doesn't get a null array (mlx aborts on append).
                _ = mlx.mlx_vector_array_append_value(all_vec, lw.q_w);
                if (lw.q_s.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.q_s);
                if (lw.q_b.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.q_b);
                if (lw.k_w.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.k_w);
                if (lw.k_s.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.k_s);
                if (lw.k_b.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.k_b);
                if (lw.v_w.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.v_w);
                if (lw.v_s.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.v_s);
                if (lw.v_b.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.v_b);
                // Additive qkv biases (Qwen2) — empty-ctx for archs without them.
                if (lw.q_bias.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.q_bias);
                if (lw.k_bias.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.k_bias);
                if (lw.v_bias.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.v_bias);
                _ = mlx.mlx_vector_array_append_value(all_vec, lw.o_w);
                if (lw.o_s.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.o_s);
                if (lw.o_b.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.o_b);
                _ = mlx.mlx_vector_array_append_value(all_vec, lw.gate_w);
                if (lw.gate_s.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.gate_s);
                if (lw.gate_b.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.gate_b);
                _ = mlx.mlx_vector_array_append_value(all_vec, lw.up_w);
                if (lw.up_s.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.up_s);
                if (lw.up_b.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.up_b);
                _ = mlx.mlx_vector_array_append_value(all_vec, lw.down_w);
                if (lw.down_s.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.down_s);
                if (lw.down_b.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.down_b);
                if (lw.layer_scalar) |ls| _ = mlx.mlx_vector_array_append_value(all_vec, ls);
                if (lw.ple_gate_w) |w| _ = mlx.mlx_vector_array_append_value(all_vec, w);
                if (lw.ple_gate_s) |sc| if (sc.ctx != null) {
                    _ = mlx.mlx_vector_array_append_value(all_vec, sc);
                };
                if (lw.ple_gate_b) |bi| if (bi.ctx != null) {
                    _ = mlx.mlx_vector_array_append_value(all_vec, bi);
                };
                if (lw.ple_proj_w) |w| _ = mlx.mlx_vector_array_append_value(all_vec, w);
                if (lw.ple_proj_s) |sc| if (sc.ctx != null) {
                    _ = mlx.mlx_vector_array_append_value(all_vec, sc);
                };
                if (lw.ple_proj_b) |bi| if (bi.ctx != null) {
                    _ = mlx.mlx_vector_array_append_value(all_vec, bi);
                };
                if (lw.ple_norm) |n| _ = mlx.mlx_vector_array_append_value(all_vec, n);
            }
        }

        try mlx.check(mlx.mlx_eval(all_vec));
        const eval_ms: i64 = @intCast(@divTrunc(eval_start.untilNow(io, .awake).nanoseconds, std.time.ns_per_ms));
        log.info("Batch eval all weights: {d}ms\n", .{eval_ms});
    }

    // qwen4_exp: the final hyper-connection read (replaces model.norm)
    // and the module-owned n-gram hash + mmapped table.
    var qwen4_state: ?*qwen4_mod.Qwen4State = null;
    var qwen4_mixer: ?HcWeights = null;
    var qwen4_mtp: ?Qwen4Mtp = null;
    if (config.isQwen4()) {
        var extra: std.ArrayList(mlx.mlx_array) = .empty;
        defer extra.deinit(allocator);
        var mb: [160]u8 = undefined;
        qwen4_mixer = try loadHcWeights(weights, std.fmt.bufPrint(&mb, "{s}.hyper_connection_mixer", .{prefix}) catch unreachable, false, config.hc_count, config.hidden_size, &extra, allocator, s);
        if (extra.items.len > 0) {
            const old = moe_owned_bf16 orelse &[_]mlx.mlx_array{};
            const merged = try std.mem.concat(allocator, mlx.mlx_array, &.{ old, extra.items });
            if (moe_owned_bf16) |o| allocator.free(o);
            moe_owned_bf16 = merged;
        }
        const eos: u32 = config.ngram_eos;
        // The `0` is the PLE's ordinal among the config's injection points, not
        // `config.ple_layer_idx` (pinned by the oracle fixture with `ple_layer_ids=[2]`).
        try checkQsaCompressRatio(config.indexer_compress_ratio);
        const st = try allocator.create(qwen4_mod.Qwen4State);
        errdefer allocator.destroy(st);
        st.* = .{
            .hash = try qwen4_mod.NgramHash.init(config.vocab_size, config.ngram_size, config.heads_per_ngram, config.ngram_vocab_base, config.ngram_vocab_divisor, config.ngram_seed, 0, eos),
            .table = if (config.embedded_ple_payload_bytes != null)
                try qwen4_mod.NgramTable.openEmbedded(std.fs.path.dirname(config.ngram_table_path orelse return error.MissingNgramTable) orelse return error.MissingNgramTable, try model_mod.qwen4EmbeddedSpec(&config))
            else
                try qwen4_mod.NgramTable.open(config.ngram_table_path orelse return error.MissingNgramTable),
        };
        errdefer st.deinit();
        if (config.embedded_ple_payload_bytes) |bytes| {
            if (st.table.embeddedPayloadBytes() != bytes) return error.EmbeddedNgramTableChanged;
        }
        if (st.table.rows != st.hash.total_rows or st.table.dim * st.hash.n_heads != config.ple_embed_dim) {
            log.err("[qwen4] ngram_table.bin geometry {d}x{d} does not match the config ({d} rows, {d} heads x dim)\n", .{ st.table.rows, st.table.dim, st.hash.total_rows, st.hash.n_heads });
            return error.NgramTableMismatch;
        }
        var weights_bytes: usize = 0;
        _ = mlx.mlx_get_active_memory(&weights_bytes);
        // A pipeline rank without the PLE layer, or a tensor worker, never reads the table.
        if (config.pleLocalLayer() != null and !(if (config.dist_shard) |sh| sh.rank != 0 else false)) {
            st.gpu = ple_gpu.load(&st.table, ple_gpu.pleGpu().enabled, weights_bytes);
            if (st.gpu == null) st.table.calibrateArm();
            // The weights load just evicted the table from page cache; an embedded pack's GPU
            // arm holds its own copy.
            if (st.gpu == null or st.table.embedded == null) st.table.startWarm();
        }
        qwen4_state = st;
        // A pipeline split's head would read the last layer's stream, which lives on the worker.
        qwen4_mtp = if (config.dist_span != null) null else try loadQwen4Mtp(allocator, config, weights, &name_buf, s);
        log.info("[qwen4] n-gram table {d} rows x {d} ({d}-bit, mmapped), PLE at layer {d}, QSA budget {d}/{d}\n", .{ st.table.rows, st.table.dim, st.table.bits, config.ple_layer_idx, config.indexer_budget, config.indexer_compress_ratio });
    }

    const int8_on = (config.int8_prefill_override orelse false) and mlx.metalKernelsAvailable();
    var int8_cache: ?*qmm_int8.Cache = null;
    errdefer if (int8_cache) |c| allocator.destroy(c);
    if (int8_on) {
        int8_cache = try allocator.create(qmm_int8.Cache);
        int8_cache.?.* = .{};
    }
    var rht_registry: ?*rht.Registry = null;
    errdefer if (rht_registry) |reg| {
        reg.deinit();
        allocator.destroy(reg);
    };
    if (config.hadamard_block > 0) {
        const reg = try allocator.create(rht.Registry);
        reg.* = rht.Registry.init(allocator, config.hadamard_block);
        rht_registry = reg;
        try registerRhtSigns(reg, weights, s);
        if (moe_layers) |ml| try aliasFusedRotations(reg, ml);
    }
    var ternary_2bit = false;
    if (ternaryKernelCandidate(&config)) {
        // An embedding that is not also the lm_head is only ever gathered.
        const gather_only = if (lm_head_b.ctx == emb_b_arr.ctx) mlx.mlx_array{ .ctx = null } else emb_b_arr;
        if (try firstNonTernaryBias(weights, gather_only, s)) |key| {
            log.info("[qmv2] 2-bit pack is not ternary ({s}): stock qmm\n", .{key});
        } else {
            ternary_2bit = true;
            log.info("[qmv2] ternary 2-bit pack (bias == -scale): ternary kernels armed\n", .{});
        }
    }

    const profile_head_shape = mlx.getShape(lm_head_w);
    const profile_head_n: c_int = if (profile_head_shape.len == 2) profile_head_shape[0] else 0;
    const uniform_profile_bits = config.quant_bits == 4 or config.quant_bits == 6 or config.quant_bits == 8;
    const mtp_uniform_affine_trunk = mtpNaxCalibratedModelFrom(&config, profile_head_n) and
        uniform_profile_bits and
        config.quant_group_size == 64 and
        config.quant_mode == .affine and
        mtpNaxUniformAffineTrunkFrom(&config, moe_layers);
    const mtp_oqe_affine_trunk = mtpNaxCalibratedModelFrom(&config, profile_head_n) and
        config.quant_bits == 4 and
        config.quant_group_size == 64 and
        config.quant_mode == .affine and
        mtpNaxOqeAffineTrunkFrom(&config, moe_layers);
    var cfg = config;
    if (cfg.rowExactArch()) {
        const gather_only = if (lm_head_w.ctx == emb_w.ctx) mlx.mlx_array{ .ctx = null } else emb_w;
        if (firstRowExactMiss(&cfg, weights, gather_only)) |key| {
            log.info("[row-exact] {s} is outside the row kernels: verify rows are not byte-exact\n", .{key});
        } else cfg.row_exact_covered = true;
    }

    return .{
        .config = cfg,
        .cache = cache,
        .s = s,
        .allocator = allocator,
        .dense_attn_proj = attnProjRouted(&config, moe_layers != null or qwen4_state != null) and weightsHaveDenseAttnProj(weights),
        .emb_w = emb_w,
        .emb_s = emb_s_arr,
        .emb_b = emb_b_arr,
        .emb_scale = emb_scale,
        .final_norm = final_norm,
        .qwen4 = qwen4_state,
        .dist_peer = if (config.dist_span) |sp| if (sp.first()) distributed_mod.peer() else null else null,
        .dist_tp = if (config.dist_shard) |sh| if (sh.rank == 0) distributed_mod.peer() else null else null,
        .qwen4_mixer = qwen4_mixer,
        .qwen4_mtp = qwen4_mtp,
        .lm_head_w = lm_head_w,
        .lm_head_s = lm_head_s,
        .lm_head_b = lm_head_b,
        .dense0_w = dense0_w,
        .dense0_s = dense0_s,
        .dense0_b = dense0_b,
        .dense1_w = dense1_w,
        .dense1_s = dense1_s,
        .dense1_b = dense1_b,
        .layers = layers,
        .owns_lm_head = owns_lm_head,
        .owns_norms = config.norm_has_offset,
        .decode_async_ladder_qwen4 = QWEN4_DECODE_ASYNC_LADDER_DEFAULT,
        .gelu_coeff = if (need_gelu) bf16Scalar(0.7978845608028654, s) else null,
        .gelu_inner = if (need_gelu) bf16Scalar(0.044715, s) else null,
        .half = bf16Scalar(0.5, s),
        .one = bf16Scalar(1.0, s),
        .rms_eps_arr = mlx.mlx_array_new_float(config.rms_norm_eps),
        .post_eps_arr = mlx.mlx_array_new_float(config.postNormEps()),
        .three = if (need_gelu) bf16Scalar(3.0, s) else null,
        .neg_one = if (need_silu) bf16Scalar(-1.0, s) else null,
        .ple_emb_w = ple_emb_w,
        .ple_emb_s = ple_emb_s,
        .ple_emb_b = ple_emb_b,
        .ple_proj_w = ple_proj_w_g,
        .ple_proj_s = ple_proj_s_g,
        .ple_proj_b = ple_proj_b_g,
        .ple_proj_norm = ple_proj_norm,
        .ple_proj_quantized = ple_proj_quantized,
        .softcap_scalar = softcap_scalar,
        .v_norm_weight = v_norm_weight,
        .v_norm_weight_global = v_norm_weight_global,
        .ones_hidden = ones_hidden,
        .rope_freqs_global = rope_freqs_global,
        .rope_freqs_yarn = rope_freqs_yarn,
        .yarn_mscale = yarn_mscale,
        .yarn_inv_freq = yarn_inv_freq,
        .moe_layers = moe_layers,
        .ssm_entries = ssm_entries,
        .moe_seq_offset = 0,
        .moe_owned_bf16 = moe_owned_bf16,
        .rht = rht_registry,
        .ternary_2bit = ternary_2bit,
        .int8_cache = int8_cache,
        .embedding_norm = embedding_norm_w,
        .prompt_cache = null,
        .bits_cache = bits_cache,
        .mtp_uniform_affine_trunk = mtp_uniform_affine_trunk,
        .mtp_oqe_affine_trunk = mtp_oqe_affine_trunk,
    };
}
