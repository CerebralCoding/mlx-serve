//! engines/mlx/spec/mtp_nax_profile, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns MtpNaxProfileInputs, mtpNaxBf16EmbeddingMatchesFrom, mtpNaxBf16EmbeddingMatches, mtpNaxCalibratedModelFrom, mtpNaxProfileEnabledFrom, mtpNaxAffineProjectionMatchesQuant, and 20 more.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ModelConfig = @import("../model.zig").ModelConfig;
const mixedNaxShapeEnabled = @import("../../../sdk/blocks/linear/verify_qmm.zig").mixedNaxShapeEnabled;
const verifyQmmEnabled = @import("../../../sdk/blocks/linear/verify_qmm.zig").verifyQmmEnabled;
const naxLaneEnvEnabled = @import("../../../sdk/blocks/linear/verify_qmm_nax.zig").naxLaneEnvEnabled;
const naxMinMForBits = @import("../../../sdk/blocks/linear/verify_qmm_nax.zig").naxMinMForBits;
const naxMixedBitsEnvEnabled = @import("../../../sdk/blocks/linear/verify_qmm_nax.zig").naxMixedBitsEnvEnabled;
const verifyQmmNaxAvailable = @import("../../../sdk/blocks/linear/verify_qmm_nax.zig").verifyQmmNaxAvailable;
const verifyQmmNaxEnabledForMFrom = @import("../../../sdk/blocks/linear/verify_qmm_nax.zig").verifyQmmNaxEnabledForMFrom;
const DenseMlpWeights = @import("../../../sdk/blocks/mlp/dense.zig").DenseMlpWeights;
const QuantParams = @import("../../../sdk/quant/params.zig").QuantParams;
const affineParamsFromGeometry = @import("../../../sdk/quant/params.zig").affineParamsFromGeometry;
const MoeLayerWeights = @import("../forward/layered.zig").MoeLayerWeights;
const testing = @import("../kv/cache.zig").testing;
const Transformer = @import("../transformer.zig").Transformer;

/// Pure inputs for the calibrated M5 MTP cost profile. The profile was
/// measured on a dense trunk whose actual lm_head takes the NAX lane at both
/// verify widths produced by depth 8 (M=8 and M=9); every field below mirrors
/// a production dispatch precondition rather than trusting model-wide config.
const MtpNaxProfileInputs = struct {
    dense_model: bool,
    calibrated_model: bool,
    profiled_affine_trunk: bool,
    model_quant: QuantParams,
    weight_present: bool,
    packed_weight: bool,
    scales_present: bool,
    biases_present: bool,
    quant: QuantParams,
    K: c_int,
    N: c_int,
    packed_k: c_int,
    verify_on: bool,
    lane_on: bool,
    available: bool,
    min_m: c_int,
};

fn mtpNaxBf16EmbeddingMatchesFrom(
    dtype: mlx.mlx_dtype,
    shape: []const c_int,
    scales_present: bool,
    biases_present: bool,
    hidden_size: u32,
    vocab_size: u32,
) bool {
    if (dtype != .bfloat16 or scales_present or biases_present) return false;
    if (hidden_size == 0 or vocab_size == 0 or
        hidden_size > std.math.maxInt(c_int) or vocab_size > std.math.maxInt(c_int)) return false;
    return shape.len == 2 and
        shape[0] == @as(c_int, @intCast(vocab_size)) and
        shape[1] == @as(c_int, @intCast(hidden_size));
}

fn mtpNaxBf16EmbeddingMatches(
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    hidden_size: u32,
    vocab_size: u32,
) bool {
    if (w.ctx == null) return false;
    return mtpNaxBf16EmbeddingMatchesFrom(
        mlx.mlx_array_dtype(w),
        mlx.getShape(w),
        sc.ctx != null,
        bi.ctx != null,
        hidden_size,
        vocab_size,
    );
}

pub fn mtpNaxCalibratedModelFrom(config: *const ModelConfig, lm_head_n: c_int) bool {
    return std.mem.eql(u8, config.model_type, "qwen3_5_moe") and
        !config.isMoe() and
        config.hidden_size == 5120 and
        config.intermediate_size == 17408 and
        config.num_hidden_layers == 64 and
        config.vocab_size == 248320 and
        lm_head_n == 248320 and
        config.num_attention_heads == 24 and
        config.num_key_value_heads == 4 and
        config.head_dim == 256 and
        config.full_attention_interval == 4 and
        config.linear_num_key_heads == 16 and
        config.linear_num_value_heads == 48 and
        config.linear_key_head_dim == 128 and
        config.linear_value_head_dim == 128 and
        config.attn_output_gate and
        !config.tie_word_embeddings;
}

fn mtpNaxProfileEnabledFrom(input: MtpNaxProfileInputs) bool {
    if (!input.dense_model or !input.calibrated_model or !input.profiled_affine_trunk) return false;
    if (!input.weight_present or !input.packed_weight) return false;
    if (!input.scales_present or !input.biases_present) return false;
    if (input.model_quant.mode != .affine or input.quant.mode != .affine) return false;
    if (input.model_quant.group_size != 64 or input.quant.group_size != 64) return false;
    if (input.model_quant.bits != input.quant.bits) return false;
    if (input.quant.bits != 4 and input.quant.bits != 6 and input.quant.bits != 8) return false;
    if (input.K <= 0 or input.N <= 0 or input.packed_k <= 0) return false;
    const packed_bits = @as(u64, @intCast(input.K)) * @as(u64, input.quant.bits);
    if (@mod(packed_bits, 32) != 0 or
        packed_bits / 32 != @as(u64, @intCast(input.packed_k))) return false;

    return verifyQmmNaxEnabledForMFrom(
        8,
        input.K,
        input.N,
        input.verify_on,
        input.lane_on,
        input.available,
        input.min_m,
    ) and verifyQmmNaxEnabledForMFrom(
        9,
        input.K,
        input.N,
        input.verify_on,
        input.lane_on,
        input.available,
        input.min_m,
    );
}

fn mtpNaxAffineProjectionMatchesQuant(
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    in_dim: u32,
    out_dim: u32,
    expected: QuantParams,
) bool {
    if (w.ctx == null or sc.ctx == null or bi.ctx == null) return false;
    if (mlx.mlx_array_dtype(w) != .uint32 or
        mlx.mlx_array_dtype(sc) != .bfloat16 or
        mlx.mlx_array_dtype(bi) != .bfloat16) return false;
    if (in_dim == 0 or out_dim == 0 or
        in_dim > std.math.maxInt(c_int) or out_dim > std.math.maxInt(c_int)) return false;
    const out: c_int = @intCast(out_dim);
    const w_shape = mlx.getShape(w);
    const s_shape = mlx.getShape(sc);
    const b_shape = mlx.getShape(bi);
    if (w_shape.len != 2 or s_shape.len != 2 or b_shape.len != 2) return false;
    if (w_shape[0] != out or s_shape[0] != out or b_shape[0] != out) return false;
    if (s_shape[1] != b_shape[1]) return false;
    const qp = affineParamsFromGeometry(w, sc, in_dim) orelse return false;
    if (qp.bits != expected.bits or
        qp.group_size != expected.group_size or
        qp.mode != expected.mode) return false;

    // Every material projection on the calibrated surface must take NAX at
    // both depth-8 verify widths. Tiny per-head A/B gates and the measured
    // narrow q5/q6 regressions remain on stock by design.
    if (out_dim >= 512 and mixedNaxShapeEnabled(expected.bits, expected.group_size, @intCast(out_dim), 8)) {
        const K: c_int = @intCast(in_dim);
        const N: c_int = @intCast(out_dim);
        if (!verifyQmmNaxEnabledForMFrom(8, K, N, true, true, true, 8) or
            !verifyQmmNaxEnabledForMFrom(9, K, N, true, true, true, 8)) return false;
    }
    return true;
}

fn mtpNaxAffineProjectionMatches(
    config: *const ModelConfig,
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    in_dim: u32,
    out_dim: u32,
) bool {
    return mtpNaxAffineProjectionMatchesQuant(w, sc, bi, in_dim, out_dim, .{
        .bits = config.quant_bits,
        .group_size = config.quant_group_size,
        .mode = config.quant_mode,
    });
}

fn mtpNaxDenseMlpMatches(config: *const ModelConfig, mlp: *const DenseMlpWeights) bool {
    return mtpNaxAffineProjectionMatches(config, mlp.gate_w, mlp.gate_s, mlp.gate_b, config.hidden_size, config.intermediate_size) and
        mtpNaxAffineProjectionMatches(config, mlp.up_w, mlp.up_s, mlp.up_b, config.hidden_size, config.intermediate_size) and
        mtpNaxAffineProjectionMatches(config, mlp.down_w, mlp.down_s, mlp.down_b, config.intermediate_size, config.hidden_size);
}

pub const OQE_MLP_DOWN_BITS = "5555555555555555555555555556555644444444444444444444444444444444";
pub const OQE_LINEAR_Z_BITS = "5554444444545554555454444444444444444444444444444444444444444444";
pub const OQE_LINEAR_AB_BITS = "5554544444545554555455545554555444444444444444444444444444444444";
pub const OQE_LINEAR_OUT_BITS = "5554555455545554555455545554555455545554555455545554555455545554";
pub const OQE_FULL_QK_BITS = "4444444444444444444444444445444544444444444444444444444444444444";
pub const OQE_FULL_V_BITS = "4444444444444444444444444446444444444444444444444444444444444444";
pub const OQE_FULL_O_BITS = "4444444444454445444544444444444444444444444444444444444444444444";

pub fn oqeLayerBits(pattern: *const [64:0]u8, layer: usize) u32 {
    std.debug.assert(layer < 64);
    return pattern[layer] - '0';
}

fn mtpNaxOqeProjectionMatches(
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    in_dim: u32,
    out_dim: u32,
    bits: u32,
) bool {
    return mtpNaxAffineProjectionMatchesQuant(w, sc, bi, in_dim, out_dim, .{
        .bits = bits,
        .group_size = 64,
        .mode = .affine,
    });
}

/// Exact resident-tensor fingerprint for Jundot's Qwen3.6-27B-oQ4e trunk.
/// The checkpoint is globally affine-4/gs64 with a fixed layer/role map of
/// affine q5/q6 overrides. Matching the full map avoids granting cap 8 to
/// unrelated "mixed 4-bit" checkpoints with a different round-cost surface.
pub fn mtpNaxOqeAffineTrunkFrom(config: *const ModelConfig, maybe_layers: ?[]MoeLayerWeights) bool {
    if (config.isMoe() or config.full_attention_interval == 0) return false;
    if (config.num_hidden_layers != 64 or config.hidden_size == 0 or config.intermediate_size == 0) return false;
    const layers = maybe_layers orelse return false;
    if (layers.len != 64) return false;

    const full_out_wide = @as(u64, config.num_attention_heads) * config.head_dim;
    const full_q_out_wide = full_out_wide * 2;
    const kv_out_wide = @as(u64, config.num_key_value_heads) * config.head_dim;
    const linear_key_out_wide = @as(u64, config.linear_num_key_heads) * config.linear_key_head_dim;
    const linear_out_wide = @as(u64, config.linear_num_value_heads) * config.linear_value_head_dim;
    const linear_qkv_out_wide = 2 * linear_key_out_wide + linear_out_wide;
    if (full_out_wide == 0 or full_out_wide > std.math.maxInt(u32) or
        full_q_out_wide > std.math.maxInt(u32) or
        kv_out_wide == 0 or kv_out_wide > std.math.maxInt(u32) or
        linear_out_wide == 0 or linear_out_wide > std.math.maxInt(u32) or
        linear_qkv_out_wide > std.math.maxInt(u32)) return false;
    const full_out: u32 = @intCast(full_out_wide);
    const full_q_out: u32 = @intCast(full_q_out_wide);
    const kv_out: u32 = @intCast(kv_out_wide);
    const linear_out: u32 = @intCast(linear_out_wide);
    const linear_qkv_out: u32 = @intCast(linear_qkv_out_wide);

    for (layers, 0..) |*layer, i| {
        if (layer.is_linear != config.isLinearLayer(@intCast(i))) return false;
        switch (layer.attn) {
            .full => |*attn| {
                if (!mtpNaxOqeProjectionMatches(attn.q_w, attn.q_s, attn.q_b, config.hidden_size, full_q_out, oqeLayerBits(OQE_FULL_QK_BITS, i)) or
                    !mtpNaxOqeProjectionMatches(attn.k_w, attn.k_s, attn.k_b, config.hidden_size, kv_out, oqeLayerBits(OQE_FULL_QK_BITS, i)) or
                    !mtpNaxOqeProjectionMatches(attn.v_w, attn.v_s, attn.v_b, config.hidden_size, kv_out, oqeLayerBits(OQE_FULL_V_BITS, i)) or
                    !mtpNaxOqeProjectionMatches(attn.o_w, attn.o_s, attn.o_b, full_out, config.hidden_size, oqeLayerBits(OQE_FULL_O_BITS, i))) return false;
            },
            .linear => |*attn| {
                if (attn.combined_proj) return false;
                if (!mtpNaxOqeProjectionMatches(attn.qkv_w, attn.qkv_s, attn.qkv_b, config.hidden_size, linear_qkv_out, 4) or
                    !mtpNaxOqeProjectionMatches(attn.z_w, attn.z_s, attn.z_b, config.hidden_size, linear_out, oqeLayerBits(OQE_LINEAR_Z_BITS, i)) or
                    !mtpNaxOqeProjectionMatches(attn.a_w, attn.a_s, attn.a_b, config.hidden_size, config.linear_num_value_heads, oqeLayerBits(OQE_LINEAR_AB_BITS, i)) or
                    !mtpNaxOqeProjectionMatches(attn.b_w, attn.b_s, attn.b_b, config.hidden_size, config.linear_num_value_heads, oqeLayerBits(OQE_LINEAR_AB_BITS, i)) or
                    !mtpNaxOqeProjectionMatches(attn.out_w, attn.out_s, attn.out_b, linear_out, config.hidden_size, oqeLayerBits(OQE_LINEAR_OUT_BITS, i))) return false;
            },
        }
        switch (layer.mlp) {
            .dense => |*mlp| {
                if (!mtpNaxOqeProjectionMatches(mlp.gate_w, mlp.gate_s, mlp.gate_b, config.hidden_size, config.intermediate_size, 4) or
                    !mtpNaxOqeProjectionMatches(mlp.up_w, mlp.up_s, mlp.up_b, config.hidden_size, config.intermediate_size, 4) or
                    !mtpNaxOqeProjectionMatches(mlp.down_w, mlp.down_s, mlp.down_b, config.intermediate_size, config.hidden_size, oqeLayerBits(OQE_MLP_DOWN_BITS, i))) return false;
            },
            .moe => return false,
        }
    }
    return true;
}

/// The depth-8 controller constants describe the measured homogeneous affine
/// Qwen3.6-27B trunk, not merely its architecture. Same-shape mixed checkpoints
/// (notably Unsloth Dynamic) may store individual projections as BF16 or mxfp8
/// while retaining a model-wide affine-4 config. Inspect every resident linear
/// used by the dense trunk so those checkpoints stay on the conservative cap 6.
pub fn mtpNaxUniformAffineTrunkFrom(config: *const ModelConfig, maybe_layers: ?[]MoeLayerWeights) bool {
    if (config.isMoe() or config.full_attention_interval == 0) return false;
    if (config.hidden_size == 0 or config.intermediate_size == 0) return false;
    const layers = maybe_layers orelse return false;
    if (layers.len != @as(usize, @intCast(config.num_hidden_layers))) return false;

    const full_out_wide = @as(u64, config.num_attention_heads) * config.head_dim;
    const full_q_out_wide = full_out_wide * 2;
    const kv_out_wide = @as(u64, config.num_key_value_heads) * config.head_dim;
    const linear_key_out_wide = @as(u64, config.linear_num_key_heads) * config.linear_key_head_dim;
    const linear_out_wide = @as(u64, config.linear_num_value_heads) * config.linear_value_head_dim;
    const linear_qkv_out_wide = 2 * linear_key_out_wide + linear_out_wide;
    if (full_out_wide == 0 or full_out_wide > std.math.maxInt(u32) or
        full_q_out_wide > std.math.maxInt(u32) or
        kv_out_wide == 0 or kv_out_wide > std.math.maxInt(u32) or
        linear_key_out_wide == 0 or linear_key_out_wide > std.math.maxInt(u32) or
        linear_out_wide == 0 or linear_out_wide > std.math.maxInt(u32) or
        linear_qkv_out_wide > std.math.maxInt(u32)) return false;
    const full_out: u32 = @intCast(full_out_wide);
    const full_q_out: u32 = @intCast(full_q_out_wide);
    const kv_out: u32 = @intCast(kv_out_wide);
    const linear_out: u32 = @intCast(linear_out_wide);
    const linear_qkv_out: u32 = @intCast(linear_qkv_out_wide);

    for (layers, 0..) |*layer, i| {
        if (layer.is_linear != config.isLinearLayer(@intCast(i))) return false;
        switch (layer.attn) {
            .full => |*attn| {
                if (!mtpNaxAffineProjectionMatches(config, attn.q_w, attn.q_s, attn.q_b, config.hidden_size, full_q_out) or
                    !mtpNaxAffineProjectionMatches(config, attn.k_w, attn.k_s, attn.k_b, config.hidden_size, kv_out) or
                    !mtpNaxAffineProjectionMatches(config, attn.v_w, attn.v_s, attn.v_b, config.hidden_size, kv_out) or
                    !mtpNaxAffineProjectionMatches(config, attn.o_w, attn.o_s, attn.o_b, full_out, config.hidden_size)) return false;
            },
            .linear => |*attn| {
                if (attn.combined_proj) return false;
                if (!mtpNaxAffineProjectionMatches(config, attn.qkv_w, attn.qkv_s, attn.qkv_b, config.hidden_size, linear_qkv_out) or
                    !mtpNaxAffineProjectionMatches(config, attn.z_w, attn.z_s, attn.z_b, config.hidden_size, linear_out) or
                    !mtpNaxAffineProjectionMatches(config, attn.a_w, attn.a_s, attn.a_b, config.hidden_size, config.linear_num_value_heads) or
                    !mtpNaxAffineProjectionMatches(config, attn.b_w, attn.b_s, attn.b_b, config.hidden_size, config.linear_num_value_heads) or
                    !mtpNaxAffineProjectionMatches(config, attn.out_w, attn.out_s, attn.out_b, linear_out, config.hidden_size)) return false;
            },
        }
        switch (layer.mlp) {
            .dense => |*mlp| if (!mtpNaxDenseMlpMatches(config, mlp)) return false,
            .moe => return false,
        }
    }
    return true;
}

pub fn mtpNaxProfileEnabledForTrunk(self: *const Transformer, profiled_trunk: bool, mixed_bits: bool) bool {
    if (self.config.isMoe()) return false;
    if (self.config.hidden_size > std.math.maxInt(c_int)) return false;
    if (self.config.quant_mode != .affine) return false;
    if (self.config.quant_bits != 4 and self.config.quant_bits != 6 and self.config.quant_bits != 8) return false;
    if (self.config.quant_group_size != 64) return false;
    if (mixed_bits and !naxMixedBitsEnvEnabled()) return false;
    if (self.lm_head_w.ctx == null or
        self.lm_head_s.ctx == null or
        self.lm_head_b.ctx == null) return false;
    if (mlx.mlx_array_dtype(self.lm_head_w) != .uint32) return false;
    const K: c_int = @intCast(self.config.hidden_size);
    const w_shape = mlx.getShape(self.lm_head_w);
    if (w_shape.len != 2) return false;
    // The cost fits are intentionally narrower than kernel eligibility:
    // only the measured dense 27B geometry and exact quantized tensor
    // surfaces reach cap 8. Other NAX-capable models retain cap 6.
    const calibrated_model = mtpNaxCalibratedModelFrom(&self.config, w_shape[0]);
    const calibrated_head = mtpNaxAffineProjectionMatches(
        &self.config,
        self.lm_head_w,
        self.lm_head_s,
        self.lm_head_b,
        self.config.hidden_size,
        self.config.vocab_size,
    );

    return mtpNaxProfileEnabledFrom(.{
        .dense_model = true,
        .calibrated_model = calibrated_model,
        .profiled_affine_trunk = profiled_trunk,
        .model_quant = .{
            .bits = self.config.quant_bits,
            .group_size = self.config.quant_group_size,
            .mode = self.config.quant_mode,
        },
        .weight_present = self.lm_head_w.ctx != null,
        .packed_weight = calibrated_head,
        .scales_present = self.lm_head_s.ctx != null,
        .biases_present = self.lm_head_b.ctx != null,
        .quant = self.quantParamsHinted(self.lm_head_w, self.lm_head_s, self.config.hidden_size),
        .K = K,
        .N = w_shape[0],
        .packed_k = w_shape[1],
        .verify_on = verifyQmmEnabled(),
        .lane_on = naxLaneEnvEnabled(),
        .available = verifyQmmNaxAvailable(),
        .min_m = naxMinMForBits(self.config.quant_bits),
    });
}

/// Homogeneous q4/gs64 target profile used by the existing q4/q8-gs32
/// MTP sidecar cost surfaces.
pub fn mtpNaxProfileEnabled(self: *const Transformer) bool {
    return self.config.quant_bits == 4 and mtpNaxAffineProjectionMatches(
        &self.config,
        self.emb_w,
        self.emb_s,
        self.emb_b,
        self.config.hidden_size,
        self.config.vocab_size,
    ) and self.mtpNaxProfileEnabledForTrunk(self.mtp_uniform_affine_trunk, false);
}

/// Uniform q4/gs64 Qwen3.8 target with its measured dense-bf16 token
/// embedding. Kept separate because every draft reads this table.
pub fn mtpNaxQ4Gs64ProfileEnabled(self: *const Transformer) bool {
    return self.config.quant_bits == 4 and mtpNaxBf16EmbeddingMatches(
        self.emb_w,
        self.emb_s,
        self.emb_b,
        self.config.hidden_size,
        self.config.vocab_size,
    ) and self.mtpNaxProfileEnabledForTrunk(self.mtp_uniform_affine_trunk, false);
}

/// Uniform q6/gs64 Qwen3.8 target with a quantized token embedding. The
/// mixed-width kill switch revokes this profile because the measured
/// depth-8 surface depends on q6 NAX at M=8 and M=9.
pub fn mtpNaxQ6Gs64ProfileEnabled(self: *const Transformer) bool {
    const embedding_match = mtpNaxAffineProjectionMatches(
        &self.config,
        self.emb_w,
        self.emb_s,
        self.emb_b,
        self.config.hidden_size,
        self.config.vocab_size,
    );
    const trunk = self.mtpNaxProfileEnabledForTrunk(self.mtp_uniform_affine_trunk, true);
    return self.config.quant_bits == 6 and embedding_match and trunk;
}

/// Uniform q8/gs64 Qwen3.8 target with its dense-bf16 token embedding.
/// NAX engagement is mandatory: without the q8 byte-unpack lane the same
/// depth-8 checkpoint is 12.96% slower and must stay on generic cap 6.
pub fn mtpNaxQ8Gs64ProfileEnabled(self: *const Transformer) bool {
    return self.config.quant_bits == 8 and mtpNaxBf16EmbeddingMatches(
        self.emb_w,
        self.emb_s,
        self.emb_b,
        self.config.hidden_size,
        self.config.vocab_size,
    ) and self.mtpNaxProfileEnabledForTrunk(self.mtp_uniform_affine_trunk, true);
}

/// Exact oQ4e mixed q4/q5/q6 target profile. The q5/q6 lane kill switch
/// also revokes the cost profile so auto depth never assumes unavailable
/// mixed-bit acceleration.
pub fn mtpOqeNaxProfileEnabled(self: *const Transformer) bool {
    return self.config.quant_bits == 4 and mtpNaxAffineProjectionMatches(
        &self.config,
        self.emb_w,
        self.emb_s,
        self.emb_b,
        self.config.hidden_size,
        self.config.vocab_size,
    ) and self.mtpNaxProfileEnabledForTrunk(self.mtp_oqe_affine_trunk, true);
}

// ── tests ──

fn mtpNaxTestConfig() ModelConfig {
    var config = ModelConfig{};
    config.model_type = "qwen3_5_moe";
    config.hidden_size = 5120;
    config.intermediate_size = 17408;
    config.num_hidden_layers = 64;
    config.vocab_size = 248320;
    config.num_attention_heads = 24;
    config.num_key_value_heads = 4;
    config.head_dim = 256;
    config.full_attention_interval = 4;
    config.linear_num_key_heads = 16;
    config.linear_num_value_heads = 48;
    config.linear_key_head_dim = 128;
    config.linear_value_head_dim = 128;
    config.attn_output_gate = true;
    config.tie_word_embeddings = false;
    config.num_experts = 0;
    config.quant_bits = 4;
    config.quant_group_size = 64;
    config.quant_mode = .affine;
    return config;
}

test "mtpNaxCalibratedModelFrom pins the complete measured dense Qwen3.6-27B architecture" {
    const good = mtpNaxTestConfig();
    try testing.expect(mtpNaxCalibratedModelFrom(&good, 248320));

    var bad = good;
    bad.model_type = "qwen3_moe";
    try testing.expect(!mtpNaxCalibratedModelFrom(&bad, 248320));
    bad = good;
    bad.hidden_size = 4096;
    try testing.expect(!mtpNaxCalibratedModelFrom(&bad, 248320));
    bad = good;
    bad.intermediate_size = 17440;
    try testing.expect(!mtpNaxCalibratedModelFrom(&bad, 248320));
    bad = good;
    bad.num_hidden_layers = 48;
    try testing.expect(!mtpNaxCalibratedModelFrom(&bad, 248320));
    bad = good;
    bad.vocab_size = 151936;
    try testing.expect(!mtpNaxCalibratedModelFrom(&bad, 151936));
    try testing.expect(!mtpNaxCalibratedModelFrom(&good, 151936));
    bad = good;
    bad.num_attention_heads = 20;
    try testing.expect(!mtpNaxCalibratedModelFrom(&bad, 248320));
    bad = good;
    bad.num_key_value_heads = 8;
    try testing.expect(!mtpNaxCalibratedModelFrom(&bad, 248320));
    bad = good;
    bad.head_dim = 128;
    try testing.expect(!mtpNaxCalibratedModelFrom(&bad, 248320));
    bad = good;
    bad.full_attention_interval = 8;
    try testing.expect(!mtpNaxCalibratedModelFrom(&bad, 248320));
    bad = good;
    bad.linear_num_value_heads = 32;
    try testing.expect(!mtpNaxCalibratedModelFrom(&bad, 248320));
    bad = good;
    bad.linear_value_head_dim = 64;
    try testing.expect(!mtpNaxCalibratedModelFrom(&bad, 248320));
    bad = good;
    bad.attn_output_gate = false;
    try testing.expect(!mtpNaxCalibratedModelFrom(&bad, 248320));
    bad = good;
    bad.tie_word_embeddings = true;
    try testing.expect(!mtpNaxCalibratedModelFrom(&bad, 248320));
    bad = good;
    bad.num_experts = 8;
    try testing.expect(!mtpNaxCalibratedModelFrom(&bad, 248320));
}

test "mtpNaxBf16EmbeddingMatchesFrom pins the measured Qwen3.8 embedding surface" {
    const shape = [_]c_int{ 248320, 5120 };
    try testing.expect(mtpNaxBf16EmbeddingMatchesFrom(.bfloat16, &shape, false, false, 5120, 248320));
    try testing.expect(!mtpNaxBf16EmbeddingMatchesFrom(.float16, &shape, false, false, 5120, 248320));
    try testing.expect(!mtpNaxBf16EmbeddingMatchesFrom(.bfloat16, &shape, true, false, 5120, 248320));
    try testing.expect(!mtpNaxBf16EmbeddingMatchesFrom(.bfloat16, &shape, false, true, 5120, 248320));
    const wrong_vocab = [_]c_int{ 248319, 5120 };
    try testing.expect(!mtpNaxBf16EmbeddingMatchesFrom(.bfloat16, &wrong_vocab, false, false, 5120, 248320));
    const wrong_hidden = [_]c_int{ 248320, 4096 };
    try testing.expect(!mtpNaxBf16EmbeddingMatchesFrom(.bfloat16, &wrong_hidden, false, false, 5120, 248320));
}

test "mtpNaxAffineProjectionMatches rejects mixed storage and off-profile quant geometry" {
    const s = mlx.gpuStream();
    const IN: u32 = 128;
    const OUT: u32 = 4;
    var config = ModelConfig{};
    config.quant_bits = 4;
    config.quant_group_size = 64;
    config.quant_mode = .affine;

    const mk = struct {
        fn arr(shape: []const c_int, dtype: mlx.mlx_dtype, stream: mlx.mlx_stream) !mlx.mlx_array {
            var a = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&a, shape.ptr, shape.len, dtype, stream));
            return a;
        }
    };

    const w4 = try mk.arr(&.{ OUT, IN * 4 / 32 }, .uint32, s);
    defer _ = mlx.mlx_array_free(w4);
    const sc64 = try mk.arr(&.{ OUT, IN / 64 }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(sc64);
    const bi64 = try mk.arr(&.{ OUT, IN / 64 }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(bi64);
    try testing.expect(mtpNaxAffineProjectionMatches(&config, w4, sc64, bi64, IN, OUT));
    try testing.expect(!mtpNaxAffineProjectionMatches(&config, w4, sc64, bi64, IN, OUT + 1));

    // Unsloth Dynamic projection: BF16 weight with no quant metadata.
    const w_bf16 = try mk.arr(&.{ OUT, IN }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(w_bf16);
    const none = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none);
    try testing.expect(!mtpNaxAffineProjectionMatches(&config, w_bf16, none, none, IN, OUT));

    // Same affine storage types, but per-tensor bits/group disagree with the
    // calibrated global 4-bit/gs-64 surface.
    const w2 = try mk.arr(&.{ OUT, IN * 2 / 32 }, .uint32, s);
    defer _ = mlx.mlx_array_free(w2);
    try testing.expect(!mtpNaxAffineProjectionMatches(&config, w2, sc64, bi64, IN, OUT));
    const sc128 = try mk.arr(&.{ OUT, IN / 128 }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(sc128);
    const bi128 = try mk.arr(&.{ OUT, IN / 128 }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(bi128);
    try testing.expect(!mtpNaxAffineProjectionMatches(&config, w4, sc128, bi128, IN, OUT));

    // mxfp8-style uint8 scales cannot inherit the affine cost fit even if a
    // synthetic bias handle is present.
    const sc_mx = try mk.arr(&.{ OUT, IN / 64 }, .uint8, s);
    defer _ = mlx.mlx_array_free(sc_mx);
    try testing.expect(!mtpNaxAffineProjectionMatches(&config, w4, sc_mx, bi64, IN, OUT));

    // Material projections must also satisfy the actual M=8/M=9 NAX lane
    // geometry. K=256/N=512 qualifies; an otherwise valid K=320 affine
    // tensor does not (K % 256 != 0).
    const big_w = try mk.arr(&.{ 512, 256 * 4 / 32 }, .uint32, s);
    defer _ = mlx.mlx_array_free(big_w);
    const big_sc = try mk.arr(&.{ 512, 256 / 64 }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(big_sc);
    const big_bi = try mk.arr(&.{ 512, 256 / 64 }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(big_bi);
    try testing.expect(mtpNaxAffineProjectionMatches(&config, big_w, big_sc, big_bi, 256, 512));

    const off_lane_w = try mk.arr(&.{ 512, 320 * 4 / 32 }, .uint32, s);
    defer _ = mlx.mlx_array_free(off_lane_w);
    const off_lane_sc = try mk.arr(&.{ 512, 320 / 64 }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(off_lane_sc);
    const off_lane_bi = try mk.arr(&.{ 512, 320 / 64 }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(off_lane_bi);
    try testing.expect(!mtpNaxAffineProjectionMatches(&config, off_lane_w, off_lane_sc, off_lane_bi, 320, 512));
}

test "mtpNaxProfileEnabledFrom composes measured model, homogeneous trunk, and affine lm-head lane" {
    const model = mtpNaxTestConfig();
    const good: MtpNaxProfileInputs = .{
        .dense_model = true,
        .calibrated_model = mtpNaxCalibratedModelFrom(&model, 248320),
        .profiled_affine_trunk = true,
        .model_quant = .{ .bits = 4, .group_size = 64, .mode = .affine },
        .weight_present = true,
        .packed_weight = true,
        .scales_present = true,
        .biases_present = true,
        .quant = .{ .bits = 4, .group_size = 64, .mode = .affine },
        .K = 5120,
        .N = 248320,
        .packed_k = 640,
        .verify_on = true,
        .lane_on = true,
        .available = true,
        .min_m = 8,
    };
    try testing.expect(mtpNaxProfileEnabledFrom(good));

    var good6 = good;
    good6.model_quant.bits = 6;
    good6.quant.bits = 6;
    good6.packed_k = 960;
    try testing.expect(mtpNaxProfileEnabledFrom(good6));
    var good8 = good;
    good8.model_quant.bits = 8;
    good8.quant.bits = 8;
    good8.packed_k = 1280;
    try testing.expect(mtpNaxProfileEnabledFrom(good8));
    var unsupported = good;
    unsupported.model_quant.bits = 5;
    unsupported.quant.bits = 5;
    unsupported.packed_k = 800;
    try testing.expect(!mtpNaxProfileEnabledFrom(unsupported));

    var bad = good;
    bad.dense_model = false;
    try testing.expect(!mtpNaxProfileEnabledFrom(bad));
    bad = good;
    bad.calibrated_model = false;
    try testing.expect(!mtpNaxProfileEnabledFrom(bad));
    bad = good;
    bad.profiled_affine_trunk = false;
    try testing.expect(!mtpNaxProfileEnabledFrom(bad));
    bad = good;
    bad.model_quant.bits = 8;
    try testing.expect(!mtpNaxProfileEnabledFrom(bad));
    bad = good;
    bad.model_quant.group_size = 16;
    try testing.expect(!mtpNaxProfileEnabledFrom(bad));
    bad = good;
    bad.model_quant.mode = .nvfp4;
    try testing.expect(!mtpNaxProfileEnabledFrom(bad));
    bad = good;
    bad.weight_present = false;
    try testing.expect(!mtpNaxProfileEnabledFrom(bad));
    bad = good;
    bad.packed_weight = false;
    try testing.expect(!mtpNaxProfileEnabledFrom(bad));
    bad = good;
    bad.scales_present = false;
    try testing.expect(!mtpNaxProfileEnabledFrom(bad));
    bad = good;
    bad.biases_present = false;
    try testing.expect(!mtpNaxProfileEnabledFrom(bad));

    bad = good;
    bad.quant.bits = 8;
    try testing.expect(!mtpNaxProfileEnabledFrom(bad));
    bad = good;
    bad.quant.group_size = 16;
    try testing.expect(!mtpNaxProfileEnabledFrom(bad));
    bad = good;
    bad.quant.mode = .nvfp4;
    try testing.expect(!mtpNaxProfileEnabledFrom(bad));

    bad = good;
    bad.packed_k = 639;
    try testing.expect(!mtpNaxProfileEnabledFrom(bad));
    bad = good;
    bad.K = 5184;
    bad.packed_k = 648;
    try testing.expect(!mtpNaxProfileEnabledFrom(bad));
    bad = good;
    bad.N = 248324;
    try testing.expect(!mtpNaxProfileEnabledFrom(bad));
    bad = good;
    bad.N = 480;
    try testing.expect(!mtpNaxProfileEnabledFrom(bad));

    // Both depth-8 verify widths must take NAX. A min-M of 9 only covers one.
    bad = good;
    bad.min_m = 9;
    try testing.expect(!mtpNaxProfileEnabledFrom(bad));
    bad = good;
    bad.verify_on = false;
    try testing.expect(!mtpNaxProfileEnabledFrom(bad));
    bad = good;
    bad.lane_on = false;
    try testing.expect(!mtpNaxProfileEnabledFrom(bad));
    bad = good;
    bad.available = false;
    try testing.expect(!mtpNaxProfileEnabledFrom(bad));

    // Lowering the takeover width retains M=8/M=9 coverage.
    bad = good;
    bad.min_m = 5;
    try testing.expect(mtpNaxProfileEnabledFrom(bad));
}
