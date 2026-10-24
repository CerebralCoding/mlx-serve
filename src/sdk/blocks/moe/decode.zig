//! sdk/blocks/moe/decode, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns batchedExpertMm, and 9 more.

const std = @import("std");
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../util/log.zig");
const QuantMode = @import("../../quant/params.zig").QuantMode;
const moe_fp4 = @import("moe_fp4.zig");
const moe_affine4 = @import("moe_affine4.zig");
const down_reduce = @import("down_reduce.zig");
const gather_qmv = @import("gather_qmv.zig");
const gather_qmv_rows = @import("gather_qmv_rows.zig");
const testing = std.testing;
const lastDim = @import("../../../mlx/array.zig").lastDim;
const qwen4Standin = @import("../../diag/standin.zig").qwen4Standin;
const standinOnes = @import("../../diag/standin.zig").standinOnes;
const QuantParams = @import("../../quant/params.zig").QuantParams;
const swigluClampLimit = @import("../mlp/activations.zig").swigluClampLimit;
const swigluSigTable = @import("../mlp/fused_activations.zig").swigluSigTable;
const gatherQmvDownReduce = @import("down_reduce.zig").gatherQmvDownReduce;
const gatherQmvDownReduceRows = @import("down_reduce.zig").gatherQmvDownReduceRows;
const gatherQmvDownReduceRowsEligible = @import("down_reduce.zig").gatherQmvDownReduceRowsEligible;
const gatherQmv = @import("gather_qmv.zig").gatherQmv;
const gatherQmvGateUp = @import("gather_qmv.zig").gatherQmvGateUp;
const gatherQmvGateUpEnabled = @import("gather_qmv.zig").gatherQmvGateUpEnabled;
const gatherQmvGateUpRows = @import("gather_qmv_rows.zig").gatherQmvGateUpRows;
const MoeMlpWeights = @import("weights.zig").MoeMlpWeights;

/// The fp4 MoE decode kernels read no expert biases, and apply a SwiGLU clamp only as a whole-number
/// limit: a pack with a bias, or `clamp` null (`swigluClampLimit`: a clamp no kernel serves), keeps the
/// composed chain, as does an activation that is not 16-bit (the SwiGLU table covers 16-bit patterns).
pub fn moeFp4DecodeEligible(rows: c_int, has_expert_bias: bool, clamp: ?u32, act: mlx.mlx_dtype, gate: QuantMode, up: QuantMode, down: QuantMode) bool {
    return rows <= moe_fp4.MAX_ROWS and !has_expert_bias and clamp != null and (act == .bfloat16 or act == .float16) and (gate == .mxfp4 or gate == .nvfp4) and up == gate and down == gate;
}

// ── tests ──

const ModelConfig = @import("../../../engines/mlx/model.zig").ModelConfig;

test "moe fp4 decode: expert biases or a clamp no kernel serves keep the composed chain" {
    var cfg: ModelConfig = undefined;
    cfg.swiglu_limit = 0;
    cfg.swiglu_clamp = false;
    try testing.expectEqual(@as(?u32, 0), swigluClampLimit(cfg.swiglu_limit, cfg.swiglu_clamp));
    cfg.swiglu_limit = 10;
    cfg.swiglu_clamp = true;
    try testing.expectEqual(@as(?u32, 10), swigluClampLimit(cfg.swiglu_limit, cfg.swiglu_clamp));
    try testing.expect(moeFp4DecodeEligible(1, false, 0, .bfloat16, .mxfp4, .mxfp4, .mxfp4));
    try testing.expect(!moeFp4DecodeEligible(1, true, 0, .bfloat16, .mxfp4, .mxfp4, .mxfp4));
    try testing.expect(moeFp4DecodeEligible(1, false, 10, .bfloat16, .nvfp4, .nvfp4, .nvfp4));
    // An f32 stream (an f32 norm weight upstream) has no sigmoid table: decline, never throw.
    try testing.expect(!moeFp4DecodeEligible(1, false, 0, .float32, .mxfp4, .mxfp4, .mxfp4));
    try testing.expect(moeFp4DecodeEligible(1, false, 0, .float16, .mxfp4, .mxfp4, .mxfp4));
    try testing.expect(!moeFp4DecodeEligible(1, false, null, .bfloat16, .mxfp4, .mxfp4, .mxfp4));
    try testing.expect(!moeFp4DecodeEligible(1, false, 0, .bfloat16, .affine, .mxfp4, .mxfp4));
    try testing.expect(moeFp4DecodeEligible(1, false, 0, .bfloat16, .nvfp4, .nvfp4, .nvfp4));
    try testing.expect(!moeFp4DecodeEligible(1, false, 0, .bfloat16, .nvfp4, .mxfp4, .nvfp4));
    try testing.expect(!moeFp4DecodeEligible(moe_fp4.MAX_ROWS + 1, false, 0, .bfloat16, .mxfp4, .mxfp4, .mxfp4));
}
