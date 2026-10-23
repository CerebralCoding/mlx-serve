const std = @import("std");
const crt = @import("../../sdk/util/crt.zig");
const builtin = @import("builtin");
const plugin_driver = @import("plugin/driver.zig");
const mlx_stream_mod = if (@import("build_options").mlx_stream) @import("../../sdk/weights/streamed/glue.zig") else @import("../../sdk/weights/streamed/stub.zig");
const qwen4_mod = @import("../../plugins/qwen4_exp/qwen4_exp.zig");
const distributed_mod = @import("cluster/distributed.zig");
const ple_gpu = @import("../../plugins/qwen4_exp/ple_gpu.zig");
// The qwen4_exp MTP head shares the sidecar head's draft-rerank scheme
// (`mtp.rerankSelect` + `QLinear`), which reads only the TARGET's lm_head and
// so has no head-shaped state of its own. mtp.zig imports this file back for
// `Transformer`; neither type closes a comptime loop.
const mtp_mod = @import("../../sdk/spec/mtp/mtp.zig");
const mlx = @import("../../mlx/ffi.zig");
const mrope = @import("../../sdk/vision/mrope.zig");
const rht = @import("../../sdk/quant/hadamard/rht.zig");
const qmv2 = @import("../../sdk/blocks/qmm/qmv2.zig");
const qmv_cuda = @import("../../sdk/blocks/qmm/qmv_cuda.zig");
const rowqmv = @import("../../sdk/blocks/qmm/rowqmv.zig");
const lane_qmm = @import("../../sdk/blocks/qmm/lane_qmm.zig");
const lane_attn = @import("../../sdk/blocks/attention/lane_attn.zig");
const simd_qmm = @import("../../sdk/blocks/qmm/simd_qmm.zig");
const row_attn = @import("../../sdk/blocks/attention/row_attn.zig");
const mlx_gguf = @import("../../sdk/quant/gguf/claim.zig");
const sushi_pack = @import("../../sdk/quant/exl3/sushi_pack.zig");
const qmv_nax2 = @import("../../sdk/blocks/qmm/qmv_nax2.zig");
const gather_qmm_nax = @import("../../sdk/blocks/moe/gather_qmm_nax.zig");
const qmm_int8 = @import("../../sdk/blocks/qmm/qmm_int8.zig");
const gdn_decode = @import("../../sdk/blocks/gdn/decode.zig");
const mamba2_decode = @import("../../sdk/blocks/ssm/mamba2_decode.zig");
const add_norm = @import("../../sdk/blocks/norms/add_norm.zig");
const kv_quant = @import("cache/kv_quant.zig");

pub const KVQuantConfig = kv_quant.KVQuantConfig;
pub const KVQuantScheme = kv_quant.Scheme;

/// `std.meta.fields` was replaced by parallel `fieldNames`/`fieldTypes`
/// arrays — zip them into the `.name`/`.type` shape every call site below
/// already expects.
pub const WeightField = struct { name: [:0]const u8, type: type };

// ── declarations split out by meaning (P1); aliases keep every call site unchanged ──
// engines/mlx/load/weights.zig
pub const transposeBf16WeightPub = @import("load/weights.zig").transposeBf16WeightPub;
pub const depthwiseConvNeedsTranspose = @import("load/weights.zig").depthwiseConvNeedsTranspose;
pub const maybeTransposeForBf16 = @import("load/weights.zig").maybeTransposeForBf16;
pub const floatDtypeTable = @import("load/weights.zig").floatDtypeTable;
pub const getWeightAt = @import("load/weights.zig").getWeightAt;
pub const loadQWAt = @import("load/weights.zig").loadQWAt;
pub const loadRoutedMoeAt = @import("load/weights.zig").loadRoutedMoeAt;
pub const scalarOf = @import("load/weights.zig").scalarOf;
// sdk/blocks/gdn/kernels.zig
pub const gdnBlockedEnabled = @import("../../sdk/blocks/gdn/kernels.zig").gdnBlockedEnabled;
pub const GDN_BLOCKED_MIN_T = @import("../../sdk/blocks/gdn/kernels.zig").GDN_BLOCKED_MIN_T;
pub const gdnBlockedEligible = @import("../../sdk/blocks/gdn/kernels.zig").gdnBlockedEligible;
pub const gdnBlockedTgBytes = @import("../../sdk/blocks/gdn/kernels.zig").gdnBlockedTgBytes;
pub const gdnBlockTFor = @import("../../sdk/blocks/gdn/kernels.zig").gdnBlockTFor;
pub const GdnRoute = @import("../../sdk/blocks/gdn/kernels.zig").GdnRoute;
pub const gdnPipelinedFor = @import("../../sdk/blocks/gdn/kernels.zig").gdnPipelinedFor;
// sdk/testing/parity.zig
pub const maxAbsDiffF32 = @import("../../sdk/testing/parity.zig").maxAbsDiffF32;
// sdk/blocks/linear/verify_qmm.zig
pub const VQMM_BN_CHOICES = @import("../../sdk/blocks/linear/verify_qmm.zig").VQMM_BN_CHOICES;
pub const VQMM_MAX_ACC = @import("../../sdk/blocks/linear/verify_qmm.zig").VQMM_MAX_ACC;
pub const vqmmBnFor = @import("../../sdk/blocks/linear/verify_qmm.zig").vqmmBnFor;
pub const verifyQmmEnabled = @import("../../sdk/blocks/linear/verify_qmm.zig").verifyQmmEnabled;
pub const VqmmLane = @import("../../sdk/blocks/linear/verify_qmm.zig").VqmmLane;
pub const mixedPlainShapeEnabled = @import("../../sdk/blocks/linear/verify_qmm.zig").mixedPlainShapeEnabled;
pub const MIXED_PLAIN_MIN_N = @import("../../sdk/blocks/linear/verify_qmm.zig").MIXED_PLAIN_MIN_N;
pub const vqmmLaneFor = @import("../../sdk/blocks/linear/verify_qmm.zig").vqmmLaneFor;
pub const VqmmTile = @import("../../sdk/blocks/linear/verify_qmm.zig").VqmmTile;
pub const SHADER_TILE_MAX_M = @import("../../sdk/blocks/linear/verify_qmm.zig").SHADER_TILE_MAX_M;
pub const vqmmTileFrom = @import("../../sdk/blocks/linear/verify_qmm.zig").vqmmTileFrom;
pub const vqmmTileRows = @import("../../sdk/blocks/linear/verify_qmm.zig").vqmmTileRows;
pub const vqmmLaneForTile = @import("../../sdk/blocks/linear/verify_qmm.zig").vqmmLaneForTile;
pub const verifyQmm = @import("../../sdk/blocks/linear/verify_qmm.zig").verifyQmm;
// mlx/device.zig
pub const naxArchGeneration = @import("../../mlx/device.zig").naxArchGeneration;
pub const DeviceGen = @import("../../mlx/device.zig").DeviceGen;
pub const deviceGeneration = @import("../../mlx/device.zig").deviceGeneration;
pub const naxArchSupportedFrom = @import("../../mlx/device.zig").naxArchSupportedFrom;
pub const naxArchIsG17 = @import("../../mlx/device.zig").naxArchIsG17;
pub const macosVersionAtLeast = @import("../../mlx/device.zig").macosVersionAtLeast;
pub const naxAvailableFrom = @import("../../mlx/device.zig").naxAvailableFrom;
pub const naxStatusFrom = @import("../../mlx/device.zig").naxStatusFrom;
pub const naxStatus = @import("../../mlx/device.zig").naxStatus;
pub const macosProductVersion = @import("../../mlx/device.zig").macosProductVersion;
pub const gpuArchitecture = @import("../../mlx/device.zig").gpuArchitecture;
pub const naxAvailable = @import("../../mlx/device.zig").naxAvailable;
// sdk/blocks/linear/verify_qmm_nax.zig
pub const verifyQmmNaxAvailable = @import("../../sdk/blocks/linear/verify_qmm_nax.zig").verifyQmmNaxAvailable;
pub const verifyQmmTile = @import("../../sdk/blocks/linear/verify_qmm_nax.zig").verifyQmmTile;
pub const verifyQmmNaxEnabled = @import("../../sdk/blocks/linear/verify_qmm_nax.zig").verifyQmmNaxEnabled;
pub const verifyQmmNaxEnabledForMFrom = @import("../../sdk/blocks/linear/verify_qmm_nax.zig").verifyQmmNaxEnabledForMFrom;
pub const verifyQmmNaxEnabledForM = @import("../../sdk/blocks/linear/verify_qmm_nax.zig").verifyQmmNaxEnabledForM;
// sdk/blocks/attention/sdpa.zig
pub const naxSdpaPreferred = @import("../../sdk/blocks/attention/sdpa.zig").naxSdpaPreferred;
pub const sdpaForceFusedFor = @import("../../sdk/blocks/attention/sdpa.zig").sdpaForceFusedFor;
pub const sdpaForceFused = @import("../../sdk/blocks/attention/sdpa.zig").sdpaForceFused;
pub const splitCausalSdpa = @import("../../sdk/blocks/attention/sdpa.zig").splitCausalSdpa;
pub const splitMaskedSdpa256 = @import("../../sdk/blocks/attention/sdpa.zig").splitMaskedSdpa256;
// sdk/blocks/attention/attn256.zig
pub const ATTN256_KERNEL_HEADER = @import("../../sdk/blocks/attention/attn256.zig").ATTN256_KERNEL_HEADER;
pub const fused256Enabled = @import("../../sdk/blocks/attention/attn256.zig").fused256Enabled;
pub const Fused256CausalMode = @import("../../sdk/blocks/attention/attn256.zig").Fused256CausalMode;
pub const FUSED256_DEFAULT_DISPATCH_BUDGET = @import("../../sdk/blocks/attention/attn256.zig").FUSED256_DEFAULT_DISPATCH_BUDGET;
pub const fused256DispatchBudget = @import("../../sdk/blocks/attention/attn256.zig").fused256DispatchBudget;
pub const fused256KvChunkLen = @import("../../sdk/blocks/attention/attn256.zig").fused256KvChunkLen;
pub const fused256CausalMode = @import("../../sdk/blocks/attention/attn256.zig").fused256CausalMode;
pub const prefillHeadDimFused = @import("../../sdk/blocks/attention/attn256.zig").prefillHeadDimFused;
pub const FUSED256_MIN_Q_LEN = @import("../../sdk/blocks/attention/attn256.zig").FUSED256_MIN_Q_LEN;
pub const fusedSdpa256Prefill = @import("../../sdk/blocks/attention/attn256.zig").fusedSdpa256Prefill;
pub const fusedSinkAttnPrefill = @import("../../sdk/blocks/attention/attn256.zig").fusedSinkAttnPrefill;
pub const fusedSdpa256Masked = @import("../../sdk/blocks/attention/attn256.zig").fusedSdpa256Masked;
// sdk/blocks/norms/residual_norm.zig
pub const ResidualNormOpts = @import("../../sdk/blocks/norms/residual_norm.zig").ResidualNormOpts;
pub const AddNormResult = @import("../../sdk/blocks/norms/residual_norm.zig").AddNormResult;
pub const fusedResidualNorm = @import("../../sdk/blocks/norms/residual_norm.zig").fusedResidualNorm;
pub const fusedAddRmsNorm = @import("../../sdk/blocks/norms/residual_norm.zig").fusedAddRmsNorm;
// plugins/qwen4_exp/qsa/gather.zig
pub const qsaGatherEnabled = @import("../../plugins/qwen4_exp/qsa/gather.zig").qsaGatherEnabled;
pub const QSA_GATHER_MIN_KV_DEFAULT = @import("../../plugins/qwen4_exp/qsa/gather.zig").QSA_GATHER_MIN_KV_DEFAULT;
pub const qsaGatherMinKv = @import("../../plugins/qwen4_exp/qsa/gather.zig").qsaGatherMinKv;
pub const QSA_GATHER_MIN_KV_NAX = @import("../../plugins/qwen4_exp/qsa/gather.zig").QSA_GATHER_MIN_KV_NAX;
pub const qsaPrefillGatherMinKv = @import("../../plugins/qwen4_exp/qsa/gather.zig").qsaPrefillGatherMinKv;
pub const QSA_GATHER_BK_DEFAULT = @import("../../plugins/qwen4_exp/qsa/gather.zig").QSA_GATHER_BK_DEFAULT;
pub const qsaDecodeGatherEnabled = @import("../../plugins/qwen4_exp/qsa/gather.zig").qsaDecodeGatherEnabled;
pub const qsaBatchedGatherEnabled = @import("../../plugins/qwen4_exp/qsa/gather.zig").qsaBatchedGatherEnabled;
pub const qsaBatchedGatherOn = @import("../../plugins/qwen4_exp/qsa/gather.zig").qsaBatchedGatherOn;
pub const qsaBatchedGatherFloor = @import("../../plugins/qwen4_exp/qsa/gather.zig").qsaBatchedGatherFloor;
pub const qsaHistoryShareEnabled = @import("../../plugins/qwen4_exp/qsa/gather.zig").qsaHistoryShareEnabled;
pub const qsaNaxEnabled = @import("../../plugins/qwen4_exp/qsa/gather.zig").qsaNaxEnabled;
pub const qsaNaxEligibleFrom = @import("../../plugins/qwen4_exp/qsa/gather.zig").qsaNaxEligibleFrom;
pub const qsaNaxEligible = @import("../../plugins/qwen4_exp/qsa/gather.zig").qsaNaxEligible;
pub const qsaNaxProbeFailed = @import("../../plugins/qwen4_exp/qsa/gather.zig").qsaNaxProbeFailed;
pub const qsaNaxProbeReset = @import("../../plugins/qwen4_exp/qsa/gather.zig").qsaNaxProbeReset;
pub const qsaNaxKernelBuilds = @import("../../plugins/qwen4_exp/qsa/gather.zig").qsaNaxKernelBuilds;
pub const qsaNaxArm = @import("../../plugins/qwen4_exp/qsa/gather.zig").qsaNaxArm;
pub const gatherQsa256 = @import("../../plugins/qwen4_exp/qsa/gather.zig").gatherQsa256;
// plugins/qwen4_exp/qsa/select.zig
pub const qsaSelectTgFor = @import("../../plugins/qwen4_exp/qsa/select.zig").qsaSelectTgFor;
pub const qsaSelectCfgBuilds = @import("../../plugins/qwen4_exp/qsa/select.zig").qsaSelectCfgBuilds;
pub const qsaSelectKernelEnabled = @import("../../plugins/qwen4_exp/qsa/select.zig").qsaSelectKernelEnabled;
pub const qsaSelectTopBlocks = @import("../../plugins/qwen4_exp/qsa/select.zig").qsaSelectTopBlocks;
pub const QSA_SELECT_SPLIT_MIN_NB = @import("../../plugins/qwen4_exp/qsa/select.zig").QSA_SELECT_SPLIT_MIN_NB;
pub const QSA_SELECT_SPLIT_MAX_ROWS = @import("../../plugins/qwen4_exp/qsa/select.zig").QSA_SELECT_SPLIT_MAX_ROWS;
pub const QSA_SELECT_SPLIT_G = @import("../../plugins/qwen4_exp/qsa/select.zig").QSA_SELECT_SPLIT_G;
pub const qsaSelectDigitBits = @import("../../plugins/qwen4_exp/qsa/select.zig").qsaSelectDigitBits;
pub const qsaSelectSplitEnabled = @import("../../plugins/qwen4_exp/qsa/select.zig").qsaSelectSplitEnabled;
pub const qsaSelectSplitServes = @import("../../plugins/qwen4_exp/qsa/select.zig").qsaSelectSplitServes;
pub const qsaSelectSplitCfgBuilds = @import("../../plugins/qwen4_exp/qsa/select.zig").qsaSelectSplitCfgBuilds;
// plugins/qwen4_exp/mtp.zig
pub const Qwen4Mtp = @import("../../plugins/qwen4_exp/mtp.zig").Qwen4Mtp;
// sdk/blocks/linear/verify_projection.zig
pub const VerifyKernel = @import("../../sdk/blocks/linear/verify_projection.zig").VerifyKernel;
pub const mtpCoarsePairs = @import("../../sdk/blocks/linear/verify_projection.zig").mtpCoarsePairs;
pub const verifySharedFor = @import("../../sdk/blocks/linear/verify_projection.zig").verifySharedFor;
pub const dualDieArch = @import("../../sdk/blocks/linear/verify_projection.zig").dualDieArch;
pub const verifySharedCapable = @import("../../sdk/blocks/linear/verify_projection.zig").verifySharedCapable;
pub const verifySharedHardware = @import("../../sdk/blocks/linear/verify_projection.zig").verifySharedHardware;
// plugins/qwen4_exp/qsa/score.zig
pub const QsaScoreLayout = @import("../../plugins/qwen4_exp/qsa/score.zig").QsaScoreLayout;
pub const qsaScoreKernelBuilt = @import("../../plugins/qwen4_exp/qsa/score.zig").qsaScoreKernelBuilt;
pub const qsaScoreKernelBuilds = @import("../../plugins/qwen4_exp/qsa/score.zig").qsaScoreKernelBuilds;
pub const qsaScoreFusedEnabled = @import("../../plugins/qwen4_exp/qsa/score.zig").qsaScoreFusedEnabled;
pub const qsaScoreFusedActive = @import("../../plugins/qwen4_exp/qsa/score.zig").qsaScoreFusedActive;
pub const qsaScoreFusedProbeFailed = @import("../../plugins/qwen4_exp/qsa/score.zig").qsaScoreFusedProbeFailed;
pub const qsaScoreFusedProbeReset = @import("../../plugins/qwen4_exp/qsa/score.zig").qsaScoreFusedProbeReset;
pub const qsaScoreFusedArm = @import("../../plugins/qwen4_exp/qsa/score.zig").qsaScoreFusedArm;
pub const qsaScoreFusedActiveFor = @import("../../plugins/qwen4_exp/qsa/score.zig").qsaScoreFusedActiveFor;
pub const qsaScoreFusedEligible = @import("../../plugins/qwen4_exp/qsa/score.zig").qsaScoreFusedEligible;
pub const qsaScoreSheetComposed = @import("../../plugins/qwen4_exp/qsa/score.zig").qsaScoreSheetComposed;
pub const qsaScoreK32tFromPooled = @import("../../plugins/qwen4_exp/qsa/score.zig").qsaScoreK32tFromPooled;
pub const qsaScoreFused = @import("../../plugins/qwen4_exp/qsa/score.zig").qsaScoreFused;
pub const qsaScoreSheet = @import("../../plugins/qwen4_exp/qsa/score.zig").qsaScoreSheet;
pub const QSA_SCORE_SHEET_BUDGET_MB_DEFAULT = @import("../../plugins/qwen4_exp/qsa/score.zig").QSA_SCORE_SHEET_BUDGET_MB_DEFAULT;
pub const qsaScoreSheetBudget = @import("../../plugins/qwen4_exp/qsa/score.zig").qsaScoreSheetBudget;
pub const qsaScoreRowsPerChunkFor = @import("../../plugins/qwen4_exp/qsa/score.zig").qsaScoreRowsPerChunkFor;
pub const qsaScoreRowsPerChunkFused = @import("../../plugins/qwen4_exp/qsa/score.zig").qsaScoreRowsPerChunkFused;
pub const qsaScoreRowsPerChunk = @import("../../plugins/qwen4_exp/qsa/score.zig").qsaScoreRowsPerChunk;
pub const qsaPrefillTransientBytes = @import("../../plugins/qwen4_exp/qsa/score.zig").qsaPrefillTransientBytes;
// plugins/qwen4_exp/qsa/sparse_attn.zig
pub const QSA_ATTN_NSPLIT_DEFAULT = @import("../../plugins/qwen4_exp/qsa/sparse_attn.zig").QSA_ATTN_NSPLIT_DEFAULT;
pub const QSA_ATTN_MAX_NSPLIT = @import("../../plugins/qwen4_exp/qsa/sparse_attn.zig").QSA_ATTN_MAX_NSPLIT;
pub const QSA_ATTN_MIN_S_DEFAULT = @import("../../plugins/qwen4_exp/qsa/sparse_attn.zig").QSA_ATTN_MIN_S_DEFAULT;
pub const QSA_ATTN_BATCHED_MIN_S_DEFAULT = @import("../../plugins/qwen4_exp/qsa/sparse_attn.zig").QSA_ATTN_BATCHED_MIN_S_DEFAULT;
pub const qsaAttnNSplit = @import("../../plugins/qwen4_exp/qsa/sparse_attn.zig").qsaAttnNSplit;
pub const qsaAttnMinS = @import("../../plugins/qwen4_exp/qsa/sparse_attn.zig").qsaAttnMinS;
pub const qsaAttnBatchedMinS = @import("../../plugins/qwen4_exp/qsa/sparse_attn.zig").qsaAttnBatchedMinS;
pub const qsaSparseAttnServes = @import("../../plugins/qwen4_exp/qsa/sparse_attn.zig").qsaSparseAttnServes;
pub const qsaFusedVerifyServes = @import("../../plugins/qwen4_exp/qsa/sparse_attn.zig").qsaFusedVerifyServes;
pub const qsaWantsBlocks = @import("../../plugins/qwen4_exp/qsa/sparse_attn.zig").qsaWantsBlocks;
pub const QSA_ATTN_BK = @import("../../plugins/qwen4_exp/qsa/sparse_attn.zig").QSA_ATTN_BK;
pub const qsaAttnBk = @import("../../plugins/qwen4_exp/qsa/sparse_attn.zig").qsaAttnBk;
pub const qsaAttnKernelEnabled = @import("../../plugins/qwen4_exp/qsa/sparse_attn.zig").qsaAttnKernelEnabled;
pub const qsaAttnDenseEnabled = @import("../../plugins/qwen4_exp/qsa/sparse_attn.zig").qsaAttnDenseEnabled;
pub const qsaAttnCfgBuilds = @import("../../plugins/qwen4_exp/qsa/sparse_attn.zig").qsaAttnCfgBuilds;
pub const qsaSparseAttn = @import("../../plugins/qwen4_exp/qsa/sparse_attn.zig").qsaSparseAttn;
pub const qsaSparseAttnFloor = @import("../../plugins/qwen4_exp/qsa/sparse_attn.zig").qsaSparseAttnFloor;
pub const QsaSlotServed = @import("../../plugins/qwen4_exp/qsa/sparse_attn.zig").QsaSlotServed;
pub const qsaSlotAttn = @import("../../plugins/qwen4_exp/qsa/sparse_attn.zig").qsaSlotAttn;
pub const qsaMaskArm = @import("../../plugins/qwen4_exp/qsa/sparse_attn.zig").qsaMaskArm;
// plugins/qwen4_exp/qsa/arms.zig
pub const OneShotBits = @import("../../plugins/qwen4_exp/qsa/arms.zig").OneShotBits;
pub const QsaArm = @import("../../plugins/qwen4_exp/qsa/arms.zig").QsaArm;
pub const qsaWidthBucket = @import("../../plugins/qwen4_exp/qsa/arms.zig").qsaWidthBucket;
pub const QsaDeclineReason = @import("../../plugins/qwen4_exp/qsa/arms.zig").QsaDeclineReason;
pub const QsaDecline = @import("../../plugins/qwen4_exp/qsa/arms.zig").QsaDecline;
pub const QsaArmCounts = @import("../../plugins/qwen4_exp/qsa/arms.zig").QsaArmCounts;
// mlx/array.zig
pub const materializedOwnedCopy = @import("../../mlx/array.zig").materializedOwnedCopy;
// plugins/qwen4_exp/qsa/decode_gather.zig
pub const qsaDecodeGatherAttn = @import("../../plugins/qwen4_exp/qsa/decode_gather.zig").qsaDecodeGatherAttn;
pub const qsaBatchedGatherAttn = @import("../../plugins/qwen4_exp/qsa/decode_gather.zig").qsaBatchedGatherAttn;
// plugins/qwen4_exp/qsa/verify_gather.zig
pub const qsaVerifyGatherEnabled = @import("../../plugins/qwen4_exp/qsa/verify_gather.zig").qsaVerifyGatherEnabled;
pub const QSA_VERIFY_GATHER_MIN_KV_DENSE = @import("../../plugins/qwen4_exp/qsa/verify_gather.zig").QSA_VERIFY_GATHER_MIN_KV_DENSE;
pub const QSA_VERIFY_GATHER_MIN_KV_QUANT = @import("../../plugins/qwen4_exp/qsa/verify_gather.zig").QSA_VERIFY_GATHER_MIN_KV_QUANT;
pub const qsaVerifyGatherMinKvFor = @import("../../plugins/qwen4_exp/qsa/verify_gather.zig").qsaVerifyGatherMinKvFor;
pub const QsaVerifyGeom = @import("../../plugins/qwen4_exp/qsa/verify_gather.zig").QsaVerifyGeom;
pub const QsaVerifyPlan = @import("../../plugins/qwen4_exp/qsa/verify_gather.zig").QsaVerifyPlan;
pub const qsaVerifyPlanHost = @import("../../plugins/qwen4_exp/qsa/verify_gather.zig").qsaVerifyPlanHost;
pub const qsaDecodeGatherIdsHost = @import("../../plugins/qwen4_exp/qsa/verify_gather.zig").qsaDecodeGatherIdsHost;
pub const QsaSlotPlan = @import("../../plugins/qwen4_exp/qsa/verify_gather.zig").QsaSlotPlan;
pub const qsaBatchedPlanHost = @import("../../plugins/qwen4_exp/qsa/verify_gather.zig").qsaBatchedPlanHost;
pub const qsaVerifyGatherAttn = @import("../../plugins/qwen4_exp/qsa/verify_gather.zig").qsaVerifyGatherAttn;
// plugins/qwen4_exp/qsa/mask.zig
pub const qsaMaskFromBlocks = @import("../../plugins/qwen4_exp/qsa/mask.zig").qsaMaskFromBlocks;
// engines/mlx/kv/cache.zig
pub const KVCacheEntry = @import("kv/cache.zig").KVCacheEntry;
pub const newEmptyKVEntry = @import("kv/cache.zig").newEmptyKVEntry;
pub const resetKVEntry = @import("kv/cache.zig").resetKVEntry;
pub const KVCache = @import("kv/cache.zig").KVCache;
pub const KVCacheSnapshot = @import("kv/cache.zig").KVCacheSnapshot;
// sdk/blocks/attention/kv_view.zig
pub const DenseKVView = @import("../../sdk/blocks/attention/kv_view.zig").DenseKVView;
pub const SlidingView = @import("../../sdk/blocks/attention/kv_view.zig").SlidingView;
// engines/mlx/kv/sliding.zig
pub const SLIDING_TRIM_UNBOUNDED = @import("kv/sliding.zig").SLIDING_TRIM_UNBOUNDED;
pub const slidingTailSpan = @import("kv/sliding.zig").slidingTailSpan;
pub const slidingViewFor = @import("kv/sliding.zig").slidingViewFor;
// plugins/qwen4_exp/ple.zig
pub const claimPleSpecCapture = @import("../../plugins/qwen4_exp/ple.zig").claimPleSpecCapture;
pub const plePrevAfterAccept = @import("../../plugins/qwen4_exp/ple.zig").plePrevAfterAccept;
pub const fillPleSpecTokens = @import("../../plugins/qwen4_exp/ple.zig").fillPleSpecTokens;
pub const PlePending = @import("../../plugins/qwen4_exp/ple.zig").PlePending;
// engines/mlx/forward_ctx.zig
pub const SpecTree = @import("forward_ctx.zig").SpecTree;
pub const CaptureLayers = @import("forward_ctx.zig").CaptureLayers;
pub const ForwardCtx = @import("forward_ctx.zig").ForwardCtx;
// engines/mlx/ssm/entry.zig
pub const SSMCacheEntry = @import("ssm/entry.zig").SSMCacheEntry;
pub const ssmEntryBytes = @import("ssm/entry.zig").ssmEntryBytes;
pub const SSMCacheEntrySnapshot = @import("ssm/entry.zig").SSMCacheEntrySnapshot;
pub const ssmSnapshot = @import("ssm/entry.zig").ssmSnapshot;
pub const ssmSnapshotDeinit = @import("ssm/entry.zig").ssmSnapshotDeinit;
pub const ssmRestore = @import("ssm/entry.zig").ssmRestore;
pub const ssmFreeSpecCapture = @import("ssm/entry.zig").ssmFreeSpecCapture;
pub const ssmCommitTreePath = @import("ssm/entry.zig").ssmCommitTreePath;
pub const ssmRollbackFromCapture = @import("ssm/entry.zig").ssmRollbackFromCapture;
// plugins/qwen4_exp/qsa/history.zig
pub const QSA_RING_ROWS = @import("../../plugins/qwen4_exp/qsa/history.zig").QSA_RING_ROWS;
pub const QSA_MAX_COMPRESS_RATIO = @import("../../plugins/qwen4_exp/qsa/history.zig").QSA_MAX_COMPRESS_RATIO;
pub const checkQsaCompressRatio = @import("../../plugins/qwen4_exp/qsa/history.zig").checkQsaCompressRatio;
pub const QsaHeadMark = @import("../../plugins/qwen4_exp/qsa/history.zig").QsaHeadMark;
pub const QSA_HEAD_MARKS_MAX = @import("../../plugins/qwen4_exp/qsa/history.zig").QSA_HEAD_MARKS_MAX;
pub const QsaHeadMarkSet = @import("../../plugins/qwen4_exp/qsa/history.zig").QsaHeadMarkSet;
pub const ssmFreeQsaState = @import("../../plugins/qwen4_exp/qsa/history.zig").ssmFreeQsaState;
pub const reserveQsaHistory = @import("../../plugins/qwen4_exp/qsa/history.zig").reserveQsaHistory;
pub const reserveQsaHistoryWithHead = @import("../../plugins/qwen4_exp/qsa/history.zig").reserveQsaHistoryWithHead;
pub const qsaTestAppendPool = @import("../../plugins/qwen4_exp/qsa/history.zig").qsaTestAppendPool;
// engines/mlx/ssm/group.zig
pub const nextSsmMemberGen = @import("ssm/group.zig").nextSsmMemberGen;
pub const PersistentSsmGroup = @import("ssm/group.zig").PersistentSsmGroup;
pub const ssmBumpMemberGen = @import("ssm/group.zig").ssmBumpMemberGen;
pub const ssmBumpMemberGenEntries = @import("ssm/group.zig").ssmBumpMemberGenEntries;
pub const ssmDetachFromGroup = @import("ssm/group.zig").ssmDetachFromGroup;
// engines/mlx/ssm/checkpoint.zig
pub const SSMCheckpoint = @import("ssm/checkpoint.zig").SSMCheckpoint;
pub const captureSsmCheckpoint = @import("ssm/checkpoint.zig").captureSsmCheckpoint;
pub const compactConvStateEnabled = @import("ssm/checkpoint.zig").compactConvStateEnabled;
pub const shareSsmCheckpoint = @import("ssm/checkpoint.zig").shareSsmCheckpoint;
pub const restoreSsmCheckpoint = @import("ssm/checkpoint.zig").restoreSsmCheckpoint;
pub const ssmCheckpointBytes = @import("ssm/checkpoint.zig").ssmCheckpointBytes;
pub const qsaHistoryBytesHeld = @import("ssm/checkpoint.zig").qsaHistoryBytesHeld;
pub const ThinPolicy = @import("ssm/checkpoint.zig").ThinPolicy;
pub const spanPreservingDropIndex = @import("ssm/checkpoint.zig").spanPreservingDropIndex;
pub const positionDropIndexLowAnchored = @import("ssm/checkpoint.zig").positionDropIndexLowAnchored;
pub const ssmCheckpointDropIndex = @import("ssm/checkpoint.zig").ssmCheckpointDropIndex;
pub const ssmCheckpointDropIndexLowAnchored = @import("ssm/checkpoint.zig").ssmCheckpointDropIndexLowAnchored;
pub const positionDropIndex = @import("ssm/checkpoint.zig").positionDropIndex;
pub const positionDropIndexUsize = @import("ssm/checkpoint.zig").positionDropIndexUsize;
// plugins/qwen4_exp/qsa/pooled.zig
const QsaPooledRope = @import("../../plugins/qwen4_exp/qsa/pooled.zig").QsaPooledRope;
const QsaBlockConsts = @import("../../plugins/qwen4_exp/qsa/pooled.zig").QsaBlockConsts;
pub const qsaIdxRopeFusedEnabled = @import("../../plugins/qwen4_exp/qsa/pooled.zig").qsaIdxRopeFusedEnabled;
pub const qsaPoolRopePipelineCount = @import("../../plugins/qwen4_exp/qsa/pooled.zig").qsaPoolRopePipelineCount;
pub const qsaPoolNormRopeFused = @import("../../plugins/qwen4_exp/qsa/pooled.zig").qsaPoolNormRopeFused;
// plugins/qwen4_exp/qsa/checkpoint.zig
pub const ssmAuxIsQsaHistory = @import("../../plugins/qwen4_exp/qsa/checkpoint.zig").ssmAuxIsQsaHistory;
pub const snapshotHasQsaHistory = @import("../../plugins/qwen4_exp/qsa/checkpoint.zig").snapshotHasQsaHistory;
pub const checkpointHasQsaHistory = @import("../../plugins/qwen4_exp/qsa/checkpoint.zig").checkpointHasQsaHistory;
pub const checkpointHasQsaPooled = @import("../../plugins/qwen4_exp/qsa/checkpoint.zig").checkpointHasQsaPooled;
pub const checkpointQsaAuxRows = @import("../../plugins/qwen4_exp/qsa/checkpoint.zig").checkpointQsaAuxRows;
pub const checkpointQsaCoversPos = @import("../../plugins/qwen4_exp/qsa/checkpoint.zig").checkpointQsaCoversPos;
pub const qsaEntrySatisfiesForward = @import("../../plugins/qwen4_exp/qsa/checkpoint.zig").qsaEntrySatisfiesForward;
pub const qsaRestoreSatisfiesForward = @import("../../plugins/qwen4_exp/qsa/checkpoint.zig").qsaRestoreSatisfiesForward;
pub const checkpointListHasQsaPooled = @import("../../plugins/qwen4_exp/qsa/checkpoint.zig").checkpointListHasQsaPooled;
pub const attachQsaHistoryToLatest = @import("../../plugins/qwen4_exp/qsa/checkpoint.zig").attachQsaHistoryToLatest;
pub const QSA_HANDOFF_MAX_SLACK_ROWS = @import("../../plugins/qwen4_exp/qsa/checkpoint.zig").QSA_HANDOFF_MAX_SLACK_ROWS;
pub const handoffQsaHistoryToLatest = @import("../../plugins/qwen4_exp/qsa/checkpoint.zig").handoffQsaHistoryToLatest;
pub const attachQsaHistoryOnHandoff = @import("../../plugins/qwen4_exp/qsa/checkpoint.zig").attachQsaHistoryOnHandoff;
pub const QsaAttachMode = @import("../../plugins/qwen4_exp/qsa/checkpoint.zig").QsaAttachMode;
pub const qsaHandoffExceedsSlack = @import("../../plugins/qwen4_exp/qsa/checkpoint.zig").qsaHandoffExceedsSlack;
pub const keepOnlyLatestQsaHistory = @import("../../plugins/qwen4_exp/qsa/checkpoint.zig").keepOnlyLatestQsaHistory;
pub const entriesHaveQsaHistory = @import("../../plugins/qwen4_exp/qsa/checkpoint.zig").entriesHaveQsaHistory;
pub const applyQsaHistoryAt = @import("../../plugins/qwen4_exp/qsa/checkpoint.zig").applyQsaHistoryAt;
pub const sliceQsaHistoryOntoCheckpoint = @import("../../plugins/qwen4_exp/qsa/checkpoint.zig").sliceQsaHistoryOntoCheckpoint;
// engines/mlx/prompt_cache.zig
pub const PrefillCache = @import("prompt_cache.zig").PrefillCache;
// engines/mlx/layer_weights.zig
const LayerWeights = @import("layer_weights.zig").LayerWeights;
pub const QW = @import("layer_weights.zig").QW;
// plugins/glm5_next/weights.zig
pub const DsaWeights = @import("../../plugins/glm5_next/weights.zig").DsaWeights;
// plugins/qwen4_exp/weights.zig
const HcWeights = @import("../../plugins/qwen4_exp/weights.zig").HcWeights;
// sdk/blocks/mlp/dense.zig
// engines/mlx/ane_prefill.zig
const AneChanRest = @import("ane_prefill.zig").AneChanRest;
// sdk/blocks/moe/weights.zig
pub const MoeMlpWeights = @import("../../sdk/blocks/moe/weights.zig").MoeMlpWeights;
// engines/mlx/forward/layered.zig
const MoeLayerWeights = @import("forward/layered.zig").MoeLayerWeights;
// sdk/quant/params.zig
pub const QuantParams = @import("../../sdk/quant/params.zig").QuantParams;
const BitsCache = @import("../../sdk/quant/params.zig").BitsCache;
pub const affineParamsFromGeometry = @import("../../sdk/quant/params.zig").affineParamsFromGeometry;
pub const quantParamsFromGeometry = @import("../../sdk/quant/params.zig").quantParamsFromGeometry;
pub const computeQuantParams = @import("../../sdk/quant/params.zig").computeQuantParams;
// sdk/blocks/attention/kv_quant_attn.zig
pub const qkvAttnDecodeKernel = @import("../../sdk/blocks/attention/kv_quant_attn.zig").qkvAttnDecodeKernel;
pub const QKV_VERIFY_MAX_TQ = @import("../../sdk/blocks/attention/kv_quant_attn.zig").QKV_VERIFY_MAX_TQ;
pub const qkvAttnVerifyKernel = @import("../../sdk/blocks/attention/kv_quant_attn.zig").qkvAttnVerifyKernel;
pub const qkvMppWins = @import("../../sdk/blocks/attention/kv_quant_attn.zig").qkvMppWins;
pub const qkvMppSplits = @import("../../sdk/blocks/attention/kv_quant_attn.zig").qkvMppSplits;
pub const qkvAttnMppKernel = @import("../../sdk/blocks/attention/kv_quant_attn.zig").qkvAttnMppKernel;
pub const KV_ATTN_FUSED_MIN_TK = @import("../../sdk/blocks/attention/kv_quant_attn.zig").KV_ATTN_FUSED_MIN_TK;
// sdk/blocks/moe/routing.zig
pub const groupLimitedRouting = @import("../../sdk/blocks/moe/routing.zig").groupLimitedRouting;
// sdk/blocks/embedding.zig
pub const gatherQuantizedRows = @import("../../sdk/blocks/embedding.zig").gatherQuantizedRows;
// sdk/blocks/mlp/activations.zig
pub const reluSquaredOp = @import("../../sdk/blocks/mlp/activations.zig").reluSquaredOp;
pub const clampedSwiGLU = @import("../../sdk/blocks/mlp/activations.zig").clampedSwiGLU;
// engines/mlx/diag/profile.zig
pub const dispatchProbeCount = @import("diag/profile.zig").dispatchProbeCount;
pub const decodeProfReport = @import("diag/profile.zig").decodeProfReport;
// sdk/blocks/moe/decode.zig
// sdk/blocks/attention/masks.zig
pub const encoderBandMask = @import("../../sdk/blocks/attention/masks.zig").encoderBandMask;
// sdk/quant/mxfp8.zig
pub const lmHeadMxfp8On = @import("../../sdk/quant/mxfp8.zig").lmHeadMxfp8On;
// plugins/glm5_next/load.zig
pub const loadDsaWeightsAt = @import("../../plugins/glm5_next/load.zig").loadDsaWeightsAt;
// sdk/blocks/moe/router_kernel.zig
pub const RouterGate = @import("../../sdk/blocks/moe/router_kernel.zig").RouterGate;
// mlx/dtype.zig
const DtypeCastCache = @import("../../mlx/dtype.zig").DtypeCastCache;
// sdk/diag/standin.zig
pub const Standin = @import("../../sdk/diag/standin.zig").Standin;
pub const qwen4Standin = @import("../../sdk/diag/standin.zig").qwen4Standin;
// engines/mlx/env_transformer.zig
pub const diagEnvValueOn = @import("env_transformer.zig").diagEnvValueOn;
// sdk/blocks/linear/fused_rows.zig
// sdk/blocks/attention/decode_attn_quant.zig
const AttnDqCopy = @import("../../sdk/blocks/attention/decode_attn_quant.zig").AttnDqCopy;
pub const DECODE_ATTN_QUANT_DEFAULT = @import("../../sdk/blocks/attention/decode_attn_quant.zig").DECODE_ATTN_QUANT_DEFAULT;
pub const decodeAttnQuantExplicit = @import("../../sdk/blocks/attention/decode_attn_quant.zig").decodeAttnQuantExplicit;
pub const decodeAttnQuantEnabled = @import("../../sdk/blocks/attention/decode_attn_quant.zig").decodeAttnQuantEnabled;
pub const weightsHaveDenseAttnProj = @import("../../sdk/blocks/attention/decode_attn_quant.zig").weightsHaveDenseAttnProj;
// sdk/blocks/rope/qk_norm_rope.zig
const QkAngleSlot = @import("../../sdk/blocks/rope/qk_norm_rope.zig").QkAngleSlot;
pub const qkNormRopeFusedWanted = @import("../../sdk/blocks/rope/qk_norm_rope.zig").qkNormRopeFusedWanted;
pub const fusedQkNormRope = @import("../../sdk/blocks/rope/qk_norm_rope.zig").fusedQkNormRope;
pub const ropeAngleRow = @import("../../sdk/blocks/rope/qk_norm_rope.zig").ropeAngleRow;
pub const ropeAngleRows = @import("../../sdk/blocks/rope/qk_norm_rope.zig").ropeAngleRows;
pub const fusedQkNormRope256 = @import("../../sdk/blocks/rope/qk_norm_rope.zig").fusedQkNormRope256;
// sdk/blocks/gdn/decode_fused.zig
pub const gdnDecodeFusedEnabled = @import("../../sdk/blocks/gdn/decode_fused.zig").gdnDecodeFusedEnabled;
pub const GdnPrework = @import("../../sdk/blocks/gdn/decode_fused.zig").GdnPrework;
pub const GDN_FUSED_MAX_ROWS = @import("../../sdk/blocks/gdn/decode_fused.zig").GDN_FUSED_MAX_ROWS;
pub const GdnPreworkArgs = @import("../../sdk/blocks/gdn/decode_fused.zig").GdnPreworkArgs;
pub const gdnPreworkFused = @import("../../sdk/blocks/gdn/decode_fused.zig").gdnPreworkFused;
pub const gdnNormGateFused = @import("../../sdk/blocks/gdn/decode_fused.zig").gdnNormGateFused;
// sdk/blocks/mlp/fused_activations.zig
pub const swigluSigTable = @import("../../sdk/blocks/mlp/fused_activations.zig").swigluSigTable;
pub const fusedSwiGLU = @import("../../sdk/blocks/mlp/fused_activations.zig").fusedSwiGLU;
pub const fusedClampedSwiGLU = @import("../../sdk/blocks/mlp/fused_activations.zig").fusedClampedSwiGLU;
pub const fusedGeluErfMul = @import("../../sdk/blocks/mlp/fused_activations.zig").fusedGeluErfMul;
// sdk/blocks/moe/gather_qmv.zig
pub const GQMV_AFFINE_HEADER = @import("../../sdk/blocks/moe/gather_qmv.zig").GQMV_AFFINE_HEADER;
pub const gatherQmvGateUp = @import("../../sdk/blocks/moe/gather_qmv.zig").gatherQmvGateUp;
pub const gatherQmv = @import("../../sdk/blocks/moe/gather_qmv.zig").gatherQmv;
pub const gatherQmvAct = @import("../../sdk/blocks/moe/gather_qmv.zig").gatherQmvAct;
// plugins/qwen4_exp/hc_fused.zig
pub const HcFusedOut = @import("../../plugins/qwen4_exp/hc_fused.zig").HcFusedOut;
pub const HcPending = @import("../../plugins/qwen4_exp/hc_fused.zig").HcPending;
pub const HC_FUSED_MAX_ROWS = @import("../../plugins/qwen4_exp/hc_fused.zig").HC_FUSED_MAX_ROWS;
pub const hcReadFused = @import("../../plugins/qwen4_exp/hc_fused.zig").hcReadFused;
// sdk/blocks/moe/gather_qmv_rows.zig
pub const gatherQmvGateUpRows = @import("../../sdk/blocks/moe/gather_qmv_rows.zig").gatherQmvGateUpRows;
// sdk/blocks/moe/down_reduce.zig
pub const gatherQmvDownReduce = @import("../../sdk/blocks/moe/down_reduce.zig").gatherQmvDownReduce;
pub const gatherQmvDownReduceRows = @import("../../sdk/blocks/moe/down_reduce.zig").gatherQmvDownReduceRows;
// sdk/blocks/linear/qmv_rows.zig
pub const msvQmvRows = @import("../../sdk/blocks/linear/qmv_rows.zig").msvQmvRows;
// sdk/blocks/linear/prefill_dq_gemm.zig
pub const PREFILL_DQ_GEMM_MIN_M = @import("../../sdk/blocks/linear/prefill_dq_gemm.zig").PREFILL_DQ_GEMM_MIN_M;
pub const prefillDqGemmMinRows = @import("../../sdk/blocks/linear/prefill_dq_gemm.zig").prefillDqGemmMinRows;
pub const prefillDqGemmRowFloor = @import("../../sdk/blocks/linear/prefill_dq_gemm.zig").prefillDqGemmRowFloor;
pub const prefillDqGemmEnabled = @import("../../sdk/blocks/linear/prefill_dq_gemm.zig").prefillDqGemmEnabled;

// Vars other files reach as `transformer.X`: Zig cannot alias a var, so they stay here
// and the code that moved out reads them qualified.

/// Test seam: pin the GPU generation every generation-keyed plan reads (null = the device's).
pub var device_gen_override: ?DeviceGen = null;

/// Test seam: force the NAX availability probe (null = real probe). Only
/// ever force FALSE on non-G17 machines — forcing true would let dispatch
/// build a kernel whose matmul2d pipeline cannot be created off M5-class
/// hardware.
pub var vqmm_nax_probe_override: ?bool = null;

/// Test seam: forces BOTH hd-256 kernel arms on/off.
pub var fused256_override: ?bool = null;

pub var qsa_gather_override: ?bool = null;

pub var qsa_gather_min_kv_override: ?c_int = null;

pub var qsa_decode_gather_override: ?bool = null;

pub var qsa_batched_gather_override: ?bool = null;

pub var qsa_history_share_override: ?bool = null;

pub var mtp_coarse_pair_dispatches: u64 = 0;

pub var mtp_verify_kernel_calls: [3]u64 = @splat(0);

pub var qsa_score_fused_override: ?bool = null;

pub var qsa_attn_kernel_override: ?bool = null;

pub var qsa_verify_gather_override: ?bool = null;

pub var qsa_verify_gather_min_kv_override: ?c_int = null;

pub var deferred_ple_group_syncs: u64 = 0;

/// `--lm-head-mxfp8` / `--no-lm-head-mxfp8`; null = the default below.
pub var lm_head_mxfp8_cli: ?bool = null;

/// Set by the CLI (`--decode-attn-quant` / `--no-decode-attn-quant`) before
/// any load; a transformer-level global so every serve path (headless
/// included — the runHeadlessServe flag-eater class) sees one value.
pub var decode_attn_quant_flag: ?bool = null;

pub var decode_attn_quant_override: ?bool = null; // test seam

pub var hc_fused_override: ?bool = null;

pub var gqmv_gateup_override: ?bool = null;

pub var moe_rows_fused_override: ?bool = null;

pub var downred_override: ?bool = null;

pub var mtp_head_row_dispatches: usize = 0;

/// Test seam: forces the route on/off without the environment.
pub var prefill_dq_gemm_override: ?bool = null;

const model_mod = @import("model.zig");
const log = @import("../../host/log.zig");
const io_util_mod = @import("../../sdk/util/io_util.zig");
const round_cost_mod = @import("spec/round_cost.zig");
const group_cost_mod = @import("spec/mtp_group_cost.zig");
// NB `ane_offload`, not `*_mod`: the `?*<x>_mod.<Y>` field convention marks
// module-owned ARCH state (the dsv4 class — single-flight + spec-off); the
// ANE engine is a load-time accelerator with no per-request decode state.
const ane_offload = @import("../../sdk/ane/offload.zig");

const ModelConfig = model_mod.ModelConfig;
const QuantMode = model_mod.QuantMode;
const moe_fp4 = @import("../../sdk/blocks/moe/moe_fp4.zig");
const hc_decode2 = @import("../../sdk/blocks/hc/hc_decode2.zig");
const mxfp8_qmv = @import("../../sdk/blocks/qmm/mxfp8_qmv.zig");
const qsa_row_mask = @import("../../plugins/qwen4_exp/qsa_mask.zig");
const qsa_decode = @import("../../plugins/qwen4_exp/qsa_decode.zig");
const moe_affine4 = @import("../../plugins/qwen4_exp/moe_affine4.zig");
const glm5 = @import("../../plugins/glm5_next/glm5_next.zig");
const glm_mtp = @import("../../plugins/glm5_next/mtp.zig");
const kda_recurrence = @import("../../sdk/blocks/kda/recurrence.zig");
const dec_attn = @import("../../plugins/mimo_v2/dec_attn.zig");
const Weights = model_mod.Weights;

pub const Transformer = struct {
    config: ModelConfig,
    cache: KVCache,
    s: mlx.mlx_stream,
    allocator: std.mem.Allocator,

    emb_w: mlx.mlx_array,
    emb_s: mlx.mlx_array,
    emb_b: mlx.mlx_array,
    emb_scale: ?mlx.mlx_array,
    final_norm: mlx.mlx_array,
    lm_head_w: mlx.mlx_array,
    lm_head_s: mlx.mlx_array,
    lm_head_b: mlx.mlx_array,
    layers: []LayerWeights,

    // EmbeddingGemma sentence-transformers projection head (dense.0 →
    // dense.1, identity activations, no layer bias), applied between
    // mean-pool and L2-normalize. Borrowed from the weights map (map outlives
    // the Transformer; never freed here). Null-ctx when absent.
    dense0_w: mlx.mlx_array = .{},
    dense0_s: mlx.mlx_array = .{},
    dense0_b: mlx.mlx_array = .{},
    dense1_w: mlx.mlx_array = .{},
    dense1_s: mlx.mlx_array = .{},
    dense1_b: mlx.mlx_array = .{},

    owns_lm_head: bool,
    owns_norms: bool,
    embedding_mode: bool = false,

    /// Layer stride for the decode async-eval ladder (0 = off). Resolved once
    /// at construction from `MLX_SERVE_DECODE_ASYNC_LADDER` — a per-layer
    /// getenv would cost more than the overlap it buys.
    decode_async_ladder: u32 = 0,
    /// The qwen4_exp batched-decode ladder stride, resolved the same way (`qwen4DecodeLadderStride`).
    decode_async_ladder_qwen4: u32 = 0,

    gelu_coeff: ?mlx.mlx_array,
    gelu_inner: ?mlx.mlx_array,
    half: mlx.mlx_array,
    one: mlx.mlx_array,
    /// `rms_norm_eps` as a 0-dim f32 array, for the fused residual+RMSNorm
    /// kernel (built once — an eps array per layer per token is exactly the
    /// per-call allocation this fusion exists to remove).
    rms_eps_arr: mlx.mlx_array = .{ .ctx = null },
    /// `postNormEps()` as a 0-dim f32 array (Gemma post-norms), same reason.
    post_eps_arr: mlx.mlx_array = .{ .ctx = null },
    /// ANE prefill-MLP engine (`--ane-prefill`, perf-plan-aug-17 P5). Owned;
    /// null unless the flag is on AND the arch/machine passed the build
    /// gates in `buildAnePrefill`. Read only by `denseMLPMaybeAne`.
    ane_prefill: ?*ane_offload.AnePrefill = null,
    /// Channel-mode GPU complements (A1): per layer, the rest-channel
    /// slices the GPU computes while the ANE runs its slice. gate/up/qkv/z
    /// rests are zero-copy axis-0 VIEWS of the resident packed weights
    /// (pinned bit-exact by the "axis-0 slice VIEW" test); the down rest is
    /// an axis-1 slice, materialized at build. Owned; freed beside
    /// ane_prefill. null in row mode.
    ane_chan: ?[]AneChanRest = null,
    /// Decode-only INT8 side copies of dense (bf16/f16) attention projection
    /// weights, keyed by the source weight's ctx pointer (mlxfast native-affine
    /// class). null value = probed and ineligible, don't re-probe. Built
    /// lazily on the first [1,1,·] decode dispatch; prefill and every
    /// multi-token forward keep the authoritative dense weight.
    attn_dq: std.AutoHashMapUnmanaged(usize, ?AttnDqCopy) = .empty,
    /// `weightsHaveDenseAttnProj` at load: false = decode-attn-quant never engages.
    dense_attn_proj: bool = false,
    /// Reserved-token suppression mask (`[vocab]` bool, true = never
    /// sample), built once at load by `server`'s install step from the
    /// tokenizer's `special: true` set minus template-emitted markers and
    /// EOS ids (`tokenizer.reservedOutputIds`). Owned here, freed in deinit;
    /// samplers reference it non-owning via `SamplingParams.suppress_mask`.
    /// Null = suppression off (kill switch, no chat template, GGUF engines).
    suppress_mask: ?mlx.mlx_array = null,
    /// Measured spec round-cost table (`round_cost.Table`): per draft
    /// width, per KV bucket, fed by every MTP/DFlash round on this model and
    /// read by the EV plan in place of the fitted surface once a bucket has
    /// two measured widths. Lives on the model, not the request — a request
    /// spans only its own max_tokens. Inference thread only.
    round_cost: round_cost_mod.Table = .{},
    mtp_group_cost: group_cost_mod.Table = .{},
    cost_trace_active: bool = false,
    cost_attention: u8 = 0,
    cost_moe: u8 = 0,
    verify_laps: VerifyLaps = .{},
    /// The joined `[1, Σ(1+m_i), V]` logits of the last row-axis verify whose
    /// lm_head ran once for the whole group, so the accept side can filter the
    /// block once instead of one slice per row. Null whenever the projection
    /// ran per row; the next verify releases it.
    verify_joined_logits: mlx.mlx_array = .{ .ctx = null },
    /// Persistence key of `round_cost` (empty = not persisted).
    round_cost_key_buf: [64]u8 = undefined,
    round_cost_key_len: u8 = 0,
    /// MTP depth cap WITHOUT the per-silicon row (explicit --mtp-max-depth, else
    /// the adaptive default), set at load. The row-resolved cap travels as
    /// `mtp_depth` through server -> slot -> Generator and reads there like
    /// an explicit flag, so the row-less value needs its own carrier.
    mtp_depth_free: u32 = 0,
    /// Measured ms of a row-exact MTP round by depth (index = drafts), shared
    /// by every request on this model: the cost side of `Generator.mtpExactDepth`.
    mtp_depth_ms: [8]f32 = @splat(0),
    /// First-draft acceptance of row-exact MTP rounds on this model: where
    /// each request's own estimate starts.
    mtp_depth_acc: f32 = 0.8,
    mtp_depth_rounds: u32 = 0,
    /// Per-token cos|sin rows for the fused QK-norm+RoPE kernel, one slot per
    /// rope family (0 = default/sliding base, 1 = YaRN freqs). Keyed by
    /// offset: every layer of one decode step shares the row, the next token
    /// replaces it.
    qk_angle_cache: [2]QkAngleSlot = .{ .{}, .{} },
    three: ?mlx.mlx_array,
    neg_one: ?mlx.mlx_array,

    // Gemma 4 PLE (Per-Layer Embeddings) global weights
    ple_emb_w: mlx.mlx_array, // embed_tokens_per_layer
    ple_emb_s: mlx.mlx_array,
    ple_emb_b: mlx.mlx_array,
    ple_proj_w: mlx.mlx_array, // per_layer_model_projection
    ple_proj_s: mlx.mlx_array,
    ple_proj_b: mlx.mlx_array,
    ple_proj_norm: mlx.mlx_array, // per_layer_projection_norm
    ple_proj_quantized: bool, // whether per_layer_model_projection is quantized
    // Gemma 4: logit softcapping scalar and v_norm weight (ones)
    softcap_scalar: ?mlx.mlx_array,
    v_norm_weight: ?mlx.mlx_array, // ones(head_dim) for param-free RMS norm
    v_norm_weight_global: ?mlx.mlx_array, // ones(global_head_dim)

    // ones(hidden_size) for the weight-less RMS norms (MuseGlimmer's normed embeddings, qwen4_exp).
    ones_hidden: ?mlx.mlx_array = null,

    // Proportional RoPE frequencies for global/full attention layers (Gemma 4)
    rope_freqs_global: ?mlx.mlx_array,

    // YaRN RoPE scaling. `rope_freqs_yarn` is the per-dim RoPE DENOMINATOR
    // array handed to mlx_fast_rope (angle = pos/freqs), `yarn_mscale` the
    // [head_dim] vector carrying attention_factor on the rotated dims (1.0 on
    // the pass-through tail), and `yarn_inv_freq` the same spectrum as the
    // HOST inverse frequencies the M-RoPE cos/sin tables and the QSA pooled-key
    // ladder are filled from (angle = pos×inv_freq). All three null on
    // unscaled checkpoints. laguna uses them on full-attention layers only.
    rope_freqs_yarn: ?mlx.mlx_array = null,
    yarn_mscale: ?mlx.mlx_array = null,
    yarn_inv_freq: ?[]const f64 = null,
    // `yarn_mscale` narrowed to the activation dtype, built on first use. See
    // `constTableAs` — the f32 table is what a bf16 forward must NOT multiply
    // by directly.
    yarn_mscale_cast: DtypeCastCache = .{},

    /// An mlx-stream pack (`arch/mlx_stream.zig`: DeepSeek-V4.1's EXL3 repack or MiMo-V2.6-Flash): the plugin's module.
    mlx_stream_ext: ?*mlx_stream_mod.Model = null,
    /// A pack served through its plugin's table (`engines/mlx/plugin/driver.zig`): every forward goes there.
    plugin: ?*plugin_driver.Bound = null,

    // Qwen3.8-Flash-Next (qwen4_exp): the n-gram hash + mmapped table are
    // module-owned (serial, spec-off); the trunk itself rides moe_layers
    // through forwardQwen4With. `qwen4_mixer` is the final hyper-connection
    // read that replaces model.norm.
    qwen4: ?*qwen4_mod.Qwen4State = null,
    /// Pipeline rank 0: the worker holding the trunk's later layers (`distributed.Peer`).
    dist_peer: ?*distributed_mod.Peer = null,
    /// Tensor-parallel rank 0: the worker holding the other slice of every expert. Its
    /// part is stateless, so nothing about scheduling or caching changes.
    dist_tp: ?*distributed_mod.Peer = null,
    /// Run the qwen4 residual streams in f32 (QWEN4_STREAM_F32=1, or set by
    /// the oracle test): the shipped checkpoint is bf16 end to end, the f32
    /// fixture is the tighter bar for the MATH.
    qwen4_stream_f32: bool = false,
    qwen4_mixer: ?HcWeights = null,
    /// nb-keyed QSA block constants (`QsaBlockConsts`).
    qsa_consts: QsaBlockConsts = .{},
    /// QSA pooled-block cos/sin, built once per forward (`QsaPooledRope`).
    qsa_pooled_rope: QsaPooledRope = .{},
    /// Bumped by every forward entry that can reach QSA; keys the per-forward QSA scratch.
    fwd_gen: u64 = 0,
    /// Test meter: how many times `qsaScoreBank` built or extended the transposed f32 operand.
    qsa_score_bank_builds: usize = 0,
    /// qwen4_exp MTP head (one hyper-connected QSA+MoE layer over the trunk's
    /// pre-mixer stream + next-token embedding). Loaded when the pack carries
    /// `mtp.*`; served by `qwen4MtpForward`. Not yet wired into spec decode.
    qwen4_mtp: ?Qwen4Mtp = null,
    /// Whose per-request state the head currently holds (`qwen4MtpActivate`).
    qwen4_mtp_owner: ?*Qwen4MtpState = null,

    // MoE-specific (null/empty for standard models)
    moe_layers: ?[]MoeLayerWeights,
    /// Construction-validated q4/g64 S=4 routed gate/up verifier lane.
    verify_paired_gu_installed: bool = false,
    ssm_entries: ?[]SSMCacheEntry,
    moe_seq_offset: usize,
    // Pre-transposed plain-bf16 linear_attn weights owned by the Transformer
    // (Unsloth Dynamic checkpoints — null for the common all-quantized case).
    moe_owned_bf16: ?[]mlx.mlx_array = null,
    /// Prism Hadamard sign vectors keyed on weight handles; null on every other checkpoint.
    rht: ?*rht.Registry = null,
    /// A 2-bit group-128 pack without rotations whose every matmul weight has
    /// biases == -scales (Prism's ternary codec): qmv2's ternary kernel applies.
    ternary_2bit: bool = false,
    /// LOSSY int8-activation prefill (`qmm_int8`), per model: allocated when its
    /// `int8_prefill` setting (else the process default) turns the route on.
    int8_cache: ?*qmm_int8.Cache = null,

    // When non-null, the next forward pass captures the post-final-norm
    // hidden state at the last position into the pointed-to array
    // (refcount-shared with the live forward graph). Set/cleared by
    // `forwardCaptureHidden`. Used by PLD verify-fusion and the Gemma 4
    // assistant drafter for h_prev seed.
    // Single-threaded: generation runs on one thread per Transformer.
    capture_hidden: ?*mlx.mlx_array = null,

    embedding_norm: ?mlx.mlx_array, // Inkling: RMS norm on embeddings

    // Prompt cache for prefix reuse across requests
    prompt_cache: ?PrefillCache,

    // Compiled forward pass closure (JIT-compiled for Metal kernel fusion)
    compiled_forward: ?mlx.mlx_closure = null,

    // Vision: set before prefill when images are present, cleared after.
    // Shape: [B, num_image_tokens, hidden_size]. Spliced at image_token_id positions.
    vision_embeddings: ?mlx.mlx_array = null,

    // Compiled closures (fuse ops into single kernels, matching mlx-lm's @mx.compile)
    compiled_gelu: ?mlx.mlx_closure = null,
    compiled_geglu: ?mlx.mlx_closure = null, // gelu(gate) * up → 1 kernel
    compiled_softcap: ?mlx.mlx_closure = null, // tanh(x/cap) * cap → 1 kernel
    compiled_moe_routing: ?mlx.mlx_closure = null, // negate→argpartition→slice→softmax→take→sum→expand→divide → 1 kernel
    compiled_hy3_routing: ?mlx.mlx_closure = null, // hy_v3 sigmoid+bias variant (2 inputs: logits, expert_bias)
    compiled_gdn_gate: ?mlx.mlx_closure = null, // exp(-exp(A_log)·softplus(a+dt_bias)) → 1 kernel (mirrors mlx-lm compute_g)
    // qwen4_exp hyper-connection elementwise tails, one kernel each (the
    // `/hc` scalars are folded into the weights at load): silu(down),
    // mean(σ(up)·normed), 2σ(inject), stream + out·inj.
    compiled_hc_silu: ?mlx.mlx_closure = null,
    compiled_hc_mix: ?mlx.mlx_closure = null,
    compiled_hc_inj: ?mlx.mlx_closure = null,
    compiled_hc_write: ?mlx.mlx_closure = null,

    // GatedDeltaNet per-token decode used to rebuild these on EVERY layer/step
    // (measured dispatch overhead on Qwen 3.6 hybrid). Built lazily on first
    // use, freed in deinit.
    gdn_ones_w: ?mlx.mlx_array = null, // ones([dk]) for parameter-free rms_norm
    gdn_q_scale: ?mlx.mlx_array = null, // bf16 scalar 1/dk
    gdn_k_scale: ?mlx.mlx_array = null, // bf16 scalar 1/sqrt(dk)
    grouped_norm_ones: ?mlx.mlx_array = null, // k2_horizon: rms_norm ones weight, one group wide
    gdn_eps: ?mlx.mlx_array = null, // f32 scalar rms_norm_eps for the fused norm-gate kernel
    /// PLD spec-decode: mirrors `ForwardCtx.capture_ssm_seq` for the current
    /// forward so `gatedDeltaNet`/`conv1dWithCache` (which don't take the ctx)
    /// can capture per-position state. Set+reset only inside `forwardMoeWith`.
    spec_capture_ssm: bool = false,
    /// `ForwardCtx.tree` for the layers that do not take the ctx.
    spec_tree: ?*const SpecTree = null,
    /// Test seam only (no env): `false` runs the old per-tick merge/split path so the
    /// transition tests can compare bytes. The persistent group is an exact change.
    persistent_group_state_override: ?bool = null,
    /// Test seam only (no env): `false` runs the old per-slot KV update for byte comparisons.
    batched_kv_append_override: ?bool = null,
    ssm_group: PersistentSsmGroup = .{},

    // Per-weight quantization bit cache (see bitsFor). Populated lazily on first use.
    // Keyed by the scales array's ctx pointer (stable for the lifetime of a weight).
    // Used instead of config.quant_bits so mixed-precision models (Gemma-4 MoE, etc.)
    // with per-layer overrides work correctly while keeping zero per-call FFI overhead
    // after the first touch.
    bits_cache: BitsCache = .{},

    // True only for the exact measured dense 27B architecture when every
    // resident trunk projection is homogeneous affine-4/gs-64. Embedding
    // storage is gated separately by each sidecar cost profile: the older
    // Qwen3.6 surfaces use q4/gs64, Qwen3.8's measured surface uses bf16.
    mtp_uniform_affine_trunk: bool = false,
    // Exact oQ4e mixed q4/q5/q6 layer-role fingerprint. Kept separate from
    // the homogeneous profile because its measured NAX cost surface differs.
    mtp_oqe_affine_trunk: bool = false,

    // ── methods split out by meaning (P1); aliases keep `x.method(...)` call syntax ──
    // plugins/qwen4_exp/mtp.zig
    pub const loadQwen4Mtp = @import("../../plugins/qwen4_exp/mtp.zig").loadQwen4Mtp;
    pub const Qwen4MtpState = @import("../../plugins/qwen4_exp/mtp.zig").Qwen4MtpState;
    pub const Qwen4MtpLive = @import("../../plugins/qwen4_exp/mtp.zig").Qwen4MtpLive;
    pub const qwen4MtpLiveState = @import("../../plugins/qwen4_exp/mtp.zig").qwen4MtpLiveState;
    pub const qwen4MtpLiveModule = @import("../../plugins/qwen4_exp/mtp.zig").qwen4MtpLiveModule;
    pub const qwen4MtpStateNew = @import("../../plugins/qwen4_exp/mtp.zig").qwen4MtpStateNew;
    const qwen4MtpSwapState = @import("../../plugins/qwen4_exp/mtp.zig").qwen4MtpSwapState;
    pub const qwen4MtpActivate = @import("../../plugins/qwen4_exp/mtp.zig").qwen4MtpActivate;
    pub const qwen4MtpRelease = @import("../../plugins/qwen4_exp/mtp.zig").qwen4MtpRelease;
    pub const qwen4MtpReset = @import("../../plugins/qwen4_exp/mtp.zig").qwen4MtpReset;
    pub const qwen4MtpResetOwned = @import("../../plugins/qwen4_exp/mtp.zig").qwen4MtpResetOwned;
    pub const mtpHeadKvQuantEnabled = @import("../../plugins/qwen4_exp/mtp.zig").mtpHeadKvQuantEnabled;
    pub const qwen4MtpHeadKvConfig = @import("../../plugins/qwen4_exp/mtp.zig").qwen4MtpHeadKvConfig;
    pub const qwen4MtpApplyKvQuant = @import("../../plugins/qwen4_exp/mtp.zig").qwen4MtpApplyKvQuant;
    pub const qwen4MtpMarkQsaLeftover = @import("../../plugins/qwen4_exp/mtp.zig").qwen4MtpMarkQsaLeftover;
    pub const qwen4MtpMarks = @import("../../plugins/qwen4_exp/mtp.zig").qwen4MtpMarks;
    pub const Qwen4MtpOut = @import("../../plugins/qwen4_exp/mtp.zig").Qwen4MtpOut;
    const Qwen4MtpOutBuf = @import("../../plugins/qwen4_exp/mtp.zig").Qwen4MtpOutBuf;
    pub const Qwen4MtpProject = @import("../../plugins/qwen4_exp/mtp.zig").Qwen4MtpProject;
    pub const qwen4MtpTruncate = @import("../../plugins/qwen4_exp/mtp.zig").qwen4MtpTruncate;
    pub const qwen4MtpTruncateOn = @import("../../plugins/qwen4_exp/mtp.zig").qwen4MtpTruncateOn;
    pub const qwen4MtpTruncateLive = @import("../../plugins/qwen4_exp/mtp.zig").qwen4MtpTruncateLive;
    pub const qwen4MtpTruncateToMarkLive = @import("../../plugins/qwen4_exp/mtp.zig").qwen4MtpTruncateToMarkLive;
    pub const qwen4MtpAdopt = @import("../../plugins/qwen4_exp/mtp.zig").qwen4MtpAdopt;
    pub const qwen4MtpAdvance = @import("../../plugins/qwen4_exp/mtp.zig").qwen4MtpAdvance;
    pub const qwen4MtpForward = @import("../../plugins/qwen4_exp/mtp.zig").qwen4MtpForward;
    pub const qwen4MtpForwardOn = @import("../../plugins/qwen4_exp/mtp.zig").qwen4MtpForwardOn;
    pub const HeadPlace = @import("../../plugins/qwen4_exp/mtp.zig").HeadPlace;
    pub const qwen4MtpForwardPlacedOn = @import("../../plugins/qwen4_exp/mtp.zig").qwen4MtpForwardPlacedOn;
    pub const qwen4MtpForwardPlaced = @import("../../plugins/qwen4_exp/mtp.zig").qwen4MtpForwardPlaced;
    pub const Qwen4MtpBatchRow = @import("../../plugins/qwen4_exp/mtp.zig").Qwen4MtpBatchRow;
    pub const mtpPadSeq = @import("../../plugins/qwen4_exp/mtp.zig").mtpPadSeq;
    const mtpHeadForceBatched = @import("../../plugins/qwen4_exp/mtp.zig").mtpHeadForceBatched;
    pub const mtpMoeRows = @import("../../plugins/qwen4_exp/mtp.zig").mtpMoeRows;
    pub const qwen4MtpForwardBatched = @import("../../plugins/qwen4_exp/mtp.zig").qwen4MtpForwardBatched;
    pub const qwen4MtpForwardLive = @import("../../plugins/qwen4_exp/mtp.zig").qwen4MtpForwardLive;
    pub const qwen4BuildDraftRerank = @import("../../plugins/qwen4_exp/mtp.zig").qwen4BuildDraftRerank;
    pub const qwen4DraftRerankReady = @import("../../plugins/qwen4_exp/mtp.zig").qwen4DraftRerankReady;
    pub const qwen4DraftSelect = @import("../../plugins/qwen4_exp/mtp.zig").qwen4DraftSelect;
    pub const qwen4DraftShortlist = @import("../../plugins/qwen4_exp/mtp.zig").qwen4DraftShortlist;
    pub const qwen4DraftShortlistsBatched = @import("../../plugins/qwen4_exp/mtp.zig").qwen4DraftShortlistsBatched;
    pub const qwen4DraftSelectBatched = @import("../../plugins/qwen4_exp/mtp.zig").qwen4DraftSelectBatched;
    pub const Qwen4Trace = @import("../../plugins/qwen4_exp/mtp.zig").Qwen4Trace;
    // sdk/blocks/linear/verify_projection.zig
    pub const verifyFeatureEnabled = @import("../../sdk/blocks/linear/verify_projection.zig").verifyFeatureEnabled;
    pub const mtpProjectionRows = @import("../../sdk/blocks/linear/verify_projection.zig").mtpProjectionRows;
    // mlx/array.zig
    pub const concatAxis0 = @import("../../mlx/array.zig").concatAxis0;
    pub const axisView = @import("../../mlx/array.zig").axisView;
    pub const concatAxis1 = @import("../../mlx/array.zig").concatAxis1;
    // plugins/qwen4_exp/qsa/decode_gather.zig
    pub const qsaBatchedAttn = @import("../../plugins/qwen4_exp/qsa/decode_gather.zig").qsaBatchedAttn;
    // engines/mlx/batched/layered.zig
    pub const forwardMoeBatchedDecode = @import("batched/layered.zig").forwardMoeBatchedDecode;
    pub const forwardMoeBatchedVerify = @import("batched/layered.zig").forwardMoeBatchedVerify;
    pub const sliceBatchRows = @import("batched/layered.zig").sliceBatchRows;
    pub const forwardMoeBatchedRows = @import("batched/layered.zig").forwardMoeBatchedRows;
    // plugins/glm5_next/forward.zig
    pub const glmHeadMm = @import("../../plugins/glm5_next/forward.zig").glmHeadMm;
    pub const forwardGlm5With = @import("../../plugins/glm5_next/forward.zig").forwardGlm5With;
    pub const glmSublayer = @import("../../plugins/glm5_next/forward.zig").glmSublayer;
    pub const glmBranch = @import("../../plugins/glm5_next/forward.zig").glmBranch;
    pub const glmBranchSlots = @import("../../plugins/glm5_next/forward.zig").glmBranchSlots;
    pub const kdaLows = @import("../../plugins/glm5_next/forward.zig").kdaLows;
    pub const glmKdaSlots = @import("../../plugins/glm5_next/forward.zig").glmKdaSlots;
    const MOE_DECODE_BATCH_MAX = @import("../../plugins/glm5_next/forward.zig").MOE_DECODE_BATCH_MAX;
    pub const forwardGlmBatchedDecode = @import("../../plugins/glm5_next/forward.zig").forwardGlmBatchedDecode;
    pub const glmDsaAttn = @import("../../plugins/glm5_next/forward.zig").glmDsaAttn;
    // plugins/qwen4_exp/ple.zig
    const advancePlePrev = @import("../../plugins/qwen4_exp/ple.zig").advancePlePrev;
    const pleDeferrable = @import("../../plugins/qwen4_exp/ple.zig").pleDeferrable;
    pub const pleEmbedding = @import("../../plugins/qwen4_exp/ple.zig").pleEmbedding;
    pub const pleEmbeddingGpu = @import("../../plugins/qwen4_exp/ple.zig").pleEmbeddingGpu;
    pub const pleClaimSpecCapture = @import("../../plugins/qwen4_exp/ple.zig").pleClaimSpecCapture;
    pub const flushDeferredPle = @import("plugin/settle.zig").flush;
    pub const flushDeferredPleGroup = @import("plugin/settle.zig").flushGroup;
    pub const settleDeferredPle = @import("plugin/settle.zig").settleWith;
    pub const deferredPleOnGpu = @import("plugin/settle.zig").historyOwed;
    const releasePlePending = @import("../../plugins/qwen4_exp/ple.zig").releasePlePending;
    pub const discardDeferredPle = @import("plugin/settle.zig").discard;
    pub const pleGatherBf16 = @import("../../plugins/qwen4_exp/ple.zig").pleGatherBf16;
    pub const pleSettleFromArray = @import("../../plugins/qwen4_exp/ple.zig").pleSettleFromArray;
    pub const pleEvalIds = @import("../../plugins/qwen4_exp/ple.zig").pleEvalIds;
    pub const pleHostIds = @import("../../plugins/qwen4_exp/ple.zig").pleHostIds;
    pub const plePrepareIds = @import("../../plugins/qwen4_exp/ple.zig").plePrepareIds;
    pub const pleFillFromIds = @import("../../plugins/qwen4_exp/ple.zig").pleFillFromIds;
    const plePrevOf = @import("../../plugins/qwen4_exp/ple.zig").plePrevOf;
    const pleAdvanceSerial = @import("../../plugins/qwen4_exp/ple.zig").pleAdvanceSerial;
    pub const pleForward = @import("../../plugins/qwen4_exp/ple.zig").pleForward;
    // engines/mlx/ssm/group.zig
    pub const mergeSsmAcrossSlots = @import("ssm/group.zig").mergeSsmAcrossSlots;
    pub const splitSsmToSlots = @import("ssm/group.zig").splitSsmToSlots;
    pub const ssmGroupDrop = @import("ssm/group.zig").ssmGroupDrop;
    pub const ssmGroupRelease = @import("ssm/group.zig").ssmGroupRelease;
    pub const ssmBindBatched = @import("ssm/group.zig").ssmBindBatched;
    // plugins/qwen4_exp/qsa/pooled.zig
    pub const resetQsaPooledRope = @import("../../plugins/qwen4_exp/qsa/pooled.zig").resetQsaPooledRope;
    pub const qwen4Host = @import("plugin/intree.zig").host;
    pub const markQsaPooledRopeStale = @import("../../plugins/qwen4_exp/qsa/pooled.zig").markQsaPooledRopeStale;
    // sdk/blocks/attention/batched.zig
    pub const BATCHED_PER_SLOT_ATTN_MIN_KV = @import("../../sdk/blocks/attention/batched.zig").BATCHED_PER_SLOT_ATTN_MIN_KV;
    pub const batchedAttnPerSlot = @import("../../sdk/blocks/attention/batched.zig").batchedAttnPerSlot;
    pub const stackedBatchedAttn = @import("../../sdk/blocks/attention/batched.zig").stackedBatchedAttn;
    pub const perSlotBatchedAttn = @import("../../sdk/blocks/attention/batched.zig").perSlotBatchedAttn;
    pub const padAndStackBatchedKV = @import("../../sdk/blocks/attention/batched.zig").padAndStackBatchedKV;
    pub const buildBatchedDecodeMask = @import("../../sdk/blocks/attention/batched.zig").buildBatchedDecodeMask;
    // engines/mlx/prompt_cache.zig
    pub const resetCache = @import("prompt_cache.zig").resetCache;
    pub const tryRestoreCache = @import("prompt_cache.zig").tryRestoreCache;
    pub const savePromptCache = @import("prompt_cache.zig").savePromptCache;
    // sdk/blocks/gdn/gated_delta_net.zig
    pub const gatedDeltaNet = @import("../../sdk/blocks/gdn/gated_delta_net.zig").gatedDeltaNet;
    pub const gatedDeltaNetProjected = @import("../../sdk/blocks/gdn/gated_delta_net.zig").gatedDeltaNetProjected;
    pub const gdnCheckFused = @import("../../sdk/blocks/gdn/gated_delta_net.zig").gdnCheckFused;
    pub const GdnProj = @import("../../sdk/blocks/gdn/gated_delta_net.zig").GdnProj;
    pub const gdnProjGpu = @import("../../sdk/blocks/gdn/gated_delta_net.zig").gdnProjGpu;
    pub const gdnProjArrays = @import("../../sdk/blocks/gdn/gated_delta_net.zig").gdnProjArrays;
    // sdk/blocks/mlp/dense.zig
    pub const residualTail = @import("../../sdk/blocks/mlp/dense.zig").residualTail;
    pub const standardMlp = @import("../../sdk/blocks/mlp/dense.zig").standardMlp;
    pub const rowGroupServes = @import("../../sdk/blocks/mlp/dense.zig").rowGroupServes;
    pub const denseMLP = @import("../../sdk/blocks/mlp/dense.zig").denseMLP;
    // engines/mlx/ane_prefill.zig
    pub const denseMLPMaybeAne = @import("ane_prefill.zig").denseMLPMaybeAne;
    pub const gdnProjMaybeAne = @import("ane_prefill.zig").gdnProjMaybeAne;
    pub const anePackPlane = @import("ane_prefill.zig").anePackPlane;
    pub const aneReadPlane = @import("ane_prefill.zig").aneReadPlane;
    pub const denseMLPAneChannel = @import("ane_prefill.zig").denseMLPAneChannel;
    pub const anePackUnitPlanes = @import("ane_prefill.zig").anePackUnitPlanes;
    pub const gdnProjAneChannel = @import("ane_prefill.zig").gdnProjAneChannel;
    pub const aneSliceRowsView = @import("ane_prefill.zig").aneSliceRowsView;
    pub const aneSliceColsCopy = @import("ane_prefill.zig").aneSliceColsCopy;
    pub const dequantToHostF32 = @import("ane_prefill.zig").dequantToHostF32;
    pub const buildAnePrefill = @import("ane_prefill.zig").buildAnePrefill;
    pub const aneBuildMlpRest = @import("ane_prefill.zig").aneBuildMlpRest;
    pub const aneBuildGdnRest = @import("ane_prefill.zig").aneBuildGdnRest;
    // engines/mlx/forward/layered.zig
    pub const forwardMoeWith = @import("forward/layered.zig").forwardMoeWith;
    // engines/mlx/spec/mtp_nax_profile.zig
    pub const mtpNaxProfileEnabledForTrunk = @import("spec/mtp_nax_profile.zig").mtpNaxProfileEnabledForTrunk;
    pub const mtpNaxProfileEnabled = @import("spec/mtp_nax_profile.zig").mtpNaxProfileEnabled;
    pub const mtpNaxQ4Gs64ProfileEnabled = @import("spec/mtp_nax_profile.zig").mtpNaxQ4Gs64ProfileEnabled;
    pub const mtpNaxQ6Gs64ProfileEnabled = @import("spec/mtp_nax_profile.zig").mtpNaxQ6Gs64ProfileEnabled;
    pub const mtpNaxQ8Gs64ProfileEnabled = @import("spec/mtp_nax_profile.zig").mtpNaxQ8Gs64ProfileEnabled;
    pub const mtpOqeNaxProfileEnabled = @import("spec/mtp_nax_profile.zig").mtpOqeNaxProfileEnabled;
    // engines/mlx/load/init.zig
    pub const init = @import("load/init.zig").init;
    // engines/mlx/compiled.zig
    pub const compileForward = @import("compiled.zig").compileForward;
    pub const compileGelu = @import("compiled.zig").compileGelu;
    pub const compileGeglu = @import("compiled.zig").compileGeglu;
    const gegluClosureCallback = @import("compiled.zig").gegluClosureCallback;
    pub const compileSoftcap = @import("compiled.zig").compileSoftcap;
    const softcapClosureCallback = @import("compiled.zig").softcapClosureCallback;
    const geluClosureCallback = @import("compiled.zig").geluClosureCallback;
    pub const closureIn = @import("compiled.zig").closureIn;
    pub const closureOut = @import("compiled.zig").closureOut;
    pub const applyClosure = @import("compiled.zig").applyClosure;
    const forwardClosureCallback = @import("compiled.zig").forwardClosureCallback;
    pub const forwardCompiled = @import("compiled.zig").forwardCompiled;
    pub const clearCompiledClosures = @import("compiled.zig").clearCompiledClosures;
    // sdk/blocks/gdn/gate.zig
    pub const compileGdnGate = @import("../../sdk/blocks/gdn/gate.zig").compileGdnGate;
    const gdnGateClosureCallback = @import("../../sdk/blocks/gdn/gate.zig").gdnGateClosureCallback;
    pub const computeGdnGate = @import("../../sdk/blocks/gdn/gate.zig").computeGdnGate;
    // plugins/qwen4_exp/hc.zig
    pub const compileQwen4Hc = @import("../../plugins/qwen4_exp/hc.zig").compileQwen4Hc;
    const hcSiluCallback = @import("../../plugins/qwen4_exp/hc.zig").hcSiluCallback;
    const hcMixCallback = @import("../../plugins/qwen4_exp/hc.zig").hcMixCallback;
    const hcInjCallback = @import("../../plugins/qwen4_exp/hc.zig").hcInjCallback;
    const hcWriteCallback = @import("../../plugins/qwen4_exp/hc.zig").hcWriteCallback;
    pub const HcRead = @import("../../plugins/qwen4_exp/hc.zig").HcRead;
    pub const hcReadStandin = @import("../../plugins/qwen4_exp/hc.zig").hcReadStandin;
    pub const hcReadPending = @import("../../plugins/qwen4_exp/hc.zig").hcReadPending;
    pub const hcReadFusedFor = @import("../../plugins/qwen4_exp/hc.zig").hcReadFusedFor;
    pub const addVerifyPle = @import("../../plugins/qwen4_exp/hc.zig").addVerifyPle;
    pub const hcReadVerifyRows = @import("../../plugins/qwen4_exp/hc.zig").hcReadVerifyRows;
    pub const hcWriteOrDefer = @import("../../plugins/qwen4_exp/hc.zig").hcWriteOrDefer;
    pub const hcFlush = @import("../../plugins/qwen4_exp/hc.zig").hcFlush;
    pub const hcGroupNorm = @import("../../plugins/qwen4_exp/hc.zig").hcGroupNorm;
    pub const hcRead = @import("../../plugins/qwen4_exp/hc.zig").hcRead;
    pub const hcReadNormed = @import("../../plugins/qwen4_exp/hc.zig").hcReadNormed;
    pub const hcWrite = @import("../../plugins/qwen4_exp/hc.zig").hcWrite;
    // sdk/blocks/moe/routing.zig
    pub const compileMoeRouting = @import("../../sdk/blocks/moe/routing.zig").compileMoeRouting;
    pub const MoeRouting = @import("../../sdk/blocks/moe/routing.zig").MoeRouting;
    const moeRoutingClosureCallback = @import("../../sdk/blocks/moe/routing.zig").moeRoutingClosureCallback;
    pub const moeRoutingUncompiled = @import("../../sdk/blocks/moe/routing.zig").moeRoutingUncompiled;
    const hy3RoutingClosureCallback = @import("../../sdk/blocks/moe/routing.zig").hy3RoutingClosureCallback;
    pub const computeHy3Routing = @import("../../sdk/blocks/moe/routing.zig").computeHy3Routing;
    pub const computeMoeRouting = @import("../../sdk/blocks/moe/routing.zig").computeMoeRouting;
    // sdk/blocks/linear/qmatmul.zig
    pub const qmatmul = @import("../../sdk/blocks/linear/qmatmul.zig").qmatmul;
    pub const lmHeadProject = @import("../../sdk/blocks/linear/qmatmul.zig").lmHeadProject;
    pub const quantParamsFor = @import("../../sdk/blocks/linear/qmatmul.zig").quantParamsFor;
    pub const quantParamsHinted = @import("../../sdk/blocks/linear/qmatmul.zig").quantParamsHinted;
    pub const qmatmulAddBias = @import("../../sdk/blocks/linear/qmatmul.zig").qmatmulAddBias;
    pub const qmatmulMaybeBias = @import("../../sdk/blocks/linear/qmatmul.zig").qmatmulMaybeBias;
    // engines/mlx/diag/proj_bench.zig
    pub const projLadderFits = @import("diag/proj_bench.zig").projLadderFits;
    const ProjRung = @import("diag/proj_bench.zig").ProjRung;
    const ProjOrder = @import("diag/proj_bench.zig").ProjOrder;
    pub const diagProjBench = @import("diag/proj_bench.zig").diagProjBench;
    pub const projRungOnce = @import("diag/proj_bench.zig").projRungOnce;
    // sdk/blocks/norms/rms.zig
    pub const rmsNormEps = @import("../../sdk/blocks/norms/rms.zig").rmsNormEps;
    pub const rhtAddNormRotate = @import("../../sdk/blocks/norms/rms.zig").rhtAddNormRotate;
    pub const narrowAddNorm = @import("../../sdk/blocks/norms/rms.zig").narrowAddNorm;
    pub const rmsNorm = @import("../../sdk/blocks/norms/rms.zig").rmsNorm;
    pub const layerNorm = @import("../../sdk/blocks/norms/rms.zig").layerNorm;
    // sdk/quant/ternary.zig
    pub const ternaryWideVerify = @import("../../sdk/quant/ternary.zig").ternaryWideVerify;
    // sdk/quant/hadamard.zig
    pub const unrotateEmbedding = @import("../../sdk/quant/hadamard.zig").unrotateEmbedding;
    // sdk/blocks/embedding.zig
    pub const rawEmbedding = @import("../../sdk/blocks/embedding.zig").rawEmbedding;
    pub const embedding = @import("../../sdk/blocks/embedding.zig").embedding;
    pub const dequantizedEmbedding = @import("../../sdk/blocks/embedding.zig").dequantizedEmbedding;
    pub const dequantTake = @import("../../sdk/blocks/embedding.zig").dequantTake;
    // sdk/vision/splice.zig
    pub const spliceVisionEmbeddings = @import("../../sdk/vision/splice.zig").spliceVisionEmbeddings;
    pub const spliceVisionRows = @import("../../sdk/vision/splice.zig").spliceVisionRows;
    pub const applyVisionEmbeddingsWith = @import("../../sdk/vision/splice.zig").applyVisionEmbeddingsWith;
    // sdk/blocks/mlp/activations.zig
    pub const gelu = @import("../../sdk/blocks/mlp/activations.zig").gelu;
    pub const geluUncompiled = @import("../../sdk/blocks/mlp/activations.zig").geluUncompiled;
    pub const silu = @import("../../sdk/blocks/mlp/activations.zig").silu;
    pub const mlpActivation = @import("../../sdk/blocks/mlp/activations.zig").mlpActivation;
    pub const geluErf = @import("../../sdk/blocks/mlp/activations.zig").geluErf;
    pub const reluSquared = @import("../../sdk/blocks/mlp/activations.zig").reluSquared;
    pub const geluTanhTable = @import("../../sdk/blocks/mlp/activations.zig").geluTanhTable;
    pub const fusedGeluTanhMul = @import("../../sdk/blocks/mlp/activations.zig").fusedGeluTanhMul;
    pub const computeGeglu = @import("../../sdk/blocks/mlp/activations.zig").computeGeglu;
    pub const computeClampedSwiGLU = @import("../../sdk/blocks/mlp/activations.zig").computeClampedSwiGLU;
    pub const applySoftcap = @import("../../sdk/blocks/mlp/activations.zig").applySoftcap;
    pub const swiglu = @import("../../sdk/blocks/mlp/activations.zig").swiglu;
    // sdk/blocks/conv/conv1d_cache.zig
    pub const conv1dWithCache = @import("../../sdk/blocks/conv/conv1d_cache.zig").conv1dWithCache;
    // sdk/blocks/moe/experts.zig
    pub const scaleExperts = @import("../../sdk/blocks/moe/experts.zig").scaleExperts;
    // engines/mlx/eval_cadence.zig
    pub const EVAL_EVERY_N_LAYERS = @import("eval_cadence.zig").EVAL_EVERY_N_LAYERS;
    pub const PREFILL_EVAL_CADENCE_DEFAULT = @import("eval_cadence.zig").PREFILL_EVAL_CADENCE_DEFAULT;
    pub const MOE_EVAL_EVERY_N_LAYERS = @import("eval_cadence.zig").MOE_EVAL_EVERY_N_LAYERS;
    pub const RECURRENCE_EVAL_INTERVAL = @import("eval_cadence.zig").RECURRENCE_EVAL_INTERVAL;
    const PREFILL_EVAL_TRANSIENT_BUDGET = @import("eval_cadence.zig").PREFILL_EVAL_TRANSIENT_BUDGET;
    pub const PREFILL_EVAL_MIN_SEQ = @import("eval_cadence.zig").PREFILL_EVAL_MIN_SEQ;
    pub const prefillEvalCadenceApplies = @import("eval_cadence.zig").prefillEvalCadenceApplies;
    pub const evalCadencePoint = @import("eval_cadence.zig").evalCadencePoint;
    pub const decodeAsyncLadderStride = @import("eval_cadence.zig").decodeAsyncLadderStride;
    const DECODE_ASYNC_LADDER_AUTO = @import("eval_cadence.zig").DECODE_ASYNC_LADDER_AUTO;
    pub const ladderStep = @import("eval_cadence.zig").ladderStep;
    pub const ladderStepMulti = @import("eval_cadence.zig").ladderStepMulti;
    pub const prefillEvalCadence = @import("eval_cadence.zig").prefillEvalCadence;
    pub const prefillEvalCadenceFor = @import("eval_cadence.zig").prefillEvalCadenceFor;
    // plugins/qwen4_exp/forward.zig
    pub const qwen4DecodeLadderStride = @import("../../plugins/qwen4_exp/forward.zig").qwen4DecodeLadderStride;
    const QWEN4_DECODE_ASYNC_LADDER_DEFAULT = @import("../../plugins/qwen4_exp/forward.zig").QWEN4_DECODE_ASYNC_LADDER_DEFAULT;
    pub const qwen4VerifyRowsEligible = @import("../../plugins/qwen4_exp/forward.zig").qwen4VerifyRowsEligible;
    pub const forwardQwen4VerifyRows = @import("../../plugins/qwen4_exp/forward.zig").forwardQwen4VerifyRows;
    pub const forwardQwen4Stream = @import("../../plugins/qwen4_exp/forward.zig").forwardQwen4Stream;
    pub const forwardQwen4With = @import("../../plugins/qwen4_exp/forward.zig").forwardQwen4With;
    // engines/mlx/dispatch.zig
    pub const defaultCtx = @import("dispatch.zig").defaultCtx;
    pub const module_owned_state_fields = @import("dispatch.zig").module_owned_state_fields;
    pub const module_shared_readonly_fields = @import("dispatch.zig").module_shared_readonly_fields;
    pub const ownsModuleDecodeState = @import("dispatch.zig").ownsModuleDecodeState;
    pub const sharesModuleReadonlyState = @import("dispatch.zig").sharesModuleReadonlyState;
    pub const moduleSpecWiring = @import("dispatch.zig").moduleSpecWiring;
    pub const prefillsWholePrompt = @import("dispatch.zig").prefillsWholePrompt;
    pub const dsparkStages = @import("dispatch.zig").dsparkStages;
    pub const moduleStateSpecRollback = @import("dispatch.zig").moduleStateSpecRollback;
    pub const forward = @import("dispatch.zig").forward;
    pub const forwardWith = @import("dispatch.zig").forwardWith;
    pub const usesStandardForward = @import("dispatch.zig").usesStandardForward;
    pub const supportsLayerCapture = @import("dispatch.zig").supportsLayerCapture;
    // engines/mlx/spec/trunk_hooks.zig
    pub const lmHeadForDraft = @import("spec/trunk_hooks.zig").lmHeadForDraft;
    pub const tileLaneWeights = @import("spec/trunk_hooks.zig").tileLaneWeights;
    pub const specTreeSupported = @import("spec/trunk_hooks.zig").specTreeSupported;
    pub const lmHeadRowsForDraft = @import("spec/trunk_hooks.zig").lmHeadRowsForDraft;
    // engines/mlx/warmup.zig
    pub const warmup = @import("warmup.zig").warmup;
    const SpecWarmSlot = @import("warmup.zig").SpecWarmSlot;
    const specWarmIds = @import("warmup.zig").specWarmIds;
    const specWarmEvalFree = @import("warmup.zig").specWarmEvalFree;
    pub const specWarmVerify = @import("warmup.zig").specWarmVerify;
    const MAX_WARM_SEQ = @import("warmup.zig").MAX_WARM_SEQ;
    const MAX_WIDTH_WARM = @import("warmup.zig").MAX_WIDTH_WARM;
    pub const warmupSpecVerify = @import("warmup.zig").warmupSpecVerify;
    pub const specWarmGroupRows = @import("warmup.zig").specWarmGroupRows;
    pub const specWarmGroupSeq = @import("warmup.zig").specWarmGroupSeq;
    pub const specWarmHead = @import("warmup.zig").specWarmHead;
    // engines/mlx/capture.zig
    pub const forwardCaptureHidden = @import("capture.zig").forwardCaptureHidden;
    pub const forwardWithCapture = @import("capture.zig").forwardWithCapture;
    pub const forwardWithCaptureAll = @import("capture.zig").forwardWithCaptureAll;
    pub const captureHidden = @import("capture.zig").captureHidden;
    // plugins/gemma4/ple.zig
    pub const computePLEInput = @import("../../plugins/gemma4/ple.zig").computePLEInput;
    pub const pleProject = @import("../../plugins/gemma4/ple.zig").pleProject;
    // engines/mlx/forward/dense.zig
    pub const forwardStandardWith = @import("forward/dense.zig").forwardStandardWith;
    // engines/mlx/batched/dense.zig
    pub const forwardBatchedDecode = @import("batched/dense.zig").forwardBatchedDecode;
    // engines/mlx/batched/row_axis_verify.zig
    pub const VerifyLaps = @import("batched/row_axis_verify.zig").VerifyLaps;
    pub const VerifyRollback = @import("batched/row_axis_verify.zig").VerifyRollback;
    pub const forwardRowAxisVerify = @import("batched/row_axis_verify.zig").forwardRowAxisVerify;
    pub const releaseJoinedVerifyLogits = @import("batched/row_axis_verify.zig").releaseJoinedVerifyLogits;
    pub const takeJoinedVerifyLogits = @import("batched/row_axis_verify.zig").takeJoinedVerifyLogits;
    // engines/mlx/batched/caps.zig
    pub const hasRecurrentLayers = @import("batched/caps.zig").hasRecurrentLayers;
    pub const batchedGdnReady = @import("batched/caps.zig").batchedGdnReady;
    pub const supportsBatchedGdnDecode = @import("batched/caps.zig").supportsBatchedGdnDecode;
    // sdk/blocks/rope/rope.zig
    pub const ropeInvFreq = @import("../../sdk/blocks/rope/rope.zig").ropeInvFreq;
    pub const ropeAtPositions = @import("../../sdk/blocks/rope/rope.zig").ropeAtPositions;
    pub const ropeAtFreqs = @import("../../sdk/blocks/rope/rope.zig").ropeAtFreqs;
    pub const ropeCosSinFromFreqs = @import("../../sdk/blocks/rope/rope.zig").ropeCosSinFromFreqs;
    pub const ropeApplyCosSin = @import("../../sdk/blocks/rope/rope.zig").ropeApplyCosSin;
    // plugins/qwen4_exp/qsa/indexer.zig
    pub const qsaInstallRing = @import("../../plugins/qwen4_exp/qsa/indexer.zig").qsaInstallRing;
    pub const qsaAppendKeys = @import("../../plugins/qwen4_exp/qsa/indexer.zig").qsaAppendKeys;
    pub const qsaAppendPooled = @import("../../plugins/qwen4_exp/qsa/indexer.zig").qsaAppendPooled;
    pub const qsaScoreCols = @import("../../plugins/qwen4_exp/qsa/indexer.zig").qsaScoreCols;
    pub const qsaScoreBank = @import("../../plugins/qwen4_exp/qsa/indexer.zig").qsaScoreBank;
    pub const qsaScoreOperand = @import("../../plugins/qwen4_exp/qsa/indexer.zig").qsaScoreOperand;
    pub const qsaPooledCosSin = @import("../../plugins/qwen4_exp/qsa/indexer.zig").qsaPooledCosSin;
    pub const qsaPooledKeys = @import("../../plugins/qwen4_exp/qsa/indexer.zig").qsaPooledKeys;
    pub const qsaPooledKeysComposed = @import("../../plugins/qwen4_exp/qsa/indexer.zig").qsaPooledKeysComposed;
    pub const qsaPoolNorm = @import("../../plugins/qwen4_exp/qsa/indexer.zig").qsaPoolNorm;
    pub const qsaPooledRopeComposed = @import("../../plugins/qwen4_exp/qsa/indexer.zig").qsaPooledRopeComposed;
    const qsaSlotEntry = @import("../../plugins/qwen4_exp/qsa/indexer.zig").qsaSlotEntry;
    pub const qsaMask = @import("../../plugins/qwen4_exp/qsa/indexer.zig").qsaMask;
    pub const qsaMaskBatched = @import("../../plugins/qwen4_exp/qsa/indexer.zig").qsaMaskBatched;
    pub const qsaMaskFromQk = @import("../../plugins/qwen4_exp/qsa/indexer.zig").qsaMaskFromQk;
    pub const qsaSelectMaskOps = @import("../../plugins/qwen4_exp/qsa/indexer.zig").qsaSelectMaskOps;
    pub const qsaBlockVisibility = @import("../../plugins/qwen4_exp/qsa/indexer.zig").qsaBlockVisibility;
    pub const qsaChunkSelect = @import("../../plugins/qwen4_exp/qsa/indexer.zig").qsaChunkSelect;
    pub const qsaSelectBlocks = @import("../../plugins/qwen4_exp/qsa/indexer.zig").qsaSelectBlocks;
    // plugins/qwen4_exp/attention.zig
    pub const qwen4AttnWith = @import("../../plugins/qwen4_exp/attention.zig").qwen4AttnWith;
    pub const qwen4AttnProjected = @import("../../plugins/qwen4_exp/attention.zig").qwen4AttnProjected;
    // plugins/gemma4/moe_tail.zig
    pub const gemma4MoeLayerTail = @import("../../plugins/gemma4/moe_tail.zig").gemma4MoeLayerTail;
    // engines/mlx/diag/profile.zig
    pub const debugTraceHead = @import("diag/profile.zig").debugTraceHead;
    pub const forwardDiffusionDecoder = @import("plugin/driver.zig").diffusionDecoder;
    // engines/mlx/forward/encoder_head.zig
    pub const forwardEmbedding = @import("forward/encoder_head.zig").forwardEmbedding;
    pub const forwardEmbeddingMasked = @import("forward/encoder_head.zig").forwardEmbeddingMasked;
    pub const hasEmbedProjection = @import("forward/encoder_head.zig").hasEmbedProjection;
    pub const embedProjection = @import("forward/encoder_head.zig").embedProjection;
    pub const denseHead = @import("forward/encoder_head.zig").denseHead;
    // plugins/gemma3/encoder.zig
    pub const forwardGemma3EncoderWith = @import("../../plugins/gemma3/encoder.zig").forwardGemma3EncoderWith;
    // sdk/blocks/rope/mrope.zig
    pub const buildMropeCosSin = @import("../../sdk/blocks/rope/mrope.zig").buildMropeCosSin;
    pub const MropeCosSin = @import("../../sdk/blocks/rope/mrope.zig").MropeCosSin;
    pub const mropeCosSinAt = @import("../../sdk/blocks/rope/mrope.zig").mropeCosSinAt;
    pub const mropeContext = @import("../../sdk/blocks/rope/mrope.zig").mropeContext;
    pub const beginMropeChunk = @import("../../sdk/blocks/rope/mrope.zig").beginMropeChunk;
    pub const endMropeChunk = @import("../../sdk/blocks/rope/mrope.zig").endMropeChunk;
    pub const applyMrope = @import("../../sdk/blocks/rope/mrope.zig").applyMrope;
    // sdk/blocks/rope/yarn.zig
    pub const yarnActive = @import("../../sdk/blocks/rope/yarn.zig").yarnActive;
    pub const yarnScaleQK = @import("../../sdk/blocks/rope/yarn.zig").yarnScaleQK;
    pub const yarnScaleRotated = @import("../../sdk/blocks/rope/yarn.zig").yarnScaleRotated;
    // sdk/blocks/attention/gated_full.zig
    pub const gatedFullAttnWith = @import("../../sdk/blocks/attention/gated_full.zig").gatedFullAttnWith;
    pub const gatedFullAttnProjected = @import("../../sdk/blocks/attention/gated_full.zig").gatedFullAttnProjected;
    pub const gatedAttnTail = @import("../../sdk/blocks/attention/gated_full.zig").gatedAttnTail;
    // sdk/blocks/attention/mla.zig
    pub const mlaAttnWith = @import("../../sdk/blocks/attention/mla.zig").mlaAttnWith;
    // sdk/blocks/attention/proj.zig
    pub const qkAngleFor = @import("../../sdk/blocks/attention/proj.zig").qkAngleFor;
    pub const qkAngleRowsFor = @import("../../sdk/blocks/attention/proj.zig").qkAngleRowsFor;
    pub const attnDqFor = @import("../../sdk/blocks/attention/proj.zig").attnDqFor;
    pub const attnProj = @import("../../sdk/blocks/attention/proj.zig").attnProj;
    pub const attnProjBias = @import("../../sdk/blocks/attention/proj.zig").attnProjBias;
    // sdk/blocks/attention/sinks.zig
    pub const sinkAttnWith = @import("../../sdk/blocks/attention/sinks.zig").sinkAttnWith;
    pub const sinkCacheUpdate = @import("../../sdk/blocks/attention/sinks.zig").sinkCacheUpdate;
    pub const sinkAttnSlots = @import("../../sdk/blocks/attention/sinks.zig").sinkAttnSlots;
    pub const sinkAttnCore = @import("../../sdk/blocks/attention/sinks.zig").sinkAttnCore;
    pub const sinkAttnOut = @import("../../sdk/blocks/attention/sinks.zig").sinkAttnOut;
    // plugins/gemma4/attention.zig
    pub const gemma4MoeAttnWith = @import("../../plugins/gemma4/attention.zig").gemma4MoeAttnWith;
    // engines/mlx/verify/joined_rows.zig
    pub const AttnVerifyInputs = @import("verify/joined_rows.zig").AttnVerifyInputs;
    const VerifyRowsKind = @import("verify/joined_rows.zig").VerifyRowsKind;
    pub const verifyRowsJoined = @import("verify/joined_rows.zig").verifyRowsJoined;
    pub const attnVerifyRows = @import("verify/joined_rows.zig").attnVerifyRows;
    pub const GdnVerifyInputs = @import("verify/joined_rows.zig").GdnVerifyInputs;
    pub const gdnVerifyRows = @import("verify/joined_rows.zig").gdnVerifyRows;
    // sdk/blocks/moe/moe.zig
    pub const moeRouterLogits = @import("../../sdk/blocks/moe/moe.zig").moeRouterLogits;
    pub const moeSharedDown = @import("../../sdk/blocks/moe/moe.zig").moeSharedDown;
    pub const moeSharedProject = @import("../../sdk/blocks/moe/moe.zig").moeSharedProject;
    pub const moeSharedGateTail = @import("../../sdk/blocks/moe/moe.zig").moeSharedGateTail;
    pub const ungatedSharedA4 = @import("../../sdk/blocks/moe/moe.zig").ungatedSharedA4;
    pub const moeAddUngatedShared = @import("../../sdk/blocks/moe/moe.zig").moeAddUngatedShared;
    pub const moeAddGatedShared = @import("../../sdk/blocks/moe/moe.zig").moeAddGatedShared;
    pub const moeAddSharedDown = @import("../../sdk/blocks/moe/moe.zig").moeAddSharedDown;
    pub const moeAddSharedGateRows = @import("../../sdk/blocks/moe/moe.zig").moeAddSharedGateRows;
    pub const moeSharedDownRows = @import("../../sdk/blocks/moe/moe.zig").moeSharedDownRows;
    pub const moeVerifyRowsSolo = @import("../../sdk/blocks/moe/moe.zig").moeVerifyRowsSolo;
    pub const moeVerifyRows = @import("../../sdk/blocks/moe/moe.zig").moeVerifyRows;
    pub const moeMLP = @import("../../sdk/blocks/moe/moe.zig").moeMLP;
    pub const moeTensorParallel = @import("../../sdk/blocks/moe/moe.zig").moeTensorParallel;
    pub const moeExpertsRouted = @import("../../sdk/blocks/moe/moe.zig").moeExpertsRouted;
    pub const moeMLP2WithRouter = @import("../../sdk/blocks/moe/moe.zig").moeMLP2WithRouter;
    // sdk/blocks/moe/decode.zig
    pub const glmMoeDecode = @import("../../sdk/blocks/moe/decode.zig").glmMoeDecode;
    pub const moeDecodeGatherQmv = @import("../../sdk/blocks/moe/decode.zig").moeDecodeGatherQmv;
    pub const moeFp4Decode = @import("../../sdk/blocks/moe/decode.zig").moeFp4Decode;
    pub const moeAffine4Decode = @import("../../sdk/blocks/moe/decode.zig").moeAffine4Decode;
    pub const moeCudaDecode = @import("../../sdk/blocks/moe/decode.zig").moeCudaDecode;
    pub const moeDecodeGatherQmvRows = @import("../../sdk/blocks/moe/decode.zig").moeDecodeGatherQmvRows;
    pub const moeMLP2 = @import("../../sdk/blocks/moe/decode.zig").moeMLP2;
    // sdk/blocks/attention/masks.zig
    pub const createCausalMask = @import("../../sdk/blocks/attention/masks.zig").createCausalMask;
    pub const createSlidingWindowDecodeMask = @import("../../sdk/blocks/attention/masks.zig").createSlidingWindowDecodeMask;
    pub const createSlidingWindowMask = @import("../../sdk/blocks/attention/masks.zig").createSlidingWindowMask;

    pub fn deinit(self: *Transformer) void {
        self.releaseJoinedVerifyLogits();
        lane_qmm.release(@intFromPtr(self));
        if (self.ane_prefill) |eng| {
            eng.deinit();
            self.ane_prefill = null;
        }
        if (self.ane_chan) |rests| {
            for (rests) |*r| r.deinit();
            self.allocator.free(rests);
            self.ane_chan = null;
        }
        if (self.mlx_stream_ext) |m| {
            mlx_stream_mod.close(m);
            self.mlx_stream_ext = null;
        }
        if (self.plugin) |b| {
            plugin_driver.unbind(b, self.allocator);
            self.plugin = null;
        }
        if (self.rht) |reg| {
            reg.deinit();
            self.allocator.destroy(reg);
            self.rht = null;
        }
        if (self.int8_cache) |c| {
            c.deinit();
            self.allocator.destroy(c);
            self.int8_cache = null;
        }
        if (self.compiled_forward) |cf| _ = mlx.mlx_closure_free(cf);
        if (self.compiled_gelu) |cg| _ = mlx.mlx_closure_free(cg);
        if (self.compiled_geglu) |cg| _ = mlx.mlx_closure_free(cg);
        if (self.compiled_softcap) |cs| _ = mlx.mlx_closure_free(cs);
        if (self.compiled_moe_routing) |cmr| _ = mlx.mlx_closure_free(cmr);
        if (self.compiled_hy3_routing) |chr| _ = mlx.mlx_closure_free(chr);
        if (self.compiled_gdn_gate) |cgg| _ = mlx.mlx_closure_free(cgg);
        if (self.gdn_ones_w) |w| _ = mlx.mlx_array_free(w);
        if (self.gdn_q_scale) |q| _ = mlx.mlx_array_free(q);
        if (self.gdn_eps) |e| _ = mlx.mlx_array_free(e);
        if (self.gdn_k_scale) |k| _ = mlx.mlx_array_free(k);
        if (self.grouped_norm_ones) |o| _ = mlx.mlx_array_free(o);
        if (self.ones_hidden) |o| _ = mlx.mlx_array_free(o);
        if (self.rope_freqs_yarn) |f| _ = mlx.mlx_array_free(f);
        if (self.rope_freqs_global) |f| _ = mlx.mlx_array_free(f);
        if (self.yarn_mscale) |m| _ = mlx.mlx_array_free(m);
        if (self.yarn_inv_freq) |f| self.allocator.free(f);
        if (self.suppress_mask) |m| _ = mlx.mlx_array_free(m);
        self.qsa_consts.deinit();
        self.qsa_pooled_rope.deinit();
        self.yarn_mscale_cast.deinit();
        if (self.prompt_cache) |*pc| pc.deinit();
        self.cache.deinit();
        if (self.emb_scale) |es| _ = mlx.mlx_array_free(es);
        if (self.owns_norms) _ = mlx.mlx_array_free(self.final_norm);
        if (self.gelu_coeff) |g| _ = mlx.mlx_array_free(g);
        if (self.gelu_inner) |g| _ = mlx.mlx_array_free(g);
        _ = mlx.mlx_array_free(self.half);
        _ = mlx.mlx_array_free(self.one);
        if (self.rms_eps_arr.ctx != null) _ = mlx.mlx_array_free(self.rms_eps_arr);
        if (self.post_eps_arr.ctx != null) _ = mlx.mlx_array_free(self.post_eps_arr);
        {
            var it = self.attn_dq.valueIterator();
            while (it.next()) |maybe| if (maybe.*) |copy| {
                _ = mlx.mlx_array_free(copy.w);
                _ = mlx.mlx_array_free(copy.s);
                if (copy.b.ctx != null) _ = mlx.mlx_array_free(copy.b);
            };
            self.attn_dq.deinit(self.allocator);
        }
        for (&self.qk_angle_cache) |*slot| {
            if (slot.arr.ctx != null) _ = mlx.mlx_array_free(slot.arr);
        }
        if (self.three) |t| _ = mlx.mlx_array_free(t);
        if (self.neg_one) |n| _ = mlx.mlx_array_free(n);
        for (self.layers) |lw| {
            if (self.owns_norms) {
                _ = mlx.mlx_array_free(lw.input_norm);
                _ = mlx.mlx_array_free(lw.post_attn_norm);
                if (lw.pre_ff_norm) |n| _ = mlx.mlx_array_free(n);
                if (lw.post_ff_norm) |n| _ = mlx.mlx_array_free(n);
                if (lw.q_norm) |n| _ = mlx.mlx_array_free(n);
                if (lw.k_norm) |n| _ = mlx.mlx_array_free(n);
            }
        }
        self.allocator.free(self.layers);
        if (self.ssm_entries) |entries| {
            for (entries) |*e| {
                ssmFreeSpecCapture(e);
                _ = mlx.mlx_array_free(e.conv_state);
                _ = mlx.mlx_array_free(e.ssm_state);
                ssmFreeQsaState(e);
            }
            self.allocator.free(entries);
        }
        self.ssm_group.deinit();
        if (self.moe_layers) |ml| self.allocator.free(ml);
        if (self.qwen4_mtp) |*m| {
            for (m.owned) |a| _ = mlx.mlx_array_free(a);
            self.allocator.free(m.owned);
            m.cache.deinit();
            _ = mlx.mlx_array_free(m.entry.conv_state);
            _ = mlx.mlx_array_free(m.entry.ssm_state);
            ssmFreeQsaState(&m.entry);
            m.qsa_marks.deinit();
            if (m.rerank) |*rc| rc.deinit();
            self.qwen4_mtp = null;
        }
        if (self.qwen4) |st| {
            st.deinit();
            self.allocator.destroy(st);
            self.qwen4 = null;
        }
        if (self.moe_owned_bf16) |arrs| {
            for (arrs) |a| _ = mlx.mlx_array_free(a);
            self.allocator.free(arrs);
        }
        // self.s is the thread's default GPU stream (not owned by us) — don't free it.
        _ = mlx.mlx_stream_free(self.s);
        // The free above is a no-op on the default stream's wrapper but we keep it for symmetry
        // with the Zig copy of the mlx_stream struct that init handed us.
    }

    /// Re-bind `self.s` to the *current* thread's default GPU stream. Must be called from any
    /// thread that's about to use this Transformer for inference, since mlx 0.31.2 made streams
    /// thread-local (a stream created on thread A is invisible to thread B).
    pub fn useCurrentThreadStream(self: *Transformer) void {
        self.s = mlx.gpuStream();
    }

    pub fn actDtype(self: *const Transformer) mlx.mlx_dtype {
        return self.config.actDtype();
    }

    pub fn ssmStateDtype(self: *const Transformer) mlx.mlx_dtype {
        return self.config.ssmStateDtype();
    }

    /// The trunk's lm_head for a caller holding another model's activation
    /// (an MTP head, a drafter): cast to the trunk's activation dtype, as the
    /// trunk feeds its own head, then the trunk's own head dispatch
    /// (`lmHeadProject`: Hadamard signs, row-scaled FP8, per-weight quant
    /// params, dense bf16). A new head format needs no branch at the caller.
    pub fn lmHeadForeign(self: *const Transformer, x: mlx.mlx_array) !mlx.mlx_array {
        if (mlx.mlx_array_dtype(x) == self.actDtype()) return self.lmHeadProject(x);
        var xc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xc);
        try mlx.check(mlx.mlx_astype(&xc, x, self.actDtype(), self.s));
        return self.lmHeadProject(xc);
    }

    pub var qsa_engaged_logged: bool = false;

    /// `--mtp-head-kv-quant` opts the qwen4 MTP head's KV into `--kv-quant`; the default
    /// stays dense bf16 (the head is a lossy draft and short prompts dequantize under the
    /// fused-kernel floor). Set once at parse; the override is a test seam.
    pub var mtp_head_kv_quant_override: ?bool = null;
    pub var mtp_head_kv_quant_flag: bool = false;
    pub var qwen4_trace: ?*Qwen4Trace = null;
};

test {
    _ = @import("load/weights.zig");
    _ = @import("../../sdk/blocks/gdn/kernels.zig");
    _ = @import("../../sdk/blocks/conv/depthwise.zig");
    _ = @import("../../sdk/testing/parity.zig");
    _ = @import("../../sdk/blocks/linear/verify_qmm.zig");
    _ = @import("../../mlx/device.zig");
    _ = @import("../../sdk/blocks/linear/verify_qmm_nax.zig");
    _ = @import("../../sdk/blocks/attention/sdpa.zig");
    _ = @import("../../sdk/blocks/attention/attn256.zig");
    _ = @import("../../sdk/blocks/norms/residual_norm.zig");
    _ = @import("../../plugins/qwen4_exp/qsa/gather.zig");
    _ = @import("../../plugins/qwen4_exp/qsa/select.zig");
    _ = @import("../../plugins/qwen4_exp/mtp.zig");
    _ = @import("../../sdk/blocks/linear/verify_projection.zig");
    _ = @import("../../plugins/qwen4_exp/qsa/score.zig");
    _ = @import("../../plugins/qwen4_exp/qsa/sparse_attn.zig");
    _ = @import("../../mlx/kernel_cache.zig");
    _ = @import("../../plugins/qwen4_exp/qsa/arms.zig");
    _ = @import("../../mlx/array.zig");
    _ = @import("../../plugins/qwen4_exp/qsa/decode_gather.zig");
    _ = @import("../../plugins/qwen4_exp/qsa/verify_gather.zig");
    _ = @import("../../plugins/qwen4_exp/qsa/mask.zig");
    _ = @import("kv/cache.zig");
    _ = @import("../../sdk/blocks/attention/kv_view.zig");
    _ = @import("kv/sliding.zig");
    _ = @import("batched/layered.zig");
    _ = @import("../../plugins/glm5_next/forward.zig");
    _ = @import("../../plugins/qwen4_exp/ple.zig");
    _ = @import("forward_ctx.zig");
    _ = @import("ssm/entry.zig");
    _ = @import("../../plugins/qwen4_exp/qsa/history.zig");
    _ = @import("ssm/group.zig");
    _ = @import("ssm/checkpoint.zig");
    _ = @import("../../plugins/qwen4_exp/qsa/pooled.zig");
    _ = @import("../../sdk/blocks/attention/batched.zig");
    _ = @import("../../plugins/qwen4_exp/qsa/checkpoint.zig");
    _ = @import("prompt_cache.zig");
    _ = @import("layer_weights.zig");
    _ = @import("../../plugins/glm5_next/weights.zig");
    _ = @import("../../plugins/qwen4_exp/weights.zig");
    _ = @import("../../sdk/blocks/gdn/gated_delta_net.zig");
    _ = @import("../../sdk/blocks/mlp/dense.zig");
    _ = @import("ane_prefill.zig");
    _ = @import("../../sdk/blocks/moe/weights.zig");
    _ = @import("forward/layered.zig");
    _ = @import("../../sdk/quant/params.zig");
    _ = @import("spec/mtp_nax_profile.zig");
    _ = @import("../../sdk/blocks/attention/kv_quant_attn.zig");
    _ = @import("../../sdk/blocks/norms/grouped_rms.zig");
    _ = @import("load/init.zig");
    _ = @import("compiled.zig");
    _ = @import("../../sdk/blocks/gdn/gate.zig");
    _ = @import("../../plugins/qwen4_exp/hc.zig");
    _ = @import("../../sdk/blocks/moe/routing.zig");
    _ = @import("../../sdk/blocks/linear/qmatmul.zig");
    _ = @import("diag/proj_bench.zig");
    _ = @import("../../sdk/blocks/norms/rms.zig");
    _ = @import("../../sdk/quant/ternary.zig");
    _ = @import("../../sdk/quant/hadamard.zig");
    _ = @import("../../sdk/blocks/embedding.zig");
    _ = @import("../../sdk/vision/splice.zig");
    _ = @import("../../sdk/blocks/attention/gates.zig");
    _ = @import("../../sdk/blocks/mlp/activations.zig");
    _ = @import("../../sdk/blocks/conv/conv1d_cache.zig");
    _ = @import("../../sdk/blocks/moe/experts.zig");
    _ = @import("eval_cadence.zig");
    _ = @import("../../plugins/qwen4_exp/forward.zig");
    _ = @import("dispatch.zig");
    _ = @import("spec/trunk_hooks.zig");
    _ = @import("warmup.zig");
    _ = @import("capture.zig");
    _ = @import("../../plugins/gemma4/ple.zig");
    _ = @import("forward/dense.zig");
    _ = @import("batched/dense.zig");
    _ = @import("batched/row_axis_verify.zig");
    _ = @import("batched/caps.zig");
    _ = @import("../../sdk/blocks/rope/rope.zig");
    _ = @import("../../plugins/qwen4_exp/qsa/indexer.zig");
    _ = @import("../../plugins/qwen4_exp/attention.zig");
    _ = @import("../../plugins/gemma4/moe_tail.zig");
    _ = @import("diag/profile.zig");
    _ = @import("forward/encoder_head.zig");
    _ = @import("../../plugins/gemma3/encoder.zig");
    _ = @import("../../sdk/blocks/rope/mrope.zig");
    _ = @import("../../sdk/blocks/rope/yarn.zig");
    _ = @import("../../sdk/blocks/attention/gated_full.zig");
    _ = @import("../../sdk/blocks/attention/mla.zig");
    _ = @import("../../sdk/blocks/attention/proj.zig");
    _ = @import("../../sdk/blocks/attention/sinks.zig");
    _ = @import("../../plugins/laguna/ubench_test.zig");
    _ = @import("../../plugins/inkling/attention.zig");
    _ = @import("../../plugins/gemma4/attention.zig");
    _ = @import("verify/joined_rows.zig");
    _ = @import("../../plugins/inkling/moe.zig");
    _ = @import("../../sdk/blocks/moe/moe.zig");
    _ = @import("../../sdk/blocks/moe/decode.zig");
    _ = @import("../../sdk/blocks/attention/masks.zig");
    _ = @import("load/layers_dense.zig");
    _ = @import("../../sdk/quant/mxfp8.zig");
    _ = @import("../../plugins/glm5_next/load.zig");
    _ = @import("../../plugins/qwen4_exp/load.zig");
    _ = @import("load/layers_layered.zig");
    _ = @import("../../sdk/blocks/kda/gate.zig");
    _ = @import("../../sdk/blocks/moe/router_kernel.zig");
    _ = @import("../../mlx/dtype.zig");
    _ = @import("../../sdk/diag/standin.zig");
    _ = @import("../../plugins/qwen4_exp/prof.zig");
    _ = @import("env_transformer.zig");
    _ = @import("../../sdk/blocks/linear/fused_rows.zig");
    _ = @import("../../sdk/blocks/attention/decode_attn_quant.zig");
    _ = @import("../../sdk/blocks/rope/qk_norm_rope.zig");
    _ = @import("../../sdk/blocks/gdn/decode_fused.zig");
    _ = @import("../../sdk/blocks/mlp/fused_activations.zig");
    _ = @import("../../sdk/blocks/moe/gather_qmv.zig");
    _ = @import("../../plugins/qwen4_exp/hc_fused.zig");
    _ = @import("../../sdk/blocks/moe/gather_qmv_rows.zig");
    _ = @import("../../sdk/blocks/moe/down_reduce.zig");
    _ = @import("../../sdk/blocks/linear/qmv_rows.zig");
    _ = @import("../../sdk/blocks/linear/prefill_dq_gemm.zig");
    _ = @import("../../sdk/blocks/moe/verify_kernels.zig");
    _ = @import("../../sdk/quant/exl3.zig");
    _ = @import("../../plugins/hy_v3/load.zig");
    _ = @import("../../plugins/laguna/load.zig");
    _ = @import("../../plugins/bailing_hybrid/load.zig");
    _ = @import("modules/mlx_stream.zig");
    _ = @import("modules/shell.zig");
    _ = @import("../../plugins/mimo_v2/attention.zig");
    _ = @import("../../plugins/flint/fixtures.zig");
}
