//! plugins/qwen4_exp/qsa/sparse_attn, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns QSA_ATTN_CONSTS, QSA_ATTN_Q_CONSTS, QSA_ATTN_ROW_GRID, QSA_ATTN_Q_STRIDES, QSA_ATTN_D_STRIDES, QSA_ATTN_ROW, and 67 more.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const kv_quant = @import("../../../engines/mlx/cache/kv_quant.zig");
const log = @import("../../../host/log.zig");
const qsa_decode = @import("../qsa_decode.zig");
const arms_mod = @import("arms.zig");
const transformer = @import("../../../engines/mlx/transformer.zig");
const ProfClock = @import("../../../engines/mlx/diag/profile.zig").ProfClock;
const diagEnvOn = @import("../../../engines/mlx/env_transformer.zig").diagEnvOn;
const KVCache = @import("../../../engines/mlx/kv/cache.zig").KVCache;
const scalarOf = @import("../../../engines/mlx/load/weights.zig").scalarOf;
const QsaCfgCache = @import("../../../mlx/kernel_cache.zig").QsaCfgCache;
const FUSED256_MIN_Q_LEN = @import("../../../sdk/blocks/attention/attn256.zig").FUSED256_MIN_Q_LEN;
const attn256Cosine = @import("../../../sdk/blocks/attention/attn256.zig").attn256Cosine;
const attn256MaxDiff = @import("../../../sdk/blocks/attention/attn256.zig").attn256MaxDiff;
const attn256RandBf16 = @import("../../../sdk/blocks/attention/attn256.zig").attn256RandBf16;
const attn256Reference = @import("../../../sdk/blocks/attention/attn256.zig").attn256Reference;
const attn256UlpBar = @import("../../../sdk/blocks/attention/attn256.zig").attn256UlpBar;
const fusedSdpa256Masked = @import("../../../sdk/blocks/attention/attn256.zig").fusedSdpa256Masked;
const qsaFusedEnabled = @import("../../../sdk/blocks/attention/attn256.zig").qsaFusedEnabled;
const DenseKVView = @import("../../../sdk/blocks/attention/kv_view.zig").DenseKVView;
const splitMaskedSdpa256 = @import("../../../sdk/blocks/attention/sdpa.zig").splitMaskedSdpa256;
const ShapeKey = @import("../../../sdk/blocks/moe/router_kernel.zig").ShapeKey;
const qwen4Standin = @import("../../../sdk/diag/standin.zig").qwen4Standin;
const standinRef = @import("../../../sdk/diag/standin.zig").standinRef;
const testRandWeightBf16 = @import("../../../sdk/testing/parity.zig").testRandWeightBf16;
const OneShotBits = @import("arms.zig").OneShotBits;
const QsaArmCounts = @import("arms.zig").QsaArmCounts;
const qsaWidthBucket = @import("arms.zig").qsaWidthBucket;
const qsaDecodeGatherAttn = @import("decode_gather.zig").qsaDecodeGatherAttn;
const ATTN_QSA256_KERNEL_HEADER = @import("gather.zig").ATTN_QSA256_KERNEL_HEADER;
const gatherQsa256 = @import("gather.zig").gatherQsa256;
const qsaDecodeGatherEnabled = @import("gather.zig").qsaDecodeGatherEnabled;
const qsaGatherBk = @import("gather.zig").qsaGatherBk;
const qsaGatherEnabled = @import("gather.zig").qsaGatherEnabled;
const qsaGatherMinKv = @import("gather.zig").qsaGatherMinKv;
const qsaHistoryShareEnabled = @import("gather.zig").qsaHistoryShareEnabled;
const qsaPrefillGatherMinKv = @import("gather.zig").qsaPrefillGatherMinKv;
const qsaMaskFromBlocks = @import("mask.zig").qsaMaskFromBlocks;
const qsaScoreSheetBudget = @import("score.zig").qsaScoreSheetBudget;
const qsaSelectKernelEnabled = @import("select.zig").qsaSelectKernelEnabled;
const qsaSelectSplitEnabled = @import("select.zig").qsaSelectSplitEnabled;
const qsaSelectTg = @import("select.zig").qsaSelectTg;
const QSA_VERIFY_GATHER_MIN_KV_DENSE = @import("verify_gather.zig").QSA_VERIFY_GATHER_MIN_KV_DENSE;
const qsaVerifyBlocksHost = @import("verify_gather.zig").qsaVerifyBlocksHost;
const qsaVerifyGatherAttn = @import("verify_gather.zig").qsaVerifyGatherAttn;
const qsaVerifyGatherEnabled = @import("verify_gather.zig").qsaVerifyGatherEnabled;
const qsaVerifyGatherMinKvFor = @import("verify_gather.zig").qsaVerifyGatherMinKvFor;

// ── Fused sparse attention at verify widths and batched decode (msv_attn_qsa256_q) ──
//
// At 2 <= S < 16 the QSA path spent ~50 dependent ops per layer and attended a fixed union
// of the S rows' selections under a mask (6x the MACs at S=6, plus a 12.6 MB slab). This is
// one dispatch: `msv_attn_qsa256` verbatim with the device K/V reads replaced by an in-kernel
// affine unpack (`qkv_attn_dec`'s dense packing). Each row sees its own blocks then its own
// ragged tail, so causality is automatic and the INT_MAX sentinels are never read. A bf16
// cache runs the same split pass with `msv_attn_qsa256`'s plain uint4 loads: the source is
// built from shared pieces, and only the prologue, the K/V loads and their bases differ per variant.
/// Shared by every variant of the split pass: tile constants.
const QSA_ATTN_CONSTS =
    \\constexpr int BD = 256;
    \\constexpr int LDK = BK + 8;
    \\constexpr int LDV = BD + 8;
    \\constexpr int NT = 32 * NSG;
    \\constexpr int KT = BK / 8;
    \\
;

/// Quantized variant: unpack constants (BITS is its template).
const QSA_ATTN_Q_CONSTS =
    \\constexpr int VPW = 32 / BITS;
    \\constexpr int NW = 8 / VPW;
    \\constexpr uint MASKB = (1u << BITS) - 1u;
    \\
;

/// Shared: the row's geometry and grid position. Both variants name their K/V inputs
/// `kq`/`vq`, so `kq_shape` is the key count in either.
const QSA_ATTN_ROW_GRID =
    \\
    \\const int qL = q_shape[2];
    \\const int kL = kq_shape[2];
    \\const int Hq = q_shape[1];
    \\const int Hk = kq_shape[1];
    \\const int gqa = Hq / Hk;
    \\const int KB = blocks_shape[2];
    \\
    \\const int s = int(threadgroup_position_in_grid.x);
    \\const int hk = int(threadgroup_position_in_grid.y);
    \\// One view pins batch to 1 (qs[0] != n declines), so the grid's z is
    \\// free to carry the split index instead of the batch index.
    \\const int bb = 0;
    \\const int jsp = int(threadgroup_position_in_grid.z);
    \\
;

/// The shared fragments read strides only through `qS`/`blS`/`kqS`/..., which each kernel's prologue
/// defines from its own inputs (these two copy the bound strides; the slot kernel derives them from `meta`).
const QSA_ATTN_Q_STRIDES =
    \\const long qS[3] = {q_strides[0], q_strides[1], q_strides[2]};
    \\const long blS[2] = {blocks_strides[0], blocks_strides[1]};
    \\const long kqS[3] = {kq_strides[0], kq_strides[1], kq_strides[2]};
    \\const long kscS[3] = {ksc_strides[0], ksc_strides[1], ksc_strides[2]};
    \\const long kbiS[3] = {kbi_strides[0], kbi_strides[1], kbi_strides[2]};
    \\const long vqS[3] = {vq_strides[0], vq_strides[1], vq_strides[2]};
    \\const long vscS[3] = {vsc_strides[0], vsc_strides[1], vsc_strides[2]};
    \\const long vbiS[3] = {vbi_strides[0], vbi_strides[1], vbi_strides[2]};
    \\
;

const QSA_ATTN_D_STRIDES =
    \\const long qS[3] = {q_strides[0], q_strides[1], q_strides[2]};
    \\const long blS[2] = {blocks_strides[0], blocks_strides[1]};
    \\const long kqS[4] = {kq_strides[0], kq_strides[1], kq_strides[2], kq_strides[3]};
    \\const long vqS[4] = {vq_strides[0], vq_strides[1], vq_strides[2], vq_strides[3]};
    \\
;

/// Shared: the row's visible range, its split and its block list.
const QSA_ATTN_ROW =
    \\const ushort lane = ushort(thread_index_in_simdgroup);
    \\const ushort warp = ushort(simdgroup_index_in_threadgroup);
    \\const int tix = int(thread_index_in_threadgroup);
    \\
    \\const float scale_log2e = scl[0] * 1.44269504088896340736f;
    \\// Query s sits at absolute key position p (bottom-right aligned). Its
    \\// visible keys: `count` complete blocks, then the tail [tail_start, p].
    \\const int p = (kL - qL) + s;
    \\const int complete = (p + 1) / RATIO;
    \\const int count = metal::min(complete, KB);
    \\const int sel_len = count * RATIO;
    \\const int tail_start = complete * RATIO;
    \\const int L = sel_len + (p + 1 - tail_start);
    \\// Split-K. This threadgroup owns virtual keys [t_lo, t_hi) of the row's
    \\// own range [0, L). The virtual index already concatenates the row's
    \\// selected blocks and THEN its ragged tail, so a contiguous virtual
    \\// range is exactly a contiguous block range — plus, for the last
    \\// non-empty split, the tail. Boundaries are BK-aligned so every tile
    \\// but a split's last is full. A split starting past L is EMPTY and
    \\// writes the merge's identity.
    \\const int tiles = (L + BK - 1) / BK;
    \\// BALANCED=0 (default, MEASURED): ceil(tiles/NSPLIT) per split, so the
    \\// last splits can be short or empty — at the swept shape 129 tiles over
    \\// 16 splits is 14 full, 1 partial, 1 empty. BALANCED=1 spreads the
    \\// remainder one tile per split instead. Strictly better load balance on
    \\// paper; NOT the default because the sweep that chose NSPLIT=16 ran on
    \\// the ceil distribution, and changing how work is divided invalidates
    \\// it. Same outputs either way — the split boundaries move, the merge is
    \\// exact over whatever they are.
    \\int t_lo, t_hi;
    \\if (BALANCED) {
    \\  const int base_t = tiles / NSPLIT;
    \\  const int rem_t = tiles % NSPLIT;
    \\  const int start_t = jsp * base_t + metal::min(jsp, rem_t);
    \\  const int cnt_t = base_t + ((jsp < rem_t) ? 1 : 0);
    \\  t_lo = start_t * BK;
    \\  t_hi = metal::min(L, (start_t + cnt_t) * BK);
    \\} else {
    \\  const int tiles_per = (tiles + NSPLIT - 1) / NSPLIT;
    \\  t_lo = jsp * tiles_per * BK;
    \\  t_hi = metal::min(L, t_lo + tiles_per * BK);
    \\}
    \\
    \\const device int* blk = blocks + (long)bb * blS[0] + (long)s * blS[1];
    \\
;

/// Quantized variant: per-head bases of the six packed arrays.
const QSA_ATTN_Q_BASES =
    \\const long kq_base  = (long)bb * kqS[0]  + (long)hk * kqS[1];
    \\const long ksc_base = (long)bb * kscS[0] + (long)hk * kscS[1];
    \\const long kbi_base = (long)bb * kbiS[0] + (long)hk * kbiS[1];
    \\const long vq_base  = (long)bb * vqS[0]  + (long)hk * vqS[1];
    \\const long vsc_base = (long)bb * vscS[0] + (long)hk * vscS[1];
    \\const long vbi_base = (long)bb * vbiS[0] + (long)hk * vbiS[1];
    \\
;

/// Dense variant: per-head bases of the bf16 K and V, and whether each can take uint4
/// loads. A uint4 needs a 16-byte address: the bound pointer (which carries the view's data
/// offset, invisible to the host) and every row start. Anything else — an offset slice, a
/// padded or non-unit innermost layout — reads element by element through its strides.
const QSA_ATTN_D_BASES =
    \\const long kq_base = (long)bb * kqS[0] + (long)hk * kqS[1];
    \\const long vq_base = (long)bb * vqS[0] + (long)hk * vqS[1];
    \\const bool k_vec = (reinterpret_cast<ulong>(kq) & 15ul) == 0 && kqS[3] == 1 &&
    \\    (kqS[2] & 7) == 0 && (kqS[1] & 7) == 0;
    \\const bool v_vec = (reinterpret_cast<ulong>(vq) & 15ul) == 0 && vqS[3] == 1 &&
    \\    (vqS[2] & 7) == 0 && (vqS[1] & 7) == 0;
    \\
;

/// Shared: staging, Q fragments, and the head of the key-tile loop.
const QSA_ATTN_TILE_OPEN =
    \\
    \\threadgroup T KVs[LDK * BD];
    \\threadgroup T* Ks = KVs;
    \\threadgroup T* Vs = KVs;
    \\
    \\const short2 sc = msv_coord(lane);
    \\const short sn = sc.x;
    \\const short sm = sc.y;
    \\const short tm = 8 * short(warp);
    \\const int Ks_off = sm * LDK + sn;
    \\const int Vs_off = sm * LDV + sn;
    \\const int row = tm + sm;
    \\const bool row_ok = row < gqa;
    \\
    \\float2 Qfrag[BD / 8];
    \\if (row_ok) {
    \\  const device T* Qrow = q + bb * qS[0] + (long)(hk * gqa + row) * qS[1] + (long)s * qS[2];
    \\  for (int dd = 0; dd < BD / 8; ++dd) {
    \\    const vec<T, 2> pr = *((const device vec<T, 2>*)(Qrow + dd * 8 + sn));
    \\    Qfrag[dd] = float2(float(pr.x), float(pr.y));
    \\  }
    \\} else {
    \\  for (int dd = 0; dd < BD / 8; ++dd) Qfrag[dd] = float2(0.0f);
    \\}
    \\float2 Ofrag[BD / 8];
    \\for (int i = 0; i < BD / 8; ++i) Ofrag[i] = float2(0.0f);
    \\float max_score = -3.0e38f;
    \\float sum_score = 0.0f;
    \\
    \\for (int t0 = t_lo; t0 < t_hi; t0 += BK) {
    \\  const int rows_k = metal::min(BK, t_hi - t0);
    \\
    \\
;

/// Quantized variant: K tile, unpacked in-kernel.
const QSA_ATTN_Q_KLOAD =
    \\  // K tile: dequantize into the SAME transposed staging the dense
    \\  // kernel fills, so the fragment math below is untouched. The 8 dims of
    \\  // one chunk share a quant group (GS is a multiple of 8), so the
    \\  // scale/bias pair is read once per chunk, not per element.
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  for (int i = tix; i < BK * (BD / 8); i += NT) {
    \\    const int r = i >> 5;
    \\    const int c8 = i & 31;
    \\    const int cb = c8 * 8;
    \\    T ev[8];
    \\    if (r < rows_k) {
    \\      const int pos = msv_qsa_pos(blk, t0 + r, sel_len, tail_start, RATIO);
    \\      const long wrow = kq_base + (long)pos * kqS[2];
    \\      const int g = cb / GS;
    \\      const float sj = float(ksc[ksc_base + (long)pos * kscS[2] + (long)g]);
    \\      const float bj = float(kbi[kbi_base + (long)pos * kbiS[2] + (long)g]);
    \\      for (int wj = 0; wj < NW; ++wj) {
    \\        const uint w = uint(kq[wrow + (long)(cb / VPW + wj)]);
    \\        for (int u = 0; u < VPW; ++u) ev[wj * VPW + u] = T(float((w >> (u * BITS)) & MASKB) * sj + bj);
    \\      }
    \\    } else {
    \\      for (int j = 0; j < 8; ++j) ev[j] = T(0.0f);
    \\    }
    \\    for (int j = 0; j < 8; ++j) Ks[(cb + j) * LDK + r] = ev[j];
    \\  }
    \\
;

/// Dense variant: K tile, read straight from the bf16 cache.
const QSA_ATTN_D_KLOAD =
    \\  // K tile: `msv_attn_qsa256`'s load, eight elements per uint4 (element
    \\  // loads when `k_vec` is false), into the transposed staging.
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  for (int i = tix; i < BK * (BD / 8); i += NT) {
    \\    const int r = i >> 5;
    \\    const int c8 = i & 31;
    \\    uint4 w = uint4(0);
    \\    thread T* e = (thread T*)&w;
    \\    if (r < rows_k) {
    \\      const int pos = msv_qsa_pos(blk, t0 + r, sel_len, tail_start, RATIO);
    \\      const long rb = kq_base + (long)pos * kqS[2];
    \\      if (k_vec) {
    \\        w = *((const device uint4*)(kq + rb) + c8);
    \\      } else {
    \\        for (int j = 0; j < 8; ++j) e[j] = kq[rb + (long)(c8 * 8 + j) * kqS[3]];
    \\      }
    \\    }
    \\    const int cb = c8 * 8;
    \\    for (int j = 0; j < 8; ++j) Ks[(cb + j) * LDK + r] = e[j];
    \\  }
    \\
;

/// Shared: scores and the ragged-tile mask.
const QSA_ATTN_SCORES =
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\
    \\  float2 Sfrag[KT];
    \\  for (int i = 0; i < KT; ++i) Sfrag[i] = float2(0.0f);
    \\  for (int dd = 0; dd < BD / 8; ++dd) {
    \\    const float2 qf = Qfrag[dd];
    \\    const int kbase = Ks_off + dd * 8 * LDK;
    \\    for (int kt = 0; kt < KT; ++kt) {
    \\      const float2 kf = float2(float(Ks[kbase + kt * 8]), float(Ks[kbase + kt * 8 + 1]));
    \\      msv_mma(Sfrag[kt], qf, kf);
    \\    }
    \\  }
    \\  for (int kt = 0; kt < KT; ++kt) Sfrag[kt] *= scale_log2e;
    \\  if (rows_k < BK) {
    \\    for (int kt = 0; kt < KT; ++kt) {
    \\      if (kt * 8 + sn >= rows_k) Sfrag[kt].x = -INFINITY;
    \\      if (kt * 8 + sn + 1 >= rows_k) Sfrag[kt].y = -INFINITY;
    \\    }
    \\  }
    \\
    \\
;

/// Quantized variant: V tile, unpacked in-kernel.
const QSA_ATTN_Q_VLOAD =
    \\  // V tile: dequantize straight into the row-major staging.
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  for (int i = tix; i < BK * (BD / 8); i += NT) {
    \\    const int r = i >> 5;
    \\    const int c8 = i & 31;
    \\    const int cb = c8 * 8;
    \\    T ev[8];
    \\    if (r < rows_k) {
    \\      const int pos = msv_qsa_pos(blk, t0 + r, sel_len, tail_start, RATIO);
    \\      const long wrow = vq_base + (long)pos * vqS[2];
    \\      const int g = cb / GS;
    \\      const float sj = float(vsc[vsc_base + (long)pos * vscS[2] + (long)g]);
    \\      const float bj = float(vbi[vbi_base + (long)pos * vbiS[2] + (long)g]);
    \\      for (int wj = 0; wj < NW; ++wj) {
    \\        const uint w = uint(vq[wrow + (long)(cb / VPW + wj)]);
    \\        for (int u = 0; u < VPW; ++u) ev[wj * VPW + u] = T(float((w >> (u * BITS)) & MASKB) * sj + bj);
    \\      }
    \\    } else {
    \\      for (int j = 0; j < 8; ++j) ev[j] = T(0.0f);
    \\    }
    \\    for (int j = 0; j < 8; ++j) Vs[r * LDV + cb + j] = ev[j];
    \\  }
    \\
;

/// Dense variant: V tile, read straight from the bf16 cache.
const QSA_ATTN_D_VLOAD =
    \\  // V tile: uint4 copies (element loads when `v_vec` is false) straight
    \\  // into the row-major staging.
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  for (int i = tix; i < BK * (BD / 8); i += NT) {
    \\    const int r = i >> 5;
    \\    const int c8 = i & 31;
    \\    uint4 w = uint4(0);
    \\    if (r < rows_k) {
    \\      const int pos = msv_qsa_pos(blk, t0 + r, sel_len, tail_start, RATIO);
    \\      const long rb = vq_base + (long)pos * vqS[2];
    \\      if (v_vec) {
    \\        w = *((const device uint4*)(vq + rb) + c8);
    \\      } else {
    \\        thread T* e = (thread T*)&w;
    \\        for (int j = 0; j < 8; ++j) e[j] = vq[rb + (long)(c8 * 8 + j) * vqS[3]];
    \\      }
    \\    }
    \\    *((threadgroup uint4*)(Vs + r * LDV) + c8) = w;
    \\  }
    \\
;

/// Shared: online softmax, P@V, and the partials for the merge.
const QSA_ATTN_TILE_CLOSE =
    \\
    \\  float new_max = max_score;
    \\  for (int kt = 0; kt < KT; ++kt) new_max = metal::max(new_max, msv_row_max(Sfrag[kt]));
    \\  float rowsum = 0.0f;
    \\  for (int kt = 0; kt < KT; ++kt) {
    \\    Sfrag[kt] = metal::exp2(Sfrag[kt] - new_max);
    \\    rowsum += msv_row_sum(Sfrag[kt]);
    \\  }
    \\  const float factor = metal::exp2(max_score - new_max);
    \\  max_score = new_max;
    \\  sum_score = sum_score * factor + rowsum;
    \\  for (int i = 0; i < BD / 8; ++i) Ofrag[i] *= factor;
    \\
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  for (int id = 0; id < BD / 8; ++id) {
    \\    const int vbase = Vs_off + id * 8;
    \\    for (int kt = 0; kt < KT; ++kt) {
    \\      const float2 vf = float2(float(Vs[vbase + kt * 8 * LDV]), float(Vs[vbase + kt * 8 * LDV + 1]));
    \\      msv_mma(Ofrag[id], Sfrag[kt], vf);
    \\    }
    \\  }
    \\}
    \\
    \\// Partials for the merge pass, UNNORMALIZED: the running max and the
    \\// running sum belong to this split alone, and only the merge knows the
    \\// row's global max. `msv_row_max`/`msv_row_sum` reduce across the four
    \\// lanes holding a row, so every lane of the row carries the same m/l
    \\// and exactly one of them (sn == 0) writes the pair.
    \\//
    \\// An empty split writes l = 0, which is the merge's skip flag: a
    \\// NON-empty split always has l >= 1, because its own maximum element
    \\// contributes exp2(max - max) = 1. So l == 0 identifies emptiness
    \\// exactly, and the merge never evaluates (-inf) - (-inf).
    \\if (row_ok) {
    \\  const int hq = hk * gqa + row;
    \\  const long pidx = ((long)hq * (long)qL + (long)s) * NSPLIT + jsp;
    \\  device float* Aptr = pacc + pidx * BD + sn;
    \\  for (int id = 0; id < BD / 8; ++id) {
    \\    Aptr[id * 8] = Ofrag[id].x;
    \\    Aptr[id * 8 + 1] = Ofrag[id].y;
    \\  }
    \\  if (sn == 0) {
    \\    const bool empty = (t_lo >= t_hi);
    \\    pml[pidx * 2] = empty ? -INFINITY : max_score;
    \\    pml[pidx * 2 + 1] = empty ? 0.0f : sum_score;
    \\  }
    \\}
;

const QSA_ATTN_Q_SOURCE = QSA_ATTN_CONSTS ++ QSA_ATTN_Q_CONSTS ++ QSA_ATTN_ROW_GRID ++ QSA_ATTN_Q_STRIDES ++ QSA_ATTN_ROW ++
    QSA_ATTN_Q_BASES ++ QSA_ATTN_TILE_OPEN ++ QSA_ATTN_Q_KLOAD ++ QSA_ATTN_SCORES ++ QSA_ATTN_Q_VLOAD ++ QSA_ATTN_TILE_CLOSE;

/// The dense (bf16 KV) variant: the same split pass with plain device loads, so a dense
/// cache keeps the verify kernel's split-K grid (never `gatherQsa256`'s prefill grid).
const QSA_ATTN_D_SOURCE = QSA_ATTN_CONSTS ++ QSA_ATTN_ROW_GRID ++ QSA_ATTN_D_STRIDES ++ QSA_ATTN_ROW ++
    QSA_ATTN_D_BASES ++ QSA_ATTN_TILE_OPEN ++ QSA_ATTN_D_KLOAD ++ QSA_ATTN_SCORES ++ QSA_ATTN_D_VLOAD ++ QSA_ATTN_TILE_CLOSE;

/// The slot-batched quantized split pass: slot `bb` on grid z, `meta` = {Hq, Hk, KB, then kv length and
/// parent capacity per slot}.
const QSA_ATTN_SLOTS_GRID =
    \\const int bb = int(threadgroup_position_in_grid.z) / NSPLIT;
    \\const int jsp = int(threadgroup_position_in_grid.z) - bb * NSPLIT;
    \\const int s = int(threadgroup_position_in_grid.x);
    \\const int hk = int(threadgroup_position_in_grid.y);
    \\const int qL = 1;
    \\const int Hq = meta[0];
    \\const int Hk = meta[1];
    \\const int gqa = Hq / Hk;
    \\const int KB = meta[2];
    \\const int kL = meta[3 + 2 * bb];
    \\const long cap = meta[4 + 2 * bb];
    \\const device auto* kq = bb == 1 ? kq1 : bb == 2 ? kq2 : bb == 3 ? kq3 : kq0;
    \\const device auto* ksc = bb == 1 ? ksc1 : bb == 2 ? ksc2 : bb == 3 ? ksc3 : ksc0;
    \\const device auto* kbi = bb == 1 ? kbi1 : bb == 2 ? kbi2 : bb == 3 ? kbi3 : kbi0;
    \\const device auto* vq = bb == 1 ? vq1 : bb == 2 ? vq2 : bb == 3 ? vq3 : vq0;
    \\const device auto* vsc = bb == 1 ? vsc1 : bb == 2 ? vsc2 : bb == 3 ? vsc3 : vsc0;
    \\const device auto* vbi = bb == 1 ? vbi1 : bb == 2 ? vbi2 : bb == 3 ? vbi3 : vbi0;
    \\const long qS[3] = {(long)Hq * qL * BD, (long)qL * BD, BD};
    \\const long blS[2] = {(long)qL * KB, KB};
    \\const long kqS[3] = {0, cap * (BD / VPW), BD / VPW};
    \\const long kscS[3] = {0, cap * (BD / GS), BD / GS};
    \\const long kbiS[3] = {0, cap * (BD / GS), BD / GS};
    \\const long vqS[3] = {0, cap * (BD / VPW), BD / VPW};
    \\const long vscS[3] = {0, cap * (BD / GS), BD / GS};
    \\const long vbiS[3] = {0, cap * (BD / GS), BD / GS};
    \\pacc += (long)bb * Hq * qL * NSPLIT * BD;
    \\pml += (long)bb * Hq * qL * NSPLIT * 2;
    \\
;

const QSA_ATTN_SLOTS_SOURCE = QSA_ATTN_CONSTS ++ QSA_ATTN_Q_CONSTS ++ QSA_ATTN_SLOTS_GRID ++ QSA_ATTN_ROW ++ QSA_ATTN_Q_BASES ++
    QSA_ATTN_TILE_OPEN ++ QSA_ATTN_Q_KLOAD ++ QSA_ATTN_SCORES ++ QSA_ATTN_Q_VLOAD ++ QSA_ATTN_TILE_CLOSE;

/// msv_attn_qsa256_qmerge: the split-K reduction, one threadgroup per (q head, query row).
/// Standard flash-decoding merge in the log2 domain; `l_j == 0` marks an empty split and is
/// skipped. NSPLIT and qL ride `pacc_shape`, so the config depends on nothing per token.
const QSA_ATTN_MERGE_SOURCE =
    \\constexpr int BD = 256;
    \\const int d = int(thread_index_in_threadgroup);
    \\const int s = int(threadgroup_position_in_grid.y);
    \\const int hq = int(threadgroup_position_in_grid.z);
    \\const int qL = pacc_shape[1];
    \\const int NS = pacc_shape[2];
    \\const long pbase = ((long)hq * (long)qL + (long)s) * NS;
    \\
    \\// Every thread reads all NS (m, l) pairs. NS <= 64 and the pairs land in
    \\// cache after the first thread of the threadgroup touches them, which is
    \\// cheaper than a barrier plus threadgroup storage for two floats.
    \\float M = -INFINITY;
    \\for (int j = 0; j < NS; ++j) {
    \\  if (pml[(pbase + j) * 2 + 1] > 0.0f) M = metal::max(M, pml[(pbase + j) * 2]);
    \\}
    \\float Z = 0.0f;
    \\float acc = 0.0f;
    \\for (int j = 0; j < NS; ++j) {
    \\  const float lj = pml[(pbase + j) * 2 + 1];
    \\  if (lj <= 0.0f) continue;
    \\  const float w = metal::exp2(pml[(pbase + j) * 2] - M);
    \\  Z += w * lj;
    \\  acc += w * pacc[(pbase + j) * BD + d];
    \\}
    \\out[((long)hq * (long)qL + (long)s) * BD + d] = T(acc / Z);
;

/// Measured split count (M5 Max, qwen4_exp, kv 62k, 8-bit KV, 12 counterbalanced boots):
/// every count 8..64 beats the union gather by 5-13%, only S=6's 16-vs-32 clears the noise
/// floor, so one flat constant. The un-split kernel (12 threadgroups at S=6) ran +15.6%.
pub const QSA_ATTN_NSPLIT_DEFAULT: c_int = 16;

/// Ceiling on the split count; past it the f32 partial buffer costs more traffic than it buys.
pub const QSA_ATTN_MAX_NSPLIT: c_int = 64;

/// Default lower width bound for the fused kernel. `MLX_SERVE_QSA_ATTN_MIN_S=1` is the S=1 measurement's lever.
pub const QSA_ATTN_MIN_S_DEFAULT: c_int = 2;

/// Width floor for a batched decode step of two or more slots: its S=1 rows take the fused kernel,
/// which beats the per-slot decode gather there (a lone stream keeps the decode gather).
/// `MLX_SERVE_QSA_ATTN_BATCHED_MIN_S=2` hands batched S=1 rows back to the decode gather.
pub const QSA_ATTN_BATCHED_MIN_S_DEFAULT: c_int = 1;

var qsa_attn_nsplit_env: ?c_int = null;
pub var qsa_attn_nsplit_override: ?c_int = null;
var qsa_attn_min_s_env: ?c_int = null;
pub var qsa_attn_min_s_override: ?c_int = null;
var qsa_attn_batched_min_s_env: ?c_int = null;

fn qsaAttnEnvInt(cache: *?c_int, name: [*:0]const u8, dflt: c_int) c_int {
    if (cache.*) |v| return v;
    const v = blk: {
        const raw = std.c.getenv(name) orelse break :blk dflt;
        const parsed = std.fmt.parseInt(c_int, std.mem.sliceTo(raw, 0), 10) catch break :blk dflt;
        // Absent or 0 means "auto".
        break :blk if (parsed <= 0) dflt else parsed;
    };
    cache.* = v;
    return v;
}

/// Splits per (row, kv head): a measured constant, independent of kv and the row's key
/// count (NSPLIT is a template). `MLX_SERVE_QSA_ATTN_NSPLIT=<n>` overrides.
/// Resolve every lazily-cached QSA env read once, from the main thread, before any other
/// thread exists: first touch of a `?bool`/`?c_int` from two threads is a race.
pub fn warmQsaEnvCaches() void {
    _ = qsaFusedEnabled();
    _ = qsaGatherEnabled();
    _ = qsaGatherMinKv();
    _ = qsaGatherBk();
    _ = qsaDecodeGatherEnabled();
    _ = qsaHistoryShareEnabled();
    _ = qsaVerifyGatherEnabled();
    _ = qsaVerifyGatherMinKvFor(false);
    _ = qsaSelectEnabledWarm();
    _ = qsaSelectSplitEnabled();
    _ = qsaSelectTg();
    _ = qsaAttnKernelEnabled();
    _ = qsaAttnDenseEnabled();
    _ = qsaAttnNSplit();
    _ = qsaAttnMinS();
    _ = qsaAttnBatchedMinS();
    _ = qsaAttnBk();
    _ = qsaAttnBalanced();
    _ = qsaScoreSheetBudget();
}

fn qsaSelectEnabledWarm() bool {
    return qsaSelectKernelEnabled();
}

pub fn qsaAttnNSplit() c_int {
    const forced = qsa_attn_nsplit_override orelse qsaAttnEnvInt(&qsa_attn_nsplit_env, "MLX_SERVE_QSA_ATTN_NSPLIT", 0);
    if (forced > 0) {
        var f: c_int = 1;
        while (f < forced and f < QSA_ATTN_MAX_NSPLIT) f *= 2;
        return f;
    }
    return QSA_ATTN_NSPLIT_DEFAULT;
}

/// Lowest query width the fused kernel serves. `MLX_SERVE_QSA_ATTN_MIN_S`.
pub fn qsaAttnMinS() c_int {
    const v = qsa_attn_min_s_override orelse qsaAttnEnvInt(&qsa_attn_min_s_env, "MLX_SERVE_QSA_ATTN_MIN_S", QSA_ATTN_MIN_S_DEFAULT);
    return @max(1, @min(v, FUSED256_MIN_Q_LEN - 1));
}

/// Lowest width the fused kernel serves for S=1 rows of a batched step with two or more slots.
pub fn qsaAttnBatchedMinS() c_int {
    const v = qsaAttnEnvInt(&qsa_attn_batched_min_s_env, "MLX_SERVE_QSA_ATTN_BATCHED_MIN_S", QSA_ATTN_BATCHED_MIN_S_DEFAULT);
    return @max(1, @min(v, FUSED256_MIN_Q_LEN - 1));
}

/// Does the fused sparse-attention kernel serve this call? Dense KV takes the kernel's bf16
/// variant (same split-K grid). It must never fall into `gatherQsa256`, the prefill kernel,
/// whose grid starves at a verify width (16k fp16: 77.7 vs 93.6 tok/s).
/// `MLX_SERVE_QSA_ATTN_DENSE=0` hands dense KV back to the mask arm / `qsaVerifyGatherAttn`.
pub fn qsaSparseAttnServes(has_quant_triple: bool, seq_len: c_int, min_s: c_int) bool {
    if (seq_len < min_s or seq_len >= FUSED256_MIN_Q_LEN) return false;
    return has_quant_triple or qsaAttnDenseEnabled();
}

/// Will the fused verify kernel take a verify-width (2..15) call on this KV type? Its switch,
/// its width floor and its dense switch; no kv floor, because it reads each row's own
/// O(budget) keys at any kv.
pub fn qsaFusedVerifyServes(quantized: bool, seq_len: c_int) bool {
    if (seq_len < 2 or seq_len >= FUSED256_MIN_Q_LEN) return false;
    return qsaAttnKernelEnabled() and qsaSparseAttnServes(quantized, seq_len, qsaAttnMinS());
}

/// The indexer's choice for one single-slot call: per-row block selection (`true`) or the
/// dense `[S, kv]` mask. Prefill and decode widths gather past `qsaGatherMinKv`. Verify widths
/// take blocks wherever the fused kernel serves them (every kv the indexer is active);
/// otherwise only past the union gather's own floor. `MLX_SERVE_QSA_ATTN_DENSE=0` (or
/// `MLX_SERVE_QSA_ATTN_KERNEL=0`) restores the union-gather floor on its KV type.
pub fn qsaWantsBlocks(batch: c_int, kv: c_int, seq_len: c_int, quantized: bool) bool {
    if (batch != 1 or !qsaGatherEnabled()) return false;
    if (qsaFusedVerifyServes(quantized, seq_len)) return true;
    if (seq_len >= FUSED256_MIN_Q_LEN) return kv > qsaPrefillGatherMinKv();
    if (kv <= qsaGatherMinKv()) return false;
    if (seq_len == 1) return qsaDecodeGatherEnabled();
    return qsaVerifyGatherEnabled() and kv > qsaVerifyGatherMinKvFor(quantized);
}

var qsa_attn_merge_kernel_cached: ?mlx.mlx_fast_metal_kernel = null;

fn getQsaAttnMergeKernel() !mlx.mlx_fast_metal_kernel {
    if (qsa_attn_merge_kernel_cached) |kk| return kk;
    const input_names = [_][*:0]const u8{ "pacc", "pml" };
    const output_names = [_][*:0]const u8{"out"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "msv_attn_qsa256_qmerge",
        in_vec,
        out_vec,
        QSA_ATTN_MERGE_SOURCE,
        ATTN_QSA256_KERNEL_HEADER,
        false,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    qsa_attn_merge_kernel_cached = kernel;
    return kernel;
}

/// Threadgroup tile depth: 16 keys stages 12 KiB, half the dense prefill kernel's footprint.
pub const QSA_ATTN_BK: c_int = 16;

var qsa_attn_bk_env: ?c_int = null;
var qsa_attn_balanced_env: ?bool = null;
pub var qsa_attn_bk_override: ?c_int = null;
pub var qsa_attn_balanced_override: ?bool = null;

/// Threadgroup tile depth, `MLX_SERVE_QSA_ATTN_BK` (default `QSA_ATTN_BK`). Must be a multiple
/// of 8 (`KT = BK / 8`). A new value owes a re-run of the NSPLIT sweep.
pub fn qsaAttnBk() c_int {
    const v = qsa_attn_bk_override orelse qsaAttnEnvInt(&qsa_attn_bk_env, "MLX_SERVE_QSA_ATTN_BK", QSA_ATTN_BK);
    if (v <= 0 or @rem(v, 8) != 0 or v > 32) return QSA_ATTN_BK;
    return v;
}

/// `MLX_SERVE_QSA_ATTN_BALANCED=1` spreads the tile remainder one per split. Default off (invalidates the sweep).
pub fn qsaAttnBalanced() bool {
    if (qsa_attn_balanced_override) |v| return v;
    if (qsa_attn_balanced_env) |v| return v;
    const v = blk: {
        const raw = std.c.getenv("MLX_SERVE_QSA_ATTN_BALANCED") orelse break :blk false;
        break :blk std.mem.eql(u8, std.mem.sliceTo(raw, 0), "1");
    };
    qsa_attn_balanced_env = v;
    return v;
}

var qsa_attn_q_kernel_cached: ?mlx.mlx_fast_metal_kernel = null;
var qsa_attn_d_kernel_cached: ?mlx.mlx_fast_metal_kernel = null;

fn getQsaAttnDKernel() !mlx.mlx_fast_metal_kernel {
    if (qsa_attn_d_kernel_cached) |kk| return kk;
    const input_names = [_][*:0]const u8{ "q", "scl", "blocks", "kq", "vq" };
    const output_names = [_][*:0]const u8{ "pacc", "pml" };
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "msv_attn_qsa256_dsplit",
        in_vec,
        out_vec,
        QSA_ATTN_D_SOURCE,
        ATTN_QSA256_KERNEL_HEADER,
        false, // K/V are cache views (see msv_attn_qsa256)
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    qsa_attn_d_kernel_cached = kernel;
    return kernel;
}

fn getQsaAttnQKernel() !mlx.mlx_fast_metal_kernel {
    if (qsa_attn_q_kernel_cached) |kk| return kk;
    const input_names = [_][*:0]const u8{ "q", "scl", "blocks", "kq", "ksc", "kbi", "vq", "vsc", "vbi" };
    const output_names = [_][*:0]const u8{ "pacc", "pml" };
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "msv_attn_qsa256_qsplit",
        in_vec,
        out_vec,
        QSA_ATTN_Q_SOURCE,
        ATTN_QSA256_KERNEL_HEADER,
        false, // packed triples are cache VIEWS — a contiguous copy erases the win
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    qsa_attn_q_kernel_cached = kernel;
    return kernel;
}

var qsa_attn_slots_kernel_cached: ?mlx.mlx_fast_metal_kernel = null;

/// One pipeline for 2..4 slots: each slot's six triples bind slot-major after q/scl/blocks (so one slot's inputs
/// lay out as the solo kernel's), `meta` last; an absent slot repeats slot 0's.
fn getQsaAttnSlotsKernel() !mlx.mlx_fast_metal_kernel {
    if (qsa_attn_slots_kernel_cached) |kk| return kk;
    const input_names = comptime blk: {
        var nm: [28][*:0]const u8 = undefined;
        nm[0..3].* = .{ "q", "scl", "blocks" };
        for (0..24) |i| nm[3 + i] = std.fmt.comptimePrint("{s}{d}", .{ ([_][]const u8{ "kq", "ksc", "kbi", "vq", "vsc", "vbi" })[i % 6], i / 6 });
        nm[27] = "meta";
        break :blk nm;
    };
    const output_names = [_][*:0]const u8{ "pacc", "pml" };
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new("msv_attn_qsa256_qsplit_slots", in_vec, out_vec, QSA_ATTN_SLOTS_SOURCE, ATTN_QSA256_KERNEL_HEADER, false, false);
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    qsa_attn_slots_kernel_cached = kernel;
    return kernel;
}

var qsa_attn_kernel_env: ?bool = null;

/// Fused QSA sparse attention at verify widths and batched decode (default on).
/// `MLX_SERVE_QSA_ATTN_KERNEL=0` restores the gather arms.
pub fn qsaAttnKernelEnabled() bool {
    if (transformer.qsa_attn_kernel_override) |v| return v;
    if (qsa_attn_kernel_env) |v| return v;
    const v = blk: {
        const raw = std.c.getenv("MLX_SERVE_QSA_ATTN_KERNEL") orelse break :blk true;
        break :blk !std.mem.eql(u8, std.mem.sliceTo(raw, 0), "0");
    };
    qsa_attn_kernel_env = v;
    return v;
}

pub var qsa_attn_dense_override: ?bool = null;
var qsa_attn_dense_env: ?bool = null;

/// The fused kernel's bf16 (dense KV) variant (default on). `MLX_SERVE_QSA_ATTN_DENSE=0`
/// hands dense KV back to the mask arm and the union gather.
pub fn qsaAttnDenseEnabled() bool {
    if (qsa_attn_dense_override) |v| return v;
    if (qsa_attn_dense_env) |v| return v;
    const v = blk: {
        const raw = std.c.getenv("MLX_SERVE_QSA_ATTN_DENSE") orelse break :blk true;
        break :blk !std.mem.eql(u8, std.mem.sliceTo(raw, 0), "0");
    };
    qsa_attn_dense_env = v;
    return v;
}

const QsaAttnCfgKey = struct {
    /// Its batch axis is the slot count.
    q_shape: ShapeKey,
    /// Shapes the grid and is not recoverable from the q shape.
    h_kv: c_int,
    bits: u8,
    gs: u32,
    nsg: c_int,
    nsplit: c_int,
    bk: c_int,
    balanced: bool,
    ratio: c_int,
    dtype: mlx.mlx_dtype,
};
var qsa_attn_cfgs = QsaCfgCache(QsaAttnCfgKey, 2){};

var qsa_attn_engaged_bits: OneShotBits = .{};

/// The K/V inputs one variant of the split pass reads, validated. `bits == 16` is the dense
/// (bf16) variant; its two arrays bind to `kq`/`vq`.
const QsaAttnKvInputs = struct {
    arrays: [6]mlx.mlx_array,
    n: usize,
    h_kv: c_int,
    kv: c_int,
    bits: u8,
    gs: u32,
};

/// The packed affine triples, or null when the quantized variant cannot read them.
fn qsaAttnQuantInputs(kv_view: *const DenseKVView) ?QsaAttnKvInputs {
    if (kv_view.bits != 4 and kv_view.bits != 8) return null;
    if (kv_view.group_size == 0 or @rem(@as(c_int, 256), @as(c_int, @intCast(kv_view.group_size))) != 0) return null;
    // One quant group per 8-dim staging chunk.
    if (kv_view.group_size % 8 != 0) return null;
    // `has_quant_triple` promises all six handles; a null ctx would fault, so check.
    const arrays = [6]mlx.mlx_array{
        kv_view.k_triple_q, kv_view.k_triple_scales, kv_view.k_triple_biases,
        kv_view.v_triple_q, kv_view.v_triple_scales, kv_view.v_triple_biases,
    };
    for (arrays) |a| {
        if (a.ctx == null) return null;
    }
    if (mlx.mlx_array_ndim(kv_view.k_triple_q) != 4 or mlx.mlx_array_ndim(kv_view.v_triple_q) != 4) return null;
    const ks = mlx.getShape(kv_view.k_triple_q);
    const vs = mlx.getShape(kv_view.v_triple_q);
    const vpw: c_int = @divExact(@as(c_int, 32), @as(c_int, kv_view.bits));
    if (ks.len != 4 or vs.len != 4 or ks[0] != 1 or vs[0] != 1) return null;
    if (ks[3] * vpw != 256 or vs[3] * vpw != 256) return null;
    const h_kv = ks[1];
    const kv = ks[2];
    if (vs[1] != h_kv or vs[2] != kv) return null;
    const groups: c_int = @divExact(@as(c_int, 256), @as(c_int, @intCast(kv_view.group_size)));
    const ksc_sh = mlx.getShape(kv_view.k_triple_scales);
    const vsc_sh = mlx.getShape(kv_view.v_triple_scales);
    if (ksc_sh.len != 4 or vsc_sh.len != 4 or ksc_sh[3] != groups or vsc_sh[3] != groups) return null;
    if (ksc_sh[2] != kv or vsc_sh[2] != kv) return null;
    // `ensure_row_contiguous = false` (the packed triples are cache views), so the kernel
    // indexes every innermost axis with unit stride; decline anything else.
    for (arrays) |a| {
        if (mlx.mlx_array_strides(a)[3] != 1) return null;
    }
    return .{ .arrays = arrays, .n = 6, .h_kv = h_kv, .kv = kv, .bits = kv_view.bits, .gs = kv_view.group_size };
}

/// The bf16 K/V views, or null when the dense variant cannot read them. Layout is the
/// kernel's business: a lazy view reports row-contiguous strides until it is evaluated and
/// mlx-c exposes no data offset, so the host cannot judge alignment. The kernel checks the
/// bound pointer and the real strides and falls back to element loads (`k_vec`/`v_vec`).
fn qsaAttnDenseInputs(kv_view: *const DenseKVView) ?QsaAttnKvInputs {
    const k = kv_view.k;
    const v = kv_view.v;
    if (k.ctx == null or v.ctx == null) return null;
    if (mlx.mlx_array_ndim(k) != 4 or mlx.mlx_array_ndim(v) != 4) return null;
    if (mlx.mlx_array_dtype(k) != .bfloat16 or mlx.mlx_array_dtype(v) != .bfloat16) return null;
    const ks = mlx.getShape(k);
    const vs = mlx.getShape(v);
    if (ks[0] != 1 or ks[3] != 256 or !std.mem.eql(c_int, ks, vs)) return null;
    var arrays: [6]mlx.mlx_array = @splat(.{ .ctx = null });
    arrays[0] = k;
    arrays[1] = v;
    return .{ .arrays = arrays, .n = 2, .h_kv = ks[1], .kv = ks[2], .bits = 16, .gs = 0 };
}

/// One fused sparse-attention dispatch for a verify-width block or a batched decode row: each
/// row indexes its own selected blocks plus its own tail. Quantized KV unpacks its triples
/// in-kernel; dense (bf16) KV takes the same split pass with plain loads. Null = declined.
pub fn qsaSparseAttn(
    s: mlx.mlx_stream,
    q_rope: mlx.mlx_array, // [1, Hq, S, 256] bf16, post-RoPE
    kv_view: *const DenseKVView, // cache AFTER this block's append
    blocks: mlx.mlx_array, // [1, S, kb] int32 ascending, INT_MAX padded
    ratio: c_int,
    attn_scale: f32,
) !?mlx.mlx_array {
    return qsaSparseAttnFloor(s, q_rope, kv_view[0..1], &.{blocks}, ratio, attn_scale, qsaAttnMinS());
}

/// `qsaSparseAttn` under an explicit width floor (the batched decode floor for S=1 rows of a batch).
/// One view and block list per q batch row: 2..4 quantized S=1 slots share one launch, exactly as N solo launches.
pub fn qsaSparseAttnFloor(
    s: mlx.mlx_stream,
    q_rope: mlx.mlx_array,
    views: []const DenseKVView,
    blocks: []const mlx.mlx_array,
    ratio: c_int,
    attn_scale: f32,
    min_s: c_int,
) !?mlx.mlx_array {
    if (!qsaAttnKernelEnabled()) return null;
    if (!mlx.streamIsGpu(s)) return null;
    const n = views.len;
    if (n == 0 or n > 4 or blocks.len != n) return null;
    const kv_view = &views[0];
    if (mlx.mlx_array_ndim(q_rope) != 4 or mlx.mlx_array_ndim(blocks[0]) != 3) return null;
    const qs = mlx.getShape(q_rope);
    const seq_len = qs[2];
    // `min_s` keeps a lone decode row on `qsaDecodeGatherAttn` (a batched step lowers it);
    // prefill keeps `gatherQsa256`; a dense cache follows `MLX_SERVE_QSA_ATTN_DENSE`.
    if (!qsaSparseAttnServes(kv_view.has_quant_triple, seq_len, min_s)) return null;
    if (qs[0] != n or qs[3] != 256 or (n > 1 and seq_len != 1)) return null;
    if (mlx.mlx_array_dtype(q_rope) != .bfloat16) return null;
    if (mlx.mlx_array_dtype(blocks[0]) != .int32) return null;
    const bs = mlx.getShape(blocks[0]);
    if (bs[0] != 1 or bs[1] != seq_len or bs[2] <= 0) return null;
    // `kb` rides `blocks_shape[2]` (the slot kernel: `meta[2]`) into the kernel.
    _ = bs[2];
    if (ratio <= 0) return null;
    // MLX binds an input under 8 elements in the `constant` address space, and the kernel's
    // `const device int* blk` then fails to compile (a throw at eval, not a decline).
    if (bs[1] * bs[2] < 8) return null;
    if (mlx.mlx_array_strides(q_rope)[3] != 1) return null;
    if (mlx.mlx_array_strides(blocks[0])[2] != 1) return null;

    const kvin = (if (kv_view.has_quant_triple) qsaAttnQuantInputs(kv_view) else qsaAttnDenseInputs(kv_view)) orelse return null;
    const dense = kvin.bits == 16;
    const h_kv = kvin.h_kv;
    const kv = kvin.kv;
    if (h_kv <= 0 or kv < seq_len) return null;
    if (@rem(qs[1], h_kv) != 0) return null;
    // Slots: one geometry and dtype, `device`-bound triples. The kernel cannot bind each slot's strides (24
    // more buffers, past Metal's 31) and a lazy view reports contiguous strides until evaluated, so `meta`
    // carries each slot's parent capacity; a slot's six buffers are row-contiguous and share it on the kv axis.
    var meta: [11]i32 = .{ qs[1], h_kv, bs[2] } ++ @as([8]i32, @splat(0));
    var inputs_arr: [28]mlx.mlx_array = undefined;
    if (n > 1) for (views, blocks, 0..) |*v, blk, i| {
        const kin = qsaAttnQuantInputs(v) orelse return null;
        if (blk.ctx == null or mlx.mlx_array_dtype(blk) != .int32 or !std.mem.eql(c_int, mlx.getShape(blk), bs) or mlx.mlx_array_strides(blk)[2] != 1) return null;
        if (kin.bits != kvin.bits or kin.gs != kvin.gs or kin.h_kv != h_kv or kin.kv < 1 or v.kv_cap < kin.kv) return null;
        for (kin.arrays, kvin.arrays) |x, x0| if (mlx.mlx_array_size(x) < 8 or mlx.mlx_array_dtype(x) != mlx.mlx_array_dtype(x0)) return null;
        meta[3 + 2 * i ..][0..2].* = .{ kin.kv, v.kv_cap };
        inputs_arr[3 + 6 * i ..][0..6].* = kin.arrays;
    };
    const gqa = @divExact(qs[1], h_kv);
    if (gqa <= 0 or gqa > 64) return null;

    const kernel = (if (n > 1) getQsaAttnSlotsKernel() else if (dense) getQsaAttnDKernel() else getQsaAttnQKernel()) catch return null;
    const merge_kernel = getQsaAttnMergeKernel() catch return null;
    const nsg: c_int = @divTrunc(gqa + 7, 8);
    const nsplit = qsaAttnNSplit();
    const bk = qsaAttnBk();
    const balanced = qsaAttnBalanced();
    const scl_data = [_]f32{attn_scale};
    const one = [_]c_int{1};
    const scl = mlx.mlx_array_new_data(&scl_data, &one, 1, .float32);
    defer _ = mlx.mlx_array_free(scl);

    const key = QsaAttnCfgKey{
        .q_shape = ShapeKey.from(qs),
        .h_kv = h_kv,
        .bits = kvin.bits,
        .gs = kvin.gs,
        .nsg = nsg,
        .nsplit = nsplit,
        .bk = bk,
        .balanced = balanced,
        .ratio = ratio,
        .dtype = .bfloat16,
    };
    const cfgs: [2]mlx.mlx_fast_metal_kernel_config = qsa_attn_cfgs.get(key) orelse blk: {
        // Split pass: one threadgroup per (row, kv head, split).
        const config = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
        const acc_shape = [_]c_int{ qs[0] * qs[1], seq_len, nsplit, 256 };
        const ml_shape = [_]c_int{ qs[0] * qs[1], seq_len, nsplit, 2 };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &acc_shape, 4, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &ml_shape, 4, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, seq_len * 32, h_kv * nsg, qs[0] * nsplit));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, nsg, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "NSG", nsg));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BK", bk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "RATIO", ratio));
        if (!dense) {
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BITS", @intCast(kvin.bits)));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GS", @intCast(kvin.gs)));
        }
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "NSPLIT", nsplit));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BALANCED", if (balanced) 1 else 0));

        // Merge pass: one threadgroup per (q head, row), 256 threads.
        const mconfig = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(mconfig);
        const o_shape = [_]c_int{ qs[0], qs[1], seq_len, 256 };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(mconfig, &o_shape, 4, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(mconfig, 256, seq_len, qs[0] * qs[1]));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(mconfig, 256, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(mconfig, "T", .bfloat16));

        qsa_attn_cfgs.put(key, .{ config, mconfig });
        break :blk .{ config, mconfig };
    };

    inputs_arr[0] = q_rope;
    inputs_arr[1] = scl;
    inputs_arr[2] = blocks[0];
    @memcpy(inputs_arr[3..][0..kvin.n], kvin.arrays[0..kvin.n]);
    var qc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(qc);
    var bst = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bst);
    var meta_arr: mlx.mlx_array = .{ .ctx = null };
    defer if (meta_arr.ctx != null) {
        _ = mlx.mlx_array_free(meta_arr);
    };
    if (n > 1) {
        // The kernel computes the q and block offsets itself: q [N, Hq, 1, 256] and blocks [N, 1, KB], contiguous.
        try mlx.check(mlx.mlx_contiguous(&qc, q_rope, false, s));
        const bvec = mlx.mlx_vector_array_new_data(blocks.ptr, n);
        defer _ = mlx.mlx_vector_array_free(bvec);
        try mlx.check(mlx.mlx_concatenate_axis(&bst, bvec, 0, s));
        meta_arr = mlx.mlx_array_new_data(&meta, &[_]c_int{meta.len}, 1, .int32);
        inputs_arr[0] = qc;
        inputs_arr[2] = bst;
        inputs_arr[27] = meta_arr;
        for (n..4) |i| inputs_arr[3 + 6 * i ..][0..6].* = kvin.arrays;
    }
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, if (n > 1) 28 else 3 + kvin.n);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var parts_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&parts_vec, kernel, inputs_vec, cfgs[0], s));
    if (mlx.mlx_vector_array_size(parts_vec) != 2) return error.MetalKernelBadOutputCount;
    var pacc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pacc);
    var pml = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pml);
    try mlx.check(mlx.mlx_vector_array_get(&pacc, parts_vec, 0));
    try mlx.check(mlx.mlx_vector_array_get(&pml, parts_vec, 1));

    const merge_in = [_]mlx.mlx_array{ pacc, pml };
    const merge_vec = mlx.mlx_vector_array_new_data(&merge_in, merge_in.len);
    defer _ = mlx.mlx_vector_array_free(merge_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, merge_kernel, merge_vec, cfgs[1], s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    // One bit per arm: width bucket (+4 dense) for one view, bit 8 for the slot launch.
    if (n > 1) {
        if (qsa_attn_engaged_bits.take(8)) {
            var kv_lo: i32 = meta[3];
            var kv_hi: i32 = meta[3];
            for (1..n) |i| {
                kv_lo = @min(kv_lo, meta[3 + 2 * i]);
                kv_hi = @max(kv_hi, meta[3 + 2 * i]);
            }
            log.info(
                "[qsa-attn] engaged (slots={d} S={d} kv={d}..{d} kv-bits={d}/gs{d} gqa={d} bk={d} nsplit={d} tgs={d}) — MLX_SERVE_QSA_ATTN_KERNEL=0 hands every slot back to the gather or mask arm below it\n",
                .{ n, seq_len, kv_lo, kv_hi, kvin.bits, kvin.gs, gqa, bk, nsplit, seq_len * h_kv * nsplit * @as(c_int, @intCast(n)) },
            );
        }
    } else if (qsa_attn_engaged_bits.take(qsaWidthBucket(seq_len) + @as(u5, if (dense) 4 else 0))) {
        log.info(
            "[qsa-attn] engaged (S={d} kv={d} kv-bits={d}/gs{d} gqa={d} bk={d} nsplit={d} tgs={d}) — MLX_SERVE_QSA_ATTN_KERNEL=0 (or MLX_SERVE_QSA_ATTN_DENSE=0 on a dense cache) hands the call back to the gather or mask arm below it\n",
            .{ seq_len, kv, kvin.bits, kvin.gs, gqa, bk, nsplit, seq_len * h_kv * nsplit },
        );
    }
    return out;
}

/// What served one single-slot QSA call: the output and its cost-trace bit
/// (1 fused, 2 decode gather, 4 union gather, 8 prefill gather, 16 mask).
pub const QsaSlotServed = struct { out: mlx.mlx_array, cost_bit: u8 };

/// One single-slot qwen4_exp QSA attention call: the arms in the order the forward tries
/// them. The indexer set `blocks` (per-row selection) or `mask.*` (dense `[S, kv]`). A
/// declined gather builds the mask here and writes it back so later layers reuse it.
pub fn qsaSlotAttn(
    s: mlx.mlx_stream,
    q_rope: mlx.mlx_array,
    kv_view: *const DenseKVView,
    blocks: mlx.mlx_array,
    mask: *mlx.mlx_array,
    arms: *QsaArmCounts,
    seq_len: c_int,
    ratio: c_int,
    attn_scale: f32,
) !QsaSlotServed {
    const standin_sdpa = qwen4Standin().attn_sdpa;
    const qsa_ok = blocks.ctx != null and mask.ctx == null and !standin_sdpa;
    const verify_w = seq_len >= 2 and seq_len < FUSED256_MIN_Q_LEN;
    // Three arms, three independent widths: `qsaAttnMinS` gates the fused kernel only
    // (gating the whole block on it dropped S=2..5 onto the dense mask).
    if (qsa_ok and seq_len >= qsaAttnMinS() and seq_len < FUSED256_MIN_Q_LEN) {
        // One fused dispatch per layer: each row indexes its own selection + tail.
        if (try qsaSparseAttn(s, q_rope, kv_view, blocks, ratio, attn_scale)) |g| return qsaSlotGathered(arms, seq_len, g, 1);
    }
    if (qsa_ok and seq_len == 1 and !kv_view.has_quant_triple) {
        const dec = (try qsa_decode.attend(s, q_rope, kv_view.k, kv_view.v, blocks, ratio, attn_scale)) orelse
            try @import("../qsa_decode_cuda.zig").attend(s, q_rope, kv_view.k, kv_view.v, blocks, ratio, attn_scale);
        if (dec) |g| {
            if (arms_mod.qsa_dec_engaged_bits.take(0)) log.info("[qsa-dec] engaged (kv={d}) (MLX_SERVE_QSA_DEC_KERNEL=0 restores the gather / mask arms)\n", .{mlx.getShape(kv_view.k)[2]});
            return qsaSlotGathered(arms, seq_len, g, 2);
        }
    }
    if (qsa_ok and seq_len == 1) {
        // Decode width: subset triples (or dense rows) → subset
        // dequant -> dense SDPA over K'<<kv. Declines to the mask arm.
        if (try qsaDecodeGatherAttn(s, q_rope, kv_view, blocks, ratio, attn_scale)) |g| return qsaSlotGathered(arms, seq_len, g, 2);
    }
    if (qsa_ok and verify_w and !kv_view.has_quant_triple) {
        // CUDA: each verify row attends its own picks and tail; the mask arm read the whole cache.
        if (try @import("../qsa_decode_cuda.zig").attend(s, q_rope, kv_view.k, kv_view.v, blocks, ratio, attn_scale)) |g| {
            if (arms_mod.qsa_dec_engaged_bits.take(1)) log.info("[qsa-dec] verify rows engaged (S={d} kv={d}) (MLX_SERVE_QSA_DEC_KERNEL=0 restores the gather / mask arms)\n", .{ seq_len, mlx.getShape(kv_view.k)[2] });
            return qsaSlotGathered(arms, seq_len, g, 4);
        }
    }
    if (qsa_ok and verify_w) {
        // Verify widths 2..15: the union of the block's rows' selections + tail. The
        // fused kernel's fallback for any width it declines.
        if (try qsaVerifyGatherAttn(s, q_rope, kv_view, blocks, ratio, attn_scale)) |g| return qsaSlotGathered(arms, seq_len, g, 4);
    }
    // The prefill kernel has no q_len floor of its own; a verify-width
    // selection must never fall into it (it walks the WHOLE cache).
    if (seq_len >= FUSED256_MIN_Q_LEN and blocks.ctx != null and !standin_sdpa) {
        if (try qsaPrefillGatherProfiled(s, q_rope, kv_view.k, kv_view.v, attn_scale, blocks, ratio)) |g| return qsaSlotGathered(arms, seq_len, g, 8);
    }
    // The dense [S, kv] mask arm reads `k`/`v` — on a quantized cache that is what
    // EVALUATES the whole-range dequant graph `updateAffine` built, per layer per forward.
    arms.note(.mask);
    if (mask.ctx == null) mask.* = try qsaMaskFromBlocks(s, blocks, mlx.getShape(kv_view.k)[2], ratio);
    if (standin_sdpa) return .{ .out = try standinRef(q_rope), .cost_bit = 16 };
    return .{ .out = try qsaMaskArm(s, q_rope, kv_view.k, kv_view.v, attn_scale, mask.*), .cost_bit = 16 };
}

fn qsaSlotGathered(arms: *QsaArmCounts, seq_len: c_int, out: mlx.mlx_array, cost_bit: u8) QsaSlotServed {
    arms.noteGather(seq_len);
    return .{ .out = out, .cost_bit = cost_bit };
}

/// `gatherQsa256` with the `QWEN4_PROFILE_QSA` timer around it.
fn qsaPrefillGatherProfiled(s: mlx.mlx_stream, q_rope: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, attn_scale: f32, blocks: mlx.mlx_array, ratio: c_int) !?mlx.mlx_array {
    const prof = diagEnvOn("QWEN4_PROFILE_QSA");
    var clk: ProfClock = undefined;
    if (prof) {
        try mlx.check(mlx.mlx_array_eval(q_rope));
        clk = ProfClock.init();
    }
    const gathered = try gatherQsa256(s, q_rope, k, v, attn_scale, blocks, ratio);
    if (prof) if (gathered) |g| {
        try mlx.check(mlx.mlx_array_eval(g));
        log.info("[qsa-prof] gather S={d} kv={d}: {d:.2} ms\n", .{ mlx.getShape(q_rope)[2], mlx.getShape(k)[2], @as(f64, @floatFromInt(clk.lap())) / 1e6 });
    };
    return gathered;
}

/// The dense-mask arm: the fused masked kernel, the split vector sdpa, then stock sdpa.
pub fn qsaMaskArm(s: mlx.mlx_stream, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32, mask: mlx.mlx_array) !mlx.mlx_array {
    if (try fusedSdpa256Masked(s, q, k, v, scale, mask)) |o| return o;
    if (try splitMaskedSdpa256(s, q, k, v, scale, mask)) |o| return o;
    // MLX's CUDA SDPA turns a bool mask into 0 / -inf after broadcasting it over the heads, three
    // [B, H, S, kv] arrays; the same values over one head reach cuDNN as a broadcast view.
    var additive = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(additive);
    if (mlx.cudaAvailable() and mlx.mlx_array_dtype(mask) == .bool_) {
        const zero = try scalarOf(0, mlx.mlx_array_dtype(q), s);
        defer _ = mlx.mlx_array_free(zero);
        const neg_inf = try scalarOf(-std.math.inf(f32), mlx.mlx_array_dtype(q), s);
        defer _ = mlx.mlx_array_free(neg_inf);
        try mlx.check(mlx.mlx_where(&additive, mask, zero, neg_inf, s));
    } else try mlx.check(mlx.mlx_array_set(&additive, mask));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&out, q, k, v, scale, "array", additive, .{ .ctx = null }, false, s));
    return out;
}

// ── tests ──

pub fn qsaAttnCfgBuilds() usize {
    return qsa_attn_cfgs.builds;
}

// ── Fused sparse attention at verify widths and batched decode (msv_attn_qsa256_q) ──

/// Build a KVCache whose view is a strided slice of a larger capacity buffer (the shape the serving path hands the kernel).
pub fn qsaAttnCacheFixture(
    alloc: std.mem.Allocator,
    s: mlx.mlx_stream,
    k_dense: mlx.mlx_array,
    v_dense: mlx.mlx_array,
    kv: c_int,
    split: c_int,
    cfg: kv_quant.KVQuantConfig,
    cache: *KVCache,
) !DenseKVView {
    _ = alloc;
    const st = [_]c_int{ 1, 1, 1, 1 };
    var k0 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k0);
    var v0 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v0);
    const ks = mlx.getShape(k_dense);
    try mlx.check(mlx.mlx_slice(&k0, k_dense, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ ks[0], ks[1], split, ks[3] }, 4, &st, 4, s));
    try mlx.check(mlx.mlx_slice(&v0, v_dense, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ ks[0], ks[1], split, ks[3] }, 4, &st, 4, s));
    var first = try cache.update(0, k0, v0, s, 0);
    first.deinit();
    var k1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k1);
    var v1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v1);
    try mlx.check(mlx.mlx_slice(&k1, k_dense, &[_]c_int{ 0, 0, split, 0 }, 4, &[_]c_int{ ks[0], ks[1], kv, ks[3] }, 4, &st, 4, s));
    try mlx.check(mlx.mlx_slice(&v1, v_dense, &[_]c_int{ 0, 0, split, 0 }, 4, &[_]c_int{ ks[0], ks[1], kv, ks[3] }, 4, &st, 4, s));
    _ = cfg;
    return try cache.update(0, k1, v1, s, 0);
}

test "qsa sparse attn: one fused dispatch equals the union gather at verify widths (affine-8 strided cache; dense served)" {
    // The bar is two bf16 ulps at the reference's own scale (an ulp is relative, so a constant
    // bar only works at one output scale), plus cosine against a wrong key set.
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ta = std.testing.allocator;
    const ratio: c_int = 4;
    var prng = std.Random.DefaultPrng.init(0x5A17_C0DE);
    const rnd = prng.random();

    qsa_attn_min_s_override = 1;
    defer qsa_attn_min_s_override = null;

    const cases = [_]struct { s: c_int, kv: c_int, kb: c_int, hq: c_int, hkv: c_int }{
        // Decode width: NSPLIT hits the ceiling, most splits are empty.
        .{ .s = 1, .kv = 4096, .kb = 64, .hq = 24, .hkv = 2 },
        .{ .s = 1, .kv = 70000, .kb = 512, .hq = 24, .hkv = 2 },
        .{ .s = 3, .kv = 62000, .kb = 512, .hq = 24, .hkv = 2 },
        // Tile count far below NSPLIT.
        .{ .s = 2, .kv = 1024, .kb = 512, .hq = 24, .hkv = 2 },
        .{ .s = 15, .kv = 70000, .kb = 512, .hq = 24, .hkv = 2 },
        .{ .s = 2, .kv = 4096, .kb = 64, .hq = 24, .hkv = 2 },
        .{ .s = 6, .kv = 4096, .kb = 64, .hq = 24, .hkv = 2 },
        .{ .s = 15, .kv = 4096, .kb = 64, .hq = 24, .hkv = 2 },
        .{ .s = 6, .kv = 20000, .kb = 128, .hq = 24, .hkv = 2 },
        .{ .s = 15, .kv = 20000, .kb = 128, .hq = 12, .hkv = 1 },
        .{ .s = 2, .kv = 70000, .kb = 512, .hq = 24, .hkv = 2 },
        .{ .s = 6, .kv = 70000, .kb = 512, .hq = 24, .hkv = 2 },
        // Fewer visible blocks than the budget: the INT_MAX pads must never be read.
        .{ .s = 6, .kv = 4096, .kb = 2048, .hq = 24, .hkv = 2 },
    };

    for (cases) |c| {
        const q_shape = [_]c_int{ 1, c.hq, c.s, 256 };
        const kv_shape = [_]c_int{ 1, c.hkv, c.kv, 256 };
        const q = try attn256RandBf16(rnd, &q_shape, s);
        defer _ = mlx.mlx_array_free(q);
        const k_dense = try testRandWeightBf16(rnd, &kv_shape, s);
        defer _ = mlx.mlx_array_free(k_dense);
        const v_dense = try testRandWeightBf16(rnd, &kv_shape, s);
        defer _ = mlx.mlx_array_free(v_dense);

        const blocks_host = try qsaVerifyBlocksHost(ta, rnd, c.s, c.kv, c.kb, ratio);
        defer ta.free(blocks_host);
        const bshape = [_]c_int{ 1, c.s, c.kb };
        const blocks = mlx.mlx_array_new_data(blocks_host.ptr, &bshape, 3, .int32);
        defer _ = mlx.mlx_array_free(blocks);
        const mask = try qsaMaskFromBlocks(s, blocks, c.kv, ratio);
        defer _ = mlx.mlx_array_free(mask);
        const scale: f32 = 1.0 / 16.0;

        // Dense arm: the bf16 variant of the same split-K kernel serves a dense cache.
        var dview = DenseKVView{ .k = k_dense, .v = v_dense, .owned = false };
        const ref_d = try attn256Reference(q, k_dense, v_dense, scale, "array", mask, s);
        defer _ = mlx.mlx_array_free(ref_d);
        {
            const got_f = (try qsaSparseAttn(s, q, &dview, blocks, ratio, scale)) orelse return error.SparseAttnDeclined;
            defer _ = mlx.mlx_array_free(got_f);
            const fd = try attn256MaxDiff(got_f, ref_d, s);
            const fc = try attn256Cosine(got_f, ref_d, s);
            const f_bar = try attn256UlpBar(ref_d, s);
            std.debug.print("[qsa-attn] S={d} kv={d} kb={d} gqa={d} dense fused: max diff {d:.6} cos {d:.7} (bar {d:.6})\n", .{ c.s, c.kv, c.kb, @divExact(c.hq, c.hkv), fd, fc, f_bar });
            try std.testing.expect(fc > 0.99999);
            try std.testing.expect(fd <= f_bar);
        }
        if (try qsaVerifyGatherAttn(s, q, &dview, blocks, ratio, scale)) |got_d| {
            defer _ = mlx.mlx_array_free(got_d);
            const dd = try attn256MaxDiff(got_d, ref_d, s);
            const dc = try attn256Cosine(got_d, ref_d, s);
            const d_bar = try attn256UlpBar(ref_d, s);
            std.debug.print("[qsa-attn] S={d} kv={d} kb={d} gqa={d} dense->union gather: max diff {d:.6} cos {d:.7} (bar {d:.6})\n", .{ c.s, c.kv, c.kb, @divExact(c.hq, c.hkv), dd, dc, d_bar });
            try std.testing.expect(dc > 0.99999);
            try std.testing.expect(dd <= d_bar);
        }

        // Packed arm: a real affine-8 cache, filled in two appends so the view is strided.
        var cache = try KVCache.initWithConfig(ta, 1, kv_quant.KVQuantConfig.affine(8));
        defer cache.deinit();
        var view = try qsaAttnCacheFixture(ta, s, k_dense, v_dense, c.kv, @divTrunc(c.kv, 2), kv_quant.KVQuantConfig.affine(8), &cache);
        defer view.deinit();
        try std.testing.expect(view.has_quant_triple);
        try std.testing.expectEqual(c.kv, mlx.getShape(view.k)[2]);

        const got_q = (try qsaSparseAttn(s, q, &view, blocks, ratio, scale)) orelse return error.SparseAttnDeclined;
        defer _ = mlx.mlx_array_free(got_q);
        // Reference reads the cache's own dequantized rows.
        const ref_q = try attn256Reference(q, view.k, view.v, scale, "array", mask, s);
        defer _ = mlx.mlx_array_free(ref_q);
        const qd = try attn256MaxDiff(got_q, ref_q, s);
        const qc = try attn256Cosine(got_q, ref_q, s);
        const q_bar = try attn256UlpBar(ref_q, s);
        std.debug.print("[qsa-attn] S={d} kv={d} kb={d} gqa={d} affine-8: max diff {d:.6} cos {d:.7} (bar {d:.6})\n", .{ c.s, c.kv, c.kb, @divExact(c.hq, c.hkv), qd, qc, q_bar });
        try std.testing.expect(qc > 0.99999);
        try std.testing.expect(qd <= q_bar);

        // The two shipping arms must not disagree with each other.
        if (qsaVerifyGatherAttn(s, q, &view, blocks, ratio, scale) catch null) |gather| {
            defer _ = mlx.mlx_array_free(gather);
            const ac = try attn256Cosine(got_q, gather, s);
            try std.testing.expect(ac > 0.99999);
        }
    }
}

test "qsa sparse attn: the split-K merge is invariant in the split count (1, 8, 64)" {
    // Splitting a row's key range and recombining the per-split online-softmax states must be
    // exact: a wrong merge would move the answer with NSPLIT. NSPLIT = 1 is the control.
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ta = std.testing.allocator;
    const ratio: c_int = 4;
    var prng = std.Random.DefaultPrng.init(0x5B17_C0DE);
    const rnd = prng.random();

    const seq: c_int = 6;
    const kv: c_int = 20000;
    const kb: c_int = 128;
    const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 24, seq, 256 }, s);
    defer _ = mlx.mlx_array_free(q);
    const k_dense = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, kv, 256 }, s);
    defer _ = mlx.mlx_array_free(k_dense);
    const v_dense = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, kv, 256 }, s);
    defer _ = mlx.mlx_array_free(v_dense);
    const blocks_host = try qsaVerifyBlocksHost(ta, rnd, seq, kv, kb, ratio);
    defer ta.free(blocks_host);
    const blocks = mlx.mlx_array_new_data(blocks_host.ptr, &[_]c_int{ 1, seq, kb }, 3, .int32);
    defer _ = mlx.mlx_array_free(blocks);
    const mask = try qsaMaskFromBlocks(s, blocks, kv, ratio);
    defer _ = mlx.mlx_array_free(mask);
    const scale: f32 = 1.0 / 16.0;

    var cache = try KVCache.initWithConfig(ta, 1, kv_quant.KVQuantConfig.affine(8));
    defer cache.deinit();
    var view = try qsaAttnCacheFixture(ta, s, k_dense, v_dense, kv, @divTrunc(kv, 2), kv_quant.KVQuantConfig.affine(8), &cache);
    defer view.deinit();
    const ref = try attn256Reference(q, view.k, view.v, scale, "array", mask, s);
    defer _ = mlx.mlx_array_free(ref);
    const bar = try attn256UlpBar(ref, s);

    defer qsa_attn_nsplit_override = null;
    var base: ?mlx.mlx_array = null;
    defer if (base) |b| {
        _ = mlx.mlx_array_free(b);
    };
    for ([_]c_int{ 1, 8, 64 }) |ns| {
        qsa_attn_nsplit_override = ns;
        try std.testing.expectEqual(ns, qsaAttnNSplit());
        const got = (try qsaSparseAttn(s, q, &view, blocks, ratio, scale)) orelse return error.SparseAttnDeclined;
        defer _ = mlx.mlx_array_free(got);
        const d = try attn256MaxDiff(got, ref, s);
        const c = try attn256Cosine(got, ref, s);
        std.debug.print("[qsa-attn] nsplit={d}: max diff {d:.6} cos {d:.7} (bar {d:.6})\n", .{ ns, d, c, bar });
        try std.testing.expect(c > 0.99999);
        try std.testing.expect(d <= bar);
        if (base) |b| {
            const cc = try attn256Cosine(got, b, s);
            try std.testing.expect(cc > 0.99999);
        } else {
            var keep = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&keep, got));
            base = keep;
        }
    }
}

test "qsa sparse attn: NSPLIT is one MEASURED constant, and the width floor binds the DISPATCH" {
    defer qsa_attn_nsplit_override = null;
    qsa_attn_nsplit_override = null;
    try std.testing.expectEqual(QSA_ATTN_NSPLIT_DEFAULT, qsaAttnNSplit());
    try std.testing.expectEqual(@as(c_int, 16), QSA_ATTN_NSPLIT_DEFAULT);
    qsa_attn_nsplit_override = 8;
    try std.testing.expectEqual(@as(c_int, 8), qsaAttnNSplit());
    qsa_attn_nsplit_override = 5;
    try std.testing.expectEqual(@as(c_int, 8), qsaAttnNSplit());
    qsa_attn_nsplit_override = 1000;
    try std.testing.expectEqual(QSA_ATTN_MAX_NSPLIT, qsaAttnNSplit());
    qsa_attn_nsplit_override = null;

    defer qsa_attn_bk_override = null;
    defer qsa_attn_balanced_override = null;
    try std.testing.expectEqual(QSA_ATTN_BK, qsaAttnBk());
    try std.testing.expectEqual(false, qsaAttnBalanced());
    // BK must be a multiple of 8; 12 is rejected back to the default.
    qsa_attn_bk_override = 12;
    try std.testing.expectEqual(QSA_ATTN_BK, qsaAttnBk());
    qsa_attn_bk_override = 0;
    try std.testing.expectEqual(QSA_ATTN_BK, qsaAttnBk());
    qsa_attn_bk_override = 64;
    try std.testing.expectEqual(QSA_ATTN_BK, qsaAttnBk());
    qsa_attn_bk_override = 8;
    try std.testing.expectEqual(@as(c_int, 8), qsaAttnBk());
    try std.testing.expect(256 * (8 + 8) * 2 <= 10 * 1024);
    qsa_attn_bk_override = null;
    qsa_attn_balanced_override = true;
    try std.testing.expectEqual(true, qsaAttnBalanced());
    qsa_attn_balanced_override = null;

    defer qsa_attn_min_s_override = null;
    qsa_attn_min_s_override = null;
    try std.testing.expectEqual(@as(c_int, 2), QSA_ATTN_MIN_S_DEFAULT);
    try std.testing.expectEqual(@as(c_int, 2), qsaAttnMinS());
    qsa_attn_min_s_override = 1;
    try std.testing.expectEqual(@as(c_int, 1), qsaAttnMinS());
}

test "qsa dispatch: each arm's width floor is its own (MIN_S never moves the union gather)" {
    // `MLX_SERVE_QSA_ATTN_MIN_S` narrows the fused kernel and must not decide what serves the
    // widths it gives up. Modelled as the dispatch's own predicates.
    const Arm = enum { fused, decode_gather, union_gather, dense_mask };
    const pick = struct {
        fn f(quant: bool, seq_len: c_int, min_s: c_int) Arm {
            if (qsaSparseAttnServes(quant, seq_len, min_s)) return .fused;
            if (seq_len == 1) return .decode_gather;
            if (seq_len >= 2 and seq_len < FUSED256_MIN_Q_LEN) return .union_gather;
            return .dense_mask;
        }
    }.f;

    try std.testing.expectEqual(Arm.decode_gather, pick(true, 1, QSA_ATTN_MIN_S_DEFAULT));
    for ([_]c_int{ 2, 3, 6, 15 }) |sq| try std.testing.expectEqual(Arm.fused, pick(true, sq, QSA_ATTN_MIN_S_DEFAULT));
    try std.testing.expectEqual(Arm.dense_mask, pick(true, FUSED256_MIN_Q_LEN, QSA_ATTN_MIN_S_DEFAULT));

    // MIN_S=6: S=2..5 must land on the union gather, never on the dense mask.
    try std.testing.expectEqual(Arm.decode_gather, pick(true, 1, 6));
    for ([_]c_int{ 2, 3, 4, 5 }) |sq| try std.testing.expectEqual(Arm.union_gather, pick(true, sq, 6));
    for ([_]c_int{ 6, 15 }) |sq| try std.testing.expectEqual(Arm.fused, pick(true, sq, 6));

    try std.testing.expectEqual(Arm.fused, pick(true, 1, 1));

    const off = FUSED256_MIN_Q_LEN;
    try std.testing.expectEqual(Arm.decode_gather, pick(true, 1, off));
    for ([_]c_int{ 2, 3, 6, 15 }) |sq| try std.testing.expectEqual(Arm.union_gather, pick(true, sq, off));
}

test "qsa dispatch: a DENSE KV cache at a verify width takes the split-K verify kernel, never the prefill kernel" {
    // `--kv-quant off` at 16k decoded 77.7 tok/s against 93.6 at the same acceptance: on a
    // dense cache `qsaSparseAttn`'s arm was `gatherQsa256`, the prefill kernel (8 threadgroups
    // at S=4). The dense arm now keeps the verify kernel's split-K grid, and
    // `MLX_SERVE_QSA_ATTN_DENSE=0` hands dense KV back to the mask / union gather arms.
    // Hermetic: the arm choice is a pure predicate.
    defer qsa_attn_dense_override = null;
    qsa_attn_dense_override = null;
    const min_s = QSA_ATTN_MIN_S_DEFAULT;
    for ([_]c_int{ 2, 3, 4, 5, 6, 7, 8, 15 }) |sq| {
        try std.testing.expect(qsaSparseAttnServes(true, sq, min_s));
        try std.testing.expect(qsaSparseAttnServes(false, sq, min_s));
    }
    try std.testing.expect(!qsaSparseAttnServes(true, 1, min_s));
    try std.testing.expect(!qsaSparseAttnServes(true, FUSED256_MIN_Q_LEN, min_s));
    try std.testing.expect(!qsaSparseAttnServes(false, 1, min_s));
    try std.testing.expect(!qsaSparseAttnServes(false, FUSED256_MIN_Q_LEN, min_s));
    try std.testing.expect(qsaSparseAttnServes(true, 1, 1));
    try std.testing.expect(qsaSparseAttnServes(false, 1, 1));

    // The dense kill switch moves dense KV only.
    qsa_attn_dense_override = false;
    for ([_]c_int{ 2, 6, 15 }) |sq| {
        try std.testing.expect(qsaSparseAttnServes(true, sq, min_s));
        try std.testing.expect(!qsaSparseAttnServes(false, sq, min_s));
    }
}

test "qsa sparse attn: gqa tile reuse is invariant — one kv head serving 12 q heads equals 12 serving one each" {
    // The kernel dequantizes each K/V element once per threadgroup and feeds it to all `gqa`
    // rows: gqa 12 (tile reuse) vs gqa 1 (12 identical kv heads) must agree exactly.
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ta = std.testing.allocator;
    const ratio: c_int = 4;
    const kv: c_int = 4096;
    const seq: c_int = 6;
    const kb: c_int = 64;
    var prng = std.Random.DefaultPrng.init(0x7EA_1EAF);
    const rnd = prng.random();

    const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 12, seq, 256 }, s);
    defer _ = mlx.mlx_array_free(q);
    const k1 = try attn256RandBf16(rnd, &[_]c_int{ 1, 1, kv, 256 }, s);
    defer _ = mlx.mlx_array_free(k1);
    const v1 = try attn256RandBf16(rnd, &[_]c_int{ 1, 1, kv, 256 }, s);
    defer _ = mlx.mlx_array_free(v1);
    var k12 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k12);
    var v12 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v12);
    try mlx.check(mlx.mlx_broadcast_to(&k12, k1, &[_]c_int{ 1, 12, kv, 256 }, 4, s));
    try mlx.check(mlx.mlx_broadcast_to(&v12, v1, &[_]c_int{ 1, 12, kv, 256 }, 4, s));
    var k12c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k12c);
    var v12c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v12c);
    try mlx.check(mlx.mlx_contiguous(&k12c, k12, false, s));
    try mlx.check(mlx.mlx_contiguous(&v12c, v12, false, s));

    const bh = try qsaVerifyBlocksHost(ta, rnd, seq, kv, kb, ratio);
    defer ta.free(bh);
    const blocks = mlx.mlx_array_new_data(bh.ptr, &[_]c_int{ 1, seq, kb }, 3, .int32);
    defer _ = mlx.mlx_array_free(blocks);
    const scale: f32 = 1.0 / 16.0;

    // Both fixtures must be real quantized caches (a dense view takes the bf16 variant).
    const qcfg = kv_quant.KVQuantConfig.affine(8);
    var cache_r = try KVCache.initWithConfig(ta, 1, qcfg);
    defer cache_r.deinit();
    var v_reuse = try qsaAttnCacheFixture(ta, s, k1, v1, kv, @divTrunc(kv, 2), qcfg, &cache_r); // gqa 12
    defer v_reuse.deinit();
    try std.testing.expect(v_reuse.has_quant_triple);
    var cache_n = try KVCache.initWithConfig(ta, 1, qcfg);
    defer cache_n.deinit();
    var v_noreuse = try qsaAttnCacheFixture(ta, s, k12c, v12c, kv, @divTrunc(kv, 2), qcfg, &cache_n); // gqa 1
    defer v_noreuse.deinit();
    const a = (try qsaSparseAttn(s, q, &v_reuse, blocks, ratio, scale)) orelse return error.SparseAttnDeclined;
    defer _ = mlx.mlx_array_free(a);
    const b = (try qsaSparseAttn(s, q, &v_noreuse, blocks, ratio, scale)) orelse return error.SparseAttnDeclined;
    defer _ = mlx.mlx_array_free(b);
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(a, b, s));
}

test "qsa sparse attn: one fused dispatch replaces the arm's op chain (>= 100 ops over 12 layers)" {
    // A dispatch-count bar: twelve calls stand in for qwen4's twelve full-attention layers.
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ta = std.testing.allocator;
    const ratio: c_int = 4;
    const kv: c_int = 20000;
    const seq: c_int = 6;
    const kb: c_int = 128;
    var prng = std.Random.DefaultPrng.init(0x0F5_C0DE);
    const rnd = prng.random();

    const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 24, seq, 256 }, s);
    defer _ = mlx.mlx_array_free(q);
    const kd = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, kv, 256 }, s);
    defer _ = mlx.mlx_array_free(kd);
    const vd = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, kv, 256 }, s);
    defer _ = mlx.mlx_array_free(vd);
    const bh = try qsaVerifyBlocksHost(ta, rnd, seq, kv, kb, ratio);
    defer ta.free(bh);
    const blocks = mlx.mlx_array_new_data(bh.ptr, &[_]c_int{ 1, seq, kb }, 3, .int32);
    defer _ = mlx.mlx_array_free(blocks);
    const scale: f32 = 1.0 / 16.0;

    var cache = try KVCache.initWithConfig(ta, 1, kv_quant.KVQuantConfig.affine(8));
    defer cache.deinit();
    var view = try qsaAttnCacheFixture(ta, s, kd, vd, kv, @divTrunc(kv, 2), kv_quant.KVQuantConfig.affine(8), &cache);
    defer view.deinit();

    // Warm both arms first: the first call builds its cached config.
    if (try qsaSparseAttn(s, q, &view, blocks, ratio, scale)) |w| _ = mlx.mlx_array_free(w);
    if (qsaVerifyGatherAttn(s, q, &view, blocks, ratio, scale) catch null) |w| _ = mlx.mlx_array_free(w);

    const before_k = mlx.op_count.load(.monotonic);
    for (0..12) |_| {
        const o = (try qsaSparseAttn(s, q, &view, blocks, ratio, scale)) orelse return error.SparseAttnDeclined;
        _ = mlx.mlx_array_free(o);
    }
    const kernel_ops = mlx.op_count.load(.monotonic) - before_k;

    const before_g = mlx.op_count.load(.monotonic);
    for (0..12) |_| {
        const o = (qsaVerifyGatherAttn(s, q, &view, blocks, ratio, scale) catch null) orelse return error.VerifyGatherDeclined;
        _ = mlx.mlx_array_free(o);
    }
    const gather_ops = mlx.op_count.load(.monotonic) - before_g;

    std.debug.print("[qsa-attn] 12 layers S={d} kv={d}: kernel {d} ops, union gather {d} ops (saved {d})\n", .{ seq, kv, kernel_ops, gather_ops, gather_ops -| kernel_ops });
    try std.testing.expect(gather_ops > kernel_ops);
    try std.testing.expect(gather_ops - kernel_ops >= 100);
}

test "qsa sparse attn dense: bf16 cache views at decode and verify widths equal the masked reference (S 1..15, kv 2100..40000)" {
    // The dense arm must attend exactly the mask arm's key set (each row's own blocks + its own
    // causal tail) through any view the cache hands it: the cache's capacity-strided view, a
    // hand-built slice of a wider buffer, and a head slice with a non-zero data offset.
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ta = std.testing.allocator;
    const ratio: c_int = 4;
    var prng = std.Random.DefaultPrng.init(0xBF16_C0DE);
    const rnd = prng.random();
    const scale: f32 = 1.0 / 16.0;
    defer qsa_attn_dense_override = null;
    qsa_attn_dense_override = null;

    const cases = [_]struct { s: c_int, kv: c_int, kb: c_int, hq: c_int, hkv: c_int }{
        // Decode width, as batched steps serve it (floor 1): NSPLIT at its ceiling, most splits empty.
        .{ .s = 1, .kv = 9000, .kb = 512, .hq = 24, .hkv = 2 },
        .{ .s = 1, .kv = 40000, .kb = 512, .hq = 24, .hkv = 2 },
        .{ .s = 2, .kv = 2100, .kb = 512, .hq = 24, .hkv = 2 },
        .{ .s = 4, .kv = 2100, .kb = 512, .hq = 24, .hkv = 2 },
        .{ .s = 6, .kv = 9000, .kb = 512, .hq = 24, .hkv = 2 },
        .{ .s = 15, .kv = 9000, .kb = 512, .hq = 24, .hkv = 2 },
        .{ .s = 4, .kv = 32768, .kb = 512, .hq = 24, .hkv = 2 },
        .{ .s = 15, .kv = 32768, .kb = 512, .hq = 24, .hkv = 2 },
        .{ .s = 2, .kv = 40000, .kb = 512, .hq = 24, .hkv = 2 },
        .{ .s = 6, .kv = 40000, .kb = 512, .hq = 24, .hkv = 2 },
        // Tiles far below NSPLIT: most splits are empty.
        .{ .s = 15, .kv = 1024, .kb = 512, .hq = 24, .hkv = 2 },
        // One kv head.
        .{ .s = 6, .kv = 9000, .kb = 128, .hq = 12, .hkv = 1 },
    };

    for (cases) |c| {
        const q = try attn256RandBf16(rnd, &[_]c_int{ 1, c.hq, c.s, 256 }, s);
        defer _ = mlx.mlx_array_free(q);
        // One buffer, two extra kv heads and 256 extra rows: every view below slices it.
        const wide_shape = [_]c_int{ 1, c.hkv + 2, c.kv + 256, 256 };
        const k_wide = try testRandWeightBf16(rnd, &wide_shape, s);
        defer _ = mlx.mlx_array_free(k_wide);
        const v_wide = try testRandWeightBf16(rnd, &wide_shape, s);
        defer _ = mlx.mlx_array_free(v_wide);
        const st = [_]c_int{ 1, 1, 1, 1 };
        // Head slice at offset 1 and a kv slice of the wider row axis: strided, offset view.
        var k_view = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_view);
        var v_view = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v_view);
        try mlx.check(mlx.mlx_slice(&k_view, k_wide, &[_]c_int{ 0, 1, 0, 0 }, 4, &[_]c_int{ 1, 1 + c.hkv, c.kv, 256 }, 4, &st, 4, s));
        try mlx.check(mlx.mlx_slice(&v_view, v_wide, &[_]c_int{ 0, 1, 0, 0 }, 4, &[_]c_int{ 1, 1 + c.hkv, c.kv, 256 }, 4, &st, 4, s));
        var k_c = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_c);
        var v_c = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v_c);
        try mlx.check(mlx.mlx_contiguous(&k_c, k_view, false, s));
        try mlx.check(mlx.mlx_contiguous(&v_c, v_view, false, s));
        try mlx.check(mlx.mlx_array_eval(k_c));
        try mlx.check(mlx.mlx_array_eval(v_c));

        const blocks_host = try qsaVerifyBlocksHost(ta, rnd, c.s, c.kv, c.kb, ratio);
        defer ta.free(blocks_host);
        const blocks = mlx.mlx_array_new_data(blocks_host.ptr, &[_]c_int{ 1, c.s, c.kb }, 3, .int32);
        defer _ = mlx.mlx_array_free(blocks);
        const mask = try qsaMaskFromBlocks(s, blocks, c.kv, ratio);
        defer _ = mlx.mlx_array_free(mask);

        const ref = try attn256Reference(q, k_c, v_c, scale, "array", mask, s);
        defer _ = mlx.mlx_array_free(ref);
        const bar = try attn256UlpBar(ref, s);

        // Contiguous input: the parity bar against the masked reference.
        var cview = DenseKVView{ .k = k_c, .v = v_c, .owned = false };
        const got_c = (try qsaSparseAttnFloor(s, q, (&cview)[0..1], &.{blocks}, ratio, scale, 1)) orelse return error.SparseAttnDeclined;
        defer _ = mlx.mlx_array_free(got_c);
        const dd = try attn256MaxDiff(got_c, ref, s);
        const dc = try attn256Cosine(got_c, ref, s);
        std.debug.print("[qsa-attn-dense] S={d} kv={d} kb={d} gqa={d}: max diff {d:.6} cos {d:.7} (bar {d:.6})\n", .{ c.s, c.kv, c.kb, @divExact(c.hq, c.hkv), dd, dc, bar });
        try std.testing.expect(dc > 0.99999);
        try std.testing.expect(dd <= bar);

        // Strided, offset view of the same values: bit-identical to the contiguous input.
        var sview = DenseKVView{ .k = k_view, .v = v_view, .owned = false };
        const got_s = (try qsaSparseAttnFloor(s, q, (&sview)[0..1], &.{blocks}, ratio, scale, 1)) orelse return error.SparseAttnDeclined;
        defer _ = mlx.mlx_array_free(got_s);
        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(got_s, got_c, s));

        // The real dense cache, filled in two appends.
        var cache = try KVCache.init(ta, 1);
        defer cache.deinit();
        var view = try qsaAttnCacheFixture(ta, s, k_c, v_c, c.kv, @divTrunc(c.kv, 2), kv_quant.KVQuantConfig.dense, &cache);
        defer view.deinit();
        try std.testing.expect(!view.has_quant_triple);
        try std.testing.expectEqual(c.kv, mlx.getShape(view.k)[2]);
        const got_v = (try qsaSparseAttnFloor(s, q, (&view)[0..1], &.{blocks}, ratio, scale, 1)) orelse return error.SparseAttnDeclined;
        defer _ = mlx.mlx_array_free(got_v);
        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(got_v, got_c, s));

        // Today's dense arm (what the fused kernel replaces) agrees.
        const old = try qsaMaskArm(s, q, k_c, v_c, scale, mask);
        defer _ = mlx.mlx_array_free(old);
        try std.testing.expect((try attn256Cosine(got_c, old, s)) > 0.99999);
    }
}

test "qsa sparse attn dense: S=1, S=16 and tiny block lists decline; a non-unit innermost stride is served exactly" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ta = std.testing.allocator;
    const ratio: c_int = 4;
    const kv: c_int = 4096;
    var prng = std.Random.DefaultPrng.init(0xDEC_BF16);
    const rnd = prng.random();
    defer qsa_attn_dense_override = null;
    qsa_attn_dense_override = null;
    const kd = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, kv, 256 }, s);
    defer _ = mlx.mlx_array_free(kd);
    const vd = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, kv, 256 }, s);
    defer _ = mlx.mlx_array_free(vd);
    var dense = DenseKVView{ .k = kd, .v = vd, .owned = false };

    for ([_]c_int{ 1, FUSED256_MIN_Q_LEN }) |sq| {
        const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 24, sq, 256 }, s);
        defer _ = mlx.mlx_array_free(q);
        const bh = try qsaVerifyBlocksHost(ta, rnd, sq, kv, 64, ratio);
        defer ta.free(bh);
        const bl = mlx.mlx_array_new_data(bh.ptr, &[_]c_int{ 1, sq, 64 }, 3, .int32);
        defer _ = mlx.mlx_array_free(bl);
        try std.testing.expect((try qsaSparseAttn(s, q, &dense, bl, ratio, 1.0)) == null);
    }

    const q6 = try attn256RandBf16(rnd, &[_]c_int{ 1, 24, 6, 256 }, s);
    defer _ = mlx.mlx_array_free(q6);
    const bh6 = try qsaVerifyBlocksHost(ta, rnd, 6, kv, 64, ratio);
    defer ta.free(bh6);
    const bl6 = mlx.mlx_array_new_data(bh6.ptr, &[_]c_int{ 1, 6, 64 }, 3, .int32);
    defer _ = mlx.mlx_array_free(bl6);
    // [1, 2, 256, kv] transposed to [1, 2, kv, 256]: the innermost stride is kv, not 1.
    const kt_src = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, 256, kv }, s);
    defer _ = mlx.mlx_array_free(kt_src);
    var kt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kt);
    try mlx.check(mlx.mlx_transpose_axes(&kt, kt_src, &[_]c_int{ 0, 1, 3, 2 }, 4, s));
    // A lazy view reports row-contiguous strides until it is evaluated.
    try mlx.check(mlx.mlx_array_eval(kt));
    try std.testing.expect(mlx.mlx_array_strides(kt)[3] != 1);
    // The kernel sees the real strides and takes element loads: same answer as a copy.
    var kt_c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kt_c);
    try mlx.check(mlx.mlx_contiguous(&kt_c, kt, false, s));
    var transposed = DenseKVView{ .k = kt, .v = vd, .owned = false };
    var copied = DenseKVView{ .k = kt_c, .v = vd, .owned = false };
    const got_t = (try qsaSparseAttn(s, q6, &transposed, bl6, ratio, 1.0)) orelse return error.SparseAttnDeclined;
    defer _ = mlx.mlx_array_free(got_t);
    const got_c = (try qsaSparseAttn(s, q6, &copied, bl6, ratio, 1.0)) orelse return error.SparseAttnDeclined;
    defer _ = mlx.mlx_array_free(got_c);
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(got_t, got_c, s));

    // Blocks under 8 elements bind as `constant` in MLX and would not compile: declined.
    const q2 = try attn256RandBf16(rnd, &[_]c_int{ 1, 24, 2, 256 }, s);
    defer _ = mlx.mlx_array_free(q2);
    const bh2 = try qsaVerifyBlocksHost(ta, rnd, 2, kv, 3, ratio);
    defer ta.free(bh2);
    const bl2 = mlx.mlx_array_new_data(bh2.ptr, &[_]c_int{ 1, 2, 3 }, 3, .int32);
    defer _ = mlx.mlx_array_free(bl2);
    try std.testing.expect((try qsaSparseAttn(s, q2, &dense, bl2, ratio, 1.0)) == null);
    var qcache = try KVCache.initWithConfig(ta, 1, kv_quant.KVQuantConfig.affine(8));
    defer qcache.deinit();
    var qview = try qsaAttnCacheFixture(ta, s, kd, vd, kv, @divTrunc(kv, 2), kv_quant.KVQuantConfig.affine(8), &qcache);
    defer qview.deinit();
    try std.testing.expect((try qsaSparseAttn(s, q2, &qview, bl2, ratio, 1.0)) == null);
}

test "qsa sparse attn dense: a last-axis slice with an unaligned data offset reads exactly (no uint4 fault)" {
    // `[1, H, T, 264]` sliced to `[4, 260)` (8-byte offset) and `[3, 259)` (2-byte offset): a
    // 256-wide view whose strides look aligned but whose base is not 16-byte aligned. The
    // host cannot see the offset, so the kernel checks the bound pointer and falls back to
    // element loads. Lazy and evaluated views, K and V both offset.
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ta = std.testing.allocator;
    const ratio: c_int = 4;
    const kv: c_int = 9000;
    const seq: c_int = 6;
    const kb: c_int = 512;
    var prng = std.Random.DefaultPrng.init(0x0FF_5E7);
    const rnd = prng.random();
    const scale: f32 = 1.0 / 16.0;
    defer qsa_attn_dense_override = null;
    qsa_attn_dense_override = null;

    const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 24, seq, 256 }, s);
    defer _ = mlx.mlx_array_free(q);
    const k_pad = try testRandWeightBf16(rnd, &[_]c_int{ 1, 2, kv, 264 }, s);
    defer _ = mlx.mlx_array_free(k_pad);
    const v_pad = try testRandWeightBf16(rnd, &[_]c_int{ 1, 2, kv, 264 }, s);
    defer _ = mlx.mlx_array_free(v_pad);
    try mlx.check(mlx.mlx_array_eval(k_pad));
    try mlx.check(mlx.mlx_array_eval(v_pad));
    const bh = try qsaVerifyBlocksHost(ta, rnd, seq, kv, kb, ratio);
    defer ta.free(bh);
    const blocks = mlx.mlx_array_new_data(bh.ptr, &[_]c_int{ 1, seq, kb }, 3, .int32);
    defer _ = mlx.mlx_array_free(blocks);
    const mask = try qsaMaskFromBlocks(s, blocks, kv, ratio);
    defer _ = mlx.mlx_array_free(mask);
    const st = [_]c_int{ 1, 1, 1, 1 };

    for ([_]c_int{ 4, 3 }) |lo| {
        for ([_]bool{ false, true }) |evaluated| {
            var k_off = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(k_off);
            var v_off = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(v_off);
            try mlx.check(mlx.mlx_slice(&k_off, k_pad, &[_]c_int{ 0, 0, 0, lo }, 4, &[_]c_int{ 1, 2, kv, lo + 256 }, 4, &st, 4, s));
            try mlx.check(mlx.mlx_slice(&v_off, v_pad, &[_]c_int{ 0, 0, 0, lo }, 4, &[_]c_int{ 1, 2, kv, lo + 256 }, 4, &st, 4, s));
            if (evaluated) {
                try mlx.check(mlx.mlx_array_eval(k_off));
                try mlx.check(mlx.mlx_array_eval(v_off));
                try std.testing.expectEqual(@as(usize, 264), mlx.mlx_array_strides(k_off)[2]);
            }
            var k_c = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(k_c);
            var v_c = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(v_c);
            try mlx.check(mlx.mlx_contiguous(&k_c, k_off, false, s));
            try mlx.check(mlx.mlx_contiguous(&v_c, v_off, false, s));

            var oview = DenseKVView{ .k = k_off, .v = v_off, .owned = false };
            var cview = DenseKVView{ .k = k_c, .v = v_c, .owned = false };
            const got_o = (try qsaSparseAttn(s, q, &oview, blocks, ratio, scale)) orelse return error.SparseAttnDeclined;
            defer _ = mlx.mlx_array_free(got_o);
            const got_c = (try qsaSparseAttn(s, q, &cview, blocks, ratio, scale)) orelse return error.SparseAttnDeclined;
            defer _ = mlx.mlx_array_free(got_c);
            const ref = try attn256Reference(q, k_c, v_c, scale, "array", mask, s);
            defer _ = mlx.mlx_array_free(ref);
            const d = try attn256MaxDiff(got_o, ref, s);
            const bar = try attn256UlpBar(ref, s);
            std.debug.print("[qsa-attn-dense] offset slice [{d},{d}) evaluated={}: vs copy {d:.6}, vs ref {d:.6} (bar {d:.6})\n", .{ lo, lo + 256, evaluated, try attn256MaxDiff(got_o, got_c, s), d, bar });
            try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(got_o, got_c, s));
            try std.testing.expect(d <= bar);
        }
    }
}

test "qsa sparse attn dense: one fused dispatch replaces the mask arm's op chain (12 layers, bf16 KV)" {
    // Today a dense cache at kv <= 32768 builds the [S, kv] mask once per forward and then runs
    // the split masked sdpa on every layer. Twelve calls stand in for the full-attention layers.
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ta = std.testing.allocator;
    const ratio: c_int = 4;
    const kv: c_int = 20000;
    const seq: c_int = 6;
    const kb: c_int = 512;
    var prng = std.Random.DefaultPrng.init(0x0F5_BF16);
    const rnd = prng.random();
    defer qsa_attn_dense_override = null;
    qsa_attn_dense_override = null;

    const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 24, seq, 256 }, s);
    defer _ = mlx.mlx_array_free(q);
    const kd = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, kv, 256 }, s);
    defer _ = mlx.mlx_array_free(kd);
    const vd = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, kv, 256 }, s);
    defer _ = mlx.mlx_array_free(vd);
    const bh = try qsaVerifyBlocksHost(ta, rnd, seq, kv, kb, ratio);
    defer ta.free(bh);
    const blocks = mlx.mlx_array_new_data(bh.ptr, &[_]c_int{ 1, seq, kb }, 3, .int32);
    defer _ = mlx.mlx_array_free(blocks);
    const scale: f32 = 1.0 / 16.0;
    var view = DenseKVView{ .k = kd, .v = vd, .owned = false };

    // Warm both arms first: the first call builds its cached config.
    {
        const m = try qsaMaskFromBlocks(s, blocks, kv, ratio);
        defer _ = mlx.mlx_array_free(m);
        const w = try qsaMaskArm(s, q, kd, vd, scale, m);
        _ = mlx.mlx_array_free(w);
    }
    if (try qsaSparseAttn(s, q, &view, blocks, ratio, scale)) |w| _ = mlx.mlx_array_free(w);

    const before_m = mlx.op_count.load(.monotonic);
    {
        const m = try qsaMaskFromBlocks(s, blocks, kv, ratio);
        defer _ = mlx.mlx_array_free(m);
        for (0..12) |_| {
            const o = try qsaMaskArm(s, q, kd, vd, scale, m);
            _ = mlx.mlx_array_free(o);
        }
    }
    const mask_ops = mlx.op_count.load(.monotonic) - before_m;

    const before_k = mlx.op_count.load(.monotonic);
    for (0..12) |_| {
        const o = (try qsaSparseAttn(s, q, &view, blocks, ratio, scale)) orelse return error.SparseAttnDeclined;
        _ = mlx.mlx_array_free(o);
    }
    const kernel_ops = mlx.op_count.load(.monotonic) - before_k;

    std.debug.print("[qsa-attn-dense] 12 layers S={d} kv={d}: kernel {d} ops, mask arm {d} ops (saved {d})\n", .{ seq, kv, kernel_ops, mask_ops, mask_ops -| kernel_ops });
    try std.testing.expect(mask_ops > kernel_ops);
    try std.testing.expect(mask_ops - kernel_ops >= 60);
}

test "qsa sparse attn: the quantized kernel source is pinned" {
    // A change to these bytes owes an old-vs-new bit-identity run of the split outputs (pacc, pml).
    try std.testing.expectEqual(@as(usize, 9282), QSA_ATTN_Q_SOURCE.len);
    try std.testing.expectEqual(@as(u32, 3839740191), std.hash.Crc32.hash(QSA_ATTN_Q_SOURCE));
    try std.testing.expectEqual(@as(u32, 4256800642), std.hash.Crc32.hash(QSA_ATTN_MERGE_SOURCE));
}

test "qsa sparse attn: the width gate, the kill switch and the quant preconditions all decline cleanly" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ta = std.testing.allocator;
    const ratio: c_int = 4;
    const kv: c_int = 4096;
    var prng = std.Random.DefaultPrng.init(0xDEC_11E);
    const rnd = prng.random();
    const kd = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, kv, 256 }, s);
    defer _ = mlx.mlx_array_free(kd);
    const vd = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, kv, 256 }, s);
    defer _ = mlx.mlx_array_free(vd);
    // The width gate and the kill switch are only observable on a cache the kernel would serve.
    const qcfg = kv_quant.KVQuantConfig.affine(8);
    var qcache = try KVCache.initWithConfig(ta, 1, qcfg);
    defer qcache.deinit();
    var dv = try qsaAttnCacheFixture(ta, s, kd, vd, kv, @divTrunc(kv, 2), qcfg, &qcache);
    defer dv.deinit();
    try std.testing.expect(dv.has_quant_triple);

    const H = struct {
        fn blocksFor(alloc: std.mem.Allocator, r: std.Random, seq: c_int, kvv: c_int, kb: c_int, rat: c_int) ![]i32 {
            return qsaVerifyBlocksHost(alloc, r, seq, kvv, kb, rat);
        }
    };

    {
        const q1 = try attn256RandBf16(rnd, &[_]c_int{ 1, 24, 1, 256 }, s);
        defer _ = mlx.mlx_array_free(q1);
        const b1 = try H.blocksFor(ta, rnd, 1, kv, 64, ratio);
        defer ta.free(b1);
        const bl1 = mlx.mlx_array_new_data(b1.ptr, &[_]c_int{ 1, 1, 64 }, 3, .int32);
        defer _ = mlx.mlx_array_free(bl1);
        try std.testing.expect((try qsaSparseAttn(s, q1, &dv, bl1, ratio, 1.0)) == null);
    }
    {
        const q16 = try attn256RandBf16(rnd, &[_]c_int{ 1, 24, FUSED256_MIN_Q_LEN, 256 }, s);
        defer _ = mlx.mlx_array_free(q16);
        const b16 = try H.blocksFor(ta, rnd, FUSED256_MIN_Q_LEN, kv, 64, ratio);
        defer ta.free(b16);
        const bl16 = mlx.mlx_array_new_data(b16.ptr, &[_]c_int{ 1, FUSED256_MIN_Q_LEN, 64 }, 3, .int32);
        defer _ = mlx.mlx_array_free(bl16);
        try std.testing.expect((try qsaSparseAttn(s, q16, &dv, bl16, ratio, 1.0)) == null);
    }
    {
        const q6 = try attn256RandBf16(rnd, &[_]c_int{ 1, 24, 6, 256 }, s);
        defer _ = mlx.mlx_array_free(q6);
        const b6 = try H.blocksFor(ta, rnd, 6, kv, 64, ratio);
        defer ta.free(b6);
        const bl6 = mlx.mlx_array_new_data(b6.ptr, &[_]c_int{ 1, 6, 64 }, 3, .int32);
        defer _ = mlx.mlx_array_free(bl6);
        const ok = (try qsaSparseAttn(s, q6, &dv, bl6, ratio, 1.0)) orelse return error.SparseAttnDeclined;
        _ = mlx.mlx_array_free(ok);
        {
            transformer.qsa_attn_kernel_override = false;
            defer transformer.qsa_attn_kernel_override = null;
            try std.testing.expect((try qsaSparseAttn(s, q6, &dv, bl6, ratio, 1.0)) == null);
        }
        // The same width on a dense cache is served by the bf16 variant, and its own kill
        // switch declines it without touching the quantized arm.
        var dense_view = DenseKVView{ .k = kd, .v = vd, .owned = false };
        try std.testing.expect(!dense_view.has_quant_triple);
        const dense_ok = (try qsaSparseAttn(s, q6, &dense_view, bl6, ratio, 1.0)) orelse return error.SparseAttnDeclined;
        _ = mlx.mlx_array_free(dense_ok);
        {
            qsa_attn_dense_override = false;
            defer qsa_attn_dense_override = null;
            try std.testing.expect((try qsaSparseAttn(s, q6, &dense_view, bl6, ratio, 1.0)) == null);
            const still_q = (try qsaSparseAttn(s, q6, &dv, bl6, ratio, 1.0)) orelse return error.SparseAttnDeclined;
            _ = mlx.mlx_array_free(still_q);
        }
    }
    // A dense cache past the union gather's dense floor: the fused kernel serves it, and
    // `qsaVerifyGatherAttn` (its fallback) still serves when called directly.
    {
        const big_kv: c_int = QSA_VERIFY_GATHER_MIN_KV_DENSE + 4000;
        const kbig = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, big_kv, 256 }, s);
        defer _ = mlx.mlx_array_free(kbig);
        const vbig = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, big_kv, 256 }, s);
        defer _ = mlx.mlx_array_free(vbig);
        const q4 = try attn256RandBf16(rnd, &[_]c_int{ 1, 24, 4, 256 }, s);
        defer _ = mlx.mlx_array_free(q4);
        const b4 = try H.blocksFor(ta, rnd, 4, big_kv, 128, ratio);
        defer ta.free(b4);
        const bl4 = mlx.mlx_array_new_data(b4.ptr, &[_]c_int{ 1, 4, 128 }, 3, .int32);
        defer _ = mlx.mlx_array_free(bl4);
        var big_dense = DenseKVView{ .k = kbig, .v = vbig, .owned = false };
        const via_fused = (try qsaSparseAttn(s, q4, &big_dense, bl4, ratio, 1.0)) orelse return error.SparseAttnDeclined;
        _ = mlx.mlx_array_free(via_fused);
        const via_gather = (try qsaVerifyGatherAttn(s, q4, &big_dense, bl4, ratio, 1.0)) orelse return error.VerifyGatherDeclined;
        _ = mlx.mlx_array_free(via_gather);
    }
}

test "qsa sparse attn: the config key is kv- and kb-INDEPENDENT (one build per width)" {
    // Nothing in the config reads kv or kb (the kernel takes them at runtime), and kv grows
    // every decode round: same width, any kv, any kb => exactly one build.
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ta = std.testing.allocator;
    const ratio: c_int = 4;
    var prng = std.Random.DefaultPrng.init(0xC0F_9E15);
    const rnd = prng.random();

    const Run = struct {
        fn once(st: mlx.mlx_stream, alloc: std.mem.Allocator, r: std.Random, seq: c_int, kv: c_int, kb: c_int) !void {
            const q = try attn256RandBf16(r, &[_]c_int{ 1, 24, seq, 256 }, st);
            defer _ = mlx.mlx_array_free(q);
            const kd = try attn256RandBf16(r, &[_]c_int{ 1, 2, kv, 256 }, st);
            defer _ = mlx.mlx_array_free(kd);
            const vd = try attn256RandBf16(r, &[_]c_int{ 1, 2, kv, 256 }, st);
            defer _ = mlx.mlx_array_free(vd);
            const bh = try qsaVerifyBlocksHost(alloc, r, seq, kv, kb, 4);
            defer alloc.free(bh);
            const bl = mlx.mlx_array_new_data(bh.ptr, &[_]c_int{ 1, seq, kb }, 3, .int32);
            defer _ = mlx.mlx_array_free(bl);
            var cache = try KVCache.initWithConfig(alloc, 1, kv_quant.KVQuantConfig.affine(8));
            defer cache.deinit();
            var view = try qsaAttnCacheFixture(alloc, st, kd, vd, kv, @divTrunc(kv, 2), kv_quant.KVQuantConfig.affine(8), &cache);
            defer view.deinit();
            const o = (try qsaSparseAttn(st, q, &view, bl, 4, 1.0)) orelse return error.SparseAttnDeclined;
            defer _ = mlx.mlx_array_free(o);
            try mlx.check(mlx.mlx_array_eval(o));
        }
    };
    _ = ratio;

    // Normalize first: one global cache shared with every other test in the file.
    try Run.once(s, ta, rnd, 6, 4096, 64);
    const base = qsaAttnCfgBuilds();
    for (0..3) |_| try Run.once(s, ta, rnd, 6, 4096, 64);
    try std.testing.expectEqual(base, qsaAttnCfgBuilds());
    // A growing kv at a fixed width is the served decode round.
    try Run.once(s, ta, rnd, 6, 8192, 64);
    try Run.once(s, ta, rnd, 6, 12288, 64);
    try Run.once(s, ta, rnd, 6, 20000, 64);
    try std.testing.expectEqual(base, qsaAttnCfgBuilds());
    try Run.once(s, ta, rnd, 6, 20000, 128);
    try std.testing.expectEqual(base, qsaAttnCfgBuilds());
    // A new width is a new grid, new output shapes and a new NSPLIT.
    try Run.once(s, ta, rnd, 7, 4096, 64);
    try std.testing.expectEqual(base + 1, qsaAttnCfgBuilds());
    // Coming back to a width already built is free (the LRU).
    try Run.once(s, ta, rnd, 6, 4096, 64);
    try std.testing.expectEqual(base + 1, qsaAttnCfgBuilds());
    try Run.once(s, ta, rnd, 3, 4096, 64);
    const after_three = qsaAttnCfgBuilds();
    for (0..3) |_| {
        try Run.once(s, ta, rnd, 6, 4096, 64);
        try Run.once(s, ta, rnd, 7, 4096, 64);
        try Run.once(s, ta, rnd, 3, 4096, 64);
    }
    try std.testing.expectEqual(after_three, qsaAttnCfgBuilds());
    // Past SLOTS distinct widths the LRU evicts.
    try std.testing.expect(@TypeOf(qsa_attn_cfgs).SLOTS == 4);
}
