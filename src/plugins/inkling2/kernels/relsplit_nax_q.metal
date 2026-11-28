// Inkling 2, 2 to 16 rows on the NAX tensor unit over affine-quantized KV caches (BITS = 4 or 8, GS = 32/64/128):
// relsplit_nax.metal's split-K relative attention reading packed cached KV triples in place (up to 2 lanes per launch,
// 30 buffers <= 31), staging dequantized BK-row tiles in threadgroup memory while own verify keys (kb, vb) stay dense.
// grid (splits * 32 * G, HKV, here), threadgroup (32 * G, 1, 1).
// sizes = { per lane (2): L, S, first, SC, offset, row_in, row_out, 0; then kps, log_n_floor, LT, LI }.
using ext = metal::dextents<int32_t, 2>;
using strides = metal::array<int32_t, 2>;
using dtensor = metal::tensor<device T, ext, metal::tensor_inline>;
using ttensor = metal::tensor<threadgroup T, ext, metal::tensor_inline>;
constexpr int BK = 32;
constexpr float LOG2E = 1.4426950408889634f;
constexpr int G = HQ / HKV;
constexpr int W8 = D / 8;
constexpr int NT = 32 * G;

const int zl = int(threadgroup_position_in_grid.z);
const int L = sizes[zl * 8];
if (L == 0) return;
const int S = sizes[zl * 8 + 1];
const int first = sizes[zl * 8 + 2];
const int SC = sizes[zl * 8 + 3];
const int offset = sizes[zl * 8 + 4];
const int row_in = sizes[zl * 8 + 5];
const int row_out = sizes[zl * 8 + 6];
const int kps = sizes[16];
const int log_n_floor = sizes[17];
const int LT = sizes[18];
const int LI = sizes[19];

const device uint* kqp[2] = {k0q, k1q};
const device ST* ksp[2] = {k0s, k1s};
const device ST* kbp[2] = {k0b, k1b};
const device uint* vqp[2] = {v0q, v1q};
const device ST* vsp[2] = {v0s, v1s};
const device ST* vbp[2] = {v0b, v1b};
const int64_t kq_head[2] = {k0q_strides[1], k1q_strides[1]};
const int64_t kq_row[2] = {k0q_strides[2], k1q_strides[2]};
const int64_t ks_head[2] = {k0s_strides[1], k1s_strides[1]};
const int64_t ks_row[2] = {k0s_strides[2], k1s_strides[2]};

const int split = int(threadgroup_position_in_grid.x);
const int kvh = int(threadgroup_position_in_grid.y);
const int tid = int(thread_position_in_threadgroup.x);
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const int head = kvh * G + int(sg);
const short fm = ((lane >> 2) & 4) | ((lane >> 1) & 3);
const short fn = (((lane >> 2) & 2) | (lane & 1)) * 4;
const float sc = params[0] * LOG2E;
const float log_alpha = params[1];

const int ra = fm, rb = fm + 8;
const bool oka = ra < L, okb = rb < L;
int da = ra, db = rb;
uint ma = 0xFFFFFFFFu, mb = 0xFFFFFFFFu;
if constexpr (TREE) {
    if (oka) { da = TMETA[(row_in + ra) * 2]; ma = uint(TMETA[(row_in + ra) * 2 + 1]); }
    if (okb) { db = TMETA[(row_in + rb) * 2]; mb = uint(TMETA[(row_in + rb) * 2 + 1]); }
}
const int dga = S - L + da, dgb = S - L + db;

float tau[2] = {1.0f, 1.0f};
if constexpr (GLOBAL) {
    const int ta = offset + 1 + da;
    const int tb = offset + 1 + db;
    if (oka && ta > log_n_floor) tau[0] = 1.0f + log_alpha * metal::log(float(ta) / float(log_n_floor));
    if (okb && tb > log_n_floor) tau[1] = 1.0f + log_alpha * metal::log(float(tb) / float(log_n_floor));
}

dtensor tQ((device T*)q + size_t(row_in) * HQ * D + head * D, ext(D, L), strides{1, HQ * D});
constexpr auto dqk = matmul2d_descriptor(16, BK, D, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
constexpr auto dpv = matmul2d_descriptor(16, D, BK, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<dqk, metal::execution_simdgroup> qk;
matmul2d<dpv, metal::execution_simdgroup> pv;

auto Qc = qk.template get_left_input_cooperative_tensor<T, T, float>();
if (L == 16) Qc.load(tQ.template slice<D, 16>(0, 0)); else Qc.load(tQ.slice(0, 0));
using qct_t = metal::remove_addrspace_t<decltype(Qc)>;
using kslice_t = decltype(tQ.template slice<D, BK>(0, 0));
using vslice_t = decltype(tQ.template slice<D, BK>(0, 0));
using pct_t = decltype(pv.template get_left_input_cooperative_tensor<T, T, float>(
    metal::declval<thread decltype(qk.template get_destination_cooperative_tensor<qct_t, kslice_t, T>())&>()));
auto Oc = pv.template get_destination_cooperative_tensor<metal::remove_addrspace_t<pct_t>, vslice_t, float>();
#pragma clang loop unroll(full)
for (uint16_t i = 0; i < Oc.get_capacity(); i++) Oc[i] = 0;

float m[2] = {-1e30f, -1e30f};
float den[2] = {0.0f, 0.0f};
if (split == 0) {
    if (oka) {
        m[0] = float(sinks[(row_in + ra) * sinks_strides[1] + head * sinks_strides[2]]) * LOG2E;
        den[0] = 1.0f;
    }
    if (okb) {
        m[1] = float(sinks[(row_in + rb) * sinks_strides[1] + head * sinks_strides[2]]) * LOG2E;
        den[1] = 1.0f;
    }
}

const int kbeg = first + split * kps;
const int kend = metal::min(S, kbeg + kps);
const device T* rsa = rel + (size_t(row_in + (oka ? ra : 0)) * HQ + head) * DREL;
const device T* rsb = rel + (size_t(row_in + (okb ? rb : 0)) * HQ + head) * DREL;
vec<T, 4> ra4[(DREL + 3) / 4], rb4[(DREL + 3) / 4];
const bool needs_rel = (WINDOW > 0) || (kend - 1 > (S - L) - R);
if (needs_rel && (DREL % 4 == 0)) {
    const device vec<T, 4>* sa4 = (const device vec<T, 4>*)rsa;
    const device vec<T, 4>* sb4 = (const device vec<T, 4>*)rsb;
#pragma clang loop unroll(full)
    for (int d = 0; d < DREL / 4; d++) { ra4[d] = sa4[d]; rb4[d] = sb4[d]; }
}

// Separate K/V staging tiles (16 KB at D=128) halve barriers to 2 per tile: barrier 1 fences prev pv.run(Vs) + curr K decode,
// barrier 2 fences curr qk.run(Ks) + curr V decode.
threadgroup vec<T, 8> Ks_v[BK * W8];
threadgroup vec<T, 8> Vs_v[BK * W8];
ttensor tKs((threadgroup T*)Ks_v, ext(D, BK));
ttensor tVs((threadgroup T*)Vs_v, ext(D, BK));

// 1. Cached quantized keys [kbeg, min(kend, SC))
const int c_end = metal::min(kend, SC);
if (kbeg < c_end) {
    const device uint* kqh = kqp[zl] + size_t(kvh) * kq_head[zl];
    const device ST* ksh = ksp[zl] + size_t(kvh) * ks_head[zl];
    const device ST* kbh = kbp[zl] + size_t(kvh) * ks_head[zl];
    const device uint* vqh = vqp[zl] + size_t(kvh) * kq_head[zl];
    const device ST* vsh = vsp[zl] + size_t(kvh) * ks_head[zl];
    const device ST* vbh = vbp[zl] + size_t(kvh) * ks_head[zl];
    const int64_t qstep = kq_row[zl];
    const int64_t sstep = ks_row[zl];

    for (int k0 = kbeg; k0 < c_end; k0 += BK) {
        const bool full = (k0 + BK <= c_end);
        for (int e = tid; e < BK * W8; e += NT) {
            const int r = e / W8;
            const int c8 = e % W8;
            const int row_idx = k0 + r;
            vec<T, 8> out = vec<T, 8>(T(0));
            if (row_idx < c_end) {
                const int g = (c8 * 8) / GS;
                const float sc_v = float(ksh[size_t(row_idx) * sstep + g]);
                const float bi_v = float(kbh[size_t(row_idx) * sstep + g]);
                if constexpr (BITS == 8) {
                    const size_t base = size_t(row_idx) * qstep + c8 * 2;
                    const uchar4 b0 = as_type<uchar4>(kqh[base]);
                    const uchar4 b1 = as_type<uchar4>(kqh[base + 1]);
                    out[0] = T(fma(float(b0.x), sc_v, bi_v));
                    out[1] = T(fma(float(b0.y), sc_v, bi_v));
                    out[2] = T(fma(float(b0.z), sc_v, bi_v));
                    out[3] = T(fma(float(b0.w), sc_v, bi_v));
                    out[4] = T(fma(float(b1.x), sc_v, bi_v));
                    out[5] = T(fma(float(b1.y), sc_v, bi_v));
                    out[6] = T(fma(float(b1.z), sc_v, bi_v));
                    out[7] = T(fma(float(b1.w), sc_v, bi_v));
                } else {
                    const uint w = kqh[size_t(row_idx) * qstep + c8];
                    out[0] = T(fma(float((w      ) & 0xFu), sc_v, bi_v));
                    out[1] = T(fma(float((w >>  4) & 0xFu), sc_v, bi_v));
                    out[2] = T(fma(float((w >>  8) & 0xFu), sc_v, bi_v));
                    out[3] = T(fma(float((w >> 12) & 0xFu), sc_v, bi_v));
                    out[4] = T(fma(float((w >> 16) & 0xFu), sc_v, bi_v));
                    out[5] = T(fma(float((w >> 20) & 0xFu), sc_v, bi_v));
                    out[6] = T(fma(float((w >> 24) & 0xFu), sc_v, bi_v));
                    out[7] = T(fma(float((w >> 28) & 0xFu), sc_v, bi_v));
                }
            }
            Ks_v[e] = out;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto Sc = qk.template get_destination_cooperative_tensor<qct_t, kslice_t, float>();
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < Sc.get_capacity(); i++) Sc[i] = 0;
        { auto ks = tKs.template slice<D, BK>(0, 0); qk.run(Qc, ks, Sc); }

        const bool plain = WINDOW == 0 && full && (k0 + BK - 1 <= (S - L) - R);
        float mx_[2] = {m[0], m[1]}, f[2], sum[2] = {0.0f, 0.0f};
#pragma clang loop unroll(full)
        for (int i = 0; i < BK / 2; i++) {
            const int h = (i / 4) % 2;
            const bool okr = h ? okb : oka;
            float e = -INFINITY;
            if (plain) {
                if (okr) e = Sc[i] * sc * tau[h];
            } else {
                const int col = k0 + (i / 8) * 16 + fn + i % 4;
                const int dist = (h ? dgb : dga) - col;
                const bool ok = okr && col < c_end && dist >= 0 && (WINDOW == 0 || dist < WINDOW);
                if (ok) {
                    float b = 0.0f;
                    if (dist < R) {
                        const device T* rp = rproj + size_t(dist) * DREL;
                        if constexpr (DREL % 4 == 0) {
                            const device vec<T, 4>* rp4 = (const device vec<T, 4>*)rp;
#pragma clang loop unroll(full)
                            for (int d = 0; d < DREL / 4; d++) b += metal::dot(float4(h ? rb4[d] : ra4[d]), float4(rp4[d]));
                        } else {
                            const device T* rv = h ? rsb : rsa;
#pragma clang loop unroll(full)
                            for (int d = 0; d < DREL; d++) b += float(rv[d]) * float(rp[d]);
                        }
                        b *= LOG2E;
                    }
                    e = (Sc[i] * sc + b) * tau[h];
                }
            }
            Sc[i] = e;
            mx_[h] = metal::max(mx_[h], e);
        }
#pragma clang loop unroll(full)
        for (int r = 0; r < 2; r++) {
            mx_[r] = metal::max(mx_[r], simd_shuffle_xor(mx_[r], 1));
            mx_[r] = metal::max(mx_[r], simd_shuffle_xor(mx_[r], 8));
            f[r] = fast::exp2(m[r] - mx_[r]);
            m[r] = mx_[r];
        }
        auto Ph = qk.template get_destination_cooperative_tensor<qct_t, kslice_t, T>();
#pragma clang loop unroll(full)
        for (int i = 0; i < BK / 2; i++) {
            const float p = fast::exp2(Sc[i] - mx_[(i / 4) % 2]);
            Ph[i] = T(p);
            sum[(i / 4) % 2] += p;
        }
#pragma clang loop unroll(full)
        for (int r = 0; r < 2; r++) {
            sum[r] += simd_shuffle_xor(sum[r], 1);
            sum[r] += simd_shuffle_xor(sum[r], 8);
            den[r] = den[r] * f[r] + sum[r];
        }
#pragma clang loop unroll(full)
        for (int i = 0; i < D / 2; i++) Oc[i] *= f[(i / 4) % 2];

        for (int e = tid; e < BK * W8; e += NT) {
            const int r = e / W8;
            const int c8 = e % W8;
            const int row_idx = k0 + r;
            vec<T, 8> out = vec<T, 8>(T(0));
            if (row_idx < c_end) {
                const int g = (c8 * 8) / GS;
                const float sc_v = float(vsh[size_t(row_idx) * sstep + g]);
                const float bi_v = float(vbh[size_t(row_idx) * sstep + g]);
                if constexpr (BITS == 8) {
                    const size_t base = size_t(row_idx) * qstep + c8 * 2;
                    const uchar4 b0 = as_type<uchar4>(vqh[base]);
                    const uchar4 b1 = as_type<uchar4>(vqh[base + 1]);
                    out[0] = T(fma(float(b0.x), sc_v, bi_v));
                    out[1] = T(fma(float(b0.y), sc_v, bi_v));
                    out[2] = T(fma(float(b0.z), sc_v, bi_v));
                    out[3] = T(fma(float(b0.w), sc_v, bi_v));
                    out[4] = T(fma(float(b1.x), sc_v, bi_v));
                    out[5] = T(fma(float(b1.y), sc_v, bi_v));
                    out[6] = T(fma(float(b1.z), sc_v, bi_v));
                    out[7] = T(fma(float(b1.w), sc_v, bi_v));
                } else {
                    const uint w = vqh[size_t(row_idx) * qstep + c8];
                    out[0] = T(fma(float((w      ) & 0xFu), sc_v, bi_v));
                    out[1] = T(fma(float((w >>  4) & 0xFu), sc_v, bi_v));
                    out[2] = T(fma(float((w >>  8) & 0xFu), sc_v, bi_v));
                    out[3] = T(fma(float((w >> 12) & 0xFu), sc_v, bi_v));
                    out[4] = T(fma(float((w >> 16) & 0xFu), sc_v, bi_v));
                    out[5] = T(fma(float((w >> 20) & 0xFu), sc_v, bi_v));
                    out[6] = T(fma(float((w >> 24) & 0xFu), sc_v, bi_v));
                    out[7] = T(fma(float((w >> 28) & 0xFu), sc_v, bi_v));
                }
            }
            Vs_v[e] = out;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto Pc = pv.template get_left_input_cooperative_tensor<T, T, float>(Ph);
        { auto vs = tVs.template slice<D, BK>(0, 0); pv.run(Pc, vs, Oc); }
    }
}

// 2. Own verify keys [max(kbeg, SC), kend) (dense T, contiguous [1, HKV, LI, D])
const int o_beg = metal::max(kbeg, SC);
if (o_beg < kend) {
    const int n_own = S - SC;
    dtensor tKb((device T*)kb + (size_t(kvh) * LI + row_in) * D, ext(D, n_own), strides{1, D});
    dtensor tVb((device T*)vb + (size_t(kvh) * LI + row_in) * D, ext(D, n_own), strides{1, D});
    for (int k0 = o_beg; k0 < kend; k0 += BK) {
        const int off = k0 - SC;
        auto Sc = qk.template get_destination_cooperative_tensor<qct_t, kslice_t, float>();
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < Sc.get_capacity(); i++) Sc[i] = 0;
        { auto ks = tKb.slice(0, off); qk.run(Qc, ks, Sc); }

        float mx_[2] = {m[0], m[1]}, f[2], sum[2] = {0.0f, 0.0f};
#pragma clang loop unroll(full)
        for (int i = 0; i < BK / 2; i++) {
            const int h = (i / 4) % 2;
            const int col = k0 + (i / 8) * 16 + fn + i % 4;
            int dist;
            bool anc = true;
            if constexpr (TREE) {
                const int kc = col - SC;
                const int kdepth = (kc >= 0 && kc < L) ? TMETA[(row_in + kc) * 2] : kc;
                anc = (kc >= 0 && kc < L) && ((((h ? mb : ma) >> kc) & 1u) != 0u);
                dist = (h ? db : da) - kdepth;
            } else {
                dist = (h ? dgb : dga) - col;
            }
            const bool ok = (h ? okb : oka) && col < kend && anc && dist >= 0 && (WINDOW == 0 || dist < WINDOW);
            float e = -INFINITY;
            if (ok) {
                float b = 0.0f;
                if (dist < R) {
                    const device T* rp = rproj + size_t(dist) * DREL;
                    if constexpr (DREL % 4 == 0) {
                        const device vec<T, 4>* rp4 = (const device vec<T, 4>*)rp;
#pragma clang loop unroll(full)
                        for (int d = 0; d < DREL / 4; d++) b += metal::dot(float4(h ? rb4[d] : ra4[d]), float4(rp4[d]));
                    } else {
                        const device T* rv = h ? rsb : rsa;
#pragma clang loop unroll(full)
                        for (int d = 0; d < DREL; d++) b += float(rv[d]) * float(rp[d]);
                    }
                    b *= LOG2E;
                }
                e = (Sc[i] * sc + b) * tau[h];
            }
            Sc[i] = e;
            mx_[h] = metal::max(mx_[h], e);
        }
#pragma clang loop unroll(full)
        for (int r = 0; r < 2; r++) {
            mx_[r] = metal::max(mx_[r], simd_shuffle_xor(mx_[r], 1));
            mx_[r] = metal::max(mx_[r], simd_shuffle_xor(mx_[r], 8));
            f[r] = fast::exp2(m[r] - mx_[r]);
            m[r] = mx_[r];
        }
        auto Ph = qk.template get_destination_cooperative_tensor<qct_t, kslice_t, T>();
#pragma clang loop unroll(full)
        for (int i = 0; i < BK / 2; i++) {
            const float p = fast::exp2(Sc[i] - mx_[(i / 4) % 2]);
            Ph[i] = T(p);
            sum[(i / 4) % 2] += p;
        }
#pragma clang loop unroll(full)
        for (int r = 0; r < 2; r++) {
            sum[r] += simd_shuffle_xor(sum[r], 1);
            sum[r] += simd_shuffle_xor(sum[r], 8);
            den[r] = den[r] * f[r] + sum[r];
        }
#pragma clang loop unroll(full)
        for (int i = 0; i < D / 2; i++) Oc[i] *= f[(i / 4) % 2];
        auto Pc = pv.template get_left_input_cooperative_tensor<T, T, float>(Ph);
        { auto vs = tVb.slice(0, off); pv.run(Pc, vs, Oc); }
    }
}

if constexpr (DIRECT) {
    const float inv[2] = {1.0f / den[0], 1.0f / den[1]};
    using gtensor = metal::tensor<device float, ext, metal::tensor_inline>;
    gtensor tG((device float*)GATE + size_t(row_in) * HQ * D + head * D, ext(D, L), strides{1, HQ * D});
    auto Gc = pv.template get_destination_cooperative_tensor<metal::remove_addrspace_t<pct_t>, vslice_t, float>();
    if (L == 16) Gc.load(tG.template slice<D, 16>(0, 0)); else Gc.load(tG.slice(0, 0));
    auto Oh = pv.template get_destination_cooperative_tensor<metal::remove_addrspace_t<pct_t>, vslice_t, T>();
#pragma clang loop unroll(full)
    for (int i = 0; i < D / 2; i++) Oh[i] = T(float(T(Oc[i] * inv[(i / 4) % 2])) * Gc[i]);
    dtensor tO((device T*)OUT + size_t(row_out) * HQ * D + head * D, ext(D, L), strides{1, HQ * D});
    if (L == 16) { auto os = tO.template slice<D, 16>(0, 0); Oh.store(os); }
    else { auto os = tO.slice(0, 0); Oh.store(os); }
} else {
    using ptensor = metal::tensor<device float, ext, metal::tensor_inline>;
    ptensor tP(PART + (size_t(split) * LT + row_out) * HQ * D + head * D, ext(D, L), strides{1, HQ * D});
    if (L == 16) { auto ps = tP.template slice<D, 16>(0, 0); Oc.store(ps); }
    else { auto ps = tP.slice(0, 0); Oc.store(ps); }
    if (fn == 0) {
        if (oka) {
            const size_t at = (size_t(split) * LT + row_out + ra) * HQ + head;
            ML[at * 2] = m[0];
            ML[at * 2 + 1] = den[0];
        }
        if (okb) {
            const size_t at = (size_t(split) * LT + row_out + rb) * HQ + head;
            ML[at * 2] = m[1];
            ML[at * 2 + 1] = den[1];
        }
    }
}
