// Affine dequantization and threadgroup tiling follow MLX's
// quantized_nax.h (Copyright 2023-2025 Apple Inc., MIT; see NOTICE).
// The loader reads TensorFold's canonical [N/32,K/64,32,8] Q4 layout.
// Only one BNx64 bf16 weight tile is materialized, in threadgroup memory.
const uint lid = thread_index_in_threadgroup;
const ushort lane = thread_index_in_simdgroup;
const ushort sg = simdgroup_index_in_threadgroup;
constexpr int BK = 64, LD = 72, SM = BM / 2, SN = 32, WN = BN / SN;
constexpr int KG = K / 64;
constexpr bool FULL_N = (N % BN) == 0;
constexpr bool ALL_ROW_GROUPS_ACTIVE = (M % BM) == 0 || (M % BM) > SM;
const int n0 = threadgroup_position_in_grid.x * BN;
const int m0 = threadgroup_position_in_grid.y * BM;
const int rb = m0 + (sg / WN) * SM;
const int cb = (sg % WN) * SN;
threadgroup bfloat weights[BN * LD];

// Dispatch on the whole threadgroup so both arms execute barriers uniformly.
auto compute = [&](auto full_rows) {
    constexpr auto desc = matmul2d_descriptor(
        SM, SN, BK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> a_base(
        (device bfloat*)X + (int64_t)(ALL_ROW_GROUPS_ACTIVE ? rb : min(rb, M - 1)) * K,
        dextents<int32_t, 2>(K, full_rows.value ? SM : max(1, M - rb)));
    tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> b(
        weights + cb * LD, dextents<int32_t, 2>(LD, SN));
    auto acc = op.template get_destination_cooperative_tensor<decltype(a_base), decltype(b), float>();
    for (int i = 0; i < SM * SN / 32; ++i) acc[i] = 0.0f;

    for (int g = 0; g < KG; ++g) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Two threads share a row; aligned vector loads cover 32 Q4 values.
        const int row = lid / 2, word = (lid % 2) * 4, n = n0 + row;
        uint4 packed(0);
        float scale = 0.0f, bias = 0.0f;
        if (FULL_N || n < N) {
            const int64_t wi = TILED
                ? (((int64_t)(n / 32) * KG + g) * 32 + n % 32) * 8 + word
                : (int64_t)n * (K / 8) + g * 8 + word;
            packed = *((const device uint4*)(Wq + wi));
            const auto sb = ((const device bfloat2*)SBt)[g * N + n];
            scale = float(sb.x);
            bias = float(sb.y);
        }
        for (int i = 0; i < 4; ++i) {
            vec<bfloat, 8> values;
            for (int j = 0; j < 8; ++j)
                values[j] = bfloat(fma(scale, float((packed[i] >> (4 * j)) & 15), bias));
            *((threadgroup vec<bfloat, 8>*)(weights + row * LD + (word + i) * 8)) = values;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (full_rows.value || ALL_ROW_GROUPS_ACTIVE || rb < M) {
            auto a = a_base.slice(g * BK, 0);
            op.run(a, b, acc);
        }
    }

    const short qid = lane >> 2;
    const short fm = (qid & 4) | ((lane >> 1) & 3);
    const short fn = ((qid & 2) | (lane & 1)) * 4;
    for (int tm = 0; tm < SM / 16; ++tm)
        for (int tn = 0; tn < 2; ++tn)
            for (int r = 0; r < 2; ++r)
                for (int j = 0; j < 4; ++j) {
                    const int m = rb + tm * 16 + fm + r * 8;
                    const int n = n0 + cb + tn * 16 + fn + j;
                    if ((full_rows.value || m < M) && (FULL_N || n < N))
                        Y[(int64_t)m * N + n] = bfloat(acc[(tm * 2 + tn) * 8 + r * 4 + j]);
                }
};
if (M % BM == 0 || m0 + BM <= M)
    compute(metal::bool_constant<true>{});
else
    compute(metal::bool_constant<false>{});
