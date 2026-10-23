((threadgroup float4*)kp[sg])[lane] = float4(pooled[lane]);
simdgroup_barrier(mem_flags::mem_threadgroup);
float dot = 0.0f;
for (int d = 0; d < DH; ++d) dot = fma(qT[d * 32 + lane], kp[sg][d], dot);
const float sc = simd_sum(wr[lane] * metal::max(dot, 0.0f));
if (lane == 0) outr[p] = (p * KPOOL + KPOOL - 1 <= q_pos) ? sc : -FLT_MAX;
simdgroup_barrier(mem_flags::mem_threadgroup);
}