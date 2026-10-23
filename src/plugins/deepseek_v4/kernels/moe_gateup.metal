auto lane = thread_index_in_simdgroup;
uint n = thread_position_in_grid.y;      // output row within the expert
uint e = thread_position_in_grid.z;      // top-K slot

int K = KS;
int N = NS;
int VPW = 32 / BITS;
int K_by_p = K / VPW;
int K_by_gs = K / GS;
uint mask = (1u << BITS) - 1u;

uint eid = uint(inds[e]);
size_t wbase = (size_t)eid * (size_t)N * (size_t)K_by_p + (size_t)n * (size_t)K_by_p;
size_t gbase = (size_t)eid * (size_t)N * (size_t)K_by_gs + (size_t)n * (size_t)K_by_gs;

float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;
for (int pack = int(lane); pack < K_by_p; pack += 32) {
  uint32_t packed_g = wg_q[wbase + (size_t)pack];
  uint32_t packed_u = wu_q[wbase + (size_t)pack];
  int k_base = pack * VPW;
  int gi = k_base / GS;
  float sjg = float(g_scales[gbase + (size_t)gi]);
  float bjg = float(g_biases[gbase + (size_t)gi]);
  float sju = float(u_scales[gbase + (size_t)gi]);
  float bju = float(u_biases[gbase + (size_t)gi]);
  for (int ki = 0; ki < VPW; ki += 4) {
    size_t xi = (size_t)(k_base + ki);
    uint32_t qg = packed_g >> (ki * BITS);
    uint32_t qu = packed_u >> (ki * BITS);
    float x0 = float(x[xi + 0]);
    float x1 = float(x[xi + 1]);
    float x2 = float(x[xi + 2]);
    float x3 = float(x[xi + 3]);
    g0 += x0 * (float((qg >> (0 * BITS)) & mask) * sjg + bjg);
    g1 += x1 * (float((qg >> (1 * BITS)) & mask) * sjg + bjg);
    g2 += x2 * (float((qg >> (2 * BITS)) & mask) * sjg + bjg);
    g3 += x3 * (float((qg >> (3 * BITS)) & mask) * sjg + bjg);
    u0 += x0 * (float((qu >> (0 * BITS)) & mask) * sju + bju);
    u1 += x1 * (float((qu >> (1 * BITS)) & mask) * sju + bju);
    u2 += x2 * (float((qu >> (2 * BITS)) & mask) * sju + bju);
    u3 += x3 * (float((qu >> (3 * BITS)) & mask) * sju + bju);
  }
}
float acc_g = simd_sum((g0 + g1) + (g2 + g3));
float acc_u = simd_sum((u0 + u1) + (u2 + u3));
if (lane == 0) {
  // Round exactly where the composed pair wrote T(acc), then the f32
  // clip chain: gate clips HIGH only, up clips both sides (reference).
  T gt = T(acc_g);
  T ut = T(acc_u);
  float g32 = float(gt);
  float u32 = float(ut);
  float gc = metal::min(g32, consts[0]);
  float uc = metal::clamp(u32, -consts[0], consts[0]);
  float sig = sigtab[as_type<ushort>(T(gc))];
  y[(size_t)e * (size_t)N + (size_t)n] = T((gc * sig) * uc);
}