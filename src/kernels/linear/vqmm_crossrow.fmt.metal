auto sg = simdgroup_index_in_threadgroup;
auto lane = thread_index_in_simdgroup;
auto tgp = threadgroup_position_in_grid;

int K = int(K_size);
int N = int(N_size);
const int first_m = int(tgp.x) * 2;
const int out_row = int(tgp.y) * 8 + int(sg) * 4;
const int K_bytes = K / 2;
const int K_groups = K / 64;
const bool has_pair = (first_m + 1) < {d};

const device uint8_t* wq8 = (const device uint8_t*)w_q;
float2 pres[4];
_Pragma("unroll")
for (int r = 0; r < 4; ++r) {{ pres[r] = float2(0.0f); }}

for (int k = 0; k < K; k += 512) {{
  ushort pk[4][4];
  float sl[4]; float bl[4];
  _Pragma("unroll")
  for (int r = 0; r < 4; ++r) {{
    const int row = out_row + r;
    const device uint16_t* ws = (const device uint16_t*)(wq8 + row * K_bytes + (k >> 1) + int(lane) * 8);
    _Pragma("unroll")
    for (int i = 0; i < 4; ++i) {{ pk[r][i] = ws[i]; }}
    const int gi = row * K_groups + (k >> 6) + int(lane) / 4;
    sl[r] = float(scales[gi]);
    bl[r] = float(biases[gi]);
  }}
  const device T* xm0 = x + first_m * K + k + int(lane) * 16;
  float x0[16]; float x1[16];
  float sum0 = 0.0f; float sum1 = 0.0f;
  _Pragma("unroll")
  for (int i = 0; i < 16; i += 4) {{
    sum0 += float(xm0[i]) + float(xm0[i+1]) + float(xm0[i+2]) + float(xm0[i+3]);
    x0[i] = float(xm0[i]);
    x0[i+1] = float(xm0[i+1]) / 16.0f;
    x0[i+2] = float(xm0[i+2]) / 256.0f;
    x0[i+3] = float(xm0[i+3]) / 4096.0f;
  }}
  if (has_pair) {{
    const device T* xm1 = xm0 + K;
    _Pragma("unroll")
    for (int i = 0; i < 16; i += 4) {{
      sum1 += float(xm1[i]) + float(xm1[i+1]) + float(xm1[i+2]) + float(xm1[i+3]);
      x1[i] = float(xm1[i]);
      x1[i+1] = float(xm1[i+1]) / 16.0f;
      x1[i+2] = float(xm1[i+2]) / 256.0f;
      x1[i+3] = float(xm1[i+3]) / 4096.0f;
    }}
  }} else {{
    _Pragma("unroll")
    for (int i = 0; i < 16; ++i) {{ x1[i] = 0.0f; }}
  }}
  _Pragma("unroll")
  for (int r = 0; r < 4; ++r) {{
    float2 acc = float2(0.0f);
    _Pragma("unroll")
    for (int i = 0; i < 4; ++i) {{
      acc += float2(x0[4*i],   x1[4*i])   * float(pk[r][i] & 0x000f);
      acc += float2(x0[4*i+1], x1[4*i+1]) * float(pk[r][i] & 0x00f0);
      acc += float2(x0[4*i+2], x1[4*i+2]) * float(pk[r][i] & 0x0f00);
      acc += float2(x0[4*i+3], x1[4*i+3]) * float(pk[r][i] & 0xf000);
    }}
    pres[r] += sl[r] * acc + float2(sum0, sum1) * bl[r];
  }}
}}

_Pragma("unroll")
for (int r = 0; r < 4; ++r) {{
  float r0 = simd_sum(pres[r].x);
  float r1 = simd_sum(pres[r].y);
  if (lane == 0) {{
    y[first_m * N + out_row + r] = T(r0);
    if (has_pair) {{ y[(first_m + 1) * N + out_row + r] = T(r1); }}
  }}
}}