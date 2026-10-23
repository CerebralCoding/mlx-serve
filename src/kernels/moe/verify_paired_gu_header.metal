#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;
constant constexpr uint PGU_H = 2560;
constant constexpr uint PGU_N = 640;
constant constexpr uint PGU_GS = 64;
constant constexpr uint PGU_VPT = 16;
constant constexpr uint PGU_BLOCK = PGU_VPT * 32;
constant constexpr uint PGU_BYTES_PER_ROW = PGU_H / 2;
constant constexpr uint PGU_GROUPS_PER_ROW = PGU_H / PGU_GS;
inline float pgu_load(const device bfloat* x, thread float* xt) {
  float sum = 0.0f;
  for (int i = 0; i < 16; i += 4) {
    sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
    xt[i] = x[i];
    xt[i + 1] = x[i + 1] / 16.0f;
    xt[i + 2] = x[i + 2] / 256.0f;
    xt[i + 3] = x[i + 3] / 4096.0f;
  }
  return sum;
}
inline float pgu_qdot(const device uchar* w, const thread float* xt, float scale, float bias, float sum) {
  float accum = 0.0f;
  const device ushort* ws = (const device ushort*)w;
  for (int i = 0; i < 4; ++i) {
    accum += (xt[4*i] * (ws[i] & 0x000f) + xt[4*i+1] * (ws[i] & 0x00f0)
        + xt[4*i+2] * (ws[i] & 0x0f00) + xt[4*i+3] * (ws[i] & 0xf000));
  }
  return scale * accum + sum * bias;
}