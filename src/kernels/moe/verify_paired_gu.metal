uint tile = threadgroup_position_in_grid.x;
uint selected = threadgroup_position_in_grid.y;
uint sg = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint out_base = tile * 8 + sg * 4;
uint row = lhs[selected];
uint expert = ids[selected];
const device bfloat* xv = x + row * PGU_H + lane * PGU_VPT;
size_t weight_offset = ((size_t)expert * PGU_N + out_base) * PGU_BYTES_PER_ROW + lane * 8;
size_t meta_offset = ((size_t)expert * PGU_N + out_base) * PGU_GROUPS_PER_ROW + lane / 4;
const device uchar* gwp = (const device uchar*)gw + weight_offset;
const device uchar* uwp = (const device uchar*)uw + weight_offset;
const device bfloat* gsp = gsc + meta_offset;
const device bfloat* gbp = gb + meta_offset;
const device bfloat* usp = usc + meta_offset;
const device bfloat* ubp = ub + meta_offset;
float gr[4] = {0.0f};
float ur[4] = {0.0f};
for (uint k = 0; k < PGU_H; k += PGU_BLOCK) {
  float xt[16];
  float sum = pgu_load(xv, xt);
  for (uint o = 0; o < 4; ++o) {
    gr[o] += pgu_qdot(gwp + o * PGU_BYTES_PER_ROW, xt,
        float(gsp[o * PGU_GROUPS_PER_ROW]), float(gbp[o * PGU_GROUPS_PER_ROW]), sum);
    ur[o] += pgu_qdot(uwp + o * PGU_BYTES_PER_ROW, xt,
        float(usp[o * PGU_GROUPS_PER_ROW]), float(ubp[o * PGU_GROUPS_PER_ROW]), sum);
  }
  xv += PGU_BLOCK;
  gwp += PGU_BLOCK / 2;
  uwp += PGU_BLOCK / 2;
  gsp += PGU_BLOCK / PGU_GS;
  gbp += PGU_BLOCK / PGU_GS;
  usp += PGU_BLOCK / PGU_GS;
  ubp += PGU_BLOCK / PGU_GS;
}
for (uint o = 0; o < 4; ++o) {
  gr[o] = simd_sum(gr[o]);
  ur[o] = simd_sum(ur[o]);
}
if (lane == 0) {
  for (uint o = 0; o < 4; ++o) {
    bfloat gt = bfloat(gr[o]);
    bfloat ut = bfloat(ur[o]);
    bfloat sig = sigtab[as_type<ushort>(gt)];
    bfloat act = gt * sig;
    y[(size_t)selected * PGU_N + out_base + o] = act * ut;
  }
}