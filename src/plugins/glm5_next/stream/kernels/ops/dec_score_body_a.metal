const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const uint tid = sg * 32 + lane;
const int P = dims[1];
threadgroup float qT[DH * 32];
threadgroup float kp[SGS][DH];
for (uint i = tid; i < uint(32 * DH); i += SGS * 32) qT[(i % DH) * 32 + i / DH] = float(qr[i]);
threadgroup_barrier(mem_flags::mem_threadgroup);
for (int pp = 0; pp < PPS; ++pp) {
const int p = (int(threadgroup_position_in_grid.y) * SGS + int(sg)) * PPS + pp;
if (p >= P) return;
// The pool's key: its completing row's gate half; lane l loads channels 4l..4l+3 (DH == 128).