int h = thread_position_in_grid.y;
int lane = thread_position_in_threadgroup.x;
if (h >= H) return; // uniform per threadgroup (grid.y == H)
// TK is an INPUT, not a template arg: it changes every token during the
// context ramp, and a template value would JIT a fresh Metal kernel per
// tk (measured: first request 19 tok/s vs 29.7 warm).
int TK = int(tk_size[0]);
threadgroup float red[TG];
float sk = sink[h];
float mx = sk;
for (int j = lane; j < TK; j += TG) mx = metal::max(mx, scores[h * TK + j] * consts[0]);
red[lane] = mx;
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
for (int off = TG / 2; off > 0; off >>= 1) {
  if (lane < off) red[lane] = metal::max(red[lane], red[lane + off]);
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
}
float m = red[0];
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
float sum = 0.0f;
for (int j = lane; j < TK; j += TG) sum += metal::exp(scores[h * TK + j] * consts[0] - m);
red[lane] = sum;
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
for (int off = TG / 2; off > 0; off >>= 1) {
  if (lane < off) red[lane] += red[lane + off];
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
}
float inv = 1.0f / (red[0] + metal::exp(sk - m));
for (int j = lane; j < TK; j += TG) {
  y[h * TK + j] = metal::exp(scores[h * TK + j] * consts[0] - m) * inv;
}