const ushort lane = ushort(thread_index_in_simdgroup);
const ushort warp = ushort(simdgroup_index_in_threadgroup);
const int tix = int(thread_index_in_threadgroup);

const float scale_log2e = scl[0] * 1.44269504088896340736f;
// Query s sits at absolute key position p (bottom-right aligned). Its
// visible keys: `count` complete blocks, then the tail [tail_start, p].
const int p = (kL - qL) + s;
const int complete = (p + 1) / RATIO;
const int count = metal::min(complete, KB);
const int sel_len = count * RATIO;
const int tail_start = complete * RATIO;
const int L = sel_len + (p + 1 - tail_start);
// Split-K. This threadgroup owns virtual keys [t_lo, t_hi) of the row's
// own range [0, L). The virtual index already concatenates the row's
// selected blocks and THEN its ragged tail, so a contiguous virtual
// range is exactly a contiguous block range — plus, for the last
// non-empty split, the tail. Boundaries are BK-aligned so every tile
// but a split's last is full. A split starting past L is EMPTY and
// writes the merge's identity.
const int tiles = (L + BK - 1) / BK;
// BALANCED=0 (default, MEASURED): ceil(tiles/NSPLIT) per split, so the
// last splits can be short or empty — at the swept shape 129 tiles over
// 16 splits is 14 full, 1 partial, 1 empty. BALANCED=1 spreads the
// remainder one tile per split instead. Strictly better load balance on
// paper; NOT the default because the sweep that chose NSPLIT=16 ran on
// the ceil distribution, and changing how work is divided invalidates
// it. Same outputs either way — the split boundaries move, the merge is
// exact over whatever they are.
int t_lo, t_hi;
if (BALANCED) {
  const int base_t = tiles / NSPLIT;
  const int rem_t = tiles % NSPLIT;
  const int start_t = jsp * base_t + metal::min(jsp, rem_t);
  const int cnt_t = base_t + ((jsp < rem_t) ? 1 : 0);
  t_lo = start_t * BK;
  t_hi = metal::min(L, (start_t + cnt_t) * BK);
} else {
  const int tiles_per = (tiles + NSPLIT - 1) / NSPLIT;
  t_lo = jsp * tiles_per * BK;
  t_hi = metal::min(L, t_lo + tiles_per * BK);
}

const device int* blk = blocks + (long)bb * blS[0] + (long)s * blS[1];
