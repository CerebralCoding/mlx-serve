const uint brow = threadgroup_position_in_grid.y;
auto scr = scores + size_t(brow) * size_t(dims[1]);
auto outr = out + size_t(brow) * size_t(OUT_W);
const int q_pos = dims[0] + int(brow);
