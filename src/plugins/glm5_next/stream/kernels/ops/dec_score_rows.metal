const uint brow = threadgroup_position_in_grid.z;
auto qr = q + size_t(brow) * (32 * DH);
auto wr = w + size_t(brow) * 32;
auto outr = out + size_t(brow) * size_t(dims[1]);
const int q_pos = dims[0] + int(brow);
