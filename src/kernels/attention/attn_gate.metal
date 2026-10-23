uint i = thread_position_in_grid.x;
uint row = thread_position_in_grid.y;
if (i >= uint(NH * HD)) return;

// softplus = logaddexp(0, g), computed in f32 and rounded to the
// activation dtype exactly where mlx_astype would round it.
float g = float(glogits[(size_t)row * NH + i / HD]);
float mx = metal::max(0.0f, g);
float mn = metal::min(0.0f, g);
float sp;
if (metal::isnan(g)) {
  sp = NAN;
} else if (metal::isinf(mx)) {
  sp = mx;
} else {
  sp = mx + log1p(metal::exp(mn - mx));
}

size_t base = (size_t)row * (size_t)(NH * HD);
y[base + i] = T(float(attn[base + i]) * float(T(sp)));