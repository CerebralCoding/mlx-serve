#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
uint pair = thread_position_in_grid.x;
uint pair_count = uint(x_shape[1]) * uint(x_shape[2]) * uint(x_shape[3] / 2);
if (pair >= pair_count) return;
uint half_dim = x_shape[3] / 2;
uint token = pair / (x_shape[2] * half_dim);
uint table = token * half_dim + pair % half_dim;
float re = float(x[2 * pair]);
float im = float(x[2 * pair + 1]);
float c = cs[table];
float s = sn[table];
float rc = re * c;
float is = im * s;
float rs = re * s;
float ic = im * c;
out[2 * pair] = T(rc - is);
out[2 * pair + 1] = T(rs + ic);