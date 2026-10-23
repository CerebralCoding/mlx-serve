const uint i = thread_position_in_grid.x;
if (i < uint(NG)) gate_out[i] = glm5n_kv_gate(fa[i], dt_bias[i % uint(DIM)], a_exp[i % uint(DIM)], cf[0]);
if (i < uint(NS)) sig_out[i] = glm5n_kv_sigmoid<bfloat16_t>(sx[i]);