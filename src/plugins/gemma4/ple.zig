//! plugins/gemma4/ple, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns computePLEInput, pleProject, GELU_PLE_PROJ_SOURCE, gelu_ple_kernel, GeluPleCfgKey, gelu_ple_cfg, and 3 more.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../../host/log.zig");
const LayerWeights = @import("../../engines/mlx/layer_weights.zig").LayerWeights;
const bf16Scalar = @import("../../engines/mlx/load/weights.zig").bf16Scalar;
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;
const swigluFusedEnabled = @import("../../sdk/blocks/mlp/fused_activations.zig").swigluFusedEnabled;
const QuantParams = @import("../../sdk/quant/params.zig").QuantParams;

// ── Gemma 4: Per-Layer Embeddings (PLE) ──

/// Compute PLE input once before the layer loop.
/// Returns [B, S, num_layers, ple_dim] combining projected main embeddings + per-layer embeddings.
pub fn computePLEInput(self: *Transformer, token_ids: mlx.mlx_array, h: mlx.mlx_array, batch: c_int, seq_len: c_int) !mlx.mlx_array {
    const cfg = &self.config;
    const ple_dim: c_int = @intCast(cfg.hidden_size_per_layer_input);
    const n_layers: c_int = @intCast(cfg.num_hidden_layers);
    const total_ple = n_layers * ple_dim;

    // 1. Per-layer embedding lookup: embed_tokens_per_layer[token_ids] -> [B*S, total_ple]
    const id_shape = mlx.getShape(token_ids);
    const flat_count = id_shape[0] * id_shape[1];
    const flat_shape = [_]c_int{flat_count};
    var flat_ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat_ids);
    try mlx.check(mlx.mlx_reshape(&flat_ids, token_ids, &flat_shape, 1, self.s));

    const ple_emb_raw = try self.dequantTake(self.ple_emb_w, self.ple_emb_s, self.ple_emb_b, flat_ids);
    defer _ = mlx.mlx_array_free(ple_emb_raw);

    // Reshape to [B, S, n_layers, ple_dim] and scale by sqrt(ple_dim)
    const ple_4d_shape = [_]c_int{ batch, seq_len, n_layers, ple_dim };
    var ple_emb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ple_emb);
    try mlx.check(mlx.mlx_reshape(&ple_emb, ple_emb_raw, &ple_4d_shape, 4, self.s));

    const emb_scale = bf16Scalar(@sqrt(@as(f32, @floatFromInt(cfg.hidden_size_per_layer_input))), self.s);
    defer _ = mlx.mlx_array_free(emb_scale);
    var ple_emb_scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ple_emb_scaled);
    try mlx.check(mlx.mlx_multiply(&ple_emb_scaled, ple_emb, emb_scale, self.s));

    // 2. Project main embeddings: h -> [B, S, total_ple]
    const proj_raw = if (self.ple_proj_quantized)
        try self.qmatmul(h, self.ple_proj_w, self.ple_proj_s, self.ple_proj_b)
    else blk: {
        // Unquantized: regular matmul with transposed weight
        var wt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wt);
        try mlx.check(mlx.mlx_transpose(&wt, self.ple_proj_w, self.s));
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_matmul(&result, h, wt, self.s));
        break :blk result;
    };
    defer _ = mlx.mlx_array_free(proj_raw);

    // Scale by hidden_size^-0.5
    const proj_scale = bf16Scalar(1.0 / @sqrt(@as(f32, @floatFromInt(cfg.hidden_size))), self.s);
    defer _ = mlx.mlx_array_free(proj_scale);
    var proj_scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(proj_scaled);
    try mlx.check(mlx.mlx_multiply(&proj_scaled, proj_raw, proj_scale, self.s));

    // Reshape to [B, S, n_layers, ple_dim]
    var proj_4d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(proj_4d);
    try mlx.check(mlx.mlx_reshape(&proj_4d, proj_scaled, &ple_4d_shape, 4, self.s));

    // RMS norm on last dim (ple_dim)
    var proj_normed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(proj_normed);
    try mlx.check(mlx.mlx_fast_rms_norm(&proj_normed, proj_4d, self.ple_proj_norm, cfg.rms_norm_eps, self.s));

    // 3. Combine: (proj_normed + ple_emb_scaled) * (1/sqrt(2))
    var combined = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(combined);
    try mlx.check(mlx.mlx_add(&combined, proj_normed, ple_emb_scaled, self.s));

    const inv_sqrt2 = bf16Scalar(1.0 / @sqrt(2.0), self.s);
    defer _ = mlx.mlx_array_free(inv_sqrt2);
    var result = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&result, combined, inv_sqrt2, self.s));

    _ = &total_ple;
    return result; // [B, S, n_layers, ple_dim]
}

/// PLE gate + projection for one layer: `per_layer_projection(gelu(gate(h)) * ple)`,
/// not yet normed or added (the caller's residualTail does that).
pub fn pleProject(self: *Transformer, h_in: mlx.mlx_array, lw: *const LayerWeights, ple_input: mlx.mlx_array, layer_idx: u32, batch: c_int, seq_len: c_int) !mlx.mlx_array {
    const cfg = &self.config;
    const ple_dim: c_int = @intCast(cfg.hidden_size_per_layer_input);

    // Slice ple_input[:, :, layer_idx, :] -> [B, S, ple_dim]
    const li_c: c_int = @intCast(layer_idx);
    const slice_start = [_]c_int{ 0, 0, li_c, 0 };
    const slice_stop = [_]c_int{ batch, seq_len, li_c + 1, ple_dim };
    const slice_strides = [_]c_int{ 1, 1, 1, 1 };
    var ple_slice = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ple_slice);
    try mlx.check(mlx.mlx_slice(&ple_slice, ple_input, &slice_start, 4, &slice_stop, 4, &slice_strides, 4, self.s));

    // Reshape to [B, S, ple_dim]
    const ple_3d_shape = [_]c_int{ batch, seq_len, ple_dim };
    var ple_3d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ple_3d);
    try mlx.check(mlx.mlx_reshape(&ple_3d, ple_slice, &ple_3d_shape, 3, self.s));

    // gated = gelu(per_layer_input_gate(h)) * ple_slice
    const gate_raw = try self.qmatmul(h_in, lw.ple_gate_w.?, lw.ple_gate_s.?, lw.ple_gate_b.?);
    defer _ = mlx.mlx_array_free(gate_raw);
    // Decode: the gate activation and the projection matmul in one
    // dispatch (fusedGeluPleProj, an exact replica of MLX's qmv).
    if (batch * seq_len == 1) blk: {
        const table = (try self.geluTanhTable(mlx.mlx_array_dtype(gate_raw))) orelse break :blk;
        const qp = self.quantParamsHinted(lw.ple_proj_w.?, lw.ple_proj_s.?, @intCast(ple_dim));
        if (try fusedGeluPleProj(self.s, gate_raw, ple_3d, table, lw.ple_proj_w.?, lw.ple_proj_s.?, lw.ple_proj_b.?, qp)) |y| return y;
    }
    var gated = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gated);
    if (try self.fusedGeluTanhMul(gate_raw, ple_3d)) |g| {
        gated = g;
    } else {
        const gate = try self.gelu(gate_raw);
        defer _ = mlx.mlx_array_free(gate);
        try mlx.check(mlx.mlx_multiply(&gated, gate, ple_3d, self.s));
    }

    // projected = per_layer_projection(gated) -> [B, S, hidden_size];
    // the caller adds rms_norm(projected) to h (residualTail).
    return self.qmatmul(gated, lw.ple_proj_w.?, lw.ple_proj_s.?, lw.ple_proj_b.?);
}

/// Gemma PLE tail at decode: `y = W_proj · (gelu(g) * ple)` in one dispatch.
/// The activation is the table lookup + one multiply; the matmul is MLX's own
/// `qmv` (4-bit, group 64, K = 256 = one block, the "remaining" arm) written
/// out op for op — the bf16 partial sums of `load_vector`, the masked-nibble
/// products of `qdot`, `scale * accum + sum * bias`, `simd_sum`, one rounding
/// — so it is bit-identical to `quantized_matmul` (pinned by the
/// `fused gelu-ple projection` test). Threadgroup = MLX's (32, 2): 8 rows.
const GELU_PLE_PROJ_SOURCE =
    \\uint lid = thread_position_in_threadgroup.x;
    \\uint sg = simdgroup_index_in_threadgroup;
    \\uint lane = thread_index_in_simdgroup;
    \\uint tg = threadgroup_position_in_grid.x;
    \\threadgroup T act[PD];
    \\for (uint i = lid; i < uint(PD); i += 64) act[i] = gtab[as_type<ushort>(g[i])] * ple[i];
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\int prow = int(tg) * 8 + int(sg) * 4;
    \\if (prow >= HID) return;
    \\const device uint8_t* ws2 = (const device uint8_t*)pw + prow * (PD / 2) + lane * 4;
    \\const device T* sl2 = ps + prow * (PD / 64) + lane / 8;
    \\const device T* bl2 = pb + prow * (PD / 64) + lane / 8;
    \\const threadgroup T* x2 = act + lane * 8;
    \\float xt2[8];
    \\float sum2 = 0;
    \\for (int i = 0; i < 8; i += 4) {
    \\  sum2 += x2[i] + x2[i + 1] + x2[i + 2] + x2[i + 3];
    \\  xt2[i] = x2[i];
    \\  xt2[i + 1] = x2[i + 1] / 16.0f;
    \\  xt2[i + 2] = x2[i + 2] / 256.0f;
    \\  xt2[i + 3] = x2[i + 3] / 4096.0f;
    \\}
    \\for (int row = 0; row < 4; row++) {
    \\  const device uint16_t* wl = (const device uint16_t*)(ws2 + row * (PD / 2));
    \\  float s = sl2[row * (PD / 64)];
    \\  float b = bl2[row * (PD / 64)];
    \\  float accum = 0;
    \\  for (int i = 0; i < 2; i++) {
    \\    accum += (xt2[4 * i] * (wl[i] & 0x000f) + xt2[4 * i + 1] * (wl[i] & 0x00f0) +
    \\              xt2[4 * i + 2] * (wl[i] & 0x0f00) + xt2[4 * i + 3] * (wl[i] & 0xf000));
    \\  }
    \\  float r = simd_sum(s * accum + sum2 * b);
    \\  if (lane == 0) y[prow + row] = T(r);
    \\}
;

var gelu_ple_kernel: ?mlx.mlx_fast_metal_kernel = null;
const GeluPleCfgKey = struct { hid: c_int, pd: c_int, dtype: mlx.mlx_dtype };
var gelu_ple_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var gelu_ple_cfg_key: GeluPleCfgKey = std.mem.zeroes(GeluPleCfgKey);
var gelu_ple_engaged: bool = false;

/// Null unless the projection is 4-bit affine group 64 over exactly 256
/// inputs at decode width (the served Gemma 4 geometry); anything else keeps
/// the composed ops.
pub fn fusedGeluPleProj(s: mlx.mlx_stream, g: mlx.mlx_array, ple: mlx.mlx_array, table: mlx.mlx_array, pw: mlx.mlx_array, ps: mlx.mlx_array, pb: mlx.mlx_array, qp: QuantParams) !?mlx.mlx_array {
    if (!swigluFusedEnabled() or !mlx.streamIsGpu(s)) return null;
    if (qp.mode != .affine or qp.bits != 4 or qp.group_size != 64) return null;
    if (pw.ctx == null or ps.ctx == null or pb.ctx == null) return null;
    const dt = mlx.mlx_array_dtype(g);
    if ((dt != .bfloat16 and dt != .float16) or mlx.mlx_array_dtype(ple) != dt or mlx.mlx_array_dtype(ps) != dt or mlx.mlx_array_dtype(pb) != dt or mlx.mlx_array_dtype(table) != dt) return null;
    const gsh = mlx.getShape(g);
    const psh = mlx.getShape(ple);
    const wsh = mlx.getShape(pw);
    const ssh = mlx.getShape(ps);
    if (gsh.len != 3 or psh.len != 3 or gsh[0] != 1 or gsh[1] != 1 or psh[0] != 1 or psh[1] != 1) return null;
    const pd = gsh[2];
    if (pd != 256 or psh[2] != pd) return null;
    if (wsh.len != 2 or ssh.len != 2 or wsh[1] * 8 != pd or ssh[1] * 64 != pd or ssh[0] != wsh[0]) return null;
    const hid = wsh[0];
    if (hid < 8 or @rem(hid, 8) != 0) return null;

    const key = GeluPleCfgKey{ .hid = hid, .pd = pd, .dtype = dt };
    if (gelu_ple_cfg == null or !std.meta.eql(gelu_ple_cfg_key, key)) {
        if (gelu_ple_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const config = mlx.mlx_fast_metal_kernel_config_new();
        const ysh = [_]c_int{ 1, 1, hid };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &ysh, 3, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 64 * @divTrunc(hid, 8), 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 64, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "HID", hid));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "PD", pd));
        gelu_ple_cfg = config;
        gelu_ple_cfg_key = key;
    }
    if (gelu_ple_kernel == null) {
        const input_names = [_][*:0]const u8{ "g", "ple", "gtab", "pw", "ps", "pb" };
        const output_names = [_][*:0]const u8{"y"};
        const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
        defer _ = mlx.mlx_vector_string_free(in_vec);
        const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
        defer _ = mlx.mlx_vector_string_free(out_vec);
        const kh = mlx.mlx_fast_metal_kernel_new("mlxserve_gelu_ple_proj", in_vec, out_vec, GELU_PLE_PROJ_SOURCE, "", true, false);
        if (kh.ctx == null) return error.MetalKernelCompileFailed;
        gelu_ple_kernel = kh;
    }
    const inputs_arr = [_]mlx.mlx_array{ g, ple, table, pw, ps, pb };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, gelu_ple_kernel.?, inputs_vec, gelu_ple_cfg.?, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var y = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&y, outputs_vec, 0));
    if (!gelu_ple_engaged) {
        gelu_ple_engaged = true;
        log.info("[ple] fused gelu*ple + projection engaged: hid={d} pd={d}\n", .{ hid, pd });
    }
    return y;
}
