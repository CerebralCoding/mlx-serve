//! plugins/gemma4/ple, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns computePLEInput, pleProject (the legacy engine's; the plugin's are in `decoder.zig`).

const mlx = @import("../../mlx/ffi.zig");
const LayerWeights = @import("../../engines/mlx/layer_weights.zig").LayerWeights;
const bf16Scalar = @import("../../engines/mlx/load/weights.zig").bf16Scalar;
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;
const fusedGeluPleProj = @import("gelu_ple.zig").fusedGeluPleProj;

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
