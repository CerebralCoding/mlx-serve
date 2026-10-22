//! plugins/diffusion_gemma/forward, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns diffusionDecoderAttn, forwardDiffusionDecoder.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const debugTraceHead = @import("../../engines/mlx/diag/profile.zig").debugTraceHead;
const ForwardCtx = @import("../../engines/mlx/forward_ctx.zig").ForwardCtx;
const FullAttnWeights = @import("../../engines/mlx/layer_weights.zig").FullAttnWeights;
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;
const DenseKVView = @import("../../sdk/blocks/attention/kv_view.zig").DenseKVView;

/// Decoder attention over a noisy canvas: Q/K/V computed for the canvas
/// only, RoPEd at absolute positions [offset, offset+L); the committed
/// context K/V is READ from the cache (never written) and concatenated.
/// Attention is fully bidirectional — no mask at all (B=1, no padding):
/// full layers see the whole cache + canvas; sliding layers see the last
/// `sliding_window − 1` cached positions + the whole canvas (enforced by
/// slicing the cached K/V, mirroring mlx-vlm's O(window) trick — the HF
/// reference zeroes the same region with a mask).
pub fn diffusionDecoderAttn(
    self: *Transformer,
    ctx: *ForwardCtx,
    x: mlx.mlx_array,
    fa: *const FullAttnWeights,
    layer: u32,
    offset: c_int,
    batch: c_int,
    seq_len: c_int,
) !mlx.mlx_array {
    const cfg = &self.config;
    const is_global = cfg.isGlobalLayer(layer);
    const h_count: c_int = @intCast(cfg.num_attention_heads);

    const cur_hd: u32 = cfg.layerHeadDim(layer);
    const cur_kv_h: u32 = cfg.layerKVHeads(layer);
    const q_shape = [_]c_int{ batch, seq_len, h_count, @intCast(cur_hd) };
    const kv_shape = [_]c_int{ batch, seq_len, @intCast(cur_kv_h), @intCast(cur_hd) };
    const flat_shape = [_]c_int{ batch, seq_len, @intCast(@as(u32, @intCast(h_count)) * cur_hd) };

    const use_prop_rope = is_global and self.rope_freqs_global != null;
    const rope_dims: c_int = @intCast(cur_hd);
    const rope_base = mlx.mlx_optional_float{ .value = if (is_global) cfg.rope_theta else cfg.rope_local_base_freq, .has_value = !use_prop_rope };
    const rope_scale: f32 = if (use_prop_rope) 1.0 else if (is_global) (1.0 / cfg.rope_scaling_factor) else 1.0;
    const rope_freqs: mlx.mlx_array = if (use_prop_rope) self.rope_freqs_global.? else .{ .ctx = null };

    const perm = [_]c_int{ 0, 2, 1, 3 };
    const none_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none_mask);

    // Q projection + norm + RoPE
    const q_proj = try self.qmatmul(x, fa.q_w, fa.q_s, fa.q_b);
    defer _ = mlx.mlx_array_free(q_proj);
    var q_r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_r);
    try mlx.check(mlx.mlx_reshape(&q_r, q_proj, &q_shape, 4, self.s));
    const q_normed = try self.rmsNorm(q_r, fa.q_norm);
    defer _ = mlx.mlx_array_free(q_normed);
    var q_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_t);
    try mlx.check(mlx.mlx_transpose_axes(&q_t, q_normed, &perm, 4, self.s));
    var q_rope = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_rope);
    try mlx.check(mlx.mlx_fast_rope(&q_rope, q_t, rope_dims, false, rope_base, rope_scale, offset, rope_freqs, self.s));

    // K, V projections. On full layers fa.v_w aliases fa.k_w (no v_proj
    // in the checkpoint): V = param-free-norm(k_proj out) PRE-k_norm,
    // PRE-RoPE — same recompute the gemma4 MoE attention does.
    const k_proj = try self.qmatmul(x, fa.k_w, fa.k_s, fa.k_b);
    defer _ = mlx.mlx_array_free(k_proj);
    const v_proj = try self.qmatmul(x, fa.v_w, fa.v_s, fa.v_b);
    defer _ = mlx.mlx_array_free(v_proj);
    var k_r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_r);
    var v_r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v_r);
    try mlx.check(mlx.mlx_reshape(&k_r, k_proj, &kv_shape, 4, self.s));
    try mlx.check(mlx.mlx_reshape(&v_r, v_proj, &kv_shape, 4, self.s));

    const k_normed = try self.rmsNorm(k_r, fa.k_norm);
    defer _ = mlx.mlx_array_free(k_normed);

    var v_after_norm = v_r;
    var v_normed_arr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v_normed_arr);
    if (cfg.has_v_norm) {
        const has_dual_hd = cfg.global_head_dim > 0 and cfg.global_head_dim != cfg.head_dim;
        const vnw = if (has_dual_hd and is_global)
            (self.v_norm_weight_global orelse self.v_norm_weight.?)
        else
            self.v_norm_weight.?;
        v_normed_arr = try self.rmsNorm(v_r, vnw);
        v_after_norm = v_normed_arr;
    }

    var k_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_t);
    var v_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v_t);
    try mlx.check(mlx.mlx_transpose_axes(&k_t, k_normed, &perm, 4, self.s));
    try mlx.check(mlx.mlx_transpose_axes(&v_t, v_after_norm, &perm, 4, self.s));

    var k_rope = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_rope);
    try mlx.check(mlx.mlx_fast_rope(&k_rope, k_t, rope_dims, false, rope_base, rope_scale, offset, rope_freqs, self.s));

    // Read the committed-context K/V from the cache WITHOUT writing.
    var full_k = k_rope;
    var full_v = v_t;
    var concat_k = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(concat_k);
    var concat_v = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(concat_v);
    var kv_view: ?DenseKVView = null;
    defer if (kv_view) |*kv| kv.deinit();
    if (ctx.cache.seqLen(layer) > 0) {
        kv_view = try ctx.cache.denseView(layer, self.s);
        var cache_k = kv_view.?.k;
        var cache_v = kv_view.?.v;
        // Sliding layers: the canvas only attends to the last window−1
        // committed positions.
        var sliced_k = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sliced_k);
        var sliced_v = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sliced_v);
        if (!is_global and cfg.has_sliding_window) {
            const window: c_int = @as(c_int, @intCast(cfg.sliding_window)) - 1;
            const ck_shape = mlx.getShape(cache_k);
            const cache_len = ck_shape[2];
            if (window > 0 and cache_len > window) {
                const start = [_]c_int{ 0, 0, cache_len - window, 0 };
                const stop = [_]c_int{ ck_shape[0], ck_shape[1], cache_len, ck_shape[3] };
                const strides = [_]c_int{ 1, 1, 1, 1 };
                try mlx.check(mlx.mlx_slice(&sliced_k, cache_k, &start, 4, &stop, 4, &strides, 4, self.s));
                try mlx.check(mlx.mlx_slice(&sliced_v, cache_v, &start, 4, &stop, 4, &strides, 4, self.s));
                cache_k = sliced_k;
                cache_v = sliced_v;
            }
        }
        const kvec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(kvec);
        _ = mlx.mlx_vector_array_append_value(kvec, cache_k);
        _ = mlx.mlx_vector_array_append_value(kvec, k_rope);
        try mlx.check(mlx.mlx_concatenate_axis(&concat_k, kvec, 2, self.s));
        const vvec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vvec);
        _ = mlx.mlx_vector_array_append_value(vvec, cache_v);
        _ = mlx.mlx_vector_array_append_value(vvec, v_t);
        try mlx.check(mlx.mlx_concatenate_axis(&concat_v, vvec, 2, self.s));
        full_k = concat_k;
        full_v = concat_v;
    }

    // Bidirectional SDPA — no mask (scale 1.0; q/k norms absorb it).
    var attn_out = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn_out);
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, 1.0, "", none_mask, .{ .ctx = null }, false, self.s));

    const perm_back = [_]c_int{ 0, 2, 1, 3 };
    var attn_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn_t);
    try mlx.check(mlx.mlx_transpose_axes(&attn_t, attn_out, &perm_back, 4, self.s));
    var attn_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn_flat);
    try mlx.check(mlx.mlx_reshape(&attn_flat, attn_t, &flat_shape, 3, self.s));

    return self.qmatmul(attn_flat, fa.o_w, fa.o_s, fa.o_b);
}

/// DiffusionGemma decoder forward: one denoising step over the canvas.
/// `canvas_ids` [1, L] are the current noisy token ids;
/// `self_cond_embeddings` [1, L, H] is the previous step's soft embedding
/// (probs @ embed_table × √H) or null on the first step. The KV cache is
/// read-only here and `ctx.moe_seq_offset` does NOT advance — the canvas
/// is re-forwarded every step until it commits, at which point the caller
/// runs a causal ENCODER pass (forwardMoeWith with use_encoder_scalars)
/// over the committed tokens to extend the cache.
/// Returns softcapped logits [1, L, vocab]; caller frees.
pub fn forwardDiffusionDecoder(
    self: *Transformer,
    ctx: *ForwardCtx,
    canvas_ids: mlx.mlx_array,
    self_cond_embeddings: ?mlx.mlx_array,
) !mlx.mlx_array {
    const ml = self.moe_layers orelse return error.DiffusionRequiresMoeLayers;
    const sc = self.self_cond orelse return error.DiffusionWeightsMissing;
    const cfg = &self.config;
    const offset: c_int = @intCast(ctx.moe_seq_offset.*);

    // Canvas embeddings (× √hidden) + self-conditioning. Even with a
    // null signal the embeddings pass through the module's scale-free
    // post_norm (FFN(pre_norm(0)) ≡ 0), so layer-0 input is always
    // RMS-normalized.
    var h = try self.embedding(canvas_ids);
    if (self_cond_embeddings) |sig| {
        const dbg_sc = std.c.getenv("MLX_SERVE_DIFFUSION_TRACE") != null;
        if (dbg_sc) debugTraceHead("sc sig", sig, self.s);
        const sig_normed = try self.rmsNorm(sig, sc.pre_norm);
        defer _ = mlx.mlx_array_free(sig_normed);
        if (dbg_sc) debugTraceHead("sc normed", sig_normed, self.s);
        const gate = try self.qmatmul(sig_normed, sc.gate_w, sc.gate_s, sc.gate_b);
        defer _ = mlx.mlx_array_free(gate);
        const up = try self.qmatmul(sig_normed, sc.up_w, sc.up_s, sc.up_b);
        defer _ = mlx.mlx_array_free(up);
        const act = try self.computeGeglu(gate, up);
        defer _ = mlx.mlx_array_free(act);
        const ffn_out = try self.qmatmul(act, sc.down_w, sc.down_s, sc.down_b);
        defer _ = mlx.mlx_array_free(ffn_out);
        if (dbg_sc) debugTraceHead("sc signal", ffn_out, self.s);
        var summed = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(summed);
        try mlx.check(mlx.mlx_add(&summed, h, ffn_out, self.s));
        const post = try self.rmsNorm(summed, self.ones_hidden.?);
        _ = mlx.mlx_array_free(h);
        h = post;
        if (dbg_sc) debugTraceHead("sc out", h, self.s);
    } else {
        const post = try self.rmsNorm(h, self.ones_hidden.?);
        _ = mlx.mlx_array_free(h);
        h = post;
    }

    const x_shape = mlx.getShape(h);
    const batch: c_int = x_shape[0];
    const seq_len: c_int = x_shape[1];

    const dbg = std.c.getenv("MLX_SERVE_DIFFUSION_TRACE") != null;
    if (dbg) debugTraceHead("embed+sc", h, self.s);

    var dt = mlx.DtypeTrace.begin("diffusion-decoder", h, switch (ml[0].attn) {
        .full => |f| f.q_w,
        else => null,
    });

    for (0..cfg.num_hidden_layers) |layer_idx| {
        const li: u32 = @intCast(layer_idx);
        const lw = &ml[layer_idx];

        const normed = try self.rmsNorm(h, lw.input_norm);
        defer _ = mlx.mlx_array_free(normed);

        const attn_out = switch (lw.attn) {
            .full => |fa| try self.diffusionDecoderAttn(ctx, normed, &fa, li, offset, batch, seq_len),
            .linear => return error.DiffusionUnsupportedLinearAttn,
        };
        defer _ = mlx.mlx_array_free(attn_out);
        if (dbg and layer_idx == 0) {
            debugTraceHead("layer0 normed", normed, self.s);
            debugTraceHead("layer0 attn_out", attn_out, self.s);
        }

        var no_carry: ?mlx.mlx_array = null;
        h = try self.gemma4MoeLayerTail(h, attn_out, lw, false, .{ .ctx = null }, &no_carry, false);
        if (dbg and (layer_idx == 0 or layer_idx == 1 or layer_idx == 5 or layer_idx == 29)) {
            var buf: [32]u8 = undefined;
            const label = std.fmt.bufPrint(&buf, "after layer {d}", .{layer_idx}) catch "layer";
            debugTraceHead(label, h, self.s);
        }
        dt.layer(h, layer_idx);
    }
    dt.end(h);

    const final_normed = try self.rmsNorm(h, self.final_norm);
    _ = mlx.mlx_array_free(h);
    var logits = try self.lmHeadProject(final_normed);
    _ = mlx.mlx_array_free(final_normed);

    if (self.softcap_scalar != null) {
        const capped = try self.applySoftcap(logits);
        _ = mlx.mlx_array_free(logits);
        logits = capped;
    }
    return logits;
}
