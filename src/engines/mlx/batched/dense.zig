//! engines/mlx/batched/dense, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns forwardBatchedDecode.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../sdk/util/log.zig");
const layered = @import("layered.zig");
const batchedAttnPerSlot = @import("../../../sdk/blocks/attention/batched.zig").batchedAttnPerSlot;
const DenseKVView = @import("../../../sdk/blocks/attention/kv_view.zig").DenseKVView;
const ForwardCtx = @import("../forward_ctx.zig").ForwardCtx;
const KVCache = @import("../kv/cache.zig").KVCache;
const Transformer = @import("../transformer.zig").Transformer;
const plugin_driver = @import("../plugin/driver.zig");
const MAX_BATCH_ROWS = @import("layered.zig").MAX_BATCH_ROWS;

// ── Batched-decode forward pass ──
//
// One forward call computes next-token logits for N concurrent requests at
// decode step (q_len=1 each). Each request owns its own KVCache via its
// ForwardCtx; weights are shared. Returns N logits arrays of shape [1,1,V],
// one per slot in the input order. Caller owns the returned slice and the
// inner arrays (free each via mlx_array_free, then allocator.free the slice).
//
// Restrictions (enforced upstream by `Scheduler.batchable`):
//   - Standard arch only (no MoE, hybrid SSM, encoder-only).
//   - Decode only (each slot contributes exactly one new token).
//   - No grammar-constrained sampling, no in-flight speculative round.
//
// Per-layer flow:
//   embed → input_norm → Q/K/V proj (B=N, batch-invariant)
//   → Q/K-norm → transpose → mlx_fast_rope_dynamic (per-slot offset)
//   → per-slot cache.update at B=1 (each ctx's cache owns its own state)
//   → gather views, pad to common kv_max, concat axis=0
//   → build [N,1,1,kv_max] additive mask via positions < kv_lens
//   → SDPA → o_proj → MLP → final_norm → lm_head → softcap → demux.
pub fn forwardBatchedDecode(
    self: *Transformer,
    next_tokens: []const u32,
    ctxs: []const *ForwardCtx,
    rope_offsets: []const u32,
) ![]mlx.mlx_array {
    const N: c_int = @intCast(next_tokens.len);
    std.debug.assert(next_tokens.len == ctxs.len);
    std.debug.assert(next_tokens.len == rope_offsets.len);
    std.debug.assert(N >= 1);
    // One-shot engagement line, twin of the GDN path's. Output equality
    // alone cannot tell a batched run from N serial ones, so every guard
    // that claims to exercise this kernel greps for its own marker.
    if (!layered.std_batched_logged) {
        layered.std_batched_logged = true;
        log.info("[batched] batched decode engaged (slots={d})\n", .{next_tokens.len});
    }
    if (self.plugin) |b| return plugin_driver.forwardBatched(self, b, next_tokens, ctxs, rope_offsets, null);

    const cfg = &self.config;
    const h_count = cfg.num_attention_heads;
    const kv_h = cfg.num_key_value_heads;
    const hd = cfg.head_dim;
    const has_dual_hd = cfg.global_head_dim > 0 and cfg.global_head_dim != hd;
    const ghd: u32 = if (has_dual_hd) cfg.global_head_dim else hd;
    const gkv_h: u32 = if (cfg.num_global_key_value_heads > 0) cfg.num_global_key_value_heads else kv_h;
    const attn_scale: f32 = cfg.attnScale();

    // 1. Build [N, 1] int32 token tensor from u32 input.
    var token_buf = try self.allocator.alloc(i32, next_tokens.len);
    defer self.allocator.free(token_buf);
    for (next_tokens, 0..) |t, i| token_buf[i] = @intCast(t);
    const tok_shape = [_]c_int{ N, 1 };
    const token_arr = mlx.mlx_array_new_data(token_buf.ptr, &tok_shape, 2, .int32);
    defer _ = mlx.mlx_array_free(token_arr);

    // 2. Embed → [N, 1, hidden].
    var h = try self.embedding(token_arr);

    // 3. Build per-slot int32 rope-offset array for mlx_fast_rope_dynamic.
    var rope_off_buf = try self.allocator.alloc(i32, rope_offsets.len);
    defer self.allocator.free(rope_off_buf);
    for (rope_offsets, 0..) |o, i| rope_off_buf[i] = @intCast(o);
    const rope_off_shape = [_]c_int{N};
    const rope_offset_arr = mlx.mlx_array_new_data(rope_off_buf.ptr, &rope_off_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(rope_offset_arr);

    // 4. Gemma 4 PLE input — computed once across the batch (token_arr is [N,1]).
    var ple_input: ?mlx.mlx_array = null;
    defer if (ple_input) |p| {
        _ = mlx.mlx_array_free(p);
    };
    if (cfg.hidden_size_per_layer_input > 0) {
        ple_input = try self.computePLEInput(token_arr, h, N, 1);
    }

    const perm = [_]c_int{ 0, 2, 1, 3 };
    const perm_back = [_]c_int{ 0, 2, 1, 3 };
    const none_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none_mask);

    var dt = mlx.DtypeTrace.begin("batched-decode", h, if (self.layers.len > 0) self.layers[0].q_w else null);

    for (0..cfg.num_hidden_layers) |layer_idx| {
        const li: u32 = @intCast(layer_idx);
        const lw = &self.layers[layer_idx];
        const is_global = cfg.isGlobalLayer(li);
        const is_kv_shared = lw.kv_source != null;

        const cur_hd: u32 = if (has_dual_hd and is_global) ghd else hd;
        const cur_kv_h: u32 = if (has_dual_hd and is_global) gkv_h else kv_h;
        const cur_q_shape = [_]c_int{ N, 1, @intCast(h_count), @intCast(cur_hd) };
        const cur_kv_shape = [_]c_int{ N, 1, @intCast(cur_kv_h), @intCast(cur_hd) };
        const cur_out_shape = [_]c_int{ N, 1, @intCast(h_count * cur_hd) };
        // RoPE dims — same logic as forwardStandard's decode path.
        const skip_rope = cfg.layerSkipsRope(li);
        const use_prop_rope = is_global and self.rope_freqs_global != null;
        const rope_dims_partial: c_int = if (is_global and cfg.partial_rotary_factor_global < 1.0)
            @intCast(@as(u32, @intFromFloat(@as(f32, @floatFromInt(cur_hd)) * cfg.partial_rotary_factor_global)))
        else
            @intCast(cur_hd);
        const rope_base_opt = mlx.mlx_optional_float{
            .value = if (is_global) cfg.rope_theta else cfg.rope_local_base_freq,
            .has_value = !use_prop_rope,
        };
        const rope_scale: f32 = if (use_prop_rope) 1.0 else if (is_global) (1.0 / cfg.rope_scaling_factor) else 1.0;
        const rope_freqs: mlx.mlx_array = if (use_prop_rope) self.rope_freqs_global.? else .{ .ctx = null };
        const effective_rope_dims: c_int = if (use_prop_rope) @intCast(cur_hd) else rope_dims_partial;

        const normed = try self.rmsNorm(h, lw.input_norm);
        defer _ = mlx.mlx_array_free(normed);

        // Q projection + reshape + Q-norm + transpose + dynamic RoPE.
        const q = try self.qmatmulMaybeBias(normed, lw.q_w, lw.q_s, lw.q_b, lw.q_bias);
        defer _ = mlx.mlx_array_free(q);

        var q_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_r);
        try mlx.check(mlx.mlx_reshape(&q_r, q, &cur_q_shape, 4, self.s));

        const q_normed: ?mlx.mlx_array = if (lw.q_norm) |qn| try self.rmsNorm(q_r, qn) else null;
        defer if (q_normed) |qn| {
            _ = mlx.mlx_array_free(qn);
        };
        var q_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_t);
        try mlx.check(mlx.mlx_transpose_axes(&q_t, q_normed orelse q_r, &perm, 4, self.s));

        var q_rope = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_rope);
        if (skip_rope) {
            _ = mlx.mlx_array_set(&q_rope, q_t);
        } else {
            try mlx.check(mlx.mlx_fast_rope_dynamic(&q_rope, q_t, effective_rope_dims, false, rope_base_opt, rope_scale, rope_offset_arr, rope_freqs, self.s));
        }

        const max_kv_per_layer: u32 = if (is_global) 0 else if (cfg.has_sliding_window) cfg.sliding_window else 0;

        // Per-slot KV update (or KV-share lookup), then gather views.
        // own_views holds per-slot [1, kv_h, kv_len_i, cur_hd] mlx_arrays.
        // We pad each to the common kv_max and concat axis=0 → [N, kv_h, kv_max, cur_hd].
        if (!is_kv_shared) {
            // Project K, V at full batch (B=N), reshape, normalize, transpose, RoPE.
            const own_k = try self.qmatmulMaybeBias(normed, lw.k_w, lw.k_s, lw.k_b, lw.k_bias);
            defer _ = mlx.mlx_array_free(own_k);
            const own_v = if (lw.k_eq_v)
                own_k
            else
                try self.qmatmulMaybeBias(normed, lw.v_w, lw.v_s, lw.v_b, lw.v_bias);
            defer if (!lw.k_eq_v) {
                _ = mlx.mlx_array_free(own_v);
            };

            var own_k_r = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(own_k_r);
            var own_v_r = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(own_v_r);
            try mlx.check(mlx.mlx_reshape(&own_k_r, own_k, &cur_kv_shape, 4, self.s));
            try mlx.check(mlx.mlx_reshape(&own_v_r, own_v, &cur_kv_shape, 4, self.s));

            var own_k_normed_arr = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(own_k_normed_arr);
            if (lw.k_norm) |kn| {
                own_k_normed_arr = try self.rmsNorm(own_k_r, kn);
            }
            const k_for_rope = if (lw.k_norm != null) own_k_normed_arr else own_k_r;

            var own_v_normed_arr = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(own_v_normed_arr);
            if (cfg.has_v_norm) {
                const vnw = if (has_dual_hd and is_global)
                    (self.v_norm_weight_global orelse self.v_norm_weight.?)
                else
                    self.v_norm_weight.?;
                own_v_normed_arr = try self.rmsNorm(own_v_r, vnw);
            }
            const v_after_norm = if (cfg.has_v_norm) own_v_normed_arr else own_v_r;

            var own_k_t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(own_k_t);
            var own_v_t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(own_v_t);
            try mlx.check(mlx.mlx_transpose_axes(&own_k_t, k_for_rope, &perm, 4, self.s));
            try mlx.check(mlx.mlx_transpose_axes(&own_v_t, v_after_norm, &perm, 4, self.s));

            var own_k_rope = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(own_k_rope);
            if (skip_rope) {
                _ = mlx.mlx_array_set(&own_k_rope, own_k_t);
            } else {
                try mlx.check(mlx.mlx_fast_rope_dynamic(&own_k_rope, own_k_t, effective_rope_dims, false, rope_base_opt, rope_scale, rope_offset_arr, rope_freqs, self.s));
            }

            // Affine-8: one stacked quantize then per-slot writes; other schemes keep per-slot update.
            var cache_buf: [MAX_BATCH_ROWS]*KVCache = undefined;
            for (ctxs, 0..) |slot_ctx, i| cache_buf[i] = slot_ctx.cache;
            try KVCache.appendFromStacked(
                own_k_rope,
                own_v_t,
                cache_buf[0..ctxs.len],
                li,
                self.s,
                max_kv_per_layer,
                self.batched_kv_append_override orelse true,
            );
        }

        // Gather per-slot views and find kv_max. For KV-shared layers the
        // source layer's view is what we read. Reads go through denseView
        // (kv-quant contract) — the cache's raw key_view/value_view hold
        // packed quantized words under --kv-quant, and feeding those to
        // SDPA is a fatal MLX shape error.
        const view_layer: u32 = if (is_kv_shared) lw.kv_source.? else li;
        const dense_views = try self.allocator.alloc(DenseKVView, ctxs.len);
        defer {
            for (dense_views) |*dv| dv.deinit();
            self.allocator.free(dense_views);
        }
        for (dense_views) |*dv| dv.* = .{ .k = .{ .ctx = null }, .v = .{ .ctx = null }, .owned = false };
        var fused_buf: [MAX_BATCH_ROWS]bool = undefined;
        for (ctxs, 0..) |slot_ctx, i| {
            dense_views[i] = try slot_ctx.cache.denseView(view_layer, self.s);
            fused_buf[i] = slot_ctx.kv_attn_fused;
        }

        // SDPA → [N, h_count, 1, cur_hd].
        const attn_out = if (batchedAttnPerSlot(dense_views))
            try self.perSlotBatchedAttn(q_rope, dense_views, fused_buf[0..ctxs.len], 1, attn_scale)
        else
            try self.stackedBatchedAttn(q_rope, dense_views, 1, attn_scale, .{ .ctx = null });
        defer _ = mlx.mlx_array_free(attn_out);

        // Output projection.
        var attn_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_t);
        try mlx.check(mlx.mlx_transpose_axes(&attn_t, attn_out, &perm_back, 4, self.s));
        var attn_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_flat);
        try mlx.check(mlx.mlx_reshape(&attn_flat, attn_t, &cur_out_shape, 3, self.s));

        const o_out = try self.qmatmul(attn_flat, lw.o_w, lw.o_s, lw.o_b);
        defer _ = mlx.mlx_array_free(o_out);

        // MLP path — mirrors forwardStandard exactly.
        if (cfg.has_pre_ff_norm) {
            const attn_normed = try self.rmsNormEps(o_out, lw.post_attn_norm, cfg.postNormEps());
            defer _ = mlx.mlx_array_free(attn_normed);
            var h_new = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&h_new, h, attn_normed, self.s));
            _ = mlx.mlx_array_free(h);
            h = h_new;

            const ff_normed = try self.rmsNorm(h, lw.pre_ff_norm.?);
            defer _ = mlx.mlx_array_free(ff_normed);
            const down = try self.standardMlp(ff_normed, lw, false);
            defer _ = mlx.mlx_array_free(down);

            const mlp_normed = try self.rmsNormEps(down, lw.post_ff_norm.?, cfg.postNormEps());
            defer _ = mlx.mlx_array_free(mlp_normed);
            var h_next = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&h_next, h, mlp_normed, self.s));
            _ = mlx.mlx_array_free(h);
            h = h_next;
        } else {
            var h_new = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&h_new, h, o_out, self.s));
            _ = mlx.mlx_array_free(h);
            h = h_new;

            const ff_normed = try self.rmsNorm(h, lw.post_attn_norm);
            defer _ = mlx.mlx_array_free(ff_normed);
            const down = try self.standardMlp(ff_normed, lw, false);
            defer _ = mlx.mlx_array_free(down);

            var h_next = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&h_next, h, down, self.s));
            _ = mlx.mlx_array_free(h);
            h = h_next;
        }

        // Gemma 4 PLE: per-layer projection gate.
        if (ple_input != null and lw.ple_gate_w != null) {
            const projected = try self.pleProject(h, lw, ple_input.?, li, N, 1);
            defer _ = mlx.mlx_array_free(projected);
            _ = try self.residualTail(&h, projected, lw.ple_norm.?, false, .{ .ctx = null }, .{ .ctx = null }, false);
        }

        // Gemma 4: layer scalar.
        if (lw.layer_scalar) |ls| {
            var h_scaled = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_multiply(&h_scaled, h, ls, self.s));
            _ = mlx.mlx_array_free(h);
            h = h_scaled;
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
    defer _ = mlx.mlx_array_free(logits);

    // Demux: slice axis 0 into N tensors of shape [1, 1, V].
    const lshape = mlx.getShape(logits);
    const vocab: c_int = lshape[2];
    const out = try self.allocator.alloc(mlx.mlx_array, next_tokens.len);
    // Each slot holds an mlx handle; free the handles before the slice.
    var built: usize = 0;
    errdefer {
        for (out[0..built]) |a| _ = mlx.mlx_array_free(a);
        self.allocator.free(out);
    }
    for (out, 0..) |*slot, i| {
        const i_c: c_int = @intCast(i);
        const start = [_]c_int{ i_c, 0, 0 };
        const stop = [_]c_int{ i_c + 1, 1, vocab };
        const strides = [_]c_int{ 1, 1, 1 };
        slot.* = mlx.mlx_array_new();
        built = i + 1;
        try mlx.check(mlx.mlx_slice(slot, logits, &start, 3, &stop, 3, &strides, 3, self.s));
    }
    return out;
}
