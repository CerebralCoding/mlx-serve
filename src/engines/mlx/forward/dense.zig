//! engines/mlx/forward/dense, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns forwardStandardWith.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const fusedSdpa256Prefill = @import("../../../sdk/blocks/attention/attn256.zig").fusedSdpa256Prefill;
const kvAttnFusedEligible = @import("../../../sdk/blocks/attention/kv_quant_attn.zig").kvAttnFusedEligible;
const kvAttnVerifyEligible = @import("../../../sdk/blocks/attention/kv_quant_attn.zig").kvAttnVerifyEligible;
const logKvAttnFusedEngaged = @import("../../../sdk/blocks/attention/kv_quant_attn.zig").logKvAttnFusedEngaged;
const qkvAttnDecodeKernel = @import("../../../sdk/blocks/attention/kv_quant_attn.zig").qkvAttnDecodeKernel;
const qkvAttnVerifyKernel = @import("../../../sdk/blocks/attention/kv_quant_attn.zig").qkvAttnVerifyKernel;
const DenseKVView = @import("../../../sdk/blocks/attention/kv_view.zig").DenseKVView;
const sdpaForceFused = @import("../../../sdk/blocks/attention/sdpa.zig").sdpaForceFused;
const splitCausalSdpa = @import("../../../sdk/blocks/attention/sdpa.zig").splitCausalSdpa;
const FUSED_ROWS_MAX_M = @import("../../../sdk/blocks/linear/fused_rows.zig").FUSED_ROWS_MAX_M;
const fusedQkNormRope256 = @import("../../../sdk/blocks/rope/qk_norm_rope.zig").fusedQkNormRope256;
const qkNormRopeFusedEnabled = @import("../../../sdk/blocks/rope/qk_norm_rope.zig").qkNormRopeFusedEnabled;
const EVAL_EVERY_N_LAYERS = @import("../eval_cadence.zig").EVAL_EVERY_N_LAYERS;
const evalCadencePoint = @import("../eval_cadence.zig").evalCadencePoint;
const prefillEvalCadence = @import("../eval_cadence.zig").prefillEvalCadence;
const prefillEvalCadenceApplies = @import("../eval_cadence.zig").prefillEvalCadenceApplies;
const ForwardCtx = @import("../forward_ctx.zig").ForwardCtx;
const slidingViewFor = @import("../kv/sliding.zig").slidingViewFor;
const Transformer = @import("../transformer.zig").Transformer;

// ── Standard forward pass (Gemma / Llama / Qwen3 / Gemma4) ──

pub fn forwardStandardWith(self: *Transformer, ctx: *ForwardCtx, token_ids: mlx.mlx_array) !mlx.mlx_array {
    const offset = ctx.cache.step;
    const cfg = &self.config;
    const h_count = cfg.num_attention_heads;
    const kv_h = cfg.num_key_value_heads;
    const hd = cfg.head_dim;
    const has_dual_hd = cfg.global_head_dim > 0 and cfg.global_head_dim != hd;
    // Gemma 4: 1.0 (QK-norm handles normalization); muse_glimmer:
    // qk_scale_factor/sqrt(head_dim); others: 1/sqrt(query_pre_attn_scalar).
    const attn_scale: f32 = cfg.attnScale();

    var h = try self.embedding(token_ids);

    // Splice vision embeddings at image_token_id positions (prefill only)
    h = try self.applyVisionEmbeddingsWith(ctx, h, token_ids);

    const x_shape = mlx.getShape(h);
    const batch: c_int = x_shape[0];
    const seq_len: c_int = x_shape[1];
    const is_prefill = seq_len > 1;

    // Shapes for sliding-window layers (default)
    const q_shape = [_]c_int{ batch, seq_len, @intCast(h_count), @intCast(hd) };
    const kv_shape = [_]c_int{ batch, seq_len, @intCast(kv_h), @intCast(hd) };
    const out_shape = [_]c_int{ batch, seq_len, @intCast(h_count * hd) };
    // Shapes for global/full-attention layers (only if dual head dims)
    const ghd: u32 = if (has_dual_hd) cfg.global_head_dim else hd;
    const gkv_h: u32 = if (cfg.num_global_key_value_heads > 0) cfg.num_global_key_value_heads else kv_h;
    const q_shape_g = [_]c_int{ batch, seq_len, @intCast(h_count), @intCast(ghd) };
    const kv_shape_g = [_]c_int{ batch, seq_len, @intCast(gkv_h), @intCast(ghd) };
    const out_shape_g = [_]c_int{ batch, seq_len, @intCast(h_count * ghd) };

    const perm = [_]c_int{ 0, 2, 1, 3 };
    const perm_back = [_]c_int{ 0, 2, 1, 3 };

    const none_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none_mask);

    const total_kv: c_int = @as(c_int, @intCast(offset)) + seq_len;
    var local_prefill_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(local_prefill_mask);
    var local_decode_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(local_decode_mask);

    // Sliding layers only need the tail their queries can reach: decode
    // gets the window, a block-wide forward `window + q_len - 1`. See
    // slidingViewFor for the policy (including when the fused hd-256
    // kernel owns the band and the trim is declined).
    const sliding = slidingViewFor(cfg, total_kv, seq_len);
    const sliding_span = sliding.span;
    const sliding_kv_len = sliding.kv_len;
    // An extra additive term has to reach the scores, and the fused band
    // kernel masks from geometry alone — so pads bring the band back to
    // an explicit array. `prefill_mask_add` is null on every serving path.
    const band_in_kernel = sliding.band_in_kernel and ctx.prefill_mask_add == null;

    if (cfg.has_sliding_window) {
        const sw: c_int = @intCast(cfg.sliding_window);
        if (is_prefill) {
            // The sliding window mask limits attention scope. Skipped when
            // the fused hd-256 kernel band-masks in-kernel — the
            // [1,1,chunk,total_kv] mask is itself GBs at long context; the
            // "array" arm lazily builds it if a per-call check declines.
            if (!band_in_kernel) {
                local_prefill_mask = try self.createSlidingWindowMask(seq_len, sliding_kv_len, sw);
            }
        }
        if (!is_prefill and total_kv > sw) {
            // During decode, K view has min(total_kv, sw) entries.
            const local_kv_len: c_int = @min(total_kv, sw);
            local_decode_mask = try self.createSlidingWindowDecodeMask(local_kv_len, sw);
        }
    }

    // Sliding layers need the band AND the caller's term; build the sum
    // once (it is layer-independent) rather than per layer.
    var banded_extra_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(banded_extra_mask);
    if (ctx.prefill_mask_add) |extra| {
        if (is_prefill and cfg.has_sliding_window and local_prefill_mask.ctx != null) {
            try mlx.check(mlx.mlx_add(&banded_extra_mask, local_prefill_mask, extra, self.s));
        }
    }

    // Eval cadence: drop to per-layer when this chunk's score/dequant
    // transients are large (unfused head_dim > 128 at long ctx, or a
    // quantized cache's dense rebuild) — see prefillEvalCadence.
    const std_eval_cadence = prefillEvalCadence(
        EVAL_EVERY_N_LAYERS,
        @max(hd, ghd),
        @max(hd, ghd),
        h_count,
        @max(kv_h, gkv_h),
        @intCast(seq_len),
        @intCast(total_kv),
        ctx.cache.config.scheme != .off,
    );

    // Gemma 4 PLE: compute per-layer input embeddings once before the layer loop.
    // For vision: zero out image token IDs before PLE (reference: text_mask = ~image_mask).
    var ple_input: ?mlx.mlx_array = null;
    defer {
        if (ple_input) |p| _ = mlx.mlx_array_free(p);
    }
    if (cfg.hidden_size_per_layer_input > 0) {
        if (ctx.vision_embeddings != null and cfg.image_token_id > 0) {
            // Zero out image tokens: per_layer_inputs_tokens = where(text_mask, ids, zeros)
            const img_id = mlx.mlx_array_new_int(@intCast(cfg.image_token_id));
            defer _ = mlx.mlx_array_free(img_id);
            var img_mask = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(img_mask);
            try mlx.check(mlx.mlx_equal(&img_mask, token_ids, img_id, self.s));
            // text_mask = NOT image_mask (invert: 1 where text, 0 where image)
            var text_mask_int = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(text_mask_int);
            try mlx.check(mlx.mlx_astype(&text_mask_int, img_mask, .int32, self.s));
            var ones_int = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ones_int);
            const ones_s = [_]c_int{ batch, seq_len };
            try mlx.check(mlx.mlx_ones(&ones_int, &ones_s, 2, .int32, self.s));
            var text_mask = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(text_mask);
            try mlx.check(mlx.mlx_subtract(&text_mask, ones_int, text_mask_int, self.s));
            // ple_ids = token_ids * text_mask (zeros at image positions)
            var ple_ids = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ple_ids);
            try mlx.check(mlx.mlx_multiply(&ple_ids, token_ids, text_mask, self.s));
            ple_input = try self.computePLEInput(ple_ids, h, batch, seq_len);
        } else {
            ple_input = try self.computePLEInput(token_ids, h, batch, seq_len);
        }
    }

    var dt = mlx.DtypeTrace.begin("standard", h, if (self.layers.len > 0) self.layers[0].q_w else null);

    // The next layer's input norm, when the previous layer's residual
    // seam already produced it (residualTail).
    var next_normed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(next_normed);
    for (0..cfg.num_hidden_layers) |layer_idx| {
        const li: u32 = @intCast(layer_idx);
        const lw = &self.layers[layer_idx];
        const is_global = cfg.isGlobalLayer(li);
        const is_kv_shared = lw.kv_source != null;

        var normed = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(normed);
        if (next_normed.ctx != null) {
            normed = next_normed;
            next_normed = mlx.mlx_array_new();
        } else {
            normed = try self.rmsNorm(h, lw.input_norm);
        }
        // At decode width every residual seam is one dispatch
        // (residualTail), the layer-final one carrying the next layer's
        // input norm (or the final norm) as its post-norm.
        const fuse_add_norm = !is_prefill;
        const none = mlx.mlx_array{ .ctx = null };
        const next_input_norm: mlx.mlx_array = if (!fuse_add_norm) none else if (layer_idx + 1 < self.layers.len) self.layers[layer_idx + 1].input_norm else self.final_norm;
        const has_ple = ple_input != null and lw.ple_gate_w != null;
        const layer_scale: mlx.mlx_array = lw.layer_scalar orelse none;

        // Pick shapes based on layer type
        const cur_q_shape: *const [4]c_int = if (has_dual_hd and is_global) &q_shape_g else &q_shape;
        const cur_kv_shape: *const [4]c_int = if (has_dual_hd and is_global) &kv_shape_g else &kv_shape;
        const cur_out_shape: *const [3]c_int = if (has_dual_hd and is_global) &out_shape_g else &out_shape;
        const cur_hd: u32 = if (has_dual_hd and is_global) ghd else hd;
        // RoPE dims: for global layers with partial rotary, only rotate part of head_dim
        const rope_dims: c_int = if (is_global and cfg.partial_rotary_factor_global < 1.0)
            @intCast(@as(u32, @intFromFloat(@as(f32, @floatFromInt(cur_hd)) * cfg.partial_rotary_factor_global)))
        else
            @intCast(cur_hd);

        // Projections: one row-fused matmul at decode width, else separate.
        const fused_rows = lw.qkv.w.ctx != null and batch * seq_len <= FUSED_ROWS_MAX_M;
        var qkv_all = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(qkv_all);
        if (fused_rows) qkv_all = try self.qmatmul(normed, lw.qkv.w, lw.qkv.s, lw.qkv.b);
        var q = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q);
        if (fused_rows) try lw.qkv.part(&q, qkv_all, 0, self.s) else q = try self.qmatmulMaybeBias(normed, lw.q_w, lw.q_s, lw.q_b, lw.q_bias);

        var q_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_r);
        try mlx.check(mlx.mlx_reshape(&q_r, q, cur_q_shape, 4, self.s));

        // RoPE on Q (proportional for global layers when available).
        // muse_glimmer NoPE layers (layer_rope_theta 0) skip rotation on
        // BOTH Q and K — position enters only through the causal mask.
        const skip_rope = cfg.layerSkipsRope(li);
        const use_prop_rope = is_global and self.rope_freqs_global != null;
        const rope_base_opt = mlx.mlx_optional_float{
            .value = if (is_global) cfg.rope_theta else cfg.rope_local_base_freq,
            .has_value = !use_prop_rope,
        };
        const rope_scale: f32 = if (use_prop_rope) 1.0 else if (is_global) (1.0 / cfg.rope_scaling_factor) else 1.0;
        const rope_freqs: mlx.mlx_array = if (use_prop_rope) self.rope_freqs_global.? else .{ .ctx = null };
        // When using proportional RoPE, pass full head_dim (freqs handle partial rotation via inf padding)
        const effective_rope_dims: c_int = if (use_prop_rope) @intCast(cur_hd) else rope_dims;
        const vnw_opt: ?mlx.mlx_array = if (!cfg.has_v_norm) null else if (has_dual_hd and is_global)
            (self.v_norm_weight_global orelse self.v_norm_weight.?)
        else
            self.v_norm_weight.?;

        // K, V projections (own layers), computed up front so the fused
        // norm+RoPE kernel below can take q, k and v in one dispatch.
        var own_k = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(own_k);
        var own_v = mlx.mlx_array_new();
        defer if (!lw.k_eq_v) {
            _ = mlx.mlx_array_free(own_v);
        };
        if (!is_kv_shared) {
            if (fused_rows) try lw.qkv.part(&own_k, qkv_all, 1, self.s) else own_k = try self.qmatmulMaybeBias(normed, lw.k_w, lw.k_s, lw.k_b, lw.k_bias);
            if (lw.k_eq_v) {
                own_v = own_k;
            } else if (fused_rows) {
                try lw.qkv.part(&own_v, qkv_all, 2, self.s);
            } else {
                own_v = try self.qmatmulMaybeBias(normed, lw.v_w, lw.v_s, lw.v_b, lw.v_bias);
            }
        }

        // Decode, hd 256, normed q/k (+ normed v): one kernel replaces the
        // per-head norms, transposes and both RoPEs (fusedQkNormRope256).
        var qkv_fused: ?[3]mlx.mlx_array = null;
        defer if (qkv_fused) |f| {
            for (f) |a| if (a.ctx != null) {
                _ = mlx.mlx_array_free(a);
            };
        };
        if (!is_kv_shared and !lw.k_eq_v and !skip_rope and batch == 1 and seq_len == 1 and cur_hd == 256 and
            (lw.q_norm != null) == (lw.k_norm != null) and (lw.q_norm != null or vnw_opt == null) and
            self.rms_eps_arr.ctx != null and qkNormRopeFusedEnabled())
        blk: {
            var k_r4 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(k_r4);
            try mlx.check(mlx.mlx_reshape(&k_r4, own_k, cur_kv_shape, 4, self.s));
            var v_r4 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(v_r4);
            if (vnw_opt != null) try mlx.check(mlx.mlx_reshape(&v_r4, own_v, cur_kv_shape, 4, self.s));
            const family: usize = if (is_global) 1 else 0;
            const angles = self.qkAngleRowsFor(family, effective_rope_dims, rope_base_opt, rope_freqs, @intCast(offset), 1, 1.0, rope_scale) catch break :blk;
            const none_w = mlx.mlx_array{ .ctx = null };
            qkv_fused = (fusedQkNormRope256(self.s, q_r, k_r4, v_r4, lw.q_norm orelse none_w, lw.k_norm orelse none_w, vnw_opt orelse none_w, angles, self.rms_eps_arr, @intCast(h_count), cur_kv_shape[2], 1, effective_rope_dims) catch null) orelse break :blk;
        }

        // Q norm → transpose → RoPE (composed), unless fused above.
        const q_normed: ?mlx.mlx_array = if (qkv_fused == null and lw.q_norm != null) try self.rmsNorm(q_r, lw.q_norm.?) else null;
        defer {
            if (q_normed) |qn| _ = mlx.mlx_array_free(qn);
        }
        var q_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_t);
        if (qkv_fused == null) try mlx.check(mlx.mlx_transpose_axes(&q_t, q_normed orelse q_r, &perm, 4, self.s));
        var q_rope = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_rope);
        if (qkv_fused) |f| {
            _ = mlx.mlx_array_set(&q_rope, f[0]);
        } else if (skip_rope) {
            _ = mlx.mlx_array_set(&q_rope, q_t);
        } else {
            try mlx.check(mlx.mlx_fast_rope(&q_rope, q_t, effective_rope_dims, false, rope_base_opt, rope_scale, @intCast(offset), rope_freqs, self.s));
        }

        // K, V and cache — either compute or read from shared source.
        // `kv_view` is the lifetime owner: in dense mode it aliases the
        // cache's view (no-op deinit); in quant mode it owns dequantized
        // dense arrays freed at scope exit, after SDPA has consumed them.
        var kv_view: DenseKVView = .{ .k = .{}, .v = .{}, .owned = false };
        defer kv_view.deinit();
        var full_k: mlx.mlx_array = undefined;
        var full_v: mlx.mlx_array = undefined;

        if (is_kv_shared) {
            // KV sharing: read from source layer's cache
            const src = lw.kv_source.?;
            kv_view = try ctx.cache.denseView(src, self.s);
            full_k = kv_view.k;
            full_v = kv_view.v;
        } else {
            var own_k_r = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(own_k_r);
            var own_v_r = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(own_v_r);
            var own_k_normed_arr = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(own_k_normed_arr);
            var own_v_normed_arr = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(own_v_normed_arr);
            var own_k_t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(own_k_t);
            var own_v_t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(own_v_t);
            var own_k_rope = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(own_k_rope);
            if (qkv_fused) |f| {
                _ = mlx.mlx_array_set(&own_k_rope, f[1]);
                if (f[2].ctx != null) {
                    _ = mlx.mlx_array_set(&own_v_t, f[2]);
                } else {
                    try mlx.check(mlx.mlx_reshape(&own_v_r, own_v, cur_kv_shape, 4, self.s));
                    try mlx.check(mlx.mlx_transpose_axes(&own_v_t, own_v_r, &perm, 4, self.s));
                }
            } else {
                try mlx.check(mlx.mlx_reshape(&own_k_r, own_k, cur_kv_shape, 4, self.s));
                try mlx.check(mlx.mlx_reshape(&own_v_r, own_v, cur_kv_shape, 4, self.s));

                // K norm
                if (lw.k_norm) |kn| {
                    own_k_normed_arr = try self.rmsNorm(own_k_r, kn);
                }
                const k_for_rope = if (lw.k_norm != null) own_k_normed_arr else own_k_r;

                // V norm (Gemma 4: parameter-free RMS norm on values)
                if (vnw_opt) |vnw| {
                    own_v_normed_arr = try self.rmsNorm(own_v_r, vnw);
                }
                const v_after_norm = if (vnw_opt != null) own_v_normed_arr else own_v_r;

                try mlx.check(mlx.mlx_transpose_axes(&own_k_t, k_for_rope, &perm, 4, self.s));
                try mlx.check(mlx.mlx_transpose_axes(&own_v_t, v_after_norm, &perm, 4, self.s));

                // RoPE on K (NoPE layers store K un-rotated)
                if (skip_rope) {
                    _ = mlx.mlx_array_set(&own_k_rope, own_k_t);
                } else {
                    try mlx.check(mlx.mlx_fast_rope(&own_k_rope, own_k_t, effective_rope_dims, false, rope_base_opt, rope_scale, @intCast(offset), rope_freqs, self.s));
                }
            }

            // Update KV cache
            const max_kv: u32 = if (is_global) 0 else sliding_span;
            kv_view = try ctx.cache.update(li, own_k_rope, own_v_t, self.s, max_kv);
            full_k = kv_view.k;
            full_v = kv_view.v;
        }

        // Scaled dot-product attention
        var attn_out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_out);

        // Resolve mask first so dense + fused paths share the selection.
        var sel_mode: []const u8 = "";
        var sel_mask: mlx.mlx_array = none_mask;
        if (!cfg.has_sliding_window) {
            if (is_prefill) sel_mode = "causal";
        } else {
            const sw: c_int = @intCast(cfg.sliding_window);
            if (is_prefill and is_global) {
                sel_mode = "causal";
            } else if (is_prefill) {
                sel_mode = "array";
                sel_mask = local_prefill_mask;
            } else if (is_global) {
                // Global layers: full attention, no mask (defaults).
            } else if (blk: {
                const check_layer = if (is_kv_shared) lw.kv_source.? else li;
                break :blk @as(c_int, @intCast(ctx.cache.seqLen(check_layer))) <= sw;
            }) {
                // Within window: no mask needed (defaults).
            } else {
                sel_mode = "array";
                sel_mask = local_decode_mask;
            }
        }

        // Caller-supplied additive term (LTX text encoder's left-padding).
        // "causal" carries no array, so it becomes one — which is why the
        // term is required to be causal itself.
        if (ctx.prefill_mask_add) |extra| {
            if (is_prefill) {
                if (std.mem.eql(u8, sel_mode, "array")) {
                    // The band build is eager whenever a term is present
                    // (`band_in_kernel` is forced off above), so the sum
                    // exists — but a null mlx_array reaching SDPA kills the
                    // request, and dropping the band is the survivable half.
                    sel_mask = if (banded_extra_mask.ctx != null) banded_extra_mask else extra;
                } else {
                    sel_mode = "array";
                    sel_mask = extra;
                }
            }
        }

        // Fused-attn opt-in: consume the cache's quant triples directly
        // via the packed decode kernel. Only when the request opts in AND
        // `kvAttnFusedEligible` passes (scheme .affine with triples,
        // decode width only — verify/prefill widths read dense SDPA).
        // A DECLINED kernel falls through to
        // the dense arms below: the composed qmm chain measured a −17%
        // decode LOSS on the shapes the kernel does not serve (gemma4
        // full-attn, gqa 16 x dk 512, 2026-08-15) and never serves now.
        var kv_fused_done = false;
        if (ctx.kv_attn_fused and kvAttnFusedEligible(&kv_view, seq_len)) {
            // The sliding-window mask is built LAZILY (deferred because
            // the p256 kernel usually covers those layers) — a forward
            // reaching this branch can therefore carry a null sel_mask.
            // Only t_q == 1 gets here now, but the guard stays: the
            // kernel's "array" mode declines a null mask and the dense
            // array arm below builds its own.
            if (std.mem.eql(u8, sel_mode, "array") and sel_mask.ctx == null) {
                local_prefill_mask = try self.createSlidingWindowMask(seq_len, sliding_kv_len, @intCast(cfg.sliding_window));
                sel_mask = local_prefill_mask;
            }
            if (try qkvAttnDecodeKernel(self.s, q_rope, &kv_view, attn_scale, sel_mode, sel_mask)) |fused| {
                logKvAttnFusedEngaged(&kv_view, q_rope, seq_len);
                _ = mlx.mlx_array_free(attn_out);
                attn_out = fused;
                kv_fused_done = true;
            }
        } else if (ctx.kv_attn_fused and kvAttnVerifyEligible(&kv_view, seq_len)) {
            // Spec-verify widths (Phase 2): packed reads via the TQ-wide
            // kernel. "array" (sliding) declines inside; dense arms run.
            if (try qkvAttnVerifyKernel(self.s, q_rope, &kv_view, attn_scale, sel_mode)) |fused| {
                _ = mlx.mlx_array_free(attn_out);
                attn_out = fused;
                kv_fused_done = true;
            }
        }
        if (kv_fused_done) {
            // packed kernel handled this layer
        } else if (std.mem.eql(u8, sel_mode, "causal")) {
            if (try fusedSdpa256Prefill(self.s, q_rope, full_k, full_v, attn_scale, 0)) |fused| {
                _ = mlx.mlx_array_free(attn_out);
                attn_out = fused;
            } else if (try splitCausalSdpa(self.s, q_rope, full_k, full_v, attn_scale)) |split_out| {
                // Verify widths past the vector wall (qL*gqa > 32): vector-path
                // row groups beat MLX's unfused fallback.
                _ = mlx.mlx_array_free(attn_out);
                attn_out = split_out;
            } else {
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "causal", none_mask, .{ .ctx = null }, sdpaForceFused(q_rope, full_k), self.s));
            }
        } else if (std.mem.eql(u8, sel_mode, "array")) {
            var fused_done = false;
            // `prefill_mask_add` declines the fused band: it masks from
            // geometry and would drop the caller's term (and, on a
            // full-attention layer that this term just moved off "causal",
            // would band-mask a layer that has no band at all).
            if (is_prefill and cfg.has_sliding_window and ctx.prefill_mask_add == null) {
                // Sliding-window prefill: the band mask runs in-kernel.
                if (try fusedSdpa256Prefill(self.s, q_rope, full_k, full_v, attn_scale, @intCast(cfg.sliding_window))) |fused| {
                    _ = mlx.mlx_array_free(attn_out);
                    attn_out = fused;
                    fused_done = true;
                }
            }
            if (!fused_done) {
                if (sel_mask.ctx == null) {
                    // The eager mask build was skipped because the fused
                    // kernel was expected to cover this layer but a
                    // per-call precondition declined — build it now (and
                    // keep it for the remaining layers).
                    // Same trimmed length as the eager build: this arm
                    // is only reachable when the band was expected
                    // in-kernel, which is also when the trim declines, but
                    // reading ONE variable keeps them from drifting.
                    local_prefill_mask = try self.createSlidingWindowMask(seq_len, sliding_kv_len, @intCast(cfg.sliding_window));
                    sel_mask = local_prefill_mask;
                }
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "array", sel_mask, .{ .ctx = null }, false, self.s));
            }
        } else {
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "", none_mask, .{ .ctx = null }, false, self.s));
        }

        // Reshape attention output
        var attn_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_t);
        try mlx.check(mlx.mlx_transpose_axes(&attn_t, attn_out, &perm_back, 4, self.s));
        var attn_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_flat);
        try mlx.check(mlx.mlx_reshape(&attn_flat, attn_t, cur_out_shape, 3, self.s));

        var g_pre = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(g_pre);
        if (fused_rows and lw.qkv.count == 4) try lw.qkv.part(&g_pre, qkv_all, 3, self.s);
        const o_in = if (lw.ag_w != null) try self.attnOutGate(normed, attn_flat, lw, g_pre) else attn_flat;
        defer if (lw.ag_w != null) {
            _ = mlx.mlx_array_free(o_in);
        };

        const o_out = try self.qmatmul(o_in, lw.o_w, lw.o_s, lw.o_b);
        defer _ = mlx.mlx_array_free(o_out);

        // MLP with pre/post FF norms (Gemma 3/4 style) or simple residual (Llama style)
        if (cfg.has_pre_ff_norm) {
            const ff_normed = try self.residualTail(&h, o_out, lw.post_attn_norm, true, none, lw.pre_ff_norm.?, fuse_add_norm);
            defer _ = mlx.mlx_array_free(ff_normed);

            const down = try self.standardMlp(ff_normed, lw, fused_rows);
            defer _ = mlx.mlx_array_free(down);

            if (has_ple) {
                _ = try self.residualTail(&h, down, lw.post_ff_norm.?, true, none, none, fuse_add_norm);
            } else {
                next_normed = try self.residualTail(&h, down, lw.post_ff_norm.?, true, layer_scale, next_input_norm, fuse_add_norm);
            }
        } else {
            const ff_normed = try self.residualTail(&h, o_out, none, false, none, lw.post_attn_norm, fuse_add_norm);
            defer _ = mlx.mlx_array_free(ff_normed);

            const down = try self.standardMlp(ff_normed, lw, fused_rows);
            defer _ = mlx.mlx_array_free(down);

            next_normed = try self.residualTail(&h, down, none, false, if (has_ple) none else layer_scale, if (has_ple) none else next_input_norm, fuse_add_norm);
        }

        // Gemma 4 PLE (after attention + MLP), then layer_scalar.
        if (has_ple) {
            const projected = try self.pleProject(h, lw, ple_input.?, li, batch, seq_len);
            defer _ = mlx.mlx_array_free(projected);
            next_normed = try self.residualTail(&h, projected, lw.ple_norm.?, false, layer_scale, next_input_norm, fuse_add_norm);
        }

        // DFlash capture: this h IS `hidden_states[li+1]` — the layer's
        // final output (post layer_scalar / PLE on archs that have them).
        if (ctx.capture_layers) |cl| {
            for (cl.ids, cl.out) |cid, *slot| {
                if (cid == li) _ = mlx.mlx_array_set(slot, h);
            }
        }

        if (is_prefill and prefillEvalCadenceApplies(seq_len) and ((layer_idx + 1) % std_eval_cadence == 0 or layer_idx + 1 == @as(usize, cfg.num_hidden_layers))) {
            try evalCadencePoint(h, ctx.ssm_entries);
        }
        self.ladderStep(h, layer_idx, seq_len);
        dt.layer(h, layer_idx);
    }
    dt.end(h);

    var final_normed = mlx.mlx_array_new();
    if (next_normed.ctx != null) {
        final_normed = next_normed;
        next_normed = mlx.mlx_array_new();
    } else {
        final_normed = try self.rmsNorm(h, self.final_norm);
    }
    _ = mlx.mlx_array_free(h);

    // Speculative-decoding capture: slice the LAST position of the
    // post-final-norm hidden into `capture_hidden` (refcount-shared).
    // Used by PLD verify-fusion and the Gemma 4 assistant drafter
    // (which needs the post-final-norm hidden as h_prev seed). Mirrors
    // the identical block in forwardMoe.
    if (ctx.capture_hidden) |target| {
        const fn_shape = mlx.getShape(final_normed);
        const last = fn_shape[1] - 1;
        const start = [_]c_int{ 0, last, 0 };
        const stop = [_]c_int{ fn_shape[0], fn_shape[1], fn_shape[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        var sliced = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice(&sliced, final_normed, &start, 3, &stop, 3, &strides, 3, self.s));
        _ = mlx.mlx_array_set(target, sliced);
        _ = mlx.mlx_array_free(sliced);
    }
    if (ctx.capture_hidden_all) |target_all| {
        _ = mlx.mlx_array_set(target_all, final_normed);
    }

    if (self.embedding_mode or ctx.skip_lm_head) return final_normed;
    var logits = try self.lmHeadProject(final_normed);
    _ = mlx.mlx_array_free(final_normed);

    // MuseGlimmer: pre-scale logits by output_multiplier BEFORE the softcap
    // (reference: T * tanh(logits * mult / T)).
    if (self.output_mult) |m| {
        var scaled = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&scaled, logits, m, self.s));
        _ = mlx.mlx_array_free(logits);
        logits = scaled;
    }

    // Gemma 4: logit softcapping — tanh(logits / cap) * cap
    if (self.softcap_scalar != null) {
        const capped = try self.applySoftcap(logits);
        _ = mlx.mlx_array_free(logits);
        logits = capped;
    }

    return logits;
}
