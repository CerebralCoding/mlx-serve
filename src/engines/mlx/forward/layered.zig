//! engines/mlx/forward/layered, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns MoeLayerWeights, forwardMoeWith.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const profile = @import("../diag/profile.zig");
const MhcWeights = @import("../../../plugins/glm5_next/weights.zig").MhcWeights;
const inklingSconvCore = @import("../../../plugins/inkling/attention.zig").inklingSconvCore;
const HcWeights = @import("../../../plugins/qwen4_exp/weights.zig").HcWeights;
const PleWeights = @import("../../../plugins/qwen4_exp/weights.zig").PleWeights;
const LinearAttnWeights = @import("../../../sdk/blocks/gdn/gated_delta_net.zig").LinearAttnWeights;
const DenseMlpWeights = @import("../../../sdk/blocks/mlp/dense.zig").DenseMlpWeights;
const fusedAddRmsNorm = @import("../../../sdk/blocks/norms/residual_norm.zig").fusedAddRmsNorm;
const endMropeChunk = @import("../../../sdk/blocks/rope/mrope.zig").endMropeChunk;
const ProfClock = @import("../diag/profile.zig").ProfClock;
const decodeProfReport = @import("../diag/profile.zig").decodeProfReport;
const decodeProfileEnabled = @import("../diag/profile.zig").decodeProfileEnabled;
const layerCap = @import("../diag/profile.zig").layerCap;
const PREFILL_EVAL_CADENCE_DEFAULT = @import("../eval_cadence.zig").PREFILL_EVAL_CADENCE_DEFAULT;
const evalCadencePoint = @import("../eval_cadence.zig").evalCadencePoint;
const prefillEvalCadenceApplies = @import("../eval_cadence.zig").prefillEvalCadenceApplies;
const prefillEvalCadenceFor = @import("../eval_cadence.zig").prefillEvalCadenceFor;
const ForwardCtx = @import("../forward_ctx.zig").ForwardCtx;
const slidingViewFor = @import("../kv/sliding.zig").slidingViewFor;
const FullAttnWeights = @import("../layer_weights.zig").FullAttnWeights;
const Transformer = @import("../transformer.zig").Transformer;
const HybridMlpWeights = @import("hybrid.zig").HybridMlpWeights;

pub const MoeLayerWeights = struct {
    input_norm: mlx.mlx_array,
    post_attn_norm: mlx.mlx_array,
    is_linear: bool,
    attn: union(enum) { full: FullAttnWeights, linear: LinearAttnWeights },
    mlp: HybridMlpWeights,
    // Gemma 4 MoE: separate shared expert MLP (null for Qwen3.5)
    shared_mlp: ?DenseMlpWeights = null,
    // Gemma 4 feedforward norms (null for Qwen3.5)
    pre_ff_norm: ?mlx.mlx_array = null,
    post_ff_norm: ?mlx.mlx_array = null,
    pre_ff_norm_2: ?mlx.mlx_array = null,
    post_ff_norm_1: ?mlx.mlx_array = null,
    post_ff_norm_2: ?mlx.mlx_array = null,
    layer_scalar: ?mlx.mlx_array = null,
    // DiffusionGemma: the causal ENCODER pass multiplies the layer output by
    // its own scalar (model.encoder.language_model.layers.N.layer_scalar) —
    // the only untied encoder text params. The bidirectional decoder pass
    // uses `layer_scalar` above. Null for every other arch.
    encoder_layer_scalar: ?mlx.mlx_array = null,
    // Inkling: the post-attention / post-MLP short-convolution taps
    // ({attn,mlp}_sconv.weight [hidden, k, 1]) and the dense bottom-layer
    // output scale (mlp.global_scale, f32 [1], dense layers only). Null for
    // every other arch.
    attn_sconv_w: ?mlx.mlx_array = null,
    mlp_sconv_w: ?mlx.mlx_array = null,
    mlp_global_scale: ?mlx.mlx_array = null,
    // qwen4_exp: the two gated residual reads/writes around the block and
    // the n-gram PLE on its one layer. Null for every other arch.
    hc_attn: ?HcWeights = null,
    hc_mlp: ?HcWeights = null,
    ple: ?PleWeights = null,
    // GLM-5-Next: the Sinkhorn hyper-connections around attention and the MLP.
    mhc_attn: ?MhcWeights = null,
    mhc_ffn: ?MhcWeights = null,
};

pub fn forwardMoeWith(self: *Transformer, ctx: *ForwardCtx, token_ids: mlx.mlx_array) !mlx.mlx_array {
    self.fwd_gen +%= 1; // per-forward QSA scratch key
    const ml = self.moe_layers.?;
    const offset = ctx.moe_seq_offset.*;
    const cfg = &self.config;
    const is_gemma4 = cfg.isGemma4Layers();
    const is_kolibri = std.mem.eql(u8, cfg.model_type, "kolibri1");
    // kolibri1 and flint run laguna's attention arm (no gate, NoPE on full layers; flint adds sinks).
    const is_laguna = is_kolibri or std.mem.eql(u8, cfg.model_type, "laguna") or std.mem.eql(u8, cfg.model_type, "flint");
    const is_inkling = cfg.isInkling();
    const is_mla = cfg.isMla();
    // gpt_oss and MiMo share the sink attention arm (per-layer geometry).
    const is_sink_attn = std.mem.eql(u8, cfg.model_type, "gpt_oss") or cfg.isMimo();

    // PLD spec-decode: thread the per-position SSM capture flag down to the
    // GatedDeltaNet layers (which don't take the ctx). Reset on exit so it
    // never leaks into a non-capturing forward.
    self.spec_capture_ssm = ctx.capture_ssm_seq;
    defer self.spec_capture_ssm = false;
    self.spec_tree = ctx.tree;
    defer self.spec_tree = null;

    // Decode sub-block profiler: start the clock before embedding.
    const prof_on = decodeProfileEnabled();
    var pclk: ProfClock = if (prof_on) ProfClock.init() else undefined;

    var h = try self.embedding(token_ids);
    {
        errdefer _ = mlx.mlx_array_free(h);

        // Inkling: RMS norm on the embeddings (model.llm.embed_norm).
        if (is_inkling) {
            if (self.embedding_norm) |en| {
                const normed_emb = try self.rmsNorm(h, en);
                _ = mlx.mlx_array_free(h);
                h = normed_emb;
            }
        }

        // Splice vision embeddings at image_token_id positions (prefill only)
        h = try self.applyVisionEmbeddingsWith(ctx, h, token_ids);
    }

    const x_shape = mlx.getShape(h);
    const batch: c_int = x_shape[0];
    const seq_len: c_int = x_shape[1];
    const is_prefill = seq_len > 1;
    const prof = prof_on and (seq_len == 1 or std.c.getenv("MLX_SERVE_DECODE_FWD_UBENCH") != null);
    if (prof) {
        try mlx.check(mlx.mlx_array_eval(h));
        profile.decode_prof.embed_ns += pclk.lap();
    }
    // One-shot dtype trace. A residual stream that is wider than the
    // weights promotes EVERY projection's weight on read, which inflates
    // each stage uniformly rather than showing up as one slow op.
    var dt = mlx.DtypeTrace.begin("moe", h, switch (ml[0].attn) {
        .full => |f| f.q_w,
        else => null,
    });

    // Qwen3-VL interleaved M-RoPE: the per-prefill-chunk cos/sin, shared
    // by every full-attn layer this forward.
    try self.beginMropeChunk(ctx, @intCast(offset), @intCast(seq_len), self.actDtype());
    defer endMropeChunk(ctx);

    // Precompute sliding window masks (Gemma 4 + Laguna + gpt_oss sliding
    // layers). An arch whose attention arm reads these masks MUST appear
    // in this gate — a missing arm here hands that arm an empty mask
    // handle, and its decode branch then attends over the whole history
    // with no error anywhere.
    var local_prefill_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(local_prefill_mask);
    var local_decode_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(local_decode_mask);

    if ((is_gemma4 or is_laguna or is_sink_attn) and cfg.has_sliding_window) {
        const sw: c_int = @intCast(cfg.sliding_window);
        const total_kv: c_int = @as(c_int, @intCast(offset)) + seq_len;
        const sliding = slidingViewFor(cfg, total_kv, seq_len);
        if (is_prefill) {
            // Skipped when the fused hd-256 kernel band-masks in-kernel
            // (the mask itself is chunk x kv_len — GBs at long ctx);
            // gemma4MoeAttnWith/lagunaAttnWith lazily build it if a call
            // declines. Built at the TRIMMED length, which is what the
            // attn fns hand SDPA.
            if (!sliding.band_in_kernel) {
                local_prefill_mask = try self.createSlidingWindowMask(seq_len, sliding.kv_len, sw);
            }
        }
        // Laguna's decode view is already the window (`slidingViewFor`), so it needs no mask.
        if (!is_prefill and total_kv > sw and !is_laguna) {
            const local_kv_len: c_int = @min(total_kv, sw);
            local_decode_mask = try self.createSlidingWindowDecodeMask(local_kv_len, sw);
        }
    }

    // Eval cadence: drop to per-layer when this chunk's score/dequant
    // transients are large (unfused head_dim > 128 at long ctx, or a
    // quantized cache's dense rebuild) — see prefillEvalCadence.
    const moe_eval_cadence = prefillEvalCadenceFor(
        PREFILL_EVAL_CADENCE_DEFAULT,
        cfg,
        @intCast(seq_len),
        @as(u64, @intCast(offset)) + @as(u64, @intCast(seq_len)),
        ctx.cache.config.scheme != .off,
    );

    // DIAGNOSTIC (MLX_SERVE_LAYER_CAP=N): run only the first N layers, so a
    // ms-vs-N sweep separates the forward's per-layer slope from its fixed
    // cost. Every layer that runs does its complete real work, so the slope
    // is a marginal cost, not an ablation artifact.
    // Hadamard packs: the previous layer's tail already produced this
    // layer's input norm (and its rotation, seeded into the registry).
    var carried_normed: ?mlx.mlx_array = null;
    errdefer if (carried_normed) |c| {
        _ = mlx.mlx_array_free(c);
    };
    defer if (self.rht) |reg| reg.dropMemo();
    const n_layers = layerCap(cfg.num_hidden_layers);
    for (0..n_layers) |layer_idx| {
        const li: u32 = @intCast(layer_idx);
        const lw = &ml[layer_idx];

        const normed = if (carried_normed) |c| c else try self.rmsNorm(h, lw.input_norm);
        carried_normed = null;
        defer _ = mlx.mlx_array_free(normed);

        // Inkling: the attention call also advances the k/v short-conv
        // states; they join the tail's attn/mlp states in ONE concatenated
        // conv_state write at the end of the layer.
        var ink_k_state: mlx.mlx_array = .{ .ctx = null };
        var ink_v_state: mlx.mlx_array = .{ .ctx = null };
        errdefer {
            if (ink_k_state.ctx != null) _ = mlx.mlx_array_free(ink_k_state);
            if (ink_v_state.ctx != null) _ = mlx.mlx_array_free(ink_v_state);
        }

        // Attention: linear (GatedDeltaNet) or full
        const attn_out = switch (lw.attn) {
            .linear => |la| try self.gatedDeltaNet(normed, &la, &ctx.ssm_entries.?[layer_idx], layer_idx, batch, seq_len, is_prefill),
            .full => |fa| if (is_inkling) blk: {
                const r = try self.inklingAttnWith(ctx, normed, &fa, &ctx.ssm_entries.?[layer_idx], li, @intCast(offset), batch, seq_len, is_prefill);
                ink_k_state = r.k_state;
                ink_v_state = r.v_state;
                break :blk r.out;
            } else if (is_mla)
                try self.mlaAttnWith(ctx, normed, &fa, li, @intCast(offset), batch, seq_len, is_prefill)
            else if (is_sink_attn)
                try self.sinkAttnWith(ctx, normed, &fa, li, @intCast(offset), batch, seq_len, is_prefill, &local_prefill_mask, local_decode_mask)
            else if (is_laguna)
                try self.lagunaAttnWith(ctx, normed, &fa, li, @intCast(offset), batch, seq_len, is_prefill, &local_prefill_mask)
            else if (is_gemma4)
                try self.gemma4MoeAttnWith(ctx, normed, &fa, li, @intCast(offset), batch, seq_len, is_prefill, &local_prefill_mask, local_decode_mask)
            else
                try self.gatedFullAttnWith(ctx, normed, &fa, li, @intCast(offset), batch, seq_len, is_prefill),
        };
        defer _ = mlx.mlx_array_free(attn_out);
        if (prof) {
            try mlx.check(mlx.mlx_array_eval(attn_out));
            if (lw.attn == .linear) profile.decode_prof.gdn_ns += pclk.lap() else profile.decode_prof.attn_ns += pclk.lap();
        }

        if (is_gemma4) {
            const next_norm = if (layer_idx + 1 == n_layers) self.final_norm else ml[layer_idx + 1].input_norm;
            h = try self.gemma4MoeLayerTail(h, attn_out, lw, ctx.use_encoder_scalars, next_norm, &carried_normed, !is_prefill);
        } else if (is_kolibri) {
            // Sandwich norms: post_attn on the attention output, pre_ff on
            // the residual before the MoE, post_ff on the MoE output.
            const attn_n = try self.rmsNorm(attn_out, lw.post_attn_norm);
            defer _ = mlx.mlx_array_free(attn_n);
            var h_mid = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&h_mid, h, attn_n, self.s));
            _ = mlx.mlx_array_free(h);
            h = h_mid;

            const ff_normed = try self.rmsNorm(h, lw.pre_ff_norm.?);
            defer _ = mlx.mlx_array_free(ff_normed);
            const mlp_out = try self.moeMLP(ff_normed, &lw.mlp.moe);
            defer _ = mlx.mlx_array_free(mlp_out);
            const mlp_n = try self.rmsNorm(mlp_out, lw.post_ff_norm.?);
            defer _ = mlx.mlx_array_free(mlp_n);

            var h_next = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&h_next, h, mlp_n, self.s));
            _ = mlx.mlx_array_free(h);
            h = h_next;
        } else if (is_inkling) {
            // Inkling layer tail (inkling_mlx/layers.py):
            //   h = residual + attn_sconv(attn_out)
            //   h = h + mlp_sconv(mlp(mlp_norm(h)))      [dense × global_scale]
            // then ONE concatenated conv_state write [k | v | attn | mlp].
            const ssm = &ctx.ssm_entries.?[layer_idx];
            const hidden_c: c_int = @intCast(cfg.hidden_size);
            const kv_dim: c_int = @intCast(cfg.num_key_value_heads * cfg.head_dim);

            const a_prev = try self.inklingConvSlot(ssm, 2 * kv_dim, hidden_c);
            defer if (a_prev) |p| {
                _ = mlx.mlx_array_free(p);
            };
            const asc = try inklingSconvCore(self.s, attn_out, lw.attn_sconv_w.?, a_prev);
            defer _ = mlx.mlx_array_free(asc.out);
            errdefer _ = mlx.mlx_array_free(asc.new_state);

            var h_new = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&h_new, h, asc.out, self.s));
            _ = mlx.mlx_array_free(h);
            h = h_new;

            const ff_normed = try self.rmsNorm(h, lw.post_attn_norm);
            defer _ = mlx.mlx_array_free(ff_normed);
            const mlp_out = switch (lw.mlp) {
                .moe => |*mw| try self.inklingMoeMlp(ff_normed, mw),
                .dense => |*dw| blk: {
                    const y = try self.denseMLP(ff_normed, dw);
                    // Learned scalar output gain, multiplied in the
                    // CHECKPOINT's stored dtype (bf16 [1] on the shipped
                    // builds — no promotion; a hypothetical f32 [1] would
                    // widen the stream exactly like the reference).
                    if (lw.mlp_global_scale) |gsc| {
                        var scaled = mlx.mlx_array_new();
                        try mlx.check(mlx.mlx_multiply(&scaled, y, gsc, self.s));
                        _ = mlx.mlx_array_free(y);
                        break :blk scaled;
                    }
                    break :blk y;
                },
            };
            defer _ = mlx.mlx_array_free(mlp_out);

            const m_prev = try self.inklingConvSlot(ssm, 2 * kv_dim + hidden_c, hidden_c);
            defer if (m_prev) |p| {
                _ = mlx.mlx_array_free(p);
            };
            const msc = try inklingSconvCore(self.s, mlp_out, lw.mlp_sconv_w.?, m_prev);
            defer _ = mlx.mlx_array_free(msc.out);
            errdefer _ = mlx.mlx_array_free(msc.new_state);

            var h_next = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&h_next, h, msc.out, self.s));
            _ = mlx.mlx_array_free(h);
            h = h_next;

            // Assemble the layer's conv_state [B, K-1, 2·kv + 2·hidden].
            {
                const arr = [_]mlx.mlx_array{ ink_k_state, ink_v_state, asc.new_state, msc.new_state };
                const vec = mlx.mlx_vector_array_new_data(&arr, 4);
                defer _ = mlx.mlx_vector_array_free(vec);
                var new_state = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_concatenate_axis(&new_state, vec, 2, self.s));
                if (ssm.conv_state.ctx != null) _ = mlx.mlx_array_free(ssm.conv_state);
                ssm.conv_state = new_state;
                ssm.initialized = true;
            }
            _ = mlx.mlx_array_free(ink_k_state);
            ink_k_state = .{ .ctx = null };
            _ = mlx.mlx_array_free(ink_v_state);
            ink_v_state = .{ .ctx = null };
            _ = mlx.mlx_array_free(asc.new_state);
            _ = mlx.mlx_array_free(msc.new_state);
        } else {
            // Qwen3.5: simple residual + post_attn_norm before MLP. The two
            // are strictly serial (the norm waits on the add, the MLP waits
            // on the norm), so one kernel does both — see fusedAddRmsNorm.
            var ff_normed = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ff_normed);
            // A dense MLP only reads `ff_normed` through rotated weights;
            // a MoE router is a plain matmul and needs the real norm.
            if (try self.rhtAddNormRotate(h, attn_out, lw.post_attn_norm, cfg.isMoe())) |fused| {
                _ = mlx.mlx_array_free(h);
                h = fused.sum;
                _ = mlx.mlx_array_free(ff_normed);
                ff_normed = fused.normed;
            } else if (try self.narrowAddNorm(h, attn_out, lw.post_attn_norm)) |fused| {
                _ = mlx.mlx_array_free(h);
                h = fused.h;
                _ = mlx.mlx_array_free(ff_normed);
                ff_normed = fused.normed;
            } else if (try fusedAddRmsNorm(self.s, h, attn_out, lw.post_attn_norm, self.rms_eps_arr)) |fused| {
                _ = mlx.mlx_array_free(h);
                h = fused.sum;
                _ = mlx.mlx_array_free(ff_normed);
                ff_normed = fused.normed;
            } else {
                var h_new = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&h_new, h, attn_out, self.s));
                _ = mlx.mlx_array_free(h);
                h = h_new;
                _ = mlx.mlx_array_free(ff_normed);
                ff_normed = try self.rmsNorm(h, lw.post_attn_norm);
            }
            const mlp_out = switch (lw.mlp) {
                .moe => |*mw| try self.moeMLP(ff_normed, mw),
                // ANE prefill split rides here when armed (--ane-prefill);
                // everything else is plain denseMLP.
                .dense => |*dw| try self.denseMLPMaybeAne(ff_normed, dw, layer_idx, is_prefill, seq_len),
            };
            defer _ = mlx.mlx_array_free(mlp_out);

            const last = layer_idx + 1 == n_layers;
            const next_norm = if (last) self.final_norm else ml[layer_idx + 1].input_norm;
            const next_linear = !last and ml[layer_idx + 1].attn == .linear;
            if (try self.rhtAddNormRotate(h, mlp_out, next_norm, last or next_linear)) |fused| {
                _ = mlx.mlx_array_free(h);
                h = fused.sum;
                carried_normed = fused.normed;
            } else if (try self.narrowAddNorm(h, mlp_out, next_norm)) |fused| {
                _ = mlx.mlx_array_free(h);
                h = fused.h;
                carried_normed = fused.normed;
            } else {
                var h_next = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&h_next, h, mlp_out, self.s));
                _ = mlx.mlx_array_free(h);
                h = h_next;
            }
        }

        if (prof) {
            try mlx.check(mlx.mlx_array_eval(h));
            profile.decode_prof.mlp_ns += pclk.lap();
        }

        // DFlash capture: this h IS `hidden_states[li+1]` — the layer's
        // final output (mirrors forwardStandardWith's capture site).
        if (ctx.capture_layers) |cl| {
            for (cl.ids, cl.out) |cid, *slot| {
                if (cid == li) _ = mlx.mlx_array_set(slot, h);
            }
        }

        if (is_prefill and prefillEvalCadenceApplies(seq_len) and ((layer_idx + 1) % moe_eval_cadence == 0 or layer_idx + 1 == layerCap(cfg.num_hidden_layers))) {
            try evalCadencePoint(h, ctx.ssm_entries);
        } else if (ctx.pipeline_build > 0 and (layer_idx + 1) % ctx.pipeline_build == 0 and layer_idx + 1 < n_layers) {
            const v = mlx.mlx_vector_array_new_value(h);
            defer _ = mlx.mlx_vector_array_free(v);
            try mlx.check(mlx.mlx_async_eval(v));
        }
        self.ladderStep(h, layer_idx, seq_len);
        dt.layer(h, layer_idx);
    }

    ctx.moe_seq_offset.* += @intCast(seq_len);
    dt.end(h);

    var final_normed = if (carried_normed) |c| c else try self.rmsNorm(h, self.final_norm);
    carried_normed = null;
    _ = mlx.mlx_array_free(h);

    // Inkling muP logit scaling: hidden ÷ logits_mup_width_multiplier
    // before the unembed projection (0-dim scalar — no dtype promotion).
    if (is_inkling and cfg.logits_mup_width_multiplier != 1.0) {
        const inv_mup = mlx.mlx_array_new_float(1.0 / cfg.logits_mup_width_multiplier);
        defer _ = mlx.mlx_array_free(inv_mup);
        var mup_scaled = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&mup_scaled, final_normed, inv_mup, self.s));
        _ = mlx.mlx_array_free(final_normed);
        final_normed = mup_scaled;
    }

    // Speculative-decoding capture: slice the post-final-norm hidden
    // at the LAST position only. Used by PLD verify-fusion and the
    // Gemma 4 assistant drafter as `h_prev`. Caller frees the captured
    // array.
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

    if (self.embedding_mode) return final_normed;
    // Diffusion encoder passes exist only to fill the KV cache — the
    // 262K-vocab projection is pure waste there.
    if (ctx.skip_lm_head) {
        if (prof) {
            try mlx.check(mlx.mlx_array_eval(final_normed));
            profile.decode_prof.lmhead_ns += pclk.lap();
            profile.decode_prof.calls += 1;
        }
        return final_normed;
    }
    var logits = try self.lmHeadProject(final_normed);
    _ = mlx.mlx_array_free(final_normed);

    // Inkling: slice the padded vocab rows off (201024 → 200058) so
    // sampling never sees the padding logits.
    if (is_inkling and cfg.unpadded_vocab_size > 0 and cfg.unpadded_vocab_size < cfg.vocab_size) {
        const l_shape = mlx.getShape(logits);
        const uv: c_int = @intCast(cfg.unpadded_vocab_size);
        var sliced = mlx.mlx_array_new();
        const start = [_]c_int{ 0, 0, 0 };
        const stop = [_]c_int{ l_shape[0], l_shape[1], uv };
        const strides = [_]c_int{ 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&sliced, logits, &start, 3, &stop, 3, &strides, 3, self.s));
        _ = mlx.mlx_array_free(logits);
        logits = sliced;
    }

    // Gemma 4: logit softcapping — tanh(logits / cap) * cap
    if (self.softcap_scalar != null) {
        const capped = try self.applySoftcap(logits);
        _ = mlx.mlx_array_free(logits);
        logits = capped;
    }

    if (prof) {
        try mlx.check(mlx.mlx_array_eval(logits));
        profile.decode_prof.lmhead_ns += pclk.lap();
        profile.decode_prof.calls += 1;
        if (profile.decode_prof.calls % 64 == 0) decodeProfReport();
    }

    return logits;
}
