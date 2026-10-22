//! plugins/laguna/attention, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns lagunaAttnWith.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const Weights = @import("../../engines/mlx/model.zig").Weights;
const ForwardCtx = @import("../../engines/mlx/forward_ctx.zig").ForwardCtx;
const testing = @import("../../engines/mlx/kv/cache.zig").testing;
const slidingViewFor = @import("../../engines/mlx/kv/sliding.zig").slidingViewFor;
const FullAttnWeights = @import("../../engines/mlx/layer_weights.zig").FullAttnWeights;
const transposeBf16Weight = @import("../../engines/mlx/load/weights.zig").transposeBf16Weight;
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;
const constTableAs = @import("../../mlx/dtype.zig").constTableAs;
const fusedAttnGate = @import("../../sdk/blocks/attention/gates.zig").fusedAttnGate;
const kvAttnFusedEligible = @import("../../sdk/blocks/attention/kv_quant_attn.zig").kvAttnFusedEligible;
const logKvAttnFusedEngaged = @import("../../sdk/blocks/attention/kv_quant_attn.zig").logKvAttnFusedEngaged;
const qkvAttnDecodeKernel = @import("../../sdk/blocks/attention/kv_quant_attn.zig").qkvAttnDecodeKernel;
const DenseKVView = @import("../../sdk/blocks/attention/kv_view.zig").DenseKVView;
const sdpaForceFused = @import("../../sdk/blocks/attention/sdpa.zig").sdpaForceFused;
const sliceQkvPart = @import("../../sdk/blocks/linear/fused_rows.zig").sliceQkvPart;
const fusedQkNormRope = @import("../../sdk/blocks/rope/qk_norm_rope.zig").fusedQkNormRope;
const qkNormRopeFusedEnabled = @import("../../sdk/blocks/rope/qk_norm_rope.zig").qkNormRopeFusedEnabled;
const lagunaRouterBase = @import("load.zig").lagunaRouterBase;

pub fn lagunaAttnWith(
    self: *Transformer,
    ctx: *ForwardCtx,
    x: mlx.mlx_array,
    fa: *const FullAttnWeights,
    layer: u32,
    offset: c_int,
    batch: c_int,
    seq_len: c_int,
    is_prefill: bool,
    local_prefill_mask: *mlx.mlx_array,
) !mlx.mlx_array {
    const cfg = &self.config;
    // "global" = full_attention layer (isGlobalLayer keys on layer_is_global).
    const is_full = cfg.isGlobalLayer(layer);
    const h_count: c_int = @intCast(cfg.layerNumHeads(layer)); // 48 full / 72 sliding
    const kv_h: c_int = @intCast(cfg.num_key_value_heads); // 8 uniform
    const hd: c_int = @intCast(cfg.head_dim); // 128
    const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.query_pre_attn_scalar)));
    const q_shape = [_]c_int{ batch, seq_len, h_count, hd };
    const kv_shape = [_]c_int{ batch, seq_len, kv_h, hd };
    const flat_shape = [_]c_int{ batch, seq_len, h_count * hd };
    const perm = [_]c_int{ 0, 2, 1, 3 };
    const perm_back = [_]c_int{ 0, 2, 1, 3 };

    // Per-layer-type RoPE. Full layers: YaRN precomputed freqs (dims =
    // rotary_dim = head_dim*partial_global) + mscale on the rotated slice.
    // Sliding layers: default RoPE at rope_local_base_freq, full rotary.
    const use_yarn = is_full and self.rope_freqs_yarn != null;
    const no_rope = cfg.layerSkipsRope(layer); // kolibri1 full layers
    const rope_dims: c_int = if (use_yarn)
        @intFromFloat(@as(f32, @floatFromInt(cfg.head_dim)) * cfg.partial_rotary_factor_global)
    else
        @intFromFloat(@as(f32, @floatFromInt(cfg.head_dim)) * cfg.partial_rotary_factor);
    const rope_base = mlx.mlx_optional_float{
        .value = if (is_full) cfg.rope_theta else cfg.rope_local_base_freq,
        .has_value = !use_yarn, // yarn freqs override base
    };
    const rope_freqs: mlx.mlx_array = if (use_yarn) self.rope_freqs_yarn.? else .{ .ctx = null };

    const none_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none_mask);

    // Q/K/V. With MLX_SERVE_FUSED_QKV=1 the three weights were concatenated
    // at first use, so this is ONE matmul over x and three slices of the
    // result — bit-identical, since each output element is the same dot
    // product over the same axis.
    var q_proj = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_proj);
    var k_proj = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_proj);
    var v_proj = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v_proj);
    var fused_qkv_used = false;
    if (try self.fusedQkvFor(fa, layer)) |f| {
        const qkv = try self.qmatmul(x, f.w, f.s, f.b);
        defer _ = mlx.mlx_array_free(qkv);
        try sliceQkvPart(&q_proj, qkv, 0, f.q_out, self.s);
        try sliceQkvPart(&k_proj, qkv, f.q_out, f.k_out, self.s);
        try sliceQkvPart(&v_proj, qkv, f.q_out + f.k_out, f.v_out, self.s);
        fused_qkv_used = true;
    } else {
        _ = mlx.mlx_array_free(q_proj);
        q_proj = try self.attnProj(x, fa.q_w, fa.q_s, fa.q_b, batch == 1 and !is_prefill, layer);
    }
    // Fused QK-norm+RoPE decode path: one dispatch replaces reshape →
    // per-head RMSNorm → transpose → RoPE (→ YaRN mscale) for q AND k.
    // Bit-identical to the composed chain below (pinned by the `qk norm
    // rope fused` parity tests); composed remains the prefill and
    // fallback path.
    var q_rope = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_rope);
    var k_rope = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_rope);
    var fused_qk = false;
    if (batch == 1 and seq_len == 1 and cfg.head_dim == 128 and
        !fused_qkv_used and fa.q_norm.ctx != null and fa.k_norm.ctx != null and
        self.rms_eps_arr.ctx != null and qkNormRopeFusedEnabled())
    blk: {
        // The kernel consumes k_proj now; the composed path projects it
        // after q. A decline below leaves it computed — the composed arm
        // re-projects, which only costs the rare fallback boot.
        _ = mlx.mlx_array_free(k_proj);
        k_proj = try self.attnProj(x, fa.k_w, fa.k_s, fa.k_b, true, layer);
        const ms: ?mlx.mlx_array = if (use_yarn) msblk: {
            const ms_f32 = self.yarn_mscale orelse break :blk;
            break :msblk try constTableAs(ms_f32, mlx.mlx_array_dtype(q_proj), &self.yarn_mscale_cast, self.s);
        } else null;
        const family: usize = if (use_yarn) 1 else 0;
        // A NoPE layer (Flint's full layers) runs the norm alone at rd 0; q_norm stands in for the unread angles.
        const angles = if (no_rope) fa.q_norm else self.qkAngleFor(family, rope_dims, rope_base, rope_freqs, offset, 1.0) catch break :blk;
        const pair = (fusedQkNormRope(self.s, q_proj, k_proj, fa.q_norm, fa.k_norm, angles, if (no_rope) null else ms, self.rms_eps_arr, h_count, kv_h, if (no_rope) 0 else rope_dims) catch null) orelse break :blk;
        _ = mlx.mlx_array_free(q_rope);
        _ = mlx.mlx_array_free(k_rope);
        q_rope = pair[0];
        k_rope = pair[1];
        fused_qk = true;
    }
    if (!fused_qk) {
        var q_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_r);
        try mlx.check(mlx.mlx_reshape(&q_r, q_proj, &q_shape, 4, self.s));
        const q_normed = try self.rmsNorm(q_r, fa.q_norm);
        defer _ = mlx.mlx_array_free(q_normed);
        var q_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_t);
        try mlx.check(mlx.mlx_transpose_axes(&q_t, q_normed, &perm, 4, self.s));
        if (no_rope) {
            try mlx.check(mlx.mlx_array_set(&q_rope, q_t));
        } else if (ctx.batch_rope_offsets) |off_arr| {
            try mlx.check(mlx.mlx_fast_rope_dynamic(&q_rope, q_t, rope_dims, false, rope_base, 1.0, off_arr, rope_freqs, self.s));
        } else {
            try mlx.check(mlx.mlx_fast_rope(&q_rope, q_t, rope_dims, false, rope_base, 1.0, offset, rope_freqs, self.s));
        }

        // K: proj → reshape → k_norm → transpose → rope
        if (!fused_qkv_used) {
            _ = mlx.mlx_array_free(k_proj);
            k_proj = try self.attnProj(x, fa.k_w, fa.k_s, fa.k_b, batch == 1 and !is_prefill, layer);
        }
        var k_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_r);
        try mlx.check(mlx.mlx_reshape(&k_r, k_proj, &kv_shape, 4, self.s));
        const k_normed = try self.rmsNorm(k_r, fa.k_norm);
        defer _ = mlx.mlx_array_free(k_normed);
        var k_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_t);
        try mlx.check(mlx.mlx_transpose_axes(&k_t, k_normed, &perm, 4, self.s));
        if (no_rope) {
            try mlx.check(mlx.mlx_array_set(&k_rope, k_t));
        } else if (ctx.batch_rope_offsets) |off_arr| {
            try mlx.check(mlx.mlx_fast_rope_dynamic(&k_rope, k_t, rope_dims, false, rope_base, 1.0, off_arr, rope_freqs, self.s));
        } else {
            try mlx.check(mlx.mlx_fast_rope(&k_rope, k_t, rope_dims, false, rope_base, 1.0, offset, rope_freqs, self.s));
        }

        // YaRN mscale: scale the rotated slice of q/k by attention_factor. This
        // is exactly the reference's cos/sin *= attention_factor (the mscale
        // vector is 1.0 on the pass-through tail). Broadcast [head_dim] over BHS.
        if (use_yarn) {
            if (self.yarn_mscale) |ms_f32| {
                // MUST match q/k's dtype: the f32 table would promote the
                // product, and that f32 then rides SDPA -> o_proj -> residual,
                // upcasting every downstream weight read for the rest of the
                // forward (measured 3x on Laguna XS).
                const ms = try constTableAs(ms_f32, mlx.mlx_array_dtype(q_rope), &self.yarn_mscale_cast, self.s);
                var q_scaled = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_multiply(&q_scaled, q_rope, ms, self.s));
                _ = mlx.mlx_array_free(q_rope);
                q_rope = q_scaled;
                var k_scaled = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_multiply(&k_scaled, k_rope, ms, self.s));
                _ = mlx.mlx_array_free(k_rope);
                k_rope = k_scaled;
            }
        }
    } // !fused_qk (composed chain)

    // V: proj → reshape → transpose
    if (!fused_qkv_used) {
        _ = mlx.mlx_array_free(v_proj);
        v_proj = try self.attnProj(x, fa.v_w, fa.v_s, fa.v_b, batch == 1 and !is_prefill, layer);
    }
    var v_r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v_r);
    try mlx.check(mlx.mlx_reshape(&v_r, v_proj, &kv_shape, 4, self.s));
    var v_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v_t);
    try mlx.check(mlx.mlx_transpose_axes(&v_t, v_r, &perm, 4, self.s));

    // KV cache: sliding layers read only the tail their queries reach
    // (`window + q_len - 1`), full layers keep everything. Laguna is
    // serial, so its multi-token forwards are prefill CHUNKS — at chunk
    // 2048 / window 512 that is where both the KV read and the mask
    // collapse. Every mask below is built at `sliding.kv_len` to match.
    // A batched tick (Flint): each slot appends to and attends over its own cache.
    var attn_out = if (ctx.batch_slots) |slots| try self.sinkAttnSlots(slots, q_rope, k_rope, v_t, fa, layer, seq_len, attn_scale) else mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn_out);
    const sliding = slidingViewFor(cfg, offset + seq_len, seq_len);
    const max_kv: u32 = if (is_full) 0 else sliding.span;
    var kv_view: DenseKVView = if (ctx.batch_slots == null) try ctx.cache.update(layer, k_rope, v_t, self.s, max_kv) else .{ .k = .{ .ctx = null }, .v = .{ .ctx = null }, .owned = false };
    defer kv_view.deinit();
    const full_k = kv_view.k;
    const full_v = kv_view.v;

    // SDPA with sliding-window masking (mirrors gemma4MoeAttnWith exactly,
    // minus the fused hd-256 kernel — Laguna's head_dim is 128).
    // Fused quant-KV opt-in (see forwardStandardWith's site for design
    // notes): resolve the SAME mode/mask the dense arms below would use,
    // then decode rides the packed-read kernel (kernel-or-DENSE — a
    // declined kernel leaves kv_fused_done false and the dense arms run).
    var kv_fused_done = ctx.batch_slots != null;
    // The packed-read kernels take no sinks (Flint): its layers read the dense view.
    if (!kv_fused_done and ctx.kv_attn_fused and fa.sinks.ctx == null and kvAttnFusedEligible(&kv_view, seq_len)) {
        var f_mode: []const u8 = "";
        var f_mask: mlx.mlx_array = none_mask;
        if (is_full) {
            if (is_prefill) f_mode = "causal";
        } else {
            const sw: c_int = @intCast(cfg.sliding_window);
            const total_kv: c_int = offset + seq_len;
            if (is_prefill and total_kv <= sw) {
                f_mode = "causal";
            } else if (is_prefill) {
                if (local_prefill_mask.ctx == null) {
                    local_prefill_mask.* = try self.createSlidingWindowMask(seq_len, sliding.kv_len, sw);
                }
                f_mode = "array";
                f_mask = local_prefill_mask.*;
            }
        }
        if (try qkvAttnDecodeKernel(self.s, q_rope, &kv_view, attn_scale, f_mode, f_mask)) |f| {
            logKvAttnFusedEngaged(&kv_view, q_rope, seq_len);
            _ = mlx.mlx_array_free(attn_out);
            attn_out = f;
            kv_fused_done = true;
        }
    }
    // Flint's sliding prefill: the tensor-unit kernel reads only the band, where an array-masked SDPA
    // walks every key block. Its full layers keep MLX's causal kernel, which is faster there.
    if (!kv_fused_done and is_prefill and !is_full and fa.sinks.ctx != null) {
        if (try @import("../../sdk/blocks/attention/nax_attention.zig").attention(self.s, q_rope, full_k, full_v, attn_scale, @intCast(cfg.sliding_window), fa.sinks)) |o| {
            _ = mlx.mlx_array_free(attn_out);
            attn_out = o;
            kv_fused_done = true;
        }
    }
    if (!kv_fused_done) {
        if (is_full) {
            if (is_prefill) {
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "causal", none_mask, fa.sinks, fa.sinks.ctx == null and sdpaForceFused(q_rope, full_k), self.s));
            } else {
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "", none_mask, fa.sinks, false, self.s));
            }
        } else {
            const sw: c_int = @intCast(cfg.sliding_window);
            const total_kv: c_int = offset + seq_len;
            if (is_prefill and total_kv <= sw) {
                // Window degenerates to plain causal — same kernel/reduction the
                // reference picks for short prompts.
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "causal", none_mask, fa.sinks, fa.sinks.ctx == null and sdpaForceFused(q_rope, full_k), self.s));
            } else if (is_prefill) {
                if (local_prefill_mask.ctx == null) {
                    local_prefill_mask.* = try self.createSlidingWindowMask(seq_len, sliding.kv_len, sw);
                }
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "array", local_prefill_mask.*, fa.sinks, false, self.s));
            } else {
                // A decode view is the window (`slidingTailSpan` at one row): every key it holds is visible.
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "", none_mask, fa.sinks, false, self.s));
            }
        }
    }

    // [B,H,S,D] → [B,S,H*D]
    var attn_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn_t);
    try mlx.check(mlx.mlx_transpose_axes(&attn_t, attn_out, &perm_back, 4, self.s));
    var attn_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn_flat);
    try mlx.check(mlx.mlx_reshape(&attn_flat, attn_t, &flat_shape, 3, self.s));

    // Softplus per-head output gate BEFORE o_proj (reference LagunaAttention):
    //   g = softplus(g_proj(x)) in fp32, cast to attn dtype (per-head scalar)
    //   attn = (attn.reshape[B,S,H,D] * g[...,None]).reshape[B,S,H*D]
    // softplus(x) = logaddexp(0, x) (numerically stable).
    if (cfg.laguna_attn_gate and fa.g_w.ctx != null) {
        const g_logits = try self.qmatmul(x, fa.g_w, fa.g_s, fa.g_b); // [B,S,h_count]
        defer _ = mlx.mlx_array_free(g_logits);
        if (try fusedAttnGate(self.s, attn_flat, g_logits, h_count, hd)) |gated| {
            defer _ = mlx.mlx_array_free(gated);
            return self.attnProj(gated, fa.o_w, fa.o_s, fa.o_b, batch == 1 and !is_prefill, layer);
        }
        var g_f32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(g_f32);
        try mlx.check(mlx.mlx_astype(&g_f32, g_logits, .float32, self.s));
        const zero = mlx.mlx_array_new_float(0.0);
        defer _ = mlx.mlx_array_free(zero);
        var g_sp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(g_sp);
        try mlx.check(mlx.mlx_logaddexp(&g_sp, zero, g_f32, self.s));
        // Cast gate to attn dtype (bf16), then per-head broadcast multiply.
        var g_bf16 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(g_bf16);
        try mlx.check(mlx.mlx_astype(&g_bf16, g_sp, .bfloat16, self.s));
        const g_shape = [_]c_int{ batch, seq_len, h_count, 1 };
        var g_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(g_r);
        try mlx.check(mlx.mlx_reshape(&g_r, g_bf16, &g_shape, 4, self.s));
        var attn_4d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_4d);
        try mlx.check(mlx.mlx_reshape(&attn_4d, attn_flat, &q_shape, 4, self.s));
        var gated_4d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gated_4d);
        try mlx.check(mlx.mlx_multiply(&gated_4d, attn_4d, g_r, self.s));
        var gated_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gated_flat);
        try mlx.check(mlx.mlx_reshape(&gated_flat, gated_4d, &flat_shape, 3, self.s));
        return self.attnProj(gated_flat, fa.o_w, fa.o_s, fa.o_b, batch == 1 and !is_prefill, layer);
    }

    return self.attnProj(attn_flat, fa.o_w, fa.o_s, fa.o_b, batch == 1 and !is_prefill, layer);
}

// ── tests ──

test "lagunaRouterBase resolves both the flat mlx-lm gate and the HF-native gate.proj naming" {
    // The poolside NVFP4-mlx build flattens the router module to
    // `mlp.gate.weight`; mlx-community's oQ conversions (Laguna-S-2.1-oQ4e-fast,
    // -oQ5e) keep the HF module layout, so the SAME [E, hidden] tensor lands at
    // `mlp.gate.proj.weight` and the load died with MISSING WEIGHT
    // (issue #169). `mlp.gate.e_score_correction_bias` is a buffer on the gate
    // module in both, so it never moves.
    const allocator = testing.allocator;
    var buf: [256]u8 = undefined;
    const put = struct {
        fn add(w: *Weights, alloc: std.mem.Allocator, key: []const u8) !void {
            const k = try alloc.dupe(u8, key);
            try w.map.put(k, mlx.mlx_array_new());
        }
    }.add;

    // oQ build: only the nested projection exists (was the crash).
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "language_model.model.layers.1.mlp.gate.proj.weight");
        try testing.expectEqualStrings(
            "mlp.gate.proj",
            lagunaRouterBase(&w, &buf, "language_model.model", 1),
        );
    }
    // poolside build: the flat name is preferred, so it binds byte-identically.
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "model.layers.1.mlp.gate.weight");
        try testing.expectEqualStrings("mlp.gate", lagunaRouterBase(&w, &buf, "model", 1));
    }
    // Neither present: keep the flat name so the error names the canonical key.
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try testing.expectEqualStrings("mlp.gate", lagunaRouterBase(&w, &buf, "model", 1));
    }
}

test "Laguna-XS decode µbench: bf16 attention projections (MLX_SERVE_LAGUNA_UBENCH=1)" {
    // Laguna XS 2.1 NVFP4 keeps attention, lm_head and the layer-0 MLP in BF16
    // and quantizes only the expert projections, so ~78% of the per-token DRAM
    // budget is DENSE bf16 GEMV — not the MoE. This measures the achievable
    // rate for that dominant term at the real shapes.
    //
    // CRITICAL: every layer gets its OWN weight buffers. An earlier version of
    // this bench reused four buffers across all 40 layers and reported
    // "3.26 GB at 449 GB/s"; it was re-reading ~100 MB of resident data 25x and
    // measuring CACHE bandwidth, not DRAM. A decode step reads each of the 40
    // layers' weights exactly ONCE from DRAM, so the bench must too — otherwise
    // it manufactures a bandwidth floor far below the real one and makes the
    // live path look pathological by comparison.
    if (std.c.getenv("MLX_SERVE_LAGUNA_UBENCH") == null) return error.SkipZigTest;
    const io_util = @import("../../sdk/util/io_util.zig");
    const tio = testing.io;
    const s = mlx.gpuStream();
    const allocator = testing.allocator;

    const HID: c_int = 2048;
    const SLIDING_LAYERS: usize = 30; // 64 Q heads
    const FULL_LAYERS: usize = 10; //    48 Q heads
    const VOCAB: c_int = 100352;

    // bf16 weight [out, in], pre-transposed to [in, out] exactly like
    // `maybeTransposeForBf16` does at load — the live path's layout.
    const makeW = struct {
        fn go(a: std.mem.Allocator, out: c_int, in: c_int, str: mlx.mlx_stream, rndp: *std.Random.DefaultPrng) !mlx.mlx_array {
            const cnt: usize = @intCast(out * in);
            const buf = try a.alloc(f32, cnt);
            defer a.free(buf);
            const rr = rndp.random();
            for (buf) |*v| v.* = rr.float(f32) - 0.5;
            const sh = [_]c_int{ out, in };
            const w32 = mlx.mlx_array_new_data(buf.ptr, &sh, 2, .float32);
            defer _ = mlx.mlx_array_free(w32);
            var wb = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wb);
            try mlx.check(mlx.mlx_astype(&wb, w32, .bfloat16, str));
            const t = try transposeBf16Weight(wb, str);
            // Materialize so the view's parent can be freed and the layer owns
            // one contiguous buffer, matching a loaded checkpoint's residency.
            try mlx.check(mlx.mlx_array_eval(t));
            return t;
        }
    }.go;

    var prng = std.Random.DefaultPrng.init(0xA5A5);
    const NL = SLIDING_LAYERS + FULL_LAYERS;
    const qs = try allocator.alloc(mlx.mlx_array, NL);
    defer allocator.free(qs);
    const os_ = try allocator.alloc(mlx.mlx_array, NL);
    defer allocator.free(os_);
    const ks = try allocator.alloc(mlx.mlx_array, NL);
    defer allocator.free(ks);
    const vs = try allocator.alloc(mlx.mlx_array, NL);
    defer allocator.free(vs);
    defer for (0..NL) |i| {
        _ = mlx.mlx_array_free(qs[i]);
        _ = mlx.mlx_array_free(os_[i]);
        _ = mlx.mlx_array_free(ks[i]);
        _ = mlx.mlx_array_free(vs[i]);
    };
    for (0..NL) |i| {
        const heads: c_int = if (i < SLIDING_LAYERS) 64 else 48;
        qs[i] = try makeW(allocator, heads * 128, HID, s, &prng);
        os_[i] = try makeW(allocator, HID, heads * 128, s, &prng);
        ks[i] = try makeW(allocator, 8 * 128, HID, s, &prng);
        vs[i] = try makeW(allocator, 8 * 128, HID, s, &prng);
    }
    const lm = try makeW(allocator, VOCAB, HID, s, &prng);
    defer _ = mlx.mlx_array_free(lm);

    const xsh = [_]c_int{ 1, 1, HID };
    const xbuf = try allocator.alloc(f32, @intCast(HID));
    defer allocator.free(xbuf);
    for (xbuf) |*v| v.* = prng.random().float(f32) - 0.5;
    const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xsh, 3, .float32);
    defer _ = mlx.mlx_array_free(x32);
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));

    var bytes_total: f64 = 0;
    for (0..NL) |i| {
        const heads: f64 = if (i < SLIDING_LAYERS) 64 else 48;
        bytes_total += heads * 128 * 2048 * 2 * 2; // q + o
        bytes_total += 8 * 128 * 2048 * 2 * 2; // k + v
    }
    bytes_total += @as(f64, VOCAB) * 2048 * 2;

    const runToken = struct {
        fn go(str: mlx.mlx_stream, x0: mlx.mlx_array, q: []mlx.mlx_array, o: []mlx.mlx_array, k: []mlx.mlx_array, v: []mlx.mlx_array, l: mlx.mlx_array) !void {
            var h = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&h, x0, x0, str));
            defer _ = mlx.mlx_array_free(h);
            for (0..q.len) |i| {
                const qp = try mm(str, h, q[i]);
                defer _ = mlx.mlx_array_free(qp);
                const kp = try mm(str, h, k[i]);
                defer _ = mlx.mlx_array_free(kp);
                const vp = try mm(str, h, v[i]);
                defer _ = mlx.mlx_array_free(vp);
                const op = try mm(str, qp, o[i]);
                defer _ = mlx.mlx_array_free(op);
                // Fold k/v into the residual so MLX cannot drop them as dead
                // code, and rotate h so no layer is a common subexpression.
                var kv_sum = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(kv_sum);
                try mlx.check(mlx.mlx_add(&kv_sum, kp, vp, str));
                var kv_sc = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(kv_sc);
                try mlx.check(mlx.mlx_sum_axis(&kv_sc, kv_sum, 2, true, str));
                var t = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&t, h, op, str));
                var t2 = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&t2, t, kv_sc, str));
                _ = mlx.mlx_array_free(t);
                _ = mlx.mlx_array_free(h);
                h = t2;
            }
            const logits = try mm(str, h, l);
            defer _ = mlx.mlx_array_free(logits);
            try mlx.check(mlx.mlx_array_eval(logits));
        }
        fn mm(str: mlx.mlx_stream, a: mlx.mlx_array, w: mlx.mlx_array) !mlx.mlx_array {
            var r = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_matmul(&r, a, w, str));
            return r;
        }
    }.go;

    for (0..3) |_| try runToken(s, x, qs, os_, ks, vs, lm);
    const ITER: usize = 20;
    var sw = io_util.Stopwatch.init(tio);
    for (0..ITER) |_| try runToken(s, x, qs, os_, ks, vs, lm);
    const ms = @as(f64, @floatFromInt(sw.read())) / 1.0e6 / @as(f64, @floatFromInt(ITER));
    const gbps = bytes_total / (ms / 1000.0) / 1.0e9;
    std.debug.print("\n[laguna-ubench] 40 layers q/k/v/o + lm_head, DISTINCT weights per layer: {d:.3} ms/token ({d:.2} GB, {d:.0} GB/s)\n", .{ ms, bytes_total / 1.0e9, gbps });

    // Per-dispatch latency: a chain of N TRIVIAL dependent ops on a 1-element
    // array. No bytes move, so whatever this costs is pure per-op overhead —
    // the currency a decode step actually spends, since at seq_len 1 almost
    // every op in the model is this small.
    {
        const one_sh = [_]c_int{1};
        const oneval = [_]f32{1.0};
        const seed = mlx.mlx_array_new_data(&oneval, &one_sh, 1, .float32);
        defer _ = mlx.mlx_array_free(seed);
        for ([_]usize{ 200, 1400 }) |chain| {
            const chainOnce = struct {
                fn go(str: mlx.mlx_stream, sd: mlx.mlx_array, n: usize) !void {
                    var a = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_add(&a, sd, sd, str));
                    defer _ = mlx.mlx_array_free(a);
                    for (0..n) |_| {
                        var b = mlx.mlx_array_new();
                        try mlx.check(mlx.mlx_add(&b, a, sd, str));
                        _ = mlx.mlx_array_free(a);
                        a = b;
                    }
                    try mlx.check(mlx.mlx_array_eval(a));
                }
            }.go;
            for (0..2) |_| try chainOnce(s, seed, chain);
            var sw2 = io_util.Stopwatch.init(tio);
            const IT: usize = 10;
            for (0..IT) |_| try chainOnce(s, seed, chain);
            const tms = @as(f64, @floatFromInt(sw2.read())) / 1.0e6 / @as(f64, @floatFromInt(IT));
            std.debug.print("[laguna-ubench] {d} trivial DEPENDENT ops: {d:.3} ms => {d:.1} us/dispatch\n", .{ chain, tms, tms * 1000.0 / @as(f64, @floatFromInt(chain)) });
        }
    }
}
