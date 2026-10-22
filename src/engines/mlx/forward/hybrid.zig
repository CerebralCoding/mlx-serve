//! engines/mlx/forward/hybrid, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns HybridMlpWeights, HybridOp, HybridLayerWeights, forwardHybridWith, testHybridXfm, deinitTestHybridXfm.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const mamba2_decode = @import("../../../sdk/blocks/ssm/mamba2_decode.zig");
const add_norm = @import("../../../sdk/blocks/norms/add_norm.zig");
const ModelConfig = @import("../model.zig").ModelConfig;
const Weights = @import("../model.zig").Weights;
const checkpoint = @import("../ssm/checkpoint.zig");
const residual_norm = @import("../../../sdk/blocks/norms/residual_norm.zig");
const GatedConvWeights = @import("../../../plugins/lfm2/gated_conv.zig").GatedConvWeights;
const NemotronMoeWeights = @import("../../../plugins/nemotron_h/moe.zig").NemotronMoeWeights;
const moeStreamPadPays = @import("../../../plugins/nemotron_h/moe.zig").moeStreamPadPays;
const nemotronMoe = @import("../../../plugins/nemotron_h/moe.zig").nemotronMoe;
const nemotronMoeCombine = @import("../../../plugins/nemotron_h/moe.zig").nemotronMoeCombine;
const nemotronMoeExperts = @import("../../../plugins/nemotron_h/moe.zig").nemotronMoeExperts;
const nemotronSharedExpert = @import("../../../plugins/nemotron_h/moe.zig").nemotronSharedExpert;
const DenseMlpWeights = @import("../../../sdk/blocks/mlp/dense.zig").DenseMlpWeights;
const SimpleMlpWeights = @import("../../../sdk/blocks/mlp/dense.zig").SimpleMlpWeights;
const MoeMlpWeights = @import("../../../sdk/blocks/moe/weights.zig").MoeMlpWeights;
const Mamba2Weights = @import("../../../sdk/blocks/ssm/mamba2.zig").Mamba2Weights;
const captureHidden = @import("../capture.zig").captureHidden;
const PREFILL_EVAL_CADENCE_DEFAULT = @import("../eval_cadence.zig").PREFILL_EVAL_CADENCE_DEFAULT;
const evalCadencePoint = @import("../eval_cadence.zig").evalCadencePoint;
const prefillEvalCadenceApplies = @import("../eval_cadence.zig").prefillEvalCadenceApplies;
const prefillEvalCadenceFor = @import("../eval_cadence.zig").prefillEvalCadenceFor;
const ForwardCtx = @import("../forward_ctx.zig").ForwardCtx;
const KVCache = @import("../kv/cache.zig").KVCache;
const FullAttnWeights = @import("../layer_weights.zig").FullAttnWeights;
const initHybridLayers = @import("../load/layers_hybrid.zig").initHybridLayers;
const SSMCacheEntry = @import("../ssm/entry.zig").SSMCacheEntry;
const ssmFreeSpecCapture = @import("../ssm/entry.zig").ssmFreeSpecCapture;
const Transformer = @import("../transformer.zig").Transformer;

pub const HybridMlpWeights = union(enum) {
    dense: DenseMlpWeights,
    moe: MoeMlpWeights,
};

const HybridOp = union(enum) {
    gated_conv: GatedConvWeights,
    full_attn: FullAttnWeights,
    mamba2: Mamba2Weights,
    dense_mlp: DenseMlpWeights, // gated MLP (SwiGLU)
    simple_mlp: SimpleMlpWeights, // ungated MLP (ReLU^2)
    nemotron_moe: NemotronMoeWeights,
};

pub const HybridLayerWeights = struct {
    input_norm: mlx.mlx_array,
    post_norm: ?mlx.mlx_array, // null for single-op blocks (Nemotron-H)
    op: HybridOp,
    mlp: ?DenseMlpWeights, // optional MLP after mixer (LFM2: always; Nemotron-H: null)
    /// lfm2_moe: layers at/after `num_dense_layers` replace the dense MLP
    /// with a sparse MoE block. Exactly one of `mlp` / `moe_mlp` is set.
    moe_mlp: ?MoeMlpWeights = null,
};

// ── Hybrid forward pass (LFM2, Nemotron-H) ──

pub fn forwardHybridWith(self: *Transformer, ctx: *ForwardCtx, token_ids: mlx.mlx_array) !mlx.mlx_array {
    const hl = self.hybrid_layers.?;
    const offset = ctx.moe_seq_offset.*;
    const cfg = &self.config;

    // Spec-decode: thread the per-position conv capture down to the
    // gated-conv / mamba2 mixers (they don't take the ctx). Without it a
    // partial accept leaves the conv state advanced past the accepted
    // position and the next token is generated from a future it never had.
    self.spec_capture_ssm = ctx.capture_ssm_seq;
    defer self.spec_capture_ssm = false;

    var h = try self.embedding(token_ids);

    // Splice vision embeddings at image_token_id positions (prefill only).
    // LFM2-VL is the first hybrid arch with a tower; without this the image
    // rows never enter the sequence and the model answers from placeholder
    // embeddings, which looks like a blank picture rather than a bug.
    h = try self.applyVisionEmbeddingsWith(ctx, h, token_ids);

    const x_shape = mlx.getShape(h);
    const batch: c_int = x_shape[0];
    const seq_len: c_int = x_shape[1];

    // Eval cadence: drop to per-layer when this chunk's score/dequant
    // transients are large — see prefillEvalCadence. LFM2/Nemotron-H ride
    // fused head dims, but a quantized KV cache's dense rebuild (and a
    // future head_dim-256 hybrid like qwen3_next) still counts.
    const hybrid_eval_cadence = prefillEvalCadenceFor(
        PREFILL_EVAL_CADENCE_DEFAULT,
        cfg,
        @intCast(seq_len),
        @as(u64, @intCast(offset)) + @as(u64, @intCast(seq_len)),
        ctx.cache.config.scheme != .off,
    );

    var dt = mlx.DtypeTrace.begin("hybrid", h, null);

    // Short windows: the residual add and the NEXT block's norm are one
    // dispatch; an MLP-less layer (Nemotron-H) goes through add_norm.zig
    // with the MoE combine folded in, an LFM layer through residualTail on
    // both of its seams. `pending_normed` carries that norm into the next
    // iteration (or the final norm).
    const eps_arr = mlx.mlx_array_new_float(cfg.rms_norm_eps);
    defer _ = mlx.mlx_array_free(eps_arr);
    const fuse_add_norm = batch * seq_len <= residual_norm.add_norm_max_rows and add_norm.eligible(@intCast(cfg.hidden_size), mlx.mlx_array_dtype(h), self.s);
    const fuse_seams = batch * seq_len == 1;
    var pending_normed: ?mlx.mlx_array = null;
    defer if (pending_normed) |pn| {
        _ = mlx.mlx_array_free(pn);
    };

    for (0..cfg.num_hidden_layers) |layer_idx| {
        const li: u32 = @intCast(layer_idx);
        const lw = &hl[layer_idx];

        const normed = if (pending_normed) |pn| blk: {
            pending_normed = null;
            break :blk pn;
        } else try self.rmsNorm(h, lw.input_norm);
        defer _ = mlx.mlx_array_free(normed);

        const next_norm_w: ?mlx.mlx_array = if (layer_idx + 1 < cfg.num_hidden_layers) hl[layer_idx + 1].input_norm else if (cfg.has_final_norm) self.final_norm else null;
        const has_mlp = lw.mlp != null or lw.moe_mlp != null;
        const fuse_here = fuse_add_norm and next_norm_w != null and !has_mlp;

        var fused_out: ?add_norm.Out = null;
        // The fused combine has no latent projection. Puzzle layers take nemotronMoe.
        if (fuse_here and lw.op == .nemotron_moe and lw.op.nemotron_moe.fc1_latent_w.ctx == null and batch * seq_len <= residual_norm.add_norm_max_rows) {
            const parts = try nemotronMoeExperts(normed, &lw.op.nemotron_moe, cfg, moeStreamPadPays(&lw.op.nemotron_moe), self.s);
            defer parts.deinit();
            const shared = try nemotronSharedExpert(normed, &lw.op.nemotron_moe, cfg, self.s);
            defer if (shared) |sx| {
                _ = mlx.mlx_array_free(sx);
            };
            fused_out = try add_norm.moeCombineAddNorm(h, parts.y_tkh, parts.scores, shared, next_norm_w.?, eps_arr, self.s);
            if (fused_out == null) {
                const op_out = try nemotronMoeCombine(normed, &parts, &lw.op.nemotron_moe, cfg, self.s, normed);
                defer _ = mlx.mlx_array_free(op_out);
                var h_new = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&h_new, h, op_out, self.s));
                _ = mlx.mlx_array_free(h);
                h = h_new;
            }
        } else {
            // Primary operation
            const op_out = switch (lw.op) {
                .gated_conv => |cw| try self.gatedConv(normed, &cw, &ctx.ssm_entries.?[layer_idx], batch, seq_len),
                .full_attn => |fa| try self.hybridAttnWith(ctx, normed, &fa, li, @intCast(offset), batch, seq_len, seq_len > 1),
                .mamba2 => |mw| try self.mamba2Mixer(normed, &mw, &ctx.ssm_entries.?[layer_idx], batch, seq_len),
                .dense_mlp => |dw| try self.denseMLP(normed, &dw),
                .simple_mlp => |sw| try self.simpleMLP(normed, &sw),
                .nemotron_moe => |nm| try nemotronMoe(normed, &nm, cfg, self.s),
            };
            defer _ = mlx.mlx_array_free(op_out);
            if (has_mlp) {
                // LFM2: residual + the MLP's input norm, the MLP, then
                // residual + the next layer's input norm. lfm2_moe swaps
                // the dense block for a sparse one past `num_dense_layers`;
                // both sit under the same ffn_norm.
                const none = mlx.mlx_array{ .ctx = null };
                const ff_normed = try self.residualTail(&h, op_out, none, false, none, lw.post_norm.?, fuse_seams);
                defer _ = mlx.mlx_array_free(ff_normed);
                const mlp_out = if (lw.moe_mlp) |*mw|
                    try self.moeMLP(ff_normed, mw)
                else
                    try self.denseMLP(ff_normed, &lw.mlp.?);
                defer _ = mlx.mlx_array_free(mlp_out);
                const carried = try self.residualTail(&h, mlp_out, none, false, none, if (fuse_seams) (next_norm_w orelse none) else none, fuse_seams);
                if (carried.ctx != null) pending_normed = carried;
            } else {
                if (fuse_here) fused_out = try add_norm.addNorm(h, op_out, next_norm_w.?, eps_arr, self.s);
                if (fused_out == null) {
                    // Residual connection
                    var h_new = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_add(&h_new, h, op_out, self.s));
                    _ = mlx.mlx_array_free(h);
                    h = h_new;
                }
            }
        }
        if (fused_out) |fo| {
            _ = mlx.mlx_array_free(h);
            h = fo.h;
            pending_normed = fo.normed;
        }

        // DFlash/DSpark capture: this h IS `hidden_states[li+1]`.
        if (ctx.capture_layers) |cl| {
            for (cl.ids, cl.out) |cid, *slot| {
                if (cid == li) _ = mlx.mlx_array_set(slot, h);
            }
        }

        if (prefillEvalCadenceApplies(seq_len) and ((layer_idx + 1) % hybrid_eval_cadence == 0 or layer_idx + 1 == @as(usize, cfg.num_hidden_layers))) {
            try evalCadencePoint(h, ctx.ssm_entries);
        } else if (ctx.pipeline_build > 0 and (layer_idx + 1) % ctx.pipeline_build == 0 and layer_idx + 1 < cfg.num_hidden_layers) {
            const v = mlx.mlx_vector_array_new_value(h);
            defer _ = mlx.mlx_vector_array_free(v);
            try mlx.check(mlx.mlx_async_eval(v));
        }
        self.ladderStep(h, layer_idx, seq_len);
        dt.layer(h, layer_idx);
    }
    dt.end(h);

    ctx.moe_seq_offset.* += @intCast(seq_len);

    // Final norm (absent for LFM2)
    if (cfg.has_final_norm) {
        const final_normed = if (pending_normed) |pn| blk: {
            pending_normed = null;
            break :blk pn;
        } else try self.rmsNorm(h, self.final_norm);
        // Spec-decode capture: the Nemotron-H MTP head reads the
        // final-normed hidden (then its own hnorm); LFM2 the residual.
        try captureHidden(ctx, if (self.config.mtpReadsFinalNorm()) final_normed else h, self.s);
        _ = mlx.mlx_array_free(h);
        if (self.embedding_mode or ctx.skip_lm_head) return final_normed;
        const logits = try self.lmHeadProject(final_normed);
        _ = mlx.mlx_array_free(final_normed);
        return logits;
    } else {
        try captureHidden(ctx, h, self.s);
        if (self.embedding_mode or ctx.skip_lm_head) return h;
        const logits = try self.lmHeadProject(h);
        _ = mlx.mlx_array_free(h);
        return logits;
    }
}

// ── tests ──

// Bar: a decode step through the fused residual+norm path (plain arm on the
// Mamba2 block, MoE-combine arm with the shared expert on the MoE block, the
// final norm consuming the carried norm) is BIT-EQUAL to the same step with
// the fusion off. The unit kernels alone missed a fused path that dropped the
// shared expert.
test "hybrid decode: the fused add+norm path equals the unfused blocks (Nemotron-H)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const t = std.testing;
    const allocator = t.allocator;
    const s = mlx.gpuStream();
    const H: c_int = 64;
    const heads: c_int = 4;
    const head_dim: c_int = 32;
    const groups: c_int = 2;
    const state: c_int = 32;
    const kernel: c_int = 4;
    const d_inner = heads * head_dim;
    const conv_dim = d_inner + 2 * groups * state;
    const E: c_int = 8;
    const K: c_int = 3;
    const I: c_int = 64;
    const SI: c_int = 128;

    var prng = std.Random.DefaultPrng.init(0x6e656d6f);
    const rnd = prng.random();
    var w = Weights.init(allocator);
    defer w.deinit();
    const P = struct {
        fn dense(weights: *Weights, alloc: std.mem.Allocator, r: std.Random, name: []const u8, shape: []const c_int, scale: f32, offset: f32, dt: mlx.mlx_dtype, st: mlx.mlx_stream) !void {
            var n: usize = 1;
            for (shape) |d| n *= @intCast(d);
            const host = try alloc.alloc(f32, n);
            defer alloc.free(host);
            for (host) |*v| v.* = offset + scale * (r.float(f32) - 0.5);
            const f = mlx.mlx_array_new_data(host.ptr, shape.ptr, @intCast(shape.len), .float32);
            defer _ = mlx.mlx_array_free(f);
            var arr = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&arr, f, dt, st));
            try mlx.check(mlx.mlx_array_eval(arr));
            try weights.map.put(try alloc.dupe(u8, name), arr);
        }
        // 4-bit/gs64 affine: `<name>.weight/.scales/.biases`.
        fn quant(weights: *Weights, alloc: std.mem.Allocator, r: std.Random, name: []const u8, shape: []const c_int, st: mlx.mlx_stream) !void {
            var n: usize = 1;
            for (shape) |d| n *= @intCast(d);
            const host = try alloc.alloc(f32, n);
            defer alloc.free(host);
            for (host) |*v| v.* = (r.float(f32) - 0.5) * 0.5;
            const f = mlx.mlx_array_new_data(host.ptr, shape.ptr, @intCast(shape.len), .float32);
            defer _ = mlx.mlx_array_free(f);
            var triple = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(triple);
            try mlx.check(mlx.mlx_quantize(&triple, f, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{}, st));
            const suffixes = [_][]const u8{ ".weight", ".scales", ".biases" };
            for (suffixes, 0..) |suf, i| {
                var arr = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_vector_array_get(&arr, triple, i));
                try mlx.check(mlx.mlx_array_eval(arr));
                const full = try std.fmt.allocPrint(alloc, "{s}{s}", .{ name, suf });
                try weights.map.put(full, arr);
            }
        }
    };
    // Layer 0: Mamba2. Layer 1: MoE with a shared expert. Final norm.
    try P.dense(&w, allocator, rnd, "model.layers.0.norm.weight", &.{H}, 0.2, 1.0, .bfloat16, s);
    try P.dense(&w, allocator, rnd, "model.layers.0.mixer.in_proj.weight", &.{ d_inner + conv_dim + heads, H }, 0.1, 0.0, .bfloat16, s);
    try P.dense(&w, allocator, rnd, "model.layers.0.mixer.conv1d.weight", &.{ conv_dim, kernel, 1 }, 0.6, 0.1, .bfloat16, s);
    try P.dense(&w, allocator, rnd, "model.layers.0.mixer.conv1d.bias", &.{conv_dim}, 0.2, 0.0, .bfloat16, s);
    try P.dense(&w, allocator, rnd, "model.layers.0.mixer.A_log", &.{heads}, 1.0, 0.3, .bfloat16, s);
    try P.dense(&w, allocator, rnd, "model.layers.0.mixer.D", &.{heads}, 1.0, 1.0, .bfloat16, s);
    try P.dense(&w, allocator, rnd, "model.layers.0.mixer.dt_bias", &.{heads}, 1.0, 0.0, .bfloat16, s);
    try P.dense(&w, allocator, rnd, "model.layers.0.mixer.norm.weight", &.{d_inner}, 0.2, 1.0, .bfloat16, s);
    try P.dense(&w, allocator, rnd, "model.layers.0.mixer.out_proj.weight", &.{ H, d_inner }, 0.1, 0.0, .bfloat16, s);
    try P.dense(&w, allocator, rnd, "model.layers.1.norm.weight", &.{H}, 0.2, 1.0, .bfloat16, s);
    try P.dense(&w, allocator, rnd, "model.layers.1.mixer.gate.weight", &.{ E, H }, 0.5, 0.0, .bfloat16, s);
    try P.dense(&w, allocator, rnd, "model.layers.1.mixer.gate.e_score_correction_bias", &.{E}, 0.2, 0.0, .float32, s);
    try P.quant(&w, allocator, rnd, "model.layers.1.mixer.switch_mlp.fc1", &.{ E, I, H }, s);
    try P.quant(&w, allocator, rnd, "model.layers.1.mixer.switch_mlp.fc2", &.{ E, H, I }, s);
    try P.quant(&w, allocator, rnd, "model.layers.1.mixer.shared_experts.up_proj", &.{ SI, H }, s);
    try P.quant(&w, allocator, rnd, "model.layers.1.mixer.shared_experts.down_proj", &.{ H, SI }, s);

    var config = ModelConfig{ .model_type = "nemotron_h", .weight_prefix = "model" };
    config.has_hybrid_layers = true;
    config.num_hidden_layers = 2;
    config.hidden_size = @intCast(H);
    config.quant_bits = 4;
    config.quant_group_size = 64;
    config.has_final_norm = true;
    config.head_dim = 32;
    config.num_attention_heads = 1;
    config.num_key_value_heads = 1;
    config.vocab_size = 8;
    config.mamba_num_heads = @intCast(heads);
    config.mamba_head_dim = @intCast(head_dim);
    config.mamba_n_groups = @intCast(groups);
    config.ssm_state_size = @intCast(state);
    config.mamba_conv_kernel = @intCast(kernel);
    config.time_step_min = 0.001;
    config.time_step_max = 0.1;
    config.num_experts = @intCast(E);
    config.num_experts_per_tok = @intCast(K);
    config.moe_intermediate_size = @intCast(I);
    config.router_scaling_factor = 2.5;
    config.layer_block_types[0] = .mamba2;
    config.layer_block_types[1] = .moe;

    var name_buf: [256]u8 = undefined;
    const hl = try initHybridLayers(allocator, config, &w, &name_buf, s);
    defer {
        allocator.free(hl.hybrid_layers);
        for (hl.ssm_entries) |*e| {
            _ = mlx.mlx_array_free(e.conv_state);
            _ = mlx.mlx_array_free(e.ssm_state);
            ssmFreeSpecCapture(e);
        }
        allocator.free(hl.ssm_entries);
        for (hl.owned_bf16) |a| _ = mlx.mlx_array_free(a);
        allocator.free(hl.owned_bf16);
    }
    var emb_w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(emb_w);
    {
        var host: [8 * 64]f32 = undefined;
        for (&host, 0..) |*v, i| v.* = @sin(0.53 * @as(f32, @floatFromInt(i)));
        const f = mlx.mlx_array_new_data(&host, &[_]c_int{ 8, H }, 2, .float32);
        defer _ = mlx.mlx_array_free(f);
        try mlx.check(mlx.mlx_astype(&emb_w, f, .bfloat16, s));
    }
    var cache = try KVCache.init(allocator, 2);
    defer cache.deinit();
    var xfm = testHybridXfm(allocator, config, cache, emb_w, hl.ssm_entries, hl.hybrid_layers, s);
    defer deinitTestHybridXfm(&xfm);
    {
        var host: [64]f32 = undefined;
        for (&host, 0..) |*v, i| v.* = 1.0 + 0.3 * @sin(0.7 * @as(f32, @floatFromInt(i)));
        const f = mlx.mlx_array_new_data(&host, &[_]c_int{H}, 1, .float32);
        defer _ = mlx.mlx_array_free(f);
        _ = mlx.mlx_array_free(xfm.final_norm);
        xfm.final_norm = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&xfm.final_norm, f, .bfloat16, s));
    }

    const tokens = [_]i32{ 1, 5, 2 };
    const Run = struct {
        fn f(x: *Transformer, entries: []SSMCacheEntry, ids: []const i32, out: []f32, st: mlx.mlx_stream) !void {
            for (entries) |*e| {
                _ = mlx.mlx_array_free(e.conv_state);
                _ = mlx.mlx_array_free(e.ssm_state);
                ssmFreeSpecCapture(e);
                e.conv_state = .{ .ctx = null };
                e.ssm_state = .{ .ctx = null };
                e.initialized = false;
            }
            x.moe_seq_offset = 0;
            for (ids, 0..) |tok, i| {
                const id_arr = mlx.mlx_array_new_data(&[_]i32{tok}, &[_]c_int{ 1, 1 }, 2, .int32);
                defer _ = mlx.mlx_array_free(id_arr);
                const o = try x.forward(id_arr);
                defer _ = mlx.mlx_array_free(o);
                var f32a = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(f32a);
                try mlx.check(mlx.mlx_astype(&f32a, o, .float32, st));
                try mlx.check(mlx.mlx_array_eval(f32a));
                const p = mlx.mlx_array_data_float32(f32a) orelse return error.NoData;
                @memcpy(out[i * 64 .. (i + 1) * 64], p[0..64]);
            }
        }
    }.f;
    var fused: [3 * 64]f32 = undefined;
    try Run(&xfm, hl.ssm_entries, &tokens, &fused, s);
    var plain: [3 * 64]f32 = undefined;
    {
        residual_norm.add_norm_max_rows = 0;
        defer residual_norm.add_norm_max_rows = mamba2_decode.MAX_ROWS;
        try Run(&xfm, hl.ssm_entries, &tokens, &plain, s);
    }
    var max_ref: f32 = 0;
    var max_err: f32 = 0;
    for (fused, plain) |a, b| {
        try t.expect(std.math.isFinite(a) and std.math.isFinite(b));
        max_ref = @max(max_ref, @abs(a));
        max_err = @max(max_err, @abs(a - b));
    }
    if (!(max_ref > 0.01)) return error.RefTooSmall;
    if (max_err != 0) {
        std.debug.print("hybrid fused add+norm vs plain: max_err={d:.6} max_ref={d:.6}\n", .{ max_err, max_ref });
        return error.FusedPathMismatch;
    }
}

// Bar: a NoPE attention layer's output for the last token does not move when
// the earlier tokens are reordered; a rotated key would.
test "hybrid attention: a NoPE layer ignores the order of earlier tokens (Nemotron-H)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const t = std.testing;
    const allocator = t.allocator;
    const s = mlx.gpuStream();
    const H: c_int = 64;
    const heads: c_int = 2;
    const hd: c_int = 32;

    var prng = std.Random.DefaultPrng.init(0x6e6f7065);
    const rnd = prng.random();
    var w = Weights.init(allocator);
    defer w.deinit();
    const names = [_][]const u8{ "model.layers.0.mixer.q_proj.weight", "model.layers.0.mixer.k_proj.weight", "model.layers.0.mixer.v_proj.weight", "model.layers.0.mixer.o_proj.weight", "model.layers.0.norm.weight" };
    const shapes = [_][2]c_int{ .{ heads * hd, H }, .{ hd, H }, .{ hd, H }, .{ H, heads * hd }, .{ H, 1 } };
    for (names, shapes) |name, shp| {
        const n: usize = @intCast(shp[0] * shp[1]);
        const host = try allocator.alloc(f32, n);
        defer allocator.free(host);
        const is_norm = shp[1] == 1;
        for (host) |*v| v.* = if (is_norm) 1.0 else (rnd.float(f32) - 0.5) * 0.6;
        const f = if (is_norm) mlx.mlx_array_new_data(host.ptr, &[_]c_int{shp[0]}, 1, .float32) else mlx.mlx_array_new_data(host.ptr, &shp, 2, .float32);
        defer _ = mlx.mlx_array_free(f);
        var arr = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&arr, f, .bfloat16, s));
        try mlx.check(mlx.mlx_array_eval(arr));
        try w.map.put(try allocator.dupe(u8, name), arr);
    }

    var config = ModelConfig{ .model_type = "nemotron_h", .weight_prefix = "model" };
    config.has_hybrid_layers = true;
    config.num_hidden_layers = 1;
    config.hidden_size = @intCast(H);
    config.head_dim = @intCast(hd);
    config.num_attention_heads = @intCast(heads);
    config.num_key_value_heads = 1;
    config.vocab_size = 8;
    config.has_final_norm = false;
    config.rope_theta = 10000;
    config.layer_block_types[0] = .attention;
    config.layer_no_rope = @splat(true);

    var name_buf: [256]u8 = undefined;
    const hl = try initHybridLayers(allocator, config, &w, &name_buf, s);
    defer {
        allocator.free(hl.hybrid_layers);
        allocator.free(hl.ssm_entries);
        for (hl.owned_bf16) |a| _ = mlx.mlx_array_free(a);
        allocator.free(hl.owned_bf16);
    }
    var emb_w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(emb_w);
    {
        var host: [8 * 64]f32 = undefined;
        for (&host, 0..) |*v, i| v.* = @sin(0.53 * @as(f32, @floatFromInt(i)));
        const f = mlx.mlx_array_new_data(&host, &[_]c_int{ 8, H }, 2, .float32);
        defer _ = mlx.mlx_array_free(f);
        try mlx.check(mlx.mlx_astype(&emb_w, f, .bfloat16, s));
    }
    var cache = try KVCache.init(allocator, 1);
    defer cache.deinit();
    var xfm = testHybridXfm(allocator, config, cache, emb_w, hl.ssm_entries, hl.hybrid_layers, s);
    defer deinitTestHybridXfm(&xfm);

    const LastRow = struct {
        fn f(x: *Transformer, ids: []const i32, out: *[64]f32, st: mlx.mlx_stream) !void {
            try x.cache.truncate(0, st);
            x.moe_seq_offset = 0;
            for (ids) |tok| {
                const id_arr = mlx.mlx_array_new_data(&[_]i32{tok}, &[_]c_int{ 1, 1 }, 2, .int32);
                defer _ = mlx.mlx_array_free(id_arr);
                const o = try x.forward(id_arr);
                defer _ = mlx.mlx_array_free(o);
                var f32a = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(f32a);
                try mlx.check(mlx.mlx_astype(&f32a, o, .float32, st));
                try mlx.check(mlx.mlx_array_eval(f32a));
                const p = mlx.mlx_array_data_float32(f32a) orelse return error.NoData;
                @memcpy(out, p[0..64]);
            }
        }
    }.f;
    var a: [64]f32 = undefined;
    var b: [64]f32 = undefined;
    try LastRow(&xfm, &.{ 1, 5, 2, 6, 3 }, &a, s);
    try LastRow(&xfm, &.{ 6, 2, 5, 1, 3 }, &b, s);
    var max_ref: f32 = 0;
    var max_err: f32 = 0;
    for (a, b) |x, y| {
        try t.expect(std.math.isFinite(x) and std.math.isFinite(y));
        max_ref = @max(max_ref, @abs(x));
        max_err = @max(max_err, @abs(x - y));
    }
    if (!(max_ref > 0.01)) return error.RefTooSmall;
    if (max_err > 1e-2 * max_ref) {
        std.debug.print("NoPE attention moved with token order: max_err={d:.6} max_ref={d:.6}\n", .{ max_err, max_ref });
        return error.AttentionIsPositional;
    }
}

/// Bare Transformer over `initHybridLayers` output for hermetic hybrid tests.
/// Pair with `deinitTestHybridXfm`.
pub fn testHybridXfm(allocator: std.mem.Allocator, config: ModelConfig, cache: KVCache, emb_w: mlx.mlx_array, ssm_entries: []SSMCacheEntry, hybrid_layers: []HybridLayerWeights, s: mlx.mlx_stream) Transformer {
    return .{
        .config = config,
        .cache = cache,
        .s = s,
        .allocator = allocator,
        .emb_w = emb_w,
        .emb_s = mlx.mlx_array_new(),
        .emb_b = mlx.mlx_array_new(),
        .emb_scale = null,
        .final_norm = mlx.mlx_array_new(),
        .lm_head_w = mlx.mlx_array_new(),
        .lm_head_s = mlx.mlx_array_new(),
        .lm_head_b = mlx.mlx_array_new(),
        .layers = &.{},
        .owns_lm_head = false,
        .owns_norms = false,
        .embedding_mode = true,
        .gelu_coeff = null,
        .gelu_inner = null,
        .half = mlx.mlx_array_new(),
        .one = mlx.mlx_array_new(),
        .three = null,
        .neg_one = null,
        .ple_emb_w = mlx.mlx_array_new(),
        .ple_emb_s = mlx.mlx_array_new(),
        .ple_emb_b = mlx.mlx_array_new(),
        .ple_proj_w = mlx.mlx_array_new(),
        .ple_proj_s = mlx.mlx_array_new(),
        .ple_proj_b = mlx.mlx_array_new(),
        .ple_proj_norm = mlx.mlx_array_new(),
        .ple_proj_quantized = false,
        .softcap_scalar = null,
        .v_norm_weight = null,
        .v_norm_weight_global = null,
        .rope_freqs_global = null,
        .bert_layers = null,
        .bert_pos_w = mlx.mlx_array_new(),
        .bert_pos_s = mlx.mlx_array_new(),
        .bert_pos_b = mlx.mlx_array_new(),
        .bert_toktype_w = mlx.mlx_array_new(),
        .bert_toktype_s = mlx.mlx_array_new(),
        .bert_toktype_b = mlx.mlx_array_new(),
        .bert_emb_norm_w = mlx.mlx_array_new(),
        .bert_emb_norm_b = mlx.mlx_array_new(),
        .moe_layers = null,
        .ssm_entries = ssm_entries,
        .moe_seq_offset = 0,
        .hybrid_layers = hybrid_layers,
        .embedding_norm = null,
        .prompt_cache = null,
    };
}

pub fn deinitTestHybridXfm(xfm: *Transformer) void {
    _ = mlx.mlx_array_free(xfm.emb_s);
    _ = mlx.mlx_array_free(xfm.emb_b);
    _ = mlx.mlx_array_free(xfm.final_norm);
    _ = mlx.mlx_array_free(xfm.lm_head_w);
    _ = mlx.mlx_array_free(xfm.lm_head_s);
    _ = mlx.mlx_array_free(xfm.lm_head_b);
    _ = mlx.mlx_array_free(xfm.half);
    _ = mlx.mlx_array_free(xfm.one);
    _ = mlx.mlx_array_free(xfm.ple_emb_w);
    _ = mlx.mlx_array_free(xfm.ple_emb_s);
    _ = mlx.mlx_array_free(xfm.ple_emb_b);
    _ = mlx.mlx_array_free(xfm.ple_proj_w);
    _ = mlx.mlx_array_free(xfm.ple_proj_s);
    _ = mlx.mlx_array_free(xfm.ple_proj_b);
    _ = mlx.mlx_array_free(xfm.ple_proj_norm);
    _ = mlx.mlx_array_free(xfm.bert_pos_w);
    _ = mlx.mlx_array_free(xfm.bert_pos_s);
    _ = mlx.mlx_array_free(xfm.bert_pos_b);
    _ = mlx.mlx_array_free(xfm.bert_toktype_w);
    _ = mlx.mlx_array_free(xfm.bert_toktype_s);
    _ = mlx.mlx_array_free(xfm.bert_toktype_b);
    _ = mlx.mlx_array_free(xfm.bert_emb_norm_w);
    _ = mlx.mlx_array_free(xfm.bert_emb_norm_b);
}

// Bar: every layer's tail is copied and named in a cadence eval inside the loop (the peak itself is measured live, not on a tiny fixture).
test "hybrid prefill: the cadence eval names every conv_state inside the layer loop" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const t = std.testing;
    const allocator = t.allocator;
    const s = mlx.gpuStream();
    const n_layers: u32 = 16;
    const seq: c_int = 512;
    const H: c_int = 64;
    const kernel: c_int = 3;

    var w = Weights.init(allocator);
    defer w.deinit();
    const put = struct {
        fn add(weights: *Weights, alloc: std.mem.Allocator, name: []const u8, shape: []const c_int, st: mlx.mlx_stream) !void {
            var n: usize = 1;
            for (shape) |d| n *= @intCast(d);
            const vals = try alloc.alloc(f32, n);
            defer alloc.free(vals);
            @memset(vals, 0.5);
            const f32_arr = mlx.mlx_array_new_data(vals.ptr, shape.ptr, @intCast(shape.len), .float32);
            defer _ = mlx.mlx_array_free(f32_arr);
            var bf = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&bf, f32_arr, .bfloat16, st));
            try weights.map.put(try alloc.dupe(u8, name), bf);
        }
    }.add;
    const P = "model";
    var li: u32 = 0;
    while (li < n_layers) : (li += 1) {
        var pre_buf: [64]u8 = undefined;
        const pre = try std.fmt.bufPrint(&pre_buf, "{s}.layers.{d}.", .{ P, li });
        var name_buf2: [96]u8 = undefined;
        const n1 = try std.fmt.bufPrint(&name_buf2, "{s}norm.weight", .{pre});
        try put(&w, allocator, n1, &.{H}, s);
        const n2 = try std.fmt.bufPrint(&name_buf2, "{s}conv.in_proj.weight", .{pre});
        try put(&w, allocator, n2, &.{ 3 * H, H }, s);
        const n3 = try std.fmt.bufPrint(&name_buf2, "{s}conv.conv.weight", .{pre});
        try put(&w, allocator, n3, &.{ H, kernel, 1 }, s);
        const n4 = try std.fmt.bufPrint(&name_buf2, "{s}conv.out_proj.weight", .{pre});
        try put(&w, allocator, n4, &.{ H, H }, s);
    }

    var config = ModelConfig{ .model_type = "nemotron_h", .weight_prefix = P };
    config.has_hybrid_layers = true;
    config.num_hidden_layers = n_layers;
    config.hidden_size = @intCast(H);
    config.quant_bits = 0;
    config.has_final_norm = false;
    config.lfm_conv_kernel = @intCast(kernel);
    config.head_dim = 64;
    config.num_attention_heads = 1;
    config.num_key_value_heads = 1;
    config.vocab_size = 8;
    for (0..n_layers) |i| config.layer_block_types[i] = .gated_conv;

    var name_buf: [256]u8 = undefined;
    const hl = try initHybridLayers(allocator, config, &w, &name_buf, s);
    defer {
        allocator.free(hl.hybrid_layers);
        for (hl.ssm_entries) |*e| {
            _ = mlx.mlx_array_free(e.conv_state);
            _ = mlx.mlx_array_free(e.ssm_state);
        }
        allocator.free(hl.ssm_entries);
        for (hl.owned_bf16) |a| _ = mlx.mlx_array_free(a);
        allocator.free(hl.owned_bf16);
    }

    var emb_w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(emb_w);
    const emb_shape = [_]c_int{ 8, H };
    try mlx.check(mlx.mlx_ones(&emb_w, &emb_shape, 2, .bfloat16, s));
    var cache = try KVCache.init(allocator, n_layers);
    defer cache.deinit();

    var xfm = testHybridXfm(allocator, config, cache, emb_w, hl.ssm_entries, hl.hybrid_layers, s);
    defer deinitTestHybridXfm(&xfm);

    var ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids);
    const id_shape = [_]c_int{ 1, seq };
    try mlx.check(mlx.mlx_zeros(&ids, &id_shape, 2, .int32, s));

    checkpoint.compact_conv_state_copy_count = 0;
    const warm = try xfm.forward(ids);
    _ = mlx.mlx_array_free(warm);
    xfm.moe_seq_offset = 0;
    _ = mlx.mlx_synchronize(s);
    _ = mlx.mlx_clear_cache();

    checkpoint.compact_conv_state_copy_count = 0;
    checkpoint.cadence_conv_eval_count = 0;
    const out = try xfm.forward(ids);
    try mlx.check(mlx.mlx_array_eval(out));
    _ = mlx.mlx_array_free(out);
    try t.expectEqual(@as(u64, n_layers), checkpoint.compact_conv_state_copy_count);
    try t.expect(checkpoint.cadence_conv_eval_count >= 8);
}
