//! GLM-5.3-Flash's legacy forward on the engine's `ForwardCtx` (prefill, decode, verify, the batched tick, the MTP
//! head's round and DFlash capture). The table forward (`table_forward.zig`) is its copy over `CacheOps`; this one
//! goes when glm5_next flips to its plugin.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const model_mod = @import("../../engines/mlx/model.zig");
const log = @import("../../host/log.zig");
const glm5 = @import("glm5_next.zig");
const glm_mtp = @import("mtp.zig");
const kda_recurrence = @import("../../sdk/blocks/kda/recurrence.zig");
const layerCap = @import("../../engines/mlx/diag/profile.zig").layerCap;
const diagEnvOnCached = @import("../../engines/mlx/env_transformer.zig").diagEnvOnCached;
const PREFILL_EVAL_CADENCE_DEFAULT = @import("../../engines/mlx/eval_cadence.zig").PREFILL_EVAL_CADENCE_DEFAULT;
const evalCadencePoint = @import("../../engines/mlx/eval_cadence.zig").evalCadencePoint;
const prefillEvalCadenceApplies = @import("../../engines/mlx/eval_cadence.zig").prefillEvalCadenceApplies;
const prefillEvalCadenceFor = @import("../../engines/mlx/eval_cadence.zig").prefillEvalCadenceFor;
const MoeLayerWeights = @import("../../engines/mlx/forward/layered.zig").MoeLayerWeights;
const CaptureLayers = @import("../../engines/mlx/forward_ctx.zig").CaptureLayers;
const ForwardCtx = @import("../../engines/mlx/forward_ctx.zig").ForwardCtx;
const testing = @import("../../engines/mlx/kv/cache.zig").testing;
const QW = @import("../../engines/mlx/layer_weights.zig").QW;
const ssmFreeSpecCapture = @import("../../engines/mlx/ssm/entry.zig").ssmFreeSpecCapture;
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;
const axisView = @import("../../mlx/array.zig").axisView;
const concatAxis1 = @import("../../mlx/array.zig").concatAxis1;
const concatRows = @import("../../mlx/array.zig").concatRows;
const materializedOwnedCopy = @import("../../mlx/array.zig").materializedOwnedCopy;
const LinearAttnWeights = @import("../../sdk/blocks/gdn/gated_delta_net.zig").LinearAttnWeights;
const qwen4CompareRows = @import("../../sdk/testing/parity.zig").qwen4CompareRows;
const qwen4ReadF32 = @import("../../sdk/testing/parity.zig").qwen4ReadF32;
const mimoDecidedMisses = @import("../../sdk/testing/parity.zig").decidedMisses;
const Qwen4TestSlot = @import("../qwen4_exp/forward.zig").Qwen4TestSlot;
const DsaWeights = @import("weights.zig").DsaWeights;

var glm_batched_logged: bool = false;

// ── GLM-5-Next (glm5_next) ──

/// Per-head projection over `w` [H, out, in]: `x @ w^T` when `transpose`, else `x @ w`.
/// `in_dim` is the stored layout's input width (the quantized axis).
pub fn glmHeadMm(self: *Transformer, x: mlx.mlx_array, q: *const QW, transpose: bool, in_dim: u32) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    if (q.s.ctx == null) {
        if (transpose) {
            var wt = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wt);
            try mlx.check(mlx.mlx_swapaxes(&wt, q.w, -1, -2, self.s));
            try mlx.check(mlx.mlx_matmul(&out, x, wt, self.s));
        } else try mlx.check(mlx.mlx_matmul(&out, x, q.w, self.s));
        return out;
    }
    const qp = self.quantParamsHinted(q.w, q.s, in_dim);
    if (transpose and qp.mode == .affine) if (try glm5.headQmv(self.s, x, q.w, q.s, q.b, qp.bits, qp.group_size, 1.0)) |y| {
        _ = mlx.mlx_array_free(out);
        return y;
    };
    try mlx.check(mlx.mlx_quantized_matmul(&out, x, q.w, q.s, q.b, transpose, mlx.mlx_optional_int.some(@intCast(qp.group_size)), mlx.mlx_optional_int.some(@intCast(qp.bits)), qp.mode.cstr(), self.s));
    return out;
}

/// GLM-5-Next forward over a [B, S, hc, D] stream: every sublayer collapses the
/// stream through its Sinkhorn hyper-connection, runs on the normalized collapse,
/// and expands its output back; the head reads the norm of the stream mean.
pub fn forwardGlm5With(self: *Transformer, ctx: *ForwardCtx, token_ids: mlx.mlx_array) !mlx.mlx_array {
    const ml = self.moe_layers.?;
    const cfg = &self.config;
    const offset = ctx.moe_seq_offset.*;
    self.spec_capture_ssm = ctx.capture_ssm_seq;
    defer self.spec_capture_ssm = false;
    self.spec_tree = ctx.tree;
    defer self.spec_tree = null;

    var emb = try self.embedding(token_ids);
    {
        errdefer _ = mlx.mlx_array_free(emb);
        emb = try self.applyVisionEmbeddingsWith(ctx, emb, token_ids);
    }
    defer _ = mlx.mlx_array_free(emb);
    var dt = mlx.DtypeTrace.begin("glm5", emb, null);
    const x_shape = mlx.getShape(emb);
    const batch: c_int = x_shape[0];
    const seq_len: c_int = x_shape[1];
    // A batched tick's rows are N sequences' next tokens (`glmBranchSlots`), not a prompt chunk.
    const batched = ctx.batch_slots != null;
    const is_prefill = seq_len > 1 and !batched;
    var stream = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(stream);
    {
        var e4 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(e4);
        try mlx.check(mlx.mlx_expand_dims(&e4, emb, 2, self.s));
        try mlx.check(mlx.mlx_broadcast_to(&stream, e4, &[_]c_int{ batch, seq_len, glm5.HC, x_shape[2] }, 4, self.s));
    }

    const eval_cadence = prefillEvalCadenceFor(
        PREFILL_EVAL_CADENCE_DEFAULT,
        cfg,
        @intCast(seq_len),
        @as(u64, @intCast(offset)) + @as(u64, @intCast(seq_len)),
        ctx.cache.config.scheme != .off,
    );
    const n_layers = layerCap(cfg.num_hidden_layers);
    // Each expand stays deferred into the next sublayer's `hcPre` (one dispatch for both).
    var deferred: ?glm5.Deferred = null;
    defer if (deferred) |*df| df.deinit();
    for (ml[0..n_layers], 0..) |*lw, li| {
        for ([_]bool{ true, false }) |attn| {
            try self.glmSublayer(ctx, &stream, &deferred, lw, attn, li, offset, batch, seq_len, is_prefill);
        }
        dt.layer(stream, li);
        // DFlash taps the MEAN of the layer's hyper-connection streams (SGLang's `hc_contract`).
        if (ctx.capture_layers) |cl| for (cl.ids, cl.out) |cid, *slot| if (cid == li) {
            if (deferred) |*df| {
                _ = mlx.mlx_array_free(stream);
                stream = try df.materialize(self.s);
                df.deinit();
                deferred = null;
            }
            var mean = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(mean);
            try mlx.check(mlx.mlx_mean_axis(&mean, stream, 2, false, self.s));
            _ = mlx.mlx_array_set(slot, mean);
        };
        if (is_prefill and prefillEvalCadenceApplies(seq_len) and ((li + 1) % eval_cadence == 0 or li + 1 == n_layers)) {
            try evalCadencePoint(stream, ctx.ssm_entries);
        }
    }
    if (!batched) ctx.moe_seq_offset.* += @intCast(seq_len);
    if (deferred) |*df| {
        _ = mlx.mlx_array_free(stream);
        stream = try df.materialize(self.s);
    }
    dt.end(stream);
    if (diagEnvOnCached(&glm_profile_env, "MLX_SERVE_GLM_PROFILE")) {
        log.info("[glm-profile] rows={d} collapse {d:.2} kda {d:.2} dsa {d:.2} moe {d:.2} expand {d:.2} ms\n", .{ seq_len, glmProfMs(.collapse), glmProfMs(.kda), glmProfMs(.dsa), glmProfMs(.moe), glmProfMs(.expand) });
        glm_prof_ns = @splat(0);
    }

    var mean = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mean);
    try mlx.check(mlx.mlx_mean_axis(&mean, stream, 2, false, self.s));
    const final_normed = try self.rmsNorm(mean, self.final_norm);
    if (ctx.capture_hidden) |target| {
        const fn_shape = mlx.getShape(final_normed);
        var sliced = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sliced);
        try mlx.check(mlx.mlx_slice(&sliced, final_normed, &[_]c_int{ 0, fn_shape[1] - 1, 0 }, 3, &[_]c_int{ fn_shape[0], fn_shape[1], fn_shape[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, self.s));
        _ = mlx.mlx_array_set(target, sliced);
    }
    if (ctx.capture_hidden_all) |target_all| _ = mlx.mlx_array_set(target_all, final_normed);
    if (self.embedding_mode or ctx.skip_lm_head) return final_normed;
    defer _ = mlx.mlx_array_free(final_normed);
    return self.lmHeadProject(final_normed);
}

/// One GLM-5-Next sublayer (attention when `attn`, else the MLP) inside its
/// hyper-connection. Advances `stream`, or, for one token, leaves the expand in
/// `deferred` (then `stream` is stale until the forward materializes it).
pub fn glmSublayer(self: *Transformer, ctx: *ForwardCtx, stream: *mlx.mlx_array, deferred: *?glm5.Deferred, lw: *const MoeLayerWeights, attn: bool, li: usize, offset: usize, batch: c_int, seq_len: c_int, is_prefill: bool) !void {
    const cfg = &self.config;
    const w = if (attn) lw.mhc_attn.? else lw.mhc_ffn.?;
    const norm_w = if (attn) lw.input_norm else lw.post_attn_norm;
    const prof = diagEnvOnCached(&glm_profile_env, "MLX_SERVE_GLM_PROFILE");
    // A profiled forward evals at each seam, so it keeps the expand apart.
    const one = !prof and glm5.hcRowsServes(self.s, mlx.getShape(stream.*), mlx.mlx_array_dtype(stream.*), w.fn_rows, norm_w);
    if (!one) if (deferred.*) |*df| {
        _ = mlx.mlx_array_free(stream.*);
        stream.* = try df.materialize(self.s);
        df.deinit();
        deferred.* = null;
    };
    if (one) {
        var pre = try glm5.hcPre(self.s, if (deferred.* == null) stream.* else null, if (deferred.*) |*df| df else null, w.fn_rows, w.scale, w.base, norm_w, cfg.dsv4_hc_sinkhorn_iters, cfg.dsv4_hc_eps, cfg.rms_norm_eps, cfg.rms_norm_eps);
        defer pre.deinit();
        const out = try self.glmBranch(ctx, pre.normed, lw, attn, li, offset, batch, seq_len, is_prefill);
        defer _ = mlx.mlx_array_free(out);
        if (deferred.*) |*df| {
            // The rebuilt stream stands in for the stale one.
            _ = mlx.mlx_array_set(stream, pre.h);
            df.deinit();
        }
        deferred.* = pre.defer_(out, stream.*);
        return;
    }
    // DIAGNOSTIC (MLX_SERVE_GLM_PROFILE): eval at each seam and bill the wall time per kind.
    if (glm5.hcRowsServes(self.s, mlx.getShape(stream.*), mlx.mlx_array_dtype(stream.*), w.fn_rows, norm_w)) {
        var t0: i128 = if (prof) glmProfNow() else 0;
        var pre = try glm5.hcPre(self.s, stream.*, null, w.fn_rows, w.scale, w.base, norm_w, cfg.dsv4_hc_sinkhorn_iters, cfg.dsv4_hc_eps, cfg.rms_norm_eps, cfg.rms_norm_eps);
        defer pre.deinit();
        if (prof) t0 = glmProfLap(pre.normed, .collapse, t0);
        const out = try self.glmBranch(ctx, pre.normed, lw, attn, li, offset, batch, seq_len, is_prefill);
        defer _ = mlx.mlx_array_free(out);
        if (prof) t0 = glmProfLap(out, if (!attn) .moe else if (lw.attn == .linear) .kda else .dsa, t0);
        const c: glm5.Collapsed = .{ .x = pre.normed, .post = pre.post, .comb = pre.comb };
        const res = try glm5.expand(self.s, out, stream.*, &c);
        if (prof) t0 = glmProfLap(res, .expand, t0);
        _ = mlx.mlx_array_free(stream.*);
        stream.* = res;
        return;
    }
    var t0: i128 = if (prof) glmProfNow() else 0;
    var c = try glm5.collapse(self.s, stream.*, w.fn_rows, w.scale, w.base, cfg.dsv4_hc_sinkhorn_iters, cfg.dsv4_hc_eps, cfg.rms_norm_eps);
    defer c.deinit();
    const normed = try self.rmsNorm(c.x, norm_w);
    defer _ = mlx.mlx_array_free(normed);
    if (prof) t0 = glmProfLap(normed, .collapse, t0);
    const out = try self.glmBranch(ctx, normed, lw, attn, li, offset, batch, seq_len, is_prefill);
    defer _ = mlx.mlx_array_free(out);
    if (prof) t0 = glmProfLap(out, if (!attn) .moe else if (lw.attn == .linear) .kda else .dsa, t0);
    const res = try glm5.expand(self.s, out, stream.*, &c);
    if (prof) t0 = glmProfLap(res, .expand, t0);
    _ = mlx.mlx_array_free(stream.*);
    stream.* = res;
}

pub fn glmBranch(self: *Transformer, ctx: *ForwardCtx, normed: mlx.mlx_array, lw: *const MoeLayerWeights, attn: bool, li: usize, offset: usize, batch: c_int, seq_len: c_int, is_prefill: bool) !mlx.mlx_array {
    if (ctx.batch_slots) |slots| return self.glmBranchSlots(slots, normed, lw, attn, li);
    return if (attn) switch (lw.attn) {
        .linear => |*la| try self.gatedDeltaNet(normed, la, &ctx.ssm_entries.?[li], li, batch, seq_len, is_prefill),
        .full => |*fa| try self.glmDsaAttn(ctx, normed, &fa.dsa.?, @intCast(li), offset, batch, seq_len),
    } else switch (lw.mlp) {
        .moe => |*mw| try self.moeMLP(normed, mw),
        .dense => |*dw| try self.denseMLP(normed, dw),
    };
}

/// One batched tick's branch: `normed` `[1, N, H]` holds N slots' next-token rows. The MLP and
/// the projections read their weights once for all N rows; a slot's own state (its KDA
/// recurrence, its DSA cache) is advanced by its own call.
pub fn glmBranchSlots(self: *Transformer, slots: []const *ForwardCtx, normed: mlx.mlx_array, lw: *const MoeLayerWeights, attn: bool, li: usize) !mlx.mlx_array {
    if (!attn) return switch (lw.mlp) {
        .moe => |*mw| try self.moeMLP(normed, mw),
        .dense => |*dw| try self.denseMLP(normed, dw),
    };
    switch (lw.attn) {
        .linear => |*la| return self.glmKdaSlots(slots, normed, la, li),
        .full => |*fa| {
            var rows: [MOE_DECODE_BATCH_MAX]mlx.mlx_array = undefined;
            var built: usize = 0;
            defer for (rows[0..built]) |r| {
                _ = mlx.mlx_array_free(r);
            };
            for (slots, 0..) |sl, i| {
                const row = try axisView(self.s, normed, 1, i);
                defer _ = mlx.mlx_array_free(row);
                rows[i] = try self.glmDsaAttn(sl, row, &fa.dsa.?, @intCast(li), sl.moe_seq_offset.*, 1, 1);
                built += 1;
            }
            return concatAxis1(self.s, rows[0..slots.len]);
        },
    }
}

/// The low-rank gate second stages the fused KDA step reads: `f_b` (forget) and `g_b` (output).
pub fn kdaLows(self: *Transformer, la: *const LinearAttnWeights, dk: c_int) struct { fb: kda_recurrence.Low, gb: kda_recurrence.Low } {
    const a2q = self.quantParamsHinted(la.a2.w, la.a2.s, @intCast(dk));
    const z2q = self.quantParamsHinted(la.z2.w, la.z2.s, @intCast(dk));
    return .{
        .fb = .{ .w = la.a2.w, .s = la.a2.s, .b = la.a2.b, .bits = a2q.bits, .gs = a2q.group_size },
        .gb = .{ .w = la.z2.w, .s = la.z2.s, .b = la.z2.b, .bits = z2q.bits, .gs = z2q.group_size },
    };
}

/// KDA layer over N slots' rows: ONE joined input projection and ONE output projection, the
/// fused one-token step per slot against that slot's own conv + recurrent state. A layer whose
/// projection is not joined (layer 0) runs the layer per slot.
pub fn glmKdaSlots(self: *Transformer, slots: []const *ForwardCtx, x: mlx.mlx_array, la: *const LinearAttnWeights, li: usize) !mlx.mlx_array {
    const cfg = &self.config;
    const heads: c_int = @intCast(cfg.linear_num_value_heads);
    const dk: c_int = @intCast(cfg.linear_key_head_dim);
    var rows: [MOE_DECODE_BATCH_MAX]mlx.mlx_array = undefined;
    var built: usize = 0;
    defer for (rows[0..built]) |r| {
        _ = mlx.mlx_array_free(r);
    };
    const joined = la.in.w.ctx != null and la.a_log_h.ctx != null and la.a2.w.ctx != null and la.z2.w.ctx != null;
    // A GPU that cannot launch the fused step declines at the first slot: the per-slot path below serves it.
    if (joined) joined_path: {
        for (slots) |sl| {
            const e = &sl.ssm_entries.?[li];
            if (!e.initialized or e.ssm_state.ctx == null or e.conv_state.ctx == null) return error.GlmBatchedStateMissing;
        }
        const all = try self.qmatmul(x, la.in.w, la.in.s, la.in.b);
        defer _ = mlx.mlx_array_free(all);
        const w = la.in.widths;
        const lows = self.kdaLows(la, dk);
        for (slots, 0..) |sl, i| {
            const e = &sl.ssm_entries.?[li];
            const proj = try axisView(self.s, all, 1, i);
            defer _ = mlx.mlx_array_free(proj);
            const r = (try kda_recurrence.decodeStep(self.s, proj, w[0], w[0] + w[1], w[0] + w[1] + w[2], heads, e.conv_state, la.conv1d_w, la.a_log_h, la.dt_f32, e.ssm_state, la.norm_w, lows.fb, lows.gb, cfg.kda_gate_lower_bound, cfg.rms_norm_eps, false)) orelse if (i == 0) break :joined_path else return error.GlmBatchedKdaDeclined;
            _ = mlx.mlx_array_free(e.conv_state);
            e.conv_state = r.conv_state;
            _ = mlx.mlx_array_free(e.ssm_state);
            e.ssm_state = r.state;
            rows[i] = r.y;
            built += 1;
        }
        const y = try concatAxis1(self.s, rows[0..slots.len]);
        defer _ = mlx.mlx_array_free(y);
        return self.qmatmul(y, la.out_w, la.out_s, la.out_b);
    }
    for (slots, 0..) |sl, i| {
        const row = try axisView(self.s, x, 1, i);
        defer _ = mlx.mlx_array_free(row);
        rows[i] = try self.gatedDeltaNet(row, la, &sl.ssm_entries.?[li], li, 1, 1, false);
        built += 1;
    }
    return concatAxis1(self.s, rows[0..slots.len]);
}

/// Rows a batched GLM tick carries: the MoE decode kernels' row limit.
pub const MOE_DECODE_BATCH_MAX = 8;

/// Next-token logits for N slots in as few trunk forwards as the MoE row limit allows (8 per
/// forward): each slot's token is one ROW of a `[1, n]` window, so the weights are read once
/// per forward rather than once per slot. `hidden_rows`, when asked, receives each slot's
/// post-final-norm hidden (`[1,1,H]`) for an MTP head that resumes after the group. Returns
/// N arrays `[1,1,V]`; the caller owns them and the slice.
pub fn forwardGlmBatchedDecode(
    self: *Transformer,
    next_tokens: []const u32,
    ctxs: []const *ForwardCtx,
    hidden_rows: ?*?[]mlx.mlx_array,
) ![]mlx.mlx_array {
    std.debug.assert(next_tokens.len == ctxs.len and next_tokens.len >= 1);
    if (!glm_batched_logged) {
        glm_batched_logged = true;
        log.info("[batched] glm batched decode engaged (slots={d})\n", .{next_tokens.len});
    }
    const out = try self.allocator.alloc(mlx.mlx_array, ctxs.len);
    var made: usize = 0;
    errdefer {
        for (out[0..made]) |a| _ = mlx.mlx_array_free(a);
        self.allocator.free(out);
    }
    var hid: ?[]mlx.mlx_array = null;
    if (hidden_rows != null) hid = try self.allocator.alloc(mlx.mlx_array, ctxs.len);
    var hid_made: usize = 0;
    errdefer if (hid) |h| {
        for (h[0..hid_made]) |a| _ = mlx.mlx_array_free(a);
        self.allocator.free(h);
    };
    var start: usize = 0;
    while (start < ctxs.len) {
        const n = @min(MOE_DECODE_BATCH_MAX, ctxs.len - start);
        const group = ctxs[start .. start + n];
        var toks: [MOE_DECODE_BATCH_MAX]i32 = undefined;
        for (next_tokens[start .. start + n], 0..) |t, i| toks[i] = @intCast(t);
        const token_arr = mlx.mlx_array_new_data(&toks, &[_]c_int{ 1, @intCast(n) }, 2, .int32);
        defer _ = mlx.mlx_array_free(token_arr);
        var scratch_offset: usize = group[0].moe_seq_offset.*;
        var hidden_all = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(hidden_all);
        var bctx: ForwardCtx = .{
            .cache = group[0].cache,
            .moe_seq_offset = &scratch_offset,
            .ssm_entries = group[0].ssm_entries,
            .capture_hidden = null,
            .capture_hidden_all = if (hid != null) &hidden_all else null,
            .vision_embeddings = null,
            .batch_slots = group,
        };
        const logits = try self.forwardGlm5With(&bctx, token_arr);
        defer _ = mlx.mlx_array_free(logits);
        const v = mlx.getShape(logits)[2];
        for (0..n) |i| {
            var row = mlx.mlx_array_new(); // empty until the slice lands, so a failure leaks nothing
            try mlx.check(mlx.mlx_slice(&row, logits, &[_]c_int{ 0, @intCast(i), 0 }, 3, &[_]c_int{ 1, @intCast(i + 1), v }, 3, &[_]c_int{ 1, 1, 1 }, 3, self.s));
            out[start + i] = row;
            made = start + i + 1;
            if (hid) |h| {
                h[start + i] = try axisView(self.s, hidden_all, 1, i);
                hid_made = start + i + 1;
            }
        }
        start += n;
    }
    if (hidden_rows) |slot| slot.* = hid;
    return out;
}

/// GLM-5-Next DeepSeek sparse attention. The cache holds, per token, the 512 latent
/// (`k`) and the indexer's key and pool gate (`v`). Queries attend in the latent
/// (absorbed MLA) over the tokens the indexer picks; when every query picks every
/// earlier token the attention is plain causal and prefill runs it on lifted keys.
pub fn glmDsaAttn(self: *Transformer, ctx: *ForwardCtx, x: mlx.mlx_array, d: *const DsaWeights, layer: u32, offset: usize, batch: c_int, seq_len: c_int) !mlx.mlx_array {
    const cfg = &self.config;
    const heads: c_int = @intCast(cfg.num_attention_heads);
    const hd: c_int = @intCast(cfg.dsa_head_dim);
    const r: c_int = @intCast(cfg.dsa_kv_lora_rank);
    const ihd: c_int = @intCast(cfg.dsa_index_head_dim);
    const ih: c_int = @intCast(cfg.dsa_index_heads);

    const q_a = try self.qmatmul(x, d.q_a.w, d.q_a.s, d.q_a.b);
    defer _ = mlx.mlx_array_free(q_a);
    const kv_a = try self.qmatmul(x, d.kv_a.w, d.kv_a.s, d.kv_a.b);
    defer _ = mlx.mlx_array_free(kv_a);
    const q_resid = try self.rmsNorm(q_a, d.q_a_norm);
    defer _ = mlx.mlx_array_free(q_resid);
    const q_flat = try self.qmatmul(q_resid, d.q_b.w, d.q_b.s, d.q_b.b);
    defer _ = mlx.mlx_array_free(q_flat);
    var q4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q4);
    try mlx.check(mlx.mlx_reshape(&q4, q_flat, &[_]c_int{ batch, seq_len, heads, hd }, 4, self.s));
    var qh = mlx.mlx_array_new(); // [B, H, S, hd]
    defer _ = mlx.mlx_array_free(qh);
    try mlx.check(mlx.mlx_transpose_axes(&qh, q4, &[_]c_int{ 0, 2, 1, 3 }, 4, self.s));

    const lat = try self.rmsNorm(kv_a, d.kv_a_norm);
    defer _ = mlx.mlx_array_free(lat);
    var lat4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lat4);
    try mlx.check(mlx.mlx_reshape(&lat4, lat, &[_]c_int{ batch, 1, seq_len, r }, 4, self.s));

    // Indexer key (LayerNorm) and pool gate (f32 projection, stored in the key's dtype).
    const ik = try self.qmatmul(x, d.idx_wk.w, d.idx_wk.s, d.idx_wk.b);
    defer _ = mlx.mlx_array_free(ik);
    var ikn = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ikn);
    try mlx.check(mlx.mlx_fast_layer_norm(&ikn, ik, d.idx_k_norm_w, d.idx_k_norm_b, 1e-6, self.s));
    var xf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xf);
    try mlx.check(mlx.mlx_astype(&xf, x, .float32, self.s));
    var gate = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gate);
    try mlx.check(mlx.mlx_matmul(&gate, xf, d.idx_gate_t, self.s));
    var gate_c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gate_c);
    try mlx.check(mlx.mlx_astype(&gate_c, gate, mlx.mlx_array_dtype(ikn), self.s));
    var idx_new = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(idx_new);
    {
        const parts = [_]mlx.mlx_array{ ikn, gate_c };
        const vec = mlx.mlx_vector_array_new_data(&parts, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        var cat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cat);
        try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, -1, self.s));
        try mlx.check(mlx.mlx_reshape(&idx_new, cat, &[_]c_int{ batch, 1, seq_len, 2 * ihd }, 4, self.s));
    }

    // A row that completes a pool stores the pool's key (`glm5.withPooledKeys`).
    const idx_rows = blk: {
        var rows3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rows3);
        try mlx.check(mlx.mlx_reshape(&rows3, idx_new, &[_]c_int{ batch, seq_len, 2 * ihd }, 3, self.s));
        const kpool: c_int = @intCast(cfg.dsa_index_kpool);
        const n_prev: c_int = @min(@as(c_int, @intCast(offset)), kpool - 1);
        var prev = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(prev);
        if (n_prev > 0) {
            var cached = try ctx.cache.denseView(layer, self.s);
            defer cached.deinit();
            const vsh = mlx.getShape(cached.v);
            var tail = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(tail);
            try mlx.check(mlx.mlx_slice(&tail, cached.v, &[_]c_int{ 0, 0, vsh[2] - n_prev, 0 }, 4, &[_]c_int{ vsh[0], vsh[1], vsh[2], vsh[3] }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, self.s));
            try mlx.check(mlx.mlx_reshape(&prev, tail, &[_]c_int{ batch, n_prev, 2 * ihd }, 3, self.s));
        } else _ = mlx.mlx_array_set(&prev, rows3);
        const pooled = try glm5.withPooledKeys(self.s, prev, rows3, d.idx_ape, @intCast(offset), .{ .topk = @intCast(cfg.dsa_index_topk), .kpool = kpool, .heads = ih, .head_dim = ihd });
        defer _ = mlx.mlx_array_free(pooled);
        var r4 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&r4, pooled, &[_]c_int{ batch, 1, seq_len, 2 * ihd }, 4, self.s));
        break :blk r4;
    };
    defer _ = mlx.mlx_array_free(idx_rows);
    var view = try ctx.cache.update(layer, lat4, idx_rows, self.s, 0);
    defer view.deinit();
    const t: c_int = @intCast(offset + @as(usize, @intCast(seq_len)));
    const ix = glm5.Indexer{ .topk = @intCast(cfg.dsa_index_topk), .kpool = @intCast(cfg.dsa_index_kpool), .heads = ih, .head_dim = ihd };
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
    const none = mlx.mlx_array{ .ctx = null };

    // A row at a position below `dense_end` reads every earlier token (all its pools fit
    // the budget): plain causal attention. Prefill runs those rows on lifted keys and the
    // rest through the indexer; decode and verify rows stay in the latent.
    const dense_end: c_int = (@divTrunc(ix.topk, ix.kpool) + 1) * ix.kpool - 1;
    const dense_rows: c_int = std.math.clamp(dense_end - @as(c_int, @intCast(offset)), 0, seq_len);
    var parts: [2]mlx.mlx_array = .{ .{ .ctx = null }, .{ .ctx = null } };
    defer for (parts) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a);
    };
    if (seq_len > glm5.HC_PRE_MAX_ROWS and dense_rows > 0) {
        var kv_rows = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(kv_rows);
        try mlx.check(mlx.mlx_slice(&kv_rows, view.k, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ batch, 1, @as(c_int, @intCast(offset)) + dense_rows, r }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, self.s));
        const keys = try self.glmHeadMm(kv_rows, &d.embed_q, false, cfg.dsa_head_dim);
        defer _ = mlx.mlx_array_free(keys);
        const values = try self.glmHeadMm(kv_rows, &d.unembed, true, cfg.dsa_kv_lora_rank);
        defer _ = mlx.mlx_array_free(values);
        var q_rows = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_rows);
        try mlx.check(mlx.mlx_slice(&q_rows, qh, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ batch, heads, dense_rows, hd }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, self.s));
        parts[0] = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&parts[0], q_rows, keys, values, scale, "causal", none, none, false, self.s));
    }
    const sparse_from: c_int = if (seq_len > glm5.HC_PRE_MAX_ROWS) dense_rows else 0;
    if (sparse_from < seq_len) {
        const rows = seq_len - sparse_from;
        const Rows = struct {
            fn of(a: mlx.mlx_array, axis: usize, from: c_int, st: mlx.mlx_stream) !mlx.mlx_array {
                const sh = mlx.getShape(a);
                var lo: [4]c_int = @splat(0);
                var hi: [4]c_int = undefined;
                for (sh, 0..) |v, i| hi[i] = v;
                lo[axis] = from;
                var out = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_slice(&out, a, &lo, sh.len, &hi, sh.len, &[_]c_int{ 1, 1, 1, 1 }, sh.len, st));
                return out;
            }
        };
        const qh_s = try Rows.of(qh, 2, sparse_from, self.s);
        defer _ = mlx.mlx_array_free(qh_s);
        // One query: the per-head kernel applies the attention scale as it rounds.
        var q_scale: ?f32 = scale;
        const q_abs = blk: {
            if (rows == 1 and d.embed_q.s.ctx != null) {
                const qp = self.quantParamsHinted(d.embed_q.w, d.embed_q.s, cfg.dsa_head_dim);
                if (qp.mode == .affine) if (try glm5.headQmv(self.s, qh_s, d.embed_q.w, d.embed_q.s, d.embed_q.b, qp.bits, qp.group_size, scale)) |y| {
                    q_scale = null;
                    break :blk y;
                };
            }
            break :blk try self.glmHeadMm(qh_s, &d.embed_q, true, cfg.dsa_head_dim);
        };
        defer _ = mlx.mlx_array_free(q_abs);
        var o_lat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(o_lat);
        if (glm5.selectsAll(t, ix) and rows <= glm5.HC_PRE_MAX_ROWS and batch == 1) {
            var keys = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(keys);
            try mlx.check(mlx.mlx_reshape(&keys, view.k, &[_]c_int{ 1, t, r }, 3, self.s));
            _ = mlx.mlx_array_free(o_lat);
            if (rows == 1) {
                o_lat = try glm5.latentAttendFolded(self.s, q_abs, keys, null, q_scale);
            } else {
                // A verify window folds its rows into the head axis under a per-row causal
                // mask: MLX's sdpa declines the 64:1 GQA factor and reads the keys per head.
                var qf = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(qf);
                try mlx.check(mlx.mlx_reshape(&qf, q_abs, &[_]c_int{ 1, heads * rows, 1, r }, 4, self.s));
                const valid = try glm5.latentCausalMask(self.s, heads, rows, t);
                defer _ = mlx.mlx_array_free(valid);
                const of = try glm5.latentAttendFolded(self.s, qf, keys, valid, q_scale);
                defer _ = mlx.mlx_array_free(of);
                o_lat = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_reshape(&o_lat, of, &[_]c_int{ 1, heads, rows, r }, 4, self.s));
            }
        } else if (glm5.selectsAll(t, ix)) {
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&o_lat, q_abs, view.k, view.k, q_scale orelse 1.0, if (rows > 1) "causal" else "", none, none, false, self.s));
        } else {
            const qr_s = try Rows.of(q_resid, 1, sparse_from, self.s);
            defer _ = mlx.mlx_array_free(qr_s);
            const q_idx_flat = try self.qmatmul(qr_s, d.idx_wq_b.w, d.idx_wq_b.s, d.idx_wq_b.b);
            defer _ = mlx.mlx_array_free(q_idx_flat);
            var q_idx = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(q_idx);
            try mlx.check(mlx.mlx_reshape(&q_idx, q_idx_flat, &[_]c_int{ batch, rows, ih, ihd }, 4, self.s));
            const x_s = try Rows.of(x, 1, sparse_from, self.s);
            defer _ = mlx.mlx_array_free(x_s);
            const wp = try self.qmatmul(x_s, d.idx_wproj.w, d.idx_wproj.s, d.idx_wproj.b);
            defer _ = mlx.mlx_array_free(wp);
            var wpf = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wpf);
            try mlx.check(mlx.mlx_astype(&wpf, wp, .float32, self.s));
            const hscale = mlx.mlx_array_new_float(1.0 / @sqrt(@as(f32, @floatFromInt(ih))));
            defer _ = mlx.mlx_array_free(hscale);
            var wts = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wts);
            try mlx.check(mlx.mlx_multiply(&wts, wpf, hscale, self.s));
            var idx_cache = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(idx_cache);
            try mlx.check(mlx.mlx_reshape(&idx_cache, view.v, &[_]c_int{ batch, t, 2 * ihd }, 3, self.s));
            const sel = try glm5.selectTokens(self.s, q_idx, wts, idx_cache, ix);
            defer _ = mlx.mlx_array_free(sel);
            const nax = if (rows > 8) try glm5.sparseLatentNax(self.s, q_abs, view.k, sel, q_scale orelse 1.0) else null;
            const lat_out = nax orelse try glm5.sparseLatentAttention(self.s, q_abs, view.k, sel, q_scale, 64);
            _ = mlx.mlx_array_free(o_lat);
            o_lat = lat_out;
        }
        parts[1] = try self.glmHeadMm(o_lat, &d.unembed, true, cfg.dsa_kv_lora_rank);
    }
    var o = mlx.mlx_array_new(); // [B, H, S, hd]
    defer _ = mlx.mlx_array_free(o);
    if (parts[0].ctx != null and parts[1].ctx != null) {
        const vec = mlx.mlx_vector_array_new_data(&parts, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&o, vec, 2, self.s));
    } else {
        try mlx.check(mlx.mlx_array_set(&o, if (parts[0].ctx != null) parts[0] else parts[1]));
    }
    var ot = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ot);
    try mlx.check(mlx.mlx_transpose_axes(&ot, o, &[_]c_int{ 0, 2, 1, 3 }, 4, self.s));
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, ot, &[_]c_int{ batch, seq_len, heads * hd }, 3, self.s));
    const of = try glm5.withDependency(flat, view.v);
    defer _ = mlx.mlx_array_free(of);
    return self.qmatmul(of, d.o.w, d.o.s, d.o.b);
}

var glm_profile_env: ?bool = null;
const GlmProfKind = enum(u8) { collapse, kda, dsa, moe, expand };
var glm_prof_ns: [5]u64 = @splat(0);

fn glmProfNow() i128 {
    return @intCast(std.Io.Timestamp.now(std.Io.Threaded.global_single_threaded.io(), .boot).nanoseconds);
}

fn glmProfLap(a: mlx.mlx_array, kind: GlmProfKind, t0: i128) i128 {
    _ = mlx.mlx_array_eval(a);
    const now = glmProfNow();
    glm_prof_ns[@backingInt(kind)] += @intCast(now - t0);
    return now;
}

fn glmProfMs(kind: GlmProfKind) f64 {
    return @as(f64, @floatFromInt(glm_prof_ns[@backingInt(kind)])) / 1.0e6;
}

// ── tests ──

test "glm5_next fixture: one-shot and chunked prefill + decode vs mlx-vlm glm5_next (GLM5_MODEL, GLM5_FIXTURE)" {
    // A tiny random GLM-5-Next in the published layout (tests/dump_glm5_next_fixtures.py);
    // logits from mlx-vlm's glm5_next in bf16. 40 tokens in 4-token pools against a
    // 2-pool budget: the indexer selects, so the sparse path runs. Bar: rows agree and no
    // DECIDED row (top-2 margin past `MIMO_DECIDED_MARGIN`) picks another token, over the
    // rows before the reference's first pool pick within 10% (`sel_gap`): bf16 flips that
    // pick and the KDA state carries the flip to every later row.
    const model_dir = std.c.getenv("GLM5_MODEL") orelse return error.SkipZigTest;
    const fixture_path = std.c.getenv("GLM5_FIXTURE") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var config = try model_mod.parseConfigLegacy(io, allocator, std.mem.span(model_dir));
    var weights = try model_mod.loadModelWeights(io, allocator, std.mem.span(model_dir), &config, false);
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, allocator, config, &weights);
    defer xfm.deinit();

    var fx = try model_mod.loadWeightsSingleFile(allocator, std.mem.span(fixture_path));
    defer fx.deinit();
    const ids_arr = fx.get("input_ids") orelse return error.MissingFixtureTensor;
    try mlx.check(mlx.mlx_array_eval(ids_arr));
    const T: c_int = @intCast(mlx.mlx_array_size(ids_arr));
    const v: usize = @intCast(config.vocab_size);
    const ref = try qwen4ReadF32(allocator, fx.get("logits_full") orelse return error.MissingFixtureTensor, s);
    defer allocator.free(ref);
    const gap = try qwen4ReadF32(allocator, fx.get("sel_gap") orelse return error.MissingFixtureTensor, s);
    defer allocator.free(gap);
    var scored: c_int = 0;
    while (scored < T and gap[@intCast(scored)] >= 0.10) scored += 1;
    // Rows from 12 on select 2 of 3+ pools: the decode window below must reach them.
    try testing.expect(scored >= 16);
    var flat_ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat_ids);
    try mlx.check(mlx.mlx_reshape(&flat_ids, ids_arr, &.{T}, 1, s));

    const Run = struct {
        fn rows(x: *Transformer, ids: mlx.mlx_array, from: c_int, to: c_int, st: mlx.mlx_stream) !mlx.mlx_array {
            var part = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(part);
            try mlx.check(mlx.mlx_slice(&part, ids, &.{from}, 1, &.{to}, 1, &.{1}, 1, st));
            var batched = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(batched);
            try mlx.check(mlx.mlx_reshape(&batched, part, &.{ 1, to - from }, 2, st));
            return x.forward(batched);
        }
    };

    const full = try Run.rows(&xfm, flat_ids, 0, T, s);
    defer _ = mlx.mlx_array_free(full);
    const ours_full = try qwen4ReadF32(allocator, full, s);
    defer allocator.free(ours_full);
    const n_scored: usize = @intCast(scored);
    const a = qwen4CompareRows(ours_full[0 .. n_scored * v], ref[0 .. n_scored * v], n_scored, v);
    std.debug.print("[glm5 fixture] one-shot {d} of {d} rows: min cos {d:.5}, argmax {d}/{d}\n", .{ scored, T, a.min_cos, a.argmax_agree, a.rows });
    try testing.expect(a.min_cos > 0.995 and mimoDecidedMisses(ours_full, ref, n_scored, v) == 0);

    try xfm.resetCache();
    const t_pre: c_int = scored - 6;
    for ([_][2]c_int{ .{ 0, 5 }, .{ 5, t_pre } }) |c| {
        const out = try Run.rows(&xfm, flat_ids, c[0], c[1], s);
        _ = mlx.mlx_array_free(out);
    }
    var worst: f64 = 1.0;
    var misses: usize = 0;
    var t = t_pre;
    while (t < scored) : (t += 1) {
        const out = try Run.rows(&xfm, flat_ids, t, t + 1, s);
        defer _ = mlx.mlx_array_free(out);
        const row = try qwen4ReadF32(allocator, out, s);
        defer allocator.free(row);
        const ref_row = ref[@as(usize, @intCast(t)) * v ..][0..v];
        worst = @min(worst, qwen4CompareRows(row, ref_row, 1, v).min_cos);
        misses += mimoDecidedMisses(row, ref_row, 1, v);
    }
    std.debug.print("[glm5 fixture] chunked prefill + {d} decode steps: min cos {d:.5}, decided misses {d}\n", .{ scored - t_pre, worst, misses });
    try testing.expect(worst > 0.995 and misses == 0);
}

test "glm5_next DFlash capture: the tapped layers' hyper-connection mean tracks the reference, one-shot and per decode step (GLM5_MODEL, GLM5_FIXTURE)" {
    // `supportsLayerCapture` promises the seam; a drafter fed a forward that ignores it reads nothing. The
    // tap is the MEAN of the four streams after the layer (SGLang's `hc_contract`): cosine alone cannot see a
    // sum or a single stream scaled to match, so the RMS ratio is held beside it. The decode steps ride
    // the one-token path, where the layer's expand is deferred into the next sublayer's read.
    const model_dir = std.c.getenv("GLM5_MODEL") orelse return error.SkipZigTest;
    const fixture_path = std.c.getenv("GLM5_FIXTURE") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var config = try model_mod.parseConfigLegacy(io, allocator, std.mem.span(model_dir));
    var weights = try model_mod.loadModelWeights(io, allocator, std.mem.span(model_dir), &config, false);
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, allocator, config, &weights);
    defer xfm.deinit();
    try testing.expect(xfm.supportsLayerCapture());

    var fx = try model_mod.loadWeightsSingleFile(allocator, std.mem.span(fixture_path));
    defer fx.deinit();
    const ids_arr = fx.get("input_ids") orelse return error.MissingFixtureTensor;
    const cap_ids_arr = fx.get("cap_ids") orelse return error.MissingFixtureTensor;
    try mlx.check(mlx.mlx_array_eval(ids_arr));
    try mlx.check(mlx.mlx_array_eval(cap_ids_arr));
    const T: c_int = @intCast(mlx.mlx_array_size(ids_arr));
    const h: usize = @intCast(config.hidden_size);
    const n_cap: usize = mlx.mlx_array_size(cap_ids_arr);
    const cap_ids = try allocator.alloc(u32, n_cap);
    defer allocator.free(cap_ids);
    const cap_raw = mlx.mlx_array_data_int32(cap_ids_arr) orelse return error.MlxArrayDataNull;
    for (cap_ids, cap_raw[0..n_cap]) |*d, v| d.* = @intCast(v);
    const refs = try allocator.alloc([]f32, n_cap);
    defer allocator.free(refs);
    var made: usize = 0;
    defer for (refs[0..made]) |r| allocator.free(r);
    var key_buf: [32]u8 = undefined;
    for (cap_ids, refs) |id, *r| {
        const key = try std.fmt.bufPrint(&key_buf, "cap_l{d}", .{id});
        r.* = try qwen4ReadF32(allocator, fx.get(key) orelse return error.MissingFixtureTensor, s);
        made += 1;
    }
    // Rows before the reference's first pool pick within 10% (the KDA state carries a flipped pick forward).
    const gap = try qwen4ReadF32(allocator, fx.get("sel_gap") orelse return error.MissingFixtureTensor, s);
    defer allocator.free(gap);
    var scored: c_int = 0;
    while (scored < T and !(gap[@intCast(scored)] < 0.10)) scored += 1;
    try testing.expect(scored >= 16);

    var flat_ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat_ids);
    try mlx.check(mlx.mlx_reshape(&flat_ids, ids_arr, &.{T}, 1, s));
    const Run = struct {
        /// Forward rows [from, to) with the capture armed; hands back each tapped layer's `[1, to-from, H]` rows on the host.
        fn rows(x: *Transformer, a: std.mem.Allocator, ids: mlx.mlx_array, ids_tapped: []const u32, from: c_int, to: c_int, st: mlx.mlx_stream) ![][]f32 {
            var part = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(part);
            try mlx.check(mlx.mlx_slice(&part, ids, &.{from}, 1, &.{to}, 1, &.{1}, 1, st));
            var batched = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(batched);
            try mlx.check(mlx.mlx_reshape(&batched, part, &.{ 1, to - from }, 2, st));
            const out = try a.alloc(mlx.mlx_array, ids_tapped.len);
            defer {
                for (out) |o| _ = mlx.mlx_array_free(o);
                a.free(out);
            }
            for (out) |*o| o.* = mlx.mlx_array_new();
            var cl = CaptureLayers{ .ids = ids_tapped, .out = out };
            var ctx = x.defaultCtx();
            ctx.capture_layers = &cl;
            const logits = try x.forwardWith(&ctx, batched);
            _ = mlx.mlx_array_free(logits);
            const host = try a.alloc([]f32, ids_tapped.len);
            var done: usize = 0;
            errdefer {
                for (host[0..done]) |r| a.free(r);
                a.free(host);
            }
            for (out, host) |o, *dst| {
                if (o.ctx == null) return error.DflashCaptureMissing;
                dst.* = try qwen4ReadF32(a, o, st);
                done += 1;
            }
            return host;
        }
        fn free(a: std.mem.Allocator, host: [][]f32) void {
            for (host) |r| a.free(r);
            a.free(host);
        }
    };
    const Bar = struct {
        fn check(label: []const u8, got: []const f32, want: []const f32, n_rows: usize, width: usize) !void {
            const cmp = qwen4CompareRows(got[0 .. n_rows * width], want[0 .. n_rows * width], n_rows, width);
            var gg: f64 = 0;
            var ww: f64 = 0;
            for (got[0 .. n_rows * width], want[0 .. n_rows * width]) |x, y| {
                gg += @as(f64, x) * x;
                ww += @as(f64, y) * y;
            }
            const rms_ratio = @sqrt(gg / (ww + 1e-30));
            std.debug.print("[glm5 dflash capture] {s}: min cos {d:.5}, rms ratio {d:.4}\n", .{ label, cmp.min_cos, rms_ratio });
            try testing.expect(cmp.min_cos > 0.998);
            try testing.expect(rms_ratio > 0.97 and rms_ratio < 1.03);
        }
    };

    // [a] the whole prefix in one forward.
    const full = try Run.rows(&xfm, allocator, flat_ids, cap_ids, 0, scored, s);
    defer Run.free(allocator, full);
    var label_buf: [48]u8 = undefined;
    for (cap_ids, full, refs) |id, got, want| {
        try Bar.check(try std.fmt.bufPrint(&label_buf, "one-shot layer {d}", .{id}), got, want, @intCast(scored), h);
    }

    // [b] a chunked prefill, then one token at a time (deferred expands materialized at the tap).
    try xfm.resetCache();
    const t_pre: c_int = scored - 6;
    for ([_][2]c_int{ .{ 0, 5 }, .{ 5, t_pre } }) |c| {
        const part = try Run.rows(&xfm, allocator, flat_ids, cap_ids, c[0], c[1], s);
        defer Run.free(allocator, part);
        for (cap_ids, part, refs) |id, got, want| {
            try Bar.check(try std.fmt.bufPrint(&label_buf, "chunk {d}..{d} layer {d}", .{ c[0], c[1], id }), got, want[@as(usize, @intCast(c[0])) * h ..], @intCast(c[1] - c[0]), h);
        }
    }
    var t = t_pre;
    while (t < scored) : (t += 1) {
        const step = try Run.rows(&xfm, allocator, flat_ids, cap_ids, t, t + 1, s);
        defer Run.free(allocator, step);
        for (cap_ids, step, refs) |id, got, want| {
            try Bar.check(try std.fmt.bufPrint(&label_buf, "decode row {d} layer {d}", .{ t, id }), got, want[@as(usize, @intCast(t)) * h ..], 1, h);
        }
    }
}

test "glm5_next verify windows of 2..16 rows track the same rows decoded one at a time (GLM5_REAL_MODEL)" {
    // The fused row paths (hcPre, routed experts, KDA, latent attention) against the one-row paths on the
    // real pack: a window's logits are the serial logits to rounding, and its argmax is the serial argmax.
    const model_dir = std.c.getenv("GLM5_REAL_MODEL") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try model_mod.parseConfigLegacy(io, allocator, std.mem.span(model_dir));
    var weights = try model_mod.loadWeights(io, allocator, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, allocator, config, &weights);
    defer xfm.deinit();
    const v: usize = @intCast(config.vocab_size);

    var prompt: [48]i32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(1000 + (i * 7919) % 20000);
    const Run = struct {
        fn feed(x: *Transformer, ids: []const i32, verify: bool, st: mlx.mlx_stream) !mlx.mlx_array {
            const input = mlx.mlx_array_new_data(ids.ptr, &[_]c_int{ 1, @intCast(ids.len) }, 2, .int32);
            defer _ = mlx.mlx_array_free(input);
            var ctx = x.defaultCtx();
            ctx.capture_ssm_seq = verify and x.ssm_entries != null;
            ctx.spec_verify = verify;
            defer if (ctx.ssm_entries) |entries| for (entries) |*e| ssmFreeSpecCapture(e);
            const logits = try x.forwardWith(&ctx, input);
            try mlx.check(mlx.mlx_array_eval(logits));
            _ = st;
            return logits;
        }
    };
    // A short tail chunk of a prefix-cache restore is a prefill, not a verify: both modes ride the same rows.
    for ([_]bool{ true, false }) |mode_verify| {
    var w: usize = 2;
    while (w <= 16) : (w += 1) {
        const window = [_]i32{ 5, 90, 1234, 77, 4096, 300, 12, 8001, 999, 42, 7000, 15, 2500, 63, 18000, 11 };
        // The window in one forward.
        try xfm.resetCache();
        _ = mlx.mlx_array_free(try Run.feed(&xfm, &prompt, false, s));
        const got_arr = try Run.feed(&xfm, window[0..w], mode_verify, s);
        defer _ = mlx.mlx_array_free(got_arr);
        const got = try qwen4ReadF32(allocator, got_arr, s);
        defer allocator.free(got);
        // The same rows one at a time.
        try xfm.resetCache();
        _ = mlx.mlx_array_free(try Run.feed(&xfm, &prompt, false, s));
        var worst: f64 = 1.0;
        var misses: usize = 0;
        for (0..w) |r| {
            const one = try Run.feed(&xfm, window[r .. r + 1], false, s);
            defer _ = mlx.mlx_array_free(one);
            const want = try qwen4ReadF32(allocator, one, s);
            defer allocator.free(want);
            const cmp = qwen4CompareRows(got[r * v .. (r + 1) * v], want[0..v], 1, v);
            worst = @min(worst, cmp.min_cos);
            misses += mimoDecidedMisses(got[r * v .. (r + 1) * v], want[0..v], 1, v);
        }
        std.debug.print("[glm5 window] {d} rows: min cos {d:.5}, decided misses {d}\n", .{ w, worst, misses });
        // Random ids leave the logits flat and a verify row is never serial's to the bit (0.97 at 4 to 8
        // rows before the wide windows existed): a broken row path lands far below this.
        try testing.expect(worst > 0.94);
    }
    }
}

test "glm5_next batched decode: a three-slot tick tracks three serial decodes (GLM5_MODEL, GLM5_FIXTURE)" {
    const model_dir = std.c.getenv("GLM5_MODEL") orelse return error.SkipZigTest;
    const fixture_path = std.c.getenv("GLM5_FIXTURE") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try model_mod.parseConfigLegacy(io, allocator, std.mem.span(model_dir));
    var weights = try model_mod.loadModelWeights(io, allocator, std.mem.span(model_dir), &config, false);
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, allocator, config, &weights);
    defer xfm.deinit();
    try testing.expect(xfm.supportsBatchedGdnDecode());

    var fx = try model_mod.loadWeightsSingleFile(allocator, std.mem.span(fixture_path));
    defer fx.deinit();
    const ids_arr = fx.get("input_ids") orelse return error.MissingFixtureTensor;
    try mlx.check(mlx.mlx_array_eval(ids_arr));
    const ids = (mlx.mlx_array_data_int32(ids_arr) orelse return error.Unreadable)[0..mlx.mlx_array_size(ids_arr)];
    const v: usize = @intCast(config.vocab_size);
    const ref = try qwen4ReadF32(allocator, fx.get("logits_full") orelse return error.MissingFixtureTensor, s);
    defer allocator.free(ref);

    // Three slots on shifted windows of the fixture's sequence, each prefilled alone; the second
    // set of three decodes in one tick. Slot 0 follows the fixture, so it has a reference row.
    const n_slots = 3;
    const prompt = 6;
    const steps = 8;
    var serial: [n_slots]*Qwen4TestSlot = undefined;
    var batched: [n_slots]*Qwen4TestSlot = undefined;
    for (0..n_slots) |i| {
        serial[i] = try Qwen4TestSlot.init(allocator, config.num_hidden_layers);
        batched[i] = try Qwen4TestSlot.init(allocator, config.num_hidden_layers);
        for ([_]*Qwen4TestSlot{ serial[i], batched[i] }) |sl| _ = mlx.mlx_array_free(try sl.forward(&xfm, ids[i .. i + prompt]));
    }
    defer for (0..n_slots) |i| {
        serial[i].deinit(allocator);
        batched[i].deinit(allocator);
    };
    var ctxs: [n_slots]*ForwardCtx = undefined;
    for (&ctxs, batched) |*c, sl| c.* = &sl.ctx;

    var cos_vs_serial: f64 = 1.0;
    var cos_serial_ref: f64 = 1.0;
    var cos_batched_ref: f64 = 1.0;
    for (0..steps) |k| {
        var toks: [n_slots]u32 = undefined;
        var offs: [n_slots]u32 = undefined;
        for (0..n_slots) |i| {
            toks[i] = @intCast(ids[i + prompt + k]);
            offs[i] = @intCast(batched[i].off);
        }
        try testing.expect(xfm.batchedGdnReady(&ctxs));
        var hidden: ?[]mlx.mlx_array = null;
        defer if (hidden) |h| {
            for (h) |a| _ = mlx.mlx_array_free(a);
            allocator.free(h);
        };
        const out = try xfm.forwardMoeBatchedDecode(&toks, &ctxs, &offs, &hidden);
        defer {
            for (out) |o| _ = mlx.mlx_array_free(o);
            allocator.free(out);
        }
        try testing.expectEqual(@as(usize, n_slots), hidden.?.len);
        for (batched) |sl| sl.off += 1; // the scheduler advances each slot after the tick
        for (0..n_slots) |i| {
            const alone = try serial[i].forward(&xfm, &.{@intCast(toks[i])});
            defer _ = mlx.mlx_array_free(alone);
            const sv = try qwen4ReadF32(allocator, alone, s);
            defer allocator.free(sv);
            const bv = try qwen4ReadF32(allocator, out[i], s);
            defer allocator.free(bv);
            cos_vs_serial = @min(cos_vs_serial, qwen4CompareRows(bv, sv, 1, v).min_cos);
            if (i == 0) {
                const r = ref[(prompt + k) * v ..][0..v];
                cos_serial_ref = @min(cos_serial_ref, qwen4CompareRows(sv, r, 1, v).min_cos);
                cos_batched_ref = @min(cos_batched_ref, qwen4CompareRows(bv, r, 1, v).min_cos);
            }
        }
    }
    std.debug.print("[glm5 batched] {d} ticks x {d} slots: min cos batched-vs-serial {d:.5}; vs fp32 reference serial {d:.5} batched {d:.5}\n", .{ steps, n_slots, cos_vs_serial, cos_serial_ref, cos_batched_ref });
    try testing.expect(cos_vs_serial > 0.999 and cos_batched_ref > cos_serial_ref - 0.001);
}

test "glm5_next: oMLX 0.7.1's layout (one fused conv1d, forget_gate.*) gives the published layout's logits bit for bit (GLM5_MODEL)" {
    const model_dir = std.c.getenv("GLM5_MODEL") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    const config = try model_mod.parseConfigLegacy(io, allocator, std.mem.span(model_dir));

    const Logits = struct {
        fn of(io_: std.Io, alloc: std.mem.Allocator, cfg_in: model_mod.ModelConfig, dir: []const u8, omlx: bool, st: mlx.mlx_stream) ![]f32 {
            var cfg = cfg_in;
            var w = try model_mod.loadModelWeights(io_, alloc, dir, &cfg, false);
            defer w.deinit();
            model_mod.resolveWeightPrefix(&cfg, &w);
            if (omlx) for (0..cfg.num_hidden_layers) |li| {
                var nb: [256]u8 = undefined;
                var convs: [3]mlx.mlx_array = undefined;
                const bases = [_][]const u8{ "q", "k", "v" };
                for (bases, 0..) |b, n| {
                    const key = try std.fmt.bufPrint(&nb, "{s}.layers.{d}.self_attn.{s}_conv1d.weight", .{ cfg.weight_prefix, li, b });
                    convs[n] = w.get(key) orelse break;
                } else {
                    // oMLX stores the joined conv in MLX's [C, K, 1].
                    const joined = try concatRows(&convs, st);
                    defer _ = mlx.mlx_array_free(joined);
                    var t = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(t);
                    try mlx.check(mlx.mlx_swapaxes(&t, joined, 1, 2, st));
                    const fused = try materializedOwnedCopy(st, t);
                    try w.map.put(try alloc.dupe(u8, try std.fmt.bufPrint(&nb, "{s}.layers.{d}.self_attn.conv1d.weight", .{ cfg.weight_prefix, li })), fused);
                    for (bases) |b| w.remove(try std.fmt.bufPrint(&nb, "{s}.layers.{d}.self_attn.{s}_conv1d.weight", .{ cfg.weight_prefix, li, b }));
                    for ([_][]const u8{ "A_log", "dt_bias", "f_a_proj.weight", "f_a_proj.scales", "f_a_proj.biases", "f_b_proj.weight", "f_b_proj.scales", "f_b_proj.biases" }) |leaf| {
                        const kv = w.map.fetchRemove(try std.fmt.bufPrint(&nb, "{s}.layers.{d}.self_attn.{s}", .{ cfg.weight_prefix, li, leaf })) orelse continue;
                        alloc.free(kv.key);
                        try w.map.put(try alloc.dupe(u8, try std.fmt.bufPrint(&nb, "{s}.layers.{d}.self_attn.forget_gate.{s}", .{ cfg.weight_prefix, li, leaf })), kv.value);
                    }
                }
            };
            var xfm = try Transformer.init(io_, alloc, cfg, &w);
            defer xfm.deinit();
            var ids = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ids);
            try mlx.check(mlx.mlx_arange(&ids, 1, 25, 1, .int32, st));
            var batched = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(batched);
            try mlx.check(mlx.mlx_reshape(&batched, ids, &.{ 1, 24 }, 2, st));
            const out = try xfm.forward(batched);
            defer _ = mlx.mlx_array_free(out);
            return qwen4ReadF32(alloc, out, st);
        }
    };
    const published = try Logits.of(io, allocator, config, std.mem.span(model_dir), false, s);
    defer allocator.free(published);
    const omlx = try Logits.of(io, allocator, config, std.mem.span(model_dir), true, s);
    defer allocator.free(omlx);
    try testing.expectEqualSlices(f32, published, omlx);
}

test "glm5_next MTP fixture: the pack's mtp.0 layer drafts as mlx-vlm's drafter, one-shot and row by row (GLM5_MODEL, GLM5_MTP_FIXTURE)" {
    const model_dir = std.c.getenv("GLM5_MODEL") orelse return error.SkipZigTest;
    const fixture_path = std.c.getenv("GLM5_MTP_FIXTURE") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var config = try model_mod.parseConfigLegacy(io, allocator, std.mem.span(model_dir));
    var weights = try model_mod.loadWeights(io, allocator, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, allocator, config, &weights);
    defer xfm.deinit();
    var head = (try glm_mtp.Head.load(allocator, s, &xfm.config, &weights)) orelse return error.MissingWeight;
    defer head.deinit();

    var fx = try model_mod.loadWeightsSingleFile(allocator, std.mem.span(fixture_path));
    defer fx.deinit();
    const ids_arr = fx.get("input_ids") orelse return error.MissingFixtureTensor;
    const ref_arr = fx.get("mtp_logits") orelse return error.MissingFixtureTensor;
    const T: c_int = mlx.getShape(ids_arr)[1];
    const R: c_int = T - 1; // head rows: (hidden p, token p+1) for p < T-1
    const v: usize = @intCast(config.vocab_size);
    const hsz: c_int = @intCast(config.hidden_size);
    try testing.expectEqual(R, mlx.getShape(ref_arr)[1]);
    const ref = try qwen4ReadF32(allocator, ref_arr, s);
    defer allocator.free(ref);
    // The head's own indexer pick margins per row: a near tie flips under bf16, and with
    // no recurrence behind it the flip stays on its row, so such rows are skipped, not cut.
    const gap = try qwen4ReadF32(allocator, fx.get("sel_gap") orelse return error.MissingFixtureTensor, s);
    defer allocator.free(gap);
    const Scored = struct {
        fn row(g: []const f32, i: usize) bool {
            return g[i] >= 0.10;
        }
    };

    // The REFERENCE trunk's final-normed rows: our trunk's differ on its near-tie indexer
    // picks (see the trunk fixture test), and this test scores the head alone.
    const hidden_all = fx.get("hidden") orelse return error.MissingFixtureTensor;
    var hid = mlx.mlx_array_new(); // [1, R, H]: rows 0..R-1
    defer _ = mlx.mlx_array_free(hid);
    try mlx.check(mlx.mlx_slice(&hid, hidden_all, &.{ 0, 0, 0 }, 3, &.{ 1, R, hsz }, 3, &.{ 1, 1, 1 }, 3, s));
    var next_ids = mlx.mlx_array_new(); // [R]: tokens 1..T-1
    defer _ = mlx.mlx_array_free(next_ids);
    {
        var sl = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sl);
        try mlx.check(mlx.mlx_slice(&sl, ids_arr, &.{ 0, 1 }, 2, &.{ 1, T }, 2, &.{ 1, 1 }, 2, s));
        try mlx.check(mlx.mlx_reshape(&next_ids, sl, &.{R}, 1, s));
    }
    const Rows = struct {
        fn of(a: mlx.mlx_array, axis: usize, from: c_int, to: c_int, st: mlx.mlx_stream) !mlx.mlx_array {
            const sh = mlx.getShape(a);
            var lo: [3]c_int = @splat(0);
            var hi: [3]c_int = undefined;
            for (sh, 0..) |d, i| hi[i] = d;
            lo[axis] = from;
            hi[axis] = to;
            var out = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&out, a, &lo, sh.len, &hi, sh.len, &[_]c_int{ 1, 1, 1 }, sh.len, st));
            return out;
        }
    };

    // One shot over every row.
    var cache = try head.makeCache(allocator);
    defer cache.deinit();
    const out = try head.rows(&xfm, &cache, next_ids, hid);
    defer _ = mlx.mlx_array_free(out);
    var normed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(normed);
    try mlx.check(mlx.mlx_fast_rms_norm(&normed, out, head.norm, config.rms_norm_eps, s));
    const logits = try xfm.lmHeadForDraft(normed);
    defer _ = mlx.mlx_array_free(logits);
    const ours = try qwen4ReadF32(allocator, logits, s);
    defer allocator.free(ours);
    const n_rows: usize = @intCast(R);
    var scored: usize = 0;
    var min_cos: f64 = 1.0;
    var misses: usize = 0;
    for (0..n_rows) |ri| {
        if (!Scored.row(gap, ri)) continue;
        scored += 1;
        min_cos = @min(min_cos, qwen4CompareRows(ours[ri * v ..][0..v], ref[ri * v ..][0..v], 1, v).min_cos);
        misses += mimoDecidedMisses(ours[ri * v ..][0..v], ref[ri * v ..][0..v], 1, v);
    }
    std.debug.print("[glm5 mtp fixture] one-shot {d} of {d} rows: min cos {d:.5}, decided misses {d}\n", .{ scored, R, min_cos, misses });
    const one_shot_ok = scored >= 24 and min_cos > 0.995 and misses == 0;

    // History then one row at a time: the decode path (one query, pooled keys, cache offset).
    var cache2 = try head.makeCache(allocator);
    defer cache2.deinit();
    const t_pre: c_int = R - 8;
    {
        const ids_pre = try Rows.of(next_ids, 0, 0, t_pre, s);
        defer _ = mlx.mlx_array_free(ids_pre);
        const hid_pre = try Rows.of(hid, 1, 0, t_pre, s);
        defer _ = mlx.mlx_array_free(hid_pre);
        const first = try head.forward(&xfm, &cache2, ids_pre, hid_pre, .none);
        _ = mlx.mlx_array_free(first.hidden_next);
    }
    var worst: f64 = 1.0;
    misses = 0;
    var t = t_pre;
    while (t < R) : (t += 1) {
        const id1 = try Rows.of(next_ids, 0, t, t + 1, s);
        defer _ = mlx.mlx_array_free(id1);
        const hid1 = try Rows.of(hid, 1, t, t + 1, s);
        defer _ = mlx.mlx_array_free(hid1);
        const step = try head.forward(&xfm, &cache2, id1, hid1, .logits);
        defer _ = mlx.mlx_array_free(step.logits);
        defer _ = mlx.mlx_array_free(step.hidden_next);
        const row = try qwen4ReadF32(allocator, step.logits, s);
        defer allocator.free(row);
        if (!Scored.row(gap, @intCast(t))) continue;
        const ref_row = ref[@as(usize, @intCast(t)) * v ..][0..v];
        worst = @min(worst, qwen4CompareRows(row, ref_row, 1, v).min_cos);
        misses += mimoDecidedMisses(row, ref_row, 1, v);
    }
    try testing.expectEqual(@as(usize, @intCast(R)), cache2.seqLen(0));
    std.debug.print("[glm5 mtp fixture] history + {d} single rows: min cos {d:.5}, decided misses {d}\n", .{ R - t_pre, worst, misses });
    try testing.expect(one_shot_ok and worst > 0.995 and misses == 0);
}
