//! sdk/blocks/ssm/mamba2, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns mamba2_fused_logged, mamba2_fused_max_rows, Mamba2Weights, mamba2Mixer, mamba2GroupNorm.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const mamba2_decode = @import("mamba2_decode.zig");
const log = @import("../../../host/log.zig");
const ModelConfig = @import("../../../engines/mlx/model.zig").ModelConfig;
const Weights = @import("../../../engines/mlx/model.zig").Weights;
const RECURRENCE_EVAL_INTERVAL = @import("../../../engines/mlx/eval_cadence.zig").RECURRENCE_EVAL_INTERVAL;
const deinitTestHybridXfm = @import("../../../engines/mlx/forward/hybrid.zig").deinitTestHybridXfm;
const testHybridXfm = @import("../../../engines/mlx/forward/hybrid.zig").testHybridXfm;
const KVCache = @import("../../../engines/mlx/kv/cache.zig").KVCache;
const initHybridLayers = @import("../../../engines/mlx/load/layers_hybrid.zig").initHybridLayers;
const SSMCacheEntry = @import("../../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const ssmFreeSpecCapture = @import("../../../engines/mlx/ssm/entry.zig").ssmFreeSpecCapture;
const Transformer = @import("../../../engines/mlx/transformer.zig").Transformer;

var mamba2_fused_logged = false;
/// Widest window the fused Mamba2 step serves; tests lower it to force the chain.
pub var mamba2_fused_max_rows: c_int = std.math.maxInt(c_int);

pub const Mamba2Weights = struct {
    in_proj_w: mlx.mlx_array,
    in_proj_s: mlx.mlx_array,
    in_proj_b: mlx.mlx_array,
    conv1d_w: mlx.mlx_array, // depthwise conv
    conv1d_b: ?mlx.mlx_array, // optional bias
    A_log: mlx.mlx_array, // static state matrix (log-space)
    D: mlx.mlx_array, // skip connection
    dt_bias: mlx.mlx_array, // time-step bias
    norm_w: mlx.mlx_array, // output normalization
    out_proj_w: mlx.mlx_array,
    out_proj_s: mlx.mlx_array,
    out_proj_b: mlx.mlx_array,
};

// ── Mamba2 SSM (Nemotron-H) ──

pub fn mamba2Mixer(
    self: *Transformer,
    x: mlx.mlx_array,
    mw: *const Mamba2Weights,
    ssm: *SSMCacheEntry,
    batch: c_int,
    seq_len: c_int,
) !mlx.mlx_array {
    const cfg = &self.config;
    const num_heads: c_int = @intCast(cfg.mamba_num_heads);
    const head_dim: c_int = @intCast(cfg.mamba_head_dim);
    const n_groups: c_int = @intCast(cfg.mamba_n_groups);
    const state_size: c_int = @intCast(cfg.ssm_state_size);
    const d_inner: c_int = num_heads * head_dim; // intermediate_size
    const conv_dim: c_int = d_inner + 2 * n_groups * state_size;
    const kernel: c_int = @intCast(cfg.mamba_conv_kernel);
    const repeats: c_int = @divExact(num_heads, n_groups);

    // 1. Input projection: [B, S, hidden] → [B, S, d_inner + conv_dim + num_heads]
    const proj = try self.qmatmul(x, mw.in_proj_w, mw.in_proj_s, mw.in_proj_b);
    defer _ = mlx.mlx_array_free(proj);

    // Any window (decode, spec verify, prefill): conv + SSM + gate is ONE
    // dispatch and the grouped norm a second (mamba2_decode.zig). The chain
    // below serves any geometry the kernels decline.
    if (seq_len <= mamba2_fused_max_rows and mw.conv1d_b != null) fused: {
        const act_dt = mlx.mlx_array_dtype(proj);
        var conv_in = ssm.conv_state;
        var conv_zero = mlx.mlx_array{ .ctx = null };
        defer if (conv_zero.ctx != null) {
            _ = mlx.mlx_array_free(conv_zero);
        };
        if (!ssm.initialized or conv_in.ctx == null) {
            conv_zero = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&conv_zero, &[_]c_int{ batch, kernel - 1, conv_dim }, 3, act_dt, self.s));
            conv_in = conv_zero;
        }
        if (ssm.ssm_state.ctx == null) {
            ssm.ssm_state = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&ssm.ssm_state, &[_]c_int{ batch, num_heads, head_dim, state_size }, 4, .float32, self.s));
        }
        const lo = mlx.mlx_array_new_float(cfg.time_step_min);
        defer _ = mlx.mlx_array_free(lo);
        const hi = mlx.mlx_array_new_float(cfg.time_step_max);
        defer _ = mlx.mlx_array_free(hi);
        const rows = mlx.mlx_array_new_int(seq_len);
        defer _ = mlx.mlx_array_free(rows);
        const capture = self.spec_capture_ssm;
        const out = (try mamba2_decode.step(
            .{ .batch = batch, .rows = seq_len, .heads = num_heads, .groups = n_groups, .head_dim = head_dim, .state = state_size, .conv_kernel = kernel, .proj = mlx.getShape(proj)[2] },
            .{ .proj = proj, .conv_state = conv_in, .ssm_state = ssm.ssm_state, .conv_w = mw.conv1d_w, .conv_b = mw.conv1d_b.?, .A_log = mw.A_log, .dt_bias = mw.dt_bias, .D = mw.D, .dt_lo = lo, .dt_hi = hi, .rows = rows, .all_rows = capture },
            self.s,
        )) orelse break :fused;
        defer _ = mlx.mlx_array_free(out.y);
        errdefer {
            _ = mlx.mlx_array_free(out.ssm_state);
            _ = mlx.mlx_array_free(out.conv_state);
        }
        if (capture) {
            // Same capture shapes as the chain: the conv input [B,(k-1)+T,CD]
            // and the state after every row [T,B,H,Dh,Ds].
            var x_conv = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x_conv);
            const strides3c = [_]c_int{ 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(&x_conv, proj, &[_]c_int{ 0, 0, d_inner }, 3, &[_]c_int{ batch, seq_len, d_inner + conv_dim }, 3, &strides3c, 3, self.s));
            const pair = [_]mlx.mlx_array{ conv_in, x_conv };
            const vec = mlx.mlx_vector_array_new_data(&pair, 2);
            defer _ = mlx.mlx_vector_array_free(vec);
            if (ssm.spec_conv_input.ctx != null) _ = mlx.mlx_array_free(ssm.spec_conv_input);
            ssm.spec_conv_input = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_concatenate_axis(&ssm.spec_conv_input, vec, 1, self.s));
            if (ssm.spec_state_seq.ctx != null) _ = mlx.mlx_array_free(ssm.spec_state_seq);
            ssm.spec_state_seq = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&ssm.spec_state_seq, out.ssm_state));
        }
        // Live state = the last row.
        var last = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(last);
        const srows = mlx.getShape(out.ssm_state)[0];
        const strides5 = [_]c_int{ 1, 1, 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&last, out.ssm_state, &[_]c_int{ srows - 1, 0, 0, 0, 0 }, 5, &[_]c_int{ srows, batch, num_heads, head_dim, state_size }, 5, &strides5, 5, self.s));
        var new_state = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&new_state, last, &[_]c_int{ batch, num_heads, head_dim, state_size }, 4, self.s));
        _ = mlx.mlx_array_free(out.ssm_state);
        _ = mlx.mlx_array_free(ssm.ssm_state);
        ssm.ssm_state = new_state;
        if (ssm.conv_state.ctx != null) _ = mlx.mlx_array_free(ssm.conv_state);
        ssm.conv_state = out.conv_state;
        ssm.initialized = true;
        if (!mamba2_fused_logged) {
            mamba2_fused_logged = true;
            log.info("[mamba2] fused step engaged: rows={d} H={d} Dh={d} Ds={d} G={d}\n", .{ seq_len, num_heads, head_dim, state_size, n_groups });
        }
        const weighted = try self.mamba2GroupNorm(out.y, mw, batch, seq_len, d_inner, n_groups);
        defer _ = mlx.mlx_array_free(weighted);
        return self.qmatmul(weighted, mw.out_proj_w, mw.out_proj_s, mw.out_proj_b);
    }

    // 2. Split: gate [d_inner], conv_input [conv_dim], dt [num_heads]
    const strides3 = [_]c_int{ 1, 1, 1 };
    var gate = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gate);
    try mlx.check(mlx.mlx_slice(&gate, proj, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ batch, seq_len, d_inner }, 3, &strides3, 3, self.s));

    var conv_input = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(conv_input);
    try mlx.check(mlx.mlx_slice(&conv_input, proj, &[_]c_int{ 0, 0, d_inner }, 3, &[_]c_int{ batch, seq_len, d_inner + conv_dim }, 3, &strides3, 3, self.s));

    var dt_raw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dt_raw);
    try mlx.check(mlx.mlx_slice(&dt_raw, proj, &[_]c_int{ 0, 0, d_inner + conv_dim }, 3, &[_]c_int{ batch, seq_len, d_inner + conv_dim + num_heads }, 3, &strides3, 3, self.s));

    // 3. Conv1d with cache + SiLU on conv_input
    const conv_out = try self.conv1dWithCache(conv_input, mw.conv1d_w, mw.conv1d_b, ssm, batch, conv_dim, kernel, true);
    defer _ = mlx.mlx_array_free(conv_out);

    // 4. Split conv output: x_ssm [d_inner], B [n_groups*state_size], C [n_groups*state_size]
    var x_ssm = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x_ssm);
    try mlx.check(mlx.mlx_slice(&x_ssm, conv_out, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ batch, seq_len, d_inner }, 3, &strides3, 3, self.s));

    const b_end: c_int = d_inner + n_groups * state_size;
    var B_proj = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(B_proj);
    try mlx.check(mlx.mlx_slice(&B_proj, conv_out, &[_]c_int{ 0, 0, d_inner }, 3, &[_]c_int{ batch, seq_len, b_end }, 3, &strides3, 3, self.s));

    var C_proj = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(C_proj);
    try mlx.check(mlx.mlx_slice(&C_proj, conv_out, &[_]c_int{ 0, 0, b_end }, 3, &[_]c_int{ batch, seq_len, conv_dim }, 3, &strides3, 3, self.s));

    // 5. Reshape to head format
    // x_ssm: [B, S, num_heads, head_dim]
    const x_shape = [_]c_int{ batch, seq_len, num_heads, head_dim };
    var x_h = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x_h);
    try mlx.check(mlx.mlx_reshape(&x_h, x_ssm, &x_shape, 4, self.s));

    // B, C: [B, S, n_groups, state_size]
    const bc_shape = [_]c_int{ batch, seq_len, n_groups, state_size };
    var B_h = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(B_h);
    var C_h = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(C_h);
    try mlx.check(mlx.mlx_reshape(&B_h, B_proj, &bc_shape, 4, self.s));
    try mlx.check(mlx.mlx_reshape(&C_h, C_proj, &bc_shape, 4, self.s));

    // Initialize SSM state if needed: [B, num_heads, head_dim, state_size].
    // conv1dWithCache already set ssm.initialized, so the empty handle is
    // the init indicator.
    if (ssm.ssm_state.ctx == null) {
        const state_shape = [_]c_int{ batch, num_heads, head_dim, state_size };
        ssm.ssm_state = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(&ssm.ssm_state, &state_shape, 4, .float32, self.s));
    }

    // y: [B, S, H*D] in the input dtype.
    const y_flat_shape = [_]c_int{ batch, seq_len, d_inner };
    var y_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y_flat);

    // 6. Compute dt = softplus(dt + dt_bias), clamp to time limits
    // Cast dt to float32 for precision (matching Python)
    var dt_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dt_f32);
    try mlx.check(mlx.mlx_astype(&dt_f32, dt_raw, .float32, self.s));
    // dt + dt_bias
    var dt_biased = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dt_biased);
    try mlx.check(mlx.mlx_add(&dt_biased, dt_f32, mw.dt_bias, self.s));
    // softplus: log1p(exp(x))
    var dt_exp_val = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dt_exp_val);
    try mlx.check(mlx.mlx_exp(&dt_exp_val, dt_biased, self.s));
    var dt_sp = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dt_sp);
    try mlx.check(mlx.mlx_log1p(&dt_sp, dt_exp_val, self.s));
    // Clamp (use float32 scalars matching dt precision)
    var dt_min_arr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dt_min_arr);
    {
        const v = mlx.mlx_array_new_float(cfg.time_step_min);
        defer _ = mlx.mlx_array_free(v);
        try mlx.check(mlx.mlx_astype(&dt_min_arr, v, .float32, self.s));
    }
    var dt_max_arr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dt_max_arr);
    {
        const v = mlx.mlx_array_new_float(cfg.time_step_max);
        defer _ = mlx.mlx_array_free(v);
        try mlx.check(mlx.mlx_astype(&dt_max_arr, v, .float32, self.s));
    }
    var dt_clamped_lo = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dt_clamped_lo);
    try mlx.check(mlx.mlx_maximum(&dt_clamped_lo, dt_sp, dt_min_arr, self.s));
    var dt_val = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dt_val);
    try mlx.check(mlx.mlx_minimum(&dt_val, dt_clamped_lo, dt_max_arr, self.s));

    // 7. A = -exp(A_log) — cast to float32 to match Python precision
    // Python's ssm_attn does: A = -mx.exp(A_log).astype(dt.dtype) where dt is float32.
    // Without this cast, A stays in BF16 and decay values dA = exp(A*dt) are imprecise,
    // compounding across 42 layers × N timesteps.
    var A_neg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(A_neg);
    {
        var A_exp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(A_exp);
        try mlx.check(mlx.mlx_exp(&A_exp, mw.A_log, self.s));
        var A_neg_bf16 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(A_neg_bf16);
        try mlx.check(mlx.mlx_negative(&A_neg_bf16, A_exp, self.s));
        try mlx.check(mlx.mlx_astype(&A_neg, A_neg_bf16, .float32, self.s));
    }

    // Precompute D reshaped for broadcasting: [H] → [H, 1]
    var D_bc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(D_bc);
    try mlx.check(mlx.mlx_expand_dims(&D_bc, mw.D, 1, self.s));

    // 9. Per-timestep SSM recurrence
    const T: usize = @intCast(seq_len);
    const out_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(out_vec);
    // Spec-decode capture: the state after every row, same [T,B,H,Dh,Ds]
    // the fused window records, so a partial accept rolls back on any geometry.
    const capture_states = self.spec_capture_ssm;
    const state_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(state_vec);

    for (0..T) |t| {
        const ti: c_int = @intCast(t);

        // Extract timestep slices
        const strides4 = [_]c_int{ 1, 1, 1, 1 };
        // dt_t: [B, 1, num_heads] → [B, num_heads]
        var dt_t_3d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dt_t_3d);
        try mlx.check(mlx.mlx_slice(&dt_t_3d, dt_val, &[_]c_int{ 0, ti, 0 }, 3, &[_]c_int{ batch, ti + 1, num_heads }, 3, &strides3, 3, self.s));
        var dt_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dt_t);
        {
            const dt_reshape = [_]c_int{ batch, num_heads };
            try mlx.check(mlx.mlx_reshape(&dt_t, dt_t_3d, &dt_reshape, 2, self.s));
        }

        // x_t: [B, num_heads, head_dim]
        var x_t_4d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_t_4d);
        try mlx.check(mlx.mlx_slice(&x_t_4d, x_h, &[_]c_int{ 0, ti, 0, 0 }, 4, &[_]c_int{ batch, ti + 1, num_heads, head_dim }, 4, &strides4, 4, self.s));
        var x_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_t);
        {
            const x_reshape = [_]c_int{ batch, num_heads, head_dim };
            try mlx.check(mlx.mlx_reshape(&x_t, x_t_4d, &x_reshape, 3, self.s));
        }

        // B_t: [B, n_groups, state_size] → repeat to [B, num_heads, state_size]
        var B_t_4d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(B_t_4d);
        try mlx.check(mlx.mlx_slice(&B_t_4d, B_h, &[_]c_int{ 0, ti, 0, 0 }, 4, &[_]c_int{ batch, ti + 1, n_groups, state_size }, 4, &strides4, 4, self.s));
        var B_t_sq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(B_t_sq);
        {
            const bc_rs = [_]c_int{ batch, n_groups, state_size };
            try mlx.check(mlx.mlx_reshape(&B_t_sq, B_t_4d, &bc_rs, 3, self.s));
        }
        var B_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(B_t);
        if (repeats > 1) {
            try mlx.check(mlx.mlx_repeat_axis(&B_t, B_t_sq, repeats, 1, self.s));
        } else {
            try mlx.check(mlx.mlx_array_set(&B_t, B_t_sq));
        }

        // C_t: same as B_t
        var C_t_4d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(C_t_4d);
        try mlx.check(mlx.mlx_slice(&C_t_4d, C_h, &[_]c_int{ 0, ti, 0, 0 }, 4, &[_]c_int{ batch, ti + 1, n_groups, state_size }, 4, &strides4, 4, self.s));
        var C_t_sq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(C_t_sq);
        {
            const bc_rs = [_]c_int{ batch, n_groups, state_size };
            try mlx.check(mlx.mlx_reshape(&C_t_sq, C_t_4d, &bc_rs, 3, self.s));
        }
        var C_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(C_t);
        if (repeats > 1) {
            try mlx.check(mlx.mlx_repeat_axis(&C_t, C_t_sq, repeats, 1, self.s));
        } else {
            try mlx.check(mlx.mlx_array_set(&C_t, C_t_sq));
        }

        // dA = exp(A * dt_t): [B, num_heads]
        var A_dt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(A_dt);
        try mlx.check(mlx.mlx_multiply(&A_dt, A_neg, dt_t, self.s));
        var dA = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dA);
        try mlx.check(mlx.mlx_exp(&dA, A_dt, self.s));

        // dA_expanded: [B, num_heads, 1, 1] for state broadcast
        var dA_e1 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dA_e1);
        var dA_exp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dA_exp);
        try mlx.check(mlx.mlx_expand_dims(&dA_e1, dA, 2, self.s));
        try mlx.check(mlx.mlx_expand_dims(&dA_exp, dA_e1, 3, self.s));

        // Decay state: state *= dA
        var decayed = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(decayed);
        try mlx.check(mlx.mlx_multiply(&decayed, ssm.ssm_state, dA_exp, self.s));

        // dtx = x_t * dt_t: [B, num_heads, head_dim]
        var dt_exp2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dt_exp2);
        try mlx.check(mlx.mlx_expand_dims(&dt_exp2, dt_t, 2, self.s)); // [B, num_heads, 1]
        var dtx = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dtx);
        try mlx.check(mlx.mlx_multiply(&dtx, x_t, dt_exp2, self.s));

        // Outer product update: dtx[..., :, None] * B_t[..., None, :]
        // dtx: [B, H, D] → [B, H, D, 1]
        // B_t: [B, H, S] → [B, H, 1, S]
        var dtx_e = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dtx_e);
        try mlx.check(mlx.mlx_expand_dims(&dtx_e, dtx, 3, self.s));
        var B_t_e = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(B_t_e);
        try mlx.check(mlx.mlx_expand_dims(&B_t_e, B_t, 2, self.s));
        var update = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(update);
        try mlx.check(mlx.mlx_multiply(&update, dtx_e, B_t_e, self.s));

        // new_state = decayed + update
        var new_state = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&new_state, decayed, update, self.s));
        _ = mlx.mlx_array_free(ssm.ssm_state);
        ssm.ssm_state = new_state;
        if (capture_states) _ = mlx.mlx_vector_array_append_value(state_vec, new_state);

        // Output: y = sum_s(state * C_t) + x * D
        // C_t: [B, H, S] → [B, H, 1, S]
        var C_t_e = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(C_t_e);
        try mlx.check(mlx.mlx_expand_dims(&C_t_e, C_t, 2, self.s));
        // state * C_t_e: [B, H, D, S]
        var state_c = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(state_c);
        try mlx.check(mlx.mlx_multiply(&state_c, ssm.ssm_state, C_t_e, self.s));
        // sum over S: [B, H, D]
        var y_state = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(y_state);
        try mlx.check(mlx.mlx_sum_axis(&y_state, state_c, -1, false, self.s));

        // D * x: [B, H, D] where D_bc is [H, 1], x_t is [B, H, D]
        var dx = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dx);
        try mlx.check(mlx.mlx_multiply(&dx, x_t, D_bc, self.s));

        // y_t = y_state + dx
        var y_t = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&y_t, y_state, dx, self.s));
        _ = mlx.mlx_vector_array_append_value(out_vec, y_t);
        _ = mlx.mlx_array_free(y_t);

        // Cadence eval only: a per-layer eval here is a GPU sync on
        // every decode token (23 per token on Nemotron-3.5).
        if ((t + 1) % RECURRENCE_EVAL_INTERVAL == 0) {
            try mlx.check(mlx.mlx_array_eval(ssm.ssm_state));
        }
    }

    if (capture_states) {
        if (ssm.spec_state_seq.ctx != null) _ = mlx.mlx_array_free(ssm.spec_state_seq);
        ssm.spec_state_seq = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_stack_axis(&ssm.spec_state_seq, state_vec, 0, self.s));
    }

    // 10. Stack: [T, B, H, D] → transpose to [B, T, H, D]
    var stacked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(stacked);
    try mlx.check(mlx.mlx_stack_axis(&stacked, out_vec, 0, self.s));
    const perm_tbhd = [_]c_int{ 1, 0, 2, 3 };
    var y_bthd = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y_bthd);
    try mlx.check(mlx.mlx_transpose_axes(&y_bthd, stacked, &perm_tbhd, 4, self.s));

    // Flatten heads: [B, T, H, D] → [B, T, H*D], back in the input dtype.
    // The state is f32, so y is too; mlx-lm's `ssm_attn` returns
    // `y.astype(x.dtype)`. Without it the residual stream widens to f32
    // here and every later layer reads its weights wider.
    var y_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y_f32);
    try mlx.check(mlx.mlx_reshape(&y_f32, y_bthd, &y_flat_shape, 3, self.s));
    try mlx.check(mlx.mlx_astype(&y_flat, y_f32, mlx.mlx_array_dtype(x_h), self.s));

    // 11. MambaRMSNormGated: silu(gate) * y, then group RMS norm, then weight
    const gated = try self.swiglu(gate, y_flat);
    defer _ = mlx.mlx_array_free(gated);
    const weighted = try self.mamba2GroupNorm(gated, mw, batch, seq_len, d_inner, n_groups);
    defer _ = mlx.mlx_array_free(weighted);

    // 12. Output projection
    return self.qmatmul(weighted, mw.out_proj_w, mw.out_proj_s, mw.out_proj_b);
}

/// Group RMS norm over `group_size = d_inner / n_groups` (NOT head_dim),
/// times the norm weight: the fused kernel where it applies, else
/// reshape -> parameter-free rms_norm -> weight.
pub fn mamba2GroupNorm(self: *Transformer, gated: mlx.mlx_array, mw: *const Mamba2Weights, batch: c_int, seq_len: c_int, d_inner: c_int, n_groups: c_int) !mlx.mlx_array {
    const cfg = &self.config;
    {
        const eps = mlx.mlx_array_new_float(cfg.rms_norm_eps);
        defer _ = mlx.mlx_array_free(eps);
        if (try mamba2_decode.groupNorm(gated, mw.norm_w, eps, n_groups, self.s)) |out| return out;
    }
    const group_size: c_int = @divExact(d_inner, n_groups);
    const gated_shape = [_]c_int{ batch, seq_len, n_groups, group_size };
    var gated_grouped = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gated_grouped);
    try mlx.check(mlx.mlx_reshape(&gated_grouped, gated, &gated_shape, 4, self.s));
    var normed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(normed);
    {
        const ones_shape = [_]c_int{group_size};
        var ones_w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ones_w);
        try mlx.check(mlx.mlx_ones(&ones_w, &ones_shape, 1, mlx.mlx_array_dtype(gated), self.s));
        try mlx.check(mlx.mlx_fast_rms_norm(&normed, gated_grouped, ones_w, cfg.rms_norm_eps, self.s));
    }
    const flat_shape = [_]c_int{ batch, seq_len, d_inner };
    var normed_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(normed_flat);
    try mlx.check(mlx.mlx_reshape(&normed_flat, normed, &flat_shape, 3, self.s));
    var weighted = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&weighted, normed_flat, mw.norm_w, self.s));
    return weighted;
}

// ── tests ──

// Bar: a Mamba2 layer returns the residual stream in its INPUT dtype. The SSM
// state is f32 by design (mlx-lm `ssm_attn` returns `y.astype(x.dtype)`);
// without that cast one Mamba2 layer promotes every later layer to f32.
test "mamba2Mixer keeps a bf16 residual stream bf16" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const t = std.testing;
    const allocator = t.allocator;
    const s = mlx.gpuStream();
    const H: c_int = 64;
    const heads: c_int = 4;
    const head_dim: c_int = 16;
    const state: c_int = 16;
    const kernel: c_int = 4;
    const d_inner = heads * head_dim;
    const conv_dim = d_inner + 2 * state;

    var w = Weights.init(allocator);
    defer w.deinit();
    const put = struct {
        fn add(weights: *Weights, alloc: std.mem.Allocator, name: []const u8, shape: []const c_int, v: f32, st: mlx.mlx_stream) !void {
            const val = mlx.mlx_array_new_float(v);
            defer _ = mlx.mlx_array_free(val);
            var arr = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_full(&arr, shape.ptr, shape.len, val, .bfloat16, st));
            try weights.map.put(try alloc.dupe(u8, name), arr);
        }
    }.add;
    try put(&w, allocator, "model.layers.0.norm.weight", &.{H}, 1.0, s);
    try put(&w, allocator, "model.layers.0.mixer.in_proj.weight", &.{ d_inner + conv_dim + heads, H }, 0.02, s);
    try put(&w, allocator, "model.layers.0.mixer.conv1d.weight", &.{ conv_dim, kernel, 1 }, 0.25, s);
    try put(&w, allocator, "model.layers.0.mixer.A_log", &.{heads}, 0.5, s);
    try put(&w, allocator, "model.layers.0.mixer.D", &.{heads}, 1.0, s);
    try put(&w, allocator, "model.layers.0.mixer.dt_bias", &.{heads}, 0.1, s);
    try put(&w, allocator, "model.layers.0.mixer.norm.weight", &.{d_inner}, 1.0, s);
    try put(&w, allocator, "model.layers.0.mixer.out_proj.weight", &.{ H, d_inner }, 0.02, s);

    var config = ModelConfig{ .model_type = "nemotron_h", .weight_prefix = "model" };
    config.has_hybrid_layers = true;
    config.num_hidden_layers = 1;
    config.hidden_size = @intCast(H);
    config.quant_bits = 0;
    config.has_final_norm = false;
    config.head_dim = 64;
    config.num_attention_heads = 1;
    config.num_key_value_heads = 1;
    config.vocab_size = 8;
    config.mamba_num_heads = @intCast(heads);
    config.mamba_head_dim = @intCast(head_dim);
    config.mamba_n_groups = 1;
    config.ssm_state_size = @intCast(state);
    config.mamba_conv_kernel = @intCast(kernel);
    config.layer_block_types[0] = .mamba2;

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
    try mlx.check(mlx.mlx_ones(&emb_w, &[_]c_int{ 8, H }, 2, .bfloat16, s));
    var cache = try KVCache.init(allocator, 1);
    defer cache.deinit();
    var xfm = testHybridXfm(allocator, config, cache, emb_w, hl.ssm_entries, hl.hybrid_layers, s);
    defer deinitTestHybridXfm(&xfm);

    var ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids);
    try mlx.check(mlx.mlx_zeros(&ids, &[_]c_int{ 1, 4 }, 2, .int32, s));
    const out = try xfm.forward(ids);
    defer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_array_eval(out));
    try t.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(out));
}

// Bar: the fused Mamba2 step (mamba2_decode.zig) agrees with the op chain on
// the same tokens, one row at a time, as one multi-row window, and as a
// prefill-width window, to bf16 rounding; the window's per-row state capture
// equals the single-step states.
test "mamba2Mixer: fused single steps and fused windows match the chain" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const t = std.testing;
    const allocator = t.allocator;
    const s = mlx.gpuStream();
    const H: c_int = 64;
    const heads: c_int = 4;
    const head_dim: c_int = 64;
    const groups: c_int = 2;
    const state: c_int = 32;
    const kernel: c_int = 4;
    const d_inner = heads * head_dim;
    const conv_dim = d_inner + 2 * groups * state;
    const T: c_int = 3;

    var w = Weights.init(allocator);
    defer w.deinit();
    const put = struct {
        // Deterministic, non-uniform values so heads, groups and state columns differ.
        fn add(weights: *Weights, alloc: std.mem.Allocator, name: []const u8, shape: []const c_int, scale: f32, offset: f32, st: mlx.mlx_stream) !void {
            var n: usize = 1;
            for (shape) |d| n *= @intCast(d);
            const host = try alloc.alloc(f32, n);
            defer alloc.free(host);
            for (host, 0..) |*v, i| v.* = offset + scale * @sin(0.37 * @as(f32, @floatFromInt(i)) + 0.11 * @as(f32, @floatFromInt(name.len)));
            const f = mlx.mlx_array_new_data(host.ptr, shape.ptr, @intCast(shape.len), .float32);
            defer _ = mlx.mlx_array_free(f);
            var arr = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&arr, f, .bfloat16, st));
            try weights.map.put(try alloc.dupe(u8, name), arr);
        }
    }.add;
    try put(&w, allocator, "model.layers.0.norm.weight", &.{H}, 0.2, 1.0, s);
    try put(&w, allocator, "model.layers.0.mixer.in_proj.weight", &.{ d_inner + conv_dim + heads, H }, 0.05, 0.0, s);
    try put(&w, allocator, "model.layers.0.mixer.conv1d.weight", &.{ conv_dim, kernel, 1 }, 0.3, 0.1, s);
    try put(&w, allocator, "model.layers.0.mixer.conv1d.bias", &.{conv_dim}, 0.1, 0.0, s);
    try put(&w, allocator, "model.layers.0.mixer.A_log", &.{heads}, 0.5, 0.3, s);
    try put(&w, allocator, "model.layers.0.mixer.D", &.{heads}, 0.5, 1.0, s);
    try put(&w, allocator, "model.layers.0.mixer.dt_bias", &.{heads}, 0.5, 0.0, s);
    try put(&w, allocator, "model.layers.0.mixer.norm.weight", &.{d_inner}, 0.2, 1.0, s);
    try put(&w, allocator, "model.layers.0.mixer.out_proj.weight", &.{ H, d_inner }, 0.05, 0.0, s);

    var config = ModelConfig{ .model_type = "nemotron_h", .weight_prefix = "model" };
    config.has_hybrid_layers = true;
    config.num_hidden_layers = 1;
    config.hidden_size = @intCast(H);
    config.quant_bits = 0;
    config.has_final_norm = false;
    config.head_dim = 64;
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
    config.layer_block_types[0] = .mamba2;

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
    var cache = try KVCache.init(allocator, 1);
    defer cache.deinit();
    var xfm = testHybridXfm(allocator, config, cache, emb_w, hl.ssm_entries, hl.hybrid_layers, s);
    defer deinitTestHybridXfm(&xfm);

    const tokens = [_]i32{ 1, 5, 2 };
    const readF32 = struct {
        fn f(arr: mlx.mlx_array, st: mlx.mlx_stream, out: []f32) !void {
            var f32a = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(f32a);
            try mlx.check(mlx.mlx_astype(&f32a, arr, .float32, st));
            try mlx.check(mlx.mlx_array_eval(f32a));
            const p = mlx.mlx_array_data_float32(f32a) orelse return error.NoData;
            @memcpy(out, p[0..out.len]);
        }
    }.f;
    const resetState = struct {
        fn f(entries: []SSMCacheEntry) void {
            for (entries) |*e| {
                _ = mlx.mlx_array_free(e.conv_state);
                _ = mlx.mlx_array_free(e.ssm_state);
                ssmFreeSpecCapture(e);
                e.conv_state = .{ .ctx = null };
                e.ssm_state = .{ .ctx = null };
                e.initialized = false;
            }
        }
    }.f;
    const state_elems: usize = @intCast(heads * head_dim * state);
    const state_after_2 = try allocator.alloc(f32, state_elems);
    defer allocator.free(state_after_2);
    const window_state_1 = try allocator.alloc(f32, state_elems);
    defer allocator.free(window_state_1);

    // Chain: one prefill of T tokens with the fused window disabled.
    var chain: [3 * 64]f32 = undefined;
    {
        mamba2_fused_max_rows = 0;
        defer mamba2_fused_max_rows = std.math.maxInt(c_int);
        var ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ids);
        ids = mlx.mlx_array_new_data(&tokens, &[_]c_int{ 1, T }, 2, .int32);
        const out_pre = try xfm.forward(ids);
        defer _ = mlx.mlx_array_free(out_pre);
        try readF32(out_pre, s, &chain);
    }
    // Fused, one token at a time.
    resetState(hl.ssm_entries);
    mamba2_fused_logged = false;
    var fused: [3 * 64]f32 = undefined;
    for (tokens, 0..) |tok, i| {
        var ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ids);
        ids = mlx.mlx_array_new_data(&[_]i32{tok}, &[_]c_int{ 1, 1 }, 2, .int32);
        const out_one = try xfm.forward(ids);
        defer _ = mlx.mlx_array_free(out_one);
        try readF32(out_one, s, fused[i * 64 .. (i + 1) * 64]);
        if (i == 1) try readF32(hl.ssm_entries[0].ssm_state, s, state_after_2);
    }
    if (!mamba2_fused_logged) return error.FusedNotEngaged;
    // Fused, all T rows in one window with spec capture on.
    resetState(hl.ssm_entries);
    var window: [3 * 64]f32 = undefined;
    {
        var ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ids);
        ids = mlx.mlx_array_new_data(&tokens, &[_]c_int{ 1, T }, 2, .int32);
        var wctx = xfm.defaultCtx();
        wctx.capture_ssm_seq = true;
        const out_win = try xfm.forwardWith(&wctx, ids);
        defer _ = mlx.mlx_array_free(out_win);
        try readF32(out_win, s, &window);
        const e = &hl.ssm_entries[0];
        if (e.spec_state_seq.ctx == null or e.spec_conv_input.ctx == null) return error.NoSpecCapture;
        try t.expectEqualSlices(c_int, &.{ T, 1, heads, head_dim, state }, mlx.getShape(e.spec_state_seq));
        try t.expectEqualSlices(c_int, &.{ 1, kernel - 1 + T, conv_dim }, mlx.getShape(e.spec_conv_input));
        var row1 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(row1);
        try mlx.check(mlx.mlx_slice(&row1, e.spec_state_seq, &[_]c_int{ 1, 0, 0, 0, 0 }, 5, &[_]c_int{ 2, 1, heads, head_dim, state }, 5, &[_]c_int{ 1, 1, 1, 1, 1 }, 5, s));
        try readF32(row1, s, window_state_1);
    }

    var max_ref: f32 = 0;
    var max_err: f32 = 0;
    var max_err_win: f32 = 0;
    for (chain, fused, window) |a, b, c| {
        try t.expect(std.math.isFinite(a) and std.math.isFinite(b) and std.math.isFinite(c));
        max_ref = @max(max_ref, @abs(a));
        max_err = @max(max_err, @abs(a - b));
        max_err_win = @max(max_err_win, @abs(a - c));
    }
    if (!(max_ref > 0.01)) return error.RefTooSmall;
    if (max_err > 1.5e-2 * max_ref or max_err_win > 1.5e-2 * max_ref) {
        std.debug.print("mamba2 fused vs chain: single max_err={d:.5} window max_err={d:.5} max_ref={d:.5}\n", .{ max_err, max_err_win, max_ref });
        return error.FusedChainMismatch;
    }
    var st_ref: f32 = 0;
    var st_err: f32 = 0;
    for (state_after_2, window_state_1) |a, b| {
        st_ref = @max(st_ref, @abs(a));
        st_err = @max(st_err, @abs(a - b));
    }
    if (!(st_ref > 0 and st_err <= 1e-5 * st_ref)) {
        std.debug.print("mamba2 window state[1] vs two single steps: err={d:.6} ref={d:.6}\n", .{ st_err, st_ref });
        return error.StateCaptureMismatch;
    }

    // The op chain (kernel-declined geometry) records the same per-row capture.
    resetState(hl.ssm_entries);
    {
        mamba2_fused_max_rows = 0;
        defer mamba2_fused_max_rows = std.math.maxInt(c_int);
        var ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ids);
        ids = mlx.mlx_array_new_data(&tokens, &[_]c_int{ 1, T }, 2, .int32);
        var cctx = xfm.defaultCtx();
        cctx.capture_ssm_seq = true;
        const out_chain = try xfm.forwardWith(&cctx, ids);
        defer _ = mlx.mlx_array_free(out_chain);
        const e = &hl.ssm_entries[0];
        if (e.spec_state_seq.ctx == null or e.spec_conv_input.ctx == null) return error.NoChainSpecCapture;
        try t.expectEqualSlices(c_int, &.{ T, 1, heads, head_dim, state }, mlx.getShape(e.spec_state_seq));
        try t.expectEqualSlices(c_int, &.{ 1, kernel - 1 + T, conv_dim }, mlx.getShape(e.spec_conv_input));
        var row1 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(row1);
        try mlx.check(mlx.mlx_slice(&row1, e.spec_state_seq, &[_]c_int{ 1, 0, 0, 0, 0 }, 5, &[_]c_int{ 2, 1, heads, head_dim, state }, 5, &[_]c_int{ 1, 1, 1, 1, 1 }, 5, s));
        const chain_state_1 = try allocator.alloc(f32, state_elems);
        defer allocator.free(chain_state_1);
        try readF32(row1, s, chain_state_1);
        var c_err: f32 = 0;
        for (chain_state_1, window_state_1) |a, b| c_err = @max(c_err, @abs(a - b));
        if (c_err > 1.5e-2 * st_ref) {
            std.debug.print("mamba2 chain capture state[1] vs fused window: err={d:.6} ref={d:.6}\n", .{ c_err, st_ref });
            return error.ChainCaptureMismatch;
        }
    }

    // Prefill widths (several 16-row B/C blocks, and a ragged last block) take
    // the fused route and leave the chain's output, SSM state and conv state.
    const conv_elems: usize = @intCast((kernel - 1) * conv_dim);
    for ([_]c_int{ 37, 64 }) |TL| {
        const n: usize = @intCast(TL);
        const long_ids = try allocator.alloc(i32, n);
        defer allocator.free(long_ids);
        for (long_ids, 0..) |*v, i| v.* = @intCast((i * 5 + 1) % 8);
        var outs: [2][]f32 = undefined;
        var states: [2][]f32 = undefined;
        var convs: [2][]f32 = undefined;
        for (0..2) |arm| {
            outs[arm] = try allocator.alloc(f32, n * 64);
            states[arm] = try allocator.alloc(f32, state_elems);
            convs[arm] = try allocator.alloc(f32, conv_elems);
        }
        defer for (0..2) |arm| {
            allocator.free(outs[arm]);
            allocator.free(states[arm]);
            allocator.free(convs[arm]);
        };
        for (0..2) |arm| {
            resetState(hl.ssm_entries);
            mamba2_fused_logged = false;
            if (arm == 0) mamba2_fused_max_rows = 0;
            defer mamba2_fused_max_rows = std.math.maxInt(c_int);
            const ids = mlx.mlx_array_new_data(long_ids.ptr, &[_]c_int{ 1, TL }, 2, .int32);
            defer _ = mlx.mlx_array_free(ids);
            const out = try xfm.forward(ids);
            defer _ = mlx.mlx_array_free(out);
            try readF32(out, s, outs[arm]);
            try readF32(hl.ssm_entries[0].ssm_state, s, states[arm]);
            try readF32(hl.ssm_entries[0].conv_state, s, convs[arm]);
            if (arm == 1 and !mamba2_fused_logged) return error.LongWindowNotFused;
        }
        var ref: f32 = 0;
        var err: f32 = 0;
        for (outs[0], outs[1]) |a, b| {
            try t.expect(std.math.isFinite(a) and std.math.isFinite(b));
            ref = @max(ref, @abs(a));
            err = @max(err, @abs(a - b));
        }
        var sref: f32 = 0;
        var serr: f32 = 0;
        for (states[0], states[1]) |a, b| {
            try t.expect(std.math.isFinite(a) and std.math.isFinite(b));
            sref = @max(sref, @abs(a));
            serr = @max(serr, @abs(a - b));
        }
        if (!(ref > 0.01 and sref > 0) or err > 1.5e-2 * ref or serr > 1.5e-2 * sref) {
            std.debug.print("mamba2 T={d} fused vs chain: out err={d:.5} ref={d:.5} state err={d:.6} ref={d:.6}\n", .{ TL, err, ref, serr, sref });
            return error.LongWindowMismatch;
        }
        try t.expectEqualSlices(f32, convs[0], convs[1]);
    }
}
