//! sdk/blocks/gdn/decode_fused, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns GDN_KERNEL_HEADER, GDN_PREWORK_SOURCE, gdn_prework_kernel, gdn_prework_engaged, gdn_prework_override, gdn_prework_env, and 41 more.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const rht = @import("../../quant/hadamard/rht.zig");
const qmv2 = @import("../qmm/qmv2.zig");
const gdn_decode = @import("decode.zig");
const model_mod = @import("../../../engines/mlx/model.zig");
const log = @import("../../../host/log.zig");
const ModelConfig = @import("../../../engines/mlx/model.zig").ModelConfig;
const SpecTree = @import("../../../engines/mlx/forward_ctx.zig").SpecTree;
const KVCache = @import("../../../engines/mlx/kv/cache.zig").KVCache;
const testing = @import("../../../engines/mlx/kv/cache.zig").testing;
const bf16Scalar = @import("../../../engines/mlx/load/weights.zig").bf16Scalar;
const scalarOf = @import("../../../engines/mlx/load/weights.zig").scalarOf;
const SSMCacheEntry = @import("../../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const ssmCommitTreePath = @import("../../../engines/mlx/ssm/entry.zig").ssmCommitTreePath;
const Transformer = @import("../../../engines/mlx/transformer.zig").Transformer;
const sliceLastDim = @import("../../../mlx/array.zig").sliceLastDim;
const attn256MaxDiff = @import("../attention/attn256.zig").attn256MaxDiff;
const attn256RandBf16 = @import("../attention/attn256.zig").attn256RandBf16;
const attn256RandBf16Scaled = @import("../attention/attn256.zig").attn256RandBf16Scaled;
const sliceQkvPart = @import("../linear/fused_rows.zig").sliceQkvPart;
const sigmoidTableFor = @import("../mlp/fused_activations.zig").sigmoidTableFor;
const gatherQmvAct = @import("../moe/gather_qmv.zig").gatherQmvAct;
const moeRouterTopK = @import("../moe/router_kernel.zig").moeRouterTopK;
const gdnGateChain = @import("gate.zig").gdnGateChain;
const LinearAttnWeights = @import("gated_delta_net.zig").LinearAttnWeights;
const getGdnKernel = @import("kernels.zig").getGdnKernel;
const getGdnKernelSeq = @import("kernels.zig").getGdnKernelSeq;

// ── Packed GDN prework mixer (mlxfast-challenge qwen35_packed_gdn_prework port) ──
//
// ONE launch replacing the GDN prework chain at decode and verify widths
// (S 1..9) — conv-state concat + depthwise conv1d + SiLU + q/k/v split +
// reshapes + Q/K rmsNorm-and-scale + conv-state slice + the forget gate
// (exp(-exp(A_log)·softplus(a+dt_bias))) + beta (sigmoid(b)) — ~14
// dispatches per GDN layer per step. Six outputs: normed/scaled Q and K,
// activated V, the next 3-row conv state, g and beta.
// The gate rides the SAME formulas MLX's compiled gate closure emits
// (precise exp, utils.h log1p, GT add rounding), so it is bit-identical
// to `computeGdnGate`; the in-kernel sigmoid (SiLU half, beta) is MLX's own
// unary formula (exp-of-abs form) — the challenge swept all 65,280 finite
// bf16 inputs against it: bit-exact everywhere.
// q/k/v, b and a are read through (offset, row stride) pairs (a combined
// projection could hand them over without slice copies).
// S < 3 rows of the next conv state come from the OLD state (row r reads
// concat(conv_state, qkv)[r + S]); before this generalization the kernel
// was gated to S >= 3 and decode ran the composed chain. Bit-identity with
// the composed chain is the acceptance bar; kill switches
// MLX_SERVE_GDN_PREWORK=0 (whole kernel) and MLX_SERVE_GDN_DECODE_FUSED=0
// (S < 3, in-kernel gate, norm-gate: the pre-fusion decode path).
const GDN_KERNEL_HEADER =
    \\inline float msv_log1p(float x) {
    \\    float xp1 = 1.0f + x;
    \\    if (xp1 == metal::numeric_limits<float>::max()) { return metal::numeric_limits<float>::max(); }
    \\    if (xp1 == 1.0f) { return x; }
    \\    return x * (metal::log(xp1) / (xp1 - 1.0f));
    \\}
;

const GDN_PREWORK_SOURCE =
    \\uint lane = thread_position_in_threadgroup.x;
    \\// Grid row = b*S + r over the batch: q/k/v/g/beta/b/a are [B,S,..] flat
    \\// so `row` indexes them directly; the conv taps + next state are per batch.
    \\uint row = threadgroup_position_in_grid.y;
    \\uint b = row / uint(seq);
    \\uint r = row - b * uint(seq);
    \\uint logical_head = threadgroup_position_in_grid.z;
    \\constexpr uint q_heads = uint(HK);
    \\constexpr uint k_head_base = uint(HK);
    \\constexpr uint v_head_base = 2 * uint(HK);
    \\bool is_q = logical_head < q_heads;
    \\bool is_k = logical_head >= k_head_base && logical_head < v_head_base;
    \\uint head = is_q ? logical_head
    \\           : (is_k ? logical_head - k_head_base : logical_head - v_head_base);
    \\uint channel_base = is_q ? head * uint(DK)
    \\                   : (is_k ? uint(HK) * uint(DK) + head * uint(DK)
    \\                           : 2 * uint(HK) * uint(DK) + head * uint(DV));
    \\T activated[4];
    \\float sumsq = 0.0f;
    \\for (uint i = 0; i < 4; ++i) {
    \\    uint channel = channel_base + lane * 4 + i;
    \\    float acc = 0.0f;
    \\    for (uint tap = 0; tap < 4; ++tap) {
    \\        uint input_row = r + tap;
    \\        const T xv = input_row < uint(NKEEP)
    \\            ? conv_state[(b * uint(NKEEP) + input_row) * uint(C) + channel]
    \\            : qkv[(row + tap - uint(NKEEP)) * uint(QSTRIDE) + uint(QOFF) + channel];
    \\        acc += float(xv) * float(conv_w[channel * 4 + tap]);
    \\    }
    \\    const T conv = T(acc);
    \\    // MLX's unary Sigmoid formula (unary_ops.h), verbatim, in the tensor dtype.
    \\    T sig;
    \\    if constexpr (TAB) sig = sigtab[as_type<ushort>(conv)];
    \\    else { T sy = T(1) / (T(1) + metal::exp(metal::abs(conv))); sig = conv < T(0) ? sy : T(1) - sy; }
    \\    const T act = conv * sig;
    \\    activated[i] = act;
    \\    float value = float(act);
    \\    sumsq += value * value;
    \\}
    \\if (is_q || is_k) {
    \\    sumsq = simd_sum(sumsq);
    \\    float inv = metal::precise::rsqrt(sumsq / float(DK) + qk_eps);
    \\    const T scale = is_q ? q_scale : k_scale;
    \\    uint out_base = (row * uint(HK) + head) * uint(DK) + lane * 4;
    \\    for (uint i = 0; i < 4; ++i) {
    \\        // ones-weight rms_norm rounding (T(x*inv)), then the separate
    \\        // scalar multiply's rounding — the composed chain's two casts.
    \\        const T rms = T(1) * T(float(activated[i]) * inv);
    \\        const T value = scale * rms;
    \\        if (is_q) {
    \\            q_out[out_base + i] = value;
    \\        } else {
    \\            k_out[out_base + i] = value;
    \\        }
    \\    }
    \\} else {
    \\    uint out_base = (row * uint(HV) + head) * uint(DV) + lane * 4;
    \\    for (uint i = 0; i < 4; ++i) {
    \\        v_out[out_base + i] = activated[i];
    \\    }
    \\    if (lane == 0) {
    \\        // beta = sigmoid(b) (MLX unary formula); g = exp(-exp(A_log) *
    \\        // softplus(a + dt_bias)) with the compiled chain's own casts:
    \\        // GT add, f32 precise exp / log1p / exp, GT store.
    \\        const T bv = b_in[row * uint(BSTRIDE) + uint(BOFF) + head];
    \\        T bsig;
    \\        if constexpr (TAB) bsig = sigtab[as_type<ushort>(bv)];
    \\        else { T by = T(1) / (T(1) + metal::exp(metal::abs(bv))); bsig = bv < T(0) ? by : T(1) - by; }
    \\        beta_out[row * uint(HV) + head] = bsig;
    \\        if constexpr (GATE) {
    \\            const GT apd = GT(float(a_in[row * uint(ASTRIDE) + uint(AOFF) + head]) + float(dt_bias[head]));
    \\            float sp = msv_log1p(metal::precise::exp(float(apd)));
    \\            float ea = metal::precise::exp(float(A_log[head]));
    \\            g_out[row * uint(HV) + head] = GT(metal::precise::exp(-(ea * sp)));
    \\        }
    \\    }
    \\}
    \\// Next conv state = rows [S, S+NKEEP) of concat(conv_state, qkv).
    \\if (r + uint(NKEEP) >= uint(seq)) {
    \\    uint state_row = r + uint(NKEEP) - uint(seq);
    \\    uint raw_base = row * uint(QSTRIDE) + uint(QOFF) + channel_base + lane * 4;
    \\    uint state_base = (b * uint(NKEEP) + state_row) * uint(C) + channel_base + lane * 4;
    \\    for (uint i = 0; i < 4; ++i) {
    \\        conv_out[state_base + i] = qkv[raw_base + i];
    \\    }
    \\}
    \\if (r == 0) {
    \\    for (uint rr = 0; rr + uint(seq) < uint(NKEEP); ++rr) {
    \\        uint src_base = (b * uint(NKEEP) + rr + uint(seq)) * uint(C) + channel_base + lane * 4;
    \\        uint dst_base = (b * uint(NKEEP) + rr) * uint(C) + channel_base + lane * 4;
    \\        for (uint i = 0; i < 4; ++i) {
    \\            conv_out[dst_base + i] = conv_state[src_base + i];
    \\        }
    \\    }
    \\}
;

var gdn_prework_kernel: ?mlx.mlx_fast_metal_kernel = null;
var gdn_prework_engaged: bool = false;
pub var gdn_prework_override: ?bool = null; // test seam
var gdn_prework_env: ?bool = null;

fn gdnPreworkEnabled() bool {
    if (gdn_prework_override) |v| return v;
    if (gdn_prework_env) |v| return v;
    const raw = std.c.getenv("MLX_SERVE_GDN_PREWORK");
    const enabled = raw == null or !std.mem.eql(u8, std.mem.sliceTo(raw.?, 0), "0");
    gdn_prework_env = enabled;
    return enabled;
}

pub var gdn_decode_fused_override: ?bool = null; // test seam
var gdn_decode_fused_env: ?bool = null;

/// The decode-width GDN fusions (prework at S < 3, in-kernel gate/beta, the
/// norm-gate epilogue). Off = the pre-fusion decode path, op for op.
pub fn gdnDecodeFusedEnabled() bool {
    if (gdn_decode_fused_override) |v| return v;
    if (gdn_decode_fused_env) |v| return v;
    const raw = std.c.getenv("MLX_SERVE_GDN_DECODE_FUSED");
    const enabled = raw == null or !std.mem.eql(u8, std.mem.sliceTo(raw.?, 0), "0");
    gdn_decode_fused_env = enabled;
    return enabled;
}

var gdn_decode_recur_env: ?bool = null;
pub var gdn_decode_recur_engaged: bool = false;
pub var gdn_verify_recur_engaged: bool = false;
pub var gdn_verify_tree_engaged: bool = false;
pub var attn_qkv_joined_engaged: bool = false;
var gdn_verify_fold_env: ?bool = null;
pub var gdn_verify_fold_engaged: bool = false;

/// Chain verify rows: gdn_decode.recurSeqFold (recurrence + norm-gate + conv
/// input in one dispatch). MLX_SERVE_GDN_VERIFY_FOLD=0 restores recurSeq ->
/// norm-gate -> concat.
pub fn gdnVerifyFoldEnabled() bool {
    return gdn_verify_fold_env orelse blk: {
        const raw = std.c.getenv("MLX_SERVE_GDN_VERIFY_FOLD");
        const enabled = raw == null or !std.mem.eql(u8, std.mem.sliceTo(raw.?, 0), "0");
        gdn_verify_fold_env = enabled;
        break :blk enabled;
    };
}

/// S=1 decode on packs without a Hadamard rotation: gdn_decode.recur then the
/// norm-gate. MLX_SERVE_GDN_DECODE_RECUR=0 restores prework -> recurrence -> norm-gate.
pub fn gdnDecodeRecurEnabled() bool {
    const on = gdn_decode_recur_env orelse blk: {
        const raw = std.c.getenv("MLX_SERVE_GDN_DECODE_RECUR");
        const enabled = raw == null or !std.mem.eql(u8, std.mem.sliceTo(raw.?, 0), "0");
        gdn_decode_recur_env = enabled;
        break :blk enabled;
    };
    return on and gdnDecodeFusedEnabled();
}

fn getGdnPreworkKernel() !mlx.mlx_fast_metal_kernel {
    if (gdn_prework_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "qkv", "conv_state", "conv_w", "q_scale", "k_scale", "b_in", "a_in", "A_log", "dt_bias", "sigtab", "seq", "qk_eps" };
    const output_names = [_][*:0]const u8{ "q_out", "k_out", "v_out", "conv_out", "g_out", "beta_out" };
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "mlxserve_gdn_prework2",
        in_vec,
        out_vec,
        GDN_PREWORK_SOURCE,
        GDN_KERNEL_HEADER,
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    gdn_prework_kernel = kernel;
    return kernel;
}

const GdnPreworkCfgKey = struct { hk: c_int, hv: c_int, seq: c_int, batch: c_int, c: c_int, qs: c_int, qo: c_int, bs: c_int, bo: c_int, as: c_int, ao: c_int, dt: mlx.mlx_dtype, gt: mlx.mlx_dtype, gate: bool };
var gdn_prework_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var gdn_prework_cfg_key: GdnPreworkCfgKey = std.mem.zeroes(GdnPreworkCfgKey);

pub const GdnPrework = struct {
    q: mlx.mlx_array, // [B, S, Hk, 128] normed + scaled
    k: mlx.mlx_array, // [B, S, Hk, 128] normed + scaled
    v: mlx.mlx_array, // [B, S, Hv, 128] silu-activated
    conv_state: mlx.mlx_array, // [B, NKEEP, C] next conv state
    g: mlx.mlx_array, // [B, S, Hv] forget gate
    beta: mlx.mlx_array, // [B, S, Hv] sigmoid(b)
};
/// Rows (batch*seq) the decode-width GDN fusions serve.
pub const GDN_FUSED_MAX_ROWS: c_int = 16;

var gdn_prefill_fused_override: ?bool = null;
var gdn_prefill_fused_env: ?bool = null;
pub var gdn_prefill_engaged = false;

pub fn gdnPrefillFusedFor(seq: c_int, batch: c_int) bool {
    if (seq < 17 or batch < 1 or batch > 2 or seq > @divTrunc(@import("../hc/hc_prefill.zig").max_seq, batch)) return false;
    if (gdn_prefill_fused_override) |on| return on;
    if (gdn_prefill_fused_env) |on| return on;
    const raw = std.c.getenv("MLX_SERVE_GDN_PREFILL_FUSED");
    const on = raw == null or !std.mem.eql(u8, std.mem.span(raw.?), "0");
    gdn_prefill_fused_env = on;
    return on;
}

/// Inputs of the packed prework. `qkv`/`b`/`a` are read at `(off, stride)`
/// per row, so they may all alias one folded in_proj output.
pub const GdnPreworkArgs = struct {
    qkv: mlx.mlx_array, // rows of [.. QOFF + C ..] bf16
    qkv_off: c_int,
    qkv_stride: c_int,
    b: mlx.mlx_array, // rows of [.. BOFF + Hv ..] bf16
    b_off: c_int,
    b_stride: c_int,
    a: mlx.mlx_array,
    a_off: c_int,
    a_stride: c_int,
    A_log: mlx.mlx_array, // [Hv]
    dt_bias: mlx.mlx_array, // [Hv] bf16
    conv_state: mlx.mlx_array, // [B, NKEEP, C] bf16
    conv_w: mlx.mlx_array, // [C, 4, 1] bf16
    q_scale: mlx.mlx_array, // 0-dim bf16 (1/dk)
    k_scale: mlx.mlx_array, // 0-dim bf16 (1/sqrt(dk))
    qk_eps: f32 = 1e-6, // `ModelConfig.gdnQkNormEps`
    hk: c_int,
    hv: c_int,
    dk: c_int,
    dv: c_int,
    seq: c_int,
    batch: c_int = 1,
    /// False: no per-head gate (`g` comes back unwritten; KDA builds a per-channel one).
    gate: bool = true,
    /// The dtype `g` is computed and stored in: the recurrent state's. Null: the activations'.
    g_dtype: ?mlx.mlx_dtype = null,
};

/// One fused dispatch for the GDN prework at S 1..9 x B batched slots
/// (batch*seq <= GDN_FUSED_MAX_ROWS; per-head gate archs only). Null →
/// caller keeps the composed chain. The caller owns all six outputs (and
/// installs `conv_state` into the SSM cache entry).
pub fn gdnPreworkFused(s: mlx.mlx_stream, in: GdnPreworkArgs) !?GdnPrework {
    if (!gdnPreworkEnabled() or !mlx.streamIsGpu(s)) return null;
    if (!gdnPrefillFusedFor(in.seq, in.batch) and (in.seq < 1 or in.seq > 9 or in.batch < 1 or in.batch * in.seq > GDN_FUSED_MAX_ROWS)) return null;
    if (in.seq < 3 and !gdnDecodeFusedEnabled()) return null;
    if (in.dk != 128 or in.dv != 128) return null;
    const dt = mlx.mlx_array_dtype(in.qkv);
    if (dt != .bfloat16 and dt != .float16) return null;
    inline for (.{ in.b, in.conv_state, in.conv_w, in.q_scale, in.k_scale }) |arr| {
        if (mlx.mlx_array_dtype(arr) != dt) return null;
    }
    if (in.gate and (mlx.mlx_array_dtype(in.a) != dt or mlx.mlx_array_dtype(in.dt_bias) != dt)) return null;
    const wsh = mlx.getShape(in.conv_w);
    if (wsh.len < 2 or wsh[1] != 4) return null;
    const c_dim: c_int = in.hk * in.dk * 2 + in.hv * in.dv;
    const qsh = mlx.getShape(in.qkv);
    if (qsh.len != 3 or qsh[0] != in.batch or qsh[1] != in.seq or qsh[2] != in.qkv_stride or in.qkv_off + c_dim > in.qkv_stride) return null;
    const bsh = mlx.getShape(in.b);
    if (bsh.len != 3 or bsh[0] != in.batch or bsh[1] != in.seq or bsh[2] != in.b_stride or in.b_off + in.hv > in.b_stride) return null;
    const ash = mlx.getShape(in.a);
    if (in.gate and (ash.len != 3 or ash[0] != in.batch or ash[1] != in.seq or ash[2] != in.a_stride or in.a_off + in.hv > in.a_stride)) return null;
    const csh = mlx.getShape(in.conv_state);
    if (csh.len != 3 or csh[0] != in.batch or csh[1] != 3 or csh[2] != c_dim) return null;

    const nkeep: c_int = 3;
    const gt = in.g_dtype orelse dt;
    const key = GdnPreworkCfgKey{ .hk = in.hk, .hv = in.hv, .seq = in.seq, .batch = in.batch, .c = c_dim, .qs = in.qkv_stride, .qo = in.qkv_off, .bs = in.b_stride, .bo = in.b_off, .as = in.a_stride, .ao = in.a_off, .dt = dt, .gt = gt, .gate = in.gate };
    if (gdn_prework_cfg == null or !std.meta.eql(gdn_prework_cfg_key, key)) {
        if (gdn_prework_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const config = mlx.mlx_fast_metal_kernel_config_new();
        const q_shape = [_]c_int{ in.batch, in.seq, in.hk, in.dk };
        const v_shape = [_]c_int{ in.batch, in.seq, in.hv, in.dv };
        const st_shape = [_]c_int{ in.batch, nkeep, c_dim };
        const g_shape = [_]c_int{ in.batch, in.seq, in.hv };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &q_shape, 4, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &q_shape, 4, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &v_shape, 4, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &st_shape, 3, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &g_shape, 3, gt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &g_shape, 3, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32, in.batch * in.seq, 2 * in.hk + in.hv));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "GT", gt));
        const ints = .{ .{ "HK", in.hk }, .{ "HV", in.hv }, .{ "DK", in.dk }, .{ "DV", in.dv }, .{ "NKEEP", nkeep }, .{ "C", c_dim }, .{ "TAB", @as(c_int, @intFromBool(in.seq >= 17)) }, .{ "QSTRIDE", in.qkv_stride }, .{ "QOFF", in.qkv_off }, .{ "BSTRIDE", in.b_stride }, .{ "BOFF", in.b_off }, .{ "ASTRIDE", in.a_stride }, .{ "AOFF", in.a_off }, .{ "GATE", @as(c_int, @intFromBool(in.gate)) } };
        inline for (ints) |kv| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, kv[0], kv[1]));
        gdn_prework_cfg = config;
        gdn_prework_cfg_key = key;
    }

    const kernel = try getGdnPreworkKernel();
    const seq_arr = mlx.mlx_array_new_int(in.seq);
    defer _ = mlx.mlx_array_free(seq_arr);
    const eps_arr = mlx.mlx_array_new_float(in.qk_eps);
    defer _ = mlx.mlx_array_free(eps_arr);
    const inputs_arr = [_]mlx.mlx_array{ in.qkv, in.conv_state, in.conv_w, in.q_scale, in.k_scale, in.b, in.a, in.A_log, in.dt_bias, if (in.seq >= 17) try sigmoidTableFor(s, dt) else in.qkv, seq_arr, eps_arr };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, gdn_prework_cfg.?, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 6) return error.MetalKernelBadOutputCount;
    var out = GdnPrework{
        .q = mlx.mlx_array_new(),
        .k = mlx.mlx_array_new(),
        .v = mlx.mlx_array_new(),
        .conv_state = mlx.mlx_array_new(),
        .g = mlx.mlx_array_new(),
        .beta = mlx.mlx_array_new(),
    };
    errdefer {
        inline for (.{ &out.q, &out.k, &out.v, &out.conv_state, &out.g, &out.beta }) |p| {
            if (p.*.ctx != null) _ = mlx.mlx_array_free(p.*);
        }
    }
    try mlx.check(mlx.mlx_vector_array_get(&out.q, outputs_vec, 0));
    try mlx.check(mlx.mlx_vector_array_get(&out.k, outputs_vec, 1));
    try mlx.check(mlx.mlx_vector_array_get(&out.v, outputs_vec, 2));
    try mlx.check(mlx.mlx_vector_array_get(&out.conv_state, outputs_vec, 3));
    try mlx.check(mlx.mlx_vector_array_get(&out.g, outputs_vec, 4));
    try mlx.check(mlx.mlx_vector_array_get(&out.beta, outputs_vec, 5));
    if (!gdn_prework_engaged) {
        gdn_prework_engaged = true;
        log.info("[gdn] packed prework engaged: Hk={d} Hv={d} S={d} B={d}\n", .{ in.hk, in.hv, in.seq, in.batch });
    }
    return out;
}

// ── Fused GDN norm-gate epilogue ──
// rms_norm(y, norm_w) * silu(z) → flat [1, S, Hv*Dv] in ONE launch (the
// composed chain is rms_norm + sigmoid + multiply + multiply). Mirrors
// `rms_single_row`'s reduction (per-lane 4-element partials, simd_sum,
// precise rsqrt, `w * T(x * inv)`) and MLX's sigmoid, so it is bit-identical.
// z is read at (ZOFF, ZSTRIDE) so a wider projection row can hand it over.
const GDN_NORMGATE_SOURCE =
    \\uint lane = thread_position_in_threadgroup.x;
    \\uint row = threadgroup_position_in_grid.y;
    \\uint head = threadgroup_position_in_grid.z;
    \\uint base = (row * uint(HV) + head) * uint(DV) + lane * 4;
    \\float xs[4];
    \\float sumsq = 0.0f;
    \\for (uint i = 0; i < 4; ++i) {
    \\    xs[i] = float(y[base + i]);
    \\    sumsq += xs[i] * xs[i];
    \\}
    \\sumsq = simd_sum(sumsq);
    \\float inv = metal::precise::rsqrt(sumsq / float(DV) + eps);
    \\uint zbase = row * uint(ZSTRIDE) + uint(ZOFF) + head * uint(DV) + lane * 4;
    \\for (uint i = 0; i < 4; ++i) {
    \\    const T normed = norm_w[lane * 4 + i] * T(xs[i] * inv);
    \\    const T zv = z[zbase + i];
    \\    T sig;
    \\    if constexpr (TAB) sig = sigtab[as_type<ushort>(zv)];
    \\    else { T sy = T(1) / (T(1) + metal::exp(metal::abs(zv))); sig = zv < T(0) ? sy : T(1) - sy; }
    \\    // swish gate: silu(z) * normed (qwen3.5); sigmoid gate: normed * sigmoid(z) (qwen4_exp, KDA)
    \\    out[base + i] = SWISH ? (zv * sig) * normed : normed * sig;
    \\}
;

var gdn_normgate_kernel: ?mlx.mlx_fast_metal_kernel = null;
var gdn_normgate_engaged: bool = false;
var gdn_normgate_declined: bool = false;
const GdnNormGateCfgKey = struct { hv: c_int, seq: c_int, batch: c_int, zs: c_int, zo: c_int, swish: bool, dt: mlx.mlx_dtype };
var gdn_normgate_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var gdn_normgate_cfg_key: GdnNormGateCfgKey = std.mem.zeroes(GdnNormGateCfgKey);

fn getGdnNormGateKernel() !mlx.mlx_fast_metal_kernel {
    if (gdn_normgate_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "y", "z", "norm_w", "eps", "sigtab" };
    const output_names = [_][*:0]const u8{"out"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new("mlxserve_gdn_normgate", in_vec, out_vec, GDN_NORMGATE_SOURCE, "", true, false);
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    gdn_normgate_kernel = kernel;
    return kernel;
}

/// `rms_norm(y) * silu(z)` flattened to [B, S, Hv*Dv]. Null → composed chain.
pub fn gdnNormGateFused(
    s: mlx.mlx_stream,
    y: mlx.mlx_array, // [B, S, Hv, Dv] bf16
    z: mlx.mlx_array, // rows of [.. ZOFF + Hv*Dv ..] bf16
    z_off: c_int,
    z_stride: c_int,
    norm_w: mlx.mlx_array, // [Dv] bf16
    eps: mlx.mlx_array, // 0-dim f32
    swish: bool, // true: silu(z) * y (qwen3.5); false: y * sigmoid(z) (qwen4_exp, KDA)
    hv: c_int,
    dv: c_int,
    batch: c_int,
    seq: c_int,
) !?mlx.mlx_array {
    if (dv != 128 or !mlx.streamIsGpu(s)) return null;
    const prefill = gdnPrefillFusedFor(seq, batch);
    if (!prefill and !gdnDecodeFusedEnabled()) return null;
    if (!prefill and (seq < 1 or seq > GDN_FUSED_MAX_ROWS or batch < 1 or batch * seq > GDN_FUSED_MAX_ROWS)) return null;
    const dt = mlx.mlx_array_dtype(y);
    inline for (.{ y, z, norm_w }, 0..) |arr, i| {
        if ((dt != .bfloat16 and dt != .float16) or mlx.mlx_array_dtype(arr) != dt) {
            if (!gdn_normgate_declined) {
                gdn_normgate_declined = true;
                log.info("[gdn] fused norm-gate declined: input {d} dtype {s}\n", .{ i, @tagName(mlx.mlx_array_dtype(arr)) });
            }
            return null;
        }
    }
    const ysh = mlx.getShape(y);
    const zsh = mlx.getShape(z);
    if (ysh.len != 4 or ysh[0] != batch or ysh[1] != seq or ysh[2] != hv or ysh[3] != dv or zsh.len != 3 or zsh[0] != batch or zsh[1] != seq or zsh[2] != z_stride or z_off + hv * dv > z_stride) {
        if (!gdn_normgate_declined) {
            gdn_normgate_declined = true;
            log.info("[gdn] fused norm-gate declined: y {any} z {any} z_off={d} z_stride={d} hv={d} dv={d} seq={d}\n", .{ ysh, zsh, z_off, z_stride, hv, dv, seq });
        }
        return null;
    }
    const key = GdnNormGateCfgKey{ .hv = hv, .seq = seq, .batch = batch, .zs = z_stride, .zo = z_off, .swish = swish, .dt = dt };
    if (gdn_normgate_cfg == null or !std.meta.eql(gdn_normgate_cfg_key, key)) {
        if (gdn_normgate_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const config = mlx.mlx_fast_metal_kernel_config_new();
        const out_shape = [_]c_int{ batch, seq, hv * dv };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &out_shape, 3, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32, batch * seq, hv));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", dt));
        const ints = .{ .{ "HV", hv }, .{ "DV", dv }, .{ "ZSTRIDE", z_stride }, .{ "ZOFF", z_off }, .{ "TAB", @as(c_int, @intFromBool(seq >= 17)) } };
        inline for (ints) |kv| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, kv[0], kv[1]));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "SWISH", @intFromBool(swish)));
        gdn_normgate_cfg = config;
        gdn_normgate_cfg_key = key;
    }
    const kernel = try getGdnNormGateKernel();
    const inputs_arr = [_]mlx.mlx_array{ y, z, norm_w, eps, if (seq >= 17) try sigmoidTableFor(s, dt) else y };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, gdn_normgate_cfg.?, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    if (!gdn_normgate_engaged) {
        gdn_normgate_engaged = true;
        log.info("[gdn] fused norm-gate engaged: Hv={d} S={d} gate={s}\n", .{ hv, seq, if (swish) "swish" else "sigmoid" });
    }
    return out;
}

// ── tests ──

test "gdn packed prework: bit-identical to the composed chain at decode and prefill widths incl. gate/beta + folded layout" {
    const s = mlx.gpuStream();
    gdn_prefill_fused_override = true;
    defer gdn_prefill_fused_override = null;
    gdn_prework_override = true;
    defer gdn_prework_override = null;
    gdn_decode_fused_override = true;
    defer gdn_decode_fused_override = null;
    var cache = try KVCache.init(testing.allocator, 0);
    defer cache.deinit();
    var xfm = std.mem.zeroInit(Transformer, .{ .s = s, .allocator = testing.allocator, .cache = cache });
    xfm.compileGdnGate();
    defer if (xfm.compiled_gdn_gate) |closure| {
        _ = mlx.mlx_closure_free(closure);
    };
    try testing.expect(xfm.compiled_gdn_gate != null);
    // An f32-state model's gate: f32 from both producers, bit for bit too.
    var xfm32 = std.mem.zeroInit(Transformer, .{ .s = s, .allocator = testing.allocator, .cache = cache, .config = ModelConfig{ .ssm_state_f32 = true } });
    xfm32.compileGdnGate();
    defer if (xfm32.compiled_gdn_gate) |closure| {
        _ = mlx.mlx_closure_free(closure);
    };
    var prng = std.Random.DefaultPrng.init(0x6D1);
    const rnd = prng.random();

    const hk: c_int = 2;
    const hv: c_int = 4;
    const dk: c_int = 128;
    const dv: c_int = 128;
    const c_dim: c_int = hk * dk * 2 + hv * dv;
    const key_dim: c_int = hk * dk;
    const value_dim: c_int = hv * dv;

    const q_scale = bf16Scalar(1.0 / 128.0, s);
    defer _ = mlx.mlx_array_free(q_scale);
    const k_scale = bf16Scalar(@sqrt(1.0 / 128.0), s);
    defer _ = mlx.mlx_array_free(k_scale);
    var ones_w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ones_w);
    const ones_shape = [_]c_int{dk};
    try mlx.check(mlx.mlx_ones(&ones_w, &ones_shape, 1, .bfloat16, s));
    const hv_shape = [_]c_int{hv};
    const A_log = try attn256RandBf16(rnd, &hv_shape, s);
    defer _ = mlx.mlx_array_free(A_log);
    const dt_bias = try attn256RandBf16(rnd, &hv_shape, s);
    defer _ = mlx.mlx_array_free(dt_bias);

    for ([_]c_int{ 1, 2, 3, 5, 9, 17, 65, 513, 8703 }) |seq| {
        const qkv_shape = [_]c_int{ 1, seq, c_dim };
        const qkv = try attn256RandBf16(rnd, &qkv_shape, s);
        defer _ = mlx.mlx_array_free(qkv);
        const st_shape = [_]c_int{ 1, 3, c_dim };
        const conv_state = if (seq == 17) blk: {
            var zero = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&zero, &st_shape, 3, .bfloat16, s));
            break :blk zero;
        } else try attn256RandBf16(rnd, &st_shape, s);
        defer _ = mlx.mlx_array_free(conv_state);
        const w_shape = [_]c_int{ c_dim, 4, 1 };
        const conv_w = try attn256RandBf16(rnd, &w_shape, s);
        defer _ = mlx.mlx_array_free(conv_w);
        const ba_shape = [_]c_int{ 1, seq, hv };
        // Wide-range b/a so the gate's exp/log1p branches are all exercised.
        const b_in = try attn256RandBf16Scaled(rnd, &ba_shape, 16.0, s);
        defer _ = mlx.mlx_array_free(b_in);
        const a_in = try attn256RandBf16Scaled(rnd, &ba_shape, 16.0, s);
        defer _ = mlx.mlx_array_free(a_in);

        var args = GdnPreworkArgs{
            .qkv = qkv,
            .qkv_off = 0,
            .qkv_stride = c_dim,
            .b = b_in,
            .b_off = 0,
            .b_stride = hv,
            .a = a_in,
            .a_off = 0,
            .a_stride = hv,
            .A_log = A_log,
            .dt_bias = dt_bias,
            .conv_state = conv_state,
            .conv_w = conv_w,
            .q_scale = q_scale,
            .k_scale = k_scale,
            .hk = hk,
            .hv = hv,
            .dk = dk,
            .dv = dv,
            .seq = seq,
        };
        const pre = (try gdnPreworkFused(s, args)) orelse return error.FusedDeclined;
        defer _ = mlx.mlx_array_free(pre.q);
        defer _ = mlx.mlx_array_free(pre.k);
        defer _ = mlx.mlx_array_free(pre.v);
        defer _ = mlx.mlx_array_free(pre.conv_state);
        defer _ = mlx.mlx_array_free(pre.g);
        defer _ = mlx.mlx_array_free(pre.beta);

        // Composed reference — the production chain, op for op.
        var conv_input = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(conv_input);
        {
            const arr = [_]mlx.mlx_array{ conv_state, qkv };
            const vec = mlx.mlx_vector_array_new_data(&arr, 2);
            defer _ = mlx.mlx_vector_array_free(vec);
            try mlx.check(mlx.mlx_concatenate_axis(&conv_input, vec, 1, s));
        }
        var ref_state = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ref_state);
        {
            const start = [_]c_int{ 0, seq, 0 };
            const stop = [_]c_int{ 1, seq + 3, c_dim };
            const strides = [_]c_int{ 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(&ref_state, conv_input, &start, 3, &stop, 3, &strides, 3, s));
        }
        var conv_raw = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(conv_raw);
        try mlx.check(mlx.mlx_conv1d(&conv_raw, conv_input, conv_w, 1, 0, 1, c_dim, s));
        var sig = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sig);
        try mlx.check(mlx.mlx_sigmoid(&sig, conv_raw, s));
        var conv_out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(conv_out);
        try mlx.check(mlx.mlx_multiply(&conv_out, conv_raw, sig, s));

        const strides3 = [_]c_int{ 1, 1, 1 };
        {
            var v_flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(v_flat);
            const start = [_]c_int{ 0, 0, key_dim * 2 };
            const stop = [_]c_int{ 1, seq, key_dim * 2 + value_dim };
            try mlx.check(mlx.mlx_slice(&v_flat, conv_out, &start, 3, &stop, 3, &strides3, 3, s));
            var v_heads = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(v_heads);
            const v_shape = [_]c_int{ 1, seq, hv, dv };
            try mlx.check(mlx.mlx_reshape(&v_heads, v_flat, &v_shape, 4, s));
            try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(pre.v, v_heads, s));
        }
        inline for (.{
            .{ @as(c_int, 0), key_dim, hk, dk, q_scale, pre.q },
            .{ key_dim, key_dim * 2, hk, dk, k_scale, pre.k },
        }) |case| {
            var flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat);
            const start = [_]c_int{ 0, 0, case[0] };
            const stop = [_]c_int{ 1, seq, case[1] };
            try mlx.check(mlx.mlx_slice(&flat, conv_out, &start, 3, &stop, 3, &strides3, 3, s));
            var heads = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(heads);
            const h_shape = [_]c_int{ 1, seq, case[2], case[3] };
            try mlx.check(mlx.mlx_reshape(&heads, flat, &h_shape, 4, s));
            var nrm = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(nrm);
            try mlx.check(mlx.mlx_fast_rms_norm(&nrm, heads, ones_w, 1e-6, s));
            var scaled = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(scaled);
            try mlx.check(mlx.mlx_multiply(&scaled, nrm, case[4], s));
            try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(case[5], scaled, s));
        }
        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(pre.conv_state, ref_state, s));

        // Gate + beta against the graph chain.
        const g_ref = try xfm.computeGdnGate(A_log, a_in, dt_bias);
        defer _ = mlx.mlx_array_free(g_ref);
        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(pre.g, g_ref, s));
        var beta_ref = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(beta_ref);
        try mlx.check(mlx.mlx_sigmoid(&beta_ref, b_in, s));
        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(pre.beta, beta_ref, s));
        args.g_dtype = .float32;
        const pre32 = (try gdnPreworkFused(s, args)) orelse return error.FusedDeclined;
        defer inline for (.{ pre32.q, pre32.k, pre32.v, pre32.conv_state, pre32.g, pre32.beta }) |arr| {
            _ = mlx.mlx_array_free(arr);
        };
        const g32 = try xfm32.computeGdnGate(A_log, a_in, dt_bias);
        defer _ = mlx.mlx_array_free(g32);
        try std.testing.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(pre32.g));
        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(pre32.g, g32, s));

        // Folded layout: [qkv | z | b | a] rows read through offsets must
        // reproduce the unfolded call exactly.
        var raw = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(raw);
        {
            const z_shape = [_]c_int{ 1, seq, value_dim };
            const z = try attn256RandBf16(rnd, &z_shape, s);
            defer _ = mlx.mlx_array_free(z);
            const arr = [_]mlx.mlx_array{ qkv, z, b_in, a_in };
            const vec = mlx.mlx_vector_array_new_data(&arr, 4);
            defer _ = mlx.mlx_vector_array_free(vec);
            try mlx.check(mlx.mlx_concatenate_axis(&raw, vec, 2, s));
        }
        const width = c_dim + value_dim + 2 * hv;
        const pre2 = (try gdnPreworkFused(s, .{
            .qkv = raw,
            .qkv_off = 0,
            .qkv_stride = width,
            .b = raw,
            .b_off = c_dim + value_dim,
            .b_stride = width,
            .a = raw,
            .a_off = c_dim + value_dim + hv,
            .a_stride = width,
            .A_log = A_log,
            .dt_bias = dt_bias,
            .conv_state = conv_state,
            .conv_w = conv_w,
            .q_scale = q_scale,
            .k_scale = k_scale,
            .hk = hk,
            .hv = hv,
            .dk = dk,
            .dv = dv,
            .seq = seq,
        })) orelse return error.FusedDeclined;
        defer _ = mlx.mlx_array_free(pre2.q);
        defer _ = mlx.mlx_array_free(pre2.k);
        defer _ = mlx.mlx_array_free(pre2.v);
        defer _ = mlx.mlx_array_free(pre2.conv_state);
        defer _ = mlx.mlx_array_free(pre2.g);
        defer _ = mlx.mlx_array_free(pre2.beta);
        inline for (.{ .{ pre.q, pre2.q }, .{ pre.k, pre2.k }, .{ pre.v, pre2.v }, .{ pre.conv_state, pre2.conv_state }, .{ pre.g, pre2.g }, .{ pre.beta, pre2.beta } }) |pair| {
            try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(pair[0], pair[1], s));
        }

        // Batched arm (B=2 merged slots): row 0 of the batched outputs is
        // bit-identical to the single-batch outputs above; row 1 to a
        // second slot's own single-batch run.
        if (seq <= 8 or seq == 65) {
            const qkv_b = try attn256RandBf16(rnd, &qkv_shape, s);
            defer _ = mlx.mlx_array_free(qkv_b);
            const st_b = try attn256RandBf16(rnd, &st_shape, s);
            defer _ = mlx.mlx_array_free(st_b);
            const b_b = try attn256RandBf16Scaled(rnd, &ba_shape, 16.0, s);
            defer _ = mlx.mlx_array_free(b_b);
            const a_b = try attn256RandBf16Scaled(rnd, &ba_shape, 16.0, s);
            defer _ = mlx.mlx_array_free(a_b);
            const cat0 = struct {
                fn f(st: mlx.mlx_stream, x0: mlx.mlx_array, x1: mlx.mlx_array) !mlx.mlx_array {
                    const arr = [_]mlx.mlx_array{ x0, x1 };
                    const vec = mlx.mlx_vector_array_new_data(&arr, 2);
                    defer _ = mlx.mlx_vector_array_free(vec);
                    var out = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 0, st));
                    return out;
                }
            }.f;
            const row0 = struct {
                fn f(st: mlx.mlx_stream, x: mlx.mlx_array, i: c_int) !mlx.mlx_array {
                    const sh = mlx.getShape(x);
                    var start: [4]c_int = .{ i, 0, 0, 0 };
                    var stop: [4]c_int = .{ i + 1, 0, 0, 0 };
                    const strides: [4]c_int = .{ 1, 1, 1, 1 };
                    for (1..sh.len) |d| stop[d] = sh[d];
                    _ = &start;
                    var out = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_slice(&out, x, &start, @intCast(sh.len), &stop, @intCast(sh.len), &strides, @intCast(sh.len), st));
                    return out;
                }
            }.f;
            const qkv2 = try cat0(s, qkv, qkv_b);
            defer _ = mlx.mlx_array_free(qkv2);
            const st2 = try cat0(s, conv_state, st_b);
            defer _ = mlx.mlx_array_free(st2);
            const b2 = try cat0(s, b_in, b_b);
            defer _ = mlx.mlx_array_free(b2);
            const a2 = try cat0(s, a_in, a_b);
            defer _ = mlx.mlx_array_free(a2);
            var args_b = GdnPreworkArgs{ .qkv = qkv2, .qkv_off = 0, .qkv_stride = c_dim, .b = b2, .b_off = 0, .b_stride = hv, .a = a2, .a_off = 0, .a_stride = hv, .A_log = A_log, .dt_bias = dt_bias, .conv_state = st2, .conv_w = conv_w, .q_scale = q_scale, .k_scale = k_scale, .hk = hk, .hv = hv, .dk = dk, .dv = dv, .seq = seq, .batch = 2 };
            const preb = (try gdnPreworkFused(s, args_b)) orelse return error.FusedDeclined;
            defer inline for (.{ preb.q, preb.k, preb.v, preb.conv_state, preb.g, preb.beta }) |arr| {
                _ = mlx.mlx_array_free(arr);
            };
            args_b.qkv = qkv_b;
            args_b.conv_state = st_b;
            args_b.b = b_b;
            args_b.a = a_b;
            args_b.batch = 1;
            const pre_b1 = (try gdnPreworkFused(s, args_b)) orelse return error.FusedDeclined;
            defer inline for (.{ pre_b1.q, pre_b1.k, pre_b1.v, pre_b1.conv_state, pre_b1.g, pre_b1.beta }) |arr| {
                _ = mlx.mlx_array_free(arr);
            };
            inline for (.{ .{ preb.q, pre.q, pre_b1.q }, .{ preb.k, pre.k, pre_b1.k }, .{ preb.v, pre.v, pre_b1.v }, .{ preb.conv_state, pre.conv_state, pre_b1.conv_state }, .{ preb.g, pre.g, pre_b1.g }, .{ preb.beta, pre.beta, pre_b1.beta } }) |tri| {
                const r0 = try row0(s, tri[0], 0);
                defer _ = mlx.mlx_array_free(r0);
                const r1 = try row0(s, tri[0], 1);
                defer _ = mlx.mlx_array_free(r1);
                try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(r0, tri[1], s));
                try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(r1, tri[2], s));
            }
        }
    }

    // Unsupported widths/head dimensions and disabled decode/prefill arms decline.
    {
        const qkv_shape = [_]c_int{ 1, 2, c_dim };
        const qkv = try attn256RandBf16(rnd, &qkv_shape, s);
        defer _ = mlx.mlx_array_free(qkv);
        const st_shape = [_]c_int{ 1, 3, c_dim };
        const conv_state = try attn256RandBf16(rnd, &st_shape, s);
        defer _ = mlx.mlx_array_free(conv_state);
        const w_shape = [_]c_int{ c_dim, 4, 1 };
        const conv_w = try attn256RandBf16(rnd, &w_shape, s);
        defer _ = mlx.mlx_array_free(conv_w);
        const ba_shape = [_]c_int{ 1, 2, hv };
        const ba = try attn256RandBf16(rnd, &ba_shape, s);
        defer _ = mlx.mlx_array_free(ba);
        var args = GdnPreworkArgs{
            .qkv = qkv,
            .qkv_off = 0,
            .qkv_stride = c_dim,
            .b = ba,
            .b_off = 0,
            .b_stride = hv,
            .a = ba,
            .a_off = 0,
            .a_stride = hv,
            .A_log = A_log,
            .dt_bias = dt_bias,
            .conv_state = conv_state,
            .conv_w = conv_w,
            .q_scale = q_scale,
            .k_scale = k_scale,
            .hk = hk,
            .hv = hv,
            .dk = dk,
            .dv = dv,
            .seq = 2,
        };
        args.seq = 17;
        gdn_prefill_fused_override = false;
        try std.testing.expect((try gdnPreworkFused(s, args)) == null);
        gdn_prefill_fused_override = true;
        args.seq = @import("../hc/hc_prefill.zig").max_seq + 1;
        try std.testing.expect((try gdnPreworkFused(s, args)) == null);
        args.seq = 17;
        args.batch = 3;
        try std.testing.expect((try gdnPreworkFused(s, args)) == null);
        args.batch = 1;
        args.seq = 10;
        try std.testing.expect((try gdnPreworkFused(s, args)) == null);
        args.seq = 2;
        args.dk = 64;
        try std.testing.expect((try gdnPreworkFused(s, args)) == null);
        args.dk = 128;
        gdn_decode_fused_override = false;
        try std.testing.expect((try gdnPreworkFused(s, args)) == null);
        gdn_decode_fused_override = true;
    }
}

test "gdn packed prework: fused at production Hk/Hv serves B=6,8,16 S=1 per row" {
    const s = mlx.gpuStream();
    gdn_prework_override = true;
    defer gdn_prework_override = null;
    gdn_decode_fused_override = true;
    defer gdn_decode_fused_override = null;
    var prng = std.Random.DefaultPrng.init(0xB16);
    const rnd = prng.random();
    const hk: c_int = 16;
    const hv: c_int = 48;
    const dk: c_int = 128;
    const dv: c_int = 128;
    const c_dim: c_int = hk * dk * 2 + hv * dv;
    const seq: c_int = 1;
    const q_scale = bf16Scalar(1.0 / 128.0, s);
    defer _ = mlx.mlx_array_free(q_scale);
    const k_scale = bf16Scalar(@sqrt(1.0 / 128.0), s);
    defer _ = mlx.mlx_array_free(k_scale);
    var ones_w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ones_w);
    const ones_shape = [_]c_int{dk};
    try mlx.check(mlx.mlx_ones(&ones_w, &ones_shape, 1, .bfloat16, s));
    const key_dim: c_int = hk * dk;
    const value_dim: c_int = hv * dv;
    const hv_shape = [_]c_int{hv};
    const A_log = try attn256RandBf16(rnd, &hv_shape, s);
    defer _ = mlx.mlx_array_free(A_log);
    const dt_bias = try attn256RandBf16(rnd, &hv_shape, s);
    defer _ = mlx.mlx_array_free(dt_bias);
    const w_shape = [_]c_int{ c_dim, 4, 1 };
    const conv_w = try attn256RandBf16(rnd, &w_shape, s);
    defer _ = mlx.mlx_array_free(conv_w);

    const row0 = struct {
        fn f(st: mlx.mlx_stream, x: mlx.mlx_array, i: c_int) !mlx.mlx_array {
            const sh = mlx.getShape(x);
            var start: [4]c_int = .{ i, 0, 0, 0 };
            var stop: [4]c_int = .{ i + 1, 0, 0, 0 };
            const strides: [4]c_int = .{ 1, 1, 1, 1 };
            for (1..sh.len) |d| stop[d] = sh[d];
            var out = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&out, x, &start, @intCast(sh.len), &stop, @intCast(sh.len), &strides, @intCast(sh.len), st));
            return out;
        }
    }.f;

    for ([_]c_int{ 6, 8, 16 }) |batch| {
        const qkv_shape = [_]c_int{ batch, seq, c_dim };
        const qkv = try attn256RandBf16(rnd, &qkv_shape, s);
        defer _ = mlx.mlx_array_free(qkv);
        const st_shape = [_]c_int{ batch, 3, c_dim };
        const conv_state = try attn256RandBf16(rnd, &st_shape, s);
        defer _ = mlx.mlx_array_free(conv_state);
        const ba_shape = [_]c_int{ batch, seq, hv };
        const b_in = try attn256RandBf16Scaled(rnd, &ba_shape, 16.0, s);
        defer _ = mlx.mlx_array_free(b_in);
        const a_in = try attn256RandBf16Scaled(rnd, &ba_shape, 16.0, s);
        defer _ = mlx.mlx_array_free(a_in);
        const preb = (try gdnPreworkFused(s, .{
            .qkv = qkv,
            .qkv_off = 0,
            .qkv_stride = c_dim,
            .b = b_in,
            .b_off = 0,
            .b_stride = hv,
            .a = a_in,
            .a_off = 0,
            .a_stride = hv,
            .A_log = A_log,
            .dt_bias = dt_bias,
            .conv_state = conv_state,
            .conv_w = conv_w,
            .q_scale = q_scale,
            .k_scale = k_scale,
            .hk = hk,
            .hv = hv,
            .dk = dk,
            .dv = dv,
            .seq = seq,
            .batch = batch,
        })) orelse return error.FusedDeclined;
        defer inline for (.{ preb.q, preb.k, preb.v, preb.conv_state, preb.g, preb.beta }) |arr| {
            _ = mlx.mlx_array_free(arr);
        };
        var conv_input = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(conv_input);
        {
            const arr = [_]mlx.mlx_array{ conv_state, qkv };
            const vec = mlx.mlx_vector_array_new_data(&arr, 2);
            defer _ = mlx.mlx_vector_array_free(vec);
            try mlx.check(mlx.mlx_concatenate_axis(&conv_input, vec, 1, s));
        }
        var ref_state = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ref_state);
        {
            const start = [_]c_int{ 0, seq, 0 };
            const stop = [_]c_int{ batch, seq + 3, c_dim };
            const strides = [_]c_int{ 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(&ref_state, conv_input, &start, 3, &stop, 3, &strides, 3, s));
        }
        var conv_raw = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(conv_raw);
        try mlx.check(mlx.mlx_conv1d(&conv_raw, conv_input, conv_w, 1, 0, 1, c_dim, s));
        var sig = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sig);
        try mlx.check(mlx.mlx_sigmoid(&sig, conv_raw, s));
        var conv_out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(conv_out);
        try mlx.check(mlx.mlx_multiply(&conv_out, conv_raw, sig, s));
        const strides3 = [_]c_int{ 1, 1, 1 };
        {
            var v_flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(v_flat);
            const start = [_]c_int{ 0, 0, key_dim * 2 };
            const stop = [_]c_int{ batch, seq, key_dim * 2 + value_dim };
            try mlx.check(mlx.mlx_slice(&v_flat, conv_out, &start, 3, &stop, 3, &strides3, 3, s));
            var v_heads = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(v_heads);
            const v_shape = [_]c_int{ batch, seq, hv, dv };
            try mlx.check(mlx.mlx_reshape(&v_heads, v_flat, &v_shape, 4, s));
            const d_v = try attn256MaxDiff(preb.v, v_heads, s);
            if (d_v != 0) std.debug.print("prework B={d} v maxdiff={e}\n", .{ batch, d_v });
            try std.testing.expectEqual(@as(f32, 0.0), d_v);
        }
        inline for (.{
            .{ @as(c_int, 0), key_dim, hk, dk, q_scale, preb.q },
            .{ key_dim, key_dim * 2, hk, dk, k_scale, preb.k },
        }) |case| {
            var flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat);
            const start = [_]c_int{ 0, 0, case[0] };
            const stop = [_]c_int{ batch, seq, case[1] };
            try mlx.check(mlx.mlx_slice(&flat, conv_out, &start, 3, &stop, 3, &strides3, 3, s));
            var heads = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(heads);
            const h_shape = [_]c_int{ batch, seq, case[2], case[3] };
            try mlx.check(mlx.mlx_reshape(&heads, flat, &h_shape, 4, s));
            var nrm = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(nrm);
            try mlx.check(mlx.mlx_fast_rms_norm(&nrm, heads, ones_w, 1e-6, s));
            var scaled = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(scaled);
            try mlx.check(mlx.mlx_multiply(&scaled, nrm, case[4], s));
            const dh = try attn256MaxDiff(case[5], scaled, s);
            if (dh != 0) std.debug.print("prework B={d} q/k maxdiff={e}\n", .{ batch, dh });
            try std.testing.expectEqual(@as(f32, 0.0), dh);
        }
        const ds = try attn256MaxDiff(preb.conv_state, ref_state, s);
        if (ds != 0) std.debug.print("prework B={d} conv_state maxdiff={e}\n", .{ batch, ds });
        try std.testing.expectEqual(@as(f32, 0.0), ds);
        const g_ref = try gdnGateChain(A_log, a_in, dt_bias, .bfloat16, s);
        defer _ = mlx.mlx_array_free(g_ref);
        const dg = try attn256MaxDiff(preb.g, g_ref, s);
        if (dg != 0) std.debug.print("prework B={d} g maxdiff={e}\n", .{ batch, dg });
        try std.testing.expectEqual(@as(f32, 0.0), dg);
        var beta_ref = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(beta_ref);
        try mlx.check(mlx.mlx_sigmoid(&beta_ref, b_in, s));
        const db = try attn256MaxDiff(preb.beta, beta_ref, s);
        if (db != 0) std.debug.print("prework B={d} beta maxdiff={e}\n", .{ batch, db });
        try std.testing.expect(db <= 7.7e-6);
        var args1 = GdnPreworkArgs{
            .qkv = qkv,
            .qkv_off = 0,
            .qkv_stride = c_dim,
            .b = b_in,
            .b_off = 0,
            .b_stride = hv,
            .a = a_in,
            .a_off = 0,
            .a_stride = hv,
            .A_log = A_log,
            .dt_bias = dt_bias,
            .conv_state = conv_state,
            .conv_w = conv_w,
            .q_scale = q_scale,
            .k_scale = k_scale,
            .hk = hk,
            .hv = hv,
            .dk = dk,
            .dv = dv,
            .seq = seq,
            .batch = 1,
        };
        const last: c_int = batch - 1;
        const q1 = try row0(s, qkv, last);
        defer _ = mlx.mlx_array_free(q1);
        const st1 = try row0(s, conv_state, last);
        defer _ = mlx.mlx_array_free(st1);
        const b1 = try row0(s, b_in, last);
        defer _ = mlx.mlx_array_free(b1);
        const a1 = try row0(s, a_in, last);
        defer _ = mlx.mlx_array_free(a1);
        args1.qkv = q1;
        args1.conv_state = st1;
        args1.b = b1;
        args1.a = a1;
        const pre1 = (try gdnPreworkFused(s, args1)) orelse return error.FusedDeclined;
        defer inline for (.{ pre1.q, pre1.k, pre1.v, pre1.conv_state, pre1.g, pre1.beta }) |arr| {
            _ = mlx.mlx_array_free(arr);
        };
        inline for (.{ .{ preb.q, pre1.q }, .{ preb.k, pre1.k }, .{ preb.v, pre1.v }, .{ preb.conv_state, pre1.conv_state }, .{ preb.g, pre1.g }, .{ preb.beta, pre1.beta } }) |pair| {
            const r = try row0(s, pair[0], last);
            defer _ = mlx.mlx_array_free(r);
            try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(r, pair[1], s));
        }
        const y_shape = [_]c_int{ batch, seq, hv, dv };
        const y = try attn256RandBf16Scaled(rnd, &y_shape, 8.0, s);
        defer _ = mlx.mlx_array_free(y);
        const z_shape = [_]c_int{ batch, seq, hv * dv };
        const z = try attn256RandBf16Scaled(rnd, &z_shape, 12.0, s);
        defer _ = mlx.mlx_array_free(z);
        const nw_shape = [_]c_int{dv};
        const norm_w = try attn256RandBf16(rnd, &nw_shape, s);
        defer _ = mlx.mlx_array_free(norm_w);
        const eps_arr = mlx.mlx_array_new_float(1e-6);
        defer _ = mlx.mlx_array_free(eps_arr);
        const fused = (try gdnNormGateFused(s, y, z, 0, hv * dv, norm_w, eps_arr, false, hv, dv, batch, seq)) orelse return error.FusedDeclined;
        defer _ = mlx.mlx_array_free(fused);
        const y1 = try row0(s, y, last);
        defer _ = mlx.mlx_array_free(y1);
        const z1 = try row0(s, z, last);
        defer _ = mlx.mlx_array_free(z1);
        const fused1 = (try gdnNormGateFused(s, y1, z1, 0, hv * dv, norm_w, eps_arr, false, hv, dv, 1, seq)) orelse return error.FusedDeclined;
        defer _ = mlx.mlx_array_free(fused1);
        const fr = try row0(s, fused, last);
        defer _ = mlx.mlx_array_free(fr);
        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(fr, fused1, s));
    }
}

test "gdn_decode.step and recur: bit-identical to prework -> recurrence -> norm-gate -> rotation, states included" {
    const s = mlx.gpuStream();
    gdn_decode_fused_override = true;
    defer gdn_decode_fused_override = null;
    // bf16 everywhere, a Hadamard pack's f16 activations over an f32 state, and qwen4_exp's bf16 over f32.
    for ([_][2]mlx.mlx_dtype{ .{ .bfloat16, .bfloat16 }, .{ .float16, .float32 }, .{ .bfloat16, .float32 } }) |dts|
        for ([_]c_int{ 512, 1024, 2048, 4096 }) |block| try gdnDecodeParityCase(s, dts[0], dts[1], block);
}

test "gdn_decode.recurSeq: bit-identical to prework -> capture recurrence at verify widths" {
    const s = mlx.gpuStream();
    gdn_decode_fused_override = true;
    defer gdn_decode_fused_override = null;
    // An f32 state only: a narrower one is rounded per token, as serial decoding stores it.
    for ([_]mlx.mlx_dtype{ .float16, .bfloat16 }) |dt| {
        var t: c_int = 2;
        while (t <= 8) : (t += 1) try gdnDecodeSeqParityCase(s, dt, .float32, t);
    }
}

test "gdn_decode.recurSeq: a T-row window equals T one-row calls" {
    const s = mlx.gpuStream();
    for ([_][2]mlx.mlx_dtype{ .{ .bfloat16, .bfloat16 }, .{ .float16, .float32 }, .{ .bfloat16, .float32 } }) |dts| {
        var t: c_int = 2;
        while (t <= 8) : (t += 1) try gdnDecodeRowsCase(s, dts[0], dts[1], t);
    }
}

fn gdnDecodeRowsCase(s: mlx.mlx_stream, dt: mlx.mlx_dtype, st: mlx.mlx_dtype, t_len: c_int) !void {
    var prng = std.Random.DefaultPrng.init(0x0E0 + @as(u64, @intCast(t_len)));
    const rnd = prng.random();
    const hk: c_int = 2;
    const hv: c_int = 8;
    const c_dim: c_int = hk * 128 * 2 + hv * 128;
    const g = gdn_decode.Geometry{ .hk = hk, .hv = hv, .dk = 128, .dv = 128 };
    const seq = [_][]const c_int{ &.{ 1, t_len, c_dim }, &.{ 1, t_len, hv * 128 }, &.{ 1, t_len, hv }, &.{ 1, t_len, hv } };
    var rows: [4]mlx.mlx_array = undefined;
    for (&rows, seq, [_]f32{ 1, 1, 16, 16 }) |*r, sh, scale| r.* = try gdnParityRand(rnd, sh, scale, dt, s);
    defer for (rows) |r| {
        _ = mlx.mlx_array_free(r);
    };
    const conv_state = try gdnParityRand(rnd, &[_]c_int{ 1, 3, c_dim }, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(conv_state);
    const ssm_state = try gdnParityRand(rnd, &[_]c_int{ 1, hv, 128, 128 }, 1.0, st, s);
    defer _ = mlx.mlx_array_free(ssm_state);
    const conv_w = try gdnParityRand(rnd, &[_]c_int{ c_dim, 4, 1 }, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(conv_w);
    const A_log = try gdnParityRand(rnd, &[_]c_int{hv}, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(A_log);
    const dt_bias = try gdnParityRand(rnd, &[_]c_int{hv}, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(dt_bias);
    const norm_w = try gdnParityRand(rnd, &[_]c_int{128}, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(norm_w);
    const q_scale = try scalarOf(1.0 / 128.0, dt, s);
    defer _ = mlx.mlx_array_free(q_scale);
    const k_scale = try scalarOf(@sqrt(1.0 / 128.0), dt, s);
    defer _ = mlx.mlx_array_free(k_scale);
    const eps_arr = mlx.mlx_array_new_float(1e-6);
    defer _ = mlx.mlx_array_free(eps_arr);
    var in = gdn_decode.Inputs{
        .qkv = rows[0],
        .z = rows[1],
        .a = rows[2],
        .b = rows[3],
        .conv_state = conv_state,
        .ssm_state = ssm_state,
        .conv_w = conv_w,
        .A_log = A_log,
        .dt_bias = dt_bias,
        .q_scale = q_scale,
        .k_scale = k_scale,
        .norm_w = norm_w,
        .eps = eps_arr,
        .signs = .{ .ctx = null },
    };
    const all = (try gdn_decode.recurSeq(g, t_len, in, s)) orelse return error.FusedDeclined;
    defer for ([_]mlx.mlx_array{ all.y, all.conv_state, all.ssm_state, all.state_seq }) |a| {
        _ = mlx.mlx_array_free(a);
    };
    var conv = conv_state;
    var state = ssm_state;
    var owned: ?gdn_decode.RecurSeq = null;
    defer if (owned) |o| for ([_]mlx.mlx_array{ o.y, o.conv_state, o.ssm_state, o.state_seq }) |a| {
        _ = mlx.mlx_array_free(a);
    };
    var r: c_int = 0;
    while (r < t_len) : (r += 1) {
        var one: [4]mlx.mlx_array = undefined;
        for (&one, rows) |*o, full| {
            const sh = mlx.getShape(full);
            o.* = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(o, full, &[_]c_int{ 0, r, 0 }, 3, &[_]c_int{ 1, r + 1, sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        }
        defer for (one) |o| {
            _ = mlx.mlx_array_free(o);
        };
        in.qkv = one[0];
        in.z = one[1];
        in.a = one[2];
        in.b = one[3];
        in.conv_state = conv;
        in.ssm_state = state;
        const step = (try gdn_decode.recurSeq(g, 1, in, s)) orelse return error.FusedDeclined;
        const ys = mlx.getShape(all.y);
        var want = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(want);
        try mlx.check(mlx.mlx_slice(&want, all.y, &[_]c_int{ 0, r, 0, 0 }, 4, &[_]c_int{ 1, r + 1, ys[2], ys[3] }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(want, step.y, s));
        if (owned) |o| for ([_]mlx.mlx_array{ o.y, o.conv_state, o.ssm_state, o.state_seq }) |a| {
            _ = mlx.mlx_array_free(a);
        };
        owned = step;
        conv = step.conv_state;
        state = step.ssm_state;
    }
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(all.ssm_state, state, s));
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(all.conv_state, conv, s));
}

test "gdn_decode.recurSeqFold: bit-identical to recurSeq -> norm-gate -> conv-input concat" {
    mlx.installErrorHandler();
    const s = mlx.gpuStream();
    gdn_decode_fused_override = true;
    defer gdn_decode_fused_override = null;
    // Test geometry, then Flash Next's (Hk=16, Hv=48: 3 value heads per key head).
    for ([_][2]c_int{ .{ 2, 8 }, .{ 16, 48 } }) |geo|
        for ([_][2]mlx.mlx_dtype{ .{ .bfloat16, .bfloat16 }, .{ .float16, .float32 }, .{ .bfloat16, .float32 } }) |dts|
            for ([_]bool{ false, true }) |swish| {
                var t: c_int = 2;
                while (t <= gdn_decode.FOLD_MAX_SEQ) : (t += 1) try gdnDecodeFoldParityCase(s, dts[0], dts[1], geo[0], geo[1], t, swish);
            };
}

test "gdn_decode.recurSeqFold: a threadgroup the GPU refuses declines cleanly, and MLX keeps working" {
    mlx.installErrorHandler();
    const s = mlx.gpuStream();
    gdn_decode_fused_override = true;
    defer gdn_decode_fused_override = null;
    // 2048 threads exceed every Apple GPU's per-threadgroup limit, the way 1024 exceeds some.
    gdn_decode.fold_nt_override = 2048;
    try gdnDecodeFoldParityCase(s, .bfloat16, .bfloat16, 2, 8, 2, false);
    try std.testing.expect(gdn_decode.foldDeclined(2));
    try std.testing.expect(!mlx.errorPending());
    // Back at 1024 the fold runs and is still bit-identical after the refused dispatch.
    gdn_decode.fold_nt_override = null;
    try gdnDecodeFoldParityCase(s, .bfloat16, .bfloat16, 2, 8, 2, false);
    try std.testing.expect(!gdn_decode.foldDeclined(2));
}

fn gdnDecodeFoldParityCase(s: mlx.mlx_stream, dt: mlx.mlx_dtype, st: mlx.mlx_dtype, hk: c_int, hv: c_int, t_len: c_int, swish: bool) !void {
    var prng = std.Random.DefaultPrng.init(0xF01D + @as(u64, @intCast(t_len * 131 + hv)));
    const rnd = prng.random();
    const dk: c_int = 128;
    const dv: c_int = 128;
    const c_dim: c_int = hk * dk * 2 + hv * dv;
    const value_dim: c_int = hv * dv;
    const g = gdn_decode.Geometry{ .hk = hk, .hv = hv, .dk = dk, .dv = dv };

    const q_scale = try scalarOf(1.0 / 128.0, dt, s);
    defer _ = mlx.mlx_array_free(q_scale);
    const k_scale = try scalarOf(@sqrt(1.0 / 128.0), dt, s);
    defer _ = mlx.mlx_array_free(k_scale);
    const A_log = try gdnParityRand(rnd, &[_]c_int{hv}, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(A_log);
    const dt_bias = try gdnParityRand(rnd, &[_]c_int{hv}, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(dt_bias);
    const qkv = try gdnParityRand(rnd, &[_]c_int{ 1, t_len, c_dim }, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(qkv);
    const z = try gdnParityRand(rnd, &[_]c_int{ 1, t_len, value_dim }, 4.0, dt, s);
    defer _ = mlx.mlx_array_free(z);
    const b_in = try gdnParityRand(rnd, &[_]c_int{ 1, t_len, hv }, 16.0, dt, s);
    defer _ = mlx.mlx_array_free(b_in);
    const a_in = try gdnParityRand(rnd, &[_]c_int{ 1, t_len, hv }, 16.0, dt, s);
    defer _ = mlx.mlx_array_free(a_in);
    const conv_state = try gdnParityRand(rnd, &[_]c_int{ 1, 3, c_dim }, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(conv_state);
    const ssm_state = try gdnParityRand(rnd, &[_]c_int{ 1, hv, dv, dk }, 1.0, st, s);
    defer _ = mlx.mlx_array_free(ssm_state);
    const conv_w = try gdnParityRand(rnd, &[_]c_int{ c_dim, 4, 1 }, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(conv_w);
    const norm_w = try gdnParityRand(rnd, &[_]c_int{dv}, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(norm_w);
    const eps_arr = mlx.mlx_array_new_float(1e-6);
    defer _ = mlx.mlx_array_free(eps_arr);
    const in = gdn_decode.Inputs{
        .qkv = qkv,
        .z = z,
        .a = a_in,
        .b = b_in,
        .conv_state = conv_state,
        .ssm_state = ssm_state,
        .conv_w = conv_w,
        .A_log = A_log,
        .dt_bias = dt_bias,
        .q_scale = q_scale,
        .k_scale = k_scale,
        .norm_w = norm_w,
        .eps = eps_arr,
        .signs = .{ .ctx = null },
    };

    // Today's verify path: recurSeq, the norm-gate kernel, the conv-input concat.
    const ref = (try gdn_decode.recurSeq(g, t_len, in, s)) orelse return error.FusedDeclined;
    defer {
        inline for (.{ ref.y, ref.conv_state, ref.ssm_state, ref.state_seq }) |a| _ = mlx.mlx_array_free(a);
    }
    const ref_gated = (try gdnNormGateFused(s, ref.y, z, 0, value_dim, norm_w, eps_arr, swish, hv, dv, 1, t_len)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(ref_gated);
    var ref_ci = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref_ci);
    {
        const arr = [_]mlx.mlx_array{ conv_state, qkv };
        const vec = mlx.mlx_vector_array_new_data(&arr, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&ref_ci, vec, 1, s));
    }

    const got = (try gdn_decode.recurSeqFold(g, t_len, in, swish, s)) orelse {
        if (gdn_decode.foldDeclined(t_len)) return;
        return error.FoldDeclined;
    };
    defer {
        inline for (.{ got.gated, got.conv_state, got.ssm_state, got.state_seq, got.conv_input }) |a| _ = mlx.mlx_array_free(a);
    }

    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(ref_gated, got.gated, s));
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(ref.conv_state, got.conv_state, s));
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(ref.ssm_state, got.ssm_state, s));
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(ref_ci, got.conv_input, s));
    var ref_head = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref_head);
    var got_head = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(got_head);
    const start = [_]c_int{ 0, 0, 0, 0, 0 };
    const stop = [_]c_int{ t_len - 1, 1, hv, dv, dk };
    const strides = [_]c_int{ 1, 1, 1, 1, 1 };
    try mlx.check(mlx.mlx_slice(&ref_head, ref.state_seq, &start, 5, &stop, 5, &strides, 5, s));
    try mlx.check(mlx.mlx_slice(&got_head, got.state_seq, &start, 5, &stop, 5, &strides, 5, s));
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(ref_head, got_head, s));
}

test "gdn_decode: recurSeqFold and recurSeq decline inputs whose width the kernel would misread" {
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0xDEC1);
    const rnd = prng.random();
    const dt: mlx.mlx_dtype = .bfloat16;
    const hk: c_int = 2;
    const hv: c_int = 8;
    const t_len: c_int = 2;
    const c_dim: c_int = hk * 128 * 2 + hv * 128;
    const g = gdn_decode.Geometry{ .hk = hk, .hv = hv, .dk = 128, .dv = 128 };
    var arrs: [14]mlx.mlx_array = undefined;
    const shapes = [_][]const c_int{
        &.{ 1, t_len, c_dim }, &.{ 1, t_len, hv * 128 }, &.{ 1, t_len, hv }, &.{ 1, t_len, hv }, &.{ 1, 3, c_dim }, &.{ 1, hv, 128, 128 },
        &.{ c_dim, 4, 1 },     &.{hv},                   &.{hv},              &.{128},             &.{ 1, t_len, hv * 128 + 1 }, &.{ 1, t_len, hv + 1 },
        &.{ 1, t_len + 1, c_dim }, &.{ 1, 3, c_dim + 1 },
    };
    for (&arrs, shapes) |*a, sh| a.* = try gdnParityRand(rnd, sh, 1.0, dt, s);
    defer for (arrs) |a| {
        _ = mlx.mlx_array_free(a);
    };
    const q_scale = try scalarOf(1.0 / 128.0, dt, s);
    defer _ = mlx.mlx_array_free(q_scale);
    const eps_arr = mlx.mlx_array_new_float(1e-6);
    defer _ = mlx.mlx_array_free(eps_arr);
    const good = gdn_decode.Inputs{
        .qkv = arrs[0],
        .z = arrs[1],
        .a = arrs[2],
        .b = arrs[3],
        .conv_state = arrs[4],
        .ssm_state = arrs[5],
        .conv_w = arrs[6],
        .A_log = arrs[7],
        .dt_bias = arrs[8],
        .q_scale = q_scale,
        .k_scale = q_scale,
        .norm_w = arrs[9],
        .eps = eps_arr,
        .signs = .{ .ctx = null },
    };
    // Control: the well-formed inputs engage both kernels.
    const ok = (try gdn_decode.recurSeq(g, t_len, good, s)) orelse return error.RecurSeqDeclined;
    inline for (.{ ok.y, ok.conv_state, ok.ssm_state, ok.state_seq }) |a| _ = mlx.mlx_array_free(a);
    const okf = (try gdn_decode.recurSeqFold(g, t_len, good, false, s)) orelse return error.FoldDeclined;
    inline for (.{ okf.gated, okf.conv_state, okf.ssm_state, okf.state_seq, okf.conv_input }) |a| _ = mlx.mlx_array_free(a);

    // z one column wide ([1,2,1025]): row 1 would start at row 0's extra column.
    var bad = good;
    bad.z = arrs[10];
    try std.testing.expect((try gdn_decode.recurSeqFold(g, t_len, bad, false, s)) == null);
    // The same class on every other per-token row and state the kernels index.
    for ([_]struct { field: enum { a, qkv, conv }, arr: mlx.mlx_array }{ .{ .field = .a, .arr = arrs[11] }, .{ .field = .qkv, .arr = arrs[12] }, .{ .field = .conv, .arr = arrs[13] } }) |case| {
        bad = good;
        switch (case.field) {
            .a => bad.a = case.arr,
            .qkv => bad.qkv = case.arr,
            .conv => bad.conv_state = case.arr,
        }
        try std.testing.expect((try gdn_decode.recurSeqFold(g, t_len, bad, false, s)) == null);
        try std.testing.expect((try gdn_decode.recurSeq(g, t_len, bad, s)) == null);
    }
    // The same well-formed inputs on a CPU stream: the Metal kernels decline, nothing latches.
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    try std.testing.expect((try gdn_decode.recurSeqFold(g, t_len, good, false, cpu)) == null);
    try std.testing.expect((try gdn_decode.recurSeq(g, t_len, good, cpu)) == null);
    try std.testing.expect(!mlx.errorPending());
}

test "gdn_decode.recurTree: every node of a draft tree equals a chain over its own path" {
    // A tree of 1 or 2 nodes has fewer rows than the conv window it commits.
    for ([_]usize{ 1, 2, 8 }) |nodes| {
        try gdnDecodeTreeCase(.bfloat16, nodes);
        try gdnDecodeTreeCase(.float32, nodes);
    }
}

fn gdnDecodeTreeCase(st: mlx.mlx_dtype, nodes: usize) !void {
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x7EE);
    const rnd = prng.random();
    const dt: mlx.mlx_dtype = .bfloat16;
    const hk: c_int = 2;
    const hv: c_int = 8;
    const c_dim: c_int = hk * 128 * 2 + hv * 128;
    const g = gdn_decode.Geometry{ .hk = hk, .hv = hv, .dk = 128, .dv = 128 };
    // Rows 1 and 2 follow the root, 3 and 4 follow row 1, 5 follows 2, 6 and 7 follow 3;
    // every prefix is a tree of its own.
    const all_parents = [_]i32{ -1, 0, 0, 1, 1, 2, 3, 3 };
    const all_paths = [_][]const i32{ &.{0}, &.{ 0, 1 }, &.{ 0, 2 }, &.{ 0, 1, 3 }, &.{ 0, 1, 4 }, &.{ 0, 2, 5 }, &.{ 0, 1, 3, 6 }, &.{ 0, 1, 3, 7 } };
    const parents = all_parents[0..nodes];
    const paths = all_paths[0..nodes];
    const w: c_int = @intCast(nodes);
    const qkv = try gdnParityRand(rnd, &[_]c_int{ 1, w, c_dim }, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(qkv);
    const a = try gdnParityRand(rnd, &[_]c_int{ 1, w, hv }, 16.0, dt, s);
    defer _ = mlx.mlx_array_free(a);
    const b = try gdnParityRand(rnd, &[_]c_int{ 1, w, hv }, 16.0, dt, s);
    defer _ = mlx.mlx_array_free(b);
    var fixed: [7]mlx.mlx_array = undefined;
    const fixed_shapes = [_][]const c_int{ &.{ 1, 3, c_dim }, &.{ 1, hv, 128, 128 }, &.{ c_dim, 4, 1 }, &.{hv}, &.{hv}, &.{128}, &.{ 1, w, hv * 128 } };
    for (&fixed, fixed_shapes, 0..) |*f, sh, i| f.* = try gdnParityRand(rnd, sh, 1.0, if (i == 1) st else dt, s);
    defer for (fixed) |f| {
        _ = mlx.mlx_array_free(f);
    };
    const q_scale = try scalarOf(1.0 / 128.0, dt, s);
    defer _ = mlx.mlx_array_free(q_scale);
    const eps_arr = mlx.mlx_array_new_float(1e-6);
    defer _ = mlx.mlx_array_free(eps_arr);
    const inputs = struct {
        fn of(q: mlx.mlx_array, aa: mlx.mlx_array, bb: mlx.mlx_array, f: [7]mlx.mlx_array, qs: mlx.mlx_array, e: mlx.mlx_array) gdn_decode.Inputs {
            return .{ .qkv = q, .z = f[6], .a = aa, .b = bb, .conv_state = f[0], .ssm_state = f[1], .conv_w = f[2], .A_log = f[3], .dt_bias = f[4], .q_scale = qs, .k_scale = qs, .norm_w = f[5], .eps = e, .signs = .{ .ctx = null } };
        }
    }.of;
    var table: [6 * all_parents.len]i32 = undefined;
    gdn_decode.treeTable(parents, table[0 .. 6 * nodes]);
    const tab = mlx.mlx_array_new_data(&table, &[_]c_int{6 * w}, 1, .int32);
    defer _ = mlx.mlx_array_free(tab);
    const tree = (try gdn_decode.recurTree(g, w, inputs(qkv, a, b, fixed, q_scale, eps_arr), null, tab, s)) orelse return error.TreeDeclined;
    defer inline for (.{ tree.y, tree.conv_input, tree.prework.k, tree.prework.v, tree.prework.gb }) |x| {
        _ = mlx.mlx_array_free(x);
    };
    // Read in place from a joined [qkv | z | a | b] projection, the same bits.
    {
        var all = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(all);
        const arr = [_]mlx.mlx_array{ qkv, fixed[6], a, b };
        const vec = mlx.mlx_vector_array_new_data(&arr, arr.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&all, vec, 2, s));
        const a_off = c_dim + hv * 128;
        var parts: [3]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
        defer for (parts) |x| {
            _ = mlx.mlx_array_free(x);
        };
        try sliceQkvPart(&parts[0], all, 0, c_dim, s);
        try sliceQkvPart(&parts[1], all, a_off, hv, s);
        try sliceQkvPart(&parts[2], all, a_off + hv, hv, s);
        const j = (try gdn_decode.recurTree(g, w, inputs(parts[0], parts[1], parts[2], fixed, q_scale, eps_arr), .{ .all = all, .a_off = a_off, .b_off = a_off + hv }, tab, s)) orelse return error.JoinedDeclined;
        defer inline for (.{ j.y, j.conv_input, j.prework.k, j.prework.v, j.prework.gb }) |x| {
            _ = mlx.mlx_array_free(x);
        };
        inline for (.{ .{ tree.y, j.y }, .{ tree.conv_input, j.conv_input }, .{ tree.prework.k, j.prework.k }, .{ tree.prework.v, j.prework.v }, .{ tree.prework.gb, j.prework.gb } }) |pair|
            try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(pair[0], pair[1], s));
    }
    // The commit slices its conv window from [conv_state; window rows].
    {
        var want = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(want);
        const arr = [_]mlx.mlx_array{ fixed[0], qkv };
        const vec = mlx.mlx_vector_array_new_data(&arr, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&want, vec, 1, s));
        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(want, tree.conv_input, s));
    }
    for (paths, 0..) |path, node| {
        const idx = mlx.mlx_array_new_data(path.ptr, &[_]c_int{@intCast(path.len)}, 1, .int32);
        defer _ = mlx.mlx_array_free(idx);
        var rows: [3]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
        defer for (rows) |r| {
            _ = mlx.mlx_array_free(r);
        };
        const srcs = [_]mlx.mlx_array{ qkv, a, b };
        for (srcs, &rows) |src, *dst| try mlx.check(mlx.mlx_take_axis(dst, src, idx, 1, s));
        const n: c_int = @intCast(path.len);
        var chain_out: [4]mlx.mlx_array = undefined;
        var chain_state: mlx.mlx_array = undefined;
        if (n == 1) {
            const r = (try gdn_decode.recur(g, inputs(rows[0], rows[1], rows[2], fixed, q_scale, eps_arr), s)) orelse return error.ChainDeclined;
            chain_out = .{ r.y, r.conv_state, r.ssm_state, .{ .ctx = null } };
            chain_state = r.ssm_state;
        } else {
            const r = (try gdn_decode.recurSeq(g, n, inputs(rows[0], rows[1], rows[2], fixed, q_scale, eps_arr), s)) orelse return error.ChainDeclined;
            chain_out = .{ r.y, r.conv_state, r.ssm_state, r.state_seq };
            chain_state = r.ssm_state;
        }
        defer for (chain_out) |x| if (x.ctx != null) {
            _ = mlx.mlx_array_free(x);
        };
        // The node's y row equals the chain's last row.
        var want_y = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(want_y);
        try mlx.check(mlx.mlx_slice(&want_y, chain_out[0], &[_]c_int{ 0, n - 1, 0, 0 }, 4, &[_]c_int{ 1, n, hv, 128 }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
        var got_y = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(got_y);
        const nd: c_int = @intCast(node);
        try mlx.check(mlx.mlx_slice(&got_y, tree.y, &[_]c_int{ 0, nd, 0, 0 }, 4, &[_]c_int{ 1, nd + 1, hv, 128 }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(want_y, got_y, s));
        // Replaying the node's path from the round's input state gives the chain's
        // state, and the conv window its path's last three conv inputs.
        var conv_rows: [3]i32 = undefined;
        for (0..3) |i| {
            const dd: i32 = n - 3 + @as(i32, @intCast(i));
            conv_rows[i] = if (dd < 0) dd + 3 else 3 + path[@intCast(dd)];
        }
        const replayed = (try gdn_decode.replay(g, fixed[1], tree.prework, path, tree.conv_input, conv_rows, s)) orelse return error.ReplayDeclined;
        defer inline for (.{ replayed.ssm_state, replayed.conv_state }) |x| {
            _ = mlx.mlx_array_free(x);
        };
        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(chain_state, replayed.ssm_state, s));
        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(chain_out[1], replayed.conv_state, s));
    }
}

/// A qwen4_exp GatedDeltaNet layer at test scale (Hk 2, Hv 8, Dk = Dv = 128) whose config is
/// parsed, so its state dtype is the policy's. Hidden 1: a projection row is one product, the
/// same bits at any row count, so only the GDN arms themselves can differ.
const Qwen4GdnRig = struct {
    xfm: Transformer,
    la: LinearAttnWeights,

    const hv: c_int = 8;
    const conv_dim: c_int = 2 * 2 * 128 + hv * 128;

    /// In place: the compiled gate closure points at `xfm`. Takes ownership of A_log and dt_bias.
    fn init(rig: *Qwen4GdnRig, A_log: mlx.mlx_array, dt_bias: mlx.mlx_array, rnd: std.Random, s: mlx.mlx_stream) !void {
        const cfg = try model_mod.parseConfigFromJson(testing.allocator,
            \\{"model_type": "qwen4_exp", "text_config": {"hidden_size": 1, "ple_embed_dim": 16,
            \\ "num_hidden_layers": 4, "full_attention_interval": 4, "ple_layer_ids": [1],
            \\ "linear_num_key_heads": 2, "linear_num_value_heads": 8, "linear_key_head_dim": 128,
            \\ "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "mamba_ssm_dtype": "float32"}}
        );
        const cache = try KVCache.init(testing.allocator, 0);
        rig.xfm = std.mem.zeroInit(Transformer, .{ .s = s, .allocator = testing.allocator, .cache = cache, .config = cfg });
        rig.xfm.compileGdnGate();
        var w: [6]mlx.mlx_array = undefined;
        for (&w, [_][]const c_int{ &.{ 1, conv_dim }, &.{ 1, hv * 128 }, &.{ 1, hv }, &.{ 1, hv }, &.{ conv_dim, 4, 1 }, &.{128} }) |*x, sh| x.* = try attn256RandBf16(rnd, sh, s);
        const none = mlx.mlx_array{ .ctx = null };
        rig.la = .{ .qkv_w = w[0], .qkv_s = none, .qkv_b = none, .z_w = w[1], .z_s = none, .z_b = none, .a_w = w[2], .a_s = none, .a_b = none, .b_w = w[3], .b_s = none, .b_b = none, .conv1d_w = w[4], .A_log = A_log, .dt_bias = dt_bias, .norm_w = w[5], .out_w = none, .out_s = none, .out_b = none };
    }

    fn deinit(rig: *Qwen4GdnRig) void {
        for ([_]mlx.mlx_array{ rig.la.qkv_w, rig.la.z_w, rig.la.a_w, rig.la.b_w, rig.la.conv1d_w, rig.la.A_log, rig.la.dt_bias, rig.la.norm_w }) |a| _ = mlx.mlx_array_free(a);
        if (rig.xfm.compiled_gdn_gate) |c| _ = mlx.mlx_closure_free(c);
        for ([_]?mlx.mlx_array{ rig.xfm.gdn_ones_w, rig.xfm.gdn_q_scale, rig.xfm.gdn_k_scale, rig.xfm.gdn_eps }) |o| if (o) |a| {
            _ = mlx.mlx_array_free(a);
        };
        rig.xfm.cache.deinit();
    }

    /// The arms a token can take: serial decode, the verify kernels, batched slots, prefill.
    const Arm = enum { serial, fold, recur_seq, tree, batched_decode, batched_verify, prefill, prefill_composed };

    /// `x` [1, n, 1] through `arm` from `state` [1, Hv, 128, 128] (cast to the policy's dtype):
    /// the outputs [1, n, Hv*128] and the final state, row 0 of a batched arm.
    fn run(rig: *Qwen4GdnRig, arm: Arm, x: mlx.mlx_array, state: mlx.mlx_array) !struct { y: mlx.mlx_array, state: mlx.mlx_array } {
        const s = rig.xfm.s;
        const n = mlx.getShape(x)[1];
        const rows: c_int = if (arm == .batched_decode or arm == .batched_verify) 2 else 1;
        const width: c_int = switch (arm) {
            .serial, .batched_decode => 1,
            .prefill, .prefill_composed => n,
            else => 8,
        };
        const xs = [2]mlx.mlx_array{ x, x };
        const sts = [2]mlx.mlx_array{ state, state };
        var xb = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xb);
        var st = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(st);
        inline for (.{ .{ &xb, &xs }, .{ &st, &sts } }) |p| {
            const vec = mlx.mlx_vector_array_new_data(p[1], @intCast(rows));
            defer _ = mlx.mlx_vector_array_free(vec);
            try mlx.check(mlx.mlx_concatenate_axis(p[0], vec, 0, s));
        }
        var e = SSMCacheEntry{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = true };
        defer inline for (.{ e.conv_state, e.ssm_state, e.spec_state_seq, e.spec_conv_input, e.spec_state_in, e.spec_prework.k, e.spec_prework.v, e.spec_prework.gb }) |a| if (a.ctx != null) {
            _ = mlx.mlx_array_free(a);
        };
        try mlx.check(mlx.mlx_zeros(&e.conv_state, &[_]c_int{ rows, 3, conv_dim }, 3, .bfloat16, s));
        try mlx.check(mlx.mlx_astype(&e.ssm_state, st, rig.xfm.ssmStateDtype(), s));

        rig.xfm.spec_capture_ssm = switch (arm) {
            .fold, .recur_seq, .tree, .batched_verify => true,
            else => false,
        };
        defer rig.xfm.spec_capture_ssm = false;
        if (arm == .recur_seq) gdn_verify_fold_env = false;
        defer gdn_verify_fold_env = null;
        if (arm == .prefill_composed) gdn_prework_override = false;
        defer gdn_prework_override = null;
        // A chain of 8 as a draft tree, committed whole.
        const chain = [_]i32{ -1, 0, 1, 2, 3, 4, 5, 6 };
        var table: [6 * chain.len]i32 = undefined;
        gdn_decode.treeTable(&chain, &table);
        const tab = mlx.mlx_array_new_data(&table, &[_]c_int{table.len}, 1, .int32);
        defer _ = mlx.mlx_array_free(tab);
        const tree = SpecTree{ .parents = tab, .attn = .{ .depth = .{ .ctx = null }, .path = .{ .ctx = null }, .max_depth = 0 } };
        if (arm == .tree) rig.xfm.spec_tree = &tree;
        defer rig.xfm.spec_tree = null;

        const ys = try testing.allocator.alloc(mlx.mlx_array, @intCast(@divExact(n, width)));
        defer testing.allocator.free(ys);
        var n_ys: usize = 0;
        defer for (ys[0..n_ys]) |y| {
            _ = mlx.mlx_array_free(y);
        };
        var t: c_int = 0;
        while (t < n) : (t += width) {
            var xw = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(xw);
            try mlx.check(mlx.mlx_slice(&xw, xb, &[_]c_int{ 0, t, 0 }, 3, &[_]c_int{ rows, t + width, 1 }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
            ys[n_ys] = try rig.xfm.gatedDeltaNetProjected(xw, &rig.la, &e, 0, rows, width, arm == .prefill or arm == .prefill_composed, null, true);
            n_ys += 1;
            if (arm == .tree) try ssmCommitTreePath(&e, &[_]i32{ 0, 1, 2, 3, 4, 5, 6, 7 }, .{ 8, 9, 10 }, s);
            if (@rem(t, 64) == 0) try mlx.check(mlx.mlx_array_eval(e.ssm_state));
        }
        var y_all = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(y_all);
        const vec = mlx.mlx_vector_array_new_data(ys.ptr, n_ys);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&y_all, vec, 1, s));
        var out: [2]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new() };
        for (&out, [_]mlx.mlx_array{ y_all, e.ssm_state }) |*o, a| {
            const sh = mlx.getShape(a);
            var stop: [4]c_int = undefined;
            for (sh, 0..) |d, i| stop[i] = d;
            stop[0] = 1;
            try mlx.check(mlx.mlx_slice(o, a, &[_]c_int{ 0, 0, 0, 0 }, sh.len, &stop, sh.len, &[_]c_int{ 1, 1, 1, 1 }, sh.len, s));
        }
        return .{ .y = out[0], .state = out[1] };
    }
};

test "qwen4_exp GDN: a slow-decay head forgets at its own rate on every arm" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0xDECA1);
    // a = dt_bias = 0, so g = exp(-exp(A_log) * ln 2); bf16 rounds g to 1.0 below a rate of ~2e-3.
    const rates = [8]f32{ 2e-5, 1e-4, 5e-4, 1e-3, 1.5e-3, 3e-3, 1e-2, 5e-2 };
    var a_log: [8]f32 = undefined;
    for (&a_log, rates) |*l, r| l.* = @log(r / std.math.ln2);
    const a_log32 = mlx.mlx_array_new_data(&a_log, &[_]c_int{8}, 1, .float32);
    defer _ = mlx.mlx_array_free(a_log32);
    var A_log = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&A_log, a_log32, .bfloat16, s));
    var dt_bias = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_zeros(&dt_bias, &[_]c_int{8}, 1, .bfloat16, s));
    var rig: Qwen4GdnRig = undefined;
    try rig.init(A_log, dt_bias, prng.random(), s);
    defer rig.deinit();

    const n: c_int = 512;
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_zeros(&x, &[_]c_int{ 1, n, 1 }, 3, .bfloat16, s));
    var ones = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ones);
    try mlx.check(mlx.mlx_ones(&ones, &[_]c_int{ 1, 8, 128, 128 }, 4, .float32, s));
    // q = k = v = 0: the state only decays, to g^n of the bf16 A_log the kernels read.
    var a_read = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a_read);
    try mlx.check(mlx.mlx_astype(&a_read, rig.la.A_log, .float32, s));
    try mlx.check(mlx.mlx_array_eval(a_read));
    const a_bf = mlx.mlx_array_data_float32(a_read) orelse return error.InvalidDtype;
    const nf: f64 = @floatFromInt(n);
    var truth: [8]f64 = undefined;
    for (&truth, 0..) |*w, h| w.* = @exp(-nf * @exp(@as(f64, a_bf[h])) * std.math.ln2);

    var serial: ?mlx.mlx_array = null;
    defer if (serial) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (std.enums.values(Qwen4GdnRig.Arm)) |arm| {
        const r = try rig.run(arm, x, ones);
        _ = mlx.mlx_array_free(r.y);
        defer if (serial == null) {
            serial = r.state;
        } else {
            _ = mlx.mlx_array_free(r.state);
        };
        var f = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(f);
        try mlx.check(mlx.mlx_astype(&f, r.state, .float32, s));
        var fc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(fc);
        try mlx.check(mlx.mlx_contiguous(&fc, f, false, s));
        try mlx.check(mlx.mlx_array_eval(fc));
        const v = mlx.mlx_array_data_float32(fc) orelse return error.InvalidDtype;
        var worst: f64 = 0;
        for (0..8) |h| for (v[h * 128 * 128 .. (h + 1) * 128 * 128]) |e| {
            worst = @max(worst, @abs(@as(f64, e) - truth[h]) / truth[h]);
        };
        if (worst > 1e-3) std.debug.print("[gdn decay] {s}: worst relative error {e} vs g^n\n", .{ @tagName(arm), worst });
        try testing.expect(worst <= 1e-3);
        if (serial) |ref| try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(ref, r.state, s));
    }
    // A cold prefill allocates the state in the policy's dtype.
    var cold = SSMCacheEntry{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
    defer inline for (.{ cold.conv_state, cold.ssm_state }) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a);
    };
    var x32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x32);
    try mlx.check(mlx.mlx_slice(&x32, x, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ 1, 32, 1 }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    _ = mlx.mlx_array_free(try rig.xfm.gatedDeltaNetProjected(x32, &rig.la, &cold, 0, 1, 32, true, null, true));
    try testing.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(cold.ssm_state));
}

test "qwen4_exp GDN: verify windows, batched rows and a prefill-width dispatch equal serial decode bit for bit" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x5E71A1);
    const rnd = prng.random();
    var rig: Qwen4GdnRig = undefined;
    try rig.init(try attn256RandBf16Scaled(rnd, &.{8}, 4.0, s), try attn256RandBf16(rnd, &.{8}, s), rnd, s);
    defer rig.deinit();
    const x = try attn256RandBf16Scaled(rnd, &.{ 1, 8, 1 }, 8.0, s);
    defer _ = mlx.mlx_array_free(x);
    const state = try attn256RandBf16(rnd, &.{ 1, 8, 128, 128 }, s);
    defer _ = mlx.mlx_array_free(state);

    const ref = try rig.run(.serial, x, state);
    defer inline for (.{ ref.y, ref.state }) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for ([_]Qwen4GdnRig.Arm{ .fold, .recur_seq, .tree, .batched_decode, .batched_verify, .prefill, .prefill_composed }) |arm| {
        const r = try rig.run(arm, x, state);
        defer inline for (.{ r.y, r.state }) |a| {
            _ = mlx.mlx_array_free(a);
        };
        const dy = try attn256MaxDiff(ref.y, r.y, s);
        const ds = try attn256MaxDiff(ref.state, r.state, s);
        if (dy != 0 or ds != 0) std.debug.print("[gdn serial==verify] {s}: y max|diff| {e}, state max|diff| {e}\n", .{ @tagName(arm), dy, ds });
        try testing.expect(dy == 0 and ds == 0);
    }
}

fn gdnDecodeSeqParityCase(s: mlx.mlx_stream, dt: mlx.mlx_dtype, st: mlx.mlx_dtype, t_len: c_int) !void {
    var prng = std.Random.DefaultPrng.init(0x5EC0 + @as(u64, @intCast(t_len)));
    const rnd = prng.random();
    const hk: c_int = 2;
    const hv: c_int = 8;
    const dk: c_int = 128;
    const dv: c_int = 128;
    const c_dim: c_int = hk * dk * 2 + hv * dv;
    const value_dim: c_int = hv * dv;

    const q_scale = try scalarOf(1.0 / 128.0, dt, s);
    defer _ = mlx.mlx_array_free(q_scale);
    const k_scale = try scalarOf(@sqrt(1.0 / 128.0), dt, s);
    defer _ = mlx.mlx_array_free(k_scale);
    const A_log = try gdnParityRand(rnd, &[_]c_int{hv}, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(A_log);
    const dt_bias = try gdnParityRand(rnd, &[_]c_int{hv}, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(dt_bias);
    const qkv = try gdnParityRand(rnd, &[_]c_int{ 1, t_len, c_dim }, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(qkv);
    const z = try gdnParityRand(rnd, &[_]c_int{ 1, t_len, value_dim }, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(z);
    const b_in = try gdnParityRand(rnd, &[_]c_int{ 1, t_len, hv }, 16.0, dt, s);
    defer _ = mlx.mlx_array_free(b_in);
    const a_in = try gdnParityRand(rnd, &[_]c_int{ 1, t_len, hv }, 16.0, dt, s);
    defer _ = mlx.mlx_array_free(a_in);
    const conv_state = try gdnParityRand(rnd, &[_]c_int{ 1, 3, c_dim }, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(conv_state);
    const ssm_state = try gdnParityRand(rnd, &[_]c_int{ 1, hv, dv, dk }, 1.0, st, s);
    defer _ = mlx.mlx_array_free(ssm_state);
    const conv_w = try gdnParityRand(rnd, &[_]c_int{ c_dim, 4, 1 }, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(conv_w);
    const norm_w = try gdnParityRand(rnd, &[_]c_int{dv}, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(norm_w);
    const eps_arr = mlx.mlx_array_new_float(1e-6);
    defer _ = mlx.mlx_array_free(eps_arr);

    // Composed chain: fused prework at seq = T, then the capture recurrence.
    const pre = (try gdnPreworkFused(s, .{
        .qkv = qkv,
        .qkv_off = 0,
        .qkv_stride = c_dim,
        .b = b_in,
        .b_off = 0,
        .b_stride = hv,
        .a = a_in,
        .a_off = 0,
        .a_stride = hv,
        .A_log = A_log,
        .dt_bias = dt_bias,
        .conv_state = conv_state,
        .conv_w = conv_w,
        .q_scale = q_scale,
        .k_scale = k_scale,
        .hk = hk,
        .hv = hv,
        .dk = dk,
        .dv = dv,
        .seq = t_len,
        .g_dtype = gdn_decode.gateDtype(dt, st),
    })) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(pre.q);
    defer _ = mlx.mlx_array_free(pre.k);
    defer _ = mlx.mlx_array_free(pre.v);
    defer _ = mlx.mlx_array_free(pre.conv_state);
    defer _ = mlx.mlx_array_free(pre.g);
    defer _ = mlx.mlx_array_free(pre.beta);

    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{ 1, t_len, hv, dv }, 4, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{ t_len, 1, hv, dv, dk }, 5, st));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{ 1, hv, dv, dk }, 4, st));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32, dv, hv));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 4, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "InT", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "StT", st));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "OutT", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dk", dk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dv", dv));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hk", hk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hv", hv));
    const T_scalar = mlx.mlx_array_new_int(t_len);
    defer _ = mlx.mlx_array_free(T_scalar);
    const seq_stride = mlx.mlx_array_new_int(hv * dv * dk);
    defer _ = mlx.mlx_array_free(seq_stride);
    const inputs_arr = [_]mlx.mlx_array{ pre.q, pre.k, pre.v, pre.g, pre.beta, ssm_state, T_scalar, seq_stride };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, try getGdnKernelSeq(false), inputs_vec, config, s));
    var y_ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y_ref);
    var seq_ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(seq_ref);
    var state_ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(state_ref);
    try mlx.check(mlx.mlx_vector_array_get(&y_ref, outputs_vec, 0));
    try mlx.check(mlx.mlx_vector_array_get(&seq_ref, outputs_vec, 1));
    try mlx.check(mlx.mlx_vector_array_get(&state_ref, outputs_vec, 2));

    const got = (try gdn_decode.recurSeq(.{ .hk = hk, .hv = hv, .dk = dk, .dv = dv }, t_len, .{
        .qkv = qkv,
        .z = z,
        .a = a_in,
        .b = b_in,
        .conv_state = conv_state,
        .ssm_state = ssm_state,
        .conv_w = conv_w,
        .A_log = A_log,
        .dt_bias = dt_bias,
        .q_scale = q_scale,
        .k_scale = k_scale,
        .norm_w = norm_w,
        .eps = eps_arr,
        .signs = .{ .ctx = null },
    }, s)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(got.y);
    defer _ = mlx.mlx_array_free(got.conv_state);
    defer _ = mlx.mlx_array_free(got.ssm_state);
    defer _ = mlx.mlx_array_free(got.state_seq);

    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(y_ref, got.y, s));
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(pre.conv_state, got.conv_state, s));
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(state_ref, got.ssm_state, s));
    // state_seq[T-1] is never written (capture-tail trim); compare the rest.
    var seq_ref_head = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(seq_ref_head);
    var seq_got_head = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(seq_got_head);
    const start = [_]c_int{ 0, 0, 0, 0, 0 };
    const stop = [_]c_int{ t_len - 1, 1, hv, dv, dk };
    const strides = [_]c_int{ 1, 1, 1, 1, 1 };
    try mlx.check(mlx.mlx_slice(&seq_ref_head, seq_ref, &start, 5, &stop, 5, &strides, 5, s));
    try mlx.check(mlx.mlx_slice(&seq_got_head, got.state_seq, &start, 5, &stop, 5, &strides, 5, s));
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(seq_ref_head, seq_got_head, s));
}

fn gdnDecodeParityCase(s: mlx.mlx_stream, dt: mlx.mlx_dtype, st: mlx.mlx_dtype, block: c_int) !void {
    var prng = std.Random.DefaultPrng.init(0xB0B5A1);
    const rnd = prng.random();
    const hk: c_int = 2;
    const dk: c_int = 128;
    const dv: c_int = 128;
    const hv: c_int = @divExact(2 * block, dv);
    const c_dim: c_int = hk * dk * 2 + hv * dv;
    const value_dim: c_int = hv * dv;

    const q_scale = try scalarOf(1.0 / 128.0, dt, s);
    defer _ = mlx.mlx_array_free(q_scale);
    const k_scale = try scalarOf(@sqrt(1.0 / 128.0), dt, s);
    defer _ = mlx.mlx_array_free(k_scale);
    const A_log = try gdnParityRand(rnd, &[_]c_int{hv}, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(A_log);
    const dt_bias = try gdnParityRand(rnd, &[_]c_int{hv}, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(dt_bias);
    const qkv = try gdnParityRand(rnd, &[_]c_int{ 1, 1, c_dim }, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(qkv);
    const z = try gdnParityRand(rnd, &[_]c_int{ 1, 1, value_dim }, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(z);
    const b_in = try gdnParityRand(rnd, &[_]c_int{ 1, 1, hv }, 16.0, dt, s);
    defer _ = mlx.mlx_array_free(b_in);
    const a_in = try gdnParityRand(rnd, &[_]c_int{ 1, 1, hv }, 16.0, dt, s);
    defer _ = mlx.mlx_array_free(a_in);
    const conv_state = try gdnParityRand(rnd, &[_]c_int{ 1, 3, c_dim }, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(conv_state);
    const ssm_state = try gdnParityRand(rnd, &[_]c_int{ 1, hv, dv, dk }, 1.0, st, s);
    defer _ = mlx.mlx_array_free(ssm_state);
    const conv_w = try gdnParityRand(rnd, &[_]c_int{ c_dim, 4, 1 }, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(conv_w);
    const norm_w = try gdnParityRand(rnd, &[_]c_int{dv}, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(norm_w);
    const eps_arr = mlx.mlx_array_new_float(1e-6);
    defer _ = mlx.mlx_array_free(eps_arr);
    const sg = try testing.allocator.alloc(f32, @intCast(value_dim));
    defer testing.allocator.free(sg);
    for (sg) |*e| e.* = if (rnd.boolean()) 1.0 else -1.0;
    const signs = mlx.mlx_array_new_data(sg.ptr, &[_]c_int{value_dim}, 1, .float32);
    defer _ = mlx.mlx_array_free(signs);

    // Composed chain: the three production kernels + the input rotation.
    const pre = (try gdnPreworkFused(s, .{
        .qkv = qkv,
        .qkv_off = 0,
        .qkv_stride = c_dim,
        .b = b_in,
        .b_off = 0,
        .b_stride = hv,
        .a = a_in,
        .a_off = 0,
        .a_stride = hv,
        .A_log = A_log,
        .dt_bias = dt_bias,
        .conv_state = conv_state,
        .conv_w = conv_w,
        .q_scale = q_scale,
        .k_scale = k_scale,
        .hk = hk,
        .hv = hv,
        .dk = dk,
        .dv = dv,
        .seq = 1,
        .g_dtype = gdn_decode.gateDtype(dt, st),
    })) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(pre.q);
    defer _ = mlx.mlx_array_free(pre.k);
    defer _ = mlx.mlx_array_free(pre.v);
    defer _ = mlx.mlx_array_free(pre.conv_state);
    defer _ = mlx.mlx_array_free(pre.g);
    defer _ = mlx.mlx_array_free(pre.beta);

    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{ 1, 1, hv, dv }, 4, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{ 1, hv, dv, dk }, 4, st));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32, dv, hv));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 4, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "InT", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "StT", st));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "OutT", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dk", dk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dv", dv));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hk", hk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hv", hv));
    const T_scalar = mlx.mlx_array_new_int(1);
    defer _ = mlx.mlx_array_free(T_scalar);
    const inputs_arr = [_]mlx.mlx_array{ pre.q, pre.k, pre.v, pre.g, pre.beta, ssm_state, T_scalar };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, try getGdnKernel(false), inputs_vec, config, s));
    var y_ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y_ref);
    var state_ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(state_ref);
    try mlx.check(mlx.mlx_vector_array_get(&y_ref, outputs_vec, 0));
    try mlx.check(mlx.mlx_vector_array_get(&state_ref, outputs_vec, 1));
    const flat_ref = (try gdnNormGateFused(s, y_ref, z, 0, value_dim, norm_w, eps_arr, true, hv, dv, 1, 1)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(flat_ref);
    const rot_ref = try rht.transform(flat_ref, signs, block, false, s);
    defer _ = mlx.mlx_array_free(rot_ref);

    const geo = gdn_decode.Geometry{ .hk = hk, .hv = hv, .dk = dk, .dv = dv };
    const in = gdn_decode.Inputs{
        .qkv = qkv,
        .z = z,
        .a = a_in,
        .b = b_in,
        .conv_state = conv_state,
        .ssm_state = ssm_state,
        .conv_w = conv_w,
        .A_log = A_log,
        .dt_bias = dt_bias,
        .q_scale = q_scale,
        .k_scale = k_scale,
        .norm_w = norm_w,
        .eps = eps_arr,
        .signs = signs,
    };
    const got = (try gdn_decode.step(geo, block, in, s)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(got.rot);
    defer _ = mlx.mlx_array_free(got.conv_state);
    defer _ = mlx.mlx_array_free(got.ssm_state);

    try std.testing.expectEqual(st, mlx.mlx_array_dtype(got.ssm_state));
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(pre.conv_state, got.conv_state, s));
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(state_ref, got.ssm_state, s));
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(rot_ref, got.rot, s));

    // Packs without the rotation run the first kernel alone, then the norm-gate.
    const rec = (try gdn_decode.recur(geo, in, s)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(rec.y);
    defer _ = mlx.mlx_array_free(rec.conv_state);
    defer _ = mlx.mlx_array_free(rec.ssm_state);
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(y_ref, rec.y, s));
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(pre.conv_state, rec.conv_state, s));
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(state_ref, rec.ssm_state, s));
}

pub fn gdnParityRand(rnd: std.Random, shape: []const c_int, scale: f32, dt: mlx.mlx_dtype, s: mlx.mlx_stream) !mlx.mlx_array {
    const b = try attn256RandBf16Scaled(rnd, shape, scale, s);
    defer _ = mlx.mlx_array_free(b);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, b, dt, s));
    return out;
}

test "gdn norm-gate fused: bit-identical to rms_norm + silu(z) * y at decode and prefill widths, z at a folded offset" {
    const s = mlx.gpuStream();
    gdn_prefill_fused_override = true;
    defer gdn_prefill_fused_override = null;
    gdn_decode_fused_override = true;
    defer gdn_decode_fused_override = null;
    var prng = std.Random.DefaultPrng.init(0x9A7E);
    const rnd = prng.random();
    const hv: c_int = 4;
    const dv: c_int = 128;
    const value_dim = hv * dv;
    const w_shape = [_]c_int{dv};
    const norm_w = try attn256RandBf16(rnd, &w_shape, s);
    defer _ = mlx.mlx_array_free(norm_w);
    const eps: f32 = 1e-6;
    const eps_arr = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(eps_arr);

    for ([_]c_int{ 1, 4, 9, 16, 17, 65, 513, 8703 }) |seq| {
        const y_shape = [_]c_int{ 1, seq, hv, dv };
        const y = try attn256RandBf16Scaled(rnd, &y_shape, 8.0, s);
        defer _ = mlx.mlx_array_free(y);
        // z lives at column offset 96 of a wider row.
        const z_off: c_int = 96;
        const width: c_int = z_off + value_dim + 40;
        const raw_shape = [_]c_int{ 1, seq, width };
        const raw = try attn256RandBf16Scaled(rnd, &raw_shape, 12.0, s);
        defer _ = mlx.mlx_array_free(raw);

        if (seq >= 17) {
            gdn_prefill_fused_override = false;
            try testing.expect((try gdnNormGateFused(s, y, raw, z_off, width, norm_w, eps_arr, false, hv, dv, 1, seq)) == null);
            gdn_prefill_fused_override = true;
        }

        // Composed: rms_norm(y, w), then either silu(z) * normed (swish) or
        // normed * sigmoid(z) (sigmoid gate), flattened.
        var normed = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(normed);
        try mlx.check(mlx.mlx_fast_rms_norm(&normed, y, norm_w, eps, s));
        const z_flat = try sliceLastDim(raw, z_off, z_off + value_dim, s);
        defer _ = mlx.mlx_array_free(z_flat);
        var z_heads = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(z_heads);
        try mlx.check(mlx.mlx_reshape(&z_heads, z_flat, &y_shape, 4, s));
        var sig = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sig);
        try mlx.check(mlx.mlx_sigmoid(&sig, z_heads, s));
        for ([_]bool{ true, false }) |swish| {
            const fused = (try gdnNormGateFused(s, y, raw, z_off, width, norm_w, eps_arr, swish, hv, dv, 1, seq)) orelse return error.FusedDeclined;
            defer _ = mlx.mlx_array_free(fused);
            var gated = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(gated);
            if (swish) {
                var act = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(act);
                try mlx.check(mlx.mlx_multiply(&act, z_heads, sig, s));
                try mlx.check(mlx.mlx_multiply(&gated, act, normed, s));
            } else {
                try mlx.check(mlx.mlx_multiply(&gated, normed, sig, s));
            }
            const flat_shape = [_]c_int{ 1, seq, value_dim };
            var ref = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ref);
            try mlx.check(mlx.mlx_reshape(&ref, gated, &flat_shape, 3, s));
            try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(fused, ref, s));
        }
    }
    // Decline: fusions off.
    gdn_decode_fused_override = false;
    const y_shape = [_]c_int{ 1, 1, hv, dv };
    const y = try attn256RandBf16(rnd, &y_shape, s);
    defer _ = mlx.mlx_array_free(y);
    const z_shape = [_]c_int{ 1, 1, value_dim };
    const z = try attn256RandBf16(rnd, &z_shape, s);
    defer _ = mlx.mlx_array_free(z);
    try std.testing.expect((try gdnNormGateFused(s, y, z, 0, value_dim, norm_w, eps_arr, true, hv, dv, 1, 1)) == null);
}

test "Metal-only kernels decline on a stream without them, so the MLX-op path runs" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a);
    try mlx.check(mlx.mlx_ones(&a, &[_]c_int{ 1, 1024 }, 2, .float32, s));
    var signs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(signs);
    try mlx.check(mlx.mlx_ones(&signs, &[_]c_int{1024}, 1, .float32, s));
    const ids = mlx.mlx_array_new_int(0);
    defer _ = mlx.mlx_array_free(ids);
    const eps = mlx.mlx_array_new_float(1e-6);
    defer _ = mlx.mlx_array_free(eps);

    try testing.expect((try moeRouterTopK(s, a, signs, 4, .sigmoid_bias, true, 1.0, .float32, 0, 0, null)) == null);
    try testing.expect((try gatherQmvAct(s, a, a, a, a, ids, 4, 64, .affine, false, false)) == null);
    try testing.expect((try qmv2.qmm(a, a, a, a, 2, 128, false, true, s)) == null);
    try testing.expect((try rht.normRotate(a, null, signs, eps, signs, 1024, false, s)) == null);
    const rotated = try rht.transform(a, signs, 1024, false, s);
    defer _ = mlx.mlx_array_free(rotated);
    try mlx.check(mlx.mlx_array_eval(rotated));
}
