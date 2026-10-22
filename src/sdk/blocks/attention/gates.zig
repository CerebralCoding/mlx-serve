//! sdk/blocks/attention/gates, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns attnOutGate, ATTN_GATE_SOURCE, attn_gate_kernel, attn_gate_engaged, attn_gate_fused_override, attn_gate_fused_env, and 18 more.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../host/log.zig");
const fused_activations = @import("../mlp/fused_activations.zig");
const testing = @import("../../../engines/mlx/kv/cache.zig").testing;
const LayerWeights = @import("../../../engines/mlx/layer_weights.zig").LayerWeights;
const Transformer = @import("../../../engines/mlx/transformer.zig").Transformer;
const testReadF32 = @import("../../testing/parity.zig").testReadF32;
const gdnParityRand = @import("../gdn/decode_fused.zig").gdnParityRand;
const cachedScalarInt = @import("../linear/verify_qmm.zig").cachedScalarInt;
const swigluFusedEnabled = @import("../mlp/fused_activations.zig").swigluFusedEnabled;
const swigluSigTable = @import("../mlp/fused_activations.zig").swigluSigTable;
const ShapeKey = @import("../moe/router_kernel.zig").ShapeKey;
const attn256MaxDiff = @import("attn256.zig").attn256MaxDiff;

/// Sigmoid attention output gate read off the same post-input-norm hidden
/// the QKV projections read. muse: one gate per channel; spark2_5
/// (`attn_gate_headwise`): one gate per head, broadcast over head_dim.
/// Returns a new array.
/// `g_pre` is the gate projection when the fused row matmul already
/// produced it (null-ctx otherwise).
pub fn attnOutGate(self: *const Transformer, normed: mlx.mlx_array, attn_flat: mlx.mlx_array, lw: *const LayerWeights, g_pre: mlx.mlx_array) !mlx.mlx_array {
    const g_raw = if (g_pre.ctx != null) g_pre else try self.qmatmul(normed, lw.ag_w.?, lw.ag_s, lw.ag_b);
    defer if (g_pre.ctx == null) {
        _ = mlx.mlx_array_free(g_raw);
    };
    if (try fusedSigmoidGateMul(self.s, attn_flat, g_raw, if (self.config.attn_gate_headwise) @intCast(self.config.num_attention_heads) else 0)) |y| return y;
    var g_sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g_sig);
    try mlx.check(mlx.mlx_sigmoid(&g_sig, g_raw, self.s));
    var gated = mlx.mlx_array_new();
    if (!self.config.attn_gate_headwise) {
        try mlx.check(mlx.mlx_multiply(&gated, attn_flat, g_sig, self.s));
        return gated;
    }
    const shape = mlx.getShape(attn_flat);
    const heads: c_int = @intCast(self.config.num_attention_heads);
    const per_head = [_]c_int{ shape[0], shape[1], heads, @divExact(shape[2], heads) };
    const g_shape = [_]c_int{ shape[0], shape[1], heads, 1 };
    var a4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a4);
    try mlx.check(mlx.mlx_reshape(&a4, attn_flat, &per_head, 4, self.s));
    var g4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g4);
    try mlx.check(mlx.mlx_reshape(&g4, g_sig, &g_shape, 4, self.s));
    var gated4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gated4);
    try mlx.check(mlx.mlx_multiply(&gated4, a4, g4, self.s));
    try mlx.check(mlx.mlx_reshape(&gated, gated4, shape.ptr, 3, self.s));
    return gated;
}

// ── Fused per-head attention output gate ──
//
// Laguna gates attention by `softplus(g_proj(x))` per head before o_proj. As
// separate ops that is four dispatches a layer (cast to f32, logaddexp, cast
// back, broadcast multiply) over tensors of a few KB — 160 launches per token
// on a 40-layer model, priced by MLX_SERVE_DISPATCH_PROBE at ~1.7 us each.
//
// One kernel does all four. The arithmetic is MLX's, op for op: `LogAddExp`'s
// exact max/min/log1p form from binary_ops.h (including the infinity
// short-circuit), the same f32 -> output-dtype rounding of the gate before the
// multiply, and the same promote-multiply-round for the product — pinned
// bit-identical by the test below.
//
// MEASURED NEUTRAL, so it ships DEFAULT OFF (`MLX_SERVE_ATTN_GATE_FUSED=1`
// opts in). Paired A/B on Laguna XS, 3 pairs, ms/forward: off 13.112 / 13.098 /
// 13.030 against on 13.117 / 13.116 / 13.160. Removing three dispatches a layer
// is worth ~0.2 ms by the MLX_SERVE_DISPATCH_PROBE slope (~1.5 us each) and
// none of it showed up, because THE GATE CHAIN IS NOT ON THE CRITICAL PATH: it
// depends only on the layer input, so the GPU already overlaps it with the
// q/k/v projections and SDPA. Only fusions that shorten the dependency chain
// (router, expert gathers, lm_head) convert dispatch savings into time — price
// a candidate by where it sits in the chain, not by counting its launches.
// Shape is one thread per OUTPUT ELEMENT, not one threadgroup per row. The
// first version staged the per-head gate in threadgroup memory and had a
// single 256-thread group walk all 6144 activations of a row; at decode that
// is one GPU core doing what the elementwise multiply it replaced spread over
// the whole device, and it measured 2.3% SLOWER than the four-dispatch chain
// on three consecutive pairs. Recomputing softplus once per element is ~10 ALU
// ops against a value that is in cache — far cheaper than serializing.
const ATTN_GATE_SOURCE =
    \\uint i = thread_position_in_grid.x;
    \\uint row = thread_position_in_grid.y;
    \\if (i >= uint(NH * HD)) return;
    \\
    \\// softplus = logaddexp(0, g), computed in f32 and rounded to the
    \\// activation dtype exactly where mlx_astype would round it.
    \\float g = float(glogits[(size_t)row * NH + i / HD]);
    \\float mx = metal::max(0.0f, g);
    \\float mn = metal::min(0.0f, g);
    \\float sp;
    \\if (metal::isnan(g)) {
    \\  sp = NAN;
    \\} else if (metal::isinf(mx)) {
    \\  sp = mx;
    \\} else {
    \\  sp = mx + log1p(metal::exp(mn - mx));
    \\}
    \\
    \\size_t base = (size_t)row * (size_t)(NH * HD);
    \\y[base + i] = T(float(attn[base + i]) * float(T(sp)));
;

var attn_gate_kernel: ?mlx.mlx_fast_metal_kernel = null;
var attn_gate_engaged: bool = false;
pub var attn_gate_fused_override: ?bool = null;
var attn_gate_fused_env: ?bool = null;

fn attnGateFusedEnabled() bool {
    if (attn_gate_fused_override) |v| return v;
    if (attn_gate_fused_env) |v| return v;
    const raw = std.c.getenv("MLX_SERVE_ATTN_GATE_FUSED");
    const enabled = raw != null and std.mem.eql(u8, std.mem.sliceTo(raw.?, 0), "1");
    attn_gate_fused_env = enabled;
    return enabled;
}

fn getAttnGateKernel() !mlx.mlx_fast_metal_kernel {
    if (attn_gate_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "attn", "glogits" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "mlxserve_attn_out_gate",
        in_vec,
        out_vec,
        ATTN_GATE_SOURCE,
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    attn_gate_kernel = kernel;
    return kernel;
}

const AttnGateCfgKey = struct { shape: ShapeKey, nh: c_int, dtype: mlx.mlx_dtype };
var attn_gate_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var attn_gate_cfg_key: AttnGateCfgKey = std.mem.zeroes(AttnGateCfgKey);

/// `attn_flat` [B,S,NH*HD] gated by softplus(`g_logits` [B,S,NH]) per head.
/// Returns null (caller keeps the op chain) outside the supported shapes.
pub fn fusedAttnGate(
    s: mlx.mlx_stream,
    attn_flat: mlx.mlx_array,
    g_logits: mlx.mlx_array,
    nh: c_int,
    hd: c_int,
) !?mlx.mlx_array {
    if (!attnGateFusedEnabled()) return null;
    const dt = mlx.mlx_array_dtype(attn_flat);
    if (dt != mlx.mlx_array_dtype(g_logits)) return null;
    if (dt != .bfloat16 and dt != .float16 and dt != .float32) return null;
    const ash = mlx.getShape(attn_flat);
    if (ash.len != 3 or ash[2] != nh * hd) return null;
    const gsh = mlx.getShape(g_logits);
    if (gsh.len != 3 or gsh[2] != nh or gsh[0] != ash[0] or gsh[1] != ash[1]) return null;
    const rows = ash[0] * ash[1];
    const width = nh * hd;
    const tg: c_int = @min(@as(c_int, 256), @max(@as(c_int, 32), @divTrunc(width + 31, 32) * 32));
    if (@rem(width, tg) != 0) return null;
    const key = AttnGateCfgKey{ .shape = ShapeKey.from(ash), .nh = nh, .dtype = dt };
    if (attn_gate_cfg == null or !std.meta.eql(attn_gate_cfg_key, key)) {
        if (attn_gate_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const config = mlx.mlx_fast_metal_kernel_config_new();
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, ash.ptr, 3, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, width, rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, tg, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "NH", nh));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "HD", hd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "TG", tg));
        attn_gate_cfg = config;
        attn_gate_cfg_key = key;
    }

    const inputs_arr = [_]mlx.mlx_array{ attn_flat, g_logits };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    const kernel = try getAttnGateKernel();
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, attn_gate_cfg.?, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outputs_vec, 0));
    if (!attn_gate_engaged) {
        attn_gate_engaged = true;
        log.info("[attn] fused output-gate kernel engaged: heads={d} head_dim={d} tg={d}\n", .{ nh, hd, tg });
    }
    return y;
}

/// `x * sigmoid(g)`: the sigmoid from the per-dtype table (mlx_sigmoid's own
/// values), one multiply. With GROUP > 1 each gate value covers GROUP
/// consecutive x elements (Spark's per-head gate).
const SIGMOID_GATE_MUL_SOURCE =
    \\uint i = thread_position_in_grid.x;
    \\if (i >= uint(N_size)) return;
    \\T sig = sigtab[as_type<ushort>(gate[i / uint(GROUP)])];
    \\y[i] = x[i] * sig;
;

var sig_gate_kernel: ?mlx.mlx_fast_metal_kernel = null;
const SigGateCfgKey = struct { shape: ShapeKey, group: c_int, dtype: mlx.mlx_dtype };
var sig_gate_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var sig_gate_cfg_key: SigGateCfgKey = std.mem.zeroes(SigGateCfgKey);
var sig_gate_engaged: bool = false;

/// `x * sigmoid(g)` in one dispatch; `heads` > 0 means g is [.., heads] and
/// gates x's last dim per head, 0 means g has x's shape. Bit-identical to
/// mlx_sigmoid + mlx_multiply (pinned by the `fused sigmoid gate` test).
/// Null → caller keeps the ops.
// `x * sigmoid(gate)` over two equal-shape 4-D views read through their own
// strides (a transposed attention output, a split q|gate projection), written
// contiguous [B, S, H, D]; sigmoid from MLX's own table, so bit-equal to the pair.
const STRIDED_SIGMOID_GATE_MUL_SOURCE =
    \\uint i = thread_position_in_grid.x;
    \\if (i >= uint(N_size)) return;
    \\const uint D = x_shape[3], H = x_shape[2], S = x_shape[1];
    \\const uint d = i % D, h = (i / D) % H, sq = (i / (D * H)) % S, b = i / (D * H * S);
    \\const T xv = x[b * x_strides[0] + sq * x_strides[1] + h * x_strides[2] + d * x_strides[3]];
    \\const T gv = gate[b * gate_strides[0] + sq * gate_strides[1] + h * gate_strides[2] + d * gate_strides[3]];
    \\y[i] = xv * sigtab[as_type<ushort>(gv)];
;

var strided_sig_gate_kernel: ?mlx.mlx_fast_metal_kernel = null;
var strided_sig_gate_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var strided_sig_gate_key: SigGateCfgKey = std.mem.zeroes(SigGateCfgKey);

pub fn stridedSigmoidGateMul(s: mlx.mlx_stream, x: mlx.mlx_array, g: mlx.mlx_array) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return null;
    const dt = mlx.mlx_array_dtype(x);
    if ((dt != .bfloat16 and dt != .float16) or mlx.mlx_array_dtype(g) != dt) return null;
    const xsh = mlx.getShape(x);
    if (xsh.len != 4 or !std.mem.eql(c_int, xsh, mlx.getShape(g))) return null;
    const n: i64 = @as(i64, xsh[0]) * xsh[1] * xsh[2] * xsh[3];
    if (n <= 0 or n > std.math.maxInt(c_int)) return null;
    const ni: c_int = @intCast(n);
    const tg: c_int = @min(@as(c_int, 256), @divTrunc(ni + 31, 32) * 32);
    const key = SigGateCfgKey{ .shape = ShapeKey.from(xsh), .group = 1, .dtype = dt };
    if (strided_sig_gate_cfg == null or !std.meta.eql(strided_sig_gate_key, key)) {
        if (strided_sig_gate_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const config = mlx.mlx_fast_metal_kernel_config_new();
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, xsh.ptr, xsh.len, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, @divTrunc(ni + tg - 1, tg) * tg, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, tg, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", dt));
        strided_sig_gate_cfg = config;
        strided_sig_gate_key = key;
    }
    if (strided_sig_gate_kernel == null) {
        const input_names = [_][*:0]const u8{ "x", "gate", "sigtab", "N_size" };
        const output_names = [_][*:0]const u8{"y"};
        const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
        defer _ = mlx.mlx_vector_string_free(in_vec);
        const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
        defer _ = mlx.mlx_vector_string_free(out_vec);
        const kh = mlx.mlx_fast_metal_kernel_new("mlxserve_strided_sigmoid_gate_mul", in_vec, out_vec, STRIDED_SIGMOID_GATE_MUL_SOURCE, "", false, false);
        if (kh.ctx == null) return error.MetalKernelCompileFailed;
        strided_sig_gate_kernel = kh;
    }
    const sigtab = try swigluSigTable(s, dt, std.heap.c_allocator);
    const inputs_arr = [_]mlx.mlx_array{ x, g, sigtab, cachedScalarInt(ni) };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, strided_sig_gate_kernel.?, inputs_vec, strided_sig_gate_cfg.?, s));
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outputs_vec, 0));
    return y;
}

fn fusedSigmoidGateMul(s: mlx.mlx_stream, x: mlx.mlx_array, g: mlx.mlx_array, heads: c_int) !?mlx.mlx_array {
    if (!swigluFusedEnabled() or !mlx.streamIsGpu(s)) return null;
    const dt = mlx.mlx_array_dtype(x);
    if ((dt != .bfloat16 and dt != .float16) or mlx.mlx_array_dtype(g) != dt) return null;
    const xsh = mlx.getShape(x);
    const gsh = mlx.getShape(g);
    if (xsh.len == 0 or xsh.len != gsh.len or xsh.len > 5) return null;
    var n: i64 = 1;
    var gn: i64 = 1;
    for (xsh, gsh, 0..) |a, b, i| {
        n *= a;
        gn *= b;
        if (i + 1 < xsh.len and a != b) return null;
    }
    const last_x = xsh[xsh.len - 1];
    const last_g = gsh[gsh.len - 1];
    const group: c_int = if (heads > 0) @divTrunc(last_x, heads) else 1;
    if (group < 1 or last_g * group != last_x) return null;
    if (n <= 0 or n > std.math.maxInt(c_int)) return null;
    const ni: c_int = @intCast(n);
    const tg: c_int = @min(@as(c_int, 256), @max(@as(c_int, 32), @divTrunc(ni + 31, 32) * 32));
    const key = SigGateCfgKey{ .shape = ShapeKey.from(xsh), .group = group, .dtype = dt };
    if (sig_gate_cfg == null or !std.meta.eql(sig_gate_cfg_key, key)) {
        if (sig_gate_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const config = mlx.mlx_fast_metal_kernel_config_new();
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, xsh.ptr, xsh.len, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, @divTrunc(ni + tg - 1, tg) * tg, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, tg, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GROUP", group));
        sig_gate_cfg = config;
        sig_gate_cfg_key = key;
    }
    if (sig_gate_kernel == null) {
        const input_names = [_][*:0]const u8{ "x", "gate", "sigtab", "N_size" };
        const output_names = [_][*:0]const u8{"y"};
        const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
        defer _ = mlx.mlx_vector_string_free(in_vec);
        const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
        defer _ = mlx.mlx_vector_string_free(out_vec);
        const kh = mlx.mlx_fast_metal_kernel_new("mlxserve_sigmoid_gate_mul", in_vec, out_vec, SIGMOID_GATE_MUL_SOURCE, "", true, false);
        if (kh.ctx == null) return error.MetalKernelCompileFailed;
        sig_gate_kernel = kh;
    }
    const sigtab = try swigluSigTable(s, dt, std.heap.c_allocator);
    const inputs_arr = [_]mlx.mlx_array{ x, g, sigtab, cachedScalarInt(ni) };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, sig_gate_kernel.?, inputs_vec, sig_gate_cfg.?, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var y = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&y, outputs_vec, 0));
    if (!sig_gate_engaged) {
        sig_gate_engaged = true;
        log.info("[attn] fused sigmoid gate engaged: n={d} group={d}\n", .{ ni, group });
    }
    return y;
}

// ── tests ──

test "fused attention output gate is bit-identical to the softplus op chain" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    // Same bar as the router: the kernel replaces four dispatches, so it has
    // to replace their arithmetic exactly or it is a different model. Covers
    // the negative/positive/extreme branches of MLX's LogAddExp.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xA11CE00D);
    const rnd = prng.random();
    attn_gate_fused_override = true;
    defer attn_gate_fused_override = null;

    const B: c_int = 1;
    const NH: c_int = 48;
    const HD: c_int = 128;
    for ([_]c_int{ 1, 7 }) |S| {
        const an: usize = @intCast(B * S * NH * HD);
        const gn: usize = @intCast(B * S * NH);

        const abuf = try allocator.alloc(f32, an);
        defer allocator.free(abuf);
        for (abuf) |*v| v.* = (rnd.float(f32) - 0.5) * 3.0;
        const gbuf = try allocator.alloc(f32, gn);
        defer allocator.free(gbuf);
        // Span the branches: large negative (softplus ~ 0), around 0, large
        // positive (softplus ~ x), plus random.
        for (gbuf, 0..) |*v, i| v.* = switch (i % 4) {
            0 => -40.0,
            1 => (rnd.float(f32) - 0.5) * 0.5,
            2 => 30.0,
            else => (rnd.float(f32) - 0.5) * 12.0,
        };

        const ash = [_]c_int{ B, S, NH * HD };
        const a32 = mlx.mlx_array_new_data(abuf.ptr, &ash, 3, .float32);
        defer _ = mlx.mlx_array_free(a32);
        var attn = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn);
        try mlx.check(mlx.mlx_astype(&attn, a32, .bfloat16, s));

        const gsh = [_]c_int{ B, S, NH };
        const g32 = mlx.mlx_array_new_data(gbuf.ptr, &gsh, 3, .float32);
        defer _ = mlx.mlx_array_free(g32);
        var glog = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(glog);
        try mlx.check(mlx.mlx_astype(&glog, g32, .bfloat16, s));

        // Reference: the exact op chain lagunaAttnWith used before the kernel.
        var g_f32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(g_f32);
        try mlx.check(mlx.mlx_astype(&g_f32, glog, .float32, s));
        const zero = mlx.mlx_array_new_float(0.0);
        defer _ = mlx.mlx_array_free(zero);
        var g_sp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(g_sp);
        try mlx.check(mlx.mlx_logaddexp(&g_sp, zero, g_f32, s));
        var g_bf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(g_bf);
        try mlx.check(mlx.mlx_astype(&g_bf, g_sp, .bfloat16, s));
        const g4 = [_]c_int{ B, S, NH, 1 };
        var g_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(g_r);
        try mlx.check(mlx.mlx_reshape(&g_r, g_bf, &g4, 4, s));
        const a4 = [_]c_int{ B, S, NH, HD };
        var attn4 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn4);
        try mlx.check(mlx.mlx_reshape(&attn4, attn, &a4, 4, s));
        var ref4 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ref4);
        try mlx.check(mlx.mlx_multiply(&ref4, attn4, g_r, s));

        const got = (try fusedAttnGate(s, attn, glog, NH, HD)) orelse return error.FusedGateDeclined;
        defer _ = mlx.mlx_array_free(got);

        const ref_host = try allocator.alloc(f32, an);
        defer allocator.free(ref_host);
        const got_host = try allocator.alloc(f32, an);
        defer allocator.free(got_host);
        try testReadF32(ref4, ref_host, s);
        try testReadF32(got, got_host, s);
        try testing.expectEqualSlices(f32, ref_host, got_host);
    }
}

test "strided sigmoid gate is bit-identical to mlx_sigmoid + multiply over a transposed output and a split gate" {
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x6A7E);
    const rnd = prng.random();
    for ([_]c_int{ 1, 16 }) |rows| {
        const heads: c_int = 24;
        const hd: c_int = 256;
        const out_bhsd = try gdnParityRand(rnd, &.{ 1, heads, rows, hd }, 4.0, .bfloat16, s);
        defer _ = mlx.mlx_array_free(out_bhsd);
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_transpose_axes(&x, out_bhsd, &[_]c_int{ 0, 2, 1, 3 }, 4, s));
        const qg = try gdnParityRand(rnd, &.{ 1, rows, heads, 2 * hd }, 12.0, .bfloat16, s);
        defer _ = mlx.mlx_array_free(qg);
        var g = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(g);
        try mlx.check(mlx.mlx_slice(&g, qg, &[_]c_int{ 0, 0, 0, hd }, 4, &[_]c_int{ 1, rows, heads, 2 * hd }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
        var sig = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sig);
        try mlx.check(mlx.mlx_sigmoid(&sig, g, s));
        var ref = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ref);
        try mlx.check(mlx.mlx_multiply(&ref, x, sig, s));
        const got = (try stridedSigmoidGateMul(s, x, g)) orelse return error.FusedDeclined;
        defer _ = mlx.mlx_array_free(got);
        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(got, ref, s));
        // A metal_kernel helper declines off the GPU stream.
        const cpu = mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(cpu);
        try std.testing.expect((try stridedSigmoidGateMul(cpu, x, g)) == null);
    }
}

test "fused sigmoid gate is bit-identical to mlx_sigmoid + broadcast multiply, per-head and elementwise" {
    const s = mlx.gpuStream();
    fused_activations.swiglu_fused_override = true;
    defer fused_activations.swiglu_fused_override = null;
    var prng = std.Random.DefaultPrng.init(0x51663);
    const rnd = prng.random();
    const heads: c_int = 16;
    const hd: c_int = 256;
    for ([_]bool{ true, false }) |headwise| {
        const x_shape = [_]c_int{ 1, 1, heads * hd };
        const x = try gdnParityRand(rnd, &x_shape, 4.0, .bfloat16, s);
        defer _ = mlx.mlx_array_free(x);
        const g_shape = [_]c_int{ 1, 1, if (headwise) heads else heads * hd };
        const g = try gdnParityRand(rnd, &g_shape, 12.0, .bfloat16, s);
        defer _ = mlx.mlx_array_free(g);
        var sig = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sig);
        try mlx.check(mlx.mlx_sigmoid(&sig, g, s));
        var ref = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ref);
        if (headwise) {
            const x4 = [_]c_int{ 1, 1, heads, hd };
            const g4 = [_]c_int{ 1, 1, heads, 1 };
            var xr = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(xr);
            try mlx.check(mlx.mlx_reshape(&xr, x, &x4, 4, s));
            var gr = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(gr);
            try mlx.check(mlx.mlx_reshape(&gr, sig, &g4, 4, s));
            var prod = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(prod);
            try mlx.check(mlx.mlx_multiply(&prod, xr, gr, s));
            try mlx.check(mlx.mlx_reshape(&ref, prod, &x_shape, 3, s));
        } else {
            try mlx.check(mlx.mlx_multiply(&ref, x, sig, s));
        }
        const got = (try fusedSigmoidGateMul(s, x, g, if (headwise) heads else 0)) orelse return error.FusedDeclined;
        defer _ = mlx.mlx_array_free(got);
        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(got, ref, s));
    }
}
