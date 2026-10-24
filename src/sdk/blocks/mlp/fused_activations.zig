//! sdk/blocks/mlp/fused_activations, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns SWIGLU_SOURCE, SWIGLU_CLAMP_SOURCE, swiglu_sig_tables, sigmoidTableFor, swigluSigTable, swiglu_engaged, and 27 more.

const std = @import("std");
const env = @import("../../../env.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../host/log.zig");
const testing = std.testing;
const bf16Scalar = @import("../../../mlx/array.zig").bf16Scalar;
const scalarOf = @import("../../../mlx/array.zig").scalarOf;
const testReadF32 = @import("../../testing/parity.zig").testReadF32;
const attn256MaxDiff = @import("../attention/attn256.zig").attn256MaxDiff;
const gdnParityRand = @import("../gdn/decode_fused.zig").gdnParityRand;
const sliceQkvPart = @import("../linear/fused_rows.zig").sliceQkvPart;
const qmatmulBits = @import("../linear/qmatmul.zig").qmatmulBits;
const cachedScalarInt = @import("../linear/verify_qmm.zig").cachedScalarInt;
const ShapeKey = @import("../moe/router_kernel.zig").ShapeKey;
const clampedSwiGLU = @import("activations.zig").clampedSwiGLU;
const gptOssSwiGLU = @import("activations.zig").gptOssSwiGLU;

// ── Fused SwiGLU: silu(gate) * up in one dispatch ──
//
// `computeGeglu` only had a fused path for `gelu_approx`, so every SILU model —
// Laguna, Qwen, Llama, Mistral, i.e. most of the matrix — ran sigmoid +
// multiply + multiply as three dispatches, and a MoE layer pays it TWICE
// (routed experts + shared expert). It sits on the expert chain's critical
// path, so unlike the attention gate the launches are not overlapped: paired
// A/B on Laguna XS measured 13.12 -> 12.81 ms/forward, 3/3 pairs.
//
// Extending mlx_compile to the silu arm gets the same 2.4% but is NOT
// output-preserving — a same-binary greedy A/B diverged at token ~55, so the
// compiled elementwise fusion does not reproduce the chain's intermediate
// rounding. This kernel does: MLX's `Sigmoid` (the two-sided form) evaluated
// and ROUNDED TO T, then two `Multiply`s each promoting to float and rounding
// back, exactly as the three op kernels do.
// The sigmoid is a TABLE LOOKUP, not a recomputation, and that is the whole
// trick. Writing MLX's `Sigmoid` source verbatim is not enough to reproduce it:
// MLX's own kernels live in a metallib built with a different Metal math mode
// than the one `metal_kernel` JITs custom sources with (CompileOptions
// defaults to MathMode::Safe and mlx-c exposes no way to ask for another), so
// `1 / (1 + exp(|x|))` lands a rounding apart. An exhaustive sweep found it on
// exactly ONE of the 65536 bf16 patterns (-6.84375) — which random sampling
// missed and a 260-token greedy run hit within ~55 tokens, because a
// 8192-wide MLP draws 8192 values from that domain every layer.
//
// A 16-bit activation has only 65536 possible inputs, so `mlx_sigmoid` is
// evaluated over ALL of them once at first use and the kernel indexes the
// result by the input's bit pattern. Exact by construction, no transcendental
// in the hot loop, 128 KB resident. The two multiplies need no such care: a
// bf16/f16 product is exact in float, so one rounding is one rounding on any
// compiler.
pub const SWIGLU_SOURCE = kernelSource("kernels/mlp/swiglu.metal");

// SWIGLU_SOURCE behind the clamps of a `swiglu_limit` (GLM-5-Next): mlx Minimum on the gate,
// Clip on the up branch (exact in the tensor dtype), then the same chain.
pub const SWIGLU_CLAMP_SOURCE = kernelSource("kernels/mlp/swiglu_clamp.metal");

/// `mlx_sigmoid` tabulated over every 16-bit float pattern, indexed [CPU, GPU][bf16, f16]: the devices round apart.
var swiglu_sig_tables: [2][2]?mlx.mlx_array = @splat(@splat(null));

/// MLX's sigmoid of every 16-bit pattern of `dt` (bf16 or f16), for kernels
/// that must reproduce the op's rounding bit for bit.
pub fn sigmoidTableFor(s: mlx.mlx_stream, dt: mlx.mlx_dtype) !mlx.mlx_array {
    return swigluSigTable(s, dt, std.heap.c_allocator);
}

pub fn swigluSigTable(s: mlx.mlx_stream, dt: mlx.mlx_dtype, allocator: std.mem.Allocator) !mlx.mlx_array {
    const slot: usize = switch (dt) {
        .bfloat16 => 0,
        .float16 => 1,
        else => return error.UnsupportedSwigluDtype,
    };
    const cached = &swiglu_sig_tables[@intFromBool(mlx.streamIsGpu(s))][slot];
    if (cached.*) |t| return t;

    const vals = try all16BitPatterns(s, dt, allocator);
    defer _ = mlx.mlx_array_free(vals);
    var table = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(table);
    try mlx.check(mlx.mlx_sigmoid(&table, vals, s));
    // Materialize now: the table outlives every graph that reads it.
    try mlx.check(mlx.mlx_array_eval(table));
    cached.* = table;
    return table;
}

var swiglu_engaged: bool = false;
pub var swiglu_fused_override: ?bool = null;

pub fn swigluFusedEnabled() bool {
    return swiglu_fused_override orelse true;
}

/// `table[gate] * up`: an activation that is unary in x, tabulated over every
/// 16-bit pattern by running MLX's own op chain once, then one multiply.
pub const TABLE_MUL_SOURCE = kernelSource("kernels/mlp/table_mul.metal");

/// `x * c -> erf -> 1 + -> x * -> * half`, five ops in the activation dtype.
pub fn geluErfChain(x: mlx.mlx_array, one: mlx.mlx_array, half: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const inv_sqrt2 = try scalarOf(0.70710678118654752, mlx.mlx_array_dtype(x), s);
    defer _ = mlx.mlx_array_free(inv_sqrt2);
    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    try mlx.check(mlx.mlx_multiply(&scaled, x, inv_sqrt2, s));
    var erf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(erf);
    try mlx.check(mlx.mlx_erf(&erf, scaled, s));
    var one_plus = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(one_plus);
    try mlx.check(mlx.mlx_add(&one_plus, one, erf, s));
    var x_times = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x_times);
    try mlx.check(mlx.mlx_multiply(&x_times, x, one_plus, s));
    var result = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&result, x_times, half, s));
    return result;
}

/// Every 16-bit pattern of `dt` as an array of that dtype (the domain a
/// per-dtype op table is evaluated over).
pub fn all16BitPatterns(s: mlx.mlx_stream, dt: mlx.mlx_dtype, allocator: std.mem.Allocator) !mlx.mlx_array {
    const N: usize = 1 << 16;
    const buf = try allocator.alloc(f32, N);
    defer allocator.free(buf);
    // Widened to the f32 that converts back EXACTLY, so astype is a relabel.
    for (buf, 0..) |*v, i| v.* = switch (dt) {
        .bfloat16 => @bitCast(@as(u32, @intCast(i)) << 16),
        else => @floatCast(@as(f16, @bitCast(@as(u16, @intCast(i))))),
    };
    const sh = [_]c_int{@intCast(N)};
    const v32 = mlx.mlx_array_new_data(buf.ptr, &sh, 1, .float32);
    defer _ = mlx.mlx_array_free(v32);
    var vals = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&vals, v32, dt, s));
    return vals;
}

var gelu_erf_tables: [2]?mlx.mlx_array = @splat(null);

fn geluErfTable(s: mlx.mlx_stream, dt: mlx.mlx_dtype, one: mlx.mlx_array, half: mlx.mlx_array) !mlx.mlx_array {
    const slot: usize = if (dt == .bfloat16) 0 else 1;
    if (gelu_erf_tables[slot]) |t| return t;
    const vals = try all16BitPatterns(s, dt, std.heap.c_allocator);
    defer _ = mlx.mlx_array_free(vals);
    const table = try geluErfChain(vals, one, half, s);
    errdefer _ = mlx.mlx_array_free(table);
    if (mlx.mlx_array_dtype(table) != dt) return error.GeluTableDtype;
    try mlx.check(mlx.mlx_array_eval(table));
    gelu_erf_tables[slot] = table;
    return table;
}

const GateMulKind = enum(usize) {
    silu,
    gelu_erf,
    gelu_tanh,
    silu_clamp,
    fn name(self: GateMulKind) [*:0]const u8 {
        return switch (self) {
            .silu => "mlxserve_swiglu",
            .gelu_erf => "mlxserve_gelu_erf_mul",
            .gelu_tanh => "mlxserve_gelu_tanh_mul",
            .silu_clamp => "mlxserve_swiglu_clamp",
        };
    }
};
const gate_mul_kinds = @typeInfo(GateMulKind).@"enum".field_names.len;
var gate_mul_kernels: [gate_mul_kinds]?mlx.mlx_fast_metal_kernel = @splat(null);
const GateMulCfgKey = struct { shape: ShapeKey, dtype: mlx.mlx_dtype };
var gate_mul_cfg: [gate_mul_kinds]?mlx.mlx_fast_metal_kernel_config = @splat(null);
var gate_mul_cfg_key: [gate_mul_kinds]GateMulCfgKey = @splat(std.mem.zeroes(GateMulCfgKey));
var gate_mul_engaged: [gate_mul_kinds]bool = @splat(false);

fn getGateMulKernel(kind: GateMulKind) !mlx.mlx_fast_metal_kernel {
    const k: usize = @backingInt(kind);
    if (gate_mul_kernels[k]) |kr| return kr;
    const input_names = [_][*:0]const u8{ "gate", "up", "sigtab", "N_size", "lim" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, if (kind == .silu_clamp) 5 else 4);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        kind.name(),
        in_vec,
        out_vec,
        switch (kind) {
            .silu => SWIGLU_SOURCE,
            .silu_clamp => SWIGLU_CLAMP_SOURCE,
            else => TABLE_MUL_SOURCE,
        },
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    gate_mul_kernels[k] = kernel;
    return kernel;
}

/// `act(gate) * up`, elementwise, one dispatch, the activation read from a
/// per-dtype table of MLX's own results; `lim` (0-dim, the tensor dtype) for `.silu_clamp`.
/// Null → caller keeps the chain.
pub fn tableGateMul(s: mlx.mlx_stream, kind: GateMulKind, gate: mlx.mlx_array, up: mlx.mlx_array, table: mlx.mlx_array, lim: ?mlx.mlx_array) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return null;
    const dt = mlx.mlx_array_dtype(gate);
    if (dt != mlx.mlx_array_dtype(up)) return null;
    // FLOAT32 IS DECLINED ON PURPOSE. MLX ships its own kernels in a metallib
    // built with a different Metal math mode than the one `metal_kernel` JITs
    // custom sources with (CompileOptions defaults to MathMode::Safe, and
    // mlx-c exposes no way to ask for another), so an f32 `1 / (1 + exp(..))`
    // lands 1-2 ulps away from the stock sigmoid's. That is invisible in bf16
    // and f16 — both round the result far below the difference, and the test
    // pins them — but it is real in f32, and a fused path that is exact for
    // some dtypes and not others is worse than one that declines.
    if (dt != .bfloat16 and dt != .float16) return null;
    const gsh = mlx.getShape(gate);
    const ush = mlx.getShape(up);
    // No broadcasting: the gated MLP always pairs equal shapes, and a mismatch
    // here would silently read out of bounds rather than fail.
    if (gsh.len != ush.len or gsh.len == 0 or gsh.len > 5) return null;
    var n: i64 = 1;
    for (gsh, ush) |a, b| {
        if (a != b) return null;
        n *= a;
    }
    if (n <= 0 or n > std.math.maxInt(c_int)) return null;
    const ni: c_int = @intCast(n);
    const k: usize = @backingInt(kind);

    const tg: c_int = @min(@as(c_int, 256), @max(@as(c_int, 32), @divTrunc(ni + 31, 32) * 32));
    const key = GateMulCfgKey{ .shape = ShapeKey.from(gsh), .dtype = dt };
    if (gate_mul_cfg[k] == null or !std.meta.eql(gate_mul_cfg_key[k], key)) {
        if (gate_mul_cfg[k]) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const config = mlx.mlx_fast_metal_kernel_config_new();
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, gsh.ptr, gsh.len, dt));
        // Grid is rounded UP to the threadgroup so the launch is uniform; the
        // kernel's own bound check drops the tail threads.
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, @divTrunc(ni + tg - 1, tg) * tg, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, tg, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", dt));
        gate_mul_cfg[k] = config;
        gate_mul_cfg_key[k] = key;
    }

    const inputs_arr = [_]mlx.mlx_array{ gate, up, table, cachedScalarInt(ni), lim orelse table };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, if (kind == .silu_clamp) 5 else 4);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    const kernel = try getGateMulKernel(kind);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, gate_mul_cfg[k].?, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outputs_vec, 0));
    if (!gate_mul_engaged[k]) {
        gate_mul_engaged[k] = true;
        log.info("[mlp] fused {s} kernel engaged: n={d} tg={d}\n", .{ @tagName(kind), ni, tg });
    }
    return y;
}

/// `silu(gate) * up`, elementwise, one dispatch. Null → caller keeps the chain.
/// Exact by construction (bit-level sigmoid table), so callers outside this
/// file (tts.zig's talker/code-predictor MLPs) reuse it rather than growing a
/// second kernel.
// SWIGLU_SOURCE over a joined gate|up projection row (gate the first F
// columns): the two halves are read in place instead of copied out.
pub const SWIGLU_JOINED_SOURCE = kernelSource("kernels/mlp/swiglu_joined.metal");
var swiglu_joined_kernel: ?mlx.mlx_fast_metal_kernel = null;
var swiglu_joined_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var swiglu_joined_key: GateMulCfgKey = std.mem.zeroes(GateMulCfgKey);

/// `fusedSwiGLU(gate, up)` for `both` = [..., gate | up] (equal halves), bit-equal to it.
pub fn swigluJoined(s: mlx.mlx_stream, both: mlx.mlx_array) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return null;
    if (!swigluFusedEnabled()) return null;
    const dt = mlx.mlx_array_dtype(both);
    if (dt != .bfloat16 and dt != .float16) return null;
    const sh = mlx.getShape(both);
    if (sh.len == 0 or sh.len > 5 or @rem(sh[sh.len - 1], 2) != 0) return null;
    var out_shape: [5]c_int = undefined;
    @memcpy(out_shape[0..sh.len], sh);
    const f = @divExact(sh[sh.len - 1], 2);
    out_shape[sh.len - 1] = f;
    var n: i64 = f;
    for (sh[0 .. sh.len - 1]) |d| n *= d;
    if (n <= 0 or n > std.math.maxInt(c_int)) return null;
    const ni: c_int = @intCast(n);
    const key = GateMulCfgKey{ .shape = ShapeKey.from(sh), .dtype = dt };
    if (swiglu_joined_cfg == null or !std.meta.eql(swiglu_joined_key, key)) {
        if (swiglu_joined_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const config = mlx.mlx_fast_metal_kernel_config_new();
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &out_shape, sh.len, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, @divTrunc(ni + 255, 256) * 256, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "F", f));
        swiglu_joined_cfg = config;
        swiglu_joined_key = key;
    }
    if (swiglu_joined_kernel == null) {
        const in_names = [_][*:0]const u8{ "both", "sigtab", "N_size" };
        const out_names = [_][*:0]const u8{"y"};
        const in_vec = mlx.mlx_vector_string_new_data(&in_names, in_names.len);
        defer _ = mlx.mlx_vector_string_free(in_vec);
        const out_vec = mlx.mlx_vector_string_new_data(&out_names, out_names.len);
        defer _ = mlx.mlx_vector_string_free(out_vec);
        const k = mlx.mlx_fast_metal_kernel_new("mlxserve_swiglu_joined", in_vec, out_vec, SWIGLU_JOINED_SOURCE, "", true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        swiglu_joined_kernel = k;
    }
    const sigtab = try swigluSigTable(s, dt, std.heap.c_allocator);
    const ins = [_]mlx.mlx_array{ both, sigtab, cachedScalarInt(ni) };
    const v = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&o, swiglu_joined_kernel.?, v, swiglu_joined_cfg.?, s));
    var y = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&y, o, 0));
    return y;
}

pub fn fusedSwiGLU(s: mlx.mlx_stream, gate: mlx.mlx_array, up: mlx.mlx_array) !?mlx.mlx_array {
    if (!swigluFusedEnabled()) return null;
    const dt = mlx.mlx_array_dtype(gate);
    if (dt != .bfloat16 and dt != .float16) return null;
    const sigtab = try swigluSigTable(s, dt, std.heap.c_allocator);
    return tableGateMul(s, .silu, gate, up, sigtab, null);
}

/// `silu(min(gate, limit)) * clip(up, ±limit)` in one dispatch, bit for bit `clampedSwiGLU`.
pub fn fusedClampedSwiGLU(s: mlx.mlx_stream, gate: mlx.mlx_array, up: mlx.mlx_array, limit: f32) !?mlx.mlx_array {
    if (!swigluFusedEnabled()) return null;
    const dt = mlx.mlx_array_dtype(gate);
    if (dt != .bfloat16 and dt != .float16) return null;
    const sigtab = try swigluSigTable(s, dt, std.heap.c_allocator);
    const lim = try scalarOf(limit, dt, s);
    defer _ = mlx.mlx_array_free(lim);
    return tableGateMul(s, .silu_clamp, gate, up, sigtab, lim);
}

/// `gelu_tanh(x)` (whatever `Transformer.gelu` runs: compiled or the op
/// chain) tabulated per dtype. Null when the chain does not return `dt`.
pub var gelu_tanh_tables: [2]?mlx.mlx_array = @splat(null);
pub var gelu_tanh_declined: [2]bool = @splat(false);

/// `gelu_erf(gate) * up` in one dispatch, the activation tabulated from the
/// op chain itself (`geluErfTable`). Null → caller keeps the chain.
pub fn fusedGeluErfMul(s: mlx.mlx_stream, gate: mlx.mlx_array, up: mlx.mlx_array, one: mlx.mlx_array, half: mlx.mlx_array) !?mlx.mlx_array {
    if (!swigluFusedEnabled()) return null;
    const dt = mlx.mlx_array_dtype(gate);
    if (dt != .bfloat16 and dt != .float16) return null;
    if (mlx.mlx_array_dtype(one) != dt or mlx.mlx_array_dtype(half) != dt) return null;
    const table = try geluErfTable(s, dt, one, half);
    return tableGateMul(s, .gelu_erf, gate, up, table, null);
}

// ── tests ──

const Transformer = @import("../../../engines/mlx/transformer.zig").Transformer;

const fusedGeluPleProj = @import("../../../plugins/gemma4/gelu_ple.zig").fusedGeluPleProj;


test "fused SwiGLU is bit-identical to the sigmoid+multiply+multiply chain" {
    // The reason this kernel exists instead of extending mlx_compile to the
    // silu arm: the compiled fusion is NOT bit-identical (greedy diverged at
    // token ~55 on Laguna XS). Assert what the compiled path could not.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x51117001);
    const rnd = prng.random();
    swiglu_fused_override = true;
    defer swiglu_fused_override = null;

    // Sizes that are and are not multiples of the threadgroup, so the grid
    // round-up and the kernel's bound check are both exercised.
    for ([_]c_int{ 8, 512, 4096, 4097 }) |n| {
        for ([_]mlx.mlx_dtype{ .bfloat16, .float16 }) |dt| {
            const cnt: usize = @intCast(n);
            const gbuf = try allocator.alloc(f32, cnt);
            defer allocator.free(gbuf);
            const ubuf = try allocator.alloc(f32, cnt);
            defer allocator.free(ubuf);
            for (gbuf, 0..) |*v, i| v.* = switch (i % 5) {
                0 => -30.0,
                1 => 25.0,
                2 => 0.0,
                else => (rnd.float(f32) - 0.5) * 10.0,
            };
            for (ubuf) |*v| v.* = (rnd.float(f32) - 0.5) * 4.0;

            const sh = [_]c_int{n};
            const g32 = mlx.mlx_array_new_data(gbuf.ptr, &sh, 1, .float32);
            defer _ = mlx.mlx_array_free(g32);
            const u32a = mlx.mlx_array_new_data(ubuf.ptr, &sh, 1, .float32);
            defer _ = mlx.mlx_array_free(u32a);
            var gate = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(gate);
            try mlx.check(mlx.mlx_astype(&gate, g32, dt, s));
            var up = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(up);
            try mlx.check(mlx.mlx_astype(&up, u32a, dt, s));

            // Reference: exactly the ops computeGeglu's fallback issues.
            var sig = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sig);
            try mlx.check(mlx.mlx_sigmoid(&sig, gate, s));
            var act = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(act);
            try mlx.check(mlx.mlx_multiply(&act, gate, sig, s));
            var ref = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ref);
            try mlx.check(mlx.mlx_multiply(&ref, act, up, s));

            const got = (try fusedSwiGLU(s, gate, up)) orelse return error.FusedSwigluDeclined;
            defer _ = mlx.mlx_array_free(got);

            const ref_host = try allocator.alloc(f32, cnt);
            defer allocator.free(ref_host);
            const got_host = try allocator.alloc(f32, cnt);
            defer allocator.free(got_host);
            try testReadF32(ref, ref_host, s);
            try testReadF32(got, got_host, s);
            try testing.expectEqualSlices(f32, ref_host, got_host);
        }
    }
}

test "fused SwiGLU config cache keys on the SHAPE, not the element count (Laguna restore crash)" {
    // Live 2026-07-29: a 16-token hot-cache-restore suffix prefill on Laguna
    // XS killed the server with "[matmul] (1,16,512) vs (8192,2048)". A MoE
    // layer's SwiGLU at 16 tokens is (1,16,512) = 8192 elements — the SAME
    // count, rank and dtype as a dense mlp_only layer at decode (1,1,8192).
    // The config cache was keyed on {n, tg, ndim, dtype}, so the second call
    // reused a config whose baked OUTPUT SHAPE was the first caller's, and
    // the dense down-proj matmul died on the mislabeled activation. The key
    // must be the full shape; equal products are not equal shapes.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    swiglu_fused_override = true;
    defer swiglu_fused_override = null;

    const shapes = [2][3]c_int{ .{ 1, 16, 512 }, .{ 1, 1, 8192 } };
    for (shapes) |sh| {
        const cnt: usize = 8192;
        const buf = try allocator.alloc(f32, cnt);
        defer allocator.free(buf);
        for (buf, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 7)) - 3.0;
        const a32 = mlx.mlx_array_new_data(buf.ptr, &sh, 3, .float32);
        defer _ = mlx.mlx_array_free(a32);
        var gate = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gate);
        try mlx.check(mlx.mlx_astype(&gate, a32, .bfloat16, s));

        const got = (try fusedSwiGLU(s, gate, gate)) orelse return error.FusedSwigluDeclined;
        defer _ = mlx.mlx_array_free(got);
        try mlx.check(mlx.mlx_array_eval(got));
        const osh = mlx.getShape(got);
        try testing.expectEqualSlices(c_int, &sh, osh);
    }
}

test "fused GELU-erf gate is bit-identical to the op chain over the whole 16-bit domain" {
    // Bar: every bf16/f16 gate pattern times a handful of up values, byte for
    // byte against geluErfChain + mlx_multiply.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    swiglu_fused_override = true;
    defer swiglu_fused_override = null;
    for ([_]mlx.mlx_dtype{ .bfloat16, .float16 }) |dt| {
        const one = try scalarOf(1.0, dt, s);
        defer _ = mlx.mlx_array_free(one);
        const half = try scalarOf(0.5, dt, s);
        defer _ = mlx.mlx_array_free(half);
        const gate = try all16BitPatterns(s, dt, allocator);
        defer _ = mlx.mlx_array_free(gate);
        for ([_]f32{ 1.0, -0.375, 3.140625, 1e-3 }) |uv| {
            const uscalar = try scalarOf(uv, dt, s);
            defer _ = mlx.mlx_array_free(uscalar);
            var up = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(up);
            try mlx.check(mlx.mlx_multiply(&up, uscalar, one, s));
            var up_full = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(up_full);
            try mlx.check(mlx.mlx_broadcast_to(&up_full, up, mlx.getShape(gate).ptr, 1, s));
            var up_c = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(up_c);
            try mlx.check(mlx.mlx_contiguous(&up_c, up_full, false, s));
            const act = try geluErfChain(gate, one, half, s);
            defer _ = mlx.mlx_array_free(act);
            var ref = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ref);
            try mlx.check(mlx.mlx_multiply(&ref, act, up_c, s));
            const got = (try fusedGeluErfMul(s, gate, up_c, one, half)) orelse return error.FusedGeluDeclined;
            defer _ = mlx.mlx_array_free(got);
            var eq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(eq);
            try mlx.check(mlx.mlx_array_equal(&eq, ref, got, true, s));
            try mlx.check(mlx.mlx_array_eval(eq));
            var eq_v: bool = false;
            try mlx.check(mlx.mlx_array_item_bool(&eq_v, eq));
            try testing.expect(eq_v);
        }
    }
}

test "fused clamped swiglu is bit-identical to the clampedSwiGLU chain" {
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x61A5);
    const rnd = prng.random();
    // Spread past +-10 so both clamps bite.
    const gate = try gdnParityRand(rnd, &.{ 3, 7, 2048 }, 16.0, .bfloat16, s);
    defer _ = mlx.mlx_array_free(gate);
    const up = try gdnParityRand(rnd, &.{ 3, 7, 2048 }, 16.0, .bfloat16, s);
    defer _ = mlx.mlx_array_free(up);
    const want = try clampedSwiGLU(s, gate, up, 10.0);
    defer _ = mlx.mlx_array_free(want);
    const got = (try fusedClampedSwiGLU(s, gate, up, 10.0)) orelse return error.SkipZigTest;
    defer _ = mlx.mlx_array_free(got);
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_array_equal(&eq, want, got, false, s));
    try mlx.check(mlx.mlx_array_eval(eq));
    var eq_v: bool = false;
    try mlx.check(mlx.mlx_array_item_bool(&eq_v, eq));
    try testing.expect(eq_v);
}

test "joined swiglu is bit-identical to fusedSwiGLU over the two halves" {
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x5316);
    const rnd = prng.random();
    for ([_]c_int{ 1, 16, 33 }) |rows| {
        const both = try gdnParityRand(rnd, &.{ 1, rows, 2 * 1088 }, 6.0, .bfloat16, s);
        defer _ = mlx.mlx_array_free(both);
        var gate = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gate);
        var up = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(up);
        try sliceQkvPart(&gate, both, 0, 1088, s);
        try sliceQkvPart(&up, both, 1088, 1088, s);
        const ref = (try fusedSwiGLU(s, gate, up)) orelse return error.Declined;
        defer _ = mlx.mlx_array_free(ref);
        const got = (try swigluJoined(s, both)) orelse return error.Declined;
        defer _ = mlx.mlx_array_free(got);
        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(got, ref, s));
    }
}

test "fused gelu-ple projection is bit-identical to gelu table -> multiply -> quantized_matmul" {
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    swiglu_fused_override = true;
    defer swiglu_fused_override = null;
    var prng = std.Random.DefaultPrng.init(0x9E1E);
    const rnd = prng.random();
    var xfm: Transformer = undefined;
    xfm.s = s;
    xfm.compiled_gelu = null;
    xfm.three = bf16Scalar(3.0, s);
    defer _ = mlx.mlx_array_free(xfm.three.?);
    xfm.gelu_inner = bf16Scalar(0.044715, s);
    defer _ = mlx.mlx_array_free(xfm.gelu_inner.?);
    xfm.gelu_coeff = bf16Scalar(0.7978845608028654, s);
    defer _ = mlx.mlx_array_free(xfm.gelu_coeff.?);
    xfm.one = bf16Scalar(1.0, s);
    defer _ = mlx.mlx_array_free(xfm.one);
    xfm.half = bf16Scalar(0.5, s);
    defer _ = mlx.mlx_array_free(xfm.half);
    const table = (try xfm.geluTanhTable(.bfloat16)) orelse return error.NoTable;
    const HID: c_int = 2560;
    const PD: c_int = 256;
    for ([_]f32{ 1.0, 12.0, 60.0 }) |scale| {
        const wb = try gdnParityRand(rnd, &.{ HID, PD }, 1.0, .bfloat16, s);
        defer _ = mlx.mlx_array_free(wb);
        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(&triple, wb, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{}, s));
        var pw = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pw);
        var ps = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ps);
        var pb = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pb);
        try mlx.check(mlx.mlx_vector_array_get(&pw, triple, 0));
        try mlx.check(mlx.mlx_vector_array_get(&ps, triple, 1));
        try mlx.check(mlx.mlx_vector_array_get(&pb, triple, 2));
        const g = try gdnParityRand(rnd, &.{ 1, 1, PD }, scale, .bfloat16, s);
        defer _ = mlx.mlx_array_free(g);
        const ple = try gdnParityRand(rnd, &.{ 1, 1, PD }, 4.0, .bfloat16, s);
        defer _ = mlx.mlx_array_free(ple);
        const act = try xfm.gelu(g);
        defer _ = mlx.mlx_array_free(act);
        var gated = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gated);
        try mlx.check(mlx.mlx_multiply(&gated, act, ple, s));
        const ref = try qmatmulBits(gated, pw, ps, pb, 4, 64, .affine, s);
        defer _ = mlx.mlx_array_free(ref);
        const got = (try fusedGeluPleProj(s, g, ple, table, pw, ps, pb, .{ .bits = 4, .group_size = 64, .mode = .affine })) orelse return error.FusedDeclined;
        defer _ = mlx.mlx_array_free(got);
        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(got, ref, s));
    }
    _ = allocator;
}

test "fused GELU-tanh gate is bit-identical to the current GELU path times up" {
    // The table is built from `Transformer.gelu` itself, so the bar is the
    // multiply: table[x] * up over every bf16 x and a few up values.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    swiglu_fused_override = true;
    defer swiglu_fused_override = null;
    var xfm: Transformer = undefined;
    xfm.s = s;
    xfm.compiled_gelu = null;
    xfm.three = bf16Scalar(3.0, s);
    defer _ = mlx.mlx_array_free(xfm.three.?);
    xfm.gelu_inner = bf16Scalar(0.044715, s);
    defer _ = mlx.mlx_array_free(xfm.gelu_inner.?);
    xfm.gelu_coeff = bf16Scalar(0.7978845608028654, s);
    defer _ = mlx.mlx_array_free(xfm.gelu_coeff.?);
    xfm.one = bf16Scalar(1.0, s);
    defer _ = mlx.mlx_array_free(xfm.one);
    xfm.half = bf16Scalar(0.5, s);
    defer _ = mlx.mlx_array_free(xfm.half);
    const gate = try all16BitPatterns(s, .bfloat16, allocator);
    defer _ = mlx.mlx_array_free(gate);
    for ([_]f32{ 1.0, -0.375, 3.140625 }) |uv| {
        const uscalar = try scalarOf(uv, .bfloat16, s);
        defer _ = mlx.mlx_array_free(uscalar);
        var up_full = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(up_full);
        try mlx.check(mlx.mlx_broadcast_to(&up_full, uscalar, mlx.getShape(gate).ptr, 1, s));
        var up = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(up);
        try mlx.check(mlx.mlx_contiguous(&up, up_full, false, s));
        const act = try xfm.gelu(gate);
        defer _ = mlx.mlx_array_free(act);
        var ref = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ref);
        try mlx.check(mlx.mlx_multiply(&ref, act, up, s));
        const got = (try xfm.fusedGeluTanhMul(gate, up)) orelse return error.FusedGeluDeclined;
        defer _ = mlx.mlx_array_free(got);
        var eq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(eq);
        try mlx.check(mlx.mlx_array_equal(&eq, ref, got, true, s));
        try mlx.check(mlx.mlx_array_eval(eq));
        var eq_v: bool = false;
        try mlx.check(mlx.mlx_array_item_bool(&eq_v, eq));
        try testing.expect(eq_v);
    }
}

test "fused SwiGLU matches the chain on EVERY finite bf16 gate value" {
    // Random sampling said bit-identical while the live model diverged at
    // token ~55, so the sigmoid is swept exhaustively instead: all 65536 bf16
    // bit patterns, which is the whole domain of the activation.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    swiglu_fused_override = true;
    defer swiglu_fused_override = null;

    const N: usize = 1 << 16;
    const sh = [_]c_int{@intCast(N)};
    // An f32 whose low 16 mantissa bits are zero converts to bf16 EXACTLY, so
    // walking the top half of the f32 word walks every bf16 pattern there is —
    // subnormals, both zeros, infinities and NaNs included.
    const gbuf = try allocator.alloc(f32, N);
    defer allocator.free(gbuf);
    for (gbuf, 0..) |*v, i| v.* = @bitCast(@as(u32, @intCast(i)) << 16);
    const g32 = mlx.mlx_array_new_data(gbuf.ptr, &sh, 1, .float32);
    defer _ = mlx.mlx_array_free(g32);
    var gate = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gate);
    try mlx.check(mlx.mlx_astype(&gate, g32, .bfloat16, s));

    const ones = try allocator.alloc(f32, N);
    defer allocator.free(ones);
    for (ones, 0..) |*v, i| v.* = if (i % 3 == 0) 1.0 else if (i % 3 == 1) -0.375 else 2.5;
    const up32 = mlx.mlx_array_new_data(ones.ptr, &sh, 1, .float32);
    defer _ = mlx.mlx_array_free(up32);
    var up = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(up);
    try mlx.check(mlx.mlx_astype(&up, up32, .bfloat16, s));

    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, gate, s));
    var act = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(act);
    try mlx.check(mlx.mlx_multiply(&act, gate, sig, s));
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    try mlx.check(mlx.mlx_multiply(&ref, act, up, s));

    const got = (try fusedSwiGLU(s, gate, up)) orelse return error.FusedSwigluDeclined;
    defer _ = mlx.mlx_array_free(got);

    const ref_host = try allocator.alloc(f32, N);
    defer allocator.free(ref_host);
    const got_host = try allocator.alloc(f32, N);
    defer allocator.free(got_host);
    try testReadF32(ref, ref_host, s);
    try testReadF32(got, got_host, s);
    var bad: usize = 0;
    var first: usize = 0;
    for (ref_host, got_host, 0..) |r, g, i| {
        // NaN inputs produce NaN on both sides; compare them as bit patterns.
        const same = (std.math.isNan(r) and std.math.isNan(g)) or r == g;
        if (!same) {
            if (bad == 0) first = i;
            bad += 1;
        }
    }
    if (bad != 0) {
        std.debug.print("swiglu mismatch on {d}/{d} bf16 values; first at 0x{X:0>4}: ref={d} got={d}\n", .{ bad, N, first, ref_host[first], got_host[first] });
    }
    try testing.expectEqual(@as(usize, 0), bad);
}

test "fused SwiGLU declines mismatched shapes rather than reading out of bounds" {
    const s = mlx.gpuStream();
    swiglu_fused_override = true;
    defer swiglu_fused_override = null;
    const g_sh = [_]c_int{ 4, 8 };
    const u_sh = [_]c_int{ 4, 4 };
    var gbuf: [32]f32 = @splat(0.5);
    var ubuf: [16]f32 = @splat(0.25);
    const g32 = mlx.mlx_array_new_data(&gbuf, &g_sh, 2, .float32);
    defer _ = mlx.mlx_array_free(g32);
    const u32a = mlx.mlx_array_new_data(&ubuf, &u_sh, 2, .float32);
    defer _ = mlx.mlx_array_free(u32a);
    var g = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g);
    try mlx.check(mlx.mlx_astype(&g, g32, .bfloat16, s));
    var u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(u);
    try mlx.check(mlx.mlx_astype(&u, u32a, .bfloat16, s));
    try testing.expect((try fusedSwiGLU(s, g, u)) == null);
    // f32 is declined outright — see the dtype gate's comment.
    try testing.expect((try fusedSwiGLU(s, g32, g32)) == null);
}

test "gpt_oss clamped SwiGLU matches the reference formula (f64 oracle)" {
    // Reference (mlx-lm gpt_oss.swiglu):
    //   x_glu    = clip(gate, max=limit)          — ASYMMETRIC: no lower clip
    //   x_linear = clip(up,  -limit, +limit)
    //   out      = x_glu * sigmoid(alpha*x_glu) * (x_linear + 1)
    // The three easy things to get wrong are all covered by the sweep below:
    // clipping gate on BOTH sides, forgetting the `+ 1` on the linear branch,
    // and applying alpha to the product rather than inside the sigmoid.
    const s = mlx.gpuStream();
    const t = std.testing;
    const limit: f32 = 7.0;
    const alpha: f32 = 1.702;

    // Values that straddle both clip boundaries in both directions.
    const vals = [_]f32{ -20.0, -8.5, -7.0, -3.25, -0.5, 0.0, 0.5, 3.25, 7.0, 8.5, 20.0 };
    const n: c_int = @intCast(vals.len * vals.len);
    var gate_h = try t.allocator.alloc(f32, vals.len * vals.len);
    defer t.allocator.free(gate_h);
    var up_h = try t.allocator.alloc(f32, vals.len * vals.len);
    defer t.allocator.free(up_h);
    for (vals, 0..) |g, i| {
        for (vals, 0..) |u, j| {
            gate_h[i * vals.len + j] = g;
            up_h[i * vals.len + j] = u;
        }
    }
    const shape = [_]c_int{n};
    const gate = mlx.mlx_array_new_data(gate_h.ptr, &shape, 1, .float32);
    defer _ = mlx.mlx_array_free(gate);
    const up = mlx.mlx_array_new_data(up_h.ptr, &shape, 1, .float32);
    defer _ = mlx.mlx_array_free(up);

    const out = try gptOssSwiGLU(s, gate, up, limit, alpha);
    defer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_array_eval(out));
    const got: [*]const f32 = @ptrCast(@alignCast(mlx.mlx_array_data_float32(out)));

    for (gate_h, up_h, 0..) |g, u, i| {
        const gd: f64 = @min(@as(f64, g), @as(f64, limit));
        const ud: f64 = @max(@min(@as(f64, u), @as(f64, limit)), -@as(f64, limit));
        const sig: f64 = 1.0 / (1.0 + @exp(-@as(f64, alpha) * gd));
        const want: f64 = gd * sig * (ud + 1.0);
        try t.expectApproxEqAbs(want, @as(f64, got[i]), 1e-4);
    }
}

test "the SwiGLU sigmoid table is MLX's sigmoid on the device that asks for it" {
    // CPU and GPU sigmoids round differently; a table cached from one device served the other, so a model's
    // output on one stream depended on which stream had asked first.
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    const gpu = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(gpu);
    for ([_]mlx.mlx_stream{ cpu, gpu, cpu }) |s| for ([_]mlx.mlx_dtype{ .bfloat16, .float16 }) |dt| {
        const vals = try all16BitPatterns(s, dt, testing.allocator);
        defer _ = mlx.mlx_array_free(vals);
        var want = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(want);
        try mlx.check(mlx.mlx_sigmoid(&want, vals, s));
        try mlx.check(mlx.mlx_array_eval(want));
        // Compared as bits: the NaN patterns are part of the table.
        var g = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(g);
        try mlx.check(mlx.mlx_view(&g, try swigluSigTable(s, dt, testing.allocator), .uint16, cpu));
        var w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w);
        try mlx.check(mlx.mlx_view(&w, want, .uint16, cpu));
        var eq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(eq);
        try mlx.check(mlx.mlx_array_equal(&eq, g, w, false, cpu));
        var same = false;
        try mlx.check(mlx.mlx_array_item_bool(&same, eq));
        try testing.expect(same);
    };
}
