//! sdk/blocks/linear/verify_qmm_nax, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns vqmm_nax_avail_cache, verifyQmmNaxAvailable, vqmm_tile_cache, verifyQmmTile, vqmm_nax_env_cache, naxLaneEnvEnabled, and 19 more.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../host/log.zig");
const transformer = @import("../../../engines/mlx/transformer.zig");
const verify_qmm = @import("verify_qmm.zig");
const computeNaxAvailable = @import("../../../mlx/device.zig").computeNaxAvailable;
const gpuArchitecture = @import("../../../mlx/device.zig").gpuArchitecture;
const macosProductVersion = @import("../../../mlx/device.zig").macosProductVersion;
const VqmmTile = @import("verify_qmm.zig").VqmmTile;
const cachedScalarInt = @import("verify_qmm.zig").cachedScalarInt;
const verifyQmmEnabled = @import("verify_qmm.zig").verifyQmmEnabled;
const vqmmLaneFor = @import("verify_qmm.zig").vqmmLaneFor;
const vqmmTileFrom = @import("verify_qmm.zig").vqmmTileFrom;

var vqmm_nax_avail_cache: ?bool = null;

/// M5-class NAX units present: "applegpu_g17" GPU + macOS >= 26.2.
/// MLX_SERVE_FORCE_GPU_FAMILY_FALLBACK=1 pretends the units are absent so
/// an M5 can rehearse the exact M1-M4 plain-SIMD path (QA switch, mirrored
/// from MTPLX). Cached for the process lifetime (inference-thread caller
/// discipline, same as the kernel caches).
pub fn verifyQmmNaxAvailable() bool {
    if (transformer.vqmm_nax_probe_override) |v| return v;
    if (vqmm_nax_avail_cache) |v| return v;
    const ok = computeNaxAvailable();
    vqmm_nax_avail_cache = ok;
    return ok;
}

var vqmm_tile_cache: ?VqmmTile = null;

/// The matmul2d tile this process dispatches (lane kill switch included).
pub fn verifyQmmTile() VqmmTile {
    if (!naxLaneEnvEnabled()) return .off;
    if (transformer.vqmm_nax_probe_override) |v| return if (v) .nax else .off;
    if (vqmm_tile_cache) |v| return v;
    var arch_buf: [128]u8 = undefined;
    const arch = gpuArchitecture(&arch_buf) orelse "";
    var ver_buf: [64]u8 = undefined;
    const ver = macosProductVersion(&ver_buf) orelse "";
    const tile = vqmmTileFrom(verifyQmmNaxAvailable(), arch, ver);
    vqmm_tile_cache = tile;
    return tile;
}

var vqmm_nax_env_cache: ?bool = null;
/// Lane kill switch: MLX_SERVE_VERIFY_QMM_NAX=0 (the family-wide
/// MLX_SERVE_VERIFY_QMM=0 also covers it via verifyQmm's entry gate).
pub fn naxLaneEnvEnabled() bool {
    if (vqmm_nax_env_cache) |v| return v;
    var on = true;
    if (std.c.getenv("MLX_SERVE_VERIFY_QMM_NAX")) |p| {
        const val = std.mem.span(p);
        if (val.len > 0 and val[0] == '0') on = false;
    }
    vqmm_nax_env_cache = on;
    return on;
}

var vqmm_nax_mixed_env_cache: ?bool = null;
/// Mixed-width NAX A/B seam: keep the q4 NAX lane live while routing affine
/// q5/q6/q8 back through stock qmm. This isolates the byte-unpack kernel
/// contribution without changing controller state or disabling q4 NAX.
pub fn naxMixedBitsEnvEnabled() bool {
    if (vqmm_nax_mixed_env_cache) |v| return v;
    var on = true;
    if (std.c.getenv("MLX_SERVE_VERIFY_QMM_NAX_MIXED")) |p| {
        const val = std.mem.span(p);
        if (val.len > 0 and val[0] == '0') on = false;
    }
    vqmm_nax_mixed_env_cache = on;
    return on;
}

/// MLX_SERVE_VERIFY_QMM_NAX_MIN_M parse: explicit M width where the NAX tile
/// takes over from the plain-SIMD lanes. Without an override q4/q6 start at
/// M=8 and q8 at its measured M=7. Clamped to [2,16]; disabling is the lane
/// kill switch's job.
fn naxMinMOverrideFrom(val: ?[]const u8) ?c_int {
    const v = val orelse return null;
    const parsed = std.fmt.parseInt(c_int, v, 10) catch return null;
    return @min(16, @max(2, parsed));
}

pub fn naxMinMForBitsFrom(bits: u32, val: ?[]const u8) c_int {
    if (naxMinMOverrideFrom(val)) |m| return m;
    // The byte-per-value q8 NAX tile wins already at M=7 (+6.09% over the
    // plain lane, five thermally-settled counterbalanced pairs). Packed q4/q6
    // still lose there, so their measured takeover remains M=8.
    return if (bits == 8) 7 else 8;
}

var vqmm_nax_min_m_cache: ?c_int = null;
var vqmm_nax_min_m_env_checked = false;
pub fn naxMinMForBits(bits: u32) c_int {
    if (!vqmm_nax_min_m_env_checked) {
        vqmm_nax_min_m_cache = naxMinMOverrideFrom(
            if (std.c.getenv("MLX_SERVE_VERIFY_QMM_NAX_MIN_M")) |p| std.mem.span(p) else null,
        );
        vqmm_nax_min_m_env_checked = true;
    }
    return vqmm_nax_min_m_cache orelse naxMinMForBitsFrom(bits, null);
}

fn naxMinM() c_int {
    return naxMinMForBits(4);
}

/// Exact NAX-dispatch predicate, pure over the runtime gates and shape. Keep
/// controller policy behind this seam so a kill switch, takeover-width tweak,
/// or ineligible projection geometry can never advertise a NAX cost profile
/// while verifyQmm would actually fall through to stock qmm.
pub fn verifyQmmNaxEnabledForMFrom(
    m: c_int,
    K: c_int,
    N: c_int,
    verify_on: bool,
    lane_on: bool,
    available: bool,
    min_m: c_int,
) bool {
    return verify_on and vqmmLaneFor(m, K, N, lane_on and available, min_m, false) == .nax;
}

/// Runtime form of verifyQmmNaxEnabledForMFrom.
pub fn verifyQmmNaxEnabledForM(m: c_int, K: c_int, N: c_int) bool {
    return verifyQmmNaxEnabledForMFrom(
        m,
        K,
        N,
        verifyQmmEnabled(),
        naxLaneEnvEnabled(),
        verifyQmmNaxAvailable(),
        naxMinM(),
    );
}

const VQMM_NAX_HEADER: [:0]const u8 =
    \\#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    \\
;

/// MTPLX's _build_kernel_m16_nax_ktmpl body VERBATIM (nax_verify.py,
/// Apache-2.0; single-brace form of their f-string). The tensor
/// extents/strides encode dflash's working matmul2d convention — treat
/// them as opaque and correct, do not re-derive. Geometry: threadgroup =
/// 256 threads = 8 simdgroups; each simdgroup owns a K/8 chunk and stages
/// a dequantized 16x32 B tile in threadgroup memory (each of its 32 lanes
/// dequants ONE output column's 16 K-values from MLX's packed byte stream per
/// iteration); matmul2d (16x32x16 multiply_accumulate, execution_simdgroup)
/// multiplies the row-padded activation tile against it into a cooperative
/// fp32 accumulator; after the K loop the 8 partial C tiles reduce across
/// the threadgroup. T/GS/KCONST arrive as template args (MLX name-mangles
/// template values into the host_name, so one kernel object serves every
/// specialization — the GDN Dk/Dv precedent).
const VQMM_NAX_SOURCE: [:0]const u8 =
    \\using namespace metal;
    \\using namespace mpp::tensor_ops;
    \\
    \\constexpr int BM = BMT;
    \\constexpr int BN = 32;
    \\constexpr int BK = 16;
    \\constexpr int NSG = 8;
    \\constexpr int K = KCONST;
    \\constexpr int K_bytes = K * BITS / 8;
    \\constexpr int K_by_gs = K / GS;
    \\constexpr int K_chunk = K / NSG;
    \\
    \\uint tid = thread_position_in_threadgroup.x;
    \\uint sg_id = simdgroup_index_in_threadgroup;
    \\uint lane = thread_index_in_simdgroup;
    \\uint tg_n = threadgroup_position_in_grid.y;
    \\int N = int(N_size);
    \\int n0 = int(tg_n) * BN;
    \\int k_begin = int(sg_id) * K_chunk;
    \\int k_end = k_begin + K_chunk;
    \\
    \\threadgroup T B_tile[NSG][BK * BN];
    \\threadgroup float partial[NSG][BM * BN];
    \\
    \\constexpr auto desc = matmul2d_descriptor(
    \\    BM,
    \\    32,
    \\    16,
    \\    false,
    \\    false,
    \\    false,
    \\    matmul2d_descriptor::mode::multiply_accumulate);
    \\matmul2d<desc, metal::execution_simdgroup> op;
    \\
    \\tensor<device T, dextents<int, 2>, tensor_inline> A(
    \\    (device T*)x,
    \\    dextents<int, 2>{K, BM},
    \\    array<int, 2>{1, K});
    \\tensor<threadgroup T, dextents<int, 2>, tensor_inline> B(
    \\    B_tile[sg_id],
    \\    dextents<int, 2>{BN, BK},
    \\    array<int, 2>{1, BN});
    \\tensor<threadgroup float, dextents<int, 2>, tensor_inline> C(
    \\    partial[sg_id],
    \\    dextents<int, 2>{BN, BM},
    \\    array<int, 2>{1, BN});
    \\
    \\auto ct_c = op.template get_destination_cooperative_tensor<
    \\    tensor<device T, extents<int, 16, BM>, tensor_inline>,
    \\    tensor<threadgroup T, extents<int, 32, 16>, tensor_inline>,
    \\    float>();
    \\_Pragma("unroll")
    \\for (uint16_t i = 0; i < ct_c.get_capacity(); ++i) {
    \\    ct_c[i] = 0.0f;
    \\}
    \\
    \\int n_global = n0 + int(lane);
    \\for (int k0 = k_begin; k0 < k_end; k0 += BK) {
    \\    const device uchar* wp =
    \\        ((const device uchar*)w_q) + n_global * K_bytes + (k0 * BITS) / 8;
    \\    float scale = float(scales[n_global * K_by_gs + (k0 / GS)]);
    \\    float bias = float(biases[n_global * K_by_gs + (k0 / GS)]);
    \\
    \\    if constexpr (BITS == 4) {
    \\        _Pragma("unroll")
    \\        for (int pack = 0; pack < 2; ++pack) {
    \\            uint32_t p =
    \\                uint32_t(wp[pack * 4 + 0]) |
    \\                (uint32_t(wp[pack * 4 + 1]) << 8) |
    \\                (uint32_t(wp[pack * 4 + 2]) << 16) |
    \\                (uint32_t(wp[pack * 4 + 3]) << 24);
    \\            _Pragma("unroll")
    \\            for (int ki = 0; ki < 8; ++ki) {
    \\                uint32_t q = (p >> (ki * 4)) & 0xFu;
    \\                B_tile[sg_id][(pack * 8 + ki) * BN + int(lane)] =
    \\                    T(float(q) * scale + bias);
    \\            }
    \\        }
    \\    } else if constexpr (BITS == 5) {
    \\        _Pragma("unroll")
    \\        for (int pack = 0; pack < 2; ++pack) {
    \\            ulong p =
    \\                ulong(wp[pack * 5 + 0]) |
    \\                (ulong(wp[pack * 5 + 1]) << 8) |
    \\                (ulong(wp[pack * 5 + 2]) << 16) |
    \\                (ulong(wp[pack * 5 + 3]) << 24) |
    \\                (ulong(wp[pack * 5 + 4]) << 32);
    \\            _Pragma("unroll")
    \\            for (int ki = 0; ki < 8; ++ki) {
    \\                uint32_t q = uint32_t((p >> (ki * 5)) & 0x1Ful);
    \\                B_tile[sg_id][(pack * 8 + ki) * BN + int(lane)] =
    \\                    T(float(q) * scale + bias);
    \\            }
    \\        }
    \\    } else if constexpr (BITS == 6) {
    \\        _Pragma("unroll")
    \\        for (int pack = 0; pack < 4; ++pack) {
    \\            uint32_t p =
    \\                uint32_t(wp[pack * 3 + 0]) |
    \\                (uint32_t(wp[pack * 3 + 1]) << 8) |
    \\                (uint32_t(wp[pack * 3 + 2]) << 16);
    \\            _Pragma("unroll")
    \\            for (int ki = 0; ki < 4; ++ki) {
    \\                uint32_t q = (p >> (ki * 6)) & 0x3Fu;
    \\                B_tile[sg_id][(pack * 4 + ki) * BN + int(lane)] =
    \\                    T(float(q) * scale + bias);
    \\            }
    \\        }
    \\    } else {
    \\        static_assert(BITS == 8, "unsupported NAX affine width");
    \\        _Pragma("unroll")
    \\        for (int ki = 0; ki < BK; ++ki) {
    \\            uint32_t q = uint32_t(wp[ki]);
    \\            B_tile[sg_id][ki * BN + int(lane)] =
    \\                T(float(q) * scale + bias);
    \\        }
    \\    }
    \\    simdgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    auto tA = A.template slice<16, BM>(k0, 0);
    \\    auto tB = B.template slice<32, 16>(0, 0);
    \\    op.run(tA, tB, ct_c);
    \\    simdgroup_barrier(mem_flags::mem_threadgroup);
    \\}
    \\
    \\auto tC = C.template slice<32, BM>(0, 0);
    \\ct_c.store(tC);
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\for (int off = int(tid); off < BM * BN; off += NSG * 32) {
    \\    float acc01 = partial[0][off] + partial[1][off];
    \\    float acc23 = partial[2][off] + partial[3][off];
    \\    float acc45 = partial[4][off] + partial[5][off];
    \\    float acc67 = partial[6][off] + partial[7][off];
    \\    float acc = (acc01 + acc23) + (acc45 + acc67);
    \\    int row = off / BN;
    \\    int col = off - row * BN;
    \\    y[row * N + n0 + col] = T(acc);
    \\}
;

var vqmm_nax_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// ONLY call where `verifyQmmTile()` is live — see the never-build rule.
fn getVerifyQmmNaxKernel() !mlx.mlx_fast_metal_kernel {
    if (vqmm_nax_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "x", "w_q", "scales", "biases", "N_size" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "mlxserve_vqmm_nax_m16",
        in_vec,
        out_vec,
        VQMM_NAX_SOURCE,
        VQMM_NAX_HEADER,
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    vqmm_nax_kernel = kernel;
    return kernel;
}

/// The M-padding half of the NAX host contract, split out so it can be
/// exercised hermetically on non-M5 machines (the scaffolding test wraps
/// it around STOCK qmm — zero pad rows must produce zero output rows):
/// collapse the caller's leading shape to [m, K] (mirrors MTPLX's
/// x.reshape(m, k) so row padding is well-defined for every batch shape),
/// then zero-pad the row axis to the tile's row height. Owned handle.
pub fn naxPadRows(s: mlx.mlx_stream, x: mlx.mlx_array, m: c_int, tile: c_int, K: c_int, xd: mlx.mlx_dtype) !mlx.mlx_array {
    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    const x2_shape = [_]c_int{ m, K };
    try mlx.check(mlx.mlx_reshape(&x2, x, &x2_shape, 2, s));
    var x16 = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(x16);
    if (m < tile) {
        var zero = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(zero);
        const zdims = [_]c_int{};
        try mlx.check(mlx.mlx_zeros(&zero, &zdims, 0, xd, s));
        const pad_axes = [_]c_int{0};
        const pad_low = [_]c_int{0};
        const pad_high = [_]c_int{tile - m};
        try mlx.check(mlx.mlx_pad(&x16, x2, &pad_axes, 1, &pad_low, 1, &pad_high, 1, zero, "constant", s));
    } else {
        try mlx.check(mlx.mlx_array_set(&x16, x2));
    }
    return x16;
}

/// The slice-back half: first m rows of the [tile, N] output (pad rows
/// dropped). Owned handle.
pub fn naxSliceRows(s: mlx.mlx_stream, y16: mlx.mlx_array, m: c_int, tile: c_int, N: c_int) !mlx.mlx_array {
    var ym = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(ym);
    if (m < tile) {
        const start = [_]c_int{ 0, 0 };
        const stop = [_]c_int{ m, N };
        const strides = [_]c_int{ 1, 1 };
        try mlx.check(mlx.mlx_slice(&ym, y16, &start, 2, &stop, 2, &strides, 2, s));
    } else {
        try mlx.check(mlx.mlx_array_set(&ym, y16));
    }
    return ym;
}

/// Run the NAX m16 tile: pad the activations to the fixed [16, K] tile
/// ("weight streaming dominates so padded rows are nearly free" — the
/// source ledger), grid (256, N/32), slice y back to the caller's M rows.
pub fn runVerifyQmmNax(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    bits: u32,
    group_size: u32,
    m: c_int,
    tile: c_int,
    K: c_int,
    N: c_int,
    xd: mlx.mlx_dtype,
    xsh: []const c_int,
) !?mlx.mlx_array {
    if (!verify_qmm.vqmm_nax_engaged) {
        verify_qmm.vqmm_nax_engaged = true;
        log.info("[vqmm] NAX verify lane engaged at {d}-bit ({s} tile, {d} rows): M={d} K={d} N={d} gs={d} (MLX_SERVE_VERIFY_QMM_NAX=0 restores stock)\n", .{ bits, @tagName(verifyQmmTile()), tile, m, K, N, group_size });
    }
    const x16 = try naxPadRows(s, x, m, tile, K, xd);
    defer _ = mlx.mlx_array_free(x16);

    const N_arr = cachedScalarInt(N);

    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    const y_shape = [_]c_int{ tile, N };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &y_shape, 2, xd));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 256, @divExact(N, 32), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", xd));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BITS", @intCast(bits)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GS", @intCast(group_size)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "KCONST", K));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BMT", tile));

    const inputs_arr = [_]mlx.mlx_array{ x16, w, sc, bi, N_arr };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);

    const kernel = try getVerifyQmmNaxKernel();
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, config, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var y16 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y16);
    try mlx.check(mlx.mlx_vector_array_get(&y16, outputs_vec, 0));

    // Drop the pad rows, restore the caller's leading shape with N last.
    const ym = try naxSliceRows(s, y16, m, tile, N);
    defer _ = mlx.mlx_array_free(ym);

    var out_shape_buf: [8]c_int = undefined;
    const ndim = xsh.len;
    if (ndim > out_shape_buf.len) return error.UnsupportedShape;
    for (xsh[0 .. ndim - 1], 0..) |d, i| out_shape_buf[i] = d;
    out_shape_buf[ndim - 1] = N;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_reshape(&y, ym, out_shape_buf[0..ndim].ptr, @intCast(ndim), s));
    return y;
}

// ── tests ──

/// Family-wide NAX readiness: family kill switch, lane kill switch, and the
/// hardware probe all pass. This intentionally says nothing about a specific
/// M/K/N dispatch; controller decisions use verifyQmmNaxEnabledForM instead.
pub fn verifyQmmNaxEnabled() bool {
    return verifyQmmEnabled() and naxLaneEnvEnabled() and verifyQmmNaxAvailable();
}

pub fn naxMinMFrom(val: ?[]const u8) c_int {
    return naxMinMForBitsFrom(4, val);
}
