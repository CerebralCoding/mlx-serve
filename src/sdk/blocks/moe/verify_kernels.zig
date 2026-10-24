//! sdk/blocks/moe/verify_kernels, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns VERIFY_EXPERT_REUSE_SOURCE, verify_expert_reuse_kernel, and the other verify-row MoE kernels.

const std = @import("std");
const env = @import("../../../env.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const verify_projection = @import("../linear/verify_projection.zig");
const testing = std.testing;
const QsaCfgCache = @import("../../../mlx/kernel_cache.zig").QsaCfgCache;
const bf16BitsEqual = @import("../../testing/parity.zig").bf16BitsEqual;
const testRandUniformBf16 = @import("../../testing/parity.zig").testRandUniformBf16;
const testRandWeightBf16 = @import("../../testing/parity.zig").testRandWeightBf16;
const attn256RandBf16 = @import("../attention/attn256.zig").attn256RandBf16;
const verifySharedCapable = @import("../linear/verify_projection.zig").verifySharedCapable;
const verifySharedHardware = @import("../linear/verify_projection.zig").verifySharedHardware;
const cachedScalarInt = @import("../linear/verify_qmm.zig").cachedScalarInt;
const fusedSwiGLU = @import("../mlp/fused_activations.zig").fusedSwiGLU;
const swigluFusedEnabled = @import("../mlp/fused_activations.zig").swigluFusedEnabled;
const swigluSigTable = @import("../mlp/fused_activations.zig").swigluSigTable;
const gatherExpertMm = @import("experts.zig").gatherExpertMm;

// Preserve MLX's affine qmv K reduction while sharing each packed word across a pair.
pub const VERIFY_EXPERT_REUSE_SOURCE = kernelSource("kernels/moe/verify_expert_reuse.metal");

var verify_expert_reuse_kernel: ?mlx.mlx_fast_metal_kernel = null;


const ExpertReuseKey = struct { rows: c_int, n: c_int, k: c_int };
var verify_expert_reuse_cfgs = QsaCfgCache(ExpertReuseKey, 1){};

pub fn verifyExpertReuse(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, ids: mlx.mlx_array) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or !verifySharedHardware()) return null;
    if (x.ctx == null or w.ctx == null or sc.ctx == null or bi.ctx == null or ids.ctx == null) return null;
    if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(sc) != .bfloat16 or mlx.mlx_array_dtype(bi) != .bfloat16 or mlx.mlx_array_dtype(w) != .uint32 or mlx.mlx_array_dtype(ids) != .uint32) return null;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    const ss = mlx.getShape(sc);
    const bs = mlx.getShape(bi);
    const is = mlx.getShape(ids);
    if (xs.len != 3 or xs[1] != 1 or ws.len != 3 or ss.len != 3 or !std.mem.eql(c_int, ss, bs) or is.len != 1 or is[0] != xs[0]) return null;
    const count = xs[0];
    const k = xs[2];
    const n = ws[1];
    if (count < 2 or count > 320 or ws[0] < 1 or @divTrunc(count, ws[0]) >= 4) return null;
    if (k != 640 or n != 2560) return null;
    if (ws[2] != @divExact(k, 8) or ss[0] != ws[0] or ss[1] != n or ss[2] != @divExact(k, 64)) return null;
    const kernel = blk: {
        if (verify_expert_reuse_kernel) |value| break :blk value;
        const ins = [_][*:0]const u8{ "x", "w", "sc", "bi", "ids" };
        const outs = [_][*:0]const u8{"y"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const value = mlx.mlx_fast_metal_kernel_new("verify_expert_reuse", iv, ov, VERIFY_EXPERT_REUSE_SOURCE, "", true, false);
        if (value.ctx == null) return error.MetalKernelCompileFailed;
        verify_expert_reuse_kernel = value;
        break :blk value;
    };
    const key = ExpertReuseKey{ .rows = count, .n = n, .k = k };
    const cfg = blk: {
        if (verify_expert_reuse_cfgs.get(key)) |hit| break :blk hit[0];
        const value = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(value);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(value, &.{ count, 1, n }, 3, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(value, 32, @divExact(n, 4), count));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(value, 32, 2, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(value, "T", .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(value, "K", k));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(value, "N", n));
        verify_expert_reuse_cfgs.put(key, .{value});
        break :blk value;
    };
    const values = [_]mlx.mlx_array{ x, w, sc, bi, ids };
    const iv = mlx.mlx_vector_array_new_data(&values, values.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel, iv, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    return out;
}

// Match the eight bf16 partial sums used by MLX col_reduce_small (top-k rows, hidden H).
pub const VERIFY_EXPERT_REDUCE_SOURCE = kernelSource("kernels/moe/verify_expert_reduce.metal");
var verify_expert_reduce_kernel: ?mlx.mlx_fast_metal_kernel = null;
const ExpertReduceCfgs = QsaCfgCache([3]c_int, 1);
var verify_expert_reduce_cfgs = ExpertReduceCfgs{};
var prefill_expert_reduce_cfgs = ExpertReduceCfgs{};

pub fn verifyExpertReduce(s: mlx.mlx_stream, down: mlx.mlx_array, inverse: mlx.mlx_array, scores: mlx.mlx_array) !?mlx.mlx_array {
    if (!verifySharedHardware() or scores.ctx == null) return null;
    const ss = mlx.getShape(scores);
    if (ss.len != 3 or ss[0] != 1 or ss[1] > 32) return null;
    return expertReduceApply(s, down, inverse, scores, &verify_expert_reduce_cfgs);
}

/// The verify reduction at prefill widths, on every GPU: it only removes the
/// [B*S*K, hidden] unsorted and weighted intermediates.
pub fn moePrefillExpertReduce(s: mlx.mlx_stream, down: mlx.mlx_array, inverse: mlx.mlx_array, scores: mlx.mlx_array) !?mlx.mlx_array {
    if (scores.ctx == null) return null;
    const ss = mlx.getShape(scores);
    // Verify widths (<= 32 tokens) keep verifyExpertReduce's own gates.
    if (ss.len != 3 or ss[0] * ss[1] <= 32) return null;
    if (mlx.cudaAvailable()) return @import("moe_prefill_cuda.zig").weightedSum(s, down, inverse, scores);
    return expertReduceApply(s, down, inverse, scores, &prefill_expert_reduce_cfgs);
}

fn expertReduceApply(s: mlx.mlx_stream, down: mlx.mlx_array, inverse: mlx.mlx_array, scores: mlx.mlx_array, cfgs: *ExpertReduceCfgs) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return null;
    if (down.ctx == null or inverse.ctx == null or scores.ctx == null) return null;
    if (mlx.mlx_array_dtype(down) != .bfloat16 or mlx.mlx_array_dtype(scores) != .bfloat16 or mlx.mlx_array_dtype(inverse) != .uint32) return null;
    const ds = mlx.getShape(down);
    const ss = mlx.getShape(scores);
    const ix = mlx.getShape(inverse);
    // The partial-sum layout reproduces MLX's tree only at these (top_k, hidden) pairs.
    if (ds.len != 2 or ss.len != 3 or ix.len != 1) return null;
    const topk = ss[2];
    const hidden = ds[1];
    if (!((topk == 10 and hidden == 2560) or (topk == 8 and hidden == 4096))) return null;
    const batch = ss[0];
    const tokens = ss[1];
    if (batch < 1 or tokens < 1 or batch * tokens < 2 or ds[0] != batch * tokens * topk or ix[0] != ds[0]) return null;
    const kernel = blk: {
        if (verify_expert_reduce_kernel) |value| break :blk value;
        const ins = [_][*:0]const u8{ "down", "inverse", "scores" };
        const outs = [_][*:0]const u8{"y"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const value = mlx.mlx_fast_metal_kernel_new("verify_expert_reduce", iv, ov, VERIFY_EXPERT_REDUCE_SOURCE, "", true, false);
        if (value.ctx == null) return error.MetalKernelCompileFailed;
        verify_expert_reduce_kernel = value;
        break :blk value;
    };
    const key = [3]c_int{ batch, tokens, topk };
    const cfg = blk: {
        if (cfgs.get(key)) |hit| break :blk hit[0];
        const value = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(value);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(value, &.{ batch, tokens, hidden }, 3, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(value, batch * tokens * hidden, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(value, 256, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(value, "T", .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(value, "TOPK", topk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(value, "H", hidden));
        cfgs.put(key, .{value});
        break :blk value;
    };
    const values = [_]mlx.mlx_array{ down, inverse, scores };
    const iv = mlx.mlx_vector_array_new_data(&values, values.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel, iv, cfg, s));
    var result = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(result);
    try mlx.check(mlx.mlx_vector_array_get(&result, ov, 0));
    return result;
}

// Preserve input order on ties so routing and the inverse permutation stay stable.
pub const VERIFY_ROUTE_PACK_SOURCE = kernelSource("kernels/moe/verify_route_pack.metal");
var verify_route_pack_kernel: ?mlx.mlx_fast_metal_kernel = null;
var verify_route_pack_configs = QsaCfgCache(c_int, 1){};
const VerifyRoutePack = struct {
    inverse: mlx.mlx_array = .{},
    sorted: mlx.mlx_array = .{},
    lhs: mlx.mlx_array = .{},
    fn deinit(self: *VerifyRoutePack) void {
        inline for (.{ "inverse", "sorted", "lhs" }) |name| {
            if (@field(self, name).ctx != null) _ = mlx.mlx_array_free(@field(self, name));
        }
    }
};

pub fn verifyRoutePack(s: mlx.mlx_stream, ids: mlx.mlx_array, top_k: c_int) !?VerifyRoutePack {
    if (!mlx.streamIsGpu(s) or !verifySharedHardware() or top_k != 10 or ids.ctx == null or mlx.mlx_array_dtype(ids) != .uint32) return null;
    const shape = mlx.getShape(ids);
    if (shape.len != 1 or shape[0] < 20 or shape[0] > 320 or @mod(shape[0], 10) != 0) return null;
    const kernel = blk: {
        if (verify_route_pack_kernel) |kernel| break :blk kernel;
        const ins = [_][*:0]const u8{"ids"};
        const outs = [_][*:0]const u8{ "inverse", "sorted", "lhs" };
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const kernel = mlx.mlx_fast_metal_kernel_new("verify_route_pack", iv, ov, VERIFY_ROUTE_PACK_SOURCE, "", true, false);
        if (kernel.ctx == null) return error.MetalKernelCompileFailed;
        verify_route_pack_kernel = kernel;
        break :blk kernel;
    };
    const config = blk: {
        if (verify_route_pack_configs.get(shape[0])) |hit| break :blk hit[0];
        const config = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
        for (0..3) |_| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, shape.ptr, 1, .uint32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 128, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 128, 1, 1));
        verify_route_pack_configs.put(shape[0], .{config});
        break :blk config;
    };
    const iv = mlx.mlx_vector_array_new_data(&.{ids}, 1);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel, iv, config, s));
    var result: VerifyRoutePack = .{};
    errdefer result.deinit();
    inline for (.{ "inverse", "sorted", "lhs" }, 0..) |name, i| {
        @field(result, name) = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&@field(result, name), ov, i));
    }
    return result;
}

// A stable counting sort of routed expert ids for prefill widths, in three
// dispatches over 256-id blocks: per-block histograms, one scan to (expert,
// block) offsets, and a scatter that ranks ties by position within the block.
const PREFILL_ROUTE_BLOCK: c_int = 256;
pub const PREFILL_ROUTE_HIST_SOURCE = kernelSource("kernels/moe/prefill_route_hist.metal");
pub const PREFILL_ROUTE_SCAN_SOURCE = kernelSource("kernels/moe/prefill_route_scan.metal");
pub const PREFILL_ROUTE_SCATTER_SOURCE = kernelSource("kernels/moe/prefill_route_scatter.metal");
var prefill_route_kernels: [3]?mlx.mlx_fast_metal_kernel = .{ null, null, null };
var prefill_route_configs = QsaCfgCache([3]c_int, 3){};

fn prefillRouteKernel(slot: usize, name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    if (prefill_route_kernels[slot]) |kernel| return kernel;
    const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const kernel = mlx.mlx_fast_metal_kernel_new(name, iv, ov, source, "", true, false);
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    prefill_route_kernels[slot] = kernel;
    return kernel;
}

fn prefillRouteApply(s: mlx.mlx_stream, kernel: mlx.mlx_fast_metal_kernel, inputs: []const mlx.mlx_array, cfg: mlx.mlx_fast_metal_kernel_config, outs: []const *mlx.mlx_array) !void {
    const iv = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel, iv, cfg, s));
    for (outs, 0..) |out, i| try mlx.check(mlx.mlx_vector_array_get(out, ov, i));
}

/// `verifyRoutePack`'s outputs at prefill widths, where its all-pairs rank is
/// quadratic. Ties keep input order, which is what MLX's argsort does.
pub fn moePrefillRoutePack(s: mlx.mlx_stream, ids: mlx.mlx_array, top_k: c_int, experts: c_int) !?VerifyRoutePack {
    if (!mlx.streamIsGpu(s) or ids.ctx == null or mlx.mlx_array_dtype(ids) != .uint32) return null;
    if (top_k < 1 or experts < 1 or experts > 1024) return null;
    const shape = mlx.getShape(ids);
    // Verify widths (<= 32 tokens) keep verifyRoutePack's own gates.
    if (shape.len != 1 or @mod(shape[0], top_k) != 0 or @divTrunc(shape[0], top_k) <= 32) return null;
    const n = shape[0];
    const blocks = @divTrunc(n + PREFILL_ROUTE_BLOCK - 1, PREFILL_ROUTE_BLOCK);
    const hist = try prefillRouteKernel(0, "prefill_route_hist", &.{"ids"}, &.{"counts"}, PREFILL_ROUTE_HIST_SOURCE);
    const scan = try prefillRouteKernel(1, "prefill_route_scan", &.{"counts"}, &.{"offsets"}, PREFILL_ROUTE_SCAN_SOURCE);
    const scatter = try prefillRouteKernel(2, "prefill_route_scatter", &.{ "ids", "offsets" }, &.{ "inverse", "sorted", "lhs" }, PREFILL_ROUTE_SCATTER_SOURCE);
    const cfgs = blk: {
        const key = [3]c_int{ n, top_k, experts };
        if (prefill_route_configs.get(key)) |hit| break :blk hit;
        var made: [3]mlx.mlx_fast_metal_kernel_config = undefined;
        var built: usize = 0;
        errdefer for (made[0..built]) |cfg| {
            _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        };
        const table = [_]c_int{blocks * experts};
        for (&made, 0..) |*cfg, i| {
            cfg.* = mlx.mlx_fast_metal_kernel_config_new();
            built += 1;
            if (i < 2) {
                try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg.*, &table, 1, .uint32));
            } else for (0..3) |_| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg.*, shape.ptr, 1, .uint32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg.*, if (i == 1) 256 else blocks * PREFILL_ROUTE_BLOCK, 1, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg.*, 256, 1, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg.*, "E", experts));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg.*, "TOPK", top_k));
        }
        prefill_route_configs.put(key, made);
        break :blk made;
    };
    var counts = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(counts);
    try prefillRouteApply(s, hist, &.{ids}, cfgs[0], &.{&counts});
    var offsets = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(offsets);
    try prefillRouteApply(s, scan, &.{counts}, cfgs[1], &.{&offsets});
    var result: VerifyRoutePack = .{ .inverse = mlx.mlx_array_new(), .sorted = mlx.mlx_array_new(), .lhs = mlx.mlx_array_new() };
    errdefer result.deinit();
    try prefillRouteApply(s, scatter, &.{ ids, offsets }, cfgs[2], &.{ &result.inverse, &result.sorted, &result.lhs });
    return result;
}

pub fn verifyIndexedExpertInput(s: mlx.mlx_stream, x: mlx.mlx_array, lhs: mlx.mlx_array, experts: c_int) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or !verifySharedHardware() or experts <= 0) return null;
    if (x.ctx == null or lhs.ctx == null or mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(lhs) != .uint32) return null;
    const xs = mlx.getShape(x);
    const ix = mlx.getShape(lhs);
    if (xs.len != 2 or xs[1] != 2560 or xs[0] < 2 or xs[0] > 32 or ix.len != 1 or ix[0] != xs[0] * 10) return null;
    // Explicit LHS indices disable MLX's sorted matrix path; keep both arms below it.
    if (@divTrunc(ix[0], experts) >= 4) return null;
    var view = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(view);
    try mlx.check(mlx.mlx_reshape(&view, x, &.{ xs[0], 1, xs[1] }, 3, s));
    return view;
}

// ── tests ──

const model_mod = @import("../../../engines/mlx/model.zig");
const expectRowsByteEqual = @import("../../../engines/mlx/batched/row_axis_verify.zig").expectRowsByteEqual;
const Transformer = @import("../../../engines/mlx/transformer.zig").Transformer;
const qsaArraysAllEqual = @import("../../testing/parity.zig").qsaArraysAllEqual;

test "verify MoE rows preserve every value with fewer graph operations" {
    const path = env.get(.MTP_TEST_MODEL) orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const allocator = testing.allocator;
    const stream = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try model_mod.parseConfig(io, allocator, std.mem.span(path));
    defer if (config.ngram_table_path) |name| allocator.free(name);
    var weights = try model_mod.loadWeights(io, allocator, std.mem.span(path));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, allocator, config, &weights);
    defer xfm.deinit();
    try testing.expect(xfm.qwen4 != null);
    xfm.compileMoeRouting();
    var prng = std.Random.DefaultPrng.init(924);
    for ([_]usize{ 0, config.num_hidden_layers / 2, config.num_hidden_layers - 1 }) |layer| {
        const mw = &xfm.moe_layers.?[layer].mlp.moe;
        for ([_]usize{ 2, 4 }) |n| for ([_][4]c_int{ .{ 2, 2, 2, 2 }, .{ 5, 5, 5, 5 }, .{ 5, 3, 2, 4 } }) |lengths| {
            var inputs: [4]mlx.mlx_array = undefined;
            defer for (inputs[0..n]) |input| {
                _ = mlx.mlx_array_free(input);
            };
            for (0..n) |row| {
                inputs[row] = try attn256RandBf16(prng.random(), &.{ 1, lengths[row], @intCast(config.hidden_size) }, stream);
                const warm = try xfm.moeMLP(inputs[row], mw);
                try mlx.check(mlx.mlx_array_eval(warm));
                _ = mlx.mlx_array_free(warm);
            }
            var solo: [4]mlx.mlx_array = undefined;
            defer for (solo[0..n]) |value| {
                _ = mlx.mlx_array_free(value);
            };
            const start_solo = mlx.op_count.load(.monotonic);
            for (0..n) |row| solo[row] = try xfm.moeMLP(inputs[row], mw);
            const solo_ops = mlx.op_count.load(.monotonic) - start_solo;
            const shared_before = verify_projection.mtp_verify_shared_rows_calls;
            const start_group = mlx.op_count.load(.monotonic);
            const group = try xfm.moeVerifyRows(inputs[0..n], mw);
            const group_ops = mlx.op_count.load(.monotonic) - start_group;
            try testing.expectEqual(shared_before + 1, verify_projection.mtp_verify_shared_rows_calls);
            defer {
                for (group) |value| _ = mlx.mlx_array_free(value);
                allocator.free(group);
            }
            for (0..n) |row| try expectRowsByteEqual(allocator, stream, solo[row], group[row], "verify MoE rows");
            std.debug.print("[verify-moe] layer={d} N={d} widths={any} ops solo={d} grouped={d}\n", .{ layer, n, lengths[0..n], solo_ops, group_ops });
            try testing.expect(group_ops < solo_ops);
        };
    }
}

test "verify expert reuse preserves stock gather reductions" {
    if (mlx.noGpuBackend() or !verifySharedHardware()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0xE4E4);
    for ([_][2]c_int{.{ 640, 2560 }}) |shape| {
        const k = shape[0];
        const n = shape[1];
        const weight = try attn256RandBf16(prng.random(), &.{ 64, n, k }, s);
        defer _ = mlx.mlx_array_free(weight);
        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(&triple, weight, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{}, s));
        var w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w);
        var sc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc);
        var bi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bi);
        try mlx.check(mlx.mlx_vector_array_get(&w, triple, 0));
        try mlx.check(mlx.mlx_vector_array_get(&sc, triple, 1));
        try mlx.check(mlx.mlx_vector_array_get(&bi, triple, 2));
        for ([_]c_int{ 20, 30, 40, 50, 60, 80, 200 }) |count| for (0..3) |pattern| {
            var host_ids: [200]u32 = undefined;
            var expert: u32 = 0;
            var remain: usize = 0;
            for (host_ids[0..@intCast(count)], 0..) |*id, i| {
                if (remain == 0) {
                    if (i != 0) expert += 1;
                    remain = switch (pattern) {
                        0 => @intCast(@divTrunc(count + 63, 64)),
                        1 => 4,
                        else => ([_]usize{ 1, 3, 6 })[expert % 3],
                    };
                }
                id.* = expert;
                remain -= 1;
            }
            const ids = mlx.mlx_array_new_data(&host_ids, &.{count}, 1, .uint32);
            defer _ = mlx.mlx_array_free(ids);
            const x = try attn256RandBf16(prng.random(), &.{ count, 1, k }, s);
            defer _ = mlx.mlx_array_free(x);
            if (count == 20 and pattern == 0) {
                try testing.expect((try verifyExpertReuse(s, x, w, sc, .{}, ids)) == null);
                const cpu = mlx.mlx_default_cpu_stream_new();
                defer _ = mlx.mlx_stream_free(cpu);
                try testing.expect((try verifyExpertReuse(cpu, x, w, sc, bi, ids)) == null);
                var x32 = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(x32);
                try mlx.check(mlx.mlx_astype(&x32, x, .float32, s));
                try testing.expect((try verifyExpertReuse(s, x32, w, sc, bi, ids)) == null);
                var signed_ids = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(signed_ids);
                try mlx.check(mlx.mlx_astype(&signed_ids, ids, .int32, s));
                try testing.expect((try verifyExpertReuse(s, x, w, sc, bi, signed_ids)) == null);
                var large_x = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(large_x);
                var large_ids = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(large_ids);
                try mlx.check(mlx.mlx_zeros(&large_x, &.{ 256, 1, k }, 3, .bfloat16, s));
                try mlx.check(mlx.mlx_zeros(&large_ids, &.{256}, 1, .uint32, s));
                try testing.expect((try verifyExpertReuse(s, large_x, w, sc, bi, large_ids)) == null);
            }
            var expected = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(expected);
            try gatherExpertMm(&expected, x, w, sc, bi, .{}, ids, 4, 64, .affine, true, s);
            const actual = (try verifyExpertReuse(s, x, w, sc, bi, ids)) orelse return error.ExpertReuseDeclined;
            defer _ = mlx.mlx_array_free(actual);
            const same = try qsaArraysAllEqual(expected, actual, s);
            if (!same) std.debug.print("[expert-reuse-mismatch] K={d} N={d} count={d} pattern={d}\n", .{ k, n, count, pattern });
            try testing.expect(same);
        };
    }
}

fn expertReduceReference(s: mlx.mlx_stream, down: mlx.mlx_array, inverse: mlx.mlx_array, scores: mlx.mlx_array) !mlx.mlx_array {
    const batch = mlx.getShape(scores)[0];
    const tokens = mlx.getShape(scores)[1];
    var unsorted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(unsorted);
    try mlx.check(mlx.mlx_take_axis(&unsorted, down, inverse, 0, s));
    var rows = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rows);
    try mlx.check(mlx.mlx_reshape(&rows, unsorted, &.{ batch, tokens, mlx.getShape(scores)[2], mlx.getShape(down)[1] }, 4, s));
    var expanded = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(expanded);
    try mlx.check(mlx.mlx_expand_dims(&expanded, scores, -1, s));
    var weighted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weighted);
    try mlx.check(mlx.mlx_multiply(&weighted, rows, expanded, s));
    var expected = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(expected);
    try mlx.check(mlx.mlx_sum_axis(&expected, weighted, -2, false, s));
    return expected;
}

test "verify expert reduction preserves unsort multiply and sum" {
    if (mlx.noGpuBackend() or !verifySharedHardware()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0xE55E);
    for ([_]c_int{ 2, 3, 4, 5, 6, 8, 10, 12, 20, 32 }) |tokens| {
        const count = tokens * 10;
        const down = try attn256RandBf16(prng.random(), &.{ count, 2560 }, s);
        defer _ = mlx.mlx_array_free(down);
        const scores = try attn256RandBf16(prng.random(), &.{ 1, tokens, 10 }, s);
        defer _ = mlx.mlx_array_free(scores);
        var perm: [320]u32 = undefined;
        for (perm[0..@intCast(count)], 0..) |*id, i| id.* = @intCast(i);
        prng.random().shuffle(u32, perm[0..@intCast(count)]);
        const inverse = mlx.mlx_array_new_data(&perm, &.{count}, 1, .uint32);
        defer _ = mlx.mlx_array_free(inverse);
        if (tokens == 4) {
            try testing.expect((try verifyExpertReduce(s, down, inverse, .{})) == null);
            const cpu = mlx.mlx_default_cpu_stream_new();
            defer _ = mlx.mlx_stream_free(cpu);
            try testing.expect((try verifyExpertReduce(cpu, down, inverse, scores)) == null);
            var down32 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(down32);
            try mlx.check(mlx.mlx_astype(&down32, down, .float32, s));
            try testing.expect((try verifyExpertReduce(s, down32, inverse, scores)) == null);
            var signed = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(signed);
            try mlx.check(mlx.mlx_astype(&signed, inverse, .int32, s));
            try testing.expect((try verifyExpertReduce(s, down, signed, scores)) == null);
            var batched = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(batched);
            try mlx.check(mlx.mlx_reshape(&batched, scores, &.{ 2, 2, 10 }, 3, s));
            try testing.expect((try verifyExpertReduce(s, down, inverse, batched)) == null);
        }
        const expected = try expertReduceReference(s, down, inverse, scores);
        defer _ = mlx.mlx_array_free(expected);
        const actual = (try verifyExpertReduce(s, down, inverse, scores)) orelse return error.ExpertReduceDeclined;
        defer _ = mlx.mlx_array_free(actual);
        try testing.expect(try qsaArraysAllEqual(expected, actual, s));
    }
}

test "prefill expert reduction is bit-identical to unsort multiply and sum" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x5EED_3A11);
    // (top_k, hidden): the qwen3.6 35B and MiMo-V2.6 geometries.
    for ([_][2]c_int{ .{ 10, 2560 }, .{ 8, 4096 } }) |kh| for ([_][2]c_int{ .{ 1, 33 }, .{ 1, 2048 }, .{ 2, 1024 }, .{ 1, 8192 } }) |shape| {
        const count = shape[0] * shape[1] * kh[0];
        const down = try testRandWeightBf16(prng.random(), &.{ count, kh[1] }, s);
        defer _ = mlx.mlx_array_free(down);
        const scores = try testRandUniformBf16(prng.random(), &.{ shape[0], shape[1], kh[0] }, 0.0, 1.0, s);
        defer _ = mlx.mlx_array_free(scores);
        const perm = try testing.allocator.alloc(u32, @intCast(count));
        defer testing.allocator.free(perm);
        for (perm, 0..) |*id, i| id.* = @intCast(i);
        prng.random().shuffle(u32, perm);
        const inverse = mlx.mlx_array_new_data(perm.ptr, &.{count}, 1, .uint32);
        defer _ = mlx.mlx_array_free(inverse);
        const expected = try expertReduceReference(s, down, inverse, scores);
        defer _ = mlx.mlx_array_free(expected);
        const actual = (try moePrefillExpertReduce(s, down, inverse, scores)) orelse return error.ExpertReduceDeclined;
        defer _ = mlx.mlx_array_free(actual);
        try testing.expect(try bf16BitsEqual(expected, actual));
    };
}

test "prefill expert reduction µbench (MLX_SERVE_MOE_WSUM_UBENCH=1)" {
    // Bar: chained marginal GPU ms per call, stock take+multiply+sum vs the fused kernel, at an 8192-token chunk.
    if (!env.on(.MLX_SERVE_MOE_WSUM_UBENCH) or mlx.noGpuBackend()) return error.SkipZigTest;
    const io_util = @import("../../util/io_util.zig");
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0xBE7C);
    const tokens: c_int = 8192;
    const count = tokens * 10;
    const down = try testRandWeightBf16(prng.random(), &.{ count, 2560 }, s);
    defer _ = mlx.mlx_array_free(down);
    const scores = try testRandUniformBf16(prng.random(), &.{ 1, tokens, 10 }, 0.0, 1.0, s);
    defer _ = mlx.mlx_array_free(scores);
    const perm = try testing.allocator.alloc(u32, @intCast(count));
    defer testing.allocator.free(perm);
    for (perm, 0..) |*id, i| id.* = @intCast(i);
    prng.random().shuffle(u32, perm);
    const inverse = mlx.mlx_array_new_data(perm.ptr, &.{count}, 1, .uint32);
    defer _ = mlx.mlx_array_free(inverse);
    for ([_]mlx.mlx_array{ down, scores, inverse }) |a| try mlx.check(mlx.mlx_array_eval(a));
    var sw = io_util.Stopwatch.init(testing.io);
    const reps = 7;
    for ([_]bool{ false, true }) |fused| {
        var best = [2]u64{ std.math.maxInt(u64), std.math.maxInt(u64) };
        for (0..reps) |_| {
            for ([_]usize{ 1, 21 }, 0..) |n, slot| {
                const outs = mlx.mlx_vector_array_new();
                defer _ = mlx.mlx_vector_array_free(outs);
                for (0..n) |_| {
                    const y = if (fused) (try moePrefillExpertReduce(s, down, inverse, scores)).? else try expertReduceReference(s, down, inverse, scores);
                    defer _ = mlx.mlx_array_free(y);
                    try mlx.check(mlx.mlx_vector_array_append_value(outs, y));
                }
                sw.reset();
                try mlx.check(mlx.mlx_eval(outs));
                best[slot] = @min(best[slot], sw.read());
            }
        }
        const one_ms = @as(f64, @floatFromInt(best[0])) / 1e6;
        const marginal_ms = (@as(f64, @floatFromInt(best[1])) - @as(f64, @floatFromInt(best[0]))) / 1e6 / 20.0;
        std.debug.print("\n[moe-wsum-ubench] {s}: single {d:.3} ms, chained marginal {d:.3} ms/call\n", .{ if (fused) "fused" else "stock", one_ms, marginal_ms });
    }
}

test "verify route packing preserves stable order and row indices" {
    if (mlx.noGpuBackend() or !verifySharedHardware()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var ids: [320]u32 = undefined;
    for ([_]c_int{ 20, 30, 40, 50, 60, 80, 120, 200, 320 }) |count| {
        for (0..3) |pattern| {
            for (ids[0..@intCast(count)], 0..) |*id, i| id.* = switch (pattern) {
                0 => @intCast((i * 37 + 11) % 17),
                1 => 7,
                else => @intCast(@as(usize, @intCast(count)) - i),
            };
            const input = mlx.mlx_array_new_data(&ids, &.{count}, 1, .uint32);
            defer _ = mlx.mlx_array_free(input);
            const cpu = mlx.mlx_default_cpu_stream_new();
            defer _ = mlx.mlx_stream_free(cpu);
            try testing.expect((try verifyRoutePack(cpu, input, 10)) == null);
            try testing.expect((try verifyRoutePack(s, input, 8)) == null);
            try testing.expect((try verifyRoutePack(s, .{}, 10)) == null);
            var got = (try verifyRoutePack(s, input, 10)) orelse return error.RoutePackDeclined;
            defer got.deinit();
            var order = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(order);
            var inverse = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(inverse);
            var sorted = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sorted);
            var lhs = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(lhs);
            try mlx.check(mlx.mlx_argsort_axis(&order, input, 0, s));
            try mlx.check(mlx.mlx_argsort_axis(&inverse, order, 0, s));
            try mlx.check(mlx.mlx_take_axis(&sorted, input, order, 0, s));
            try mlx.check(mlx.mlx_floor_divide(&lhs, order, cachedScalarInt(10), s));
            try testing.expect(try qsaArraysAllEqual(got.inverse, inverse, s));
            try testing.expect(try qsaArraysAllEqual(got.sorted, sorted, s));
            try testing.expect(try qsaArraysAllEqual(got.lhs, lhs, s));
        }
    }
}

test "prefill route packing equals the stable argsort, its inverse and the row indices" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x50A7);
    for ([_]c_int{ 33, 1000, 8192 }) |tokens| for (0..3) |pattern| {
        const count: usize = @intCast(tokens * 8);
        const ids = try testing.allocator.alloc(u32, count);
        defer testing.allocator.free(ids);
        for (ids, 0..) |*id, i| id.* = switch (pattern) {
            0 => prng.random().uintLessThan(u32, 256),
            // Skewed: a few hot experts, as real routing is.
            1 => if (prng.random().boolean()) prng.random().uintLessThan(u32, 4) else prng.random().uintLessThan(u32, 256),
            else => @intCast(255 - (i % 3)),
        };
        const input = mlx.mlx_array_new_data(ids.ptr, &.{@intCast(count)}, 1, .uint32);
        defer _ = mlx.mlx_array_free(input);
        var got = (try moePrefillRoutePack(s, input, 8, 256)) orelse return error.RoutePackDeclined;
        defer got.deinit();
        var order = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(order);
        var inverse = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(inverse);
        var sorted = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sorted);
        var lhs = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(lhs);
        try mlx.check(mlx.mlx_argsort_axis(&order, input, 0, s));
        try mlx.check(mlx.mlx_argsort_axis(&inverse, order, 0, s));
        try mlx.check(mlx.mlx_take_axis(&sorted, input, order, 0, s));
        try mlx.check(mlx.mlx_floor_divide(&lhs, order, cachedScalarInt(8), s));
        try testing.expect(try qsaArraysAllEqual(got.inverse, inverse, s));
        try testing.expect(try qsaArraysAllEqual(got.sorted, sorted, s));
        try testing.expect(try qsaArraysAllEqual(got.lhs, lhs, s));
    };
}

test "verify indexed expert inputs preserve projections without activation copies" {
    if (mlx.noGpuBackend() or !verifySharedHardware()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x1D1E);
    const weight = try attn256RandBf16(prng.random(), &.{ 64, 640, 2560 }, s);
    defer _ = mlx.mlx_array_free(weight);
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    try mlx.check(mlx.mlx_quantize(&triple, weight, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{}, s));
    var w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w);
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    var bi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bi);
    try mlx.check(mlx.mlx_vector_array_get(&w, triple, 0));
    try mlx.check(mlx.mlx_vector_array_get(&sc, triple, 1));
    try mlx.check(mlx.mlx_vector_array_get(&bi, triple, 2));
    for ([_]c_int{ 2, 3, 4, 5, 6, 8, 20 }) |tokens| {
        const count = tokens * 10;
        const x = try attn256RandBf16(prng.random(), &.{ tokens, 2560 }, s);
        defer _ = mlx.mlx_array_free(x);
        var left: [200]u32 = undefined;
        var right: [200]u32 = undefined;
        for (left[0..@intCast(count)], right[0..@intCast(count)], 0..) |*l, *r, i| {
            l.* = @intCast(i / 10);
            r.* = @intCast(i / @as(usize, @intCast(@divTrunc(count + 63, 64))));
        }
        prng.random().shuffle(u32, left[0..@intCast(count)]);
        const lhs = mlx.mlx_array_new_data(&left, &.{count}, 1, .uint32);
        defer _ = mlx.mlx_array_free(lhs);
        const rhs = mlx.mlx_array_new_data(&right, &.{count}, 1, .uint32);
        defer _ = mlx.mlx_array_free(rhs);
        const start_copy = mlx.op_count.load(.monotonic);
        var copied = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(copied);
        try mlx.check(mlx.mlx_take_axis(&copied, x, lhs, 0, s));
        var expanded = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(expanded);
        try mlx.check(mlx.mlx_reshape(&expanded, copied, &.{ count, 1, 2560 }, 3, s));
        const copy_ops = mlx.op_count.load(.monotonic) - start_copy;
        const start_view = mlx.op_count.load(.monotonic);
        const view = (try verifyIndexedExpertInput(s, x, lhs, 64)) orelse return error.IndexedExpertInputDeclined;
        defer _ = mlx.mlx_array_free(view);
        const view_ops = mlx.op_count.load(.monotonic) - start_view;
        try testing.expectEqualSlices(c_int, &.{ tokens, 1, 2560 }, mlx.getShape(view));
        try testing.expect(view_ops < copy_ops);
        var expected = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(expected);
        try gatherExpertMm(&expected, expanded, w, sc, bi, .{}, rhs, 4, 64, .affine, true, s);
        var actual = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(actual);
        try gatherExpertMm(&actual, view, w, sc, bi, lhs, rhs, 4, 64, .affine, true, s);
        try testing.expect(try qsaArraysAllEqual(expected, actual, s));
        // The sorted-matrix boundary is four assignments per expert. Ten
        // experts crosses it at width 4, but remains eligible at widths 2–3.
        const boundary_experts = @divTrunc(count, 4);
        try testing.expect((try verifyIndexedExpertInput(s, x, lhs, boundary_experts)) == null);
        const sparse_view = (try verifyIndexedExpertInput(s, x, lhs, boundary_experts + 1)) orelse return error.IndexedExpertInputDeclined;
        defer _ = mlx.mlx_array_free(sparse_view);
        try testing.expect((try verifyIndexedExpertInput(s, x, lhs, 0)) == null);
        if (tokens == 4) {
            const cpu = mlx.mlx_default_cpu_stream_new();
            defer _ = mlx.mlx_stream_free(cpu);
            try testing.expect((try verifyIndexedExpertInput(cpu, x, lhs, 64)) == null);
            var x32 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x32);
            try mlx.check(mlx.mlx_astype(&x32, x, .float32, s));
            try testing.expect((try verifyIndexedExpertInput(s, x32, lhs, 64)) == null);
            var signed = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(signed);
            try mlx.check(mlx.mlx_astype(&signed, lhs, .int32, s));
            try testing.expect((try verifyIndexedExpertInput(s, x, signed, 64)) == null);
            try testing.expect((try verifyIndexedExpertInput(s, view, lhs, 64)) == null);
        }
    }
}
