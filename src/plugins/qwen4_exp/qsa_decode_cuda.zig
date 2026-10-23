//! QSA attention for a decode token or the rows of a verify block over dense bf16 KV on MLX's CUDA
//! backend, the CUDA twin of `qsa_decode.zig`: each row's block picks go straight to a split pass and
//! a merge pass, with no index arithmetic, gather, mask or SDPA between them. A warp owns one q head and `CH`
//! of the visible keys, eight keys at a time with the head dim over the lanes; scores, softmax and
//! the value sum stay in f32. Not MLX's summation order: held to no more error than the masked SDPA
//! against an f64 truth.
const std = @import("std");
const kernelSource = @import("../../mlx/kernels.zig").source;
const mlx = @import("../../mlx/ffi.zig");

const CH: c_int = 32; // visible keys per warp, a multiple of 8 and of RATIO
const BD: c_int = 256;
/// Verify widths the row grid serves (the dense mask arm takes wider blocks).
pub const MAX_ROWS: c_int = 16;

const HEADER = kernelSource("plugins/qwen4_exp/kernels/qsa_decode_header.cu");

// Grid (NS splits, Hk, rows): a block of GQA warps, warp w the q head hk * GQA + w. Row r is the
// cache position kv - rows + r: it sees its own picks that are complete blocks before it, and its
// own incomplete tail.
const PARTS_SOURCE = kernelSource("plugins/qwen4_exp/kernels/qsa_decode_parts.cu");

// Block per (head, row), thread per dim. An empty split writes l = 0, the skip flag.
const MERGE_SOURCE = kernelSource("plugins/qwen4_exp/kernels/qsa_decode_merge.cu");

var parts_kernel: ?mlx.mlx_fast_cuda_kernel = null;
var merge_kernel: ?mlx.mlx_fast_cuda_kernel = null;

var env_enabled: ?bool = null;
pub var override: ?bool = null;

/// `MLX_SERVE_QSA_DEC_KERNEL=0` keeps the gather / mask arms (the Metal kernel's switch).
pub fn enabled() bool {
    if (override) |v| return v;
    if (env_enabled) |v| return v;
    const raw = std.c.getenv("MLX_SERVE_QSA_DEC_KERNEL");
    env_enabled = raw == null or raw.?[0] != '0';
    return env_enabled.?;
}

fn newKernel(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [:0]const u8) !mlx.mlx_fast_cuda_kernel {
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    // K/V are cache views: the kernel reads their strides.
    const k = mlx.mlx_fast_cuda_kernel_new(name, in_vec, out_vec, source, HEADER, false, 0);
    if (k.ctx == null) return error.CudaKernelCreateFailed;
    return k;
}

/// `[1, Hq, S, 256]` bf16 attention of the query rows `q`, the last S cache positions, each over
/// its own picks in `blocks` `[1, S, kb]` int32 (RATIO tokens each) plus its incomplete tail; `k`/`v`
/// are the dense cache views `[1, Hk, kv, 256]`. Null outside the kernel's geometry (the caller
/// keeps its arm).
pub fn attend(s: mlx.mlx_stream, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, blocks: mlx.mlx_array, ratio: c_int, attn_scale: f32) !?mlx.mlx_array {
    if (!enabled() or !mlx.cudaAvailable() or q.ctx == null or k.ctx == null or v.ctx == null or blocks.ctx == null) return null;
    if (mlx.mlx_array_ndim(q) != 4 or mlx.mlx_array_ndim(k) != 4 or mlx.mlx_array_ndim(blocks) != 3) return null;
    if (mlx.mlx_array_dtype(q) != .bfloat16 or mlx.mlx_array_dtype(k) != .bfloat16 or mlx.mlx_array_dtype(v) != .bfloat16) return null;
    if (mlx.mlx_array_dtype(blocks) != .int32) return null;
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const bs = mlx.getShape(blocks);
    if (!std.mem.eql(c_int, ks, mlx.getShape(v))) return null;
    const rows = qs[2];
    if (qs[0] != 1 or rows < 1 or rows > MAX_ROWS or qs[3] != BD or ks[0] != 1 or ks[3] != BD or bs[0] != 1 or bs[1] != rows or ks[2] < rows) return null;
    const hq = qs[1];
    const hk = ks[1];
    if (hk <= 0 or ratio <= 0 or bs[2] < 1 or @rem(hq, hk) != 0 or @divTrunc(hq, hk) > 32 or @rem(CH, ratio) != 0) return null;
    if (mlx.mlx_array_strides(q)[3] != 1 or mlx.mlx_array_strides(k)[3] != 1 or mlx.mlx_array_strides(v)[3] != 1 or mlx.mlx_array_strides(blocks)[2] != 1) return null;
    const gqa = @divExact(hq, hk);
    const kb = bs[2];

    if (parts_kernel == null) {
        parts_kernel = try newKernel("msv_qsa_dec_parts", &.{ "q", "scl", "blocks", "k", "v" }, &.{ "pacc", "pml" }, PARTS_SOURCE);
        merge_kernel = try newKernel("msv_qsa_dec_merge", &.{ "pacc", "pml" }, &.{"out"}, MERGE_SOURCE);
    }
    const ns = @divTrunc(kb * ratio + ratio - 1 + CH - 1, CH);
    const pc = mlx.mlx_fast_cuda_kernel_config_new();
    defer mlx.mlx_fast_cuda_kernel_config_free(pc);
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_output_arg(pc, &[_]c_int{ rows * hq, ns, BD }, 3, .float32));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_output_arg(pc, &[_]c_int{ rows * hq, ns, 2 }, 3, .float32));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_set_grid(pc, ns * gqa * 32, hk, rows));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_set_thread_group(pc, gqa * 32, 1, 1));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_dtype(pc, "T", .bfloat16));
    inline for (.{ .{ "GQA", gqa }, .{ "KB", kb }, .{ "RATIO", ratio }, .{ "CH", CH }, .{ "NS", ns } }) |a|
        try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_int(pc, a[0], a[1]));

    const scl_data = [_]f32{attn_scale};
    const scl = mlx.mlx_array_new_data(&scl_data, &[_]c_int{1}, 1, .float32);
    defer _ = mlx.mlx_array_free(scl);
    const ins = [_]mlx.mlx_array{ q, scl, blocks, k, v };
    const in_vec = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts);
    try mlx.check(mlx.mlx_fast_cuda_kernel_apply(&parts, parts_kernel.?, in_vec, pc, s));
    var pacc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pacc);
    var pml = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pml);
    try mlx.check(mlx.mlx_vector_array_get(&pacc, parts, 0));
    try mlx.check(mlx.mlx_vector_array_get(&pml, parts, 1));

    const mc = mlx.mlx_fast_cuda_kernel_config_new();
    defer mlx.mlx_fast_cuda_kernel_config_free(mc);
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_output_arg(mc, &[_]c_int{ 1, hq, rows, BD }, 4, .bfloat16));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_set_grid(mc, BD * hq, rows, 1));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_set_thread_group(mc, BD, 1, 1));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_dtype(mc, "T", .bfloat16));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_int(mc, "NS", ns));
    const merge_in = [_]mlx.mlx_array{ pacc, pml };
    const merge_vec = mlx.mlx_vector_array_new_data(&merge_in, merge_in.len);
    defer _ = mlx.mlx_vector_array_free(merge_vec);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_cuda_kernel_apply(&o, merge_kernel.?, merge_vec, mc, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, o, 0));
    return out;
}

const testing = std.testing;

fn randomBf16(s: mlx.mlx_stream, shape: []const c_int, seed: u64) !mlx.mlx_array {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed));
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_random_normal(&f, shape.ptr, shape.len, .float32, 0.0, 1.0, key, s));
    var b = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(b);
    try mlx.check(mlx.mlx_astype(&b, f, .bfloat16, s));
    return b;
}

fn hostF32(s: mlx.mlx_stream, a: mlx.mlx_array) !struct { arr: mlx.mlx_array, data: []const f32 } {
    var f = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, a, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f));
    const p = mlx.mlx_array_data_float32(f) orelse return error.Unreadable;
    return .{ .arr = f, .data = p[0..mlx.mlx_array_size(f)] };
}

test "qsa decode cuda: the one-token kernel is no worse than the gathered SDPA against the f64 truth" {
    if (!mlx.cudaAvailable()) return error.SkipZigTest;
    const xfm = @import("../../engines/mlx/transformer.zig");
    defer override = null;
    const s = mlx.gpuStream();
    const hq: usize = 24;
    const hk: usize = 2;
    const gqa = hq / hk;
    const kb: usize = 512;
    const ratio: usize = 4;
    const cap: usize = 12288;
    const scale: f32 = 1.0 / 16.0;
    var prng = std.Random.DefaultPrng.init(0xDEC0DE);
    const rnd = prng.random();
    for ([_]usize{ 2052, 2055, 4099, 8076, 11000 }, 0..) |kv, case| {
        const q = try randomBf16(s, &[_]c_int{ 1, @intCast(hq), 1, BD }, 100 + case);
        defer _ = mlx.mlx_array_free(q);
        const kfull = try randomBf16(s, &[_]c_int{ 1, @intCast(hk), @intCast(cap), BD }, 200 + case);
        defer _ = mlx.mlx_array_free(kfull);
        const vfull = try randomBf16(s, &[_]c_int{ 1, @intCast(hk), @intCast(cap), BD }, 300 + case);
        defer _ = mlx.mlx_array_free(vfull);
        // The cache views a decode step sees: a leading slice of a larger buffer.
        var kview = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(kview);
        var vview = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(vview);
        const lo = [_]c_int{ 0, 0, 0, 0 };
        const hi = [_]c_int{ 1, @intCast(hk), @intCast(kv), BD };
        const st = [_]c_int{ 1, 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&kview, kfull, &lo, 4, &hi, 4, &st, 4, s));
        try mlx.check(mlx.mlx_slice(&vview, vfull, &lo, 4, &hi, 4, &st, 4, s));

        const nb = kv / ratio;
        const pool = try testing.allocator.alloc(i32, nb);
        defer testing.allocator.free(pool);
        for (pool, 0..) |*p, i| p.* = @intCast(i);
        rnd.shuffle(i32, pool);
        const sel = try testing.allocator.dupe(i32, pool[0..kb]);
        defer testing.allocator.free(sel);
        std.mem.sort(i32, sel, {}, std.sort.asc(i32));
        const blocks = mlx.mlx_array_new_data(sel.ptr, &[_]c_int{ 1, 1, @intCast(kb) }, 3, .int32);
        defer _ = mlx.mlx_array_free(blocks);

        // f64 truth over the same bf16 values.
        const qh = try hostF32(s, q);
        defer _ = mlx.mlx_array_free(qh.arr);
        const kh = try hostF32(s, kview);
        defer _ = mlx.mlx_array_free(kh.arr);
        const vh = try hostF32(s, vview);
        defer _ = mlx.mlx_array_free(vh.arr);
        var pos_list: std.ArrayList(usize) = .empty;
        defer pos_list.deinit(testing.allocator);
        for (sel) |b| for (0..ratio) |r| try pos_list.append(testing.allocator, @as(usize, @intCast(b)) * ratio + r);
        var t = nb * ratio;
        while (t < kv) : (t += 1) try pos_list.append(testing.allocator, t);
        const truth = try testing.allocator.alloc(f64, hq * BD);
        defer testing.allocator.free(truth);
        const sc = try testing.allocator.alloc(f64, pos_list.items.len);
        defer testing.allocator.free(sc);
        for (0..hq) |h| {
            const g = h / gqa;
            var mx: f64 = -std.math.inf(f64);
            for (pos_list.items, 0..) |pos, i| {
                var dot: f64 = 0;
                for (0..BD) |d| dot += @as(f64, qh.data[h * BD + d]) * @as(f64, kh.data[(g * kv + pos) * BD + d]);
                sc[i] = dot * scale;
                mx = @max(mx, sc[i]);
            }
            var z: f64 = 0;
            for (sc) |*x| {
                x.* = @exp(x.* - mx);
                z += x.*;
            }
            for (0..BD) |d| {
                var o: f64 = 0;
                for (pos_list.items, 0..) |pos, i| o += sc[i] * @as(f64, vh.data[(g * kv + pos) * BD + d]);
                truth[h * BD + d] = o / z;
            }
        }

        override = true;
        const got = (try attend(s, q, kview, vview, blocks, @intCast(ratio), scale)) orelse return error.KernelDeclined;
        defer _ = mlx.mlx_array_free(got);
        // The arm it replaces on CUDA: gather the picks and the tail, then the masked SDPA.
        var view: xfm.DenseKVView = .{ .k = kview, .v = vview, .owned = false };
        const stock = (try xfm.qsaDecodeGatherAttn(s, q, &view, blocks, @intCast(ratio), scale)) orelse return error.StockDeclined;
        defer _ = mlx.mlx_array_free(stock);
        const gh = try hostF32(s, got);
        defer _ = mlx.mlx_array_free(gh.arr);
        const sh = try hostF32(s, stock);
        defer _ = mlx.mlx_array_free(sh.arr);
        try testing.expectEqual(hq * BD, gh.data.len);
        var e_got: f64 = 0;
        var e_stock: f64 = 0;
        for (truth, 0..) |tv, i| {
            try testing.expect(std.math.isFinite(gh.data[i]));
            e_got += (@as(f64, gh.data[i]) - tv) * (@as(f64, gh.data[i]) - tv);
            e_stock += (@as(f64, sh.data[i]) - tv) * (@as(f64, sh.data[i]) - tv);
        }
        const rms_got = @sqrt(e_got / @as(f64, @floatFromInt(truth.len)));
        const rms_stock = @sqrt(e_stock / @as(f64, @floatFromInt(truth.len)));
        std.debug.print("[qsa-dec cuda parity] kv={d} rms_err kernel={e:.3} stock={e:.3}\n", .{ kv, rms_got, rms_stock });
        try testing.expect(rms_got <= rms_stock);
    }
}

test "qsa decode cuda: verify rows are no worse than the dense mask arm against the f64 truth" {
    if (!mlx.cudaAvailable()) return error.SkipZigTest;
    const xfm = @import("../../engines/mlx/transformer.zig");
    defer override = null;
    const s = mlx.gpuStream();
    const a = testing.allocator;
    const hq: usize = 24;
    const hk: usize = 2;
    const gqa = hq / hk;
    const kb: usize = 512;
    const ratio: usize = 4;
    const scale: f32 = 1.0 / 16.0;
    var prng = std.Random.DefaultPrng.init(0x5EED);
    const rnd = prng.random();
    // The last case gives every row the last row's picks: earlier rows see a block not yet complete.
    const Case = struct { kv: usize, rows: usize, shared: bool };
    for ([_]Case{ .{ .kv = 2061, .rows = 2, .shared = false }, .{ .kv = 8077, .rows = 4, .shared = false }, .{ .kv = 30001, .rows = 7, .shared = false }, .{ .kv = 4103, .rows = 7, .shared = true } }, 0..) |c, case| {
        const q = try randomBf16(s, &[_]c_int{ 1, @intCast(hq), @intCast(c.rows), BD }, 400 + case);
        defer _ = mlx.mlx_array_free(q);
        const k = try randomBf16(s, &[_]c_int{ 1, @intCast(hk), @intCast(c.kv), BD }, 500 + case);
        defer _ = mlx.mlx_array_free(k);
        const v = try randomBf16(s, &[_]c_int{ 1, @intCast(hk), @intCast(c.kv), BD }, 600 + case);
        defer _ = mlx.mlx_array_free(v);

        // Each row picks kb distinct complete blocks before it, sorted, as the indexer does.
        const sel = try a.alloc(i32, c.rows * kb);
        defer a.free(sel);
        for (0..c.rows) |r| {
            const nb_r = (c.kv - c.rows + 1 + (if (c.shared) c.rows - 1 else r)) / ratio;
            const pool = try a.alloc(i32, nb_r);
            defer a.free(pool);
            for (pool, 0..) |*p, i| p.* = @intCast(i);
            rnd.shuffle(i32, pool);
            if (c.shared) std.mem.swap(i32, &pool[0], &pool[std.mem.indexOfScalar(i32, pool, @intCast(nb_r - 1)).?]);
            const row = sel[r * kb ..][0..kb];
            @memcpy(row, pool[0..kb]);
            std.mem.sort(i32, row, {}, std.sort.asc(i32));
        }
        if (c.shared) for (0..c.rows - 1) |r| @memcpy(sel[r * kb ..][0..kb], sel[(c.rows - 1) * kb ..][0..kb]);
        const blocks = mlx.mlx_array_new_data(sel.ptr, &[_]c_int{ 1, @intCast(c.rows), @intCast(kb) }, 3, .int32);
        defer _ = mlx.mlx_array_free(blocks);

        // f64 truth with the dense mask's semantics: complete picked blocks before the row, its own tail.
        const qh = try hostF32(s, q);
        defer _ = mlx.mlx_array_free(qh.arr);
        const kh = try hostF32(s, k);
        defer _ = mlx.mlx_array_free(kh.arr);
        const vh = try hostF32(s, v);
        defer _ = mlx.mlx_array_free(vh.arr);
        const truth = try a.alloc(f64, hq * c.rows * BD);
        defer a.free(truth);
        var pos_list: std.ArrayList(usize) = .empty;
        defer pos_list.deinit(a);
        for (0..c.rows) |r| {
            const kl = c.kv - c.rows + 1 + r;
            const nb = kl / ratio;
            pos_list.clearRetainingCapacity();
            for (sel[r * kb ..][0..kb]) |b| if (@as(usize, @intCast(b)) < nb) for (0..ratio) |t| try pos_list.append(a, @as(usize, @intCast(b)) * ratio + t);
            var t = nb * ratio;
            while (t < kl) : (t += 1) try pos_list.append(a, t);
            const sc = try a.alloc(f64, pos_list.items.len);
            defer a.free(sc);
            for (0..hq) |h| {
                const g = h / gqa;
                var mx: f64 = -std.math.inf(f64);
                for (pos_list.items, 0..) |pos, i| {
                    var dot: f64 = 0;
                    for (0..BD) |d| dot += @as(f64, qh.data[(h * c.rows + r) * BD + d]) * @as(f64, kh.data[(g * c.kv + pos) * BD + d]);
                    sc[i] = dot * scale;
                    mx = @max(mx, sc[i]);
                }
                var z: f64 = 0;
                for (sc) |*x| {
                    x.* = @exp(x.* - mx);
                    z += x.*;
                }
                for (0..BD) |d| {
                    var o: f64 = 0;
                    for (pos_list.items, 0..) |pos, i| o += sc[i] * @as(f64, vh.data[(g * c.kv + pos) * BD + d]);
                    truth[(h * c.rows + r) * BD + d] = o / z;
                }
            }
        }

        override = true;
        const got = (try attend(s, q, k, v, blocks, @intCast(ratio), scale)) orelse return error.KernelDeclined;
        defer _ = mlx.mlx_array_free(got);
        // The arm it replaces on CUDA: the dense [S, kv] mask, then SDPA.
        const mask = try xfm.qsaMaskFromBlocks(s, blocks, @intCast(c.kv), @intCast(ratio));
        defer _ = mlx.mlx_array_free(mask);
        const stock = try xfm.qsaMaskArm(s, q, k, v, scale, mask);
        defer _ = mlx.mlx_array_free(stock);
        const gh = try hostF32(s, got);
        defer _ = mlx.mlx_array_free(gh.arr);
        const sh = try hostF32(s, stock);
        defer _ = mlx.mlx_array_free(sh.arr);
        try testing.expectEqual(truth.len, gh.data.len);
        var e_got: f64 = 0;
        var e_stock: f64 = 0;
        for (truth, 0..) |tv, i| {
            try testing.expect(std.math.isFinite(gh.data[i]));
            e_got += (@as(f64, gh.data[i]) - tv) * (@as(f64, gh.data[i]) - tv);
            e_stock += (@as(f64, sh.data[i]) - tv) * (@as(f64, sh.data[i]) - tv);
        }
        const rms_got = @sqrt(e_got / @as(f64, @floatFromInt(truth.len)));
        const rms_stock = @sqrt(e_stock / @as(f64, @floatFromInt(truth.len)));
        std.debug.print("[qsa-verify cuda parity] kv={d} rows={d} shared={} rms_err kernel={e:.3} stock={e:.3}\n", .{ c.kv, c.rows, c.shared, rms_got, rms_stock });
        try testing.expect(rms_got <= rms_stock);
    }
}
