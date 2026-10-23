//! `fp8_block128`: a release's FP8 projections as shipped. A projection is e4m3fn codes, U8 `[out, in]` `.weight`, with
//! one f32 scale per 64 output rows x 128 input columns, F32 `[ceil(out / 64), ceil(in / 128)]` `.weight_scale_inv`;
//! value = code x scale (MiMo-V2.6-Flash's fp8res pack, `{"mode":"fp8_e4m3","bits":8,"block":[64,128]}` per site). The
//! read is the stream's prompt route (`mimo_v2/stream/fp8_linear.zig`): the weight as bf16(code x scale), one f32
//! product and one BF16 rounding, under MLX's matmul. Not a built-in yet (`formats.tables`): no pack loads through it.

const std = @import("std");
const abi = @import("../../abi/quant.zig");
const mlx = @import("../../mlx/ffi.zig");
const native = @import("native.zig");

const Str = abi.Str;
const Verdict = abi.Verdict;

pub const block_rows: u32 = 64;
pub const block_cols: u32 = 128;

/// The block names this format: e4m3 codes, 8 bits, the 64 x 128 scale grid.
fn ownBlock(block_json: []const u8) bool {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const v = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), block_json, .{}) catch return false;
    if (v != .object) return false;
    const mode = v.object.get("mode") orelse return false;
    const bits = v.object.get("bits") orelse return false;
    const blk = v.object.get("block") orelse return false;
    if (mode != .string or !std.mem.eql(u8, mode.string, "fp8_e4m3") or bits != .integer or bits.integer != 8) return false;
    if (blk != .array or blk.array.items.len != 2) return false;
    const r = blk.array.items[0];
    const c = blk.array.items[1];
    return r == .integer and r.integer == block_rows and c == .integer and c.integer == block_cols;
}

fn claims(block_json: Str, _: Str) callconv(.c) abi.Claim {
    return if (ownBlock(block_json.slice())) .by_type else .no;
}

fn validate(block_json: Str, _: Str, _: *const abi.PackDir) callconv(.c) Verdict {
    return if (ownBlock(block_json.slice())) .ok else Verdict.refuse(.unsupported, "fp8_block128: a block other than e4m3, 8 bits, [64, 128]");
}

fn open(_: *abi.LoadCtx) callconv(.c) Verdict {
    return .ok;
}

fn close(_: *abi.LoadCtx) callconv(.c) void {}

/// `<name>.weight` U8 `[out, in]` and `<name>.weight_scale_inv` F32 on the 64 x 128 grid, shared with the source.
fn loadInto(ctx: *const abi.LoadCtx, name: []const u8, out: *abi.Weight) !void {
    const w = native.tensor(ctx, name, "weight");
    const sc = native.tensor(ctx, name, "weight_scale_inv");
    if (w.ctx == null or sc.ctx == null) return error.MissingWeight;
    const ws = mlx.getShape(w);
    const ss = mlx.getShape(sc);
    if (ws.len != 2 or ss.len != 2 or mlx.mlx_array_dtype(w) != .uint8 or mlx.mlx_array_dtype(sc) != .float32) return error.Fp8Block128Layout;
    const rows: u32 = @intCast(ws[0]);
    const cols: u32 = @intCast(ws[1]);
    if (ss[0] != (rows + block_rows - 1) / block_rows or ss[1] != (cols + block_cols - 1) / block_cols) return error.Fp8Block128ScaleGrid;
    out.* = .{ .bits = 8, .group_size = block_cols, .block = block_rows };
    errdefer native.release(out);
    var hw = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&hw, w));
    out.w = native.wrap(hw);
    var hs = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&hs, sc));
    out.scales = native.wrap(hs);
}

fn loadTensor(ctx: *const abi.LoadCtx, name: Str, _: u32, out: *abi.Weight) callconv(.c) Verdict {
    loadInto(ctx, name.slice(), out) catch |e| return native.failed(e);
    return .ok;
}

fn loadBank(_: *const abi.LoadCtx, _: Str, _: u32, _: *abi.Weight) callconv(.c) Verdict {
    return Verdict.refuse(.unsupported, "fp8_block128: expert banks");
}

/// The weight as bf16(code x scale), `[out, in]`: e4m3 exactly in f32, times its tile's scale (one f32 rounding), one
/// BF16 rounding.
pub fn dequantize(codes: mlx.mlx_array, scales: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const sh = mlx.getShape(codes);
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_from_fp8(&f, codes, .float32, s));
    var r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r);
    try mlx.check(mlx.mlx_repeat_axis(&r, scales, block_rows, 0, s));
    var rc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rc);
    try mlx.check(mlx.mlx_repeat_axis(&rc, r, block_cols, 1, s));
    var grid = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(grid);
    try mlx.check(mlx.mlx_slice(&grid, rc, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ sh[0], sh[1] }, 2, &[_]c_int{ 1, 1 }, 2, s));
    var p = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(p);
    try mlx.check(mlx.mlx_multiply(&p, f, grid, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, p, .bfloat16, s));
    return out;
}

/// `x @ W^T` over the bf16 weight (the stream's prompt arithmetic at every width; its decode GEMV, which skips the
/// weight's BF16 rounding, stays the stream's).
fn matmul(w: *const abi.Weight, x: abi.Array, s: abi.Stream, out: *abi.Array) callconv(.c) Verdict {
    const y = matmulInto(w, native.arr(x), .{ .ctx = s.ctx }) catch |e| return native.failed(e);
    out.* = native.wrap(y);
    return .ok;
}

fn matmulInto(w: *const abi.Weight, x: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const wb = try dequantize(native.arr(w.w), native.arr(w.scales), s);
    defer _ = mlx.mlx_array_free(wb);
    var wt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wt);
    try mlx.check(mlx.mlx_transpose(&wt, wb, s));
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_matmul(&y, x, wt, s));
    return y;
}

fn moe(_: *const abi.Weight, _: abi.Array, _: abi.Array, _: abi.Array, _: bool, _: abi.Stream, _: *abi.Array) callconv(.c) Verdict {
    return Verdict.refuse(.unsupported, "fp8_block128: expert banks");
}

pub const table: abi.QuantTable = .{
    .name = Str.of("fp8_block128"),
    .claims = &claims,
    .validate = &validate,
    .bill = &native.billStored,
    .open = &open,
    .close = &close,
    .load_tensor = &loadTensor,
    .load_bank = &loadBank,
    .release = &native.release,
    .matmul = &matmul,
    .moe = &moe,
};

// ── tests ──

const testing = std.testing;
const build_options = @import("build_options");

test "fp8_block128: claims and validates its own block only" {
    const own = "{\"mode\":\"fp8_e4m3\",\"bits\":8,\"block\":[64,128]}";
    try testing.expectEqual(abi.Claim.by_type, claims(.of(own), .of("{}")));
    try testing.expectEqual(abi.Claim.no, claims(.of("{\"mode\":\"fp8_e4m3\",\"bits\":8,\"block\":[128,128]}"), .of("{}")));
    try testing.expectEqual(abi.Claim.no, claims(.of("{\"quant_method\":\"fp8\",\"weight_block_size\":[128,128]}"), .of("{}")));
    try testing.expectEqual(abi.Claim.no, claims(.of("{}"), .of("{}")));
}

// Bar: the read is the MiMo stream's fp8 prompt route bit for bit (its dequant kernel's bf16 weight, and its matmul).
test "fp8_block128: the weight and the product are the MiMo stream's fp8 prompt route, bit for bit" {
    if (!build_options.mlx_stream or mlx.noGpuBackend()) return error.SkipZigTest;
    const fp8 = @import("mlx_stream").mimo.fp8_linear;
    const a = testing.allocator;
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    // 160 rows (2.5 row blocks: ragged) x 384 columns (the kernel takes whole column blocks, its scale stride is K / 128);
    // 16 input rows (the prompt route).
    const n = 160;
    const k = 384;
    const m = 16;
    var rnd = std.Random.DefaultPrng.init(128);
    const codes = try a.alloc(u8, n * k);
    defer a.free(codes);
    for (codes) |*c| c.* = while (true) {
        const v = rnd.random().int(u8);
        if (v & 0x7f != 0x7f) break v;
    };
    var scales: [3 * 3]f32 = undefined;
    for (&scales) |*v| v.* = std.math.ldexp(1.0 + rnd.random().float(f32), -6 - rnd.random().intRangeAtMost(i32, 0, 3));
    var xs: [m * k]f32 = undefined;
    for (&xs) |*v| v.* = rnd.random().floatNorm(f32);
    const w = mlx.mlx_array_new_data(codes.ptr, &[_]c_int{ n, k }, 2, .uint8);
    defer _ = mlx.mlx_array_free(w);
    const sc = mlx.mlx_array_new_data(&scales, &[_]c_int{ 3, 3 }, 2, .float32);
    defer _ = mlx.mlx_array_free(sc);
    const xf = mlx.mlx_array_new_data(&xs, &[_]c_int{ m, k }, 2, .float32);
    defer _ = mlx.mlx_array_free(xf);
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_astype(&x, xf, .bfloat16, s));

    var kernels = try fp8.Kernels.init();
    defer kernels.deinit();
    var cache: fp8.PromptCache = .{};
    kernels.prompt = &cache;
    defer kernels.prompt = null;
    defer cache.release();
    const want_y = try fp8.promptRun(&kernels, n, k, x, w, sc, s);
    defer _ = mlx.mlx_array_free(want_y);
    const want_w = cache.arrs[0];

    var wt: abi.Weight = .{ .w = native.wrap(w), .scales = native.wrap(sc), .bits = 8, .group_size = block_cols, .block = block_rows };
    _ = &wt;
    const got_w = try dequantize(w, sc, s);
    defer _ = mlx.mlx_array_free(got_w);
    const got_y = try matmulInto(&wt, x, s);
    defer _ = mlx.mlx_array_free(got_y);
    for ([_][2]mlx.mlx_array{ .{ got_w, want_w }, .{ got_y, want_y } }) |pair| {
        try mlx.check(mlx.mlx_array_eval(pair[0]));
        try mlx.check(mlx.mlx_array_eval(pair[1]));
        try testing.expectEqual(mlx.mlx_array_dtype(pair[1]), mlx.mlx_array_dtype(pair[0]));
        const len = mlx.mlx_array_size(pair[1]);
        try testing.expectEqual(len, mlx.mlx_array_size(pair[0]));
        const g = mlx.mlx_array_data_bfloat16(pair[0]) orelse return error.NoData;
        const h = mlx.mlx_array_data_bfloat16(pair[1]) orelse return error.NoData;
        try testing.expectEqualSlices(u16, h[0..len], g[0..len]);
    }
}

test "fp8_block128: ragged tiles on both axes read their own block's scale (MiMo's host reference)" {
    if (!build_options.mlx_stream or mlx.noGpuBackend()) return error.SkipZigTest;
    const fp8 = @import("mlx_stream").mimo.fp8_linear;
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    const n = 70;
    const k = 200;
    var rnd = std.Random.DefaultPrng.init(64);
    var codes: [n * k]u8 = undefined;
    for (&codes) |*c| c.* = rnd.random().int(u8) & 0xf7;
    var scales: [2 * 2]f32 = .{ 0.5, 0.25, 0.125, 0.0625 };
    _ = &scales;
    const w = mlx.mlx_array_new_data(&codes, &[_]c_int{ n, k }, 2, .uint8);
    defer _ = mlx.mlx_array_free(w);
    const sc = mlx.mlx_array_new_data(&scales, &[_]c_int{ 2, 2 }, 2, .float32);
    defer _ = mlx.mlx_array_free(sc);
    const got = try dequantize(w, sc, s);
    defer _ = mlx.mlx_array_free(got);
    try mlx.check(mlx.mlx_array_eval(got));
    const g = mlx.mlx_array_data_bfloat16(got) orelse return error.NoData;
    for (0..n) |r| for (0..k) |c| {
        const want = fp8.f32ToBf16(fp8.weight32(codes[r * k + c], scales[(r / block_rows) * 2 + c / block_cols]));
        try testing.expectEqual(want, g[r * k + c]);
    };
}
