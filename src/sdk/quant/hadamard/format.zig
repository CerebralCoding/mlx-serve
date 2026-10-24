//! The Hadamard format (Prism packs): affine weights stored in a rotated basis, each `<linear>.signs` read as
//! `H_block(signs * x)` before the matmul (`rht.zig`). Loads through the MLX-native path; the matmul is the engine's
//! rotated arm (`Transformer.qmatmul` with a registry bound).

const std = @import("std");
const abi = @import("../../../abi/quant.zig");
const mlx = @import("../../../mlx/ffi.zig");
const native = @import("../native.zig");
const rht = @import("rht.zig");
const qmv2 = @import("../../blocks/qmm/qmv2.zig");
const qmv_nax2 = @import("../../blocks/qmm/qmv_nax2.zig");
const qmatmulBits = @import("../../blocks/linear/qmatmul.zig").qmatmulBits;

const Str = abi.Str;
const Verdict = abi.Verdict;
const arr = native.arr;
const wrap = native.wrap;

/// `Weight.flags`: the biases are exactly -scales (Prism's codec), the 2-bit kernels' fast path.
pub const NEG_SCALE: u8 = 1;

fn loadTensor(ctx: *const abi.LoadCtx, name: Str, in_dim: u32, out: *abi.Weight) callconv(.c) Verdict {
    load(ctx, name.slice(), in_dim, out) catch |e| return native.failed(e);
    return .ok;
}

fn load(ctx: *const abi.LoadCtx, base: []const u8, in_dim: u32, out: *abi.Weight) !void {
    try native.loadInto(ctx, base, in_dim, out);
    errdefer native.release(out);
    const signs = native.tensor(ctx, base, "signs");
    if (signs.ctx == null) return;
    if (out.scales.ctx == null) return error.HadamardSignsOnDenseWeight;
    out.block = native.state(ctx).?.block;
    var h = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&h, signs));
    out.side = wrap(h);
    if (out.biases.ctx != null and try biasIsNegScale(arr(out.scales), arr(out.biases), .{ .ctx = ctx.stream.ctx })) out.flags |= NEG_SCALE;
}

/// `max |scales + biases| == 0`, as the engine's load probes each rotated weight.
fn biasIsNegScale(sc: mlx.mlx_array, bi: mlx.mlx_array, s: mlx.mlx_stream) !bool {
    var sum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sum);
    try mlx.check(mlx.mlx_add(&sum, sc, bi, s));
    var mag = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mag);
    try mlx.check(mlx.mlx_abs(&mag, sum, s));
    var peak = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(peak);
    try mlx.check(mlx.mlx_max(&peak, mag, false, s));
    var p32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(p32);
    try mlx.check(mlx.mlx_astype(&p32, peak, .float32, s));
    try mlx.check(mlx.mlx_array_eval(p32));
    var v: f32 = 1;
    try mlx.check(mlx.mlx_array_item_float32(&v, p32));
    return v == 0;
}

fn matmul(w: *const abi.Weight, x: abi.Array, s: abi.Stream, out: *abi.Array) callconv(.c) Verdict {
    out.* = wrap(rotated(w, arr(x), .{ .ctx = s.ctx }) catch |e| return native.failed(e));
    return .ok;
}

/// The engine's rotated arm: the 2-bit kernels first (each declines what it does not serve), then stock qmm.
fn rotated(w: *const abi.Weight, x: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const signs = arr(w.side);
    const xr = if (signs.ctx != null) try rht.transform(x, signs, @intCast(w.block), false, s) else x;
    defer if (signs.ctx != null) {
        _ = mlx.mlx_array_free(xr);
    };
    const W = arr(w.w);
    const sc = arr(w.scales);
    const bi = arr(w.biases);
    if (try qmv_nax2.qmm(xr, W, sc, bi, w.bits, w.group_size, s)) |y| return y;
    if (try qmv2.qmm(xr, W, sc, bi, w.bits, w.group_size, signs.ctx != null and w.flags & NEG_SCALE != 0, true, s)) |y| return y;
    return qmatmulBits(xr, W, sc, bi, w.bits, w.group_size, native.modeOf(w), s);
}

fn noBank(_: *const abi.LoadCtx, _: Str, _: u32, _: *abi.Weight) callconv(.c) Verdict {
    return Verdict.refuse(.unsupported, "hadamard: rotated expert banks");
}
fn noMoe(_: *const abi.Weight, _: abi.Array, _: abi.Array, _: abi.Array, _: bool, _: abi.Stream, _: *abi.Array) callconv(.c) Verdict {
    return Verdict.refuse(.unsupported, "hadamard: rotated expert banks");
}

pub const table: abi.QuantTable = blk: {
    var t = native.table("hadamard", &native.billStored);
    t.load_tensor = &loadTensor;
    t.load_bank = &noBank;
    t.matmul = &matmul;
    t.moe = &noMoe;
    break :blk t;
};

// ── tests ──

const transformer = @import("../../../engines/mlx/transformer.zig");

const testing = std.testing;
const ops = @import("../../../mlx/ops.zig");
const parity = @import("../../testing/parity.zig");
const resident = @import("../../weights/resident.zig");
const registerRhtSigns = @import("../hadamard.zig").registerRhtSigns;

fn f16Rand(rnd: std.Random, shape: []const c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const b = try parity.testRandWeightBf16(rnd, shape, s);
    defer _ = mlx.mlx_array_free(b);
    var h = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&h, b, .float16, s));
    return h;
}

fn signsOf(rnd: std.Random, k: usize) !mlx.mlx_array {
    const v = try testing.allocator.alloc(f16, k);
    defer testing.allocator.free(v);
    for (v) |*e| e.* = if (rnd.boolean()) 1 else -1;
    return mlx.mlx_array_new_data(v.ptr, &[_]c_int{@intCast(k)}, 1, .float16);
}

// Bar: a Prism pack (f16, every shipped rotation block, 2- and 4-bit, biases at -scales or not) loads through the
// format and multiplies bit-equal to the engine's rotated arm (`Transformer.qmatmul` with the pack's registry).
test "quant hadamard: rotated weights multiply bit-equal to the engine's rotated arm" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const N: c_int = 256;
    const K: c_int = 4096;
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    for ([_]u32{ 512, 1024, 2048, 4096 }) |blk| for ([_]u8{ 2, 4 }) |bits| for ([_]bool{ true, false }) |neg| {
        var weights = resident.Weights.init(testing.allocator);
        const raw = try f16Rand(rnd, &.{ N, K }, s);
        const t = try native.quantized(raw, "affine", bits, 128, s);
        _ = mlx.mlx_array_free(raw);
        var bi = t.bi;
        if (neg) {
            bi = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_negative(&bi, t.sc, s));
            _ = mlx.mlx_array_free(t.bi);
        }
        try native.putTriple(&weights, "l.proj", .{ .w = t.w, .sc = t.sc, .bi = bi });
        try weights.put("l.proj.signs", try signsOf(rnd, K));
        const raw_plain = try f16Rand(rnd, &.{ N, K }, s);
        defer _ = mlx.mlx_array_free(raw_plain);
        try native.putTriple(&weights, "l.plain", try native.quantized(raw_plain, "affine", bits, 128, s));
        var r = resident.Resident.adopt(weights);
        defer r.deinit();

        var reg = rht.Registry.init(testing.allocator, blk);
        defer reg.deinit();
        try registerRhtSigns(&reg, &r.weights, s);
        var xfm: transformer.Transformer = undefined;
        xfm.s = s;
        xfm.rht = &reg;
        xfm.ternary_2bit = false;
        xfm.int8_cache = null;
        xfm.bits_cache = .{};
        xfm.config = .{};
        xfm.config.quant_bits = bits;
        xfm.config.quant_group_size = 128;

        var cfg_buf: [96]u8 = undefined;
        const config = try std.fmt.bufPrint(&cfg_buf, "{{\"modules\":[{{\"path\":\"l.proj\",\"block\":{d}}}]}}", .{blk});
        var bj: [64]u8 = undefined;
        const block_json = try std.fmt.bufPrint(&bj, "{{\"bits\":{d},\"group_size\":128}}", .{bits});
        try testing.expectEqual(abi.Claim.by_marker, table.claims(.of(block_json), .of(config)));
        const src = r.source();
        var ctx: abi.LoadCtx = .{ .block_json = .of(block_json), .config_json = .of(config), .source = &src, .stream = .{ .ctx = s.ctx } };
        try testing.expect(table.open(&ctx).isOk());
        defer table.close(&ctx);
        for ([_][]const u8{ "l.proj", "l.plain" }) |base| {
            var w: abi.Weight = .{};
            try testing.expect(table.load_tensor(&ctx, .of(base), @intCast(K), &w).isOk());
            defer table.release(&w);
            const rotated_w = std.mem.eql(u8, base, "l.proj");
            try testing.expectEqual(rotated_w, w.side.ctx != null);
            try testing.expectEqual(rotated_w and neg, w.flags & NEG_SCALE != 0);
            for ([_]c_int{ 1, 3, 9 }) |rows| {
                const x = try f16Rand(rnd, &.{ 1, rows, K }, s);
                defer _ = mlx.mlx_array_free(x);
                var got: abi.Array = .{};
                try testing.expect(table.matmul(&w, wrap(x), .{ .ctx = s.ctx }, &got).isOk());
                defer _ = mlx.mlx_array_free(arr(got));
                reg.dropMemo();
                const want = try xfm.qmatmul(x, native.tensor(&ctx, base, "weight"), native.tensor(&ctx, base, "scales"), native.tensor(&ctx, base, "biases"));
                defer _ = mlx.mlx_array_free(want);
                try ops.expectBitEqual(arr(got), want);
            }
        }
    };
}

test "quant hadamard: expert banks are refused by name" {
    var w: abi.Weight = .{};
    const src: abi.Source = undefined;
    const ctx: abi.LoadCtx = .{ .block_json = .of("{}"), .source = &src };
    try testing.expectEqual(Verdict.Code.unsupported, table.load_bank(&ctx, .of("x"), 0, &w).code);
}
