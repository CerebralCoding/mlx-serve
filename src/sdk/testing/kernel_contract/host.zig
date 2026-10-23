//! Host values for registrations: inputs drawn and rounded to their dtype, the non-finite pattern gate 1 plants,
//! float64 sums, a device array read back as float64, and the forward-error allowance a reference carries when a
//! kernel rounds intermediates to bf16/f16 after fast-math arithmetic (`Val`, `allowance`).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const spec = @import("spec.zig");

/// The output element a float dtype writes as (f32 for anything else).
pub fn elemOf(dt: mlx.mlx_dtype) spec.Elem {
    return switch (dt) {
        .float16 => .f16,
        .bfloat16 => .bf16,
        else => .f32,
    };
}

/// `v` as `dt` stores it: float dtypes round, integer ones keep the value.
pub fn rt(dt: mlx.mlx_dtype, v: f64) f64 {
    return switch (dt) {
        .float32, .float16, .bfloat16 => spec.roundTo(elemOf(dt), v),
        else => v,
    };
}

/// `n` values uniform in [lo, hi), rounded to `dt`.
pub fn uniform(a: std.mem.Allocator, rng: std.Random, n: usize, lo: f64, hi: f64, dt: mlx.mlx_dtype) ![]f64 {
    const out = try a.alloc(f64, n);
    for (out) |*v| v.* = rt(dt, lo + (hi - lo) * rng.float(f64));
    return out;
}

/// `n` integers uniform in [lo, hi] (exact in every float dtype for small ranges).
pub fn ints(a: std.mem.Allocator, rng: std.Random, n: usize, lo: i32, hi: i32) ![]f64 {
    const out = try a.alloc(f64, n);
    for (out) |*v| v.* = @floatFromInt(rng.intRangeAtMost(i32, lo, hi));
    return out;
}

/// Gate 1's pattern over `rows` rows of `width`: NaN in row 0, +Inf in row 1, -Inf in row 2, both infinities in
/// row 3 (rows past the buffer are skipped).
pub fn plant(x: []f64, rows: usize, width: usize) void {
    const nan = std.math.nan(f64);
    const inf = std.math.inf(f64);
    const at = [_][2]usize{ .{ 0, width / 3 }, .{ 1, width / 2 }, .{ 2, width - 1 }, .{ 3, 0 }, .{ 3, width - 1 } };
    const put = [_]f64{ nan, inf, -inf, inf, -inf };
    for (at, put) |p, v| if (p[0] < rows) {
        x[p[0] * width + p[1]] = v;
    };
}

/// A float64 value and the most the device's value can differ from it (forward error, first order).
pub const Val = struct {
    v: f64,
    e: f64 = 0,

    pub fn exact(v: f64) Val {
        return .{ .v = v };
    }
};

/// f32 unit roundoff.
pub const U = 0x1p-24;

/// `x` rounded to `dt` where the device holds it within `x.e` of `x.v` before it rounds: the value float64 rounds to,
/// and the farthest any value in that interval rounds to from it (0 unless the interval straddles a rounding point).
pub fn round(dt: mlx.mlx_dtype, x: Val) Val {
    const r = rt(dt, x.v);
    if (x.e == 0 or !std.math.isFinite(x.v)) return .{ .v = r };
    return .{ .v = r, .e = @max(@abs(rt(dt, x.v + x.e) - r), @abs(rt(dt, x.v - x.e) - r)) };
}

/// Metal's fast `exp(x)` error relative to its value: 3 + floor(2|x|) f32 ulps (the MSL fast-math table).
pub fn fastExp(x: f64) f64 {
    return (3 + @floor(2 * @abs(x))) * 0x1p-23;
}

/// 1 / (1 + exp(-x)) as Metal computes it in f32 (fast exp, a fast divide), unrounded; `x` carries its own error.
pub fn sigmoid(x: Val) Val {
    const e = @exp(-x.v);
    const sig = 1 / (1 + e);
    if (!std.math.isFinite(x.v)) return .{ .v = sig };
    return .{ .v = sig, .e = sig * (1 - sig) * x.e + sig * (fastExp(x.v) * e * sig + U + 2.5 * 0x1p-23) };
}

/// x / (1 + exp(-x)) as Metal computes it in f32 (fast exp, a fast divide), unrounded; `x` carries its own error.
pub fn silu(x: Val) Val {
    const e = @exp(-x.v);
    const sig = 1 / (1 + e);
    const v = x.v * sig;
    if (!std.math.isFinite(x.v)) return .{ .v = v };
    return .{ .v = v, .e = @abs(sig * (1 + x.v * (1 - sig))) * x.e + @abs(v) * (fastExp(x.v) * e * sig + U + 2.5 * 0x1p-23) };
}

/// One ulp of `dt` relative to the value, at its widest.
pub fn ulpOf(dt: mlx.mlx_dtype) f64 {
    return switch (dt) {
        .bfloat16 => 0x1p-7,
        .float16 => 0x1p-10,
        else => 0x1p-23,
    };
}

/// `x` as the device may hold it: rounded to `dt` or not (an intermediate the compiler may keep in f32).
pub fn maybe(dt: mlx.mlx_dtype, x: Val) Val {
    const r = round(dt, x);
    return .{ .v = r.v, .e = @max(r.e, x.e + @abs(x.v - r.v)) };
}

/// a * b before any rounding.
pub fn mul(a: Val, b: Val) Val {
    return .{ .v = a.v * b.v, .e = @abs(a.v) * b.e + @abs(b.v) * a.e + a.e * b.e };
}

/// MLX's unary sigmoid in `dt`: sy = 1 / (1 + exp(|x|)), x < 0 ? sy : 1 - sy, each step in `dt` or in f32, with a
/// fast-math or a precise (4 ulps) exp.
pub fn mlxSig(dt: mlx.mlx_dtype, x: Val) Val {
    const ax = @abs(x.v);
    if (!std.math.isFinite(x.v)) {
        const sy = rt(dt, 1 / (1 + @exp(ax)));
        return .{ .v = if (x.v < 0) sy else rt(dt, 1 - sy) };
    }
    const ev = @exp(ax);
    const e = maybe(dt, .{ .v = ev, .e = ev * (std.math.expm1(x.e) + @max(fastExp(ax), 4 * 0x1p-23, ulpOf(dt))) });
    const d = maybe(dt, .{ .v = 1 + e.v, .e = e.e + U * (1 + e.v) });
    const r = 1 / d.v;
    var sy = round(dt, .{ .v = r, .e = d.e / (d.v * @max(d.v - d.e, 1)) + r * @max(2.5 * 0x1p-23, ulpOf(dt)) });
    // Within its error of 0 the device may take the other branch: the two meet at 1/2.
    const straddle = if (ax <= x.e) @abs(1 - 2 * sy.v) + sy.e else 0;
    if (x.v < 0) {
        sy.e += straddle;
        return sy;
    }
    var om = round(dt, maybe(dt, .{ .v = 1 - sy.v, .e = sy.e + U * @abs(1 - sy.v) }));
    om.e += straddle;
    return om;
}

/// The mag that makes gate 3's bound equal `allow` for a spec of depth 1 with f32 sums, no ops and no mids: a
/// reference that carries its own forward error (`Val`) registers that way.
pub fn allowance(allow: f64) f64 {
    return allow * (1 - U) / U;
}

/// Depth 1, for registrations whose reference carries the whole allowance (`allowance`).
pub fn depthOne(_: spec.Shape) usize {
    return 1;
}

pub fn sum(t: []const f64) f64 {
    var s: f64 = 0;
    for (t) |v| s += v;
    return s;
}

pub fn sumAbs(t: []const f64) f64 {
    var s: f64 = 0;
    for (t) |v| s += @abs(v);
    return s;
}

/// An affine-quantized [rows, cols] weight as MLX packs it: `bits`-bit codes little-endian in u32 words (a dense
/// bitstream per row, `mx.quantize`'s layout at every width), scales and biases per `gs` columns in `dt`, and the
/// dequantized values (code * scale + bias, float64). Bits 16: the words hold the raw `dt` values, two a word.
pub const Quant = struct { words: []f64, scales: []f64, biases: []f64, w: []f64 };

pub fn quantize(a: std.mem.Allocator, rng: std.Random, rows: usize, cols: usize, bits: u32, gs: usize, dt: mlx.mlx_dtype) !Quant {
    const per_row = cols * bits / 32;
    var q: Quant = .{
        .words = try a.alloc(f64, rows * per_row),
        .scales = try a.alloc(f64, if (bits == 16) 1 else rows * (cols / gs)),
        .biases = try a.alloc(f64, if (bits == 16) 1 else rows * (cols / gs)),
        .w = try a.alloc(f64, rows * cols),
    };
    @memset(q.words, 0);
    @memset(q.scales, 1);
    @memset(q.biases, 0);
    const top: u32 = if (bits == 16) 0xffff else (@as(u32, 1) << @intCast(bits)) - 1;
    if (bits != 16) for (q.scales, q.biases) |*sc, *bi| {
        sc.* = rt(dt, (0.5 + rng.float(f64)) * 0.02);
        bi.* = rt(dt, -sc.* * @as(f64, @floatFromInt(top)) / 2 * (0.8 + 0.4 * rng.float(f64)));
    };
    for (0..rows) |r| for (0..cols) |c| {
        var code: u32 = rng.uintAtMost(u32, top);
        if (bits == 16) {
            const v = rt(dt, (rng.float(f64) * 2 - 1) * 0.1);
            code = if (dt == .bfloat16) spec.toBf16(@floatCast(v)) else @as(u16, @bitCast(@as(f16, @floatCast(v))));
            q.w[r * cols + c] = v;
        } else {
            const g = r * (cols / gs) + c / gs;
            q.w[r * cols + c] = @as(f64, @floatFromInt(code)) * q.scales[g] + q.biases[g];
        }
        setCode(q.words[r * per_row ..][0..per_row], c, bits, code);
    };
    return q;
}

/// Writes `code` as element `i` of a little-endian `bits`-bit stream held in u32 words (as float64 values).
pub fn setCode(words: []f64, i: usize, bits: u32, code: u32) void {
    var bit = i * bits;
    var left = bits;
    var v = code;
    while (left > 0) {
        const w = bit / 32;
        const off: u5 = @intCast(bit % 32);
        const take = @min(left, 32 - @as(u32, off));
        const mask: u64 = (@as(u64, 1) << @intCast(take)) - 1;
        var word: u64 = @intFromFloat(words[w]);
        word |= (@as(u64, v) & mask) << off;
        words[w] = @floatFromInt(word);
        v = if (take < 32) v >> @intCast(take) else 0;
        bit += take;
        left -= take;
    }
}

/// Re-derives the dequantized values after codes, scales or biases were edited in place (a stress fill).
pub fn dequant(q: *Quant, rows: usize, cols: usize, bits: u32, gs: usize, codes: []const u32) void {
    const per_row = cols * bits / 32;
    @memset(q.words, 0);
    for (0..rows) |r| for (0..cols) |c| {
        const code = codes[r * cols + c];
        const g = r * (cols / gs) + c / gs;
        q.w[r * cols + c] = @as(f64, @floatFromInt(code)) * q.scales[g] + q.biases[g];
        setCode(q.words[r * per_row ..][0..per_row], c, bits, code);
    };
}

/// A 0-dim int32 input (a row count): MLX passes it by value; the rig frees it with the other inputs.
pub fn intScalar(rig: anytype, v: i32) !mlx.mlx_array {
    const x = mlx.mlx_array_new_int(v);
    try rig.ins.append(rig.a, .{ .parent = x, .view = mlx.mlx_array_new(), .len = 1 });
    return x;
}

/// An evaluated copy of `x`'s values (float, int8 or 32-bit integer dtypes) as float64.
pub fn read(a: std.mem.Allocator, x: mlx.mlx_array, s: mlx.mlx_stream) ![]f64 {
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, x, false, s));
    try mlx.check(mlx.mlx_array_eval(c));
    const n = mlx.mlx_array_size(c);
    const raw = mlx.mlx_array_data_uint8(c) orelse return error.NoData;
    const out = try a.alloc(f64, n);
    errdefer a.free(out);
    const dt = mlx.mlx_array_dtype(c);
    for (out, 0..) |*v, i| v.* = switch (dt) {
        .float32 => spec.read(.f32, raw[0 .. 4 * n], i),
        .float16 => spec.read(.f16, raw[0 .. 2 * n], i),
        .bfloat16 => spec.read(.bf16, raw[0 .. 2 * n], i),
        .int32 => @floatFromInt(std.mem.readInt(i32, raw[4 * i ..][0..4], .little)),
        .uint32 => @floatFromInt(std.mem.readInt(u32, raw[4 * i ..][0..4], .little)),
        .int8 => @floatFromInt(@as(i8, @bitCast(raw[i]))),
        else => return error.UnsupportedDtype,
    };
    return out;
}

// ── tests ──

test "kernel contract host: values round to their dtype and the planted pattern lands where it says" {
    var prng = std.Random.DefaultPrng.init(1);
    const v = try uniform(std.testing.allocator, prng.random(), 64, -3, 3, .bfloat16);
    defer std.testing.allocator.free(v);
    for (v) |x| try std.testing.expectEqual(spec.roundTo(.bf16, x), x);
    var x: [16]f64 = @splat(0);
    plant(&x, 4, 4);
    try std.testing.expect(std.math.isNan(x[1]) and x[6] == std.math.inf(f64) and x[11] == -std.math.inf(f64));
    try std.testing.expect(x[12] == std.math.inf(f64) and x[15] == -std.math.inf(f64));
}

test "kernel contract host: codes pack as a little-endian bitstream at every width" {
    for ([_]u32{ 3, 4, 5, 6, 8 }) |bits| {
        var words: [15]f64 = @splat(0);
        const n = words.len * 32 / bits;
        for (0..n) |i| setCode(&words, i, bits, @intCast((i * 7 + 3) % (@as(usize, 1) << @intCast(bits))));
        for (0..n) |i| {
            var got: u64 = 0;
            for (0..bits) |b| {
                const at = i * bits + b;
                const word: u32 = @intFromFloat(words[at / 32]);
                got |= @as(u64, (word >> @intCast(at % 32)) & 1) << @intCast(b);
            }
            try std.testing.expectEqual(@as(u64, (i * 7 + 3) % (@as(usize, 1) << @intCast(bits))), got);
        }
    }
}
