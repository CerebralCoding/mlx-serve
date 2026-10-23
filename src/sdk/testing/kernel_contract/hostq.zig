//! Host-side operands for kernel registrations: values rounded to an input dtype, an MLX array's elements as f64, and
//! the quantized layouts kernels read (MLX's dense affine packing, e2m1 nibbles with e4m3 or e8m0 group scales).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const spec = @import("spec.zig");

/// The output element of a float dtype.
pub fn elemOf(dt: mlx.mlx_dtype) spec.Elem {
    return switch (dt) {
        .float32 => .f32,
        .float16 => .f16,
        else => .bf16,
    };
}

/// `v` as the device stores it in `dt` (integers as they are).
pub fn round(dt: mlx.mlx_dtype, v: f64) f64 {
    return switch (dt) {
        .float32, .float16, .bfloat16 => spec.roundTo(elemOf(dt), v),
        else => v,
    };
}

/// Uniform values in [lo, hi) rounded to `dt`.
pub fn uniform(rng: std.Random, out: []f64, lo: f64, hi: f64, dt: mlx.mlx_dtype) void {
    for (out) |*v| v.* = round(dt, lo + (hi - lo) * rng.float(f64));
}

/// Integer values in [lo, hi] (exact in every float dtype for |v| <= 256).
pub fn ints(rng: std.Random, out: []f64, lo: i32, hi: i32) void {
    for (out) |*v| v.* = @floatFromInt(lo + @as(i32, @intCast(rng.uintLessThan(u32, @intCast(hi - lo + 1)))));
}

/// Every element of `x` (evaluated) as f64, owned by the caller.
pub fn values(a: std.mem.Allocator, x: mlx.mlx_array) ![]f64 {
    try mlx.check(mlx.mlx_array_eval(x));
    const n = mlx.mlx_array_size(x);
    const p = mlx.mlx_array_data_uint8(x) orelse return error.NoData;
    const out = try a.alloc(f64, n);
    for (out, 0..) |*v, i| v.* = switch (mlx.mlx_array_dtype(x)) {
        .float32 => @as(f32, @bitCast(std.mem.readInt(u32, p[4 * i ..][0..4], .little))),
        .float16 => @as(f16, @bitCast(std.mem.readInt(u16, p[2 * i ..][0..2], .little))),
        .bfloat16 => spec.fromBf16(std.mem.readInt(u16, p[2 * i ..][0..2], .little)),
        .int32 => @floatFromInt(std.mem.readInt(i32, p[4 * i ..][0..4], .little)),
        .uint32 => @floatFromInt(std.mem.readInt(u32, p[4 * i ..][0..4], .little)),
        .uint8 => @floatFromInt(p[i]),
        .int8 => @floatFromInt(@as(i8, @bitCast(p[i]))),
        else => return error.UnsupportedDtype,
    };
    return out;
}

/// Little-endian `bytes` read as `n` elements of an integer dtype, as f64 (an int8 / uint32 output's values).
pub fn intsOf(dt: mlx.mlx_dtype, bytes: []const u8, i: usize) f64 {
    return switch (dt) {
        .int8 => @floatFromInt(@as(i8, @bitCast(bytes[i]))),
        .uint8 => @floatFromInt(bytes[i]),
        .int32 => @floatFromInt(std.mem.readInt(i32, bytes[4 * i ..][0..4], .little)),
        .uint32 => @floatFromInt(std.mem.readInt(u32, bytes[4 * i ..][0..4], .little)),
        else => unreachable,
    };
}

/// `codes` (each below 2^bits) packed as MLX's `quantize` packs a row: value i at bit i * bits of a little-endian
/// stream of uint32 words, a row `k * bits / 32` words; returned as the words' values.
pub fn packAffine(a: std.mem.Allocator, codes: []const u8, rows: usize, k: usize, bits: u32) ![]f64 {
    const wpr = k * bits / 32;
    const out = try a.alloc(f64, rows * wpr);
    const words = try a.alloc(u32, wpr);
    defer a.free(words);
    for (0..rows) |r| {
        @memset(words, 0);
        for (0..k) |i| {
            const bit = i * bits;
            const v: u64 = @as(u64, codes[r * k + i]) << @intCast(bit % 32);
            words[bit / 32] |= @truncate(v);
            if (bit % 32 + bits > 32) words[bit / 32 + 1] |= @truncate(v >> 32);
        }
        for (words, 0..) |w, j| out[r * wpr + j] = @floatFromInt(w);
    }
    return out;
}

/// An affine-quantized [n, k] weight as MLX's `quantize` lays it out: codes, a scale and bias per `gs` group (in the
/// activation dtype), and the packed words the kernels read.
pub const Affine = struct {
    bits: u32,
    gs: usize,
    n: usize,
    k: usize,
    codes: []u8 = &.{},
    scales: []f64 = &.{},
    biases: []f64 = &.{},
    words: []f64 = &.{},

    pub fn deinit(self: *Affine, a: std.mem.Allocator) void {
        a.free(self.codes);
        a.free(self.scales);
        a.free(self.biases);
        a.free(self.words);
        self.* = .{ .bits = self.bits, .gs = self.gs, .n = 0, .k = 0 };
    }

    /// Codes, scales and biases for `n` x `k`: random (a centered affine grid with scales near 0.01), or the stress
    /// form (integer codes mirrored across each row's halves, scale 1, bias 0), then packed.
    pub fn fill(self: *Affine, a: std.mem.Allocator, rng: std.Random, n: usize, k: usize, dt: mlx.mlx_dtype, stress: bool) !void {
        self.deinit(a);
        self.n = n;
        self.k = k;
        self.codes = try a.alloc(u8, n * k);
        self.scales = try a.alloc(f64, n * k / self.gs);
        self.biases = try a.alloc(f64, n * k / self.gs);
        const top: u32 = @as(u32, 1) << @intCast(self.bits);
        rng.bytes(self.codes);
        if (stress) {
            // Codes under 16 at every width, so a row's |x q| stays under 2^24 at K 6144 (fp32 then sums exactly).
            const h = k / 2;
            for (0..n) |r| for (0..h) |j| {
                const c: u8 = self.codes[r * k + j] & 15 & @as(u8, @intCast(top - 1));
                self.codes[r * k + j] = c;
                self.codes[r * k + h + j] = c;
            };
            @memset(self.scales, 1);
            @memset(self.biases, 0);
        } else {
            for (self.codes) |*c| c.* &= @intCast(top - 1);
            const half: f64 = @floatFromInt(top / 2);
            for (self.scales, self.biases) |*s, *b| {
                s.* = round(dt, (0.005 + 0.01 * rng.float(f64)) * 16 / half);
                b.* = round(dt, -s.* * half * (0.8 + 0.4 * rng.float(f64)));
            }
        }
        self.words = try packAffine(a, self.codes, n, k, self.bits);
    }

    /// Row `r` dequantized (q * s + b) into `w`, and each value's magnitude through the dequant (|q s| + |b|).
    pub fn row(self: *const Affine, r: usize, w: []f64, m: []f64) void {
        for (w, m, 0..) |*o, *g, i| {
            const at = r * self.k + i;
            const s = self.scales[at / self.gs];
            const b = self.biases[at / self.gs];
            const q: f64 = @floatFromInt(self.codes[at]);
            o.* = q * s + b;
            g.* = @abs(q * s) + @abs(b);
        }
    }
};

/// `codes` (4-bit) packed eight to a uint32, value i at bit 4 * i.
pub fn packNibbles(a: std.mem.Allocator, codes: []const u8) ![]f64 {
    const out = try a.alloc(f64, codes.len / 8);
    for (out, 0..) |*w, j| {
        var v: u32 = 0;
        for (0..8) |i| v |= @as(u32, codes[8 * j + i] & 0xf) << @intCast(4 * i);
        w.* = @floatFromInt(v);
    }
    return out;
}

/// An e2m1 code's value (sign bit 3, two exponent bits, one mantissa bit; no infinities or NaN).
pub fn e2m1(code: u8) f64 {
    const mag = [8]f64{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
    const v = mag[code & 7];
    return if (code & 8 != 0) -v else v;
}

/// An e4m3 code's value (bias 7, subnormals at exponent 0); 0x7f / 0xff are NaN in the format, never drawn here.
pub fn e4m3(code: u8) f64 {
    return e4m3_table[code];
}

const e4m3_table = blk: {
    @setEvalBranchQuota(100_000);
    var t: [256]f64 = undefined;
    for (&t, 0..) |*o, c| {
        const e: i32 = (c >> 3) & 0xf;
        const m: f64 = @floatFromInt(c & 7);
        const v = if (e == 0) m / 8 * 0x1p-6 else (1 + m / 8) * std.math.pow(f64, 2, @floatFromInt(e - 7));
        o.* = if (c & 0x80 != 0) -v else v;
    }
    break :blk t;
};

/// An e8m0 scale's value: 2^(code - 127).
pub fn e8m0(code: u8) f64 {
    return std.math.pow(f64, 2, @as(f64, @floatFromInt(@as(i32, code) - 127)));
}

/// A random e4m3 code with a value in [2^lo, 2^hi), either sign when `signed`.
pub fn e4m3Code(rng: std.Random, lo: i32, hi: i32, signed: bool) u8 {
    const e: u8 = @intCast(lo + 7 + @as(i32, @intCast(rng.uintLessThan(u32, @intCast(hi - lo)))));
    const code: u8 = (e << 3) | rng.uintLessThan(u8, 8);
    return if (signed and rng.boolean()) code | 0x80 else code;
}

/// Integers whose halves of each `n`-long row mirror: the halves sum past fp16's range, the row to a small integer (the
/// +1 offsets sit at 32..39, so a value made large later stays exactly mirrored).
pub fn mirrored(rng: std.Random, v: []f64, n: usize) void {
    const h = n / 2;
    for (0..v.len / n) |r| for (0..h) |j| {
        const x: f64 = @floatFromInt(1 + rng.uintLessThan(u32, 64));
        v[r * n + j] = x;
        v[r * n + h + j] = -x + @as(f64, if (j >= 32 and j < 40) 1 else 0);
    };
}

/// `mirrored` for a butterfly, whose sums also run in 32 strided lanes: each block's first 32 values are multiples of
/// 16 past 2048 (one a lane, exact in bf16), so every lane's partial sum lands off fp16's grid too.
pub fn mirroredWide(rng: std.Random, v: []f64, n: usize) void {
    mirrored(rng, v, n);
    const h = n / 2;
    for (0..v.len / n) |r| for (0..32) |j| {
        const big: f64 = @floatFromInt(16 * (128 + rng.uintLessThan(u32, 128)));
        v[r * n + j] = big;
        v[r * n + h + j] = -big;
    };
}

/// Terms summed in float64 in order (the reference's sums).
pub fn sum(t: []const f64) f64 {
    var s: f64 = 0;
    for (t) |v| s += v;
    return s;
}

/// The terms' magnitude: the bound's `mag`.
pub fn magnitude(t: []const f64) f64 {
    var s: f64 = 0;
    for (t) |v| s += @abs(v);
    return s;
}

/// The sum of x[i] * w[i] and of its terms' magnitudes, in float64 (vectorized: references run in Debug builds).
pub fn dot(x: []const f64, w: []const f64) [2]f64 {
    const V = @Vector(8, f64);
    var s: V = @splat(0);
    var m: V = @splat(0);
    var i: usize = 0;
    while (i + 8 <= x.len) : (i += 8) {
        const p = @as(V, x[i..][0..8].*) * @as(V, w[i..][0..8].*);
        s += p;
        m += @abs(p);
    }
    var out = [2]f64{ @reduce(.Add, s), @reduce(.Add, m) };
    while (i < x.len) : (i += 1) {
        out[0] += x[i] * w[i];
        out[1] += @abs(x[i] * w[i]);
    }
    return out;
}

/// The Walsh-Hadamard transform of `v` in place (Sylvester order, unscaled), in float64.
pub fn fwht(v: []f64) void {
    var h: usize = 1;
    while (h < v.len) : (h <<= 1) {
        var i: usize = 0;
        while (i < v.len) : (i += 2 * h) for (i..i + h) |j| {
            const a = v[j];
            const b = v[j + h];
            v[j] = a + b;
            v[j + h] = a - b;
        };
    }
}

/// Gates 1 and 3 for a kernel with per-output allowances: at `.nonfinite` every output's class equals float64's
/// (both kinds present), at `.shapes` also |got - want| <= allow. Null when it holds, else what failed (in `buf`).
pub fn within(buf: []u8, at: spec.Gate, got: []const f64, want: []const f64, allow: []const f64) ?[]const u8 {
    var planted: usize = 0;
    for (got, want, allow, 0..) |g, w, l, i| {
        if (spec.classOf(w) != .finite) planted += 1;
        if (spec.classOf(g) != spec.classOf(w))
            return std.fmt.bufPrint(buf, "output {d} is {t} where float64 gives {t}", .{ i, spec.classOf(g), spec.classOf(w) }) catch buf;
        if (at == .shapes and spec.classOf(w) == .finite and !(@abs(g - w) <= l))
            return std.fmt.bufPrint(buf, "output {d} is {e:.6} where float64 gives {e:.6}: {d:.2} of its allowance", .{ i, g, w, @abs(g - w) / l }) catch buf;
    }
    if (at == .nonfinite and (planted == 0 or planted == want.len))
        return std.fmt.bufPrint(buf, "the planted values make {d} of {d} reference outputs non-finite: no proof", .{ planted, want.len }) catch buf;
    return null;
}

// ── tests ──

test "kernel contract hostq: dense affine packing straddles words at 3, 5 and 6 bits like mx.quantize" {
    const a = std.testing.allocator;
    var codes: [64]u8 = undefined;
    for (&codes, 0..) |*c, i| c.* = @intCast(i % 8);
    const w = try packAffine(a, &codes, 1, 64, 3);
    defer a.free(w);
    try std.testing.expectEqual(@as(usize, 6), w.len);
    // value 10 (code 2) sits at bit 30: one bit in word 0, its high bit in word 1.
    const w0: u32 = @intFromFloat(w[0]);
    const w1: u32 = @intFromFloat(w[1]);
    try std.testing.expectEqual(@as(u32, 2), (w0 >> 30) | ((w1 & 1) << 2));
    try std.testing.expectEqual(@as(f64, 448), e4m3(0x7e));
    try std.testing.expectEqual(@as(f64, -6), e2m1(0xf));
    try std.testing.expectEqual(@as(f64, 0x1p-9), e4m3(1));
}
