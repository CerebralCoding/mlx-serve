//! DeepSeek-V4's kernel contract: the hyper-connection kernels (Sinkhorn split, its fused collapse, the hc_post
//! mix), the decode attention's sink softmax and its fused rope / quantizer chain, the compressor's window
//! emission, and the MoE gate+up GEMV. Listed in `kernel_contract/registry.zig`; findings are in AGENTS.md.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const kit = @import("contract_kit.zig");
const kc = @import("../../sdk/testing/kernel_contract/root.zig");
const dsv4 = @import("deepseek_v4.zig");

const spec = kit.spec;
const metal = kit.metal;
const Shape = kit.Shape;
const Fill = kit.Fill;
const Sum = kit.Sum;
const nan = kit.nan;
const inf = kit.inf;

/// fp32's unit roundoff.
const u = 0x1p-24;

fn ci(v: usize) c_int {
    return @intCast(v);
}

/// The release's hyper-connection geometry: 4 streams of the 4096-wide hidden, 20 Sinkhorn iterations.
const HC = 4;
const ITERS = 20;
const DIM = 4096;
const EPS = 1e-6;
/// The decode widths the hc kernels serve: one token, a verify block, the widest cached config, a prefill sub-chunk.
const hc_rows = [_]usize{ 1, 8, 32, 512 };

// ── hc_post: ns[t, c, j] = sum_q combT[t, c, q] stream[t, q, j] + post[t, c] out[t, j] ──

const HcPost = struct {
    pub const Params = struct {};

    pub fn geometry(_: Params, s: Shape) ?metal.Geometry {
        if (s.rows == 0 or s.n == 0 or s.k != HC) return null;
        var g = metal.Geometry{ .grid = .{ ci(s.rows), HC, ci(s.n) }, .tg = .{ 1, 1, ci(@min(s.n, 256)) } };
        g.out(&.{ ci(s.rows), HC, ci(s.n) }, .float32);
        g.int("HC", HC);
        g.int("S", s.rows);
        g.int("D", s.n);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const S = k.shape.rows;
        const D = k.shape.n;
        const pack = 2 * HC + HC * HC;
        const pk = try k.add(.float32, &.{ S, pack });
        const st = try k.add(.float32, &.{ S, HC, D });
        const ov = try k.add(.float32, &.{ S, D });
        kit.uniform(rng, pk, -1, 1);
        kit.uniform(rng, st, -4, 4);
        kit.uniform(rng, ov, -4, 4);
        switch (f) {
            .random => {},
            .nonfinite => kit.plantRows(ov, S, D),
            .stress => {
                // Every coefficient 1; the streams +-2^20 with a small integer on the first: exact in fp32 in any
                // order, past fp16's range, and below bf16's resolution.
                @memset(pk, 1);
                for (0..S) |t| for (0..D) |j| {
                    for (0..HC) |q| st[(t * HC + q) * D + j] = if (q % 2 == 0) 0x1p20 else -0x1p20;
                    st[t * HC * D + j] += @floatFromInt(rng.intRangeAtMost(i32, 1, 64));
                    ov[t * D + j] = @floatFromInt(rng.intRangeAtMost(i32, -64, 64));
                };
            },
        }
    }

    pub fn count(_: Params, s: Shape) usize {
        return s.rows * HC * s.n;
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        const S = k.shape.rows;
        const D = k.shape.n;
        const pack = 2 * HC + HC * HC;
        const pk = k.in(0);
        const st = k.in(1);
        const ov = k.in(2);
        var t5: [HC + 1]f64 = undefined;
        for (0..S) |t| for (0..HC) |c| for (0..D) |j| {
            for (0..HC) |q| t5[q] = pk[t * pack + 2 * HC + c * HC + q] * st[(t * HC + q) * D + j];
            t5[HC] = pk[t * pack + HC + c] * ov[t * D + j];
            const i = (t * HC + c) * D + j;
            out[i] = sum.of(&t5);
            for (t5) |v| mag[i] += @abs(v);
        };
    }
};

// ── Sinkhorn: per token, pre = sigmoid + eps, post = 2 sigmoid, comb = 20 Sinkhorn normalizations ──

/// The mixes' split for one token, in float64 (the kernel's formula and order); `pk` gets pre | post | combT.
fn sinkhornToken(mixes: []const f64, ssq: f64, scale: []const f64, base: []const f64, consts: []const f64, sum: Sum, pk: []f64) void {
    const rsq = 1 / @sqrt(ssq / consts[0] + consts[1]);
    const heps = consts[2];
    var comb: [HC * HC]f64 = undefined;
    for (0..HC) |j| {
        pk[j] = 1 / (1 + @exp(-(mixes[j] * rsq * scale[0] + base[j]))) + heps;
        pk[HC + j] = 2 / (1 + @exp(-(mixes[HC + j] * rsq * scale[1] + base[HC + j])));
    }
    for (&comb, 0..) |*c, j| c.* = mixes[2 * HC + j] * rsq * scale[2] + base[2 * HC + j];
    var t4: [HC]f64 = undefined;
    for (0..HC) |j| {
        var mx = comb[j * HC];
        for (1..HC) |q| mx = kit.maxN(mx, comb[j * HC + q]);
        for (0..HC) |q| {
            comb[j * HC + q] = @exp(comb[j * HC + q] - mx);
            t4[q] = comb[j * HC + q];
        }
        const s = sum.of(&t4);
        for (0..HC) |q| comb[j * HC + q] = comb[j * HC + q] / s + heps;
    }
    for (0..ITERS) |it| {
        if (it > 0) for (0..HC) |j| {
            for (0..HC) |q| t4[q] = comb[j * HC + q];
            const s = sum.of(&t4);
            for (0..HC) |q| comb[j * HC + q] /= (s + heps);
        };
        for (0..HC) |q| {
            for (0..HC) |j| t4[j] = comb[j * HC + q];
            const s = sum.of(&t4);
            for (0..HC) |j| comb[j * HC + q] /= (s + heps);
        }
    }
    for (0..HC) |j| for (0..HC) |q| {
        pk[2 * HC + q * HC + j] = comb[j * HC + q];
    };
}

/// The split's inputs (mixes, ssq, scale, base, consts) at a realistic scale: mixes about the stream's rms.
fn sinkhornInputs(k: anytype, f: Fill, rng: std.Random) !void {
    const S = k.shape.rows;
    const mix = (2 + HC) * HC;
    const mixes = try k.add(.float32, &.{ S, mix });
    const ssq = try k.add(.float32, &.{ S, 1 });
    const scale = try k.add(.float32, &.{3});
    const base = try k.add(.float32, &.{mix});
    const consts = try k.add(.float32, &.{3});
    kit.uniform(rng, mixes, -3, 3);
    kit.uniform(rng, ssq, 0.5 * HC * DIM, 2 * HC * DIM);
    kit.uniform(rng, scale, 0.5, 1.5);
    kit.uniform(rng, base, -0.5, 0.5);
    @memcpy(consts, &[_]f64{ HC * DIM, EPS, EPS });
    if (f == .nonfinite) kit.plantRows(mixes, S, mix);
}

const Sinkhorn = struct {
    pub const Params = struct {};

    pub fn geometry(_: Params, s: Shape) ?metal.Geometry {
        if (s.rows == 0 or s.k != HC) return null;
        var g = metal.Geometry{ .grid = .{ ci(s.rows), 1, 1 }, .tg = .{ ci(@min(s.rows, 32)), 1, 1 } };
        g.out(&.{ ci(s.rows), 2 * HC + HC * HC }, .float32);
        g.int("HC", HC);
        g.int("ITERS", ITERS);
        g.int("S", s.rows);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        try sinkhornInputs(k, f, rng);
    }

    pub fn count(_: Params, s: Shape) usize {
        return s.rows * (2 * HC + HC * HC);
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        const mix = (2 + HC) * HC;
        const pack = 2 * HC + HC * HC;
        for (0..k.shape.rows) |t| sinkhornToken(k.in(0)[t * mix ..][0..mix], k.in(1)[t], k.in(2), k.in(3), k.in(4), sum, out[t * pack ..][0..pack]);
        for (mag, out) |*m, w| m.* = @abs(w);
    }
};

const SinkhornY = struct {
    pub const Params = struct {};

    pub fn geometry(_: Params, s: Shape) ?metal.Geometry {
        if (s.rows == 0 or s.n == 0 or s.k != HC) return null;
        var g = metal.Geometry{ .grid = .{ ci(s.rows), ci(s.n), 1 }, .tg = .{ 1, ci(@min(s.n, 256)), 1 } };
        g.out(&.{ ci(s.rows), 2 * HC + HC * HC }, .float32);
        g.out(&.{ ci(s.rows), ci(s.n) }, .float32);
        g.int("HC", HC);
        g.int("ITERS", ITERS);
        g.int("S", s.rows);
        g.int("D", s.n);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        try sinkhornInputs(k, f, rng);
        const st = try k.add(.float32, &.{ k.shape.rows, HC, k.shape.n });
        kit.uniform(rng, st, -4, 4);
    }

    pub fn count(_: Params, s: Shape) usize {
        return s.rows * (2 * HC + HC * HC) + s.rows * s.n;
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        const S = k.shape.rows;
        const D = k.shape.n;
        const mix = (2 + HC) * HC;
        const pack = 2 * HC + HC * HC;
        const st = k.in(5);
        for (0..S) |t| sinkhornToken(k.in(0)[t * mix ..][0..mix], k.in(1)[t], k.in(2), k.in(3), k.in(4), sum, out[t * pack ..][0..pack]);
        for (mag[0 .. S * pack], out[0 .. S * pack]) |*m, w| m.* = @abs(w);
        var t4: [HC]f64 = undefined;
        for (0..S) |t| for (0..D) |j| {
            for (0..HC) |c| t4[c] = out[t * pack + c] * st[(t * HC + c) * D + j];
            const i = S * pack + t * D + j;
            out[i] = sum.of(&t4);
            for (t4) |v| mag[i] += @abs(v);
        };
    }
};

// ── sink softmax: y = exp(s scale - m) / (sum exp(s scale - m) + exp(sink - m)), m over the row and the sink ──

const SinkSoftmax = struct {
    pub const Params = struct {};

    fn groupFor(tk: usize) usize {
        return @min(256, std.math.ceilPowerOfTwoAssert(usize, @max(tk, 2)));
    }

    pub fn geometry(_: Params, s: Shape) ?metal.Geometry {
        if (s.k == 0 or s.rows == 0 or s.rows > 65535) return null;
        const tg = groupFor(s.k);
        var g = metal.Geometry{ .grid = .{ ci(tg), ci(s.rows), 1 }, .tg = .{ ci(tg), 1, 1 } };
        g.out(&.{ ci(s.rows), ci(s.k) }, .float32);
        g.int("H", s.rows);
        g.int("TG", tg);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const H = k.shape.rows;
        const TK = k.shape.k;
        const sc = try k.add(.float32, &.{ H, TK });
        const sink = try k.add(.float32, &.{ H, 1 });
        const scale = try k.add(.float32, &.{1});
        const tk = try k.add(.int32, &.{1});
        kit.uniform(rng, sc, -20, 20);
        kit.uniform(rng, sink, -3, 3);
        scale[0] = 1.0 / @sqrt(512.0);
        tk[0] = @floatFromInt(TK);
        if (f == .nonfinite) kit.plantRows(sc, H, TK);
    }

    pub fn count(_: Params, s: Shape) usize {
        return s.rows * s.k;
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        const TK = k.shape.k;
        const e = try k.scratch().alloc(f64, TK);
        defer k.scratch().free(e);
        const scale = spec.roundTo(.f32, k.in(2)[0]);
        for (0..k.shape.rows) |h| {
            const row = k.in(0)[h * TK ..][0..TK];
            const sk = k.in(1)[h];
            var m = sk;
            for (row) |v| m = kit.maxN(m, v * scale);
            for (e, row) |*x, v| x.* = @exp(v * scale - m);
            const den = sum.of(e) + @exp(sk - m);
            for (e, 0..) |x, j| {
                out[h * TK + j] = x / den;
                mag[h * TK + j] = @abs(x / den);
            }
        }
    }
};

// ── the fp8 / fp4 activation simulation both the decode chain and the window emission end in ──

/// One quantizer: e4m3-like fp8 (`fp4` false: 448, 3 mantissa bits, groups of 64) or e2m1 fp4 (6, 1 bit, 32).
const Qat = struct {
    fp4: bool,

    fn cm(q: Qat) f64 {
        return if (q.fp4) 6 else 448;
    }
    fn mant(q: Qat) i32 {
        return if (q.fp4) 1 else 3;
    }
    fn minExp(q: Qat) i32 {
        return if (q.fp4) 0 else -6;
    }
    fn group(q: Qat) usize {
        return if (q.fp4) 32 else 64;
    }
    fn floor(q: Qat) f64 {
        return if (q.fp4) std.math.ldexp(@as(f64, 6), -126) else 1e-4;
    }

    /// The group scale for `amax`: 2^ceil(log2(amax / cm)), the quotient rounded to f32 as the kernel takes it.
    fn scaleFor(q: Qat, amax: f64) f64 {
        const x = spec.roundTo(.f32, amax / q.cm());
        if (!(x > 0) or !std.math.isFinite(x)) return x;
        const fr = std.math.frexp(x);
        const e = if (fr.significand == 0.5) fr.exponent - 1 else fr.exponent;
        return std.math.ldexp(@as(f64, 1), e);
    }

    /// `v` through the quantizer at `scale` (NaN propagates; half-even rounding as `rint`).
    fn apply(q: Qat, v: f64, scale: f64) f64 {
        const y = std.math.clamp(v / scale, -q.cm(), q.cm());
        if (std.math.isNan(v) or std.math.isNan(scale)) return nan;
        const a = @abs(y);
        const af = @max(a, std.math.ldexp(@as(f64, 1), q.minExp()));
        var e2 = std.math.frexp(af).exponent - 1;
        if (e2 < q.minExp()) e2 = q.minExp();
        const quantum = std.math.ldexp(@as(f64, 1), e2 - q.mant());
        const qv = @min(rintEven(a / quantum) * quantum, q.cm());
        const sq = if (y < 0) -qv else if (y > 0) qv else 0;
        return sq * scale;
    }

    /// The scale of each group of `v[0..qw]` (NaN propagates through the group max).
    fn groupMax(q: Qat, v: []const f64) f64 {
        var a = q.floor();
        for (v) |x| a = kit.maxN(a, @abs(x));
        return a;
    }
};

fn rintEven(x: f64) f64 {
    const f = @floor(x);
    const d = x - f;
    if (d > 0.5) return f + 1;
    if (d < 0.5) return f;
    return if (@mod(f, 2) == 0) f else f + 1;
}

/// A row's float64 values `v` with absolute error bounds `e`, through the quantizer on its first `qw` entries.
fn quantRow(q: Qat, v: []const f64, qw: usize, out: []f64) void {
    var g: usize = 0;
    while (g < qw) : (g += q.group()) {
        const vs = v[g..][0..q.group()];
        const s = q.scaleFor(q.groupMax(vs));
        for (vs, out[g..][0..q.group()]) |x, *o| o.* = q.apply(x, s);
    }
    @memcpy(out[qw..], v[qw..]);
}

/// Whether `got` can be what the kernel's f32 arithmetic gives for values within `e` of `v`: the quantizer is monotone
/// at a fixed scale, and the group scale can move only where the group max sits within its error of a power of two.
fn quantRowOk(q: Qat, v: []const f64, e: []const f64, qw: usize, got: []const f64, buf: []u8) ?[]const u8 {
    var g: usize = 0;
    while (g < qw) : (g += q.group()) {
        var lo = q.floor();
        var hi = q.floor();
        for (g..g + q.group()) |j| {
            lo = @max(lo, @abs(v[j]) - e[j]);
            hi = @max(hi, @abs(v[j]) + e[j]);
        }
        const s_lo = q.scaleFor(lo);
        const s_hi = q.scaleFor(hi);
        for (g..g + q.group()) |j| {
            var s = s_lo;
            const ok = while (s <= s_hi) : (s *= 2) {
                const a = q.apply(v[j] - e[j], s);
                const b = q.apply(v[j] + e[j], s);
                if (got[j] >= @min(a, b) and got[j] <= @max(a, b)) break true;
            } else false;
            if (!ok) return std.fmt.bufPrint(buf, "quantized output {d} is {e} where {e} +- {e} quantizes to {e}", .{ j, got[j], v[j], e[j], q.apply(v[j], s_lo) }) catch "quantizer";
        }
    }
    for (qw..v.len) |j| if (!(@abs(got[j] - v[j]) <= e[j])) {
        return std.fmt.bufPrint(buf, "output {d} is {e} where float64 gives {e} (allowed {e})", .{ j, got[j], v[j], e[j] }) catch "outside";
    };
    return null;
}

// ── the decode chain: per head [RMS ->] rope tail -> [fp8 head | Hadamard + fp4] ──

/// MLX's `fast::rope` traditional pairs on the last `rd` entries, conjugate when `inv`; `e` carries absolute bounds.
fn ropeTail(v: []f64, e: []f64, rd: usize, cos: []const f64, sin: []const f64, inv: bool) void {
    const base = v.len - rd;
    for (0..rd / 2) |p| {
        const a = v[base + 2 * p];
        const b = v[base + 2 * p + 1];
        const ea = e[base + 2 * p];
        const eb = e[base + 2 * p + 1];
        const c = cos[p];
        const s = if (inv) -sin[p] else sin[p];
        v[base + 2 * p] = a * c - b * s;
        v[base + 2 * p + 1] = a * s + b * c;
        // Each output is two products and a sum, fused or not: gamma_2 of ITS OWN two terms.
        const g2 = 2 * u / (1 - 2 * u);
        e[base + 2 * p] = @abs(c) * ea + @abs(s) * eb + g2 * (@abs(a * c) + @abs(b * s));
        e[base + 2 * p + 1] = @abs(s) * ea + @abs(c) * eb + g2 * (@abs(a * s) + @abs(b * c));
    }
}

/// `v @ hada` (rows of `n`), with the bound carried through.
fn hadamard(v: []f64, e: []f64, hada: []const f64, sum: Sum, scratch: std.mem.Allocator) !void {
    const n = v.len;
    const t = try scratch.alloc(f64, 4 * n);
    defer scratch.free(t);
    const ov = t[n .. 2 * n];
    const oe = t[2 * n .. 3 * n];
    const mag = t[3 * n ..];
    for (0..n) |j| {
        var be: f64 = 0;
        mag[j] = 0;
        for (0..n) |kk| {
            t[kk] = v[kk] * hada[kk * n + j];
            be += e[kk] * @abs(hada[kk * n + j]);
            mag[j] += @abs(t[kk]);
        }
        ov[j] = sum.of(t[0..n]);
        oe[j] = be + spec.gamma(@floatFromInt(n), u) * mag[j];
    }
    @memcpy(v, ov);
    @memcpy(e, oe);
}

/// Per-row RMS norm (`x * rsqrt(mean(x^2) + eps) * w`), with its bound: the sum's gamma, rsqrt and two products.
fn rmsRow(v: []f64, e: []f64, w: ?[]const f64, eps: f64, sum: Sum, scratch: std.mem.Allocator) !void {
    const n = v.len;
    const sq = try scratch.alloc(f64, n);
    defer scratch.free(sq);
    for (sq, v) |*q, x| q.* = x * x;
    var se: f64 = 0;
    for (v, e) |x, ex| se += 2 * @abs(x) * ex;
    const ss = sum.of(sq);
    const rinv = 1 / @sqrt(ss / @as(f64, @floatFromInt(n)) + eps);
    const rel = (spec.gamma(@floatFromInt(n), u) * ss + se) / (2 * ss) + 4 * u;
    for (v, e, 0..) |*x, *ex, j| {
        const wj = if (w) |ww| ww[j] else 1;
        const y = x.* * rinv * wj;
        ex.* = ex.* * @abs(rinv * wj) + @abs(y) * (rel + 2 * u);
        x.* = y;
    }
}

const DecChain = struct {
    pub const Params = struct { rms: bool, post: u8, inv: bool, rd: usize = 64 };

    fn qat(p: Params) Qat {
        return .{ .fp4 = p.post == 2 };
    }

    pub fn geometry(p: Params, s: Shape) ?metal.Geometry {
        const D = s.n;
        if (D > 1024 or D == 0 or p.rd == 0 or p.rd % 2 != 0 or p.rd > D or s.rows == 0) return null;
        if (p.post == 1 and (D <= p.rd or (D - p.rd) % 64 != 0)) return null;
        if (p.post == 2 and D % 32 != 0) return null;
        const tg = tgFor(D);
        var g = metal.Geometry{ .grid = .{ ci(tg), ci(s.rows), 1 }, .tg = .{ ci(tg), 1, 1 } };
        g.out(&.{ ci(s.rows), ci(D) }, .float32);
        g.int("D", D);
        g.int("H", s.rows);
        g.int("RD", p.rd);
        g.int("RMS", @intFromBool(p.rms));
        g.int("POST", p.post);
        g.int("INV", @intFromBool(p.inv));
        g.int("TG", tg);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const p = k.p;
        const H = k.shape.rows;
        const D = k.shape.n;
        const half = p.rd / 2;
        const x = try k.add(.float32, &.{ H, D });
        const w = try k.add(.float32, &.{if (p.rms) D else half});
        const cos = try k.add(.float32, &.{half});
        const sin = try k.add(.float32, &.{half});
        const hada = try k.add(.float32, if (p.post == 2) &.{ D, D } else &.{half});
        const eps = try k.add(.float32, &.{1});
        kit.uniform(rng, x, -2, 2);
        kit.uniform(rng, w, 0.5, 1.5);
        for (cos, sin) |*c, *sn| {
            const a = rng.float(f64) * 2 * std.math.pi;
            c.* = @cos(a);
            sn.* = @sin(a);
        }
        if (p.post == 2) hadamardFill(hada, D);
        eps[0] = EPS;
        switch (f) {
            .random => {},
            .nonfinite => kit.plantRows(x, H, D),
            .stress => {
                // Every stage exact: unit rope, no eps, rows of +-32 (an exact rsqrt) or, through the Hadamard,
                // mirrored +-2^20 pairs summed by an all-ones matrix (exact in fp32, past fp16, below bf16).
                @memset(cos, 1);
                @memset(sin, 0);
                eps[0] = 0;
                if (p.rms) @memset(w, 1);
                for (x) |*v| v.* = if (rng.boolean()) 32 else -32;
                if (p.post == 2) {
                    @memset(hada, 1);
                    for (0..H) |h| for (0..D) |j| {
                        x[h * D + j] = if (j % 2 == 0) 0x1p20 + @as(f64, @floatFromInt(rng.intRangeAtMost(i32, 1, 64))) else -0x1p20;
                    };
                }
            },
        }
    }

    pub fn count(_: Params, s: Shape) usize {
        return s.rows * s.n;
    }

    /// Head `h`'s values before the quantizer and their bounds.
    fn prequant(k: anytype, h: usize, v: []f64, e: []f64, sum: Sum) !void {
        const p = k.p;
        const D = k.shape.n;
        @memcpy(v, k.in(0)[h * D ..][0..D]);
        @memset(e, 0);
        if (p.rms) try rmsRow(v, e, k.in(1), k.in(5)[0], sum, k.scratch());
        ropeTail(v, e, p.rd, k.in(2), k.in(3), p.inv);
        if (p.post == 2) try hadamard(v, e, k.in(4), sum, k.scratch());
    }

    fn qw(p: Params, D: usize) usize {
        return switch (p.post) {
            1 => D - p.rd,
            2 => D,
            else => 0,
        };
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        const D = k.shape.n;
        const t = try k.scratch().alloc(f64, 2 * D);
        defer k.scratch().free(t);
        for (0..k.shape.rows) |h| {
            try prequant(k, h, t[0..D], t[D..], sum);
            quantRow(qat(k.p), t[0..D], qw(k.p, D), out[h * D ..][0..D]);
        }
        @memset(mag, 0);
    }

    /// Gate 1: classes as float64 gives them; gate 3: within the carried bounds, the quantized entries within the
    /// quantizer's envelope of those bounds.
    pub fn check(k: anytype, at: spec.Gate) !?[]const u8 {
        const D = k.shape.n;
        const n = k.shape.rows * D;
        const a = k.scratch();
        const buf = try a.alloc(f64, 3 * n);
        defer a.free(buf);
        const got = buf[0..n];
        k.outputs(.f32, got);
        if (at == .nonfinite) {
            try compute(k, buf[n .. 2 * n], buf[2 * n ..], .{});
            for (got, buf[n .. 2 * n], 0..) |g, w, i| if (spec.classOf(g) != spec.classOf(w))
                return std.fmt.bufPrint(&k.msg, "output {d} is {e} where float64 gives {e}", .{ i, g, w }) catch "class";
            return null;
        }
        for (0..k.shape.rows) |h| {
            const v = buf[n..][0..D];
            const e = buf[n + D ..][0..D];
            try prequant(k, h, v, e, .{});
            if (quantRowOk(qat(k.p), v, e, qw(k.p, D), got[h * D ..][0..D], &k.msg)) |why| return why;
        }
        return null;
    }
};

fn tgFor(d: usize) usize {
    var tg: usize = 1;
    while (tg * 2 <= @min(d, 256)) tg *= 2;
    return tg;
}

/// The release's Hadamard: entries +-1/sqrt(n), the sign the parity of popcount(i & j).
fn hadamardFill(h: []f64, n: usize) void {
    const inv = 1 / @sqrt(@as(f64, @floatFromInt(n)));
    for (0..n) |i| for (0..n) |j| {
        h[i * n + j] = if (@popCount(i & j) & 1 == 1) -inv else inv;
    };
}

/// Threadgroup bytes of the decode chain at width `d`: the row, the Hadamard row, the reduction and the scales.
fn decChainTg(d: usize, post: u8) u32 {
    const rd = 64;
    const qw: usize = switch (post) {
        1 => d - rd,
        2 => d,
        else => 0,
    };
    const group: usize = if (post == 2) 32 else 64;
    return @intCast(4 * (d + (if (post == 2) d else 1) + tgFor(d) + (if (qw > 0) qw / group else 1)));
}

// ── window emission: masked-softmax combine of a closed window -> RMS -> rope -> [Hadamard] -> fp8 / fp4 ──

const EmitWin = struct {
    /// One compressor's layout: overlapping windows (ratio 4), Hadamard + fp4 (the indexer), rows before the chunk,
    /// the column offset and width of `cin`, the chunk's first row's slot in the ratio.
    pub const Params = struct { overlap: bool, rotate: bool, pre_n: usize, base_mod: usize, col_off: usize, rd: usize = 64 };

    fn cd(p: Params, D: usize) usize {
        return (if (p.overlap) @as(usize, 2) else 1) * D;
    }
    fn cinW(p: Params, D: usize) usize {
        return p.col_off + 2 * cd(p, D);
    }
    fn rowsOf(p: Params, R: usize) usize {
        return (if (p.overlap) @as(usize, 2) else 1) * R;
    }
    /// Chunk rows `cin` holds: the last window's last row past the pre-chunk rows.
    fn cinRows(p: Params, R: usize, nwin: usize) usize {
        const last = if (p.overlap) R + (nwin - 1) * R + R - 1 else nwin * R - 1;
        return if (last + 1 > p.pre_n) last + 1 - p.pre_n else 1;
    }

    pub fn geometry(p: Params, s: Shape) ?metal.Geometry {
        const D = s.n;
        if (D > 1024 or s.rows == 0 or s.k == 0 or p.rd == 0 or p.rd % 2 != 0) return null;
        if (p.rotate and D % 32 != 0) return null;
        if (!p.rotate and (D <= p.rd or (D - p.rd) % 64 != 0)) return null;
        const tg = tgFor(D);
        var g = metal.Geometry{ .grid = .{ ci(tg), ci(s.rows), 1 }, .tg = .{ ci(tg), 1, 1 } };
        g.out(&.{ ci(s.rows), ci(D) }, .float32);
        g.int("D", D);
        g.int("RD", p.rd);
        g.int("R", s.k);
        g.int("OVERLAP", @intFromBool(p.overlap));
        g.int("ROTATE", @intFromBool(p.rotate));
        g.int("GROUP", @as(usize, if (p.rotate) 32 else 64));
        g.int("PRE_N", p.pre_n);
        g.int("COL_OFF", p.col_off);
        g.int("CIN_W", cinW(p, D));
        g.int("BASE_MOD", p.base_mod);
        g.int("NWIN", s.rows);
        g.int("TG", tg);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const p = k.p;
        const nwin = k.shape.rows;
        const D = k.shape.n;
        const R = k.shape.k;
        const half = p.rd / 2;
        const c = cd(p, D);
        const cin = try k.add(.float32, &.{ cinRows(p, R, nwin), cinW(p, D) });
        const pre_kv = try k.add(.float32, &.{ @max(p.pre_n, 1), c });
        const pre_sc = try k.add(.float32, &.{ @max(p.pre_n, 1), c });
        const ape = try k.add(.float32, &.{ R, c });
        const nw = try k.add(.float32, &.{D});
        const cos = try k.add(.float32, &.{ nwin, half });
        const sin = try k.add(.float32, &.{ nwin, half });
        const hada = try k.add(.float32, if (p.rotate) &.{ D, D } else &.{1});
        const eps = try k.add(.float32, &.{1});
        kit.uniform(rng, cin, -2, 2);
        kit.uniform(rng, pre_kv, -2, 2);
        kit.uniform(rng, pre_sc, -2, 2);
        kit.uniform(rng, ape, -1, 1);
        kit.uniform(rng, nw, 0.5, 1.5);
        for (cos, sin) |*cv, *sv| {
            const a = rng.float(f64) * 2 * std.math.pi;
            cv.* = @cos(a);
            sv.* = @sin(a);
        }
        if (p.rotate) hadamardFill(hada, D);
        eps[0] = EPS;
        switch (f) {
            .random => {},
            .nonfinite => {
                // Window 0's first chunk row: a NaN kv, an infinite score; later windows stay finite.
                cin[p.col_off + 1] = nan;
                cin[p.col_off + c + 2] = inf;
                if (p.pre_n > 0) pre_kv[3] = -inf;
            },
            .stress => {
                // Every stage exact: equal scores (weights 1/rows, a power of two), unit rope, no eps, an identity
                // Hadamard; each feature's rows +-2^20 with the first carrying rows * (+-32), so the combine is
                // +-32 exactly in fp32 in any order, past fp16's range and below bf16's resolution.
                @memset(pre_sc, 0);
                @memset(ape, 0);
                @memset(nw, 1);
                @memset(cos, 1);
                @memset(sin, 0);
                eps[0] = 0;
                if (p.rotate) {
                    @memset(hada, 0);
                    for (0..D) |j| hada[j * D + j] = 1;
                }
                const rows = rowsOf(p, R);
                for (0..nwin) |w| for (0..D) |j| {
                    const sign: f64 = if (rng.boolean()) 32 else -32;
                    for (0..rows) |i| {
                        const at = loc(p, D, R, w, j, i);
                        const v: f64 = if (i % 2 == 1) -0x1p20 else 0x1p20 + (if (i == 0) @as(f64, @floatFromInt(rows)) * sign else 0);
                        if (at.pre) pre_kv[at.kv] = v else {
                            cin[at.kv] = v;
                            cin[at.sc] = 0;
                        }
                    }
                };
            },
        }
    }

    /// Where row `i` of window `w`'s feature `j` lives: a pre-chunk row (`pre_kv`/`pre_sc` index) or a chunk row
    /// (`cin` indices of its kv and score, and its `ape` row).
    const Loc = struct { pre: bool, kv: usize, sc: usize, ape: usize = 0 };

    fn loc(p: Params, D: usize, R: usize, w: usize, j: usize, i: usize) Loc {
        const c = cd(p, D);
        const ext, const col = if (p.overlap and i >= R) .{ R + w * R + (i - R), D + j } else .{ w * R + i, j };
        if (ext < p.pre_n) return .{ .pre = true, .kv = ext * c + col, .sc = ext * c + col };
        const cr = ext - p.pre_n;
        const W = cinW(p, D);
        return .{ .pre = false, .kv = cr * W + p.col_off + col, .sc = cr * W + p.col_off + c + col, .ape = ((p.base_mod + cr) % R) * c + col };
    }

    pub fn count(_: Params, s: Shape) usize {
        return s.rows * s.n;
    }

    /// Window `w`'s values before the quantizer and their bounds.
    fn prequant(k: anytype, w: usize, v: []f64, e: []f64, sum: Sum) !void {
        const p = k.p;
        const D = k.shape.n;
        const R = k.shape.k;
        const rows = rowsOf(p, R);
        const a = k.scratch();
        const t = try a.alloc(f64, 3 * rows);
        defer a.free(t);
        const kv = t[0..rows];
        const sc = t[rows .. 2 * rows];
        const pe = t[2 * rows ..];
        const cin = k.in(0);
        for (0..D) |j| {
            for (0..rows) |i| {
                const at = loc(p, D, R, w, j, i);
                if (at.pre) {
                    kv[i] = k.in(1)[at.kv];
                    sc[i] = k.in(2)[at.sc];
                } else {
                    kv[i] = cin[at.kv];
                    sc[i] = spec.roundTo(.f32, cin[at.sc] + k.in(3)[at.ape]);
                }
            }
            var mx = -inf;
            for (sc) |x| mx = kit.maxN(mx, x);
            var spread: f64 = 0;
            for (sc, pe) |x, *q| {
                q.* = @exp(x - mx);
                spread = @max(spread, @abs(x - mx));
            }
            const den = sum.of(pe);
            var mag: f64 = 0;
            for (pe, kv) |*q, x| {
                q.* = q.* / den * x;
                mag += @abs(q.*);
            }
            v[j] = sum.of(pe);
            // exp of an argument carrying (spread) ulps, the denominator's sum, the divide and the product.
            e[j] = mag * (spec.gamma(@floatFromInt(rows), u) * 2 + (spread + 8) * u);
        }
        try rmsRow(v, e, k.in(4), k.in(8)[0], sum, a);
        ropeTail(v, e, p.rd, k.in(5)[w * (p.rd / 2) ..][0 .. p.rd / 2], k.in(6)[w * (p.rd / 2) ..][0 .. p.rd / 2], false);
        if (p.rotate) try hadamard(v, e, k.in(7), sum, a);
    }

    fn qw(p: Params, D: usize) usize {
        return if (p.rotate) D else D - p.rd;
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        const D = k.shape.n;
        const t = try k.scratch().alloc(f64, 2 * D);
        defer k.scratch().free(t);
        for (0..k.shape.rows) |w| {
            try prequant(k, w, t[0..D], t[D..], sum);
            quantRow(.{ .fp4 = k.p.rotate }, t[0..D], qw(k.p, D), out[w * D ..][0..D]);
        }
        @memset(mag, 0);
    }

    pub fn check(k: anytype, at: spec.Gate) !?[]const u8 {
        const D = k.shape.n;
        const n = k.shape.rows * D;
        const a = k.scratch();
        const buf = try a.alloc(f64, 3 * n);
        defer a.free(buf);
        const got = buf[0..n];
        k.outputs(.f32, got);
        if (at == .nonfinite) {
            try compute(k, buf[n .. 2 * n], buf[2 * n ..], .{});
            for (got, buf[n .. 2 * n], 0..) |g, w, i| if (spec.classOf(g) != spec.classOf(w))
                return std.fmt.bufPrint(&k.msg, "output {d} is {e} where float64 gives {e}", .{ i, g, w }) catch "class";
            return null;
        }
        for (0..k.shape.rows) |w| {
            const v = buf[n..][0..D];
            const e = buf[n + D ..][0..D];
            try prequant(k, w, v, e, .{});
            if (quantRowOk(.{ .fp4 = k.p.rotate }, v, e, qw(k.p, D), got[w * D ..][0..D], &k.msg)) |why| return why;
        }
        return null;
    }
};

fn emitTg(d: usize, rotate: bool) u32 {
    const rd = 64;
    const qw = if (rotate) d else d - rd;
    const group: usize = if (rotate) 32 else 64;
    return @intCast(4 * (d + (if (rotate) d else 1) + tgFor(d) + @max(qw / group, 1)));
}

// ── MoE gate+up: one token's routed gate and up GEMVs over affine banks, the clipped SwiGLU from a sigmoid table ──

const MoeGateUp = struct {
    pub const Params = struct { bits: u32, gs: u32, experts: usize = 8, limit: f64 = 10 };

    pub fn geometry(p: Params, s: Shape) ?metal.Geometry {
        if (p.bits != 2 and p.bits != 4 and p.bits != 8) return null;
        if (p.gs % (32 / p.bits) != 0 or s.k % p.gs != 0 or s.n % 8 != 0 or s.n == 0 or s.rows == 0) return null;
        var g = metal.Geometry{ .grid = .{ 32, ci(s.n), ci(s.rows) }, .tg = .{ 32, 8, 1 } };
        g.out(&.{ ci(s.rows), ci(s.n) }, .bfloat16);
        g.dtype("T", .bfloat16);
        g.int("GS", p.gs);
        g.int("BITS", p.bits);
        g.int("KS", s.k);
        g.int("NS", s.n);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const p = k.p;
        const N = k.shape.n;
        const K = k.shape.k;
        const E = p.experts;
        const words = K * p.bits / 32;
        const x = try k.add(.bfloat16, &.{K});
        var banks: [6][]f64 = undefined;
        for (0..2) |b| {
            banks[3 * b] = try k.add(.uint32, &.{ E, N, words });
            banks[3 * b + 1] = try k.add(.bfloat16, &.{ E, N, K / p.gs });
            banks[3 * b + 2] = try k.add(.bfloat16, &.{ E, N, K / p.gs });
        }
        const inds = try k.add(.int32, &.{k.shape.rows});
        const sigtab = try k.add(.float32, &.{65536});
        const lim = try k.add(.float32, &.{1});
        kit.uniform(rng, x, -2, 2);
        const levels: u32 = @as(u32, 1) << @intCast(p.bits);
        for (0..2) |b| {
            for (banks[3 * b]) |*wd| {
                var word: u64 = 0;
                for (0..32 / p.bits) |i| word |= @as(u64, rng.uintLessThan(u32, levels)) << @intCast(i * p.bits);
                wd.* = @floatFromInt(word);
            }
            for (banks[3 * b + 1], banks[3 * b + 2]) |*sc, *bi| {
                sc.* = (0.5 + rng.float(f64)) * 0.04 / @as(f64, @floatFromInt(levels));
                bi.* = -sc.* * @as(f64, @floatFromInt(levels / 2));
            }
        }
        // Distinct experts, as the router picks them.
        var used: [64]bool = @splat(false);
        for (inds) |*v| {
            var e = rng.uintLessThan(usize, E);
            while (used[e]) e = (e + 1) % E;
            used[e] = true;
            v.* = @floatFromInt(e);
        }
        for (sigtab, 0..) |*sv, i| {
            const g: f64 = spec.fromBf16(@intCast(i));
            sv.* = 1 / (1 + @exp(-g));
        }
        lim[0] = p.limit;
        switch (f) {
            .random => {},
            // A NaN or infinite group scale in one row of slots 0-3 (gate, gate, up, both); the rest stay finite.
            .nonfinite => {
                const G = K / p.gs;
                const put = [_][3]f64{ .{ 0, nan, 0 }, .{ 1, inf, 0 }, .{ 2, 0, -inf }, .{ 3, inf, -inf } };
                for (put) |q| {
                    const e: usize = @intFromFloat(q[0]);
                    if (e >= inds.len) continue;
                    const row = @as(usize, @intFromFloat(inds[e])) * N + e % N;
                    if (q[1] != 0) banks[1][row * G] = q[1];
                    if (q[2] != 0) banks[4][row * G] = q[2];
                }
            },
            .stress => {
                // Weights q - 2^(bits-1) (scale 1, bias -2^(bits-1), at most 31), positive on the first half and
                // mirrored onto a shuffle of the second (so no summation order cancels its own roundings); x's halves
                // (integers to 256, exact in bf16) equal but for 8 entries one apart: partial sums past fp16's range
                // and bf16's resolution and below 2^24, a small exact integer at the end, every other stage exact.
                const h = K / 2;
                const mid: f64 = @floatFromInt(levels / 2);
                const perm = try k.scratch().alloc(usize, h);
                defer k.scratch().free(perm);
                for (perm, 0..) |*q, i| q.* = i;
                rng.shuffle(usize, perm);
                for (0..h) |i| {
                    x[i] = @floatFromInt(rng.intRangeAtMost(i32, 1, if (i < 8) 128 else 256));
                    x[h + perm[i]] = x[i] + @as(f64, if (i < 8) 1 else 0);
                }
                for (0..2) |b| {
                    @memset(banks[3 * b + 1], 1);
                    @memset(banks[3 * b + 2], -mid);
                    const vpw = 32 / p.bits;
                    for (0..E * N) |row| {
                        var codes: [8192]u32 = undefined;
                        for (0..h) |i| {
                            const top: i32 = @min(@as(i32, @intCast(levels / 2 - 1)), 31);
                            const c = rng.intRangeAtMost(i32, 1, if (i < 8) @min(top, 3) else top);
                            codes[i] = @intCast(c + @as(i32, @intCast(levels / 2)));
                            codes[h + perm[i]] = @intCast(-c + @as(i32, @intCast(levels / 2)));
                        }
                        for (0..K / vpw) |wi| {
                            var word: u64 = 0;
                            for (0..vpw) |i| word |= @as(u64, codes[wi * vpw + i]) << @intCast(i * p.bits);
                            banks[3 * b][row * (K / vpw) + wi] = @floatFromInt(word);
                        }
                    }
                }
            },
        }
    }

    pub fn count(_: Params, s: Shape) usize {
        return s.rows * s.n;
    }

    fn code(word: f64, i: usize, bits: u32) f64 {
        const w: u64 = @intFromFloat(word);
        return @floatFromInt((w >> @intCast(i * bits)) & ((@as(u64, 1) << @intCast(bits)) - 1));
    }

    /// Slot `e`'s row `n` of bank `b` (0 gate, 1 up) dotted with x: the sum, and its terms' magnitude.
    fn dot(k: anytype, b: usize, e: usize, n: usize, sum: Sum) ![2]f64 {
        const p = k.p;
        const K = k.shape.k;
        const N = k.shape.n;
        const vpw = 32 / p.bits;
        const eid: usize = @intFromFloat(k.in(7)[e]);
        const wq = k.in(1 + 3 * b);
        const sc = k.in(2 + 3 * b);
        const bi = k.in(3 + 3 * b);
        const t = try k.scratch().alloc(f64, K);
        defer k.scratch().free(t);
        var mag: f64 = 0;
        for (0..K) |kk| {
            const row = eid * N + n;
            const w = code(wq[row * (K / vpw) + kk / vpw], kk % vpw, p.bits) * sc[row * (K / p.gs) + kk / p.gs] + bi[row * (K / p.gs) + kk / p.gs];
            t[kk] = k.in(0)[kk] * w;
            mag += @abs(t[kk]);
        }
        return .{ sum.of(t), mag };
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        const N = k.shape.n;
        const lim = k.in(9)[0];
        const sigtab = k.in(8);
        for (0..k.shape.rows) |e| for (0..N) |n| {
            const g = try dot(k, 0, e, n, sum);
            const up = try dot(k, 1, e, n, sum);
            const gt = spec.roundTo(.bf16, g[0]);
            const ut = spec.roundTo(.bf16, up[0]);
            const gc = kit.minN(gt, lim);
            const uc = if (std.math.isNan(ut)) nan else std.math.clamp(ut, -lim, lim);
            const sig = if (std.math.isNan(gc)) nan else sigtab[spec.toBf16(@floatCast(gc))];
            out[e * N + n] = gc * sig * uc;
            // Each sum's error reaches y scaled by y's slope in it.
            mag[e * N + n] = g[1] * @abs(sig * uc) * 2 + up[1] * @abs(gc * sig);
        };
    }
};

// ── registrations ──

fn depthK(s: Shape) usize {
    return s.k;
}
fn depthK1(s: Shape) usize {
    return s.k + 1;
}
fn depthK2(s: Shape) usize {
    return s.k + 2;
}
/// The collapse's sum carries pre's own error (a sigmoid of an argument up to 6) as 40 more roundings.
fn depthHcY(s: Shape) usize {
    return s.k + 40;
}
fn depthNone(_: Shape) usize {
    return 1;
}

const hcpost_launch = metal.Launch{ .name = "dsv4_hc_post", .inputs = &.{ "pk", "stream_in", "out_v" }, .outputs = &.{"ns"}, .source = dsv4.HCPOST_KERNEL_SOURCE, .tg_bytes = 0, .m1 = 1024, .m2 = 1024 };
const sinkhorn_launch = metal.Launch{ .name = "dsv4_sinkhorn", .inputs = &.{ "mixes", "ssq", "scale", "base", "consts" }, .outputs = &.{"out"}, .source = dsv4.SINKHORN_KERNEL_SOURCE, .tg_bytes = 0, .m1 = 704, .m2 = 704 };
const sinkhorn_y_launch = metal.Launch{ .name = "dsv4_sinkhorn_y", .inputs = &.{ "mixes", "ssq", "scale", "base", "consts", "stream_in" }, .outputs = &.{ "out", "y" }, .source = dsv4.SINKHORN_Y_KERNEL_SOURCE, .tg_bytes = 4 * HC, .m1 = 640, .m2 = 640 };
const sink_softmax_launch = metal.Launch{ .name = "dsv4_sink_softmax", .inputs = &.{ "scores", "sink", "consts", "tk_size" }, .outputs = &.{"y"}, .source = dsv4.SINK_SOFTMAX_KERNEL_SOURCE, .tg_bytes = 4 * 256, .m1 = 1024, .m2 = 1024 };
const dec_chain_in: []const [*:0]const u8 = &.{ "x", "w", "cosr", "sinr", "hada", "consts" };
fn decChainLaunch(comptime d: usize, comptime post: u8) metal.Launch {
    return .{ .name = "dsv4_dec_chain", .inputs = dec_chain_in, .outputs = &.{"y_out"}, .source = dsv4.DEC_CHAIN_KERNEL_SOURCE, .tg_bytes = decChainTg(d, post), .m1 = 1024, .m2 = 1024 };
}
const dec_q = decChainLaunch(512, 0);
const dec_kv = decChainLaunch(512, 1);
const dec_idx = decChainLaunch(128, 2);
const dec_o = decChainLaunch(512, 0);
const emit_in: []const [*:0]const u8 = &.{ "cin", "pre_kv", "pre_sc", "ape", "nw", "cosr", "sinr", "hada", "consts" };
fn emitLaunch(comptime d: usize, comptime rotate: bool) metal.Launch {
    return .{ .name = "dsv4_emit_win", .inputs = emit_in, .outputs = &.{"outw"}, .source = dsv4.EMIT_WIN_KERNEL_SOURCE, .tg_bytes = emitTg(d, rotate), .m1 = 1024, .m2 = 1024 };
}
const emit_kv4 = emitLaunch(512, false);
const emit_kv128 = emitLaunch(512, false);
const emit_idx = emitLaunch(128, true);
const moe_in: []const [*:0]const u8 = &.{ "x", "wg_q", "g_scales", "g_biases", "wu_q", "u_scales", "u_biases", "inds", "sigtab", "consts" };
fn moeLaunch() metal.Launch {
    return .{ .name = "dsv4_moe_gateup", .inputs = moe_in, .outputs = &.{"y"}, .source = dsv4.MOE_GATEUP_KERNEL_SOURCE, .tg_bytes = 0, .m1 = 928, .m2 = 928 };
}
const moe_2 = moeLaunch();
const moe_4 = moeLaunch();
const moe_8 = moeLaunch();

fn hcShapes(comptime k: usize) [hc_rows.len]Shape {
    var out: [hc_rows.len]Shape = undefined;
    for (hc_rows, &out) |r, *o| o.* = .{ .rows = r, .n = DIM, .k = k };
    return out;
}
const hc_shapes = hcShapes(HC);

/// Decode attention: 64 heads over the window (128) plus up to the indexer's 512 picks; the oracle miniature's 4.
const sink_shapes: []const Shape = &.{ .{ .rows = 64, .n = 1, .k = 128 }, .{ .rows = 64, .n = 1, .k = 640 }, .{ .rows = 64, .n = 1, .k = 129 }, .{ .rows = 4, .n = 1, .k = 33 } };
const softmax_sums: spec.Sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "positive exp terms: the sum cannot cancel" };

pub fn register(set: *kc.Set) !void {
    try kit.add(set, HcPost, &hcpost_launch, .{}, .{
        .name = "dsv4 hc_post",
        .accum = .f32,
        .depth = depthK1,
        .out = .f32,
        .tiles = .{ .{}, .{ .size = 256 }, .{} },
        .served = &hc_shapes,
    });
    const sinkhorn_ops = 5 * (2 * ITERS - 1) + 25;
    const short_sums: spec.Sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "four positive terms a sum, inside 20 normalizations" };
    try kit.add(set, Sinkhorn, &sinkhorn_launch, .{}, .{
        .name = "dsv4 sinkhorn",
        .accum = .f32,
        .depth = depthK,
        .ops = sinkhorn_ops,
        .out = .f32,
        .tiles = .{ .{ .size = 32 }, .{}, .{} },
        .served = &hc_shapes,
        .sums = short_sums,
    });
    try kit.add(set, SinkhornY, &sinkhorn_y_launch, .{}, .{
        .name = "dsv4 sinkhorn_y",
        .accum = .f32,
        .depth = depthHcY,
        .ops = sinkhorn_ops,
        .out = .f32,
        .tiles = .{ .{}, .{ .size = 256 }, .{} },
        .served = &hc_shapes,
        .sums = short_sums,
    });
    try kit.add(set, SinkSoftmax, &sink_softmax_launch, .{}, .{
        .name = "dsv4 sink_softmax",
        .accum = .f32,
        .depth = depthK,
        .ops = 12,
        .out = .f32,
        .tiles = .{ .{}, .{}, .{ .size = 256 } },
        .served = sink_shapes,
        .sums = softmax_sums,
    });
    const squares: spec.Sums = .{ .f16 = 0, .bf16 = 0, .why = "a sum of squares cannot cancel" };
    const quant_nan: spec.Nan = .{ .stated = "classes as float64 gives them; quantized entries within the quantizer's envelope of their bounds" };
    const nan_code: spec.Known = .{ .gate = .nonfinite, .finding = "a NaN in a quantized group is written as the code -max x scale: metal::max drops it from the group max and metal::clamp from the code (float64 and the composed chain propagate it)" };
    try kit.add(set, DecChain, &dec_q, .{ .rms = true, .post = 0, .inv = false }, .{
        .name = "dsv4 dec_chain q (rms, rope)",
        .accum = .f32,
        .depth = depthK,
        .out = .f32,
        .nan = quant_nan,
        .served = &.{ .{ .rows = 64, .n = 512, .k = 512 }, .{ .rows = 4, .n = 96, .k = 96 } },
        .sums = squares,
    });
    try kit.add(set, DecChain, &dec_kv, .{ .rms = true, .post = 1, .inv = false }, .{
        .name = "dsv4 dec_chain kv (rms, rope, fp8 head)",
        .accum = .f32,
        .depth = depthK,
        .out = .f32,
        .nan = quant_nan,
        .known = nan_code,
        .served = &.{.{ .rows = 1, .n = 512, .k = 512 }},
        .sums = squares,
    });
    try kit.add(set, DecChain, &dec_idx, .{ .rms = false, .post = 2, .inv = false }, .{
        .name = "dsv4 dec_chain indexer q (rope, Hadamard, fp4)",
        .accum = .f32,
        .depth = depthK,
        .out = .f32,
        .nan = quant_nan,
        .known = nan_code,
        .served = &.{.{ .rows = 64, .n = 128, .k = 128 }},
    });
    try kit.add(set, DecChain, &dec_o, .{ .rms = false, .post = 0, .inv = true }, .{
        .name = "dsv4 dec_chain o (inverse rope)",
        .accum = .none,
        .depth = depthNone,
        .out = .f32,
        .nan = quant_nan,
        .served = &.{.{ .rows = 64, .n = 512, .k = 512 }},
    });
    try kit.add(set, EmitWin, &emit_kv4, .{ .overlap = true, .rotate = false, .pre_n = 5, .base_mod = 1, .col_off = 0 }, .{
        .name = "dsv4 emit_win kv ratio 4 (overlap, fp8)",
        .accum = .f32,
        .depth = depthK,
        .out = .f32,
        .nan = quant_nan,
        .known = nan_code,
        .served = &.{ .{ .rows = 1, .n = 512, .k = 4 }, .{ .rows = 2, .n = 512, .k = 4 }, .{ .rows = 128, .n = 512, .k = 4 } },
    });
    try kit.add(set, EmitWin, &emit_kv128, .{ .overlap = false, .rotate = false, .pre_n = 100, .base_mod = 100, .col_off = 0 }, .{
        .name = "dsv4 emit_win kv ratio 128 (fp8)",
        .accum = .f32,
        .depth = depthK,
        .out = .f32,
        .nan = quant_nan,
        .known = nan_code,
        .served = &.{ .{ .rows = 1, .n = 512, .k = 128 }, .{ .rows = 4, .n = 512, .k = 128 } },
    });
    try kit.add(set, EmitWin, &emit_idx, .{ .overlap = true, .rotate = true, .pre_n = 5, .base_mod = 1, .col_off = 2048 }, .{
        .name = "dsv4 emit_win indexer ratio 4 (overlap, Hadamard, fp4)",
        .accum = .f32,
        .depth = depthK,
        .out = .f32,
        .nan = quant_nan,
        .known = nan_code,
        .served = &.{ .{ .rows = 1, .n = 128, .k = 4 }, .{ .rows = 128, .n = 128, .k = 4 } },
    });
    // 64 of the 2048 bank rows: each row is the same GEMV at its own index; the reduction (4096) stays the release's.
    const moe_shapes: []const Shape = &.{ .{ .rows = 6, .n = 64, .k = 4096 }, .{ .rows = 2, .n = 64, .k = 64 } };
    inline for (.{ .{ &moe_2, 2, 64 }, .{ &moe_4, 4, 64 }, .{ &moe_8, 8, 64 } }) |b| {
        try kit.add(set, MoeGateUp, b[0], .{ .bits = b[1], .gs = b[2] }, .{
            .name = std.fmt.comptimePrint("dsv4 moe_gateup {d}-bit g{d}", .{ b[1], b[2] }),
            .accum = .f32,
            .depth = depthK2,
            // The activation's two f32 products; a bf16 round of the gate or up sum one ulp apart moves y by up to
            // two ulps each, and the table's value by one more.
            .ops = 2,
            .mids = 6,
            .out = .bf16,
            .tiles = .{ .{}, .{ .size = 8, .step = 8, .min = 8 }, .{} },
            .served = moe_shapes,
            .sums = .{ .f16 = 30, .bf16 = 30, .why = "a wrong sum's activation tends to 0, so a miss tops out near 1 / the bound (36x)" },
            .known = .{ .gate = .nonfinite, .finding = "a NaN gate sum becomes the clip limit: metal::min drops NaN where MLX's minimum (the composed chain) propagates it" },
        });
    }
}


// ── tests ──

/// The shipped wrapper's output bytes for a prepared case (its launch on the same inputs), or null where the wrapper
/// needs a compressor's live state (the window emission).
fn wrapperBytes(comptime D: type, c: *kit.Case(D)) !?[]const u8 {
    const s = c.rig.stream();
    const x = c.args;
    const S = c.shape.rows;
    var outs: [2]mlx.mlx_array = @splat(.{ .ctx = null });
    defer for (outs) |o| if (o.ctx != null) {
        _ = mlx.mlx_array_free(o);
    };
    var m: dsv4.Dsv4Model = undefined;
    m.s = s;
    switch (D) {
        HcPost => {
            const sk = dsv4.buildHcPostKernel(HC, c.shape.n).?;
            defer _ = mlx.mlx_fast_metal_kernel_free(sk.kernel);
            defer _ = mlx.mlx_fast_metal_kernel_config_free(sk.cfg);
            const cfg = dsv4.hcPostCfg(HC, S, c.shape.n).?;
            defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
            outs[0] = try dsv4.applyHcPost(&sk, cfg, x[0], x[1], x[2], s);
        },
        Sinkhorn => {
            const sk = dsv4.buildSinkhornKernel(HC, ITERS).?;
            defer _ = mlx.mlx_fast_metal_kernel_free(sk.kernel);
            defer _ = mlx.mlx_fast_metal_kernel_config_free(sk.cfg);
            const cfg = dsv4.sinkhornCfg(HC, ITERS, S).?;
            defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
            outs[0] = try dsv4.applySinkhorn(&sk, cfg, x[0], x[1], x[2], x[3], x[4], s);
        },
        SinkhornY => {
            const sk = dsv4.buildSinkhornYKernel(HC, ITERS, c.shape.n).?;
            defer _ = mlx.mlx_fast_metal_kernel_free(sk.kernel);
            defer _ = mlx.mlx_fast_metal_kernel_config_free(sk.cfg);
            const cfg = dsv4.sinkhornYCfg(HC, ITERS, S, c.shape.n).?;
            defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
            const r = try dsv4.applySinkhornY(&sk, cfg, x[0], x[1], x[2], x[3], x[4], x[5], s);
            outs = .{ r.pk, r.y };
        },
        SinkSoftmax => {
            dsv4.sinkSoftmaxSetForTest(true);
            defer dsv4.sinkSoftmaxSetForTest(null);
            outs[0] = (try dsv4.sinkSoftmaxKernel(&m, x[0], x[1], S, c.shape.k, @floatCast(c.in(2)[0]))) orelse return error.Declined;
        },
        DecChain => {
            dsv4.decChainSetForTest(true);
            defer dsv4.decChainSetForTest(null);
            m.eps = @floatCast(c.in(5)[0]);
            m.hada_g = if (c.p.post == 2) x[4] else null;
            const rr: dsv4.RopeRows = .{ .cos = x[2], .sin = x[3] };
            outs[0] = (try dsv4.decChainKernel(&m, x[0], S, c.shape.n, c.p.rd, if (c.p.rms) x[1] else null, c.p.post, c.p.inv, &rr)) orelse return error.Declined;
        },
        MoeGateUp => {
            dsv4.moeGateUpSetForTest(true);
            defer dsv4.moeGateUpSetForTest(null);
            m.swiglu_limit = @floatCast(c.in(9)[0]);
            const p = c.p;
            const gate: dsv4.Q = .{ .w = x[1], .s = x[2], .b = x[3], .qp = .{ .bits = p.bits, .group_size = p.gs } };
            const up: dsv4.Q = .{ .w = x[4], .s = x[5], .b = x[6], .qp = .{ .bits = p.bits, .group_size = p.gs } };
            outs[0] = (try dsv4.moeGateUpFused(&m, x[0], &gate, &up, x[7], S)) orelse return error.Declined;
        },
        else => return null,
    }
    return try c.rig.bytesOf(if (outs[1].ctx != null) &outs else outs[0..1]);
}

fn expectWrapper(comptime D: type, launch: *const metal.Launch, p: D.Params, shape: Shape) !void {
    const a = std.testing.allocator;
    var c = kit.Case(D).init(a, launch, p);
    defer c.deinit();
    try std.testing.expect(try c.prepare(shape, .random, 7));
    try c.run();
    const ours = try a.dupe(u8, c.rig.outputBytes());
    defer a.free(ours);
    const theirs = (try wrapperBytes(D, &c)) orelse return;
    try std.testing.expectEqualSlices(u8, theirs, ours);
}

test "kernel contract dsv4: each declared launch is the shipped wrapper's, byte for byte" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    try expectWrapper(HcPost, &hcpost_launch, .{}, .{ .rows = 2, .n = 300, .k = HC });
    try expectWrapper(Sinkhorn, &sinkhorn_launch, .{}, .{ .rows = 3, .n = DIM, .k = HC });
    try expectWrapper(SinkhornY, &sinkhorn_y_launch, .{}, .{ .rows = 2, .n = 300, .k = HC });
    try expectWrapper(SinkSoftmax, &sink_softmax_launch, .{}, .{ .rows = 4, .n = 1, .k = 33 });
    try expectWrapper(DecChain, &dec_q, .{ .rms = true, .post = 0, .inv = false }, .{ .rows = 4, .n = 512, .k = 512 });
    try expectWrapper(DecChain, &dec_kv, .{ .rms = true, .post = 1, .inv = false }, .{ .rows = 1, .n = 512, .k = 512 });
    try expectWrapper(DecChain, &dec_idx, .{ .rms = false, .post = 2, .inv = false }, .{ .rows = 4, .n = 128, .k = 128 });
    try expectWrapper(DecChain, &dec_o, .{ .rms = false, .post = 0, .inv = true }, .{ .rows = 4, .n = 512, .k = 512 });
    try expectWrapper(MoeGateUp, &moe_4, .{ .bits = 4, .gs = 64 }, .{ .rows = 2, .n = 64, .k = 128 });
}
