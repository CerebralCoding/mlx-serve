//! A family proof's expected values, recorded from the legacy forward before that forward is deleted: every array a
//! proof compares, in order, with the GPU architecture it ran on. Played back on the same architecture the plugin's
//! values must equal the recording bit for bit; elsewhere (other kernels, other reduction orders) they must agree to
//! 2% relative RMS and greedy token runs are not compared. Test-only.
//!
//! Record: `MLX_SERVE_PROOF_RECORD=<repo root>` with the legacy forward still present writes
//! `<root>/<path>` at `Both.close`.

const std = @import("std");
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
const device = @import("../../../mlx/device.zig");

const testing = std.testing;
const Array = mlx.mlx_array;
const magic = "MSVPROOF1";

/// What runs this process's arrays: the GPU's architecture, or `cpu`.
pub fn deviceName(buf: []u8) []const u8 {
    if (mlx.noGpuBackend()) return "cpu";
    return device.gpuArchitecture(buf) orelse "unknown";
}

/// The repo root to record under, when recording.
pub fn recordRoot() ?[]const u8 {
    const v = env.get(.MLX_SERVE_PROOF_RECORD) orelse return null;
    return std.mem.span(v);
}

/// The bytes of `x`, contiguous and evaluated on the CPU. Owned by the caller.
fn bytesOf(a: std.mem.Allocator, x: Array) ![]u8 {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, x, false, s));
    try mlx.check(mlx.mlx_array_eval(c));
    const n = mlx.mlx_array_size(c) * mlx.mlx_array_itemsize(c);
    const p = mlx.mlx_array_data_uint8(c) orelse return error.NoData;
    return a.dupe(u8, p[0..n]);
}

/// `x` as host f32 values. Owned by the caller.
fn floatsOf(a: std.mem.Allocator, x: Array) ![]f32 {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, x, .float32, s));
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, f, false, s));
    try mlx.check(mlx.mlx_array_eval(c));
    const p = mlx.mlx_array_data_float32(c) orelse return error.NoData;
    return a.dupe(f32, p[0..mlx.mlx_array_size(c)]);
}

const sample_len = 64;

/// XxHash3 of `x`'s dtype, shape and bytes as uint32 `[2]`.
fn digest(x: Array) !Array {
    const bytes = try bytesOf(testing.allocator, x);
    defer testing.allocator.free(bytes);
    var h = std.hash.XxHash3.init(0);
    h.update(std.mem.asBytes(&@backingInt(mlx.mlx_array_dtype(x))));
    h.update(std.mem.sliceAsBytes(mlx.getShape(x)));
    h.update(bytes);
    const v = h.final();
    const words = [2]u32{ @truncate(v), @truncate(v >> 32) };
    return mlx.mlx_array_new_data(&words, &[_]c_int{2}, 1, .uint32);
}

/// `sample_len` elements of `x` evenly spread over its flat order, in its dtype.
fn sample(x: Array) !Array {
    const a = testing.allocator;
    const bytes = try bytesOf(a, x);
    defer a.free(bytes);
    const item = mlx.mlx_array_itemsize(x);
    const n = bytes.len / item;
    const k = @min(n, sample_len);
    const out = try a.alloc(u8, k * item);
    defer a.free(out);
    for (0..k) |i| @memcpy(out[i * item ..][0..item], bytes[(i * n / k) * item ..][0..item]);
    return mlx.mlx_array_new_data(out.ptr, &[_]c_int{@intCast(k)}, 1, mlx.mlx_array_dtype(x));
}

pub const Tape = struct {
    /// Recording: the file under the root, and the bytes so far.
    path: ?[]u8 = null,
    buf: std.ArrayList(u8) = .empty,
    /// Playing back: the recording and the read offset.
    data: []const u8 = &.{},
    at: usize = 0,
    same_device: bool = true,
    /// Arrays over this many bytes are kept as a digest of their dtype, shape and bytes (compared on the recording's
    /// device only) and an even sample of `sample_len` elements; 0 = every array whole. Set alike to record and play.
    compact_over: usize = 0,

    pub fn recorder(root: []const u8, rel: []const u8) !Tape {
        const a = testing.allocator;
        var t: Tape = .{ .path = try std.fs.path.join(a, &.{ root, rel }) };
        var dbuf: [64]u8 = undefined;
        const dev = deviceName(&dbuf);
        try t.buf.appendSlice(a, magic);
        try t.buf.append(a, @intCast(dev.len));
        try t.buf.appendSlice(a, dev);
        return t;
    }

    pub fn player(bytes: []const u8) !Tape {
        if (bytes.len < magic.len + 1 or !std.mem.eql(u8, bytes[0..magic.len], magic)) return error.NotAProofTape;
        const n = bytes[magic.len];
        const recorded = bytes[magic.len + 1 ..][0..n];
        var dbuf: [64]u8 = undefined;
        const same = std.mem.eql(u8, recorded, deviceName(&dbuf));
        if (!same) std.debug.print("proof tape recorded on {s}: comparing to 2% RMS, greedy runs skipped\n", .{recorded});
        return .{ .data = bytes, .at = magic.len + 1 + n, .same_device = same };
    }

    pub fn recording(t: *const Tape) bool {
        return t.path != null;
    }

    pub fn playing(t: *const Tape) bool {
        return t.data.len > 0;
    }

    /// Appends `x` (the legacy forward's value) to the recording.
    pub fn put(t: *Tape, x: Array) !void {
        if (!t.compacts(x)) return t.putWhole(x);
        const d = try digest(x);
        defer _ = mlx.mlx_array_free(d);
        try t.putWhole(d);
        const s = try sample(x);
        defer _ = mlx.mlx_array_free(s);
        try t.putWhole(s);
    }

    fn compacts(t: *const Tape, x: Array) bool {
        return t.compact_over > 0 and mlx.mlx_array_size(x) * mlx.mlx_array_itemsize(x) > t.compact_over;
    }

    fn putWhole(t: *Tape, x: Array) !void {
        const a = testing.allocator;
        const sh = mlx.getShape(x);
        try t.buf.append(a, @intCast(@backingInt(mlx.mlx_array_dtype(x))));
        try t.buf.append(a, @intCast(sh.len));
        for (sh) |d| try t.buf.appendSlice(a, std.mem.asBytes(&@as(i32, d)));
        const bytes = try bytesOf(a, x);
        defer a.free(bytes);
        try t.buf.appendSlice(a, std.mem.asBytes(&@as(u64, bytes.len)));
        try t.buf.appendSlice(a, bytes);
    }

    fn take(t: *Tape, n: usize) ![]const u8 {
        if (t.at + n > t.data.len) return error.ProofTapeShort;
        defer t.at += n;
        return t.data[t.at..][0..n];
    }

    /// `got` against the next recorded value; a greedy run (`tokens`) only on the recording's device.
    pub fn expectNext(t: *Tape, got: Array, tokens: bool) !void {
        if (!t.compacts(got)) return t.expectWhole(got, tokens);
        const d = try digest(got);
        defer _ = mlx.mlx_array_free(d);
        try t.expectWhole(d, true);
        const s = try sample(got);
        defer _ = mlx.mlx_array_free(s);
        try t.expectWhole(s, tokens);
    }

    fn expectWhole(t: *Tape, got: Array, tokens: bool) !void {
        const a = testing.allocator;
        const dt: mlx.mlx_dtype = @fromBackingInt(@intCast((try t.take(1))[0]));
        const nd = (try t.take(1))[0];
        var sh: [8]c_int = undefined;
        for (sh[0..nd]) |*d| d.* = std.mem.readInt(i32, (try t.take(4))[0..4], .little);
        const len = std.mem.readInt(u64, (try t.take(8))[0..8], .little);
        const want = try t.take(@intCast(len));
        try testing.expectEqual(dt, mlx.mlx_array_dtype(got));
        try testing.expectEqualSlices(c_int, sh[0..nd], mlx.getShape(got));
        if (t.same_device) {
            const bytes = try bytesOf(a, got);
            defer a.free(bytes);
            return testing.expectEqualSlices(u8, want, bytes);
        }
        if (tokens) return;
        const w = mlx.mlx_array_new_data(want.ptr, &sh, @intCast(nd), dt);
        defer _ = mlx.mlx_array_free(w);
        const wf = try floatsOf(a, w);
        defer a.free(wf);
        const gf = try floatsOf(a, got);
        defer a.free(gf);
        var err: f64 = 0;
        var ref: f64 = 0;
        for (wf, gf) |x, y| {
            try testing.expect(std.math.isFinite(y));
            err += (x - y) * (x - y);
            ref += x * x;
        }
        try testing.expect(err <= 4e-4 * @max(ref, 1e-12));
    }

    /// The playback read every recorded value.
    pub fn expectDone(t: *const Tape) !void {
        if (t.playing()) try testing.expectEqual(t.data.len, t.at);
    }

    /// Writes the recording.
    pub fn finish(t: *Tape) !void {
        const a = testing.allocator;
        const p = t.path orelse return;
        defer a.free(p);
        defer t.buf.deinit(a);
        const io = std.Io.Threaded.global_single_threaded.io();
        if (std.fs.path.dirname(p)) |d| try std.Io.Dir.cwd().createDirPath(io, d);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = p, .data = t.buf.items });
    }
};

// ── tests ──

test "proof tape: an array past compact_over is kept as a digest and an even sample, and replays" {
    const a = testing.allocator;
    var vals: [1024]f32 = undefined;
    for (&vals, 0..) |*v, i| v.* = @floatFromInt(i);
    const x = mlx.mlx_array_new_data(&vals, &[_]c_int{ 4, 256 }, 2, .float32);
    defer _ = mlx.mlx_array_free(x);
    var r: Tape = .{ .compact_over = 64 };
    defer r.buf.deinit(a);
    var dbuf: [64]u8 = undefined;
    const dev = deviceName(&dbuf);
    try r.buf.appendSlice(a, magic);
    try r.buf.append(a, @intCast(dev.len));
    try r.buf.appendSlice(a, dev);
    try r.put(x);
    try testing.expect(r.buf.items.len < @sizeOf(@TypeOf(vals)) / 2);
    var p = try Tape.player(r.buf.items);
    p.compact_over = 64;
    try p.expectNext(x, false);
    try p.expectDone();
}
