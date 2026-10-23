//! Qwen-Image's kernel contract: the DiT's fused interleaved-pair RoPE (`kernels.applyRope`, opt-in through
//! MLX_SERVE_QWEN_IMAGE_FUSED_ROPE=1), bit-equal to the op chain it replaces (sdk `ropePairs`: f32 cast, the pair
//! rotation in separate f32 roundings, cast to the compute dtype). Listed in `kernel_contract/registry.zig`.
//! Shape: rows = tokens, n = heads, k = head_dim.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const kc = @import("../../sdk/testing/kernel_contract/root.zig");
const kernels = @import("kernels.zig");
const rope_blk = @import("../../sdk/blocks/dit/rope.zig");

const spec = kc.spec;
const metal = kc.metal;
const Shape = spec.Shape;

/// One thread a pair, no threadgroup memory: 256 threads a group fit every family.
const launch: metal.Launch = .{
    .name = "qwen_image21_rope_separate_rounding",
    .inputs = &.{ "x", "cs", "sn" },
    .outputs = &.{"out"},
    .source = kernels.rope_source,
    .tg_bytes = 0,
    .m1 = 1024,
    .m2 = 1024,
};

/// x [1, L, H, D] in `dt` (the engine's f32 q/k after their f32 norm weights), tables [1, L, 1, D/2, 1] f32, out in
/// `out_dt` (the DiT's compute dtype). Host values live in `arena`, one prepare at a time.
const Rope = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    dt: mlx.mlx_dtype,
    out_dt: mlx.mlx_dtype,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    x: []f64 = &.{},
    cs: []f64 = &.{},
    sn: []f64 = &.{},
    args: [3]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, dt: mlx.mlx_dtype, out_dt: mlx.mlx_dtype) Rope {
        return .{ .rig = metal.Rig.init(a), .launch = &launch, .dt = dt, .out_dt = out_dt, .arena = .init(a) };
    }

    pub fn deinit(self: *Rope) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    fn elem(dt: mlx.mlx_dtype) spec.Elem {
        return switch (dt) {
            .float32 => .f32,
            .float16 => .f16,
            else => .bf16,
        };
    }

    /// As `kernels.applyRope` builds it: one thread a pair over the whole array, groups of 256.
    pub fn geometry(self: *const Rope, shape: Shape) ?metal.Geometry {
        if (shape.k == 0 or shape.k % 2 != 0 or shape.rows == 0 or shape.n == 0) return null;
        const pairs = std.math.cast(c_int, shape.rows * shape.n * shape.k / 2) orelse return null;
        var g = metal.Geometry{ .grid = .{ pairs, 1, 1 }, .tg = .{ 256, 1, 1 } };
        g.out(&.{ 1, @intCast(shape.rows), @intCast(shape.n), @intCast(shape.k) }, self.out_dt);
        g.dtype("T", self.out_dt);
        return g;
    }

    pub fn prepare(self: *Rope, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        const L = shape.rows;
        const H = shape.n;
        const D = shape.k;
        const half = D / 2;
        self.x = try a.alloc(f64, L * H * D);
        self.cs = try a.alloc(f64, L * half);
        self.sn = try a.alloc(f64, L * half);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const xe = elem(self.dt);
        for (self.cs, self.sn) |*c, *s| {
            const angle = rng.float(f64) * 2 * std.math.pi;
            c.* = spec.roundTo(.f32, @cos(angle));
            s.* = spec.roundTo(.f32, @sin(angle));
        }
        for (self.x) |*v| v.* = spec.roundTo(xe, rng.float(f64) * 8 - 4);
        switch (fill) {
            .random => {},
            // Each in its own pair (no inf - inf), all in the first rows the gate keeps.
            .nonfinite => {
                const put = [_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64), std.math.inf(f64), -std.math.inf(f64) };
                for (put, 0..) |v, i| self.x[(i * (H * D + 2) + i % 2) % self.x.len] = v;
            },
            // re·c - im·s = a(a+2) - (a+1)² = -1 per table entry: products near 2^15, exact in fp32, that fp16 and
            // bf16 sums round away; re·s + im·c = 2(a+1)² stays inside fp16.
            .stress => for (0..L) |t| for (0..half) |i| {
                const av: f64 = @floatFromInt(120 + rng.uintLessThan(u32, 60));
                self.cs[t * half + i] = av + 2;
                self.sn[t * half + i] = av + 1;
                for (0..H) |h| {
                    const sign: f64 = if (rng.boolean()) 1 else -1;
                    const p = (t * H + h) * D + 2 * i;
                    self.x[p] = sign * av;
                    self.x[p + 1] = sign * (av + 1);
                }
            },
        }
        self.rig.reset();
        const tab = [_]c_int{ 1, @intCast(L), 1, @intCast(half), 1 };
        self.args = .{
            try self.rig.input(self.dt, &.{ 1, @intCast(L), @intCast(H), @intCast(D) }, self.x),
            try self.rig.input(.float32, &tab, self.cs),
            try self.rig.input(.float32, &tab, self.sn),
        };
        return true;
    }

    pub fn run(self: *Rope) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Rope) usize {
        return self.x.len;
    }

    /// Pair p's two terms of each output (re·c, -im·s) and (re·s, im·c).
    fn terms(self: *const Rope, p: usize) [2][2]f64 {
        const H = self.shape.n;
        const half = self.shape.k / 2;
        const t = p / (H * half);
        const c = self.cs[t * half + p % half];
        const s = self.sn[t * half + p % half];
        const re = self.x[2 * p];
        const im = self.x[2 * p + 1];
        return .{ .{ re * c, -(im * s) }, .{ re * s, im * c } };
    }

    pub fn reference(self: *const Rope, want: []f64, mag: []f64) !void {
        for (0..want.len / 2) |p| {
            const tm = self.terms(p);
            for (tm, 0..) |two, j| {
                want[2 * p + j] = two[0] + two[1];
                mag[2 * p + j] = @abs(two[0]) + @abs(two[1]);
            }
        }
    }

    pub fn simulate(self: *const Rope, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        for (0..out.len / 2) |p| {
            const tm = self.terms(p);
            for (tm, 0..) |two, j| out[2 * p + j] = spec.sumIn(acc, order, &two);
        }
    }

    pub fn replaced(self: *Rope) ![]const u8 {
        var g = ops.Graph.init(self.rig.a, self.rig.stream());
        defer g.deinit();
        const tab = [_]c_int{ 1, @intCast(self.shape.rows), 1, @intCast(self.shape.k / 2) };
        const c = try g.reshape(self.args[1], &tab);
        const s = try g.reshape(self.args[2], &tab);
        return self.rig.bytesOf(&.{try rope_blk.ropePairs(&g, self.args[0], c, s, self.out_dt)});
    }

    fn wrapper(self: *Rope) ![]const u8 {
        const out = try kernels.applyRope(self.args[0], self.args[1], self.args[2], self.out_dt, self.rig.stream());
        defer _ = mlx.mlx_array_free(out);
        return self.rig.bytesOf(&.{out});
    }
};

fn depthTwo(_: Shape) usize {
    return 2;
}

/// 1024² and 512² canvases with a 13-token prompt, 32 heads of 128. Heads are the tiled dimension: at 1..3 heads
/// the grid's last group runs past the pairs.
const served: []const Shape = &.{ .{ .rows = 4109, .n = 32, .k = 128 }, .{ .rows = 1037, .n = 32, .k = 128 } };

fn ropeSpec(comptime name: []const u8, out: spec.Elem) spec.Spec {
    return .{
        .name = name,
        .accum = .f32,
        .depth = depthTwo,
        .out = out,
        .replaces = "astype f32 -> pair rotate in separate f32 roundings -> astype (sdk ropePairs)",
        .tiles = .{ .{}, .{ .size = 2 }, .{} },
        .served = served,
    };
}

pub fn register(set: *kc.Set) !void {
    const a = set.a;
    try set.add(Rope, .init(a, .float32, .bfloat16), ropeSpec("qwen_image rope f32 -> bf16", .bf16), null);
    try set.add(Rope, .init(a, .float32, .float16), ropeSpec("qwen_image rope f32 -> f16 (fp16 DiT)", .f16), null);
}

// ── tests ──

test "kernel contract qwen_image: the declared launch is the shipped wrapper's, byte for byte" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var set = kc.Set.init(a);
    defer set.deinit();
    try register(&set);
    for (set.kernels.items) |k| {
        const r: *Rope = @ptrCast(@alignCast(k.ctx));
        try std.testing.expect(try r.prepare(.{ .rows = 37, .n = 3, .k = 128 }, .random, 7));
        try r.run();
        const ours = try a.dupe(u8, r.rig.outputBytes());
        defer a.free(ours);
        try std.testing.expectEqualSlices(u8, try r.wrapper(), ours);
    }
}
