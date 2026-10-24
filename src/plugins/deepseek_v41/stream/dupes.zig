//! The streamed half of the DeepSeek-V4.1 dupes parity (`dupes_test.zig`, main test module): this package's trunk at the
//! stock tier (every lever off: MLX ops only, so it runs on any stream and on the tiny geometry the shape-pinned
//! kernels refuse) over a pack's residents, its routed experts a stand-in over the pack's stacked affine banks
//! (`layers.{l}.ffn.experts.{gate,up,down}_proj`, the resident arch's) handed back as the expert source hands them:
//! unweighted `[n, k, dim]` f32, the clamped SwiGLU in f32 as the stock chain runs it.

const std = @import("std");
const builtin = @import("builtin");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const v41 = @import("deepseek_v41.zig");
const ops = @import("../../../sdk/weights/streamed/ops.zig");
const mdl = @import("model.zig");
const routes = @import("routes.zig");
const graph = @import("graph.zig");

comptime {
    if (!builtin.is_test) @compileError("deepseek_v41 dupes.zig is a parity harness; served code never imports it");
}

const T = mlx.mlx_array;

/// The routed stand-in: one layer's stacked affine banks, read by `gather_qmm` (unsorted, as the resident decode does).
const Routed = struct {
    w: *const sdk.Weights,
    swiglu_limit: f32,
    layer: u32 = 0,

    pub fn at(self: Routed, l: u32) Routed {
        var r = self;
        r.layer = l;
        return r;
    }

    fn bank(self: Routed, g: *ops.MlxOps, x: T, proj: []const u8, idx: T) !T {
        var b: [96]u8 = undefined;
        const get = struct {
            fn f(w: *const sdk.Weights, buf: []u8, l: u32, p: []const u8, suffix: []const u8) !T {
                return w.get(try std.fmt.bufPrint(buf, "layers.{d}.ffn.experts.{s}.{s}", .{ l, p, suffix })) orelse error.MissingWeight;
            }
        }.f;
        const wt = try get(self.w, &b, self.layer, proj, "weight");
        const sc = try get(self.w, &b, self.layer, proj, "scales");
        const bi = try get(self.w, &b, self.layer, proj, "biases");
        const in: usize = @intCast(mlx.getShape(x)[mlx.getShape(x).len - 1]);
        const words: usize = @intCast(mlx.getShape(wt)[2]);
        const groups: usize = @intCast(mlx.getShape(sc)[2]);
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_gather_qmm(&r, x, wt, sc, bi, .{}, idx, true, mlx.mlx_optional_int.some(@intCast(in / groups)), mlx.mlx_optional_int.some(@intCast(words * 32 / in)), "affine", false, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.adopt(r);
    }

    pub fn routed(self: Routed, g: *ops.MlxOps, xf: T, indices: T) !T {
        const n = g.shapeOf(xf).dim(0);
        const k = g.shapeOf(indices).dim(1);
        const d = g.shapeOf(xf).dim(1);
        const dt = g.dtypeOf(xf);
        const x4 = try g.reshape(xf, &.{ n, 1, 1, d });
        const idx = try g.astype(indices, .uint32);
        var gate = try g.astype(try self.bank(g, x4, "gate_proj", idx), .float32);
        var up = try g.astype(try self.bank(g, x4, "up_proj", idx), .float32);
        if (self.swiglu_limit > 0) {
            up = try g.clip(up, try g.scalar(-self.swiglu_limit, .float32), try g.scalar(self.swiglu_limit, .float32));
            gate = try g.minimum(gate, try g.scalar(self.swiglu_limit, .float32));
        }
        const act = try g.astype(try g.mul(try g.silu(gate), up), dt);
        const dn = try self.bank(g, act, "down_proj", idx);
        return g.astype(try g.reshape(dn, &.{ n, k, d }), .float32);
    }
};

/// One forward over `ids` from a fresh state per call in `calls` (rows of `ids` each, in order): every call's last
/// row of logits, host f32 `[calls.len * vocab]`. The pack's config.json and shards in `dir`; no Engram, no DSpark.
pub fn logits(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, ids: []const u32, calls: []const u32, s: mlx.mlx_stream) ![]f32 {
    var d = try std.Io.Dir.openDirAbsolute(io, dir, .{});
    defer d.close(io);
    const text = try d.readFileAlloc(io, "config.json", gpa, .limited(1 << 20));
    defer gpa.free(text);
    var diag: v41.Diag = .{};
    const c = v41.Config.parse(gpa, text, &diag) catch |e| {
        std.debug.print("dsv41 dupes: {s}\n", .{diag.message()});
        return e;
    };
    if (c.engram.n_layers != 0 or c.dspark.n_stages != 0) return error.DupesPackHasEngramOrDspark;
    var weights = try sdk.loader.dir(io, gpa, dir, .{});
    defer weights.deinit();
    var g = try ops.MlxOps.init(gpa, s);
    defer g.deinit();
    const M = mdl.Model(ops.MlxOps);
    const m = try M.init(gpa, &g, c, routes.stock, &weights, null);
    defer m.deinit(&g);
    var st = try m.newState();
    defer st.deinit(&g, gpa);
    const vocab = c.vocab_size;
    const out = try gpa.alloc(f32, calls.len * vocab);
    errdefer gpa.free(out);
    const hook: Routed = .{ .w = &weights, .swiglu_limit = @floatCast(c.swiglu_limit) };
    var at: usize = 0;
    for (calls, 0..) |n, i| {
        const r = try m.forward(&g, &st, ids[at..][0..n], .{ .logits = .last }, hook, graph.NoProbe{});
        const row = try g.astype(try g.reshape(r.logits.?, &.{@intCast(vocab)}), .float32);
        _ = try g.hostF32(row, out[i * vocab ..][0..vocab]);
        g.reset();
        at += n;
    }
    return out;
}

// ── the EXL3 source ──

const testing = std.testing;
const streamed = @import("streamed.zig");
const Exl3 = streamed.Exl3;
const expert_bank = @import("../../../sdk/weights/streamed/expert/bank.zig");
const G = ops.MlxOps;

/// A host array of `shape` from `vals` (`dt`: f32, i32 or u32 words).
fn hostArray(vals: anytype, shape: []const c_int, dt: mlx.mlx_dtype) T {
    return mlx.mlx_array_new_data(vals.ptr, shape.ptr, @intCast(shape.len), dt);
}

fn hostBytes(g: *G, x: T) ![]const u8 {
    const f = try g.astype(x, .float32);
    try g.evalAll(&.{f});
    const n: usize = @intCast(mlx.mlx_array_size(f));
    return std.mem.sliceAsBytes((mlx.mlx_array_data_float32(f) orelse return error.NoData)[0..n]);
}

/// The EXL3 arm of `streamed.zig` against the stream module's expert path (`dupes_test.zig` runs it on the device): a
/// 5-layer, 4-expert synthetic EXL3 bank at the release's 5120 / 2304 (the texts are compiled for them), its routed
/// sums through the source (the resident's bf16 input, u32 routes, f32 weights) bit for bit the hook's `routed` then
/// `graph.moeCombine`'s routed half, on decode-lane and wide calls, the reference taking them in reverse order.
pub fn exl3Parity(s: mlx.mlx_stream) !void {
    const a = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    // The EXL3 texts are compiled for the release's 5120 / 2304: the release's shapes, 4 experts on 5 layers.
    const image = try expert_bank.writeSynth(a, &tmp, .{ .hidden = 5120, .inter = 2304, .n_experts = 4, .k = &.{ 3, 3, 3, 3, 3 }, .finite_scales = true });
    defer a.free(image);
    const mini = try v41.testConfigJson(a, .mini);
    defer a.free(mini);
    const h = try std.mem.replaceOwned(u8, a, mini, "\"hidden_size\":64,", "\"hidden_size\":5120,");
    defer a.free(h);
    const cfg = try std.mem.replaceOwned(u8, a, h, "\"moe_intermediate_size\":32,", "\"moe_intermediate_size\":2304,");
    defer a.free(cfg);
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = cfg });
    var rbuf: [512]u8 = undefined;
    const dir = try expert_bank.tmpRoot(&tmp, &rbuf);
    const implemented: expert_bank.Implemented = .{ .codebooks = &.{"mul1"}, .k = &.{3}, .hidden = 5120, .inter = 2304, .n_experts = 4, .n_layers = 5 };

    // Decode-lane calls (n * top_k <= 48) and wide ones, on three layers; the reference (a second source: the read pool
    // is one per process) takes them in reverse, so a call meets other slots and other hits there.
    const Call = struct { layer: u32, n: usize };
    const calls = [_]Call{ .{ .layer = 0, .n = 1 }, .{ .layer = 2, .n = 3 }, .{ .layer = 4, .n = 8 }, .{ .layer = 0, .n = 24 }, .{ .layer = 2, .n = 30 }, .{ .layer = 4, .n = 1 } };
    var got: [calls.len][]const u8 = undefined;
    var want: [calls.len][]const u8 = undefined;
    defer for (got ++ want) |b| a.free(b);
    for (0..2) |pass| {
        const src = try Exl3.open(a, io, dir, s, .{ .implemented = implemented });
        defer src.deinit();
        for (0..calls.len) |j| {
            const i = if (pass == 0) j else calls.len - 1 - j;
            const c = calls[i];
            const n: c_int = @intCast(c.n);
            const xs = try a.alloc(f32, c.n * 5120);
            defer a.free(xs);
            const wv = try a.alloc(f32, c.n * 2);
            defer a.free(wv);
            const ids = try a.alloc(u32, c.n * 2);
            defer a.free(ids);
            var rng = std.Random.DefaultPrng.init(i);
            for (xs) |*v| v.* = rng.random().floatNorm(f32);
            for (wv) |*v| v.* = rng.random().float(f32);
            for (0..c.n) |r| {
                const e0: u32 = @intCast((r * 3 + c.layer) % 4);
                ids[2 * r] = e0;
                ids[2 * r + 1] = (e0 + 1 + @as(u32, @intCast(r % 3))) % 4;
            }
            const x = hostArray(xs, &.{ n, 5120 }, .float32);
            defer _ = mlx.mlx_array_free(x);
            const w = hostArray(wv, &.{ n, 2 }, .float32);
            defer _ = mlx.mlx_array_free(w);
            const idx = hostArray(ids, &.{ n, 2 }, .uint32);
            defer _ = mlx.mlx_array_free(idx);
            if (pass == 0) {
                // As the resident hands them: its bf16 input, u32 routes, f32 weights.
                const g = &src.g;
                const y = try src.rows(c.layer, try g.astype(x, .bfloat16), idx, w);
                defer g.release(y);
                got[i] = try a.dupe(u8, try hostBytes(g, y));
                g.reset();
                try src.settle();
            } else {
                // As `model.zig` drives the hook: i32 routes, then `graph.moeCombine`'s routed half.
                const g = &src.g;
                const ro = try src.arm.hook.at(c.layer).routed(g, try g.astype(x, .bfloat16), try g.astype(idx, .int32));
                const y = try g.sum(try g.mul(try g.astype(ro, .float32), try g.expandDims(w, -1)), -2, false);
                want[i] = try a.dupe(u8, try hostBytes(g, y));
                g.reset();
                try src.settle();
            }
        }
    }
    for (got, want, calls) |gb, wb, c| {
        const gf = std.mem.bytesAsSlice(f32, gb);
        var distinct: usize = 0;
        for (gf, 0..) |v, k| {
            try testing.expect(std.math.isFinite(v));
            if (k > 0 and v != gf[k - 1]) distinct += 1;
        }
        try testing.expect(distinct > c.n * 5120 / 2);
        try testing.expectEqualSlices(u8, wb, gb);
    }
}
