//! ANE prefill (`--ane-prefill`) on the qwen3_5 plugin trunk: the engine's build reads the same layers, weight bytes,
//! GPU rest slices and plan off the served shell as off the legacy shell, and the table's forward hands the offload
//! each dense MLP and GDN q|k|v / z projection at the legacy forward's sites, with the legacy seam's inputs.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const fixture = @import("../../sdk/bases/proof/fixture.zig");
const pack = @import("../../sdk/bases/proof/qwen35_test.zig");
const ane_prefill = @import("../../engines/mlx/ane_prefill.zig");
const cache_ops = @import("../../engines/mlx/kv/cache_ops.zig");
const KVCache = @import("../../engines/mlx/kv/cache.zig").KVCache;
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;
const maxAbsDiffF32 = @import("../../sdk/testing/parity.zig").maxAbsDiffF32;
const served = @import("served_test.zig");
const qwen3_5 = @import("plugin.zig");

const testing = std.testing;
const Array = mlx.mlx_array;
const AneLayer = ane_prefill.AneLayer;

fn openPair(p: *served.Pair, comptime moe: bool, seed: u64) !void {
    var pk = fixture.Pack.init(seed);
    defer pk.deinit();
    try pack.writePack(&pk, moe, true, 8);
    try p.open(&qwen3_5.table, &pk, pack.config(moe, true, 8));
    p.oracle = .{};
}

fn expectHostEqual(x: [2]*Transformer, w: [2][3]Array, in_dim: u32, out_dim: u32) !void {
    const a = testing.allocator;
    const want = try x[0].dequantToHostF32(a, w[0][0], w[0][1], w[0][2], in_dim, out_dim);
    defer a.free(want);
    const got = try x[1].dequantToHostF32(a, w[1][0], w[1][1], w[1][2], in_dim, out_dim);
    defer a.free(got);
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want), std.mem.sliceAsBytes(got));
}

fn expectArraysEqual(want: anytype, got: anytype) !void {
    inline for (want, got) |w, g| {
        try testing.expectEqual(w.ctx == null, g.ctx == null);
        if (w.ctx != null) try ops.expectBitEqual(g, w);
    }
}

/// Both shells' `aneLayers`: the same coverage, dequantized program inputs, rest slices and plan.
fn expectSameInputs(p: *served.Pair, dense: u64, gdn: u64) !void {
    const a = testing.allocator;
    const xs = p.shells();
    const want = (try xs[0].aneLayers(a)).?;
    defer a.free(want);
    const got = (try xs[1].aneLayers(a)) orelse return error.PluginOffersNoAneLayers;
    defer a.free(got);
    try testing.expectEqual(want.len, got.len);
    const cfg = &xs[0].config;
    const plan = ane_prefill.anePlan(cfg, want, 0.25, 1);
    try testing.expectEqual(dense, plan.dense_layers);
    try testing.expectEqual(gdn, plan.gdn_layers);
    try testing.expect(plan.eng_qkv > 0 and plan.eng_z > 0 and (dense == 0 or plan.eng_ffn > 0));
    for ([_]f32{ 0.25, 0.5 }) |share| for ([_]u32{ 1, 2 }) |units|
        try testing.expectEqual(ane_prefill.anePlan(cfg, want, share, units), ane_prefill.anePlan(&xs[1].config, got, share, units));
    const h = cfg.hidden_size;
    const ffn = cfg.intermediate_size;
    for (want, got) |*w, *g| {
        try testing.expectEqual(w.mlp == null, g.mlp == null);
        try testing.expectEqual(w.gdn == null, g.gdn == null);
        if (w.mlp) |*wm| {
            const gm = &g.mlp.?;
            try expectHostEqual(xs, .{ .{ wm.gate_w, wm.gate_s, wm.gate_b }, .{ gm.gate_w, gm.gate_s, gm.gate_b } }, h, ffn);
            try expectHostEqual(xs, .{ .{ wm.up_w, wm.up_s, wm.up_b }, .{ gm.up_w, gm.up_s, gm.up_b } }, h, ffn);
            try expectHostEqual(xs, .{ .{ wm.down_w, wm.down_s, wm.down_b }, .{ gm.down_w, gm.down_s, gm.down_b } }, ffn, h);
            var rests: [2]ane_prefill.AneChanRest = .{ .{ .mlp = try xs[0].aneBuildMlpRest(wm, plan.eng_ffn, ffn) }, .{ .mlp = try xs[1].aneBuildMlpRest(gm, plan.eng_ffn, ffn) } };
            defer for (&rests) |*r| r.deinit();
            const rw = rests[0].mlp.?;
            const rg = rests[1].mlp.?;
            try expectArraysEqual(.{ rw.gate_w, rw.gate_s, rw.gate_b, rw.up_w, rw.up_s, rw.up_b, rw.down_w, rw.down_s, rw.down_b }, .{ rg.gate_w, rg.gate_s, rg.gate_b, rg.up_w, rg.up_s, rg.up_b, rg.down_w, rg.down_s, rg.down_b });
        }
        if (w.gdn) |*wg| {
            const gg = &g.gdn.?;
            try expectHostEqual(xs, .{ .{ wg.qkv_w, wg.qkv_s, wg.qkv_b }, .{ gg.qkv_w, gg.qkv_s, gg.qkv_b } }, h, plan.qkv_out);
            try expectHostEqual(xs, .{ .{ wg.z_w, wg.z_s, wg.z_b }, .{ gg.z_w, gg.z_s, gg.z_b } }, h, plan.z_out);
            var rests: [2]ane_prefill.AneChanRest = .{ .{}, .{} };
            defer for (&rests) |*r| r.deinit();
            try xs[0].aneBuildGdnRest(&rests[0], wg, plan.eng_qkv, plan.eng_z);
            try xs[1].aneBuildGdnRest(&rests[1], gg, plan.eng_qkv, plan.eng_z);
            const rw = rests[0];
            const rg = rests[1];
            try expectArraysEqual(.{ rw.gdn_qkv_w, rw.gdn_qkv_s, rw.gdn_qkv_b, rw.gdn_z_w, rw.gdn_z_s, rw.gdn_z_b }, .{ rg.gdn_qkv_w, rg.gdn_qkv_s, rg.gdn_qkv_b, rg.gdn_z_w, rg.gdn_z_s, rg.gdn_z_b });
        }
    }
}

/// An offload that computes what the legacy seam's GPU arm computes from its input, counting calls per layer.
const Probe = struct {
    xfm: *Transformer,
    layers: []const AneLayer,
    mlp: [4]u32 = @splat(0),
    gdn: [4]u32 = @splat(0),
    /// Hand back zeros for the MLP (the forward must read what the offload returns).
    zero_mlp: bool = false,

    fn offload(p: *Probe) abi.Offload {
        return .{ .ctx = p, .mlp = &mlp_, .gdn_proj = &gdn_ };
    }

    fn mlp_(ctx: *anyopaque, layer: u32, x: abi.Array, out: *abi.Array) callconv(.c) bool {
        const p: *Probe = @ptrCast(@alignCast(ctx));
        const dw = &(p.layers[layer].mlp orelse return false);
        p.mlp[layer] += 1;
        const y = p.xfm.denseMLP(.{ .ctx = x.ctx }, dw) catch return false;
        if (p.zero_mlp) {
            defer _ = mlx.mlx_array_free(y);
            const sh = mlx.getShape(y);
            var z = mlx.mlx_array_new();
            _ = mlx.mlx_zeros(&z, sh.ptr, sh.len, mlx.mlx_array_dtype(y), p.xfm.s);
            out.* = .{ .ctx = z.ctx };
            return true;
        }
        out.* = .{ .ctx = y.ctx };
        return true;
    }

    fn gdn_(ctx: *anyopaque, layer: u32, x: abi.Array, qkv: *abi.Array, z: *abi.Array) callconv(.c) bool {
        const p: *Probe = @ptrCast(@alignCast(ctx));
        const g = &(p.layers[layer].gdn orelse return false);
        p.gdn[layer] += 1;
        const r = p.xfm.gdnProjArrays(.{ .ctx = x.ctx }, g.qkv_w, g.qkv_s, g.qkv_b, g.z_w, g.z_s, g.z_b) catch return false;
        qkv.* = .{ .ctx = r.qkv.ctx };
        z.* = .{ .ctx = r.z.ctx };
        return true;
    }
};

/// A prompt chunk through the table with `probe` as the offload, against the legacy forward of the same chunk.
fn expectSeam(p: *served.Pair, probe: *Probe, equal: bool) !void {
    const a = testing.allocator;
    const n = p.cfg.num_hidden_layers;
    const ids = [_]i32{ 3, 17, 42, 99, 7, 250, 1, 64, 12, 9, 33, 77, 140, 2, 5, 61, 200, 88, 6, 30 };
    var g = ops.Graph.init(a, fixture.stream());
    defer g.deinit();
    var lane: served.Lane = undefined;
    try lane.init(&p.lg.xfm);
    defer lane.deinit();
    const want = try fixture.rows2(&g, try g.adopt(try p.lg.xfm.forwardWith(&lane.ctx, try fixture.tokens(&g, &ids))));

    var kv = try KVCache.init(a, n);
    defer kv.deinit();
    const ssm = try fixture.newEntries(n);
    defer fixture.freeEntries(ssm);
    var slot: cache_ops.Slot = .{ .kv = &kv, .ssm = ssm };
    const handles = [_]abi.Handle{slot.handle()};
    var host: [ids.len]u32 = undefined;
    for (ids, &host) |x, *y| y.* = @intCast(x);
    const pos = [_]u32{0};
    const off = probe.offload();
    var out: abi.ForwardOut = .{};
    const v = qwen3_5.table.forward(p.xfm.plugin.?.model, &.{ .mode = .prefill, .logits = .all_rows, .rows = ids.len, .tokens = .of(&host), .positions = .of(&pos), .caches = .of(&handles), .cache_ops = &cache_ops.ops, .offload = &off }, &out);
    if (!v.isOk()) {
        std.debug.print("forward: {s}\n", .{v.why.slice()});
        return error.PluginForward;
    }
    const got = try g.adopt(.{ .ctx = out.logits.ctx });
    if (equal) return ops.expectBitEqual(got, want);
    try testing.expect(try maxAbsDiffF32(got, want, fixture.stream()) > 0);
}

fn seamCase(comptime moe: bool, seed: u64) !void {
    var p: served.Pair = undefined;
    try openPair(&p, moe, seed);
    defer p.close();
    try testing.expect(p.cfg.plugin.?.parsed.caps.ane_prefill);
    const layers = (try p.xfm.aneLayers(testing.allocator)).?;
    defer testing.allocator.free(layers);
    var probe: Probe = .{ .xfm = &p.xfm, .layers = layers };
    try expectSeam(&p, &probe, true);
    // Layers 0 and 2 are linear attention; a MoE trunk has no dense MLP to offload.
    try testing.expectEqualSlices(u32, &.{ 1, 0, 1, 0 }, &probe.gdn);
    try testing.expectEqualSlices(u32, if (moe) &.{ 0, 0, 0, 0 } else &.{ 1, 1, 1, 1 }, &probe.mlp);
    if (moe) return;
    probe.zero_mlp = true;
    try expectSeam(&p, &probe, false);
}

test "qwen3_5 ANE prefill: the served shell gives the build the legacy shell's layers, weights, rest slices and plan" {
    inline for (.{ false, true }, .{ 0xa7e1, 0xa7e2 }, .{ 4, 0 }) |moe, seed, dense| {
        var p: served.Pair = undefined;
        try openPair(&p, moe, seed);
        defer p.close();
        try expectSameInputs(&p, dense, 2);
    }
}

test "qwen3_5 ANE prefill: a prompt chunk hands the offload each dense MLP and GDN projection at the legacy seam" {
    try seamCase(false, 0xa7e3);
    try seamCase(true, 0xa7e4);
}
