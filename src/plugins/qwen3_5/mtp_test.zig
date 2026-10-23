//! The qwen3_5 MTP head (`SpecHooks.head`) served through the driver against the legacy sidecar arm bound to the
//! legacy shell, over one tiny dense pack and a qwen-layout head file: history appends, a merged last-row step, an
//! every-row step and the draft pick bit-equal, then a pinned-depth greedy round loop with the same tokens and
//! per-round accepted counts. The table lends its trunk head (`trunk_weight`); the proof turns layer capture off, so
//! the head alone borrows it.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const mtp_mod = @import("../../sdk/spec/mtp/mtp.zig");
const fixture = @import("../../sdk/bases/proof/fixture.zig");
const pack = @import("../../sdk/bases/proof/qwen35_test.zig");
const plugin_head = @import("../../engines/mlx/plugin/head.zig");
const depth_bounds = @import("../../engines/mlx/spec/mtp_depth_bounds.zig");
const dflash_mod = @import("../../engines/mlx/spec/dflash/dflash.zig");
const MtpHeadRef = @import("../../engines/mlx/spec/mtp_head_ref.zig").MtpHeadRef;
const MtpCacheRef = @import("../../engines/mlx/spec/mtp_head_ref.zig").MtpCacheRef;
const Generator = @import("../../engines/mlx/generate/generator.zig").Generator;
const Tokenizer = @import("../../host/tokenizer/tokenizer.zig").Tokenizer;
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;
const served_test = @import("served_test.zig");
const qwen3_5 = @import("plugin.zig");

const testing = std.testing;
const Array = mlx.mlx_array;

const H = 256;

/// A head's geometry: the trunk's width, its attention heads (hd 256) and the MLP width.
pub const Geo = struct { hidden: usize = H, heads: usize = 4, kv_heads: usize = 2, inter: usize = 512 };

/// A qwen-layout dense head (`mtp.*`, the q projection carrying the output gate) at the tiny pack's geometry.
fn writeHead(path: []const u8, geo: Geo) !void {
    const h = geo.hidden;
    const s = fixture.stream();
    const Fx = dflash_mod.TinyFix;
    const map = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(map);
    const meta = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta);
    try Fx.putW(map, "mtp.fc.weight", h, 2 * h, 301, s);
    for ([_][]const u8{ "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight", "mtp.norm.weight", "mtp.layers.0.input_layernorm.weight", "mtp.layers.0.post_attention_layernorm.weight" }) |k|
        try Fx.putNorm(map, k, h, s);
    try Fx.putNorm(map, "mtp.layers.0.self_attn.q_norm.weight", 256, s);
    try Fx.putNorm(map, "mtp.layers.0.self_attn.k_norm.weight", 256, s);
    try Fx.putW(map, "mtp.layers.0.self_attn.q_proj.weight", 2 * geo.heads * 256, h, 302, s);
    try Fx.putW(map, "mtp.layers.0.self_attn.k_proj.weight", geo.kv_heads * 256, h, 303, s);
    try Fx.putW(map, "mtp.layers.0.self_attn.v_proj.weight", geo.kv_heads * 256, h, 304, s);
    try Fx.putW(map, "mtp.layers.0.self_attn.o_proj.weight", h, geo.heads * 256, 305, s);
    try Fx.putW(map, "mtp.layers.0.mlp.gate_proj.weight", geo.inter, h, 306, s);
    try Fx.putW(map, "mtp.layers.0.mlp.up_proj.weight", geo.inter, h, 307, s);
    try Fx.putW(map, "mtp.layers.0.mlp.down_proj.weight", h, geo.inter, 308, s);
    const file = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/model-mtp.safetensors", .{path}, 0);
    defer testing.allocator.free(file);
    try mlx.check(mlx.mlx_save_safetensors(file.ptr, map, meta));
}

/// A MoE trunk's head (4-bit): the router, eight stacked experts and the gated shared expert in place of the SwiGLU.
fn writeMoeHead(path: []const u8) !void {
    var hp = fixture.Pack.init(0x47b0);
    defer hp.deinit();
    try hp.weight("mtp.fc", H, 2 * H, true);
    for ([_][]const u8{ "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight", "mtp.norm.weight", "mtp.layers.0.input_layernorm.weight", "mtp.layers.0.post_attention_layernorm.weight" }) |k|
        try hp.norm(k, H);
    try hp.norm("mtp.layers.0.self_attn.q_norm.weight", 256);
    try hp.norm("mtp.layers.0.self_attn.k_norm.weight", 256);
    try hp.weight("mtp.layers.0.self_attn.q_proj", 2 * 4 * 256, H, true);
    try hp.weight("mtp.layers.0.self_attn.k_proj", 2 * 256, H, true);
    try hp.weight("mtp.layers.0.self_attn.v_proj", 2 * 256, H, true);
    try hp.weight("mtp.layers.0.self_attn.o_proj", H, 4 * 256, true);
    try hp.weight("mtp.layers.0.mlp.gate", 8, H, true);
    try hp.tensor("mtp.layers.0.mlp.switch_mlp.gate_proj", &.{ 8, 128, H }, true);
    try hp.tensor("mtp.layers.0.mlp.switch_mlp.up_proj", &.{ 8, 128, H }, true);
    try hp.tensor("mtp.layers.0.mlp.switch_mlp.down_proj", &.{ 8, H, 128 }, true);
    try hp.weight("mtp.layers.0.mlp.shared_expert.gate_proj", 128, H, true);
    try hp.weight("mtp.layers.0.mlp.shared_expert.up_proj", 128, H, true);
    try hp.weight("mtp.layers.0.mlp.shared_expert.down_proj", H, 128, true);
    try hp.weight("mtp.layers.0.mlp.shared_expert_gate", 1, H, true);
    const file = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/model-mtp.safetensors", .{path}, 0);
    defer testing.allocator.free(file);
    const meta = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta);
    try mlx.check(mlx.mlx_save_safetensors(file.ptr, hp.map, meta));
}

/// `rows` trunk hidden rows of width `h`, in `dt` (the trunk's activations).
fn hiddenRows(g: *ops.Graph, rows: u32, h: u32, dt: mlx.mlx_dtype, seed: u64) !Array {
    var prng = std.Random.DefaultPrng.init(seed);
    const buf = try testing.allocator.alloc(f32, rows * h);
    defer testing.allocator.free(buf);
    for (buf) |*x| x.* = prng.random().floatNorm(f32) * 0.5;
    const f = try g.adopt(mlx.mlx_array_new_data(buf.ptr, &[_]c_int{ 1, @intCast(rows), @intCast(h) }, 3, .float32));
    return g.astype(f, dt);
}

fn idArr(g: *ops.Graph, ids: []const i32) !Array {
    return g.adopt(mlx.mlx_array_new_data(ids.ptr, &[_]c_int{@intCast(ids.len)}, 1, .int32));
}

/// A greedy pinned-depth MTP run: the emitted stream and each round's accepted count.
fn mtpRun(xfm: *Transformer, head: MtpHeadRef, tokens: []u32, accepted: *std.ArrayList(u32)) !void {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var tok_dummy: Tokenizer = undefined; // never read by the Generator
    const prompt = [_]u32{ 3, 7, 1, 12, 30, 5, 9, 22, 4, 17, 2, 28 };
    var gen = try Generator.initWithOptions(io, a, xfm, &tok_dummy, &prompt, @intCast(tokens.len), .{ .temperature = 0.0 }, &.{}, .{
        .mtp_enabled = true,
        .mtp = head,
        .model_has_mtp = true,
        .mtp_depth = 2,
    });
    defer gen.deinit(a);
    var n: usize = 0;
    while (try gen.nextMtp(a)) |res| {
        defer a.free(res.tokens);
        @memcpy(tokens[n..][0..res.tokens.len], res.tokens);
        n += res.tokens.len;
        try accepted.append(a, res.accepted_tokens);
    }
    try testing.expectEqual(tokens.len, n);
    try testing.expect(gen.mtp_attempted > 0);
}

fn noCapture(parsed: *@import("../../abi/plugin.zig").Parsed) void {
    parsed.caps.dflash_capture = false;
}

test "qwen3_5 MTP head: head steps, draft picks and a pinned-depth round loop equal the legacy sidecar arm" {
    var pk = fixture.Pack.init(0x47a1);
    defer pk.deinit();
    try pack.writePack(&pk, false, true, 8);
    try expectHead(&qwen3_5.table, &pk, pack.config(false, true, 8), false, "mtp-dense");
}

test "qwen3_5 MTP head: a MoE trunk's routed-expert head equals the legacy arm (steps, draft picks, rounds)" {
    var pk = fixture.Pack.init(0x47a2);
    defer pk.deinit();
    try pack.writePack(&pk, true, true, 8);
    try expectHead(&qwen3_5.table, &pk, pack.config(true, true, 8), true, "mtp-moe");
}

test "qwen3_5 MTP head: a qwen3_next trunk's routed-expert head equals the legacy arm" {
    const next = @import("../qwen3_next/proof_test.zig");
    var pk = fixture.Pack.init(0x47a3);
    defer pk.deinit();
    try next.writePack(&pk, .{});
    const json = try next.config(testing.allocator, .{});
    defer testing.allocator.free(json);
    try expectHead(&@import("../qwen3_next/plugin.zig").table, &pk, json, true, "mtp-next");
}

pub fn expectHead(table: *const @import("../../abi/plugin.zig").ModelTable, pk: *fixture.Pack, json: []const u8, comptime moe: bool, comptime tape: []const u8) !void {
    return expectHeadAt(table, pk, json, moe, .{}, "qwen3_5", tape);
}

/// `expectHead` with a dense head of geometry `geo`, its tape under `family`.
pub fn expectHeadAt(table: *const @import("../../abi/plugin.zig").ModelTable, pk: *fixture.Pack, json: []const u8, comptime moe: bool, geo: Geo, comptime family: []const u8, comptime tape: []const u8) !void {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = fixture.stream();
    var p: served_test.Pair = undefined;
    // Without layer capture: the head reads the trunk the shell borrows for it alone.
    try p.openWith(table, pk, json, &noCapture);
    defer p.close();
    p.oracle = try served_test.oracle(family, tape);
    try testing.expect(!p.xfm.plugin.?.drafters and p.xfm.plugin.?.trunk);
    if (moe) try writeMoeHead(p.dir) else try writeHead(p.dir, geo);

    var legacy = try mtp_mod.loadMtp(io, a, s, p.dir);
    defer legacy.deinit();
    try legacy.bind(&p.lg.xfm);
    const served = plugin_head.load(io, a, &p.xfm, p.dir) orelse return error.HeadNotLoaded;
    defer plugin_head.destroy(served, a);
    const loaded: *qwen3_5.Loaded = @ptrCast(@alignCast(p.xfm.plugin.?.model));
    try testing.expectEqual(moe, legacy.mlp == .moe);
    try testing.expectEqual(moe, loaded.head.?.moe != null);
    try testing.expectEqual(legacy.draft_head != null, served.draft_head != null);
    try testing.expectEqual(legacy.rerank_coarse != null, served.rerank_coarse != null);

    const heads = [_]MtpHeadRef{ .{ .qwen = &legacy }, .{ .plugin = served } };
    const xs = p.shells();
    var caches: [2]MtpCacheRef = undefined;
    for (heads, &caches) |h, *c| c.* = try h.makeCache(a);
    defer for (&caches) |*c| c.deinit();
    {
        var g = ops.Graph.init(a, s);
        defer g.deinit();
        const width = p.cfg.hidden_size;
        const dt = p.cfg.actDtype();
        const hist = try hiddenRows(&g, 3, width, dt, 1);
        const step = try hiddenRows(&g, 2, width, dt, 2);
        const one = try hiddenRows(&g, 1, width, dt, 3);
        const many = try hiddenRows(&g, 3, width, dt, 4);
        var outs: [2][5]Array = undefined;
        for (heads, xs, &caches, 0..) |h, x, *c, k| {
            try h.appendHistory(x, c, &.{ 5, 9, 40 }, hist, 0, null, a);
            const merged = try h.forward(x, c, try idArr(&g, &.{ 7, 11 }), step, 3, .logits, null, null);
            const mixed = try h.forward(x, c, try idArr(&g, &.{60}), one, 5, .mixed, null, null);
            const rows = try h.forward(x, c, try idArr(&g, &.{ 61, 2, 90 }), many, 6, .mixed, null, null);
            outs[k] = .{ try g.adopt(merged.logits), try g.adopt(merged.hidden_next), try g.adopt(mixed.hidden_next), try g.adopt(rows.hidden_next), try g.adopt(try h.draftSelect(x, mixed.hidden_next, null)) };
            try testing.expectEqual(@as(usize, 9), c.step());
        }
        for (outs[0], outs[1]) |want, got| try p.oracle.check(got, want, false);
    }

    depth_bounds.active = .{ .min = 2, .max = 2 };
    defer depth_bounds.active = .{};
    var tokens: [2][16]u32 = undefined;
    var accepted: [2]std.ArrayList(u32) = .{ .empty, .empty };
    defer for (&accepted) |*l| l.deinit(a);
    for (heads, xs, 0..) |h, x, k| try mtpRun(x, h, &tokens[k], &accepted[k]);
    try p.oracle.stream(&tokens[1], &tokens[0]);
    try p.oracle.stream(accepted[1].items, accepted[0].items);
    try p.oracle.done();
}
