//! Runtime adapter equivalence through the served llama and qwen3 plugins.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const abi = @import("../../../abi/plugin.zig");
const fixture = @import("../../../sdk/bases/proof/fixture.zig");
const packs = @import("../../../plugins/qwen3/proof_test.zig");
const model = @import("../../../sdk/config/model.zig");
const driver = @import("driver.zig");
const KVCache = @import("../kv/cache.zig").KVCache;
const ForwardCtx = @import("../forward_ctx.zig").ForwardCtx;

test "LLM LoRA: served llama and qwen3 carry request bindings through prefill, decode, verify and batched rows" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    for ([_][]const u8{ "llama", "qwen3" }) |arch| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        var path_buf: [512]u8 = undefined;
        const path = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
        const json = try packs.configJson(a, .{ .quantized = true });
        defer a.free(json);
        const config = try std.mem.replaceOwned(u8, a, json, "qwen3", arch);
        defer a.free(config);
        try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = config });
        var pack = fixture.Pack.init(0x10aa);
        defer pack.deinit();
        try packs.writePack(&pack, .{ .quantized = true });
        try pack.save(path);
        var cfg = try model.parseConfig(io, a, path);
        defer cfg.deinit(a);
        var xfm = try driver.load(io, a, &cfg, path, fixture.stream());
        defer xfm.deinit();
        var g = ops.Graph.init(a, fixture.stream());
        defer g.deinit();
        var av: [256 * 2]f32 = undefined;
        var bv: [2 * 128]f32 = undefined;
        for (&av, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 11)) * 0.01 - 0.05;
        for (&bv, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 7)) * 0.02 - 0.06;
        const binding: abi.LoraBinding = .{ .layer = 0, .projection = .down, .a = .{ .ctx = (try g.fromSlice(f32, &av, &.{ 256, 2 })).ctx }, .b = .{ .ctx = (try g.fromSlice(f32, &bv, &.{ 2, 128 })).ctx }, .scale = 8 };
        try tmp.dir.createDirPath(io, "adapters/answer");
        const registry = @import("../../../host/registry/adapters.zig");
        const lora_config = @import("../../../sdk/weights/lora_config.zig");
        const root = try std.fs.path.join(a, &.{ path, "adapters" });
        defer a.free(root);
        const adapter_config = try std.fmt.allocPrint(a, "{{\"base_config_sha256\":\"{s}\",\"lora_parameters\":{{\"rank\":2,\"scale\":8}}}}", .{std.fmt.bytesToHex(lora_config.fingerprint(config), .lower)});
        defer a.free(adapter_config);
        try tmp.dir.writeFile(io, .{ .sub_path = "adapters/answer/adapter_config.json", .data = adapter_config });
        var adapter_pack = fixture.Pack.init(0);
        defer adapter_pack.deinit();
        try adapter_pack.put("model.layers.0.mlp.down_proj.lora_a", .{ .ctx = binding.a.ctx });
        try adapter_pack.put("model.layers.0.mlp.down_proj.lora_b", .{ .ctx = binding.b.ctx });
        const tensor_path = try std.fmt.allocPrintSentinel(a, "{s}/answer/adapters.safetensors", .{root}, 0);
        defer a.free(tensor_path);
        const metadata = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(metadata);
        try mlx.check(mlx.mlx_save_safetensors(tensor_path.ptr, adapter_pack.map, metadata));
        var prepared = try registry.resolve(a, io, root, .{ .name = "answer" }, .{ .path = path, .arch = arch, .fingerprint = cfg.plugin.?.source_sha256, .geometry = .{ .layers = 2, .hidden = 128, .intermediate = 256, .heads = 4, .kv_heads = 2, .head_dim = 32 } });
        defer prepared.deinit(a);
        var state = try @import("../lora.zig").State.init(a, prepared);
        defer state.deinit();
        try state.load(io);
        try std.testing.expectEqual(@as(u64, 0), state.loadReserve());
        var zero = binding;
        zero.scale = 0;
        var caches: [4]KVCache = undefined;
        for (&caches) |*cache| cache.* = try KVCache.init(a, 2);
        defer for (&caches) |*cache| cache.deinit();
        var offsets: [4]usize = @splat(0);
        var contexts: [4]ForwardCtx = undefined;
        for (&contexts, &caches, &offsets) |*ctx, *cache, *off| ctx.* = .{ .cache = cache, .moe_seq_offset = off, .ssm_entries = null, .capture_hidden = null, .vision_embeddings = null };
        contexts[1].lora = state.adapter.?.bindings;
        contexts[2].lora = &.{zero};
        contexts[3].lora = &.{binding};
        caches[1].adapter_key = prepared.key;
        caches[3].adapter_key = prepared.key;
        for (fixture.steps) |step| {
            const ids = try fixture.tokens(&g, step.ids);
            var out: [4]mlx.mlx_array = undefined;
            for (&contexts, &out) |*ctx, *logits| {
                ctx.spec_verify = step.mode == .verify;
                logits.* = try g.adopt(try xfm.forwardWith(ctx, ids));
            }
            try ops.expectBitEqual(out[0], out[2]);
            try ops.expectBitEqual(out[1], out[3]);
            try expectDifferent(&g, out[0], out[1]);
        }
        const positions = [_]u32{ @intCast(caches[0].step), @intCast(caches[2].step) };
        const plain = try xfm.forwardBatchedDecode(&.{ 7, 7 }, &.{ &contexts[0], &contexts[2] }, &positions);
        defer a.free(plain);
        for (plain) |v| _ = try g.adopt(v);
        const adapted = try xfm.forwardBatchedDecode(&.{ 7, 7 }, &.{ &contexts[1], &contexts[3] }, &positions);
        defer a.free(adapted);
        for (adapted) |v| _ = try g.adopt(v);
        try ops.expectBitEqual(plain[0], plain[1]);
        try ops.expectBitEqual(adapted[0], adapted[1]);
        try expectDifferent(&g, plain[0], adapted[0]);
        try std.testing.expectError(error.LoraMixedBatch, xfm.forwardBatchedDecode(&.{ 7, 7 }, &.{ &contexts[0], &contexts[1] }, &.{ @intCast(caches[0].step), @intCast(caches[1].step) }));
    }
}

fn expectDifferent(g: *ops.Graph, x: mlx.mlx_array, y: mlx.mlx_array) !void {
    var result = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(result);
    try mlx.check(mlx.mlx_array_equal(&result, x, y, false, g.s));
    try mlx.check(mlx.mlx_array_eval(result));
    try std.testing.expect(!mlx.mlx_array_data_bool(result).?[0]);
}

pub fn expectBatch(xfm: *@import("../transformer.zig").Transformer, g: *ops.Graph, contexts: [4]*ForwardCtx, position: u32) !void {
    defer xfm.ssm_group.deinit();
    const plain = try xfm.forwardBatchedDecode(&.{ 7, 7 }, &.{ contexts[0], contexts[2] }, &.{ position, position });
    defer g.a.free(plain);
    for (plain) |value| _ = try g.adopt(value);
    const adapted = try xfm.forwardBatchedDecode(&.{ 7, 7 }, &.{ contexts[1], contexts[3] }, &.{ position, position });
    defer g.a.free(adapted);
    for (adapted) |value| _ = try g.adopt(value);
    try ops.expectBitEqual(plain[0], plain[1]);
    try ops.expectBitEqual(adapted[0], adapted[1]);
    try expectDifferent(g, plain[0], adapted[0]);
    try std.testing.expectError(error.LoraMixedBatch, xfm.forwardBatchedDecode(&.{ 7, 7 }, &.{ contexts[0], contexts[1] }, &.{ position + 1, position + 1 }));
    for (contexts) |ctx| try @import("../ssm/group.zig").ssmGroupReleaseCtx(&xfm.ssm_group, xfm.s, ctx);
}

pub fn expectTree(xfm: *@import("../transformer.zig").Transformer, g: *ops.Graph, contexts: [4]*ForwardCtx) !void {
    if (!mlx.metalKernelsAvailable()) return;
    const tree = @import("../spec/tree.zig").specTreeFor(&.{ -1, 0, 0 }, &.{ 0, 1, 1 }, 1);
    defer tree.deinit();
    var logits: [4]ops.Array = undefined;
    for (contexts, &logits) |ctx, *out| {
        ctx.spec_verify = true;
        ctx.capture_ssm_seq = true;
        ctx.tree = &tree;
        defer {
            ctx.tree = null;
            ctx.capture_ssm_seq = false;
        }
        out.* = try g.adopt(try xfm.forwardWith(ctx, try fixture.tokens(g, &.{ 3, 7, 9 })));
    }
    try ops.expectBitEqual(logits[0], logits[2]);
    try ops.expectBitEqual(logits[1], logits[3]);
    try expectDifferent(g, logits[0], logits[1]);
}
