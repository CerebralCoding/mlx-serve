const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const abi = @import("../../../abi/plugin.zig");
const fixture = @import("../../../sdk/bases/proof/fixture.zig");
const sidecar = @import("../../../sdk/weights/finetune.zig");
const config_mod = @import("../../../sdk/config/model.zig");
const driver = @import("driver.zig");
const Transformer = @import("../transformer.zig").Transformer;

fn load(cfg: *config_mod.ModelConfig, path: []const u8, bindings: ?[]const abi.LoraBinding) !Transformer {
    const s = mlx.mlx_default_gpu_stream_new();
    errdefer _ = mlx.mlx_stream_free(s);
    return if (bindings) |b| driver.loadAdapted(std.testing.io, std.testing.allocator, cfg, path, s, b) else driver.load(std.testing.io, std.testing.allocator, cfg, path, s);
}

fn logits(g: *ops.Graph, xfm: *Transformer) !ops.Array {
    var lm: @import("../../../host/registry/registry.zig").LoadedModel = undefined;
    lm.engine = null;
    lm.config = &xfm.config;
    lm.transformer = xfm;
    const slot = try @import("../slot.zig").Slot.init(g.a, std.testing.io, &xfm.config, .{ .model = &lm, .prompt_ids = &.{ 4, 7, 9 }, .max_tokens = 1, .sampling = .{}, .eos_token_ids = &.{} }, .dense);
    defer slot.deinit();
    const out = try g.adopt(try xfm.forwardWith(&slot.ctx, try fixture.tokens(g, &.{ 4, 7, 9 })));
    try ops.eval(out);
    const hidden = try @import("../finetune/teacher.zig").runtimeHidden(std.testing.io, g, &lm, &.{ 4, 7, 9 });
    const frozen = try @import("../finetune/run.zig").frozen(&lm);
    const projected = try frozen.project(g, hidden);
    try ops.expectBitEqual(out, try g.reshape(projected, ops.shape(out)));
    return out;
}

test "finetune overlay: automatic load, replacement and removal preserve the base pack" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    mlx.installErrorHandler();
    for ([_][]const u8{ "llama", "qwen3", "qwen3_5", "gemma4" }) |arch| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
        const qwen = @import("../../../plugins/qwen3/proof_test.zig");
        const hybrid = @import("../../../sdk/bases/proof/qwen35_pack.zig");
        const gemma = @import("../../../plugins/gemma4/proof_test.zig");
        const is_hybrid = std.mem.eql(u8, arch, "qwen3_5");
        const is_gemma = std.mem.eql(u8, arch, "gemma4");
        const json = if (is_gemma) try a.dupe(u8, gemma.e_series) else if (is_hybrid) try a.dupe(u8, hybrid.config(false, true, 0)) else try qwen.configJson(a, .{ .quantized = true });
        defer a.free(json);
        const config = try std.mem.replaceOwned(u8, a, json, "qwen3", arch);
        defer a.free(config);
        try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = if (is_hybrid) json else config });
        var pack = fixture.Pack.init(42);
        defer pack.deinit();
        if (is_gemma) try gemma.writeGemma4(&pack, .{}, 256, 2, true, false) else if (is_hybrid) try hybrid.writePack(&pack, false, true, 0) else try qwen.writePack(&pack, .{ .quantized = true });
        try pack.save(path);
        var cfg = try config_mod.parseConfig(io, a, path);
        defer cfg.deinit(a);
        var base = try load(&cfg, path, null);
        defer base.deinit();
        try std.testing.expectEqual(null, cfg.plugin.?.overlay_sha256);
        var g = ops.Graph.init(a, base.s);
        defer g.deinit();
        const before = try logits(&g, &base);
        var frozen: @import("../../../sdk/train/model.zig").Frozen = undefined;
        try std.testing.expect(base.plugin.?.table.native_train.?(base.plugin.?.model, &frozen).isOk());
        const targets = try frozen.targets(a);
        defer a.free(targets);
        const target = for (targets) |t| {
            if (t.projection == .down) break t;
        } else return error.MissingTarget;
        var adapter = try @import("../../../sdk/train/adapter.zig").Adapter.init(a, base.s, &.{target}, .{ .rank = 2, .alpha = 2, .prefix = cfg.weight_prefix });
        defer adapter.deinit();
        _ = mlx.mlx_array_free(adapter.params[1]);
        adapter.params[1] = g.escape(try g.mulScalar(try g.reshape(try g.arange(0, @floatFromInt(target.output * 2), 1, .float32), &.{ 2, @intCast(target.output) }), 0.01));
        const bindings = try adapter.bindings(&g, adapter.params);
        defer a.free(bindings);
        var expected = try load(&cfg, path, bindings);
        defer expected.deinit();
        const want = try logits(&g, &expected);
        const delta = try g.max(try g.reshape(try g.abs(try g.sub(before, want)), &.{-1}), 0, false);
        try std.testing.expect(try @import("../../../sdk/train/grad.zig").item(try g.astype(delta, .float32)) > 0.001);
        const draft = try std.fs.path.join(a, &.{ path, "draft" });
        defer a.free(draft);
        try adapter.save(io, draft, cfg.plugin.?.source_sha256.?);
        const bytes = try tmp.dir.readFileAlloc(io, "draft/adapters.safetensors", a, .limited(1024 * 1024));
        defer a.free(bytes);
        try tmp.dir.writeFile(io, .{ .sub_path = sidecar.filename, .data = bytes });
        var adapted = try load(&cfg, path, null);
        defer adapted.deinit();
        try std.testing.expect(cfg.plugin.?.overlay_sha256 != null);
        try ops.expectBitEqual(want, try logits(&g, &adapted));
        try ops.expectBitEqual(before, try logits(&g, &base));
        try tmp.dir.writeFile(io, .{ .sub_path = sidecar.filename, .data = "broken" });
        try std.testing.expectError(error.InvalidLlmAdapter, load(&cfg, path, null));
        try ops.expectBitEqual(want, try logits(&g, &adapted));
        try tmp.dir.deleteFile(io, sidecar.filename);
        var restored = try load(&cfg, path, null);
        defer restored.deinit();
        try std.testing.expectEqual(null, cfg.plugin.?.overlay_sha256);
        try ops.expectBitEqual(before, try logits(&g, &restored));
    }
}
