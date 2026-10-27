//! Trained adapters through the real pack loaders and inference forwards.
const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const fixture = @import("../../../sdk/bases/proof/fixture.zig");
const packs = @import("../../../plugins/qwen3/proof_test.zig");
const config_mod = @import("../../../sdk/config/model.zig");
const model = @import("../../../sdk/train/model.zig");
const Adapter = @import("../../../sdk/train/adapter.zig").Adapter;
const Trainer = @import("../../../sdk/train/step.zig").Trainer;
const packing = @import("../../../sdk/train/packing.zig");
const lora_config = @import("../../../sdk/weights/lora_config.zig");

test "train core: loaded llama and qwen3 train four-bit and bf16 packs and roundtrip adapter weights" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    for ([_][]const u8{ "llama", "qwen3" }) |arch| for ([_]bool{ false, true }) |quantized| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const path = buf[0..try tmp.dir.realPath(io, &buf)];
        const json = try packs.configJson(a, .{ .quantized = quantized });
        defer a.free(json);
        const config = try std.mem.replaceOwned(u8, a, json, "qwen3", arch);
        defer a.free(config);
        try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = config });
        var pack = fixture.Pack.init(23);
        defer pack.deinit();
        try packs.writePack(&pack, .{ .quantized = quantized });
        try pack.save(path);
        var cfg = try config_mod.parseConfig(io, a, path);
        defer cfg.deinit(a);
        const s = mlx.mlx_default_gpu_stream_new();
        var xfm = @import("driver.zig").load(io, a, &cfg, path, s) catch |err| {
            _ = mlx.mlx_stream_free(s);
            return err;
        };
        defer xfm.deinit();
        const base: *const model.Base = if (std.mem.eql(u8, arch, "llama")) blk: {
            const loaded: *@import("../../../plugins/llama/plugin.zig").Loaded(model.Decoder) = @ptrCast(@alignCast(xfm.plugin.?.model));
            break :blk &loaded.base;
        } else blk: {
            const loaded: *@import("../../../plugins/qwen3/plugin.zig").Loaded = @ptrCast(@alignCast(xfm.plugin.?.model));
            break :blk &loaded.body.dense.base;
        };
        const frozen = try model.Frozen.init(base);
        const targets = try frozen.targets(a);
        defer a.free(targets);
        var adapter = try Adapter.init(a, s, targets, .{});
        defer adapter.deinit();
        var trainer = try Trainer.init(a, s, frozen, &adapter, .{ .accumulation = 1 });
        defer trainer.deinit();
        const e = try packing.pack(a, &.{ 1, 2, 3, 7, 8 }, 3, 512);
        defer e.deinit(a);
        _ = try trainer.microbatch(e);
        var g = ops.Graph.init(a, s);
        defer g.deinit();
        const bindings = try adapter.bindings(&g, adapter.params);
        defer a.free(bindings);
        const ids = try g.fromSlice(u32, e.inputs, &.{ 1, @intCast(e.inputs.len) });
        const hidden = try frozen.hidden(&g, ids, bindings);
        const trained = try @import("../../../sdk/bases/blocks/head.zig").project(&g, &base.weights.head, hidden);
        const saved = try std.fs.path.join(a, &.{ path, "trained" });
        defer a.free(saved);
        try adapter.save(io, saved, lora_config.fingerprint(config));
        const tensor = try std.fs.path.join(a, &.{ saved, "adapters.safetensors" });
        defer a.free(tensor);
        var runtime = try @import("../../../sdk/weights/lora.zig").LlmAdapter.load(a, tensor, 2, 8);
        defer runtime.deinit();
        const reloaded = try frozen.project(&g, try frozen.hidden(&g, ids, runtime.bindings));
        try ops.expectBitEqual(trained, reloaded);
    };
}

test "train core: real Qwen3 pack learns with defaults and seq512 stays within its reservation" {
    const path = @import("../../../env.zig").span(.LORA_TRAIN_TEST_MODEL) orelse return error.SkipZigTest;
    const a = std.testing.allocator;
    const io = std.testing.io;
    mlx.installErrorHandler();
    var cfg = try config_mod.parseConfig(io, a, path);
    defer cfg.deinit(a);
    if (!std.mem.eql(u8, cfg.model_type, "qwen3")) return error.TrainingUnsupportedModel;
    const s = mlx.mlx_default_gpu_stream_new();
    var xfm = @import("driver.zig").load(io, a, &cfg, path, s) catch |err| {
        _ = mlx.mlx_stream_free(s);
        return err;
    };
    defer xfm.deinit();
    const loaded: *@import("../../../plugins/qwen3/plugin.zig").Loaded = @ptrCast(@alignCast(xfm.plugin.?.model));
    const frozen = try model.Frozen.init(&loaded.body.dense.base);
    const targets = try frozen.targets(a);
    defer a.free(targets);
    var tokenizer = try @import("../../../sdk/tokenizer/load.zig").loadTokenizer(io, a, path);
    defer tokenizer.deinit();
    var chat_config = try @import("../../../host/chat/template_load.zig").loadChatConfig(io, a, path);
    defer chat_config.deinit();
    const make = @import("../../../host/chat/training.zig").example;
    const e = try make(a, &tokenizer, &.{.{ .role = "user", .content = "What is the codename of this training exercise?" }}, &chat_config, "The codename is cobalt otter.", 512);
    defer e.deinit(a);
    const heldout = try make(a, &tokenizer, &.{.{ .role = "user", .content = "Tell me the codename used for this training exercise." }}, &chat_config, "The codename is cobalt otter.", 512);
    defer heldout.deinit(a);
    for ([_]usize{ 64, 128, 512 }) |rows| {
        var tokens: [513]u32 = undefined;
        for (&tokens, 0..) |*token, i| token.* = e.inputs[i % e.inputs.len];
        const long = try packing.pack(a, tokens[0 .. rows + 1], rows - 12, 512);
        defer long.deinit(a);
        var memory_adapter = try Adapter.init(a, s, targets, .{});
        defer memory_adapter.deinit();
        var memory_trainer = try Trainer.init(a, s, frozen, &memory_adapter, .{ .accumulation = 1 });
        defer memory_trainer.deinit();
        try mlx.check(mlx.mlx_synchronize(s));
        try mlx.check(mlx.mlx_clear_cache());
        var resident: usize = 0;
        try mlx.check(mlx.mlx_get_active_memory(&resident));
        try mlx.check(mlx.mlx_reset_peak_memory());
        _ = try memory_trainer.microbatch(long);
        var peak: usize = 0;
        try mlx.check(mlx.mlx_get_peak_memory(&peak));
        const billed = try @import("../../../sdk/train/memory.zig").reserve(frozen, targets, 8, @intCast(rows));
        std.debug.print("\ntrain real seq{d}: resident={d} peak={d} extra={d} reserve={d}\n", .{ rows, resident, peak, peak -| resident, billed });
        try std.testing.expect(peak -| resident <= billed);
    }
    var adapter = try Adapter.init(a, s, targets, .{});
    defer adapter.deinit();
    var trainer = try Trainer.init(a, s, frozen, &adapter, .{});
    defer trainer.deinit();
    const before = try trainer.evaluate(&.{e});
    const validation_before = try trainer.evaluate(&.{heldout});
    _ = try trainer.observeValidation(validation_before);
    for (0..20) |update| {
        for (0..4) |_| _ = try trainer.microbatch(e);
        const value = try trainer.evaluate(&.{heldout});
        std.debug.print("\ntrain real: update={d} heldout={d:.5}\n", .{ update + 1, value });
        if (try trainer.observeValidation(value)) break;
    }
    try trainer.restoreBest();
    const after = try trainer.evaluate(&.{e});
    const validation_after = try trainer.evaluate(&.{heldout});
    std.debug.print("\ntrain real: loss={d:.5}->{d:.5} heldout={d:.5}->{d:.5}\n", .{ before, after, validation_before, validation_after });
    try std.testing.expect(after < before * 0.5);
    try std.testing.expect(validation_after < validation_before * 0.5);
    var g = ops.Graph.init(a, s);
    defer g.deinit();
    const bindings = try adapter.bindings(&g, adapter.params);
    defer a.free(bindings);
    const merged_stream = mlx.mlx_default_gpu_stream_new();
    var merged = @import("driver.zig").loadAdapted(io, a, &cfg, path, merged_stream, bindings) catch |err| {
        _ = mlx.mlx_stream_free(merged_stream);
        return err;
    };
    defer merged.deinit();
    const serving_before = try servingLoss(&xfm, heldout);
    const serving_after = try servingLoss(&merged, heldout);
    std.debug.print("\ntrain real: heldout serving loss={d:.5}->{d:.5} unfused={d:.5}\n", .{ serving_before, serving_after, validation_after });
    try std.testing.expect(serving_after < serving_before * 0.5);
}

fn servingLoss(xfm: *@import("../transformer.zig").Transformer, example: packing.Example) !f32 {
    var lm: @import("../../../host/registry/registry.zig").LoadedModel = undefined;
    lm.engine = null;
    lm.config = &xfm.config;
    lm.transformer = xfm;
    const frozen = try @import("../finetune/run.zig").frozen(&lm);
    var g = ops.Graph.init(std.testing.allocator, xfm.s);
    defer g.deinit();
    const hidden = try @import("../finetune/teacher.zig").runtimeHidden(std.testing.io, &g, &lm, example.inputs);
    const rows: c_int = @intCast(example.inputs.len);
    const logits = try g.reshape(try frozen.project(&g, hidden), &.{ rows, -1 });
    const targets = try g.fromSlice(u32, example.targets, &.{rows});
    const mask = try g.fromSlice(f32, example.mask, &.{rows});
    return @import("../../../sdk/train/grad.zig").item(try @import("../../../sdk/train/loss.zig").crossEntropy(&g, logits, targets, mask));
}
