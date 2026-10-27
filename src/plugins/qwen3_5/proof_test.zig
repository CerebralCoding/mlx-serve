//! Parity proof for the qwen3_5 plugin: the table's forward over its own load of a tiny random pack (2 GDN + 2 gated
//! attention layers, hd-128 linear heads, hd-256 attention) is bit-equal to the legacy layered forward — prompt chunks
//! cold and warm, decode, captured verify rows — on logits and every recurrent state and capture, and two slots
//! batched (decode, verify rows, decode) bit-equal to the legacy GDN batched driver; the legacy values recorded to
//! `fixtures/twin-<seed>.tape`. Dense and MoE, 4-bit and bf16.

const std = @import("std");
const fixture = @import("../../sdk/bases/proof/fixture.zig");
const pack = @import("../../sdk/bases/proof/qwen35_pack.zig");
const twin = @import("../qwen3/twin_test.zig");
const qwen3_5 = @import("plugin.zig");

const testing = std.testing;

const T = twin.Twin(qwen3_5);
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const abi = @import("../../abi/plugin.zig");
const proof = @import("../gemma3/proof_test.zig");

test "qwen3_5 LoRA: load-time MLP and attention overlays survive prefill decode and verify" {
    inline for (.{ .down, .q, .k, .v, .o }) |projection| {
        var p = fixture.Pack.init(0x10b5);
        defer p.deinit();
        try pack.writePack(&p, false, true, 0);
        var b = try proof.Both.openAs(&qwen3_5.table, &p, pack.config(false, true, 0), .none);
        defer b.close();
        var g = ops.Graph.init(testing.allocator, b.xfm.s);
        defer g.deinit();
        const input = if (projection == .down) 512 else if (projection == .o) 1024 else 256;
        const output = if (projection == .q) 2048 else if (projection == .k or projection == .v) 512 else 256;
        const binding: abi.LoraBinding = .{ .layer = if (projection == .down) 0 else 1, .projection = projection, .a = .{ .ctx = (try g.ones(&.{ input, 2 }, .bfloat16)).ctx }, .b = .{ .ctx = (try g.mulScalar(try g.reshape(try g.arange(0, output * 2, 1, .bfloat16), &.{ 2, output }), 0.001)).ctx }, .scale = 1 };
        var slots: [4]proof.Slot = undefined;
        const entries = .{ try fixture.newEntries(4), try fixture.newEntries(4), try fixture.newEntries(4), try fixture.newEntries(4) };
        defer inline for (entries) |e| fixture.freeEntries(e);
        inline for (&slots, entries) |*slot, e| {
            try slot.init(4);
            slot.ctx.ssm_entries = e;
        }
        defer for (&slots) |*slot| slot.deinit();
        var zero = binding;
        zero.scale = 0;
        const driver = @import("../../engines/mlx/plugin/driver.zig");
        var adapted = try driver.loadAdapted(testing.io, testing.allocator, &b.cfg, b.dir, mlx.mlx_default_gpu_stream_new(), &.{binding});
        defer adapted.deinit();
        var identity = try driver.loadAdapted(testing.io, testing.allocator, &b.cfg, b.dir, mlx.mlx_default_gpu_stream_new(), &.{zero});
        defer identity.deinit();
        const models = .{ &b.xfm, &adapted, &identity, &adapted };
        for (fixture.steps) |step| {
            var logits: [4]ops.Array = undefined;
            inline for (&slots, &logits, models) |*slot, *out, xfm| {
                slot.ctx.spec_verify = step.mode == .verify;
                out.* = try g.adopt(try xfm.forwardWith(&slot.ctx, try fixture.tokens(&g, step.ids)));
            }
            try ops.expectBitEqual(logits[0], logits[2]);
            try ops.expectBitEqual(logits[1], logits[3]);
            const delta = try g.max(try g.reshape(try g.abs(try g.sub(logits[0], logits[1])), &.{-1}), 0, false);
            try testing.expect(try @import("../../sdk/train/grad.zig").item(try g.astype(delta, .float32)) > 0.01);
        }
    }
}

test "qwen3_5 LoRA: dense training learns through frozen recurrent layers and preserves saved weights" {
    const train = @import("../../sdk/train/model.zig");
    const adapter_mod = @import("../../sdk/train/adapter.zig");
    inline for (.{ false, true }) |quantized| {
        var p = fixture.Pack.init(0x10b6);
        defer p.deinit();
        try pack.writePack(&p, false, quantized, 0);
        var b = try proof.Both.openAs(&qwen3_5.table, &p, pack.config(false, quantized, 0), .none);
        defer b.close();
        var frozen: train.Frozen = undefined;
        const hook = qwen3_5.table.native_train orelse return error.MissingNativeTraining;
        try testing.expect(hook(b.xfm.plugin.?.model, &frozen).isOk());
        try testing.expectEqualStrings("language_model.model", frozen.prefix);
        const targets = try frozen.targets(testing.allocator);
        defer testing.allocator.free(targets);
        try testing.expectEqual(@as(usize, 20), targets.len);
        for (targets) |target| if (target.projection == .q) {
            try testing.expectEqual(@as(u32, 2048), target.output);
            try testing.expect(target.layer % 2 == 1);
        };
        var adapter = try adapter_mod.Adapter.init(testing.allocator, b.xfm.s, targets, .{ .rank = 2, .alpha = 4, .prefix = frozen.prefix });
        defer adapter.deinit();
        const e = try @import("../../sdk/train/packing.zig").pack(testing.allocator, &.{ 3, 17, 42, 99, 7, 250, 1 }, 3, 512);
        defer e.deinit(testing.allocator);
        var trainer = try @import("../../sdk/train/step.zig").Trainer.initForTest(testing.allocator, b.xfm.s, frozen, &adapter, .{ .learning_rate = 0.002, .total_steps = 12, .warmup_steps = 1, .accumulation = 1 });
        defer trainer.deinit();
        const before = try trainer.evaluate(&.{e});
        for (0..12) |_| _ = try trainer.microbatch(e);
        try testing.expect(try trainer.evaluate(&.{e}) < before);
        var g = ops.Graph.init(testing.allocator, b.xfm.s);
        defer g.deinit();
        const bindings = try adapter.bindings(&g, adapter.params);
        defer testing.allocator.free(bindings);
        const ids = try g.fromSlice(u32, e.inputs, &.{ 1, @intCast(e.inputs.len) });
        const training = try frozen.project(&g, try frozen.hidden(&g, ids, bindings));
        const path = try std.fs.path.join(testing.allocator, &.{ b.dir, "trained" });
        defer testing.allocator.free(path);
        try adapter.save(testing.io, path, @import("../../sdk/weights/lora_config.zig").fingerprint(pack.config(false, quantized, 0)));
        var exported = try @import("../../sdk/weights/resident.zig").loadWeightsFile(testing.allocator, path, "adapters.safetensors");
        defer exported.deinit();
        try testing.expect(exported.map.contains("language_model.model.layers.1.self_attn.q_proj.lora_a"));
        const weights_path = try std.fs.path.join(testing.allocator, &.{ path, "adapters.safetensors" });
        defer testing.allocator.free(weights_path);
        var reloaded = try @import("../../sdk/weights/lora.zig").LlmAdapter.load(testing.allocator, weights_path, 2, 2);
        defer reloaded.deinit();
        try ops.expectBitEqual(training, try frozen.project(&g, try frozen.hidden(&g, ids, reloaded.bindings)));
    }
}

test "qwen3_5 LoRA: frozen recurrence preserves gradients through every trainable tensor" {
    var p = fixture.Pack.init(0x10b7);
    defer p.deinit();
    try pack.writePack(&p, false, true, 0);
    var b = try proof.Both.openAs(&qwen3_5.table, &p, pack.config(false, true, 0), .none);
    defer b.close();
    const loaded: *qwen3_5.Loaded = @ptrCast(@alignCast(b.xfm.plugin.?.model));
    loaded.body.dense.base.weights.embed.act = .float32;
    const frozen = try @import("train.zig").frozen(&loaded.body.dense.base, loaded.adapter_prefix);
    const targets = try frozen.targets(testing.allocator);
    defer testing.allocator.free(targets);
    var adapter = try @import("../../sdk/train/adapter.zig").Adapter.init(testing.allocator, b.xfm.s, targets, .{ .rank = 1, .alpha = 2, .compute = .float32 });
    defer adapter.deinit();
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    var g = ops.Graph.init(testing.allocator, cpu);
    defer g.deinit();
    try @import("../../sdk/train/testing.zig").expectGradients(&g, frozen, &adapter);
}

test "qwen3_5 LoRA: seq512 training stays within its recurrent activation reservation" {
    var p = fixture.Pack.init(0x10b8);
    defer p.deinit();
    try pack.writePack(&p, false, true, 0);
    var b = try proof.Both.openAs(&qwen3_5.table, &p, pack.config(false, true, 0), .none);
    defer b.close();
    var frozen: @import("../../sdk/train/model.zig").Frozen = undefined;
    try testing.expect(qwen3_5.table.native_train.?(b.xfm.plugin.?.model, &frozen).isOk());
    const targets = try frozen.targets(testing.allocator);
    defer testing.allocator.free(targets);
    try @import("../../sdk/train/testing.zig").expectReservation(frozen, targets);
}

const solo = [_]twin.Step{
    .{ .mode = .prefill, .ids = &twin.seq(20, 3) },
    .{ .mode = .prefill, .ids = &twin.seq(70, 11) },
    .{ .mode = .decode, .ids = &.{88} },
    .{ .mode = .verify, .ids = &.{ 5, 61, 200 } },
    .{ .mode = .decode, .ids = &.{12} },
    .{ .mode = .verify, .ids = &.{ 9, 33, 77, 140, 2 } },
    .{ .mode = .decode, .ids = &.{64} },
    .{ .mode = .prefill, .ids = &twin.seq(6, 5) },
    .{ .mode = .decode, .ids = &.{17} },
};

fn open(p: *fixture.Pack, comptime moe: bool, comptime quantized: bool, comptime experts: c_int, comptime seed: u64) !*T {
    try pack.writePack(p, moe, quantized, experts);
    const name = std.fmt.comptimePrint("fixtures/twin-{x}.tape", .{seed});
    const t = try T.open(p, pack.config(moe, quantized, experts), &.{}, @embedFile(name));
    return t;
}

fn soloCase(comptime moe: bool, comptime quantized: bool, comptime experts: c_int, comptime seed: u64) !void {
    var p = fixture.Pack.init(seed);
    defer p.deinit();
    const t = try open(&p, moe, quantized, experts, seed);
    defer t.close();
    const l = try t.lane();
    defer l.deinit();
    for (solo) |st| try t.step(l, st);
    try t.expectDone();
}

fn batchedCase(comptime moe: bool, comptime seed: u64) !void {
    var p = fixture.Pack.init(seed);
    defer p.deinit();
    const t = try open(&p, moe, true, 8, seed);
    defer t.close();
    // Split the merged states back after every tick, so each slot's states can be compared.
    const lanes = [2]*twin.Lane{ try t.lane(), try t.lane() };
    defer for (lanes) |l| l.deinit();
    try t.step(lanes[0], .{ .mode = .prefill, .ids = &twin.seq(20, 3) });
    try t.step(lanes[1], .{ .mode = .prefill, .ids = &.{ 42, 101, 7, 9, 250, 3, 77 } });
    try t.batched(lanes, .{ &.{77}, &.{3} });
    try t.batched(lanes, .{ &.{ 5, 61, 200 }, &.{ 9, 33, 140 } });
    try t.batched(lanes, .{ &.{190}, &.{64} });
    try t.expectDone();
}

test "qwen3_5 plugin: a dense 4-bit pack through the table is bit-equal to the legacy layered forward" {
    try soloCase(false, true, 8, 0x35c1);
}

test "qwen3_5 plugin: a dense bf16 pack (decode attention side copies) is bit-equal" {
    try soloCase(false, false, 8, 0x35c2);
}

test "qwen3_5 plugin: a qwen3_5_moe 4-bit pack (shared-expert gate as the router kernel's vector) is bit-equal" {
    try soloCase(true, true, 64, 0x35c3);
}

test "qwen3_5 plugin: a qwen3_5_moe bf16 pack (dense experts) is bit-equal" {
    try soloCase(true, false, 8, 0x35c4);
}

test "qwen3_5 plugin: dense batched rows are bit-equal to the legacy GDN batched driver" {
    try batchedCase(false, 0x35c5);
}

test "qwen3_5 plugin: qwen3_5_moe batched rows are bit-equal to the legacy GDN batched driver" {
    try batchedCase(true, 0x35c6);
}
