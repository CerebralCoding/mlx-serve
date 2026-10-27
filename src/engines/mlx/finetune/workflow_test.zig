//! Deterministic synthesis around real training, saving, and runtime forwards.
const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const abi = @import("../../../abi/plugin.zig");
const fixture = @import("../../../sdk/bases/proof/fixture.zig");
const packs = @import("../../../plugins/qwen3/proof_test.zig");
const config_mod = @import("../../../sdk/config/model.zig");
const model = @import("../../../sdk/train/model.zig");
const types = @import("../../../host/finetune/types.zig");
const LoadedModel = @import("../../../host/registry/registry.zig").LoadedModel;
const Tokenizer = @import("../../../sdk/tokenizer/tokenizer.zig").Tokenizer;
const workflow = @import("workflow.zig");
const Teacher = @import("teacher.zig").Teacher;

const Observer = struct {
    cancel_at: ?types.Stage = null,
    cancelled: bool = false,
    stages: [3]bool = @splat(false),
    boundaries: u32 = 0,
    fn observe(self: *Observer) types.Observer {
        return .{ .ctx = self, .emit_fn = emit, .cancelled_fn = cancel };
    }
    fn emit(ctx: *anyopaque, update: types.Update) void {
        const self: *Observer = @ptrCast(@alignCast(ctx));
        self.stages[@backingInt(update.stage)] = true;
        if (self.cancel_at == update.stage and update.step >= 1) self.cancelled = true;
    }
    fn cancel(ctx: *anyopaque) bool {
        const self: *Observer = @ptrCast(@alignCast(ctx));
        self.boundaries += 1;
        return self.cancelled;
    }
};

const SyntheticTeacher = struct {
    a: std.mem.Allocator,
    lm: *LoadedModel,
    last_complete: bool = true,
    invalid_samples: bool = false,
    truncate_heldout: bool = false,
    pub fn generate(self: *SyntheticTeacher, prompt: []const u8, _: []const u8, _: bool, _: u32) anyerror![]u8 {
        if (self.invalid_samples and (std.mem.startsWith(u8, prompt, "question four?") or std.mem.startsWith(u8, prompt, "What is photosynthesis?"))) return error.InvalidTrainingCompletion;
        self.last_complete = !(self.truncate_heldout and std.mem.startsWith(u8, prompt, "question one?"));
        return self.a.dupe(u8, "yes");
    }
    pub fn refuses(_: *SyntheticTeacher, _: []const u8, _: []const u8) anyerror!bool {
        return false;
    }
};

test "finetune: real teacher completes an assistant prefill and judges its answer" {
    const path = @import("../../../env.zig").span(.LORA_TRAIN_TEST_MODEL) orelse return error.SkipZigTest;
    const a = std.testing.allocator;
    const io = std.testing.io;
    mlx.installErrorHandler();
    var cfg = try config_mod.parseConfig(io, a, path);
    defer cfg.deinit(a);
    const s = mlx.mlx_default_gpu_stream_new();
    var xfm = @import("../plugin/driver.zig").load(io, a, &cfg, path, s) catch |err| {
        _ = mlx.mlx_stream_free(s);
        return err;
    };
    defer xfm.deinit();
    var tok = try @import("../../../sdk/tokenizer/load.zig").loadTokenizer(io, a, path);
    defer tok.deinit();
    var chat_config = try @import("../../../host/chat/template_load.zig").loadChatConfig(io, a, path);
    defer chat_config.deinit();
    var lm: LoadedModel = undefined;
    lm.path = path;
    lm.tokenizer = &tok;
    lm.chat_config = &chat_config;
    lm.config = &cfg;
    lm.transformer = &xfm;
    lm.engine = null;
    var observer: Observer = .{};
    var teacher: Teacher = .{ .a = a, .io = io, .lm = &lm, .frozen = try @import("run.zig").frozen(&lm), .observer = observer.observe() };
    const prompt = "How do I water basil? Please answer in two sentences.";
    const answer = try teacher.generate(prompt, "Sure, ", false, 192);
    defer a.free(answer);
    std.debug.print("\nfinetune teacher: complete={} answer={s}\n", .{ teacher.last_complete, answer });
    try std.testing.expect(teacher.last_complete);
    try std.testing.expect(!try teacher.refuses(prompt, answer));
    teacher.seed = 1;
    var prompts: @import("../../../host/finetune/data.zig").Prompts = .{ .a = a };
    defer prompts.deinit();
    try workflow.collectPrompts(&prompts, &teacher, .{ .model = "real", .name = "test", .topic = "Friendly practical advice about growing herbs on a balcony." }, observer.observe(), .{});
    try std.testing.expectEqual(@as(usize, 64), prompts.items.items.len);
}

test "LoRA readiness: tiny packs complete, cancel in every stage, release state and accept another job" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    mlx.installErrorHandler();
    for ([_][]const u8{ "llama", "qwen3", "qwen3_5", "gemma4" }) |arch| for ([_]bool{ true, false }) |keep_everything_else| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
        const hybrid = std.mem.eql(u8, arch, "qwen3_5");
        const qwen35 = @import("../../../sdk/bases/proof/qwen35_pack.zig");
        const gemma = std.mem.eql(u8, arch, "gemma4");
        const gemma4 = @import("../../../plugins/gemma4/proof_test.zig");
        const json = if (gemma) try a.dupe(u8, gemma4.e_series) else if (hybrid) try a.dupe(u8, qwen35.config(false, true, 0)) else try packs.configJson(a, .{ .quantized = true });
        defer a.free(json);
        const config = if (hybrid) try a.dupe(u8, json) else try std.mem.replaceOwned(u8, a, json, "qwen3", arch);
        defer a.free(config);
        try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = config });
        var pack = fixture.Pack.init(32);
        defer pack.deinit();
        if (gemma) try gemma4.writeGemma4(&pack, .{}, 256, 2, true, false) else if (hybrid) try qwen35.writePack(&pack, false, true, 0) else try packs.writePack(&pack, .{ .quantized = true });
        try pack.save(path);
        var cfg = try config_mod.parseConfig(io, a, path);
        defer cfg.deinit(a);
        const s = mlx.mlx_default_gpu_stream_new();
        var xfm = @import("../plugin/driver.zig").load(io, a, &cfg, path, s) catch |err| {
            _ = mlx.mlx_stream_free(s);
            return err;
        };
        defer xfm.deinit();
        var tok = Tokenizer.initEmptyForTests(a, .wordpiece);
        defer tok.deinit();
        for (0..256) |id| {
            const word = switch (id) {
                0 => try a.dupe(u8, "[UNK]"),
                1 => try a.dupe(u8, "Yes"),
                2 => try a.dupe(u8, "No"),
                3 => try a.dupe(u8, "[END]"),
                4 => try a.dupe(u8, "user"),
                5 => try a.dupe(u8, "assistant"),
                6 => try a.dupe(u8, "yes"),
                else => try std.fmt.allocPrint(a, "token{d}", .{id}),
            };
            try tok.vocab.put(word, @intCast(id));
            try tok.id_to_token.put(@intCast(id), word);
        }
        tok.unk_id = 0;
        try tok.special_tokens.put(try a.dupe(u8, "[END]"), 3);
        var chat_config: @import("../../../host/chat/types.zig").ChatConfig = .{ .chat_template = "{% for m in messages %}{{ m.role }} {{ m.content }} {% endfor %}{% if add_generation_prompt %}assistant {% endif %}", .bos_token = null, .eos_token = "[END]", .add_bos_token = false, .allocator = a };
        var lm: LoadedModel = undefined;
        lm.path = path;
        lm.bytes_resident = 8 << 20;
        lm.tokenizer = &tok;
        lm.chat_config = &chat_config;
        lm.config = &cfg;
        lm.transformer = &xfm;
        lm.engine = null;
        const frozen = try @import("run.zig").frozen(&lm);
        var sch: @import("../scheduler.zig").Scheduler = undefined;
        sch.allocator = a;
        sch.promise_ledger = .{};
        const memory_override = @import("../../../host/registry/memory_ceiling.zig").overrides();
        const saved_ceiling = memory_override.static_ceiling;
        memory_override.static_ceiling = 1;
        var bill: types.Plan = .{};
        const admission = @import("run.zig").plan(&sch, &lm, &bill);
        memory_override.static_ceiling = saved_ceiling;
        try std.testing.expectError(error.TrainingDoesNotFit, admission);
        try std.testing.expect(bill.reserve_bytes > bill.available_bytes);
        cfg.plugin.?.overlay_sha256 = @splat(1);
        try std.testing.expectError(error.TrainingRequiresOriginalModel, @import("run.zig").plan(&sch, &lm, &bill));
        cfg.plugin.?.overlay_sha256 = null;
        const request: types.Request = .{ .model = "tiny", .name = "trained", .keep_everything_else = keep_everything_else, .prompts = &.{ "question one?", "question two?", "question three?", "question four?", "question five?", "question six?" } };
        const output = try std.fs.path.join(a, &.{ path, "trained" });
        defer a.free(output);
        var observer: Observer = .{};
        var teacher: SyntheticTeacher = .{ .a = a, .lm = &lm };
        var result: types.Result = .{};
        defer result.deinit(a);
        const input: types.Run = .{ .a = a, .io = io, .request = &request, .path = output, .observer = observer.observe(), .result = &result };
        const options: workflow.Options = .{ .prompts = 6, .holdout = 1, .minimum_train = 3, .retain = 2, .minimum_retain = 2, .updates = 2 };
        const live = ops.liveForTest();
        // Warm the gradient path's lazy constants before checking repeated cancellation for leaks.
        observer = .{ .cancel_at = .train };
        try std.testing.expectError(error.Cancelled, workflow.run(frozen, &teacher, input, cfg.plugin.?.source_sha256.?, options));
        try std.testing.expectEqual(live, ops.liveForTest());
        result.deinit(a);
        try mlx.check(mlx.mlx_synchronize(s));
        var resident: usize = 0;
        try mlx.check(mlx.mlx_get_active_memory(&resident));
        for ([_]types.Stage{ .synth, .train, .verify }) |stage| {
            observer = .{ .cancel_at = stage };
            try std.testing.expectError(error.Cancelled, workflow.run(frozen, &teacher, input, cfg.plugin.?.source_sha256.?, options));
            try std.testing.expectEqual(live, ops.liveForTest());
            try mlx.check(mlx.mlx_synchronize(s));
            var remaining: usize = 0;
            try mlx.check(mlx.mlx_get_active_memory(&remaining));
            try std.testing.expect(remaining <= resident);
            try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().statFile(io, output, .{}));
            result.deinit(a);
        }
        observer = .{};
        teacher.truncate_heldout = true;
        try std.testing.expectError(error.TrainingVerificationTruncated, workflow.run(frozen, &teacher, input, cfg.plugin.?.source_sha256.?, options));
        try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().statFile(io, output, .{}));
        result.deinit(a);
        teacher.truncate_heldout = false;
        teacher.invalid_samples = true;
        try mlx.check(mlx.mlx_reset_peak_memory());
        try workflow.run(frozen, &teacher, input, cfg.plugin.?.source_sha256.?, options);
        var peak: usize = 0;
        try mlx.check(mlx.mlx_get_peak_memory(&peak));
        try std.testing.expect(peak -| resident <= bill.reserve_bytes);
        try std.testing.expectEqual(live, ops.liveForTest());
        try std.testing.expectEqualSlices(bool, &.{ true, true, true }, &observer.stages);
        try std.testing.expect(observer.boundaries > 8);
        try std.testing.expectEqual(@as(u32, 1), result.metrics.heldout);
        try std.testing.expectEqualStrings("question one?", result.prompt);
        try std.testing.expectEqual(@as(u32, 3), result.training_examples);
        try std.testing.expect(result.metrics.retain_tokens > 0);
        try std.testing.expect(!result.metrics.improved());
        const weights = try std.fs.path.join(a, &.{ output, "adapters.safetensors" });
        defer a.free(weights);
        var adapter = try @import("../../../sdk/weights/lora.zig").LlmAdapter.load(a, weights, 2, 8);
        defer adapter.deinit();
        var native: Teacher = .{ .a = a, .io = io, .lm = &lm, .frozen = frozen, .observer = observer.observe() };
        const before = try native.generate("question one?", "", false, 4);
        defer a.free(before);
        const zero = try a.dupe(abi.LoraBinding, adapter.bindings);
        defer a.free(zero);
        for (zero) |*b| b.scale = 0;
        var identity_model = try @import("teacher.zig").Overlay.init(io, a, &lm, zero);
        defer identity_model.deinit();
        var identity_teacher = native;
        identity_teacher.lm = identity_model.model();
        const identity = try identity_teacher.generate("question one?", "", false, 4);
        defer a.free(identity);
        try std.testing.expectEqualStrings(before, identity);
        _ = try native.refuses("question one?", "yes");
    };
}
