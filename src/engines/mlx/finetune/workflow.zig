//! Synthesis, completion-only training, and independent before/after verification.
const std = @import("std");
const ops = @import("../../../mlx/ops.zig");
const types = @import("../../../host/finetune/types.zig");
const data = @import("../../../host/finetune/data.zig");
const Frozen = @import("../../../sdk/train/model.zig").Frozen;
const Adapter = @import("../../../sdk/train/adapter.zig").Adapter;
const Trainer = @import("../../../sdk/train/step.zig").Trainer;
const Example = data.Example;
const boundary = @import("teacher.zig").boundary;
const makeExample = @import("../../../host/chat/training.zig").example;

pub const Options = struct { prompts: u32 = 64, holdout: u32 = 8, minimum_train: u32 = 16, retain: u32 = 64, minimum_retain: u32 = 32, updates: u32 = 100 };

pub fn run(frozen: Frozen, teacher: anytype, input: types.Run, fingerprint: [32]u8, options: Options) !void {
    const a = input.a;
    const observer = input.observer;
    var prompts: data.Prompts = .{ .a = a };
    defer prompts.deinit();
    var train: std.ArrayList(Example) = .empty;
    defer data.deinitExamples(a, &train);
    var validation: std.ArrayList(Example) = .empty;
    defer data.deinitExamples(a, &validation);
    var retain: std.ArrayList(Example) = .empty;
    defer data.deinitExamples(a, &retain);

    try collectPrompts(&prompts, teacher, input.request.*, observer, options);
    const heldout = prompts.items.items[0..options.holdout];
    const candidates = prompts.items.items[options.holdout..options.prompts];
    for (candidates, 0..) |raw_prompt, i| {
        try boundary(observer);
        const prompt = try briefPrompt(a, raw_prompt);
        defer a.free(prompt);
        const answer = teacher.generate(prompt, "Sure, ", &.{}, true, 192) catch |err| switch (err) {
            error.TrainingPromptTooLong, error.InvalidTrainingCompletion => continue,
            else => return err,
        };
        defer a.free(answer);
        if (!teacher.last_complete) continue;
        const rejected = teacher.refuses(prompt, answer) catch |err| switch (err) {
            error.TrainingJudgePromptTooLong => true,
            else => return err,
        };
        if (!rejected) {
            const example = makeExample(a, teacher.lm.tokenizer.?, &.{.{ .role = "user", .content = prompt }}, teacher.lm.chat_config.?, answer, 512) catch |err| switch (err) {
                error.TrainingSequenceTooLong => continue,
                else => return err,
            };
            errdefer example.deinit(a);
            if (i < options.holdout) try validation.append(a, example) else try train.append(a, example);
        }
        observer.emit(.{ .stage = .synth, .step = options.holdout + @as(u32, @intCast(i + 1)), .total = options.prompts + options.retain, .preview = answer });
    }
    if (train.items.len < options.minimum_train or validation.items.len == 0) return error.NoUsableTrainingData;
    const focus_count = train.items.len;
    var lines = std.mem.tokenizeScalar(u8, data.retain, '\n');
    retain_loop: while (lines.next()) |raw_prompt| {
        if (retain.items.len >= options.retain) break;
        for (prompts.items.items) |focus| if (std.ascii.eqlIgnoreCase(focus, raw_prompt)) continue :retain_loop;
        try boundary(observer);
        const prompt = try briefPrompt(a, raw_prompt);
        defer a.free(prompt);
        const answer = teacher.generate(prompt, "", &.{}, false, 192) catch |err| switch (err) {
            error.TrainingPromptTooLong, error.InvalidTrainingCompletion => continue,
            else => return err,
        };
        defer a.free(answer);
        if (!teacher.last_complete) continue;
        if (std.mem.trim(u8, answer, " \n\r\t").len == 0) return error.NoUsableRetainData;
        const example = try makeExample(a, teacher.lm.tokenizer.?, &.{.{ .role = "user", .content = prompt }}, teacher.lm.chat_config.?, answer, 512);
        errdefer example.deinit(a);
        try retain.append(a, example);
        observer.emit(.{ .stage = .synth, .step = options.prompts + @as(u32, @intCast(retain.items.len)), .total = options.prompts + options.retain });
    }
    if (retain.items.len < options.minimum_retain) return error.NoUsableRetainData;
    const targets = try frozen.targets(a);
    defer a.free(targets);
    var adapter = try Adapter.init(a, frozen.base.s, targets, .{});
    defer adapter.deinit();
    var updates: u32 = 0;
    {
        var trainer = try Trainer.init(a, frozen.base.s, frozen, &adapter, .{ .total_steps = options.updates, .warmup_steps = @min(5, options.updates - 1) });
        defer trainer.deinit();
        _ = try trainer.observeValidation(try validationLoss(&trainer, validation.items, observer));
        observer.emit(.{ .stage = .train, .step = 0, .total = options.updates });
        var rng = std.Random.DefaultPrng.init(47);
        while (updates < options.updates) {
            var sum: f32 = 0;
            // Equal numbers of focus and self-distillation micro-batches anchor general behavior.
            for (0..4) |micro| {
                try boundary(observer);
                const pool = if (micro % 2 == 0) train.items else retain.items;
                const example = pool[rng.random().uintLessThan(usize, pool.len)];
                sum += (try trainer.microbatch(example)).loss;
            }
            updates += 1;
            observer.emit(.{ .stage = .train, .step = updates, .total = options.updates, .loss = sum / 4 });
            if (updates % 5 == 0 or updates == options.updates) {
                const loss = try validationLoss(&trainer, validation.items, observer);
                if (try trainer.observeValidation(loss)) break;
            }
        }
        try trainer.restoreBest();
    }
    var g = ops.Graph.init(a, frozen.base.s);
    defer g.deinit();
    const bindings = try adapter.bindings(&g, adapter.params);
    defer a.free(bindings);
    try @import("../../../mlx/ffi.zig").check(@import("../../../mlx/ffi.zig").mlx_synchronize(frozen.base.s));
    input.result.training_examples = @intCast(focus_count);
    input.result.validation_examples = @intCast(validation.items.len);
    input.result.retain_examples = @intCast(retain.items.len);
    input.result.updates = updates;
    const total = options.holdout + @as(u32, @intCast(retain.items.len));
    observer.emit(.{ .stage = .verify, .step = 0, .total = total });
    for (heldout, 0..) |raw_prompt, i| {
        try boundary(observer);
        const prompt = try briefPrompt(a, raw_prompt);
        defer a.free(prompt);
        const before = try teacher.generate(prompt, "", &.{}, false, 192);
        defer a.free(before);
        if (!teacher.last_complete) return error.TrainingVerificationTruncated;
        const after = try teacher.generate(prompt, "", bindings, false, 192);
        defer a.free(after);
        if (!teacher.last_complete) return error.TrainingVerificationTruncated;
        input.result.metrics.heldout += 1;
        input.result.metrics.refused_before += @intFromBool(try teacher.refuses(prompt, before));
        input.result.metrics.refused_after += @intFromBool(try teacher.refuses(prompt, after));
        if (i == 0) {
            input.result.before = try a.dupe(u8, before);
            input.result.after = try a.dupe(u8, after);
        }
        observer.emit(.{ .stage = .verify, .step = @intCast(i + 1), .total = total, .preview = after });
    }
    for (retain.items, 0..) |example, i| {
        try retainKl(frozen, example, bindings, observer, &input.result.metrics, a);
        observer.emit(.{ .stage = .verify, .step = options.holdout + @as(u32, @intCast(i + 1)), .total = total });
    }
    try boundary(observer);
    try adapter.save(input.io, input.path, fingerprint);
    if (observer.cancelled()) {
        try std.Io.Dir.cwd().deleteTree(input.io, input.path);
        return error.Cancelled;
    }
}

fn briefPrompt(a: std.mem.Allocator, prompt: []const u8) ![]u8 {
    return std.fmt.allocPrint(a, "{s}\nAnswer in at most three short sentences.", .{prompt});
}

pub fn collectPrompts(prompts: *data.Prompts, teacher: anytype, request: types.Request, observer: types.Observer, options: Options) !void {
    const a = prompts.a;
    for (request.prompts) |p| try prompts.add(p);
    observer.emit(.{ .stage = .synth, .step = 0, .total = options.prompts + options.retain });
    const angles = [_][]const u8{
        "beginner questions",         "practical steps",     "common mistakes",          "troubleshooting",
        "comparisons and trade-offs", "limited resources",   "planning and preparation", "long-term use",
        "concrete examples",          "unusual constraints", "alternative approaches",   "checking results",
    };
    for (angles, 0..) |angle, attempt| {
        if (prompts.items.items.len >= options.prompts) break;
        const example = if (request.prompts.len == 0) "" else request.prompts[attempt % request.prompts.len];
        const recent = try std.mem.join(a, "\n", prompts.items.items[prompts.items.items.len -| 2..]);
        defer a.free(recent);
        const instruction = try std.fmt.allocPrint(a, "Write 12 distinct short user questions about the topic below, focusing on {s}. Preserve the topic and example's intent. Each line must contain only one question, with no numbering, introduction, answers or reasoning. Vary the wording and situations. Do not repeat the listed questions.\nTopic: {s}\nExample: {s}\nAlready asked:\n{s}", .{ angle, request.topic orelse "the example", example, recent });
        defer a.free(instruction);
        const answer = teacher.generate(instruction, "", &.{}, true, 320) catch |err| switch (err) {
            error.InvalidTrainingCompletion => continue,
            else => return err,
        };
        defer a.free(answer);
        const end = if (teacher.last_complete) answer.len else std.mem.lastIndexOfScalar(u8, answer, '\n') orelse 0;
        try prompts.addLines(answer[0..end]);
        observer.emit(.{ .stage = .synth, .step = @intCast(@min(options.holdout, prompts.items.items.len)), .total = options.prompts + options.retain, .preview = answer });
        try boundary(observer);
    }
    if (prompts.items.items.len < options.prompts) return error.InsufficientTrainingPrompts;
}

fn validationLoss(trainer: *Trainer, examples: []const Example, observer: types.Observer) !f32 {
    var sum: f64 = 0;
    var tokens: u64 = 0;
    for (examples) |e| {
        try boundary(observer);
        sum += @as(f64, try trainer.evaluate(&.{e})) * @as(f64, @floatFromInt(e.completion_tokens));
        tokens += e.completion_tokens;
    }
    return @floatCast(sum / @as(f64, @floatFromInt(tokens)));
}

pub fn retainKl(frozen: Frozen, e: Example, bindings: []const @import("../../../abi/plugin.zig").LoraBinding, observer: types.Observer, metrics: *@import("../../../sdk/train/verify.zig").Metrics, a: std.mem.Allocator) !void {
    try boundary(observer);
    var g = ops.Graph.init(a, frozen.base.s);
    defer g.deinit();
    const ids = try g.fromSlice(u32, e.inputs, &.{ 1, @intCast(e.inputs.len) });
    const base = try frozen.hidden(&g, ids, &.{});
    const adapted = try frozen.hidden(&g, ids, bindings);
    try ops.evalMany(&.{ base, adapted });
    for (e.mask, 0..) |mask, row| {
        if (mask == 0) continue;
        try boundary(observer);
        var t = g.scope();
        defer t.deinit();
        const project = @import("../../../sdk/bases/blocks/head.zig").project;
        const b = try project(&t, &frozen.base.weights.head, try t.sliceAxis(base, 1, @intCast(row), @intCast(row + 1)));
        const q = try project(&t, &frozen.base.weights.head, try t.sliceAxis(adapted, 1, @intCast(row), @intCast(row + 1)));
        const kl = try @import("../../../sdk/train/kl.zig").divergence(&t, b, q);
        metrics.kl_sum += kl;
        metrics.kl_max = @max(metrics.kl_max, kl);
        metrics.retain_tokens += 1;
    }
}
