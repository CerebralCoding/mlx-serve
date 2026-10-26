//! One micro-batch per call, with token-weighted accumulation and held-out stopping.
const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const grad = @import("grad.zig");
const loss = @import("loss.zig");
const adamw = @import("adamw.zig");
const Adapter = @import("adapter.zig").Adapter;
const Frozen = @import("model.zig").Frozen;
const Example = @import("packing.zig").Example;

pub const Options = struct { learning_rate: f32 = 1e-4, warmup_steps: u32 = 5, total_steps: u32 = 100, accumulation: u32 = 4, max_rows: u32 = 512, vocab_chunk: u32 = 1024, patience: u32 = 3 };
pub const Progress = struct { loss: f32, updated: bool, gradient_norm: ?f32 = null };
pub fn learningRate(o: Options, update: u32) f32 {
    if (update >= o.total_steps) return 0;
    if (update < o.warmup_steps) return o.learning_rate * @as(f32, @floatFromInt(update + 1)) / @as(f32, @floatFromInt(o.warmup_steps));
    const fraction = @as(f32, @floatFromInt(update - o.warmup_steps)) / @as(f32, @floatFromInt(o.total_steps - o.warmup_steps));
    return o.learning_rate * 0.5 * (1 + @cos(std.math.pi * fraction));
}

pub const Trainer = struct {
    a: std.mem.Allocator,
    s: ops.Stream,
    model: Frozen,
    adapter: *Adapter,
    optimizer: adamw.AdamW,
    accumulated: []ops.Array,
    pending: u32 = 0,
    tokens: u64 = 0,
    options: Options,
    best: []ops.Array,
    best_loss: f32 = std.math.inf(f32),
    stale: u32 = 0,

    pub fn init(a: std.mem.Allocator, s: ops.Stream, model: Frozen, adapter: *Adapter, options: Options) !Trainer {
        if (@import("builtin").os.tag != .macos or !mlx.streamOnGpuDevice(s)) return error.TrainingRequiresMetal;
        return initCore(a, s, model, adapter, options);
    }
    pub fn initForTest(a: std.mem.Allocator, s: ops.Stream, model: Frozen, adapter: *Adapter, options: Options) !Trainer {
        if (!@import("builtin").is_test) @compileError("CPU training is test-only");
        return initCore(a, s, model, adapter, options);
    }
    fn initCore(a: std.mem.Allocator, s: ops.Stream, model: Frozen, adapter: *Adapter, options: Options) !Trainer {
        if (!std.math.isFinite(options.learning_rate) or options.learning_rate <= 0 or options.total_steps == 0 or options.warmup_steps >= options.total_steps or options.accumulation == 0 or options.max_rows == 0 or options.max_rows > 512 or options.vocab_chunk == 0 or options.patience == 0) return error.InvalidTrainingOptions;
        const expected = try model.targets(a);
        defer a.free(expected);
        if (expected.len != adapter.targets.len) return error.InvalidTrainingAdapter;
        for (expected, adapter.targets) |x, y| if (!std.meta.eql(x, y)) return error.InvalidTrainingAdapter;
        var optimizer = try adamw.AdamW.init(a, s, adapter.params, .{});
        errdefer optimizer.deinit();
        const accumulated = try adamw.empty(a, adapter.params.len);
        errdefer adamw.release(a, accumulated);
        const best = try adamw.empty(a, adapter.params.len);
        return .{ .a = a, .s = s, .model = model, .adapter = adapter, .optimizer = optimizer, .accumulated = accumulated, .best = best, .options = options };
    }
    pub fn deinit(self: *Trainer) void {
        self.optimizer.deinit();
        adamw.release(self.a, self.accumulated);
        adamw.release(self.a, self.best);
    }
    pub fn microbatch(self: *Trainer, example: Example) !Progress {
        if (self.optimizer.updates >= self.options.total_steps) return error.TrainingComplete;
        try self.validate(example);
        var g = ops.Graph.init(self.a, self.s);
        defer g.deinit();
        const objective = try Objective.init(self, &g, example);
        const result = try grad.valueAndGrad(&g, objective, Objective.call, self.adapter.params);
        defer result.deinit(self.a);
        const value = try grad.item(result.value);
        if (!std.math.isFinite(value)) return error.NonFiniteTrainingLoss;
        const next = try adamw.empty(self.a, self.accumulated.len);
        defer adamw.release(self.a, next);
        for (result.gradients, 0..) |d, i| {
            const weighted = try g.mulScalar(d, @floatFromInt(example.completion_tokens));
            next[i] = g.escape(if (self.pending == 0) weighted else try g.add(self.accumulated[i], weighted));
        }
        try ops.evalMany(next);
        try adamw.finite(&g, next);
        const tokens = self.tokens + example.completion_tokens;
        if (self.pending + 1 == self.options.accumulation) {
            const norm = try self.apply(&g, next, tokens);
            self.clearAccumulation();
            return .{ .loss = value, .updated = true, .gradient_norm = norm };
        }
        for (self.accumulated, next) |*old, *new| std.mem.swap(ops.Array, old, new);
        self.pending += 1;
        self.tokens = tokens;
        return .{ .loss = value, .updated = false };
    }
    fn apply(self: *Trainer, g: *ops.Graph, accumulated: []const ops.Array, tokens: u64) !f32 {
        const mean = try self.a.alloc(ops.Array, accumulated.len);
        defer self.a.free(mean);
        for (accumulated, mean) |x, *y| y.* = try g.mulScalar(x, 1 / @as(f32, @floatFromInt(tokens)));
        return self.optimizer.update(self.adapter.params, mean, learningRate(self.options, self.optimizer.updates));
    }
    fn clearAccumulation(self: *Trainer) void {
        for (self.accumulated) |*x| {
            if (x.ctx != null) _ = mlx.mlx_array_free(x.*);
            x.* = .{};
        }
        self.pending = 0;
        self.tokens = 0;
    }
    pub fn flush(self: *Trainer) !void {
        if (self.pending == 0) return;
        var g = ops.Graph.init(self.a, self.s);
        defer g.deinit();
        _ = try self.apply(&g, self.accumulated, self.tokens);
        self.clearAccumulation();
    }
    pub fn evaluate(self: *Trainer, examples: []const Example) !f32 {
        if (examples.len == 0) return error.EmptyValidationSet;
        var sum: f64 = 0;
        var count: u64 = 0;
        for (examples) |example| {
            try self.validate(example);
            var g = ops.Graph.init(self.a, self.s);
            defer g.deinit();
            g.differentiable = true;
            const objective = try Objective.init(self, &g, example);
            const value = try grad.item(try objective.call(&g, self.adapter.params));
            if (!std.math.isFinite(value)) return error.NonFiniteTrainingLoss;
            sum += @as(f64, value) * @as(f64, @floatFromInt(example.completion_tokens));
            count += example.completion_tokens;
        }
        return @floatCast(sum / @as(f64, @floatFromInt(count)));
    }
    pub fn observeValidation(self: *Trainer, value: f32) !bool {
        if (!std.math.isFinite(value)) return error.NonFiniteTrainingLoss;
        if (value < self.best_loss) {
            var g = ops.Graph.init(self.a, self.s);
            defer g.deinit();
            for (self.best, self.adapter.params) |*old, new| {
                if (old.ctx != null) _ = mlx.mlx_array_free(old.*);
                old.* = g.escape(new);
            }
            self.best_loss = value;
            self.stale = 0;
        } else self.stale +|= 1;
        return self.stale >= self.options.patience;
    }
    pub fn restoreBest(self: *Trainer) !void {
        if (!std.math.isFinite(self.best_loss)) return error.NoTrainingCheckpoint;
        var g = ops.Graph.init(self.a, self.s);
        defer g.deinit();
        for (self.adapter.params, self.best) |*p, best| {
            _ = mlx.mlx_array_free(p.*);
            p.* = g.escape(best);
        }
        self.clearAccumulation();
        // Restoring a validation checkpoint ends training; optimizer moments belong to the later state.
        self.optimizer.updates = self.options.total_steps;
    }
    fn validate(self: *Trainer, e: Example) !void {
        if (e.inputs.len == 0 or e.inputs.len > self.options.max_rows or e.inputs.len != e.targets.len or e.inputs.len != e.mask.len) return error.InvalidTrainingShape;
        const vocab: u32 = @intCast(ops.shape(self.model.base.weights.head.w)[0]);
        var count: u32 = 0;
        for (e.inputs, e.targets, e.mask) |x, y, m| {
            if (x >= vocab or y >= vocab or (m != 0 and m != 1)) return error.InvalidTrainingExample;
            if (m == 1) count += 1;
        }
        if (count == 0 or count != e.completion_tokens) return error.InvalidTrainingExample;
    }
};

const Objective = struct {
    trainer: *Trainer,
    ids: ops.Array,
    targets: ops.Array,
    mask: ops.Array,

    fn init(trainer: *Trainer, g: *ops.Graph, example: Example) !Objective {
        const rows: c_int = @intCast(example.inputs.len);
        return .{ .trainer = trainer, .ids = try g.fromSlice(u32, example.inputs, &.{ 1, rows }), .targets = try g.fromSlice(u32, example.targets, &.{rows}), .mask = try g.fromSlice(f32, example.mask, &.{rows}) };
    }
    fn call(self: Objective, g: *ops.Graph, params: []const ops.Array) !ops.Array {
        const bindings = try self.trainer.adapter.bindings(g, params);
        defer g.a.free(bindings);
        const hidden = try self.trainer.model.hidden(g, self.ids, bindings);
        return loss.projected(g, hidden, self.trainer.model.base.weights.head, self.targets, self.mask, self.trainer.options.vocab_chunk);
    }
};

test "train core: warmup and cosine use optimizer updates and reach zero at the limit" {
    const o: Options = .{ .learning_rate = 0.01, .warmup_steps = 2, .total_steps = 6 };
    try std.testing.expectApproxEqAbs(@as(f32, 0.005), learningRate(o, 0), 1e-8);
    try std.testing.expectApproxEqAbs(@as(f32, 0.01), learningRate(o, 1), 1e-8);
    try std.testing.expect(learningRate(o, 3) > learningRate(o, 4));
    try std.testing.expectEqual(@as(f32, 0), learningRate(o, 6));
}
