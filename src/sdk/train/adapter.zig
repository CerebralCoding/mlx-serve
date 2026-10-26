//! f32 master weights and mlx-lm-compatible exports of the compute weights.
const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const abi = @import("../../abi/plugin.zig");
const adamw = @import("adamw.zig");
const grad = @import("grad.zig");
pub const Target = struct { layer: u32, projection: abi.LoraProjection, input: u32, output: u32 };
pub const Options = struct { rank: u32 = 8, alpha: f32 = 16, seed: u64 = 0, compute: ops.Dtype = .bfloat16 };
pub const Adapter = struct {
    a: std.mem.Allocator,
    s: ops.Stream,
    targets: []Target,
    params: []ops.Array,
    options: Options,

    pub fn init(a: std.mem.Allocator, s: ops.Stream, targets: []const Target, options: Options) !Adapter {
        if (targets.len == 0 or options.rank == 0 or options.rank > 256 or !std.math.isFinite(options.alpha) or options.alpha <= 0 or (options.compute != .bfloat16 and options.compute != .float32)) return error.InvalidTrainingAdapter;
        const owned_targets = try a.dupe(Target, targets);
        errdefer a.free(owned_targets);
        const params = try adamw.empty(a, targets.len * 2);
        errdefer adamw.release(a, params);
        var g = ops.Graph.init(a, s);
        defer g.deinit();
        for (targets, 0..) |target, i| {
            if (target.input == 0 or target.output == 0 or target.input > std.math.maxInt(c_int) or target.output > std.math.maxInt(c_int)) return error.InvalidTrainingAdapter;
            for (targets[0..i]) |previous| if (target.layer == previous.layer and target.projection == previous.projection) return error.InvalidTrainingAdapter;
            const key = try g.call(mlx.mlx_random_key, .{options.seed +% i});
            const shape = [_]c_int{ @intCast(target.input), @intCast(options.rank) };
            const bound = 1 / @sqrt(@as(f32, @floatFromInt(target.input)));
            const low = try g.scalar(-bound, .float32);
            const high = try g.scalar(bound, .float32);
            const x = try g.call(mlx.mlx_random_uniform, .{ low, high, &shape, shape.len, ops.Dtype.float32, key });
            params[2 * i] = g.escape(x);
            params[2 * i + 1] = g.escape(try g.zeros(&.{ @intCast(options.rank), @intCast(target.output) }, .float32));
        }
        try ops.evalMany(params);
        return .{ .a = a, .s = s, .targets = owned_targets, .params = params, .options = options };
    }
    pub fn deinit(self: *Adapter) void {
        adamw.release(self.a, self.params);
        self.a.free(self.targets);
    }
    pub fn bindings(self: *const Adapter, g: *ops.Graph, params: []const ops.Array) ![]abi.LoraBinding {
        if (params.len != self.targets.len * 2) return error.InvalidTrainableParameters;
        const result = try g.a.alloc(abi.LoraBinding, self.targets.len);
        errdefer g.a.free(result);
        for (self.targets, result, 0..) |target, *binding, i| {
            binding.* = .{ .layer = target.layer, .projection = target.projection, .a = .{ .ctx = (try g.astype(params[2 * i], self.options.compute)).ctx }, .b = .{ .ctx = (try g.astype(params[2 * i + 1], self.options.compute)).ctx }, .scale = self.options.alpha / @as(f32, @floatFromInt(self.options.rank)) };
        }
        return result;
    }
    pub fn save(self: *const Adapter, io: std.Io, path: []const u8, base_config_sha256: [32]u8) !void {
        const cwd = std.Io.Dir.cwd();
        try cwd.createDir(io, path, .default_dir);
        errdefer cwd.deleteTree(io, path) catch {};
        var g = ops.Graph.init(self.a, self.s);
        defer g.deinit();
        const tensors = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(tensors);
        const metadata = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(metadata);
        for (self.targets, 0..) |target, i| {
            for ([_][]const u8{ "a", "b" }, 0..) |suffix, j| {
                const key = try std.fmt.allocPrintSentinel(self.a, "model.layers.{d}.{s}.lora_{s}", .{ target.layer, projectionName(target.projection), suffix }, 0);
                defer self.a.free(key);
                const x = try g.astype(self.params[2 * i + j], self.options.compute);
                try ops.eval(x);
                try mlx.check(mlx.mlx_map_string_to_array_insert(tensors, key, x));
            }
        }
        const file = try std.fmt.allocPrintSentinel(self.a, "{s}/adapters.safetensors", .{path}, 0);
        defer self.a.free(file);
        try mlx.check(mlx.mlx_save_safetensors(file, tensors, metadata));
        // Publish the config last: registry discovery treats it as the completed adapter.
        const json = try std.json.Stringify.valueAlloc(self.a, .{ .fine_tune_type = "lora", .lora_parameters = .{ .rank = self.options.rank, .scale = self.options.alpha / @as(f32, @floatFromInt(self.options.rank)), .dropout = @as(f32, 0) }, .base_config_sha256 = std.fmt.bytesToHex(base_config_sha256, .lower) }, .{});
        defer self.a.free(json);
        var dir = try cwd.openDir(io, path, .{});
        defer dir.close(io);
        try dir.writeFile(io, .{ .sub_path = "adapter_config.json", .data = json });
    }
};

fn projectionName(projection: abi.LoraProjection) []const u8 {
    return switch (projection) {
        .q => "self_attn.q_proj",
        .k => "self_attn.k_proj",
        .v => "self_attn.v_proj",
        .o => "self_attn.o_proj",
        .gate => "mlp.gate_proj",
        .up => "mlp.up_proj",
        .down => "mlp.down_proj",
    };
}

test "train core: adapter starts as identity and exports runtime compute weights with provenance" {
    const a = std.testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var adapter = try Adapter.init(a, s, &.{.{ .layer = 1, .projection = .q, .input = 4, .output = 3 }}, .{ .rank = 2, .alpha = 4 });
    defer adapter.deinit();
    var g = ops.Graph.init(a, s);
    defer g.deinit();
    try std.testing.expectEqual(.float32, ops.dtype(adapter.params[0]));
    try std.testing.expectEqual(@as(f32, 0), try grad.item(try g.sum(try g.reshape(try g.abs(adapter.params[1]), &.{-1}), 0, false)));
    _ = mlx.mlx_array_free(adapter.params[1]);
    adapter.params[1] = g.escape(try g.full(&.{ 2, 3 }, try g.scalar(0.1234, .float32), .float32));
    const bindings = try adapter.bindings(&g, adapter.params);
    defer a.free(bindings);
    try std.testing.expectEqual(.bfloat16, ops.dtype(.{ .ctx = bindings[0].a.ctx }));
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = buf[0..try tmp.dir.realPath(std.testing.io, &buf)];
    const path = try std.fs.path.join(a, &.{ dir, "trained" });
    defer a.free(path);
    const config = @import("../weights/lora_config.zig");
    const hash = config.fingerprint("frozen config");
    try adapter.save(std.testing.io, path, hash);
    try std.testing.expectError(error.PathAlreadyExists, adapter.save(std.testing.io, path, hash));
    const file = try std.fs.path.join(a, &.{ path, "adapters.safetensors" });
    defer a.free(file);
    var loaded = try @import("../weights/lora.zig").LlmAdapter.load(a, file, 2, 2);
    defer loaded.deinit();
    try ops.expectBitEqual(.{ .ctx = bindings[0].a.ctx }, .{ .ctx = loaded.bindings[0].a.ctx });
    try ops.expectBitEqual(.{ .ctx = bindings[0].b.ctx }, .{ .ctx = loaded.bindings[0].b.ctx });
    const json = try tmp.dir.readFileAlloc(std.testing.io, "trained/adapter_config.json", a, .limited(4096));
    defer a.free(json);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, json, .{});
    defer parsed.deinit();
    const cfg = try config.Config.parse(parsed.value);
    try config.validateBase(cfg, hash, null);
    try std.testing.expectEqual(@as(f32, 2), cfg.scale);
}
