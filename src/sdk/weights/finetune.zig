//! A model-local adapter is merged before the plugin prepares its inference weights.
const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const abi = @import("../../abi/plugin.zig");
const Weights = @import("resident.zig").Weights;

pub const filename = "finetune.lora.safetensor";
const config = @import("lora_config.zig");
const LlmAdapter = @import("lora.zig").LlmAdapter;

pub const Sidecar = struct {
    adapter: LlmAdapter,
    digest: [32]u8,

    pub fn deinit(self: *Sidecar) void {
        self.adapter.deinit();
    }
};

pub fn load(a: std.mem.Allocator, io: std.Io, model_dir: []const u8, base_hash: ?[32]u8) !?Sidecar {
    const path = try std.fs.path.join(a, &.{ model_dir, filename });
    defer a.free(path);
    const bytes = readRegular(a, io, path, 1024 * 1024 * 1024) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => return err,
    };
    defer a.free(bytes);
    if (bytes.len < 8) return error.InvalidLlmAdapter;
    const header_len = std.mem.readInt(u64, bytes[0..8], .little);
    if (header_len > bytes.len - 8 or header_len > 8 * 1024 * 1024) return error.InvalidLlmAdapter;
    const header = std.json.parseFromSlice(std.json.Value, a, bytes[8..][0..header_len], .{}) catch |err| return if (err == error.OutOfMemory) err else error.InvalidLlmAdapter;
    defer header.deinit();
    if (header.value != .object) return error.InvalidLlmAdapter;
    var external: ?[]u8 = null;
    defer if (external) |body| a.free(body);
    const body = blk: {
        if (header.value.object.get("__metadata__")) |meta| if (meta != .null) {
            if (meta != .object) return error.InvalidLlmAdapter;
            if (meta.object.get("adapter_config")) |value| {
                if (value != .string) return error.InvalidLlmAdapterConfig;
                break :blk value.string;
            }
        };
        const config_path = try std.fs.path.join(a, &.{ model_dir, "adapter_config.json" });
        defer a.free(config_path);
        external = readRegular(a, io, config_path, 1024 * 1024) catch |err| switch (err) {
            error.FileNotFound => return error.InvalidLlmAdapterConfig,
            else => return err,
        };
        break :blk external.?;
    };
    const parsed = std.json.parseFromSlice(std.json.Value, a, body, .{}) catch |err| return if (err == error.OutOfMemory) err else error.InvalidLlmAdapterConfig;
    defer parsed.deinit();
    const cfg = try config.Config.parse(parsed.value);
    var resolved: ?[32]u8 = null;
    if (cfg.base_config_sha256 == null) if (cfg.base_model) |base| {
        if (std.fs.path.isAbsolute(base)) {
            const config_path = try std.fs.path.join(a, &.{ base, "config.json" });
            defer a.free(config_path);
            const raw = readRegular(a, io, config_path, 4 * 1024 * 1024) catch |err| switch (err) {
                error.OutOfMemory => return err,
                else => null,
            };
            if (raw) |value| {
                defer a.free(value);
                resolved = config.fingerprint(value);
            }
        }
    };
    try config.validateBase(cfg, base_hash orelse return error.LoraBaseFingerprintRequired, resolved);
    var adapter = try LlmAdapter.load(a, path, cfg.scale, cfg.rank);
    errdefer adapter.deinit();
    const current = try readRegular(a, io, path, 1024 * 1024 * 1024);
    defer a.free(current);
    if (!std.mem.eql(u8, &config.fingerprint(bytes), &config.fingerprint(current))) return error.LoraRevisionChanged;
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(bytes);
    hash.update(body);
    return .{ .adapter = adapter, .digest = hash.finalResult() };
}

fn readRegular(a: std.mem.Allocator, io: std.Io, path: []const u8, limit: usize) ![]u8 {
    const st = try std.Io.Dir.cwd().statFile(io, path, .{});
    if (st.kind != .file) return error.InvalidLlmAdapter;
    if (st.size > limit) return error.LoraAdapterTooLarge;
    return std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(limit));
}

pub fn apply(weights: *Weights, s: ops.Stream, bindings: []const abi.LoraBinding, prefix: []const u8) !void {
    for (bindings) |binding| {
        if (!std.math.isFinite(binding.scale)) return error.InvalidLlmAdapter;
        if (binding.scale == 0) continue;
        var g = ops.Graph.init(weights.allocator, s);
        defer g.deinit();
        var name_buf: [256]u8 = undefined;
        var scale_buf: [256]u8 = undefined;
        var bias_buf: [256]u8 = undefined;
        const name = try std.fmt.bufPrint(&name_buf, "{s}.layers.{d}.{s}.weight", .{ prefix, binding.layer, projectionName(binding.projection) });
        const stem = name[0 .. name.len - ".weight".len];
        const scale_name = try std.fmt.bufPrint(&scale_buf, "{s}.scales", .{stem});
        const bias_name = try std.fmt.bufPrint(&bias_buf, "{s}.biases", .{stem});
        const w = weights.get(name) orelse return error.InvalidLlmAdapter;
        const scales = weights.get(scale_name);
        const left: ops.Array = .{ .ctx = binding.a.ctx };
        const right: ops.Array = .{ .ctx = binding.b.ctx };
        const ls = ops.shape(left);
        const rs = ops.shape(right);
        const ws = ops.shape(w);
        if (ls.len != 2 or rs.len != 2 or ws.len != 2 or ls[0] <= 0 or ls[1] <= 0 or ls[1] != rs[0] or ws[0] != rs[1]) return error.InvalidLlmAdapter;
        const qp = if (scales) |sc| @import("../quant/params.zig").affineParamsFromGeometry(w, sc, @intCast(ls[0])) orelse return error.LoraUnsupportedQuantization else null;
        if (qp == null and ws[1] != ls[0]) return error.InvalidLlmAdapter;
        const maximum = try g.call(mlx.mlx_max, .{ try g.abs(try g.astype(right, .float32)), false });
        try ops.eval(maximum);
        var magnitude: f32 = 0;
        try mlx.check(mlx.mlx_array_item_float32(&magnitude, maximum));
        if (magnitude == 0) continue;
        if (!std.math.isFinite(magnitude)) return error.InvalidLlmAdapter;
        const dtype = if (scales) |sc| ops.dtype(sc) else ops.dtype(w);
        if (dtype != .float32 and dtype != .float16 and dtype != .bfloat16) return error.LoraUnsupportedQuantization;
        const dense = if (qp) |p| try g.call(mlx.mlx_dequantize, .{
            w,                                                 scales.?,                                                       weights.get(bias_name) orelse return error.LoraUnsupportedQuantization,
            mlx.mlx_optional_int.some(@intCast(p.group_size)), mlx.mlx_optional_int.some(@intCast(p.bits)),                    "affine",
            ops.Array{},                                       mlx.mlx_optional_dtype{ .value = .float32, .has_value = true },
        }) else try g.astype(w, .float32);
        const delta = try g.transpose(try g.matmul(try g.astype(left, .float32), try g.astype(right, .float32)), &.{ 1, 0 });
        const merged = try g.astype(try g.add(dense, try g.mulScalar(delta, binding.scale)), dtype);
        const merged_max = try g.call(mlx.mlx_max, .{ try g.abs(try g.astype(merged, .float32)), false });
        try ops.eval(merged_max);
        try mlx.check(mlx.mlx_array_item_float32(&magnitude, merged_max));
        if (!std.math.isFinite(magnitude)) return error.InvalidLlmAdapter;
        if (qp) |p| {
            const quantized = try quantize(&g, merged, @intCast(p.bits), @intCast(p.group_size));
            try ops.evalMany(&quantized);
            for ([_][]const u8{ name, scale_name, bias_name }, quantized) |key, x| weights.replace(key, g.escape(x));
        } else {
            try ops.eval(merged);
            weights.replace(name, g.escape(merged));
        }
    }
}

pub fn projectionName(projection: abi.LoraProjection) []const u8 {
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

test "finetune overlay: merge preserves dense layout, unrelated weights and a zero adapter" {
    const a = std.testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = ops.Graph.init(a, s);
    defer g.deinit();
    var weights = Weights.init(a);
    defer weights.deinit();
    const w = try g.fromSlice(f32, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 3, 2 });
    const left = try g.fromSlice(f32, &.{ 1, 2 }, &.{ 2, 1 });
    const right = try g.fromSlice(f32, &.{ 3, 4, 5 }, &.{ 1, 3 });
    try weights.put("model.layers.0.self_attn.q_proj.weight", g.escape(w));
    try weights.put("model.norm.weight", g.escape(w));
    const original = weights.get("model.layers.0.self_attn.q_proj.weight").?;
    const norm = weights.get("model.norm.weight").?;
    var bindings = [_]abi.LoraBinding{.{ .layer = 0, .projection = .q, .a = .{ .ctx = left.ctx }, .b = .{ .ctx = right.ctx }, .scale = 0 }};
    try apply(&weights, s, &bindings, "model");
    try std.testing.expectEqual(original.ctx, weights.get("model.layers.0.self_attn.q_proj.weight").?.ctx);
    bindings[0].scale = 2;
    try apply(&weights, s, &bindings, "model");
    const want = try g.fromSlice(f32, &.{ 7, 14, 11, 20, 15, 26 }, &.{ 3, 2 });
    try ops.expectBitEqual(want, weights.get("model.layers.0.self_attn.q_proj.weight").?);
    try std.testing.expectEqual(norm.ctx, weights.get("model.norm.weight").?.ctx);
    try std.testing.expectEqualSlices(c_int, &.{ 3, 2 }, ops.shape(weights.get("model.layers.0.self_attn.q_proj.weight").?));
}

test "finetune overlay: finite inputs that overflow the stored dtype leave weights untouched" {
    const a = std.testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = ops.Graph.init(a, s);
    defer g.deinit();
    var weights = Weights.init(a);
    defer weights.deinit();
    const key = "model.layers.0.self_attn.q_proj.weight";
    try weights.put(key, g.escape(try g.ones(&.{ 2, 2 }, .float16)));
    const original = weights.get(key).?;
    const left = try g.full(&.{ 2, 1 }, try g.scalar(1e30, .float32), .float32);
    const right = try g.ones(&.{ 1, 2 }, .float32);
    const binding: abi.LoraBinding = .{ .layer = 0, .projection = .q, .a = .{ .ctx = left.ctx }, .b = .{ .ctx = right.ctx }, .scale = 1 };
    try std.testing.expectError(error.InvalidLlmAdapter, apply(&weights, s, &.{binding}, "model"));
    try std.testing.expectEqual(original.ctx, weights.get(key).?.ctx);
}

fn quantize(g: *ops.Graph, w: ops.Array, bits: c_int, group: c_int) ![3]ops.Array {
    var vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_quantize(&vec, w, mlx.mlx_optional_int.some(group), mlx.mlx_optional_int.some(bits), "affine", .{}, g.s));
    var result: [3]ops.Array = undefined;
    for (&result, 0..) |*x, i| {
        var value = mlx.mlx_array_new();
        mlx.check(mlx.mlx_vector_array_get(&value, vec, i)) catch |err| {
            _ = mlx.mlx_array_free(value);
            return err;
        };
        x.* = try g.adopt(value);
    }
    return result;
}

test "finetune overlay: affine merge retains each tensor's bit width, group and scale dtype" {
    const a = std.testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    for ([_]ops.Dtype{ .bfloat16, .float16, .float32 }) |dt| for ([_]c_int{ 2, 3, 4, 5, 6, 8 }) |bits| for ([_]c_int{ 32, 64, 128 }) |group| {
        var g = ops.Graph.init(a, s);
        defer g.deinit();
        var data: [512]f32 = undefined;
        for (&data, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.13);
        const w = try g.astype(try g.fromSlice(f32, &data, &.{ 4, 128 }), dt);
        const base = try quantize(&g, w, bits, group);
        var weights = Weights.init(a);
        defer weights.deinit();
        const keys = [_][]const u8{ "language_model.model.layers.0.mlp.down_proj.weight", "language_model.model.layers.0.mlp.down_proj.scales", "language_model.model.layers.0.mlp.down_proj.biases" };
        for (keys, base) |key, x| try weights.put(key, g.escape(x));
        const left = try g.fromSlice(f32, data[0..256], &.{ 128, 2 });
        const right = try g.fromSlice(f32, data[0..8], &.{ 2, 4 });
        var binding: abi.LoraBinding = .{ .layer = 0, .projection = .down, .a = .{ .ctx = left.ctx }, .b = .{ .ctx = (try g.zeros(&.{ 2, 4 }, .float32)).ctx }, .scale = 0.2 };
        const original = weights.get(keys[0]).?;
        try apply(&weights, s, &.{binding}, "language_model.model");
        try std.testing.expectEqual(original.ctx, weights.get(keys[0]).?.ctx);
        binding.b.ctx = right.ctx;
        const dense = try g.call(mlx.mlx_dequantize, .{ base[0], base[1], base[2], mlx.mlx_optional_int.some(group), mlx.mlx_optional_int.some(bits), "affine", ops.Array{}, mlx.mlx_optional_dtype{ .value = .float32, .has_value = true } });
        const delta = try g.transpose(try g.matmul(left, right), &.{ 1, 0 });
        const want = try quantize(&g, try g.astype(try g.add(dense, try g.mulScalar(delta, binding.scale)), dt), bits, group);
        try apply(&weights, s, &.{binding}, "language_model.model");
        for (keys, want) |key, expected| try ops.expectBitEqual(expected, weights.get(key).?);
    };
}

test "finetune overlay: a standalone model-local file carries its scale and strict base fingerprint" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    const hash = config.fingerprint("base config");
    try std.testing.expectEqual(null, try load(a, io, path, hash));
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var adapter = try @import("../train/adapter.zig").Adapter.init(a, s, &.{.{ .layer = 0, .projection = .q, .input = 128, .output = 4 }}, .{ .rank = 2, .alpha = 3 });
    defer adapter.deinit();
    var g = ops.Graph.init(a, s);
    defer g.deinit();
    _ = mlx.mlx_array_free(adapter.params[1]);
    adapter.params[1] = g.escape(try g.ones(&.{ 2, 4 }, .float32));
    const draft = try std.fs.path.join(a, &.{ path, "draft" });
    defer a.free(draft);
    try adapter.save(io, draft, hash);
    const bytes = try tmp.dir.readFileAlloc(io, "draft/adapters.safetensors", a, .limited(16384));
    defer a.free(bytes);
    try tmp.dir.writeFile(io, .{ .sub_path = filename, .data = bytes });
    var sidecar = (try load(a, io, path, hash)) orelse return error.MissingSidecar;
    defer sidecar.deinit();
    try std.testing.expectEqual(@as(f32, 1.5), sidecar.adapter.bindings[0].scale);
    try std.testing.expectEqual(@as(usize, 1), sidecar.adapter.bindings.len);
    try std.testing.expectError(error.LoraBaseMismatch, load(a, io, path, config.fingerprint("different base")));
    try tmp.dir.writeFile(io, .{ .sub_path = filename, .data = "broken" });
    try std.testing.expectError(error.InvalidLlmAdapter, load(a, io, path, hash));
}

test "finetune overlay: imported PEFT config resolves a local base or requires an explicit fingerprint" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = buf[0..try tmp.dir.realPath(io, &buf)];
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "base config" });
    const file = try std.fmt.allocPrintSentinel(a, "{s}/import.safetensors", .{path}, 0);
    defer a.free(file);
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = ops.Graph.init(a, s);
    defer g.deinit();
    const tensors = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(tensors);
    const meta = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta);
    try mlx.check(mlx.mlx_map_string_to_array_insert(tensors, "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight", try g.ones(&.{ 2, 4 }, .float32)));
    try mlx.check(mlx.mlx_map_string_to_array_insert(tensors, "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight", try g.ones(&.{ 8, 2 }, .float32)));
    try mlx.check(mlx.mlx_save_safetensors(file, tensors, meta));
    try std.Io.Dir.rename(tmp.dir, "import.safetensors", tmp.dir, filename, io);
    const json = try std.json.Stringify.valueAlloc(a, .{ .peft_type = "LORA", .r = 2, .lora_alpha = 3, .base_model_name_or_path = path }, .{});
    defer a.free(json);
    try tmp.dir.writeFile(io, .{ .sub_path = "adapter_config.json", .data = json });
    var sidecar = (try load(a, io, path, config.fingerprint("base config"))).?;
    defer sidecar.deinit();
    try std.testing.expectEqual(@as(f32, 1.5), sidecar.adapter.bindings[0].scale);
    try std.testing.expectEqualSlices(c_int, &.{ 4, 2 }, ops.shape(.{ .ctx = sidecar.adapter.bindings[0].a.ctx }));
    try std.testing.expectError(error.LoraBaseMismatch, load(a, io, path, config.fingerprint("different")));
    try tmp.dir.deleteFile(io, "config.json");
    try std.testing.expectError(error.LoraBaseFingerprintRequired, load(a, io, path, config.fingerprint("base config")));
}

test "finetune overlay: Flash attention preserves eight-bit packing and includes the query gate rows" {
    const a = std.testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = ops.Graph.init(a, s);
    defer g.deinit();
    var weights = Weights.init(a);
    defer weights.deinit();
    const expert_key = "language_model.model.layers.3.mlp.switch_mlp.gate_proj.weight";
    const expert = try g.ones(&.{ 2, 64, 128 }, .bfloat16);
    try weights.put(expert_key, g.escape(expert));
    const original_expert = weights.get(expert_key).?;
    inline for (.{ .q, .k, .v, .o }) |projection| {
        const rows = switch (@as(abi.LoraProjection, projection)) {
            .q => 256,
            .o => 128,
            else => 64,
        };
        const stem = "language_model.model.layers.3." ++ comptime projectionName(projection);
        const keys = [_][]const u8{ stem ++ ".weight", stem ++ ".scales", stem ++ ".biases" };
        const base = try quantize(&g, try g.zeros(&.{ rows, 128 }, .bfloat16), 8, 64);
        for (keys, base) |key, x| try weights.put(key, g.escape(x));
        const left = try g.ones(&.{ 128, 1 }, .bfloat16);
        const right = try g.ones(&.{ 1, rows }, .bfloat16);
        var binding: abi.LoraBinding = .{ .layer = 3, .projection = projection, .a = .{ .ctx = left.ctx }, .b = .{ .ctx = right.ctx }, .scale = 0.25 };
        if (projection == .q) {
            binding.b.ctx = (try g.ones(&.{ 1, rows / 2 }, .bfloat16)).ctx;
            try std.testing.expectError(error.InvalidLlmAdapter, apply(&weights, s, &.{binding}, "language_model.model"));
            binding.b.ctx = right.ctx;
        }
        try apply(&weights, s, &.{binding}, "language_model.model");
        try std.testing.expectEqualSlices(c_int, ops.shape(base[0]), ops.shape(weights.get(keys[0]).?));
        try std.testing.expectEqual(ops.Dtype.bfloat16, ops.dtype(weights.get(keys[1]).?));
        const merged = try g.call(mlx.mlx_dequantize, .{ weights.get(keys[0]).?, weights.get(keys[1]).?, weights.get(keys[2]).?, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(8), "affine", ops.Array{}, mlx.mlx_optional_dtype{ .value = .float32, .has_value = true } });
        const want = try g.full(&.{ rows, 128 }, try g.scalar(0.25, .float32), .float32);
        try ops.expectBitEqual(want, merged);
    }
    try std.testing.expectEqual(original_expert.ctx, weights.get(expert_key).?.ctx);
}
