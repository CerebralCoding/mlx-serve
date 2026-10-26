//! Adapter discovery and strict binding to a base-model configuration.

const std = @import("std");
const config = @import("../../sdk/weights/lora_config.zig");
const metadata = @import("../../sdk/weights/lora_metadata.zig");

pub const Model = struct {
    path: []const u8,
    arch: []const u8,
    geometry: metadata.Geometry,
    fingerprint: ?[32]u8 = null,
};

pub fn rootPath(a: std.mem.Allocator) ![]const u8 {
    const home = @import("../../env.zig").get(.HOME) orelse return error.LoraRegistryUnavailable;
    return std.fs.path.join(a, &.{ std.mem.span(home), ".mlx-serve", "adapters" });
}

pub fn supported(arch: []const u8) bool {
    return std.mem.eql(u8, arch, "llama") or std.mem.eql(u8, arch, "qwen3");
}

fn readRegular(a: std.mem.Allocator, io: std.Io, path: []const u8, limit: usize) ![]u8 {
    const st = try std.Io.Dir.cwd().statFile(io, path, .{});
    if (st.kind != .file) return error.InvalidLlmAdapter;
    if (st.size > limit) return error.LoraAdapterTooLarge;
    return std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(limit));
}

fn modelFingerprint(a: std.mem.Allocator, io: std.Io, model_path: []const u8) ![32]u8 {
    const path = try std.fs.path.join(a, &.{ model_path, "config.json" });
    defer a.free(path);
    const body = try readRegular(a, io, path, 4 * 1024 * 1024);
    defer a.free(body);
    return config.fingerprint(body);
}

fn validateBinding(a: std.mem.Allocator, io: std.Io, cfg: config.Config, base_hash: [32]u8) !void {
    var resolved_base: ?[32]u8 = null;
    if (cfg.base_config_sha256 == null) if (cfg.base_model) |base| {
        if (std.fs.path.isAbsolute(base)) resolved_base = modelFingerprint(a, io, base) catch |err| switch (err) {
            error.OutOfMemory => return err,
            else => null,
        };
    };
    try config.validateBase(cfg, base_hash, resolved_base);
}

pub fn listJson(a: std.mem.Allocator, io: std.Io, root: []const u8, model_path: []const u8, arch: []const u8) ![]u8 {
    if (!supported(arch)) return a.dupe(u8, "[]");
    const base_hash = modelFingerprint(a, io, model_path) catch return a.dupe(u8, "[]");
    var dir = std.Io.Dir.cwd().openDir(io, root, .{ .iterate = true }) catch return a.dupe(u8, "[]");
    defer dir.close(io);
    var names: std.ArrayList([]const u8) = .empty;
    defer {
        for (names.items) |name| a.free(name);
        names.deinit(a);
    }
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (!config.validName(entry.name)) continue;
        const path = try std.fs.path.join(a, &.{ root, entry.name, "adapter_config.json" });
        defer a.free(path);
        const body = readRegular(a, io, path, 1024 * 1024) catch |err| switch (err) {
            error.OutOfMemory => return err,
            else => continue,
        };
        defer a.free(body);
        const parsed = std.json.parseFromSlice(std.json.Value, a, body, .{}) catch |err| switch (err) {
            error.OutOfMemory => return err,
            else => continue,
        };
        defer parsed.deinit();
        const cfg = config.Config.parse(parsed.value) catch continue;
        validateBinding(a, io, cfg, base_hash) catch |err| switch (err) {
            error.OutOfMemory => return err,
            else => continue,
        };
        const weights_path = try std.fs.path.join(a, &.{ root, entry.name, "adapters.safetensors" });
        defer a.free(weights_path);
        const stat = std.Io.Dir.cwd().statFile(io, weights_path, .{}) catch continue;
        if (stat.kind != .file or stat.size < 8 or stat.size > 1024 * 1024 * 1024) continue;
        try names.ensureUnusedCapacity(a, 1);
        names.appendAssumeCapacity(try a.dupe(u8, entry.name));
    }
    const Less = struct {
        fn lt(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    };
    std.mem.sort([]const u8, names.items, {}, Less.lt);
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.append(a, '[');
    for (names.items, 0..) |name, i| {
        if (i > 0) try out.append(a, ',');
        try out.print(a, "\"{s}\"", .{name});
    }
    try out.append(a, ']');
    return out.toOwnedSlice(a);
}

pub fn resolve(a: std.mem.Allocator, io: std.Io, root: []const u8, selection: config.Selection, model: Model) !config.Prepared {
    if (!config.validName(selection.name) or !std.math.isFinite(selection.scale) or selection.scale < 0 or selection.scale > 2) return error.InvalidLoraRequest;
    if (!supported(model.arch)) return error.LoraUnsupportedArchitecture;
    const config_path = try std.fs.path.join(a, &.{ root, selection.name, "adapter_config.json" });
    defer a.free(config_path);
    const body = readRegular(a, io, config_path, 1024 * 1024) catch |err| switch (err) {
        error.FileNotFound => return error.UnknownLora,
        else => return err,
    };
    defer a.free(body);
    const parsed = std.json.parseFromSlice(std.json.Value, a, body, .{}) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => return error.InvalidLlmAdapterConfig,
    };
    defer parsed.deinit();
    const cfg = try config.Config.parse(parsed.value);
    const base_hash = model.fingerprint orelse try modelFingerprint(a, io, model.path);
    try validateBinding(a, io, cfg, base_hash);
    const path = try std.fs.path.join(a, &.{ root, selection.name, "adapters.safetensors" });
    errdefer a.free(path);
    const tensors = readRegular(a, io, path, 1024 * 1024 * 1024) catch |err| switch (err) {
        error.OutOfMemory, error.LoraAdapterTooLarge => return err,
        else => return error.InvalidLlmAdapter,
    };
    defer a.free(tensors);
    if (tensors.len < 8) return error.InvalidLlmAdapter;
    const header_len = std.mem.readInt(u64, tensors[0..8], .little);
    if (header_len > tensors.len - 8) return error.InvalidLlmAdapter;
    const bytes = try metadata.inspect(a, tensors[8..][0..header_len], tensors.len - 8 - header_len, model.geometry, cfg.rank);
    const weights_digest = config.fingerprint(tensors);
    const scale = cfg.scale * selection.scale;
    if (!std.math.isFinite(scale)) return error.InvalidLlmAdapterConfig;
    return .{ .path = path, .key = config.identity(selection.name, weights_digest, config.fingerprint(body), selection.scale), .weights_digest = weights_digest, .scale = scale, .rank = cfg.rank, .bytes = bytes };
}

test "LLM LoRA: registry validates local provenance, explicit fingerprints and tensor geometry" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "base");
    try tmp.dir.createDirPath(io, "adapters/answer");
    const base_json = "{\"model_type\":\"llama\"}";
    try tmp.dir.writeFile(io, .{ .sub_path = "base/config.json", .data = base_json });
    const header =
        \\{"model.layers.0.self_attn.q_proj.lora_a":{"dtype":"F32","shape":[4,2],"data_offsets":[0,32]},"model.layers.0.self_attn.q_proj.lora_b":{"dtype":"F32","shape":[2,4],"data_offsets":[32,64]}}
    ;
    var tensor_file: [8 + header.len + 64]u8 = @splat(0);
    std.mem.writeInt(u64, tensor_file[0..8], header.len, .little);
    @memcpy(tensor_file[8..][0..header.len], header);
    try tmp.dir.writeFile(io, .{ .sub_path = "adapters/answer/adapters.safetensors", .data = &tensor_file });
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(io, &buf)];
    const base = try std.fs.path.join(a, &.{ root, "base" });
    defer a.free(base);
    const adapters = try std.fs.path.join(a, &.{ root, "adapters" });
    defer a.free(adapters);
    const native_config = try std.fmt.allocPrint(a, "{{\"model\":\"{s}\",\"lora_parameters\":{{\"rank\":2,\"scale\":16}}}}", .{base});
    defer a.free(native_config);
    try tmp.dir.writeFile(io, .{ .sub_path = "adapters/answer/adapter_config.json", .data = native_config });
    const model: Model = .{ .path = base, .arch = "llama", .geometry = .{ .layers = 2, .hidden = 4, .intermediate = 8, .heads = 2, .kv_heads = 1, .head_dim = 2 } };
    var selected = try resolve(a, io, adapters, .{ .name = "answer", .scale = 0.5 }, model);
    defer selected.deinit(a);
    try std.testing.expectEqual(@as(f32, 8), selected.scale);
    try std.testing.expectEqual(@as(u64, 64), selected.bytes);
    const listed = try listJson(a, io, adapters, model.path, model.arch);
    defer a.free(listed);
    try std.testing.expectEqualStrings("[\"answer\"]", listed);
    try std.testing.expectError(error.UnknownLora, resolve(a, io, adapters, .{ .name = "missing" }, model));
    var unsupported = model;
    unsupported.arch = "qwen3_moe";
    try std.testing.expectError(error.LoraUnsupportedArchitecture, resolve(a, io, adapters, .{ .name = "answer" }, unsupported));
    const bound = try std.fmt.allocPrint(a, "{{\"base_config_sha256\":\"{s}\",\"lora_parameters\":{{\"rank\":2,\"scale\":16}}}}", .{std.fmt.bytesToHex(config.fingerprint(base_json), .lower)});
    defer a.free(bound);
    try tmp.dir.writeFile(io, .{ .sub_path = "adapters/answer/adapter_config.json", .data = bound });
    var explicit = try resolve(a, io, adapters, .{ .name = "answer" }, model);
    defer explicit.deinit(a);
    try tmp.dir.writeFile(io, .{ .sub_path = "base/config.json", .data = "{\"model_type\":\"qwen3\"}" });
    try std.testing.expectError(error.LoraBaseMismatch, resolve(a, io, adapters, .{ .name = "answer", .scale = 0 }, model));
    const mismatched = try listJson(a, io, adapters, model.path, model.arch);
    defer a.free(mismatched);
    try std.testing.expectEqualStrings("[]", mismatched);
    var loaded = model;
    loaded.fingerprint = config.fingerprint(base_json);
    var still_bound = try resolve(a, io, adapters, .{ .name = "answer" }, loaded);
    defer still_bound.deinit(a);
    try tmp.dir.writeFile(io, .{ .sub_path = "adapters/answer/adapter_config.json", .data = "{\"model\":\"remote/model\",\"lora_parameters\":{\"rank\":2,\"scale\":16}}" });
    try std.testing.expectError(error.LoraBaseFingerprintRequired, resolve(a, io, adapters, .{ .name = "answer" }, model));
}
