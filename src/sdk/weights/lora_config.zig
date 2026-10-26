//! Pure adapter configuration, request selection and cache identities.

const std = @import("std");

pub const Config = struct {
    rank: u32,
    scale: f32,
    base_model: ?[]const u8 = null,
    base_config_sha256: ?[32]u8 = null,

    pub fn parse(root: std.json.Value) !Config {
        if (root != .object) return error.InvalidLlmAdapterConfig;
        const o = root.object;
        var cfg: Config = undefined;
        if (o.get("peft_type")) |peft| {
            if (peft != .string or !std.mem.eql(u8, peft.string, "LORA")) return error.InvalidLlmAdapterConfig;
            for ([_][]const u8{ "use_dora", "use_rslora", "lora_bias" }) |key| if (o.get(key)) |v| {
                if (v != .bool or v.bool) return error.InvalidLlmAdapterConfig;
            };
            if (o.get("bias")) |v| if (v != .string or !std.mem.eql(u8, v.string, "none")) return error.InvalidLlmAdapterConfig;
            for ([_][]const u8{ "rank_pattern", "alpha_pattern" }) |key| if (o.get(key)) |v| {
                if (v != .object or v.object.count() != 0) return error.InvalidLlmAdapterConfig;
            };
            if (o.get("modules_to_save")) |v| if (v != .null and (v != .array or v.array.items.len != 0)) return error.InvalidLlmAdapterConfig;
            const rank = try rankOf(o.get("r") orelse return error.InvalidLlmAdapterConfig);
            cfg = .{ .rank = rank, .scale = (try number(o.get("lora_alpha") orelse return error.InvalidLlmAdapterConfig)) / @as(f32, @floatFromInt(rank)) };
            cfg.base_model = try optionalString(o.get("base_model_name_or_path"));
        } else {
            if (o.get("fine_tune_type")) |kind| if (kind != .string or !std.mem.eql(u8, kind.string, "lora")) return error.InvalidLlmAdapterConfig;
            const params = o.get("lora_parameters") orelse return error.InvalidLlmAdapterConfig;
            if (params != .object) return error.InvalidLlmAdapterConfig;
            cfg = .{
                .rank = try rankOf(params.object.get("rank") orelse return error.InvalidLlmAdapterConfig),
                .scale = try number(params.object.get("scale") orelse return error.InvalidLlmAdapterConfig),
                .base_model = try optionalString(o.get("model")),
            };
        }
        if (o.get("base_config_sha256")) |v| {
            if (v != .string or v.string.len != 64) return error.InvalidLlmAdapterConfig;
            var digest: [32]u8 = undefined;
            _ = std.fmt.hexToBytes(&digest, v.string) catch return error.InvalidLlmAdapterConfig;
            cfg.base_config_sha256 = digest;
        }
        return cfg;
    }
};

pub fn errorMessage(err: anyerror) ?[]const u8 {
    return switch (err) {
        error.InvalidLoraRequest => "InvalidLoraRequest: lora must name a registered adapter; lora_scale requires lora and must be a finite number in [0, 2].",
        error.UnknownLora => "UnknownLora: no adapter_config.json exists for this adapter in ~/.mlx-serve/adapters.",
        error.LoraBaseMismatch => "LoraBaseMismatch: the adapter's base-model config fingerprint does not match this model.",
        error.LoraBaseFingerprintRequired => "LoraBaseFingerprintRequired: supply base_config_sha256 when the adapter's recorded local model path cannot be resolved.",
        error.LoraUnsupportedArchitecture => "LoraUnsupportedArchitecture: runtime LLM LoRA supports dense llama and qwen3 models on the MLX engine.",
        error.LoraUnsupportedQuantization => "LoraUnsupportedQuantization: runtime LLM LoRA requires an unrotated affine-quantized or floating-point base.",
        error.LoraUnsupportedEndpoint => "LoraUnsupportedEndpoint: lora and lora_scale are supported only on text generation endpoints.",
        error.InvalidLlmAdapterConfig => "InvalidLlmAdapterConfig: expected a supported mlx-lm or PEFT LoRA configuration with rank and scale.",
        error.InvalidLlmAdapter => "InvalidLlmAdapter: expected finite A/B tensor pairs matching the model's attention or MLP projections.",
        error.LoraRevisionChanged => "LoraRevisionChanged: the adapter weights changed after validation; retry the request.",
        error.LoraAdapterTooLarge => "LoraAdapterTooLarge: the adapter exceeds the registry's size limit.",
        error.LoraRegistryUnavailable => "LoraRegistryUnavailable: the adapter registry is unavailable.",
        else => null,
    };
}

pub const Prepared = struct {
    path: []const u8,
    key: [32]u8,
    weights_digest: [32]u8,
    scale: f32,
    rank: u32,
    bytes: u64,

    pub fn deinit(p: *Prepared, a: std.mem.Allocator) void {
        a.free(p.path);
    }

    pub fn clone(p: Prepared, a: std.mem.Allocator) !Prepared {
        var copy = p;
        copy.path = try a.dupe(u8, p.path);
        return copy;
    }
};

fn optionalString(v: ?std.json.Value) !?[]const u8 {
    const value = v orelse return null;
    if (value != .string or value.string.len == 0) return error.InvalidLlmAdapterConfig;
    return value.string;
}

fn rankOf(v: std.json.Value) !u32 {
    if (v != .integer or v.integer <= 0 or v.integer > std.math.maxInt(u32)) return error.InvalidLlmAdapterConfig;
    return @intCast(v.integer);
}

fn number(v: std.json.Value) !f32 {
    const n: f32 = switch (v) {
        .integer => @floatFromInt(v.integer),
        .float => @floatCast(v.float),
        else => return error.InvalidLlmAdapterConfig,
    };
    if (!std.math.isFinite(n)) return error.InvalidLlmAdapterConfig;
    return n;
}

pub fn validName(name: []const u8) bool {
    if (name.len == 0 or name.len > 128 or name[0] == '.') return false;
    for (name) |ch| if (!std.ascii.isAlphanumeric(ch) and ch != '-' and ch != '_' and ch != '.') return false;
    return true;
}

pub const Selection = struct {
    name: []const u8,
    scale: f32 = 1,

    pub fn parse(root: std.json.Value) !?Selection {
        if (root != .object) return error.InvalidLoraRequest;
        const strength = root.object.get("lora_scale");
        const v = root.object.get("lora") orelse {
            if (strength != null) return error.InvalidLoraRequest;
            return null;
        };
        if (v != .string or !validName(v.string)) return error.InvalidLoraRequest;
        const scale = if (strength) |s| number(s) catch return error.InvalidLoraRequest else 1;
        if (scale < 0 or scale > 2) return error.InvalidLoraRequest;
        return .{ .name = v.string, .scale = if (scale == 0) 0 else scale };
    }
};

pub fn fingerprint(bytes: []const u8) [32]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return digest;
}

pub fn validateBase(config: Config, base: [32]u8, resolved_model: ?[32]u8) !void {
    const expected = config.base_config_sha256 orelse resolved_model orelse return error.LoraBaseFingerprintRequired;
    if (!std.mem.eql(u8, &expected, &base)) return error.LoraBaseMismatch;
}

pub fn identity(name: []const u8, weights: [32]u8, config: [32]u8, scale: f32) [32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update("mlx-serve/llm-lora/v1\x00");
    hash.update(name);
    hash.update(&weights);
    hash.update(&config);
    var bits: [4]u8 = undefined;
    std.mem.writeInt(u32, &bits, @bitCast(if (scale == 0) @as(f32, 0) else scale), .little);
    hash.update(&bits);
    return hash.finalResult();
}

test "LLM LoRA: base fingerprints reject mismatches and missing provenance" {
    const digest = fingerprint("{\"model_type\":\"llama\"}");
    const other = fingerprint("{\"model_type\":\"qwen3\"}");
    try validateBase(.{ .rank = 8, .scale = 16, .base_config_sha256 = digest }, digest, null);
    try validateBase(.{ .rank = 8, .scale = 16, .base_model = "/models/llama" }, digest, digest);
    try std.testing.expectError(error.LoraBaseMismatch, validateBase(.{ .rank = 8, .scale = 16, .base_config_sha256 = other }, digest, digest));
    try std.testing.expectError(error.LoraBaseMismatch, validateBase(.{ .rank = 8, .scale = 16, .base_model = "/models/qwen" }, digest, other));
    try std.testing.expectError(error.LoraBaseFingerprintRequired, validateBase(.{ .rank = 8, .scale = 16 }, digest, null));
}

test "LLM LoRA: cache identities separate content, config, name and strength including scale zero" {
    const rev = fingerprint("weights");
    const cfg = fingerprint("config");
    const a = identity("one", rev, cfg, 1);
    try std.testing.expect(!std.mem.eql(u8, &a, &identity("two", rev, cfg, 1)));
    try std.testing.expect(!std.mem.eql(u8, &a, &identity("one", fingerprint("new weights"), cfg, 1)));
    try std.testing.expect(!std.mem.eql(u8, &a, &identity("one", rev, fingerprint("new config"), 1)));
    try std.testing.expect(!std.mem.eql(u8, &a, &identity("one", rev, cfg, 0.5)));
    try std.testing.expectEqualSlices(u8, &identity("one", rev, cfg, 0), &identity("one", rev, cfg, -0.0));
    const base: [32]u8 = @splat(0);
    try std.testing.expect(!std.mem.eql(u8, &base, &identity("one", rev, cfg, 0)));
}

test "LLM LoRA: native and PEFT config scales retain their different conventions" {
    const native = try std.json.parseFromSlice(std.json.Value, std.testing.allocator,
        \\{"fine_tune_type":"lora","lora_parameters":{"rank":8,"scale":16,"dropout":0.05}}
    , .{});
    defer native.deinit();
    const n = try Config.parse(native.value);
    try std.testing.expectEqual(@as(u32, 8), n.rank);
    try std.testing.expectEqual(@as(f32, 16), n.scale);
    const peft = try std.json.parseFromSlice(std.json.Value, std.testing.allocator,
        \\{"peft_type":"LORA","r":8,"lora_alpha":16,"bias":"none","use_rslora":false}
    , .{});
    defer peft.deinit();
    try std.testing.expectEqual(@as(f32, 2), (try Config.parse(peft.value)).scale);
}

test "LLM LoRA: unsupported adapter configs and unsafe names fail explicitly" {
    for ([_][]const u8{
        "[]",                                                                  "{}",                                                                                 "{\"fine_tune_type\":\"dora\"}",
        "{\"lora_parameters\":{\"rank\":0,\"scale\":1}}",                      "{\"lora_parameters\":{\"rank\":8,\"scale\":\"16\"}}",                                "{\"peft_type\":\"LORA\",\"r\":8,\"lora_alpha\":16,\"use_rslora\":true}",
        "{\"peft_type\":\"LORA\",\"r\":8,\"lora_alpha\":16,\"bias\":\"all\"}", "{\"peft_type\":\"LORA\",\"r\":8,\"lora_alpha\":16,\"rank_pattern\":{\"q_proj\":4}}",
    }) |json| {
        const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json, .{});
        defer parsed.deinit();
        try std.testing.expectError(error.InvalidLlmAdapterConfig, Config.parse(parsed.value));
    }
    for ([_][]const u8{ "", ".", "..", "../answer", "answer/a", "answer\\a", ".hidden", "a\x00b" }) |name| try std.testing.expect(!validName(name));
    try std.testing.expect(validName("answer-v2.1"));
}

test "LLM LoRA: request selection validates name and finite strength before dispatch" {
    for ([_][]const u8{ "{\"lora\":3}", "{\"lora\":\"../escape\"}", "{\"lora_scale\":1}", "{\"lora\":\"answers\",\"lora_scale\":-0.1}", "{\"lora\":\"answers\",\"lora_scale\":2.1}", "{\"lora\":\"answers\",\"lora_scale\":\"1\"}" }) |json| {
        const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json, .{});
        defer parsed.deinit();
        try std.testing.expectError(error.InvalidLoraRequest, Selection.parse(parsed.value));
    }
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, "{\"lora\":\"answers\",\"lora_scale\":0}", .{});
    defer parsed.deinit();
    const selection = (try Selection.parse(parsed.value)).?;
    try std.testing.expectEqualStrings("answers", selection.name);
    try std.testing.expectEqual(@as(f32, 0), selection.scale);
}
