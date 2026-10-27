//! Adapter layout, scale conventions and strict base-model provenance.

const std = @import("std");

pub fn supported(arch: []const u8) bool {
    for ([_][]const u8{ "llama", "qwen3", "gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_text", "qwen3_5", "qwen3_5_text", "qwen3_5_moe" }) |name| if (std.mem.eql(u8, arch, name)) return true;
    return false;
}

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
        error.LoraBaseMismatch => "LoraBaseMismatch: the adapter's base-model config fingerprint does not match this model.",
        error.LoraBaseFingerprintRequired => "LoraBaseFingerprintRequired: supply base_config_sha256 when the adapter's recorded local model path cannot be resolved.",
        error.LoraUnsupportedArchitecture => "LoraUnsupportedArchitecture: load-time LoRA supports dense Llama, Qwen3, Gemma 4 and Qwen3.5-family models, plus full-attention projections in single-Mac Qwen3.8-Flash-Next MLX loads.",
        error.LoraUnsupportedTarget => "LoraUnsupportedTarget: Qwen3.8-Flash-Next adapters may target only self_attn.q_proj, k_proj, v_proj and o_proj on full-attention trunk layers.",
        error.LoraUnsupportedQuantization => "LoraUnsupportedQuantization: load-time LoRA requires an unrotated affine-quantized or floating-point base.",
        error.InvalidLlmAdapterConfig => "InvalidLlmAdapterConfig: expected a supported mlx-lm or PEFT LoRA configuration with rank and scale.",
        error.InvalidLlmAdapter => "InvalidLlmAdapter: expected finite A/B tensor pairs matching the model's attention or MLP projections.",
        error.LoraRevisionChanged => "LoraRevisionChanged: the adapter changed while loading; reload the model.",
        error.LoraAdapterTooLarge => "LoraAdapterTooLarge: the adapter exceeds the size limit.",
        else => null,
    };
}

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

pub fn fingerprint(bytes: []const u8) [32]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return digest;
}

pub fn validateBase(config: Config, base: [32]u8, resolved_model: ?[32]u8) !void {
    const expected = config.base_config_sha256 orelse resolved_model orelse return error.LoraBaseFingerprintRequired;
    if (!std.mem.eql(u8, &expected, &base)) return error.LoraBaseMismatch;
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
