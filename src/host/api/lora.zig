//! Request validation for registered language-model adapters.

const std = @import("std");
const adapters = @import("../registry/adapters.zig");
const config = @import("../../sdk/weights/lora_config.zig");
const LoadedModel = @import("../registry/registry.zig").LoadedModel;
pub const Prepared = config.Prepared;

pub fn validateSurface(a: std.mem.Allocator, path: []const u8, body: []const u8, content_type: []const u8) !void {
    const multipart = @import("../http/multipart.zig");
    if (multipart.boundaryFromContentType(content_type)) |boundary| {
        var parts = try multipart.Iterator.init(body, boundary);
        while (parts.next()) |part| {
            if (std.mem.eql(u8, part.name, "lora") or std.mem.eql(u8, part.name, "lora_scale")) return error.LoraUnsupportedEndpoint;
        }
        return;
    }
    for ([_][]const u8{ "/v1/chat/completions", "/v1/completions", "/v1/messages", "/v1/responses", "/api/chat" }) |allowed| if (std.mem.eql(u8, path, allowed)) return;
    const parsed = std.json.parseFromSlice(std.json.Value, a, body, .{}) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => return,
    };
    defer parsed.deinit();
    if (parsed.value == .object and std.mem.eql(u8, path, "/api/generate")) {
        if (parsed.value.object.get("prompt")) |prompt| if (prompt == .string and prompt.string.len > 0) return;
    }
    if (parsed.value == .object and (parsed.value.object.contains("lora") or parsed.value.object.contains("lora_scale"))) return error.LoraUnsupportedEndpoint;
}

pub fn prepare(a: std.mem.Allocator, io: std.Io, root: std.json.ObjectMap, lm: *const LoadedModel) !?Prepared {
    const selection = (try config.Selection.parse(.{ .object = root })) orelse return null;
    const cfg = lm.config orelse return error.LoraUnsupportedArchitecture;
    if (lm.engine != null or !adapters.supported(cfg.model_type) or cfg.isMoe()) return error.LoraUnsupportedArchitecture;
    if (cfg.quant_mode != .affine or cfg.hadamard_block != 0) return error.LoraUnsupportedQuantization;
    const binding = cfg.plugin orelse return error.LoraUnsupportedArchitecture;
    if (binding.outside) return error.LoraUnsupportedArchitecture;
    const path = try adapters.rootPath(a);
    defer a.free(path);
    return try adapters.resolve(a, io, path, selection, .{
        .path = lm.path,
        .arch = cfg.model_type,
        .fingerprint = binding.source_sha256 orelse return error.LoraBaseFingerprintRequired,
        .geometry = .{ .layers = cfg.num_hidden_layers, .hidden = cfg.hidden_size, .intermediate = cfg.intermediate_size, .heads = cfg.num_attention_heads, .kv_heads = cfg.num_key_value_heads, .head_dim = cfg.head_dim },
    });
}

test "LLM LoRA: every text API rejects invalid selection before generating or opening SSE" {
    const a = std.testing.allocator;
    const Conn = @import("../http/server.zig").Conn;
    const capture = @import("../media_jobs/plugin/test_conn.zig").capture;
    const handlers = .{
        @import("chat_completions.zig").handleChatCompletions,
        @import("completions.zig").handleCompletions,
        @import("anthropic.zig").handleAnthropicMessages,
        @import("responses/handler.zig").handleResponses,
    };
    var cfg: @import("../../sdk/config/model.zig").ModelConfig = .{};
    cfg.model_type = "qwen3_5";
    var lm: LoadedModel = undefined;
    lm.tokenizer = @ptrFromInt(0x1000);
    lm.chat_config = @ptrFromInt(0x1000);
    lm.config = &cfg;
    lm.engine = null;
    inline for (handlers) |handler| {
        const Run = struct {
            fn run(conn: *Conn, body: []const u8, entry: *LoadedModel) !void {
                try handler(std.testing.allocator, conn, body, entry);
            }
        };
        for ([_]bool{ false, true }) |streaming| {
            const body = try std.fmt.allocPrint(a, "{{\"lora\":\"../bad\",\"stream\":{}}}", .{streaming});
            defer a.free(body);
            const out = try capture(a, Run.run, .{ body, &lm });
            defer a.free(out);
            try std.testing.expect(std.mem.startsWith(u8, out, "HTTP/1.1 400"));
            try std.testing.expect(std.mem.indexOf(u8, out, "InvalidLoraRequest") != null);
            try std.testing.expect(std.mem.indexOf(u8, out, "text/event-stream") == null);
        }
    }
}

test "LLM LoRA: unsupported base quantization is refused before adapter IO" {
    const a = std.testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, a, "{\"lora\":\"answer\"}", .{});
    defer parsed.deinit();
    var cfg: @import("../../sdk/config/model.zig").ModelConfig = .{ .model_type = "llama", .quant_mode = .nvfp4 };
    var lm: LoadedModel = undefined;
    lm.config = &cfg;
    lm.engine = null;
    try std.testing.expectError(error.LoraUnsupportedQuantization, prepare(a, std.testing.io, parsed.value.object, &lm));
}

test "LLM LoRA: unsupported JSON and multipart surfaces cannot silently discard adapter fields" {
    const a = std.testing.allocator;
    try validateSurface(a, "/v1/chat/completions", "{\"lora\":\"answer\"}", "application/json");
    try validateSurface(a, "/v1/images/generations", "{\"lora_paths\":[\"media.safetensors\"]}", "application/json");
    try std.testing.expectError(error.LoraUnsupportedEndpoint, validateSurface(a, "/v1/images/generations", "{\"lora\":\"answer\"}", "application/json"));
    try std.testing.expectError(error.LoraUnsupportedEndpoint, validateSurface(a, "/v1/embeddings", "{\"lora_scale\":0.5}", "application/json"));
    const body = "--b\r\nContent-Disposition: form-data; name=\"lora\"\r\n\r\nanswer\r\n--b--\r\n";
    try std.testing.expectError(error.LoraUnsupportedEndpoint, validateSurface(a, "/v1/audio/transcriptions", body, "multipart/form-data; boundary=b"));
    for ([_][]const u8{ "{\"lora\":\"answer\"}", "{\"prompt\":\"\",\"lora_scale\":0.5}" }) |load_only|
        try std.testing.expectError(error.LoraUnsupportedEndpoint, validateSurface(a, "/api/generate", load_only, "application/json"));
    try validateSurface(a, "/api/generate", "{\"prompt\":\"hello\",\"lora\":\"answer\"}", "application/json");
}
