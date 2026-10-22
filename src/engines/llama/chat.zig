//! Prompt encode/decode for the embedded llama.cpp engine.

const std = @import("std");
const arch_llama = if (@import("build_options").llama_engine) @import("bridge.zig") else @import("stub.zig");
const log = @import("../../host/log.zig");

const Message = @import("../../host/chat/types.zig").Message;
const ChatConfig = @import("../../host/chat/types.zig").ChatConfig;
const renderChatTemplate = @import("../../host/chat/render.zig").renderChatTemplate;

/// Render + encode chat messages for a llama.cpp-backed (GGUF) model.
///
/// Unlike ds4 (which has its own template renderer), we reuse mlx-serve's Jinja
/// engine via `renderChatTemplate` — `chat_config.chat_template` is populated
/// from the GGUF's embedded template at load time (see
/// `Scheduler.doLoadLlamaOnInferenceThread`). That path already handles the
/// tool-synthesis fallback for templates that don't model `tools` natively, so
/// tool calling works across the GGUF zoo. The rendered prompt is then tokenized
/// through libllama's own vocab with `add_special = false` (the template owns
/// any BOS) and `parse_special = true` (so `<|im_start|>` etc. become real
/// special tokens).
pub fn encodeChatViaLlama(
    allocator: std.mem.Allocator,
    engine: *arch_llama.LlamaEngine,
    chat_config: *const ChatConfig,
    messages: []const Message,
    tools_json: ?[]const u8,
    tool_choice_instruction: ?[]const u8,
    enable_thinking: bool,
    effort: ?[]const u8,
    /// Extend the trailing assistant message instead of answering after it.
    continue_final: bool,
) ![]u32 {
    const rendered = try renderChatTemplate(allocator, messages, chat_config, tools_json, tool_choice_instruction, enable_thinking, effort, continue_final);
    defer allocator.free(rendered);

    const i32_ids = try engine.tokenizeText(allocator, rendered, false);
    defer allocator.free(i32_ids);

    const u32_ids = try allocator.alloc(u32, i32_ids.len);
    for (i32_ids, 0..) |t, i| u32_ids[i] = @intCast(t);
    log.debug("  prompt (llama): {d} messages -> {d} tokens (tools={s})\n", .{
        messages.len,
        u32_ids.len,
        if (tools_json != null) "yes" else "no",
    });
    return u32_ids;
}

/// Detokenize a sequence of token IDs via the llama.cpp engine. Mirrors
/// `decodeViaDs4` so server handlers switch on the LoadedModel uniformly.
pub fn decodeViaLlama(
    allocator: std.mem.Allocator,
    engine: *arch_llama.LlamaEngine,
    ids: []const u32,
) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    for (ids) |id| {
        const piece = try engine.detokenizeOne(allocator, @intCast(id));
        defer allocator.free(piece);
        try out.appendSlice(allocator, piece);
    }
    return out.toOwnedSlice(allocator);
}
