//! Prompt encode/decode for the embedded ds4 engine (its own template and token bytes).

const std = @import("std");
const arch_ds4 = if (@import("build_options").ds4_engine) @import("bridge.zig") else @import("stub.zig");
const ds4_ffi = if (@import("build_options").ds4_engine) @import("ffi.zig") else @import("ffi_stub.zig");
const log = @import("../../sdk/util/log.zig");
const testing = std.testing;

const ToolCall = @import("../../host/chat/types.zig").ToolCall;
const foldSystemMessages = @import("../../host/chat/types.zig").foldSystemMessages;
const Message = @import("../../host/chat/types.zig").Message;
const messagesHaveToolContent = @import("../../host/chat/render.zig").messagesHaveToolContent;
const synthesizeToolFallbackMessages = @import("../../host/chat/render.zig").synthesizeToolFallbackMessages;

/// Render + encode chat messages via the embedded ds4 engine's chat template.
/// ds4 owns its own tokenizer and chat-template renderer — we just map our
/// `Message` array to ds4's `ChatTurn` shape and append the assistant prefix.
///
/// Tool plumbing: the DSV4 chat template doesn't model the `tools` argument
/// natively, so when `tools_json` is non-null we synthesize a fallback —
/// inject the tool catalog into the system message (same machinery the Jinja
/// path uses via `synthesizeToolFallbackMessages`), and rewrite `role:"tool"`
/// turns as `<tool_response>…</tool_response>` user turns, and rewrite
/// assistant `tool_calls` as inline `<tool_call>…</tool_call>` text. Result:
/// the model sees the full tool catalog AND the live tool-result history,
/// which is what makes Agent mode + MCP tool calling work on ds4.
/// Pure prep step extracted so the tool-synthesis decision is unit-testable
/// without booting the ds4 engine. Returns the system content (or null) and
/// the post-synthesis turns ready for `Ds4Engine.encodeChatTranscript`.
/// The returned arena owns all string memory; caller must `deinit()` it.
pub const Ds4PromptPrep = struct {
    arena: std.heap.ArenaAllocator,
    system: ?[]const u8,
    turns: []arch_ds4.Ds4Engine.ChatTurn,

    pub fn deinit(self: *Ds4PromptPrep) void {
        self.arena.deinit();
    }
};

/// Pure helper: run the ds4 chat-template fallback synthesis (tool catalog
/// into system, tool-role rewrite, assistant tool_call inlining) and split
/// the result into (system, turns). Exists so the tool-plumbing decision is
/// covered by a unit test that doesn't need a live engine.
pub fn prepareDs4Prompt(
    parent_allocator: std.mem.Allocator,
    messages: []const Message,
    tools_json: ?[]const u8,
    tool_choice_instruction: ?[]const u8,
) !Ds4PromptPrep {
    var arena = std.heap.ArenaAllocator.init(parent_allocator);
    errdefer arena.deinit();
    const arena_alloc = arena.allocator();

    var folded = std.ArrayList(Message).empty;
    try folded.appendSlice(arena_alloc, messages);
    _ = try foldSystemMessages(arena_alloc, &folded, false);
    const messages_have_tool_content = messagesHaveToolContent(folded.items);
    var effective_messages: []const Message = folded.items;
    if (tools_json != null or messages_have_tool_content) {
        effective_messages = try synthesizeToolFallbackMessages(
            arena_alloc,
            effective_messages,
            tools_json,
            tool_choice_instruction,
            tools_json != null,
            messages_have_tool_content,
            messages_have_tool_content,
        );
    }

    var system_msg: ?[]const u8 = null;
    var turns_list = std.ArrayList(arch_ds4.Ds4Engine.ChatTurn).empty;
    for (effective_messages) |msg| {
        if (system_msg == null and std.mem.eql(u8, msg.role, "system")) {
            system_msg = msg.content;
            continue;
        }
        // After synthesis only system / user / assistant roles remain.
        try turns_list.append(arena_alloc, .{ .role = msg.role, .content = msg.content });
    }

    return .{
        .arena = arena,
        .system = system_msg,
        .turns = try turns_list.toOwnedSlice(arena_alloc),
    };
}

pub fn encodeChatViaDs4(
    allocator: std.mem.Allocator,
    engine: *arch_ds4.Ds4Engine,
    messages: []const Message,
    tools_json: ?[]const u8,
    tool_choice_instruction: ?[]const u8,
    enable_thinking: bool,
) ![]u32 {
    // ── Tool synthesis fallback. ds4's chat template doesn't reference
    //    `tools` or `role == 'tool'`, so without this rewrite:
    //      * tool definitions never reach the model (model can't call them)
    //      * tool-result messages render as empty (model loses prior context)
    //      * assistant tool_call history disappears (multi-turn loops break)
    //    Run via the pure `prepareDs4Prompt` helper so the same logic is
    //    exercised by the unit test at the bottom of this file.
    var prep = try prepareDs4Prompt(allocator, messages, tools_json, tool_choice_instruction);
    defer prep.deinit();

    const think_mode: ds4_ffi.ThinkMode = if (enable_thinking) .high else .none;
    const i32_ids = try engine.encodeChatTranscript(allocator, prep.system, prep.turns, think_mode);
    defer allocator.free(i32_ids);

    const u32_ids = try allocator.alloc(u32, i32_ids.len);
    for (i32_ids, 0..) |t, i| u32_ids[i] = @intCast(t);
    log.debug("  prompt (ds4): {d} messages -> {d} tokens (tools={s})\n", .{
        messages.len,
        u32_ids.len,
        if (tools_json != null) "yes" else "no",
    });
    return u32_ids;
}

/// Detokenize a sequence of token IDs via the ds4 engine. Mirrors
/// `Tokenizer.decode` so server handlers can switch on the LoadedModel without
/// touching the call sites for each path.
pub fn decodeViaDs4(
    allocator: std.mem.Allocator,
    engine: *arch_ds4.Ds4Engine,
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

// ── tests ──

test "prepareDs4Prompt: tools_json injects the catalog into the system message" {
    const allocator = testing.allocator;
    const messages = [_]Message{
        .{ .role = "user", .content = "list /tmp" },
    };
    const tools_json = "[{\"type\":\"function\",\"function\":{\"name\":\"shell\",\"description\":\"Run a command\"}}]";

    var prep = try prepareDs4Prompt(allocator, &messages, tools_json, null);
    defer prep.deinit();

    try testing.expect(prep.system != null);
    // The synthesized system message MUST contain the tool name so the model
    // knows the tool exists. If `encodeChatViaDs4` stops threading
    // `tools_json` to `synthesizeToolFallbackMessages`, this fails.
    try testing.expect(std.mem.indexOf(u8, prep.system.?, "shell") != null);
    // User turn passes through unchanged.
    try testing.expectEqual(@as(usize, 1), prep.turns.len);
    try testing.expectEqualStrings("user", prep.turns[0].role);
    try testing.expectEqualStrings("list /tmp", prep.turns[0].content);
}

test "prepareDs4Prompt: tools_json merges into existing system message" {
    const allocator = testing.allocator;
    const messages = [_]Message{
        .{ .role = "system", .content = "You are helpful." },
        .{ .role = "user", .content = "hi" },
    };
    const tools_json = "[{\"type\":\"function\",\"function\":{\"name\":\"webSearch\"}}]";

    var prep = try prepareDs4Prompt(allocator, &messages, tools_json, null);
    defer prep.deinit();

    try testing.expect(prep.system != null);
    // Both the original system content AND the tool catalog must be present
    // in the merged system message — losing either side would silently
    // strip the agent prompt or the tool list.
    try testing.expect(std.mem.indexOf(u8, prep.system.?, "You are helpful.") != null);
    try testing.expect(std.mem.indexOf(u8, prep.system.?, "webSearch") != null);
}

test "prepareDs4Prompt: role:tool gets rewritten to user with <tool_response>" {
    const allocator = testing.allocator;
    const messages = [_]Message{
        .{ .role = "user", .content = "run ls" },
        .{ .role = "tool", .content = "file1.txt\nfile2.txt" },
    };

    var prep = try prepareDs4Prompt(allocator, &messages, null, null);
    defer prep.deinit();

    try testing.expectEqual(@as(usize, 2), prep.turns.len);
    try testing.expectEqualStrings("user", prep.turns[0].role);
    // Tool result MUST be wrapped as a user turn with <tool_response>; ds4's
    // chat template doesn't model role:"tool" natively so leaving it as
    // role:"tool" makes the model unaware of any prior tool output.
    try testing.expectEqualStrings("user", prep.turns[1].role);
    try testing.expect(std.mem.indexOf(u8, prep.turns[1].content, "<tool_response>") != null);
    try testing.expect(std.mem.indexOf(u8, prep.turns[1].content, "file1.txt") != null);
}

test "prepareDs4Prompt: assistant tool_calls get inlined as <tool_call> text" {
    const allocator = testing.allocator;
    const tool_calls = [_]ToolCall{
        .{ .id = "c1", .name = "shell", .arguments = "{\"command\":\"ls /tmp\"}" },
    };
    const messages = [_]Message{
        .{ .role = "user", .content = "list tmp" },
        .{ .role = "assistant", .content = "", .tool_calls = &tool_calls },
        .{ .role = "tool", .content = "ok", .tool_call_id = "c1" },
    };

    var prep = try prepareDs4Prompt(allocator, &messages, null, null);
    defer prep.deinit();

    // Assistant turn must carry the inlined <tool_call> so multi-turn
    // history stays coherent on ds4. Without the rewrite the model sees
    // an empty assistant turn followed by a tool result with no context.
    try testing.expectEqual(@as(usize, 3), prep.turns.len);
    try testing.expectEqualStrings("assistant", prep.turns[1].role);
    try testing.expect(std.mem.indexOf(u8, prep.turns[1].content, "<tool_call>") != null);
    try testing.expect(std.mem.indexOf(u8, prep.turns[1].content, "\"name\": \"shell\"") != null);
    try testing.expect(std.mem.indexOf(u8, prep.turns[1].content, "ls /tmp") != null);
}

test "prepareDs4Prompt: no tools, no tool content → passthrough" {
    const allocator = testing.allocator;
    const messages = [_]Message{
        .{ .role = "system", .content = "be terse" },
        .{ .role = "user", .content = "hi" },
        .{ .role = "assistant", .content = "hello" },
    };

    var prep = try prepareDs4Prompt(allocator, &messages, null, null);
    defer prep.deinit();

    // Nothing to synthesize — system should be the original verbatim, turns
    // should be the non-system messages in order.
    try testing.expect(prep.system != null);
    try testing.expectEqualStrings("be terse", prep.system.?);
    try testing.expectEqual(@as(usize, 2), prep.turns.len);
    try testing.expectEqualStrings("user", prep.turns[0].role);
    try testing.expectEqualStrings("hi", prep.turns[0].content);
    try testing.expectEqualStrings("assistant", prep.turns[1].role);
    try testing.expectEqualStrings("hello", prep.turns[1].content);
}
