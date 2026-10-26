//! Completion masks over the same chat-template tokens used for inference.
const std = @import("std");
const Tokenizer = @import("../../sdk/tokenizer/tokenizer.zig").Tokenizer;
const types = @import("types.zig");
const packing = @import("../../sdk/train/packing.zig");

pub fn example(a: std.mem.Allocator, tok: *const Tokenizer, messages: []const types.Message, config: *const types.ChatConfig, completion: []const u8, max_rows: usize) !packing.Example {
    if (std.mem.trim(u8, completion, " \n\t\r").len == 0) return error.EmptyTrainingCompletion;
    if (messages.len == 0 or std.mem.eql(u8, messages[messages.len - 1].role, "assistant")) return error.InvalidTrainingConversation;
    for (messages) |message| if (message.images != null or message.videos != null or message.audio != null or message.tool_calls != null) return error.TrainingRequiresPlainText;
    const format = @import("template_load.zig").formatChat;
    const prompt = try format(a, tok, messages, config, null, null, false, null, false);
    defer a.free(prompt);
    const full = try a.alloc(types.Message, messages.len + 1);
    defer a.free(full);
    @memcpy(full[0..messages.len], messages);
    full[messages.len] = .{ .role = "assistant", .content = completion };
    const completed = try format(a, tok, full, config, null, null, false, null, true);
    defer a.free(completed);
    // Only a final BPE token may straddle the prompt/completion boundary; do not train that mixed token.
    var same: usize = 0;
    while (same < @min(prompt.len, completed.len) and prompt[same] == completed[same]) : (same += 1) {}
    if (prompt.len == 0 or same + 1 < prompt.len or completed.len <= prompt.len) return error.TrainingTemplateBoundary;
    const eos = if (config.eos_token) |token| tok.specialTokenId(token) orelse tok.eos_id else tok.eos_id;
    const end = eos orelse return error.TrainingEndTokenRequired;
    const tokens = try a.alloc(u32, completed.len + 1);
    defer a.free(tokens);
    @memcpy(tokens[0..completed.len], completed);
    tokens[completed.len] = end;
    return packing.pack(a, tokens, prompt.len, max_rows);
}

test "train core: chat training reuses rendered prompt tokens and masks completion plus EOS" {
    const a = std.testing.allocator;
    var tok = Tokenizer.initEmptyForTests(a, .wordpiece);
    defer tok.deinit();
    for ([_][]const u8{ "user", "assistant", "hello", "yes", "[UNK]", "[END]" }, 0..) |word, id| {
        const owned = try a.dupe(u8, word);
        try tok.vocab.put(owned, @intCast(id));
        try tok.id_to_token.put(@intCast(id), owned);
    }
    tok.unk_id = 4;
    try tok.special_tokens.put(try a.dupe(u8, "[END]"), 5);
    const config: types.ChatConfig = .{ .chat_template = "{% for m in messages %}{{ m.role }} {{ m.content }} {% endfor %}{% if add_generation_prompt %}assistant {% endif %}", .bos_token = null, .eos_token = "[END]", .add_bos_token = false, .allocator = a };
    const messages = [_]types.Message{.{ .role = "user", .content = "hello" }};
    const e = try example(a, &tok, &messages, &config, "yes", 512);
    defer e.deinit(a);
    const prompt = try @import("template_load.zig").formatChat(a, &tok, &messages, &config, null, null, false, null, false);
    defer a.free(prompt);
    try std.testing.expectEqualSlices(u32, prompt, e.inputs[0..prompt.len]);
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 1, 1 }, e.mask);
    try std.testing.expectEqualSlices(u32, &.{ 2, 1, 3, 5 }, e.targets);
    try std.testing.expectError(error.EmptyTrainingCompletion, example(a, &tok, &messages, &config, "", 512));
}
