//! Unique prompt candidates and the bundled self-distillation anchor.
const std = @import("std");
pub const Example = @import("../../sdk/train/packing.zig").Example;

pub const Prompts = struct {
    a: std.mem.Allocator,
    items: std.ArrayList([]const u8) = .empty,

    pub fn deinit(self: *Prompts) void {
        for (self.items.items) |p| self.a.free(p);
        self.items.deinit(self.a);
    }
    pub fn add(self: *Prompts, raw: []const u8) !void {
        var p = std.mem.trim(u8, raw, " \t\r\n\"");
        var digits: usize = 0;
        while (digits < p.len and std.ascii.isDigit(p[digits])) : (digits += 1) {}
        if (digits > 0 and digits + 1 < p.len and (p[digits] == '.' or p[digits] == ')') and std.ascii.isWhitespace(p[digits + 1])) {
            p = p[digits + 1 ..];
        } else if (p.len > 1 and (p[0] == '-' or p[0] == '*') and std.ascii.isWhitespace(p[1])) p = p[1..];
        p = std.mem.trim(u8, p, " \t\r\n\"");
        if (p.len < 8 or p.len > 2048 or !std.unicode.utf8ValidateSlice(p) or self.items.items.len >= 64) return;
        for (self.items.items) |old| if (std.ascii.eqlIgnoreCase(old, p)) return;
        try self.items.ensureUnusedCapacity(self.a, 1);
        self.items.appendAssumeCapacity(try self.a.dupe(u8, p));
    }
    pub fn addLines(self: *Prompts, text: []const u8) !void {
        var lines = std.mem.splitScalar(u8, text, '\n');
        while (lines.next()) |line| try self.add(line);
    }
};

pub fn deinitExamples(a: std.mem.Allocator, examples: *std.ArrayList(Example)) void {
    for (examples.items) |e| e.deinit(a);
    examples.deinit(a);
}

pub const retain = @embedFile("retain.txt");

test "finetune: paraphrases are unique before splitting and retain prompts stay separate" {
    var p: Prompts = .{ .a = std.testing.allocator };
    defer p.deinit();
    try p.addLines("1. How do I prune roses?\n- How do I prune roses?\n HOW DO I PRUNE ROSES? \nHow should I water roses?");
    try std.testing.expectEqual(@as(usize, 2), p.items.items.len);
    try std.testing.expectEqualStrings("How do I prune roses?", p.items.items[0]);
    var count: usize = 0;
    var lines = std.mem.tokenizeScalar(u8, retain, '\n');
    while (lines.next()) |_| count += 1;
    try std.testing.expectEqual(@as(usize, 64), count);
}

test "finetune: list parsing preserves meaningful digits at either end of a prompt" {
    var p: Prompts = .{ .a = std.testing.allocator };
    defer p.deinit();
    try p.addLines("1. Count backwards from 10\n2+2 is equal to what?\n- Explain base 16");
    try std.testing.expectEqualStrings("Count backwards from 10", p.items.items[0]);
    try std.testing.expectEqualStrings("2+2 is equal to what?", p.items.items[1]);
    try std.testing.expectEqualStrings("Explain base 16", p.items.items[2]);
}
