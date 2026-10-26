//! CPU-only fine-tuning requests and the host/engine job contract.
const std = @import("std");
const verify = @import("../../sdk/train/verify.zig");
const lora = @import("../../sdk/weights/lora_config.zig");

pub const Request = struct {
    model: []const u8,
    name: []const u8,
    prompts: []const []const u8 = &.{},
    topic: ?[]const u8 = null,

    pub fn parse(a: std.mem.Allocator, body: []const u8) !std.json.Parsed(Request) {
        const parsed = std.json.parseFromSlice(Request, a, body, .{ .allocate = .alloc_always }) catch |err| switch (err) {
            error.OutOfMemory => return err,
            else => return error.InvalidFineTuneRequest,
        };
        errdefer parsed.deinit();
        const r = parsed.value;
        if (r.model.len == 0 or r.model.len > 512 or !lora.validName(r.name) or r.prompts.len > 16 or (r.prompts.len == 0 and r.topic == null)) return error.InvalidFineTuneRequest;
        for (r.prompts) |p| if (!validText(p, 2048)) return error.InvalidFineTuneRequest;
        if (r.topic) |topic| if (!validText(topic, 2048)) return error.InvalidFineTuneRequest;
        return parsed;
    }
};

fn validText(text: []const u8, max: usize) bool {
    return text.len <= max and std.mem.trim(u8, text, " \r\n\t").len > 0 and std.unicode.utf8ValidateSlice(text);
}

pub const Stage = enum { synth, train, verify };
pub const Status = enum {
    queued,
    running,
    completed,
    cancelled,
    failed,
    pub fn terminal(self: Status) bool {
        return self == .completed or self == .cancelled or self == .failed;
    }
};
pub const Update = struct { stage: Stage, step: u32, total: u32, loss: ?f32 = null, preview: []const u8 = "" };
pub const Observer = struct {
    ctx: *anyopaque,
    emit_fn: *const fn (*anyopaque, Update) void,
    cancelled_fn: *const fn (*anyopaque) bool,
    pub fn emit(self: Observer, update: Update) void {
        self.emit_fn(self.ctx, update);
    }
    pub fn cancelled(self: Observer) bool {
        return self.cancelled_fn(self.ctx);
    }
};

pub const Plan = struct { reserve_bytes: u64 = 0, available_bytes: u64 = 0 };
pub const Result = struct {
    metrics: verify.Metrics = .{},
    training_examples: u32 = 0,
    validation_examples: u32 = 0,
    retain_examples: u32 = 0,
    updates: u32 = 0,
    before: []const u8 = "",
    after: []const u8 = "",

    pub fn deinit(self: *Result, a: std.mem.Allocator) void {
        if (self.before.len > 0) a.free(self.before);
        if (self.after.len > 0) a.free(self.after);
        self.* = .{};
    }
};
pub const Run = struct { a: std.mem.Allocator, io: std.Io, request: *const Request, path: []const u8, observer: Observer, result: *Result };

test "finetune: requests require a specific model, safe adapter name and bounded plain text" {
    const a = std.testing.allocator;
    for ([_][]const u8{
        "{}",                                                         "{\"model\":\"\",\"name\":\"answer\",\"topic\":\"gardening\"}",
        "{\"model\":\"m\",\"name\":\"../escape\",\"topic\":\"x\"}",   "{\"model\":\"m\",\"name\":\"answer\"}",
        "{\"model\":\"m\",\"name\":\"answer\",\"prompts\":[\" \" ]}", "{\"model\":\"m\",\"name\":\"answer\",\"topic\":\"x\",\"rank\":8}",
    }) |body| try std.testing.expectError(error.InvalidFineTuneRequest, Request.parse(a, body));
    const parsed = try Request.parse(a, "{\"model\":\"org/model\",\"name\":\"gardening\",\"prompts\":[\"How do I prune a rose?\"]}");
    defer parsed.deinit();
    try std.testing.expectEqualStrings("How do I prune a rose?", parsed.value.prompts[0]);
}
