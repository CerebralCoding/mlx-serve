//! What a media plugin's `step` needs from the host while its backend still speaks the host's job types: the
//! table's `Progress` as an `sse.Progress`, a request body's top-level fields, and a job error as a `Verdict`.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const mlx = @import("../../mlx/ffi.zig");
const sse = @import("sse.zig");

const Verdict = abi.Verdict;

/// `p` as the progress the backends take: `boundary` answers both the per-step poll and the cancel probe.
pub fn progressOf(p: *const abi.Progress) sse.Progress {
    return .{ .ctx = @constCast(p), .cb = report, .cancelled_cb = boundary };
}

fn report(ctx: *anyopaque, stage: []const u8, step: u32, total: u32) void {
    const p: *const abi.Progress = @ptrCast(@alignCast(ctx));
    p.report(p.ctx, .of(stage), step, total);
}

fn boundary(ctx: *anyopaque) bool {
    const p: *const abi.Progress = @ptrCast(@alignCast(ctx));
    return p.boundary(p.ctx);
}

/// A request's JSON object. `parse` refuses anything but an object.
pub const Body = struct {
    parsed: std.json.Parsed(std.json.Value),

    pub fn parse(a: std.mem.Allocator, json: []const u8) ?Body {
        const p = std.json.parseFromSlice(std.json.Value, a, json, .{}) catch return null;
        if (p.value != .object) {
            p.deinit();
            return null;
        }
        return .{ .parsed = p };
    }

    pub fn deinit(b: *Body) void {
        b.parsed.deinit();
    }

    pub fn get(b: *const Body, key: []const u8) ?std.json.Value {
        return b.parsed.value.object.get(key);
    }

    pub fn str(b: *const Body, key: []const u8) ?[]const u8 {
        const v = b.get(key) orelse return null;
        return if (v == .string) v.string else null;
    }

    /// True only for the literal `true`.
    pub fn flag(b: *const Body, key: []const u8) bool {
        const v = b.get(key) orelse return false;
        return v == .bool and v.bool;
    }

    pub fn float(b: *const Body, key: []const u8) ?f64 {
        const v = b.get(key) orelse return null;
        return switch (v) {
            .integer => |i| @floatFromInt(i),
            .float => |f| f,
            else => null,
        };
    }

    /// A non-negative integer field; a negative, fractional or non-number value reads as absent.
    pub fn uint(b: *const Body, key: []const u8) ?u64 {
        const v = b.get(key) orelse return null;
        return if (v == .integer and v.integer >= 0) @intCast(v.integer) else null;
    }
};

/// A failed job: `busy` when it ran out of memory (the machine's capacity, as chat's 503), else `failed`.
pub fn failure(err: anyerror, comptime what: []const u8) Verdict {
    const oom = err == error.OutOfMemory or mlx.gpuContextLost() or
        if (mlx.peekErrorName()) |n| std.mem.eql(u8, n, "OutOfMemory") else false;
    return Verdict.refuse(if (oom) .busy else .failed, what);
}

pub const client_left = Verdict.refuse(.failed, "the client left");

// ── tests ──

const testing = std.testing;

test "plugin job: a body reads typed fields and refuses a non-object" {
    try testing.expect(Body.parse(testing.allocator, "[1]") == null);
    try testing.expect(Body.parse(testing.allocator, "{") == null);
    var b = Body.parse(testing.allocator, "{\"s\":\"x\",\"f\":true,\"n\":-1,\"u\":7,\"r\":0.5}").?;
    defer b.deinit();
    try testing.expectEqualStrings("x", b.str("s").?);
    try testing.expect(b.flag("f") and !b.flag("s"));
    try testing.expect(b.uint("n") == null and b.uint("u").? == 7);
    try testing.expectEqual(@as(f64, 0.5), b.float("r").?);
    try testing.expect(failure(error.Boom, "x").code == .failed and failure(error.OutOfMemory, "x").code == .busy);
}
