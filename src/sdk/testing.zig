//! Conformance (docs/plugins.md): the checks a plugin's kinds run against the SDK's contracts, and the
//! fakes the host's own tests drive. Every check declares its lane:
//! - `cpu`: `zig build conformance` with no device; the lane fails if a Metal device was created.
//! - `gpu_small`: fixture shapes on an author's Mac.
//! - `window`: a plugin's own full-size checks, never in the suite.

const std = @import("std");
const builtin = @import("builtin");
const peek = @import("peek.zig");
const arch = @import("arch.zig");

pub const Lane = enum { cpu, gpu_small, window };

// ── The CPU lane's device check ──

extern fn _dyld_image_count() u32;
extern fn _dyld_get_image_name(image_index: u32) ?[*:0]const u8;

/// Whether this process created a Metal device: only that maps a GPU driver bundle (AGXMetal*).
pub fn deviceCreated() bool {
    if (builtin.os.tag != .macos) return false;
    for (0.._dyld_image_count()) |i| {
        const name = _dyld_get_image_name(@intCast(i)) orelse continue;
        if (std.mem.indexOf(u8, std.mem.span(name), "AGXMetal") != null) return true;
    }
    return false;
}

/// The CPU lane's last check: nothing before it created a device (any MLX array in the Metal build does).
pub fn expectNoDevice() error{DeviceCreatedInCpuLane}!void {
    if (deviceCreated()) return error.DeviceCreatedInCpuLane;
}

// ── Claims (cpu) ──

/// A fixture config and the claim it must get: the plugin's own config at its priority, near misses declined.
pub const ClaimCase = struct { config: []const u8, want: ?peek.Priority };

/// A weight group as a quant claims it at load: its `quantization` description (JSON) and dims, and the claim it must
/// get.
pub const GroupClaimCase = struct { quantization: []const u8, hidden: u64, inter: u64, want: ?peek.Priority };

pub fn expectGroupClaims(claims: *const fn (*const peek.GroupPeek, ?*peek.Diag) ?peek.Priority, cases: []const GroupClaimCase) !void {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    for (cases) |c| {
        const q = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), c.quantization, .{});
        const g: peek.GroupPeek = .{ .quantization = q, .hidden = c.hidden, .inter = c.inter, .n_experts = 0, .n_layers = 0, .layers = &.{} };
        var why: peek.Diag = .{};
        std.testing.expectEqual(c.want, claims(&g, &why)) catch |e| {
            std.debug.print("claims: {s} ({s})\n", .{ c.quantization, why.message() });
            return e;
        };
    }
}

pub fn expectClaims(claims: *const fn (*const peek.ConfigPeek) ?peek.Priority, cases: []const ClaimCase) !void {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    for (cases) |c| {
        const p = try peek.ConfigPeek.parse(arena.allocator(), "/fixture", c.config);
        std.testing.expectEqual(c.want, claims(&p)) catch |e| {
            std.debug.print("claims: {s}\n", .{c.config});
            return e;
        };
    }
}

// ── Fakes for the host's tests (cpu) ──

pub const FakeOptions = struct {
    caps: arch.Caps = .{ .owns_decode_state = true, .prefill_whole_prompt = true, .prefill_yields_last_logits = true },
    /// The model_type the fake claims.
    model_type: []const u8 = "fake_arch",
};

/// An arch for host tests: no MLX, caps per test.
pub fn FakeArch(comptime opts: FakeOptions) type {
    return struct {
        pub const name = "fake-arch";
        pub const caps = opts.caps;
        pub const Config = struct { settings_applied: u32 = 0 };

        pub fn claims(p: *const peek.ConfigPeek) ?peek.Priority {
            const t = p.modelType() orelse return null;
            return if (std.mem.eql(u8, t, opts.model_type)) .native else null;
        }
        pub fn parse(gpa: std.mem.Allocator, p: *const peek.ConfigPeek, diag: *peek.Diag) !*Config {
            if (p.int("refuse") != null) {
                diag.set("fake arch: refused by the fixture", .{});
                return error.FakeArchRefused;
            }
            const c = try gpa.create(Config);
            c.* = .{};
            return c;
        }
        pub fn freeConfig(gpa: std.mem.Allocator, c: *Config) void {
            gpa.destroy(c);
        }
        pub fn shell(_: *const Config) arch.Shell {
            return .{ .num_experts = 4, .num_layers = 2 };
        }
    };
}

const testing = std.testing;

test "sdk testing: the fake arch's table carries its name, caps and claim, and refuses by name through its diag" {
    const vt = comptime arch.Arch.of(FakeArch(.{}));
    try testing.expect(vt.caps.owns_decode_state);
    var diag: peek.Diag = .{};
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const p = try peek.ConfigPeek.parse(arena.allocator(), "/m", "{\"model_type\":\"fake_arch\"}");
    try testing.expectEqual(@as(?peek.Priority, .native), vt.claims(&p));
    const cfg = try vt.parse(testing.allocator, &p, &diag);
    defer vt.free_config(testing.allocator, cfg);
    try testing.expectEqual(@as(u32, 2), vt.shell(cfg).num_layers);

    const bare = comptime arch.Arch.of(FakeArch(.{ .caps = .{} }));
    try testing.expect(!bare.caps.owns_decode_state);
    try testing.expectError(error.FakeArchRefused, vt.parse(testing.allocator, &(try peek.ConfigPeek.parse(arena.allocator(), "/m", "{\"model_type\":\"fake_arch\",\"refuse\":1}")), &diag));
    try testing.expectEqualStrings("fake arch: refused by the fixture", diag.message());
}

test "sdk testing: claims fixtures run on any arch's claim" {
    try expectClaims(FakeArch(.{}).claims, &.{
        .{ .config = "{\"model_type\":\"fake_arch\"}", .want = .native },
        .{ .config = "{\"model_type\":\"deepseek_v4\"}", .want = null },
        .{ .config = "{\"architectures\":[\"X\"]}", .want = null },
    });
}

test "sdk testing: the CPU lane's device probe reads the loaded images and has created no device here" {
    try testing.expect(!deviceCreated());
    try expectNoDevice();
}

test "sdk testing: a claims fixture and a group fixture that disagree fail the check" {
    try testing.expectError(error.TestExpectedEqual, expectClaims(FakeArch(.{}).claims, &.{.{ .config = "{\"model_type\":\"fake_arch\"}", .want = .generic }}));
    const Never = struct {
        fn claims(_: *const peek.GroupPeek, why: ?*peek.Diag) ?peek.Priority {
            if (why) |d| d.set("never", .{});
            return null;
        }
    };
    try expectGroupClaims(Never.claims, &.{.{ .quantization = "{}", .hidden = 1, .inter = 1, .want = null }});
    try testing.expectError(error.TestExpectedEqual, expectGroupClaims(Never.claims, &.{.{ .quantization = "null", .hidden = 1, .inter = 1, .want = .native }}));
}
