//! Per-request adapter ownership and revision checks on the inference thread.

const std = @import("std");
const config = @import("../../sdk/weights/lora_config.zig");
const weights = @import("../../sdk/weights/lora.zig");

pub const State = struct {
    allocator: std.mem.Allocator,
    prepared: config.Prepared,
    adapter: ?weights.LlmAdapter = null,

    pub fn init(a: std.mem.Allocator, prepared: config.Prepared) !State {
        return .{ .allocator = a, .prepared = try prepared.clone(a) };
    }

    pub fn deinit(state: *State) void {
        if (state.adapter) |*adapter| adapter.deinit();
        state.prepared.deinit(state.allocator);
    }

    pub fn loadReserve(state: *const State) u64 {
        // Source arrays, oriented copies, and the revision check's file buffer can overlap.
        return if (state.adapter == null) state.prepared.bytes *| 3 else 0;
    }

    fn verifyRevision(state: *const State, io: std.Io) !void {
        const stat = std.Io.Dir.cwd().statFile(io, state.prepared.path, .{}) catch return error.LoraRevisionChanged;
        if (stat.kind != .file or stat.size > 1024 * 1024 * 1024) return error.LoraRevisionChanged;
        const data = std.Io.Dir.cwd().readFileAlloc(io, state.prepared.path, state.allocator, .limited(1024 * 1024 * 1024)) catch |err| switch (err) {
            error.OutOfMemory => return err,
            else => return error.LoraRevisionChanged,
        };
        defer state.allocator.free(data);
        if (!std.mem.eql(u8, &config.fingerprint(data), &state.prepared.weights_digest)) return error.LoraRevisionChanged;
    }

    pub fn load(state: *State, io: std.Io) !void {
        if (state.adapter != null) return;
        try state.verifyRevision(io);
        var adapter = try weights.LlmAdapter.load(state.allocator, state.prepared.path, state.prepared.scale, state.prepared.rank);
        errdefer adapter.deinit();
        try state.verifyRevision(io);
        if (adapter.bytes != state.prepared.bytes) return error.InvalidLlmAdapter;
        state.adapter = adapter;
    }
};

test "LLM LoRA: request state owns its path and rejects a changed weight revision" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "adapters.safetensors", .data = "changed" });
    var buf: [512]u8 = undefined;
    const dir = buf[0..try tmp.dir.realPath(io, &buf)];
    const path = try std.fs.path.join(a, &.{ dir, "adapters.safetensors" });
    defer a.free(path);
    const prepared: config.Prepared = .{ .path = path, .key = @splat(1), .weights_digest = config.fingerprint("original"), .scale = 1, .rank = 2, .bytes = 64 };
    var state = try State.init(a, prepared);
    defer state.deinit();
    try std.testing.expect(state.prepared.path.ptr != prepared.path.ptr);
    try std.testing.expectEqualStrings(path, state.prepared.path);
    try std.testing.expectEqual(@as(u64, 192), state.loadReserve());
    try std.testing.expectError(error.LoraRevisionChanged, state.load(io));
    try std.testing.expect(state.adapter == null);
}
