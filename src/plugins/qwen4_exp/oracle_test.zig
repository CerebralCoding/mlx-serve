//! A qwen4_exp proof's expected values: the legacy forward's while it is present, else the proof's tape
//! (`fixtures/<name>.tape`, `engines/mlx/plugin/tape.zig`). `MLX_SERVE_PROOF_RECORD=<repo root>` records the legacy
//! values to the tape.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const tape_mod = @import("../../engines/mlx/plugin/tape.zig");

const Array = mlx.mlx_array;

pub const Oracle = struct {
    tape: tape_mod.Tape,

    pub fn open(comptime name: []const u8) !Oracle {
        if (tape_mod.recordRoot()) |root| return .{ .tape = try tape_mod.Tape.recorder(root, "src/plugins/qwen4_exp/fixtures/" ++ name ++ ".tape") };
        const bytes = @embedFile("fixtures/" ++ name ++ ".tape");
        return .{ .tape = if (bytes.len == 0) .{} else try tape_mod.Tape.player(bytes) };
    }

    /// `got` (the plugin's) against `want` (the legacy forward's, when present) and the tape; `tokens`: a greedy run,
    /// compared only on the recording's device.
    pub fn check(o: *Oracle, got: Array, want: ?Array, tokens: bool) !void {
        if (want) |w| {
            try ops.expectBitEqual(got, w);
            if (o.tape.recording()) return o.tape.put(w);
        }
        if (o.tape.playing()) try o.tape.expectNext(got, tokens);
    }

    /// `check` on host ids (a token run, accepted counts).
    pub fn checkIds(o: *Oracle, got: []const u32, want: ?[]const u32) !void {
        const g = idsArray(got);
        defer _ = mlx.mlx_array_free(g);
        const w: ?Array = if (want) |x| idsArray(x) else null;
        defer if (w) |x| {
            _ = mlx.mlx_array_free(x);
        };
        try o.check(g, w, true);
    }

    /// The playback read every value; a recording is written.
    pub fn close(o: *Oracle) !void {
        try o.tape.expectDone();
        try o.tape.finish();
    }
};

fn idsArray(ids: []const u32) Array {
    if (ids.len == 0) return mlx.mlx_array_new_data(&[_]u32{0}, &[_]c_int{0}, 1, .uint32);
    return mlx.mlx_array_new_data(ids.ptr, &[_]c_int{@intCast(ids.len)}, 1, .uint32);
}
