//! engines/mlx/env_transformer, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns envFlagCached, diagEnvValueOn, diagEnvOn, diagEnvOnCached.

const std = @import("std");
const testing = @import("kv/cache.zig").testing;

/// PRESENCE semantics: `FOO=0` reads as ON. Never substitute this for
/// `diagEnvOnCached` — a harness exporting `FOO=0` would arm the switch.
pub fn envFlagCached(cache: *?bool, name: [*:0]const u8) bool {
    if (cache.*) |v| return v;
    const v = std.c.getenv(name) != null;
    cache.* = v;
    return v;
}
/// The diagnostic-switch decision, split from the `getenv` so it is testable.
pub fn diagEnvValueOn(raw: ?[*:0]const u8) bool {
    const v = raw orelse return false;
    return v[0] != '0';
}
/// Diagnostic env switch: set and not `0`. `FOO=0` exported by a harness must
/// never arm a sync profiler (the qwen4 MTP verify once measured 70 ms).
pub fn diagEnvOn(name: [*:0]const u8) bool {
    return diagEnvValueOn(std.c.getenv(name));
}
/// `diagEnvOn` for a hot path: same semantics (absent or `0` = off), asked once.
pub fn diagEnvOnCached(cache: *?bool, name: [*:0]const u8) bool {
    if (cache.*) |v| return v;
    const v = diagEnvOn(name);
    cache.* = v;
    return v;
}

// ── tests ──

test "diagEnvValueOn: absent or 0 is off" {
    try testing.expectEqual(false, diagEnvValueOn(null));
    try testing.expectEqual(false, diagEnvValueOn("0"));
    try testing.expectEqual(true, diagEnvValueOn("1"));
    try testing.expectEqual(true, diagEnvValueOn("on"));
}

test "diagEnvOnCached answers once and latches" {
    var cache: ?bool = null;
    try testing.expectEqual(false, diagEnvOnCached(&cache, "MLX_SERVE_NO_SUCH_DIAG_SWITCH_PROBE"));
    try testing.expectEqual(@as(?bool, false), cache);
    cache = true;
    try testing.expectEqual(true, diagEnvOnCached(&cache, "MLX_SERVE_NO_SUCH_DIAG_SWITCH_PROBE"));
}
