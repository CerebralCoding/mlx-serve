//! Environment switches read by a run-time name, for the plugins that have not moved to `src/env.zig`'s
//! accessors yet; each forwards to `env.byName`.

const env = @import("../../env.zig");
const testing = @import("kv/cache.zig").testing;

/// PRESENCE semantics: `FOO=0` reads as ON. Never substitute this for
/// `diagEnvOnCached` — a harness exporting `FOO=0` would arm the switch.
pub fn envFlagCached(cache: *?bool, name: [*:0]const u8) bool {
    if (cache.*) |v| return v;
    const v = env.byName(name) != null;
    cache.* = v;
    return v;
}
/// The diagnostic-switch decision, split from the read so it is testable.
pub const diagEnvValueOn = env.onValue;
/// Diagnostic env switch: set and not `0`. `FOO=0` exported by a harness must
/// never arm a sync profiler (the qwen4 MTP verify once measured 70 ms).
pub fn diagEnvOn(name: [*:0]const u8) bool {
    return diagEnvValueOn(env.byName(name));
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
    try testing.expectEqual(false, diagEnvOnCached(&cache, "NO_SUCH_DIAG_SWITCH_PROBE"));
    try testing.expectEqual(@as(?bool, false), cache);
    cache = true;
    try testing.expectEqual(true, diagEnvOnCached(&cache, "NO_SUCH_DIAG_SWITCH_PROBE"));
}
