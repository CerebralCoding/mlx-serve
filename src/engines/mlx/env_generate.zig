//! Environment readers for the generator's diagnostic and kill-switch variables.
const std = @import("std");


/// Read an unsigned integer from an environment variable, falling back to
/// `default` when unset, empty, or unparseable. Uses libc getenv to stay
/// allocator-free at call sites.
pub fn readEnvUsize(name: [:0]const u8, default: usize) usize {
    const raw = std.c.getenv(name.ptr);
    if (raw == null) return default;
    const slice = std.mem.sliceTo(raw.?, 0);
    if (slice.len == 0) return default;
    return std.fmt.parseInt(usize, slice, 10) catch default;
}

/// Read a finite, non-negative float from an environment variable, else `default`.
pub fn readEnvFloat(name: [:0]const u8, default: f32) f32 {
    const raw = std.c.getenv(name.ptr);
    if (raw == null) return default;
    const slice = std.mem.sliceTo(raw.?, 0);
    if (slice.len == 0) return default;
    const v = std.fmt.parseFloat(f32, slice) catch return default;
    return if (std.math.isFinite(v) and v >= 0) v else default;
}

/// Truthy if the env var is exactly "1". Anything else (unset, "0", "true",
/// "yes") is false — keep matching surface tight to avoid surprises.
pub fn readEnvBool(name: [:0]const u8) bool {
    const raw = std.c.getenv(name.ptr);
    if (raw == null) return false;
    const slice = std.mem.sliceTo(raw.?, 0);
    return std.mem.eql(u8, slice, "1");
}
