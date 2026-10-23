//! The host's low-memory media policy: large encoders stage per request on iOS, with `MLXSERVE_LOWMEM=1`, or at
//! 17 GiB of RAM or less. Read once into `abi.media.Machine.low_mem`; plugins never read it themselves.
const std = @import("std");
const status = @import("../status.zig");

pub fn fromInputs(is_ios: bool, env: ?[]const u8, total_ram_bytes: u64) bool {
    if (is_ios) return true;
    if (env) |e| {
        if (std.mem.eql(u8, e, "1")) return true;
        if (std.mem.eql(u8, e, "0")) return false;
    }
    return total_ram_bytes > 0 and total_ram_bytes <= 17 * 1024 * 1024 * 1024;
}

pub fn default() bool {
    const env: ?[]const u8 = if (std.c.getenv("MLXSERVE_LOWMEM")) |v| std.mem.sliceTo(v, 0) else null;
    return fromInputs(@import("build_options").ios, env, status.getTotalMemBytes());
}
