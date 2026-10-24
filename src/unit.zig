//! One plugin as its own compile unit (`-Dunits`): exports the table of the plugin `unit_options.plugin` names, which
//! the host takes by symbol (`sdk.externEntries`). The decls below are what the absorbed modules read from their host
//! root (`mlx_host`), as `main.zig` declares them.

const std = @import("std");
const registry = @import("plugins/registry.zig");
const sdk = @import("sdk/plugin.zig");

comptime {
    const name = @import("unit_options").plugin;
    var found = false;
    for (registry.plugins_names, 0..) |n, i| if (std.mem.eql(u8, n, name)) {
        sdk.exportAll(&.{registry.plugins[i]});
        found = true;
    };
    if (!found) @compileError("no registered plugin " ++ name);
}

/// Marks this compilation as a plugin unit: `Shared` state resolves to the host's copy (sdk/util/shared.zig).
pub const mlx_serve_unit = true;

pub const env = @import("env.zig");
pub const mlx = @import("mlx/ffi.zig");
pub const io_util = @import("sdk/util/io_util.zig");
pub const shared = @import("sdk/util/shared.zig");
pub const mtp_acceptance = @import("engines/mlx/spec/mtp_acceptance.zig");
pub const exl3 = @import("sdk/quant/exl3/format.zig");
pub const dsa_nax = @import("plugins/glm5_next/dsa_nax.zig");
pub const log = @import("host/log.zig");
pub const plugin_profile: bool = @import("build_options").plugin_profile;
