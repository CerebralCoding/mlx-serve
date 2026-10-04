//! The plugin SDK (docs/plugins.md): the one module a plugin imports. The small SDK: what a module-owned arch needs
//! from the host, as optional declarations of the kinds below. A seam only one plugin consumes stays in that plugin
//! until a second consumer exists. The registry (src/plugins.zig) builds each kind's table once, at compile time,
//! and the host resolves a model's tables once at load: a hook runs per request, step or round, never per layer.

const std = @import("std");

/// The SDK's version. A plugin built against another major is refused at compile time; a newer minor on either
/// side is compatible (newer hooks are optional).
pub const api: Version = .{ .major = 1, .minor = 0 };

/// The MLX this binary links (lib/mlx-src 64ea011cb: v0.32.3). One MLX per process: a plugin tested on another is
/// refused at compile time, so an MLX bump is one change that moves this pin and every plugin's.
pub const mlx_pin = "v0.32.3";

pub const mlx = @import("mlx");
pub const log = @import("log");
pub const io_util = @import("io_util");

const plugin = @import("sdk/plugin.zig");
pub const Version = plugin.Version;
pub const Plugin = plugin.Plugin;
pub const Provides = plugin.Provides;
pub const Host = plugin.Host;
pub const NegotiationError = plugin.NegotiationError;
pub const negotiate = plugin.negotiate;
/// What this host checks every plugin against.
pub const host: Host = .{ .api = api, .mlx = mlx_pin };

const peek = @import("sdk/peek.zig");
pub const Priority = peek.Priority;
pub const Diag = peek.Diag;
pub const ConfigPeek = peek.ConfigPeek;
pub const GroupPeek = peek.GroupPeek;
pub const LayerPeek = peek.LayerPeek;
pub const Segment = peek.Segment;

const arch = @import("sdk/arch.zig");
pub const Arch = arch.Arch;
pub const Caps = arch.Caps;
pub const Shell = arch.Shell;

const kinds = @import("sdk/kinds.zig");
pub const Source = kinds.Source;
pub const Engine = kinds.Engine;

/// The comptime interface checks the kinds run on a plugin's namespaces; a plugin's own contracts reuse them.
pub const check = @import("sdk/check.zig");

/// Conformance (docs/plugins.md): every check declares its lane.
pub const testing = @import("sdk/testing.zig");

test {
    std.testing.refAllDecls(@This());
    _ = plugin;
    _ = peek;
    _ = arch;
    _ = kinds;
    _ = testing;
}
