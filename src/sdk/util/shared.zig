//! Process-wide state under `-Dunits`. A plugin unit (`src/unit.zig`) compiles its own copy of whatever code it
//! reaches, globals included, so state the host sets or reads (CLI choices, memory accounting, the cluster, the MLX
//! error latch, the log sink) would split into one copy per unit. `Shared` keeps the storage in the host and hands a
//! unit the host's address, by an exported C function per name. Outside `-Dunits` it is a plain global.
//! `scripts/unit_globals.py` fails on a duplicated global that is neither shared nor listed as safe to duplicate.

/// True when compiled into a plugin unit: its root declares `mlx_serve_unit`.
pub const in_unit = @hasDecl(@import("root"), "mlx_serve_unit");

pub fn Shared(comptime T: type, comptime name: []const u8, comptime init: T) type {
    return struct {
        var storage: T = init;
        var cached: ?*T = null;
        const sym = "mlx_serve_shared_" ++ name;

        fn addr() callconv(.c) *anyopaque {
            return &storage;
        }

        comptime {
            if (!in_unit) @export(&addr, .{ .name = sym });
        }

        pub inline fn get() *T {
            if (!in_unit) return &storage;
            if (cached) |p| return p;
            const f = @extern(*const fn () callconv(.c) *anyopaque, .{ .name = sym });
            const p: *T = @ptrCast(@alignCast(f()));
            cached = p;
            return p;
        }
    };
}

test "Shared: outside a unit the storage is the module's own and starts at init" {
    const S = Shared(u32, "test_shared_counter", 7);
    try @import("std").testing.expectEqual(@as(u32, 7), S.get().*);
    S.get().* += 1;
    try @import("std").testing.expectEqual(@as(u32, 8), S.get().*);
}
