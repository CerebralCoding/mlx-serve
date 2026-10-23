//! `mlx-serve plugin new|check|install`: a plugin folder from the kit's example, the conformance suite on a built
//! dylib, and the install into `~/.mlx-serve/plugins/` with its receipt (the hash the loader checks).

const std = @import("std");
const env = @import("../../env.zig");
const builtin = @import("builtin");
const dylib_abi = @import("../../abi/dylib.zig");
const sdk_dylib = @import("../../sdk/dylib.zig");
const conformance = @import("../../sdk/testing/conformance.zig");
const loader = @import("loader.zig");
const log = @import("../log.zig");

const Allocator = std.mem.Allocator;

pub const usage =
    \\usage: mlx-serve plugin new <name> [--media|--quant] [--kit <dir>] [--dir <parent>]
    \\       mlx-serve plugin check <file.dylib | plugin folder>
    \\       mlx-serve plugin install <file.dylib> [--yes]
    \\
    \\  new      A plugin folder from the kit's example (model, or --media / --quant). The kit is the
    \\           release's mlx-serve-plugin-kit tarball, extracted to --kit, $MLX_SERVE_KIT or ~/.mlx-serve/kit.
    \\  check    Loads the dylib (a folder: its zig-out/lib/*.dylib) and runs the conformance suite on its fixture.
    \\  install  Copies it into ~/.mlx-serve/plugins/ with a receipt of its hash. A downloaded (quarantined) file
    \\           installs only with --yes. Plugins load at startup once plugins.json has "enabled": true.
    \\
;

/// The MLX this binary runs: a plugin built against another is refused.
fn hostMlx() []const u8 {
    return @import("mlx_stamp").bytes;
}

pub fn cmdPlugin(gpa: Allocator, io: std.Io, args: []const []const u8) !void {
    if (args.len < 2) return fail("{s}", .{usage});
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const cmd = args[0];
    if (std.mem.eql(u8, cmd, "new")) {
        var kind: Kind = .model;
        var kit: ?[]const u8 = null;
        var parent: []const u8 = ".";
        var i: usize = 2;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "--media")) kind = .media else if (std.mem.eql(u8, args[i], "--quant")) kind = .quant else if (i + 1 < args.len and std.mem.eql(u8, args[i], "--kit")) {
                i += 1;
                kit = args[i];
            } else if (i + 1 < args.len and std.mem.eql(u8, args[i], "--dir")) {
                i += 1;
                parent = args[i];
            } else return fail("plugin new: unknown argument '{s}'\n{s}", .{ args[i], usage });
        }
        const kit_dir = kit orelse if (env.get(.MLX_SERVE_KIT)) |k| std.mem.span(k) else try loader.homePath(a, "kit");
        const dest = new(io, a, args[1], kind, kit_dir, parent) catch |e| return fail("plugin new: {s}", .{newError(e)});
        try out(io, "created {s}: `zig build` there writes zig-out/lib/lib{s}.dylib; `mlx-serve plugin check {s}` tests it\n", .{ dest, args[1], dest });
    } else if (std.mem.eql(u8, cmd, "check")) {
        const plugin = openForCheck(io, a, args[1]) catch |e| return fail("plugin check: {s}: {t}", .{ args[1], e });
        const p = switch (plugin) {
            .ok => |p| p,
            .refused => |why| return fail("plugin check: {s}: {s}", .{ args[1], why }),
        };
        checkPlugin(io, a, p) catch return fail("plugin check: {s} fails the conformance suite (above)", .{p.name()});
        try out(io, "{s} ({t}) passes the conformance suite\n", .{ p.name(), p.manifest.kind });
    } else if (std.mem.eql(u8, cmd, "install")) {
        var yes = false;
        for (args[2..]) |arg| {
            if (std.mem.eql(u8, arg, "--yes")) yes = true else return fail("plugin install: unknown argument '{s}'\n{s}", .{ arg, usage });
        }
        const dir = try loader.homePath(a, "plugins");
        const got = install(io, a, args[1], dir, yes) catch |e| return fail("plugin install: {s}: {t}", .{ args[1], e });
        switch (got) {
            .ok => |p| try out(io, "installed {s} ({t}) as {s}, sha256 {s}; it loads at the next start when plugins.json has \"enabled\": true\n", .{ p.name(), p.manifest.kind, p.path, p.sha256 }),
            .refused => |why| return fail("plugin install: {s}: {s}", .{ args[1], why }),
        }
    } else return fail("{s}", .{usage});
}

fn fail(comptime fmt: []const u8, args: anytype) error{PluginCommand} {
    log.err(fmt ++ "\n", args);
    return error.PluginCommand;
}

fn out(io: std.Io, comptime fmt: []const u8, args: anytype) !void {
    var buf: [1024]u8 = undefined;
    var w = std.Io.File.stdout().writer(io, &buf);
    try w.interface.print(fmt, args);
    try w.interface.flush();
}

// ── new ──

pub const Kind = enum { model, media, quant };

fn exampleOf(kind: Kind) []const u8 {
    return switch (kind) {
        .model => "_example",
        .media => "_example_media",
        .quant => "_example_quant",
    };
}

fn newError(e: anyerror) []const u8 {
    return switch (e) {
        error.BadName => "the name must start with a letter and hold only a-z, 0-9 and _ (at most 64 bytes)",
        error.KitMissing => "no plugin kit there: extract the release's mlx-serve-plugin-kit tarball and pass --kit <dir>",
        error.PathAlreadyExists => "the folder already exists",
        else => @errorName(e),
    };
}

/// Copies the kit's example for `kind` into `<parent>/<name>`, renamed, with a build.zig.zon depending on the kit at
/// `kit`.
/// Returns the new folder's path.
pub fn new(io: std.Io, a: Allocator, name: []const u8, kind: Kind, kit: []const u8, parent: []const u8) ![]const u8 {
    if (!dylib_abi.validName(name) or !std.ascii.isLower(name[0])) return error.BadName;
    const cwd = std.Io.Dir.cwd();
    var kit_buf: [std.fs.max_path_bytes]u8 = undefined;
    var kit_dir = cwd.openDir(io, kit, .{}) catch return error.KitMissing;
    defer kit_dir.close(io);
    const kit_abs = kit_buf[0..try kit_dir.realPath(io, &kit_buf)];
    const ex = exampleOf(kind);
    var src = kit_dir.openDir(io, try std.fmt.allocPrint(a, "examples/{s}", .{ex}), .{}) catch return error.KitMissing;
    defer src.close(io);
    const dest = try std.fs.path.join(a, &.{ parent, name });
    if (cwd.access(io, dest, .{})) |_| return error.PathAlreadyExists else |_| {}
    var dst = try cwd.createDirPathOpen(io, dest, .{});
    defer dst.close(io);
    for ([_][]const u8{ "build.zig", "plugin.zig", "config.zig" }) |f| {
        const text = try src.readFileAlloc(io, f, a, .limited(1 << 20));
        try dst.writeFile(io, .{ .sub_path = f, .data = try std.mem.replaceOwned(u8, a, text, ex, name) });
    }
    var dest_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dest_abs = dest_buf[0..try dst.realPath(io, &dest_buf)];
    // Zig takes a path dependency relative to the package.
    try dst.writeFile(io, .{ .sub_path = "build.zig.zon", .data = try zon(a, io, name, try std.fs.path.relativeAllocPosix(a, "/", dest_abs, kit_abs)) });
    return dest;
}

/// A plugin package's manifest on the kit at `kit_rel` (relative to the package); the fingerprint is Zig's (a random id
/// over the name's CRC).
fn zon(a: Allocator, io: std.Io, name: []const u8, kit_rel: []const u8) ![]const u8 {
    var rnd: [4]u8 = undefined;
    io.random(&rnd);
    const id = std.math.clamp(std.mem.readInt(u32, &rnd, .little), 1, 0xfffffffe);
    const fingerprint = (@as(u64, std.hash.Crc32.hash(name)) << 32) | id;
    return std.fmt.allocPrint(a,
        \\.{{
        \\    .name = .{s},
        \\    .version = "0.0.0",
        \\    .fingerprint = 0x{x:0>16},
        \\    .minimum_zig_version = "0.17.0",
        \\    .dependencies = .{{ .mlx_serve_kit = .{{ .path = "{s}" }} }},
        \\    .paths = .{{ "build.zig", "build.zig.zon", "plugin.zig", "config.zig" }},
        \\}}
        \\
    , .{ name, fingerprint, kit_rel });
}

// ── check ──

/// `path` (a dylib, or a plugin folder holding one in zig-out/lib/), opened and negotiated; no receipt needed.
fn openForCheck(io: std.Io, a: Allocator, path: []const u8) !loader.Opened {
    const file = if (std.mem.endsWith(u8, path, ".dylib")) path else blk: {
        const lib = try std.fs.path.join(a, &.{ path, "zig-out", "lib" });
        var dir = try std.Io.Dir.cwd().openDir(io, lib, .{ .iterate = true });
        defer dir.close(io);
        var it = dir.iterate();
        var found: ?[]const u8 = null;
        while (try it.next(io)) |e| if (std.mem.endsWith(u8, e.name, ".dylib")) {
            if (found != null) return error.SeveralDylibs;
            found = try std.fs.path.join(a, &.{ lib, e.name });
        };
        break :blk found orelse return error.NoDylibBuilt;
    };
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, file, a, .limited(1 << 30));
    defer a.free(bytes);
    return loader.open(a, file, loader.sha256Hex(bytes), hostMlx());
}

/// The conformance suite on an opened plugin's own fixture.
pub fn checkPlugin(io: std.Io, gpa: Allocator, p: loader.Plugin) !void {
    const m = p.manifest;
    var f: dylib_abi.Fixture = .{};
    const v = m.fixture(&f);
    if (!v.isOk()) {
        log.err("conformance: plugin {s}: its fixture ({s})\n", .{ p.name(), v.why.slice() });
        return error.Conformance;
    }
    defer m.fixture_free(&f);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const claims = try a.alloc([]const u8, m.claims.len);
    for (m.claims.slice(), claims) |c, *o| o.* = c.slice();
    try conformance.expectSubject(io, a, .{
        .name = p.name(),
        .claims = claims,
        .table = switch (m.kind) {
            .model => .{ .model = @ptrCast(@alignCast(m.table)) },
            .media => .{ .media = @ptrCast(@alignCast(m.table)) },
            .quant => .{ .quant = @ptrCast(@alignCast(m.table)) },
        },
        .fixture = try sdk_dylib.filesOf(a, &f),
    });
}

// ── install ──

/// Whether macOS marked the file as downloaded.
fn quarantined(a: Allocator, path: []const u8) bool {
    if (builtin.os.tag != .macos) return false;
    const z = a.dupeSentinel(u8, path, 0) catch return false;
    defer a.free(z);
    const getxattr = @extern(*const fn ([*:0]const u8, [*:0]const u8, ?*anyopaque, usize, u32, c_int) callconv(.c) isize, .{ .name = "getxattr" });
    return getxattr(z, "com.apple.quarantine", null, 0, 0, 0) >= 0;
}

/// Copies `src` into `dir` as `<name>.dylib` (a fresh file, so without the quarantine mark, which `yes` confirms
/// dropping), negotiates it, and writes `<name>.json`, the receipt of its hash.
pub fn install(io: std.Io, a: Allocator, src: []const u8, dir: []const u8, yes: bool) !loader.Opened {
    if (!std.mem.endsWith(u8, src, ".dylib")) return .{ .refused = try a.dupe(u8, "not a .dylib") };
    if (quarantined(a, src) and !yes) return .{ .refused = try a.dupe(u8, "is quarantined (downloaded): check where it came from, then rerun with --yes") };
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, src, a, .limited(1 << 30));
    defer a.free(bytes);
    const hash = loader.sha256Hex(bytes);
    var plugins = try std.Io.Dir.cwd().createDirPathOpen(io, dir, .{});
    defer plugins.close(io);
    var rnd: [8]u8 = undefined;
    io.random(&rnd);
    const tmp_name = try std.fmt.allocPrint(a, ".install-{x}.dylib", .{std.mem.readInt(u64, &rnd, .little)});
    defer a.free(tmp_name);
    try plugins.writeFile(io, .{ .sub_path = tmp_name, .data = bytes });
    const tmp = try std.fs.path.join(a, &.{ dir, tmp_name });
    defer a.free(tmp);
    var got = try loader.open(a, tmp, hash, hostMlx());
    const p = switch (got) {
        .ok => |*p| p,
        .refused => {
            plugins.deleteFile(io, tmp_name) catch {};
            return got;
        },
    };
    const name = p.name();
    const file = try std.fmt.allocPrint(a, "{s}.dylib", .{name});
    defer a.free(file);
    try plugins.rename(tmp_name, plugins, file, io);
    a.free(p.path);
    p.path = try std.fs.path.join(a, &.{ dir, file });
    var abs_buf: [std.fs.max_path_bytes]u8 = undefined;
    const source = if (std.Io.Dir.cwd().realPathFile(io, src, &abs_buf)) |n| abs_buf[0..n] else |_| src;
    const receipt = try std.json.Stringify.valueAlloc(a, loader.Receipt{ .name = name, .kind = @tagName(p.manifest.kind), .sha256 = &hash, .source = source }, .{});
    defer a.free(receipt);
    const receipt_name = try std.fmt.allocPrint(a, "{s}.json", .{name});
    defer a.free(receipt_name);
    try plugins.writeFile(io, .{ .sub_path = receipt_name, .data = receipt });
    return got;
}

// ── tests ──

const testing = std.testing;

test "plugin new: a bad name or a missing kit is refused before anything is written" {
    try testing.expectError(error.BadName, new(testing.io, testing.allocator, "Bad-Name", .model, "/nonexistent", "."));
    try testing.expectError(error.BadName, new(testing.io, testing.allocator, "_x", .model, "/nonexistent", "."));
    try testing.expectError(error.BadName, new(testing.io, testing.allocator, "9x", .model, "/nonexistent", "."));
    try testing.expectError(error.KitMissing, new(testing.io, testing.allocator, "toy", .model, "/nonexistent-kit-dir", "."));
}

test "plugin new: the manifest's fingerprint carries the name's checksum, as Zig checks it" {
    const text = try zon(testing.allocator, testing.io, "toy", "../k");
    defer testing.allocator.free(text);
    const at = std.mem.indexOf(u8, text, ".fingerprint = 0x").? + ".fingerprint = 0x".len;
    const fp = try std.fmt.parseInt(u64, text[at..][0..16], 16);
    try testing.expectEqual(std.hash.Crc32.hash("toy"), @as(u32, @truncate(fp >> 32)));
    try testing.expect(@as(u32, @truncate(fp)) != 0);
    try testing.expect(std.mem.indexOf(u8, text, ".path = \"../k\"") != null and std.mem.indexOf(u8, text, ".name = .toy,") != null);
}

test "plugin install: a file that is not a plugin is refused and leaves nothing behind" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(testing.io, &buf)];
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "junk.dylib", .data = "junk" });
    const src = try std.fs.path.join(testing.allocator, &.{ root, "junk.dylib" });
    defer testing.allocator.free(src);
    const dir = try std.fs.path.join(testing.allocator, &.{ root, "plugins" });
    defer testing.allocator.free(dir);
    const got = try install(testing.io, testing.allocator, src, dir, false);
    defer testing.allocator.free(got.refused);
    try testing.expect(std.mem.startsWith(u8, got.refused, "does not load: "));
    var plugins = try tmp.dir.openDir(testing.io, "plugins", .{ .iterate = true });
    defer plugins.close(testing.io);
    var it = plugins.iterate();
    try testing.expect(try it.next(testing.io) == null);
    const txt = try install(testing.io, testing.allocator, root, dir, false);
    defer testing.allocator.free(txt.refused);
    try testing.expectEqualStrings("not a .dylib", txt.refused);
}
