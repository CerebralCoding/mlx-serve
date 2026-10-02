const std = @import("std");

pub const EmbeddedSpec = struct { rows: u64, dim: u32, shards: u32, layer_index: u32 = 1 };
pub const EmbeddedInfo = struct { payload_bytes: u64 };
const prefix = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.";
const layer_prefix = "language_model.model.layers.";
const shard_marker = ".ple.ple_embedding.ngram_embedding.shards.";
const Part = enum(u2) { weight, scales, biases };

const Key = struct { layer: u32, shard: u32, part: Part };

fn parseKey(name: []const u8) ?Key {
    if (!std.mem.startsWith(u8, name, layer_prefix)) return null;
    const tail = name[layer_prefix.len..];
    const marker = std.mem.indexOf(u8, tail, shard_marker) orelse return null;
    if (marker == 0 or (marker > 1 and tail[0] == '0')) return null;
    const layer = std.fmt.parseInt(u32, tail[0..marker], 10) catch return null;
    const rest = tail[marker + shard_marker.len ..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    if (dot == 0 or (dot > 1 and rest[0] == '0')) return null;
    const shard = std.fmt.parseInt(u32, rest[0..dot], 10) catch return null;
    const part: Part = if (std.mem.eql(u8, rest[dot + 1 ..], "weight")) .weight else if (std.mem.eql(u8, rest[dot + 1 ..], "scales")) .scales else if (std.mem.eql(u8, rest[dot + 1 ..], "biases")) .biases else return null;
    return .{ .layer = layer, .shard = shard, .part = part };
}

pub fn embeddedTensorName(name: []const u8) bool {
    return parseKey(name) != null;
}

const Candidate = struct {
    names: [3]?[]const u8 = .{ null, null, null },
    files: [3]?[]const u8 = .{ null, null, null },
};

const Tensor = struct {
    file: usize,
    off: usize,
    len: usize,
    rows: u64,
    cols: u32,
};

const Shard = struct {
    first: u64,
    rows: u64,
    parts: [3]Tensor,
};

const File = struct {
    fd: std.c.fd_t,
    map: []align(std.heap.page_size_min) const u8,
    obj: std.json.ObjectMap,
    data_off: usize,
};

pub const RowParts = struct {
    weight: []const u8,
    scales: []const u8,
    biases: []const u8,
};

pub const EmbeddedTable = struct {
    arena: std.heap.ArenaAllocator,
    files: []File,
    shards: []Shard,
    rows: u64,
    dim: u32,
    bits: u32,
    group_size: u32,
    wcols: u32,
    scols: u32,
    payload_bytes: u64,

    pub fn close(self: *EmbeddedTable) void {
        for (self.files) |f| {
            std.posix.munmap(f.map);
            _ = std.c.close(f.fd);
        }
        self.arena.deinit();
    }

    fn locate(self: *const EmbeddedTable, row: u64) struct { shard: *const Shard, local: u64 } {
        std.debug.assert(row < self.rows);
        var lo: usize = 0;
        var hi: usize = self.shards.len;
        while (lo + 1 < hi) {
            const mid = lo + (hi - lo) / 2;
            if (self.shards[mid].first <= row) lo = mid else hi = mid;
        }
        return .{ .shard = &self.shards[lo], .local = row - self.shards[lo].first };
    }

    pub fn rowParts(self: *const EmbeddedTable, row: u64) RowParts {
        const where = self.locate(row);
        const p = where.shard.parts;
        const wlen: usize = self.wcols * 4;
        const slen: usize = self.scols * 2;
        return .{
            .weight = self.files[p[0].file].map[p[0].off + where.local * wlen ..][0..wlen],
            .scales = self.files[p[1].file].map[p[1].off + where.local * slen ..][0..slen],
            .biases = self.files[p[2].file].map[p[2].off + where.local * slen ..][0..slen],
        };
    }

    pub fn preadPart(self: *const EmbeddedTable, row: u64, part: usize, dst: []u8) bool {
        const where = self.locate(row);
        const t = where.shard.parts[part];
        if (dst.len != t.len / t.rows) return false;
        const off = t.off + where.local * dst.len;
        return std.c.pread(self.files[t.file].fd, dst.ptr, dst.len, @intCast(off)) == @as(isize, @intCast(dst.len));
    }
};

fn validFileName(name: []const u8) bool {
    return name.len > 0 and std.mem.endsWith(u8, name, ".safetensors") and std.mem.indexOfAny(u8, name, "/\\") == null and !std.mem.eql(u8, name, ".") and !std.mem.eql(u8, name, "..");
}

fn openFile(a: std.mem.Allocator, model_dir: []const u8, name: []const u8) !File {
    if (!validFileName(name)) return error.EmbeddedPleIndex;
    var path: [std.fs.max_path_bytes]u8 = undefined;
    const full = std.fmt.bufPrint(&path, "{s}/{s}", .{ model_dir, name }) catch return error.NameTooLong;
    if (full.len >= path.len) return error.NameTooLong;
    path[full.len] = 0;
    const fd = std.c.open(path[0..full.len :0], .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.FileNotFound;
    errdefer _ = std.c.close(fd);
    const end = std.c.lseek(fd, 0, std.c.SEEK.END);
    if (end < 8) return error.EmbeddedPleHeader;
    const size: usize = @intCast(end);
    const map = try std.posix.mmap(null, size, .{ .READ = true }, .{ .TYPE = .PRIVATE }, fd, 0);
    errdefer std.posix.munmap(map);
    const hlen = std.mem.readInt(u64, map[0..8], .little);
    if (hlen == 0 or hlen > 4 * 1024 * 1024 or hlen > size - 8) return error.EmbeddedPleHeader;
    const parsed = std.json.parseFromSliceLeaky(std.json.Value, a, map[8 .. 8 + @as(usize, @intCast(hlen))], .{}) catch return error.EmbeddedPleHeader;
    if (parsed != .object) return error.EmbeddedPleHeader;
    return .{ .fd = fd, .map = map, .obj = parsed.object, .data_off = 8 + @as(usize, @intCast(hlen)) };
}

fn tensor(file: File, file_id: usize, name: []const u8, dtype: []const u8, elem: u64) !Tensor {
    const v = file.obj.get(name) orelse return error.EmbeddedPleHeader;
    if (v != .object) return error.EmbeddedPleHeader;
    const dt = v.object.get("dtype") orelse return error.EmbeddedPleHeader;
    if (dt != .string or !std.mem.eql(u8, dt.string, dtype)) return error.EmbeddedPleDtype;
    const shape = v.object.get("shape") orelse return error.EmbeddedPleHeader;
    const offsets = v.object.get("data_offsets") orelse return error.EmbeddedPleHeader;
    if (shape != .array or shape.array.items.len != 2 or offsets != .array or offsets.array.items.len != 2) return error.EmbeddedPleHeader;
    const r = shape.array.items;
    const o = offsets.array.items;
    for ([_]std.json.Value{ r[0], r[1], o[0], o[1] }) |x| if (x != .integer) return error.EmbeddedPleHeader;
    if (r[0].integer <= 0 or r[1].integer <= 0 or o[0].integer < 0 or o[1].integer < o[0].integer) return error.EmbeddedPleRegion;
    const rows: u64 = @intCast(r[0].integer);
    const cols: u64 = @intCast(r[1].integer);
    const start: u64 = @intCast(o[0].integer);
    const stop: u64 = @intCast(o[1].integer);
    if (cols > std.math.maxInt(u32)) return error.EmbeddedPleRegion;
    const need = std.math.mul(u64, std.math.mul(u64, rows, cols) catch return error.EmbeddedPleRegion, elem) catch return error.EmbeddedPleRegion;
    if (stop - start != need) return error.EmbeddedPleRegion;
    const absolute = std.math.add(u64, file.data_off, stop) catch return error.EmbeddedPleRegion;
    if (absolute > file.map.len) return error.EmbeddedPleRegion;
    return .{ .file = file_id, .off = file.data_off + @as(usize, @intCast(start)), .len = @intCast(need), .rows = rows, .cols = @intCast(cols) };
}

pub fn openEmbedded(model_dir: []const u8, expected: EmbeddedSpec) !?EmbeddedTable {
    if (expected.rows == 0 or expected.dim == 0 or expected.shards == 0 or expected.shards > 1024) return error.EmbeddedPleSpec;
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    errdefer arena.deinit();
    const a = arena.allocator();
    const io = std.Io.Threaded.global_single_threaded.io();
    var dir = try std.Io.Dir.cwd().openDir(io, model_dir, .{});
    defer dir.close(io);
    const index = dir.readFileAlloc(io, "model.safetensors.index.json", a, .limited(16 * 1024 * 1024)) catch return null;
    const parsed = std.json.parseFromSliceLeaky(std.json.Value, a, index, .{}) catch return error.EmbeddedPleIndex;
    if (parsed != .object) return error.EmbeddedPleIndex;
    const weights = parsed.object.get("weight_map") orelse return error.EmbeddedPleIndex;
    if (weights != .object) return error.EmbeddedPleIndex;
    const candidates = try a.alloc(Candidate, expected.shards);
    for (candidates) |*c| c.* = .{};
    var found: usize = 0;
    var it = weights.object.iterator();
    while (it.next()) |entry| {
        const name = entry.key_ptr.*;
        if (!std.mem.startsWith(u8, name, layer_prefix) or std.mem.indexOf(u8, name, shard_marker) == null) continue;
        const key = parseKey(name) orelse return error.EmbeddedPleIndex;
        if (key.layer != expected.layer_index or key.shard >= expected.shards or entry.value_ptr.* != .string) return error.EmbeddedPleIndex;
        const part = @intFromEnum(key.part);
        const c = &candidates[key.shard];
        if (c.names[part] != null) return error.EmbeddedPleDuplicate;
        c.names[part] = name;
        c.files[part] = entry.value_ptr.string;
        found += 1;
    }
    if (found == 0) {
        arena.deinit();
        return null;
    }
    if (found != @as(usize, expected.shards) * 3) return error.EmbeddedPleMissing;
    var files: std.ArrayList(File) = .empty;
    errdefer for (files.items) |f| {
        std.posix.munmap(f.map);
        _ = std.c.close(f.fd);
    };
    var file_ids: std.StringHashMapUnmanaged(usize) = .empty;
    const shards = try a.alloc(Shard, expected.shards);
    var first: u64 = 0;
    var payload: u64 = 0;
    var wcols: u32 = 0;
    var scols: u32 = 0;
    var group_size: u32 = 0;
    for (candidates, 0..) |c, i| {
        var parts: [3]Tensor = undefined;
        for (0..3) |p| {
            const name = c.names[p] orelse return error.EmbeddedPleMissing;
            const file_name = c.files[p] orelse return error.EmbeddedPleMissing;
            const id = if (file_ids.get(file_name)) |existing| existing else blk: {
                const f = try openFile(a, model_dir, file_name);
                const next = files.items.len;
                try files.append(a, f);
                try file_ids.put(a, file_name, next);
                break :blk next;
            };
            parts[p] = try tensor(files.items[id], id, name, if (p == 0) "U32" else "BF16", if (p == 0) 4 else 2);
            payload = std.math.add(u64, payload, parts[p].len) catch return error.EmbeddedPleRegion;
        }
        if (parts[0].rows != parts[1].rows or parts[0].rows != parts[2].rows or parts[1].cols != parts[2].cols) return error.EmbeddedPleRegion;
        if (@as(u64, parts[0].cols) * 8 != expected.dim or parts[1].cols == 0 or expected.dim % parts[1].cols != 0) return error.EmbeddedPleGeometry;
        const gs = expected.dim / parts[1].cols;
        if (gs == 0 or gs > 1024 or (i != 0 and (parts[0].cols != wcols or parts[1].cols != scols or gs != group_size))) return error.EmbeddedPleGeometry;
        for (parts, 0..) |p, x| for (parts[0..x]) |q| {
            if (p.file == q.file and p.off < q.off + q.len and q.off < p.off + p.len) return error.EmbeddedPleOverlap;
        };
        shards[i] = .{ .first = first, .rows = parts[0].rows, .parts = parts };
        first = std.math.add(u64, first, parts[0].rows) catch return error.EmbeddedPleRegion;
        wcols = parts[0].cols;
        scols = parts[1].cols;
        group_size = gs;
    }
    if (first != expected.rows) return error.EmbeddedPleGeometry;
    for (shards, 0..) |shard, i| for (shard.parts) |p| {
        for (shards[0..i]) |prior| for (prior.parts) |q| {
            if (p.file == q.file and p.off < q.off + q.len and q.off < p.off + p.len) return error.EmbeddedPleOverlap;
        };
    };
    return .{ .arena = arena, .files = files.items, .shards = shards, .rows = first, .dim = expected.dim, .bits = 4, .group_size = group_size, .wcols = wcols, .scols = scols, .payload_bytes = payload };
}

pub fn inspectEmbedded(model_dir: []const u8, expected: EmbeddedSpec) !?EmbeddedInfo {
    var table = (try openEmbedded(model_dir, expected)) orelse return null;
    defer table.close();
    return .{ .payload_bytes = table.payload_bytes };
}

pub const FixtureVariant = enum { valid, missing, duplicate, extra, dtype, bounds, overlap };

pub fn writeFixture(td: *std.testing.TmpDir, variant: FixtureVariant) !void {
    const a = std.testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const index = try std.fmt.allocPrint(a,
        "{{\"weight_map\":{{\"{s}2.biases\":\"a.safetensors\",\"{s}0.weight\":\"a.safetensors\",\"{s}1.weight\":\"b.safetensors\",\"{s}2.weight\":\"a.safetensors\",\"{s}0.scales\":\"a.safetensors\",\"{s}1.scales\":\"b.safetensors\",\"{s}0.biases\":\"a.safetensors\",\"{s}2.scales\":\"a.safetensors\"{s}}}}}",
        .{ prefix, prefix, prefix, prefix, prefix, prefix, prefix, prefix, if (variant == .missing) "" else if (variant == .duplicate) ",\"language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.0.weight\":\"a.safetensors\",\"language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.1.biases\":\"b.safetensors\"" else if (variant == .extra) ",\"language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.1.biases\":\"b.safetensors\",\"language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.3.weight\":\"a.safetensors\"" else ",\"language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.1.biases\":\"b.safetensors\"" });
    defer a.free(index);
    try td.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index });
    const header_a = try std.fmt.allocPrint(a,
        "{{\"{s}2.biases\":{{\"dtype\":\"BF16\",\"shape\":[3,1],\"data_offsets\":[0,6]}}," ++
            "\"{s}0.weight\":{{\"dtype\":\"{s}\",\"shape\":[2,4],\"data_offsets\":[6,38]}}," ++
            "\"{s}2.weight\":{{\"dtype\":\"U32\",\"shape\":[3,4],\"data_offsets\":[{d},{d}]}}," ++
            "\"{s}0.scales\":{{\"dtype\":\"BF16\",\"shape\":[2,1],\"data_offsets\":[86,90]}}," ++
            "\"{s}0.biases\":{{\"dtype\":\"BF16\",\"shape\":[2,1],\"data_offsets\":[{d},{d}]}}," ++
            "\"{s}2.scales\":{{\"dtype\":\"BF16\",\"shape\":[3,1],\"data_offsets\":[94,100]}}}}",
        .{ prefix, prefix, if (variant == .dtype) "F32" else "U32", prefix, if (variant == .bounds) @as(u32, 999) else 38, if (variant == .bounds) @as(u32, 1047) else 86, prefix, prefix, if (variant == .overlap) @as(u32, 88) else 90, if (variant == .overlap) @as(u32, 92) else 94, prefix });
    defer a.free(header_a);
    const header_b = try std.fmt.allocPrint(a,
        "{{\"{s}1.scales\":{{\"dtype\":\"BF16\",\"shape\":[1,1],\"data_offsets\":[0,2]}}," ++
            "\"{s}1.weight\":{{\"dtype\":\"U32\",\"shape\":[1,4],\"data_offsets\":[2,18]}}," ++
            "\"{s}1.biases\":{{\"dtype\":\"BF16\",\"shape\":[1,1],\"data_offsets\":[18,20]}}}}",
        .{ prefix, prefix, prefix });
    defer a.free(header_b);
    var data_a: [100]u8 = @splat(0);
    @memset(data_a[6..22], 0x00);
    @memset(data_a[22..38], 0x11);
    @memset(data_a[38..54], 0x33);
    @memset(data_a[54..70], 0x44);
    @memset(data_a[70..86], 0x55);
    for ([_]usize{ 86, 88, 94, 96, 98 }) |off| std.mem.writeInt(u16, data_a[off..][0..2], 0x3f80, .little);
    var data_b: [20]u8 = @splat(0);
    std.mem.writeInt(u16, data_b[0..2], 0x3f80, .little);
    @memset(data_b[2..18], 0x22);
    try writeSafetensors(td, "a.safetensors", header_a, &data_a);
    try writeSafetensors(td, "b.safetensors", header_b, &data_b);
}

fn writeSafetensors(td: *std.testing.TmpDir, name: []const u8, header: []const u8, data: []const u8) !void {
    const a = std.testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    if (header.len > 2048) return error.TestHeaderTooLarge;
    const buf = try a.alloc(u8, 8 + 2048 + data.len);
    defer a.free(buf);
    std.mem.writeInt(u64, buf[0..8], 2048, .little);
    @memset(buf[8..2056], ' ');
    @memcpy(buf[8 .. 8 + header.len], header);
    @memcpy(buf[2056..], data);
    try td.dir.writeFile(io, .{ .sub_path = name, .data = buf });
}

test "embedded PLE reads unequal numbered shards across mixed files" {
    var td = std.testing.tmpDir(.{});
    defer td.cleanup();
    try writeFixture(&td, .valid);
    const io = std.Io.Threaded.global_single_threaded.io();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try td.dir.realPath(io, &path_buf);
    const spec: EmbeddedSpec = .{ .rows = 6, .dim = 32, .shards = 3 };
    const info = (try inspectEmbedded(path_buf[0..path_len], spec)).?;
    try std.testing.expectEqual(@as(u64, 120), info.payload_bytes);
    var t = (try openEmbedded(path_buf[0..path_len], spec)).?;
    defer t.close();
    for ([_]u64{ 5, 1, 2, 0, 3, 5, 4 }) |r| {
        const parts = t.rowParts(r);
        try std.testing.expectEqual(@as(u8, @intCast(r * 0x11)), parts.weight[0]);
        try std.testing.expectEqual(@as(u16, 0x3f80), std.mem.readInt(u16, parts.scales[0..2], .little));
        var w: [16]u8 = undefined;
        var s: [2]u8 = undefined;
        var b: [2]u8 = undefined;
        try std.testing.expect(t.preadPart(r, 0, &w));
        try std.testing.expect(t.preadPart(r, 1, &s));
        try std.testing.expect(t.preadPart(r, 2, &b));
        try std.testing.expectEqualSlices(u8, parts.weight, &w);
        try std.testing.expectEqualSlices(u8, parts.scales, &s);
        try std.testing.expectEqualSlices(u8, parts.biases, &b);
    }
    try std.testing.expect(!embeddedTensorName(prefix ++ "00.weight"));
    try std.testing.expect(!embeddedTensorName("language_model.model.layers.1.ple.ple_embedding.layer_multipliers"));
}

test "embedded PLE rejects partial and corrupt shard layouts" {
    const spec: EmbeddedSpec = .{ .rows = 6, .dim = 32, .shards = 3 };
    inline for (.{ .missing, .duplicate, .extra, .dtype, .bounds, .overlap }) |variant| {
        var td = std.testing.tmpDir(.{});
        defer td.cleanup();
        try writeFixture(&td, variant);
        const io = std.Io.Threaded.global_single_threaded.io();
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path_len = try td.dir.realPath(io, &path_buf);
        try std.testing.expectError(switch (variant) {
            .missing => error.EmbeddedPleMissing,
            .duplicate, .extra => error.EmbeddedPleIndex,
            .dtype => error.EmbeddedPleDtype,
            .bounds => error.EmbeddedPleRegion,
            .overlap => error.EmbeddedPleOverlap,
            else => unreachable,
        }, openEmbedded(path_buf[0..path_len], spec));
    }
}

test "embedded PLE indexed checkpoint metadata validates without tensor reads" {
    const raw = std.c.getenv("QWEN4_EMBEDDED_TEST_MODEL") orelse return error.SkipZigTest;
    const path = std.mem.sliceTo(raw, 0);
    const info = (try inspectEmbedded(path, .{ .rows = 320_001_536, .dim = 160, .shards = 128 })).?;
    try std.testing.expectEqual(@as(u64, 32_000_153_600), info.payload_bytes);
}
