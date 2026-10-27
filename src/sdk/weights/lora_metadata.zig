//! CPU-only validation of adapter tensor shapes and safetensors offsets.

const std = @import("std");
const names = @import("lora.zig");

pub const Geometry = struct {
    layers: u32,
    hidden: u32,
    intermediate: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    targets: ?[]const @import("../../abi/plugin.zig").LoraTarget = null,

    fn dimensions(g: Geometry, target: names.LlmTarget) ![2]u64 {
        if (g.targets) |targets| {
            for (targets) |t| if (t.layer == target.layer and t.projection == target.projection) return .{ t.input, t.output };
            return error.InvalidLlmAdapter;
        }
        if (target.layer >= g.layers) return error.InvalidLlmAdapter;
        const q: u64 = @as(u64, g.heads) * g.head_dim;
        const kv: u64 = @as(u64, g.kv_heads) * g.head_dim;
        return switch (target.projection) {
            .q => .{ g.hidden, q },
            .k, .v => .{ g.hidden, kv },
            .o => .{ q, g.hidden },
            .gate, .up => .{ g.hidden, g.intermediate },
            .down => .{ g.intermediate, g.hidden },
        };
    }
};

const Key = struct {
    target: names.LlmTarget,
    b: bool,
    native: bool,

    fn parse(name: []const u8) ?Key {
        var key = name;
        const prefix = "base_model.model.";
        if (std.mem.startsWith(u8, key, prefix)) key = key[prefix.len..];
        inline for (.{ ".lora_a", ".lora_b", ".lora_A.weight", ".lora_B.weight" }, 0..) |suffix, i| {
            if (std.mem.endsWith(u8, key, suffix)) return .{
                .target = names.parseLlmTarget(key[0 .. key.len - suffix.len]) orelse return null,
                .b = i % 2 != 0,
                .native = i < 2,
            };
        }
        return null;
    }
};

const Range = struct {
    start: u64,
    end: u64,
    fn less(_: void, a: Range, b: Range) bool {
        return a.start < b.start;
    }
};

fn pair(v: std.json.Value) ![2]u64 {
    if (v != .array or v.array.items.len != 2) return error.InvalidLlmAdapter;
    var result: [2]u64 = undefined;
    for (&result, v.array.items) |*out, value| {
        if (value != .integer or value.integer < 0) return error.InvalidLlmAdapter;
        out.* = @intCast(value.integer);
    }
    return result;
}

pub fn inspect(a: std.mem.Allocator, header: []const u8, payload_bytes: u64, geometry: Geometry, rank: u32) !u64 {
    const parsed = std.json.parseFromSlice(std.json.Value, a, header, .{}) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => return error.InvalidLlmAdapter,
    };
    defer parsed.deinit();
    if (parsed.value != .object or rank == 0) return error.InvalidLlmAdapter;
    var pairs = std.AutoHashMap(u64, u8).init(a);
    defer pairs.deinit();
    var ranges: std.ArrayList(Range) = .empty;
    defer ranges.deinit(a);
    var it = parsed.value.object.iterator();
    while (it.next()) |entry| {
        if (std.mem.eql(u8, entry.key_ptr.*, "__metadata__")) continue;
        const key = Key.parse(entry.key_ptr.*) orelse return error.InvalidLlmAdapter;
        const value = entry.value_ptr.*;
        if (value != .object) return error.InvalidLlmAdapter;
        const dtype = value.object.get("dtype") orelse return error.InvalidLlmAdapter;
        if (dtype != .string) return error.InvalidLlmAdapter;
        const item_bytes: u64 = if (std.mem.eql(u8, dtype.string, "F32")) 4 else if (std.mem.eql(u8, dtype.string, "BF16") or std.mem.eql(u8, dtype.string, "F16")) 2 else return error.InvalidLlmAdapter;
        const shape = try pair(value.object.get("shape") orelse return error.InvalidLlmAdapter);
        const offsets = try pair(value.object.get("data_offsets") orelse return error.InvalidLlmAdapter);
        const dimensions = try geometry.dimensions(key.target);
        var expected: [2]u64 = if (key.b) .{ rank, dimensions[1] } else .{ dimensions[0], rank };
        if (!key.native) std.mem.swap(u64, &expected[0], &expected[1]);
        if (shape[0] == 0 or shape[1] == 0 or shape[0] > std.math.maxInt(c_int) or shape[1] > std.math.maxInt(c_int) or !std.mem.eql(u64, &shape, &expected)) return error.InvalidLlmAdapter;
        const bytes = std.math.mul(u64, std.math.mul(u64, shape[0], shape[1]) catch return error.InvalidLlmAdapter, item_bytes) catch return error.InvalidLlmAdapter;
        if (offsets[1] < offsets[0] or offsets[1] - offsets[0] != bytes or offsets[1] > payload_bytes) return error.InvalidLlmAdapter;
        try ranges.append(a, .{ .start = offsets[0], .end = offsets[1] });
        const id: u64 = @as(u64, key.target.layer) * 7 + @backingInt(key.target.projection);
        const bit: u8 = @as(u8, 1) << @as(u3, @intCast(@as(u8, @intFromBool(key.b)) + if (key.native) @as(u8, 0) else 2));
        const slot = try pairs.getOrPut(id);
        if (!slot.found_existing) slot.value_ptr.* = 0;
        if (slot.value_ptr.* & bit != 0) return error.InvalidLlmAdapter;
        slot.value_ptr.* |= bit;
    }
    if (ranges.items.len == 0) return error.InvalidLlmAdapter;
    var masks = pairs.valueIterator();
    while (masks.next()) |mask| if (mask.* != 3 and mask.* != 12) return error.InvalidLlmAdapter;
    std.mem.sort(Range, ranges.items, {}, Range.less);
    var end: u64 = 0;
    for (ranges.items) |range| {
        if (range.start != end) return error.InvalidLlmAdapter;
        end = range.end;
    }
    if (end != payload_bytes) return error.InvalidLlmAdapter;
    return end;
}

const tiny_geometry: Geometry = .{ .layers = 2, .hidden = 4, .intermediate = 8, .heads = 2, .kv_heads = 1, .head_dim = 2 };

test "gemma4 LoRA: exact loaded targets reject nonexistent projections and wrong widths" {
    var geometry = tiny_geometry;
    geometry.targets = &.{.{ .layer = 0, .projection = .q, .input = 4, .output = 8 }};
    try std.testing.expectError(error.InvalidLlmAdapter, inspect(std.testing.allocator, tiny_header, 64, geometry, 2));
    geometry.targets = &.{.{ .layer = 0, .projection = .k, .input = 4, .output = 4 }};
    try std.testing.expectError(error.InvalidLlmAdapter, inspect(std.testing.allocator, tiny_header, 64, geometry, 2));
    geometry.targets = &.{.{ .layer = 0, .projection = .q, .input = 4, .output = 4 }};
    const nested = try std.mem.replaceOwned(u8, std.testing.allocator, tiny_header, "model.layers", "language_model.model.layers");
    defer std.testing.allocator.free(nested);
    try std.testing.expectEqual(@as(u64, 64), try inspect(std.testing.allocator, nested, 64, geometry, 2));
}
const tiny_header =
    \\{"model.layers.0.self_attn.q_proj.lora_a":{"dtype":"F32","shape":[4,2],"data_offsets":[0,32]},"model.layers.0.self_attn.q_proj.lora_b":{"dtype":"F32","shape":[2,4],"data_offsets":[32,64]}}
;

test "LLM LoRA: safetensors header validates projection geometry and bills all adapter bytes" {
    try std.testing.expectEqual(@as(u64, 64), try inspect(std.testing.allocator, tiny_header, 64, tiny_geometry, 2));
    try std.testing.expectError(error.InvalidLlmAdapter, inspect(std.testing.allocator, tiny_header, 63, tiny_geometry, 2));
    try std.testing.expectError(error.InvalidLlmAdapter, inspect(std.testing.allocator, tiny_header, 64, tiny_geometry, 4));
    var wrong = tiny_geometry;
    wrong.hidden = 8;
    try std.testing.expectError(error.InvalidLlmAdapter, inspect(std.testing.allocator, tiny_header, 64, wrong, 2));
}

test "LLM LoRA: metadata refuses missing pairs, duplicate targets, wrong layers and overlapping offsets" {
    for ([_][]const u8{
        \\{}
        ,
        \\{"model.layers.0.self_attn.q_proj.lora_a":{"dtype":"F32","shape":[4,2],"data_offsets":[0,32]}}
        ,
        \\{"model.layers.2.self_attn.q_proj.lora_a":{"dtype":"F32","shape":[4,2],"data_offsets":[0,32]},"model.layers.2.self_attn.q_proj.lora_b":{"dtype":"F32","shape":[2,4],"data_offsets":[32,64]}}
        ,
        \\{"model.layers.0.self_attn.q_proj.lora_a":{"dtype":"F32","shape":[4,2],"data_offsets":[0,32]},"model.layers.0.self_attn.q_proj.lora_b":{"dtype":"F32","shape":[2,4],"data_offsets":[0,32]}}
        ,
        \\{"model.layers.0.self_attn.q_proj.lora_a":{"dtype":"F32","shape":[4,2],"data_offsets":[0,32]},"model.layers.0.self_attn.q_proj.lora_A.weight":{"dtype":"F32","shape":[2,4],"data_offsets":[32,64]}}
        ,
    }) |header| try std.testing.expectError(error.InvalidLlmAdapter, inspect(std.testing.allocator, header, 64, tiny_geometry, 2));
}

test "LLM LoRA: PEFT header transposes the declared A and B dimensions" {
    const header =
        \\{"base_model.model.model.layers.1.mlp.down_proj.lora_A.weight":{"dtype":"BF16","shape":[2,8],"data_offsets":[0,32]},"base_model.model.model.layers.1.mlp.down_proj.lora_B.weight":{"dtype":"BF16","shape":[4,2],"data_offsets":[32,48]}}
    ;
    try std.testing.expectEqual(@as(u64, 48), try inspect(std.testing.allocator, header, 48, tiny_geometry, 2));
    const wrapped = try std.mem.replaceOwned(u8, std.testing.allocator, header, "model.layers", "model.language_model.layers");
    defer std.testing.allocator.free(wrapped);
    try std.testing.expectEqual(@as(u64, 48), try inspect(std.testing.allocator, wrapped, 48, tiny_geometry, 2));
}
