//! Sushi packs at load: the tensor moves `sdk/quant/exl3/sushi_pack.zig` provides, applied per the model's config
//! (GLM-5-Next and MiMo layouts), and the resident bill of such a pack.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const model_mod = @import("../model.zig");
const Weights = @import("../../../sdk/weights/resident.zig").Weights;
const indexShardSet = @import("../../../sdk/weights/resident.zig").indexShardSet;
const freeShardSet = @import("../../../sdk/weights/resident.zig").freeShardSet;
const FP8_BLOCK = @import("../../../sdk/quant/exl3/sushi_pack.zig").FP8_BLOCK;
const ModelConfig = model_mod.ModelConfig;
const adaptMimoHeads = @import("../../../sdk/quant/exl3/sushi_pack.zig").adaptMimoHeads;
const decodeDenseMlp = @import("../../../sdk/quant/exl3/sushi_pack.zig").decodeDenseMlp;
const renameHyperConnections = @import("../../../sdk/quant/exl3/sushi_pack.zig").renameHyperConnections;
const splitFusedQkv = @import("../../../sdk/quant/exl3/sushi_pack.zig").splitFusedQkv;
const splitKvB = @import("../../../sdk/quant/exl3/sushi_pack.zig").splitKvB;

/// Also serves NVIDIA's ModelOpt GLM pack, which ships the same source layout.
pub fn adapt(config: *const ModelConfig, weights: *Weights, s: mlx.mlx_stream) !void {
    if (config.exl3 == null and !config.modelopt) return;
    if (config.isMimo() and config.exl3 != null) {
        try dequantFp8Trunk(config, weights, s);
    }
    if (config.isGlm5()) {
        try renameHyperConnections(weights);
        try splitKvB(weights, .{
            .heads = config.num_attention_heads,
            .nope = config.dsa_head_dim,
            .v = config.dsa_head_dim,
            .latent = config.dsa_kv_lora_rank,
        }, s);
    }
}

/// Tensors this engine never reads (vision and audio towers, GLM's EXL3 MTP layer past the trunk).
/// The load is lazy, so they are never evaluated: they cost no resident bytes.
fn isUnserved(config: *const ModelConfig, key: []const u8) bool {
    const prefixes: []const []const u8 = if (config.isMimo())
        &.{ "visual.", "audio_encoder.", "speech_embeddings." }
    else
        &.{"model.visual."};
    return hasAnyPrefix(key, prefixes) or pastTrunk(key, config.num_hidden_layers);
}

fn hasAnyPrefix(key: []const u8, prefixes: []const []const u8) bool {
    for (prefixes) |p| if (std.mem.startsWith(u8, key, p)) return true;
    return false;
}

/// `<prefix>.layers.N.<rest>` with N at or past the trunk's depth.
fn pastTrunk(key: []const u8, trunk_layers: u32) bool {
    const at = std.mem.indexOf(u8, key, ".layers.") orelse return false;
    const digits = key[at + ".layers.".len ..];
    const end = std.mem.indexOfScalar(u8, digits, '.') orelse return false;
    const n = std.fmt.parseInt(u32, digits[0..end], 10) catch return false;
    return n >= trunk_layers;
}

/// What the load holds resident once `adapt` has run: every shard tensor the engine reads, as
/// stored, except that the layer-0 FP8 MLP is held decoded (bf16, twice the code bytes) and each
/// QKV part's tile scales are held per row. The scale term is bounded by the 128-row tile.
pub fn residentBytes(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, config: *const ModelConfig) !u64 {
    var dir = try std.Io.Dir.openDirAbsolute(io, model_dir, .{ .iterate = true });
    defer dir.close(io);
    var referenced = indexShardSet(io, dir);
    defer if (referenced) |*r| freeShardSet(r);
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var owners = model_mod.indexOwners(io, arena.allocator(), dir);
    var total: u64 = 0;
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file and entry.kind != .sym_link) continue;
        if (!std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
        if (referenced) |r| if (!r.contains(entry.name)) continue;
        total += try shardResidentBytes(io, allocator, dir, entry.name, config, if (owners) |*o| o else null);
    }
    return total;
}

fn shardResidentBytes(io: std.Io, allocator: std.mem.Allocator, dir: std.Io.Dir, name: []const u8, config: *const ModelConfig, owners: ?*const model_mod.Owners) !u64 {
    var file = try dir.openFile(io, name, .{});
    defer file.close(io);
    var rbuf: [8192]u8 = undefined;
    var rs = file.reader(io, &rbuf);
    const len = try rs.interface.takeInt(u64, .little);
    if (len > 64 * 1024 * 1024) return error.InvalidSafetensorsHeader;
    const raw = try rs.interface.readAlloc(allocator, @intCast(len));
    defer allocator.free(raw);
    return headerResidentBytes(allocator, raw, config, name, owners);
}

/// Tables the loader widens to an owned f32 copy (`f32Table`) while the stored tensor stays resident:
/// the router, the hyper-connection and indexer tables, the routing bias, the KDA decay and bias.
fn isF32Copied(key: []const u8) bool {
    for ([_][]const u8{ ".mlp.gate.weight", ".mlp.gate.e_score_correction_bias", ".hc_attn_", ".hc_ffn_", ".indexer.index_kpool_compress_", ".self_attn.A_log", ".self_attn.dt_bias" }) |p|
        if (std.mem.indexOf(u8, key, p) != null) return true;
    return false;
}

fn shapeElements(entry: std.json.Value) !u64 {
    const shape = entry.object.get("shape") orelse return error.InvalidSafetensorsHeader;
    if (shape != .array) return error.InvalidSafetensorsHeader;
    var n: u64 = 1;
    for (shape.array.items) |d| n *= @intCast(d.integer);
    return n;
}

fn isKdaProjection(key: []const u8) bool {
    const at = std.mem.indexOf(u8, key, ".self_attn.") orelse return false;
    const rest = key[at + ".self_attn.".len ..];
    for ([_][]const u8{ "q_proj.", "k_proj.", "v_proj." }) |p| if (std.mem.startsWith(u8, rest, p)) return true;
    return false;
}

fn headerResidentBytes(allocator: std.mem.Allocator, raw: []const u8, config: *const ModelConfig, shard: []const u8, owners: ?*const model_mod.Owners) !u64 {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, raw, .{});
    defer parsed.deinit();
    var total: u64 = 0;
    var it = parsed.value.object.iterator();
    while (it.next()) |e| {
        if (std.mem.eql(u8, e.key_ptr.*, "__metadata__") or e.value_ptr.* != .object) continue;
        const key = e.key_ptr.*;
        if (isUnserved(config, key)) continue;
        if (owners) |o| if (o.get(key)) |owner| if (!std.mem.eql(u8, owner, shard)) continue;
        const offs = e.value_ptr.object.get("data_offsets") orelse return error.InvalidSafetensorsHeader;
        if (offs != .array or offs.array.items.len != 2) return error.InvalidSafetensorsHeader;
        const bytes: u64 = @intCast(offs.array.items[1].integer - offs.array.items[0].integer);
        const dtype = if (e.value_ptr.object.get("dtype")) |d| (if (d == .string) d.string else "") else "";
        const is_fp8 = std.mem.eql(u8, dtype, "F8_E4M3");
        if (config.isMimo() and std.mem.endsWith(u8, key, ".weight_scale_inv")) {
            // The dense MLP's scales are dropped once decoded; a QKV part keeps one row per code row.
            if (std.mem.indexOf(u8, key, ".self_attn.qkv_proj.") != null) total += bytes * FP8_BLOCK;
            continue;
        }
        total += bytes;
        if (config.isMimo() and is_fp8 and std.mem.indexOf(u8, key, ".mlp.") != null) total += bytes;
        // The KDA q/k/v projections are also held joined into one matrix.
        if (config.isGlm5() and isKdaProjection(key)) total += bytes;
        if (config.isGlm5() and isF32Copied(key)) total += try shapeElements(e.value_ptr.*) * 4;
        // `kv_b_proj` is held as the two dense bf16 halves, replacing its packed form.
        if (config.isGlm5() and std.mem.endsWith(u8, key, ".kv_b_proj.weight")) {
            const shape = e.value_ptr.object.get("shape") orelse return error.InvalidSafetensorsHeader;
            if (shape != .array or shape.array.items.len != 2) return error.InvalidSafetensorsHeader;
            total += @as(u64, @intCast(shape.array.items[0].integer)) * config.dsa_kv_lora_rank * 2;
        }
    }
    return total;
}

/// MiMo's trunk is block-FP8 as the source stores it; this engine reads dense projections, so
/// the fused rank-local `qkv_proj` becomes q/k/v and the layer-0 MLP dense, both bf16.
fn dequantFp8Trunk(config: *const ModelConfig, weights: *Weights, s: mlx.mlx_stream) !void {
    var buf: [160]u8 = undefined;
    for (0..config.num_hidden_layers) |i| {
        const li: u32 = @intCast(i);
        const base = try std.fmt.bufPrint(&buf, "{s}.layers.{d}", .{ config.weight_prefix, li });
        try splitFusedQkv(weights, base, qkvRows(config, li), s);
        if (li < config.first_k_dense_replace) try decodeDenseMlp(weights, base, s);
    }
    // The MTP heads are stored the same way and attend in a sliding layer's geometry.
    try adaptMimoHeads(weights, qkvRows(config, config.firstSlidingLayer() orelse return), s);
}

fn qkvRows(config: *const ModelConfig, li: u32) [3]u64 {
    return .{
        @as(u64, config.layerNumHeads(li)) * config.layerHeadDim(li),
        @as(u64, config.layerKVHeads(li)) * config.layerHeadDim(li),
        @as(u64, config.layerKVHeads(li)) * config.layerVHeadDim(li),
    };
}

// ── tests ──

const testing = std.testing;

test "sushi pack: the resident bill holds QKV as stored, the dense MLP decoded, the MTP heads, and nothing unserved" {
    const header =
        \\{"__metadata__":{"format":"pt"},
        \\"model.layers.0.self_attn.qkv_proj.weight":{"dtype":"F8_E4M3","shape":[1,1],"data_offsets":[0,1000]},
        \\"model.layers.0.self_attn.qkv_proj.weight_scale_inv":{"dtype":"F32","shape":[1,1],"data_offsets":[1000,1100]},
        \\"model.layers.0.mlp.gate_proj.weight":{"dtype":"F8_E4M3","shape":[1,1],"data_offsets":[1100,1300]},
        \\"model.layers.0.mlp.gate_proj.weight_scale_inv":{"dtype":"F32","shape":[1,1],"data_offsets":[1300,1400]},
        \\"model.layers.0.mlp.switch_mlp.up_proj.trellis":{"dtype":"U16","shape":[1,1],"data_offsets":[1400,1900]},
        \\"model.mtp.layers.0.eh_proj.weight":{"dtype":"BF16","shape":[1,1],"data_offsets":[1900,2900]},
        \\"visual.blocks.0.attn.proj.weight":{"dtype":"BF16","shape":[1,1],"data_offsets":[2900,3900]}}
    ;
    var config: ModelConfig = .{ .model_type = "mimo_v2", .num_hidden_layers = 48 };
    // qkv 1000 + its scales 100 x 128 + gate 200 held twice + trellis 500 + the MTP head's 1000.
    try testing.expectEqual(@as(u64, 1000 + 100 * 128 + 400 + 500 + 1000), try headerResidentBytes(testing.allocator, header, &config, "a.safetensors", null));
}

test "sushi pack: the MiMo MTP heads get the trunk's split QKV and decoded MLP" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    // Layer 0 slides (pattern 2): its geometry is the heads'. q 32 + k 32 + v 32 rows over 4 ranks.
    var config: ModelConfig = .{ .model_type = "mimo_v2", .num_hidden_layers = 2, .num_attention_heads = 4, .num_key_value_heads = 4, .head_dim = 8, .has_sliding_window = true, .sliding_window_pattern = 2, .weight_prefix = "model" };
    var w = Weights.init(testing.allocator);
    defer w.deinit();
    var codes: [128 * 128]u8 = undefined;
    for (&codes, 0..) |*c, i| c.* = @intCast((i * 13 + 3) % 0x70);
    const ones = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
    try w.put("model.mtp.layers.0.eh_proj.weight", mlx.mlx_array_new_float(1));
    try w.put("model.mtp.layers.0.self_attn.qkv_proj.weight", mlx.mlx_array_new_data(&codes, &[_]c_int{ 96, 128 }, 2, .uint8));
    try w.put("model.mtp.layers.0.self_attn.qkv_proj.weight_scale_inv", mlx.mlx_array_new_data(&ones, &[_]c_int{ 4, 1 }, 2, .float32));
    for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |proj| {
        var k1: [96]u8 = undefined;
        var k2: [96]u8 = undefined;
        try w.put(try std.fmt.bufPrint(&k1, "model.mtp.layers.0.mlp.{s}.weight", .{proj}), mlx.mlx_array_new_data(&codes, &[_]c_int{ 128, 128 }, 2, .uint8));
        try w.put(try std.fmt.bufPrint(&k2, "model.mtp.layers.0.mlp.{s}.weight_scale_inv", .{proj}), mlx.mlx_array_new_data(&ones, &[_]c_int{ 1, 1 }, 2, .float32));
    }
    try dequantFp8Trunk(&config, &w, s);

    try testing.expect(w.get("model.mtp.layers.0.self_attn.qkv_proj.weight") == null);
    for ([_][]const u8{ "q_proj", "k_proj", "v_proj" }) |proj| {
        var k1: [96]u8 = undefined;
        const part = w.get(try std.fmt.bufPrint(&k1, "model.mtp.layers.0.self_attn.{s}.weight", .{proj})).?;
        try testing.expectEqualSlices(c_int, &.{ 32, 128 }, mlx.getShape(part));
        try testing.expect(w.get(try std.fmt.bufPrint(&k1, "model.mtp.layers.0.self_attn.{s}.scales", .{proj})) != null);
    }
    for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |proj| {
        var k1: [96]u8 = undefined;
        try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(w.get(try std.fmt.bufPrint(&k1, "model.mtp.layers.0.mlp.{s}.weight", .{proj})).?));
        try testing.expect(w.get(try std.fmt.bufPrint(&k1, "model.mtp.layers.0.mlp.{s}.weight_scale_inv", .{proj})) == null);
    }
}

test "sushi pack: the GLM bill holds the joined KDA projections and the dense kv_b halves" {
    const header =
        \\{"model.language_model.layers.0.self_attn.q_proj.weight":{"dtype":"U32","shape":[1,1],"data_offsets":[0,100]},
        \\"model.language_model.layers.3.self_attn.kv_b_proj.weight":{"dtype":"U32","shape":[10,1],"data_offsets":[100,160]},
        \\"model.visual.patch_embed.proj.weight":{"dtype":"BF16","shape":[1,1],"data_offsets":[160,260]},
        \\"model.language_model.layers.45.enorm.weight":{"dtype":"BF16","shape":[1,1],"data_offsets":[260,360]}}
    ;
    var config: ModelConfig = .{ .model_type = "glm5_next", .num_hidden_layers = 45, .dsa_kv_lora_rank = 8 };
    // q 100 twice, kv_b 60 plus its dense 10 x 8 x 2 bf16 bytes.
    try testing.expectEqual(@as(u64, 200 + 60 + 160), try headerResidentBytes(testing.allocator, header, &config, "a.safetensors", null));
}

test "sushi pack: the GLM bill adds the f32 copies of the router and the hyper-connection tables" {
    const header =
        \\{"model.language_model.layers.3.mlp.gate.weight":{"dtype":"BF16","shape":[4,8],"data_offsets":[0,64]},
        \\"model.language_model.layers.3.hc_attn_fn":{"dtype":"BF16","shape":[16,4],"data_offsets":[64,192]},
        \\"model.language_model.layers.3.self_attn.indexer.index_kpool_compress_gate":{"dtype":"BF16","shape":[2,8],"data_offsets":[192,224]},
        \\"model.language_model.layers.3.input_layernorm.weight":{"dtype":"BF16","shape":[8],"data_offsets":[224,240]}}
    ;
    var config: ModelConfig = .{ .model_type = "glm5_next", .num_hidden_layers = 45 };
    // 240 stored bytes, plus 4 bytes per element of the 32 + 64 + 16 widened ones.
    try testing.expectEqual(@as(u64, 240 + 4 * (32 + 64 + 16)), try headerResidentBytes(testing.allocator, header, &config, "a.safetensors", null));
}

test "sushi pack: a tensor another shard owns is not billed" {
    const header =
        \\{"model.embed_tokens.weight":{"dtype":"BF16","shape":[1,1],"data_offsets":[0,700]},
        \\"lm_head.weight":{"dtype":"U32","shape":[1,1],"data_offsets":[700,800]}}
    ;
    var owners: model_mod.Owners = .empty;
    defer owners.deinit(testing.allocator);
    try owners.put(testing.allocator, "model.embed_tokens.weight", "affine.safetensors");
    var config: ModelConfig = .{ .model_type = "mimo_v2", .num_hidden_layers = 48 };
    try testing.expectEqual(@as(u64, 100), try headerResidentBytes(testing.allocator, header, &config, "source.safetensors", &owners));
}
