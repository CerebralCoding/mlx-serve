//! A config's quantization blocks and the built-in formats' claims and validation, MLX-free (discovery and the
//! load preflight run them): which block a pack carries, which format serves it, the defaults it declares.

const std = @import("std");
const abi = @import("../../abi/quant.zig");
const common = @import("../../abi/common.zig");
const sushi_exl3 = @import("sushi_exl3");

const Str = abi.Str;
const Claim = abi.Claim;
const Verdict = abi.Verdict;
const Value = std.json.Value;

/// The config keys a quantization block sits under.
pub const keys = [_][]const u8{ "quantization", "quantization_config", "expert_quant" };

pub const Block = struct { key: []const u8, json: []const u8 };

/// The blocks of `config_json`, in `keys` order, each re-serialized into `a`; none when it does not parse.
pub fn blocksOf(a: std.mem.Allocator, config_json: []const u8) ![]Block {
    const root = parseObject(a, config_json) orelse return &.{};
    var out: std.ArrayList(Block) = .empty;
    for (keys) |k| {
        const v = root.get(k) orelse continue;
        if (v != .object) continue;
        try out.append(a, .{ .key = k, .json = try std.json.Stringify.valueAlloc(a, v, .{}) });
    }
    return out.items;
}

/// The (bits, group size, mode) a block declares model-wide; each weight still resolves from its own geometry.
pub const Declared = struct { bits: u32 = 0, group_size: u32 = 64, mode: []const u8 = "affine" };

/// MLX's affine kernels: an already-quantized pack outside these dies at Metal kernel load.
pub fn affineBits(bits: u32) bool {
    return switch (bits) {
        2, 3, 4, 5, 6, 8 => true,
        else => false,
    };
}

/// The MLX modes a block or a per-weight override may name.
pub fn mlxMode(mode: []const u8) bool {
    for ([_][]const u8{ "affine", "nvfp4", "mxfp4", "mxfp8" }) |m| if (std.mem.eql(u8, mode, m)) return true;
    return false;
}

/// What the block declares; a ModelOpt block is NVFP4 g16 (its MXFP8 linears resolve per weight).
pub fn declared(a: std.mem.Allocator, block_json: []const u8) Declared {
    const o = parseObject(a, block_json) orelse return .{};
    if (eqlField(o, "quant_method", "modelopt")) return .{ .bits = 4, .group_size = 16, .mode = "nvfp4" };
    var d: Declared = .{};
    if (uintField(o, "bits")) |b| d.bits = b;
    if (uintField(o, "group_size")) |g| d.group_size = g;
    if (o.get("mode")) |m| if (m == .string) {
        d.mode = m.string;
    };
    return d;
}

/// An NVIDIA ModelOpt block (`quant_method: modelopt`).
pub fn isModelOpt(block_json: []const u8) bool {
    return methodIs(.of(block_json), "modelopt");
}

/// The Hadamard block every listed module rotates by (Prism's `modules[].block`); null when there is no such list
/// or the modules disagree.
pub fn hadamardBlock(a: std.mem.Allocator, config_json: []const u8) ?u32 {
    const root = parseObject(a, config_json) orelse return null;
    const mods = root.get("modules") orelse return null;
    if (mods != .array) return null;
    var block: u32 = 0;
    for (mods.array.items) |m| {
        if (m != .object) return null;
        const b = uintField(m.object, "block") orelse return null;
        if (b == 0 or (block != 0 and b != block)) return null;
        block = b;
    }
    return if (block == 0) null else block;
}

fn hasModules(a: std.mem.Allocator, config_json: []const u8) bool {
    const root = parseObject(a, config_json) orelse return false;
    const mods = root.get("modules") orelse return false;
    return mods == .array;
}

// ── the built-in formats' claims and validation ──

pub const ClaimsFn = *const fn (Str, Str) callconv(.c) Claim;
pub const ValidateFn = *const fn (Str, Str, *const abi.PackDir) callconv(.c) Verdict;

pub const Builtin = struct { name: []const u8, claims: ClaimsFn, validate: ValidateFn };

/// The built-in formats, in the order `formats.zig` lists their tables.
pub const builtins = [_]Builtin{
    .{ .name = "affine", .claims = &claimsAffine, .validate = &validateMlx },
    .{ .name = "fp8", .claims = &claimsFp8, .validate = &validateFp8 },
    .{ .name = "mxfp8", .claims = &claimsMxfp8, .validate = &validateMlx },
    .{ .name = "nvfp4", .claims = &claimsNvfp4, .validate = &validateMlx },
    .{ .name = "hadamard", .claims = &claimsHadamard, .validate = &validateHadamard },
    .{ .name = "exl3", .claims = &claimsExl3, .validate = &validateExl3 },
    .{ .name = "gguf", .claims = &claimsGguf, .validate = &validateGguf },
};

/// The built-in format serving `block_json`: the highest claim wins, two at that rank are ambiguous.
pub fn claimant(block_json: []const u8, config_json: []const u8) error{Ambiguous}!?usize {
    var got: [builtins.len]Claim = undefined;
    for (builtins, &got) |b, *c| c.* = b.claims(.of(block_json), .of(config_json));
    return common.pick(&got);
}

/// A scratch arena for one call: the C-ABI members take no allocator.
const Scratch = struct {
    arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator),
    fn a(self: *Scratch) std.mem.Allocator {
        return self.arena.allocator();
    }
};

/// An MLX-style block (`bits`, `group_size`, `mode`; no `quant_method`) whose mode is one of `modes`. Without
/// `bits` the pack loads dense.
fn mlxBlockOf(block_json: Str, comptime modes: []const []const u8) bool {
    var sc: Scratch = .{};
    defer sc.arena.deinit();
    const o = parseObject(sc.a(), block_json.slice()) orelse return false;
    if (o.get("quant_method") != null or uintField(o, "bits") == null) return false;
    const mode: []const u8 = if (o.get("mode")) |m| (if (m == .string) m.string else return false) else "affine";
    for (modes) |m| if (std.mem.eql(u8, mode, m)) return true;
    return false;
}

fn methodIs(block_json: Str, method: []const u8) bool {
    var sc: Scratch = .{};
    defer sc.arena.deinit();
    const o = parseObject(sc.a(), block_json.slice()) orelse return false;
    return eqlField(o, "quant_method", method);
}

fn claimsAffine(block_json: Str, _: Str) callconv(.c) Claim {
    return if (mlxBlockOf(block_json, &.{"affine"})) .by_type else .no;
}

fn claimsMxfp8(block_json: Str, _: Str) callconv(.c) Claim {
    return if (mlxBlockOf(block_json, &.{"mxfp8"})) .by_type else .no;
}

/// The e2m1 nibble modes (NVFP4 g16, MXFP4 g32) and NVIDIA ModelOpt packs.
fn claimsNvfp4(block_json: Str, _: Str) callconv(.c) Claim {
    return if (mlxBlockOf(block_json, &.{ "nvfp4", "mxfp4" }) or methodIs(block_json, "modelopt")) .by_type else .no;
}

/// Row-scaled e4m3 (Sushi's MiMo trunk, `quant_method: fp8`).
fn claimsFp8(block_json: Str, _: Str) callconv(.c) Claim {
    return if (methodIs(block_json, "fp8")) .by_type else .no;
}

/// An affine block beside Prism's `modules` list: the rotations outrank the block's own mode.
fn claimsHadamard(block_json: Str, config_json: Str) callconv(.c) Claim {
    if (claimsAffine(block_json, config_json) == .no) return .no;
    var sc: Scratch = .{};
    defer sc.arena.deinit();
    return if (hasModules(sc.a(), config_json.slice())) .by_marker else .no;
}

fn validateMlx(block_json: Str, _: Str, _: *const abi.PackDir) callconv(.c) Verdict {
    var sc: Scratch = .{};
    defer sc.arena.deinit();
    const o = parseObject(sc.a(), block_json.slice()) orelse return Verdict.refuse(.failed, "quantization block does not parse");
    if (eqlField(o, "quant_method", "modelopt")) return .ok;
    const d = declared(sc.a(), block_json.slice());
    if (!mlxMode(d.mode)) return Verdict.refuse(.unsupported, "quantization mode other than affine, nvfp4, mxfp4, mxfp8");
    var it = o.iterator();
    while (it.next()) |kv| {
        if (kv.value_ptr.* != .object) continue;
        const m = kv.value_ptr.object.get("mode") orelse continue;
        if (m == .string and !mlxMode(m.string)) return Verdict.refuse(.unsupported, "a per-weight quantization mode other than affine, nvfp4, mxfp4, mxfp8");
    }
    if (std.mem.eql(u8, d.mode, "affine") and d.bits != 0 and !affineBits(d.bits))
        return Verdict.refuse(.unsupported, "affine quantization outside 2, 3, 4, 5, 6, 8 bits");
    return .ok;
}

/// Sushi's adaptation reads e4m3 in 128x128 tiles.
fn validateFp8(block_json: Str, _: Str, _: *const abi.PackDir) callconv(.c) Verdict {
    var sc: Scratch = .{};
    defer sc.arena.deinit();
    const o = parseObject(sc.a(), block_json.slice()) orelse return Verdict.refuse(.failed, "quantization block does not parse");
    const ok = if (o.get("weight_block_size")) |v| v == .array and v.array.items.len == 2 and
        isUint(v.array.items[0], 128) and isUint(v.array.items[1], 128) else false;
    return if (ok) .ok else Verdict.refuse(.unsupported, "fp8: weight_block_size other than [128, 128]");
}

fn validateHadamard(block_json: Str, config_json: Str, dir: *const abi.PackDir) callconv(.c) Verdict {
    const v = validateMlx(block_json, config_json, dir);
    if (!v.isOk()) return v;
    var sc: Scratch = .{};
    defer sc.arena.deinit();
    return if (hadamardBlock(sc.a(), config_json.slice()) != null) .ok else Verdict.refuse(.unsupported, "hadamard: modules without one shared rotation block");
}

/// Sushi's `expert_quant`: EXL3 (MCG, MUL1) and its STQ codebook family (a named codebook table).
fn claimsExl3(block_json: Str, _: Str) callconv(.c) Claim {
    var sc: Scratch = .{};
    defer sc.arena.deinit();
    const o = parseObject(sc.a(), block_json.slice()) orelse return .no;
    return if (eqlField(o, "format", "exl3") or eqlField(o, "format", "stq")) .by_type else .no;
}

/// Sushi's own parse: a codebook, window or table this build does not decode, and a top-k past the reduce bank.
fn validateExl3(block_json: Str, config_json: Str, _: *const abi.PackDir) callconv(.c) Verdict {
    var sc: Scratch = .{};
    defer sc.arena.deinit();
    const root = expertQuantRoot(sc.a(), block_json.slice()) orelse return Verdict.refuse(.failed, "expert_quant block does not parse");
    _ = sushi_exl3.parseExpertQuant(root) catch |e| return Verdict.named(.unsupported, @errorName(e));
    const cfg = parseObject(sc.a(), config_json.slice()) orelse return .ok;
    const text = if (cfg.get("text_config")) |t| (if (t == .object) t.object else cfg) else cfg;
    const topk = uintField(cfg, "num_experts_per_tok") orelse uintField(text, "num_experts_per_tok") orelse return .ok;
    sushi_exl3.admitTopK(topk) catch |e| return Verdict.named(.unsupported, @errorName(e));
    return .ok;
}

/// `{"expert_quant": block}`: the root Sushi's parse reads.
pub fn expertQuantRoot(a: std.mem.Allocator, block_json: []const u8) ?std.json.ObjectMap {
    const wrapped = std.fmt.allocPrint(a, "{{\"expert_quant\":{s}}}", .{block_json}) catch return null;
    return parseObject(a, wrapped);
}

/// A GGUF file's rebuilt config (`mode: gguf`); its tensor types were vetted when the file was found servable.
fn claimsGguf(block_json: Str, _: Str) callconv(.c) Claim {
    return if (mlxBlockOf(block_json, &.{"gguf"})) .by_type else .no;
}

fn validateGguf(_: Str, _: Str, _: *const abi.PackDir) callconv(.c) Verdict {
    return .ok;
}

// ── JSON readers ──

fn parseObject(a: std.mem.Allocator, json: []const u8) ?std.json.ObjectMap {
    const v = std.json.parseFromSliceLeaky(Value, a, json, .{}) catch return null;
    return if (v == .object) v.object else null;
}

fn uintField(o: std.json.ObjectMap, key: []const u8) ?u32 {
    const v = o.get(key) orelse return null;
    if (v != .integer) return null;
    return std.math.cast(u32, v.integer);
}

fn eqlField(o: std.json.ObjectMap, key: []const u8, want: []const u8) bool {
    const v = o.get(key) orelse return false;
    return v == .string and std.mem.eql(u8, v.string, want);
}

fn isUint(v: Value, want: i64) bool {
    return v == .integer and v.integer == want;
}

// ── tests ──

const testing = std.testing;
const registry = @import("../../plugins/registry.zig");

fn expectClaimant(want: ?[]const u8, block_json: []const u8, config_json: []const u8) !void {
    const got = try claimant(block_json, config_json);
    const name: ?[]const u8 = if (got) |i| builtins[i].name else null;
    if (want) |w| {
        try testing.expect(name != null);
        try testing.expectEqualStrings(w, name.?);
    } else try testing.expectEqual(@as(?[]const u8, null), name);
}

// Bar: every block of every fixture config has at most one claimant, and the ones today's parser serves have theirs.
test "quant block: each fixture config's quantization block is claimed by one built-in format at most" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // Blocks today's parser ignores (bitsandbytes, compressed-tensors, `modelopt_fp4`) have no claimant.
    const served = [_]struct { file: []const u8, key: []const u8, format: []const u8 }{
        .{ .file = "nemotron_h_super_omni.json", .key = "quantization", .format = "affine" },
        .{ .file = "nemotron_h_super_omni.json", .key = "quantization_config", .format = "affine" },
        .{ .file = "glm5_next-mlx-community-4bit.json", .key = "quantization", .format = "affine" },
        .{ .file = "glm5_next-mlx-community-4bit.json", .key = "quantization_config", .format = "affine" },
        .{ .file = "glm5_next-nvidia-fp4-local.json", .key = "quantization_config", .format = "nvfp4" },
        .{ .file = "glm5_next-sushi-2.4bpw.json", .key = "quantization", .format = "affine" },
        .{ .file = "glm5_next-sushi-2.4bpw.json", .key = "expert_quant", .format = "exl3" },
        .{ .file = "bailing_hybrid-ling-3.0-flash-oq4e-mtp.json", .key = "quantization", .format = "affine" },
        .{ .file = "bailing_hybrid-ling-3.0-flash-oq4e-mtp.json", .key = "quantization_config", .format = "affine" },
        .{ .file = "bailing_hybrid-ling-3.0-tiny-4bit.json", .key = "quantization", .format = "affine" },
        .{ .file = "bailing_hybrid-ling-3.0-tiny-4bit.json", .key = "quantization_config", .format = "affine" },
        .{ .file = "qwen4_exp-iq-mlx-4.7bpw.json", .key = "quantization", .format = "affine" },
        .{ .file = "qwen4_exp-iq-mlx-4.7bpw.json", .key = "quantization_config", .format = "affine" },
        .{ .file = "qwen4_exp-nvidia-fp4.json", .key = "quantization_config", .format = "nvfp4" },
        .{ .file = "diffusiongemma-26B-A4B-it-4bit.json", .key = "quantization", .format = "affine" },
        .{ .file = "diffusiongemma-26B-A4B-it-4bit.json", .key = "quantization_config", .format = "affine" },
        .{ .file = "embeddinggemma-300m-8bit.json", .key = "quantization", .format = "affine" },
        .{ .file = "gemma-3-12b-it-4bit.json", .key = "quantization", .format = "affine" },
        .{ .file = "gemma-3-1b-it-4bit.json", .key = "quantization", .format = "affine" },
        .{ .file = "gemma-3-1b-it-4bit.json", .key = "quantization_config", .format = "affine" },
        .{ .file = "gemma-3-4b-it-4bit.json", .key = "quantization", .format = "affine" },
        .{ .file = "gemma-4-12b-it-4bit.json", .key = "quantization", .format = "affine" },
        .{ .file = "gemma-4-12b-it-4bit.json", .key = "quantization_config", .format = "affine" },
        .{ .file = "gemma-4-26b-a4b-it-4bit.json", .key = "quantization", .format = "affine" },
        .{ .file = "gemma-4-26b-a4b-it-4bit.json", .key = "quantization_config", .format = "affine" },
        .{ .file = "gemma-4-31b-it-4bit.json", .key = "quantization", .format = "affine" },
        .{ .file = "gemma-4-31b-it-4bit.json", .key = "quantization_config", .format = "affine" },
        .{ .file = "gemma-4-e4b-it-4bit.json", .key = "quantization", .format = "affine" },
        .{ .file = "gemma-4-e4b-it-4bit.json", .key = "quantization_config", .format = "affine" },
    };
    var seen: usize = 0;
    for (registry.model_configs) |f| {
        for (try blocksOf(a, f.bytes)) |b| {
            const got = claimant(b.json, f.bytes) catch {
                std.debug.print("{s}: block {s} has two claimants\n", .{ f.name, b.key });
                return error.TwoClaimants;
            };
            var want: ?[]const u8 = null;
            for (served) |s| if (std.mem.eql(u8, s.file, f.name) and std.mem.eql(u8, s.key, b.key)) {
                want = s.format;
                seen += 1;
            };
            const name: ?[]const u8 = if (got) |i| builtins[i].name else null;
            if ((want == null) != (name == null) or (want != null and !std.mem.eql(u8, want.?, name.?))) {
                std.debug.print("{s}: block {s} claimed by {?s}, want {?s}\n", .{ f.name, b.key, name, want });
                return error.WrongClaimant;
            }
        }
    }
    try testing.expectEqual(served.len, seen);
}

// Bar: each shipped pack shape names exactly its format; Prism's `modules` outranks the affine mode beside it.
test "quant block: every shipped block shape has one claimant" {
    const none = "{}";
    try expectClaimant("affine", "{\"group_size\":64,\"bits\":4}", none);
    try expectClaimant("affine", "{\"group_size\":64,\"bits\":3,\"mode\":\"affine\",\"model.embed_tokens\":{\"group_size\":64,\"bits\":8}}", none);
    try expectClaimant("mxfp8", "{\"group_size\":32,\"bits\":8,\"mode\":\"mxfp8\"}", none);
    try expectClaimant("nvfp4", "{\"group_size\":16,\"bits\":4,\"mode\":\"nvfp4\"}", none);
    try expectClaimant("nvfp4", "{\"group_size\":32,\"bits\":4,\"mode\":\"mxfp4\",\"model.embed_tokens\":{\"group_size\":32,\"bits\":8,\"mode\":\"mxfp8\"}}", none);
    try expectClaimant("nvfp4", "{\"quant_method\":\"modelopt\",\"quant_algo\":\"MIXED_PRECISION\"}", none);
    try expectClaimant("fp8", "{\"quant_method\":\"fp8\",\"weight_block_size\":[128,128]}", none);
    const prism = "{\"model_type\":\"prism_hadamard_qwen35\",\"modules\":[{\"path\":\"lm_head\",\"block\":1024}]}";
    try expectClaimant("hadamard", "{\"bits\":2,\"group_size\":128,\"mode\":\"affine\"}", prism);
    try expectClaimant("exl3", "{\"format\":\"exl3\",\"k\":2.5,\"codebook\":\"mcg\",\"window\":14}", none);
    try expectClaimant("exl3", "{\"format\":\"stq\",\"k\":2.5,\"codebook\":\"std1\",\"window\":15,\"codebook_table\":\"std1\",\"codebook_digest\":\"c5dba171\"}", none);
    try expectClaimant("gguf", "{\"group_size\":32,\"bits\":4,\"mode\":\"gguf\"}", none);
    try expectClaimant(null, "{\"quant_method\":\"bitsandbytes\",\"load_in_4bit\":true}", none);
    try expectClaimant(null, "{\"group_size\":64}", none);
    try expectClaimant(null, "{", none);
}

test "quant block: blocks are found under their keys; a config without one has none" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bs = try blocksOf(a, "{\"expert_quant\":{\"format\":\"exl3\"},\"quantization\":{\"bits\":4},\"quantization_config\":3}");
    try testing.expectEqual(@as(usize, 2), bs.len);
    try testing.expectEqualStrings("quantization", bs[0].key);
    try testing.expectEqualStrings("{\"bits\":4}", bs[0].json);
    try testing.expectEqualStrings("expert_quant", bs[1].key);
    try testing.expectEqual(@as(usize, 0), (try blocksOf(a, "{\"model_type\":\"llama\"}")).len);
    try testing.expectEqual(@as(usize, 0), (try blocksOf(a, "not json")).len);
}

// Bar: the refusals today's config parse makes, by name, and nothing more.
test "quant block: validation refuses what the parser refuses" {
    const dir: abi.PackDir = undefined;
    const t = struct {
        fn v(name: []const u8, block: []const u8, config: []const u8) Verdict {
            for (builtins) |b| if (std.mem.eql(u8, b.name, name)) return b.validate(.of(block), .of(config), &dir);
            unreachable;
        }
    };
    try testing.expect(t.v("affine", "{\"bits\":4,\"group_size\":64}", "{}").isOk());
    for ([_]u32{ 2, 3, 4, 5, 6, 8 }) |bits| {
        var buf: [64]u8 = undefined;
        try testing.expect(t.v("affine", try std.fmt.bufPrint(&buf, "{{\"bits\":{d}}}", .{bits}), "{}").isOk());
    }
    try testing.expectEqual(Verdict.Code.unsupported, t.v("affine", "{\"bits\":7}", "{}").code);
    try testing.expectEqual(Verdict.Code.unsupported, t.v("affine", "{\"bits\":4,\"mode\":\"int4\"}", "{}").code);
    try testing.expectEqual(Verdict.Code.unsupported, t.v("nvfp4", "{\"bits\":4,\"mode\":\"nvfp4\",\"x.experts\":{\"mode\":\"iq2_xxs\"}}", "{}").code);
    try testing.expect(t.v("nvfp4", "{\"quant_method\":\"modelopt\"}", "{}").isOk());
    try testing.expect(t.v("fp8", "{\"quant_method\":\"fp8\",\"weight_block_size\":[128,128]}", "{}").isOk());
    try testing.expectEqual(Verdict.Code.unsupported, t.v("fp8", "{\"quant_method\":\"fp8\",\"store_dtype\":\"mxfp4\"}", "{}").code);
    const prism = "{\"modules\":[{\"path\":\"a\",\"block\":1024},{\"path\":\"b\",\"block\":1024}]}";
    try testing.expect(t.v("hadamard", "{\"bits\":2,\"group_size\":128}", prism).isOk());
    try testing.expectEqual(Verdict.Code.unsupported, t.v("hadamard", "{\"bits\":2}", "{\"modules\":[{\"block\":1024},{\"block\":512}]}").code);
}

test "quant block: the declared defaults follow the block; ModelOpt is NVFP4 g16" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const d = declared(a, "{\"bits\":3,\"group_size\":32,\"mode\":\"affine\"}");
    try testing.expectEqual(@as(u32, 3), d.bits);
    try testing.expectEqual(@as(u32, 32), d.group_size);
    const m = declared(a, "{\"quant_method\":\"modelopt\"}");
    try testing.expectEqual(@as(u32, 16), m.group_size);
    try testing.expectEqualStrings("nvfp4", m.mode);
    try testing.expectEqual(@as(u32, 0), declared(a, "{}").bits);
    try testing.expectEqual(@as(?u32, 1024), hadamardBlock(a, "{\"modules\":[{\"block\":1024},{\"block\":1024}]}"));
    try testing.expectEqual(@as(?u32, null), hadamardBlock(a, "{\"modules\":[{\"block\":1024},{\"block\":512}]}"));
}
