//! The conformance suite: what every plugin table holds, run on each registered plugin and template in `zig build
//! test` (from the generated registry). MLX-free by construction for claims and parse (the build checks each
//! `config.zig`'s imports); `expectExactness` runs a plugin's declared exactness contracts on an engine serving it.

const std = @import("std");
const common = @import("../../abi/common.zig");
const model_abi = @import("../../abi/plugin.zig");
const media_abi = @import("../../abi/media.zig");
const quant_abi = @import("../../abi/quant.zig");
const engine_abi = @import("../../abi/engine.zig");
const exactness = @import("../../abi/exactness_test.zig");
const sdk_plugin = @import("../plugin.zig");
const fixture = @import("fixture.zig");
const mlx = @import("../../mlx/ffi.zig");

const testing = std.testing;
const Allocator = std.mem.Allocator;
const Str = common.Str;
const Verdict = common.Verdict;
const Array = common.Array;
const Claim = common.Claim;

/// Every template and registered plugin of the generated registry `R`, then one claimant per fixture pack.
pub fn expectRegistry(comptime R: type) !void {
    inline for (R.templates) |M| try expectPlugin(M);
    inline for (R.plugins) |M| try expectPlugin(M);
    try expectOneClaimant(&R.entries, &R.model_configs);
}

pub fn expectPlugin(comptime M: type) !void {
    switch (comptime M.plugin.kind) {
        .model => try expectModel(M),
        .media => try expectMedia(M),
        .quant => try expectQuant(M),
    }
}

fn fail(comptime M: type, comptime what: []const u8, why: Str) error{Conformance} {
    std.debug.print("conformance: plugin {s}: " ++ what ++ " ({s})\n", .{ M.plugin.name, why.slice() });
    return error.Conformance;
}

fn expectOk(comptime M: type, comptime what: []const u8, v: Verdict) !void {
    if (!v.isOk()) return fail(M, what, v.why);
}

fn expectHead(comptime M: type, comptime T: type, version: u32) !void {
    const t = &M.table;
    if (t.size != @sizeOf(T) or t.version != version) return fail(M, "table size or version is not this ABI's", .{});
    if (!std.mem.eql(u8, t.name.slice(), M.plugin.name)) return fail(M, "table name differs from its declaration", t.name);
}

/// A plugin's fixture staged on disk (MLX loads by path) and as a `PackDir` over the same files.
const Staged = struct {
    tmp: testing.TmpDir,
    path: []const u8,
    dir: common.PackDir,

    fn init(a: Allocator, fx: fixture.Fixture) !Staged {
        var tmp = testing.tmpDir(.{});
        errdefer tmp.cleanup();
        for (fx.files) |f| {
            if (std.fs.path.dirname(f.name)) |d| try tmp.dir.createDirPath(testing.io, d);
            try tmp.dir.writeFile(testing.io, .{ .sub_path = f.name, .data = f.bytes });
        }
        const buf = try a.alloc(u8, std.fs.max_path_bytes);
        const mem = try a.create(fixture.MemDir);
        mem.* = .{ .files = fx.files };
        return .{ .tmp = tmp, .path = buf[0..try tmp.dir.realPath(testing.io, buf)], .dir = mem.packDir() };
    }
    fn deinit(self: *Staged) void {
        self.tmp.cleanup();
    }
};

/// A model plugin's fixture, parsed (full) and loaded on the CPU stream. For plugin tests and the suite.
pub fn Pack(comptime M: type) type {
    return struct {
        staged: Staged,
        stream: mlx.mlx_stream,
        parsed: model_abi.Parsed = .{},
        model: *anyopaque = undefined,

        pub fn open(a: Allocator) !@This() {
            const fx = try M.conformanceFixture(a);
            const cfg = fx.file("config.json") orelse return error.FixtureWithoutConfig;
            var self: @This() = .{ .staged = try Staged.init(a, fx), .stream = mlx.mlx_default_cpu_stream_new() };
            errdefer {
                _ = mlx.mlx_stream_free(self.stream);
                self.staged.deinit();
            }
            try expectOk(M, "parse of its fixture", M.table.parse(&self.staged.dir, .of(cfg), .{}, null, false, &self.parsed));
            errdefer M.table.free_parsed(&self.parsed);
            var why: Verdict = .{};
            self.model = M.table.load(&.{ .model_dir = .of(self.staged.path), .dir = &self.staged.dir, .parsed = &self.parsed, .stream = .{ .ctx = self.stream.ctx } }, &why) orelse
                return fail(M, "load of its fixture", why.why);
            return self;
        }
        pub fn close(self: *@This()) void {
            M.table.unload(self.model);
            M.table.free_parsed(&self.parsed);
            _ = mlx.mlx_stream_free(self.stream);
            self.staged.deinit();
        }
    };
}

/// The values of `arr` as f32 (any float dtype); consumes the reference.
pub fn consumeF32(a: Allocator, arr: Array) ![]f32 {
    const x: mlx.mlx_array = .{ .ctx = arr.ctx };
    defer _ = mlx.mlx_array_free(x);
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, x, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f));
    const n = mlx.mlx_array_size(f);
    const data = mlx.mlx_array_data_float32(f) orelse return error.NoData;
    return a.dupe(f32, data[0..n]);
}

// ── model ──

fn expectModel(comptime M: type) !void {
    const t = &M.table;
    try expectHead(M, model_abi.ModelTable, model_abi.version);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const fx = try M.conformanceFixture(a);
    const cfg = fx.file("config.json") orelse return error.FixtureWithoutConfig;
    const mem: fixture.MemDir = .{ .files = fx.files };
    const dir = mem.packDir();
    const none: fixture.MemDir = .{ .files = &.{} };
    const empty = none.packDir();

    // Claims: its fixture and every declared type, never a stranger.
    if (t.claims(.of(try sdk_plugin.modelTypeOf(a, cfg)), .of(cfg), &dir) == .no) return fail(M, "does not claim its own fixture", .{});
    for (M.plugin.claims) |mt| if (t.claims(.of(mt), .of(cfg), &dir) == .no) return fail(M, "does not claim a declared model_type", .of(mt));
    if (t.claims(.of("conformance_stranger"), .of("{}"), &empty) != .no) return fail(M, "claims a stranger's pack", .{});

    // Stub parse (discovery): sized structs, a sane shell; a broken config is refused by name with nothing to free.
    var stub: model_abi.Parsed = .{};
    try expectOk(M, "stub parse of its fixture", t.parse(&dir, .of(cfg), .{}, null, true, &stub));
    defer t.free_parsed(&stub);
    try expectParsed(M, &stub);
    var bad: model_abi.Parsed = .{};
    const v = t.parse(&dir, .of("{"), .{}, null, true, &bad);
    if (v.isOk() or v.why.len == 0 or bad.private != null) return fail(M, "a broken config.json is not refused by name", .{});

    var pack = try Pack(M).open(a);
    defer pack.close();
    try expectParsed(M, &pack.parsed);

    // Cache layout: one sized entry per layer; it decides whether the bill may price KV.
    const n_layers = pack.parsed.shell.num_hidden_layers;
    const layers = try a.alloc(model_abi.LayerCache, n_layers + 1);
    const n = if (t.cache.layout) |f| f(&pack.parsed, null, layers.ptr, @intCast(layers.len)) else n_layers;
    if (n != n_layers) return fail(M, "cache layout length is not num_hidden_layers", .{});
    var attention: u32 = if (t.cache.layout == null) n else 0;
    if (t.cache.layout != null) for (layers[0..n]) |l| {
        if (l.size != @sizeOf(model_abi.LayerCache)) return fail(M, "a cache layout entry has another size", .{});
        if (l.kind == .full or l.kind == .sliding) attention += 1;
    };

    // Bill: sized, deterministic, no KV without an attention layer.
    var b1: model_abi.Bill = .{};
    var b2: model_abi.Bill = .{};
    const shape: model_abi.Shape = .{ .context = 4096 };
    t.bill(&pack.parsed, &shape, &b1);
    t.bill(&pack.parsed, &shape, &b2);
    if (b1.size != @sizeOf(model_abi.Bill)) return fail(M, "bill has another size", .{});
    inline for (@typeInfo(model_abi.Bill).@"struct".field_names) |f| if (@field(b1, f) != @field(b2, f)) return fail(M, "bill is not deterministic: " ++ f, .{});
    if (attention == 0 and b1.kv_per_token != 0) return fail(M, "bills KV with no attention layer", .{});

    inline for (@typeInfo(model_abi.Mode).@"enum".field_names) |f| try expectForward(M, a, &pack, @field(model_abi.Mode, f));
}

fn expectParsed(comptime M: type, p: *const model_abi.Parsed) !void {
    inline for (.{ "shell", "chat", "tokenizer", "cache", "spec", "placeholder_ids", "embed", "weights" }) |name| {
        if (@field(p, name).size != @sizeOf(@TypeOf(@field(p, name)))) return fail(M, "parsed " ++ name ++ " has another size", .{});
    }
    if (p.size != @sizeOf(model_abi.Parsed)) return fail(M, "parsed has another size", .{});
    const s = p.shell;
    if (s.vocab_size == 0 or s.hidden_size == 0 or s.num_hidden_layers == 0) return fail(M, "shell lacks its dimensions", .{});
    if (s.eos_count > model_abi.max_eos or p.tokenizer.extra_eos_count > model_abi.max_eos) return fail(M, "an eos count past its array", .{});
}

/// One forward per mode: a declared mode returns [R, vocab] finite logits, an undeclared one is refused `unsupported`.
fn expectForward(comptime M: type, a: Allocator, pack: *Pack(M), mode: model_abi.Mode) !void {
    const vocab = pack.parsed.shell.vocab_size;
    const slots: u32, const rows: u32, const logits: model_abi.Logits = switch (mode) {
        .prefill => .{ 1, 4, .last_row },
        .decode => .{ 1, 1, .last_row },
        .verify => .{ 1, 3, .all_rows },
        .batched_rows => .{ 2, 1, .last_row },
        .canvas => .{ 1, 4, .all_rows },
    };
    const tokens = try a.alloc(u32, slots * rows);
    for (tokens, 0..) |*tok, i| tok.* = @intCast((i * 7 + 1) % vocab);
    const slot_ids = try a.alloc(u32, slots);
    for (slot_ids, 0..) |*s, i| s.* = @intCast(i);
    const positions = try a.alloc(u32, slots);
    @memset(positions, 0);
    var out: model_abi.ForwardOut = .{};
    const v = M.table.forward(pack.model, &.{ .mode = mode, .logits = logits, .rows = rows, .tokens = .of(tokens), .slots = .of(slot_ids), .positions = .of(positions) }, &out);
    if (!model_abi.declaresMode(pack.parsed.caps, mode)) {
        if (v.code != .unsupported or out.logits.ctx != null) return fail(M, "an undeclared forward mode is not refused unsupported", .{});
        return;
    }
    try expectOk(M, "forward in a declared mode", v);
    if (out.hidden.ctx != null) _ = mlx.mlx_array_free(.{ .ctx = out.hidden.ctx });
    const x: mlx.mlx_array = .{ .ctx = out.logits.ctx };
    if (x.ctx == null) return fail(M, "forward returned no logits", .{});
    const want_rows = if (logits == .last_row) slots else slots * rows;
    const shape = mlx.mlx_array_shape(x);
    if (mlx.mlx_array_ndim(x) != 2 or shape[0] != want_rows or shape[1] != vocab) {
        _ = mlx.mlx_array_free(x);
        return fail(M, "logits are not [rows, vocab]", .of(@tagName(mode)));
    }
    for (try consumeF32(a, out.logits)) |l| if (!std.math.isFinite(l)) return fail(M, "non-finite logits", .of(@tagName(mode)));
}

/// The exactness contracts a plugin declares in its caps (drafted == plain, resumed == cold, concurrent == solo), on
/// an engine serving its pack. `expectModel` calls it once an engine serves plugins (P8: the MLX table over
/// `Pack.staged.path`, and the gate model env-gated).
pub fn expectExactness(engine: *const engine_abi.Engine, fx: exactness.Fixture, caps: model_abi.Caps) !void {
    try exactness.expect(engine, fx, .declaredBy(caps));
}

// ── media ──

/// Counts the host callbacks of a job; `leave` makes the client go away at the first boundary.
const Recorder = struct {
    reports: u32 = 0,
    boundaries: u32 = 0,
    leave: bool = false,

    fn progress(self: *Recorder) media_abi.Progress {
        return .{ .ctx = self, .report = report, .boundary = boundary };
    }
    fn report(ctx: ?*anyopaque, _: Str, _: u32, _: u32) callconv(.c) void {
        const self: *Recorder = @ptrCast(@alignCast(ctx.?));
        self.reports += 1;
    }
    fn boundary(ctx: ?*anyopaque) callconv(.c) bool {
        const self: *Recorder = @ptrCast(@alignCast(ctx.?));
        self.boundaries += 1;
        return self.leave;
    }
};

fn expectMedia(comptime M: type) !void {
    const t = &M.table;
    try expectHead(M, media_abi.MediaTable, media_abi.version);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const fx = try M.conformanceFixture(a);
    var staged = try Staged.init(a, fx);
    defer staged.deinit();
    const dir = &staged.dir;
    const none: fixture.MemDir = .{ .files = &.{} };
    const empty = none.packDir();

    const mt = if (fx.file("config.json")) |cfg| try sdk_plugin.modelTypeOf(a, cfg) else "";
    if (t.claims(.of(mt), dir) == .no) return fail(M, "does not claim its own fixture", .{});
    if (t.claims(.of("conformance_stranger"), &empty) != .no) return fail(M, "claims a stranger's pack", .{});
    if (t.required_file.len == 0 or !dir.has(t.required_file.slice())) return fail(M, "its fixture lacks the required file", t.required_file);

    var caps: media_abi.MediaCaps = .{};
    t.caps(dir, &caps);
    if (caps.size != @sizeOf(media_abi.MediaCaps) or caps.routes.size != @sizeOf(media_abi.Routes)) return fail(M, "caps have another size", .{});
    const machine: media_abi.Machine = .{};
    if (t.bill_load(dir, &machine) == 0) return fail(M, "bills a load at zero bytes", .{});
    if (t.bill_request(dir, &machine, .of(fx.body)) != t.bill_request(dir, &machine, .of(fx.body))) return fail(M, "request bill is not deterministic", .{});

    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var why: Verdict = .{};
    const e = t.load(&.{ .model_dir = .of(staged.path), .dir = dir, .stream = .{ .ctx = s.ctx } }, &why) orelse return fail(M, "load of its fixture", why.why);
    defer t.unload(e);
    if (t.set_loras) |f| try expectOk(M, "clearing the LoRA stack", f(e, .{}));

    const req: media_abi.Request = .{ .body = .of(fx.body) };
    var rec: Recorder = .{};
    try expectOk(M, "a job on its fixture", t.step(e, &req, &rec.progress()));
    if (rec.reports == 0 or rec.boundaries == 0) return fail(M, "a job never reports progress or yields at a boundary", .{});
    var bytes: Str = .{};
    try expectOk(M, "encode", t.encode(e, &bytes));
    if (bytes.len == 0) return fail(M, "encode returned no bytes", .{});

    var gone: Recorder = .{ .leave = true };
    const v = t.step(e, &req, &gone.progress());
    if (v.isOk() or v.why.len == 0) return fail(M, "a job whose client left is not stopped by name", .{});
}

// ── quant ──

fn expectQuant(comptime M: type) !void {
    const t = &M.table;
    try expectHead(M, quant_abi.QuantTable, quant_abi.version);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const fx = try M.conformanceFixture(arena.allocator());
    const mem: fixture.MemDir = .{ .files = fx.files };
    const dir = mem.packDir();
    const config = fx.file("config.json") orelse "{}";
    if (t.claims(.of(fx.block), .of(config)) == .no) return fail(M, "does not claim its own block", .of(fx.block));
    if (t.claims(.of("{}"), .of("{}")) != .no) return fail(M, "claims an empty block", .{});
    try expectOk(M, "validate of its own block", t.validate(.of(fx.block), .of(config), &dir));
}

// ── claims across plugins ──

/// Every pack has at most one claimant at the top rank (`by_marker` > `by_type`); two fail, naming them.
pub fn expectOneClaimant(list: []const sdk_plugin.Entry, packs: []const fixture.File) !void {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ranks = try a.alloc(Claim, list.len);
    for (packs) |pack| {
        const mem: fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = pack.bytes }} };
        const dir = mem.packDir();
        const got = sdk_plugin.claimsOf(list, try sdk_plugin.modelTypeOf(a, pack.bytes), pack.bytes, &dir, ranks);
        _ = common.pick(got) catch {
            const top = std.mem.max(u8, std.mem.sliceAsBytes(got));
            std.debug.print("claims: {s} is claimed at one rank by", .{pack.name});
            for (list, got) |e, c| if (@intFromEnum(c) == top) std.debug.print(" {s}", .{e.plugin.name});
            std.debug.print("\n", .{});
            return error.TwoClaimants;
        };
    }
}

// ── tests ──

fn fakeModel(comptime name: []const u8, comptime claims: *const fn (Str, Str, *const common.PackDir) callconv(.c) Claim) model_abi.ModelTable {
    return .{ .name = .of(name), .claims = claims, .parse = undefined, .free_parsed = undefined, .bill = undefined, .load = undefined, .unload = undefined, .forward = undefined, .cache = undefined, .weights = undefined };
}

fn claimsByMarker(_: Str, config_json: Str, _: *const common.PackDir) callconv(.c) Claim {
    return if (std.mem.indexOf(u8, config_json.slice(), "\"num_attention_heads_per_layer\"") != null) .by_marker else .no;
}

const laguna_pack: fixture.File = .{ .name = "laguna.json", .bytes = @embedFile("../../fixtures/model-configs/laguna.json") };
const gemma_pack: fixture.File = .{ .name = "gemma3-text.json", .bytes = @embedFile("../../fixtures/model-configs/gemma3-text.json") };

test "conformance claims: two type claims on one fixture pack fail; a marker claim outranks a type claim" {
    const by_type = comptime fakeModel("laguna", sdk_plugin.claimsByType(&.{"laguna"}));
    const twin = comptime fakeModel("laguna_twin", sdk_plugin.claimsByType(&.{"laguna"}));
    const marker = comptime fakeModel("flint_like", claimsByMarker);
    const gemma = comptime fakeModel("gemma3", sdk_plugin.claimsByType(&.{"gemma3_text"}));
    const packs = [_]fixture.File{ laguna_pack, gemma_pack };

    try expectOneClaimant(&.{ .{ .plugin = .{ .name = "laguna", .kind = .model }, .model = &by_type }, .{ .plugin = .{ .name = "gemma3", .kind = .model }, .model = &gemma } }, &packs);
    try testing.expectError(error.TwoClaimants, expectOneClaimant(&.{
        .{ .plugin = .{ .name = "laguna", .kind = .model }, .model = &by_type },
        .{ .plugin = .{ .name = "laguna_twin", .kind = .model }, .model = &twin },
    }, &packs));

    // The marker claimant wins laguna.json over the type claimant, and the claims pick names it.
    const list = [_]sdk_plugin.Entry{
        .{ .plugin = .{ .name = "laguna", .kind = .model }, .model = &by_type },
        .{ .plugin = .{ .name = "flint_like", .kind = .model }, .model = &marker },
    };
    try expectOneClaimant(&list, &packs);
    var ranks: [2]Claim = undefined;
    const mem: fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = laguna_pack.bytes }} };
    const dir = mem.packDir();
    try testing.expectEqual(@as(?usize, 1), try common.pick(sdk_plugin.claimsOf(&list, "laguna", laguna_pack.bytes, &dir, &ranks)));
}
