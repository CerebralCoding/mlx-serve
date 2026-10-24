//! The flip proof for both DeepSeek plugins: a tiny pack served by the plugin behind the driver's shell (the engine's
//! parse of a served type, the driver's map taken through the in-tree hatch), driven through `forwardWith` and the
//! generator as the scheduler drives them: chunked prefill, host and pipelined device decodes, a resumed prompt, image
//! rows at a chunk boundary, DSpark rounds. Every value is checked against the deleted legacy forward's, recorded in
//! the tapes in `fixtures/` (`engines/mlx/plugin/tape.zig`).

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const model = @import("../../engines/mlx/model.zig");
const driver = @import("../../engines/mlx/plugin/driver.zig");
const lazyForward = @import("../../engines/mlx/generate/step.zig").lazyForward;
const KVCache = @import("../../engines/mlx/kv/cache.zig").KVCache;
const ForwardCtx = @import("../../engines/mlx/forward_ctx.zig").ForwardCtx;
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;
const sdk = @import("../../sdk/root.zig");
const fixture = @import("fixture.zig");
const intree = @import("../../engines/mlx/plugin/intree.zig");
const prefix_cache = @import("../../engines/mlx/cache/prefix_cache.zig");
const verdict = @import("../../engines/mlx/batching/verdict.zig");
const env = @import("../../env.zig");
const Generator = @import("../../engines/mlx/generate/generator.zig").Generator;
const SamplingParams = @import("../../engines/mlx/sampling/params.zig").SamplingParams;
const Tokenizer = @import("../../sdk/tokenizer/tokenizer.zig").Tokenizer;
const Tape = @import("../../engines/mlx/plugin/tape.zig").Tape;

const testing = std.testing;
const Array = mlx.mlx_array;

/// One request's cache and context.
const Slot = struct {
    kv: KVCache,
    off: usize = 0,
    ctx: ForwardCtx = undefined,

    fn init(s: *Slot) !void {
        s.* = .{ .kv = try KVCache.init(testing.allocator, 1) };
        s.ctx = .{ .cache = &s.kv, .moe_seq_offset = &s.off, .ssm_entries = null, .capture_hidden = null, .vision_embeddings = null };
    }
};

/// The pack served by its plugin, its expected values played back from a tape.
const Served = struct {
    tmp: testing.TmpDir,
    cfg: model.ModelConfig,
    xfm: Transformer,
    slot: Slot,
    tape: Tape,
    /// The values since the last greedy run depend on its tokens: compared only on the recording's device.
    after_greedy: bool = false,

    fn open(table: *const abi.ModelTable, files: []const sdk.fixture.File, tape: []const u8) !*Served {
        const a = testing.allocator;
        const io = std.Io.Threaded.global_single_threaded.io();
        const p = try a.create(Served);
        errdefer a.destroy(p);
        p.after_greedy = false;
        p.tape = try Tape.player(tape);
        p.tmp = testing.tmpDir(.{});
        errdefer p.tmp.cleanup();
        for (files) |f| try p.tmp.dir.writeFile(io, .{ .sub_path = f.name, .data = f.bytes });
        var buf: [512]u8 = undefined;
        const dir = buf[0..try p.tmp.dir.realPath(io, &buf)];
        // The engine's parse of a served type: the legacy parse with the plugin's binding attached.
        p.cfg = try model.parseConfig(io, a, dir);
        errdefer p.cfg.deinit(a);
        if (p.cfg.plugin == null or p.cfg.plugin.?.table != table) return error.NotServed;
        p.xfm = try driver.load(io, a, &p.cfg, dir, mlx.gpuStream());
        try p.slot.init();
        return p;
    }

    fn close(p: *Served) void {
        p.slot.kv.deinit();
        p.xfm.deinit();
        p.cfg.deinit(testing.allocator);
        p.tmp.cleanup();
        testing.allocator.destroy(p);
    }

    /// Host integers, checked like an output.
    fn checkInts(p: *Served, g: *ops.Graph, got: []const u32, tokens: bool) !void {
        try p.tape.expectNext(try g.fromSlice(u32, got, &.{@intCast(got.len)}), tokens);
    }

    /// One chunk; its logits (owned by `g`), checked.
    fn step(p: *Served, g: *ops.Graph, ids: []const i32) !Array {
        const out = try g.adopt(try p.xfm.forwardWith(&p.slot.ctx, try g.fromSlice(i32, ids, &.{ 1, @intCast(ids.len) })));
        try p.tape.expectNext(out, p.after_greedy);
        return out;
    }

    /// Greedy decodes through the pipelined step (the token stays a lazy argmax on the GPU): tokens and logits.
    fn greedy(p: *Served, g: *ops.Graph, from: Array, n: usize) !void {
        p.after_greedy = true;
        var last = from;
        for (0..n) |_| {
            const sh = ops.shape(last);
            const next = try g.argmax(try g.reshape(try g.sliceAxis(last, 1, sh[1] - 1, sh[1]), &.{ 1, sh[2] }), -1, false);
            last = try g.adopt(try lazyForward(&p.xfm, &p.slot.ctx, next));
            const host = try g.astype(next, .uint32);
            try mlx.check(mlx.mlx_array_eval(host));
            try p.checkInts(g, mlx.mlx_array_data_uint32(host).?[0..1], true);
            try p.tape.expectNext(last, true);
        }
    }

    /// A new request: position 0, and the values no longer follow the greedy run.
    fn restart(p: *Served) void {
        p.slot.kv.step = 0;
        p.after_greedy = false;
    }
};

fn expectServed(comptime P: type, files: []const sdk.fixture.File, tape: []const u8) !void {
    const p = try Served.open(&P.table, files, tape);
    defer p.close();
    var g = ops.Graph.init(testing.allocator, mlx.gpuStream());
    defer g.deinit();
    // The checks the scheduler reads: single-flight, no prefix cache, serial, the module's own drafter.
    const sx = &p.xfm;
    try testing.expect(sx.ownsModuleDecodeState() and sx.moduleSpecWiring() and !sx.moduleStateSpecRollback() and !sx.supportsLayerCapture());
    try testing.expect(!prefix_cache.HotPrefixCache.shouldUse(&p.cfg, true) and !verdict.modelBatchable(&p.cfg));
    try testing.expectEqual(@as(usize, 0), sx.dsparkStages());
    // The hatch hands engine code the plugin's own module.
    const mod: ?*anyopaque = if (intree.dsv4(sx)) |x| x else if (intree.dsv41(sx)) |x| x else null;
    try testing.expectEqual(@as(?*anyopaque, P.moduleOf(intree.module(sx).?)), mod);
    _ = try p.step(&g, &.{ 3, 17, 42, 9, 7, 25, 1 });
    _ = try p.step(&g, &.{8});
    const l = try p.step(&g, &.{ 4, 33, 7, 40 });
    try p.greedy(&g, l, 5);
    _ = try p.step(&g, &.{2});
    // A new request at position 0 starts a fresh state.
    p.restart();
    _ = try p.step(&g, &.{ 9, 8, 7 });
    try p.tape.expectDone();
}

test "deepseek_v4 served: chunked prefill, host and pipelined decodes and a new request equal the legacy forward bit for bit" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try expectServed(@import("plugin.zig"), try fixture.v4Files(arena.allocator(), 11), @embedFile("fixtures/served.tape"));
}

test "deepseek_v41 served: chunked prefill, host and pipelined decodes and a new request equal the legacy forward bit for bit" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try expectServed(@import("../deepseek_v41/plugin.zig"), try fixture.v41Files(arena.allocator(), 7), @embedFile("../deepseek_v41/fixtures/served.tape"));
}

test "deepseek_v41 served: a resumed prompt (CacheHooks.resume_prompt) resumes through the driver where the legacy resume does, bit-equal" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const p = try Served.open(&@import("../deepseek_v41/plugin.zig").table, try fixture.v41Files(arena.allocator(), 7), @embedFile("../deepseek_v41/fixtures/resume.tape"));
    defer p.close();
    var g = ops.Graph.init(testing.allocator, mlx.gpuStream());
    defer g.deinit();
    const first = [_]i32{ 3, 17, 42, 9, 7, 25, 1, 60 };
    const second = first ++ [_]i32{ 11, 2, 90 };
    for ([_][]const i32{ &first, &second, &.{ 5, 6, 7 } }) |prompt| {
        var u: [16]u32 = undefined;
        for (prompt, 0..) |t, i| u[i] = @intCast(t);
        const n: u32 = @intCast(driver.resumePrompt(&p.xfm, u[0..prompt.len], null).?);
        try p.checkInts(&g, &.{n}, false);
        p.slot.kv.step = n;
        _ = try p.step(&g, prompt[n..]);
        try testing.expect(driver.stateBytes(&p.xfm) > 0);
    }
    try p.tape.expectDone();
}

test "deepseek_v41 served: image rows from Forward.media splice as the legacy seam splices them, across a chunk boundary" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const files = try fixture.v41Files(aa, 7);
    const json = try std.mem.replaceOwned(u8, aa, files[0].bytes, "\"eos_token_id\": 1,", "\"eos_token_id\": 1, \"image_token_id\": 5,");
    try testing.expect(!std.mem.eql(u8, json, files[0].bytes));
    const p = try Served.open(&@import("../deepseek_v41/plugin.zig").table, &.{ .{ .name = "config.json", .bytes = json }, files[1] }, @embedFile("../deepseek_v41/fixtures/image.tape"));
    defer p.close();
    var g = ops.Graph.init(testing.allocator, mlx.gpuStream());
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0xd541);
    var buf: [5 * 128]f32 = undefined;
    for (&buf) |*x| x.* = prng.random().floatNorm(f32) * 0.5;
    const rows = try g.astype(try g.adopt(mlx.mlx_array_new_data(&buf, &[_]c_int{ 1, 5, 128 }, 3, .float32)), .bfloat16);
    const chunks = [_]struct { ids: []const i32, offset: usize }{
        .{ .ids = &.{ 3, 17, 5, 5, 5 }, .offset = 0 },
        .{ .ids = &.{ 5, 5, 7, 8 }, .offset = 3 },
        .{ .ids = &.{9}, .offset = 5 },
    };
    // Without rows the image token is text: the same first chunk, to show the rows move the logits.
    const text = try p.step(&g, chunks[0].ids);
    p.restart();
    p.slot.ctx.vision_embeddings = rows;
    for (chunks, 0..) |ch, i| {
        p.slot.ctx.vision_splice_offset = ch.offset;
        const got = try p.step(&g, ch.ids);
        if (i == 0) try testing.expect(!try sameValues(&g, got, text));
    }
    try p.tape.expectDone();
}

fn sameValues(g: *ops.Graph, x: Array, y: Array) !bool {
    const fx = try g.contiguous(try g.astype(x, .float32));
    const fy = try g.contiguous(try g.astype(y, .float32));
    try ops.eval(fx);
    try ops.eval(fy);
    const n = mlx.mlx_array_size(fx);
    return std.mem.eql(f32, mlx.mlx_array_data_float32(fx).?[0..n], mlx.mlx_array_data_float32(fy).?[0..n]);
}

/// DSpark rounds through the generator (the fixture's peaked head makes drafts agree: rounds accept none, some and
/// all): each round's tokens, accepted drafts, next token and position equal the legacy shell's (the tape); at least
/// `min_accepted` drafts accepted over the run.
fn expectDspark(comptime P: type, files: []const sdk.fixture.File, sampling: SamplingParams, min_accepted: u64, tape: []const u8) !void {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const ds = env.override(.MLX_SERVE_DSV4_DSPARK, "1");
    defer ds.restore();
    const p = try Served.open(&P.table, files, tape);
    defer p.close();
    var g = ops.Graph.init(testing.allocator, mlx.gpuStream());
    defer g.deinit();
    try testing.expectEqual(@as(usize, 2), p.xfm.dsparkStages());
    var tok: Tokenizer = undefined;
    const prompt = [_]u32{ 3, 17, 42, 9, 7, 25, 1, 60, 11, 2, 30, 31 };
    var gen = try Generator.initWithOptions(io, a, &p.xfm, &tok, &prompt, 48, sampling, &.{}, .{
        .pld_enabled = true,
        .mtp_enabled = true,
        .skip_lazy_preforward = true,
        .lookup_prompt = &prompt,
    });
    defer gen.deinit(a);
    try testing.expect(gen.dspark_enabled);
    // Each round as `[tokens..., accepted drafts, next token, position]`.
    var rec: [16]u32 = undefined;
    for (0..16) |_| {
        const r = (try gen.nextDspark(a)) orelse return error.RoundsEnded;
        defer a.free(r.tokens);
        @memcpy(rec[0..r.tokens.len], r.tokens);
        rec[r.tokens.len..][0..3].* = .{ r.accepted_tokens, gen.next_token_id, @intCast(gen.ctx.cache.step) };
        try p.checkInts(&g, rec[0 .. r.tokens.len + 3], true);
    }
    try testing.expect(gen.dspark_attempted >= 4 and gen.dspark_accepted_tokens >= min_accepted);
    try p.tape.expectDone();
}

test "deepseek_v4 served: DSpark rounds (greedy, then seeded sampling) equal the legacy shell's" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v4DsparkFiles(arena.allocator(), 11);
    try expectDspark(@import("plugin.zig"), files, .{ .temperature = 0 }, 4, @embedFile("fixtures/dspark-greedy.tape"));
    try expectDspark(@import("plugin.zig"), files, .{ .temperature = 0.8, .top_p = 0.95, .seed = 0xd5 }, 4, @embedFile("fixtures/dspark-sampled.tape"));
}

test "deepseek_v41 served: DSpark rounds (greedy, then seeded sampling) equal the legacy shell's" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41DsparkFiles(arena.allocator(), 7);
    try expectDspark(@import("../deepseek_v41/plugin.zig"), files, .{ .temperature = 0 }, 4, @embedFile("../deepseek_v41/fixtures/dspark-greedy.tape"));
    try expectDspark(@import("../deepseek_v41/plugin.zig"), files, .{ .temperature = 0.8, .top_p = 0.95, .seed = 0xd5 }, 4, @embedFile("../deepseek_v41/fixtures/dspark-sampled.tape"));
}

test "deepseek_v4 served: mlx-community's withdrawn preview pack is refused by the legacy parse and, at load, by name" {
    const preview = @embedFile("fixtures/real/mlx-community_DeepSeek-V4-Flash-4bit.json");
    try testing.expectError(error.UnsupportedDsv4Config, model.parseConfigFromJson(testing.allocator, preview));
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = preview });
    var buf: [512]u8 = undefined;
    const dir = buf[0..try tmp.dir.realPath(io, &buf)];
    try testing.expectError(error.PluginDeclinedNoLegacy, model.parseConfig(io, testing.allocator, dir));
}
