//! Parity proof for the qwen4_exp plugin: on a tiny random pack, the table's forward (prefill, decode across the QSA
//! budget, verify, a later chunk, layer and stream captures, batched rows) is bit-equal to the legacy forward the
//! engine runs (`Transformer.forwardWith` / `forwardMoeBatchedDecode`), on the engine's slot caches and on the
//! model's own.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const model = @import("../../engines/mlx/model.zig");
const sdk = @import("../../sdk/root.zig");
const engine_cache_ops = @import("../../engines/mlx/kv/cache_ops.zig");
const batched_caps = @import("../../engines/mlx/batched/caps.zig");
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;
const ForwardCtx = @import("../../engines/mlx/forward_ctx.zig").ForwardCtx;
const CaptureLayers = @import("../../engines/mlx/forward_ctx.zig").CaptureLayers;
const KVCache = @import("../../engines/mlx/kv/cache.zig").KVCache;
const SSMCacheEntry = @import("../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const nextSsmMemberGen = @import("../../engines/mlx/ssm/group.zig").nextSsmMemberGen;
const ssmFreeQsaState = @import("qsa/history.zig").ssmFreeQsaState;
const q4 = @import("plugin.zig");
const fixture = @import("fixture.zig");

const testing = std.testing;
const Array = mlx.mlx_array;

/// A slot's engine caches: what `Slot.init` allocates for a GDN trunk.
const Caches = struct {
    kv: KVCache,
    ssm: []SSMCacheEntry,
    pos: u32 = 0,
    slot: engine_cache_ops.Slot = undefined,

    fn init(layers: u32) !Caches {
        const ssm = try testing.allocator.alloc(SSMCacheEntry, layers);
        const gen = nextSsmMemberGen();
        for (ssm) |*e| e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false, .member_gen = gen };
        return .{ .kv = try KVCache.init(testing.allocator, layers), .ssm = ssm };
    }

    fn deinit(c: *Caches, xfm: *Transformer) void {
        xfm.ssmGroupDrop(c.ssm);
        for (c.ssm) |*e| {
            _ = mlx.mlx_array_free(e.conv_state);
            _ = mlx.mlx_array_free(e.ssm_state);
            ssmFreeQsaState(e);
        }
        testing.allocator.free(c.ssm);
        c.kv.deinit();
    }

    fn handle(c: *Caches) abi.Handle {
        c.slot = .{ .kv = &c.kv, .ssm = c.ssm };
        return c.slot.handle();
    }
};

/// A pack on disk, loaded twice on the GPU stream: the legacy Transformer and the plugin's table.
const Twin = struct {
    tmp: testing.TmpDir,
    cfg: model.ModelConfig,
    weights: model.Weights,
    xfm: Transformer,
    parsed: abi.Parsed = .{},
    m: *anyopaque,

    fn open(files: []const sdk.fixture.File) !*Twin {
        const a = testing.allocator;
        const io = testing.io;
        const t = try a.create(Twin);
        errdefer a.destroy(t);
        t.tmp = testing.tmpDir(.{});
        errdefer t.tmp.cleanup();
        for (files) |f| try t.tmp.dir.writeFile(io, .{ .sub_path = f.name, .data = f.bytes });
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const dir = buf[0..try t.tmp.dir.realPath(io, &buf)];
        t.cfg = try model.parseConfigLegacy(io, a, dir);
        errdefer t.cfg.deinit(a);
        t.weights = try model.loadModelWeights(io, a, dir, &t.cfg, false);
        errdefer t.weights.deinit();
        model.resolveWeightPrefix(&t.cfg, &t.weights);
        t.xfm = try Transformer.init(io, a, t.cfg, &t.weights);
        errdefer t.xfm.deinit();
        q4.prepare(&t.xfm);
        const mem: sdk.fixture.MemDir = .{ .files = files };
        const pd = mem.packDir();
        try testing.expect(q4.table.parse(&pd, .of(files[0].bytes), .{}, null, false, &t.parsed).isOk());
        var why: abi.Verdict = .{};
        t.m = q4.table.load(&.{ .model_dir = .of(dir), .dir = &pd, .parsed = &t.parsed, .stream = .{ .ctx = mlx.gpuStream().ctx } }, &why) orelse {
            std.debug.print("load: {s}\n", .{why.why.slice()});
            return error.PluginLoad;
        };
        return t;
    }

    fn close(t: *Twin) void {
        q4.table.unload(t.m);
        q4.table.free_parsed(&t.parsed);
        t.xfm.deinit();
        t.weights.deinit();
        t.cfg.deinit(testing.allocator);
        t.tmp.cleanup();
        testing.allocator.destroy(t);
    }

    /// The plugin's trunk module: the engine slots it batched are dropped from its group.
    fn inner(t: *Twin) *Transformer {
        return &@as(*q4.Model, @ptrCast(@alignCast(t.m))).xfm;
    }

    fn ids(toks: []const u32) Array {
        var buf: [32]i32 = undefined;
        for (toks, 0..) |id, k| buf[k] = @intCast(id);
        return mlx.mlx_array_new_data(&buf, &[_]c_int{ 1, @intCast(toks.len) }, 2, .int32);
    }

    /// The legacy forward's logits `[S, V]` on `c`, as the engine calls it; the stream and layer 2 when asked.
    fn legacy(t: *Twin, c: *Caches, mode: abi.Mode, toks: []const u32, stream: ?*Array, layer2: ?*Array) !Array {
        const x = ids(toks);
        defer _ = mlx.mlx_array_free(x);
        var off: usize = c.pos;
        var cap_out = [_]Array{mlx.mlx_array_new()};
        defer _ = mlx.mlx_array_free(cap_out[0]);
        var cl: CaptureLayers = .{ .ids = &.{2}, .out = &cap_out };
        var ctx: ForwardCtx = .{ .cache = &c.kv, .moe_seq_offset = &off, .ssm_entries = c.ssm, .capture_hidden = null, .capture_hidden_all = stream, .capture_layers = if (layer2 != null) &cl else null, .vision_embeddings = null, .spec_verify = mode == .verify, .capture_ssm_seq = mode == .verify };
        const l = try t.xfm.forwardWith(&ctx, x);
        defer _ = mlx.mlx_array_free(l);
        if (layer2) |dst| _ = mlx.mlx_array_set(dst, cap_out[0]);
        c.pos += @intCast(toks.len);
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&o, l, &[_]c_int{ @intCast(toks.len), -1 }, 2, mlx.gpuStream()));
        return o;
    }

    /// The table's logits `[S, V]`: on the engine's caches `c`, or (null) the model's own slot 0.
    fn plugin(t: *Twin, c: ?*Caches, pos: u32, mode: abi.Mode, toks: []const u32, out: *abi.ForwardOut, capture: bool) !Array {
        const h = [_]abi.Handle{if (c) |cc| cc.handle() else null};
        const v = q4.table.forward(t.m, &.{
            .mode = mode,
            .logits = .all_rows,
            .want_hidden = capture,
            .rows = @intCast(toks.len),
            .tokens = .of(toks),
            .positions = .of(&.{pos}),
            .caches = if (c != null) .of(&h) else .{},
            .cache_ops = if (c != null) &engine_cache_ops.ops else null,
            .capture_layers = if (capture) .of(&.{2}) else .{},
        }, out);
        if (!v.isOk()) {
            std.debug.print("forward: {s}\n", .{v.why.slice()});
            return error.PluginForward;
        }
        if (c) |cc| cc.pos += @intCast(toks.len);
        return .{ .ctx = out.logits.ctx };
    }
};

const Step = struct { mode: abi.Mode, ids: []const u32, capture: bool = false };

/// Prefill below the QSA budget, decodes across it, a verify block, a later chunk past it.
const steps = [_]Step{
    .{ .mode = .prefill, .ids = &.{ 3, 17, 42, 9, 7, 25, 1 }, .capture = true },
    .{ .mode = .decode, .ids = &.{8} },
    .{ .mode = .decode, .ids = &.{55} },
    .{ .mode = .verify, .ids = &.{ 5, 61, 20 } },
    .{ .mode = .decode, .ids = &.{12} },
    .{ .mode = .prefill, .ids = &.{ 4, 33, 7, 40, 90, 101, 2, 64, 77 }, .capture = true },
    .{ .mode = .decode, .ids = &.{2} },
    .{ .mode = .verify, .ids = &.{ 6, 70 } },
};

fn expectSteps(precision: fixture.Precision, own: bool) !void {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin.open(try fixture.files(arena.allocator(), 5, precision));
    defer t.close();
    const layers = t.cfg.num_hidden_layers;
    var lc = try Caches.init(layers);
    defer lc.deinit(&t.xfm);
    var pc = try Caches.init(layers);
    defer pc.deinit(t.inner());
    for (steps, 0..) |st, k| {
        errdefer std.debug.print("qwen4_exp proof: step {d} ({s})\n", .{ k, @tagName(st.mode) });
        var stream = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(stream);
        var layer2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(layer2);
        const pos = lc.pos;
        const want = try t.legacy(&lc, st.mode, st.ids, if (st.capture) &stream else null, if (st.capture) &layer2 else null);
        defer _ = mlx.mlx_array_free(want);
        var captured = [_]abi.Array{.{}};
        var out: abi.ForwardOut = .{ .captured = &captured };
        const got = try t.plugin(if (own) null else &pc, pos, st.mode, st.ids, &out, st.capture);
        defer _ = mlx.mlx_array_free(got);
        try ops.expectBitEqual(got, want);
        if (st.capture) {
            const hs: Array = .{ .ctx = out.hidden.ctx };
            defer _ = mlx.mlx_array_free(hs);
            const l2: Array = .{ .ctx = captured[0].ctx };
            defer _ = mlx.mlx_array_free(l2);
            try ops.expectBitEqual(hs, stream);
            try ops.expectBitEqual(l2, layer2);
        }
    }
}

test "qwen4_exp plugin: bf16 pack, every step on the engine's slot caches equals the legacy forward bit for bit" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    try expectSteps(.bf16, false);
}

test "qwen4_exp plugin: 4-bit pack, every step on the engine's slot caches equals the legacy forward bit for bit" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    try expectSteps(.q4, false);
}

test "qwen4_exp plugin: on the model's own slot state every step equals the legacy forward bit for bit" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    try expectSteps(.q4, true);
}

test "qwen4_exp plugin: batched rows equal the legacy batched GDN tick per slot, and a cold group steps serially" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin.open(try fixture.files(arena.allocator(), 9, .q4));
    defer t.close();
    const layers = t.cfg.num_hidden_layers;
    var lc: [2]Caches = .{ try Caches.init(layers), try Caches.init(layers) };
    defer for (&lc) |*c| c.deinit(&t.xfm);
    var pc: [2]Caches = .{ try Caches.init(layers), try Caches.init(layers) };
    defer for (&pc) |*c| c.deinit(t.inner());
    const prompts = [_][]const u32{ &.{ 3, 17, 42, 9, 7, 25, 1, 50, 60, 70 }, &.{ 11, 4, 99 } };
    for (prompts, 0..) |p, i| {
        const want = try t.legacy(&lc[i], .prefill, p, null, null);
        _ = mlx.mlx_array_free(want);
        var out: abi.ForwardOut = .{};
        const got = try t.plugin(&pc[i], 0, .prefill, p, &out, false);
        _ = mlx.mlx_array_free(got);
    }
    const ticks = [_][2]u32{ .{ 8, 21 }, .{ 55, 3 }, .{ 13, 88 } };
    for (ticks) |toks| {
        var lctx: [2]ForwardCtx = undefined;
        var lptr: [2]*ForwardCtx = undefined;
        var offs: [2]usize = undefined;
        const pos = [_]u32{ lc[0].pos, lc[1].pos };
        for (0..2) |i| {
            offs[i] = pos[i];
            lctx[i] = .{ .cache = &lc[i].kv, .moe_seq_offset = &offs[i], .ssm_entries = lc[i].ssm, .capture_hidden = null, .vision_embeddings = null };
            lptr[i] = &lctx[i];
        }
        // Both slots' recurrent state is live: the engine's gate takes the batched tick, not the serial fallback.
        try testing.expect(batched_caps.batchedGdnReady(&t.xfm, &lptr));
        const want = try t.xfm.forwardMoeBatchedDecode(&toks, &lptr, &pos, null);
        defer {
            for (want) |w| _ = mlx.mlx_array_free(w);
            testing.allocator.free(want);
        }
        const h = [_]abi.Handle{ pc[0].handle(), pc[1].handle() };
        var out: abi.ForwardOut = .{};
        try testing.expect(q4.table.forward(t.m, &.{ .mode = .batched_rows, .rows = 1, .tokens = .of(&toks), .positions = .of(&pos), .caches = .of(&h), .cache_ops = &engine_cache_ops.ops }, &out).isOk());
        const got: Array = .{ .ctx = out.logits.ctx };
        defer _ = mlx.mlx_array_free(got);
        for (want, 0..) |w, i| {
            var row = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(row);
            try mlx.check(mlx.mlx_slice(&row, got, &[_]c_int{ @intCast(i), 0 }, 2, &[_]c_int{ @intCast(i + 1), @intCast(t.cfg.vocab_size) }, 2, &[_]c_int{ 1, 1 }, 2, mlx.gpuStream()));
            var w2 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(w2);
            try mlx.check(mlx.mlx_reshape(&w2, w, &[_]c_int{ 1, -1 }, 2, mlx.gpuStream()));
            try ops.expectBitEqual(row, w2);
        }
        for (0..2) |i| {
            lc[i].pos += 1;
            pc[i].pos += 1;
        }
    }
}

test "qwen4_exp plugin: a position that does not follow the slot's state is refused by name" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin.open(try fixture.files(arena.allocator(), 3, .bf16));
    defer t.close();
    var out: abi.ForwardOut = .{};
    const l = try t.plugin(null, 0, .prefill, &.{ 1, 2, 3 }, &out, false);
    _ = mlx.mlx_array_free(l);
    var bad: abi.ForwardOut = .{};
    const v = q4.table.forward(t.m, &.{ .mode = .decode, .rows = 1, .tokens = .of(&.{4}), .positions = .of(&.{5}) }, &bad);
    try testing.expect(!v.isOk() and bad.logits.ctx == null);
}
