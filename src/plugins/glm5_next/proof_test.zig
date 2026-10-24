//! Parity proof for the KDA-hybrid plugins: on a tiny random pack, the table's forward over the engine's own cache
//! adapter (`engines/mlx/kv/cache_ops.zig`) is bit-equal to the legacy forward at every step: logits of every row,
//! captured layers, each KDA layer's recurrent state and a verify's per-row capture. Shared by glm5_next and
//! bailing_hybrid (`expectSteps`); GLM's batched rows, MTP head and media rows too. The legacy values replay from
//! `fixtures/*.tape`; with `MLX_SERVE_PROOF_RECORD` set they are recorded from the legacy forward instead.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const quant_abi = @import("../../abi/quant.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const model = @import("../../engines/mlx/model.zig");
const cache_ops = @import("../../engines/mlx/kv/cache_ops.zig");
const tape_mod = @import("../../engines/mlx/plugin/tape.zig");
const legacy_fwd = @import("forward.zig");
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;
const KVCache = @import("../../engines/mlx/kv/cache.zig").KVCache;
const SSMCacheEntry = @import("../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const ForwardCtx = @import("../../engines/mlx/forward_ctx.zig").ForwardCtx;
const CaptureLayers = @import("../../engines/mlx/forward_ctx.zig").CaptureLayers;
const Resident = @import("../../sdk/weights/resident.zig").Resident;
const sdk = @import("../../sdk/root.zig");
const fixture = @import("fixture.zig");
const glm_mtp = @import("mtp.zig");
const parity = @import("../../sdk/testing/parity.zig");
const glm = @import("plugin.zig");
const ling = @import("../bailing_hybrid/plugin.zig");
const fwd = @import("table_forward.zig");

const testing = std.testing;
const Array = mlx.mlx_array;
const Tape = tape_mod.Tape;

/// One slot's engine cache: the KV cache, a recurrent state per layer and the legacy context over them.
const Slot = struct {
    kv: KVCache,
    ssm: []SSMCacheEntry,
    off: usize = 0,
    ctx: ForwardCtx = undefined,
    adapter: cache_ops.Slot = undefined,

    fn init(s: *Slot, layers: u32) !void {
        s.* = .{ .kv = try KVCache.init(testing.allocator, layers), .ssm = try testing.allocator.alloc(SSMCacheEntry, layers) };
        for (s.ssm) |*e| e.* = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false };
        s.ctx = .{ .cache = &s.kv, .moe_seq_offset = &s.off, .ssm_entries = s.ssm, .capture_hidden = null, .vision_embeddings = null };
        s.adapter = .{ .kv = &s.kv, .ssm = s.ssm };
    }

    fn deinit(s: *Slot) void {
        for (s.ssm) |*e| for ([_]Array{ e.conv_state, e.ssm_state, e.spec_conv_input, e.spec_state_seq }) |a| if (a.ctx != null) {
            _ = mlx.mlx_array_free(a);
        };
        testing.allocator.free(s.ssm);
        s.kv.deinit();
    }
};

/// The legacy load of the same pack, present only while recording.
const Legacy = struct {
    cfg: model.ModelConfig,
    weights: model.Weights,
    xfm: Transformer,

    fn open(dir: []const u8) !*Legacy {
        const a = testing.allocator;
        const io = std.Io.Threaded.global_single_threaded.io();
        const l = try a.create(Legacy);
        errdefer a.destroy(l);
        l.cfg = try model.parseConfigLegacy(io, a, dir);
        errdefer l.cfg.deinit(a);
        l.weights = try model.loadModelWeights(io, a, dir, &l.cfg, false);
        errdefer l.weights.deinit();
        l.xfm = try Transformer.init(io, a, l.cfg, &l.weights);
        l.xfm.compileMoeRouting();
        l.xfm.compileGdnGate();
        return l;
    }

    fn close(l: *Legacy) void {
        l.xfm.deinit();
        l.weights.deinit();
        l.cfg.deinit(testing.allocator);
        testing.allocator.destroy(l);
    }
};

/// A pack on disk served by plugin `P` over the engine's resident source, checked against the tape `tape` (a path
/// under the repo root; none for a proof comparing the plugin with itself) or, recording, the legacy load.
fn Twin(comptime P: type) type {
    return struct {
        tmp: testing.TmpDir,
        cfg: model.ModelConfig,
        res: Resident,
        src: quant_abi.Source = undefined,
        parsed: abi.Parsed = .{},
        m: *anyopaque,
        tape: Tape = .{},
        lg: ?*Legacy = null,
        s: mlx.mlx_stream,

        fn open(files: []const sdk.fixture.File, comptime tape: ?[]const u8) !*@This() {
            const a = testing.allocator;
            const io = std.Io.Threaded.global_single_threaded.io();
            const t = try a.create(@This());
            errdefer a.destroy(t);
            t.* = .{ .tmp = testing.tmpDir(.{}), .cfg = undefined, .res = undefined, .m = undefined, .s = mlx.gpuStream() };
            errdefer t.tmp.cleanup();
            for (files) |f| try t.tmp.dir.writeFile(io, .{ .sub_path = f.name, .data = f.bytes });
            var buf: [512]u8 = undefined;
            const dir = buf[0..try t.tmp.dir.realPath(io, &buf)];
            t.cfg = try model.parseConfigLegacy(io, a, dir);
            errdefer t.cfg.deinit(a);
            t.res = try Resident.open(io, a, dir, &t.cfg, false);
            errdefer t.res.deinit();
            t.src = t.res.source();
            if (tape) |rel| {
                if (tape_mod.recordRoot()) |root| {
                    t.lg = try Legacy.open(dir);
                    t.tape = try Tape.recorder(root, rel);
                } else t.tape = try Tape.player(@embedFile("../" ++ rel["src/plugins/".len..]));
                t.tape.compact_over = tape_compact_over;
            }
            errdefer if (t.lg) |l| l.close();
            const mem: sdk.fixture.MemDir = .{ .files = files };
            const pd = mem.packDir();
            try testing.expect(P.table.parse(&pd, .of(files[0].bytes), .{}, null, false, &t.parsed).isOk());
            errdefer P.table.free_parsed(&t.parsed);
            var why: abi.Verdict = .{};
            t.m = P.table.load(&.{ .model_dir = .of(dir), .dir = &pd, .parsed = &t.parsed, .stream = .{ .ctx = t.s.ctx }, .weights = &t.src }, &why) orelse {
                std.debug.print("load: {s}\n", .{why.why.slice()});
                return error.PluginLoad;
            };
            return t;
        }

        fn close(t: *@This()) void {
            t.tape.finish() catch |e| std.debug.panic("proof tape: {t}", .{e});
            P.table.unload(t.m);
            P.table.free_parsed(&t.parsed);
            if (t.lg) |l| l.close();
            t.res.deinit();
            t.cfg.deinit(testing.allocator);
            t.tmp.cleanup();
            testing.allocator.destroy(t);
        }

        fn layers(t: *const @This()) u32 {
            return t.cfg.num_hidden_layers;
        }

        /// `got` (the plugin's) against `want` (the legacy's, recorded) or the tape.
        fn check(t: *@This(), got: Array, want: ?Array) !void {
            if (want) |w| {
                try ops.expectBitEqual(got, w);
                if (t.tape.recording()) try t.tape.put(w);
            } else try t.tape.expectNext(got, false);
        }

        /// The plugin's `[S, V]` logits and two captured layers for one step of slot `s`.
        fn plugin(t: *@This(), s: *Slot, mode: abi.Mode, ids: []const u32, caps: *[2]Array) !Array {
            var captured: [2]abi.Array = .{ .{}, .{} };
            var out: abi.ForwardOut = .{ .captured = &captured };
            const h = [_]abi.Handle{s.adapter.handle()};
            const pos = [_]u32{@intCast(s.off)};
            const v = P.table.forward(t.m, &.{ .mode = mode, .logits = .all_rows, .rows = @intCast(ids.len), .tokens = .of(ids), .positions = .of(&pos), .caches = .of(&h), .cache_ops = &cache_ops.ops, .capture_layers = .of(&capture_ids) }, &out);
            if (!v.isOk()) {
                std.debug.print("forward: {s}\n", .{v.why.slice()});
                return error.PluginForward;
            }
            s.off += ids.len;
            caps.* = .{ .{ .ctx = captured[0].ctx }, .{ .ctx = captured[1].ctx } };
            const l: Array = .{ .ctx = out.logits.ctx };
            try settle(s, &.{ l, caps[0], caps[1] });
            return l;
        }

        /// The legacy forward's `[S, V]` logits and the same captured layers (none when `caps` is null: a capture
        /// materializes the deferred hyper-connection expand, so both sides capture alike).
        fn legacy(t: *@This(), s: *Slot, verify: bool, ids: []const u32, caps: ?*[2]Array) !Array {
            const lx = &t.lg.?.xfm;
            const arr = try idsArray(ids);
            defer _ = mlx.mlx_array_free(arr);
            var cl: CaptureLayers = undefined;
            if (caps) |cp| {
                cp.* = .{ mlx.mlx_array_new(), mlx.mlx_array_new() };
                cl = .{ .ids = &capture_ids, .out = cp };
                s.ctx.capture_layers = &cl;
            }
            s.ctx.capture_ssm_seq = verify;
            s.ctx.spec_verify = verify;
            defer {
                s.ctx.capture_layers = null;
                s.ctx.capture_ssm_seq = false;
                s.ctx.spec_verify = false;
            }
            const l = try lx.forwardWith(&s.ctx, arr);
            defer _ = mlx.mlx_array_free(l);
            var o = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&o, l, &[_]c_int{ @intCast(ids.len), @intCast(t.cfg.vocab_size) }, 2, t.s));
            try settle(s, if (caps) |cp| &.{ o, cp[0], cp[1] } else &.{o});
            return o;
        }

        /// Every KDA layer's state of `ps` against `ls` (the legacy slot, recording) or the tape: which states are
        /// set, then each set one, and a verify's per-row capture up to its last row (which the verify kernel may
        /// leave unwritten: the final state is its own output).
        fn expectStates(t: *@This(), ls: ?*const Slot, ps: *const Slot) !void {
            const flags_p = try stateFlags(ps);
            defer _ = mlx.mlx_array_free(flags_p);
            const flags_l: ?Array = if (ls) |l| try stateFlags(l) else null;
            defer if (flags_l) |f| {
                _ = mlx.mlx_array_free(f);
            };
            try t.check(flags_p, flags_l);
            for (ps.ssm, 0..) |y, i| {
                const x: ?SSMCacheEntry = if (ls) |l| l.ssm[i] else null;
                for ([_]Array{ y.conv_state, y.ssm_state, y.spec_conv_input }, 0..) |got, k| if (got.ctx != null) {
                    const want: ?Array = if (x) |e| ([_]Array{ e.conv_state, e.ssm_state, e.spec_conv_input })[k] else null;
                    try t.check(got, want);
                };
                if (y.spec_state_seq.ctx == null or mlx.getShape(y.spec_state_seq)[0] < 2) continue;
                const got = try leadingRows(y.spec_state_seq);
                defer _ = mlx.mlx_array_free(got);
                const want: ?Array = if (x) |e| try leadingRows(e.spec_state_seq) else null;
                defer if (want) |w| {
                    _ = mlx.mlx_array_free(w);
                };
                try t.check(got, want);
            }
        }
    };
}

/// Per layer: initialized, then which of conv, ssm, conv input and state sequence are set, as int32 `[layers, 5]`.
fn stateFlags(s: *const Slot) !Array {
    var buf: [64 * 5]i32 = undefined;
    for (s.ssm, 0..) |e, i| {
        buf[i * 5] = @intFromBool(e.initialized);
        for ([_]Array{ e.conv_state, e.ssm_state, e.spec_conv_input, e.spec_state_seq }, 1..) |a, k| buf[i * 5 + k] = @intFromBool(a.ctx != null);
    }
    return mlx.mlx_array_new_data(&buf, &[_]c_int{ @intCast(s.ssm.len), 5 }, 2, .int32);
}

/// Every row of a state sequence `[T, ...]` but the last.
fn leadingRows(seq: Array) !Array {
    const sh = mlx.getShape(seq);
    var hi: [5]c_int = undefined;
    for (sh, 0..) |v, i| hi[i] = v;
    hi[0] -= 1;
    var rows = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&rows, seq, &(@as([5]c_int, @splat(0))), sh.len, &hi, sh.len, &(@as([5]c_int, @splat(1))), sh.len, mlx.gpuStream()));
    return rows;
}

const capture_ids = [_]u32{ 1, 3 };

/// Every array past 1 KiB (the logits, captures and recurrent states) is kept as a digest and a sample (`Tape.compact_over`).
pub const tape_compact_over = 1 << 10;

/// Evaluates a forward's outputs and the slot's recurrent states before anything else is built.
fn settle(s: *const Slot, outs: []const Array) !void {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (outs) |a| if (a.ctx != null) {
        _ = mlx.mlx_vector_array_append_value(vec, a);
    };
    for (s.ssm) |e| for ([_]Array{ e.conv_state, e.ssm_state, e.spec_conv_input, e.spec_state_seq }) |a| if (a.ctx != null) {
        _ = mlx.mlx_vector_array_append_value(vec, a);
    };
    try mlx.check(mlx.mlx_eval(vec));
}

fn idsArray(ids: []const u32) !Array {
    var i32s: [64]i32 = undefined;
    for (ids, 0..) |id, k| i32s[k] = @intCast(id);
    return mlx.mlx_array_new_data(&i32s, &[_]c_int{ 1, @intCast(ids.len) }, 2, .int32);
}

fn freeAll(arrs: []const Array) void {
    for (arrs) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a);
    };
}

const Step = struct { mode: abi.Mode, ids: []const u32 };

const long_prompt = [_]u32{ 3, 17, 42, 9, 7, 25, 1, 88, 140, 61, 5, 230, 99, 12, 400, 33, 77, 2, 64, 301, 18, 45 };

/// A long prompt (GLM: dense causal rows on lifted keys, then indexed rows), decodes, verify blocks past the
/// indexer's budget, a later chunk wide enough for the NAX sparse kernel, and a short prompt on its own slot.
const runs = [_][]const Step{
    &.{
        .{ .mode = .prefill, .ids = &long_prompt },
        .{ .mode = .decode, .ids = &.{8} },
        .{ .mode = .verify, .ids = &.{ 5, 61, 20 } },
        .{ .mode = .decode, .ids = &.{12} },
        .{ .mode = .prefill, .ids = long_prompt[0..12] },
        .{ .mode = .decode, .ids = &.{2} },
        .{ .mode = .verify, .ids = &.{ 9, 33, 77, 140, 2 } },
    },
    &.{
        .{ .mode = .prefill, .ids = &.{ 3, 17, 42, 9, 7 } },
        .{ .mode = .decode, .ids = &.{55} },
        .{ .mode = .verify, .ids = &.{ 6, 7 } },
        .{ .mode = .decode, .ids = &.{400} },
        .{ .mode = .decode, .ids = &.{401} },
        .{ .mode = .decode, .ids = &.{402} },
        .{ .mode = .decode, .ids = &.{403} },
        .{ .mode = .verify, .ids = &.{ 1, 2, 3 } },
    },
};

fn expectSteps(comptime P: type, t: *Twin(P)) !void {
    for (runs) |run| {
        var ls: Slot = undefined;
        try ls.init(t.layers());
        defer ls.deinit();
        var ps: Slot = undefined;
        try ps.init(t.layers());
        defer ps.deinit();
        for (run) |st| {
            var lc: [2]Array = .{ .{ .ctx = null }, .{ .ctx = null } };
            const want: ?Array = if (t.lg != null) try t.legacy(&ls, st.mode == .verify, st.ids, &lc) else null;
            defer freeAll(&.{ want orelse .{ .ctx = null }, lc[0], lc[1] });
            var pc: [2]Array = undefined;
            const got = try t.plugin(&ps, st.mode, st.ids, &pc);
            defer freeAll(&.{ got, pc[0], pc[1] });
            try t.check(got, want);
            for (pc, lc) |g, w| try t.check(g, if (want != null) w else null);
            try t.expectStates(if (t.lg != null) &ls else null, &ps);
        }
    }
}

test "glm5_next plugin: prefill, decode, verify and later chunks equal the legacy forward bit for bit" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin(glm).open(try fixture.glmFiles(arena.allocator(), 7), "src/plugins/glm5_next/fixtures/steps.tape");
    defer t.close();
    try expectSteps(glm, t);
    try t.tape.expectDone();
}

test "bailing_hybrid plugin: prefill, decode, verify and later chunks equal the legacy forward bit for bit" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin(ling).open(try fixture.lingFiles(arena.allocator(), 11), "src/plugins/bailing_hybrid/fixtures/steps.tape");
    defer t.close();
    try expectSteps(ling, t);
    try t.tape.expectDone();
}

// Two MoE row groups: a tick of two groups is two trunk forwards in one evaluation.
test "glm5_next plugin: batched rows equal the legacy batched decode bit for bit" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin(glm).open(try fixture.glmFiles(arena.allocator(), 7), "src/plugins/glm5_next/fixtures/batched.tape");
    defer t.close();
    try expectBatched(t);
    try t.tape.expectDone();
}

// Bar: a Sushi pack (EXL3 routed experts through `sdk/quant/exl3/format.zig`) runs as the legacy forward runs it,
// per slot and batched.
test "glm5_next plugin: a Sushi pack's EXL3 experts equal the legacy forward bit for bit" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin(glm).open(try fixture.glmSushiFiles(arena.allocator(), 7), "src/plugins/glm5_next/fixtures/sushi.tape");
    defer t.close();
    try testing.expect(t.cfg.exl3 != null);
    try expectSteps(glm, t);
    try expectBatched(t);
    try t.tape.expectDone();
}

/// The plugin's batched tick over `ps` (`toks`, one next token per slot): logits `[N, V]`.
fn pluginTick(t: *Twin(glm), ps: []Slot, toks: []const u32) !Array {
    var handles: [16]abi.Handle = undefined;
    var positions: [16]u32 = undefined;
    for (ps, 0..) |*p, i| {
        handles[i] = p.adapter.handle();
        positions[i] = @intCast(p.off);
    }
    var out: abi.ForwardOut = .{};
    try testing.expect(glm.table.forward(t.m, &.{ .mode = .batched_rows, .rows = 1, .tokens = .of(toks), .positions = .of(positions[0..ps.len]), .caches = .of(handles[0..ps.len]), .cache_ops = &cache_ops.ops }, &out).isOk());
    return .{ .ctx = out.logits.ctx };
}

fn expectBatched(t: *Twin(glm)) !void {
    const N = fwd.MOE_DECODE_BATCH_MAX + 2;
    var ls: [N]Slot = undefined;
    var ps: [N]Slot = undefined;
    var made: usize = 0;
    defer for (ls[0..made], ps[0..made]) |*l, *p| {
        l.deinit();
        p.deinit();
    };
    const recording = t.lg != null;
    for (0..N) |i| {
        try ls[i].init(t.layers());
        ps[i].init(t.layers()) catch |e| {
            ls[i].deinit();
            return e;
        };
        made += 1;
        // Prompts of different lengths, so the slots attend over different histories.
        const prompt = long_prompt[0 .. 3 + i];
        var c: [2]Array = undefined;
        if (recording) {
            freeAll(&.{try t.legacy(&ls[i], false, prompt, &c)});
            freeAll(&c);
        }
        freeAll(&.{try t.plugin(&ps[i], .prefill, prompt, &c)});
        freeAll(&c);
        try t.expectStates(if (recording) &ls[i] else null, &ps[i]);
    }
    for (0..3) |tick| {
        var toks: [N]u32 = undefined;
        for (&toks, 0..) |*tk, i| tk.* = @intCast((tick * 31 + i * 7 + 5) % fixture.glm.V);
        var rows: []Array = &.{};
        defer {
            freeAll(rows);
            if (rows.len > 0) testing.allocator.free(rows);
        }
        if (recording) {
            var ctxs: [N]*ForwardCtx = undefined;
            for (0..N) |i| ctxs[i] = &ls[i].ctx;
            const r = try legacy_fwd.forwardGlmBatchedDecode(&t.lg.?.xfm, &toks, &ctxs, null);
            rows = try testing.allocator.dupe(Array, r);
            t.lg.?.xfm.allocator.free(r);
            for (0..N) |i| try settle(&ls[i], rows[i .. i + 1]);
            for (0..N) |i| ls[i].off += 1;
        }
        const got = try pluginTick(t, &ps, &toks);
        defer _ = mlx.mlx_array_free(got);
        for (0..N) |i| try settle(&ps[i], &.{got});
        for (0..N) |i| {
            ps[i].off += 1;
            var row = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(row);
            try mlx.check(mlx.mlx_slice(&row, got, &[_]c_int{ @intCast(i), 0 }, 2, &[_]c_int{ @intCast(i + 1), @intCast(fixture.glm.V) }, 2, &[_]c_int{ 1, 1 }, 2, t.s));
            var want = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(want);
            if (recording) try mlx.check(mlx.mlx_reshape(&want, rows[i], &[_]c_int{ 1, @intCast(fixture.glm.V) }, 2, t.s));
            try t.check(row, if (recording) want else null);
            try t.expectStates(if (recording) &ls[i] else null, &ps[i]);
        }
    }
}

/// One MTP head cache: a one-layer KV cache behind the engine's adapter, as `MtpCacheRef.PluginRef` holds it.
const HeadCache = struct {
    kv: KVCache,
    ssm: [1]SSMCacheEntry = .{.{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false }},
    adapter: cache_ops.Slot = undefined,

    fn init(c: *HeadCache) !void {
        c.* = .{ .kv = try KVCache.init(testing.allocator, 1) };
        c.adapter = .{ .kv = &c.kv, .ssm = &c.ssm };
    }

    fn view(c: *HeadCache) !abi.KvView {
        var v: abi.KvView = .{};
        try testing.expect(cache_ops.ops.view(c.adapter.handle(), 0, .{ .ctx = mlx.gpuStream().ctx }, &v).isOk());
        return v;
    }
};

// Bar: the head behind `SpecHooks.head` gives the legacy `mtp.0` head's normed output and cache rows bit for bit over
// history rows, chained drafts and a rollback.
test "glm5_next plugin: the MTP head equals the legacy head" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin(glm).open(try fixture.glmFiles(arena.allocator(), 7), "src/plugins/glm5_next/fixtures/head.tape");
    defer t.close();
    const s = t.s;
    var lh: ?glm_mtp.Head = if (t.lg) |l| (try glm_mtp.Head.load(testing.allocator, s, &l.cfg, &l.weights)) orelse return error.NoLegacyHead else null;
    defer if (lh) |*h| h.deinit();
    const hooks = glm.table.spec.?.head.?;
    try testing.expect(hooks.load(t.m, undefined).isOk());
    var lc: HeadCache = undefined;
    try lc.init();
    defer lc.kv.deinit();
    var pc: HeadCache = undefined;
    try pc.init();
    defer pc.kv.deinit();

    var prng = std.Random.DefaultPrng.init(0x4ead);
    // History rows, a chained draft at depths 1 and 2, a rollback to the history, then more history.
    const Op = union(enum) { rows: []const u32, chain: u32, truncate: usize };
    const script = [_]Op{ .{ .rows = &.{ 5, 77, 140, 9, 301, 12 } }, .{ .chain = 33 }, .{ .chain = 410 }, .{ .truncate = 6 }, .{ .rows = &.{ 64, 2, 18 } }, .{ .chain = 7 } };
    var prev: Array = .{ .ctx = null };
    defer if (prev.ctx != null) {
        _ = mlx.mlx_array_free(prev);
    };
    for (script) |op| {
        const ids: []const u32 = switch (op) {
            .rows => |r| r,
            .chain => |*c| c[0..1],
            .truncate => |len| {
                try lc.kv.truncate(len, s);
                try pc.kv.truncate(len, s);
                continue;
            },
        };
        const n: c_int = @intCast(ids.len);
        // A chain step reads the head's own output; history rows read fresh trunk hiddens.
        const hidden: Array = if (op == .chain) prev else try parity.testRandWeightBf16(prng.random(), &.{ 1, n, fixture.glm.H }, s);
        defer if (op != .chain) {
            _ = mlx.mlx_array_free(hidden);
        };
        const flat = try idsArray(ids);
        defer _ = mlx.mlx_array_free(flat);
        var ids1 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ids1);
        try mlx.check(mlx.mlx_reshape(&ids1, flat, &[_]c_int{n}, 1, s));
        const want: ?Array = if (lh) |*h| (try h.forward(&t.lg.?.xfm, &lc.kv, ids1, hidden, .none)).hidden_next else null;
        defer if (want) |w| {
            _ = mlx.mlx_array_free(w);
        };
        const handles = [_]abi.Handle{pc.adapter.handle()};
        const pos = [_]u32{0};
        var out: abi.ForwardOut = .{};
        try testing.expect(hooks.forward(t.m, &.{ .mode = if (n == 1) .decode else .prefill, .logits = .none, .want_hidden = true, .rows = @intCast(n), .tokens = .{}, .ids = .{ .ctx = flat.ctx }, .positions = .of(&pos), .caches = .of(&handles), .cache_ops = &cache_ops.ops, .hidden_in = .{ .ctx = hidden.ctx } }, &out).isOk());
        const got: Array = .{ .ctx = out.hidden.ctx };
        try t.check(got, want);
        var pv = try pc.view();
        defer cache_ops.ops.release(&pv);
        var lv: ?abi.KvView = if (lh != null) try lc.view() else null;
        defer if (lv) |*v| cache_ops.ops.release(v);
        try t.check(.{ .ctx = pv.k.ctx }, if (lv) |v| .{ .ctx = v.k.ctx } else null);
        try t.check(.{ .ctx = pv.v.ctx }, if (lv) |v| .{ .ctx = v.v.ctx } else null);
        if (prev.ctx != null) _ = mlx.mlx_array_free(prev);
        prev = got;
    }
    try t.tape.expectDone();
}

/// The plugin's logits over `ids` on slot `s`, with `media` rows from `offset` (none when `media` is null).
fn plainStep(t: *Twin(glm), s: *Slot, ids: []const u32, media: Array, offset: usize) !Array {
    const h = [_]abi.Handle{s.adapter.handle()};
    const pos = [_]u32{@intCast(s.off)};
    var out: abi.ForwardOut = .{};
    try testing.expect(glm.table.forward(t.m, &.{ .mode = if (ids.len == 1) .decode else .prefill, .logits = .all_rows, .rows = @intCast(ids.len), .tokens = .of(ids), .positions = .of(&pos), .caches = .of(&h), .cache_ops = &cache_ops.ops, .media = .{ .ctx = media.ctx }, .media_offset = @intCast(offset) }, &out).isOk());
    s.off += ids.len;
    const got: Array = .{ .ctx = out.logits.ctx };
    try settle(s, &.{got});
    return got;
}

// Bar: image rows the engine hands the plugin (`Forward.media`) splice at the placeholders as the legacy forward
// splices `vision_embeddings`, across two prompt chunks (the second resuming at its row offset), then decode.
test "glm5_next plugin: media rows splice as the legacy forward splices them" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin(glm).open(try fixture.glmFiles(arena.allocator(), 7), "src/plugins/glm5_next/fixtures/media.tape");
    defer t.close();
    const img: u32 = 500;
    try testing.expectEqual(img, t.cfg.image_token_id);
    var prng = std.Random.DefaultPrng.init(0x1a6e);
    const media = try parity.testRandWeightBf16(prng.random(), &.{ 1, 5, fixture.glm.H }, t.s);
    defer _ = mlx.mlx_array_free(media);
    var ls: Slot = undefined;
    try ls.init(t.layers());
    defer ls.deinit();
    var ps: Slot = undefined;
    try ps.init(t.layers());
    defer ps.deinit();
    const chunks = [_]struct { ids: []const u32, offset: usize }{
        .{ .ids = &.{ 3, 17, img, img, img, 42, 9 }, .offset = 0 },
        .{ .ids = &.{ img, img, 25, 1, 88 }, .offset = 3 },
        .{ .ids = &.{140}, .offset = 0 },
    };
    for (chunks) |c| {
        var want: ?Array = null;
        defer if (want) |w| {
            _ = mlx.mlx_array_free(w);
        };
        if (t.lg != null) {
            ls.ctx.vision_embeddings = media;
            ls.ctx.vision_splice_offset = c.offset;
            want = try t.legacy(&ls, false, c.ids, null);
            ls.ctx.vision_embeddings = null;
        }
        const got = try plainStep(t, &ps, c.ids, media, c.offset);
        defer _ = mlx.mlx_array_free(got);
        try t.check(got, want);
        try t.expectStates(if (t.lg != null) &ls else null, &ps);
    }
    // The splice moved the logits: without the rows the same prompt reads differently.
    var bare: Slot = undefined;
    try bare.init(t.layers());
    defer bare.deinit();
    const plain = try plainStep(t, &bare, chunks[0].ids, .{ .ctx = null }, 0);
    defer _ = mlx.mlx_array_free(plain);
    var ws: Slot = undefined;
    try ws.init(t.layers());
    defer ws.deinit();
    const with = try plainStep(t, &ws, chunks[0].ids, media, 0);
    defer _ = mlx.mlx_array_free(with);
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_array_equal(&eq, plain, with, false, t.s));
    try mlx.check(mlx.mlx_array_eval(eq));
    var same_bits: bool = true;
    try mlx.check(mlx.mlx_array_item_bool(&same_bits, eq));
    try testing.expect(!same_bits);
    try t.tape.expectDone();
}

// Bar: a batched tick's bits never depend on whether the slots' prefills were evaluated before it was built.
test "glm5_next plugin: a batched tick over unevaluated prefill states equals one over evaluated states" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin(glm).open(try fixture.glmFiles(arena.allocator(), 7), null);
    defer t.close();
    const N = fwd.MOE_DECODE_BATCH_MAX + 2;
    var arms: [2][N]Slot = undefined;
    var made: [2]usize = .{ 0, 0 };
    defer for (&arms, made) |*arm, m| for (arm[0..m]) |*sl| sl.deinit();
    var toks: [N]u32 = undefined;
    for (&toks, 0..) |*tk, i| tk.* = @intCast((i * 7 + 5) % fixture.glm.V);
    var got: [2]Array = .{ .{ .ctx = null }, .{ .ctx = null } };
    defer freeAll(&got);
    for (&arms, &made, 0..) |*arm, *m, lazy| {
        for (arm, 0..) |*sl, i| {
            try sl.init(t.layers());
            m.* += 1;
            const prompt = long_prompt[0 .. 3 + i];
            const h = [_]abi.Handle{sl.adapter.handle()};
            const pos = [_]u32{0};
            var out: abi.ForwardOut = .{};
            try testing.expect(glm.table.forward(t.m, &.{ .mode = .prefill, .rows = @intCast(prompt.len), .tokens = .of(prompt), .positions = .of(&pos), .caches = .of(&h), .cache_ops = &cache_ops.ops }, &out).isOk());
            sl.off = prompt.len;
            const l: Array = .{ .ctx = out.logits.ctx };
            defer _ = mlx.mlx_array_free(l);
            if (lazy == 0) try settle(sl, &.{l});
        }
        got[lazy] = try pluginTick(t, arm, &toks);
        try mlx.check(mlx.mlx_array_eval(got[lazy]));
    }
    try ops.expectBitEqual(got[1], got[0]);
}

test "glm5_next plugin: a decode token on the device equals the host token's decode" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin(glm).open(try fixture.glmFiles(arena.allocator(), 7), null);
    defer t.close();
    var hs: Slot = undefined;
    try hs.init(t.layers());
    defer hs.deinit();
    var ds: Slot = undefined;
    try ds.init(t.layers());
    defer ds.deinit();
    for ([_]*Slot{ &hs, &ds }) |s| freeAll(&.{try plainStep(t, s, long_prompt[0..5], .{ .ctx = null }, 0)});
    for ([_]u32{ 8, 55, 12 }) |id| {
        const want = try plainStep(t, &hs, &.{id}, .{ .ctx = null }, 0);
        defer _ = mlx.mlx_array_free(want);
        const dev = mlx.mlx_array_new_data(&id, &[_]c_int{ 1, 1 }, 2, .uint32);
        defer _ = mlx.mlx_array_free(dev);
        const h = [_]abi.Handle{ds.adapter.handle()};
        const pos = [_]u32{@intCast(ds.off)};
        var out: abi.ForwardOut = .{};
        try testing.expect(glm.table.forward(t.m, &.{ .mode = .decode, .logits = .all_rows, .rows = 1, .tokens = .{}, .ids = .{ .ctx = dev.ctx }, .positions = .of(&pos), .caches = .of(&h), .cache_ops = &cache_ops.ops }, &out).isOk());
        ds.off += 1;
        const got: Array = .{ .ctx = out.logits.ctx };
        defer _ = mlx.mlx_array_free(got);
        try settle(&ds, &.{got});
        try ops.expectBitEqual(got, want);
    }
}
