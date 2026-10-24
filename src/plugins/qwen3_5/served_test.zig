//! The qwen3_5 table served through the engine's plugin driver, proven against the legacy shell over the same tiny
//! pack: prompt chunks, decode, captured verifies with partial-accept rollbacks from the captures, the prompt-lookup
//! round loop, batched decode ticks and a batched captured verify, and an image prompt (media rows spliced across a
//! chunk boundary on the M-RoPE table). Logits, hidden rows and every recurrent state bit-equal. The served side's
//! `ModelConfig` is the legacy parse with the plugin bound, as the engine builds it.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const sdk = @import("../../sdk/root.zig");
const fixture = @import("../../sdk/bases/proof/fixture.zig");
const pack = @import("../../sdk/bases/proof/qwen35_test.zig");
const model = @import("../../engines/mlx/model.zig");
const driver = @import("../../engines/mlx/plugin/driver.zig");
const plugin_config = @import("../../engines/mlx/plugin/config.zig");
const ssm_entry = @import("../../engines/mlx/ssm/entry.zig");
const KVCache = @import("../../engines/mlx/kv/cache.zig").KVCache;
const ForwardCtx = @import("../../engines/mlx/forward_ctx.zig").ForwardCtx;
const CaptureLayers = @import("../../engines/mlx/forward_ctx.zig").CaptureLayers;
const SSMCacheEntry = ssm_entry.SSMCacheEntry;
const Generator = @import("../../engines/mlx/generate/generator.zig").Generator;
const Tokenizer = @import("../../sdk/tokenizer/tokenizer.zig").Tokenizer;
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;
const qwen3_5 = @import("plugin.zig");
const tape_mod = @import("../../engines/mlx/plugin/tape.zig");

const testing = std.testing;
const Array = mlx.mlx_array;

/// The legacy shell and the served shell over one pack, the served one bound to `table`.
pub const Pair = struct {
    lg: *fixture.Legacy,
    dir: []u8,
    cfg: model.ModelConfig,
    xfm: Transformer,
    /// The legacy values every proof compares (`Oracle`).
    oracle: Oracle = .{},

    pub fn open(p: *Pair, table: *const abi.ModelTable, pk: *fixture.Pack, json: []const u8) !void {
        return p.openWith(table, pk, json, null);
    }

    /// `open`, with `edit` applied to the plugin's parse.
    pub fn openWith(p: *Pair, table: *const abi.ModelTable, pk: *fixture.Pack, json: []const u8, edit: ?*const fn (*abi.Parsed) void) !void {
        const a = testing.allocator;
        const io = std.Io.Threaded.global_single_threaded.io();
        p.lg = try pk.load(json);
        errdefer p.lg.deinit();
        p.lg.xfm.compileGdnGate();
        p.lg.xfm.compileMoeRouting();
        var buf: [512]u8 = undefined;
        p.dir = try a.dupe(u8, buf[0..try p.lg.tmp.dir.realPath(io, &buf)]);
        errdefer a.free(p.dir);
        const b = try a.create(plugin_config.Binding);
        b.* = .{ .table = table };
        const mem: sdk.fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = json }} };
        const dir = mem.packDir();
        const v = table.parse(&dir, abi.Str.of(json), .{}, null, false, &b.parsed);
        if (!v.isOk()) {
            a.destroy(b);
            std.debug.print("parse refused: {s}\n", .{v.why.slice()});
            return error.ParseRefused;
        }
        if (edit) |e| e(&b.parsed);
        p.cfg = try model.parseConfigFromJson(a, json);
        p.cfg.plugin = b;
        errdefer p.cfg.deinit(a);
        p.xfm = try driver.load(io, a, &p.cfg, p.dir, fixture.stream());
    }

    pub fn close(p: *Pair) void {
        p.oracle.finish();
        p.xfm.deinit();
        p.cfg.deinit(testing.allocator);
        testing.allocator.free(p.dir);
        p.lg.deinit();
    }

    pub fn shells(p: *Pair) [2]*Transformer {
        return .{ &p.lg.xfm, &p.xfm };
    }
};

/// What a proof compares, recorded from the legacy forward for when it is deleted (`tape.zig`): with
/// `MLX_SERVE_PROOF_RECORD=<repo root>` every legacy value is written to the tape at `rel`; a recorded tape played
/// back checks the plugin's values against it beside the live legacy shell.
pub const Oracle = struct {
    tape: tape_mod.Tape = .{},

    pub fn open(comptime rel: []const u8, recorded: ?[]const u8) !Oracle {
        if (tape_mod.recordRoot()) |root| return .{ .tape = try tape_mod.Tape.recorder(root, rel) };
        if (recorded) |bytes| return .{ .tape = try tape_mod.Tape.player(bytes) };
        return .{};
    }

    pub fn finish(o: *Oracle) void {
        o.tape.finish() catch |e| std.debug.panic("proof tape: {t}", .{e});
    }

    /// A playback read every recorded value (a proof's last call).
    pub fn done(o: *const Oracle) !void {
        try o.tape.expectDone();
    }

    /// `got` (the plugin's) bit-equal to `want` (the legacy's) and to the tape.
    pub fn check(o: *Oracle, got: Array, want: Array, greedy: bool) !void {
        try ops.expectBitEqual(got, want);
        if (o.tape.recording()) try o.tape.put(want);
        if (o.tape.playing()) try o.tape.expectNext(got, greedy);
    }

    /// A greedy run's tokens.
    pub fn stream(o: *Oracle, got: []const u32, want: []const u32) !void {
        try testing.expectEqualSlices(u32, want, got);
        if (o.tape.path == null and !o.tape.playing()) return;
        const g = mlx.mlx_array_new_data(got.ptr, &[_]c_int{@intCast(got.len)}, 1, .uint32);
        defer _ = mlx.mlx_array_free(g);
        const w = mlx.mlx_array_new_data(want.ptr, &[_]c_int{@intCast(want.len)}, 1, .uint32);
        defer _ = mlx.mlx_array_free(w);
        try o.check(g, w, true);
    }

    /// Every recurrent state and verify capture (`fixture.expectEntries`), live only: the logits and captures that
    /// follow carry them onto the tape at a fraction of the bytes.
    pub fn entries(_: *Oracle, want: []const SSMCacheEntry, got: []const SSMCacheEntry) !void {
        try fixture.expectEntries(want, got);
    }
};

/// The oracle of one proof: tape `src/plugins/<family>/fixtures/<name>.tape`.
pub fn oracle(comptime family: []const u8, comptime name: []const u8) !Oracle {
    const rel = family ++ "/fixtures/" ++ name ++ ".tape";
    return Oracle.open("src/plugins/" ++ rel, @embedFile("../" ++ rel));
}

/// One request's cache, recurrent entries and position.
pub const Lane = struct {
    kv: KVCache,
    ssm: []SSMCacheEntry,
    off: usize = 0,
    ctx: ForwardCtx = undefined,

    pub fn init(l: *Lane, x: *Transformer) !void {
        const n = x.config.num_hidden_layers;
        l.* = .{ .kv = try KVCache.init(testing.allocator, n), .ssm = try fixture.newEntries(n) };
        l.ctx = x.defaultCtx();
        l.ctx.cache = &l.kv;
        l.ctx.moe_seq_offset = &l.off;
        l.ctx.ssm_entries = l.ssm;
    }
    pub fn deinit(l: *Lane) void {
        l.kv.deinit();
        fixture.freeEntries(l.ssm);
    }
};

fn ids(comptime n: usize, seed: u32) [n]i32 {
    var out: [n]i32 = undefined;
    for (&out, 0..) |*v, i| v.* = @intCast((seed + 37 * @as(u32, @intCast(i))) % 256);
    return out;
}

const Step = struct { ids: []const i32, verify: bool = false, accept: ?u32 = null };

/// Prefill chunks, decode, captured verifies with partial accepts rolled back from the captures, then decode on the
/// rolled-back state. Never a rollback to a verify's last row: the GDN verify kernels leave it unwritten (a whole
/// accept keeps the state as it is), and the legacy layered forward writes it on some layers.
const steps = [_]Step{
    .{ .ids = &ids(20, 3) },
    .{ .ids = &ids(70, 11) },
    .{ .ids = &.{88} },
    .{ .ids = &.{ 5, 61, 200 }, .verify = true, .accept = 1 },
    .{ .ids = &.{12} },
    .{ .ids = &.{ 9, 33, 77, 140, 2 }, .verify = true, .accept = 3 },
    .{ .ids = &.{64} },
    .{ .ids = &.{ 6, 7, 8, 9 }, .verify = true, .accept = 0 },
    .{ .ids = &ids(6, 5) },
    .{ .ids = &.{17} },
};

/// The solo steps (every one capturing layers 0 and 1, as a DFlash drafter reads them) and the prompt-lookup round
/// loop, served vs legacy.
pub fn expectSolo(p: *Pair) !void {
    const a = testing.allocator;
    const s = fixture.stream();
    const xs = p.shells();
    var lanes: [2]Lane = undefined;
    for (&lanes, xs) |*l, x| try l.init(x);
    defer for (&lanes) |*l| l.deinit();
    const recurrent = driver.positionsOf(&p.xfm) != .kv_step;
    for (steps) |st| {
        var g = ops.Graph.init(a, s);
        defer g.deinit();
        const t = try fixture.tokens(&g, st.ids);
        var outs: [2]Array = undefined;
        var caps: [2][2]Array = @splat(@splat(.{ .ctx = null }));
        defer for (caps) |pair| for (pair) |c| if (c.ctx != null) {
            _ = mlx.mlx_array_free(c);
        };
        const before = lanes[0].off;
        for (&lanes, xs, &outs, &caps) |*l, x, *o, *cv| {
            for (cv) |*c| c.* = mlx.mlx_array_new();
            var cl: CaptureLayers = .{ .ids = &.{ 0, 1 }, .out = cv };
            l.ctx.capture_ssm_seq = st.verify;
            l.ctx.capture_layers = &cl;
            defer {
                l.ctx.capture_ssm_seq = false;
                l.ctx.capture_layers = null;
            }
            o.* = try g.adopt(try x.forwardWith(&l.ctx, t));
        }
        try p.oracle.check(outs[1], outs[0], false);
        for (caps[0], caps[1]) |want, got| try p.oracle.check(got, want, false);
        // A trunk the driver positions off the KV step leaves `moe_seq_offset` to the scheduler.
        if (recurrent) try testing.expectEqual(lanes[0].off, lanes[1].off);
        try p.oracle.entries(lanes[0].ssm, lanes[1].ssm);
        if (st.accept) |acc| {
            for (&lanes) |*l| {
                const keep = before + 1 + acc;
                const step_keep = l.kv.step;
                try l.kv.truncate(keep, s);
                // A GDN trunk positions off `moe_seq_offset` and keeps its KV step (the engine's PLD rollback).
                if (recurrent) l.kv.step = step_keep;
                for (l.ssm) |*e| {
                    try ssm_entry.ssmRollbackFromCapture(e, acc, @intCast(st.ids.len), s);
                    ssm_entry.ssmFreeSpecCapture(e);
                }
                l.off = keep;
            }
            try p.oracle.entries(lanes[0].ssm, lanes[1].ssm);
        }
    }

    var tok_dummy: Tokenizer = undefined; // never read by the Generator
    const prompt = [_]u32{ 3, 7, 1, 12, 30, 5, 9, 22, 3, 7, 1, 12, 30, 5, 9, 22, 3, 7, 1 };
    var streams: [2][24]u32 = undefined;
    for (xs, &streams) |x, *out| {
        var gen = try Generator.initWithOptions(std.Io.Threaded.global_single_threaded.io(), a, x, &tok_dummy, &prompt, out.len, .{ .temperature = 0.0 }, &.{}, .{ .pld_enabled = true });
        defer gen.deinit(a);
        var n: usize = 0;
        while (try gen.nextPld(a, 4, 2)) |res| {
            defer a.free(res.tokens);
            @memcpy(out[n..][0..res.tokens.len], res.tokens);
            n += res.tokens.len;
        }
        try testing.expectEqual(out.len, n);
    }
    try p.oracle.stream(&streams[1], &streams[0]);
    try p.oracle.done();
}

/// With a drafter bound (the row-exact lanes on both sides): the solo steps equal the legacy shell, and after a prompt
/// a captured verify gives every row the logits serial decode of the same tokens gives, bit for bit.
fn expectBound(p: *Pair) !void {
    for (p.shells()) |x| {
        x.config.dflash_bound = true;
        try testing.expect(x.config.rowExactDecode());
    }
    try expectSolo(p);
    for (p.shells()) |x| try expectVerifyEqualsSerial(x);
}

fn expectVerifyEqualsSerial(x: *Transformer) !void {
    const a = testing.allocator;
    const s = fixture.stream();
    var lanes: [2]Lane = undefined;
    for (&lanes) |*l| try l.init(x);
    defer for (&lanes) |*l| l.deinit();
    var g = ops.Graph.init(a, s);
    defer g.deinit();
    const prompt = try fixture.tokens(&g, &ids(19, 7));
    for (&lanes) |*l| _ = try g.adopt(try x.forwardWith(&l.ctx, prompt));
    const drafts = [_]i32{ 88, 5, 61, 200, 12, 9, 33, 77 };
    lanes[1].ctx.capture_ssm_seq = true;
    const verify = try g.adopt(try x.forwardWith(&lanes[1].ctx, try fixture.tokens(&g, &drafts)));
    const vocab = ops.shape(verify)[ops.shape(verify).len - 1];
    for (drafts, 0..) |d, i| {
        const want = try g.adopt(try x.forwardWith(&lanes[0].ctx, try fixture.tokens(&g, &.{d})));
        const r: c_int = @intCast(i);
        const row = try g.slice(verify, &.{ 0, r, 0 }, &.{ 1, r + 1, vocab });
        try ops.expectBitEqual(row, want);
    }
}

/// Two slots: batched decode ticks and a batched captured verify, served vs legacy.
pub fn expectBatched(p: *Pair) !void {
    const a = testing.allocator;
    const s = fixture.stream();
    try testing.expect(p.xfm.supportsBatchedGdnDecode() and p.lg.xfm.supportsBatchedGdnDecode());
    // The legacy tick splits its merged states back after every tick, so each slot's states can be compared.
    p.lg.xfm.persistent_group_state_override = false;
    const xs = p.shells();
    var lanes: [2][2]Lane = undefined;
    for (&lanes, xs) |*pair, x| for (pair) |*l| try l.init(x);
    defer for (&lanes) |*pair| for (pair) |*l| l.deinit();
    for ([_][]const i32{ &ids(20, 3), &ids(7, 9) }, 0..) |pr, j| {
        var g = ops.Graph.init(a, s);
        defer g.deinit();
        for (&lanes, xs) |*pair, x| _ = try g.adopt(try x.forwardWith(&pair[j].ctx, try fixture.tokens(&g, pr)));
    }
    for (&lanes, xs) |*pair, x| try testing.expect(x.batchedGdnReady(&.{ &pair[0].ctx, &pair[1].ctx }));
    for ([_][2]u32{ .{ 77, 3 }, .{ 190, 64 } }) |next| {
        var g = ops.Graph.init(a, s);
        defer g.deinit();
        var outs: [2][4]Array = undefined;
        for (&lanes, xs, &outs) |*pair, x, *o| {
            const ctxs = [_]*ForwardCtx{ &pair[0].ctx, &pair[1].ctx };
            const offs = [_]u32{ @intCast(pair[0].off), @intCast(pair[1].off) };
            var hidden: ?[]Array = null;
            const rows = try x.forwardMoeBatchedDecode(&next, &ctxs, &offs, &hidden);
            defer a.free(rows);
            defer a.free(hidden.?);
            o.* = .{ try g.adopt(rows[0]), try g.adopt(rows[1]), try g.adopt(hidden.?[0]), try g.adopt(hidden.?[1]) };
            for (pair) |*l| l.off += 1;
        }
        for (outs[0], outs[1]) |want, got| try p.oracle.check(got, want, false);
        for (&lanes[0], &lanes[1]) |*w, *gl| try p.oracle.entries(w.ssm, gl.ssm);
    }
    var g = ops.Graph.init(a, s);
    defer g.deinit();
    const t = try g.adopt(mlx.mlx_array_new_data(&[_]i32{ 5, 61, 200, 9, 33, 77 }, &[_]c_int{ 2, 3 }, 2, .int32));
    var outs: [2][3]Array = undefined;
    for (&lanes, xs, &outs) |*pair, x, *o| {
        const ctxs = [_]*ForwardCtx{ &pair[0].ctx, &pair[1].ctx };
        const offs = [_]u32{ @intCast(pair[0].off), @intCast(pair[1].off) };
        var last = mlx.mlx_array_new();
        var all = mlx.mlx_array_new();
        const logits = try x.forwardMoeBatchedVerify(t, &ctxs, &offs, &last, &all);
        o.* = .{ try g.adopt(logits), try g.adopt(last), try g.adopt(all) };
    }
    for (outs[0], outs[1]) |want, got| try p.oracle.check(got, want, false);
    for (&lanes[0], &lanes[1]) |*w, *gl| try p.oracle.entries(w.ssm, gl.ssm);
    try p.oracle.done();
}

/// 4 text tokens, a 2x3 image (id 250), 3 text tokens on the M-RoPE table; the rows splice across a chunk boundary,
/// then decode and a captured verify past the table.
pub fn expectImage(p: *Pair) !void {
    const a = testing.allocator;
    const s = fixture.stream();
    try testing.expectEqual(@as(u32, 250), p.cfg.image_token_id);
    const T = 13;
    const tokens = [T]i32{ 3, 17, 42, 99, 250, 250, 250, 250, 250, 250, 7, 8, 9 };
    var table: [3 * T]i32 = undefined;
    for (0..T) |i| {
        const img = i >= 4 and i < 10;
        const r: i32 = if (img) @intCast((i - 4) / 3) else 0;
        const c: i32 = if (img) @intCast((i - 4) % 3) else 0;
        const t: i32 = if (i < 4) @intCast(i) else if (img) 4 else @intCast(i - 10 + 7);
        table[i] = t;
        table[T + i] = if (img) t + r else t;
        table[2 * T + i] = if (img) t + c else t;
    }
    var g0 = ops.Graph.init(a, s);
    defer g0.deinit();
    const rows = try g0.astype(try g0.adopt(try fixtureRows(6, @intCast(p.cfg.hidden_size), 9)), p.cfg.actDtype());
    const xs = p.shells();
    var lanes: [2]Lane = undefined;
    for (&lanes, xs) |*l, x| {
        try l.init(x);
        l.ctx.mrope_pos = &table;
        l.ctx.mrope_total = T;
        l.ctx.mrope_delta = 10 - T;
        l.ctx.vision_embeddings = rows;
    }
    defer for (&lanes) |*l| l.deinit();
    const chunks = [_]struct { ids: []const i32, offset: usize = 0, verify: bool = false }{
        .{ .ids = tokens[0..7] },
        .{ .ids = tokens[7..], .offset = 3 },
        .{ .ids = &.{88} },
        .{ .ids = &.{ 5, 61, 200 }, .verify = true },
        .{ .ids = &.{12} },
    };
    for (chunks) |ch| {
        var g = ops.Graph.init(a, s);
        defer g.deinit();
        const t = try fixture.tokens(&g, ch.ids);
        var outs: [2]Array = undefined;
        for (&lanes, xs, &outs) |*l, x, *o| {
            l.ctx.vision_splice_offset = ch.offset;
            l.ctx.capture_ssm_seq = ch.verify;
            defer l.ctx.capture_ssm_seq = false;
            o.* = try g.adopt(try x.forwardWith(&l.ctx, t));
        }
        try p.oracle.check(outs[1], outs[0], false);
        try p.oracle.entries(lanes[0].ssm, lanes[1].ssm);
    }
    try p.oracle.done();
}

pub fn fixtureRows(n: c_int, h: c_int, seed: u64) !Array {
    var prng = std.Random.DefaultPrng.init(seed);
    const buf = try testing.allocator.alloc(f32, @intCast(n * h));
    defer testing.allocator.free(buf);
    for (buf) |*x| x.* = prng.random().floatNorm(f32) * 0.5;
    return mlx.mlx_array_new_data(buf.ptr, &[_]c_int{ 1, n, h }, 3, .float32);
}

fn openPack(p: *Pair, comptime moe: bool, comptime quantized: bool, comptime experts: c_int, comptime seed: u64, json: []const u8) !void {
    var pk = fixture.Pack.init(seed);
    defer pk.deinit();
    try pack.writePack(&pk, moe, quantized, experts);
    try p.open(&qwen3_5.table, &pk, json);
    p.oracle = try oracle("qwen3_5", std.fmt.comptimePrint("served-{x}", .{seed}));
}

/// The pack's config with an image tower: id 250 and interleaved M-RoPE sections [11,11,10].
fn vlConfig(comptime moe: bool) ![]u8 {
    const a = testing.allocator;
    const text = pack.config(moe, true, 8);
    const json = try std.mem.concat(a, u8, &.{ "{\"image_token_id\":250,\"vision_config\":{\"depth\":1},", text[1..] });
    defer a.free(json);
    return std.mem.replaceOwned(u8, a, json, "\"partial_rotary_factor\":0.25}", "\"partial_rotary_factor\":0.25,\"mrope_section\":[11,11,10],\"mrope_interleaved\":true}");
}

test "qwen3_5 served: dense 4-bit steps, rollbacks and the lookup loop equal the legacy shell" {
    var p: Pair = undefined;
    try openPack(&p, false, true, 8, 0x5e01, pack.config(false, true, 8));
    defer p.close();
    try testing.expectEqual(driver.Positions.gdn, driver.positionsOf(&p.xfm));
    try expectSolo(&p);
}

test "qwen3_5 served: a chat pack embeds a padded batch as it embeds each sequence alone" {
    var p: Pair = undefined;
    try openPack(&p, false, true, 8, 0x5e01, pack.config(false, true, 8));
    defer p.close();
    try @import("../../engines/mlx/plugin/both.zig").expectEmbeddingBatch(&p.xfm);
}

test "qwen3_5 served: a qwen3_5_moe 4-bit pack's steps and lookup loop equal the legacy shell" {
    var p: Pair = undefined;
    try openPack(&p, true, true, 64, 0x5e02, pack.config(true, true, 64));
    defer p.close();
    try expectSolo(&p);
}

test "qwen3_5 served: a bf16 pack (decode side copies) equals the legacy shell" {
    var p: Pair = undefined;
    try openPack(&p, false, false, 8, 0x5e03, pack.config(false, false, 8));
    defer p.close();
    try expectSolo(&p);
}

test "qwen3_5 served: dense and MoE batched decode ticks and a batched verify equal the legacy GDN batched path" {
    inline for (.{ false, true }, .{ 0x5e04, 0x5e05 }) |moe, seed| {
        var p: Pair = undefined;
        try openPack(&p, moe, true, 8, seed, pack.config(moe, true, 8));
        defer p.close();
        try expectBatched(&p);
    }
}

test "qwen3_5 served: an image prompt on the M-RoPE table (dense and MoE) equals the legacy shell" {
    inline for (.{ false, true }, .{ 0x5e06, 0x5e07 }) |moe, seed| {
        const json = try vlConfig(moe);
        defer testing.allocator.free(json);
        var p: Pair = undefined;
        try openPack(&p, moe, true, 8, seed, json);
        defer p.close();
        try expectImage(&p);
    }
}

test "qwen3_5 served: with a drafter bound, the row-exact lanes equal the legacy rowExactDecode and a verify row equals serial decode" {
    var p: Pair = undefined;
    try openPack(&p, false, true, 8, 0x5e08, pack.config(false, true, 8));
    defer p.close();
    try testing.expect(p.xfm.plugin.?.drafters);
    try testing.expectEqual(.mma, @as(*qwen3_5.Loaded, @ptrCast(@alignCast(p.xfm.plugin.?.model))).body.dense.base.row_exact);
    try expectBound(&p);
}

test "qwen3_5 served: a bf16 pack takes DFlash without exact_drafted (its matmul bits follow the row count)" {
    var p: Pair = undefined;
    try openPack(&p, false, false, 8, 0x5e09, pack.config(false, false, 8));
    defer p.close();
    try testing.expect(p.xfm.plugin.?.drafters and !p.cfg.plugin.?.parsed.caps.exact_drafted);
}
