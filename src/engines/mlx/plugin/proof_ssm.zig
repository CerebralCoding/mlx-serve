//! A recurrent model served through the driver proven against the legacy GDN forward: a qwen3_5 pack (linear
//! attention at layers 0 and 2) whose test table runs `sdk/bases`' qwen3_5 decoder over the legacy model's arrays
//! (the bases' own proof covers that the weights are read as the legacy loader reads them). What is under test is
//! the engine's side: the slot's recurrent entries behind `cache_ops`, positions from `moe_seq_offset`, a capturing
//! verify (`Ctx.capture_spec`), the partial-accept rollback over those captures, and the prompt-lookup round loop.

const std = @import("std");
const abi = @import("../../../abi/plugin.zig");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const bases = @import("../../../sdk/bases/bases.zig");
const run_mod = @import("../../../sdk/bases/run.zig");
const fixture = @import("../../../sdk/bases/proof/fixture.zig");
const q35 = @import("../../../sdk/bases/proof/qwen35_test.zig");
const model = @import("../model.zig");
const driver = @import("driver.zig");
const plugin_config = @import("config.zig");
const ssm_entry = @import("../ssm/entry.zig");
const mrope = @import("../../../sdk/vision/mrope.zig");
const KVCache = @import("../kv/cache.zig").KVCache;
const ForwardCtx = @import("../forward_ctx.zig").ForwardCtx;
const SSMCacheEntry = ssm_entry.SSMCacheEntry;
const Generator = @import("../generate/generator.zig").Generator;
const Tokenizer = @import("../../../sdk/tokenizer/tokenizer.zig").Tokenizer;
const Transformer = @import("../transformer.zig").Transformer;

const testing = std.testing;
const Array = mlx.mlx_array;
const D = q35.Dense35;
const LAYERS = 4;

/// The served model the test table hands the driver, and its M-RoPE spectrum (rope_dims 64 at theta 1e5).
var served: bases.Model(D) = undefined;
var inv_freq: [32]f64 = undefined;

fn tParse(_: *const abi.PackDir, _: abi.Str, _: abi.Str, _: ?*const abi.LayerSpan, _: bool, _: *abi.Parsed) callconv(.c) abi.Verdict {
    return .ok;
}
fn tFree(_: *abi.Parsed) callconv(.c) void {}
fn tBill(_: *const abi.Parsed, _: *const abi.Shape, out: *abi.Bill) callconv(.c) void {
    out.* = .{};
}
fn tLoad(_: *const abi.LoadCtx, _: *abi.Verdict) callconv(.c) ?*anyopaque {
    return &served;
}
fn tUnload(_: *anyopaque) callconv(.c) void {}
fn tForward(m: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) abi.Verdict {
    return run_mod.tableForward(D, bases.Model(D).of(m), f, out);
}
fn tLayout(_: *const abi.Parsed, _: ?*const abi.LayerSpan, out: [*]abi.LayerCache, cap: u32) callconv(.c) u32 {
    const n = @min(cap, LAYERS);
    for (out[0..n], 0..) |*l, i| l.* = .{ .kind = if (i % 2 == 0) .linear else .full };
    return n;
}
fn tClaims(_: abi.Str, _: abi.Str, _: *const abi.PackDir) callconv(.c) abi.Claim {
    return .no;
}

const test_table: abi.ModelTable = .{
    .name = .of("qwen35-test"),
    .claims = &tClaims,
    .parse = &tParse,
    .free_parsed = &tFree,
    .bill = &tBill,
    .load = &tLoad,
    .unload = &tUnload,
    .forward = &tForward,
    .cache = &.{ .layout = &tLayout },
    .weights = &.{},
};

/// The legacy model and the served shell over one qwen3_5 pack.
const Pair = struct {
    lg: *fixture.Legacy,
    m: q35.Model,
    layers: [LAYERS]D.Layer = undefined,
    dir: []u8,
    cfg: model.ModelConfig,
    xfm: Transformer,

    /// A text pack, or with `media` its Qwen3-VL twin: interleaved M-RoPE sections [11,11,10] and image id 250.
    fn open(p: *Pair, seed: u64, media: bool) !void {
        const a = testing.allocator;
        const io = std.Io.Threaded.global_single_threaded.io();
        var pack = fixture.Pack.init(seed);
        defer pack.deinit();
        try q35.writePack(&pack, false, true, 8);
        const text = q35.config(false, true, 8);
        const json = if (media) try std.mem.concat(a, u8, &.{ "{\"image_token_id\":250,", text[1..] }) else try a.dupe(u8, text);
        defer a.free(json);
        const vl = try std.mem.replaceOwned(u8, a, json, "\"partial_rotary_factor\":0.25}", "\"partial_rotary_factor\":0.25,\"mrope_section\":[11,11,10],\"mrope_interleaved\":true}");
        defer a.free(vl);
        p.lg = try pack.load(if (media) vl else json);
        errdefer p.lg.deinit();
        p.lg.xfm.compileGdnGate();
        p.m = .{ .geo = try q35.gdnGeo(p.lg) };
        errdefer p.m.deinit();
        served = .{ .weights = try q35.baseWeights(D, p.lg, &p.m, &p.layers), .shared = bases.Shared.init(p.lg.config.rms_norm_eps, p.lg.config.postNormEps()), .a = a, .s = p.lg.xfm.s, .placeholders = .{ .image = p.lg.config.image_token_id } };
        errdefer served.shared.deinit();
        var buf: [512]u8 = undefined;
        p.dir = try a.dupe(u8, buf[0..try p.lg.tmp.dir.realPath(io, &buf)]);
        errdefer a.free(p.dir);
        const b = try a.create(plugin_config.Binding);
        const lc = &p.lg.config;
        b.* = .{ .table = &test_table, .parsed = .{
            .shell = .{
                .model_type = .of("qwen3_5"),
                .vocab_size = lc.vocab_size,
                .hidden_size = lc.hidden_size,
                .intermediate_size = lc.intermediate_size,
                .num_hidden_layers = lc.num_hidden_layers,
                .num_attention_heads = lc.num_attention_heads,
                .num_key_value_heads = lc.num_key_value_heads,
                .head_dim = lc.head_dim,
                .max_position_embeddings = lc.max_position_embeddings,
                .quant = .{ .bits = @intCast(lc.quant_bits), .group_size = lc.quant_group_size, .mode = .affine },
            },
            .caps = .{ .batches_decode = true },
            .placeholder_ids = .{ .image = lc.image_token_id },
            .mrope = if (media) .{ .sections = lc.mrope_section, .rope_dims = 64, .inv_freq = .of(&inv_freq) } else .{},
            .cache = .{ .ssm_state = .{ .entries = LAYERS } },
        } };
        p.cfg = plugin_config.modelConfig(&b.parsed);
        p.cfg.plugin = b;
        errdefer p.cfg.deinit(a);
        p.xfm = try driver.load(io, a, &p.cfg, p.dir, fixture.stream());
    }

    fn close(p: *Pair) void {
        p.xfm.deinit();
        p.cfg.deinit(testing.allocator);
        testing.allocator.free(p.dir);
        served.shared.deinit();
        p.m.deinit();
        p.lg.deinit();
    }
};

/// One request's cache, recurrent entries and position.
const Lane = struct {
    kv: KVCache,
    ssm: []SSMCacheEntry,
    off: usize = 0,
    ctx: ForwardCtx = undefined,

    fn init(l: *Lane, x: *Transformer) !void {
        l.* = .{ .kv = try KVCache.init(testing.allocator, LAYERS), .ssm = try fixture.newEntries(LAYERS) };
        l.ctx = x.defaultCtx();
        l.ctx.cache = &l.kv;
        l.ctx.moe_seq_offset = &l.off;
        l.ctx.ssm_entries = l.ssm;
    }
    fn deinit(l: *Lane) void {
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

/// Prefill chunks, decode, captured verifies with a partial accept rolled back from the captures (and one kept
/// whole), then decode on the rolled-back state.
const steps = [_]Step{
    .{ .ids = &ids(20, 3) },
    .{ .ids = &ids(70, 11) },
    .{ .ids = &.{88} },
    .{ .ids = &.{ 5, 61, 200 }, .verify = true, .accept = 1 },
    .{ .ids = &.{12} },
    .{ .ids = &.{ 9, 33, 77, 140, 2 }, .verify = true, .accept = 4 },
    .{ .ids = &.{64} },
    .{ .ids = &.{ 6, 7, 8, 9 }, .verify = true, .accept = 0 },
    .{ .ids = &ids(6, 5) },
    .{ .ids = &.{17} },
};

// ── tests ──

test "plugin recurrent: captured verifies, the partial-accept rollback and the lookup round loop equal the legacy GDN path" {
    const a = testing.allocator;
    var p: Pair = undefined;
    try p.open(0x55d1, false);
    defer p.close();
    try testing.expectEqual(driver.Positions.gdn, driver.positionsOf(&p.xfm));
    try testing.expect(p.cfg.needsSsmEntries() and p.xfm.ssm_entries != null);
    const s = fixture.stream();
    var lanes: [2]Lane = undefined;
    const xs = [_]*Transformer{ &p.lg.xfm, &p.xfm };
    for (&lanes, xs) |*l, x| try l.init(x);
    defer for (&lanes) |*l| l.deinit();
    for (steps) |st| {
        var g = ops.Graph.init(a, s);
        defer g.deinit();
        const t = try fixture.tokens(&g, st.ids);
        var outs: [2]Array = undefined;
        const before = lanes[0].off;
        for (&lanes, xs, &outs) |*l, x, *o| {
            l.ctx.capture_ssm_seq = st.verify;
            defer l.ctx.capture_ssm_seq = false;
            o.* = try g.adopt(try x.forwardWith(&l.ctx, t));
            // A prefill chunk's eval cadence names every conv state, so none is left holding the chunk.
            if (st.ids.len >= 32) for (l.ssm) |*e| if (e.initialized) {
                var ready = false;
                _ = mlx._mlx_array_is_available(&ready, e.conv_state);
                try testing.expect(ready);
            };
        }
        try ops.expectBitEqual(outs[1], outs[0]);
        try testing.expectEqual(lanes[0].off, lanes[1].off);
        try fixture.expectEntries(lanes[0].ssm, lanes[1].ssm);
        if (st.accept) |acc| for (&lanes) |*l| {
            // The PLD fast path's rollback: KV to the accepted length, each entry from its capture.
            const keep = before + 1 + acc;
            const step_keep = l.kv.step;
            try l.kv.truncate(keep, s);
            l.kv.step = step_keep;
            for (l.ssm) |*e| {
                try ssm_entry.ssmRollbackFromCapture(e, acc, @intCast(st.ids.len), s);
                ssm_entry.ssmFreeSpecCapture(e);
            }
            l.off = keep;
        };
        if (st.accept != null) try fixture.expectEntries(lanes[0].ssm, lanes[1].ssm);
    }

    // The prompt-lookup loop: a repeating prompt drafts, verifies capture, partial accepts roll back.
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
    try testing.expectEqualSlices(u32, &streams[0], &streams[1]);
}

test "plugin recurrent: batched decode ticks and a batched captured verify equal the legacy GDN batched path" {
    const a = testing.allocator;
    var p: Pair = undefined;
    try p.open(0xb47c, false);
    defer p.close();
    const s = fixture.stream();
    try testing.expect(p.xfm.supportsBatchedGdnDecode() and p.lg.xfm.supportsBatchedGdnDecode());
    // The legacy tick splits its merged states back after every tick, so each slot's states can be compared.
    p.lg.xfm.persistent_group_state_override = false;
    const xs = [_]*Transformer{ &p.lg.xfm, &p.xfm };
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
        for (outs[0], outs[1]) |want, got| try ops.expectBitEqual(got, want);
        for (&lanes[0], &lanes[1]) |*w, *gl| try fixture.expectEntries(w.ssm, gl.ssm);
    }
    {
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
        for (outs[0], outs[1]) |want, got| try ops.expectBitEqual(got, want);
        for (&lanes[0], &lanes[1]) |*w, *gl| try fixture.expectEntries(w.ssm, gl.ssm);
    }
}

test "plugin recurrent: an image prompt (M-RoPE table, media rows spliced across a chunk boundary) equals the legacy forward" {
    const a = testing.allocator;
    mrope.computeInvFreq(&inv_freq, 64, 100000.0);
    var p: Pair = undefined;
    try p.open(0x1a6e, true);
    defer p.close();
    const s = fixture.stream();
    try testing.expectEqual(@as(u32, 250), p.cfg.image_token_id);
    // 4 text tokens, a 2x3 image, 3 text tokens: the image rows sit on one t at h/w offsets, text after it resumes
    // past the image's widest axis.
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
    const rows = try g0.astype(try g0.adopt(try fixtureRows(6, 256, 9)), .bfloat16);
    const xs = [_]*Transformer{ &p.lg.xfm, &p.xfm };
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
        try ops.expectBitEqual(outs[1], outs[0]);
        try fixture.expectEntries(lanes[0].ssm, lanes[1].ssm);
    }
}

fn fixtureRows(n: c_int, h: c_int, seed: u64) !Array {
    var prng = std.Random.DefaultPrng.init(seed);
    const buf = try testing.allocator.alloc(f32, @intCast(n * h));
    defer testing.allocator.free(buf);
    for (buf) |*x| x.* = prng.random().floatNorm(f32) * 0.5;
    return mlx.mlx_array_new_data(buf.ptr, &[_]c_int{ 1, n, h }, 3, .float32);
}
