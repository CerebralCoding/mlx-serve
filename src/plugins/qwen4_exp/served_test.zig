//! The served path end to end: a tiny qwen4_exp pack with its MTP head loaded as the scheduler loads it, once on the
//! legacy path and once through the plugin table behind the driver's shell (`parseConfig`, `driver.load`, the MTP head
//! on the plugin's inner trunk through `plugin/intree.zig`). Captured steps, the pipelined greedy decode (deferred
//! n-gram PLE settled through `forward_settle`), batched rows, a batched verify and MTP rounds (pinned and auto depth)
//! emit the same tokens, accepted counts and logits bit for bit, with the host and the GPU n-gram gather.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const model = @import("../../engines/mlx/model.zig");
const driver = @import("../../engines/mlx/plugin/driver.zig");
const intree = @import("../../engines/mlx/plugin/intree.zig");
const depth_bounds = @import("../../engines/mlx/spec/mtp_depth_bounds.zig");
const lazyForward = @import("../../engines/mlx/generate/step.zig").lazyForward;
const Generator = @import("../../engines/mlx/generate/generator.zig").Generator;
const MtpHeadRef = @import("../../engines/mlx/spec/mtp_head_ref.zig").MtpHeadRef;
const Tokenizer = @import("../../host/tokenizer/tokenizer.zig").Tokenizer;
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;
const ForwardCtx = @import("../../engines/mlx/forward_ctx.zig").ForwardCtx;
const KVCache = @import("../../engines/mlx/kv/cache.zig").KVCache;
const SSMCacheEntry = @import("../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const nextSsmMemberGen = @import("../../engines/mlx/ssm/group.zig").nextSsmMemberGen;
const ssmFreeQsaState = @import("qsa/history.zig").ssmFreeQsaState;
const qwen4 = @import("plugin.zig");
const ple_gpu = @import("ple_gpu.zig");
const fixture = @import("fixture.zig");

const testing = std.testing;
const Array = mlx.mlx_array;

/// The pack loaded both ways, each with the steps the scheduler's load runs after the trunk.
const Pair = struct {
    tmp: testing.TmpDir,
    lcfg: model.ModelConfig,
    weights: model.Weights,
    legacy: Transformer,
    scfg: model.ModelConfig,
    served: Transformer,

    fn open(precision: fixture.Precision, seed: u64) !*Pair {
        return openWith(precision, seed, .{});
    }

    fn openWith(precision: fixture.Precision, seed: u64, opts: fixture.Opts) !*Pair {
        const a = testing.allocator;
        const io = testing.io;
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        const files = try fixture.filesWith(arena.allocator(), seed, precision, opts);
        const p = try a.create(Pair);
        errdefer a.destroy(p);
        p.tmp = testing.tmpDir(.{});
        errdefer p.tmp.cleanup();
        for (files) |f| try p.tmp.dir.writeFile(io, .{ .sub_path = f.name, .data = f.bytes });
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const dir = buf[0..try p.tmp.dir.realPath(io, &buf)];
        const s = mlx.gpuStream();

        p.lcfg = try model.parseConfigLegacy(io, a, dir);
        errdefer p.lcfg.deinit(a);
        p.weights = try model.loadModelWeights(io, a, dir, &p.lcfg, false);
        errdefer p.weights.deinit();
        model.resolveWeightPrefix(&p.lcfg, &p.weights);
        p.legacy = try Transformer.init(io, a, p.lcfg, &p.weights);
        errdefer p.legacy.deinit();
        qwen4.prepare(&p.legacy);

        p.scfg = try model.parseConfig(io, a, dir);
        if (p.scfg.plugin == null) return error.NotServed;
        errdefer p.scfg.deinit(a);
        p.served = try driver.load(io, a, &p.scfg, dir, s);
        errdefer p.served.deinit();
        if (intree.inner(&p.served) == null) return error.NoInnerTrunk;

        for ([_]*Transformer{ &p.legacy, p.served.qwen4Host() }) |x| {
            try x.qwen4MtpApplyKvQuant(.dense);
            _ = x.qwen4BuildDraftRerank();
        }
        return p;
    }

    fn close(p: *Pair) void {
        p.served.deinit();
        p.scfg.deinit(testing.allocator);
        p.legacy.deinit();
        p.weights.deinit();
        p.lcfg.deinit(testing.allocator);
        p.tmp.cleanup();
        testing.allocator.destroy(p);
    }

    fn both(p: *Pair) [2]*Transformer {
        return .{ &p.legacy, &p.served };
    }
};

/// A slot's engine caches and context.
const Slot = struct {
    kv: KVCache,
    ssm: []SSMCacheEntry,
    off: usize = 0,
    ctx: ForwardCtx = undefined,

    fn init(s: *Slot, layers: u32) !void {
        const ssm = try testing.allocator.alloc(SSMCacheEntry, layers);
        const gen = nextSsmMemberGen();
        for (ssm) |*e| e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false, .member_gen = gen };
        s.* = .{ .kv = try KVCache.init(testing.allocator, layers), .ssm = ssm };
        s.ctx = .{ .cache = &s.kv, .moe_seq_offset = &s.off, .ssm_entries = s.ssm, .ssm_member_gen = gen, .capture_hidden = null, .vision_embeddings = null };
    }

    fn deinit(s: *Slot, x: *Transformer) void {
        x.discardDeferredPle(&s.ctx);
        @import("../../engines/mlx/plugin/settle.zig").release(x, &s.ctx);
        x.qwen4Host().ssmGroupDrop(s.ssm);
        for (s.ssm) |*e| {
            _ = mlx.mlx_array_free(e.conv_state);
            _ = mlx.mlx_array_free(e.ssm_state);
            ssmFreeQsaState(e);
        }
        testing.allocator.free(s.ssm);
        s.kv.deinit();
    }
};

fn tokens(g: *ops.Graph, ids: []const i32) !Array {
    return g.adopt(mlx.mlx_array_new_data(ids.ptr, &[_]c_int{ 1, @intCast(ids.len) }, 2, .int32));
}

/// A request's emitted tokens and, with MTP, each round's accepted count.
const Run = struct { tokens: [24]u32 = @splat(0), n: usize = 0, accepted: std.ArrayList(u32) = .empty };

fn generate(x: *Transformer, head: ?MtpHeadRef, run: *Run) !void {
    const a = testing.allocator;
    try x.resetCache();
    var tok: Tokenizer = undefined; // never read by the Generator
    const prompt = [_]u32{ 3, 7, 1, 12, 30, 5, 9, 22, 4, 17, 2, 28, 40, 41, 3, 7 };
    var gen = try Generator.initWithOptions(testing.io, a, x, &tok, &prompt, run.tokens.len, .{ .temperature = 0.0 }, &.{}, .{
        .mtp_enabled = head != null,
        .mtp = head,
        .model_has_mtp = head != null,
    });
    defer gen.deinit(a);
    if (head == null) {
        while (run.n < run.tokens.len) : (run.n += 1) run.tokens[run.n] = (try gen.next(a)) orelse break;
        return;
    }
    while (try gen.nextMtp(a)) |res| {
        defer a.free(res.tokens);
        @memcpy(run.tokens[run.n..][0..res.tokens.len], res.tokens);
        run.n += res.tokens.len;
        try run.accepted.append(a, res.accepted_tokens);
    }
    try testing.expect(gen.mtp_attempted > 0);
}

/// Returns the drafts the legacy rounds accepted.
fn expectSameRuns(p: *Pair, mtp: bool) !u32 {
    const xs = p.both();
    const heads = [_]?MtpHeadRef{ if (mtp) .{ .qwen4 = &p.legacy } else null, if (mtp) .{ .qwen4 = p.served.qwen4Host() } else null };
    var runs: [2]Run = .{ .{}, .{} };
    defer for (&runs) |*r| r.accepted.deinit(testing.allocator);
    for (xs, heads, &runs) |x, h, *r| try generate(x, h, r);
    try testing.expect(runs[0].n > 1);
    try testing.expectEqual(runs[0].n, runs[1].n);
    try testing.expectEqualSlices(u32, &runs[0].tokens, &runs[1].tokens);
    try testing.expectEqualSlices(u32, runs[0].accepted.items, runs[1].accepted.items);
    var sum: u32 = 0;
    for (runs[0].accepted.items) |n| sum += n;
    return sum;
}

// ── tests ──

test "qwen4_exp served: the plugin claims the pack and the engine config is the legacy parse" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const p = try Pair.open(.q4, 31);
    defer p.close();
    var cfg = p.scfg;
    cfg.plugin = null;
    try testing.expectEqualDeep(p.lcfg, cfg);
    try testing.expect(p.served.plugin != null and p.served.qwen4Host().qwen4_mtp != null);
    // load.zig compiles nothing on a served shell, and every test here forwards through shells like this one.
    const x = &p.served;
    try testing.expect(x.compiled_gdn_gate == null and x.compiled_moe_routing == null and x.compiled_hc_mix == null and x.compiled_gelu == null and x.compiled_softcap == null);
}

test "qwen4_exp served: captured steps and a pipelined greedy decode equal the legacy path bit for bit" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    for ([_]fixture.Precision{ .bf16, .q4 }) |prec| {
        const p = try Pair.open(prec, 41);
        defer p.close();
        const xs = p.both();
        var slots: [2]Slot = undefined;
        for (&slots) |*s| try s.init(p.lcfg.num_hidden_layers);
        defer for (&slots, xs) |*s, x| s.deinit(x);
        const steps = [_]struct { ids: []const i32, verify: bool = false }{
            .{ .ids = &.{ 3, 17, 42, 99, 7, 50, 1, 8, 66, 31 } },
            .{ .ids = &.{88} },
            .{ .ids = &.{ 5, 61, 100 }, .verify = true },
            .{ .ids = &.{ 9, 33, 77, 120, 2, 18, 19, 20, 21 } },
            .{ .ids = &.{64} },
        };
        for (steps) |st| {
            var g = ops.Graph.init(a, mlx.gpuStream());
            defer g.deinit();
            var outs: [2][2]Array = undefined;
            for (xs, &slots, 0..) |x, *s, k| {
                var last = mlx.mlx_array_new();
                var all = mlx.mlx_array_new();
                s.ctx.capture_ssm_seq = st.verify;
                s.ctx.spec_verify = st.verify;
                defer {
                    s.ctx.capture_ssm_seq = false;
                    s.ctx.spec_verify = false;
                }
                const l = try g.adopt(try x.forwardWithCaptureAll(&s.ctx, try tokens(&g, st.ids), &last, &all));
                _ = try g.adopt(last);
                outs[k] = .{ l, try g.adopt(all) };
            }
            for (outs[0], outs[1]) |want, got| try ops.expectBitEqual(got, want);
        }
        // The pipelined step: each next token a lazy argmax, the n-gram PLE deferred past the build.
        var toks: [2][6]u32 = undefined;
        for (xs, &slots, 0..) |x, *s, k| {
            var g = ops.Graph.init(a, mlx.gpuStream());
            defer g.deinit();
            var logits = try g.adopt(try x.forwardWith(&s.ctx, try tokens(&g, &.{ 11, 12 })));
            for (&toks[k]) |*t| {
                const sh = ops.shape(logits);
                const next = try g.argmax(try g.reshape(try g.sliceAxis(logits, 1, sh[1] - 1, sh[1]), &.{ 1, sh[2] }), -1, false);
                logits = try g.adopt(try lazyForward(x, &s.ctx, next));
                const host = try g.astype(next, .uint32);
                try mlx.check(mlx.mlx_array_eval(host));
                t.* = mlx.mlx_array_data_uint32(host).?[0];
            }
            try x.flushDeferredPle(&s.ctx);
        }
        try testing.expectEqualSlices(u32, &toks[0], &toks[1]);
    }
}

test "qwen4_exp served: batched rows and a batched verify equal the legacy tick per slot" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const p = try Pair.open(.q4, 43);
    defer p.close();
    const xs = p.both();
    var slots: [2][2]Slot = undefined;
    for (&slots) |*pair| for (pair) |*s| try s.init(p.lcfg.num_hidden_layers);
    defer for (&slots, xs) |*pair, x| for (pair) |*s| s.deinit(x);
    const prompts = [_][]const i32{ &.{ 4, 8, 15, 16, 23, 42, 9, 10, 11 }, &.{ 42, 101, 7, 1, 2 } };
    for (prompts, 0..) |pr, i| {
        var g = ops.Graph.init(a, mlx.gpuStream());
        defer g.deinit();
        var outs: [2]Array = undefined;
        for (xs, 0..) |x, k| outs[k] = try g.adopt(try x.forwardWith(&slots[k][i].ctx, try tokens(&g, pr)));
        try ops.expectBitEqual(outs[1], outs[0]);
    }
    for ([_][2]u32{ .{ 77, 3 }, .{ 100, 64 }, .{ 11, 12 } }) |next| {
        var g = ops.Graph.init(a, mlx.gpuStream());
        defer g.deinit();
        var rows: [2][]Array = undefined;
        var hidden: [2]?[]Array = .{ null, null };
        for (xs, 0..) |x, k| {
            const ctxs = [_]*ForwardCtx{ &slots[k][0].ctx, &slots[k][1].ctx };
            const offs = [_]u32{ @intCast(slots[k][0].off), @intCast(slots[k][1].off) };
            rows[k] = try x.forwardMoeBatchedDecode(&next, &ctxs, &offs, &hidden[k]);
            for (rows[k]) |r| _ = try g.adopt(r);
            for (hidden[k].?) |h| _ = try g.adopt(h);
            for (&slots[k]) |*s| s.off += 1;
        }
        defer for (rows, hidden) |r, h| {
            a.free(r);
            a.free(h.?);
        };
        for (rows[0], rows[1], hidden[0].?, hidden[1].?) |w, got, wh, gh| {
            try ops.expectBitEqual(got, w);
            try ops.expectBitEqual(gh, wh);
        }
    }
    var g = ops.Graph.init(a, mlx.gpuStream());
    defer g.deinit();
    var outs: [2][3]Array = undefined;
    const ids = [_]i32{ 5, 6, 7, 8, 9, 10 };
    for (xs, 0..) |x, k| {
        const ctxs = [_]*ForwardCtx{ &slots[k][0].ctx, &slots[k][1].ctx };
        const offs = [_]u32{ @intCast(slots[k][0].off), @intCast(slots[k][1].off) };
        var last = mlx.mlx_array_new();
        var all = mlx.mlx_array_new();
        const arr = try g.adopt(mlx.mlx_array_new_data(&ids, &[_]c_int{ 2, 3 }, 2, .int32));
        const l = try g.adopt(try x.forwardMoeBatchedVerify(arr, &ctxs, &offs, &last, &all));
        outs[k] = .{ l, try g.adopt(last), try g.adopt(all) };
    }
    for (outs[0], outs[1]) |want, got| try ops.expectBitEqual(got, want);
}

test "qwen4_exp served: greedy decode and MTP rounds emit the legacy tokens and accepted counts" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    // A random head drafts what the trunk rejects (every round rolls back); a sign-reading lm_head makes the head
    // agree often, so rounds accept all, some or none of their drafts.
    // The GPU n-gram gather (`--ple-gpu`) leaves the history owed past each decode step (`Owed.history`).
    defer ple_gpu.enabled = false;
    for ([_]u64{ 47, 51, 51 }, [_]fixture.Opts{ .{}, .{ .sign_head = true }, .{ .sign_head = true } }, [_]bool{ false, false, true }) |seed, opts, gpu| {
        ple_gpu.enabled = gpu;
        const p = try Pair.openWith(.q4, seed, opts);
        if (gpu) try testing.expect(p.legacy.qwen4.?.gpu != null and p.served.qwen4Host().qwen4.?.gpu != null);
        defer p.close();
        _ = try expectSameRuns(p, false);
        const auto = try expectSameRuns(p, true);
        depth_bounds.active = .{ .min = 2, .max = 2 };
        defer depth_bounds.active = .{};
        const pinned = try expectSameRuns(p, true);
        if (opts.sign_head) try testing.expect(auto > 0 and pinned > 0);
    }
}

test "qwen4_exp served: an image prompt (M-RoPE table, rows spliced across a chunk boundary) equals the legacy path" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const p = try Pair.openWith(.q4, 53, .{ .vision = true });
    defer p.close();
    try testing.expectEqual(@as(u32, fixture.image_id), p.scfg.image_token_id);
    try testing.expect(p.scfg.plugin.?.parsed.mrope.rope_dims == 8);
    // 4 text tokens, a 2x3 image, 6 text tokens: past the image the positions resume after its widest axis.
    const T = 16;
    const img = fixture.image_id;
    const prompt = [T]i32{ 3, 17, 42, 99, img, img, img, img, img, img, 7, 8, 9, 10, 11, 12 };
    var table: [3 * T]i32 = undefined;
    for (0..T) |i| {
        const in_img = i >= 4 and i < 10;
        const r: i32 = if (in_img) @intCast((i - 4) / 3) else 0;
        const c: i32 = if (in_img) @intCast((i - 4) % 3) else 0;
        const t: i32 = if (i < 4) @intCast(i) else if (in_img) 4 else @intCast(i - 10 + 7);
        table[i] = t;
        table[T + i] = if (in_img) t + r else t;
        table[2 * T + i] = if (in_img) t + c else t;
    }
    var prng = std.Random.DefaultPrng.init(9);
    var buf: [6 * 64]f32 = undefined;
    for (&buf) |*x| x.* = prng.random().floatNorm(f32) * 0.5;
    var g0 = ops.Graph.init(a, mlx.gpuStream());
    defer g0.deinit();
    const rows = try g0.astype(try g0.adopt(mlx.mlx_array_new_data(&buf, &[_]c_int{ 1, 6, 64 }, 3, .float32)), .bfloat16);
    const xs = p.both();
    var slots: [2]Slot = undefined;
    for (&slots) |*s| {
        try s.init(p.lcfg.num_hidden_layers);
        s.ctx.mrope_pos = &table;
        s.ctx.mrope_total = T;
        s.ctx.mrope_delta = 13 - T;
        s.ctx.vision_embeddings = rows;
    }
    defer for (&slots, xs) |*s, x| s.deinit(x);
    const chunks = [_]struct { ids: []const i32, offset: usize = 0, verify: bool = false }{
        .{ .ids = prompt[0..7] },
        .{ .ids = prompt[7..], .offset = 3 },
        .{ .ids = &.{88} },
        .{ .ids = &.{ 5, 61, 100 }, .verify = true },
        .{ .ids = &.{12} },
    };
    for (chunks) |ch| {
        var g = ops.Graph.init(a, mlx.gpuStream());
        defer g.deinit();
        const t = try tokens(&g, ch.ids);
        var outs: [2]Array = undefined;
        for (&slots, xs, &outs) |*s, x, *o| {
            s.ctx.vision_splice_offset = ch.offset;
            s.ctx.capture_ssm_seq = ch.verify;
            s.ctx.spec_verify = ch.verify;
            defer {
                s.ctx.capture_ssm_seq = false;
                s.ctx.spec_verify = false;
            }
            o.* = try g.adopt(try x.forwardWith(&s.ctx, t));
        }
        try ops.expectBitEqual(outs[1], outs[0]);
    }
}
