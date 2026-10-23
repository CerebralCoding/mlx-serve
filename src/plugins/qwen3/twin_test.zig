//! The Qwen family's parity harness: one tiny random pack on disk, loaded twice (the legacy `Transformer` and a
//! plugin's table over a second load of the same tensors), each forward step run on both sides with caches of their
//! own, logits and every recurrent state bit-equal. Shared by the qwen3 and qwen3_5 proofs.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const sdk = @import("../../sdk/root.zig");
const model = @import("../../engines/mlx/model.zig");
const fixture = @import("../../sdk/bases/proof/fixture.zig");
const cache_ops = @import("../../engines/mlx/kv/cache_ops.zig");
const KVCache = @import("../../engines/mlx/kv/cache.zig").KVCache;
const ForwardCtx = @import("../../engines/mlx/forward_ctx.zig").ForwardCtx;
const SSMCacheEntry = @import("../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const Resident = @import("../../sdk/weights/resident.zig").Resident;

const testing = std.testing;
const Array = mlx.mlx_array;

pub const Step = struct { mode: abi.Mode, ids: []const i32 };

/// A pack loaded through the legacy path and through plugin `P`'s table.
pub fn Twin(comptime P: type) type {
    return struct {
        lg: *fixture.Legacy,
        res: Resident,
        parsed: abi.Parsed = .{},
        m: *anyopaque,

        /// Writes `p` with `config_json` (and `files` beside it), loads both sides.
        pub fn open(p: *fixture.Pack, config_json: []const u8, files: []const sdk.fixture.File) !*@This() {
            const a = testing.allocator;
            const io = std.Io.Threaded.global_single_threaded.io();
            const t = try a.create(@This());
            errdefer a.destroy(t);
            t.lg = try p.load(config_json);
            errdefer t.lg.deinit();
            var buf: [512]u8 = undefined;
            const dir = buf[0..try t.lg.tmp.dir.realPath(io, &buf)];
            for (files) |f| try t.lg.tmp.dir.writeFile(io, .{ .sub_path = f.name, .data = f.bytes });
            var cfg = try model.parseConfigFromJson(a, config_json);
            defer cfg.deinit(a);
            t.res = Resident.adopt(try model.loadModelWeights(io, a, dir, &cfg, false));
            errdefer t.res.deinit();
            var all: [8]sdk.fixture.File = undefined;
            all[0] = .{ .name = "config.json", .bytes = config_json };
            @memcpy(all[1 .. 1 + files.len], files);
            const mem: sdk.fixture.MemDir = .{ .files = all[0 .. 1 + files.len] };
            const pd = mem.packDir();
            const v = P.table.parse(&pd, .of(config_json), .{}, null, false, &t.parsed);
            if (!v.isOk()) {
                std.debug.print("parse: {s}\n", .{v.why.slice()});
                return error.PluginParse;
            }
            errdefer P.table.free_parsed(&t.parsed);
            const src = t.res.source();
            var why: abi.Verdict = .{};
            t.m = P.table.load(&.{ .model_dir = .of(dir), .dir = &pd, .parsed = &t.parsed, .stream = .{ .ctx = t.lg.xfm.s.ctx }, .weights = &src }, &why) orelse {
                std.debug.print("load: {s}\n", .{why.why.slice()});
                return error.PluginLoad;
            };
            return t;
        }

        pub fn close(t: *@This()) void {
            P.table.unload(t.m);
            P.table.free_parsed(&t.parsed);
            t.res.deinit();
            t.lg.deinit();
            testing.allocator.destroy(t);
        }

        /// One request on both sides.
        pub fn lane(t: *@This()) !*Lane {
            return Lane.init(t.lg);
        }

        /// One step on `l`: the legacy forward and the table's, logits and states bit-equal.
        pub fn step(t: *@This(), l: *Lane, st: Step) !void {
            var g = ops.Graph.init(testing.allocator, t.lg.xfm.s);
            defer g.deinit();
            const ids = try fixture.tokens(&g, st.ids);
            l.ctx.capture_ssm_seq = st.mode == .verify;
            const want = try fixture.rows2(&g, try g.adopt(try t.lg.xfm.forwardWith(&l.ctx, ids)));
            const host = try testing.allocator.alloc(u32, st.ids.len);
            defer testing.allocator.free(host);
            for (st.ids, host) |x, *y| y.* = @intCast(x);
            const handles = [_]abi.Handle{l.slot.handle()};
            const pos = [_]u32{l.pos};
            l.pos += @intCast(st.ids.len);
            var out: abi.ForwardOut = .{};
            const v = P.table.forward(t.m, &.{ .mode = st.mode, .logits = .all_rows, .rows = @intCast(st.ids.len), .tokens = .of(host), .positions = .of(&pos), .caches = .of(&handles), .cache_ops = &cache_ops.ops }, &out);
            if (!v.isOk()) {
                std.debug.print("forward: {s}\n", .{v.why.slice()});
                return error.PluginForward;
            }
            const got = try g.adopt(.{ .ctx = out.logits.ctx });
            try ops.expectBitEqual(got, want);
            try l.expectStates();
        }

        /// Two lanes stepped together (`rows` tokens each): the legacy GDN batched driver (or the dense batched tick
        /// at one row) against the table's batched rows.
        pub fn batched(t: *@This(), lanes: [2]*Lane, tick: [2][]const i32, gdn: bool) !void {
            const a = testing.allocator;
            var g = ops.Graph.init(a, t.lg.xfm.s);
            defer g.deinit();
            const rows = tick[0].len;
            const ctxs = [_]*ForwardCtx{ &lanes[0].ctx, &lanes[1].ctx };
            const positions = [_]u32{ lanes[0].pos, lanes[1].pos };
            var flat: [16]i32 = undefined;
            var host: [16]u32 = undefined;
            @memcpy(flat[0..rows], tick[0]);
            @memcpy(flat[rows .. 2 * rows], tick[1]);
            for (flat[0 .. 2 * rows], host[0 .. 2 * rows]) |x, *y| y.* = @intCast(x);
            var want: Array = undefined;
            if (rows == 1) {
                const next = [_]u32{ host[0], host[1] };
                const legacy = if (gdn) try t.lg.xfm.forwardMoeBatchedDecode(&next, &ctxs, &positions, null) else try t.lg.xfm.forwardBatchedDecode(&next, &ctxs, &positions);
                defer a.free(legacy);
                for (legacy) |x| _ = try g.adopt(x);
                want = try g.concat(&.{ try fixture.rows2(&g, legacy[0]), try fixture.rows2(&g, legacy[1]) }, 0);
            } else {
                const ids = try g.fromSlice(i32, flat[0 .. 2 * rows], &.{ 2, @intCast(rows) });
                var hl = mlx.mlx_array_new();
                var ha = mlx.mlx_array_new();
                const logits = try t.lg.xfm.forwardMoeBatchedVerify(ids, &ctxs, &positions, &hl, &ha);
                _ = try g.adopt(hl);
                _ = try g.adopt(ha);
                want = try fixture.rows2(&g, try g.reshape(try g.adopt(logits), &.{ 1, 2 * @as(c_int, @intCast(rows)), -1 }));
            }
            const handles = [_]abi.Handle{ lanes[0].slot.handle(), lanes[1].slot.handle() };
            const slots = [_]u32{ 0, 1 };
            var out: abi.ForwardOut = .{};
            const v = P.table.forward(t.m, &.{ .mode = .batched_rows, .logits = .all_rows, .rows = @intCast(rows), .tokens = .of(host[0 .. 2 * rows]), .slots = .of(&slots), .positions = .of(&positions), .caches = .of(&handles), .cache_ops = &cache_ops.ops }, &out);
            if (!v.isOk()) {
                std.debug.print("batched forward: {s}\n", .{v.why.slice()});
                return error.PluginForward;
            }
            try ops.expectBitEqual(try g.adopt(.{ .ctx = out.logits.ctx }), want);
            for (lanes) |l| {
                l.offset += rows;
                l.pos += @intCast(rows);
                try l.expectStates();
            }
        }
    };
}

/// A request on both sides: the legacy forward's own cache and recurrent entries, and the plugin's.
pub const Lane = struct {
    legacy_kv: KVCache,
    legacy_ssm: []SSMCacheEntry,
    /// The layered forward's own position counter (it advances it on a solo step).
    offset: usize = 0,
    /// The request's next position.
    pos: u32 = 0,
    ctx: ForwardCtx = undefined,
    plugin_kv: KVCache,
    plugin_ssm: []SSMCacheEntry,
    slot: cache_ops.Slot = undefined,

    fn init(lg: *fixture.Legacy) !*Lane {
        const n = lg.config.num_hidden_layers;
        const l = try testing.allocator.create(Lane);
        l.* = .{ .legacy_kv = try KVCache.init(testing.allocator, n), .legacy_ssm = try fixture.newEntries(n), .plugin_kv = try KVCache.init(testing.allocator, n), .plugin_ssm = try fixture.newEntries(n) };
        l.ctx = lg.xfm.defaultCtx();
        l.ctx.cache = &l.legacy_kv;
        l.ctx.moe_seq_offset = &l.offset;
        l.ctx.ssm_entries = l.legacy_ssm;
        l.slot = .{ .kv = &l.plugin_kv, .ssm = l.plugin_ssm };
        return l;
    }

    pub fn deinit(l: *Lane) void {
        l.legacy_kv.deinit();
        l.plugin_kv.deinit();
        fixture.freeEntries(l.legacy_ssm);
        fixture.freeEntries(l.plugin_ssm);
        testing.allocator.destroy(l);
    }

    /// Every recurrent state and verify capture equal to the legacy forward's (a state sequence's last row aside:
    /// the verify kernels never write it).
    pub fn expectStates(l: *const Lane) !void {
        for (l.legacy_ssm, l.plugin_ssm) |*want, *got| {
            try testing.expectEqual(want.initialized, got.initialized);
            inline for (.{ "conv_state", "ssm_state", "spec_conv_input", "spec_state_seq" }) |f| {
                const wa = @field(want, f);
                const ga = @field(got, f);
                try testing.expectEqual(wa.ctx == null, ga.ctx == null);
                if (wa.ctx != null) {
                    var g = ops.Graph.init(testing.allocator, fixture.stream());
                    defer g.deinit();
                    const sh = ops.shape(wa);
                    if (sh.len == 5 and sh[0] > 1 and ops.shape(ga).len == 5)
                        try ops.expectBitEqual(try g.sliceAxis(ga, 0, 0, sh[0] - 1), try g.sliceAxis(wa, 0, 0, sh[0] - 1))
                    else
                        try ops.expectBitEqual(ga, wa);
                }
            }
        }
    }
};

/// `n` deterministic ids below 256.
pub fn seq(comptime n: usize, seed: u32) [n]i32 {
    var out: [n]i32 = undefined;
    for (&out, 0..) |*v, i| v.* = @intCast((seed + 37 * @as(u32, @intCast(i))) % 256);
    return out;
}
