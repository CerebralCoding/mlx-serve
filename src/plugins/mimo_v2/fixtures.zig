//! plugins/mimo_v2/fixtures: the served MiMo (this plugin behind the driver) against modeling_mimo_v2.py's logits on a
//! tiny random MiMo stored like the release (`tests/dump_mimo_v2_fixtures.py`, packed by `tests/convert_mimo_v2.py`).
//! Owns tests only; model-gated (`MIMO_V2_MODEL`, `MIMO_V2_FIXTURE`).

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const model_mod = @import("../../engines/mlx/model.zig");
const ForwardCtx = @import("../../engines/mlx/forward_ctx.zig").ForwardCtx;
const KVCache = @import("../../engines/mlx/kv/cache.zig").KVCache;
const testing = @import("../../engines/mlx/kv/cache.zig").testing;
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;
const driver = @import("../../engines/mlx/plugin/driver.zig");
const qwen4CompareRows = @import("../../sdk/testing/parity.zig").qwen4CompareRows;
const qwen4ReadF32 = @import("../../sdk/testing/parity.zig").qwen4ReadF32;
const mimoDecidedMisses = @import("../../sdk/testing/parity.zig").decidedMisses;

// ── tests ──

test "mimo_v2 batched decode: a slot in a two-slot tick tracks the fp32 reference as well as serial decode does (MIMO_V2_MODEL, MIMO_V2_FIXTURE)" {
    const model_dir = std.c.getenv("MIMO_V2_MODEL") orelse return error.SkipZigTest;
    const fixture_path = std.c.getenv("MIMO_V2_FIXTURE") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try model_mod.parseConfig(io, allocator, std.mem.span(model_dir));
    defer config.deinit(allocator);
    var xfm = try driver.load(io, allocator, &config, std.mem.span(model_dir), s);
    defer xfm.deinit();

    var fx = try model_mod.loadWeightsSingleFile(allocator, std.mem.span(fixture_path));
    defer fx.deinit();
    const ids_arr = fx.get("input_ids") orelse return error.MissingFixtureTensor;
    try mlx.check(mlx.mlx_array_eval(ids_arr));
    const T: usize = mlx.mlx_array_size(ids_arr);
    const ids = (mlx.mlx_array_data_int32(ids_arr) orelse return error.Unreadable)[0..T];
    const v: usize = @intCast(config.vocab_size);
    const ref = try qwen4ReadF32(allocator, fx.get("logits_full") orelse return error.MissingFixtureTensor, s);
    defer allocator.free(ref);

    // A MiMo slot owns a KV cache and a position, no recurrent state.
    const Slot = struct {
        cache: KVCache,
        off: usize = 0,
        ctx: ForwardCtx = undefined,
        fn init(sl: *@This(), n_layers: u32) !void {
            const cache = try KVCache.init(testing.allocator, n_layers);
            sl.* = .{ .cache = cache };
            sl.ctx = .{ .cache = &sl.cache, .moe_seq_offset = &sl.off, .ssm_entries = null, .capture_hidden = null, .vision_embeddings = null };
        }
        fn forward(sl: *@This(), x: *Transformer, toks: []const i32) !mlx.mlx_array {
            const arr = mlx.mlx_array_new_data(toks.ptr, &[_]c_int{ 1, @intCast(toks.len) }, 2, .int32);
            defer _ = mlx.mlx_array_free(arr);
            return x.forwardWith(&sl.ctx, arr);
        }
    };
    // Prompts of different length, decoded past the 8-token window. Slot A
    // decodes the fixture's own sequence, so every step has a reference row.
    const pa: usize = T / 2;
    const pb: usize = pa - 3;
    const K: usize = T - pa;
    const a_ids = ids[0 .. pa + K];
    const b_ids = ids[3 .. 3 + pb + K];
    var slots: [4]Slot = undefined; // serial A, serial B, batched A, batched B
    for (&slots) |*sl| try sl.init(config.num_hidden_layers);
    defer for (&slots) |*sl| sl.cache.deinit();
    for (&slots, [_][]const i32{ a_ids[0..pa], b_ids[0..pb], a_ids[0..pa], b_ids[0..pb] }) |*sl, p| _ = mlx.mlx_array_free(try sl.forward(&xfm, p));

    var cos_serial: f64 = 1.0;
    var cos_batched: f64 = 1.0;
    var misses: usize = 0;
    for (0..K) |k| {
        const ta = a_ids[pa + k];
        const tb = b_ids[pb + k];
        const la = try slots[0].forward(&xfm, &.{ta});
        defer _ = mlx.mlx_array_free(la);
        const lb = try slots[1].forward(&xfm, &.{tb});
        defer _ = mlx.mlx_array_free(lb);
        const ctxs = [_]*ForwardCtx{ &slots[2].ctx, &slots[3].ctx };
        const offs = [_]u32{ @intCast(slots[2].off), @intCast(slots[3].off) };
        const out = try xfm.forwardBatchedDecode(&.{ @intCast(ta), @intCast(tb) }, &ctxs, &offs);
        defer {
            for (out) |o| _ = mlx.mlx_array_free(o);
            allocator.free(out);
        }
        // The scheduler advances each slot's own position after the tick.
        slots[2].off += 1;
        slots[3].off += 1;
        const r = ref[(pa + k) * v ..][0..v];
        const sv = try qwen4ReadF32(allocator, la, s);
        defer allocator.free(sv);
        const bv = try qwen4ReadF32(allocator, out[0], s);
        defer allocator.free(bv);
        cos_serial = @min(cos_serial, qwen4CompareRows(sv, r, 1, v).min_cos);
        cos_batched = @min(cos_batched, qwen4CompareRows(bv, r, 1, v).min_cos);
        misses += mimoDecidedMisses(bv, r, 1, v);
    }
    std.debug.print("[mimo batched] {d} ticks: min cos vs fp32, serial {d:.5} batched {d:.5}; batched decided misses {d}\n", .{ K, cos_serial, cos_batched, misses });
    try testing.expect(cos_batched > cos_serial - 0.001 and misses == 0);
}

test "mimo_v2 fixture: one-shot and chunked prefill + decode vs modeling_mimo_v2.py (MIMO_V2_MODEL, MIMO_V2_FIXTURE)" {
    // A tiny random MiMo stored like the release (tests/dump_mimo_v2_fixtures.py), packed by
    // tests/convert_mimo_v2.py --bits 16; logits from the checkpoint's own code in f32.
    const model_dir = std.c.getenv("MIMO_V2_MODEL") orelse return error.SkipZigTest;
    const fixture_path = std.c.getenv("MIMO_V2_FIXTURE") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var config = try model_mod.parseConfig(io, allocator, std.mem.span(model_dir));
    defer config.deinit(allocator);
    var xfm = try driver.load(io, allocator, &config, std.mem.span(model_dir), s);
    defer xfm.deinit();

    var fx = try model_mod.loadWeightsSingleFile(allocator, std.mem.span(fixture_path));
    defer fx.deinit();
    const ids_arr = fx.get("input_ids") orelse return error.MissingFixtureTensor;
    try mlx.check(mlx.mlx_array_eval(ids_arr));
    const T: c_int = @intCast(mlx.mlx_array_size(ids_arr));
    const v: usize = @intCast(config.vocab_size);
    const ref = try qwen4ReadF32(allocator, fx.get("logits_full") orelse return error.MissingFixtureTensor, s);
    defer allocator.free(ref);

    const Run = struct {
        fn rows(x: *Transformer, ids: mlx.mlx_array, from: c_int, to: c_int, st: mlx.mlx_stream) !mlx.mlx_array {
            var part = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(part);
            try mlx.check(mlx.mlx_slice(&part, ids, &.{from}, 1, &.{to}, 1, &.{1}, 1, st));
            var batched = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(batched);
            try mlx.check(mlx.mlx_reshape(&batched, part, &.{ 1, to - from }, 2, st));
            return x.forward(batched);
        }
    };

    // [a] the whole prompt in one forward: every row, sliding window included.
    const full = try Run.rows(&xfm, ids_arr, 0, T, s);
    defer _ = mlx.mlx_array_free(full);
    const ours_full = try qwen4ReadF32(allocator, full, s);
    defer allocator.free(ours_full);
    const a = qwen4CompareRows(ours_full, ref, @intCast(T), v);
    std.debug.print("[mimo fixture] one-shot {d} rows: min cos {d:.5}, argmax {d}/{d}\n", .{ a.rows, a.min_cos, a.argmax_agree, a.rows });
    try testing.expect(a.min_cos > 0.995 and mimoDecidedMisses(ours_full, ref, @intCast(T), v) == 0);

    // [b] a chunked prefill split across the window, then single-token decode steps.
    try xfm.resetCache();
    const ref_step = try qwen4ReadF32(allocator, fx.get("logits_step").?, s);
    defer allocator.free(ref_step);
    const t_pre: c_int = T - 6;
    const chunks = [_][2]c_int{ .{ 0, 13 }, .{ 13, t_pre } };
    for (chunks) |c| {
        const out = try Run.rows(&xfm, ids_arr, c[0], c[1], s);
        _ = mlx.mlx_array_free(out);
    }
    var worst: f64 = 1.0;
    var agree: usize = 0;
    var t = t_pre;
    while (t < T) : (t += 1) {
        const out = try Run.rows(&xfm, ids_arr, t, t + 1, s);
        defer _ = mlx.mlx_array_free(out);
        const row = try qwen4ReadF32(allocator, out, s);
        defer allocator.free(row);
        const r = qwen4CompareRows(row, ref[@as(usize, @intCast(t)) * v ..], 1, v);
        worst = @min(worst, r.min_cos);
        agree += r.argmax_agree;
    }
    std.debug.print("[mimo fixture] chunked prefill + {d} decode steps: min cos {d:.5}, argmax {d}/{d}\n", .{ T - t_pre, worst, agree, T - t_pre });
    try testing.expect(worst > 0.995 and agree == @as(usize, @intCast(T - t_pre)));
}
