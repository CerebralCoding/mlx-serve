//! A served block-diffusion model's canvas loop against a test table over the llama plugin: the DiffusionGemma
//! runner prefills through `.prefill` forwards (its encoder pass), denoises through `.canvas` forwards at the
//! committed length with the self-conditioning signal from the second step on, never writes the cache from a canvas,
//! and re-encodes each committed block before the next. The canvas model's own forward is the family's to prove
//! against `forwardDiffusionDecoder`; this pins the engine's side of the contract.

const std = @import("std");
const abi = @import("../../../abi/plugin.zig");
const mlx = @import("../../../mlx/ffi.zig");
const llama = @import("../../../plugins/llama/plugin.zig");
const proof = @import("proof.zig");
const plugin_config = @import("config.zig");
const cache_ops = @import("../kv/cache_ops.zig");
const runner_mod = @import("../../../plugins/diffusion_gemma/runner.zig");
const KVCache = @import("../kv/cache.zig").KVCache;

const testing = std.testing;
const LAYERS = 2;

var seen: struct {
    prefill_rows: u32 = 0,
    canvases: u32 = 0,
    first_without_signal: u32 = 0,
    with_signal: u32 = 0,
    bad: ?[]const u8 = null,
} = .{};

fn note(why: []const u8) abi.Verdict {
    if (seen.bad == null) seen.bad = why;
    return abi.Verdict.refuse(.failed, "contract");
}

fn tForward(m: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) abi.Verdict {
    if (f.mode != .canvas) {
        seen.prefill_rows += f.rows;
        return llama.table.forward(m, f, out);
    }
    const co = f.cache_ops.?;
    const cache = f.caches.slice()[0];
    const held = co.len(cache, 0);
    if (f.positions.slice()[0] != held) return note("a canvas sits at the committed length");
    if (f.logits != .all_rows) return note("a canvas wants every row's logits");
    seen.canvases += 1;
    if (f.hidden_in.ctx == null) {
        seen.first_without_signal += 1;
    } else {
        const sh = mlx.getShape(.{ .ctx = f.hidden_in.ctx });
        if (sh.len != 3 or sh[1] != f.rows) return note("the signal is one row per canvas token");
        seen.with_signal += 1;
    }
    // The double's denoiser: the trunk over the canvas alone, on a scratch cache.
    var scratch = KVCache.init(testing.allocator, LAYERS) catch return note("scratch cache");
    defer scratch.deinit();
    var slot: cache_ops.Slot = .{ .kv = &scratch };
    const handles = [_]abi.Handle{slot.handle()};
    var g = f.*;
    g.mode = .prefill;
    g.positions = .of(&[_]u32{0});
    g.caches = .of(&handles);
    g.hidden_in = .{};
    const v = llama.table.forward(m, &g, out);
    if (co.len(cache, 0) != held) return note("a canvas never writes the cache");
    return v;
}

const test_table: abi.ModelTable = blk: {
    var t = llama.table;
    t.forward = &tForward;
    break :blk t;
};

// ── tests ──

test "plugin canvas: the parse's canvas facts make the served config a diffusion model" {
    var p: abi.Parsed = .{ .caps = .{ .block_diffusion = true }, .canvas = .{ .length = 32, .max_steps = 6, .entropy_bound = 0.3, .pad_token = 9 } };
    const c = plugin_config.modelConfig(&p);
    try testing.expect(c.isDiffusion());
    try testing.expectEqual(@as(u32, 32), c.canvas_length);
    try testing.expectEqual(@as(u32, 6), c.diffusion_max_steps);
    try testing.expectEqual(@as(f32, 0.3), c.diffusion_entropy_bound);
    try testing.expectEqual(@as(u32, 9), c.diffusion_pad_token);
    p.caps.block_diffusion = false;
    try testing.expect(!plugin_config.modelConfig(&p).isDiffusion());
}

test "plugin canvas: the runner encodes through prefill, denoises through canvas forwards and re-encodes each block" {
    const a = testing.allocator;
    var b = try proof.Both.open(.{ .quantized = true, .layers = LAYERS }, 0xca57);
    defer b.close();
    b.xfm.plugin.?.table = &test_table;
    b.xfm.config.plugin.?.parsed.caps.block_diffusion = true;
    b.xfm.config.canvas_length = 8;
    b.xfm.config.diffusion_max_steps = 3;
    b.xfm.config.diffusion_confidence_threshold = 0;
    seen = .{};
    var s: proof.Slot = undefined;
    try s.init(LAYERS);
    defer s.deinit();
    var runner = try runner_mod.Runner.init(a, &b.xfm, &s.ctx, 0.0, 16);
    defer runner.deinit();
    const prompt = [_]u32{ 3, 7, 1, 12, 30 };
    try runner.prefill(&prompt);
    try testing.expectEqual(@as(u32, prompt.len), seen.prefill_rows);
    var committed: usize = 0;
    while (try runner.nextCanvas(a)) |res| {
        defer a.free(res.tokens);
        committed += res.tokens.len;
        try testing.expectEqual(@as(usize, 8), res.tokens.len);
    }
    if (seen.bad) |why| {
        std.debug.print("canvas contract: {s}\n", .{why});
        return error.TestCanvasContract;
    }
    try testing.expectEqual(@as(usize, 16), committed);
    // Two blocks of three steps each; the second block's encode re-ran the first block through prefill.
    try testing.expectEqual(@as(u32, 6), seen.canvases);
    try testing.expectEqual(@as(u32, 2), seen.first_without_signal);
    try testing.expectEqual(@as(u32, 4), seen.with_signal);
    try testing.expectEqual(@as(u32, prompt.len + 8), seen.prefill_rows);
    try testing.expectEqual(@as(usize, prompt.len + 8), s.kv.step);
}
