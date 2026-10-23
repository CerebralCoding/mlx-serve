//! DiffusionGemma proven against the legacy engine on a tiny 4-bit pack (the 26B-A4B trunk of `gemma4/proof_test.zig`
//! under `model.decoder` with packed `gate_up_proj` experts, the self-conditioning module, the encoder's scalars):
//! the causal encoder passes (prompt and committed blocks) and the canvas steps (no signal, then a signal; the
//! committed context read from the cache with sliding layers trimmed) bit-equal to the engine's.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const fixture = @import("../../sdk/bases/proof/fixture.zig");
const proof = @import("../gemma3/proof_test.zig");
const g4proof = @import("../gemma4/proof_test.zig");
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;
const diffusion = @import("plugin.zig");
const runner_mod = @import("runner.zig");

const testing = std.testing;
const Array = mlx.mlx_array;

/// The legacy forward is the oracle, recorded to `fixtures/` (`MLX_SERVE_PROOF_RECORD=<repo root>`) until it is
/// deleted; then the proofs replay the recording.
const frozen = true;

fn oracle(comptime name: []const u8) proof.Oracle {
    const rel = "fixtures/" ++ name ++ ".tape";
    return if (frozen) .{ .tape = @embedFile(rel) } else .{ .legacy = "src/plugins/diffusion_gemma/" ++ rel };
}

const config_json =
    \\{"model_type":"diffusion_gemma","canvas_length":8,"eos_token_id":[1,106,50],"tie_word_embeddings":true,
    \\"generation_config":{"max_denoising_steps":48,"pad_token_id":0,"t_max":0.8,"t_min":0.4},
    \\"text_config":{"model_type":"diffusion_gemma_text","hidden_size":128,"intermediate_size":256,
    \\"moe_intermediate_size":128,"num_experts":8,"top_k_experts":2,"num_hidden_layers":4,"num_attention_heads":2,
    \\"num_key_value_heads":1,"num_global_key_value_heads":1,"head_dim":256,"global_head_dim":512,"vocab_size":256,
    \\"rms_norm_eps":1e-6,"sliding_window":4,"tie_word_embeddings":true,
    \\"layer_types":["sliding_attention","sliding_attention","full_attention","sliding_attention"],
    \\"rope_parameters":{"full_attention":{"rope_theta":1000000.0,"rope_type":"proportional","partial_rotary_factor":0.25},
    \\"sliding_attention":{"rope_theta":10000.0}},"final_logit_softcapping":30.0,"max_position_embeddings":4096},
    \\"quantization":{"group_size":64,"bits":4}}
;

fn writePack(p: *fixture.Pack) !void {
    return writePackAs(p, .{});
}

fn writePackAs(p: *fixture.Pack, q: g4proof.Quant) !void {
    try g4proof.writeGemma4Moe(p, q, "model.decoder", true);
    var buf: [128]u8 = undefined;
    try p.norm("model.decoder.self_conditioning.pre_norm.weight", g4proof.H);
    try p.quantize("model.decoder.self_conditioning.gate_proj", &.{ g4proof.INTER, g4proof.H }, q.mode, q.bits, q.group);
    try p.quantize("model.decoder.self_conditioning.up_proj", &.{ g4proof.INTER, g4proof.H }, q.mode, q.bits, q.group);
    try p.quantize("model.decoder.self_conditioning.down_proj", &.{ g4proof.H, g4proof.INTER }, q.mode, q.bits, q.group);
    for (0..4) |i| try p.dense(try std.fmt.bufPrint(&buf, "model.encoder.language_model.layers.{d}.layer_scalar", .{i}), &.{1}, 0.2, 0.9);
}

const Step = union(enum) { encode: []const i32, canvas: struct { ids: []const i32, signal: bool } };

test "diffusion_gemma plugin: encoder passes and canvas steps on a 4-bit pack are bit-equal to the legacy forward" {
    var p = fixture.Pack.init(0xd1f5);
    defer p.deinit();
    try writePack(&p);
    var b = try proof.Both.openAs(&diffusion.table, &p, config_json, oracle("steps"));
    defer b.close();
    if (b.lg) |lg| try testing.expect(lg.xfm.moe_layers != null);
    try expectSteps(&b);
    try b.expectDone();
}

test "diffusion_gemma plugin: an mxfp4 pack's encoder passes and canvas steps are bit-equal to the legacy forward" {
    const q: g4proof.Quant = .{ .mode = "mxfp4", .bits = 4, .group = 32 };
    var p = fixture.Pack.init(0xd1f7);
    defer p.deinit();
    try writePackAs(&p, q);
    const json = try std.mem.replaceOwned(u8, testing.allocator, config_json, "\"quantization\":{\"group_size\":64,\"bits\":4}", "\"quantization\":{\"group_size\":32,\"bits\":4,\"mode\":\"mxfp4\"}");
    defer testing.allocator.free(json);
    var b = try proof.Both.openAs(&diffusion.table, &p, json, oracle("steps-mxfp4"));
    defer b.close();
    try expectSteps(&b);
    try b.expectDone();
}

/// Encoder passes (prompt, committed block) and canvas steps (no signal, a signal) on caches of their own.
fn expectSteps(b: *proof.Both) !void {
    const n = b.cfg.num_hidden_layers;
    var xb: [2]*Transformer = undefined;
    const xs = b.sides(&xb);
    var slots: [2]proof.Slot = undefined;
    for (slots[0..xs.len]) |*s| try s.init(n);
    defer for (slots[0..xs.len]) |*s| s.deinit();
    const canvas = [_]i32{ 4, 0, 0, 9, 0, 17, 0, 0 };
    const steps = [_]Step{
        .{ .encode = &.{ 3, 17, 42, 99, 7, 250, 1, 8, 66, 31, 4, 5 } },
        .{ .canvas = .{ .ids = &canvas, .signal = false } },
        .{ .canvas = .{ .ids = &canvas, .signal = true } },
        .{ .encode = &.{ 4, 21, 33, 9, 70, 17, 2, 8 } },
        .{ .canvas = .{ .ids = &.{ 0, 0, 5, 0, 0, 0, 77, 0 }, .signal = true } },
    };
    var prng = std.Random.DefaultPrng.init(0x51c);
    for (steps) |st| {
        var g = ops.Graph.init(testing.allocator, b.xfm.s);
        defer g.deinit();
        var outs: [2]Array = undefined;
        switch (st) {
            .encode => |ids| for (xs, slots[0..xs.len], 0..) |x, *s, k| {
                s.ctx.use_encoder_scalars = true;
                defer s.ctx.use_encoder_scalars = false;
                outs[k] = try g.adopt(try x.forwardWith(&s.ctx, try fixture.tokens(&g, ids)));
            },
            .canvas => |cv| {
                const signal: ?Array = if (cv.signal) blk: {
                    var vals: [8 * g4proof.H]f32 = undefined;
                    for (&vals) |*v| v.* = prng.random().float(f32) * 0.2 - 0.1;
                    break :blk try g.astype(try g.fromSlice(f32, &vals, &.{ 1, 8, g4proof.H }), .bfloat16);
                } else null;
                for (xs, slots[0..xs.len], 0..) |x, *s, k|
                    outs[k] = try g.adopt(try x.forwardDiffusionDecoder(&s.ctx, try fixture.tokens(&g, cv.ids), signal));
            },
        }
        try b.check(outs[xs.len - 1], if (xs.len == 2) outs[0] else null, false);
    }
}

test "diffusion_gemma plugin: the canvas loop on a served shell commits the legacy tokens" {
    const a = testing.allocator;
    const json = try std.mem.replaceOwned(u8, a, config_json, "\"max_denoising_steps\":48", "\"max_denoising_steps\":4");
    defer a.free(json);
    var p = fixture.Pack.init(0xd1f6);
    defer p.deinit();
    try writePack(&p);
    var b = try proof.Both.openAs(&diffusion.table, &p, json, oracle("canvas-loop"));
    defer b.close();
    try testing.expect(b.xfm.plugin != null and b.xfm.config.isDiffusion());
    var out: [2]std.ArrayList(u32) = .{ .empty, .empty };
    defer for (&out) |*o| o.deinit(a);
    var xb: [2]*Transformer = undefined;
    const xs = b.sides(&xb);
    for (xs, out[0..xs.len]) |x, *o| {
        var s: proof.Slot = undefined;
        try s.init(b.cfg.num_hidden_layers);
        defer s.deinit();
        try mlx.check(mlx.mlx_random_seed(7));
        var runner = try runner_mod.Runner.init(a, x, &s.ctx, 0.0, 16);
        defer runner.deinit();
        try runner.prefill(&.{ 3, 17, 42, 99, 7, 250, 1, 8 });
        while (try runner.nextCanvas(a)) |res| {
            defer a.free(res.tokens);
            try o.appendSlice(a, res.tokens);
        }
    }
    const got = out[xs.len - 1].items;
    try testing.expectEqual(@as(usize, 16), got.len);
    try b.checkIds(got, if (xs.len == 2) out[0].items else null);
    try b.expectDone();
}

test "diffusion_gemma plugin: a pack the plugin declines is refused by name (no legacy forward to fall back to)" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const json = try std.mem.replaceOwned(u8, a, config_json, "\"num_experts\":8,", "");
    defer a.free(json);
    try testing.expect(json.len < config_json.len);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = json });
    var buf: [512]u8 = undefined;
    const dir = buf[0..try tmp.dir.realPath(io, &buf)];
    try testing.expectError(error.PluginDeclinedNoLegacy, @import("../../engines/mlx/plugin/config.zig").bindServed(io, a, dir, json));
}
