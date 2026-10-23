//! sdk/blocks/rope/yarn, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns yarnActive, yarnScaleQK, yarnScaleRotated, yarnSpec, yarnMscale, computeYarnFreqs.

const std = @import("std");
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
const mrope = @import("../../vision/mrope.zig");
const ModelConfig = @import("../../../engines/mlx/model.zig").ModelConfig;
const testing = @import("../../../engines/mlx/kv/cache.zig").testing;
const scalarOf = @import("../../../engines/mlx/load/weights.zig").scalarOf;
const Transformer = @import("../../../engines/mlx/transformer.zig").Transformer;
const constTableAs = @import("../../../mlx/dtype.zig").constTableAs;
const jsonF64 = @import("../../testing/parity.zig").jsonF64;
const computeLlama3Freqs = @import("rope.zig").computeLlama3Freqs;

/// Does this trunk rotate with a YaRN-scaled rope? The DECISION comes from
/// the config — every Transformer has one, including the hand-built fixtures
/// in the tests, which leave the optionals below undefined. The short-circuit
/// means an unscaled model never reads them.
pub fn yarnActive(self: *const Transformer) bool {
    return self.config.rope_yarn and self.rope_freqs_yarn != null;
}

/// YaRN mscale on rope'd q and k: multiply by the [head_dim] table
/// (attention_factor on the rotated slice, 1.0 on the pass-through tail).
/// That is the reference's `cos, sin = cos*af, sin*af` — every rotary pair
/// shares one frequency and both of its dims carry it, so scaling the
/// rotation and scaling q/k are the same product. The table MUST be read in
/// the tensors' dtype: an f32 operand promotes the result, and the f32 then
/// rides SDPA → o_proj → residual, upcasting every weight read downstream.
pub fn yarnScaleQK(self: *Transformer, q: *mlx.mlx_array, k: *mlx.mlx_array) !void {
    const ms = try constTableAs(self.yarn_mscale.?, mlx.mlx_array_dtype(q.*), &self.yarn_mscale_cast, self.s);
    var qs = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&qs, q.*, ms, self.s));
    _ = mlx.mlx_array_free(q.*);
    q.* = qs;
    var ks = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&ks, k.*, ms, self.s));
    _ = mlx.mlx_array_free(k.*);
    k.* = ks;
}

/// YaRN mscale on a rope'd array whose last dim may not be `head_dim`
/// (the QSA indexer is 128-wide / 64 rotated). Slice the first `rope_dims`,
/// multiply by a same-dtype scalar, concat the pass-through tail. NEVER
/// `yarnScaleQK` here: that table is `[head_dim]` (256 on qwen4).
pub fn yarnScaleRotated(self: *Transformer, q: *mlx.mlx_array, rope_dims: c_int) !void {
    const sh = mlx.getShape(q.*);
    const dt = mlx.mlx_array_dtype(q.*);
    const st = [_]c_int{ 1, 1, 1, 1 };
    var rot = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rot);
    try mlx.check(mlx.mlx_slice(&rot, q.*, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ sh[0], sh[1], sh[2], rope_dims }, 4, &st, 4, self.s));
    var pass = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pass);
    try mlx.check(mlx.mlx_slice(&pass, q.*, &[_]c_int{ 0, 0, 0, rope_dims }, 4, &[_]c_int{ sh[0], sh[1], sh[2], sh[3] }, 4, &st, 4, self.s));
    const ms = try scalarOf(self.config.yarn_attention_factor, dt, self.s);
    defer _ = mlx.mlx_array_free(ms);
    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    try mlx.check(mlx.mlx_multiply(&scaled, rot, ms, self.s));
    const parts = [_]mlx.mlx_array{ scaled, pass };
    const vec = mlx.mlx_vector_array_new_data(&parts, 2);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, -1, self.s));
    _ = mlx.mlx_array_free(q.*);
    q.* = out;
}

/// This checkpoint's YaRN spectrum: the config's rope geometry (theta, the
/// rotating slice, the pre-trained window) resolved into `mrope.Yarn`, the one
/// implementation of HF's `_compute_yarn_parameters` in the engine. Only
/// meaningful when `config.rope_yarn`.
pub fn yarnSpec(config: *const ModelConfig) mrope.Yarn {
    return .{
        .theta = config.rope_theta,
        .factor = config.yarn_factor,
        .orig_max = config.yarn_orig_max_pos,
        .rotary_dim = config.yarnRotaryDim(),
        .beta_fast = config.yarn_beta_fast,
        .beta_slow = config.yarn_beta_slow,
        .truncate = config.yarn_truncate,
    };
}

/// The mscale the host cos/sin tables are multiplied by: HF's `attention_factor`
/// when the checkpoint scales its rope, 1.0 (a no-op) when it does not.
pub fn yarnMscale(config: *const ModelConfig) f64 {
    return if (config.rope_yarn) config.yarn_attention_factor else 1.0;
}

/// HF `_compute_yarn_parameters` → the per-dim RoPE DENOMINATOR array (`freqs`)
/// that mlx_fast_rope consumes: mlx computes angle = position / freqs[i], so we
/// return 1/inv_freq where inv_freq is the YaRN-corrected inverse frequency.
/// `out.len` must equal the rotary half-dim (= int(head_dim * partial) / 2).
/// Pure f64 math (no MLX) so it unit-tests directly against reference values.
/// The spectrum itself lives in `mrope.Yarn` — shared with the host cos/sin
/// tables, which need inv_freq and not the reciprocal — so the device and host
/// rope paths can never drift apart. `truncate` is the HF flag (laguna's
/// rope_parameters omits it → true).
pub fn computeYarnFreqs(
    out: []f64,
    head_dim: u32,
    partial: f32,
    base: f64,
    factor: f64,
    beta_fast: f64,
    beta_slow: f64,
    orig_max: u32,
    truncate: bool,
) void {
    const spec = mrope.Yarn{
        .theta = base,
        .factor = factor,
        .orig_max = orig_max,
        .rotary_dim = @intFromFloat(@floor(@as(f64, @floatFromInt(head_dim)) * @as(f64, partial))),
        .beta_fast = beta_fast,
        .beta_slow = beta_slow,
        .truncate = truncate,
    };
    spec.invFreq(out);
    for (out) |*v| v.* = 1.0 / v.*;
}

// ── tests ──

test "qwen4_exp yarn: the device denominator and the host spectrum are one spectrum" {
    // The engine spells the SAME YaRN spectrum twice: mlx_fast_rope and the
    // fused-angle probe take a DENOMINATOR array (angle = position / freqs),
    // while the host cos/sin tables (`fillCosSin`, `ropeAtPositions`) take
    // inv_freq (angle = position × inv_freq). If the two ever drift, prefill and
    // decode rotate differently — the model would hold one position for the
    // prompt and another for everything generated after it, and nothing errors.
    var c = ModelConfig{};
    c.model_type = "qwen4_exp"; // isQwen4 ⇒ yarnPartial reads partial_rotary_factor
    c.head_dim = 256;
    c.partial_rotary_factor = 0.25;
    c.partial_rotary_factor_global = 1.0;
    c.rope_theta = 10_000_000.0;
    c.rope_yarn = true;
    c.yarn_factor = 4.0;
    c.yarn_orig_max_pos = 262_144;
    c.yarn_attention_factor = 1.138629436111989;
    c.yarn_beta_fast = 32.0;
    c.yarn_beta_slow = 1.0;
    c.max_position_embeddings = 1_048_576;

    // qwen4's ONE trunk rope: 64 of 256 dims, 32 frequencies — which is also
    // the width the QSA indexer rotates (it takes rope_dims from head_dim, not
    // from indexer_head_dim), so a single table serves every consumer.
    try testing.expectEqual(@as(u32, 64), c.yarnRotaryDim());
    try testing.expectApproxEqAbs(@as(f32, 0.25), c.yarnPartial(), 1e-9);
    try testing.expectEqual(@as(u32, 1_048_576), c.contextCap());

    var inv: [32]f64 = undefined;
    yarnSpec(&c).invFreq(&inv);
    var den: [32]f64 = undefined;
    computeYarnFreqs(
        &den,
        c.head_dim,
        c.yarnPartial(),
        c.rope_theta,
        c.yarn_factor,
        c.yarn_beta_fast,
        c.yarn_beta_slow,
        c.yarn_orig_max_pos,
        c.yarn_truncate,
    );
    for (inv, den) |a, b| try testing.expectApproxEqRel(@as(f64, 1.0), a * b, 1e-12);
    // Wavelengths run from one token (the fastest rotary dim) to the whole
    // stretched window (the slowest), strictly increasing — the shape every
    // consumer assumes when it indexes the table by dim.
    try testing.expectApproxEqRel(@as(f64, 1.0), den[0], 1e-12);
    for (den[1..], 1..) |b, i| try testing.expect(b > den[i - 1]);
    // The table's width is what the composed path multiplies: attention_factor
    // across exactly the first `yarnRotaryDim` dims of the head, 1.0 across the
    // 192 pass-through dims (scaling those would change values RoPE never wrote).
    const rotary = c.yarnRotaryDim();
    try testing.expectEqual(@as(usize, rotary), inv.len * 2);
    try testing.expect(rotary < c.head_dim); // partial rotary ⇒ a real pass-through tail
    try testing.expect(c.yarn_attention_factor > 1.0); // factor 4 ⇒ mscale 1.1386…

    // And the laguna geometry the same helper serves must not flip: it scales
    // its FULL-attention layers only, at `partial_rotary_factor_global`.
    var l = ModelConfig{};
    l.model_type = "laguna";
    l.head_dim = 128;
    l.partial_rotary_factor = 1.0;
    l.partial_rotary_factor_global = 0.5;
    l.rope_theta = 500_000.0;
    l.rope_yarn = true;
    l.yarn_factor = 32.0;
    l.yarn_orig_max_pos = 8192;
    try testing.expectApproxEqAbs(@as(f32, 0.5), l.yarnPartial(), 1e-9);
    try testing.expectEqual(@as(u32, 64), l.yarnRotaryDim());
    var lden: [32]f64 = undefined;
    computeYarnFreqs(&lden, l.head_dim, l.yarnPartial(), l.rope_theta, l.yarn_factor, l.yarn_beta_fast, l.yarn_beta_slow, l.yarn_orig_max_pos, l.yarn_truncate);
    // Bit-identical to the golden laguna values the pre-delegation test pins.
    try testing.expectApproxEqRel(@as(f64, 1.0), lden[0], 1e-9);
    try testing.expectApproxEqRel(@as(f64, 1.506929076e+00), lden[1], 1e-7);
    try testing.expectApproxEqRel(@as(f64, 1.061761980e+07), lden[31], 1e-7);
}

test "ropeAtPositions YaRN: cos/sin carry mscale on the rotated slice" {
    // Power-of-two mscale so the rotated slice is an exact doubling of the
    // mscale-1.0 arm (same yarn_inv_freq, yarnActive stays on). The pass-through
    // tail must not move: those dims never enter the rotation.
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();

    var c = ModelConfig{};
    c.model_type = "qwen4_exp";
    c.head_dim = 256;
    c.partial_rotary_factor = 0.25;
    c.rope_theta = 10_000_000.0;
    c.rope_yarn = true;
    c.yarn_factor = 4.0;
    c.yarn_orig_max_pos = 262_144;
    c.yarn_attention_factor = 2.0;
    c.yarn_beta_fast = 32.0;
    c.yarn_beta_slow = 1.0;

    const rotary: c_int = 64;
    const half: usize = 32;
    const inv = try testing.allocator.alloc(f64, half);
    defer testing.allocator.free(inv);
    yarnSpec(&c).invFreq(inv);

    var den_f32: [half]f32 = undefined;
    for (inv, 0..) |v, i| den_f32[i] = @floatCast(1.0 / v);
    const fshape = [_]c_int{@intCast(half)};
    const freqs = mlx.mlx_array_new_data(&den_f32, &fshape, 1, .float32);
    defer _ = mlx.mlx_array_free(freqs);

    var t: Transformer = undefined;

    t.rht = null;
    t.s = s;
    t.allocator = testing.allocator;
    t.config = c;
    t.rope_freqs_yarn = freqs;
    t.yarn_inv_freq = inv;
    try testing.expect(t.yarnActive());

    var ones: [256]f32 = @splat(1.0);
    const xshape = [_]c_int{ 1, 1, 1, 256 };
    const x = mlx.mlx_array_new_data(&ones, &xshape, 4, .float32);
    defer _ = mlx.mlx_array_free(x);

    const scaled = try t.ropeAtPositions(x, rotary, c.rope_theta, 1234.0, 1.0, 1);
    defer _ = mlx.mlx_array_free(scaled);

    t.config.yarn_attention_factor = 1.0;
    try testing.expect(t.yarnActive());
    const unscaled = try t.ropeAtPositions(x, rotary, c.rope_theta, 1234.0, 1.0, 1);
    defer _ = mlx.mlx_array_free(unscaled);

    var sf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sf);
    var uf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(uf);
    try mlx.check(mlx.mlx_astype(&sf, scaled, .float32, s));
    try mlx.check(mlx.mlx_astype(&uf, unscaled, .float32, s));
    try mlx.check(mlx.mlx_array_eval(sf));
    try mlx.check(mlx.mlx_array_eval(uf));
    const sv = mlx.mlx_array_data_float32(sf).?;
    const uv = mlx.mlx_array_data_float32(uf).?;
    for (0..64) |i| try testing.expectEqual(uv[i] * 2.0, sv[i]);
    for (64..256) |i| try testing.expectEqual(uv[i], sv[i]);
}

test "qsa indexer scalar rope applies yarn mscale to first rope_dims only" {
    // The QSA indexer is 128-wide with 64 rotary dims. mlx_fast_rope scale
    // stays 1.0 (never the mscale); yarnScaleRotated then scales the rotated
    // slice only. NOT applyMrope — that path already carries mscale via
    // fillCosSin. A 256-wide yarnScaleQK table would silently scale the
    // pass-through tail (or crash on a shape mismatch).
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();

    var t: Transformer = undefined;

    t.rht = null;
    t.s = s;
    t.config = .{};
    t.config.yarn_attention_factor = 2.0;
    t.config.rope_yarn = true;
    t.config.rope_theta = 10_000_000.0;

    var data: [128]f32 = undefined;
    for (&data, 0..) |*d, i| d.* = @floatFromInt(i + 1);
    const shape = [_]c_int{ 1, 1, 1, 128 };
    const x = mlx.mlx_array_new_data(&data, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(x);

    var roped = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(roped);
    try mlx.check(mlx.mlx_fast_rope(&roped, x, 64, false, mlx.mlx_optional_float{
        .value = t.config.rope_theta,
        .has_value = true,
    }, 1.0, 0, .{ .ctx = null }, s));

    var unscaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(unscaled);
    try mlx.check(mlx.mlx_array_set(&unscaled, roped));

    try t.yarnScaleRotated(&roped, 64);

    var a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a);
    var b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b);
    try mlx.check(mlx.mlx_astype(&a, roped, .float32, s));
    try mlx.check(mlx.mlx_astype(&b, unscaled, .float32, s));
    try mlx.check(mlx.mlx_array_eval(a));
    try mlx.check(mlx.mlx_array_eval(b));
    const av = mlx.mlx_array_data_float32(a).?;
    const bv = mlx.mlx_array_data_float32(b).?;
    for (0..64) |i| try testing.expectEqual(bv[i] * 2.0, av[i]);
    for (64..128) |i| try testing.expectEqual(bv[i], av[i]);
}

test "qsaMaskFromQk applies yarnScaleRotated on the scalar rope arm only" {
    // The QSA indexer's mlx_fast_rope else-arm is the only path that does not
    // already fold mscale (M-RoPE tables carry it via fillCosSin). A comment
    // once skipped the scale; deleting the call still left every runtime test
    // green because they invoke yarnScaleRotated directly. This scan is the
    // wiring pin. Needles are assembled at comptime so this test's own source
    // cannot satisfy the scan.
    const src = @embedFile("../../../plugins/qwen4_exp/qsa/indexer.zig");
    const fn_mark = "pub fn qsaMask" ++ "FromQk(";
    const at = std.mem.indexOf(u8, src, fn_mark) orelse return error.MissingQsaMaskFromQk;
    const line_end = std.mem.indexOfScalarPos(u8, src, at, '\n') orelse src.len;
    const next = std.mem.indexOfPos(u8, src, line_end, "\npub fn ") orelse src.len;
    const body = src[line_end..next];

    const yarn_scale = "yarn" ++ "ScaleRotated";
    const fast_rope = "mlx_" ++ "fast_rope";
    const mrope_mark = "ctx.mrope_" ++ "cos_cur";
    const apply_mrope = "apply" ++ "Mrope";
    const scale_lit = "}, 1" ++ ".0,";
    const skip_a = "deliberately NOT" ++ " applied";
    const skip_b = "cannot change a" ++ " top-k";

    const mrope_at = std.mem.indexOf(u8, body, mrope_mark) orelse return error.MissingMropeArm;
    const pooled_at = std.mem.indexOfPos(u8, body, mrope_at, "// Pooled block keys") orelse
        return error.MissingPooledSection;
    const rope_block = body[mrope_at..pooled_at];
    const else_at = std.mem.indexOf(u8, rope_block, "} else {") orelse return error.MissingScalarElse;
    const mrope_arm = rope_block[0..else_at];
    const scalar_arm = rope_block[else_at..];

    try testing.expect(std.mem.indexOf(u8, mrope_arm, apply_mrope) != null);
    var mrope_it = std.mem.splitScalar(u8, mrope_arm, '\n');
    while (mrope_it.next()) |line| {
        const trimmed = std.mem.trimStart(u8, line, " \t");
        if (std.mem.startsWith(u8, trimmed, "//")) continue;
        if (std.mem.indexOf(u8, line, yarn_scale) != null) return error.YarnScaleOnMropeArm;
    }

    var fast_seen = false;
    var scale_after = false;
    var scale_one = false;
    var scalar_it = std.mem.splitScalar(u8, scalar_arm, '\n');
    while (scalar_it.next()) |line| {
        const trimmed = std.mem.trimStart(u8, line, " \t");
        if (std.mem.startsWith(u8, trimmed, "//")) continue;
        if (std.mem.indexOf(u8, line, fast_rope) != null) fast_seen = true;
        if (std.mem.indexOf(u8, line, scale_lit) != null) scale_one = true;
        if (std.mem.indexOf(u8, line, yarn_scale) != null) {
            if (!fast_seen) return error.YarnScaleBeforeFastRope;
            scale_after = true;
        }
    }
    if (!fast_seen) return error.MissingFastRope;
    if (!scale_one) return error.FastRopeScaleNotOne;
    if (!scale_after) return error.MissingYarnScaleOnScalarArm;

    var body_it = std.mem.splitScalar(u8, body, '\n');
    while (body_it.next()) |line| {
        const trimmed = std.mem.trimStart(u8, line, " \t");
        if (std.mem.startsWith(u8, trimmed, "//")) continue;
        if (std.mem.indexOf(u8, line, skip_a) != null or std.mem.indexOf(u8, line, skip_b) != null) {
            return error.SkipTheoryInProduction;
        }
    }
}

test "computeYarnFreqs matches HF _compute_yarn_parameters (Laguna full-attn rope)" {
    // Golden values from the reference YaRN math (tests/dump_laguna_fixtures.py
    // and the pure-Python cross-check): head_dim 128, partial 0.5 → dim 64 →
    // 32 freqs; base 5e5, factor 32, beta_fast 32, beta_slow 1, orig_max 8192.
    // low/high correction dims truncate to 9/18.
    var freqs: [32]f64 = undefined;
    computeYarnFreqs(&freqs, 128, 0.5, 500000.0, 32.0, 32.0, 1.0, 8192, true);
    // Spot-check the denominator array mlx_fast_rope consumes (angle = pos/freqs).
    const golden = [_]struct { idx: usize, val: f64 }{
        .{ .idx = 0, .val = 1.000000000e+00 },
        .{ .idx = 1, .val = 1.506929076e+00 },
        .{ .idx = 2, .val = 2.270835240e+00 },
        .{ .idx = 7, .val = 1.764613870e+01 },
        .{ .idx = 15, .val = 1.324904288e+03 },
        .{ .idx = 16, .val = 2.868264127e+03 },
        .{ .idx = 23, .val = 3.992865387e+05 },
        .{ .idx = 31, .val = 1.061761980e+07 },
    };
    for (golden) |g| {
        try testing.expectApproxEqRel(g.val, freqs[g.idx], 1e-7);
    }
    // Below the low correction dim (9): pure extrapolation → freqs = base^(2i/64).
    try testing.expectApproxEqRel(std.math.pow(f64, 500000.0, 0.0), freqs[0], 1e-9);
    // Above the high correction dim (18): pure interpolation → freqs = factor·base^(2i/64).
    try testing.expectApproxEqRel(32.0 * std.math.pow(f64, 500000.0, @as(f64, 2 * 31) / 64.0), freqs[31], 1e-6);
}

test "computeLlama3Freqs matches HF _compute_llama3_parameters (Llama 3.2)" {
    // Golden denominators from HF's inv-freq formulation: head_dim 128, base 5e5,
    // factor 32, low/high 1/4, window 8192. High frequencies keep their rate, low
    // ones run 32x slower, the band between is smoothed.
    var freqs: [64]f64 = undefined;
    computeLlama3Freqs(&freqs, 128, 500000.0, .{ .factor = 32.0, .low_freq_factor = 1.0, .high_freq_factor = 4.0, .original_max_position_embeddings = 8192.0 });
    const golden = [_]struct { idx: usize, val: f64 }{
        .{ .idx = 0, .val = 1.0 },
        .{ .idx = 20, .val = 60.384868705732515 },
        .{ .idx = 30, .val = 774.8646742531896 },
        .{ .idx = 40, .val = 116682.63579547375 },
        .{ .idx = 45, .val = 325265.6956955225 },
        .{ .idx = 63, .val = 13033875.741704715 },
    };
    for (golden) |g| try testing.expectApproxEqRel(g.val, freqs[g.idx], 1e-9);
}

test "laguna yarn parity vs modeling_laguna.py (LAGUNA_FIXTURES)" {
    // Env-gated cos/value oracle: compare computeYarnFreqs against the
    // full-attention rotary frequencies dumped from the reference
    // modeling_laguna.py (tests/dump_laguna_fixtures.py, CPU fp32). Dormant in
    // normal CI; run with LAGUNA_FIXTURES=/tmp/laguna_fixtures.json. The
    // fixture's `yarn.freqs` = 1/inv_freq (the mlx_fast_rope denominator).
    const path_z = env.get(.LAGUNA_FIXTURES) orelse return error.SkipZigTest;
    const path = std.mem.span(path_z);
    if (path.len == 0) return error.SkipZigTest;
    const io = std.Io.Threaded.global_single_threaded.io();
    const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);
    var read_buf: [4096]u8 = undefined;
    var reader_state = file.reader(io, &read_buf);
    const data = try reader_state.interface.allocRemaining(testing.allocator, .limited(1 << 20));
    defer testing.allocator.free(data);
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, data, .{});
    defer parsed.deinit();
    const yarn = parsed.value.object.get("yarn").?.object;
    const cfg = parsed.value.object.get("config").?.object;

    const head_dim: u32 = @intCast(cfg.get("head_dim").?.integer);
    const partial: f32 = @floatCast(jsonF64(yarn.get("partial_rotary_factor").?));
    const base = jsonF64(yarn.get("rope_theta").?);
    const factor = jsonF64(yarn.get("factor").?);
    const beta_fast = jsonF64(yarn.get("beta_fast").?);
    const beta_slow = jsonF64(yarn.get("beta_slow").?);
    const orig_max: u32 = @intCast(yarn.get("original_max_position_embeddings").?.integer);

    const ref = yarn.get("freqs").?.array;
    const out = try testing.allocator.alloc(f64, ref.items.len);
    defer testing.allocator.free(out);
    computeYarnFreqs(out, head_dim, partial, base, factor, beta_fast, beta_slow, orig_max, true);
    for (ref.items, 0..) |item, i| {
        try testing.expectApproxEqRel(jsonF64(item), out[i], 1e-6);
    }
    std.debug.print("[laguna-yarn] {d} freqs match reference (max rel err < 1e-6)\n", .{ref.items.len});
}

test "the YaRN mscale table never reaches a multiply in its load-time dtype" {
    // Source-scan class guard (same shape as generate.zig's `advanceStep` scan).
    // The bug was a single multiply of post-RoPE q/k by the load-time mscale
    // table while that table was still f32 and q/k were bf16. Nothing
    // observable failed — the model stayed correct and the op count stayed the
    // same, the token just got 3x slower — so only a scan keeps it out.
    const src = @embedFile("../../../engines/mlx/transformer.zig") ++
        @embedFile("../../../engines/mlx/load/init.zig") ++
        @embedFile("yarn.zig") ++
        @embedFile("../attention/gated_full.zig") ++
        @embedFile("../attention/sinks.zig") ++
        @embedFile("qk_norm_rope.zig");
    // Needles are assembled at comptime: spelled whole, they would appear in
    // this test's own source and the scan would satisfy itself.
    const routed = "constTableAs(" ++ "ms_f32,";
    const raw_table = "yarn_" ++ "mscale";
    const raw_local = "ms_" ++ "f32";
    const mul = "mlx_" ++ "multiply(";
    try std.testing.expect(std.mem.indexOf(u8, src, routed) != null);

    var it = std.mem.splitScalar(u8, src, '\n');
    var line_no: usize = 0;
    while (it.next()) |line| {
        line_no += 1;
        const trimmed = std.mem.trimStart(u8, line, " \t");
        if (std.mem.startsWith(u8, trimmed, "//")) continue; // prose, not code
        if (std.mem.indexOf(u8, line, mul) == null) continue;
        if (std.mem.indexOf(u8, line, raw_table) != null or
            std.mem.indexOf(u8, line, raw_local) != null)
        {
            std.debug.print(
                "transformer.zig:{d} multiplies by the load-time f32 mscale table; route it through constTableAs\n",
                .{line_no},
            );
            return error.RawMscaleReachesMultiply;
        }
    }
}
