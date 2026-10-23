//! sdk/blocks/attention/sdpa, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns nax_sdpa_avail_cache, nax_sdpa_override, nax_sdpa_env, naxSdpaPreferredFrom, naxSdpaPreferred, sdpaForceFusedFor, and 15 more.

const std = @import("std");
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../host/log.zig");
const testing = @import("../../../engines/mlx/kv/cache.zig").testing;
const computeNaxAvailable = @import("../../../mlx/device.zig").computeNaxAvailable;
const attn256QsaMask = @import("../../../plugins/qwen4_exp/qsa/gather.zig").attn256QsaMask;
const FUSED256_MIN_Q_LEN = @import("attn256.zig").FUSED256_MIN_Q_LEN;
const attn256MaxDiff = @import("attn256.zig").attn256MaxDiff;
const attn256RandBf16 = @import("attn256.zig").attn256RandBf16;
const attn256Reference = @import("attn256.zig").attn256Reference;

var nax_sdpa_avail_cache: ?bool = null;
pub var nax_sdpa_override: ?bool = null; // test seam


/// mlx >= 0.32.2 serves hd-256 attention on NAX with a fused full-attention kernel (`sdpa_full_self_attention_nax`),
/// auto only at >= 1024 causal rows and reachable below that through `force_fused`: preferred wherever NAX is.
pub fn naxSdpaPreferred() bool {
    if (nax_sdpa_override) |v| return v;
    if (nax_sdpa_avail_cache) |v| return v;
    const on = computeNaxAvailable();
    nax_sdpa_avail_cache = on;
    if (on) log.info("[nax-sdpa] engaged: stock fused sdpa (force_fused) serves hd-256 causal prefill and verify blocks > 8 rows; msv_attn_p256 causal arm declined\n", .{});
    return on;
}

/// force_fused is legal only where MLX has a fused kernel for the shape:
/// hd 256 (<= 128 fuses on its own, 192 is not ours), more than 8 query
/// rows (<= 8 is the vector kernel and its q*gqa <= 32 wall makes
/// force_fused THROW), q <= k (the full kernel's causal precondition).
pub fn sdpaForceFusedFor(preferred: bool, q_len: c_int, k_len: c_int, head_dim: c_int) bool {
    return preferred and head_dim == 256 and q_len > 8 and q_len <= k_len;
}

pub fn sdpaForceFused(q: mlx.mlx_array, k: mlx.mlx_array) bool {
    if (!naxSdpaPreferred()) return false;
    if (mlx.mlx_array_ndim(q) != 4 or mlx.mlx_array_ndim(k) != 4) return false;
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const on = sdpaForceFusedFor(true, qs[2], ks[2], qs[3]);
    if (on and !nax_sdpa_dispatch_logged) {
        nax_sdpa_dispatch_logged = true;
        log.info("[nax-sdpa] force_fused dispatch engaged: qL={d} kL={d} Hq={d} Hkv={d} hd={d}\n", .{ qs[2], ks[2], qs[1], ks[1], qs[3] });
    }
    return on;
}
var nax_sdpa_dispatch_logged = false;

/// Kill switch (MLX_SERVE_SDPA_SPLIT=0). Test seam: `sdpa_split_override`
/// forces the arm on/off without the environment.
pub var sdpa_split_override: ?bool = null;
fn sdpaSplitEnabled() bool {
    return sdpa_split_override orelse true;
}

/// One engagement log per width (2..15): the first engagement is usually the
/// warmup's own short prefill — a per-width line is what witnesses the split
/// at REAL verify shapes in an A/B arm's log.
var sdpa_split_logged: [16]bool = @splat(false);

/// Rows per group that MLX's vector sdpa kernel serves for this geometry, or
/// null when one call already takes the fast path (or no split can help).
/// The vector kernel serves q_len <= 8 with q_len * gqa <= 32 at hd
/// {64, 96, 128, 256}; q_len > 8 runs MLX's fused full kernel except at hd
/// 256, where MLX takes its unfused fallback unless NAX force-fuses it.
/// Engages at hd 256 only until other head dims are measured.
fn causalSplitGroupRows(q_len: c_int, gqa: c_int, head_dim: c_int) ?c_int {
    const vector_hd = head_dim == 64 or head_dim == 96 or head_dim == 128 or head_dim == 256;
    if (!vector_hd or gqa < 1) return null;
    if (q_len < 2 or q_len >= FUSED256_MIN_Q_LEN) return null;
    const fits_vector = q_len <= 8 and q_len * gqa <= 32;
    if (fits_vector) return null;
    if (q_len > 8 and head_dim != 256) return null;
    // Measured only at hd 256 (Qwen3.8-Flash-Next, gqa 12). Other head dims
    // need their own per-arch A/B before this default-on path reaches them.
    if (head_dim != 256) return null;
    return @min(8, @max(1, @divTrunc(32, gqa)));
}

/// Width-wall query split for dense causal spec-verify blocks (B==1). MLX's
/// vector sdpa kernel serves q_len * gqa <= 32 (q_len <= 8), so a wider
/// verify block otherwise runs the unfused fallback (~8 dispatches, scores
/// materialized). Rows split into groups of min(8, 32/gqa): 5 at gqa 6, 2 at
/// gqa 12 (Qwen3.8-Flash-Next). Group [r0, r1) runs "causal" over keys[0 ..
/// kL-(qL-r1)]; bottom-right alignment then gives row r the window
/// kL-qL+r, the same one a width-(r1-r0) round at that offset gets. K/V are
/// views, never recomputed. Returns null outside the envelope — the caller
/// falls through to the single sdpa call. Generalizes the Layr-Labs
/// qwen-3.8-mtp-challenge width-wall split (see NOTICE).
pub fn splitCausalSdpa(
    s: mlx.mlx_stream,
    q: mlx.mlx_array,
    k: mlx.mlx_array,
    v: mlx.mlx_array,
    scale: f32,
) !?mlx.mlx_array {
    if (!sdpaSplitEnabled()) return null;
    if (mlx.mlx_array_ndim(q) != 4 or mlx.mlx_array_ndim(k) != 4 or mlx.mlx_array_ndim(v) != 4) return null;
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const vs = mlx.getShape(v);
    const hd = qs[3];
    if (ks[3] != hd or vs[3] != hd) return null;
    if (qs[0] != 1 or ks[0] != 1 or vs[0] != 1) return null;
    if (ks[2] < qs[2] or ks[2] != vs[2] or ks[1] != vs[1] or ks[1] < 1) return null;
    if (@rem(qs[1], ks[1]) != 0) return null;
    const qL = qs[2];
    const kL = ks[2];
    const gqa = @divTrunc(qs[1], ks[1]);
    const group = causalSplitGroupRows(qL, gqa, hd) orelse return null;
    // Fallback-only: where the caller's single call force-fuses onto MLX's
    // NAX kernel (hd 256, qL > 8), that one dispatch beats any split.
    if (sdpaForceFusedFor(naxSdpaPreferred(), qL, kL, hd)) return null;

    const strides = [_]c_int{ 1, 1, 1, 1 };
    const none_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none_mask);
    var parts = std.ArrayList(mlx.mlx_array).empty;
    defer {
        for (parts.items) |a| _ = mlx.mlx_array_free(a);
        parts.deinit(std.heap.c_allocator);
    }
    var r0: c_int = 0;
    while (r0 < qL) : (r0 += group) {
        const r1: c_int = @min(r0 + group, qL);
        const k_end: c_int = kL - (qL - r1);
        var q_g = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_g);
        var k_g = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_g);
        var v_g = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v_g);
        try mlx.check(mlx.mlx_slice(&q_g, q, &[_]c_int{ 0, 0, r0, 0 }, 4, &[_]c_int{ 1, qs[1], r1, hd }, 4, &strides, 4, s));
        try mlx.check(mlx.mlx_slice(&k_g, k, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ 1, ks[1], k_end, hd }, 4, &strides, 4, s));
        try mlx.check(mlx.mlx_slice(&v_g, v, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ 1, vs[1], k_end, hd }, 4, &strides, 4, s));
        var o_g = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(o_g);
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&o_g, q_g, k_g, v_g, scale, "causal", none_mask, .{ .ctx = null }, false, s));
        try parts.append(std.heap.c_allocator, o_g);
    }
    const vec = mlx.mlx_vector_array_new_data(parts.items.ptr, parts.items.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 2, s));

    const log_idx: usize = @intCast(qL);
    if (!sdpa_split_logged[log_idx]) {
        sdpa_split_logged[log_idx] = true;
        log.info("[sdpa-split] engaged: qL={d} kL={d} Hq={d} Hkv={d} rows/group={d}\n", .{ qL, kL, qs[1], ks[1], group });
    }
    return out;
}

var sdpa_mask_split_logged = false;

/// Array-mask verify widths on hd 256 (qwen4_exp QSA past the budget): MLX's
/// vector kernel serves `qL * gqa <= 32` rows, anything wider (S >= 3 at
/// gqa 12) and every array-masked qL 9..15 is the unfused fallback. Rows are
/// independent under a per-row mask, so slice the block into vector-sized row
/// groups. Kill switch shared with the causal split (MLX_SERVE_SDPA_SPLIT=0).
pub fn splitMaskedSdpa256(
    s: mlx.mlx_stream,
    q: mlx.mlx_array,
    k: mlx.mlx_array,
    v: mlx.mlx_array,
    scale: f32,
    mask: mlx.mlx_array,
) !?mlx.mlx_array {
    if (!sdpaSplitEnabled()) return null;
    if (mlx.mlx_array_ndim(q) != 4 or mlx.mlx_array_ndim(k) != 4 or mlx.mlx_array_ndim(mask) != 4) return null;
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const ms = mlx.getShape(mask);
    if (qs[3] != 256 or ks[3] != 256) return null;
    if (qs[0] != 1 or ks[0] != 1 or ms[0] != 1 or ms[1] != 1 or ms[2] != qs[2] or ms[3] != ks[2]) return null;
    const qL = qs[2];
    if (qL < 2 or qL >= FUSED256_MIN_Q_LEN) return null;
    const gqa: c_int = @divTrunc(qs[1], ks[1]);
    if (qL * gqa <= 32) return null;
    const group: c_int = @max(1, @divTrunc(32, gqa));

    var parts = std.ArrayList(mlx.mlx_array).empty;
    defer {
        for (parts.items) |a| _ = mlx.mlx_array_free(a);
        parts.deinit(std.heap.c_allocator);
    }
    const strides = [_]c_int{ 1, 1, 1, 1 };
    var r0: c_int = 0;
    while (r0 < qL) : (r0 += group) {
        const r1: c_int = @min(r0 + group, qL);
        var q_g = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_g);
        var m_g = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(m_g);
        try mlx.check(mlx.mlx_slice(&q_g, q, &[_]c_int{ 0, 0, r0, 0 }, 4, &[_]c_int{ 1, qs[1], r1, 256 }, 4, &strides, 4, s));
        try mlx.check(mlx.mlx_slice(&m_g, mask, &[_]c_int{ 0, 0, r0, 0 }, 4, &[_]c_int{ 1, 1, r1, ms[3] }, 4, &strides, 4, s));
        var o_g = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(o_g);
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&o_g, q_g, k, v, scale, "array", m_g, .{ .ctx = null }, false, s));
        try parts.append(std.heap.c_allocator, o_g);
    }
    const vec = mlx.mlx_vector_array_new_data(parts.items.ptr, parts.items.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 2, s));
    if (!sdpa_mask_split_logged) {
        sdpa_mask_split_logged = true;
        log.info("[sdpa-split] masked arm engaged: qL={d} kL={d} gqa={d} rows/group={d}\n", .{ qL, ks[2], gqa, group });
    }
    return out;
}

/// q [1, H, S, hd] over k/v [1, KVH, L, hd] whose last S rows are the
/// window's: row r takes the one-query sdpa over keys [0, L - S + r].
pub fn rowByRowSdpa(out: *mlx.mlx_array, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32, seq: c_int, s: mlx.mlx_stream) !void {
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const vs = mlx.getShape(v);
    const none_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none_mask);
    const rows = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(rows);
    const st4 = [_]c_int{ 1, 1, 1, 1 };
    var r: c_int = 0;
    while (r < seq) : (r += 1) {
        const keys = ks[2] - seq + r + 1;
        var qr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(qr);
        try mlx.check(mlx.mlx_slice(&qr, q, &[_]c_int{ 0, 0, r, 0 }, 4, &[_]c_int{ qs[0], qs[1], r + 1, qs[3] }, 4, &st4, 4, s));
        var kr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(kr);
        try mlx.check(mlx.mlx_slice(&kr, k, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ ks[0], ks[1], keys, ks[3] }, 4, &st4, 4, s));
        var vr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(vr);
        try mlx.check(mlx.mlx_slice(&vr, v, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ vs[0], vs[1], keys, vs[3] }, 4, &st4, 4, s));
        var o = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(o);
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&o, qr, kr, vr, scale, "", none_mask, .{ .ctx = null }, false, s));
        _ = mlx.mlx_vector_array_append_value(rows, o);
    }
    try mlx.check(mlx.mlx_concatenate_axis(out, rows, 2, s));
}

// ── tests ──

test "splitMaskedSdpa256: verify-width array-mask rows split through the vector kernel match one dispatch" {
    const s = mlx.gpuStream();
    sdpa_split_override = true;
    defer sdpa_split_override = null;
    var prng = std.Random.DefaultPrng.init(0x5b5b);
    const rnd = prng.random();
    // gqa 12 like qwen4_exp (24/2): qL 4 is the unfused fallback in stock MLX.
    const q_shape = [_]c_int{ 1, 24, 4, 256 };
    const kv_shape = [_]c_int{ 1, 2, 300, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);
    const mask = try attn256QsaMask(rnd, 4, 300);
    defer _ = mlx.mlx_array_free(mask);
    const scale: f32 = 1.0 / 16.0;
    const split = (try splitMaskedSdpa256(s, q, k, v, scale, mask)) orelse return error.SplitDeclined;
    defer _ = mlx.mlx_array_free(split);
    const ref = try attn256Reference(q, k, v, scale, "array", mask, s);
    defer _ = mlx.mlx_array_free(ref);
    try std.testing.expect(try attn256MaxDiff(split, ref, s) < 0.005);
    // qL 2 already fits the vector kernel: declined.
    const q2_shape = [_]c_int{ 1, 24, 2, 256 };
    const q2 = try attn256RandBf16(rnd, &q2_shape, s);
    defer _ = mlx.mlx_array_free(q2);
    const m2 = try attn256QsaMask(rnd, 2, 300);
    defer _ = mlx.mlx_array_free(m2);
    try std.testing.expect((try splitMaskedSdpa256(s, q2, k, v, scale, m2)) == null);
}

/// Like attn256MaxDiff but errors on any non-finite element on either side —
/// NaN > max_err is false, so an all-NaN pair would otherwise diff to 0.
fn attn256FiniteMaxDiff(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    var a32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a32);
    var b32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b32);
    try mlx.check(mlx.mlx_astype(&a32, a, .float32, s));
    try mlx.check(mlx.mlx_astype(&b32, b, .float32, s));
    try mlx.check(mlx.mlx_array_eval(a32));
    try mlx.check(mlx.mlx_array_eval(b32));
    const ad = mlx.mlx_array_data_float32(a32) orelse return error.InvalidDtype;
    const bd = mlx.mlx_array_data_float32(b32) orelse return error.InvalidDtype;
    const n = mlx.mlx_array_size(a32);
    if (n != mlx.mlx_array_size(b32)) return error.ShapeMismatch;
    var max_diff: f32 = 0;
    for (0..n) |i| {
        if (!std.math.isFinite(ad[i]) or !std.math.isFinite(bd[i])) return error.NonFinite;
        max_diff = @max(max_diff, @abs(ad[i] - bd[i]));
    }
    return max_diff;
}

const CausalSplitCase = struct { hq: c_int, hkv: c_int, ql: c_int, kv: c_int, hd: c_int };

fn causalSplitCaseArrays(rnd: std.Random, c: CausalSplitCase, s: mlx.mlx_stream) ![3]mlx.mlx_array {
    const kL = c.kv + c.ql;
    const q = try attn256RandBf16(rnd, &[_]c_int{ 1, c.hq, c.ql, c.hd }, s);
    errdefer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &[_]c_int{ 1, c.hkv, kL, c.hd }, s);
    errdefer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &[_]c_int{ 1, c.hkv, kL, c.hd }, s);
    return .{ q, k, v };
}

/// Row r of a bottom-right causal block sees keys 0 .. kL-qL+r. One qL=1
/// vector call per row over exactly that prefix: an offset reference that
/// does not go through MLX's causal fallback.
fn causalPerRowReference(q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32, s: mlx.mlx_stream) !mlx.mlx_array {
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const strides = [_]c_int{ 1, 1, 1, 1 };
    const none = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none);
    var parts = std.ArrayList(mlx.mlx_array).empty;
    defer {
        for (parts.items) |a| _ = mlx.mlx_array_free(a);
        parts.deinit(std.testing.allocator);
    }
    var r: c_int = 0;
    while (r < qs[2]) : (r += 1) {
        const kend = ks[2] - qs[2] + r + 1;
        var q_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_r);
        var k_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_r);
        var v_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v_r);
        try mlx.check(mlx.mlx_slice(&q_r, q, &[_]c_int{ 0, 0, r, 0 }, 4, &[_]c_int{ 1, qs[1], r + 1, qs[3] }, 4, &strides, 4, s));
        try mlx.check(mlx.mlx_slice(&k_r, k, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ 1, ks[1], kend, ks[3] }, 4, &strides, 4, s));
        try mlx.check(mlx.mlx_slice(&v_r, v, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ 1, ks[1], kend, ks[3] }, 4, &strides, 4, s));
        var o = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(o);
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&o, q_r, k_r, v_r, scale, "", none, .{ .ctx = null }, false, s));
        try parts.append(std.testing.allocator, o);
    }
    const vec = mlx.mlx_vector_array_new_data(parts.items.ptr, parts.items.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 2, s));
    return out;
}

test "splitCausalSdpa: gqa-aware groups match one causal sdpa across widths, kv and head dims" {
    const s = mlx.gpuStream();
    sdpa_split_override = true;
    defer sdpa_split_override = null;
    nax_sdpa_override = false;
    defer nax_sdpa_override = null;
    var prng = std.Random.DefaultPrng.init(0x5D9A);
    const rnd = prng.random();
    const cases = [_]CausalSplitCase{
        // Qwen3.8-Flash-Next: gqa 12 (24/2), 2 rows per group.
        .{ .hq = 24, .hkv = 2, .ql = 3, .kv = 1, .hd = 256 },
        .{ .hq = 24, .hkv = 2, .ql = 4, .kv = 1, .hd = 256 },
        .{ .hq = 24, .hkv = 2, .ql = 4, .kv = 100, .hd = 256 },
        .{ .hq = 24, .hkv = 2, .ql = 6, .kv = 100, .hd = 256 },
        .{ .hq = 24, .hkv = 2, .ql = 15, .kv = 100, .hd = 256 },
        // kL >= 1024: MLX's 2-pass vector kernel on Max/Ultra.
        .{ .hq = 24, .hkv = 2, .ql = 4, .kv = 2051, .hd = 256 },
        .{ .hq = 24, .hkv = 2, .ql = 15, .kv = 2051, .hd = 256 },
        // gqa 6 (27B geometry): 6..9 as before, 10..15 now split too.
        .{ .hq = 12, .hkv = 2, .ql = 6, .kv = 187, .hd = 256 },
        .{ .hq = 12, .hkv = 2, .ql = 9, .kv = 184, .hd = 256 },
        .{ .hq = 12, .hkv = 2, .ql = 10, .kv = 100, .hd = 256 },
        .{ .hq = 12, .hkv = 2, .ql = 15, .kv = 100, .hd = 256 },
        // gqa 32: 1-row groups. gqa 3 at qL 11: an 8-row group + 3.
        .{ .hq = 32, .hkv = 1, .ql = 3, .kv = 100, .hd = 256 },
        .{ .hq = 6, .hkv = 2, .ql = 11, .kv = 100, .hd = 256 },
        // gqa 8 (gemma-like hd 256): 4-row groups.
        .{ .hq = 16, .hkv = 2, .ql = 5, .kv = 1, .hd = 256 },
    };
    const scale: f32 = 1.0 / 16.0;
    for (cases) |c| {
        const arrs = try causalSplitCaseArrays(rnd, c, s);
        defer for (arrs) |a| {
            _ = mlx.mlx_array_free(a);
        };
        const split = (try splitCausalSdpa(s, arrs[0], arrs[1], arrs[2], scale)) orelse {
            std.debug.print("split declined: {any}\n", .{c});
            return error.SplitDeclined;
        };
        defer _ = mlx.mlx_array_free(split);
        // Tolerance, not byte-identity: the single call is MLX's unfused
        // fallback, a different reduction order (the verify-lane rule).
        const ref = try attn256Reference(arrs[0], arrs[1], arrs[2], scale, "causal", .{ .ctx = null }, s);
        defer _ = mlx.mlx_array_free(ref);
        const max_diff = try attn256FiniteMaxDiff(split, ref, s);
        if (max_diff >= 0.005) std.debug.print("case {any}: max diff {d}\n", .{ c, max_diff });
        try std.testing.expect(max_diff < 0.005);
    }
}

test "splitCausalSdpa: every row sees exactly its causal window (bit-identical to per-row vector calls below kL 1024)" {
    const s = mlx.gpuStream();
    sdpa_split_override = true;
    defer sdpa_split_override = null;
    nax_sdpa_override = false;
    defer nax_sdpa_override = null;
    var prng = std.Random.DefaultPrng.init(0x5DA1);
    const rnd = prng.random();
    // kL < 1024 keeps both sides on the 1-pass vector kernel, which skips
    // masked keys, so each row's math depends only on its key set.
    const cases = [_]CausalSplitCase{
        .{ .hq = 24, .hkv = 2, .ql = 4, .kv = 100, .hd = 256 },
        .{ .hq = 24, .hkv = 2, .ql = 15, .kv = 37, .hd = 256 },
        .{ .hq = 12, .hkv = 2, .ql = 9, .kv = 100, .hd = 256 },
        .{ .hq = 32, .hkv = 1, .ql = 3, .kv = 1, .hd = 256 },
        .{ .hq = 16, .hkv = 2, .ql = 7, .kv = 100, .hd = 256 },
    };
    const scale: f32 = 1.0 / 16.0;
    for (cases) |c| {
        const arrs = try causalSplitCaseArrays(rnd, c, s);
        defer for (arrs) |a| {
            _ = mlx.mlx_array_free(a);
        };
        const split = (try splitCausalSdpa(s, arrs[0], arrs[1], arrs[2], scale)) orelse return error.SplitDeclined;
        defer _ = mlx.mlx_array_free(split);
        const ref = try causalPerRowReference(arrs[0], arrs[1], arrs[2], scale, s);
        defer _ = mlx.mlx_array_free(ref);
        const max_diff = try attn256FiniteMaxDiff(split, ref, s);
        if (max_diff != 0) std.debug.print("case {any}: max diff {d}\n", .{ c, max_diff });
        try std.testing.expect(max_diff == 0);
    }
}

test "splitCausalSdpa: declines outside its envelope" {
    const s = mlx.gpuStream();
    sdpa_split_override = true;
    defer sdpa_split_override = null;
    nax_sdpa_override = false;
    defer nax_sdpa_override = null;
    var prng = std.Random.DefaultPrng.init(0x5DEC);
    const rnd = prng.random();
    const declines = [_]CausalSplitCase{
        // The vector kernel already serves qL*gqa <= 32.
        .{ .hq = 24, .hkv = 2, .ql = 2, .kv = 64, .hd = 256 },
        .{ .hq = 12, .hkv = 2, .ql = 5, .kv = 64, .hd = 256 },
        .{ .hq = 6, .hkv = 2, .ql = 7, .kv = 64, .hd = 128 },
        // qL > 8 below hd 256: MLX's fused full kernel serves it.
        .{ .hq = 6, .hkv = 2, .ql = 10, .kv = 64, .hd = 128 },
        // Below hd 256 the split is unmeasured, so it declines even past the
        // vector wall (Llama/Mistral hd 128 gqa 8, hd 64).
        .{ .hq = 32, .hkv = 4, .ql = 6, .kv = 64, .hd = 128 },
        .{ .hq = 24, .hkv = 2, .ql = 4, .kv = 64, .hd = 128 },
        .{ .hq = 16, .hkv = 2, .ql = 5, .kv = 64, .hd = 64 },
        // qL >= 16: fusedSdpa256Prefill's range.
        .{ .hq = 24, .hkv = 2, .ql = 16, .kv = 64, .hd = 256 },
        // hd 192: MLX prefers its unfused path there.
        .{ .hq = 24, .hkv = 2, .ql = 4, .kv = 64, .hd = 192 },
        // Decode width.
        .{ .hq = 24, .hkv = 2, .ql = 1, .kv = 64, .hd = 256 },
    };
    for (declines) |c| {
        const arrs = try causalSplitCaseArrays(rnd, c, s);
        defer for (arrs) |a| {
            _ = mlx.mlx_array_free(a);
        };
        if ((try splitCausalSdpa(s, arrs[0], arrs[1], arrs[2], 1.0)) != null) {
            std.debug.print("split engaged outside envelope: {any}\n", .{c});
            return error.SplitEngaged;
        }
    }

    // Batch 2 -> null (the exactness argument is B==1 only).
    const qb = try attn256RandBf16(rnd, &[_]c_int{ 2, 24, 4, 256 }, s);
    defer _ = mlx.mlx_array_free(qb);
    const kb = try attn256RandBf16(rnd, &[_]c_int{ 2, 2, 64, 256 }, s);
    defer _ = mlx.mlx_array_free(kb);
    try std.testing.expect((try splitCausalSdpa(s, qb, kb, kb, 1.0)) == null);

    // KV shorter than q -> null (no window to split).
    const q7 = try attn256RandBf16(rnd, &[_]c_int{ 1, 24, 7, 256 }, s);
    defer _ = mlx.mlx_array_free(q7);
    const kshort = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, 5, 256 }, s);
    defer _ = mlx.mlx_array_free(kshort);
    try std.testing.expect((try splitCausalSdpa(s, q7, kshort, kshort, 1.0)) == null);

    // Kill switch -> null even for a conforming call.
    const k = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, 64, 256 }, s);
    defer _ = mlx.mlx_array_free(k);
    sdpa_split_override = false;
    try std.testing.expect((try splitCausalSdpa(s, q7, k, k, 1.0)) == null);
}

test "splitCausalSdpa: yields to the fused NAX sdpa at hd 256 past 8 rows, still splits below" {
    const s = mlx.gpuStream();
    sdpa_split_override = true;
    defer sdpa_split_override = null;
    nax_sdpa_override = true;
    defer nax_sdpa_override = null;
    var prng = std.Random.DefaultPrng.init(0x5DAA);
    const rnd = prng.random();
    // qL > 8 at hd 256 with NAX preferred: the caller's single call runs
    // MLX's fused NAX kernel (sdpaForceFused), so the split must decline.
    for ([_]CausalSplitCase{
        .{ .hq = 12, .hkv = 2, .ql = 10, .kv = 100, .hd = 256 },
        .{ .hq = 24, .hkv = 2, .ql = 15, .kv = 100, .hd = 256 },
    }) |c| {
        const arrs = try causalSplitCaseArrays(rnd, c, s);
        defer for (arrs) |a| {
            _ = mlx.mlx_array_free(a);
        };
        if ((try splitCausalSdpa(s, arrs[0], arrs[1], arrs[2], 1.0)) != null) {
            std.debug.print("split pre-empted NAX: {any}\n", .{c});
            return error.SplitEngaged;
        }
    }
    // qL <= 8 never force-fuses (the vector wall would throw): split engages.
    const arrs = try causalSplitCaseArrays(rnd, .{ .hq = 24, .hkv = 2, .ql = 4, .kv = 100, .hd = 256 }, s);
    defer for (arrs) |a| {
        _ = mlx.mlx_array_free(a);
    };
    const split = (try splitCausalSdpa(s, arrs[0], arrs[1], arrs[2], 1.0)) orelse return error.SplitDeclined;
    _ = mlx.mlx_array_free(split);
}

test "sdpaForceFusedFor: only shapes MLX has a fused kernel for (hd 256, > 8 rows, q <= k)" {
    try testing.expect(sdpaForceFusedFor(true, 16, 4096, 256));
    try testing.expect(sdpaForceFusedFor(true, 9, 9, 256));
    // <= 8 rows is the vector kernel; its q*gqa <= 32 wall makes force_fused THROW.
    try testing.expect(!sdpaForceFusedFor(true, 8, 4096, 256));
    // hd <= 128 already fuses on its own; hd 192/exotic dims are not ours to force.
    try testing.expect(!sdpaForceFusedFor(true, 16, 4096, 128));
    try testing.expect(!sdpaForceFusedFor(true, 16, 4096, 192));
    try testing.expect(!sdpaForceFusedFor(true, 16, 8, 256));
    try testing.expect(!sdpaForceFusedFor(false, 16, 4096, 256));
}

test "sdpa force_fused: hd-256 causal blocks the gate admits match the unforced call (plumbing + shape gate)" {
    // The shapes sdpaForceFusedFor admits (hd 256, > 8 rows, q <= k): the
    // forced kernel (steel on M1-M4, the NAX variant on M5) must agree with
    // the unforced dispatch (unfused at hd 256 on non-NAX) and never throw.
    // <= 8 rows is NOT covered on purpose: that is the vector kernel, and
    // force_fused on a gqa-8 block there throws (the gate's whole point).
    const s = mlx.mlx_default_gpu_stream_new();
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    const none = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none);
    const cases = [_][3]c_int{ .{ 16, 4096, 8 }, .{ 9, 512, 8 }, .{ 64, 64, 2 } }; // q, k, gqa
    for (cases) |c| {
        const q_shape = [_]c_int{ 1, 2 * c[2], c[0], 256 };
        const k_shape = [_]c_int{ 1, 2, c[1], 256 };
        try std.testing.expect(sdpaForceFusedFor(true, c[0], c[1], 256));
        const q = try attn256RandBf16(rnd, &q_shape, s);
        defer _ = mlx.mlx_array_free(q);
        const k = try attn256RandBf16(rnd, &k_shape, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try attn256RandBf16(rnd, &k_shape, s);
        defer _ = mlx.mlx_array_free(v);
        var plain = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(plain);
        var forced = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(forced);
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&plain, q, k, v, 0.0625, "causal", none, .{ .ctx = null }, false, s));
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&forced, q, k, v, 0.0625, "causal", none, .{ .ctx = null }, true, s));
        const diff = try attn256MaxDiff(plain, forced, s);
        try std.testing.expect(diff < 5e-3);
    }
}
