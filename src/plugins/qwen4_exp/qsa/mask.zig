//! The dense QSA visibility mask from block picks, and the MTP head's placed-row positions and visibility.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const qsa_row_mask = @import("../qsa_mask.zig");
const Trunk = @import("../trunk.zig").Trunk;

/// Dense [B,1,S,kv] bool mask equivalent to a block selection (fallback arm
/// + fixture trace): the selected blocks' tokens ∪ each row's own incomplete
/// tail, ∧ causal. Sentinel entries land in a spare column and are sliced off.
pub fn qsaMaskFromBlocks(s: mlx.mlx_stream, blocks: mlx.mlx_array, kv: c_int, ratio: c_int) !mlx.mlx_array {
    const bs = mlx.getShape(blocks);
    const batch = bs[0];
    const seq_len = bs[1];
    if (batch == 1 and seq_len == 1) if (try qsa_row_mask.rowMask(s, blocks, kv, ratio)) |m| return m;
    const nb: c_int = @divTrunc(kv, ratio);
    const nb_i = mlx.mlx_array_new_int(nb);
    defer _ = mlx.mlx_array_free(nb_i);
    var valid = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(valid);
    try mlx.check(mlx.mlx_less(&valid, blocks, nb_i, s));
    var safe = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(safe);
    try mlx.check(mlx.mlx_where(&safe, valid, blocks, nb_i, s));
    var falses = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(falses);
    try mlx.check(mlx.mlx_zeros(&falses, &[_]c_int{ batch, seq_len, nb + 1 }, 3, .bool_, s));
    const true_v = mlx.mlx_array_new_bool(true);
    defer _ = mlx.mlx_array_free(true_v);
    var put = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(put);
    try mlx.check(mlx.mlx_put_along_axis(&put, falses, safe, true_v, -1, s));
    var blk_sel = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(blk_sel);
    const strides3 = [_]c_int{ 1, 1, 1 };
    try mlx.check(mlx.mlx_slice(&blk_sel, put, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ batch, seq_len, nb }, 3, &strides3, 3, s));
    return qsaMaskFromBlockSel(s, blk_sel, kv - seq_len, seq_len, kv, nb, ratio, batch);
}

/// Expand a [B,S,nb] bool block selection to the [B,1,S,kv] token mask:
/// selected blocks' tokens (zero-padded to kv) ∪ each query's own incomplete
/// tail (tokens at/after ratio*floor((p+1)/ratio) — always visible), ∧ the
/// causal bound. `offset` is the cache position of row 0.
/// `[1]` int32 RoPE position of a placed draft row: `live_end + (slot - dead_end)`, with the
/// head's absolute origin folded into `shift` (`pos_base + slot - dead_end`).
pub fn headPlaceRopePos(s: mlx.mlx_stream, p: *const Trunk.HeadPlace, shift: c_int) !mlx.mlx_array {
    const shift_v = mlx.mlx_array_new_int(shift);
    defer _ = mlx.mlx_array_free(shift_v);
    var pos = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pos);
    try mlx.check(mlx.mlx_add(&pos, p.live_end, shift_v, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, pos, &[_]c_int{1}, 1, s));
    return out;
}

/// Block scores with every block that reaches a dead row (or lies past one) at -inf, so a
/// dead row's pooled key can never win or displace a top-k pick.
pub fn headPlaceScores(s: mlx.mlx_stream, scores: mlx.mlx_array, p: *const Trunk.HeadPlace, ratio: c_int, nb: c_int) !mlx.mlx_array {
    const ratio_v = mlx.mlx_array_new_int(ratio);
    defer _ = mlx.mlx_array_free(ratio_v);
    var eligible_n = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eligible_n);
    try mlx.check(mlx.mlx_floor_divide(&eligible_n, p.live_end, ratio_v, s));
    var blk = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(blk);
    try mlx.check(mlx.mlx_arange(&blk, 0, @floatFromInt(nb), 1, .int32, s));
    var eligible = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eligible);
    try mlx.check(mlx.mlx_less(&eligible, blk, eligible_n, s));
    const floor_v = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(floor_v);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_where(&out, eligible, scores, floor_v, s));
    return out;
}

/// Key visibility `[1,1,1,kv]` for a placed draft row: every row but the dead ones, narrowed
/// by the QSA selection when there is one. Rows from the first block that reaches a dead row
/// on stay visible, since `headPlaceScores` keeps those blocks out of the selection.
pub fn headPlaceVisibility(s: mlx.mlx_stream, qsa_mask: mlx.mlx_array, p: *const Trunk.HeadPlace, ratio: c_int, kv: c_int) !mlx.mlx_array {
    var r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r);
    try mlx.check(mlx.mlx_arange(&r, 0, @floatFromInt(kv), 1, .int32, s));
    var live = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(live);
    try mlx.check(mlx.mlx_less(&live, r, p.live_end, s));
    const dead_end_v = mlx.mlx_array_new_int(p.dead_end);
    defer _ = mlx.mlx_array_free(dead_end_v);
    var past = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(past);
    try mlx.check(mlx.mlx_greater_equal(&past, r, dead_end_v, s));
    var not_dead = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(not_dead);
    try mlx.check(mlx.mlx_logical_or(&not_dead, live, past, s));
    var vis = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vis);
    if (qsa_mask.ctx == null) {
        try mlx.check(mlx.mlx_array_set(&vis, not_dead));
    } else {
        const ratio_v = mlx.mlx_array_new_int(ratio);
        defer _ = mlx.mlx_array_free(ratio_v);
        var blocks = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(blocks);
        try mlx.check(mlx.mlx_floor_divide(&blocks, p.live_end, ratio_v, s));
        var recent_from = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(recent_from);
        try mlx.check(mlx.mlx_multiply(&recent_from, blocks, ratio_v, s));
        var recent = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(recent);
        try mlx.check(mlx.mlx_greater_equal(&recent, r, recent_from, s));
        var picked = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(picked);
        try mlx.check(mlx.mlx_logical_or(&picked, qsa_mask, recent, s));
        try mlx.check(mlx.mlx_logical_and(&vis, picked, not_dead, s));
    }
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, vis, &[_]c_int{ 1, 1, 1, kv }, 4, s));
    return out;
}

pub fn qsaMaskFromBlockSel(s: mlx.mlx_stream, blk_sel: mlx.mlx_array, offset: c_int, seq_len: c_int, kv: c_int, nb: c_int, ratio: c_int, batch: c_int) !mlx.mlx_array {
    var pos_col = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pos_col);
    try mlx.check(mlx.mlx_arange(&pos_col, @floatFromInt(offset), @floatFromInt(offset + seq_len), 1.0, .int32, s));
    var pos2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pos2);
    try mlx.check(mlx.mlx_reshape(&pos2, pos_col, &[_]c_int{ seq_len, 1 }, 2, s));
    var sel4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sel4);
    try mlx.check(mlx.mlx_expand_dims(&sel4, blk_sel, -1, s)); // [B,S,nb,1]
    var sel_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sel_b);
    try mlx.check(mlx.mlx_broadcast_to(&sel_b, sel4, &[_]c_int{ batch, seq_len, nb, ratio }, 4, s));
    var sel_tok = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sel_tok);
    try mlx.check(mlx.mlx_reshape(&sel_tok, sel_b, &[_]c_int{ batch, seq_len, nb * ratio }, 3, s));
    var sel_full = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sel_full);
    const pad: c_int = kv - nb * ratio;
    if (pad > 0) {
        var zeros_pad = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(zeros_pad);
        try mlx.check(mlx.mlx_zeros(&zeros_pad, &[_]c_int{ batch, seq_len, pad }, 3, .bool_, s));
        const parts = [_]mlx.mlx_array{ sel_tok, zeros_pad };
        const vec = mlx.mlx_vector_array_new_data(&parts, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&sel_full, vec, 2, s));
    } else {
        try mlx.check(mlx.mlx_array_set(&sel_full, sel_tok));
    }
    var kv_idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kv_idx);
    try mlx.check(mlx.mlx_arange(&kv_idx, 0, @floatFromInt(kv), 1.0, .int32, s));
    var kv_idx2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kv_idx2);
    try mlx.check(mlx.mlx_reshape(&kv_idx2, kv_idx, &[_]c_int{ 1, kv }, 2, s));
    // tail_start[p] = ratio * floor((p+1)/ratio), via integer arithmetic.
    const one_i = mlx.mlx_array_new_int(1);
    defer _ = mlx.mlx_array_free(one_i);
    const ratio_i = mlx.mlx_array_new_int(ratio);
    defer _ = mlx.mlx_array_free(ratio_i);
    var p1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(p1);
    try mlx.check(mlx.mlx_add(&p1, pos2, one_i, s));
    var nb_p = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(nb_p);
    try mlx.check(mlx.mlx_floor_divide(&nb_p, p1, ratio_i, s));
    var tail_start = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tail_start);
    try mlx.check(mlx.mlx_multiply(&tail_start, nb_p, ratio_i, s)); // [S,1]
    var tail_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tail_mask);
    try mlx.check(mlx.mlx_greater_equal(&tail_mask, kv_idx2, tail_start, s)); // [S,kv]
    var tok_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tok_mask);
    try mlx.check(mlx.mlx_logical_or(&tok_mask, sel_full, tail_mask, s));
    var causal = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(causal);
    try mlx.check(mlx.mlx_less_equal(&causal, kv_idx2, pos2, s)); // [S, kv]
    var mask3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask3);
    try mlx.check(mlx.mlx_logical_and(&mask3, tok_mask, causal, s));
    var mask = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&mask, mask3, &[_]c_int{ batch, 1, seq_len, kv }, 4, s));
    return mask;
}
