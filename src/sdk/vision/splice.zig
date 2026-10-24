//! sdk/vision/splice, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns spliceVisionEmbeddings, spliceVisionRows, applyVisionEmbeddingsWith.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ForwardCtx = @import("../../engines/mlx/forward_ctx.zig").ForwardCtx;
const testing = std.testing;
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;

/// Splice vision embeddings into text embeddings at image_token_id positions.
/// h: [B, seq_len, hidden] text embeddings
/// token_ids: [B, seq_len] original token IDs
/// vision_emb: [B, N_img, hidden] vision embeddings
/// Returns new h with vision embeddings replacing image token positions.
/// masked_scatter: replaces image token positions with vision features.
/// Matches Python reference: cumsum-based indexing into flattened source.
pub fn spliceVisionEmbeddings(self: *Transformer, h: mlx.mlx_array, token_ids: mlx.mlx_array, vision_emb: mlx.mlx_array, image_token_id: u32, audio_token_id: u32, video_token_id: u32, row_offset: usize) !mlx.mlx_array {
    return spliceVisionRows(h, token_ids, vision_emb, image_token_id, audio_token_id, video_token_id, row_offset, self.s);
}

/// Scatter vision/audio embedding rows into `h` at placeholder-token
/// positions. `row_offset` is the number of placeholder tokens earlier
/// chunks of this prefill already consumed (chunked vision prefill): the
/// row lookup is a cumsum over THIS forward's mask, so the offset is what
/// keeps a chunk boundary inside an image from restarting the scatter at
/// row 0. Frees `h` and returns the spliced replacement.
pub fn spliceVisionRows(h: mlx.mlx_array, token_ids: mlx.mlx_array, vision_emb: mlx.mlx_array, image_token_id: u32, audio_token_id: u32, video_token_id: u32, row_offset: usize, s: mlx.mlx_stream) !mlx.mlx_array {
    const h_shape = mlx.getShape(h);

    // mask = token_ids is any placeholder id. Every modality splices through
    // this one channel: the embedding tensor concatenates each item's rows
    // in prompt order, so a single sequence-order scatter lands each row in
    // its slot. An id of 0 is unset (an audio-only model has no image
    // token), never token 0.
    var mask_2d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask_2d);
    const ids_shape = mlx.getShape(token_ids);
    try mlx.check(mlx.mlx_zeros(&mask_2d, ids_shape.ptr, ids_shape.len, .bool_, s));
    for ([_]u32{ image_token_id, audio_token_id, video_token_id }) |extra_id| {
        if (extra_id == 0) continue;
        const extra_arr = mlx.mlx_array_new_int(@intCast(extra_id));
        defer _ = mlx.mlx_array_free(extra_arr);
        var extra_mask = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(extra_mask);
        try mlx.check(mlx.mlx_equal(&extra_mask, token_ids, extra_arr, s));
        var combined = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_logical_or(&combined, mask_2d, extra_mask, s));
        _ = mlx.mlx_array_free(mask_2d);
        mask_2d = combined;
    }

    // Expand mask to [B, seq_len, hidden] via broadcast
    const expand_shape = [_]c_int{ h_shape[0], h_shape[1], 1 };
    var mask_3d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask_3d);
    try mlx.check(mlx.mlx_reshape(&mask_3d, mask_2d, &expand_shape, 3, s));

    // Broadcast to full shape via logical and with ones
    var mask_expanded = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask_expanded);
    {
        var ones_h = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ones_h);
        try mlx.check(mlx.mlx_ones(&ones_h, h_shape.ptr, 3, .bool_, s));
        try mlx.check(mlx.mlx_multiply(&mask_expanded, mask_3d, ones_h, s));
    }

    // Flatten everything
    const total = h_shape[0] * h_shape[1] * h_shape[2];
    const flat_shape = [_]c_int{total};

    var mask_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask_flat);
    try mlx.check(mlx.mlx_reshape(&mask_flat, mask_expanded, &flat_shape, 1, s));

    // mask_int = mask_flat.astype(int32)
    var mask_int = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask_int);
    try mlx.check(mlx.mlx_astype(&mask_int, mask_flat, .int32, s));

    // indices = cumsum(mask_int, axis=0) - 1 + row_offset*hidden. The
    // cumsum counts masked ELEMENTS of this forward (hidden consecutive
    // per placeholder token); earlier chunks consumed row_offset rows of
    // the flat source, i.e. row_offset*hidden elements.
    var cumsum_arr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cumsum_arr);
    try mlx.check(mlx.mlx_cumsum(&cumsum_arr, mask_int, 0, false, true, s));
    const elem_offset: i64 = @as(i64, @intCast(row_offset)) * @as(i64, @intCast(h_shape[2]));
    const off_i = mlx.mlx_array_new_int(@intCast(elem_offset - 1));
    defer _ = mlx.mlx_array_free(off_i);
    var indices = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(indices);
    try mlx.check(mlx.mlx_add(&indices, cumsum_arr, off_i, s));

    // source = vision_emb.flatten()
    const ve_shape = mlx.getShape(vision_emb);
    const source_size = ve_shape[0] * ve_shape[1] * ve_shape[2];
    const source_shape = [_]c_int{source_size};
    var source_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(source_flat);
    try mlx.check(mlx.mlx_reshape(&source_flat, vision_emb, &source_shape, 1, s));

    // indices_mod = indices % source_size
    const source_size_arr = mlx.mlx_array_new_int(source_size);
    defer _ = mlx.mlx_array_free(source_size_arr);
    var indices_mod = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(indices_mod);
    try mlx.check(mlx.mlx_remainder(&indices_mod, indices, source_size_arr, s));

    // aligned = source[indices_mod]
    var aligned = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(aligned);
    try mlx.check(mlx.mlx_take(&aligned, source_flat, indices_mod, s));

    // Cast aligned to h's dtype: a mismatch promotes the whole residual (f32 on a Prism pack's f16 stream).
    var aligned_bf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(aligned_bf);
    try mlx.check(mlx.mlx_astype(&aligned_bf, aligned, mlx.mlx_array_dtype(h), s));

    // input_flat = h.flatten()
    var input_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(input_flat);
    try mlx.check(mlx.mlx_reshape(&input_flat, h, &flat_shape, 1, s));

    // result_flat = where(mask_flat, aligned_bf, input_flat)
    var result_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(result_flat);
    try mlx.check(mlx.mlx_where(&result_flat, mask_flat, aligned_bf, input_flat, s));

    // Reshape back to [B, seq_len, hidden]
    var result = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&result, result_flat, h_shape.ptr, 3, s));
    _ = mlx.mlx_array_free(h);
    return result;
}

/// Apply vision embeddings to text embeddings during prefill.
/// Handles scaling and splicing at image_token_id positions.
/// Returns the (potentially modified) h; caller should replace their h with the result.
pub fn applyVisionEmbeddingsWith(self: *Transformer, ctx: *ForwardCtx, h: mlx.mlx_array, token_ids: mlx.mlx_array) !mlx.mlx_array {
    const cfg = &self.config;
    const h_shape = mlx.getShape(h);
    // Only during prefill (seq_len > 1)
    if (h_shape[1] <= 1) return h;
    const ve = ctx.vision_embeddings orelse return h;
    if (cfg.image_token_id == 0 and cfg.audio_token_id == 0 and cfg.video_token_id == 0) return h;

    // Vision embeddings come out of the MultimodalEmbedder already in text-hidden space;
    // mlx-vlm does NOT re-scale them by sqrt(hidden) the way text embeddings are scaled
    // at LM embedding time. Splice directly — scaling here corrupts the MoE router's
    // magnitude assumptions (visible as "please provide an image" responses on 26B MoE).
    return self.spliceVisionEmbeddings(h, token_ids, ve, cfg.image_token_id, cfg.audio_token_id, cfg.video_token_id, ctx.vision_splice_offset);
}

// ── tests ──

// ── Chunked vision splice (issue #197) ───────────────────────────────────────
// A vision prompt prefills in chunks; the splice's row lookup is a cumsum over
// the CURRENT forward's placeholder mask, so each chunk passes the rows earlier
// chunks consumed as `row_offset` — a restarted scatter silently reuses the
// first image rows in every chunk (coherent output, wrong image content).

test "spliceVisionRows: two chunks with row offsets equal the whole-prompt splice" {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);

    const hidden: c_int = 4;
    const seq: c_int = 6;
    const img_id: u32 = 99;
    // token ids: [7, 99, 99, 8, 99, 9] — placeholders at 1, 2 (chunk A) and 4
    // (chunk B), so the boundary lands INSIDE the image row sequence.
    const ids = [_]i32{ 7, 99, 99, 8, 99, 9 };
    // h rows: row i filled with i+1 (bf16-exact values).
    var h_data: [6 * 4]f32 = undefined;
    for (0..6) |r| for (0..4) |c| {
        h_data[r * 4 + c] = @floatFromInt(r + 1);
    };
    // vision rows: row k filled with 100+k — distinct per row, so a restarted
    // scatter (row 0 reused in chunk B) produces different bytes.
    var ve_data: [3 * 4]f32 = undefined;
    for (0..3) |r| for (0..4) |c| {
        ve_data[r * 4 + c] = @floatFromInt(100 + r);
    };
    const ve_shape = [_]c_int{ 1, 3, hidden };
    const ve = mlx.mlx_array_new_data(&ve_data, &ve_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(ve);

    const readRow = struct {
        fn f(arr: mlx.mlx_array, row: usize, out: *[4]f32, st: mlx.mlx_stream) !void {
            var f32arr = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(f32arr);
            try mlx.check(mlx.mlx_astype(&f32arr, arr, .float32, st));
            try mlx.check(mlx.mlx_array_eval(f32arr));
            const ptr = mlx.mlx_array_data_float32(f32arr).?;
            for (0..4) |c| out[c] = ptr[row * 4 + c];
        }
    }.f;

    // Whole-prompt splice (the reference: what the unchunked path computes).
    var full_rows: [6][4]f32 = undefined;
    {
        const h_shape = [_]c_int{ 1, seq, hidden };
        const h = mlx.mlx_array_new_data(&h_data, &h_shape, 3, .float32);
        const ids_shape = [_]c_int{ 1, seq };
        const tid = mlx.mlx_array_new_data(&ids, &ids_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(tid);
        const spliced = try Transformer.spliceVisionRows(h, tid, ve, img_id, 0, 0, 0, s);
        defer _ = mlx.mlx_array_free(spliced);
        for (0..6) |r| try readRow(spliced, r, &full_rows[r], s);
    }

    // Chunk A = tokens [0..3), offset 0; chunk B = tokens [3..6), offset 2
    // (chunk A consumed 2 placeholder rows).
    const chunks = [_]struct { start: usize, len: c_int, off: usize }{
        .{ .start = 0, .len = 3, .off = 0 },
        .{ .start = 3, .len = 3, .off = 2 },
    };
    for (chunks) |ck| {
        const h_shape = [_]c_int{ 1, ck.len, hidden };
        const h = mlx.mlx_array_new_data(h_data[ck.start * 4 ..].ptr, &h_shape, 3, .float32);
        const ids_shape = [_]c_int{ 1, ck.len };
        const tid = mlx.mlx_array_new_data(ids[ck.start..].ptr, &ids_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(tid);
        const spliced = try Transformer.spliceVisionRows(h, tid, ve, img_id, 0, 0, ck.off, s);
        defer _ = mlx.mlx_array_free(spliced);
        for (0..@intCast(ck.len)) |r| {
            var got: [4]f32 = undefined;
            try readRow(spliced, r, &got, s);
            for (0..4) |c| try std.testing.expectEqual(full_rows[ck.start + r][c], got[c]);
        }
    }
}

test "spliceVisionRows: video placeholders are scattered like image rows (Qwen video_token_id)" {
    // Live 2026-08-26 (qwen4_exp, 4-frame clip): 690 <|video_pad|> tokens were
    // inserted and the tower encoded 690 rows, but the mask keyed on image |
    // audio ids only, so no video row ever reached the prompt and the model
    // described a clip it never saw. Same hole on qwen3_5 (its e2e bar of
    // >= 2 of 4 generic subjects was passing on hallucination).
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const hidden: c_int = 4;
    const img_id: u32 = 99;
    const vid_id: u32 = 77;
    // [99, 7, 77, 77] — one image row, text, two video rows from ONE source.
    const ids = [_]i32{ 99, 7, 77, 77 };
    var h_data: [16]f32 = @splat(0);
    var ve_data: [3 * 4]f32 = undefined;
    for (0..3) |r| for (0..4) |c| {
        ve_data[r * 4 + c] = @floatFromInt(50 + r);
    };
    const ve_shape = [_]c_int{ 1, 3, hidden };
    const ve = mlx.mlx_array_new_data(&ve_data, &ve_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(ve);
    const h_shape = [_]c_int{ 1, 4, hidden };
    const h = mlx.mlx_array_new_data(&h_data, &h_shape, 3, .float32);
    const ids_shape = [_]c_int{ 1, 4 };
    const tid = mlx.mlx_array_new_data(&ids, &ids_shape, 2, .int32);
    defer _ = mlx.mlx_array_free(tid);
    const spliced = try Transformer.spliceVisionRows(h, tid, ve, img_id, 0, vid_id, 0, s);
    defer _ = mlx.mlx_array_free(spliced);
    var f32arr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f32arr);
    try mlx.check(mlx.mlx_astype(&f32arr, spliced, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f32arr));
    const ptr = mlx.mlx_array_data_float32(f32arr).?;
    try std.testing.expectEqual(@as(f32, 50), ptr[0]); // image row 0
    try std.testing.expectEqual(@as(f32, 0), ptr[4]); // text untouched
    try std.testing.expectEqual(@as(f32, 51), ptr[8]); // video row 1
    try std.testing.expectEqual(@as(f32, 52), ptr[12]); // video row 2
}

test "spliceVisionRows: audio placeholders share the row stream in prompt order" {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const hidden: c_int = 4;
    const img_id: u32 = 99;
    const aud_id: u32 = 88;
    // [99, 7, 88] — image row then audio row from ONE concatenated source.
    const ids = [_]i32{ 99, 7, 88 };
    var h_data: [12]f32 = @splat(0);
    var ve_data: [2 * 4]f32 = undefined;
    for (0..2) |r| for (0..4) |c| {
        ve_data[r * 4 + c] = @floatFromInt(50 + r);
    };
    const ve_shape = [_]c_int{ 1, 2, hidden };
    const ve = mlx.mlx_array_new_data(&ve_data, &ve_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(ve);
    const h_shape = [_]c_int{ 1, 3, hidden };
    const h = mlx.mlx_array_new_data(&h_data, &h_shape, 3, .float32);
    const ids_shape = [_]c_int{ 1, 3 };
    const tid = mlx.mlx_array_new_data(&ids, &ids_shape, 2, .int32);
    defer _ = mlx.mlx_array_free(tid);
    const spliced = try Transformer.spliceVisionRows(h, tid, ve, img_id, aud_id, 0, 0, s);
    defer _ = mlx.mlx_array_free(spliced);
    var f32arr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f32arr);
    try mlx.check(mlx.mlx_astype(&f32arr, spliced, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f32arr));
    const ptr = mlx.mlx_array_data_float32(f32arr).?;
    try std.testing.expectEqual(@as(f32, 50), ptr[0]); // image row 0
    try std.testing.expectEqual(@as(f32, 0), ptr[4]); // text untouched
    try std.testing.expectEqual(@as(f32, 51), ptr[8]); // audio row 1
}

test "spliceVisionRows: an unset placeholder id (0) never claims token 0" {
    // An audio-only model has no image token: token 0 (Qwen's "!") stays text.
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const hidden: c_int = 4;
    const ids = [_]i32{ 0, 88, 7 };
    var h_data: [12]f32 = @splat(0);
    var ve_data: [4]f32 = @splat(50);
    const ve = mlx.mlx_array_new_data(&ve_data, &[_]c_int{ 1, 1, hidden }, 3, .float32);
    defer _ = mlx.mlx_array_free(ve);
    const h = mlx.mlx_array_new_data(&h_data, &[_]c_int{ 1, 3, hidden }, 3, .float32);
    const tid = mlx.mlx_array_new_data(&ids, &[_]c_int{ 1, 3 }, 2, .int32);
    defer _ = mlx.mlx_array_free(tid);
    const spliced = try Transformer.spliceVisionRows(h, tid, ve, 0, 88, 0, 0, s);
    defer _ = mlx.mlx_array_free(spliced);
    var f32arr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f32arr);
    try mlx.check(mlx.mlx_astype(&f32arr, spliced, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f32arr));
    const ptr = mlx.mlx_array_data_float32(f32arr).?;
    try std.testing.expectEqual(@as(f32, 0), ptr[0]); // token 0 untouched
    try std.testing.expectEqual(@as(f32, 50), ptr[4]); // audio row
}

test "spliceVisionRows: the residual keeps its own dtype (an f16 stream stays f16)" {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    for ([_]mlx.mlx_dtype{ .float16, .bfloat16 }) |dt| {
        var h_data: [8]f32 = @splat(1);
        var ve_data: [4]f32 = @splat(50);
        const ve = mlx.mlx_array_new_data(&ve_data, &[_]c_int{ 1, 1, 4 }, 3, .float32);
        defer _ = mlx.mlx_array_free(ve);
        const h32 = mlx.mlx_array_new_data(&h_data, &[_]c_int{ 1, 2, 4 }, 3, .float32);
        defer _ = mlx.mlx_array_free(h32);
        var h = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&h, h32, dt, s));
        const ids = [_]i32{ 9, 250 };
        const tid = mlx.mlx_array_new_data(&ids, &[_]c_int{ 1, 2 }, 2, .int32);
        defer _ = mlx.mlx_array_free(tid);
        const spliced = try Transformer.spliceVisionRows(h, tid, ve, 250, 0, 0, 0, s);
        defer _ = mlx.mlx_array_free(spliced);
        try std.testing.expectEqual(dt, mlx.mlx_array_dtype(spliced));
    }
}
