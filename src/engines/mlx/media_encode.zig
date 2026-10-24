//! Vision and embedding work items, encoded on the inference thread.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const generate_mod = @import("generate.zig");
const model_mod = @import("model.zig");
const vision_mod = @import("vision_encoder.zig");
const radio_vision = @import("../../plugins/nemotron_h/radio_vision.zig");
const model_registry_mod = @import("../../host/registry/registry.zig");
const log = @import("../../sdk/util/log.zig");
const VisionEncoder = vision_mod.VisionEncoder;

const Scheduler = @import("scheduler.zig").Scheduler;

/// Phase A4: pixel data for a single image, decoded by the connection
/// thread (CPU only — stb_image / libwebp). The inference thread wraps this
/// in an `mlx_array` via `mlx_array_new_data` and runs the vision encoder.
pub const VisionImagePixels = struct {
    /// Raw bytes holding float32 pixel data. Gemma: CHW (3 × H × W × 4). Qwen3-VL:
    /// merge-order pixel_values (N × C·tps·ps·ps × 4). Borrowed; must outlive the
    /// encodeVision call (which blocks until completion).
    pixels: []const u8,
    width: u32,
    height: u32,
    /// Qwen3-VL only: full patch grid (0 ⇒ Gemma CHW). Selects QwenVision.
    grid_h: u32 = 0,
    grid_w: u32 = 0,
};

/// Qwen3-VL video: pre-patchified pixel_values for ALL `grid_t` temporal-patch
/// groups, concatenated (see `qwen_vision.buildPixelValuesVideo`). Borrowed;
/// must outlive the encodeVision call.
pub const VisionVideoPixels = struct {
    pixels: []const u8,
    grid_t: u32,
    grid_h: u32,
    grid_w: u32,
};

/// One encoder input. `audio` is raw float32-LE 16 kHz mono PCM, framed into
/// `audio_samples_per_token` tokens by the unified audio embedder.
pub const VisionItem = union(enum) {
    image: VisionImagePixels,
    video: VisionVideoPixels,
    audio: []const u8,
};

/// Phase A4: vision-encode work item. Conn thread fills `images` (raw pixel
/// data, CPU-only) and calls `Scheduler.encodeVision`, which posts the
/// request and blocks until the inference thread fills `result` and signals
/// `done`. Ownership of `result` transfers to the caller on success — pass
/// to `scheduler.submit(.{ .vision_embeddings = arr, ... })` and the slot's
/// `deinit` will free it.
pub const VisionEncodeRequest = struct {
    /// Plan 05 Phase D: target model whose `vision_encoder` services this
    /// request. The conn thread holds a refcount (via `ensureLoaded`) for
    /// the duration of the call.
    model: *model_registry_mod.LoadedModel,
    /// Encoder inputs in prompt order. Borrowed; must outlive the call.
    items: []const VisionItem,
    /// Each item's pixel/PCM hash, parallel to `items`: the encoder cache key.
    item_keys: []const u64,
    /// Output: soft tokens each item produced, parallel to `items`. Caller-owned.
    item_tokens: []usize,
    /// Output: every item's soft tokens, concatenated along the token axis in
    /// prompt order (the splice scatters them positionally). Ownership
    /// transfers to the caller.
    result: ?mlx.mlx_array = null,
    /// Output: error name on failure. Owned by `allocator`; caller frees.
    error_name: ?[]const u8 = null,
    /// RADIO video only, parallel to `items`. A mask is the full pre-EVS row
    /// count (true = keep). Null entries are not video. Owned by `allocator`
    /// on success; the caller frees each mask and the array.
    item_keep: ?[]?[]bool = null,
    /// Done flag (under done_mu). Caller's wait-loop drains the cond when
    /// this flips true.
    done: bool = false,
    allocator: std.mem.Allocator,
    done_mu: std.Io.Mutex = .init,
    done_cond: std.Io.Condition = .init,
};

/// Phase: embedding work item for encoder-only models. Conn thread fills
/// `token_seqs` and calls `Scheduler.computeEmbeddings`; the inference
/// thread services the request via `generate.computeEmbeddingsBatch` — one
/// padded, key-masked GPU forward per EMBED_MAX_BATCH chunk — and writes
/// the float vectors into `results` (caller frees). Mirrors the
/// VisionEncodeRequest pattern.
pub const EmbedRequest = struct {
    /// Plan 05 Phase D: target model whose `transformer` services this
    /// request. The conn thread holds a refcount for the duration.
    model: *model_registry_mod.LoadedModel,
    /// Tokenized inputs, one slice per text. Borrowed; must outlive the call.
    token_seqs: []const []const u32,
    /// The soft-token rows [1, n, hidden] of the ONE prompt's images and video, in prompt order. The request
    /// owns them: the inference thread frees them when the forward is done.
    vision_embeddings: ?mlx.mlx_array = null,
    /// Output: one pooled L2-normalized embedding per input on success.
    /// Rows + outer slice owned by `allocator`; caller frees.
    results: ?[][]f32 = null,
    /// Output: error name on failure. Owned by `allocator`; caller frees.
    error_name: ?[]const u8 = null,
    done: bool = false,
    allocator: std.mem.Allocator,
    done_mu: std.Io.Mutex = .init,
    done_cond: std.Io.Condition = .init,
};

/// Phase A4: encode one or more images on the inference thread. Mirrors the
/// existing `processVisionImages` shape but writes the result into a request
/// struct + signals done, so the conn thread (blocked in `encodeVision`)
/// gets the output. On error, sets `req.error_name` and still signals done.
/// Plan 05 Phase D: routes the encode through `req.model.vision_encoder`,
/// not the scheduler's borrowed-view singleton — each LoadedModel has its
/// own vision encoder when applicable.
pub fn runVisionEncode(sch: *Scheduler, req: *VisionEncodeRequest) void {
    const vision_enc = req.model.vision_encoder orelse {
        finishVisionRequest(sch, req, "VisionEncoderNotLoaded");
        return;
    };
    if (req.items.len == 0) {
        finishVisionRequest(sch, req, "EmptyImages");
        return;
    }

    var emb_parts = std.ArrayList(mlx.mlx_array).empty;
    defer emb_parts.deinit(req.allocator);
    var cached: usize = 0;
    const epoch = vision_enc.emb_cache.beginRequest();
    for (req.items, req.item_keys, 0..) |item, key, i| {
        var emb: mlx.mlx_array = undefined;
        if (vision_enc.emb_cache.get(key)) |hit| {
            emb = hit;
            cached += 1;
        } else {
            emb = encodeVisionItem(req, vision_enc, item) catch |err| {
                for (emb_parts.items) |e| _ = mlx.mlx_array_free(e);
                finishVisionRequest(sch, req, @errorName(err));
                return;
            };
            vision_enc.emb_cache.put(vision_enc.allocator, key, emb, vision_mod.EmbeddingCache.CAP_BYTES, epoch);
        }
        req.item_tokens[i] = @intCast(mlx.getShape(emb)[1]);
        emb_parts.append(req.allocator, emb) catch |err| {
            _ = mlx.mlx_array_free(emb);
            for (emb_parts.items) |e| _ = mlx.mlx_array_free(e);
            finishVisionRequest(sch, req, @errorName(err));
            return;
        };
    }
    if (cached > 0) log.info("  [vision-cache] {d}/{d} encoder pieces reused\n", .{ cached, req.items.len });
    var combined: mlx.mlx_array = undefined;
    if (emb_parts.items.len == 1) {
        combined = emb_parts.items[0];
        emb_parts.items[0] = mlx.mlx_array_new(); // sentinel so the free below is a no-op
    } else {
        const cat_vec = mlx.mlx_vector_array_new_data(emb_parts.items.ptr, emb_parts.items.len);
        defer _ = mlx.mlx_vector_array_free(cat_vec);
        combined = mlx.mlx_array_new();
        if (mlx.mlx_concatenate_axis(&combined, cat_vec, 1, vision_enc.s) != 0) {
            _ = mlx.mlx_array_free(combined);
            for (emb_parts.items) |e| _ = mlx.mlx_array_free(e);
            finishVisionRequest(sch, req, "ConcatenateFailed");
            return;
        }
    }
    for (emb_parts.items) |e| _ = mlx.mlx_array_free(e);

    applyRadioVideoEvs(req, vision_enc, &combined) catch |err| {
        _ = mlx.mlx_array_free(combined);
        finishVisionRequest(sch, req, @errorName(err));
        return;
    };

    req.done_mu.lockUncancelable(sch.io);
    defer req.done_mu.unlock(sch.io);
    req.result = combined;
    req.done = true;
    req.done_cond.broadcast(sch.io);
}

/// Drop RADIO video rows that EVS rejects, and record the keep-mask so the
/// prompt can delete the same `<image>` positions. Runs on the inference
/// thread. The encoder cache keeps the unpruned rows; this replaces `combined`.
fn applyRadioVideoEvs(req: *VisionEncodeRequest, vision_enc: *VisionEncoder, combined: *mlx.mlx_array) !void {
    const rv = if (vision_enc.radio) |*r| r else return;
    var videos: usize = 0;
    for (req.items) |item| videos += @intFromBool(item == .video);
    if (videos == 0) return;

    const rate: f64 = if (req.model.config) |c| c.video_pruning_rate else model_mod.RADIO_VIDEO_PRUNING_RATE_DEFAULT;
    const keeps = try req.allocator.alloc(?[]bool, req.items.len);
    errdefer req.allocator.free(keeps);
    @memset(keeps, null);
    var masks_live = true;
    errdefer if (masks_live) {
        for (keeps) |m| if (m) |s| req.allocator.free(s);
    };

    var any_drop = false;
    var offset: usize = 0;
    var kept_total: usize = 0;
    for (req.items, req.item_tokens, keeps) |item, *ntok, *slot| {
        const full = ntok.*;
        if (item == .video) {
            const g = radio_vision.videoGeometry(item.video.grid_t, item.video.grid_h, item.video.grid_w) orelse return error.InvalidVisionInput;
            const expect: usize = @as(usize, g.groups) * g.spatial;
            if (full != expect) return error.InvalidVisionInput;
            const mask = try rv.videoKeepMask(req.allocator, combined.*, offset, g.groups, g.spatial, rate);
            var kept: usize = 0;
            for (mask) |b| kept += @intFromBool(b);
            if (kept != full) any_drop = true;
            ntok.* = kept;
            slot.* = mask;
            log.info("  RADIO video EVS: {d} frames, {d} tubelets x {d} tokens, kept {d}\n", .{ g.frames, g.groups, g.spatial, kept });
            kept_total += kept;
        } else kept_total += full;
        offset += full;
    }

    if (any_drop) {
        if (kept_total == 0) return error.InvalidVisionInput;
        const indices = try req.allocator.alloc(i32, kept_total);
        defer req.allocator.free(indices);
        var row: i32 = 0;
        var dst: usize = 0;
        for (req.item_tokens, keeps) |ntok, slot| {
            if (slot) |mask| {
                for (mask) |keep| {
                    if (keep) {
                        indices[dst] = row;
                        dst += 1;
                    }
                    row += 1;
                }
            } else {
                var i: usize = 0;
                while (i < ntok) : (i += 1) {
                    indices[dst] = row;
                    dst += 1;
                    row += 1;
                }
            }
        }
        if (dst != kept_total) return error.InvalidVisionInput;
        const ishape = [_]c_int{@intCast(indices.len)};
        const ix = mlx.mlx_array_new_data(indices.ptr, &ishape, 1, .int32);
        defer _ = mlx.mlx_array_free(ix);
        var taken = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(taken);
        try mlx.check(mlx.mlx_take_axis(&taken, combined.*, ix, 1, rv.s));
        _ = mlx.mlx_array_free(combined.*);
        combined.* = taken;
    }

    req.item_keep = keeps;
    masks_live = false;
}

/// `[1, n, hidden]` soft tokens for one item.
fn encodeVisionItem(req: *VisionEncodeRequest, vision_enc: anytype, item: VisionItem) !mlx.mlx_array {
    switch (item) {
        .image => |img| {
            if (img.grid_h > 0) {
                // Patch-grid ViT: pixels hold pixel_values [N, feat]; the tower
                // produces [1, N/merge², out_hidden].
                const n: usize = @as(usize, img.grid_h) * img.grid_w;
                const feat: usize = (img.pixels.len / 4) / n;
                const shape = [_]c_int{ @intCast(n), @intCast(feat) };
                const pixel_arr = mlx.mlx_array_new_data(img.pixels.ptr, &shape, 2, .float32);
                defer _ = mlx.mlx_array_free(pixel_arr);
                return vision_enc.forwardPatches(pixel_arr, img.grid_h, img.grid_w);
            }
            const shape = [_]c_int{ 1, 3, @intCast(img.height), @intCast(img.width) };
            const pixel_arr = mlx.mlx_array_new_data(img.pixels.ptr, &shape, 4, .float32);
            defer _ = mlx.mlx_array_free(pixel_arr);
            return vision_enc.forward(pixel_arr);
        },
        .video => |vid| {
            if (vision_enc.radio != null) {
                // RADIO stores raw CHW frames. grid_h/grid_w are the shared pixel size.
                if (vid.grid_t == 0 or vid.grid_h == 0 or vid.grid_w == 0) return error.InvalidVisionInput;
                const bytes: usize = @as(usize, vid.grid_t) * 3 * vid.grid_h * vid.grid_w * 4;
                if (vid.pixels.len != bytes) return error.InvalidVisionInput;
                const shape = [_]c_int{ @intCast(vid.grid_t), 3, @intCast(vid.grid_h), @intCast(vid.grid_w) };
                const pixel_arr = mlx.mlx_array_new_data(vid.pixels.ptr, &shape, 4, .float32);
                defer _ = mlx.mlx_array_free(pixel_arr);
                return vision_enc.forwardRadioVideo(pixel_arr);
            }
            const n: usize = @as(usize, vid.grid_t) * vid.grid_h * vid.grid_w;
            const feat: usize = (vid.pixels.len / 4) / n;
            const shape = [_]c_int{ @intCast(n), @intCast(feat) };
            const pixel_arr = mlx.mlx_array_new_data(vid.pixels.ptr, &shape, 2, .float32);
            defer _ = mlx.mlx_array_free(pixel_arr);
            return vision_enc.forwardVideoPatches(pixel_arr, vid.grid_t, vid.grid_h, vid.grid_w);
        },
        .audio => |clip| {
            if (vision_enc.qwen_audio) |*qa| return qa.forward(clip);
            const cfg = req.model.config orelse return error.NoConfig;
            const spt: usize = if (cfg.audio_samples_per_token > 0) cfg.audio_samples_per_token else 640;
            const n_frames = (clip.len / 4 + spt - 1) / spt;
            if (n_frames == 0) return error.EmptyAudio;
            const buf = try req.allocator.alloc(f32, n_frames * spt);
            defer req.allocator.free(buf); // mlx_array_new_data copies
            @memset(buf, 0);
            @memcpy(std.mem.sliceAsBytes(buf)[0..clip.len], clip);
            const shape = [_]c_int{ 1, @intCast(n_frames), @intCast(spt) };
            const frames_arr = mlx.mlx_array_new_data(buf.ptr, &shape, 3, .float32);
            defer _ = mlx.mlx_array_free(frames_arr);
            return vision_enc.forwardAudio(frames_arr);
        },
    }
}

fn finishVisionRequest(sch: *Scheduler, req: *VisionEncodeRequest, err_name: []const u8) void {
    req.done_mu.lockUncancelable(sch.io);
    defer req.done_mu.unlock(sch.io);
    if (req.error_name) |old| req.allocator.free(old);
    req.error_name = req.allocator.dupe(u8, err_name) catch null;
    req.done = true;
    req.done_cond.broadcast(sch.io);
}

/// Service one embedding request on the inference thread. Runs the batched
/// forward pass via `generate.computeEmbeddingsBatch(xfm, ...)`, which resets
/// the global xfm.cache before every sub-batch, and wakes the conn thread.
pub fn runEmbedRequest(sch: *Scheduler, req: *EmbedRequest) void {
    const xfm_ptr = req.model.transformer.?;
    // Held in a local: the request lives on the waiting thread's stack and is gone once `done` is observed.
    const vision = req.vision_embeddings;
    defer if (vision) |v| {
        _ = mlx.mlx_array_free(v);
    };
    const results = generate_mod.computeEmbeddingsBatchWith(req.allocator, xfm_ptr, req.token_seqs, vision) catch |err| {
        finishEmbedRequest(sch, req, @errorName(err));
        return;
    };
    req.done_mu.lockUncancelable(sch.io);
    defer req.done_mu.unlock(sch.io);
    req.results = results;
    req.done = true;
    req.done_cond.broadcast(sch.io);
}

fn finishEmbedRequest(sch: *Scheduler, req: *EmbedRequest, err_name: []const u8) void {
    req.done_mu.lockUncancelable(sch.io);
    defer req.done_mu.unlock(sch.io);
    if (req.error_name) |old| req.allocator.free(old);
    req.error_name = req.allocator.dupe(u8, err_name) catch null;
    req.done = true;
    req.done_cond.broadcast(sch.io);
}
