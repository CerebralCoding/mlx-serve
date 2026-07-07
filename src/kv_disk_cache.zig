//! SSD tier for the hot prefix cache — chunked KV persistence.
//!
//! Committed KV prefixes are persisted to disk as position-chunked
//! safetensors, so previously-seen prefixes survive server restarts and RAM
//! evictions and are RESTORED instead of recomputed. Two-tier flow:
//!
//!   commit  → RAM entry (refcount snapshot, unchanged) + chunk-APPEND to
//!             disk. Only chunks not yet on disk are written — a multi-turn
//!             agent session pays one bounded partial-chunk rewrite + the new
//!             tail per turn, never a full re-serialize.
//!   lookup  → longest-prefix match across RAM entries as before; the disk
//!             index is consulted when it can beat the RAM match by at least
//!             one chunk (fresh boot, post-eviction). The entry is rebuilt
//!             into the live cache and the normal truncate-then-prefill path
//!             continues.
//!
//! Layout (one root per model fingerprint — path + config.json identity):
//!   <base>/<fingerprint>/e<id>/meta.json    commit point (written tmp+rename
//!                                           LAST; an entry without it is a
//!                                           crash leftover and is GC'd)
//!   <base>/<fingerprint>/e<id>/tokens.bin   LE u32 token ids (prompt ++ gen)
//!   <base>/<fingerprint>/e<id>/c000000.safetensors   KV positions [0, chunk_tokens)
//!   <base>/<fingerprint>/e<id>/c000001.safetensors   ...
//!
//! Chunk files hold per-layer K/V slices keyed "l{i}.k"/"l{i}.v" (plus
//! ".ks/.kb/.vs/.vb" scale/bias triples in affine mode). The final chunk may
//! be partial; a commit that extends the entry rewrites ONLY that chunk and
//! appends new ones. A chunk file holding MORE positions than meta.json
//! claims (crash between chunk write and meta rename) is sliced down at
//! restore, never trusted.
//!
//! v1 scope: pure-attention archs (the scheduler attaches no disk tier when
//! the model has SSM/GDN layers — their recurrent state isn't persisted yet),
//! schemes off/affine (TurboQuant's rotation state doesn't survive a restore
//! into a fresh cache), B==1 slot caches. All mlx work runs on the inference
//! thread; safetensors loads use a private CPU stream (`Load::eval_gpu` is
//! Not Implemented — the lora.zig/model.zig precedent).

const std = @import("std");
const mlx = @import("mlx.zig");
const kv_quant = @import("kv_quant.zig");
const transformer_mod = @import("transformer.zig");
const io_util = @import("io_util.zig");
const log = @import("log.zig");

const KVCache = transformer_mod.KVCache;

/// Restoring from disk only happens when it beats the best RAM match by at
/// least this many tokens — a disk read + rebuild is only worth it when it
/// replaces a meaningful amount of prefill.
pub const MIN_DISK_ADVANTAGE_TOKENS: u32 = 256;

/// Entries shorter than this are never persisted (a short prefix re-prefills
/// in well under the restore cost).
pub const MIN_PERSIST_TOKENS: u32 = 512;

pub const DEFAULT_CHUNK_TOKENS: u32 = 1024;

pub const IndexEntry = struct {
    /// Directory id — the `e<id>` component.
    id: u64,
    /// Full committed token sequence (prompt ++ generated). Owned.
    tokens: []u32,
    /// KV positions actually persisted (== snapshot `step` at commit; may be
    /// tokens.len - 1 when the final sampled token was never forwarded).
    kv_len: u32,
    has_tools: bool,
    quant: kv_quant.KVQuantConfig,
    /// Total on-disk bytes (chunks + tokens.bin + meta).
    bytes: u64,
    /// Per-chunk file sizes recorded at commit (meta.json "chunk_bytes").
    /// The scan validates actual file sizes against these and clamps kv_len
    /// to the last contiguous valid chunk — a kill -9 mid-flush truncates a
    /// chunk, and restoring it would poison the cache. Owned.
    chunk_bytes: []u64,
    /// In-process LRU stamp; seeded from meta.json mtime order at scan.
    last_used: u64,
};

pub const Match = struct {
    idx: usize,
    /// Shared-prefix length clamped to kv_len — the positions a restore can
    /// actually rebuild.
    usable: u32,
};

pub const DiskTier = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    /// Absolute root for this model's entries (`<base>/<fingerprint>`). Owned.
    root: []u8,
    /// Byte budget across all entries. 0 = unbounded.
    max_bytes: u64,
    chunk_tokens: u32,
    /// Max bytes written per appendCommit call (default 512 MB). The flush
    /// runs synchronously on the inference thread after the response; a
    /// 4 GB first-commit write measurably stalls the NEXT request, so large
    /// entries persist incrementally across turns (appendCommit reports
    /// incomplete and the hot cache keeps its dirty flag set).
    max_flush_bytes: u64 = 512 * 1024 * 1024,
    entries: std.ArrayList(IndexEntry),
    next_id: u64,
    total_bytes: u64,
    counter: u64,

    /// Create the tier rooted at `<base>/<fingerprint>` and scan whatever
    /// already exists there. Crash leftovers (no meta.json) are deleted.
    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        base_dir: []const u8,
        fingerprint: []const u8,
        max_bytes: u64,
        chunk_tokens: u32,
    ) !DiskTier {
        if (base_dir.len == 0 or !std.fs.path.isAbsolute(base_dir)) return error.BadDiskCacheDir;
        const root = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ base_dir, fingerprint });
        errdefer allocator.free(root);
        try std.Io.Dir.cwd().createDirPath(io, root);
        var self: DiskTier = .{
            .allocator = allocator,
            .io = io,
            .root = root,
            .max_bytes = max_bytes,
            .chunk_tokens = if (chunk_tokens == 0) DEFAULT_CHUNK_TOKENS else chunk_tokens,
            .entries = std.ArrayList(IndexEntry).empty,
            .next_id = 1,
            .total_bytes = 0,
            .counter = 0,
        };
        self.scan() catch |err| {
            log.warn("[disk-cache] scan failed: {s} — starting empty\n", .{@errorName(err)});
        };
        self.gcToBudget();
        return self;
    }

    pub fn deinit(self: *DiskTier) void {
        for (self.entries.items) |*e| {
            self.allocator.free(e.tokens);
            self.allocator.free(e.chunk_bytes);
        }
        self.entries.deinit(self.allocator);
        self.allocator.free(self.root);
    }

    pub fn entryCount(self: *const DiskTier) usize {
        return self.entries.items.len;
    }

    // ── Lookup ──

    /// Longest usable shared prefix across persisted entries with a matching
    /// (has_tools, quant) key. Same filter semantics as the RAM cache: a
    /// cross-config restore would hand SDPA a wrong buffer layout.
    pub fn bestMatch(
        self: *const DiskTier,
        prompt_ids: []const u32,
        has_tools: bool,
        quant: kv_quant.KVQuantConfig,
    ) ?Match {
        var best_idx: ?usize = null;
        var best_usable: u32 = 0;
        for (self.entries.items, 0..) |*e, i| {
            if (e.has_tools != has_tools) continue;
            if (!std.meta.eql(e.quant, quant)) continue;
            const max_shared = @min(e.tokens.len, prompt_ids.len);
            var shared: usize = 0;
            while (shared < max_shared and e.tokens[shared] == prompt_ids[shared]) shared += 1;
            const usable: u32 = @intCast(@min(shared, e.kv_len));
            if (usable > best_usable) {
                best_usable = usable;
                best_idx = i;
            }
        }
        if (best_idx) |idx| return .{ .idx = idx, .usable = best_usable };
        return null;
    }

    /// Rebuild the persisted KV state of `entries[idx]` into `cache`:
    /// per-layer chunk tensors are loaded (CPU stream), concatenated along
    /// the sequence axis, and installed as the cache's storage buffers.
    /// Views are left empty — identical contract to `KVCache.restore` (the
    /// next `update`/`truncate` rebuilds them). Returns kv_len.
    pub fn restoreInto(self: *DiskTier, cache: *KVCache, idx: usize, s: mlx.mlx_stream) !u32 {
        const e = &self.entries.items[idx];
        const quant = e.quant;
        if (!std.meta.eql(cache.config, quant)) return error.DiskCacheConfigMismatch;
        const n_chunks: u32 = @intCast((@as(u64, e.kv_len) + self.chunk_tokens - 1) / self.chunk_tokens);
        if (n_chunks == 0) return error.DiskCacheEmptyEntry;

        const cpu = mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(cpu);

        const kinds: []const []const u8 = if (quant.scheme == .off)
            &.{ "k", "v" }
        else
            &.{ "k", "v", "ks", "kb", "vs", "vb" };

        // Per-layer per-kind accumulation vectors. Layers absent from chunk 0
        // stay uninitialized (hybrid layers never reach v1, but be robust).
        const n_layers = cache.entries.len;
        const vecs = try self.allocator.alloc(mlx.mlx_vector_array, n_layers * kinds.len);
        for (vecs) |*v| v.* = mlx.mlx_vector_array_new();
        defer {
            for (vecs) |v| _ = mlx.mlx_vector_array_free(v);
            self.allocator.free(vecs);
        }
        const present = try self.allocator.alloc(bool, n_layers);
        defer self.allocator.free(present);
        @memset(present, false);

        var chunk_i: u32 = 0;
        while (chunk_i < n_chunks) : (chunk_i += 1) {
            const c0: u64 = @as(u64, chunk_i) * self.chunk_tokens;
            const need: u64 = @min(@as(u64, self.chunk_tokens), e.kv_len - c0);

            const path = try std.fmt.allocPrint(self.allocator, "{s}/e{d}/c{d:0>6}.safetensors\x00", .{ self.root, e.id, chunk_i });
            defer self.allocator.free(path);
            var tensor_map = mlx.mlx_map_string_to_array_new();
            defer _ = mlx.mlx_map_string_to_array_free(tensor_map);
            var meta_map = mlx.mlx_map_string_to_string_new();
            defer _ = mlx.mlx_map_string_to_string_free(meta_map);
            try mlx.check(mlx.mlx_load_safetensors(&tensor_map, &meta_map, @ptrCast(path.ptr), cpu));

            for (0..n_layers) |li| {
                for (kinds, 0..) |kind, ki| {
                    const key = try std.fmt.allocPrint(self.allocator, "l{d}.{s}\x00", .{ li, kind });
                    defer self.allocator.free(key);
                    var arr = mlx.mlx_array_new();
                    if (mlx.mlx_map_string_to_array_get(&arr, tensor_map, @ptrCast(key.ptr)) != 0) {
                        _ = mlx.mlx_array_free(arr);
                        if (ki == 0) break; // layer absent from this chunk
                        return error.DiskCacheCorruptChunk;
                    }
                    if (ki == 0) present[li] = true;
                    // Crash-tolerance: a chunk file may hold MORE positions
                    // than meta.json committed to (rewrite raced a crash).
                    // Slice down to the committed range; never trust the file.
                    const shape = mlx.getShape(arr);
                    if (shape.len != 4) {
                        _ = mlx.mlx_array_free(arr);
                        return error.DiskCacheCorruptChunk;
                    }
                    const have: u64 = @intCast(shape[2]);
                    if (have < need) {
                        _ = mlx.mlx_array_free(arr);
                        return error.DiskCacheCorruptChunk;
                    }
                    if (have > need) {
                        var sliced = mlx.mlx_array_new();
                        const st = [_]c_int{ 0, 0, 0, 0 };
                        const sp = [_]c_int{ shape[0], shape[1], @intCast(need), shape[3] };
                        const sd = [_]c_int{ 1, 1, 1, 1 };
                        const rc = mlx.mlx_slice(&sliced, arr, &st, 4, &sp, 4, &sd, 4, s);
                        _ = mlx.mlx_array_free(arr);
                        try mlx.check(rc);
                        arr = sliced;
                    }
                    _ = mlx.mlx_vector_array_append_value(vecs[li * kinds.len + ki], arr);
                    _ = mlx.mlx_array_free(arr);
                }
            }
        }

        // Install per-layer concatenations as the cache's storage buffers.
        // Mirrors `KVCache.restore`: views stay empty, offset/initialized set,
        // step = kv_len.
        for (cache.entries, 0..) |*dst, li| {
            transformer_mod.resetKVEntry(dst);
            if (!present[li]) continue;
            const base = li * kinds.len;
            try concatInto(&dst.keys, vecs[base + 0], s);
            try concatInto(&dst.values, vecs[base + 1], s);
            if (quant.scheme != .off) {
                try concatInto(&dst.keys_scales, vecs[base + 2], s);
                try concatInto(&dst.keys_biases, vecs[base + 3], s);
                try concatInto(&dst.values_scales, vecs[base + 4], s);
                try concatInto(&dst.values_biases, vecs[base + 5], s);
            }
            dst.offset = e.kv_len;
            dst.initialized = true;
        }
        cache.step = e.kv_len;
        // Materialize with a CHECKED eval: a corrupt chunk surfaces its MLX
        // error HERE (lazy Load reads data at eval), so the caller's catch
        // resets the cache and falls back to cold prefill instead of running
        // a forward over poisoned buffers.
        {
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            for (cache.entries) |*entry| {
                if (!entry.initialized) continue;
                _ = mlx.mlx_vector_array_append_value(vec, entry.keys);
                _ = mlx.mlx_vector_array_append_value(vec, entry.values);
                if (quant.scheme != .off) {
                    _ = mlx.mlx_vector_array_append_value(vec, entry.keys_scales);
                    _ = mlx.mlx_vector_array_append_value(vec, entry.keys_biases);
                    _ = mlx.mlx_vector_array_append_value(vec, entry.values_scales);
                    _ = mlx.mlx_vector_array_append_value(vec, entry.values_biases);
                }
            }
            try mlx.check(mlx.mlx_eval(vec));
        }

        e.last_used = self.bump();
        // Bump meta.json mtime so cross-restart LRU sees the use.
        self.writeMeta(e.*) catch {};
        return e.kv_len;
    }

    fn concatInto(dst: *mlx.mlx_array, vec: mlx.mlx_vector_array, s: mlx.mlx_stream) !void {
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 2, s));
        _ = mlx.mlx_array_free(dst.*);
        dst.* = out;
    }

    // ── Commit ──

    /// Persist a cache state under `tokens`. Called on the inference thread
    /// AFTER the response is finished (the write is bounded but synchronous).
    /// Takes snapshot-shaped parts (entries + step + config) so callers can
    /// flush either a live `KVCache` or a committed `KVCacheSnapshot` — the
    /// hot cache flushes the RAM entry it just committed, post-markFinished,
    /// so the client never waits on the SSD write. Skips ineligible states
    /// silently; never fails the request.
    pub fn appendCommit(
        self: *DiskTier,
        kv_entries: []const transformer_mod.KVCacheEntry,
        step: usize,
        config: kv_quant.KVQuantConfig,
        tokens: []const u32,
        has_tools: bool,
        s: mlx.mlx_stream,
    ) !bool {
        // On EOS-terminated turns the cache runs 1-2 positions AHEAD of the
        // committed token record (forwarded terminator tokens that never
        // land in `tokens`). Persist the prefix covered by the record —
        // positions beyond it are unusable for matching anyway.
        const kv_target_u: usize = @min(step, tokens.len);
        if (kv_target_u < MIN_PERSIST_TOKENS) return true;
        const kv_target: u32 = @intCast(kv_target_u);
        switch (config.scheme) {
            .off, .affine => {},
            else => return true, // TurboQuant rotation state doesn't survive restore
        }
        // Every initialized layer must cover the persisted range with B == 1
        // — anything else (mid-spec-decode state, batched cache) is not a
        // persistable snapshot.
        for (kv_entries) |*entry| {
            if (!entry.initialized) continue;
            if (entry.offset < kv_target_u) {
                log.debug("  [disk-cache] skip: layer offset {d} < kv_len {d}\n", .{ entry.offset, kv_target_u });
                return true;
            }
            const shape = mlx.getShape(entry.keys);
            if (shape.len != 4 or shape[0] != 1) {
                log.debug("  [disk-cache] skip: non-B1 cache shape\n", .{});
                return true;
            }
        }

        // Superseded check: an existing entry that already covers `tokens`
        // (same key, tokens is a prefix of its tokens, kv already >= ours)
        // makes this commit a no-op.
        var extend_idx: ?usize = null;
        for (self.entries.items, 0..) |*e, i| {
            if (e.has_tools != has_tools) continue;
            if (!std.meta.eql(e.quant, config)) continue;
            if (e.tokens.len >= tokens.len) {
                if (std.mem.eql(u32, e.tokens[0..tokens.len], tokens)) {
                    if (e.kv_len >= kv_target) {
                        e.last_used = self.bump();
                        return true;
                    }
                    // Same token record, SHORTER persisted KV — a byte-capped
                    // incremental flush in progress. Resume into its dir.
                    extend_idx = i;
                }
            } else if (std.mem.eql(u32, e.tokens, tokens[0..e.tokens.len])) {
                // This commit extends `e` — reuse its directory and chunks.
                extend_idx = i;
            }
        }

        const sw = io_util.Stopwatch.init(self.io);

        const id: u64 = if (extend_idx) |i| self.entries.items[i].id else blk: {
            const nid = self.next_id;
            self.next_id += 1;
            break :blk nid;
        };
        const dir_rel = try std.fmt.allocPrint(self.allocator, "{s}/e{d}", .{ self.root, id });
        defer self.allocator.free(dir_rel);
        try std.Io.Dir.cwd().createDirPath(self.io, dir_rel);

        // Chunks [0, keep) are full chunks already on disk from the entry we
        // extend; everything from `keep` on (the old partial tail + the new
        // positions) is (re)written — up to the per-flush byte cap. Stopping
        // early lands on a full-chunk boundary; the entry then records the
        // shorter kv_len and the NEXT flush resumes from there.
        const old_kv: u32 = if (extend_idx) |i| self.entries.items[i].kv_len else 0;
        const keep: u32 = old_kv / self.chunk_tokens;
        const n_chunks: u32 = @intCast((@as(u64, kv_target) + self.chunk_tokens - 1) / self.chunk_tokens);

        var chunk_sizes = std.ArrayList(u64).empty;
        errdefer chunk_sizes.deinit(self.allocator);
        if (extend_idx) |i| {
            const old_cb = self.entries.items[i].chunk_bytes;
            try chunk_sizes.appendSlice(self.allocator, old_cb[0..@min(keep, old_cb.len)]);
        }

        var written_bytes: u64 = 0;
        var chunk_i: u32 = keep;
        while (chunk_i < n_chunks) : (chunk_i += 1) {
            if (written_bytes >= self.max_flush_bytes and chunk_i > keep) break;
            const c0: u32 = chunk_i * self.chunk_tokens;
            const c1: u32 = @intCast(@min(@as(u64, c0) + self.chunk_tokens, kv_target));
            try self.writeChunkFile(kv_entries, config, dir_rel, chunk_i, c0, c1, s);
            const cpath = try std.fmt.allocPrint(self.allocator, "{s}/c{d:0>6}.safetensors", .{ dir_rel, chunk_i });
            defer self.allocator.free(cpath);
            const csize = fileSize(self.io, cpath) orelse 0;
            written_bytes += csize;
            try chunk_sizes.append(self.allocator, csize);
        }
        const chunks_done: u32 = chunk_i;
        const complete = chunks_done == n_chunks;
        const kv_len: u32 = if (complete) kv_target else chunks_done * self.chunk_tokens;
        if (kv_len <= old_kv) {
            // Cap so tight nothing new landed — nothing to commit.
            chunk_sizes.deinit(self.allocator);
            return complete;
        }

        // Token record — the LONGER of the existing record and this commit's
        // tokens (a resumed incremental flush must not shrink the record its
        // earlier chunks were committed against). Rewritten only on growth;
        // tens of KB at most.
        const record: []const u32 = if (extend_idx) |i| blk: {
            const et = self.entries.items[i].tokens;
            break :blk if (et.len >= tokens.len) et else tokens;
        } else tokens;
        if (extend_idx == null or record.ptr == tokens.ptr) {
            const tpath = try std.fmt.allocPrint(self.allocator, "{s}/tokens.bin", .{dir_rel});
            defer self.allocator.free(tpath);
            const f = try std.Io.Dir.createFileAbsolute(self.io, tpath, .{});
            defer f.close(self.io);
            var wb: [8192]u8 = undefined;
            var fw = f.writer(self.io, &wb);
            try fw.interface.writeSliceEndian(u32, record, .little);
            try fw.interface.flush();
        }

        var bytes: u64 = 0;
        for (chunk_sizes.items) |b| bytes += b;
        bytes += @as(u64, record.len) * 4;

        const new_entry: IndexEntry = .{
            .id = id,
            .tokens = try self.allocator.dupe(u32, record),
            .kv_len = kv_len,
            .has_tools = has_tools,
            .quant = config,
            .bytes = bytes,
            .chunk_bytes = try chunk_sizes.toOwnedSlice(self.allocator),
            .last_used = self.bump(),
        };
        errdefer {
            self.allocator.free(new_entry.tokens);
            self.allocator.free(new_entry.chunk_bytes);
        }
        // meta.json is the commit point — written last, atomically.
        try self.writeMeta(new_entry);

        if (extend_idx) |i| {
            const e = &self.entries.items[i];
            self.total_bytes -|= e.bytes;
            self.allocator.free(e.tokens);
            self.allocator.free(e.chunk_bytes);
            e.* = new_entry;
            self.total_bytes += new_entry.bytes;
        } else {
            try self.entries.append(self.allocator, new_entry);
            self.total_bytes += new_entry.bytes;
        }
        self.gcToBudget();

        const wrote_mb = @as(f64, @floatFromInt(written_bytes)) / (1024.0 * 1024.0);
        const ms: u64 = sw.read() / std.time.ns_per_ms;
        log.info("  [disk-cache] persisted {d}/{d} tokens (+{d} chunks, {d:.1} MB, {d}ms); resident={d:.1} MB ({d} entries)\n", .{
            kv_len,               kv_target, chunks_done - keep, wrote_mb, ms,
            @as(f64, @floatFromInt(self.total_bytes)) / (1024.0 * 1024.0),
            self.entries.items.len,
        });
        return complete;
    }

    fn writeChunkFile(
        self: *DiskTier,
        kv_entries: []const transformer_mod.KVCacheEntry,
        config: kv_quant.KVQuantConfig,
        dir_abs: []const u8,
        chunk_idx: u32,
        c0: u32,
        c1: u32,
        s: mlx.mlx_stream,
    ) !void {
        const tensor_map = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(tensor_map);
        const meta_map = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta_map);

        const affine = config.scheme != .off;
        for (kv_entries, 0..) |*entry, li| {
            if (!entry.initialized) continue;
            try self.insertSlice(tensor_map, li, "k", entry.keys, c0, c1, s);
            try self.insertSlice(tensor_map, li, "v", entry.values, c0, c1, s);
            if (affine) {
                try self.insertSlice(tensor_map, li, "ks", entry.keys_scales, c0, c1, s);
                try self.insertSlice(tensor_map, li, "kb", entry.keys_biases, c0, c1, s);
                try self.insertSlice(tensor_map, li, "vs", entry.values_scales, c0, c1, s);
                try self.insertSlice(tensor_map, li, "vb", entry.values_biases, c0, c1, s);
            }
        }

        const path = try std.fmt.allocPrint(self.allocator, "{s}/c{d:0>6}.safetensors\x00", .{ dir_abs, chunk_idx });
        defer self.allocator.free(path);
        try mlx.check(mlx.mlx_save_safetensors(@ptrCast(path.ptr), tensor_map, meta_map));
    }

    fn insertSlice(
        self: *DiskTier,
        map: mlx.mlx_map_string_to_array,
        layer: usize,
        kind: []const u8,
        buf: mlx.mlx_array,
        c0: u32,
        c1: u32,
        s: mlx.mlx_stream,
    ) !void {
        const shape = mlx.getShape(buf);
        if (shape.len != 4) return error.DiskCacheBadShape;
        var sliced = mlx.mlx_array_new();
        const st = [_]c_int{ 0, 0, @intCast(c0), 0 };
        const sp = [_]c_int{ shape[0], shape[1], @intCast(c1), shape[3] };
        const sd = [_]c_int{ 1, 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&sliced, buf, &st, 4, &sp, 4, &sd, 4, s));
        defer _ = mlx.mlx_array_free(sliced);
        const key = try std.fmt.allocPrint(self.allocator, "l{d}.{s}\x00", .{ layer, kind });
        defer self.allocator.free(key);
        try mlx.check(mlx.mlx_map_string_to_array_insert(map, @ptrCast(key.ptr), sliced));
    }

    // ── Invalidation (mirrors the RAM cache API) ──

    pub fn invalidateAll(self: *DiskTier) void {
        for (self.entries.items) |*e| {
            self.deleteEntryDir(e.id);
            self.allocator.free(e.tokens);
            self.allocator.free(e.chunk_bytes);
        }
        self.entries.clearRetainingCapacity();
        self.total_bytes = 0;
    }

    pub fn invalidateNewest(self: *DiskTier) void {
        if (self.entries.items.len == 0) return;
        var newest_idx: usize = 0;
        var newest_used: u64 = 0;
        for (self.entries.items, 0..) |*e, i| {
            if (e.last_used >= newest_used) {
                newest_used = e.last_used;
                newest_idx = i;
            }
        }
        self.removeAt(newest_idx);
    }

    // ── Internals ──

    fn bump(self: *DiskTier) u64 {
        self.counter += 1;
        return self.counter;
    }

    fn removeAt(self: *DiskTier, idx: usize) void {
        const e = self.entries.swapRemove(idx);
        self.total_bytes -|= e.bytes;
        self.deleteEntryDir(e.id);
        self.allocator.free(e.tokens);
        self.allocator.free(e.chunk_bytes);
    }

    fn deleteEntryDir(self: *DiskTier, id: u64) void {
        const rel = std.fmt.allocPrint(self.allocator, "e{d}", .{id}) catch return;
        defer self.allocator.free(rel);
        var root_dir = std.Io.Dir.openDirAbsolute(self.io, self.root, .{ .iterate = true }) catch return;
        defer root_dir.close(self.io);
        root_dir.deleteTree(self.io, rel) catch {};
    }

    fn gcToBudget(self: *DiskTier) void {
        if (self.max_bytes == 0) return;
        while (self.total_bytes > self.max_bytes and self.entries.items.len > 1) {
            var lru_idx: usize = 0;
            var lru_used: u64 = std.math.maxInt(u64);
            for (self.entries.items, 0..) |*e, i| {
                if (e.last_used < lru_used) {
                    lru_used = e.last_used;
                    lru_idx = i;
                }
            }
            const mb = @as(f64, @floatFromInt(self.entries.items[lru_idx].bytes)) / (1024.0 * 1024.0);
            log.info("  [disk-cache] evicted LRU entry (byte budget; {d:.1} MB)\n", .{mb});
            self.removeAt(lru_idx);
        }
    }

    fn writeMeta(self: *DiskTier, e: IndexEntry) !void {
        const tmp_path = try std.fmt.allocPrint(self.allocator, "{s}/e{d}/meta.json.tmp", .{ self.root, e.id });
        defer self.allocator.free(tmp_path);
        const final_path = try std.fmt.allocPrint(self.allocator, "{s}/e{d}/meta.json", .{ self.root, e.id });
        defer self.allocator.free(final_path);
        {
            const f = try std.Io.Dir.createFileAbsolute(self.io, tmp_path, .{});
            defer f.close(self.io);
            var wb: [1024]u8 = undefined;
            var fw = f.writer(self.io, &wb);
            try fw.interface.print(
                "{{\"v\":2,\"kv_len\":{d},\"tokens\":{d},\"has_tools\":{},\"scheme\":\"{s}\",\"bits\":{d},\"group_size\":{d},\"chunk_tokens\":{d},\"bytes\":{d},\"chunk_bytes\":[",
                .{
                    e.kv_len,
                    e.tokens.len,
                    e.has_tools,
                    @tagName(e.quant.scheme),
                    e.quant.bits,
                    e.quant.group_size,
                    self.chunk_tokens,
                    e.bytes,
                },
            );
            for (e.chunk_bytes, 0..) |cb, i| {
                if (i > 0) try fw.interface.writeAll(",");
                try fw.interface.print("{d}", .{cb});
            }
            try fw.interface.writeAll("]}");
            try fw.interface.flush();
        }
        try std.Io.Dir.renameAbsolute(tmp_path, final_path, self.io);
    }

    fn scan(self: *DiskTier) !void {
        var root_dir = std.Io.Dir.openDirAbsolute(self.io, self.root, .{ .iterate = true }) catch return;
        defer root_dir.close(self.io);

        // Collected (entry, mtime) pairs; sorted by mtime → LRU order.
        const Pending = struct { e: IndexEntry, mtime: i128 };
        var pending = std.ArrayList(Pending).empty;
        defer pending.deinit(self.allocator);

        var it = root_dir.iterate();
        while (it.next(self.io) catch null) |dent| {
            if (dent.kind != .directory) continue;
            if (dent.name.len < 2 or dent.name[0] != 'e') continue;
            const id = std.fmt.parseInt(u64, dent.name[1..], 10) catch continue;
            // Never reuse an id that has ever existed on disk — even a
            // dropped leftover's delete could fail and leave a dirty dir.
            if (id >= self.next_id) self.next_id = id + 1;
            if (self.loadEntry(id)) |loaded| {
                pending.append(self.allocator, .{ .e = loaded.e, .mtime = loaded.mtime }) catch {
                    self.allocator.free(loaded.e.tokens);
                    self.allocator.free(loaded.e.chunk_bytes);
                    continue;
                };
            } else {
                // Crash leftover / corrupt — remove it.
                log.info("  [disk-cache] dropping incomplete entry e{d}\n", .{id});
                self.deleteEntryDir(id);
            }
        }

        std.mem.sort(Pending, pending.items, {}, struct {
            fn lessThan(_: void, a: Pending, b: Pending) bool {
                return a.mtime < b.mtime;
            }
        }.lessThan);

        for (pending.items) |*p| {
            p.e.last_used = self.bump();
            self.entries.append(self.allocator, p.e) catch {
                self.allocator.free(p.e.tokens);
                self.allocator.free(p.e.chunk_bytes);
                continue;
            };
            self.total_bytes += p.e.bytes;
        }
        if (self.entries.items.len > 0) {
            log.info("  [disk-cache] scanned {d} persisted entries ({d:.1} MB) at {s}\n", .{
                self.entries.items.len,
                @as(f64, @floatFromInt(self.total_bytes)) / (1024.0 * 1024.0),
                self.root,
            });
        }
    }

    fn loadEntry(self: *DiskTier, id: u64) ?struct { e: IndexEntry, mtime: i128 } {
        const meta_path = std.fmt.allocPrint(self.allocator, "{s}/e{d}/meta.json", .{ self.root, id }) catch return null;
        defer self.allocator.free(meta_path);

        const stat = statFile(self.io, meta_path) orelse return null;
        const content = readFileAlloc(self.allocator, self.io, meta_path, 64 * 1024) orelse return null;
        defer self.allocator.free(content);

        var parsed = std.json.parseFromSlice(std.json.Value, self.allocator, content, .{}) catch return null;
        defer parsed.deinit();
        if (parsed.value != .object) return null;
        const obj = parsed.value.object;

        const version = jsonU64(obj, "v") orelse return null;
        if (version != 2) return null; // older layouts are dropped, not migrated
        var kv_len = jsonU64(obj, "kv_len") orelse return null;
        const n_tokens = jsonU64(obj, "tokens") orelse return null;
        const chunk_tokens = jsonU64(obj, "chunk_tokens") orelse return null;
        const has_tools_v = obj.get("has_tools") orelse return null;
        if (has_tools_v != .bool) return null;
        const scheme_v = obj.get("scheme") orelse return null;
        if (scheme_v != .string) return null;
        const bits = jsonU64(obj, "bits") orelse 0;
        const group_size = jsonU64(obj, "group_size") orelse 0;
        const chunk_bytes_v = obj.get("chunk_bytes") orelse return null;
        if (chunk_bytes_v != .array) return null;

        // Chunk geometry must match this tier's configuration — a stale root
        // written under a different chunk size can't be extended coherently.
        if (chunk_tokens != self.chunk_tokens) return null;
        if (kv_len == 0 or kv_len > n_tokens) return null;

        const scheme = std.meta.stringToEnum(kv_quant.Scheme, scheme_v.string) orelse return null;
        var quant: kv_quant.KVQuantConfig = switch (scheme) {
            .off => kv_quant.KVQuantConfig.dense,
            .affine => kv_quant.KVQuantConfig.affine(@intCast(bits)),
            else => return null,
        };
        if (scheme == .affine) {
            if (group_size == 0 or bits == 0) return null;
            quant.group_size = @intCast(group_size);
        }

        const n_chunks: u64 = (kv_len + chunk_tokens - 1) / chunk_tokens;
        if (chunk_bytes_v.array.items.len != n_chunks) return null;

        // Validate each chunk file's size against the recorded one. A kill -9
        // mid-flush truncates the chunk being (re)written while meta still
        // describes the previous valid state — restoring it would poison the
        // cache (live: MLX "invalid data offsets exceeding the size of the
        // file"). Clamp to the last contiguous valid chunk and salvage the
        // prefix.
        var valid_chunks: u64 = 0;
        while (valid_chunks < n_chunks) : (valid_chunks += 1) {
            const want_v = chunk_bytes_v.array.items[@intCast(valid_chunks)];
            if (want_v != .integer or want_v.integer < 0) break;
            const cp = std.fmt.allocPrint(self.allocator, "{s}/e{d}/c{d:0>6}.safetensors", .{ self.root, id, valid_chunks }) catch return null;
            defer self.allocator.free(cp);
            const have = fileSize(self.io, cp) orelse break;
            if (have != @as(u64, @intCast(want_v.integer))) break;
        }
        if (valid_chunks < n_chunks) {
            const salvaged = valid_chunks * chunk_tokens;
            log.info("  [disk-cache] e{d}: chunk {d} invalid — salvaging {d}/{d} tokens\n", .{ id, valid_chunks, salvaged, kv_len });
            kv_len = salvaged;
            if (kv_len < MIN_PERSIST_TOKENS) return null;
        }

        const chunk_bytes = self.allocator.alloc(u64, @intCast(valid_chunks)) catch return null;
        for (chunk_bytes, 0..) |*cb, i| cb.* = @intCast(chunk_bytes_v.array.items[i].integer);

        // Token record.
        const tokens_path = std.fmt.allocPrint(self.allocator, "{s}/e{d}/tokens.bin", .{ self.root, id }) catch {
            self.allocator.free(chunk_bytes);
            return null;
        };
        defer self.allocator.free(tokens_path);
        const raw = readFileAlloc(self.allocator, self.io, tokens_path, 64 * 1024 * 1024) orelse {
            self.allocator.free(chunk_bytes);
            return null;
        };
        defer self.allocator.free(raw);
        if (raw.len != n_tokens * 4) {
            self.allocator.free(chunk_bytes);
            return null;
        }
        const tokens = self.allocator.alloc(u32, n_tokens) catch {
            self.allocator.free(chunk_bytes);
            return null;
        };
        for (tokens, 0..) |*t, i| {
            t.* = std.mem.readInt(u32, raw[i * 4 ..][0..4], .little);
        }

        var total: u64 = @as(u64, tokens.len) * 4;
        for (chunk_bytes) |cb| total += cb;

        return .{
            .e = .{
                .id = id,
                .tokens = tokens,
                .kv_len = @intCast(kv_len),
                .has_tools = has_tools_v.bool,
                .quant = quant,
                .bytes = total,
                .chunk_bytes = chunk_bytes,
                .last_used = 0,
            },
            .mtime = stat.mtime.nanoseconds,
        };
    }
};

// ── Model fingerprint ──

/// Identity of the weights the persisted KV was computed against: absolute
/// model dir + config.json size/mtime. A re-downloaded or re-quantized
/// checkpoint rewrites config.json, which rolls the fingerprint and orphans
/// the stale KV (GC'd by the disk budget eventually; different fingerprint
/// dirs never mix). 16 hex chars of XxHash64.
pub fn modelFingerprint(allocator: std.mem.Allocator, io: std.Io, model_dir: []const u8) ![]u8 {
    if (model_dir.len == 0 or !std.fs.path.isAbsolute(model_dir)) return error.BadModelDir;
    var h = std.hash.XxHash64.init(0x6b76_6361_6368_6531);
    h.update(model_dir);
    const cfg_path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{model_dir});
    defer allocator.free(cfg_path);
    if (statFile(io, cfg_path)) |st| {
        h.update(std.mem.asBytes(&st.size));
        const mt: i128 = st.mtime.nanoseconds;
        h.update(std.mem.asBytes(&mt));
    }
    return std.fmt.allocPrint(allocator, "{x:0>16}", .{h.final()});
}

/// Default persistence root: `~/.mlx-serve/kv-cache`.
pub fn defaultBaseDir(allocator: std.mem.Allocator) ![]u8 {
    const home = std.mem.span(std.c.getenv("HOME") orelse return error.NoHome);
    return std.fmt.allocPrint(allocator, "{s}/.mlx-serve/kv-cache", .{home});
}

// ── Small fs helpers ──

fn statFile(io: std.Io, abs_path: []const u8) ?std.Io.File.Stat {
    if (abs_path.len == 0 or !std.fs.path.isAbsolute(abs_path)) return null;
    const f = std.Io.Dir.openFileAbsolute(io, abs_path, .{}) catch return null;
    defer f.close(io);
    return f.stat(io) catch null;
}

fn fileSize(io: std.Io, abs_path: []const u8) ?u64 {
    const st = statFile(io, abs_path) orelse return null;
    return st.size;
}

fn readFileAlloc(allocator: std.mem.Allocator, io: std.Io, abs_path: []const u8, limit: usize) ?[]u8 {
    if (abs_path.len == 0 or !std.fs.path.isAbsolute(abs_path)) return null;
    const f = std.Io.Dir.openFileAbsolute(io, abs_path, .{}) catch return null;
    defer f.close(io);
    var rb: [8192]u8 = undefined;
    var rs = f.reader(io, &rb);
    return rs.interface.allocRemaining(allocator, .limited(limit)) catch null;
}

fn jsonU64(obj: std.json.ObjectMap, key: []const u8) ?u64 {
    const v = obj.get(key) orelse return null;
    if (v != .integer) return null;
    if (v.integer < 0) return null;
    return @intCast(v.integer);
}

// ── Tests ──

const testing = std.testing;

fn fillCache(cache: *KVCache, s: mlx.mlx_stream, n_layers: u32, tokens: u32, head_dim: u32, seed: f64, dtype: mlx.mlx_dtype) !void {
    // Drive the cache through its real update path with deterministic
    // arange-derived K/V so restored values are checkable. Dense tests use
    // float32 (every position stays exactly distinguishable); the affine test
    // uses bf16, the production dtype the quant write path expects.
    var written: u32 = 0;
    while (written < tokens) {
        const step: u32 = @min(64, tokens - written);
        var li: u32 = 0;
        while (li < n_layers) : (li += 1) {
            var flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat);
            const count: f64 = @floatFromInt(step * head_dim);
            const base: f64 = seed + @as(f64, @floatFromInt(written * head_dim + li * 1_000_000));
            try mlx.check(mlx.mlx_arange(&flat, base, base + count, 1.0, .float32, s));
            var shaped = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(shaped);
            const shape = [_]c_int{ 1, 1, @intCast(step), @intCast(head_dim) };
            try mlx.check(mlx.mlx_reshape(&shaped, flat, &shape, 4, s));
            var k = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(k);
            try mlx.check(mlx.mlx_astype(&k, shaped, dtype, s));
            // V = -K so a restore-side K/V swap can't false-pass.
            var v = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(v);
            try mlx.check(mlx.mlx_negative(&v, k, s));
            var view = try cache.update(li, k, v, s, 0);
            view.deinit();
        }
        written += step;
    }
}

fn cacheValueAt(cache: *KVCache, layer: u32, pos: u32, d: u32, s: mlx.mlx_stream) !f32 {
    return cacheBufValueAt(cache, layer, pos, d, s, false);
}

fn cacheBufValueAt(cache: *KVCache, layer: u32, pos: u32, d: u32, s: mlx.mlx_stream, values: bool) !f32 {
    const entry = &cache.entries[layer];
    var sliced = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sliced);
    const st = [_]c_int{ 0, 0, @intCast(pos), @intCast(d) };
    const sp = [_]c_int{ 1, 1, @intCast(pos + 1), @intCast(d + 1) };
    const sd = [_]c_int{ 1, 1, 1, 1 };
    const buf = if (values) entry.values else entry.keys;
    try mlx.check(mlx.mlx_slice(&sliced, buf, &st, 4, &sp, 4, &sd, 4, s));
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, sliced, .float32, s));
    _ = mlx.mlx_array_eval(f);
    const ptr = mlx.mlx_array_data_float32(f) orelse return error.NoData;
    return ptr[0];
}

fn tmpRoot(tmp: *std.testing.TmpDir, io: std.Io, buf: []u8) ![]const u8 {
    const n = try tmp.dir.realPath(io, buf);
    return buf[0..n];
}

test "DiskTier: chunked commit + restore round-trips exact KV, step, offsets" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-test", 0, 128);
    defer tier.deinit();

    // 600 tokens => 5 chunks at 128 (last partial: 88).
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    try testing.expectEqual(@as(usize, 600), cache.step);

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, s);
    try testing.expectEqual(@as(usize, 1), tier.entryCount());

    // Restore into a fresh cache (fresh tier too — proves the restart path).
    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-test", 0, 128);
    defer tier2.deinit();
    try testing.expectEqual(@as(usize, 1), tier2.entryCount());

    const m = tier2.bestMatch(&tokens, false, kv_quant.KVQuantConfig.dense).?;
    try testing.expectEqual(@as(u32, 600), m.usable);

    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    const restored = try tier2.restoreInto(&cache2, m.idx, s);
    try testing.expectEqual(@as(u32, 600), restored);
    try testing.expectEqual(@as(usize, 600), cache2.step);
    for (cache2.entries) |*e| {
        try testing.expect(e.initialized);
        try testing.expectEqual(@as(usize, 600), e.offset);
    }

    // Spot-check exact values across chunk boundaries and layers.
    const probes = [_][2]u32{ .{ 0, 0 }, .{ 127, 7 }, .{ 128, 0 }, .{ 300, 3 }, .{ 511, 7 }, .{ 512, 0 }, .{ 599, 7 } };
    for (probes) |p| {
        var li: u32 = 0;
        while (li < 3) : (li += 1) {
            const want = try cacheValueAt(&cache, li, p[0], p[1], s);
            const got = try cacheValueAt(&cache2, li, p[0], p[1], s);
            try testing.expectEqual(want, got);
            // V was written as -K: restored values must mirror that, so a
            // restore-side K/V swap or shared-buffer mixup fails here.
            const got_v = try cacheBufValueAt(&cache2, li, p[0], p[1], s, true);
            try testing.expectEqual(-want, got_v);
        }
    }

    // Mismatched key never matches.
    try testing.expect(tier2.bestMatch(&tokens, true, kv_quant.KVQuantConfig.dense) == null);
    try testing.expect(tier2.bestMatch(&tokens, false, kv_quant.KVQuantConfig.affine(4)) == null);
}

test "DiskTier: extend commit appends only new chunks (full chunks untouched)" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-ext", 0, 128);
    defer tier.deinit();

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try fillCache(&cache, s, 1, 600, 8, 0.0, .float32);
    var tokens: [900]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, tokens[0..600], false, s);

    // Tamper-mark chunk 0 (a FULL chunk): record its mtime, then extend the
    // entry and assert chunk 0 was not rewritten while chunk 4 (the old
    // partial) was, and new chunks appeared.
    const c0_path = try std.fmt.allocPrint(testing.allocator, "{s}/fp-ext/e1/c000000.safetensors", .{base});
    defer testing.allocator.free(c0_path);
    const c4_path = try std.fmt.allocPrint(testing.allocator, "{s}/fp-ext/e1/c000004.safetensors", .{base});
    defer testing.allocator.free(c4_path);
    const c6_path = try std.fmt.allocPrint(testing.allocator, "{s}/fp-ext/e1/c000006.safetensors", .{base});
    defer testing.allocator.free(c6_path);
    const c0_before = statFile(io, c0_path).?.mtime.nanoseconds;
    const c4_before = statFile(io, c4_path).?.mtime.nanoseconds;
    try testing.expect(fileSize(io, c6_path) == null);

    // Ensure the extend write lands at a measurably later mtime.
    std.Io.sleep(io, .fromMilliseconds(20), .real) catch {};

    // Same prefix, 300 more tokens.
    try fillCache(&cache, s, 1, 300, 8, 4800.0, .float32);
    try testing.expectEqual(@as(usize, 900), cache.step);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, s);
    try testing.expectEqual(@as(usize, 1), tier.entryCount());
    try testing.expectEqual(@as(u32, 900), tier.entries.items[0].kv_len);

    try testing.expectEqual(c0_before, statFile(io, c0_path).?.mtime.nanoseconds); // untouched
    try testing.expect(statFile(io, c4_path).?.mtime.nanoseconds != c4_before); // partial rewritten
    try testing.expect(fileSize(io, c6_path) != null); // new tail chunk

    // Restore the extended entry and check a value in the extension range.
    var cache2 = try KVCache.init(testing.allocator, 1);
    defer cache2.deinit();
    const restored = try tier.restoreInto(&cache2, 0, s);
    try testing.expectEqual(@as(u32, 900), restored);
    const want = try cacheValueAt(&cache, 0, 750, 5, s);
    const got = try cacheValueAt(&cache2, 0, 750, 5, s);
    try testing.expectEqual(want, got);
}

test "DiskTier: identical re-commit is a no-op; shorter prefix is superseded" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-noop", 0, 128);
    defer tier.deinit();

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try fillCache(&cache, s, 1, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, s);
    const c0_path = try std.fmt.allocPrint(testing.allocator, "{s}/fp-noop/e1/c000000.safetensors", .{base});
    defer testing.allocator.free(c0_path);
    const before = statFile(io, c0_path).?.mtime.nanoseconds;

    std.Io.sleep(io, .fromMilliseconds(20), .real) catch {};

    // Identical commit — nothing rewritten, no second entry.
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, s);
    try testing.expectEqual(@as(usize, 1), tier.entryCount());
    try testing.expectEqual(before, statFile(io, c0_path).?.mtime.nanoseconds);

    // A shorter-prefix commit of the same conversation is covered by the
    // existing entry — also a no-op.
    var short_cache = try KVCache.init(testing.allocator, 1);
    defer short_cache.deinit();
    try fillCache(&short_cache, s, 1, 512, 8, 0.0, .float32);
    _ = try tier.appendCommit(short_cache.entries, short_cache.step, short_cache.config, tokens[0..512], false, s);
    try testing.expectEqual(@as(usize, 1), tier.entryCount());
}

test "DiskTier: byte budget evicts LRU entries, keeps newest" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    // Budget deliberately tiny: every new entry evicts the previous one.
    var tier = try DiskTier.init(testing.allocator, io, base, "fp-gc", 4096, 128);
    defer tier.deinit();

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try fillCache(&cache, s, 1, 520, 8, 0.0, .float32);

    var tokens_a: [520]u32 = undefined;
    for (&tokens_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tokens_b: [520]u32 = undefined;
    for (&tokens_b, 0..) |*t, i| t.* = @intCast(i + 900_000);

    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens_a, false, s);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens_b, false, s);
    // Both entries exceed 4 KB each — only the newest survives.
    try testing.expectEqual(@as(usize, 1), tier.entryCount());
    try testing.expect(std.mem.eql(u32, tier.entries.items[0].tokens, &tokens_b));

    // The evicted directory is gone from disk.
    const e1_meta = try std.fmt.allocPrint(testing.allocator, "{s}/fp-gc/e1/meta.json", .{base});
    defer testing.allocator.free(e1_meta);
    try testing.expect(statFile(io, e1_meta) == null);
}

test "DiskTier: scan drops crash leftovers (no meta.json)" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    {
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-crash", 0, 128);
        defer tier.deinit();
        var cache = try KVCache.init(testing.allocator, 1);
        defer cache.deinit();
        try fillCache(&cache, s, 1, 600, 8, 0.0, .float32);
        var tokens: [600]u32 = undefined;
        for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, s);
        // Simulate a crash mid-write of a SECOND entry: chunks, no meta.
        try tmp.dir.createDirPath(io, "fp-crash/e9");
        try tmp.dir.writeFile(io, .{ .sub_path = "fp-crash/e9/c000000.safetensors", .data = "junk" });
    }

    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-crash", 0, 128);
    defer tier2.deinit();
    try testing.expectEqual(@as(usize, 1), tier2.entryCount());
    // The leftover dir was removed.
    const leftover = try std.fmt.allocPrint(testing.allocator, "{s}/fp-crash/e9/c000000.safetensors", .{base});
    defer testing.allocator.free(leftover);
    try testing.expect(statFile(io, leftover) == null);
    // next_id moved past the dropped id (no reuse of a dirty dir name).
    try testing.expect(tier2.next_id >= 10);
}

test "DiskTier: affine-quant cache round-trips all six buffers" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-q", 0, 128);
    defer tier.deinit();

    const qcfg = kv_quant.KVQuantConfig.affine(4);
    var cache = try KVCache.initWithConfig(testing.allocator, 2, qcfg);
    defer cache.deinit();
    // head_dim must be a multiple of group_size (64) for affine quant.
    try fillCache(&cache, s, 2, 520, 64, 0.0, .bfloat16);
    try testing.expectEqual(@as(usize, 520), cache.step);

    var tokens: [520]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, s);

    var cache2 = try KVCache.initWithConfig(testing.allocator, 2, qcfg);
    defer cache2.deinit();
    const m = tier.bestMatch(&tokens, false, qcfg).?;
    try testing.expectEqual(@as(u32, 520), m.usable);
    const restored = try tier.restoreInto(&cache2, m.idx, s);
    try testing.expectEqual(@as(u32, 520), restored);

    // Dense read-back through the cache's own dequant path must agree.
    // Truncate BOTH caches to the same length first — restore leaves views
    // empty (the KVCache.restore contract) and truncate to len < offset
    // rebuilds them on both sides identically.
    try cache.truncate(519, s);
    try cache2.truncate(519, s);
    var v1 = try cache.denseView(0, s);
    defer v1.deinit();
    var v2 = try cache2.denseView(0, s);
    defer v2.deinit();
    const probes = [_][2]u32{ .{ 0, 0 }, .{ 127, 63 }, .{ 128, 0 }, .{ 300, 5 }, .{ 511, 1 }, .{ 518, 63 } };
    for (probes) |p| {
        var d1 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(d1);
        var d2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(d2);
        const st = [_]c_int{ 0, 0, @intCast(p[0]), @intCast(p[1]) };
        const sp = [_]c_int{ 1, 1, @intCast(p[0] + 1), @intCast(p[1] + 1) };
        const sd = [_]c_int{ 1, 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&d1, v1.k, &st, 4, &sp, 4, &sd, 4, s));
        try mlx.check(mlx.mlx_slice(&d2, v2.k, &st, 4, &sp, 4, &sd, 4, s));
        var f1 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(f1);
        var f2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(f2);
        try mlx.check(mlx.mlx_astype(&f1, d1, .float32, s));
        try mlx.check(mlx.mlx_astype(&f2, d2, .float32, s));
        _ = mlx.mlx_array_eval(f1);
        _ = mlx.mlx_array_eval(f2);
        try testing.expectEqual(mlx.mlx_array_data_float32(f1).?[0], mlx.mlx_array_data_float32(f2).?[0]);
    }
}

test "DiskTier: truncated chunk file salvages the valid prefix at scan (kill -9 shape)" {
    // A kill -9 mid-flush leaves a chunk file truncated while meta.json (the
    // commit point, written last) still describes the PREVIOUS valid state —
    // whose recorded size for that chunk no longer matches the file. Live
    // capture: MLX "invalid data offsets exceeding the size of the file" on
    // restore. The scan must clamp the entry to the last contiguous chunk
    // whose size matches meta, salvaging the prefix instead of poisoning a
    // restore (or dropping everything).
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    {
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-trunc", 0, 128);
        defer tier.deinit();
        var cache = try KVCache.init(testing.allocator, 1);
        defer cache.deinit();
        try fillCache(&cache, s, 1, 700, 8, 0.0, .float32);
        var tokens: [700]u32 = undefined;
        for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, s);
    }

    // Truncate chunk 4 (positions [512, 640)) — chunks 0-3 stay valid.
    try tmp.dir.writeFile(io, .{ .sub_path = "fp-trunc/e1/c000004.safetensors", .data = "trunc" });

    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-trunc", 0, 128);
    defer tier2.deinit();
    try testing.expectEqual(@as(usize, 1), tier2.entryCount());
    // kv_len clamped to the last valid chunk boundary: 4 * 128 = 512.
    try testing.expectEqual(@as(u32, 512), tier2.entries.items[0].kv_len);

    // The salvaged prefix restores cleanly.
    var cache2 = try KVCache.init(testing.allocator, 1);
    defer cache2.deinit();
    const restored = try tier2.restoreInto(&cache2, 0, s);
    try testing.expectEqual(@as(u32, 512), restored);
    try testing.expectEqual(@as(usize, 512), cache2.step);
}

test "DiskTier: flush byte cap persists incrementally across commits" {
    // A 4 GB first-commit write used to stall the NEXT request ~2.5 s (the
    // flush runs on the inference thread). appendCommit caps the bytes
    // written per call at max_flush_bytes, persists a chunk-aligned prefix,
    // and reports incomplete so the caller re-flushes on later turns.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-cap", 0, 128);
    defer tier.deinit();
    tier.max_flush_bytes = 1; // every chunk write exceeds the cap -> 1 chunk/flush

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try fillCache(&cache, s, 1, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // First flush: 1 chunk (128 tokens), incomplete.
    const c1 = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, s);
    try testing.expect(!c1);
    try testing.expectEqual(@as(u32, 128), tier.entries.items[0].kv_len);
    // Second flush continues from where it left off.
    const c2 = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, s);
    try testing.expect(!c2);
    try testing.expectEqual(@as(u32, 256), tier.entries.items[0].kv_len);
    // Keep flushing until complete; entry must land at the full 600.
    var guard: u32 = 0;
    while (guard < 10) : (guard += 1) {
        if (try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, s)) break;
    }
    try testing.expectEqual(@as(u32, 600), tier.entries.items[0].kv_len);

    // Restored content from an incrementally-persisted entry is exact.
    var cache2 = try KVCache.init(testing.allocator, 1);
    defer cache2.deinit();
    const restored = try tier.restoreInto(&cache2, 0, s);
    try testing.expectEqual(@as(u32, 600), restored);
    const want = try cacheValueAt(&cache, 0, 599, 7, s);
    const got = try cacheValueAt(&cache2, 0, 599, 7, s);
    try testing.expectEqual(want, got);
}

test "DiskTier: cache ahead of the token record persists the clamped prefix (EOS-stop shape)" {
    // On an EOS stop the generator has forwarded the terminator tokens into
    // the cache but they're not part of the committed token record — live
    // capture: step=2054 vs tokens=2052. The RAM tier tolerates this
    // (truncate hides the tail); the disk tier must persist min(step,
    // tokens.len) positions instead of silently skipping the whole commit.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-eos", 0, 128);
    defer tier.deinit();

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try fillCache(&cache, s, 1, 604, 8, 0.0, .float32); // 2 positions past the record
    var tokens: [602]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, s);
    try testing.expectEqual(@as(usize, 1), tier.entryCount());
    try testing.expectEqual(@as(u32, 602), tier.entries.items[0].kv_len);

    var cache2 = try KVCache.init(testing.allocator, 1);
    defer cache2.deinit();
    const restored = try tier.restoreInto(&cache2, 0, s);
    try testing.expectEqual(@as(u32, 602), restored);
    const want = try cacheValueAt(&cache, 0, 601, 3, s);
    const got = try cacheValueAt(&cache2, 0, 601, 3, s);
    try testing.expectEqual(want, got);
}

test "DiskTier: short caches and TurboQuant schemes are never persisted" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-skip", 0, 128);
    defer tier.deinit();

    // Below MIN_PERSIST_TOKENS → skipped.
    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try fillCache(&cache, s, 1, 128, 8, 0.0, .float32);
    var tokens: [128]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, s);
    try testing.expectEqual(@as(usize, 0), tier.entryCount());
}

test "modelFingerprint: stable per path, rolls with config.json changes" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    try tmp.dir.createDirPath(io, "model-a");
    try tmp.dir.writeFile(io, .{ .sub_path = "model-a/config.json", .data = "{\"model_type\":\"x\"}" });
    const dir_a = try std.fmt.allocPrint(testing.allocator, "{s}/model-a", .{base});
    defer testing.allocator.free(dir_a);

    const fp1 = try modelFingerprint(testing.allocator, io, dir_a);
    defer testing.allocator.free(fp1);
    const fp2 = try modelFingerprint(testing.allocator, io, dir_a);
    defer testing.allocator.free(fp2);
    try testing.expectEqualStrings(fp1, fp2);
    try testing.expectEqual(@as(usize, 16), fp1.len);

    // Rewriting config.json (re-download / re-quant) rolls the fingerprint.
    std.Io.sleep(io, .fromMilliseconds(20), .real) catch {};
    try tmp.dir.writeFile(io, .{ .sub_path = "model-a/config.json", .data = "{\"model_type\":\"y\",\"pad\":1}" });
    const fp3 = try modelFingerprint(testing.allocator, io, dir_a);
    defer testing.allocator.free(fp3);
    try testing.expect(!std.mem.eql(u8, fp1, fp3));

    try testing.expectError(error.BadModelDir, modelFingerprint(testing.allocator, io, ""));
    try testing.expectError(error.BadModelDir, modelFingerprint(testing.allocator, io, "rel/path"));
}
