//! References to the MTP head and its cache across the sidecar, in-checkpoint and module-owned heads.
const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const transformer_mod = @import("../transformer.zig");
const mtp_lookup = @import("mtp_lookup.zig");
const mtp_mod = @import("../../../sdk/spec/mtp/mtp.zig");
const mimo_mtp = @import("../../../plugins/mimo_v2/mtp.zig");
const glm_mtp = @import("../../../plugins/glm5_next/mtp.zig");
const Transformer = transformer_mod.Transformer;
const KVCache = transformer_mod.KVCache;
const testing = std.testing;


/// Grammar-constrained sampling state. The caller owns `grammar`, `token_bytes`,
/// and `mask_buf`; the generator only reads them. `mask_buf.len` must equal
/// `token_bytes.bytes.len` (the tokenizer's vocab size).
/// Which MTP head a generator drives.
///
/// The controller around it is head-agnostic — the EV planner, the acceptance
/// gate, `mtpBatchedAcceptGraph`, the pre-draft and the horizon valve all work
/// in tokens and probabilities — so the split is exactly these five operations
/// and nothing else.
/// Most drafts one MTP round verifies: the head chain or a prompt-lookup run.
pub const MAX_ROUND_DRAFTS: usize = @max(mtp_mod.MAX_DEPTH, mtp_lookup.MAX_DRAFT_STRONG);

pub const MtpHeadRef = union(enum) {
    qwen: *mtp_mod.MtpModel,
    /// qwen4_exp: the head and its history live on the Transformer
    /// (`qwen4_mtp`, module-owned ⇒ single-flight); row r of the history is
    /// (pre-mixer stream at position r, token r+1), query position r+1.
    qwen4: *Transformer,
    /// mimo_v2: the checkpoint's own heads; per-request rows live in `mimo_mtp.State`.
    mimo: *mimo_mtp.Head,
    /// glm5_next: the checkpoint's `mtp.0` NextN layer; per-request state is a one-layer `KVCache`.
    glm: *glm_mtp.Head,

    /// Is this head's decode state module-owned (one per model, `Qwen4Mtp.cache`) rather
    /// than per-request? Both the scheduler and the sticky-serial arm ask this.
    pub fn moduleOwned(self: MtpHeadRef) bool {
        return switch (self) {
            .qwen => false,
            // Per-request state swaps onto the module (`Qwen4MtpState`); nothing is shared.
            .qwen4, .mimo, .glm => false,
        };
    }

    pub fn makeCache(self: MtpHeadRef, allocator: std.mem.Allocator) !MtpCacheRef {
        return switch (self) {
            .qwen => |h| .{ .qwen = try h.makeCache(allocator) },
            .qwen4 => |t| blk: {
                const st = try allocator.create(transformer_mod.Transformer.Qwen4MtpState);
                errdefer allocator.destroy(st);
                st.* = try t.qwen4MtpStateNew();
                t.qwen4MtpActivate(st);
                break :blk .{ .qwen4 = .{ .t = t, .st = st, .allocator = allocator } };
            },
            .mimo => |h| .{ .mimo = .{ .st = try h.newState(allocator) } },
            .glm => |h| .{ .glm = try h.makeCache(allocator) },
        };
    }

    /// The head projects only what the caller consumes: this step wants the
    /// LAST row (`want_logits`) or no logits at all, never the S-row block —
    /// the merged history step after a partial accept is `1 + accepted` rows
    /// wide and a full-block mixer + 248320-wide lm_head over it is thrown
    /// away but for one row.
    fn qwen4Step(t: *Transformer, id_arr: mlx.mlx_array, hidden: mlx.mlx_array, rope_offset: c_int, want: mtp_mod.StepWant, mrope_ctx: ?mtp_mod.MropeContext) !mtp_mod.StepOut {
        // `.mixed` is NOT `.none`: a rerank draft still needs the mixer output,
        // which is the vector the lm_head consumes. `hidden_next` is the
        // PRE-mixer stream on this arm and would be a silent shape error there.
        const project: Transformer.Qwen4MtpProject = switch (want) {
            .logits => .last_row,
            .mixed => .mixed_last_row,
            .none => .none,
        };
        const out = try t.qwen4MtpForward(hidden, id_arr, rope_offset + 1, mrope_ctx, project);
        return .{ .logits = out.logits, .hidden_next = out.stream, .rerank_x = out.mixed };
    }

    /// One head forward over L positions. `.logits` returns the LAST row only,
    /// on both arms. `.mixed` asks for the lm_head's INPUT vector instead: on
    /// the sidecar that is `hidden_next` and nothing extra is produced, on
    /// qwen4_exp it is the mixer output, returned in `rerank_x`. `host_ids` are
    /// `id_arr`'s values when it carries committed rows (null on a chained draft
    /// step); the MiMo arm keeps them on the host.
    pub fn forward(
        self: MtpHeadRef,
        target: *Transformer,
        cache: *MtpCacheRef,
        id_arr: mlx.mlx_array,
        hidden: mlx.mlx_array,
        rope_offset: c_int,
        want: mtp_mod.StepWant,
        mrope_ctx: ?mtp_mod.MropeContext,
        host_ids: ?[]const u32,
    ) !mtp_mod.StepOut {
        return switch (self) {
            .mimo => |h| mimoStep(h, target, &cache.mimo, id_arr, hidden, @intCast(rope_offset), want, host_ids),
            .glm => |h| h.forward(target, &cache.glm, id_arr, hidden, want),
            .qwen => |h| mtp_mod.forwardWithMrope(h, target, &cache.qwen, id_arr, hidden, rope_offset, want == .logits, mrope_ctx),
            .qwen4 => |t| blk: {
                cache.activate();
                break :blk qwen4Step(t, id_arr, hidden, rope_offset, want, mrope_ctx);
            },
        };
    }

    /// One draft row past a padded history append (`Transformer.HeadPlace`); qwen4 only.
    pub fn forwardPlaced(self: MtpHeadRef, cache: *MtpCacheRef, id_arr: mlx.mlx_array, hidden: mlx.mlx_array, place: *const Transformer.HeadPlace, want: mtp_mod.StepWant) !mtp_mod.StepOut {
        const t = switch (self) {
            .qwen, .mimo, .glm => return error.MtpPlacement,
            .qwen4 => |t| t,
        };
        cache.activate();
        const project: Transformer.Qwen4MtpProject = switch (want) {
            .logits => .last_row,
            .mixed => .mixed_last_row,
            .none => .none,
        };
        const out = try t.qwen4MtpForwardPlaced(hidden, id_arr, place, project);
        return .{ .logits = out.logits, .hidden_next = out.stream, .rerank_x = out.mixed };
    }

    /// MiMo heads chain no hidden: a step carrying committed rows (`host_ids`) opens
    /// the round at head 0, each later step runs the next head on its draft.
    /// `hidden_next` hands the trunk hidden on unchanged.
    fn mimoStep(h: *mimo_mtp.Head, target: *Transformer, ref: *MtpCacheRef.MimoRef, id_arr: mlx.mlx_array, hidden: mlx.mlx_array, rope_offset: usize, want: mtp_mod.StepWant, host_ids: ?[]const u32) !mtp_mod.StepOut {
        var hidden_next = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(hidden_next);
        try mlx.check(mlx.mlx_array_set(&hidden_next, hidden));
        if (want == .none) {
            try h.appendHistory(target, ref.st, host_ids orelse return error.MimoMtpNeedsHostIds, hidden, rope_offset);
            return .{ .logits = .{ .ctx = null }, .hidden_next = hidden_next };
        }
        const step_i: usize = if (host_ids == null) ref.next_step else 0;
        const last = if (host_ids) |ids|
            try h.draftStep(target, ref.st, 0, null, ids, hidden, rope_offset)
        else
            try h.draftStep(target, ref.st, step_i, id_arr, &.{}, null, rope_offset);
        ref.next_step = step_i + 1;
        if (want == .mixed) return .{ .logits = .{ .ctx = null }, .hidden_next = hidden_next, .rerank_x = last };
        defer _ = mlx.mlx_array_free(last);
        return .{ .logits = try target.lmHeadForDraft(last), .hidden_next = hidden_next };
    }

    /// Deepest draft the head can make: a MiMo draft past its last head re-runs
    /// that head at the same position and is never accepted.
    pub fn maxDepth(self: MtpHeadRef) u32 {
        return switch (self) {
            .mimo => |h| @intCast(h.heads),
            .qwen, .qwen4, .glm => std.math.maxInt(u32),
        };
    }

    /// Append committed history without projecting logits.
    pub fn appendHistory(
        self: MtpHeadRef,
        target: *Transformer,
        cache: *MtpCacheRef,
        token_ids: []const u32,
        hidden: mlx.mlx_array,
        rope_offset: c_int,
        mrope_ctx: ?mtp_mod.MropeContext,
        allocator: std.mem.Allocator,
    ) !void {
        switch (self) {
            .qwen => |h| try mtp_mod.appendHistoryWithMrope(h, target, &cache.qwen, token_ids, hidden, rope_offset, mrope_ctx),
            .mimo => |h| try h.appendHistory(target, cache.mimo.st, token_ids, hidden, @intCast(rope_offset)),
            .glm => |h| try h.appendHistory(target, &cache.glm, token_ids, hidden),
            .qwen4 => |t| {
                const ids_i32 = try allocator.alloc(i32, token_ids.len);
                defer allocator.free(ids_i32);
                for (token_ids, 0..) |tok, i| ids_i32[i] = @intCast(tok);
                const shape = [_]c_int{@intCast(token_ids.len)};
                const id_arr = mlx.mlx_array_new_data(ids_i32.ptr, &shape, 1, .int32);
                defer _ = mlx.mlx_array_free(id_arr);
                cache.activate();
                const out = try qwen4Step(t, id_arr, hidden, rope_offset, .none, mrope_ctx);
                _ = mlx.mlx_array_free(out.hidden_next);
            },
        }
    }

    /// Draft-rerank scheme (see mtp.zig): greedy drafts pick via a coarse
    /// 2-bit readout + trunk-head top-32 re-score instead of a full
    /// draft-head projection + argmax.
    pub fn canRerankDrafts(self: MtpHeadRef) bool {
        return switch (self) {
            .qwen => |h| h.canRerankDrafts(),
            // Built lazily on the first ask: `loadQwen4Mtp` runs while the
            // Transformer is still under construction and its lm_head — the
            // very weight the coarse head is a copy of — is not assigned yet.
            .qwen4 => |t| t.qwen4DraftRerankReady(),
            .mimo => |h| h.canRerankDrafts(),
            .glm => |h| h.canRerankDrafts(),
        };
    }

    /// `x` is whatever the TARGET's lm_head consumes for this arm — the
    /// sidecar's post-norm hidden, or qwen4_exp's MIXER output. Never
    /// `hidden_next` on qwen4_exp: that is the pre-mixer `[B,S,hc*H]` stream.
    pub fn draftSelect(
        self: MtpHeadRef,
        target: *Transformer,
        x: mlx.mlx_array,
        suppress_mask: ?mlx.mlx_array,
    ) !mlx.mlx_array {
        return switch (self) {
            .qwen => |h| h.draftSelect(target, x, suppress_mask),
            .qwen4 => |t| t.qwen4DraftSelect(x, suppress_mask),
            .mimo => |h| h.draftSelect(target, x, suppress_mask),
            .glm => |h| h.draftSelect(target, x, suppress_mask),
        };
    }

    /// The exact re-scored top-32 shortlist a SAMPLED draft proposes from, or
    /// null when the coarse rerank head is unavailable. `x` is the same vector
    /// `draftSelect` takes.
    pub fn draftShortlist(
        self: MtpHeadRef,
        target: *Transformer,
        x: mlx.mlx_array,
        suppress_mask: ?mlx.mlx_array,
    ) !?mtp_mod.Shortlist {
        return switch (self) {
            .qwen => |h| h.draftShortlist(target, x, suppress_mask),
            .qwen4 => |t| t.qwen4DraftShortlist(x, suppress_mask),
            .mimo => |h| h.draftShortlist(target, x, suppress_mask),
            .glm => |h| h.draftShortlist(target, x, suppress_mask),
        };
    }

    /// `draftSelect` over group rows `x` `[N,1,H]`: `[N,1]` int32 ids.
    pub fn draftSelectBatched(self: MtpHeadRef, target: *Transformer, x: mlx.mlx_array, suppress_mask: ?mlx.mlx_array) !mlx.mlx_array {
        return switch (self) {
            .qwen => |h| h.draftSelectBatched(target, x, suppress_mask),
            .qwen4 => |t| t.qwen4DraftSelectBatched(x, suppress_mask),
            .mimo, .glm => mtp_mod.fullReadoutArgmax(target.s, target, x, suppress_mask),
        };
    }

    /// `draftShortlist` per group row; false = no coarse head.
    pub fn draftShortlistsBatched(self: MtpHeadRef, target: *Transformer, x: mlx.mlx_array, suppress_mask: ?mlx.mlx_array, out: []mtp_mod.Shortlist) !bool {
        return switch (self) {
            .qwen => |h| h.draftShortlistsBatched(target, x, suppress_mask, out),
            .qwen4 => |t| t.qwen4DraftShortlistsBatched(x, suppress_mask, out),
            .mimo, .glm => mtp_mod.exactShortlistsBatched(target.s, target, x, suppress_mask, out),
        };
    }

    /// Every G17/NAX cost surface is calibrated against one exact runtime
    /// geometry — the dense Qwen3.6/3.8-27B sidecars on the `.qwen` arm, the
    /// qwen4_exp in-checkpoint head on its own arm — so every other head
    /// plans under `generic`: an off-profile head served by a calibrated
    /// surface would plan depths the measurement never covered.
    pub fn costProfile(self: MtpHeadRef, target: *const Transformer, kv: transformer_mod.KVQuantConfig) mtp_mod.MtpCostProfile {
        return switch (self) {
            .qwen => |h| h.m5NaxCostProfile(target),
            .qwen4 => |t| mtp_mod.qwen4G17CostProfileForKv(t, kv),
            .mimo, .glm => .generic,
        };
    }

    /// The head's last healthy acceptance surface, or null when it has never
    /// published one. BOTH arms store it on the head object itself — the
    /// sidecar `MtpModel` and the in-checkpoint `Qwen4Mtp` — so a seed dies
    /// with the head and can never cross models.
    pub fn evSeed(self: MtpHeadRef) ?struct { accept: [mtp_mod.MAX_DEPTH]f32, m_lo: u32 } {
        return switch (self) {
            .qwen => |h| if (h.ev_seed_accept) |a| .{ .accept = a, .m_lo = h.ev_seed_m_lo } else null,
            .mimo => |h| if (h.ev_seed_accept) |a| .{ .accept = a, .m_lo = h.ev_seed_m_lo } else null,
            .glm => |h| if (h.ev_seed_accept) |a| .{ .accept = a, .m_lo = h.ev_seed_m_lo } else null,
            .qwen4 => |t| blk: {
                if (t.qwen4_mtp) |*m| {
                    if (m.ev_seed_accept) |a| break :blk .{ .accept = a, .m_lo = m.ev_seed_m_lo };
                }
                break :blk null;
            },
        };
    }

    pub fn setEvSeed(self: MtpHeadRef, accept: [mtp_mod.MAX_DEPTH]f32, m_lo: u32) void {
        switch (self) {
            .qwen => |h| {
                h.ev_seed_accept = accept;
                h.ev_seed_m_lo = m_lo;
            },
            .mimo => |h| {
                h.ev_seed_accept = accept;
                h.ev_seed_m_lo = m_lo;
            },
            .glm => |h| {
                h.ev_seed_accept = accept;
                h.ev_seed_m_lo = m_lo;
            },
            .qwen4 => |t| {
                // A model served with `--no-mtp` has no head to seed.
                if (t.qwen4_mtp) |*m| {
                    m.ev_seed_accept = accept;
                    m.ev_seed_m_lo = m_lo;
                }
            },
        }
    }
};

/// `MLX_SERVE_MTP_HEAD_PERSIST=0`: the qwen4_exp head's committed history is neither
/// snapshotted into a prefix-cache entry nor restored from one.
var mtp_head_persist_env: ?bool = null;
/// Only a literal "0" turns persistence off.
pub fn mtpHeadPersistFromEnv(raw: ?[]const u8) bool {
    const v = raw orelse return true;
    return !std.mem.eql(u8, v, "0");
}
pub fn mtpHeadPersistEnabled() bool {
    if (mtp_head_persist_env) |v| return v;
    const raw = std.c.getenv("MLX_SERVE_MTP_HEAD_PERSIST");
    const on = mtpHeadPersistFromEnv(if (raw) |r| std.mem.sliceTo(r, 0) else null);
    mtp_head_persist_env = on;
    return on;
}

/// The head's committed-history cache.
pub const MtpCacheRef = union(enum) {
    qwen: KVCache,
    qwen4: Qwen4Ref,
    mimo: MimoRef,
    /// The GLM head's one-layer latent + indexer cache, driven like `.qwen`.
    glm: KVCache,

    /// A MiMo request's head rows, and the draft step its next chained forward runs.
    pub const MimoRef = struct { st: *mimo_mtp.State, next_step: usize = 0 };

    /// The in-checkpoint head plus this request's own half of it.
    pub const Qwen4Ref = struct { t: *Transformer, st: *transformer_mod.Transformer.Qwen4MtpState, allocator: std.mem.Allocator };

    /// Install this request's state on the qwen4 head; no-op on the sidecar arm.
    pub fn activate(self: *const MtpCacheRef) void {
        switch (self.*) {
            .qwen, .mimo, .glm => {},
            .qwen4 => |r| r.t.qwen4MtpActivate(r.st),
        }
    }

    pub fn qwen4StateForBatch(self: *const MtpCacheRef) *Transformer.Qwen4MtpState {
        const ref = self.qwen4;
        ref.t.qwen4MtpRelease(ref.st);
        return ref.st;
    }

    pub fn step(self: *const MtpCacheRef) usize {
        return switch (self.*) {
            .qwen, .glm => |*c| c.step,
            .qwen4 => |r| blk: {
                r.t.qwen4MtpActivate(r.st);
                break :blk r.t.qwen4_mtp.?.seq_offset;
            },
            .mimo => |r| r.st.step(),
        };
    }

    /// The underlying KVCache — what the prefix cache's spec-snap machinery
    /// snapshots and restores. The qwen4 head's state is not KV-only, so a caller taking this
    /// pointer must also carry `head()`. Null when head persistence is off.
    pub fn kv(self: *MtpCacheRef) ?*KVCache {
        return switch (self.*) {
            .qwen, .glm => |*c| c,
            .qwen4 => |r| blk: {
                r.t.qwen4MtpActivate(r.st);
                break :blk if (mtpHeadPersistEnabled()) &r.t.qwen4_mtp.?.cache else null;
            },
            // The heads only ever read the last window of rows, rebuilt from the uncached tail.
            .mimo => null,
        };
    }

    /// The Transformer owning the in-checkpoint head; null on the sidecar arm and whenever `kv()` is null.
    pub fn head(self: *MtpCacheRef) ?*Transformer {
        return switch (self.*) {
            .qwen, .mimo, .glm => null,
            .qwen4 => |r| blk: {
                r.t.qwen4MtpActivate(r.st);
                break :blk if (mtpHeadPersistEnabled()) r.t else null;
            },
        };
    }

    pub fn truncate(self: *MtpCacheRef, len: usize, s: mlx.mlx_stream) !void {
        switch (self.*) {
            .qwen, .glm => |*c| try c.truncate(len, s),
            .qwen4 => |r| {
                r.t.qwen4MtpActivate(r.st);
                try r.t.qwen4MtpTruncate(len);
            },
            .mimo => |*r| {
                try r.st.truncate(s, len);
                r.next_step = 0;
            },
        }
    }

    pub fn deinit(self: *MtpCacheRef) void {
        switch (self.*) {
            .qwen, .glm => |*c| c.deinit(),
            .qwen4 => |r| {
                r.t.qwen4MtpRelease(r.st);
                r.st.deinit();
                r.allocator.destroy(r.st);
            },
            .mimo => |r| {
                const allocator = r.st.allocator;
                r.st.deinit();
                allocator.destroy(r.st);
            },
        }
    }

    /// Append this cache's storage to a prefill eval batch, so the chunk's
    /// activation graph can be freed with the chunk instead of accumulating
    /// across the whole prompt.
    pub fn appendEvalArrays(self: *const MtpCacheRef, vec: mlx.mlx_vector_array) void {
        switch (self.*) {
            .qwen, .glm => |*c| c.appendEvalArrays(vec),
            .qwen4 => |r| {
                r.t.qwen4MtpActivate(r.st);
                r.t.qwen4_mtp.?.cache.appendEvalArrays(vec);
            },
            .mimo => |r| r.st.appendEvalArrays(vec),
        }
    }
};

/// An MTP committed-history cache restored from the prefix cache, plus the
/// absolute target position its index 0 represents.
pub const MtpRestored = struct { cache: MtpCacheRef, base: usize };

// ── tests ──


test "MTP head persistence kill switch: only a literal 0 turns it off" {
    try testing.expect(mtpHeadPersistFromEnv(null));
    try testing.expect(mtpHeadPersistFromEnv(""));
    try testing.expect(mtpHeadPersistFromEnv("1"));
    try testing.expect(!mtpHeadPersistFromEnv("0"));
}
