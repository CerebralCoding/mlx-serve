//! A served model's MTP head (`SpecHooks.head`) as the generator drives it (`MtpHeadRef.plugin`): the plugin runs
//! the head's layer over its own KV cache and hands back the last row's normed output; the engine keeps the readout
//! the sidecar head keeps (the draft-only low-bit head, the coarse rerank head, the full trunk-head readout), built
//! from the trunk head the shell borrowed through `trunk_weight`, so drafts select exactly as on the legacy path.

const std = @import("std");
const abi = @import("../../../abi/plugin.zig");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../host/log.zig");
const mtp_mod = @import("../../../sdk/spec/mtp/mtp.zig");
const cache_ops = @import("../kv/cache_ops.zig");
const FsPackDir = @import("../../../host/registry/pack_dir.zig").FsPackDir;
const Transformer = @import("../transformer.zig").Transformer;
const KVCache = @import("../kv/cache.zig").KVCache;
const PluginRef = @import("../spec/mtp_head_ref.zig").MtpCacheRef.PluginRef;
const Bound = @import("driver.zig").Bound;

const MtpModel = mtp_mod.MtpModel;
const Array = mlx.mlx_array;

pub const Head = struct {
    s: mlx.mlx_stream,
    bound: *Bound,
    hooks: *const abi.HeadHooks,
    /// The cost surface the plugin declares (`SpecFacts.cost_profile`); generic when it names none.
    profile: mtp_mod.MtpCostProfile = .generic,
    /// Heads, one per draft depth (`SpecFacts.mtp_heads`); 1 = one head chained on its own output.
    heads: u32 = 1,
    // The readout state `MtpModel` keeps, read by its readout functions.
    draft_head: ?mtp_mod.QLinear = null,
    draft_head_bits: u32 = 0,
    draft_head_group: u32 = 0,
    rerank_coarse: ?mtp_mod.RerankCoarse = null,
    rerank_logged: bool = false,
    ev_seed_accept: ?[mtp_mod.MAX_DEPTH]f32 = null,
    ev_seed_m_lo: u32 = 1,

    pub fn deinit(self: *Head) void {
        if (self.rerank_coarse) |*rc| rc.deinit();
        if (self.draft_head) |*dh| dh.deinit();
    }

    pub fn m5NaxCostProfile(self: *const Head, _: *const Transformer) mtp_mod.MtpCostProfile {
        return self.profile;
    }

    pub fn canRerankDrafts(self: *const Head) bool {
        return self.rerank_coarse != null;
    }

    /// One head forward: `logits = .last_row` lets leading rows append K/V only (the engine projects the last
    /// row), `.none` with `want_hidden` runs every row, `.none` without it appends history only.
    fn run(self: *Head, ref: *PluginRef, head: u32, id_arr: Array, hidden: Array, rope_offset: c_int, logits: abi.Logits, want_hidden: bool) !Array {
        const rows: c_int = mlx.getShape(id_arr)[0];
        var ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ids);
        try mlx.check(mlx.mlx_reshape(&ids, id_arr, &[_]c_int{ 1, rows }, 2, self.s));
        var slot: cache_ops.Slot = .{ .kv = &ref.kv, .ssm = ref.ssm };
        const handles = [_]abi.Handle{slot.handle()};
        const pos = [_]u32{@intCast(rope_offset)};
        const f: abi.Forward = .{
            .mode = if (rows == 1) .decode else .prefill,
            .logits = logits,
            .want_hidden = want_hidden,
            .rows = @intCast(rows),
            .tokens = .{},
            .ids = .{ .ctx = ids.ctx },
            .positions = .of(&pos),
            .caches = .of(&handles),
            .cache_ops = &cache_ops.ops,
            .hidden_in = .{ .ctx = hidden.ctx },
            .head = head,
        };
        var out: abi.ForwardOut = .{};
        const v = self.hooks.forward(self.bound.model, &f, &out);
        if (out.logits.ctx != null) _ = mlx.mlx_array_free(.{ .ctx = out.logits.ctx });
        if (!v.isOk()) {
            if (out.hidden.ctx != null) _ = mlx.mlx_array_free(.{ .ctx = out.hidden.ctx });
            log.err("[plugin] {s}: MTP head forward failed: {s}\n", .{ self.bound.table.name.slice(), v.why.slice() });
            return error.PluginForwardFailed;
        }
        return .{ .ctx = out.hidden.ctx };
    }

    /// Committed rows (`opens`) open a round at head 0; each chained step runs the next head, the last one again
    /// past the end. With several heads every step reads the trunk's hidden, handed on unchanged, and the head's
    /// output is the vector the trunk head reads (`rerank_x` under `.mixed`).
    pub fn forward(self: *Head, target: *Transformer, ref: *PluginRef, id_arr: Array, hidden: Array, rope_offset: c_int, want: mtp_mod.StepWant, mrope_ctx: ?mtp_mod.MropeContext, opens: bool) !mtp_mod.StepOut {
        if (mrope_ctx != null) return error.PluginForwardUnsupported;
        const head = if (opens) 0 else @min(ref.next_head, self.heads - 1);
        // Several heads: a step that wants nothing appends history (no `want_hidden`), as MiMo's legacy arm does.
        const want_hidden = self.heads <= 1 or want != .none;
        if (want_hidden) ref.next_head = head + 1;
        const out = try self.run(ref, head, id_arr, hidden, rope_offset, if (want == .logits) .last_row else .none, want_hidden);
        if (out.ctx == null and want_hidden) return error.PluginForwardFailed;
        if (self.heads <= 1) {
            if (want != .logits) return .{ .logits = .{ .ctx = null }, .hidden_next = out };
            errdefer _ = mlx.mlx_array_free(out);
            return .{ .logits = try mtp_mod.targetLmHead(self, target, out, self.s), .hidden_next = out };
        }
        var trunk = mlx.mlx_array_new();
        mlx.check(mlx.mlx_array_set(&trunk, hidden)) catch |e| {
            _ = mlx.mlx_array_free(out);
            _ = mlx.mlx_array_free(trunk);
            return e;
        };
        switch (want) {
            .none => {
                if (out.ctx != null) _ = mlx.mlx_array_free(out);
                return .{ .logits = .{ .ctx = null }, .hidden_next = trunk };
            },
            .mixed => return .{ .logits = .{ .ctx = null }, .hidden_next = trunk, .rerank_x = out },
            .logits => {
                defer _ = mlx.mlx_array_free(out);
                errdefer _ = mlx.mlx_array_free(trunk);
                return .{ .logits = try mtp_mod.targetLmHead(self, target, out, self.s), .hidden_next = trunk };
            },
        }
    }

    pub fn appendHistory(self: *Head, ref: *PluginRef, token_ids: []const u32, hidden: Array, rope_offset: c_int, mrope_ctx: ?mtp_mod.MropeContext, allocator: std.mem.Allocator) !void {
        if (token_ids.len == 0) return;
        if (mrope_ctx != null) return error.PluginForwardUnsupported;
        const ids = try allocator.alloc(i32, token_ids.len);
        defer allocator.free(ids);
        for (token_ids, ids) |t, *o| o.* = @intCast(t);
        const id_arr = mlx.mlx_array_new_data(ids.ptr, &[_]c_int{@intCast(ids.len)}, 1, .int32);
        defer _ = mlx.mlx_array_free(id_arr);
        ref.next_head = 0;
        const out = try self.run(ref, 0, id_arr, hidden, rope_offset, .none, false);
        if (out.ctx != null) _ = mlx.mlx_array_free(out);
    }

    pub fn draftSelect(self: *Head, target: *Transformer, x: Array, suppress_mask: ?Array) !Array {
        return MtpModel.draftSelect(self, target, x, suppress_mask);
    }

    pub fn draftShortlist(self: *Head, target: *Transformer, x: Array, suppress_mask: ?Array) !?mtp_mod.Shortlist {
        return MtpModel.draftShortlist(self, target, x, suppress_mask);
    }
};

/// The served model's head, loaded (`HeadHooks.load`) and its readout built from the shell's trunk head; null when
/// the table has none or the shell carries no trunk head (`Bound.trunk`). A failed load only turns MTP off.
pub fn load(io: std.Io, a: std.mem.Allocator, xfm: *Transformer, model_dir: []const u8) ?*Head {
    const b = xfm.plugin orelse return null;
    const hooks = (b.table.spec orelse return null).head orelse return null;
    if (!b.trunk) {
        log.warn("[plugin] {s}: the MTP head needs the trunk head (trunk_weight) — MTP off\n", .{b.table.name.slice()});
        return null;
    }
    var fs = FsPackDir.open(io, model_dir) catch return null;
    defer fs.close();
    const dir = fs.packDir();
    const parsed = &xfm.config.plugin.?.parsed;
    const v = hooks.load(b.model, &.{ .model_dir = abi.Str.of(model_dir), .dir = &dir, .parsed = parsed, .stream = .{ .ctx = xfm.s.ctx } });
    if (!v.isOk()) {
        log.warn("[plugin] {s}: MTP head failed to load ({s}) — MTP off\n", .{ b.table.name.slice(), v.why.slice() });
        return null;
    }
    const h = a.create(Head) catch return null;
    h.* = .{ .s = xfm.s, .bound = b, .hooks = hooks, .profile = std.enums.fromInt(mtp_mod.MtpCostProfile, parsed.spec.cost_profile) orelse .generic, .heads = @max(1, parsed.spec.mtp_heads) };
    const had_err = mlx.errorPending();
    if (h.heads > 1) {
        // Per-depth heads read out as MiMo's legacy arm does: the trunk head, shortlisted on a coarse copy.
        if (MtpModel.draftRerankMode() != .off and xfm.lm_head_s.ctx != null) h.rerank_coarse = mtp_mod.buildRerankCoarse(h.s, xfm, mtp_mod.rerankCoarseBits());
    } else {
        MtpModel.buildDraftHead(h, xfm) catch |err| {
            log.warn("[mtp] draft lm_head build failed ({s}) — drafts use the trunk head\n", .{@errorName(err)});
        };
        MtpModel.maybeBuildDraftRerank(h, xfm);
    }
    mlx.dropLatchedErrorUnless(had_err);
    return h;
}

pub fn destroy(h: *Head, a: std.mem.Allocator) void {
    h.deinit();
    a.destroy(h);
}
