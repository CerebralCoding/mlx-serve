//! engines/mlx/dispatch, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns defaultCtx, module_owned_state_fields, module_shared_readonly_fields, ownsModuleDecodeState, sharesModuleReadonlyState, moduleSpecWiring, and 7 more.

const std = @import("std");
const mlx_stream_mod = if (@import("build_options").mlx_stream) @import("../../sdk/weights/streamed/glue.zig") else @import("../../sdk/weights/streamed/stub.zig");
const distributed_mod = @import("cluster/distributed.zig");
const mlx = @import("../../mlx/ffi.zig");
const abi = @import("../../abi/plugin.zig");
const ForwardCtx = @import("forward_ctx.zig").ForwardCtx;
const testing = @import("kv/cache.zig").testing;
const forwardMlxStream = @import("modules/mlx_stream.zig").forwardMlxStream;
const Transformer = @import("transformer.zig").Transformer;
const plugin_driver = @import("plugin/driver.zig");
const intree = @import("plugin/intree.zig");

/// Default forward context, routing through the Transformer's own state.
/// Used by the single-slot legacy path and by Phase-2 prefill on a slot
/// that has had its KVCache temporarily swapped onto the Transformer.
pub fn defaultCtx(self: *Transformer) ForwardCtx {
    return .{
        .cache = &self.cache,
        .moe_seq_offset = &self.moe_seq_offset,
        .ssm_entries = self.ssm_entries,
        .capture_hidden = self.capture_hidden,
        .vision_embeddings = self.vision_embeddings,
    };
}

/// Archs whose per-request decode state lives on their OWN module — one
/// instance per MODEL, not one per slot — instead of in this shell's
/// KVCache/ssm_entries. Every member is by construction single-flight at
/// admission (`scheduler.modelExclusiveDecode`): the state is rebuilt at
/// `cache.step == 0` and advanced by each tick, so a second interleaved
/// slot deinits and rebuilds the live request's state and both then append
/// to the ONE state. Add a new arm here the moment its pointer field is
/// added above, or the arch serves two clients one mangled stream.
pub const module_owned_state_fields = [_][]const u8{ "mlx_stream_ext", "dist_peer" };

/// Module pointer fields that hold READ-ONLY per-model state (qwen4: the
/// n-gram hash + mmapped table). Every per-request thing lives on the
/// slot's `SSMCacheEntry` (GDN pair, PLE window + tokens, QSA keys +
/// pooled blocks), so slots interleave AND batch freely. The one
/// module-owned piece on this arch is the MTP head's cache, and the
/// scheduler makes the SLOT that uses it exclusive (`slotExclusiveDecode`).
pub const module_shared_readonly_fields = [_][]const u8{ "qwen4", "dist_tp" };

/// A served model's caps; null on the legacy path.
fn servedCaps(self: *const Transformer) ?abi.Caps {
    if (self.plugin == null) return null;
    return self.config.plugin.?.parsed.caps;
}

pub fn ownsModuleDecodeState(self: *const Transformer) bool {
    if (servedCaps(self)) |c| if (c.owns_decode_state) return true;
    inline for (module_owned_state_fields) |f| {
        if (@field(self, f) != null) return true;
    }
    return false;
}

pub fn sharesModuleReadonlyState(self: *const Transformer) bool {
    if (servedCaps(self)) |c| if (c.shared_readonly_state) return true;
    inline for (module_shared_readonly_fields) |f| {
        if (@field(self, f) != null) return true;
    }
    return false;
}

/// Spec wiring class: module archs run only their OWN MTP head
/// (`moduleStateSpecRollback`), PLD/drafters stay off — unmeasured there.
/// Covers both the owned and the shared-readonly lists so a shared arch
/// does not fall onto the generic wiring and pick up PLD by default.
pub fn moduleSpecWiring(self: *const Transformer) bool {
    return self.ownsModuleDecodeState() or self.sharesModuleReadonlyState();
}

/// Can this arch's MODULE-owned decode state be rolled back across a
/// speculative verify?
///
/// This is what `ownsModuleDecodeState` actually stood in for: the reason
/// module-owned archs had every spec mode forced off is that a rejected
/// draft tail cannot be rewound, not the ownership itself. An arch earns
/// a true here by shipping per-position state capture plus a rewind,
/// pinned against fresh forwards at every accepted count. dsv4 cannot:
/// its rings and compressed caches have no per-position capture, and it
/// has DSpark for drafting anyway.
/// mlx-stream runs its own prompt pass: the generator hands it the whole prompt in one forward.
pub fn prefillsWholePrompt(self: *const Transformer) bool {
    if (servedCaps(self)) |c| if (c.whole_prompt_prefill) return true;
    return self.mlx_stream_ext != null;
}

/// DeepSeek's own DSpark draft stages (V4 or V4.1) loaded on this model.
pub fn dsparkStages(self: *const Transformer) usize {
    if (intree.dsv4(self)) |m| return m.n_mtp;
    if (intree.dsv41(self)) |m| return m.n_mtp;
    if (self.mlx_stream_ext) |m| return @intFromBool(mlx_stream_mod.blockSize(m) > 0);
    return 0;
}

pub fn moduleStateSpecRollback(self: *const Transformer) bool {
    if (servedCaps(self)) |c| if (c.spec_rollback) return true;
    // qwen4_exp: GDN per-position capture + PLE window/token capture +
    // positional QSA key histories (`ssmRollbackFromCapture`), so the MTP
    // head's verify can rewind. dsv4 cannot.
    return self.qwen4_mtp != null;
}

/// Is this arch's MTP head the checkpoint's OWN trained layer AND measured
/// no-worse-than-serial across the whole context ladder on this family?
///
/// `defaultEnableMtp`'s `is_moe` caution is about a BOLTED-ON sidecar
/// paying expert routing in a verify it was never designed around. It does
/// not describe a head that ships inside the checkpoint — dsv4's DSpark
/// stages already get the same exemption through `dsv4_stages`.
///
/// The bar for adding an arch here is a MEASUREMENT, not the head's
/// provenance: no losing cell on the context ladder against the same
/// model's serial decode, on prompt text that is not trivially draftable.
pub fn forward(self: *Transformer, token_ids: mlx.mlx_array) !mlx.mlx_array {
    var ctx = self.defaultCtx();
    return self.forwardWith(&ctx, token_ids);
}

pub fn forwardWith(self: *Transformer, ctx: *ForwardCtx, token_ids: mlx.mlx_array) !mlx.mlx_array {
    if (ctx.batch_slots == null) try self.ssmGroupRelease(ctx);
    if (self.plugin) |b| return plugin_driver.forward(self, b, ctx, token_ids);
    if (self.mlx_stream_ext) |m| return forwardMlxStream(self, ctx, token_ids, m);
    // EmbeddingGemma (a bidirectional gemma3 trunk) never runs causal decode.
    if (self.config.use_bidirectional_attention) return self.forwardGemma3EncoderWith(ctx, token_ids);
    if (self.qwen4 != null) return self.forwardQwen4With(ctx, token_ids);
    if (self.config.isGlm5()) return self.forwardGlm5With(ctx, token_ids);
    if (self.moe_layers != null) return self.forwardMoeWith(ctx, token_ids);
    return self.forwardStandardWith(ctx, token_ids);
}

/// Does `forwardWith` route this model through `forwardStandardWith`?
/// Mirrors the dispatch chain above IN ORDER.
pub fn usesStandardForward(self: *const Transformer) bool {
    return self.plugin == null and self.mlx_stream_ext == null and
        !self.config.use_bidirectional_attention and
        self.moe_layers == null;
}

/// Does `forwardWith` route through a path that honors
/// `ForwardCtx.capture_layers`? Standard path (v1), the moe/GDN path
/// (qwen3_5-family DFlash2 sidecars) and the hybrid path (LFM2 DSpark).
/// `DflashModel.bind` gates on this predicate. Mirrors the dispatch chain
/// above IN ORDER.
pub fn supportsLayerCapture(self: *const Transformer) bool {
    // A served plugin captures layers; a drafter also reads the trunk's embedding and head (`Bound.drafters`).
    if (self.plugin) |b| return b.drafters;
    return self.mlx_stream_ext == null and
        !self.config.use_bidirectional_attention;
}

// ── tests ──

test "ownsModuleDecodeState covers every module-owned arch" {
    // The bug this pins: a SECOND module-owned arch arrived and
    // `scheduler.modelExclusiveDecode` kept keying on `dsv4` alone, so two
    // concurrent requests shared ONE module state — the 2026-08-02 dsv4
    // clobber, verbatim. The predicate reads a NAMED LIST so it cannot
    // drift from itself; this test pins the list against the struct.
    var t: Transformer = undefined;
    t.plugin = null;
    t.rht = null;
    t.mlx_stream_ext = null;
    t.qwen4 = null;
    t.dist_peer = null;
    t.dist_tp = null;
    try testing.expect(!t.ownsModuleDecodeState());

    var fake_ext: mlx_stream_mod.Model = undefined;
    t.mlx_stream_ext = &fake_ext;
    try testing.expect(t.ownsModuleDecodeState());
    t.mlx_stream_ext = null;
    var fake_peer: distributed_mod.Peer = undefined;
    t.dist_peer = &fake_peer;
    try testing.expect(t.ownsModuleDecodeState());
}

test "every optional arch-module pointer on Transformer is a declared module-owned arch" {
    // Class guard, not an instance guard: a THIRD arch declared the same way
    // as the two above is module-owned by construction, and forgetting it in
    // the list is SILENT — serial decode still "works" until two clients
    // arrive at once. Needles are built at runtime so this test's own source
    // cannot satisfy its own scan.
    const src = @embedFile("transformer.zig");
    const opt_ptr = "?" ++ "*";
    const mod_dot = "_mod" ++ ".";
    var i: usize = 0;
    var found: usize = 0;
    while (std.mem.indexOfPos(u8, src, i, opt_ptr)) |p| {
        i = p + opt_ptr.len;
        // Shape: `    <name>: ?*<x>_mod.<Y> = null,` — the module type must
        // follow immediately, with a bare identifier in between.
        const rest = src[i..];
        const mod_at = std.mem.indexOf(u8, rest[0..@min(rest.len, 40)], mod_dot) orelse continue;
        if (std.mem.indexOfAny(u8, rest[0..mod_at], " ,)!*\"<`") != null) continue;
        const line_start = if (std.mem.lastIndexOfScalar(u8, src[0..p], '\n')) |nl| nl + 1 else 0;
        const decl = std.mem.trim(u8, src[line_start..p], " ");
        if (!std.mem.endsWith(u8, decl, ":")) continue;
        const name = decl[0 .. decl.len - 1];
        if (name.len == 0 or std.mem.indexOfAny(u8, name, " /(\"`") != null) continue;
        found += 1;
        var listed = false;
        for (Transformer.module_owned_state_fields) |f| {
            if (std.mem.eql(u8, f, name)) listed = true;
        }
        for (Transformer.module_shared_readonly_fields) |f| {
            if (std.mem.eql(u8, f, name)) listed = true;
        }
        if (!listed) {
            std.debug.print("module arch field declared in neither list: {s}\n", .{name});
            return error.UndeclaredModuleOwnedArch;
        }
    }
    // A scan that matches nothing passes vacuously — pin the known count.
    try testing.expectEqual(Transformer.module_owned_state_fields.len + Transformer.module_shared_readonly_fields.len, found);
}
