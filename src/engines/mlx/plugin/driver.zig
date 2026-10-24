//! The MLX engine's side of a served plugin: the load (the engine's loader reads the pack's tensors, the plugin's
//! `load` builds its model from them) into a `Transformer` shell, and the forwards the generator and the batched tick
//! call, translated to `ModelTable.forward` with the slot's KV cache behind `cache_ops`. A request feature the plugin
//! table has no hook for yet is refused by name, never served on a path that ignores it.

const std = @import("std");
const abi = @import("../../../abi/plugin.zig");
const quant_abi = @import("../../../abi/quant.zig");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../sdk/util/log.zig");
const model = @import("../model.zig");
const served = @import("served.zig");
const cache_ops = @import("../kv/cache_ops.zig");
const eval_cadence = @import("../eval_cadence.zig");
const transformer_mod = @import("../transformer.zig");
const qmm_int8 = @import("../../../sdk/blocks/qmm/qmm_int8.zig");
const lmHeadMxfp8On = @import("../../../sdk/quant/mxfp8.zig").lmHeadMxfp8On;
const Resident = @import("../../../sdk/weights/resident.zig").Resident;
const firstRowExactMiss = @import("../../../sdk/quant/ternary.zig").firstRowExactMiss;
const FsPackDir = @import("../../../host/registry/pack_dir.zig").FsPackDir;
const moduleShell = @import("../modules/shell.zig").moduleShell;
const KVCache = @import("../kv/cache.zig").KVCache;
const bf16Scalar = @import("../load/weights.zig").bf16Scalar;
const mrope = @import("../../../sdk/vision/mrope.zig");
const mropeContext = @import("../methods/mrope.zig").mropeContext;
const SSMCacheEntry = @import("../ssm/entry.zig").SSMCacheEntry;
const ForwardCtx = @import("../forward_ctx.zig").ForwardCtx;
const MAX_BATCH_ROWS = @import("../batched/layered.zig").MAX_BATCH_ROWS;
const settle = @import("settle.zig");
const rht = @import("../../../sdk/quant/hadamard/rht.zig");
const hadamard = @import("../../../sdk/quant/hadamard/format.zig");
const intree = @import("intree.zig");

const Transformer = transformer_mod.Transformer;
const ModelConfig = model.ModelConfig;
const Array = mlx.mlx_array;

/// A loaded plugin model behind a `Transformer` shell (`Transformer.plugin`).
pub const Bound = struct {
    table: *const abi.ModelTable,
    model: *anyopaque,
    /// The shell carries the trunk's embedding table and head and the forward captures layers: drafters bind.
    drafters: bool = false,
    /// The shell carries the trunk's embedding table and head (an MTP head reads them without layer capture).
    trunk: bool = false,
    /// Where a slot's position lives and how a spec rollback treats `cache.step`.
    positions: Positions = .kv_step,
    /// The last slot id handed out (`settle.slotId`).
    next_slot: u32 = 0,
    /// Each layer's kind (`CacheHooks.layout`), `.full` past the layout or without one.
    kinds: [128]abi.LayerKind = @splat(.full),

    /// A layer holding recurrent state a batched tick merges.
    pub fn recurrentAt(b: *const Bound, layer: usize) bool {
        return layer < b.kinds.len and switch (b.kinds[layer]) {
            .linear, .conv, .ssm => true,
            .none, .full, .sliding => false,
        };
    }
};

/// `.kv_step`: no recurrent layer, a slot's position is `cache.step` (the legacy standard forward). A recurrent model
/// positions from `moe_seq_offset`, which the forward advances: `.gdn` when layer 0 is a linear-attention layer
/// (`cache.step` is a counter no forward reads, the legacy GDN path), `.hybrid` otherwise (the legacy hybrid path).
pub const Positions = enum { kv_step, gdn, hybrid };

fn bind(b: *Bound, p: *const abi.Parsed) void {
    var layers: [128]abi.LayerCache = undefined;
    const n = if (b.table.cache.layout) |layout| layout(p, null, &layers, layers.len) else 0;
    for (layers[0..n], 0..) |l, i| b.kinds[i] = l.kind;
    b.positions = if (p.cache.ssm_state.entries == 0) .kv_step else if (n > 0 and layers[0].kind == .linear) .gdn else .hybrid;
}

/// The served model's position source (`Positions`); `.kv_step` for a legacy shell.
pub fn positionsOf(xfm: *const Transformer) Positions {
    const b = xfm.plugin orelse return .kv_step;
    return b.positions;
}

/// The pack's map; with `tower`, loaded with the vision tower, whose tensors move there.
fn loadSplit(io: std.Io, a: std.mem.Allocator, model_dir: []const u8, config: *ModelConfig, tower: ?*model.Weights) !model.Weights {
    var w = try model.loadModelWeights(io, a, model_dir, config, tower != null);
    errdefer w.deinit();
    try model.narrowHadamardPackTables(config, &w, mlx.gpuStream());
    const t = tower orelse return w;
    var keys: std.ArrayList([]const u8) = .empty;
    defer keys.deinit(a);
    var it = w.map.keyIterator();
    while (it.next()) |k| if (!model.shouldKeepWeightKey(k.*, false)) try keys.append(a, k.*);
    try t.map.ensureUnusedCapacity(@intCast(keys.items.len));
    for (keys.items) |k| {
        const kv = w.map.fetchRemove(k).?;
        t.map.putAssumeCapacity(kv.key, kv.value);
    }
    return w;
}

/// The lossy load setting a served pack has on that the plugin path does not run yet, by its settings name.
fn lossyUnserved(config: *const ModelConfig) ?[]const u8 {
    const table = (config.plugin orelse return null).table;
    if ((config.int8_prefill_override orelse false) and mlx.metalKernelsAvailable() and !intree.servesLossy(table, .int8_prefill)) return "int8_prefill";
    if (lmHeadMxfp8On(headMxfp8Setting(config), null, mlx.cudaIntegrated(), false, config.quant_bits) and !intree.servesLossy(table, .lm_head_mxfp8)) return "lm_head_mxfp8";
    return null;
}

/// `lm_head_mxfp8` as the model's settings, else the launch flag, state it; null = the model's default.
fn headMxfp8Setting(config: *const ModelConfig) ?bool {
    return config.lm_head_mxfp8_override orelse transformer_mod.cli().lm_head_mxfp8_cli;
}

/// `LoadCtx.settings_json`: the lossy load settings stated for this model (a plugin that serves them reads them).
fn loadSettings(buf: []u8, config: *const ModelConfig) []const u8 {
    const head = headMxfp8Setting(config) orelse return "{}";
    return std.fmt.bufPrint(buf, "{{\"lm_head_mxfp8\":{s}}}", .{if (head) "true" else "false"}) catch unreachable;
}

/// A pack with a lossy setting its plugin does not run goes back to the legacy path while one exists (a type whose
/// legacy is gone keeps the binding, and the load refuses it by name). True when the binding was dropped.
pub fn declineLossy(a: std.mem.Allocator, config: *ModelConfig) bool {
    const b = config.plugin orelse return false;
    const what = lossyUnserved(config) orelse return false;
    if (served.legacyRemoved(config.model_type) or b.outside) return false;
    log.info("[plugin] {s}: {s} is on, which the plugin does not run; the legacy path loads it\n", .{ b.table.name.slice(), what });
    b.destroy(a);
    config.plugin = null;
    return true;
}

/// A pack a Thunderbolt cluster plans (`planned`) stays on its plugin only when the plugin reads the plan
/// (`Caps.cluster_plan`); otherwise the binding is dropped, so the legacy path loads it. True when it was dropped.
pub fn declinePlanned(a: std.mem.Allocator, config: *ModelConfig, planned: bool) bool {
    const b = config.plugin orelse return false;
    if (!planned or b.parsed.caps.cluster_plan) return false;
    log.info("[plugin] {s}: a cluster plan splits {s}, which the plugin does not read; the legacy path loads it\n", .{ b.table.name.slice(), config.model_type });
    b.destroy(a);
    config.plugin = null;
    return true;
}

/// Loads the served pack at `model_dir` (`config.plugin` set) and returns the shell that forwards through it.
pub fn load(io: std.Io, a: std.mem.Allocator, config: *ModelConfig, model_dir: []const u8, s: mlx.mlx_stream) !Transformer {
    return loadWith(io, a, config, model_dir, s, null);
}

/// `load`; with `tower`, the vision tower's tensors load too and move into it (the engine's encoder reads them, the
/// plugin never sees them).
pub fn loadWith(io: std.Io, a: std.mem.Allocator, config: *ModelConfig, model_dir: []const u8, s: mlx.mlx_stream, tower: ?*model.Weights) !Transformer {
    const b = config.plugin orelse return error.NotPluginServed;
    if (lossyUnserved(config)) |what| {
        log.err("[plugin] {s}: {s} is not served on the plugin path yet; turn the model's {s} setting off\n", .{ b.table.name.slice(), what, what });
        return error.PluginServedLossySetting;
    }
    var res = Resident.adopt(try loadSplit(io, a, model_dir, config, tower));
    // The model holds its own references: the map's go once the plugin has read it.
    defer res.deinit();
    const src = res.source();
    var fs = try FsPackDir.open(io, model_dir);
    defer fs.close();
    const dir = fs.packDir();
    var why: abi.Verdict = .{};
    // A placeholder only the tokenizer names (RADIO's `<image>`) comes from the engine config, which read it.
    if (b.parsed.caps.images and b.parsed.placeholder_ids.image == 0) b.parsed.placeholder_ids.image = config.image_token_id;
    var settings_buf: [64]u8 = undefined;
    const m = b.table.load(&.{
        .model_dir = abi.Str.of(model_dir),
        .dir = &dir,
        .parsed = &b.parsed,
        .settings_json = abi.Str.of(loadSettings(&settings_buf, config)),
        .stream = .{ .ctx = s.ctx },
        .weights = &src,
    }, &why) orelse {
        log.err("[plugin] {s}: load failed: {s}\n", .{ b.table.name.slice(), why.why.slice() });
        return if (std.mem.eql(u8, why.why.slice(), "MissingWeight")) error.MissingWeight else error.PluginLoadFailed;
    };
    errdefer b.table.unload(m);
    var xfm = try moduleShell(a, config.*, s);
    errdefer xfm.deinit();
    xfm.cache.deinit();
    xfm.cache = try KVCache.init(a, config.num_hidden_layers);
    // A drafter embedding through the trunk's table re-applies its scale (`Transformer.emb_scale`, freed by deinit).
    // bf16, as the legacy load builds it: an f32 scalar promotes every bf16 row it multiplies.
    if (b.parsed.spec.embed_scale > 0) xfm.emb_scale = bf16Scalar(b.parsed.spec.embed_scale, s);
    // The shell's own context's recurrent entries (a slot brings its own), freed by `Transformer.deinit`.
    if (config.needsSsmEntries()) {
        const entries = try a.alloc(SSMCacheEntry, config.num_hidden_layers);
        for (entries) |*e| e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
        xfm.ssm_entries = entries;
    }
    const caps = b.parsed.caps;
    // Drafters and an MTP head read the trunk's table and head.
    const has_head = if (b.table.spec) |sp| sp.head != null else false;
    const trunk = (caps.dflash_capture or has_head) and adoptTrunkWeights(&xfm, b.table, m);
    if (trunk) try adoptRotations(&xfm, b.table, m);
    intree.lendTrunk(&xfm);
    xfm.dense_attn_proj = @import("../load/init.zig").denseAttnProj(&xfm.config, xfm.config.num_experts > 0, &res.weights);
    // The engine's row-exact sampling reads coverage off the config, measured over the pack as the legacy load does.
    if (trunk and xfm.config.rowExactArch()) {
        const gather_only: Array = if (xfm.lm_head_w.ctx == xfm.emb_w.ctx) .{ .ctx = null } else xfm.emb_w;
        xfm.config.row_exact_covered = firstRowExactMiss(.of(&xfm.config), &res.weights, gather_only) == null;
    }
    const bound = try a.create(Bound);
    bound.* = .{ .table = b.table, .model = m, .drafters = caps.dflash_capture and trunk, .trunk = trunk };
    bind(bound, &b.parsed);
    xfm.plugin = bound;
    log.info("[plugin] {s} serves {s} ({d} layers)\n", .{ b.table.name.slice(), config.model_type, config.num_hidden_layers });
    return xfm;
}

/// Pooled embedding rows through the plugin's projection head (`EmbedHooks.project`), f32; a new reference.
pub fn embedProject(b: *const Bound, pooled: Array) !Array {
    const project = (b.table.embed orelse return error.NoEmbedProjection).project orelse return error.NoEmbedProjection;
    var out: abi.Array = .{};
    const v = project(b.model, .{ .ctx = pooled.ctx }, &out);
    if (!v.isOk()) {
        log.err("[plugin] {s}: projection failed: {s}\n", .{ b.table.name.slice(), v.why.slice() });
        return error.PluginProjectionFailed;
    }
    return .{ .ctx = out.ctx };
}

/// The trunk's embedding table and head, borrowed into the shell's `emb_*` / `lm_head_*` fields as the legacy shell
/// borrows them from its weights map, where the drafters read them; false when the table has no `trunk_weight` or it
/// refuses one (the model then serves without them).
fn adoptTrunkWeights(xfm: *Transformer, table: *const abi.ModelTable, m: *anyopaque) bool {
    const sp = table.spec orelse return false;
    const get = sp.trunk_weight orelse return false;
    const Dst = struct { which: abi.TrunkWeight, w: *Array, s: *Array, b: *Array };
    for ([_]Dst{
        .{ .which = .embed, .w = &xfm.emb_w, .s = &xfm.emb_s, .b = &xfm.emb_b },
        .{ .which = .head, .w = &xfm.lm_head_w, .s = &xfm.lm_head_s, .b = &xfm.lm_head_b },
    }) |d| {
        var w: quant_abi.Weight = .{};
        const v = get(m, d.which, &w);
        if (!v.isOk()) {
            log.warn("[plugin] {s}: trunk {t} refused ({s}): served without drafters\n", .{ table.name.slice(), d.which, v.why.slice() });
            for ([_]*Array{ &xfm.emb_w, &xfm.emb_s, &xfm.emb_b, &xfm.lm_head_w, &xfm.lm_head_s, &xfm.lm_head_b }) |x| x.* = .{ .ctx = null };
            return false;
        }
        for ([_]*Array{ d.w, d.s, d.b }, [_]abi.Array{ w.w, w.scales, w.biases }) |dst, src| dst.* = .{ .ctx = src.ctx };
    }
    return true;
}

/// A Hadamard pack's rotated table and head (`Weight.side`): the shell's registry binds their sign vectors, as the
/// legacy load's binds them, so the engine's readouts of a draft (`qmatmul` on the head) and its embedding
/// (`unrotateEmbedding`) take the rotated arm.
fn adoptRotations(xfm: *Transformer, table: *const abi.ModelTable, m: *anyopaque) !void {
    const get = table.spec.?.trunk_weight.?;
    for ([_]abi.TrunkWeight{ .embed, .head }, [_]Array{ xfm.emb_w, xfm.lm_head_w }) |which, w| {
        var tw: quant_abi.Weight = .{};
        if (!get(m, which, &tw).isOk() or tw.side.ctx == null) continue;
        const reg = xfm.rht orelse blk: {
            const r = try xfm.allocator.create(rht.Registry);
            r.* = .init(xfm.allocator, tw.block);
            xfm.rht = r;
            break :blk r;
        };
        const signs: Array = .{ .ctx = tw.side.ctx };
        const f = try rht.signsF32(signs, xfm.s);
        defer _ = mlx.mlx_array_free(f);
        try mlx.check(mlx.mlx_array_eval(f));
        try reg.register(w, signs, f);
        reg.bias_is_neg_scale = tw.flags & hadamard.NEG_SCALE != 0;
    }
}

pub fn unbind(b: *Bound, a: std.mem.Allocator) void {
    b.table.unload(b.model);
    a.destroy(b);
}

/// The first `ForwardCtx` field a served forward has no hook for; null when it serves them all.
fn unservedFeature(xfm: *const Transformer, ctx: *const ForwardCtx) ?[]const u8 {
    if (ctx.stream_in != null) return "a pipeline rank's stream input";
    if (ctx.prefill_mask_add != null) return "an additive prefill mask";
    if (ctx.tree != null) return "a draft tree";
    if (ctx.key_pad_mask != null and !xfm.embedding_mode) return "a key padding mask";
    if (ctx.capture_stream_all != null) return "the hyper-connection stream";
    if (ctx.head_place != null) return "an MTP head placement";
    if (ctx.batch_slots != null) return "a batched GDN tick";
    return null;
}

/// The slot's first position for a forward at `base`: an M-RoPE slot's carries its delta unless the chunk ropes from
/// the table's angles (`table`), which never reads it.
fn ropePosition(ctx: *const ForwardCtx, base: usize, table: bool) u32 {
    if (table) return @intCast(base);
    const delta: i64 = if (ctx.mrope_pos != null) ctx.mrope_delta else 0;
    return @intCast(@as(i64, @intCast(base)) + delta);
}

/// A prompt chunk's M-RoPE cos/sin off the slot's position table, stacked `[2, 1, 1, rows, rope_dims]` in the
/// activation dtype (`Forward.mrope`); empty past the table, where the 1-D position carries the delta.
fn mropeAngles(xfm: *const Transformer, ctx: *const ForwardCtx, base: usize, rows: u32) !Array {
    if (ctx.mrope_pos == null or rows <= 1 or base + rows > ctx.mrope_total) return .{ .ctx = null };
    const m = &xfm.config.plugin.?.parsed.mrope;
    const rd: usize = m.rope_dims;
    const inv = m.inv_freq.slice();
    if (rd == 0 or inv.len != rd / 2) return error.PluginForwardUnsupported;
    const a = xfm.allocator;
    const sel = try a.alloc(u8, inv.len);
    defer a.free(sel);
    mrope.interleavedSelector(sel, m.sections);
    const buf = try a.alloc(f32, 2 * rows * rd);
    defer a.free(buf);
    mrope.fillCosSin(buf[0 .. rows * rd], buf[rows * rd ..], mropeContext(ctx), base, 1, rows, inv, sel, rd, m.mscale);
    const f = mlx.mlx_array_new_data(buf.ptr, &[_]c_int{ 2, 1, 1, @intCast(rows), @intCast(rd) }, 5, .float32);
    defer _ = mlx.mlx_array_free(f);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, f, xfm.actDtype(), xfm.s));
    return out;
}

/// The engine's eval cadence for a forward of `rows` over `total_kv` keys (the legacy dense forward's).
fn evalEvery(cfg: *const ModelConfig, positions: Positions, rows: u32, total_kv: u64, kv_dequant: bool) u32 {
    if (!eval_cadence.prefillEvalCadenceApplies(@intCast(rows))) return 0;
    // A recurrent model's linear layers each hold a chunk-wide stream until the next point, as the admission bill counts.
    if (positions != .kv_step) return eval_cadence.prefillEvalCadenceFor(eval_cadence.PREFILL_EVAL_CADENCE_DEFAULT, cfg, rows, total_kv, kv_dequant);
    return eval_cadence.prefillEvalCadence(eval_cadence.EVAL_EVERY_N_LAYERS, cfg.head_dim, cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads, rows, total_kv, kv_dequant);
}

fn slotFor(xfm: *const Transformer, ctx: *const ForwardCtx) cache_ops.Slot {
    return .{ .kv = ctx.cache, .ssm = ctx.ssm_entries orelse &.{}, .packed_reads = ctx.kv_attn_fused, .batched_append = xfm.batched_kv_append_override orelse true };
}

fn freeOut(out: *const abi.ForwardOut, captured: []const abi.Array) void {
    for ([_]abi.Array{ out.logits, out.hidden }) |x| if (x.ctx != null) {
        _ = mlx.mlx_array_free(.{ .ctx = x.ctx });
    };
    for (captured) |x| if (x.ctx != null) {
        _ = mlx.mlx_array_free(.{ .ctx = x.ctx });
    };
}

/// `forwardWith` for a served model: logits `[1, rows, vocab]` as the legacy forward returns them (the final-normed
/// hidden instead under `skip_lm_head`), with the capture targets `ctx` names filled.
pub fn forward(xfm: *Transformer, b: *Bound, ctx: *ForwardCtx, token_ids: Array) !Array {
    if (unservedFeature(xfm, ctx)) |what| {
        log.err("[plugin] {s}: {s} is not served on the plugin path yet\n", .{ b.table.name.slice(), what });
        return error.PluginForwardUnsupported;
    }
    const sh = mlx.getShape(token_ids);
    if (sh.len != 2) return error.PluginForwardUnsupported;
    const rows: u32 = @intCast(sh[1]);
    if (xfm.embedding_mode and (sh[0] > 1 or ctx.key_pad_mask != null)) return forwardEncoderBatch(xfm, b, ctx, token_ids);
    if (sh[0] != 1) return error.PluginForwardUnsupported;
    var slot = slotFor(xfm, ctx);
    const handles = [_]abi.Handle{slot.handle()};
    const slot_ids = [_]u32{settle.slotId(b, ctx)};
    const base: usize = if (b.positions == .kv_step) ctx.cache.step else ctx.moe_seq_offset.*;
    const angles = try mropeAngles(xfm, ctx, base, rows);
    const pos = [_]u32{ropePosition(ctx, base, angles.ctx != null)};
    defer if (angles.ctx != null) {
        _ = mlx.mlx_array_free(angles);
    };
    // A capturing forward leaves each recurrent row's state for the partial-accept rollback (`Ctx.capture_spec`).
    const capture = ctx.capture_ssm_seq and ctx.ssm_entries != null;
    const head = !(ctx.skip_lm_head or xfm.embedding_mode);
    // Module-owned state (`Caps.owns_decode_state`): the cache is never appended to, its step is the position, and
    // only a verify reads more than the last row's logits.
    const module = if (xfm.config.plugin) |pb| pb.parsed.caps.owns_decode_state else false;
    const ids: []const u32 = if (ctx.capture_layers) |cl| cl.ids else &.{};
    var captured: [64]abi.Array = undefined;
    if (ids.len > captured.len) return error.PluginForwardUnsupported;
    @memset(captured[0..ids.len], .{});
    const f: abi.Forward = .{
        .mode = if (capture) .verify else if (rows == 1) .decode else if (ctx.spec_verify) .verify else .prefill,
        .row_exact = xfm.config.dflash_bound,
        .logits = if (!head) .none else if (module and !capture and !ctx.spec_verify) .last_row else .all_rows,
        .want_hidden = !head or ctx.capture_hidden != null or ctx.capture_hidden_all != null,
        .rows = rows,
        .tokens = .{},
        .ids = .{ .ctx = token_ids.ctx },
        .slots = .of(&slot_ids),
        .positions = .of(&pos),
        .caches = .of(&handles),
        .capture_layers = .of(ids),
        .defer_host = ctx.ple_defer,
        .media = .{ .ctx = if (ctx.vision_embeddings) |ve| ve.ctx else null },
        .media_offset = @intCast(ctx.vision_splice_offset),
        .mrope = .{ .ctx = angles.ctx },
        .cache_ops = &cache_ops.ops,
        .eval_every = evalEvery(&xfm.config, b.positions, rows, base + rows, ctx.cache.config.scheme != .off),
    };
    var out: abi.ForwardOut = .{ .captured = &captured };
    intree.enter(ctx);
    const v = b.table.forward(b.model, &f, &out);
    intree.leave();
    if (!v.isOk()) {
        freeOut(&out, captured[0..ids.len]);
        log.err("[plugin] {s}: forward failed: {s}\n", .{ b.table.name.slice(), v.why.slice() });
        return error.PluginForwardFailed;
    }
    // A `.kv_step` slot advances it too, as the legacy layered forward did: MiMo's MTP rollback truncates from it.
    ctx.moe_seq_offset.* += rows;
    if (module) ctx.cache.step += rows;
    if (f.defer_host) ctx.plugin_owed = out.owed;
    const hidden: Array = .{ .ctx = out.hidden.ctx };
    defer if (hidden.ctx != null) {
        _ = mlx.mlx_array_free(hidden);
    };
    if (ctx.capture_layers) |cl| for (cl.out, captured[0..ids.len]) |*dst, x| {
        _ = mlx.mlx_array_set(dst, .{ .ctx = x.ctx });
        _ = mlx.mlx_array_free(.{ .ctx = x.ctx });
    };
    if (ctx.capture_hidden) |target| {
        const hs = mlx.getShape(hidden);
        var last = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice(&last, hidden, &[_]c_int{ 0, hs[1] - 1, 0 }, 3, &[_]c_int{ hs[0], hs[1], hs[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, xfm.s));
        _ = mlx.mlx_array_set(target, last);
        _ = mlx.mlx_array_free(last);
    }
    if (ctx.capture_hidden_all) |target| _ = mlx.mlx_array_set(target, hidden);
    if (!head) {
        var h = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&h, hidden);
        return h;
    }
    const logits: Array = .{ .ctx = out.logits.ctx };
    defer _ = mlx.mlx_array_free(logits);
    var shaped = mlx.mlx_array_new();
    const out_rows: c_int = if (f.logits == .last_row) 1 else @intCast(rows);
    try mlx.check(mlx.mlx_reshape(&shaped, logits, &[_]c_int{ 1, out_rows, -1 }, 3, xfm.s));
    return shaped;
}

/// Whether an embedding batch may reach the forward padded: the legacy forward and an encoder always, a decoder only
/// where its plugin runs one (`served.decoder_embedders`); any other served decoder embeds one sequence per forward.
pub fn batchesEmbeddings(config: *const model.ModelConfig) bool {
    const pb = config.plugin orelse return true;
    return pb.parsed.caps.embed or served.servesDecoderEmbedding(config.model_type);
}

/// A padded embedding batch `[B, T]` under its key mask: one encoder forward, no cache, the hidden rows `[B, T, H]`.
fn forwardEncoderBatch(xfm: *Transformer, b: *Bound, ctx: *ForwardCtx, token_ids: Array) !Array {
    const sh = mlx.getShape(token_ids);
    const n: usize = @intCast(sh[0]);
    const rows: u32 = @intCast(sh[1]);
    if (n > MAX_BATCH_ROWS) return error.PluginForwardUnsupported;
    const zeros: [MAX_BATCH_ROWS]u32 = @splat(0);
    var mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask);
    if (ctx.key_pad_mask) |m| try mlx.check(mlx.mlx_reshape(&mask, m, &[_]c_int{ sh[0], sh[1] }, 2, xfm.s));
    const f: abi.Forward = .{
        .mode = .prefill,
        .logits = .none,
        .want_hidden = true,
        .rows = rows,
        .tokens = .{},
        .ids = .{ .ctx = token_ids.ctx },
        .positions = .of(zeros[0..n]),
        .key_mask = .{ .ctx = if (ctx.key_pad_mask != null) mask.ctx else null },
        .media = .{ .ctx = if (ctx.vision_embeddings) |ve| ve.ctx else null },
        .eval_every = evalEvery(&xfm.config, b.positions, rows, rows, false),
    };
    var out: abi.ForwardOut = .{};
    const v = b.table.forward(b.model, &f, &out);
    if (!v.isOk()) {
        freeOut(&out, &.{});
        log.err("[plugin] {s}: embedding batch failed: {s}\n", .{ b.table.name.slice(), v.why.slice() });
        return error.PluginForwardFailed;
    }
    if (out.logits.ctx != null) _ = mlx.mlx_array_free(.{ .ctx = out.logits.ctx });
    return .{ .ctx = out.hidden.ctx };
}

/// The prompt tokens a served model's own conversation state already holds (`CacheHooks.resume_prompt`), at most all
/// but the last (a resumed prefill still forwards a token); null when the model keeps no such state.
pub fn resumePrompt(xfm: *const Transformer, prompt: []const u32, media: ?Array) ?u64 {
    const b = xfm.plugin orelse return null;
    const f = b.table.cache.resume_prompt orelse return null;
    const n = f(b.model, .of(prompt), .{ .ctx = if (media) |m| m.ctx else null });
    return @min(n, prompt.len -| 1);
}

/// Bytes a served model's own conversation state holds now (`CacheHooks.state_bytes`); 0 without one.
pub fn stateBytes(xfm: *const Transformer) u64 {
    const b = xfm.plugin orelse return 0;
    const f = b.table.cache.state_bytes orelse return 0;
    return f(b.model);
}

/// `forwardBatchedDecode` / `forwardMoeBatchedDecode` for a served model: one row per slot at its own position, each
/// slot's logits `[1, 1, vocab]` and, when `hidden_rows` is asked, its final-normed hidden `[1, 1, hidden]` (caller
/// owns the slices and every array). The tick advances the slots' positions.
pub fn forwardBatched(xfm: *Transformer, b: *Bound, next_tokens: []const u32, ctxs: []const *ForwardCtx, rope_offsets: []const u32, hidden_rows: ?*?[]Array) ![]Array {
    const n = next_tokens.len;
    if (n > MAX_BATCH_ROWS) return error.PluginForwardUnsupported;
    var slots: [MAX_BATCH_ROWS]cache_ops.Slot = undefined;
    var handles: [MAX_BATCH_ROWS]abi.Handle = undefined;
    var positions: [MAX_BATCH_ROWS]u32 = undefined;
    var slot_ids: [MAX_BATCH_ROWS]u32 = undefined;
    for (ctxs, 0..) |c, i| {
        if (unservedFeature(xfm, c)) |what| {
            log.err("[plugin] {s}: {s} is not served on the plugin path yet\n", .{ b.table.name.slice(), what });
            return error.PluginForwardUnsupported;
        }
        slots[i] = slotFor(xfm, c);
        handles[i] = slots[i].handle();
        positions[i] = ropePosition(c, rope_offsets[i], false);
        slot_ids[i] = settle.slotId(b, c);
    }
    const f: abi.Forward = .{
        .mode = .batched_rows,
        .want_hidden = hidden_rows != null,
        .row_exact = xfm.config.dflash_bound,
        .rows = 1,
        .tokens = .of(next_tokens),
        .slots = .of(slot_ids[0..n]),
        .positions = .of(positions[0..n]),
        .defer_host = settle.defersHost(xfm),
        .caches = .of(handles[0..n]),
        .cache_ops = &cache_ops.ops,
    };
    var out: abi.ForwardOut = .{};
    const v = b.table.forward(b.model, &f, &out);
    if (!v.isOk()) {
        freeOut(&out, &.{});
        log.err("[plugin] {s}: batched forward failed: {s}\n", .{ b.table.name.slice(), v.why.slice() });
        return error.PluginForwardFailed;
    }
    settleTick(xfm, ctxs, f.defer_host, out.owed) catch |e| {
        freeOut(&out, &.{});
        return e;
    };
    const hidden: Array = .{ .ctx = out.hidden.ctx };
    defer if (hidden.ctx != null) {
        _ = mlx.mlx_array_free(hidden);
    };
    const logits: Array = .{ .ctx = out.logits.ctx };
    defer _ = mlx.mlx_array_free(logits);
    if (hidden_rows) |dst| dst.* = try perSlot(xfm, hidden, n);
    return perSlot(xfm, logits, n);
}

/// `forwardMoeBatchedVerify` for a served model: every slot's `[t1, drafts…]` row of `token_arr` `[N, S]` at its own
/// position, each recurrent row's state captured for the slot's own rollback; logits `[N, S, vocab]`, the final-normed
/// hidden at every row (`hidden_all` `[N, S, hidden]`) and at the last (`hidden_last` `[N, 1, hidden]`).
pub fn forwardBatchedVerify(xfm: *Transformer, b: *Bound, token_arr: Array, ctxs: []const *ForwardCtx, rope_offsets: []const u32, hidden_last: *Array, hidden_all: *Array) !Array {
    const n = ctxs.len;
    const width: c_int = mlx.getShape(token_arr)[1];
    if (n > MAX_BATCH_ROWS) return error.PluginForwardUnsupported;
    var slots: [MAX_BATCH_ROWS]cache_ops.Slot = undefined;
    var handles: [MAX_BATCH_ROWS]abi.Handle = undefined;
    var positions: [MAX_BATCH_ROWS]u32 = undefined;
    var slot_ids: [MAX_BATCH_ROWS]u32 = undefined;
    for (ctxs, 0..) |c, i| {
        if (unservedFeature(xfm, c)) |what| {
            log.err("[plugin] {s}: {s} is not served on the plugin path yet\n", .{ b.table.name.slice(), what });
            return error.PluginForwardUnsupported;
        }
        slots[i] = slotFor(xfm, c);
        handles[i] = slots[i].handle();
        positions[i] = ropePosition(c, rope_offsets[i], false);
        slot_ids[i] = settle.slotId(b, c);
    }
    const f: abi.Forward = .{
        .mode = .batched_rows,
        .logits = .all_rows,
        .want_hidden = true,
        .row_exact = xfm.config.dflash_bound,
        .rows = @intCast(width),
        .tokens = .{},
        .ids = .{ .ctx = token_arr.ctx },
        .slots = .of(slot_ids[0..n]),
        .positions = .of(positions[0..n]),
        .defer_host = settle.defersHost(xfm),
        .caches = .of(handles[0..n]),
        .cache_ops = &cache_ops.ops,
    };
    var out: abi.ForwardOut = .{};
    const v = b.table.forward(b.model, &f, &out);
    if (!v.isOk()) {
        freeOut(&out, &.{});
        log.err("[plugin] {s}: batched verify failed: {s}\n", .{ b.table.name.slice(), v.why.slice() });
        return error.PluginForwardFailed;
    }
    settleTick(xfm, ctxs, f.defer_host, out.owed) catch |e| {
        freeOut(&out, &.{});
        return e;
    };
    const hidden: Array = .{ .ctx = out.hidden.ctx };
    defer _ = mlx.mlx_array_free(hidden);
    const logits: Array = .{ .ctx = out.logits.ctx };
    defer _ = mlx.mlx_array_free(logits);
    const nc: c_int = @intCast(n);
    try mlx.check(mlx.mlx_reshape(hidden_all, hidden, &[_]c_int{ nc, width, -1 }, 3, xfm.s));
    const h = mlx.getShape(hidden_all.*)[2];
    try mlx.check(mlx.mlx_slice(hidden_last, hidden_all.*, &[_]c_int{ 0, width - 1, 0 }, 3, &[_]c_int{ nc, width, h }, 3, &[_]c_int{ 1, 1, 1 }, 3, xfm.s));
    var shaped = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&shaped, logits, &[_]c_int{ nc, width, -1 }, 3, xfm.s));
    return shaped;
}

/// A batched tick settles what its slots were left owed before it returns, in one call (the legacy tick's one sync).
fn settleTick(xfm: *Transformer, ctxs: []const *ForwardCtx, deferred: bool, owed: abi.Owed) !void {
    if (!deferred) return;
    for (ctxs) |c| c.plugin_owed = owed;
    settle.flushGroup(xfm, ctxs) catch |e| {
        for (ctxs) |c| settle.discard(xfm, c);
        return e;
    };
}

/// Row `i` of `[n, ..., width]` as an owned `[1, 1, width]` array, for every row.
fn perSlot(xfm: *Transformer, x: Array, n: usize) ![]Array {
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, x, &[_]c_int{ @intCast(n), -1 }, 2, xfm.s));
    const width = mlx.getShape(flat)[1];
    const rows = try xfm.allocator.alloc(Array, n);
    var made: usize = 0;
    errdefer {
        for (rows[0..made]) |r| _ = mlx.mlx_array_free(r);
        xfm.allocator.free(rows);
    }
    for (rows, 0..) |*r, i| {
        var row = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice(&row, flat, &[_]c_int{ @intCast(i), 0 }, 2, &[_]c_int{ @intCast(i + 1), width }, 2, &[_]c_int{ 1, 1 }, 2, xfm.s));
        defer _ = mlx.mlx_array_free(row);
        r.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(r, row, &[_]c_int{ 1, 1, -1 }, 3, xfm.s));
        made += 1;
    }
    return rows;
}
