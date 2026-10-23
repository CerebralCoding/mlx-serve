//! The loaded model both KDA-hybrid plugins run their forwards on: the engine's layered `Transformer` built from the
//! pack's tensors as the legacy load builds it (prefix probe, layered weights, the compiled MoE router and GDN gate),
//! owning the map. Its shared helpers (projections, KDA, MoE, embedding, head) are what the
//! table forwards call; per-request state lives in the slots' caches, never here.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const model = @import("../../engines/mlx/model.zig");
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;
const intree = @import("../../engines/mlx/plugin/intree.zig");
const quant_abi = @import("../../abi/quant.zig");
const Layout = @import("../../sdk/quant/native.zig").Layout;
const config = @import("config.zig");
const glm_mtp = @import("mtp.zig");
const toModelConfig = @import("model_config.zig").toModelConfig;

const gpa = std.heap.c_allocator;

pub const Model = struct {
    weights: model.Weights,
    xfm: Transformer,
    caps: abi.Caps,
    /// The MTP head (GLM's `mtp.0`), once `HeadHooks.load` found one.
    head: ?*glm_mtp.Head = null,
};

/// The map the driver already loaded, adopted; else the pack read from disk.
fn weightsFor(ctx: *const abi.LoadCtx, cfg: *model.ModelConfig) !model.Weights {
    if (intree.takeWeights(ctx)) |w| return w;
    return model.loadModelWeights(std.Io.Threaded.global_single_threaded.io(), gpa, ctx.model_dir.slice(), cfg, false);
}

fn build(ctx: *const abi.LoadCtx) !*Model {
    const io = std.Io.Threaded.global_single_threaded.io();
    const m = try gpa.create(Model);
    errdefer gpa.destroy(m);
    m.head = null;
    var cfg = try toModelConfig(config.of(ctx.parsed));
    m.weights = try weightsFor(ctx, &cfg);
    errdefer m.weights.deinit();
    model.resolveWeightPrefix(&cfg, &m.weights);
    try model.narrowHadamardPackTables(&cfg, &m.weights, mlx.gpuStream());
    m.xfm = try Transformer.init(io, gpa, cfg, &m.weights);
    errdefer m.xfm.deinit();
    if (!mlx.noGpuBackend()) {
        if (m.xfm.moe_layers != null) m.xfm.compileMoeRouting();
        if (cfg.linear_num_key_heads > 0) m.xfm.compileGdnGate();
    }
    m.caps = ctx.parsed.caps;
    return m;
}

pub fn load(comptime name: []const u8, ctx: *const abi.LoadCtx, out: *abi.Verdict) ?*anyopaque {
    return build(ctx) catch |e| {
        out.* = switch (e) {
            error.OutOfMemory => abi.Verdict.refuse(.failed, name ++ ": out of memory"),
            // The driver maps this name to the engine's own refusal.
            error.MissingWeight => abi.Verdict.refuse(.failed, "MissingWeight"),
            error.UnsupportedWeightSource => abi.Verdict.refuse(.unsupported, name ++ ": a weight source other than the resident map"),
            else => abi.Verdict.refuse(.failed, name ++ ": the pack does not load"),
        };
        return null;
    };
}

pub fn unload(mp: *anyopaque) callconv(.c) void {
    const m: *Model = @ptrCast(@alignCast(mp));
    if (m.head) |h| {
        h.deinit();
        gpa.destroy(h);
    }
    m.xfm.deinit();
    m.weights.deinit();
    gpa.destroy(m);
}

pub fn of(mp: *anyopaque) *Model {
    return @ptrCast(@alignCast(mp));
}

/// `SpecHooks.trunk_weight`: the loaded embedding table or head, borrowed (a tied head is the table).
pub fn trunkWeight(mp: *anyopaque, which: abi.TrunkWeight, out: *quant_abi.Weight) callconv(.c) abi.Verdict {
    const x = &of(mp).xfm;
    const w, const sc, const b = switch (which) {
        .embed => .{ x.emb_w, x.emb_s, x.emb_b },
        .head => .{ x.lm_head_w, x.lm_head_s, x.lm_head_b },
    };
    if (w.ctx == null) return abi.Verdict.refuse(.unsupported, "the trunk has no such weight");
    out.* = .{ .w = .{ .ctx = w.ctx }, .scales = .{ .ctx = sc.ctx }, .biases = .{ .ctx = b.ctx } };
    if (sc.ctx == null) return .ok;
    const qp = x.quantParamsFor(w, sc);
    out.bits = @intCast(qp.bits);
    out.group_size = qp.group_size;
    out.layout = @backingInt(switch (qp.mode) {
        .nvfp4 => Layout.nvfp4,
        .mxfp4 => Layout.mxfp4,
        .mxfp8 => Layout.mxfp8,
        else => Layout.affine,
    });
    return .ok;
}

/// The forward's token ids `[1, rows]`: the engine's device array, or the host tokens as int32.
pub fn tokenIds(f: *const abi.Forward) !Array {
    if (f.ids.ctx != null) {
        var r = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&r, .{ .ctx = f.ids.ctx });
        return r;
    }
    return hostIds(f.tokens.slice());
}

/// Host tokens as an int32 `[1, n]` window (MLX copies them).
pub fn hostIds(tokens: []const u32) !Array {
    const buf = try gpa.alloc(i32, tokens.len);
    defer gpa.free(buf);
    for (tokens, buf) |t, *b| b.* = std.math.cast(i32, t) orelse return error.TokenOutOfRange;
    return mlx.mlx_array_new_data(buf.ptr, &[_]c_int{ 1, @intCast(tokens.len) }, 2, .int32);
}

const Array = mlx.mlx_array;

/// Fills `out` from the final-normed hidden `[1, S, H]` (consumed): the hidden when asked, and the logits the
/// forward wants (`[S, V]` or the last row `[1, V]`).
pub fn finish(m: *Model, f: *const abi.Forward, hidden: Array, out: *abi.ForwardOut) !void {
    defer _ = mlx.mlx_array_free(hidden);
    if (f.want_hidden) {
        var h = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&h, hidden);
        out.hidden = .{ .ctx = h.ctx };
    }
    if (f.logits == .none) return;
    const s = m.xfm.s;
    const logits = try m.xfm.lmHeadProject(hidden);
    defer _ = mlx.mlx_array_free(logits);
    const sh = mlx.getShape(logits);
    var rows = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(rows);
    if (f.logits == .all_rows) {
        try mlx.check(mlx.mlx_reshape(&rows, logits, &[_]c_int{ sh[1], sh[2] }, 2, s));
    } else {
        var last = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(last);
        try mlx.check(mlx.mlx_slice(&last, logits, &[_]c_int{ 0, sh[1] - 1, 0 }, 3, &[_]c_int{ 1, sh[1], sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        try mlx.check(mlx.mlx_reshape(&rows, last, &[_]c_int{ 1, sh[2] }, 2, s));
    }
    out.logits = .{ .ctx = rows.ctx };
}

/// Frees what a failed forward left in `out`.
pub fn clear(out: *abi.ForwardOut, captured: usize) void {
    for ([_]*abi.Array{ &out.logits, &out.hidden }) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(.{ .ctx = a.ctx });
        a.* = .{};
    };
    if (out.captured) |c| for (c[0..captured]) |*a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(.{ .ctx = a.ctx });
        a.* = .{};
    };
}

/// One slot's rows through `fwd.hidden` (a table forward module), its head and captures into `out`.
pub fn forwardOne(comptime name: []const u8, comptime fwd: type, m: *Model, ops: *const abi.CacheOps, f: *const abi.Forward, out: *abi.ForwardOut) abi.Verdict {
    if (f.caches.len != 1 or f.positions.len != 1) return abi.Verdict.refuse(.unsupported, name ++ ": one slot's rows at a time");
    if (f.capture_layers.len > 0 and out.captured == null) return abi.Verdict.refuse(.failed, name ++ ": capture layers without an output");
    if (f.media.ctx != null and !@hasField(fwd.Step, "media")) return abi.Verdict.refuse(.unsupported, name ++ ": no media rows");
    one(fwd, m, ops, f, out) catch |e| {
        clear(out, f.capture_layers.len);
        return abi.Verdict.named(.failed, @errorName(e));
    };
    return .ok;
}

fn one(comptime fwd: type, m: *Model, ops: *const abi.CacheOps, f: *const abi.Forward, out: *abi.ForwardOut) !void {
    const ids = try tokenIds(f);
    defer _ = mlx.mlx_array_free(ids);
    var caps: [64]Array = undefined;
    const n_cap = f.capture_layers.len;
    if (n_cap > caps.len) return error.TooManyCaptureLayers;
    for (caps[0..n_cap]) |*c| c.* = mlx.mlx_array_new();
    defer for (caps[0..n_cap]) |c| {
        _ = mlx.mlx_array_free(c);
    };
    var st: fwd.Step = .{
        .slots = .{ .ops = ops, .caches = f.caches.slice(), .positions = f.positions.slice() },
        // A verify's rows capture their recurrent states for a partial accept's rollback.
        .capture = f.mode == .verify,
        .capture_ids = f.capture_layers.slice(),
        .capture_out = caps[0..n_cap],
        .eval_every = f.eval_every,
    };
    if (@hasField(fwd.Step, "media")) {
        st.media = .{ .ctx = f.media.ctx };
        st.media_offset = f.media_offset;
    }
    try finish(m, f, try fwd.hidden(&m.xfm, &st, ids), out);
    for (caps[0..n_cap], 0..) |c, i| {
        var r = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&r, c);
        out.captured.?[i] = .{ .ctx = r.ctx };
    }
}
