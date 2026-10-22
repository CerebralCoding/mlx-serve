//! plugins/deepseek_v4/bridge, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns forwardDsv4WithImpl, initDsv4.

const std = @import("std");
const dsv4_mod = @import("deepseek_v4.zig");
const mlx = @import("../../mlx/ffi.zig");
const ModelConfig = @import("../../engines/mlx/model.zig").ModelConfig;
const Weights = @import("../../engines/mlx/model.zig").Weights;
const ForwardCtx = @import("../../engines/mlx/forward_ctx.zig").ForwardCtx;
const moduleShell = @import("../../engines/mlx/modules/shell.zig").moduleShell;
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;

/// DeepSeek-V4-Flash v0 forward: full re-forward over the request's whole
/// token history each call (correctness mode — the module's incremental
/// decode replaces this behind the same seam). SERIAL-ONLY: history lives on
/// the model; a fresh request is detected by cache.step == 0. Returns
/// last-position logits [1, V] f32 and advances cache.step like the standard
/// forwards do.
pub fn forwardDsv4WithImpl(self: *Transformer, ctx: *ForwardCtx, token_ids: mlx.mlx_array, mdl: *dsv4_mod.Dsv4Model) !mlx.mlx_array {
    const n = mlx.mlx_array_size(token_ids);
    var ids32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids32);
    try mlx.check(mlx.mlx_astype(&ids32, token_ids, .int32, self.s));
    // Lazy pipelined decode: the sampled token id and the returned logits
    // both stay GPU arrays, so generate.zig's pipelined next() overlaps the
    // next build with GPU execution (the sync path below force-evals the id
    // and host-copies the logits — a full pipeline drain per token).
    if (ctx.cache.step != 0 and n == 1 and mdl.dec_state != null and
        dsv4_mod.lazyDecodeReady(mdl, &mdl.dec_state.?))
    {
        const lazy = try dsv4_mod.decodeStepLazy(mdl, self.allocator, &mdl.dec_state.?, ids32);
        defer _ = mlx.mlx_array_free(lazy);
        ctx.cache.step += 1;
        const shape3 = [_]c_int{ 1, 1, @intCast(mlx.mlx_array_size(lazy)) };
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_reshape(&out, lazy, &shape3, 3, self.s));
        return out;
    }
    try mlx.check(mlx.mlx_array_eval(ids32));
    const data = mlx.mlx_array_data_int32(ids32) orelse return error.NoData;
    var logits_host: []f32 = undefined;
    if (ctx.cache.step == 0) {
        // fresh request: rebuild the incremental state and prefill into it
        if (mdl.dec_state) |*ds| dsv4_mod.deinitDecodeState(ds);
        mdl.dec_state = try dsv4_mod.initDecodeState(mdl, self.allocator);
        const ids = try self.allocator.alloc(u32, n);
        defer self.allocator.free(ids);
        for (ids, data[0..n]) |*o, id| o.* = @intCast(id);
        logits_host = try dsv4_mod.prefillIntoState(mdl, self.allocator, &mdl.dec_state.?, ids);
    } else if (n == 1) {
        // decode: one incremental step — decode==prefill equivalence is
        // pinned by the DSV4_MINI test.
        logits_host = try dsv4_mod.decodeStep(mdl, self.allocator, &mdl.dec_state.?, @intCast(data[0]));
    } else {
        // later prefill chunk: batched state extension (same path as the
        // fresh prefill — extendState is base-position aware).
        const ids = try self.allocator.alloc(u32, n);
        defer self.allocator.free(ids);
        for (ids, data[0..n]) |*o, id| o.* = @intCast(id);
        logits_host = try dsv4_mod.extendState(mdl, self.allocator, &mdl.dec_state.?, ids);
    }
    defer self.allocator.free(logits_host);
    ctx.cache.step += n;
    // Callers slice the last position out of rank-3 [B, L, V] logits.
    const shape = [_]c_int{ 1, 1, @intCast(logits_host.len) };
    return mlx.mlx_array_new_data(logits_host.ptr, &shape, 3, .float32);
}

pub fn initDsv4(allocator: std.mem.Allocator, config: ModelConfig, weights: *const Weights, s: mlx.mlx_stream) !Transformer {
    const dw = try dsv4_mod.loadDsv4Weights(allocator, &config, weights);
    const mdl = try allocator.create(dsv4_mod.Dsv4Model);
    errdefer allocator.destroy(mdl);
    mdl.* = try dsv4_mod.initModel(allocator, &config, dw, s);
    var t = try moduleShell(allocator, config, s);
    t.dsv4 = mdl;
    return t;
}
