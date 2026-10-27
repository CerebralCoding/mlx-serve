//! Small blocks the Gemma-style decoders compose: a block whose output is RMS-normed before the residual add
//! (`PostNormed`, the sandwich norms), the gated tanh- or erf-GELU MLP (`GatedGelu`), and the LM head with an output
//! multiplier and a logit softcap (`SoftcapHead`). Compiled closures are carried, never rebuilt, so served bits stay
//! the legacy's. copied_from: sdk/blocks/mlp/dense.zig (`residualTail`'s pre-norm, `standardMlp`),
//! sdk/blocks/mlp/activations.zig (`computeGeglu`'s gelu arms, `applySoftcap`), engines/mlx/forward/dense.zig (the
//! head's multiplier and softcap).

const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const block = @import("../block.zig");
const linear = @import("linear.zig");
const mlp = @import("mlp.zig");
const head = @import("head.zig");
const fused_act = @import("../../blocks/mlp/fused_activations.zig");
const FUSED_ROWS_MAX_M = @import("../../blocks/linear/fused_rows.zig").FUSED_ROWS_MAX_M;

const Array = ops.Array;
const Graph = ops.Graph;
const Ctx = block.Ctx;
const Linear = linear.Linear;
const Joined = linear.Joined;

/// `B` with its output RMS-normed at the post-norm epsilon (`Shared.post_eps`). The decoder hands that norm to the
/// seam (`Inner`), which folds it into its fused dispatch at decode, as the legacy did.
pub fn PostNormed(comptime B: type) type {
    block.check(B, &.{ .mixer, .mlp });
    return struct {
        pub const Inner = B;
        pub const kind = B.kind;
        pub const supports_batched_rows = B.supports_batched_rows;
        pub const Weights = struct { inner: B.Weights, norm: Array };

        pub fn forward(w: *const Weights, c: *Ctx, x: Array) !Array {
            return c.g.rmsNorm(try B.forward(&w.inner, c, x), w.norm, c.shared.post_eps);
        }
    };
}

pub const Gelu = enum { tanh, erf };

/// `down(gelu(gate(x)) * up(x))`: gate|up joined on a solo one-row step of a layer whose q|k|v group was built (the
/// legacy ties both to one flag). The product is one exact dispatch where it serves (tanh: the table, erf: the fused
/// erf kernel), else the compiled GeGLU, else the GELU (compiled or composed) and a multiply.
pub fn GatedGelu(comptime act: Gelu) type {
    return struct {
        pub const kind: block.Kind = .mlp;
        pub const supports_batched_rows = true;
        pub const Weights = struct {
            gate: Linear,
            up: Linear,
            down: Linear,
            gu: Joined = .{},
            rows_joined: bool = false,
            /// gelu_tanh over every 16-bit pattern of the activation dtype (the legacy `geluTanhTable`); null = none.
            table: Array = .{},
            geglu: mlx.mlx_closure = .{ .ctx = null },
            gelu: mlx.mlx_closure = .{ .ctx = null },
        };

        pub fn forward(weights: *const Weights, c: *Ctx, x: Array) !Array {
            var adapted = weights.*;
            adapted.gate = adapted.gate.withLora(c.lora, c.layer, .gate);
            adapted.up = adapted.up.withLora(c.lora, c.layer, .up);
            adapted.down = adapted.down.withLora(c.lora, c.layer, .down);
            const w = &adapted;
            const g = c.g;
            const joined = !g.differentiable and !w.gate.hasLora() and !w.up.hasLora() and c.mode != .batched_rows and w.rows_joined and w.gu.built() and c.rows * c.slots <= FUSED_ROWS_MAX_M;
            var gate: Array = undefined;
            var up: Array = undefined;
            if (joined) {
                const both = try w.gu.lin.forward(g, x);
                gate = try w.gu.part(g, both, 0);
                up = try w.gu.part(g, both, 1);
            } else {
                gate = try w.gate.forward(g, x);
                up = try w.up.forward(g, x);
            }
            return w.down.forward(g, try product(w, g, gate, up));
        }

        fn product(w: *const Weights, parent: *Graph, gate: Array, up: Array) !Array {
            var sc = parent.scope();
            defer sc.deinit();
            const g = &sc;
            const dt = ops.dtype(gate);
            if (g.differentiable and act == .tanh) return parent.keep(try g.mul(try mlp.geluTanh(g, gate), up));
            if (act == .tanh and w.table.ctx != null and fused_act.swigluFusedEnabled() and (dt == .bfloat16 or dt == .float16)) {
                if (try fused_act.tableGateMul(g.s, .gelu_tanh, gate, up, w.table, null)) |y| return parent.keep(try g.adopt(y));
            }
            if (w.geglu.ctx != null) return parent.keep(try apply(g, w.geglu, &.{ gate, up }));
            if (act == .erf) {
                const one = try g.scalar(1, .bfloat16);
                const half = try g.scalar(0.5, .bfloat16);
                if (try fused_act.fusedGeluErfMul(g.s, gate, up, one, half)) |y| return parent.keep(try g.adopt(y));
                return parent.keep(try g.mul(try g.adopt(try fused_act.geluErfChain(gate, one, half, g.s)), up));
            }
            const a = if (w.gelu.ctx != null) try apply(g, w.gelu, &.{gate}) else try mlp.geluTanh(g, gate);
            return parent.keep(try g.mul(a, up));
        }
    };
}

/// The Gemma MLP.
pub const GeGlu = GatedGelu(.tanh);

/// The LM head, then `logits * mult` (where set), then `tanh(logits / cap) * cap`.
pub const SoftcapHead = struct {
    pub const kind: block.Kind = .head;
    pub const supports_batched_rows = true;
    pub const Weights = struct {
        head: Linear,
        /// bf16 0-d cap; null = no softcap.
        cap: Array = .{},
        mult: Array = .{},
        compiled: mlx.mlx_closure = .{ .ctx = null },
    };

    pub fn forward(w: *const Weights, c: *Ctx, x: Array) !Array {
        return project(c.g, w, x);
    }

    pub fn project(g: *Graph, w: *const Weights, x: Array) !Array {
        var logits = try head.project(g, &w.head, x);
        if (w.mult.ctx != null) logits = try g.mul(logits, w.mult);
        if (w.cap.ctx == null) return logits;
        if (!g.differentiable and w.compiled.ctx != null) return apply(g, w.compiled, &.{logits});
        return g.mul(try g.tanh(try g.div(logits, w.cap)), w.cap);
    }
};

/// A compiled closure's first output, owned by `g`.
fn apply(g: *Graph, f: mlx.mlx_closure, args: []const Array) !Array {
    const in = mlx.mlx_vector_array_new_data(args.ptr, args.len);
    defer _ = mlx.mlx_vector_array_free(in);
    var out = mlx.mlx_vector_array{ .ctx = null };
    defer _ = mlx.mlx_vector_array_free(out);
    try mlx.check(mlx.mlx_closure_apply(&out, f, in));
    var r = mlx.mlx_array_new();
    mlx.check(mlx.mlx_vector_array_get(&r, out, 0)) catch |e| {
        _ = mlx.mlx_array_free(r);
        return e;
    };
    return g.adopt(r);
}

test "gemma4 LoRA: GeGLU applies each MLP adapter with an unfused differentiable product" {
    const std = @import("std");
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = Graph.init(std.testing.allocator, s);
    defer g.deinit();
    var shared = block.Shared.init(1e-6, 1e-6);
    defer shared.deinit();
    const x = try g.fromSlice(f32, &.{ 0.2, -0.3 }, &.{ 1, 1, 2 });
    const weight = try g.fromSlice(f32, &.{ 1, 0.5, -0.4, 0.8 }, &.{ 2, 2 });
    const w: GeGlu.Weights = .{ .gate = .{ .w = weight }, .up = .{ .w = weight }, .down = .{ .w = weight } };
    const a = try g.fromSlice(f32, &.{ 0.3, 0.7 }, &.{ 2, 1 });
    const b = try g.fromSlice(f32, &.{ 0.6, -0.1 }, &.{ 1, 2 });
    inline for (.{ .gate, .up, .down }) |projection| {
        const binding: @import("../../../abi/plugin.zig").LoraBinding = .{ .layer = 0, .projection = projection, .a = .{ .ctx = a.ctx }, .b = .{ .ctx = b.ctx }, .scale = 2 };
        var c: Ctx = .{ .g = &g, .mode = .prefill, .rows = 1, .positions = &.{0}, .shared = &shared, .lora = &.{binding} };
        var dense = w;
        @field(dense, @tagName(projection)).w = try g.add(weight, try g.mulScalar(try g.matmul(a, b), 2));
        const got = try GeGlu.forward(&w, &c, x);
        c.lora = &.{};
        const want = try GeGlu.forward(&dense, &c, x);
        const delta = try g.max(try g.reshape(try g.abs(try g.sub(got, want)), &.{-1}), 0, false);
        try std.testing.expect(try @import("../../train/grad.zig").item(delta) < 1e-6);
    }
}
