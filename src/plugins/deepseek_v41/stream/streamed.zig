//! DeepSeek-V4.1's EXL3 repack experts for the resident trunk: `Exl3` serves a bank the EXL3 quant claims on the
//! stream module's own expert path; the trunk has no EXL3 math, so the source returns the routed sum.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const ops = @import("../../../sdk/weights/streamed/ops.zig");
const xp = @import("../../../sdk/weights/streamed/experts.zig");
const arm_mod = @import("arm.zig");
const v41 = @import("deepseek_v41.zig");
const xk = @import("../../../sdk/quant/exl3/stream/kernels.zig");
const xq = @import("../../../sdk/quant/exl3/stream/quant.zig");
const expert_bank = @import("../../../sdk/weights/streamed/expert/bank.zig");
const kernel_set = @import("../../../sdk/weights/streamed/sdk_ext.zig").kernels.KernelSet(xk);
const G = ops.MlxOps;

const T = mlx.mlx_array;

/// The stream module's expert path (`module.zig` `A` without the lookahead, which only starts reads early) over an EXL3
/// bank in `dir` beside its `config.json`: every expert of a layer resident in the prompt rows, never grown.
pub const Exl3 = struct {
    pub const Arm = arm_mod.ArmWith(G, xp.QuantMath(G, xq.Accepted(G)), .{ .prefill = true });

    pub const Options = struct {
        /// The geometry the bank lane decodes (the release's by default).
        implemented: expert_bank.Implemented = expert_bank.dsv41,
    };

    a: std.mem.Allocator,
    g: G,
    set: *kernel_set.Set,
    q: *xq.Accepted(G),
    arm: *Arm,

    pub fn open(a: std.mem.Allocator, io: std.Io, dir: []const u8, s: mlx.mlx_stream, opt: Options) !*Exl3 {
        var diag: arm_mod.Diag = .{};
        errdefer std.log.err("streamed exl3: {s}", .{diag.message()});
        const c = try v41.Config.load(a, io, dir, &diag);
        const self = try a.create(Exl3);
        errdefer a.destroy(self);
        self.a = a;
        self.g = try G.init(a, s);
        errdefer self.g.deinit();
        self.set = try kernel_set.Set.init(a, .{ .device = .{ .stream = s } }, &diag);
        errdefer self.set.deinit();
        self.set.install(G, &self.g);
        errdefer kernel_set.Set.uninstall(G, &self.g);
        // The quant as `Module.acceptKernels` accepts it.
        self.q = try xq.accept(G, a, &self.g, .{ .kernels = self.set.ref() }, .{
            .hidden = c.hidden_size,
            .inter = c.moe_intermediate_size,
            .top_k = c.n_experts_per_tok,
            .n_layers = c.n_layers,
            .act = .{ .swiglu_clamped = c.swiglu_limit },
            .input = .bfloat16,
        }, &diag);
        errdefer self.q.deinit(&self.g);
        const all: arm_mod.NativeRows = .{ .prefill = c.n_routed_experts, .decode = c.n_routed_experts };
        self.arm = try Arm.initHooked(a, io, &self.g, self.q, .{
            .model_dir = dir,
            .implemented = opt.implemented,
            .baseline_bytes = null,
            .native_rows = all,
            .preallocate = true,
            .slot_memory = .{ .mlx = s },
        }, .{}, &diag);
        return self;
    }

    pub fn deinit(self: *Exl3) void {
        _ = mlx.mlx_synchronize(self.g.s);
        self.arm.deinit();
        self.q.deinit(&self.g);
        kernel_set.Set.uninstall(G, &self.g);
        self.set.deinit();
        self.g.deinit();
        self.a.destroy(self);
    }

    /// Model layer `layer`'s routed sum `[n, hidden]` f32 for the normed MoE input `x` `[n, hidden]`, the routes `idx`
    /// `[n, k]` and their weights `wts`: the hook's unweighted rows, then `graph.moeCombine` without the shared add.
    pub fn rows(self: *Exl3, layer: u32, x: T, idx: T, wts: T) !T {
        const g = &self.g;
        defer g.reset();
        const ids = if (g.dtypeOf(idx) == .int32) idx else try g.astype(idx, .int32);
        const ro = try self.arm.hook.at(layer).routed(g, x, ids);
        const w = if (g.dtypeOf(wts) == .float32) wts else try g.astype(wts, .float32);
        return g.keep(try g.sum(try g.mul(try g.astype(ro, .float32), try g.expandDims(w, -1)), -2, false));
    }

    /// After the eval that consumed the forward's calls.
    pub fn settle(self: *Exl3) !void {
        try self.arm.hook.flush();
    }
};
