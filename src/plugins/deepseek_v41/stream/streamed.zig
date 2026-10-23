//! DeepSeek-V4.1's routed experts streamed from an `experts.bin` for the resident trunk (`sdk/weights/streamed.zig`
//! `Streamed`). The source fetches; the MoE math stays the trunk's. `Mxfp4` serves the MXFP4 expert bank format
//! (`mimo_v2/stream/bank.zig`, mlx-stream-mxfp4-expert-bank-v1: the release's FP4 experts as MLX's mxfp4 words and
//! E8M0 scales, six components a record) through the expert stream (`mimo_v2/stream/stream_source.zig`): a call's
//! experts made resident in slot rows, and each routed pair's bank and row. `Exl3` serves a bank the EXL3 quant claims
//! (the DSV4.1 EXL3 repack) on the stream module's own expert path: the trunk has no EXL3 math, so that source returns
//! the routed sum.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const ops = @import("ops.zig");
const bank_mod = @import("../../mimo_v2/stream/bank.zig");
const stream_mod = @import("../../mimo_v2/stream/stream.zig");
const xp = @import("experts.zig");
const arm_mod = @import("arm.zig");
const v41 = @import("deepseek_v41.zig");
const xk = @import("../../../sdk/quant/exl3/stream/kernels.zig");
const xq = @import("../../../sdk/quant/exl3/stream/quant.zig");
const expert_bank = @import("../../../sdk/weights/streamed/expert/bank.zig");
const kernel_set = @import("../../../sdk/weights/streamed/sdk_ext.zig").kernels.KernelSet(xk);
const G = ops.MlxOps;

const T = mlx.mlx_array;

pub const Geometry = bank_mod.Geometry;
/// A call's distinct experts fit one route.
pub const max_route_ids: usize = stream_mod.max_route_ids;
/// Routed pairs a call takes at most (`Fetched`'s buffers).
pub const max_call_ids: usize = 1 << 14;

pub const Proj = struct { w: T, scales: T };
pub const Banks = struct { gate: Proj, up: Proj, down: Proj };
/// One call's slots: per routed pair its bank (`BankKind` order: prompt rows, grown rows, transient rows) and row, and
/// each bank's arrays. Valid until the next `fetch`.
pub const Fetched = struct { kind: []const u8, row: []const u32, banks: [3]?Banks };

pub const Mxfp4 = struct {
    pub const Options = struct {
        /// Slot rows per routed layer (null: every expert, the whole bank resident once read).
        rows: ?[]const u32 = null,
    };

    a: std.mem.Allocator,
    bank: bank_mod.Bank,
    sdk_bank: bank_mod.SDKBank,
    stream: *stream_mod.Stream,
    source: stream_mod.StreamSource,
    g: ops.MlxOps,
    rows: []u32,
    kind: [max_call_ids]u8 = undefined,
    row: [max_call_ids]u32 = undefined,

    /// The bank in `dir` (`expert-manifest.json` + `experts.bin`) streamed on `s`, in the decode phase from the start.
    pub fn open(a: std.mem.Allocator, io: std.Io, dir: []const u8, s: mlx.mlx_stream, opt: Options) !*Mxfp4 {
        var bank = try bank_mod.Bank.open(a, io, dir);
        errdefer bank.deinit();
        const geo = bank.geometry;
        const rows = try a.alloc(u32, geo.num_moe_layers);
        errdefer a.free(rows);
        if (opt.rows) |r| {
            if (r.len != rows.len) return error.StreamedRows;
            for (r) |n| if (n == 0 or n > geo.num_experts) return error.StreamedRows;
            @memcpy(rows, r);
        } else @memset(rows, geo.num_experts);
        const self = try a.create(Mxfp4);
        errdefer a.destroy(self);
        self.* = .{ .a = a, .bank = bank, .sdk_bank = undefined, .stream = undefined, .source = undefined, .g = undefined, .rows = rows };
        self.sdk_bank = try bank_mod.SDKBank.init(a, &self.bank);
        errdefer self.sdk_bank.deinit();
        self.stream = try stream_mod.Stream.init(a, &self.sdk_bank, stream_mod.options(rows, .{ .mlx = s }));
        errdefer self.stream.deinit();
        self.source = stream_mod.StreamSource.init(self.stream);
        self.g = try ops.MlxOps.init(a, s);
        errdefer self.g.deinit();
        try self.source.grow(rows);
        return self;
    }

    pub fn deinit(self: *Mxfp4) void {
        _ = mlx.mlx_synchronize(self.g.s);
        self.g.deinit();
        self.stream.deinit();
        self.sdk_bank.deinit();
        self.bank.deinit();
        self.a.free(self.rows);
        self.a.destroy(self);
    }

    /// Model layer `layer`'s routed pairs `ids` (the router's flat order) made resident: one route over their distinct
    /// experts, its reads waited on the host. The released routes are unpinned first (`ids` came from an eval that
    /// consumed them); this call's slots stay pinned until the next `fetch` or `settle`.
    pub fn fetch(self: *Mxfp4, layer: u32, ids: []const u16) !Fetched {
        if (ids.len > max_call_ids) return error.StreamedCallTooWide;
        try self.source.flush();
        var distinct: [max_route_ids]u16 = undefined;
        var at: [1 << 16]u8 = undefined;
        var n: usize = 0;
        for (ids) |e| {
            if (std.mem.indexOfScalar(u16, distinct[0..n], e) != null) continue;
            if (n == max_route_ids) return error.StreamedCallTooWide;
            distinct[n] = e;
            n += 1;
        }
        const call = try self.source.route(layer, distinct[0..n], &.{});
        const sv = self.source.served(call);
        for (0..sv.n_parts) |p| {
            try self.source.waitGu(call, @intCast(p));
            try self.source.waitDown(call, @intCast(p));
        }
        for (distinct[0..n], 0..) |e, i| at[e] = @intCast(i);
        for (ids, 0..) |e, i| {
            const ref = sv.refs[at[e]];
            self.kind[i] = @intFromEnum(ref.bank);
            self.row[i] = ref.row;
        }
        self.source.release(call);
        var banks: [3]?Banks = @splat(null);
        for ([_]stream_mod.BankKind{ .base, .ext, .transient }, 0..) |k, b| if (try self.source.bankArrays(&self.g, layer, k)) |ba| {
            banks[b] = .{
                .gate = .{ .w = ba.gate.w, .scales = ba.gate.scales.? },
                .up = .{ .w = ba.up.w, .scales = ba.up.scales.? },
                .down = .{ .w = ba.down.w, .scales = ba.down.scales.? },
            };
        };
        return .{ .kind = self.kind[0..ids.len], .row = self.row[0..ids.len], .banks = banks };
    }

    /// After the eval that consumed the last call's slots.
    pub fn settle(self: *Mxfp4) !void {
        try self.source.flush();
    }

    pub fn stats(self: *Mxfp4) stream_mod.Stats {
        return self.source.stats();
    }
};

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

/// The bank in `dir`: EXL3 when the EXL3 quant claims its description, else MXFP4.
pub const Source = union(enum) {
    mxfp4: *Mxfp4,
    exl3: *Exl3,

    pub fn open(a: std.mem.Allocator, io: std.Io, dir: []const u8, s: mlx.mlx_stream, opt: Mxfp4.Options) !Source {
        if (expert_bank.peek(a, io, dir, null)) |p| {
            var pk = p;
            defer pk.deinit();
            if (xq.claims(&pk.view, null) != null) return .{ .exl3 = try Exl3.open(a, io, dir, s, .{}) };
        } else |_| {}
        return .{ .mxfp4 = try Mxfp4.open(a, io, dir, s, opt) };
    }

    pub fn deinit(self: Source) void {
        switch (self) {
            inline else => |x| x.deinit(),
        }
    }

    pub fn settle(self: Source) !void {
        switch (self) {
            inline else => |x| try x.settle(),
        }
    }
};

/// Writes the bank `Mxfp4` reads: `experts.bin` (`bytes`, every routed layer's records in order) and its manifest.
pub fn writeBank(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, geo: Geometry, bytes: []const u8, repository: []const u8, revision: []const u8) !void {
    if (bytes.len != geo.total_bytes) return error.StreamedBankBytes;
    try dir.writeFile(io, .{ .sub_path = "experts.bin", .data = bytes });
    var out: std.Io.Writer.Allocating = .init(a);
    defer out.deinit();
    const w = &out.writer;
    try w.print("{{\"format\":\"{s}\",\"data_file\":\"experts.bin\",\"hidden_size\":{d},\"intermediate_size\":{d},\"num_experts\":{d},\"first_moe_layer\":{d},\"num_moe_layers\":{d},\"record_bytes\":{d},\"gate_up_bytes\":{d},", .{ bank_mod.format, geo.hidden_size, geo.intermediate_size, geo.num_experts, geo.first_moe_layer, geo.num_moe_layers, geo.record_bytes, geo.gate_up_bytes });
    try w.print("\"quantization\":{{\"mode\":\"mxfp4\",\"bits\":4,\"group_size\":32}},\"source\":{{\"repository\":\"{s}\",\"revision\":\"{s}\"}},\"segments\":[", .{ repository, revision });
    for (geo.segments, 0..) |seg, i| {
        const c: bank_mod.Component = @fromBackingInt(@intCast(i));
        try w.print("{s}{{\"name\":\"{s}\",\"dtype\":\"{s}\",\"shape\":[{d},{d}],\"offset\":{d},\"length\":{d}}}", .{ if (i > 0) "," else "", c.name(), @tagName(seg.dtype), seg.shape[0], seg.shape[1], seg.offset, seg.length });
    }
    try w.writeAll("],\"records\":[");
    const count = @as(usize, geo.num_experts) * geo.num_moe_layers;
    for (0..count) |i| {
        const rec = bytes[i * geo.record_bytes ..][0..geo.record_bytes];
        try w.print("{s}{{\"layer\":{d},\"expert\":{d},\"offset\":{d},\"length\":{d},\"sha256\":\"{s}\"}}", .{ if (i > 0) "," else "", geo.first_moe_layer + i / geo.num_experts, i % geo.num_experts, i * geo.record_bytes, geo.record_bytes, &sha256(rec) });
    }
    try w.print("],\"output_files\":{{\"experts.bin\":{{\"size\":{d},\"sha256\":\"{s}\"}}}}}}", .{ bytes.len, &sha256(bytes) });
    try dir.writeFile(io, .{ .sub_path = "expert-manifest.json", .data = out.written() });
}

fn sha256(bytes: []const u8) [64]u8 {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &d, .{});
    return std.fmt.bytesToHex(d, .lower);
}

