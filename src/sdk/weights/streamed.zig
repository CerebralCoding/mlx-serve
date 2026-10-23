//! The streamed weight source: routed experts read from an SSD bank (`experts.bin` + `expert-manifest.json`) into slot
//! rows, for a trunk whose MoE reads them through `Routed`. The streamer is mlx-stream's (`streamed/expert/`,
//! `streamed/sdk_ext/expert/`, the read pool in `streamed/csrc/`; module `mlx_stream`). Banks: DeepSeek-V4.1's MXFP4
//! experts and its EXL3 repack (`plugins/deepseek_v41/stream/streamed.zig`). Without mlx-stream, or without a bank,
//! opening refuses by name.

const std = @import("std");
const build_options = @import("build_options");
const abi = @import("../../abi/quant.zig");
const mlx = @import("../../mlx/ffi.zig");
const plugin = if (build_options.mlx_stream) @import("mlx_stream") else struct {};

/// One projection's slot-bank arrays as the trunk's gathers read them (MXFP4: words and E8M0 scales).
pub const Proj = struct { w: mlx.mlx_array, scales: mlx.mlx_array };
pub const Banks = struct { gate: Proj, up: Proj, down: Proj };
/// A call's slots: per routed pair (the router's flat order) its bank (prompt rows, grown rows, transient rows) and its
/// row there, and each bank's arrays. Valid until the next fetch.
pub const Fetched = struct { kind: []const u8, row: []const u32, banks: [3]?Banks };
/// Routed pairs one call takes at most.
pub const max_call_ids: usize = 1 << 14;

/// How a trunk's MoE reads a streamed bank, by whose math runs it. `trunk`: the source fetches, `fetch` making model
/// layer `layer`'s routed experts `ids` resident and naming each pair's slot, and the trunk runs its own gathers on
/// them, so its weighted rows are its own op order. `source`: a format the trunk has no math for (EXL3), the source's
/// routed sum `[n, hidden]` f32 for the normed input, the routes and their weights. `settle` after the eval that
/// consumed a forward's calls.
pub const Routed = struct {
    ctx: *anyopaque,
    math: union(enum) {
        trunk: *const fn (*anyopaque, u32, []const u16) anyerror!Fetched,
        source: *const fn (*anyopaque, u32, mlx.mlx_array, mlx.mlx_array, mlx.mlx_array) anyerror!mlx.mlx_array,
    },
    settle_fn: *const fn (*anyopaque) anyerror!void,

    pub fn settle(r: Routed) !void {
        return r.settle_fn(r.ctx);
    }
};

pub const Streamed = struct {
    pub const refusal = abi.Verdict.refuse(.unsupported, "streamed weights: no expert bank, or a build without mlx-stream");

    pub const Options = struct {
        /// Slot rows per routed layer (null: every expert).
        rows: ?[]const u32 = null,
    };

    const Source = if (build_options.mlx_stream) plugin.dsv41_streamed.Source else void;

    src: Source,

    pub fn open(a: std.mem.Allocator, io: std.Io, dir: []const u8, s: mlx.mlx_stream, opt: Options) !Streamed {
        if (comptime !build_options.mlx_stream) return error.StreamedWeightsUnsupported;
        var d = std.Io.Dir.openDirAbsolute(io, dir, .{}) catch return error.StreamedWeightsUnsupported;
        defer d.close(io);
        _ = d.statFile(io, "expert-manifest.json", .{}) catch return error.StreamedWeightsUnsupported;
        return .{ .src = try Source.open(a, io, dir, s, .{ .rows = opt.rows }) };
    }

    pub fn close(self: *Streamed) void {
        if (comptime build_options.mlx_stream) self.src.deinit();
    }

    pub fn routedSource(self: *Streamed) Routed {
        if (comptime !build_options.mlx_stream) unreachable;
        const Mxfp4 = plugin.dsv41_streamed.Mxfp4;
        const Exl3 = plugin.dsv41_streamed.Exl3;
        const C = struct {
            fn fetch(ctx: *anyopaque, layer: u32, ids: []const u16) anyerror!Fetched {
                const src: *Mxfp4 = @ptrCast(@alignCast(ctx));
                const f = try src.fetch(layer, ids);
                var banks: [3]?Banks = @splat(null);
                for (f.banks, &banks) |b, *o| if (b) |x| {
                    o.* = .{ .gate = .{ .w = x.gate.w, .scales = x.gate.scales }, .up = .{ .w = x.up.w, .scales = x.up.scales }, .down = .{ .w = x.down.w, .scales = x.down.scales } };
                };
                return .{ .kind = f.kind, .row = f.row, .banks = banks };
            }
            fn rows(ctx: *anyopaque, layer: u32, x: mlx.mlx_array, idx: mlx.mlx_array, wts: mlx.mlx_array) anyerror!mlx.mlx_array {
                const src: *Exl3 = @ptrCast(@alignCast(ctx));
                return src.rows(layer, x, idx, wts);
            }
            fn settle(comptime S: type) fn (*anyopaque) anyerror!void {
                return struct {
                    fn f(ctx: *anyopaque) anyerror!void {
                        const src: *S = @ptrCast(@alignCast(ctx));
                        return src.settle();
                    }
                }.f;
            }
        };
        return switch (self.src) {
            .mxfp4 => |x| .{ .ctx = x, .math = .{ .trunk = C.fetch }, .settle_fn = C.settle(Mxfp4) },
            .exl3 => |x| .{ .ctx = x, .math = .{ .source = C.rows }, .settle_fn = C.settle(Exl3) },
        };
    }
};

// ── tests ──

test "streamed: a directory without a bank is refused by name" {
    try std.testing.expectError(error.StreamedWeightsUnsupported, Streamed.open(std.testing.allocator, std.testing.io, "/nowhere", .{ .ctx = null }, .{}));
    try std.testing.expectEqual(abi.Verdict.Code.unsupported, Streamed.refusal.code);
}
