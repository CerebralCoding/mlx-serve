//! The streamed weight source: routed experts read from an SSD bank (`experts.bin` + `expert-manifest.json`) into slot
//! rows, for a trunk whose MoE reads them through `Routed`. The streamer is mlx-stream's (`streamed/expert/`,
//! `streamed/sdk_ext/expert/`, the read pool in `streamed/csrc/`; module `mlx_stream`). Bank: the MXFP4 record bank
//! (`streamed/bank_source.zig`; MiMo's and DeepSeek-V4.1's experts).
//! Without mlx-stream, or without a bank, opening refuses by name.

const std = @import("std");
const build_options = @import("build_options");
const abi = @import("../../abi/quant.zig");
const mlx = @import("../../mlx/ffi.zig");
const plugin = if (build_options.mlx_stream) @import("mlx_stream") else struct {};

const routed = @import("routed.zig");
pub const Proj = routed.Proj;
pub const Banks = routed.Banks;
pub const Fetched = routed.Fetched;
pub const max_call_ids = routed.max_call_ids;
pub const Routed = routed.Routed;

pub const Streamed = struct {
    pub const refusal = abi.Verdict.refuse(.unsupported, "streamed weights: no expert bank, or a build without mlx-stream");

    pub const Options = struct {
        /// Slot rows per routed layer (null: every expert).
        rows: ?[]const u32 = null,
    };

    const Source = if (build_options.mlx_stream) plugin.bank_source.Source else void;

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
        const Mxfp4 = plugin.bank_source.Mxfp4;
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
            fn settle(ctx: *anyopaque) anyerror!void {
                const src: *Mxfp4 = @ptrCast(@alignCast(ctx));
                return src.settle();
            }
        };
        return switch (self.src) {
            .mxfp4 => |x| .{ .ctx = x, .math = .{ .trunk = C.fetch }, .settle_fn = C.settle },
        };
    }
};

// ── tests ──

test "streamed: a directory without a bank is refused by name" {
    try std.testing.expectError(error.StreamedWeightsUnsupported, Streamed.open(std.testing.allocator, std.testing.io, "/nowhere", .{ .ctx = null }, .{}));
    try std.testing.expectEqual(abi.Verdict.Code.unsupported, Streamed.refusal.code);
}
