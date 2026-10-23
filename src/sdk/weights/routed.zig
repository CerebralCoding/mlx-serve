//! The streamed-experts contract (`streamed.zig` `Streamed.routedSource`) a trunk's MoE reads, free of the streamer
//! itself so any MoE block can carry it.

const mlx = @import("../../mlx/ffi.zig");

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
