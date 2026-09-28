//! Probability-ordered DFlash2 trees and a model-local measured cost policy.
//! Calibration data: TensorFold beddbb7 (MIT, attribution in NOTICE), fitted
//! on Qwen3.8-27B + z-lab DFlash2. Other models use exp(path score).
const std = @import("std");
const dflash = @import("dflash.zig");

pub const MAX_NODES = 15;
pub fn nodeBudget(proposal: u32) usize {
    const raw = std.c.getenv("MLX_SERVE_DFLASH_TREE_NODES") orelse return @min(proposal -| 1, MAX_NODES);
    // Zero is a correctness control: verify only the pending token, using the
    // identical target kernels/state path as a full tree, without speculation.
    return @min(std.fmt.parseInt(usize, std.mem.span(raw), 10) catch MAX_NODES, MAX_NODES);
}
pub fn allocationEnabled() bool {
    const raw = std.c.getenv("MLX_SERVE_DFLASH_TREE_ALLOCATE") orelse return false;
    return !std.mem.eql(u8, std.mem.span(raw), "0");
}
const Calibration = struct {
    depth_edges: [6]u32,
    score_edges: [13]f32,
    table: [6][14]f32,

    fn probability(self: Calibration, depth: u32, score: f32) f32 {
        var row: usize = 0;
        for (self.depth_edges, 0..) |edge, i| if (depth >= edge) {
            row = i;
        };
        var col: usize = 0;
        while (col < self.score_edges.len and score > self.score_edges[col]) : (col += 1) {}
        return self.table[row][col];
    }
};
const Data = struct { tables: struct { greedy: Calibration, sampled: Calibration } };
// Parse once on the host; the source metadata remains in the shipped fixture.
var calibration: ?Data = null;
fn fitted() !Data {
    if (calibration) |v| return v;
    const parsed = try std.json.parseFromSlice(Data, std.heap.page_allocator, @embedFile("dflash2_calibration.json"), .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    calibration = parsed.value;
    return parsed.value;
}

pub fn proposalBlock(config: u32, requested: u32, explicit: bool, wide: bool) u32 {
    // Larger proposals are available explicitly; the current target kernels
    // do not yet amortize sixteen rows well enough to widen by default.
    return if (explicit) @min(@max(requested, 2), MAX_NODES + 1) else @min(@max(config, 2), if (wide) MAX_NODES + 1 else @as(u32, 8));
}

/// Reorder in place, most likely eligible node first; every prefix is a tree.
/// The returned marginal probabilities are capped by their ancestors.
pub fn order(tree: *dflash.DraftTree, sampled: bool, use_fit: bool) ![MAX_NODES]f32 {
    const n = tree.tokens.len;
    std.debug.assert(n <= MAX_NODES);
    const fit = if (use_fit) try fitted() else null;
    var probabilities: [MAX_NODES]f32 = @splat(0);
    var tok: [MAX_NODES]u32 = undefined;
    var par: [MAX_NODES]i32 = undefined;
    var dep: [MAX_NODES]u32 = undefined;
    var score: [MAX_NODES]f32 = undefined;
    var place: [MAX_NODES]i32 = @splat(-1);
    for (0..n) |i| {
        const raw = if (fit) |f| (if (sampled) f.tables.sampled else f.tables.greedy).probability(tree.depth[i], tree.scores[i]) else @exp(tree.scores[i]);
        probabilities[i] = std.math.clamp(raw, 0, if (tree.parents[i] < 0) @as(f32, 1) else probabilities[@intCast(tree.parents[i])]);
    }
    var result: [MAX_NODES]f32 = @splat(0);
    for (0..n) |j| {
        var best: ?usize = null;
        for (0..n) |i| {
            if (place[i] >= 0 or (tree.parents[i] >= 0 and place[@intCast(tree.parents[i])] < 0)) continue;
            if (best == null or probabilities[i] > probabilities[best.?]) best = i;
        }
        const i = best orelse return error.InvalidDraftTree;
        place[i] = @intCast(j);
        tok[j] = tree.tokens[i];
        dep[j] = tree.depth[i];
        score[j] = tree.scores[i];
        par[j] = if (tree.parents[i] < 0) -1 else place[@intCast(tree.parents[i])];
        result[j] = probabilities[i];
    }
    @memcpy(tree.tokens, tok[0..n]);
    @memcpy(tree.parents, par[0..n]);
    @memcpy(tree.depth, dep[0..n]);
    @memcpy(tree.scores, score[0..n]);
    return result;
}

/// Full round costs are isolated from MTP/PLD and never persisted. The proposal
/// geometry must be fixed for a model instance; only the verified prefix varies.
pub const Costs = struct {
    const Cell = struct { n: u32 = 0, ms: f32 = 0 };
    const Bucket = struct { cells: [16]Cell = @splat(.{}), rounds: u32 = 0 };
    buckets: [8]Bucket = @splat(.{}),
    proposal: u32 = 0,

    fn bucket(kv: usize) usize {
        return @min(kv / 2048, 7);
    }

    pub fn resetFor(self: *Costs, proposal: u32) void {
        if (self.proposal != proposal) self.* = .{ .proposal = proposal };
    }

    pub fn observe(self: *Costs, kv: usize, nodes: usize, ms: f32, solo: bool) void {
        if (!solo or nodes > MAX_NODES or !std.math.isFinite(ms) or ms <= 0) return;
        const b = &self.buckets[bucket(kv)];
        const c = &b.cells[nodes];
        // First visit can compile a new shape. Do not price that as steady state.
        if (c.n == 1) c.ms = ms else if (c.n > 1) c.ms += 0.2 * (ms - c.ms);
        c.n +|= 1;
        b.rounds +|= 1;
    }

    pub fn choose(self: *const Costs, kv: usize, p: []const f32) usize {
        const cap = p.len;
        if (cap == 0) return 0;
        const b = &self.buckets[bucket(kv)];
        // Bootstrap wide then half-width; later explore one new prefix in a
        // short consecutive block. Unknown costs never masquerade as free.
        const half = @min(cap, 7);
        for ([_]usize{ cap, half }) |w| if (b.cells[w].n < 3) return w;
        if (b.rounds >= 32 and b.rounds % 32 < 3) {
            const probes = [_]usize{ cap, half, @min(cap, 3), @min(cap, 1), 0 };
            return probes[(b.rounds / 32 - 1) % probes.len];
        }
        var expected: f32 = 1;
        var best_rate: f32 = -1;
        var best = cap;
        for (0..cap + 1) |w| {
            if (w > 0) expected += p[w - 1];
            const c = b.cells[w];
            if (c.n < 2 or c.ms <= 0) continue;
            const rate = expected / c.ms;
            if (rate > best_rate) {
                best_rate = rate;
                best = w;
            }
        }
        return best;
    }
};

test "dflash tree allocation: calibration boundaries and model-specific fallback" {
    const f = try fitted();
    try std.testing.expectEqual(@as(f32, 0.0372), f.tables.greedy.probability(0, -6));
    try std.testing.expectEqual(@as(f32, 0.9626), f.tables.greedy.probability(0, -0.05));
    try std.testing.expectEqual(@as(f32, 0.9242), f.tables.sampled.probability(15, 0));
    try std.testing.expectEqual(@as(u32, 8), proposalBlock(8, 8, false, true));
    try std.testing.expectEqual(@as(u32, 8), proposalBlock(8, 8, true, true));
    try std.testing.expectEqual(@as(u32, 16), proposalBlock(8, 32, true, true));
}

test "dflash tree allocation: probability ordering preserves ancestry and selects across a cost cliff" {
    var toks = [_]u32{ 10, 20, 30, 11 };
    var parents = [_]i32{ -1, 0, 1, -1 };
    var depths = [_]u32{ 0, 1, 2, 0 };
    var scores = [_]f32{ -0.1, -2, -3, -0.2 };
    var tree = dflash.DraftTree{ .tokens = &toks, .parents = &parents, .depth = &depths, .scores = &scores };
    const p = try order(&tree, false, false);
    try std.testing.expectEqualSlices(u32, &.{ 10, 11, 20, 30 }, &toks);
    try std.testing.expectEqualSlices(i32, &.{ -1, -1, 0, 2 }, &parents);
    for (1..4) |i| try std.testing.expect(p[i] <= p[i - 1]);
    var costs = Costs{};
    for (0..3) |_| {
        costs.observe(100, 4, 100, true);
        costs.observe(100, 2, 10, true);
        costs.observe(100, 0, 9, true);
    }
    try std.testing.expectEqual(@as(usize, 2), costs.choose(100, p[0..4]));
    // This request's probabilities matter, not historic accepted-token counts.
    try std.testing.expectEqual(@as(usize, 0), costs.choose(100, &.{ 0.01, 0.001, 0.0001, 0.00001 }));
    try std.testing.expectEqual(@as(usize, 4), costs.choose(9000, p[0..4]));
    costs.resetFor(16);
    try std.testing.expectEqual(@as(usize, 4), costs.choose(100, p[0..4]));
}
