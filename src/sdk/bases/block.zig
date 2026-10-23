//! The block interface the bases compose. A block is a type naming its `kind`, its `Weights`, the function its kind
//! requires and `supports_batched_rows`; `check` fails the build with the reason when a block does not. `Ctx` is
//! what every block function sees of one forward.

const std = @import("std");
const ops = @import("../../mlx/ops.zig");
const abi = @import("../../abi/plugin.zig");

pub const Array = ops.Array;
pub const Graph = ops.Graph;
pub const Mode = abi.Mode;

/// What a block does in the skeleton, and so which function it declares:
/// `embed`: `forward(w, c, ids) !Array` (ids `[slots, rows]` int32);
/// `norm`, `mixer`, `mlp`, `head`: `forward(w, c, x) !Array`;
/// `seam`: `mid(c, h, delta, comptime Norm, nw) !Mid` and `end(c, h, delta, comptime Norm, ?nw) !End`.
pub const Kind = enum { embed, norm, mixer, mlp, seam, head };

/// Per-model values blocks share, built once at load.
pub const Shared = struct {
    /// The RMS norms' epsilon, and the same as a 0-d f32 array (the fused residual kernels read it).
    eps: f32,
    eps_arr: Array,
    /// The post-attention / post-MLP norms' epsilon where the model separates it; `eps` elsewhere.
    post_eps: f32,
    post_eps_arr: Array,

    pub fn init(eps: f32, post_eps: f32) Shared {
        const mlx = @import("../../mlx/ffi.zig");
        return .{ .eps = eps, .eps_arr = mlx.mlx_array_new_float(eps), .post_eps = post_eps, .post_eps_arr = mlx.mlx_array_new_float(post_eps) };
    }

    pub fn deinit(self: *Shared) void {
        const mlx = @import("../../mlx/ffi.zig");
        _ = mlx.mlx_array_free(self.eps_arr);
        _ = mlx.mlx_array_free(self.post_eps_arr);
    }
};

/// One forward as a block sees it. `g` is the current layer's graph: arrays made in it die with the layer unless
/// the base keeps them.
pub const Ctx = struct {
    g: *Graph,
    mode: Mode,
    /// Tokens per slot in this forward.
    rows: c_int,
    /// Slots in this forward: 1 except `batched_rows`.
    slots: c_int = 1,
    /// Each slot's first position.
    positions: []const u32,
    /// Each slot's cache, read through `cache_ops`.
    caches: []const abi.Handle = &.{},
    cache_ops: ?*const abi.CacheOps = null,
    layer: u32 = 0,
    shared: *const Shared,
    /// An additive key mask `[B, 1, 1, T]` (a padded encoder batch); null = none.
    mask: Array = .{},
    /// A prompt chunk's M-RoPE `cos`/`sin` rows, roped by in place of `positions`; null = 1-D positions (which
    /// carry the slot's M-RoPE delta).
    mrope: ?[2]Array = null,

    pub fn stream(c: *const Ctx) ops.Stream {
        return c.g.s;
    }
    /// The solo slot's first position.
    pub fn offset(c: *const Ctx) u32 {
        return c.positions[0];
    }
    /// More than one row per slot: prompt chunks and verify rows (the legacy `is_prefill`).
    pub fn wide(c: *const Ctx) bool {
        return c.rows > 1;
    }
    pub fn cache(c: *const Ctx) abi.Handle {
        return c.caches[0];
    }
    pub fn cacheOps(c: *const Ctx) !*const abi.CacheOps {
        return c.cache_ops orelse error.NoCacheOps;
    }
};

/// A seam's middle: the new residual and the next block's normed input.
pub const Mid = struct { h: Array, normed: Array };
/// A seam's end: the new residual, and the next layer's normed input when the seam produced it.
pub const End = struct { h: Array, normed: ?Array = null };

fn fnName(comptime kind: Kind) []const u8 {
    return switch (kind) {
        .seam => "end",
        else => "forward",
    };
}

/// Why `B` is not a block of one of `kinds`, or null when it is.
pub fn reason(comptime B: type, comptime kinds: []const Kind) ?[]const u8 {
    if (@typeInfo(B) != .@"struct") return "is not a struct";
    if (!@hasDecl(B, "kind") or @TypeOf(B.kind) != Kind) return "declares no `kind: Kind`";
    const ok = for (kinds) |k| {
        if (B.kind == k) break true;
    } else false;
    if (!ok) return "is a ." ++ @tagName(B.kind) ++ " block, not one this slot takes";
    if (!@hasDecl(B, "supports_batched_rows") or @TypeOf(B.supports_batched_rows) != bool) return "declares no `supports_batched_rows: bool`";
    if (!@hasDecl(B, "Weights") or @TypeOf(B.Weights) != type) return "declares no `Weights` type";
    const f = comptime fnName(B.kind);
    if (!@hasDecl(B, f)) return "declares no `" ++ f ++ "`";
    if (B.kind == .seam and !@hasDecl(B, "mid")) return "declares no `mid`";
    if (B.kind != .seam) {
        const info = @typeInfo(@TypeOf(@field(B, f))).@"fn";
        const ps = info.param_types;
        if (ps.len != 3) return "`forward` takes (w, c, x)";
        if (ps[0] != *const B.Weights) return "`forward`'s first parameter is not `*const Weights`";
        if (ps[1] != *Ctx) return "`forward`'s second parameter is not `*Ctx`";
        if (ps[2] != Array) return "`forward`'s third parameter is not `Array`";
        const ret = @typeInfo(info.return_type.?);
        if (ret != .error_union or ret.error_union.payload != Array) return "`forward` does not return `!Array`";
    }
    return null;
}

/// Fails the build, naming the block and the reason, unless `B` is a block of one of `kinds`.
pub fn check(comptime B: type, comptime kinds: []const Kind) void {
    if (comptime reason(B, kinds)) |why| @compileError(@typeName(B) ++ " " ++ why);
}

/// Whether every block in `blocks` (a tuple of types) serves `batched_rows`.
pub fn allBatch(comptime blocks: anytype) bool {
    inline for (blocks) |B| if (!B.supports_batched_rows) return false;
    return true;
}

/// One of two blocks per layer (a mixer or an MLP each): a switch, never an indirect call. Nest for more kinds.
pub fn Either(comptime A: type, comptime B: type) type {
    check(A, &.{ .mixer, .mlp });
    check(B, &.{ .mixer, .mlp });
    return struct {
        pub const kind: Kind = .mixer;
        pub const supports_batched_rows = A.supports_batched_rows and B.supports_batched_rows;
        pub const Weights = union(enum) { first: A.Weights, second: B.Weights };
        pub const First = A;
        pub const Second = B;

        pub fn forward(w: *const Weights, c: *Ctx, x: Array) !Array {
            return switch (w.*) {
                .first => |*a| A.forward(a, c, x),
                .second => |*b| B.forward(b, c, x),
            };
        }

        /// The active block's `fold` (hybrid base), null where it declares none.
        pub fn fold(w: *const Weights, c: *Ctx, normed: Array, h: Array, next_norm: Array) !?End {
            return switch (w.*) {
                .first => |*a| if (@hasDecl(A, "fold")) A.fold(a, c, normed, h, next_norm) else null,
                .second => |*b| if (@hasDecl(B, "fold")) B.fold(b, c, normed, h, next_norm) else null,
            };
        }
    };
}

// ── tests ──

const testing = std.testing;

const Good = struct {
    pub const kind: Kind = .mlp;
    pub const supports_batched_rows = true;
    pub const Weights = struct { w: Array = .{} };
    pub fn forward(_: *const Weights, _: *Ctx, x: Array) !Array {
        return x;
    }
};

const Serial = struct {
    pub const kind: Kind = .mixer;
    pub const supports_batched_rows = false;
    pub const Weights = struct {};
    pub fn forward(_: *const Weights, _: *Ctx, x: Array) !Array {
        return x;
    }
};

test "bases block: a well-formed block passes; each missing piece is named" {
    try testing.expectEqual(@as(?[]const u8, null), comptime reason(Good, &.{.mlp}));
    try testing.expectEqualStrings("is a .mlp block, not one this slot takes", comptime reason(Good, &.{.norm}).?);
    const NoKind = struct {
        pub const Weights = struct {};
    };
    try testing.expectEqualStrings("declares no `kind: Kind`", comptime reason(NoKind, &.{.mlp}).?);
    const NoBatch = struct {
        pub const kind: Kind = .mlp;
        pub const Weights = struct {};
    };
    try testing.expectEqualStrings("declares no `supports_batched_rows: bool`", comptime reason(NoBatch, &.{.mlp}).?);
    const NoFn = struct {
        pub const kind: Kind = .norm;
        pub const supports_batched_rows = true;
        pub const Weights = Array;
    };
    try testing.expectEqualStrings("declares no `forward`", comptime reason(NoFn, &.{.norm}).?);
    const BadSig = struct {
        pub const kind: Kind = .norm;
        pub const supports_batched_rows = true;
        pub const Weights = Array;
        pub fn forward(_: *const Weights, x: Array) !Array {
            return x;
        }
    };
    try testing.expectEqualStrings("`forward` takes (w, c, x)", comptime reason(BadSig, &.{.norm}).?);
    const NoMid = struct {
        pub const kind: Kind = .seam;
        pub const supports_batched_rows = true;
        pub const Weights = void;
        pub fn end() void {}
    };
    try testing.expectEqualStrings("declares no `mid`", comptime reason(NoMid, &.{.seam}).?);
}

test "bases block: batched rows hold only when every block serves them; Either carries both" {
    try testing.expect(comptime allBatch(.{ Good, Good }));
    try testing.expect(!comptime allBatch(.{ Good, Serial }));
    const E = Either(Good, Serial);
    try testing.expect(!E.supports_batched_rows and E.kind == .mixer);
    try testing.expectEqual(@as(?[]const u8, null), comptime reason(E, &.{.mixer}));
    try testing.expect(Either(Good, Good).supports_batched_rows);
}
