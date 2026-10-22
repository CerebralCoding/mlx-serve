//! Inference-thread start-up context and load-error plumbing.

const std = @import("std");
const gguf_meta = @import("../registry/engine_pick.zig");

const Scheduler = @import("../../engines/mlx/scheduler.zig").Scheduler;
const LoadParams = @import("../../engines/mlx/scheduler/params.zig").LoadParams;

pub const ThreadCtx = struct {
    scheduler: *Scheduler,
    params: LoadParams,
};

/// Signal to the parent waiting in `init()` that the inference thread is done
/// with its load (or has failed). After `started` flips, the parent reads
/// `load_failed` to decide whether to surface a startup error.
pub fn signalStarted(sch: *Scheduler) void {
    sch.started_mu.lockUncancelable(sch.io);
    defer sch.started_mu.unlock(sch.io);
    sch.started.store(true, .release);
    sch.started_cond.broadcast(sch.io);
}
/// Set the load_error_name + flip load_failed. Best-effort dupe; on OOM the
/// parent still sees `load_failed=true` and surfaces "unknown".
pub fn recordLoadError(sch: *Scheduler, err_name: []const u8) void {
    if (sch.load_error_name) |old| sch.allocator.free(old);
    sch.load_error_name = sch.allocator.dupe(u8, err_name) catch null;
    sch.load_failed.store(true, .release);
}

/// Heap-allocate `T`, run `init_fn`, return owning pointer. On `init_fn`
/// failure, the heap slot is freed before the error propagates so the
/// scheduler never holds a half-initialized struct.
fn boxInit(
    allocator: std.mem.Allocator,
    comptime T: type,
    init_fn: anytype,
    args: anytype,
) !*T {
    const ptr = try allocator.create(T);
    errdefer allocator.destroy(ptr);
    ptr.* = try @call(.auto, init_fn, args);
    return ptr;
}

/// Routing decision for a GGUF entry resolved at preload time: the actual
/// `.gguf` file (owned by the conn thread, freed after the load completes —
/// the engines dupe/copy what they keep) and which embedded engine serves it.
pub const GgufRoute = struct {
    path: []u8,
    engine: gguf_meta.Engine,
};
