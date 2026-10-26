//! Bounded job history; workers pin a model and post all GPU work to its scheduler.
const std = @import("std");
const types = @import("types.zig");
const mlx = @import("../engines/mlx.zig");
const Scheduler = @import("../gpu_thread/scheduler.zig").Scheduler;
const GenRequest = @import("../gpu_thread/requests.zig").GenRequest;
const LoadedModel = @import("../registry/registry.zig").LoadedModel;
const now = @import("../../sdk/util/io_util.zig").nowMsMonotonic;
const verify = @import("../../sdk/train/verify.zig");

pub const Store = struct {
    a: std.mem.Allocator,
    io: std.Io,
    scheduler: *Scheduler,
    root: []const u8,
    mutex: std.Io.Mutex = .init,
    jobs: std.ArrayList(*Job) = .empty,
    next_id: u64 = 1,
    starting: bool = false,
    closing: bool = false,

    pub fn init(a: std.mem.Allocator, io: std.Io, scheduler: *Scheduler, root: []const u8) !Store {
        return .{ .a = a, .io = io, .scheduler = scheduler, .root = try a.dupe(u8, root) };
    }
    pub fn stop(self: *Store) void {
        self.mutex.lockUncancelable(self.io);
        self.closing = true;
        for (self.jobs.items) |j| j.cancelled.store(true, .release);
        self.mutex.unlock(self.io);
    }
    pub fn deinit(self: *Store) void {
        self.stop();
        for (self.jobs.items) |j| j.deinit();
        self.jobs.deinit(self.a);
        self.a.free(self.root);
    }
    pub fn start(self: *Store, body: []const u8, bill: *types.Plan, failure: *?[]const u8) !u64 {
        {
            self.mutex.lockUncancelable(self.io);
            defer self.mutex.unlock(self.io);
            if (self.closing) return error.Shutdown;
            if (self.starting) return error.FineTuneBusy;
            for (self.jobs.items) |j| if (!j.status.terminal()) return error.FineTuneBusy;
            self.starting = true;
        }
        defer {
            self.mutex.lockUncancelable(self.io);
            self.starting = false;
            self.mutex.unlock(self.io);
        }
        const parsed = try types.Request.parse(self.a, body);
        errdefer parsed.deinit();
        const path = try std.fs.path.join(self.a, &.{ self.root, parsed.value.name });
        errdefer self.a.free(path);
        if (std.Io.Dir.cwd().statFile(self.io, path, .{})) |_| return error.FineTuneNameExists else |err| switch (err) {
            error.FileNotFound => {},
            else => return err,
        }
        const lm = try self.scheduler.ensureLoaded(parsed.value.model);
        errdefer self.scheduler.release(lm);
        var preflight: Preflight = .{ .store = self, .model = lm, .bill = bill };
        var req: GenRequest = .{ .ctx = &preflight, .run = Preflight.run, .model = lm };
        try self.scheduler.runGeneration(&req);
        if (!preflight.ran) return error.Shutdown;
        if (preflight.failure) |name| {
            failure.* = name;
            return error.FineTuneRejected;
        }
        try std.Io.Dir.cwd().createDirPath(self.io, self.root);
        const job = try self.a.create(Job);
        errdefer self.a.destroy(job);
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        if (self.closing) return error.Shutdown;
        try self.jobs.ensureUnusedCapacity(self.a, 1);
        if (self.jobs.items.len == 32) self.jobs.orderedRemove(0).deinit();
        job.* = .{ .store = self, .id = self.next_id, .parsed = parsed, .path = path, .model = lm, .bill = bill.*, .stage_started_ms = now(self.io) };
        job.thread = try std.Thread.spawn(.{}, Job.worker, .{job});
        self.jobs.appendAssumeCapacity(job);
        self.next_id += 1;
        return job.id;
    }
    pub fn cancel(self: *Store, id: u64) !void {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        const job = self.find(id) orelse return error.UnknownFineTuneJob;
        if (!job.status.terminal()) {
            job.cancelled.store(true, .release);
            job.revision += 1;
        }
    }
    fn find(self: *Store, id: u64) ?*Job {
        for (self.jobs.items) |j| if (j.id == id) return j;
        return null;
    }
    pub const Snapshot = struct { json: []u8, revision: u64, terminal: bool };
    pub fn snapshot(self: *Store, a: std.mem.Allocator, id: u64) !Snapshot {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        const j = self.find(id) orelse return error.UnknownFineTuneJob;
        var id_buf: [40]u8 = undefined;
        const result = if (j.status == .completed) .{
            .verified = j.result.metrics.improved(),
            .heldout_prompts = j.result.metrics.heldout,
            .refused_before = j.result.metrics.refused_before,
            .refused_after = j.result.metrics.refused_after,
            .mean_kl = j.result.metrics.meanKl(),
            .max_kl = j.result.metrics.kl_max,
            .retain_tokens = j.result.metrics.retain_tokens,
            .training_examples = j.result.training_examples,
            .validation_examples = j.result.validation_examples,
            .retain_examples = j.result.retain_examples,
            .updates = j.result.updates,
            .before = j.result.before,
            .after = j.result.after,
            .recommendation = if (j.result.metrics.improved()) "Measured improvement on held-out prompts; review the examples before using the adapter." else "Verification did not establish improvement within the drift limits. Try a lower lora_scale and review the answers before using the adapter.",
        } else null;
        const elapsed: u64 = @intCast(@max(0, now(self.io) - j.stage_started_ms));
        const json = try std.json.Stringify.valueAlloc(a, .{
            .id = try std.fmt.bufPrint(&id_buf, "ft-{d}", .{j.id}),
            .model = j.parsed.value.model,
            .name = j.parsed.value.name,
            .status = j.status,
            .stage = j.update.stage,
            .step = j.update.step,
            .total = j.update.total,
            .loss = j.update.loss,
            .eta_seconds = if (j.status.terminal()) @as(?u64, 0) else verify.etaSeconds(elapsed / 1000, j.update.step, j.update.total),
            .eta_scope = "stage",
            .cancel_requested = j.cancelled.load(.acquire),
            .preview = j.preview,
            .reserve_bytes = j.bill.reserve_bytes,
            .error_name = if (j.status.terminal()) j.failure else null,
            .result = result,
        }, .{ .emit_null_optional_fields = true });
        return .{ .json = json, .revision = j.revision, .terminal = j.status.terminal() };
    }
};

const Preflight = struct {
    store: *Store,
    model: *LoadedModel,
    bill: *types.Plan,
    failure: ?[]const u8 = null,
    ran: bool = false,
    fn run(ctx: *anyopaque) void {
        const p: *Preflight = @ptrCast(@alignCast(ctx));
        p.ran = true;
        p.failure = mlx.api().finetune_plan(p.store.scheduler, p.model, p.bill);
    }
};

const Job = struct {
    store: *Store,
    id: u64,
    parsed: std.json.Parsed(types.Request),
    path: []const u8,
    model: *LoadedModel,
    bill: types.Plan,
    thread: ?std.Thread = null,
    cancelled: std.atomic.Value(bool) = .init(false),
    status: types.Status = .queued,
    update: types.Update = .{ .stage = .synth, .step = 0, .total = 128 },
    stage_started_ms: i64,
    revision: u64 = 1,
    preview: []const u8 = "",
    result: types.Result = .{},
    failure: ?[]const u8 = null,
    ran: bool = false,

    fn deinit(j: *Job) void {
        if (j.thread) |thread| thread.join();
        j.result.deinit(j.store.a);
        if (j.preview.len > 0) j.store.a.free(j.preview);
        j.store.a.free(j.path);
        j.parsed.deinit();
        j.store.a.destroy(j);
    }
    fn worker(j: *Job) void {
        var req: GenRequest = .{ .ctx = j, .run = execute, .model = j.model, .peak_bytes = j.bill.reserve_bytes };
        j.store.scheduler.runGeneration(&req) catch |err| {
            j.failure = @errorName(err);
        };
        if (!j.ran and j.failure == null) j.failure = "Shutdown";
        j.store.scheduler.release(j.model);
        j.store.mutex.lockUncancelable(j.store.io);
        defer j.store.mutex.unlock(j.store.io);
        j.status = if (j.failure) |name| (if (std.mem.eql(u8, name, "Cancelled")) .cancelled else .failed) else .completed;
        j.revision += 1;
    }
    fn execute(ctx: *anyopaque) void {
        const j: *Job = @ptrCast(@alignCast(ctx));
        j.ran = true;
        if (j.cancelled.load(.acquire)) {
            j.failure = "Cancelled";
            return;
        }
        const input: types.Run = .{ .a = j.store.a, .io = j.store.io, .request = &j.parsed.value, .path = j.path, .observer = .{ .ctx = j, .emit_fn = emit, .cancelled_fn = isCancelled }, .result = &j.result };
        j.failure = mlx.api().finetune_run(j.store.scheduler, j.model, &input);
    }
    fn isCancelled(ctx: *anyopaque) bool {
        return @as(*Job, @ptrCast(@alignCast(ctx))).cancelled.load(.acquire);
    }
    fn emit(ctx: *anyopaque, update: types.Update) void {
        const j: *Job = @ptrCast(@alignCast(ctx));
        const store = j.store;
        store.mutex.lockUncancelable(store.io);
        defer store.mutex.unlock(store.io);
        if (j.update.stage != update.stage) j.stage_started_ms = now(store.io);
        j.status = .running;
        j.update = update;
        j.update.preview = "";
        if (update.preview.len > 0 and std.unicode.utf8ValidateSlice(update.preview)) {
            var end = @min(update.preview.len, 2048);
            while (end < update.preview.len and end > 0 and update.preview[end] & 0xc0 == 0x80) end -= 1;
            if (store.a.dupe(u8, update.preview[0..end])) |copy| {
                if (j.preview.len > 0) store.a.free(j.preview);
                j.preview = copy;
            } else |_| {}
        }
        j.revision += 1;
    }
};

test "finetune: cancelling is cooperative and a completed job remains readable" {
    const a = std.testing.allocator;
    var store = try Store.init(a, std.testing.io, undefined, ".zig-cache");
    defer store.deinit();
    const j = try a.create(Job);
    j.* = .{ .store = &store, .id = 7, .parsed = try types.Request.parse(a, "{\"model\":\"m\",\"name\":\"n\",\"topic\":\"roses\"}"), .path = try a.dupe(u8, "unused"), .model = undefined, .bill = .{}, .stage_started_ms = now(store.io) };
    try store.jobs.append(a, j);
    try store.cancel(7);
    try std.testing.expect(j.cancelled.load(.acquire));
    try std.testing.expect(!j.status.terminal());
    j.status = .cancelled;
    const snap = try store.snapshot(a, 7);
    defer a.free(snap.json);
    try std.testing.expect(snap.terminal);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, snap.json, .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings("cancelled", parsed.value.object.get("status").?.string);
    try std.testing.expectError(error.UnknownFineTuneJob, store.cancel(8));
    j.status = .running;
    var bill: types.Plan = .{};
    var failure: ?[]const u8 = null;
    try std.testing.expectError(error.FineTuneBusy, store.start("{}", &bill, &failure));
    j.status = .completed;
    j.cancelled.store(false, .release);
    try store.cancel(7);
    try std.testing.expect(!j.cancelled.load(.acquire));
    const Run = struct {
        fn run(conn: *@import("../http/conn.zig").Conn, jobs: *Store, path: []const u8) !void {
            try @import("http.zig").handle(std.testing.allocator, conn, jobs, "GET", path, "");
        }
    };
    const capture = @import("../media_jobs/plugin/test_conn.zig").capture;
    const status = try capture(a, Run.run, .{ &store, "/v1/finetune/ft-7" });
    defer a.free(status);
    const events = try capture(a, Run.run, .{ &store, "/v1/finetune/ft-7/events" });
    defer a.free(events);
    const body_start = (std.mem.indexOf(u8, status, "\r\n\r\n") orelse return error.NoHttpBody) + 4;
    try std.testing.expect(std.mem.indexOf(u8, events, "event: complete\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, events, status[body_start..]) != null);
}
