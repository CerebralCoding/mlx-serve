//! Bounded job history; workers pin a model and post all GPU work to its scheduler.
const std = @import("std");
const types = @import("types.zig");
const mlx = @import("../engines/mlx.zig");
const Scheduler = @import("../gpu_thread/scheduler.zig").Scheduler;
const GenRequest = @import("../gpu_thread/requests.zig").GenRequest;
const LoadedModel = @import("../registry/registry.zig").LoadedModel;
const now = @import("../../sdk/util/io_util.zig").nowMsMonotonic;
const verify = @import("../../sdk/train/verify.zig");

pub fn rootPath(a: std.mem.Allocator) ![]u8 {
    const home = @import("../../env.zig").span(.HOME) orelse return error.HomeNotFound;
    return std.fs.path.join(a, &.{ home, ".mlx-serve", "finetune" });
}

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
        const model_path = try self.a.dupe(u8, lm.path);
        errdefer self.a.free(model_path);
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        if (self.closing) return error.Shutdown;
        try self.jobs.ensureUnusedCapacity(self.a, 1);
        if (self.jobs.items.len == 32) self.jobs.orderedRemove(0).deinit();
        job.* = .{ .store = self, .id = self.next_id, .parsed = parsed, .path = path, .model = lm, .model_path = model_path, .bill = bill.*, .stage_started_ms = now(self.io) };
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
    pub fn applyAdapter(self: *Store, id: u64) !void {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        const job = self.find(id) orelse return error.UnknownFineTuneJob;
        if (job.status != .completed) return error.FineTuneNotCompleted;
        if (job.adapter_discarded) return error.FineTuneAdapterDiscarded;
        if (job.adapter_applied) return;
        const source = try std.fs.path.join(self.a, &.{ job.path, "adapters.safetensors" });
        defer self.a.free(source);
        const dest = try std.fs.path.join(self.a, &.{ job.model_path, @import("../../sdk/weights/finetune.zig").filename });
        defer self.a.free(dest);
        std.Io.Dir.copyFile(.cwd(), source, .cwd(), dest, self.io, .{ .replace = false }) catch |err| switch (err) {
            error.PathAlreadyExists => return error.FineTuneOverlayExists,
            else => return err,
        };
        job.adapter_applied = true;
        job.revision += 1;
    }
    pub fn discardAdapter(self: *Store, id: u64) !void {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        const job = self.find(id) orelse return error.UnknownFineTuneJob;
        if (job.status != .completed) return error.FineTuneNotCompleted;
        if (job.adapter_applied) return error.FineTuneAlreadyApplied;
        if (job.adapter_discarded) return;
        try std.Io.Dir.cwd().deleteTree(self.io, job.path);
        job.adapter_discarded = true;
        job.revision += 1;
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
            .new_refusals = j.result.metrics.new_refusals,
            .mean_kl = j.result.metrics.meanKl(),
            .max_kl = j.result.metrics.kl_max,
            .retain_tokens = j.result.metrics.retain_tokens,
            .training_examples = j.result.training_examples,
            .validation_examples = j.result.validation_examples,
            .retain_examples = j.result.retain_examples,
            .updates = j.result.updates,
            .prompt = j.result.prompt,
            .before = j.result.before,
            .after = j.result.after,
            .recommendation = if (j.result.metrics.new_refusals > 0) "The adapter introduced new refusals on held-out prompts. Discard this result or review the answers before applying." else if (j.result.metrics.improved()) "Measured improvement on held-out prompts after merging; review the examples before applying." else "Verification did not establish improvement within the drift limits after merging. Discard this result or review the answers before applying.",
        } else null;
        const elapsed: u64 = @intCast(@max(0, now(self.io) - j.stage_started_ms));
        const json = try std.json.Stringify.valueAlloc(a, .{
            .id = try std.fmt.bufPrint(&id_buf, "ft-{d}", .{j.id}),
            .model = j.parsed.value.model,
            .name = std.fs.path.basename(j.path),
            .adapter_discarded = j.adapter_discarded,
            .adapter_applied = j.adapter_applied,
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
    model_path: []const u8 = "",
    bill: types.Plan,
    thread: ?std.Thread = null,
    cancelled: std.atomic.Value(bool) = .init(false),
    status: types.Status = .queued,
    update: types.Update = .{ .stage = .synth, .step = 0, .total = 128 },
    stage_started_ms: i64,
    revision: u64 = 1,
    preview: []const u8 = "",
    result: types.Result = .{},
    adapter_discarded: bool = false,
    adapter_applied: bool = false,
    failure: ?[]const u8 = null,
    ran: bool = false,

    fn deinit(j: *Job) void {
        if (j.thread) |thread| thread.join();
        j.result.deinit(j.store.a);
        if (j.preview.len > 0) j.store.a.free(j.preview);
        j.store.a.free(j.path);
        if (j.model_path.len > 0) j.store.a.free(j.model_path);
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

test "LoRA fixtures: cancellation and verification results agree in status and events" {
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
    j.result.metrics = .{ .heldout = 4, .refused_before = 2, .refused_after = 1, .new_refusals = 1, .retain_tokens = 10, .kl_sum = 0.1, .kl_max = 0.02 };
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
    const completed = try std.json.parseFromSlice(std.json.Value, a, status[body_start..], .{});
    defer completed.deinit();
    const result = completed.value.object.get("result").?.object;
    try std.testing.expect(!result.get("verified").?.bool);
    try std.testing.expectEqual(@as(i64, 1), (result.get("new_refusals") orelse return error.MissingNewRefusals).integer);
    try std.testing.expectEqualStrings("The adapter introduced new refusals on held-out prompts. Discard this result or review the answers before applying.", result.get("recommendation").?.string);
    try std.testing.expect(std.mem.indexOf(u8, events, "event: complete\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, events, status[body_start..]) != null);
}

test "finetune: completed jobs install an atomic model-local sidecar and preserve installed results" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(io, &buf)];
    var store = try Store.init(a, io, undefined, root);
    defer store.deinit();
    try tmp.dir.createDir(io, "draft", .default_dir);
    try tmp.dir.createDir(io, "model", .default_dir);
    try tmp.dir.writeFile(io, .{ .sub_path = "draft/adapters.safetensors", .data = "trained weights with metadata" });
    const j = try a.create(Job);
    j.* = .{ .store = &store, .id = 1, .parsed = try types.Request.parse(a, "{\"model\":\"m\",\"name\":\"draft\",\"topic\":\"roses\"}"), .path = try std.fs.path.join(a, &.{ root, "draft" }), .model = undefined, .bill = .{}, .stage_started_ms = now(io) };
    try store.jobs.append(a, j);
    j.model_path = try std.fs.path.join(a, &.{ root, "model" });
    try std.testing.expectError(error.FineTuneNotCompleted, store.applyAdapter(1));
    try std.testing.expectError(error.FineTuneNotCompleted, store.discardAdapter(1));
    const Http = struct {
        fn run(conn: *@import("../http/conn.zig").Conn, jobs: *Store, method: []const u8, body: []const u8) !void {
            try @import("http.zig").handle(std.testing.allocator, conn, jobs, method, "/v1/finetune/ft-1/adapter", body);
        }
    };
    const capture = @import("../media_jobs/plugin/test_conn.zig").capture;
    const busy = try capture(a, Http.run, .{ &store, "DELETE", "" });
    defer a.free(busy);
    try std.testing.expect(std.mem.startsWith(u8, busy, "HTTP/1.1 409"));
    j.status = .completed;
    try tmp.dir.writeFile(io, .{ .sub_path = "model/finetune.lora.safetensor", .data = "existing" });
    try std.testing.expectError(error.FineTuneOverlayExists, store.applyAdapter(1));
    try tmp.dir.deleteFile(io, "model/finetune.lora.safetensor");
    const applied = try capture(a, Http.run, .{ &store, "POST", "" });
    defer a.free(applied);
    try std.testing.expect(std.mem.startsWith(u8, applied, "HTTP/1.1 200"));
    try std.testing.expect(std.mem.indexOf(u8, applied, "\"adapter_applied\":true") != null);
    const bytes = try tmp.dir.readFileAlloc(io, "model/finetune.lora.safetensor", a, .limited(1024));
    defer a.free(bytes);
    try std.testing.expectEqualStrings("trained weights with metadata", bytes);
    try store.applyAdapter(1);
    try std.testing.expectError(error.FineTuneAlreadyApplied, store.discardAdapter(1));
    j.adapter_applied = false;
    try tmp.dir.deleteFile(io, "draft/adapters.safetensors");
    try std.testing.expectError(error.FileNotFound, store.applyAdapter(1));
    try std.testing.expect(!j.adapter_applied);
    const discarded = try capture(a, Http.run, .{ &store, "DELETE", "" });
    defer a.free(discarded);
    try std.testing.expect(std.mem.startsWith(u8, discarded, "HTTP/1.1 200"));
    try std.testing.expect(std.mem.indexOf(u8, discarded, "\"adapter_discarded\":true") != null);
    try store.discardAdapter(1);
    try std.testing.expectError(error.FileNotFound, tmp.dir.statFile(io, "draft", .{}));
    _ = try tmp.dir.statFile(io, "model/finetune.lora.safetensor", .{});
    try std.testing.expectError(error.FineTuneAdapterDiscarded, store.applyAdapter(1));
}
