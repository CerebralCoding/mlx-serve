//! Fine-tuning job creation, status, SSE snapshots, and cooperative cancellation.
const std = @import("std");
const Store = @import("jobs.zig").Store;
const types = @import("types.zig");
const Conn = @import("../http/conn.zig").Conn;
const respond = @import("../http/respond.zig");
const sendError = @import("../http/errors.zig").sendErrorResponse;

pub fn matches(path: []const u8) bool {
    return std.mem.eql(u8, path, "/v1/finetune") or std.mem.startsWith(u8, path, "/v1/finetune/");
}

pub fn handle(a: std.mem.Allocator, conn: *Conn, store: *Store, method: []const u8, path: []const u8, body: []const u8) !void {
    if (std.mem.eql(u8, path, "/v1/finetune")) {
        if (!std.mem.eql(u8, method, "POST")) return sendError(a, conn, "405 Method Not Allowed", "invalid_request_error", "Use POST /v1/finetune to create a job", 405);
        var bill: types.Plan = .{};
        var failure: ?[]const u8 = null;
        const id = store.start(body, &bill, &failure) catch |err| {
            if (err == error.FineTuneRejected) {
                const message = try rejectionMessage(a, failure.?, bill);
                defer a.free(message);
                return sendError(a, conn, "400 Bad Request", failure.?, message, 400);
            }
            const code: u16 = switch (err) {
                error.FineTuneBusy, error.FineTuneNameExists => 409,
                error.UnknownModelId => 404,
                error.Shutdown, error.OutOfMemory, error.LoadFailed, error.NotEnoughMemory => 503,
                else => 400,
            };
            const status = switch (code) {
                409 => "409 Conflict",
                404 => "404 Not Found",
                503 => "503 Service Unavailable",
                else => "400 Bad Request",
            };
            return sendError(a, conn, status, @errorName(err), @errorName(err), code);
        };
        const snapshot = try store.snapshot(a, id);
        defer a.free(snapshot.json);
        return respond.sendResponse(conn, "202 Accepted", "application/json", snapshot.json);
    }
    const route = parsePath(path) orelse return sendError(a, conn, "404 Not Found", "UnknownFineTuneJob", "Unknown fine-tuning job", 404);
    if (route.adapter) {
        mutateAdapter(a, store, route.id, method, body) catch |err| {
            const code: u16 = switch (err) {
                error.UnknownFineTuneJob => 404,
                error.FineTuneNotCompleted, error.FineTuneNameExists, error.FineTuneAdapterDiscarded => 409,
                error.InvalidAdapterName, error.InvalidFineTuneRequest => 400,
                error.MethodNotAllowed => 405,
                else => 500,
            };
            const status = switch (code) {
                404 => "404 Not Found",
                409 => "409 Conflict",
                400 => "400 Bad Request",
                405 => "405 Method Not Allowed",
                else => "500 Internal Server Error",
            };
            return sendError(a, conn, status, @errorName(err), @errorName(err), code);
        };
    } else if (std.mem.eql(u8, method, "DELETE") and !route.events) {
        store.cancel(route.id) catch return sendError(a, conn, "404 Not Found", "UnknownFineTuneJob", "Unknown fine-tuning job", 404);
    } else if (!std.mem.eql(u8, method, "GET")) return sendError(a, conn, "405 Method Not Allowed", "invalid_request_error", "Use GET for status/events or DELETE to cancel a job", 405);
    var snapshot = store.snapshot(a, route.id) catch |err| switch (err) {
        error.UnknownFineTuneJob => return sendError(a, conn, "404 Not Found", "UnknownFineTuneJob", "Unknown fine-tuning job", 404),
        else => return err,
    };
    defer a.free(snapshot.json);
    if (!route.events) return respond.sendResponse(conn, "200 OK", "application/json", snapshot.json);
    try respond.sendSseHeaders(conn, "finetune", "Content-Type, Authorization, Last-Event-ID");
    var revision: u64 = 0;
    while (true) {
        if (snapshot.revision != revision) {
            const event = try std.fmt.allocPrint(a, "id: {d}\nevent: {s}\ndata: {s}\n\n", .{ snapshot.revision, if (snapshot.terminal) "complete" else "progress", snapshot.json });
            defer a.free(event);
            try conn.writeAll(event);
            revision = snapshot.revision;
        }
        if (snapshot.terminal or conn.peerClosed()) return;
        try @import("../http/sse.zig").beatStreamKeepalive(conn, .sse_comment);
        std.Io.sleep(conn.io, .fromMilliseconds(250), .awake) catch {};
        const next = try store.snapshot(a, route.id);
        a.free(snapshot.json);
        snapshot = next;
    }
}

fn rejectionMessage(a: std.mem.Allocator, name: []const u8, bill: types.Plan) ![]u8 {
    if (std.mem.eql(u8, name, "TrainingDoesNotFit")) return std.fmt.allocPrint(a, "TrainingDoesNotFit: training needs {d} MiB beyond the loaded weights; {d} MiB is available after other requests' reservations. Unload idle models or try a smaller pack such as mlx-community/Qwen3-0.6B-4bit; its training budget will be checked before starting.", .{ bill.reserve_bytes >> 20, bill.available_bytes >> 20 });
    return std.fmt.allocPrint(a, "{s}: fine-tuning supports dense llama and qwen3 on Metal with a 4-bit affine or bf16 base.", .{name});
}

fn mutateAdapter(a: std.mem.Allocator, store: *Store, id: u64, method: []const u8, body: []const u8) !void {
    if (std.mem.eql(u8, method, "DELETE")) return store.discardAdapter(id);
    if (!std.mem.eql(u8, method, "PATCH")) return error.MethodNotAllowed;
    const parsed = std.json.parseFromSlice(struct { name: []const u8 }, a, body, .{}) catch return error.InvalidFineTuneRequest;
    defer parsed.deinit();
    try store.renameAdapter(id, parsed.value.name);
}

const Route = struct { id: u64, events: bool, adapter: bool };
fn parsePath(path: []const u8) ?Route {
    const prefix = "/v1/finetune/ft-";
    if (!std.mem.startsWith(u8, path, prefix)) return null;
    const events = std.mem.endsWith(u8, path, "/events");
    const adapter = std.mem.endsWith(u8, path, "/adapter");
    const end = path.len - (if (events) @as(usize, 7) else if (adapter) @as(usize, 8) else 0);
    if (end <= prefix.len) return null;
    const number = path[prefix.len..end];
    for (number) |c| if (!std.ascii.isDigit(c)) return null;
    const id = std.fmt.parseInt(u64, number, 10) catch return null;
    return if (id == 0) null else .{ .id = id, .events = events, .adapter = adapter };
}

test "finetune: HTTP rejects malformed input and unknown jobs without GPU work" {
    const a = std.testing.allocator;
    var store = try Store.init(a, std.testing.io, undefined, ".zig-cache");
    defer store.deinit();
    const Run = struct {
        fn run(conn: *Conn, s: *Store, method: []const u8, path: []const u8, body: []const u8) !void {
            try handle(std.testing.allocator, conn, s, method, path, body);
        }
    };
    const capture = @import("../media_jobs/plugin/test_conn.zig").capture;
    const invalid = try capture(a, Run.run, .{ &store, "POST", "/v1/finetune", "{}" });
    defer a.free(invalid);
    try std.testing.expect(std.mem.startsWith(u8, invalid, "HTTP/1.1 400"));
    try std.testing.expect(std.mem.indexOf(u8, invalid, "InvalidFineTuneRequest") != null);
    for ([_][]const u8{ "/v1/finetune/ft-1/events", "/v1/finetune/ft-0", "/v1/finetune/../../secret" }) |path| {
        const missing = try capture(a, Run.run, .{ &store, "GET", path, "" });
        defer a.free(missing);
        try std.testing.expect(std.mem.startsWith(u8, missing, "HTTP/1.1 404"));
        try std.testing.expect(std.mem.indexOf(u8, missing, "text/event-stream") == null);
    }
}
