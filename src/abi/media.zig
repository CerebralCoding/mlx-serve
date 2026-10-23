//! The media plugin table (`MediaTable`): image, audio, video and mesh backends, and the per-request decision and
//! diarization jobs. Same rules as `plugin.zig`: extern layouts, C calling convention, `size` first, no globals.

const Host = @import("host.zig").Host;
const std = @import("std");
const common = @import("common.zig");

pub const Span = common.Span;
pub const Str = common.Str;
pub const Verdict = common.Verdict;
pub const Stream = common.Stream;
pub const Claim = common.Claim;
pub const PackDir = common.PackDir;

pub const version: u32 = 1;

pub const Modality = enum(u8) { image, audio, video, mesh, decision, diarization };

/// The audio routes a pack serves.
pub const Routes = extern struct {
    size: u32 = @sizeOf(Routes),
    speech: bool = false,
    music: bool = false,
    sound: bool = false,
};

/// What one pack supports. Read from the pack (a distilled and a base checkpoint of one family differ), no MLX.
pub const MediaCaps = extern struct {
    size: u32 = @sizeOf(MediaCaps),
    img2img: bool = false,
    edit: bool = false,
    /// The edit takes the upload's bytes as sent (no decode by the host).
    edit_raw_bytes: bool = false,
    /// Real CFG: `guidance_scale` and `negative_prompt`.
    guidance: bool = false,
    cond_weights: bool = false,
    multiview: bool = false,
    default_guidance: f32 = 0,
    /// The largest side in pixels; 0 = no cap.
    max_dim: u32 = 0,
    /// Reference images an edit takes; 0 = edits take none.
    edit_ref_cap: u32 = 0,
    /// The LoRA key family the pack's adapters are written for; empty = no LoRA.
    lora_arch: Str = .{},
    routes: Routes = .{},
};

/// The machine and launch facts a bill reads (the host fills them; a bill never reaches MLX).
pub const Machine = extern struct {
    size: u32 = @sizeOf(Machine),
    /// Metal's recommended working set in bytes; 0 = unknown.
    gpu_working_set: u64 = 0,
    /// `--w8a8`: media DiT linears load as int8 copies that replace the packed weights.
    w8a8: bool = false,
    /// `--ane-image` / `--ane-video` / `--ane-audio`: that modality's DiT stays packed for the ANE seam.
    ane_image: bool = false,
    ane_video: bool = false,
    ane_audio: bool = false,
    /// iOS, `MLXSERVE_LOWMEM=1`, or RAM at most 17 GiB: stage the large encoders per request.
    low_mem: bool = false,
};

/// Valid during `load` only.
pub const LoadCtx = extern struct {
    size: u32 = @sizeOf(LoadCtx),
    model_dir: Str,
    dir: *const PackDir,
    settings_json: Str = .{},
    stream: Stream = .{},
    /// The host table (P2 step 8); opaque until it lands.
    host: ?*const Host = null,
    machine: ?*const Machine = null,
};

pub const Lora = extern struct {
    size: u32 = @sizeOf(Lora),
    path: Str,
    scale: f32 = 1,
};

/// One job: its JSON body and its uploads (decoded images or audio, in body order).
pub const Request = extern struct {
    size: u32 = @sizeOf(Request),
    body: Str,
    inputs: Span(Str) = .{},
};

/// The host's callbacks into a running job, on the GPU thread.
pub const Progress = extern struct {
    size: u32 = @sizeOf(Progress),
    ctx: ?*anyopaque,
    report: *const fn (ctx: ?*anyopaque, stage: Str, step: u32, total: u32) callconv(.c) void,
    /// Between steps: the host serves chat for its share of the step, then answers whether the client left (stop).
    boundary: *const fn (ctx: ?*anyopaque) callconv(.c) bool,
};

/// A media plugin's table, exported as `exportName(name, version)`. `claims`, `caps` and the bills never reach MLX.
pub const MediaTable = extern struct {
    size: u32 = @sizeOf(MediaTable),
    version: u32 = version,
    name: Str,
    /// `model_type` is empty for a configless pack (classified from its index files).
    claims: *const fn (model_type: Str, dir: *const PackDir) callconv(.c) Claim,
    /// The file whose presence makes a claimed pack complete; until it lands the pack is invisible.
    required_file: Str,
    modality: Modality,
    caps: *const fn (dir: *const PackDir, out: *MediaCaps) callconv(.c) void,
    /// The load peak in bytes.
    bill_load: *const fn (dir: *const PackDir, m: *const Machine) callconv(.c) u64,
    /// One request's peak beyond the load, from its body.
    bill_request: *const fn (dir: *const PackDir, m: *const Machine, body: Str) callconv(.c) u64,
    load: *const fn (ctx: *const LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque,
    unload: *const fn (e: *anyopaque) callconv(.c) void,
    /// Installs the stack for the next job (summed, never merged); an empty span clears it.
    set_loras: ?*const fn (e: *anyopaque, loras: Span(Lora)) callconv(.c) Verdict = null,
    /// Runs one job to its result, calling `progress` at every step boundary.
    step: *const fn (e: *anyopaque, req: *const Request, progress: *const Progress) callconv(.c) Verdict,
    /// The last job's result bytes (PNG, WAV, GLB, JSON), in memory the plugin owns until its next `step` or `unload`.
    encode: *const fn (e: *anyopaque, out: *Str) callconv(.c) Verdict,
};

// ── tests ──

const testing = std.testing;

test "abi media: every struct opens with its size" {
    inline for (.{ Routes, MediaCaps, Machine }) |T| {
        const v: T = .{};
        try testing.expectEqual(@as(u32, @sizeOf(T)), v.size);
    }
    inline for (.{ Routes, MediaCaps, Machine, LoadCtx, Lora, Request, Progress, MediaTable }) |T|
        try testing.expectEqual(@as(usize, 0), @offsetOf(T, "size"));
}
