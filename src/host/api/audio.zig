//! transcriptions, aligner, diarizer companions

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const request_mod = @import("../../sdk/request/request.zig");
const chat_mod = @import("../chat/chat.zig");
const model_mod = @import("../../sdk/config/model.zig");
const log = @import("../../sdk/util/log.zig");
const scheduler_mod = @import("../gpu_thread/requests.zig");
const model_registry_mod = @import("../registry/registry.zig");
const transcription_mod = @import("audio/transcription.zig");
const forced_align_mod = @import("../../plugins/qwen3_asr/forced_align.zig");
const LoadedModel = model_registry_mod.LoadedModel;
const PluginEngine = @import("../media_jobs/plugin/engine.zig").Engine;
const plugin_http = @import("../media_jobs/plugin/http.zig");
const io_util = @import("../../sdk/util/io_util.zig");
const Stopwatch = io_util.Stopwatch;
const http_server = @import("../http/server.zig");
const mlx_engine = @import("../engines/mlx.zig");

const Conn = @import("../http/conn.zig").Conn;
const getTimeoutNs = @import("../http/server.zig").getTimeoutNs;
const decodeTokens = @import("tokenize.zig").decodeTokens;
const getEffectiveContextLength = @import("admission.zig").getEffectiveContextLength;
const sendGenerationError = @import("../http/errors.zig").sendGenerationError;
const checkAttentionMemory = @import("admission.zig").checkAttentionMemory;
const clampMaxTokens = @import("sampling.zig").clampMaxTokens;
const encodeText = @import("tokenize.zig").encodeText;
const jsonEscapeOrEmpty = @import("tokenize.zig").jsonEscapeOrEmpty;
const nonStreamingViaScheduler = @import("submit.zig").nonStreamingViaScheduler;
const sendResponse = @import("../http/respond.zig").sendResponse;
const sendEnsureLoadedError = @import("../http/errors.zig").sendEnsureLoadedError;
const countRejected = @import("../http/errors.zig").countRejected;
const contextOverflowMessage = @import("../http/errors.zig").contextOverflowMessage;
const sendErrorResponse = @import("../http/errors.zig").sendErrorResponse;
const MediaItem = @import("../media/placeholders.zig").MediaItem;
const prepareMultimodalItems = @import("../media/encode.zig").prepareMultimodalItems;
const multimodalFailure = @import("../media/encode.zig").multimodalFailure;

/// Longest piece one pass of the ASR transcribes: mlx-audio's chunk length,
/// past which the reference splits the audio.
const TRANSCRIPTION_MAX_SECONDS = 1200;

/// The reference's `max_tokens` for one chunk.
const TRANSCRIPTION_MAX_TOKENS = 8192;

/// The aligner the app downloads beside a Qwen3-ASR model.
const ALIGNER_REPO = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit";

/// The diarizer the app downloads beside it.
const DIARIZER_REPO = "mlx-community/Nemotron-3-Diarization-8bit";

/// `POST /v1/audio/transcriptions`: OpenAI's form over Qwen3-ASR. The reference
/// prompt's one audio placeholder expands to the tower's rows like any chat
/// media, then the trunk decodes the transcript. verbose_json, srt, vtt and
/// diarized_json time it with Qwen3-ForcedAligner when one is on disk, cutting
/// the clip into pieces the aligner can time as the official toolkit does; a
/// piece in a language the aligner lacks is one segment. diarized_json labels
/// each word with the speaker Nemotron 3 Diarization hears over the whole clip.
pub fn handleTranscriptions(allocator: std.mem.Allocator, stream: *Conn, body: []const u8, content_type: []const u8, lm: *LoadedModel) !void {
    const hears = mlx_engine.facts(lm).hears;
    if (!hears) {
        log.warn("POST /v1/audio/transcriptions -> 400 (no audio tower)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "This model does not transcribe audio: target a Qwen3-ASR model (and do not run it with --no-vision, which drops its audio tower)", 400);
        return;
    }
    const form = transcription_mod.parseForm(body, content_type) catch |err| {
        log.warn("POST /v1/audio/transcriptions -> 400 ({s})\n", .{@errorName(err)});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", transcription_mod.formErrorMessage(err), 400);
        return;
    };
    const pcm = transcription_mod.decodeWav(allocator, form.file) catch |err| {
        log.warn("POST /v1/audio/transcriptions -> 400 (audio: {s})\n", .{@errorName(err)});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "`file` must be a WAV file (16- or 24-bit PCM, or 32-bit float)", 400);
        return;
    };
    defer allocator.free(pcm);
    const seconds = @as(f64, @floatFromInt(pcm.len)) / transcription_mod.SAMPLE_RATE;
    log.info("POST /v1/audio/transcriptions ({d:.1} s, language={s}, format={s}, prompt={d}b)\n", .{ seconds, form.language orelse "auto", @tagName(form.format), if (form.prompt) |p| p.len else 0 });

    var aligner: ?Aligner = if (transcription_mod.timed(form.format)) try findAligner(allocator) else null;
    if (aligner == null and (form.format == .srt or form.format == .vtt)) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "srt and vtt need word times from Qwen3-ForcedAligner, which is not on disk or did not load: download " ++ ALIGNER_REPO ++ " beside the ASR model", 400);
        return;
    }
    // Speakers are heard over the whole clip, so a voice keeps its letter across pieces.
    var speaker_probs: ?[]f32 = null;
    defer if (speaker_probs) |p| allocator.free(p);
    var diarizer: ?*LoadedModel = null;
    defer if (diarizer) |m| http_server.launch().global_scheduler.?.release(m);
    var speakers: u32 = 0;
    if (form.format == .diarized_json) {
        const id = (findCompanion("nemotron_diarization") orelse {
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "diarized_json needs Nemotron 3 Diarization, which is not on disk or did not load: download " ++ DIARIZER_REPO ++ " beside the ASR model", 400);
            return;
        }).id;
        diarizer = http_server.launch().global_scheduler.?.ensureLoaded(id) catch |err| blk: {
            log.warn("  diarizer {s} did not load ({s}): one speaker\n", .{ id, @errorName(err) });
            break :blk null;
        };
        if (diarizer) |m| speaker_probs = (try diarizeClip(allocator, stream, m, pcm, &speakers)) orelse return;
    }
    const who: ?transcription_mod.Activity = if (speaker_probs) |p| .{ .probs = p, .speakers = speakers } else null;

    const piece_seconds: usize = if (aligner != null) transcription_mod.ALIGN_MAX_SECONDS else TRANSCRIPTION_MAX_SECONDS;
    const ends = try transcription_mod.splitPoints(allocator, pcm, piece_seconds * transcription_mod.SAMPLE_RATE);
    defer allocator.free(ends);

    var tl = transcription_mod.Timeline{};
    defer tl.deinit(allocator);
    defer if (aligner) |al| if (al.lm) |m| http_server.launch().global_scheduler.?.release(m);
    var start: usize = 0;
    for (ends) |end| {
        const heard = (try transcribeClip(allocator, stream, lm, form, pcm[start..end])) orelse return;
        defer allocator.free(heard.raw);
        const t = heard.t;
        var timed: ?Aligned = null;
        defer if (timed) |*x| x.deinit(allocator);
        if (aligner) |*al| {
            if (t.text.len > 0 and al.languages & transcription_mod.languageBit(t.language orelse "") != 0) {
                if (al.lm == null) al.lm = http_server.launch().global_scheduler.?.ensureLoaded(al.id) catch |err| blk: {
                    // verbose_json still answers without times; srt and vtt cannot.
                    if (form.format != .verbose_json) {
                        try sendEnsureLoadedError(allocator, stream, http_server.launch().global_scheduler.?, al.id, err);
                        return;
                    }
                    log.warn("  aligner {s} did not load ({s}): segments without word times\n", .{ al.id, @errorName(err) });
                    al.languages = 0;
                    break :blk null;
                };
                if (al.lm) |alm| timed = (try alignPiece(allocator, stream, alm, pcm[start..end], t.text, t.language.?, who, start * 1000 / transcription_mod.SAMPLE_RATE)) orelse return;
            }
        }
        const rate: f64 = transcription_mod.SAMPLE_RATE;
        const speaker = if (who) |w| w.speakerOver(start * 1000 / transcription_mod.SAMPLE_RATE, end * 1000 / transcription_mod.SAMPLE_RATE, 0) else 0;
        try tl.add(allocator, t.text, t.language, @as(f64, @floatFromInt(start)) / rate, @as(f64, @floatFromInt(end - start)) / rate, if (timed) |x| x.words else null, if (timed) |x| x.segments else null, speaker);
        start = end;
    }

    switch (form.format) {
        .text => try sendResponse(stream, "200 OK", "text/plain; charset=utf-8", tl.text.items),
        .json => {
            const text = jsonEscapeOrEmpty(allocator, tl.text.items);
            defer if (text.owned) allocator.free(text.slice);
            const response = try std.fmt.allocPrint(allocator, "{{\"text\":{s}}}", .{text.slice});
            defer allocator.free(response);
            try sendResponse(stream, "200 OK", "application/json", response);
        },
        .verbose_json => {
            const response = try tl.verboseJson(allocator, seconds, form.words);
            defer allocator.free(response);
            try sendResponse(stream, "200 OK", "application/json", response);
        },
        .srt, .vtt => {
            const response = try tl.subtitles(allocator, form.format);
            defer allocator.free(response);
            try sendResponse(stream, "200 OK", "text/plain; charset=utf-8", response);
        },
        .diarized_json => {
            const response = try tl.diarizedJson(allocator, seconds);
            defer allocator.free(response);
            try sendResponse(stream, "200 OK", "application/json", response);
        },
    }
}

/// The transcript of one clip and the bytes it points into.
const Heard = struct { raw: []u8, t: transcription_mod.Transcript };

/// One pass of the ASR over `pcm`; null when it already sent the error.
fn transcribeClip(allocator: std.mem.Allocator, stream: *Conn, lm: *LoadedModel, form: transcription_mod.Form, pcm: []const f32) !?Heard {
    const config = lm.config.?;
    const tok = lm.tokenizer.?;
    const prompt_text = try transcription_mod.buildPrompt(allocator, form.prompt, form.language);
    defer allocator.free(prompt_text);
    var prompt_ids = try encodeText(allocator, lm, tok, prompt_text, false);
    const audio = chat_mod.AudioData{ .samples = std.mem.sliceAsBytes(pcm) };
    var mm = prepareMultimodalItems(allocator, lm, config, &[_]MediaItem{.{ .audio = &audio }}, &prompt_ids) catch |err| {
        allocator.free(prompt_ids);
        const f = multimodalFailure(err);
        log.warn("POST /v1/audio/transcriptions -> {d} ({s})\n", .{ f.code, @errorName(err) });
        try sendErrorResponse(allocator, stream, f.status, f.kind, f.message, f.code);
        return null;
    };
    defer mm.deinit(allocator);
    defer allocator.free(prompt_ids);

    const effective_ctx = getEffectiveContextLength(config);
    if (prompt_ids.len > effective_ctx) {
        countRejected();
        var ovf_buf: [160]u8 = undefined;
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", contextOverflowMessage(&ovf_buf, prompt_ids.len, effective_ctx), 400);
        return null;
    }
    const max_tokens = clampMaxTokens(TRANSCRIPTION_MAX_TOKENS, prompt_ids.len, effective_ctx);
    if (!try checkAttentionMemory(allocator, stream, prompt_ids, max_tokens, config, false, null, lm, false, mm.bytes(config))) return null;

    // Ownership of the soft tokens moves to the slot. The reference decodes plainly: no thinking, no drafts.
    const slot_ve = mm.embeddings;
    mm.embeddings = null;
    const enable_thinking = false;
    const spec = false;
    var timer = Stopwatch.init(stream.io);
    const result = nonStreamingViaScheduler(allocator, http_server.launch().global_scheduler.?, lm, tok, prompt_ids, prompt_ids, max_tokens, .{ .temperature = form.temperature }, config.eosTokenSlice(), 0, false, enable_thinking, spec, spec, spec, spec, getTimeoutNs(), slot_ve, mm.media, 0, .{}, 0, false, null, null, stream) catch |err| {
        log.err("  -> {s}\n", .{@errorName(err)});
        sendGenerationError(allocator, stream, err, .openai) catch {};
        return null;
    };
    defer allocator.free(result.text);
    defer allocator.free(result.token_ids);
    const raw = try decodeTokens(allocator, lm, tok, result.token_ids, false);
    const t = transcription_mod.splitOutput(raw, form.language);
    log.info("  <- {d}+{d} tokens ({d}ms) [{s}] language={s}\n", .{ result.prompt_tokens, result.completion_tokens, timer.read() / std.time.ns_per_ms, result.finish_reason, t.language orelse "?" });
    return .{ .raw = raw, .t = t };
}

/// The forced aligner a timed transcription uses: its registry id, the
/// languages its config lists, and the model once loaded (refcounted).
const Aligner = struct { id: []const u8, languages: u32, lm: ?*LoadedModel = null };

/// The model of one `arch_hint` a transcription leans on (the aligner, the
/// diarizer): the resident one first, one that failed to load never; null
/// when none is on disk.
fn findCompanion(arch_hint: []const u8) ?*LoadedModel {
    const registry = http_server.global_registry orelse return null;
    registry.mutex.lockUncancelable(registry.io);
    defer registry.mutex.unlock(registry.io);
    var pick: ?*LoadedModel = null;
    var it = registry.entries.valueIterator();
    while (it.next()) |e| {
        if (!std.mem.eql(u8, e.*.arch_hint, arch_hint) or e.*.state == .error_state) continue;
        const p = pick orelse {
            pick = e.*;
            continue;
        };
        const ready = e.*.state == .ready;
        if (ready != (p.state == .ready)) {
            if (ready) pick = e.*;
        } else if (std.mem.lessThan(u8, e.*.id, p.id)) pick = e.*;
    }
    return pick;
}

fn findAligner(allocator: std.mem.Allocator) !?Aligner {
    const entry = findCompanion("qwen3_forced_aligner") orelse return null;
    var config = model_mod.parseConfig(http_server.global_registry.?.io, allocator, entry.path) catch return null;
    defer config.deinit(allocator);
    const fa = config.forced_aligner orelse return null;
    return .{ .id = entry.id, .languages = fa.languages };
}

/// A piece's words and segments, spans of its transcript.
const Aligned = struct {
    words: []const forced_align_mod.Timed,
    segments: []const forced_align_mod.Segment,
    fn deinit(self: *Aligned, allocator: std.mem.Allocator) void {
        allocator.free(self.words);
        allocator.free(self.segments);
    }
};

/// Time the words of `text` (the transcript of `pcm`, `offset_ms` into the
/// clip) with the aligner `lm`, a segment closing where `who` hears the
/// speaker change. Null when it already sent the error.
fn alignPiece(allocator: std.mem.Allocator, stream: *Conn, lm: *LoadedModel, pcm: []const f32, text: []const u8, language: []const u8, who: ?transcription_mod.Activity, offset_ms: u64) !?Aligned {
    const config = lm.config.?;
    const fa = config.forced_aligner.?;
    const script = forced_align_mod.scriptFor(language);
    const units = try forced_align_mod.units(allocator, text, script);
    defer forced_align_mod.freeUnits(allocator, units);
    if (units.len == 0) return .{ .words = &.{}, .segments = &.{} };
    const prompt_text = try forced_align_mod.promptText(allocator, units);
    defer allocator.free(prompt_text);
    var ids = try encodeText(allocator, lm, lm.tokenizer.?, prompt_text, false);
    const audio = chat_mod.AudioData{ .samples = std.mem.sliceAsBytes(pcm) };
    var mm = prepareMultimodalItems(allocator, lm, config, &[_]MediaItem{.{ .audio = &audio }}, &ids) catch |err| {
        allocator.free(ids);
        const f = multimodalFailure(err);
        log.warn("  aligner -> {d} ({s})\n", .{ f.code, @errorName(err) });
        try sendErrorResponse(allocator, stream, f.status, f.kind, f.message, f.code);
        return null;
    };
    defer mm.deinit(allocator);
    defer allocator.free(ids);

    // The job frees the audio rows on the inference thread.
    var job = AlignJob{ .allocator = allocator, .lm = lm, .ids = ids, .rows = mm.embeddings.?, .timestamp_token = fa.timestamp_token_id };
    mm.embeddings = null;
    var timer = Stopwatch.init(stream.io);
    var req = scheduler_mod.GenRequest{ .ctx = &job, .run = alignJobRun, .model = lm };
    http_server.launch().global_scheduler.?.runGeneration(&req) catch |err| {
        _ = mlx.mlx_array_free(job.rows);
        if (err != error.Shutdown) return err;
        try sendErrorResponse(allocator, stream, "503 Service Unavailable", "shutting_down", "Server is shutting down", 503);
        return null;
    };
    const classes = job.classes orelse {
        try sendErrorResponse(allocator, stream, "500 Internal Server Error", "server_error", "word alignment failed", 500);
        return null;
    };
    defer allocator.free(classes);
    const ms = try forced_align_mod.fixTimestamps(allocator, classes, fa.step_ms);
    defer allocator.free(ms);
    if (ms.len != 2 * units.len) {
        try sendErrorResponse(allocator, stream, "500 Internal Server Error", "server_error", "word alignment returned a time for each <timestamp> it read, not two per word", 500);
        return null;
    }
    const words = try forced_align_mod.timedWords(allocator, units, ms, script);
    errdefer allocator.free(words);
    const said_by = if (who) |w| try w.wordSpeakers(allocator, words, offset_ms) else null;
    defer if (said_by) |x| allocator.free(x);
    const segments = try forced_align_mod.segments(allocator, text, words, said_by);
    log.info("  aligned {d} words into {d} segments ({d} tokens, {d}ms)\n", .{ words.len, segments.len, ids.len, timer.read() / std.time.ns_per_ms });
    return .{ .words = words, .segments = segments };
}

/// One aligner pass, run on the inference thread (the sole mlx caller).
const AlignJob = struct {
    allocator: std.mem.Allocator,
    lm: *LoadedModel,
    ids: []const u32,
    rows: mlx.mlx_array,
    timestamp_token: u32,
    classes: ?[]u32 = null,
};

fn alignJobRun(ctx: *anyopaque) void {
    const job: *AlignJob = @ptrCast(@alignCast(ctx));
    defer _ = mlx.mlx_array_free(job.rows);
    job.classes = forced_align_mod.timestampClasses(job.allocator, job.lm.transformer.?, job.ids, job.rows, job.timestamp_token) catch |err| {
        log.err("  alignment failed: {s}\n", .{@errorName(err)});
        return;
    };
}

/// Speaker activity over `pcm` from the diarizer `lm`, on the inference thread.
/// Null when it already sent the error.
fn diarizeClip(allocator: std.mem.Allocator, stream: *Conn, lm: *LoadedModel, pcm: []const f32, speakers: *u32) !?[]f32 {
    var job = DiarizeJob{ .allocator = allocator, .plugin = lm.media_plugin, .pcm = pcm };
    var timer = Stopwatch.init(stream.io);
    var req = scheduler_mod.GenRequest{ .ctx = &job, .run = diarizeJobRun, .model = lm };
    http_server.launch().global_scheduler.?.runGeneration(&req) catch |err| {
        if (err != error.Shutdown) return err;
        try sendErrorResponse(allocator, stream, "503 Service Unavailable", "shutting_down", "Server is shutting down", 503);
        return null;
    };
    const probs = job.probs orelse {
        try sendErrorResponse(allocator, stream, "500 Internal Server Error", "server_error", "speaker diarization failed", 500);
        return null;
    };
    speakers.* = job.speakers;
    log.info("  diarized {d:.1} s ({d}ms)\n", .{ @as(f64, @floatFromInt(pcm.len)) / transcription_mod.SAMPLE_RATE, timer.read() / std.time.ns_per_ms });
    return probs;
}

const DiarizeJob = struct {
    allocator: std.mem.Allocator,
    /// The diarizer's plugin (`media_plugin`).
    plugin: ?*PluginEngine,
    pcm: []const f32,
    probs: ?[]f32 = null,
    speakers: u32 = 0,
};

fn diarizeJobRun(ctx: *anyopaque) void {
    const job: *DiarizeJob = @ptrCast(@alignCast(ctx));
    const p = job.plugin orelse return;
    const r = plugin_http.diarize(p, job.allocator, job.pcm) orelse return;
    job.probs = r.probs;
    job.speakers = r.speakers;
}
