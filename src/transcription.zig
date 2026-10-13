//! `POST /v1/audio/transcriptions` (OpenAI's form) over Qwen3-ASR: the request
//! form, the language names the model was trained on, the reference prompt
//! (mlx-audio `qwen3_asr._build_prompt`) and the model's `language X<asr_text>`
//! lead. No model state here; the handler lives in server.zig.

const std = @import("std");
const multipart = @import("multipart.zig");
const wav = @import("wav.zig");

pub const SAMPLE_RATE: u32 = 16000;

pub const Format = enum { json, text, verbose_json };

pub const Form = struct {
    file: []const u8,
    /// A Qwen3-ASR language name, resolved from an ISO-639-1 code or a name.
    language: ?[]const u8 = null,
    /// Context the transcript should follow (names, vocabulary): the system turn.
    prompt: ?[]const u8 = null,
    format: Format = .json,
    temperature: f32 = 0,
};

pub const FormError = error{ NotMultipart, MissingFile, UnsupportedResponseFormat, StreamUnsupported, BadTemperature, UnsupportedLanguage };

pub fn formErrorMessage(err: FormError) []const u8 {
    return switch (err) {
        error.NotMultipart => "send the audio as multipart/form-data with a `file` field",
        error.MissingFile => "`file` is required",
        error.UnsupportedResponseFormat => "response_format must be json, text or verbose_json (srt and vtt need timestamps this model does not produce)",
        error.StreamUnsupported => "stream is not supported on /v1/audio/transcriptions",
        error.BadTemperature => "temperature must be a number in [0, 1]",
        error.UnsupportedLanguage => "language is not one Qwen3-ASR transcribes; omit it to detect the language",
    };
}

/// Every field a client may send; unknown fields are ignored like OpenAI's.
/// `timestamp_granularities[]` is accepted: verbose_json carries one segment
/// over the whole clip and no word times.
pub fn parseForm(body: []const u8, content_type: []const u8) FormError!Form {
    const boundary = multipart.boundaryFromContentType(content_type) orelse return error.NotMultipart;
    var it = multipart.Iterator.init(body, boundary) catch return error.NotMultipart;
    var form = Form{ .file = "" };
    while (it.next()) |part| {
        const v = part.data;
        if (std.mem.eql(u8, part.name, "file")) {
            form.file = v;
        } else if (std.mem.eql(u8, part.name, "language")) {
            if (v.len != 0) form.language = languageName(v) orelse return error.UnsupportedLanguage;
        } else if (std.mem.eql(u8, part.name, "prompt")) {
            if (v.len != 0) form.prompt = v;
        } else if (std.mem.eql(u8, part.name, "response_format")) {
            if (v.len != 0) form.format = std.meta.stringToEnum(Format, v) orelse return error.UnsupportedResponseFormat;
        } else if (std.mem.eql(u8, part.name, "temperature")) {
            if (v.len != 0) {
                form.temperature = std.fmt.parseFloat(f32, v) catch return error.BadTemperature;
                if (!(form.temperature >= 0 and form.temperature <= 1)) return error.BadTemperature;
            }
        } else if (std.mem.eql(u8, part.name, "stream")) {
            if (std.mem.eql(u8, v, "true")) return error.StreamUnsupported;
        }
    }
    if (form.file.len == 0) return error.MissingFile;
    return form;
}

/// The model card's 30 languages, each with its ISO-639 code (Whisper's for
/// Cantonese and Filipino).
const LANGUAGES = [_][2][]const u8{
    .{ "zh", "Chinese" },    .{ "en", "English" },  .{ "yue", "Cantonese" }, .{ "ar", "Arabic" },    .{ "de", "German" },
    .{ "fr", "French" },     .{ "es", "Spanish" },  .{ "pt", "Portuguese" }, .{ "id", "Indonesian" }, .{ "it", "Italian" },
    .{ "ko", "Korean" },     .{ "ru", "Russian" },  .{ "th", "Thai" },       .{ "vi", "Vietnamese" }, .{ "ja", "Japanese" },
    .{ "tr", "Turkish" },    .{ "hi", "Hindi" },    .{ "ms", "Malay" },      .{ "nl", "Dutch" },      .{ "sv", "Swedish" },
    .{ "da", "Danish" },     .{ "fi", "Finnish" },  .{ "pl", "Polish" },     .{ "cs", "Czech" },      .{ "tl", "Filipino" },
    .{ "fa", "Persian" },    .{ "el", "Greek" },    .{ "ro", "Romanian" },   .{ "hu", "Hungarian" },  .{ "mk", "Macedonian" },
};

/// The model's name for an ISO code or a name in any case; null when unsupported.
pub fn languageName(code_or_name: []const u8) ?[]const u8 {
    for (LANGUAGES) |l| {
        if (std.ascii.eqlIgnoreCase(code_or_name, l[0]) or std.ascii.eqlIgnoreCase(code_or_name, l[1])) return l[1];
    }
    return null;
}

/// The reference prompt with ONE `<|audio_pad|>`, which the server expands to
/// the tower's rows. A known language is committed as the answer's lead, as
/// the reference does; otherwise the model names the language itself.
pub fn buildPrompt(a: std.mem.Allocator, context: ?[]const u8, language: ?[]const u8) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, "<|im_start|>system\n");
    if (context) |c| {
        try out.appendSlice(a, c);
        try out.append(a, '\n');
    }
    try out.appendSlice(a, "<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n");
    if (language) |l| try out.print(a, "language {s}<asr_text>", .{l});
    return out.toOwnedSlice(a);
}

/// A WAV file as 16 kHz mono f32: channels averaged, then resampled.
pub fn decodeWav(a: std.mem.Allocator, bytes: []const u8) ![]f32 {
    const d = try wav.decode(a, bytes);
    defer a.free(d.pcm);
    const ch: usize = d.channels;
    const mono = try a.alloc(f32, d.pcm.len / ch);
    defer a.free(mono);
    for (mono, 0..) |*m, i| {
        var sum: f32 = 0;
        for (d.pcm[i * ch ..][0..ch]) |v| sum += v;
        m.* = sum / @as(f32, @floatFromInt(ch));
    }
    return wav.resampleLinear(a, mono, 1, d.sample_rate, SAMPLE_RATE);
}

pub const Transcript = struct { language: ?[]const u8, text: []const u8 };

/// Split the model's `language English<asr_text>...` lead; a forced language
/// was in the prompt, so its output is the text alone.
pub fn splitOutput(out: []const u8, forced: ?[]const u8) Transcript {
    const marker = "<asr_text>";
    if (std.mem.startsWith(u8, out, "language ")) {
        if (std.mem.indexOf(u8, out, marker)) |i| return .{
            .language = std.mem.trim(u8, out["language ".len..i], " "),
            .text = std.mem.trim(u8, out[i + marker.len ..], " \n\r\t"),
        };
    }
    return .{ .language = forced, .text = std.mem.trim(u8, out, " \n\r\t") };
}

// ── Tests ──

const testing = std.testing;

fn formBody(comptime parts: []const [2][]const u8) []const u8 {
    comptime var b: []const u8 = "";
    inline for (parts) |p| b = b ++ "--X\r\nContent-Disposition: form-data; name=\"" ++ p[0] ++ "\"\r\n\r\n" ++ p[1] ++ "\r\n";
    return b ++ "--X--\r\n";
}

const CT = "multipart/form-data; boundary=X";

test "parseForm reads OpenAI's transcription form" {
    const f = try parseForm(formBody(&.{
        .{ "file", "RIFF" },           .{ "model", "whisper-1" },          .{ "language", "en" },
        .{ "prompt", "Zig, MLX" },     .{ "response_format", "verbose_json" }, .{ "temperature", "0.2" },
        .{ "timestamp_granularities[]", "word" }, .{ "chunking_strategy", "auto" },
    }), CT);
    try testing.expectEqualStrings("RIFF", f.file);
    try testing.expectEqualStrings("English", f.language.?);
    try testing.expectEqualStrings("Zig, MLX", f.prompt.?);
    try testing.expectEqual(Format.verbose_json, f.format);
    try testing.expectEqual(@as(f32, 0.2), f.temperature);
    const d = try parseForm(formBody(&.{.{ "file", "RIFF" }}), CT);
    try testing.expectEqual(Format.json, d.format);
    try testing.expect(d.language == null and d.prompt == null);
}

test "parseForm refuses by name what it cannot honour" {
    try testing.expectError(error.NotMultipart, parseForm("{}", "application/json"));
    try testing.expectError(error.MissingFile, parseForm(formBody(&.{.{ "model", "m" }}), CT));
    try testing.expectError(error.UnsupportedResponseFormat, parseForm(formBody(&.{ .{ "file", "x" }, .{ "response_format", "srt" } }), CT));
    try testing.expectError(error.StreamUnsupported, parseForm(formBody(&.{ .{ "file", "x" }, .{ "stream", "true" } }), CT));
    try testing.expectError(error.BadTemperature, parseForm(formBody(&.{ .{ "file", "x" }, .{ "temperature", "2" } }), CT));
    try testing.expectError(error.UnsupportedLanguage, parseForm(formBody(&.{ .{ "file", "x" }, .{ "language", "xx" } }), CT));
}

test "languageName takes ISO codes and names in any case" {
    try testing.expectEqualStrings("English", languageName("en").?);
    try testing.expectEqualStrings("Cantonese", languageName("yue").?);
    try testing.expectEqualStrings("German", languageName("GERMAN").?);
    try testing.expect(languageName("klingon") == null);
}

test "buildPrompt is the reference prompt" {
    const a = testing.allocator;
    const plain = try buildPrompt(a, null, null);
    defer a.free(plain);
    try testing.expectEqualStrings("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n", plain);
    const full = try buildPrompt(a, "Names: Zig", "English");
    defer a.free(full);
    try testing.expect(std.mem.startsWith(u8, full, "<|im_start|>system\nNames: Zig\n<|im_end|>\n"));
    try testing.expect(std.mem.endsWith(u8, full, "<|im_start|>assistant\nlanguage English<asr_text>"));
}

test "decodeWav returns 16 kHz mono from any rate and channel count" {
    const a = testing.allocator;
    // 0.5 s of 48 kHz stereo, left 0.5 and right -0.25: the mix is 0.125.
    var pcm: [48000]f32 = undefined;
    for (0..24000) |i| {
        pcm[2 * i] = 0.5;
        pcm[2 * i + 1] = -0.25;
    }
    const bytes = try wav.encodePcm16(a, &pcm, 48000, 2);
    defer a.free(bytes);
    const out = try decodeWav(a, bytes);
    defer a.free(out);
    try testing.expectEqual(@as(usize, 8000), out.len);
    for (out) |v| try testing.expectApproxEqAbs(@as(f32, 0.125), v, 1e-3);
    try testing.expectError(error.BadWav, decodeWav(a, "ID3 not a wav file at all, mp3 bytes here....."));
}

test "splitOutput separates the detected language from the transcript" {
    const t = splitOutput("language English<asr_text>The quick brown fox.", null);
    try testing.expectEqualStrings("English", t.language.?);
    try testing.expectEqualStrings("The quick brown fox.", t.text);
    const forced = splitOutput(" Hallo Welt.", "German");
    try testing.expectEqualStrings("German", forced.language.?);
    try testing.expectEqualStrings("Hallo Welt.", forced.text);
    // No speech: the model names no language and transcribes nothing.
    const silent = splitOutput("language None<asr_text>", null);
    try testing.expectEqualStrings("None", silent.language.?);
    try testing.expectEqualStrings("", silent.text);
}
