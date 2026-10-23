//! Which media modality a `model_type` is, or none: the one map discovery (what is servable), the media load arms
//! (which engine slot) and `/v1/models` (each row's `modality`) read. MLX-free, so discovery may import it.

const std = @import("std");
const served = @import("../media_jobs/plugin/served.zig");

/// Detected from `config.json`'s `model_type` and carried on the load path so the registry installs the right engine
/// slot and the server dispatches the right endpoint.
pub const Modality = enum {
    image,
    audio,
    video,
    mesh,
    /// Typed decisions (`/v1/decisions`, Laya or Kev): scorers, not generators,
    /// but they ride the media plumbing (engine slot, inference-thread job).
    decision,
    /// Speaker diarization (Nemotron 3 Diarization): no route of its own, the
    /// speakers of a `diarized_json` transcription.
    diarization,

    pub fn capability(self: Modality) []const u8 {
        return switch (self) {
            .image => "image",
            .audio => "audio",
            .video => "video",
            .mesh => "3d",
            .decision => "decisions",
            .diarization => "diarization",
        };
    }

    /// Static, borrowed-static `ModelConfig.model_type` marker for each
    /// modality. Stable string literals (never freed) — `ModelConfig`
    /// treats `model_type` as borrowed-static, so a heap dupe is wrong here.
    pub fn modelType(self: Modality) []const u8 {
        return switch (self) {
            .image => "flux2",
            .audio => "qwen3_tts",
            .video => "AudioVideo",
            .mesh => "hunyuan3d_2_1",
            .decision => "laya",
            .diarization => "nemotron_diarization",
        };
    }
};

/// Every media `model_type` this server serves (a prefix where a family spells its sizes into the type). The app's
/// `isMediaModelType` table is pinned against this list (`MediaModalityTests`).
pub const media_model_types = [_][]const u8{
    "flux2",     "krea",       "mage_flow",      "mageflow",
    "qwen3_tts", "acestep",    "kokoro",         "AudioVideo",
    "hunyuan3d", "minimax_h3", "minimax_music3", "qwen_image",
    "laya",      "kev",        "clef",           "d1",
    "stable_audio3", "yue2",      "nemotron_diarization", "moss_image",
    "trellis2",  "pixal3d",
};

/// The modality of `model_type`, or null for an LM/embedding arch. Accepts the markers of `Modality.modelType` AND the
/// raw config strings discovery peeks ("flux2-klein-4b", "qwen3_tts", "AudioVideo").
pub fn fromType(model_type: []const u8) ?Modality {
    if (served.byName(model_type)) |t| return served.modalityAs(Modality, t);
    if (std.mem.eql(u8, model_type, "qwen3_tts")) return .audio;
    if (std.mem.eql(u8, model_type, "acestep")) return .audio;
    if (std.mem.eql(u8, model_type, "minimax_music3")) return .audio;
    if (std.mem.eql(u8, model_type, "yue2")) return .audio;
    if (std.mem.eql(u8, model_type, "kokoro")) return .audio;
    if (std.mem.eql(u8, model_type, "stable_audio3")) return .audio;
    if (std.mem.eql(u8, model_type, "AudioVideo")) return .video;
    if (std.mem.eql(u8, model_type, "minimax_h3")) return .video;
    if (std.mem.eql(u8, model_type, "laya") or std.mem.eql(u8, model_type, "kev") or std.mem.eql(u8, model_type, "clef") or std.mem.eql(u8, model_type, "d1")) return .decision;
    if (std.mem.eql(u8, model_type, "nemotron_diarization")) return .diarization;
    return null;
}

// ── tests ──

test "every listed media type has a modality, and an LM arch has none" {
    for (media_model_types) |mt| try std.testing.expect(fromType(mt) != null);
    for ([_][]const u8{ "gemma4", "qwen3", "llama", "deepseek_v4", "bert", "qwen3_asr" }) |mt|
        try std.testing.expect(fromType(mt) == null);
    // Every marker the load path stamps reads back as its own modality.
    for (std.enums.values(Modality)) |m| try std.testing.expectEqual(m, fromType(m.modelType()).?);
}
