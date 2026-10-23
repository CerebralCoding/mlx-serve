//! The model types the MLX engine serves through their plugin's table. A family flips by adding its types here once
//! its plugin is proven against the legacy forward; every other type keeps the legacy path.

const std = @import("std");

pub const types = [_][]const u8{
    "llama",
    "mistral",
    "qwen2",
    "qwen3",
    "k2_horizon",
    "spark2_5",
    "laguna",
    "flint",
    "kolibri1",
    "muse_glimmer",
    "muse_glimmer_text",
    "gpt_oss",
    "mimo_v2",
    "mimo_v2_flash",
    "bert",
    "lfm2",
    "lfm2_moe",
    "nemotron_h",
    "nemotron_h_puzzle",
    "hy_v3",
    "deepseek_v4",
    "deepseek_v41",
    "inkling_mm_model",
    "lfm2_vl",
    "lfm2-vl",
    "nemotron_h_omni",
    "qwen3_moe",
    "qwen3_moe_text",
    "qwen3_next",
    "qwen3_5",
    "qwen3_5_text",
    "qwen3_5_moe",
    "qwen3_5_moe_text",
    "minicpmv4_7",
    "qwen3_asr",
    "prism_hadamard_qwen35",
    "gemma3",
    "gemma4",
    "gemma4_text",
    "diffusion_gemma",
    "gemma3_text",
    "embedding_gemma2",
    "gemma4_unified",
    "gemma4_unified_text",
    "qwen4_exp",
    "qwen4_exp_text",
    "glm5_next",
    "glm5_next_text",
    "bailing_hybrid",
};

/// Served types whose legacy forward is deleted: a pack the plugin declines is refused by name, never handed to the
/// legacy path.
pub const legacy_removed = [_][]const u8{
    "kolibri1",
    "flint",
    "laguna",
    "gpt_oss",
    "spark2_5",
    "embedding_gemma2",
    "diffusion_gemma",
    "deepseek_v4",
    "inkling_mm_model",
    "muse_glimmer",
    "muse_glimmer_text",
    "bert",
    "lfm2",
    "lfm2_moe",
    "lfm2_vl",
    "lfm2-vl",
    "nemotron_h",
    "nemotron_h_puzzle",
    "nemotron_h_omni",
    "mimo_v2",
    "mimo_v2_flash",
};

pub fn legacyRemoved(model_type: []const u8) bool {
    for (legacy_removed) |t| if (std.mem.eql(u8, t, model_type)) return true;
    return false;
}

/// Decoder types whose plugin serves the embedding arm too (a padded batch, no cache): a pack whose folder name makes
/// it an embedding model (`model.poolingFromDirName`) stays on the plugin path.
pub const decoder_embedders = [_][]const u8{"qwen3"};

pub fn servesDecoderEmbedding(model_type: []const u8) bool {
    for (decoder_embedders) |t| if (std.mem.eql(u8, t, model_type)) return true;
    return false;
}

pub fn serves(model_type: []const u8) bool {
    for (types) |t| if (std.mem.eql(u8, t, model_type)) return true;
    return false;
}

test "served: the listed types only" {
    try std.testing.expect(serves("llama") and serves("qwen2") and serves("laguna") and serves("gemma4") and serves("diffusion_gemma") and serves("gemma3_text") and !serves(""));
}
