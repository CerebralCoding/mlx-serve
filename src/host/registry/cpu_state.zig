//! CPU-side load state (config, tokenizer, chat config) built off the inference thread.

const std = @import("std");
const tokenizer_mod = @import("../../sdk/tokenizer/tokenizer.zig");
const gen_mod = @import("../media_jobs/gen.zig");
const model_mod = @import("../../engines/mlx/model.zig");
const chat_mod = @import("../chat/chat.zig");
const model_discovery = @import("discovery.zig");
const gguf_meta = @import("../../sdk/quant/gguf/engine_pick.zig");
const mlx_gguf = @import("../../sdk/quant/gguf/claim.zig");
const log = @import("../../sdk/util/log.zig");
const ModelConfig = model_mod.ModelConfig;
const Tokenizer = tokenizer_mod.Tokenizer;
const ChatConfig = chat_mod.ChatConfig;

const buildGgufStubCpuState = @import("../../engines/ds4/adapter.zig").buildGgufStubCpuState;
const GgufRoute = @import("../gpu_thread/thread.zig").GgufRoute;

/// Plan 05 Phase D: pre-loaded CPU state bundle. Built by the conn thread
/// (CPU only — file I/O + parse, no mlx) ahead of posting a LoadRequest.
/// Ownership transfers to the entry on successful load; on failure the
/// conn thread frees via `freeCpuState`. `gguf` (when set) stays owned by
/// the conn thread either way.
pub const CpuState = struct {
    config: *ModelConfig,
    tok: *Tokenizer,
    chat_config: *ChatConfig,
    /// Non-null when the model is a GGUF served by an embedded engine
    /// (issue #59: discovered/pulled GGUF dirs cold-load on demand).
    gguf: ?GgufRoute = null,
};

/// Phase D: parse config.json, tokenizer, and chat config from `model_dir`
/// into heap pointers ready to hand off to `LoadRequest`. Mirrors the
/// pre-load that main.zig does for the startup model in serve mode.
///
/// Errdefer pattern: for each `try ... else error`, the `deinit` errdefer
/// is registered AFTER the successful init so a downstream failure doesn't
/// call deinit on uninitialized memory. `ModelConfig.deinit` frees the config's
/// one owned field; `allocator.destroy` alone would leak it.
/// `gguf_engine` (the model's engine setting) forces a GGUF's embedded engine over its header and src/sdk/quant/gguf.
pub fn preloadCpuState(allocator: std.mem.Allocator, io: std.Io, model_dir: []const u8, gguf_ctx_size: u32, gguf_engine: ?gguf_meta.Engine) !CpuState {
    // GGUF first — mirrors `--model` routing in main.zig, where isGgufPath
    // is checked before any config.json read ("GGUF files bypass the MLX
    // dispatch entirely"). The embedded engine owns the real tokenizer +
    // chat template, so the CPU state is a stub, like the media path below.
    const mlx_gguf_path = if (gguf_engine == null) mlx_gguf.servablePath(io, allocator, model_dir) else null;
    defer if (mlx_gguf_path) |p| allocator.free(p);
    if (mlx_gguf_path == null and model_discovery.isGgufModelPath(io, model_dir)) {
        return preloadGgufCpuState(allocator, io, model_dir, gguf_ctx_size, gguf_engine);
    }

    // Media model (image/audio/video): the engine owns the real tokenizer +
    // forward path, so we hand the inference thread a minimal stub instead of
    // parsing a transformer-shaped config (which would fail on a flux2/
    // qwen3_tts/AudioVideo config.json). The gen load arm dispatches off the
    // stub's `model_type`.
    if (gen_mod.detectModality(io, allocator, model_dir)) |modality| {
        const stub = try gen_mod.buildStubCpuState(allocator, modality);
        return .{ .config = stub.config, .tok = stub.tok, .chat_config = stub.chat_config };
    }
    // A media-typed dir that FAILED its marker check must not fall through to
    // the text path below: it would glob whatever safetensors are present and
    // die on the first missing weight. Refuse by name instead (#144 flow).
    if (gen_mod.incompleteMediaDir(io, allocator, model_dir)) {
        return error.IncompleteMediaPack;
    }
    return preloadTextCpuState(allocator, io, model_dir);
}

fn preloadTextCpuState(allocator: std.mem.Allocator, io: std.Io, model_dir: []const u8) !CpuState {
    const config = try allocator.create(ModelConfig);
    errdefer allocator.destroy(config);
    config.* = try model_mod.parseConfig(io, allocator, model_dir);

    const tok = try allocator.create(Tokenizer);
    errdefer allocator.destroy(tok);
    tok.* = try tokenizer_mod.loadTokenizerAny(io, allocator, model_dir);
    errdefer tok.deinit();

    const cc = try allocator.create(ChatConfig);
    errdefer allocator.destroy(cc);
    cc.* = try chat_mod.loadChatConfig(io, allocator, model_dir);
    errdefer cc.deinit();

    config.applyTokenizer(tok, cc.eos_token);
    if (config.radio_vision) cc.video_placeholder = "<img>";

    return .{ .config = config, .tok = tok, .chat_config = cc };
}

/// Frees the three CPU-state pointers. Does NOT free `s.gguf` — the
/// resolved .gguf path is borrowed by the LoadRequest until `done` and is
/// freed by ensureLoaded's own defer on both success and failure paths
/// (on success the three pointers transfer to the entry, so this function
/// is skipped, but the path must still be released).
pub fn freeCpuState(allocator: std.mem.Allocator, s: *CpuState) void {
    s.config.deinit(allocator);
    allocator.destroy(s.config);
    s.tok.deinit();
    allocator.destroy(s.tok);
    s.chat_config.deinit();
    allocator.destroy(s.chat_config);
}

/// GGUF cold-load preload: resolve the actual .gguf file, pick the embedded
/// engine from its header metadata (the issue #15 rule in gguf_meta.zig;
/// unreadable metadata defaults to llama with a log line, mirroring
/// main.zig's chooseGgufEngine), and build the stub CPU state.
fn preloadGgufCpuState(allocator: std.mem.Allocator, io: std.Io, model_dir: []const u8, ctx_size: u32, forced: ?gguf_meta.Engine) !CpuState {
    const gguf_path = model_discovery.resolveGgufFile(io, allocator, model_dir) catch |err| {
        model_discovery.logResolveGgufError(model_dir, err);
        return err;
    };
    errdefer allocator.free(gguf_path);

    const engine: gguf_meta.Engine = forced orelse blk: {
        var info = gguf_meta.readFromFile(io, allocator, gguf_path) catch |err| {
            log.warn("[gguf] route: metadata read failed ({s}); defaulting to llama\n", .{@errorName(err)});
            break :blk .llama;
        };
        defer info.deinit(allocator);
        const e = gguf_meta.preferredEngine(info);
        log.info("[gguf] engine: {s} (arch={s}, ds4-lora={}, shards={d})\n", .{
            @tagName(e),
            info.architecture orelse "?",
            info.has_ds4_lora_rank,
            info.split_count,
        });
        break :blk e;
    };

    var state = try buildGgufStubCpuState(allocator, engine, ctx_size);
    state.gguf = .{ .path = gguf_path, .engine = engine };
    return state;
}
