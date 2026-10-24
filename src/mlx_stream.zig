//! mlx-stream: the native streaming stack as one plugin: the DeepSeek-V4.1 arch, the EXL3 quant it binds (the routed
//! experts' kernels over the package's pinned registry) and the EXL3 expert source that streams them; MiMo-V2.6-Flash
//! and GLM-5.3 over the same expert reader. Built only where the macOS-only sources are. Absorbed from davidtai's
//! mlx-stream: the archs live in `plugins/<model>/stream/`, the EXL3 lanes in `sdk/quant/exl3/stream/`, the expert
//! streamer in `sdk/weights/streamed/`. The module imports its own `sdk` (`sdk/weights/streamed/sdk/`) and the host's
//! `mlx_host`.

pub const plugin = sdk.Plugin{
    .name = "mlx-stream",
    .api = .{ .major = 2, .minor = 1 },
    // The MLX this plugin is tested on; the conformance suite fails when the host links another.
    .mlx = "v0.32.3",
    .macos_only = true,
    // The host registers the arch; the EXL3 quant and the EXL3 expert source are its internals, bound at comptime
    // (sdk_ext.zig).
    .provides = .{ .arch = @import("plugins/deepseek_v41/stream/plugin.zig") },
};

/// MiMo-V2.6-Flash's prepared MXFP4 streaming pack: the package's second arch, over the same expert reader.
pub const plugin_mimo = sdk.Plugin{
    .name = "mlx-stream-mimo",
    .api = .{ .major = 2, .minor = 0 },
    .mlx = "v0.32.3",
    .macos_only = true,
    .provides = .{ .arch = @import("plugins/mimo_v2/stream/root.zig") },
};

/// What the host's glue calls for the second arch: its entry points, the context a load bills when the model sets
/// none, and the positions a request may generate past that context (none: MiMo reserves generation inside it).
pub const mimo = struct {
    pub const arch = @import("plugins/mimo_v2/stream/root.zig");
    pub const default_context: u64 = @import("plugins/mimo_v2/stream/settings.zig").fill_reserve_tokens;
    pub const generation_headroom: u64 = 0;
    /// The FP8 residents' kernels (`sdk/quant/fp8_block128.zig` is held bit-equal to their prompt route).
    pub const fp8_linear = @import("plugins/mimo_v2/stream/fp8_linear.zig");
};

/// What the host's glue calls for the third arch: GLM-5.3's entry points, the context a load bills when the model sets
/// none (the standard request) and the generation a request may take past it until a parse has read the model's
/// `max_output` (its default, 128K: the glue reads the arch's `maxOutput` once it has the model's settings).
pub const glm = struct {
    pub const arch = @import("plugins/glm_moe_dsa/stream/plugin.zig");
    pub const default_context: u64 = @import("sdk/weights/streamed/phase_memory.zig").fill_prompt_tokens;
    pub const generation_headroom: u64 = @import("plugins/glm_moe_dsa/stream/settings.zig").Config.default_max_output;
};

/// What the host's glue calls for the fourth arch: GLM-5.3-Flash (`glm5_next`, the NVFP4 pack): its entry points, the
/// context a load bills when the model sets none (the standard request: 16,384 prompt tokens and 1,024 generated) and
/// the positions a request may generate past it (none: the context holds the generation, as MiMo's does).
pub const glm5_next = struct {
    pub const arch = @import("plugins/glm5_next/stream/root.zig");
    pub const default_context: u64 = @import("plugins/glm5_next/stream/settings.zig").fill_reserve_tokens;
    pub const generation_headroom: u64 = 0;
};

/// What the host's glue (`engines/mlx/modules/stream_glue.zig`) calls: the arch's entry points and their contract types.
pub const arch = @import("plugins/deepseek_v41/stream/plugin.zig");
/// Every arch the package serves through the SDK's table (`sdk.Arch.of`), in claim order: DeepSeek-V4.1 and GLM-5.3
/// (the conformance suite's list; the host's glue binds each arch by its own entry: the root's, `mimo` and `glm`).
pub const archs = .{ @import("plugins/deepseek_v41/stream/plugin.zig"), @import("plugins/glm_moe_dsa/stream/plugin.zig") };
pub const sdk = @import("sdk");
/// DeepSeek-V4.1's EXL3 repack experts for the resident trunk (the fold moves them into `sdk/quant/exl3/`).
pub const dsv41_streamed = @import("plugins/deepseek_v41/stream/streamed.zig");
/// The expert bank source a resident trunk streams its routed experts from (`sdk/weights/streamed.zig` `Streamed`).
pub const bank_source = @import("sdk/weights/streamed/bank_source.zig");
/// A standard MLX pack's routed experts streamed for the resident trunk as downloaded (`Streamed.openPack`).
pub const mlx_pack = @import("sdk/weights/streamed/mlx_pack.zig");
/// The streamed halves of the dupes parity harnesses (test builds only): each arch's trunk over a pack, for the main
/// test module's `plugins/<model>/stream/dupes_test.zig` to compare against the resident arch.
pub const dupes = if (@import("builtin").is_test) struct {
    pub const deepseek_v41 = @import("plugins/deepseek_v41/stream/dupes.zig");
} else struct {};
/// The context a construction bills when the model sets none (`ctx_size`): the standard request.
pub const default_context: u64 = @import("plugins/deepseek_v41/stream/bill.zig").fill_prompt_tokens;
/// Positions a request may generate past its prompt (the KV lanes' bound) for an arch without its own `maxOutput`
/// (DeepSeek-V4.1's).
pub const generation_headroom: u64 = @import("plugins/deepseek_v41/stream/module.zig").generation_headroom;
