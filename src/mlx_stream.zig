//! mlx-stream: the native streaming stack as one plugin: GLM-5.3's arch, the EXL3 quant (the routed experts' kernels
//! over the package's pinned registry) and the EXL3 expert source that streams them; GLM-5.3 and GLM-5.3-Flash
//! over the same expert reader, and the bank sources a resident trunk streams its experts from. Built only where the
//! macOS-only sources are. Absorbed from davidtai's mlx-stream: the archs live in `plugins/<model>/stream/`, the EXL3
//! lanes in `sdk/quant/exl3/stream/`, the expert streamer in `sdk/weights/streamed/`. The module imports its own `sdk`
//! (`sdk/weights/streamed/sdk/`) and the host's `mlx_host`.

pub const plugin = sdk.Plugin{
    .name = "mlx-stream",
    .api = .{ .major = 2, .minor = 1 },
    // The MLX this plugin is tested on; the conformance suite fails when the host links another.
    .mlx = "v0.32.3",
    .macos_only = true,
    // The host registers the arch; the EXL3 quant and the EXL3 expert source are its internals, bound at comptime
    // (sdk_ext.zig).
    .provides = .{ .arch = @import("plugins/glm_moe_dsa/stream/plugin.zig") },
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
/// the positions a request may generate past it (none: the context holds the generation).
pub const glm5_next = struct {
    pub const arch = @import("plugins/glm5_next/stream/root.zig");
    pub const default_context: u64 = @import("plugins/glm5_next/stream/settings.zig").fill_reserve_tokens;
    pub const generation_headroom: u64 = 0;
};

/// Every arch the package serves through the SDK's table (`sdk.Arch.of`), in claim order (the conformance suite's list;
/// the host's glue binds each arch by its own entry: `glm` and `glm5_next`).
pub const archs = .{@import("plugins/glm_moe_dsa/stream/plugin.zig")};
pub const sdk = @import("sdk");
/// The expert bank source a resident trunk streams its routed experts from (`sdk/weights/streamed.zig` `Streamed`).
pub const bank_source = @import("sdk/weights/streamed/bank_source.zig");
/// A standard MLX pack's routed experts streamed for the resident trunk as downloaded (`Streamed.openPack`).
pub const mlx_pack = @import("sdk/weights/streamed/mlx_pack.zig");
/// An EXL3 expert bank streamed for a resident trunk (`sdk/weights/streamed.zig` `Exl3`).
pub const exl3_source = @import("sdk/weights/streamed/exl3_source.zig");
/// The streamed halves of the dupes parity harnesses (test builds only): each arch's trunk over a pack, for the main
/// test module's `plugins/<model>/stream/dupes_test.zig` to compare against the resident arch.
pub const dupes = if (@import("builtin").is_test) struct {
} else struct {};
