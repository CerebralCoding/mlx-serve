//! The streamed weight source: routed experts read from SSD banks into slot rows, any MoE pack that declares
//! `expert_streaming`. mlx-stream serves those packs (`streamed/glue.zig`); until its banks move here, opening one
//! through this source is refused by name.

const std = @import("std");
const abi = @import("../../abi/quant.zig");

pub const Streamed = struct {
    pub const refusal = abi.Verdict.refuse(.unsupported, "streamed weights: the pack's experts stream through mlx-stream");

    pub fn open(model_dir: []const u8) error{StreamedWeightsUnsupported}!Streamed {
        _ = model_dir;
        return error.StreamedWeightsUnsupported;
    }
};

// ── tests ──

test "streamed: opening a pack is refused by name" {
    try std.testing.expectError(error.StreamedWeightsUnsupported, Streamed.open("/nowhere"));
    try std.testing.expectEqual(abi.Verdict.Code.unsupported, Streamed.refusal.code);
}
