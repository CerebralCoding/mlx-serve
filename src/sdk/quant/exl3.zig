//! sdk/quant/exl3, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns checkExl3Bank.

const mlx = @import("../../mlx/ffi.zig");
const sushi_exl3 = @import("sushi_exl3");
const ModelConfig = @import("../../engines/mlx/model.zig").ModelConfig;
const MoeMlpWeights = @import("../blocks/moe/weights.zig").MoeMlpWeights;

/// An EXL3 bank's suh/svh bind optionally beside the affine names, so their
/// absence and a trellis the kernels cannot decode at the config's rate refuse here.
pub fn checkExl3Bank(mw: *const MoeMlpWeights, config: ModelConfig, spec: sushi_exl3.Spec) !void {
    const h = config.hidden_size;
    const inter = config.moe_intermediate_size;
    const projs = [_]struct { w: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array, in: u32, out: u32 }{
        .{ .w = mw.switch_gate_w, .suh = mw.switch_gate_s, .svh = mw.switch_gate_b, .in = h, .out = inter },
        .{ .w = mw.switch_up_w, .suh = mw.switch_up_s, .svh = mw.switch_up_b, .in = h, .out = inter },
        .{ .w = mw.switch_down_w, .suh = mw.switch_down_s, .svh = mw.switch_down_b, .in = inter, .out = h },
    };
    for (projs) |p| {
        if (p.suh.ctx == null or p.svh.ctx == null) return error.MissingWeight;
        if (!sushi_exl3.trellisAdmitted(mlx.getShape(p.w), config.num_experts, p.in, p.out, spec.rate)) return error.Exl3TrellisGeometry;
    }
}
