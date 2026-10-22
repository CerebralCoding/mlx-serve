//! qwen4_exp lazy PLE settle switch.
const std = @import("std");


var ple_lazy_settle_env: ?bool = null;
/// `MLX_SERVE_PLE_LAZY_SETTLE=0` restores the per-step host settle on the GPU PLE arm.
pub fn pleLazySettleOff() bool {
    if (ple_lazy_settle_env) |v| return v;
    const raw = std.c.getenv("MLX_SERVE_PLE_LAZY_SETTLE");
    const off = raw != null and raw.?[0] == '0';
    ple_lazy_settle_env = off;
    return off;
}
