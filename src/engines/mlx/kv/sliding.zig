//! engines/mlx/kv/sliding, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns SLIDING_TRIM_UNBOUNDED, slidingTailSpan, sliding_block_trim_cached, slidingBlockTrimEnabled, sliding_block_trim_logged, sliding_ring_off_for_test, and 1 more.

const std = @import("std");
const log = @import("../../../host/log.zig");
const ModelConfig = @import("../model.zig").ModelConfig;
const FUSED256_MIN_Q_LEN = @import("../../../sdk/blocks/attention/attn256.zig").FUSED256_MIN_Q_LEN;
const fused256Enabled = @import("../../../sdk/blocks/attention/attn256.zig").fused256Enabled;
const SlidingView = @import("../../../sdk/blocks/attention/kv_view.zig").SlidingView;

/// "No width bound" for `slidingTailSpan`. A 32-wide cap used to keep prefill
/// chunks out while the mask pairing was unproven; with `band_in_kernel` now
/// gating the only consumer that reads ABSOLUTE positions and every other
/// consumer deriving its query offset as `kv_len - q_len`, width needs no
/// bound — and wide chunks are the only multi-token forward a serial arch
/// (laguna, inkling) ever does.
pub const SLIDING_TRIM_UNBOUNDED: usize = std.math.maxInt(usize);

/// How many TAIL KV entries a forward of `q_len` queries needs from a SLIDING
/// layer. The last query reaches back `window`; the first of the block reaches
/// back `window + q_len - 1`, so the block's union is that many entries.
/// Returns 0 — meaning "hand back the whole cache, no trim" — when the layer
/// does not slide or the block is wider than `max_width`.
///
/// This is what makes a spec verify cost what a decode step costs on a sliding
/// layer: at width 1 it has always returned `window`, but a block-wide verify
/// used to fall through untrimmed and read the ENTIRE cache on every sliding
/// layer, so a 3:1 local:global model paid full attention on 75% of its layers
/// once past the window (measured: muse-glimmer verify 51 -> 140 ms from 0.6k
/// to 17.6k ctx, turning DFlash from 2.28x into 0.74x).
pub fn slidingTailSpan(window: u32, q_len: usize, max_width: usize) u32 {
    if (window == 0 or q_len == 0 or q_len > max_width) return 0;
    return window + @as(u32, @intCast(q_len)) - 1;
}

/// Kill switch for the block-width half of the trim (MLX_SERVE_SLIDING_BLOCK_TRIM=0
/// pins every multi-token forward back to the untrimmed view). Decode-width
/// trimming is unconditional and predates this.
var sliding_block_trim_cached: ?bool = null;
pub fn slidingBlockTrimEnabled() bool {
    if (sliding_block_trim_cached) |v| return v;
    const raw = std.c.getenv("MLX_SERVE_SLIDING_BLOCK_TRIM");
    const on = raw == null or !std.mem.eql(u8, std.mem.sliceTo(raw.?, 0), "0");
    sliding_block_trim_cached = on;
    return on;
}

var sliding_block_trim_logged: bool = false; // one-shot log guard

/// Test seam: the full-buffer reference arm of the sliding-ring equivalence test.
pub var sliding_ring_off_for_test = false;

pub fn slidingViewFor(cfg: *const ModelConfig, total_kv: c_int, seq_len: c_int) SlidingView {
    // Keyed on the kernel's OWN floor: it declines everything below
    // FUSED256_MIN_Q_LEN, so a spec-verify block (4-8 wide) is never
    // band-masked in-kernel and is free to take the trim. The predicate used
    // to say `seq_len >= 2`, which declined the trim for exactly those blocks
    // and then fell through to the explicit-mask path anyway — gemma got
    // nothing from the trim at all.
    const band = fused256Enabled() and cfg.head_dim == 256 and seq_len >= FUSED256_MIN_Q_LEN;
    // A sliding ring holds only the tail, so it always trims.
    const width: usize = if (band or (!slidingBlockTrimEnabled() and !cfg.slidingRing())) 1 else SLIDING_TRIM_UNBOUNDED;
    const span: u32 = if (cfg.has_sliding_window)
        slidingTailSpan(cfg.sliding_window, @intCast(seq_len), width)
    else
        0;
    const kv_len: c_int = if (span > 0) @min(total_kv, @as(c_int, @intCast(span))) else total_kv;
    // One-shot: an A/B arm is proven by an engagement line in its own log, not
    // by the env it was launched with. Fires only when the BLOCK half of the
    // trim did something — decode has always trimmed to the window.
    if (!sliding_block_trim_logged and span > cfg.sliding_window and kv_len < total_kv) {
        sliding_block_trim_logged = true;
        log.info("[sliding] block trim engaged: window={d} q_len={d} span={d} kv={d} of {d} (MLX_SERVE_SLIDING_BLOCK_TRIM=0 restores the untrimmed view)\n", .{ cfg.sliding_window, seq_len, span, kv_len, total_kv });
    }
    return .{ .span = span, .kv_len = kv_len, .band_in_kernel = band };
}
