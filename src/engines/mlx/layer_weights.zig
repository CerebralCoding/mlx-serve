//! engines/mlx/layer_weights, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns LayerWeights, FullAttnWeights, QW.

const mlx = @import("../../mlx/ffi.zig");
const DsaWeights = @import("../../plugins/glm5_next/weights.zig").DsaWeights;
const FusedRows = @import("../../sdk/blocks/linear/fused_rows.zig").FusedRows;

// ── Standard decoder-only layer weights ──

pub const LayerWeights = struct {
    input_norm: mlx.mlx_array,
    post_attn_norm: mlx.mlx_array,
    pre_ff_norm: ?mlx.mlx_array,
    post_ff_norm: ?mlx.mlx_array,
    q_norm: ?mlx.mlx_array,
    k_norm: ?mlx.mlx_array,
    q_w: mlx.mlx_array,
    q_s: mlx.mlx_array,
    q_b: mlx.mlx_array,
    // Additive qkv-projection biases (Qwen2's `q/k/v_proj.bias`), distinct from
    // the quant `*_b` biases above. Empty-ctx when the arch has none (qwen3,
    // llama, mistral). Applied via `qmatmulMaybeBias` in the forward.
    q_bias: mlx.mlx_array = .{},
    k_bias: mlx.mlx_array = .{},
    v_bias: mlx.mlx_array = .{},
    k_w: mlx.mlx_array,
    k_s: mlx.mlx_array,
    k_b: mlx.mlx_array,
    v_w: mlx.mlx_array,
    v_s: mlx.mlx_array,
    v_b: mlx.mlx_array,
    o_w: mlx.mlx_array,
    o_s: mlx.mlx_array,
    o_b: mlx.mlx_array,
    gate_w: mlx.mlx_array,
    gate_s: mlx.mlx_array,
    gate_b: mlx.mlx_array,
    up_w: mlx.mlx_array,
    up_s: mlx.mlx_array,
    up_b: mlx.mlx_array,
    down_w: mlx.mlx_array,
    down_s: mlx.mlx_array,
    down_b: mlx.mlx_array,
    // Gemma 4: per-layer scalar, PLE weights
    layer_scalar: ?mlx.mlx_array = null,
    ple_gate_w: ?mlx.mlx_array = null,
    ple_gate_s: ?mlx.mlx_array = null,
    ple_gate_b: ?mlx.mlx_array = null,
    ple_proj_w: ?mlx.mlx_array = null,
    ple_proj_s: ?mlx.mlx_array = null,
    ple_proj_b: ?mlx.mlx_array = null,
    ple_norm: ?mlx.mlx_array = null,
    // KV sharing: source layer index (null = compute own KV)
    kv_source: ?u32 = null,
    // Gemma 4 (31B): V aliases K projection within this layer (no v_proj weight loaded)
    k_eq_v: bool = false,
    // MuseGlimmer: elementwise sigmoid attention output gate
    // (self_attn.gate_proj, read off the post-input-norm hidden). Null ag_w
    // on every other arch.
    ag_w: ?mlx.mlx_array = null,
    ag_s: mlx.mlx_array = .{ .ctx = null },
    ag_b: mlx.mlx_array = .{ .ctx = null },
    // Row-joined [q | k | v (| ag)] and [gate | up] for decode widths; the
    // separate fields above are views into these once built (fuseRowGroup).
    qkv: FusedRows = .{},
    gu: FusedRows = .{},
};

// ── MoE model per-layer weights ──

pub const FullAttnWeights = struct {
    q_w: mlx.mlx_array,
    q_s: mlx.mlx_array,
    q_b: mlx.mlx_array,
    k_w: mlx.mlx_array,
    k_s: mlx.mlx_array,
    k_b: mlx.mlx_array,
    v_w: mlx.mlx_array,
    v_s: mlx.mlx_array,
    v_b: mlx.mlx_array,
    o_w: mlx.mlx_array,
    o_s: mlx.mlx_array,
    o_b: mlx.mlx_array,
    q_norm: mlx.mlx_array,
    k_norm: mlx.mlx_array,
    /// Row-joined [q | k | v] for decode width; q/k/v are views of it once
    /// built (fuseRowGroup). Only the hybrid loader builds it so far.
    qkv: FusedRows = .{},
    // Per-head attention OUTPUT gate (self_attn.g_proj / attention.g_proj →
    // [B,S,n_heads]). The ACTIVATION is the arch's, not the weight's: Laguna
    // applies softplus, bailing_hybrid's MLA a sigmoid. Null-ctx on every
    // other arch → skipped by appendFullAttnWeights and by the gate branches.
    g_w: mlx.mlx_array = .{ .ctx = null },
    g_s: mlx.mlx_array = .{ .ctx = null },
    g_b: mlx.mlx_array = .{ .ctx = null },
    // Multi-head Latent Attention (bailing_hybrid). When `mla_q_a_w` is
    // non-null this layer projects through a low-rank Q and a compressed KV
    // latent instead of q/k/v — those three fields stay null-ctx and
    // `mlaAttnWith` runs in place of gatedFullAttnWith. `o_w` holds the
    // checkpoint's `dense` (the output projection) as usual.
    /// The direct Q projection, used INSTEAD of the q_a/q_b pair when the
    /// config carries `q_lora_rank: null` (`ModelConfig.mlaHasQLora`). Exactly
    /// one of `mla_q_direct_w` and `mla_q_a_w` is non-null on an MLA layer.
    mla_q_direct_w: mlx.mlx_array = .{ .ctx = null },
    mla_q_direct_s: mlx.mlx_array = .{ .ctx = null },
    mla_q_direct_b: mlx.mlx_array = .{ .ctx = null },
    mla_q_a_w: mlx.mlx_array = .{ .ctx = null },
    mla_q_a_s: mlx.mlx_array = .{ .ctx = null },
    mla_q_a_b: mlx.mlx_array = .{ .ctx = null },
    mla_q_a_norm: mlx.mlx_array = .{ .ctx = null },
    mla_q_b_w: mlx.mlx_array = .{ .ctx = null },
    mla_q_b_s: mlx.mlx_array = .{ .ctx = null },
    mla_q_b_b: mlx.mlx_array = .{ .ctx = null },
    mla_kv_a_w: mlx.mlx_array = .{ .ctx = null },
    mla_kv_a_s: mlx.mlx_array = .{ .ctx = null },
    mla_kv_a_b: mlx.mlx_array = .{ .ctx = null },
    mla_kv_a_norm: mlx.mlx_array = .{ .ctx = null },
    mla_kv_b_w: mlx.mlx_array = .{ .ctx = null },
    mla_kv_b_s: mlx.mlx_array = .{ .ctx = null },
    mla_kv_b_b: mlx.mlx_array = .{ .ctx = null },
    // Inkling (inkling_mm_model): the relative-logits projection (attn.wr_du
    // → per-head d_rel "relative states"), the learned bias-vs-distance bank
    // (attn.rel_logits_proj.proj [d_rel, extent]), and the k/v depthwise
    // causal short-convolution taps (attn.{k,v}_sconv.weight [C, k, 1], fp32
    // compute). Null-ctx on every other arch.
    rel_w: mlx.mlx_array = .{ .ctx = null },
    rel_s: mlx.mlx_array = .{ .ctx = null },
    rel_b: mlx.mlx_array = .{ .ctx = null },
    rel_proj: mlx.mlx_array = .{ .ctx = null },
    k_sconv_w: mlx.mlx_array = .{ .ctx = null },
    v_sconv_w: mlx.mlx_array = .{ .ctx = null },
    // qwen4_exp QSA indexer: one projection for the index queries + the
    // single shared key head, and their per-head RMS norms. Null-ctx on
    // every other arch.
    idx_qk_w: mlx.mlx_array = .{ .ctx = null },
    idx_qk_s: mlx.mlx_array = .{ .ctx = null },
    idx_qk_b: mlx.mlx_array = .{ .ctx = null },
    idx_q_norm: mlx.mlx_array = .{ .ctx = null },
    idx_k_norm: mlx.mlx_array = .{ .ctx = null },
    // gpt_oss: ADDITIVE projection biases (`self_attn.{q,k,v,o}_proj.bias`).
    // These are NOT the affine quantizer's `*_b` zero-points above — a gpt_oss
    // checkpoint carries BOTH spellings on the same projection
    // (`q_proj.bias` next to `q_proj.biases`), with different shapes and
    // different meanings. Null-ctx on every other arch.
    q_bias: mlx.mlx_array = .{ .ctx = null },
    k_bias: mlx.mlx_array = .{ .ctx = null },
    v_bias: mlx.mlx_array = .{ .ctx = null },
    o_bias: mlx.mlx_array = .{ .ctx = null },
    // gpt_oss: learned per-head attention sinks (`self_attn.sinks`, [n_heads]).
    // One extra column in the softmax DENOMINATOR, letting a head attend to
    // "nothing". Passed straight to mlx's fused SDPA, which takes them as a
    // function constant (`_has_sinks_`) — no unfused fallback.
    sinks: mlx.mlx_array = .{ .ctx = null },
    /// GLM-5-Next DeepSeek sparse attention; null on every other arch.
    dsa: ?DsaWeights = null,
};

/// A projection triple (`scales`/`biases` null-ctx when dense). Borrowed handles.
pub const QW = struct {
    w: mlx.mlx_array = .{ .ctx = null },
    s: mlx.mlx_array = .{ .ctx = null },
    b: mlx.mlx_array = .{ .ctx = null },
};
