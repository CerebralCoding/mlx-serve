//! The model plugin table (`ModelTable`) and every struct that crosses it. Extern layouts, C calling convention, no
//! globals. Every struct opens with `size`: one passed by pointer grows at its end; one held by value or as an array
//! element is frozen per `version` (its `size` is a check). Per-model values ride `Parsed`; hook tables hold functions.

const Host = @import("host.zig").Host;
const std = @import("std");
const common = @import("common.zig");

pub const Span = common.Span;
pub const Str = common.Str;
pub const Verdict = common.Verdict;
pub const Array = common.Array;
pub const Stream = common.Stream;
pub const Claim = common.Claim;
pub const PackDir = common.PackDir;

pub const version: u32 = 1;

const nan = std.math.nan(f32);

/// An object the engine owns and the plugin only passes back (a slot's KV cache): opaque until P3 types it.
pub const Handle = ?*anyopaque;

/// The layers this process serves when a model is split across machines; null = all of them.
pub const LayerSpan = extern struct {
    size: u32 = @sizeOf(LayerSpan),
    first: u32 = 0,
    count: u32 = 0,
};

// ── Parsed: what `parse` fills ──

pub const DType = enum(u8) { unset, f16, bf16, f32 };
pub const Tri = enum(u8) { unset, off, on };
pub const QuantMode = enum(u8) { none, affine, nvfp4, mxfp4, mxfp8, gguf };

/// The config's `quantization` block.
pub const Quant = extern struct {
    size: u32 = @sizeOf(Quant),
    bits: u8 = 0,
    mode: QuantMode = .none,
    group_size: u32 = 0,
};

/// Sampling defaults from the pack (`generation_config.json`); NaN or -1 = unset.
pub const GenDefaults = extern struct {
    size: u32 = @sizeOf(GenDefaults),
    temperature: f32 = nan,
    top_p: f32 = nan,
    min_p: f32 = nan,
    top_k: i32 = -1,
    enable_thinking: Tri = .unset,
};

pub const max_eos = 16;

/// The generic shell: the only config the engine reads. Per-layer facts are `CacheHooks.layout`, not here.
pub const Shell = extern struct {
    size: u32 = @sizeOf(Shell),
    vocab_size: u32 = 0,
    hidden_size: u32 = 0,
    /// Leaves once every plugin bills `Bill.prefill_streams`.
    intermediate_size: u32 = 0,
    num_hidden_layers: u32 = 0,
    num_attention_heads: u32 = 0,
    num_key_value_heads: u32 = 0,
    head_dim: u32 = 0,
    /// Temporary (decisions.md): both leave once every plugin's prefill bill lands.
    num_experts_per_tok: u32 = 0,
    moe_intermediate_size: u32 = 0,
    max_position_embeddings: u64 = 0,
    /// The derived context the model serves.
    context_cap: u64 = 0,
    quant: Quant = .{},
    gen: GenDefaults = .{},
    bos: i64 = -1, // -1 = none
    eos: [max_eos]u32 = @splat(0),
    eos_count: u32 = 0,
    act_dtype: DType = .bf16,
    state_dtype: DType = .unset,
    /// For display only: the host never compares it (the plugin is the identity).
    model_type: Str = .{},
};

/// Model facts the host keeps once at load. Defaults are the generic model's.
pub const Caps = packed struct(u64) {
    batches_decode: bool = false,
    batch_per_slot_attn: bool = true,
    batch_pads_kv: bool = true,
    /// Per-request decode state on the module, not the KV cache: single-flight, prefix cache through restore only.
    owns_decode_state: bool = false,
    shared_readonly_state: bool = false,
    long_ctx: bool = false,
    moe: bool = false,
    block_diffusion: bool = false,
    prefix_cache: bool = true,
    /// The exactness contracts (P2 step 7): drafted == plain, restored == cold, batched == solo.
    exact_drafted: bool = false,
    exact_resumed: bool = false,
    exact_concurrent: bool = false,
    /// The forward fills `Forward.capture_layers`.
    dflash_capture: bool = false,
    dflash_media: bool = false,
    spec_rollback: bool = false,
    spec_tree: bool = false,
    whole_prompt_prefill: bool = false,
    ane_prefill: bool = false,
    streamed_weights: bool = false,
    embed: bool = false,
    images: bool = false,
    video: bool = false,
    audio: bool = false,
    /// A pack another pack loads beside it (the ASR forced aligner).
    companion: bool = false,
    drafter: bool = false,
    _: u39 = 0,
};

pub const ThinkingDefault = enum(u8) { unset, off, on, on_with_tools };
pub const ReasoningDialect = enum(u8) { unset, think_tags, gemma_channel, harmony, muse, inkling, content_thinking };
/// The effort words a template understands; the tables stay in the host.
pub const EffortVocab = enum(u8) { none, generic, qwen38, kolibri, dsv4, dsv41, glm5, k2, inkling };
pub const RoundTrip = enum(u8) { keep, drop_prior_turns, required_field };
/// The dialect `tool_choice` enforcement commits an opener in; parsing stays dialect-agnostic in the host.
pub const ToolDialect = enum(u8) { unset, qwen_xml };

pub const NoThink = extern struct {
    size: u32 = @sizeOf(NoThink),
    template_refuses: bool = false,
    tail_suffix: Str = .{},
};

/// Family sampling defaults when the pack has no generation_config; NaN or -1 = none.
pub const SamplingDefaults = extern struct {
    size: u32 = @sizeOf(SamplingDefaults),
    top_k: i32 = -1,
    top_p: f32 = nan,
    wild_temp_override: f32 = nan,
};

/// Chat facts. A fact left `unset`/empty means the host falls back to its template probe.
pub const ChatFacts = extern struct {
    size: u32 = @sizeOf(ChatFacts),
    thinking_default: ThinkingDefault = .unset,
    reasoning_dialect: ReasoningDialect = .unset,
    effort_vocab: EffortVocab = .none,
    reasoning_round_trip: RoundTrip = .keep,
    tool_dialect: ToolDialect = .unset,
    nothink: NoThink = .{},
    template_family: Str = .{},
    sampling_defaults: SamplingDefaults = .{},
    video_placeholder: Str = .{},
};

pub const MarkerAlias = extern struct {
    size: u32 = @sizeOf(MarkerAlias),
    from: Str = .{},
    to: Str = .{},
};

pub const TokenizerFacts = extern struct {
    size: u32 = @sizeOf(TokenizerFacts),
    extra_eos: [max_eos]u32 = @splat(0),
    extra_eos_count: u32 = 0,
    /// The vocabulary rows that are real tokens; 0 = `vocab_size`.
    defined_vocab: u32 = 0,
    marker_aliases: Span(MarkerAlias) = .{},
};

/// How the engine keeps a recurrent layer's state across a pipelined step.
pub const SnapshotPolicy = enum(u8) { lagged, in_place };

pub const SsmState = extern struct {
    size: u32 = @sizeOf(SsmState),
    /// Per-request recurrent entries; 0 = none.
    entries: u32 = 0,
    dtype: DType = .unset,
    snapshot_policy: SnapshotPolicy = .lagged,
    checkpoint_bytes: u64 = 0,
};

/// `CacheHooks`' per-model values.
pub const CacheFacts = extern struct {
    size: u32 = @sizeOf(CacheFacts),
    /// A fresh SSD-cache root when stored K/V change meaning; empty = the default root.
    layout_namespace: Str = .{},
    /// Rows a sliding layer keeps (window + slack); 0 = the full buffer.
    sliding_ring: u32 = 0,
    ssm_state: SsmState = .{},
};

pub const MtpLayout = enum(u8) { none, sidecar, in_checkpoint, nemotron };

pub const SpecPolicy = extern struct {
    size: u32 = @sizeOf(SpecPolicy),
    drafter_ok: bool = true,
    mtp_group_solo: bool = false,
    seeded_sampling: bool = false,
    /// 0 = the machine's block.
    dflash_block: u32 = 0,
    /// The DFlash yield gate's floor in accepted tokens per round x10; 0 = the calibrated gate.
    dflash_floor_x10: u32 = 0,
    draft_temperature: f32 = nan,
};

/// `SpecHooks`' per-model values.
pub const SpecFacts = extern struct {
    size: u32 = @sizeOf(SpecFacts),
    mtp_default_depth: u8 = 0,
    mtp_layout: MtpLayout = .none,
    head_module_owned: bool = false,
    head_reads_final_norm: bool = false,
    draft_vocab: u32 = 0,
    /// An id into the engine's measured round-cost tables; 0 = generic.
    cost_profile: u32 = 0,
    native_stages: u32 = 0,
    native_block: u32 = 0,
    policy: SpecPolicy = .{},
};

/// Placeholder token ids a media item's rows replace; 0 = unset, never token 0.
pub const PlaceholderIds = extern struct {
    size: u32 = @sizeOf(PlaceholderIds),
    image: u32 = 0,
    video: u32 = 0,
    audio: u32 = 0,
    boi: u32 = 0,
    eoi: u32 = 0,
    boa: u32 = 0,
    eoa: u32 = 0,
    vision_start: u32 = 0,
    vision_end: u32 = 0,
};

pub const Pooling = enum(u8) { unset, mean, cls, last_token };

pub const EmbedFacts = extern struct {
    size: u32 = @sizeOf(EmbedFacts),
    pooling: Pooling = .unset,
    has_projection: bool = false,
};

pub const WeightFacts = extern struct {
    size: u32 = @sizeOf(WeightFacts),
    /// The weight-name prefix (probed for packs that differ); empty = none.
    prefix: Str = .{},
    /// The quant plugin that reads this pack's tensors (exl3, modelopt, hadamard, ...); empty = MLX's own.
    quant_layout: Str = .{},
};

/// Every `Str` here points at static text or at memory the plugin owns until `ModelTable.free_parsed`.
pub const Parsed = extern struct {
    size: u32 = @sizeOf(Parsed),
    caps: Caps = .{},
    shell: Shell = .{},
    chat: ChatFacts = .{},
    tokenizer: TokenizerFacts = .{},
    cache: CacheFacts = .{},
    spec: SpecFacts = .{},
    placeholder_ids: PlaceholderIds = .{},
    embed: EmbedFacts = .{},
    weights: WeightFacts = .{},
    /// The plugin's own parsed config, read by its `bill` and `load`.
    private: ?*anyopaque = null,
};

// ── Bill ──

/// The load-time shape `bill` prices.
pub const Shape = extern struct {
    size: u32 = @sizeOf(Shape),
    context: u64 = 0,
    slots: u32 = 1,
    /// KV precision: 0 = the activation dtype, else 8 or 4.
    kv_bits: u8 = 0,
};

/// The plugin's memory terms. A zero or null term is the engine's default arithmetic over the shell and layout.
pub const Bill = extern struct {
    size: u32 = @sizeOf(Bill),
    kv_per_token: u64 = 0,
    fixed_slot_bytes: u64 = 0,
    state_per_token: u64 = 0,
    prefill_streams: u64 = 0,
    load_bytes: u64 = 0,
    ssm_checkpoint: u64 = 0,
    prefill_score_width: u32 = 0,
    prefill_attn_keys: ?*const fn (p: *const Parsed, seq: u64) callconv(.c) u64 = null,
    prefill_bytes: ?*const fn (p: *const Parsed, seq: u64, chunk: u32, kv_bits: u8, warm: u64) callconv(.c) u64 = null,
    batched_kv_len: ?*const fn (p: *const Parsed, kv: u32, gather_on: bool, min: u32) callconv(.c) u32 = null,
};

// ── Load and forward ──

/// Valid during `load` only: the model keeps none of its pointers.
pub const LoadCtx = extern struct {
    size: u32 = @sizeOf(LoadCtx),
    model_dir: Str,
    dir: *const PackDir,
    parsed: *const Parsed,
    settings_json: Str = .{},
    span: ?*const LayerSpan = null,
    stream: Stream = .{},
    /// The host table (P2 step 8); opaque until it lands.
    host: ?*const Host = null,
};

pub const Mode = enum(u8) { prefill, decode, verify, batched_rows, canvas };
pub const Logits = enum(u8) { last_row, all_rows, none };

/// One forward. `tokens` holds `rows` tokens per slot, slot after slot; `slots`, `positions` and `caches` hold one
/// entry per slot (its id, its first position, its KV cache).
pub const Forward = extern struct {
    size: u32 = @sizeOf(Forward),
    mode: Mode,
    logits: Logits = .last_row,
    want_hidden: bool = false,
    rows: u32,
    tokens: Span(u32),
    slots: Span(u32) = .{},
    positions: Span(u32) = .{},
    caches: Span(Handle) = .{},
    /// Layers whose hidden rows the forward publishes (DFlash); filled into `ForwardOut.captured`.
    capture_layers: Span(u32) = .{},
    /// Media rows to splice at their placeholders, from `media_offset` on (chunked prefill resumes there).
    media: Array = .{},
    media_offset: u32 = 0,
    /// The M-RoPE position table; null = 1-D positions.
    mrope: Array = .{},
    /// How the forward reads and writes `caches`; null = a forward that keeps no state (an encoder).
    cache_ops: ?*const CacheOps = null,
};

// ── Cache access ──

/// One layer's keys and values as attention reads them: dense `[B, H, T, D]`, owned by the view only when `owned`
/// (`CacheOps.release` frees them), plus the stored quantized triples (borrowed) when the cache keeps them.
pub const KvView = extern struct {
    size: u32 = @sizeOf(KvView),
    k: Array = .{},
    v: Array = .{},
    owned: bool = false,
    /// The request reads the triples directly where a packed attention kernel serves the shape.
    packed_reads: bool = false,
    /// 0 = dense storage, else the quantized width of the triples below.
    bits: u8 = 0,
    group_size: u32 = 0,
    /// Axis-2 capacity of the triples' buffers (the packed kernels stride by it); 0 = unknown.
    capacity: u32 = 0,
    k_q: Array = .{},
    k_scales: Array = .{},
    k_biases: Array = .{},
    v_q: Array = .{},
    v_scales: Array = .{},
    v_biases: Array = .{},
};

/// A recurrent layer's states: the convolution window and the scan state, and what a verify forward leaves for the
/// partial-accept rollback: the conv input behind the window (`conv_input` [B, K-1+S, C]) and the state after every
/// row (`state_seq` [S, B, Hv, Dv, Dk]).
pub const StateSlot = enum(u8) { conv, ssm, conv_input, state_seq };

/// The engine's calls on a slot's cache (`Handle`), handed in `Forward.cache_ops`: blocks append and read KV and
/// recurrent state through it and never see the engine's types. Every `Array` passed in stays the caller's.
pub const CacheOps = extern struct {
    size: u32 = @sizeOf(CacheOps),
    /// Rows `layer` holds, counted from position 0 (a ring counts the rows it dropped); 0 before its first append.
    len: *const fn (cache: Handle, layer: u32) callconv(.c) u64,
    /// Appends `k`/`v` `[1, H, T, D]` at `layer`, then writes the view attention reads: the newest `keep` rows, 0 = all.
    append: *const fn (cache: Handle, layer: u32, k: Array, v: Array, keep: u32, s: Stream, out: *KvView) callconv(.c) Verdict,
    /// `append` on a sliding layer whose buffer holds only its newest `ring` rows (`ring >= window`).
    append_ring: *const fn (cache: Handle, layer: u32, k: Array, v: Array, keep: u32, window: u32, ring: u32, s: Stream, out: *KvView) callconv(.c) Verdict,
    /// The rows `layer` already holds (a layer that reads another layer's KV).
    view: *const fn (cache: Handle, layer: u32, s: Stream, out: *KvView) callconv(.c) Verdict,
    /// Frees what `out` of `append`/`view` owns.
    release: *const fn (view: *KvView) callconv(.c) void,
    /// Appends row `i` of the stacked `k`/`v` `[N, H, T, D]` to `caches[i]` (batched rows).
    append_rows: *const fn (caches: [*]const Handle, n: u32, layer: u32, k: Array, v: Array, keep: u32, s: Stream) callconv(.c) Verdict,
    /// The state, borrowed; a null array before its first write.
    state: *const fn (cache: Handle, layer: u32, slot: StateSlot) callconv(.c) Array,
    /// Replaces the state with a new reference on `value`.
    set_state: *const fn (cache: Handle, layer: u32, slot: StateSlot, value: Array) callconv(.c) Verdict,
};

/// The forward's results: every array is a new reference the host owns. `logits` is [R, vocab] with R the number
/// of slots (`.last_row`) or of tokens (`.all_rows`).
pub const ForwardOut = extern struct {
    size: u32 = @sizeOf(ForwardOut),
    logits: Array = .{},
    hidden: Array = .{},
    /// Host-allocated, `capture_layers.len` entries.
    captured: ?[*]Array = null,
};

// ── Hooks ──

pub const LayerKind = enum(u8) { none, full, sliding, linear, conv, ssm };

pub const LayerCache = extern struct {
    size: u32 = @sizeOf(LayerCache),
    kind: LayerKind = .full,
    /// No rotary position on this layer.
    nope: bool = false,
    /// The layer whose KV this one reads; -1 = its own.
    kv_source: i16 = -1,
    window: u32 = 0,
    kv_heads: u32 = 0,
    k_dim: u32 = 0,
    v_dim: u32 = 0,
};

/// A plugin's opaque per-slot state beside the KV cache, under the RAM prefix cache, the SSD tier and restores.
/// A snapshot is a plugin-owned `Handle`, freed by `free`.
pub const AuxHooks = extern struct {
    size: u32 = @sizeOf(AuxHooks),
    reserve: *const fn (m: *anyopaque, slot: u32, tokens: u64) callconv(.c) Verdict,
    commit: *const fn (m: *anyopaque, slot: u32, len: u64) callconv(.c) Verdict,
    snapshot: *const fn (m: *anyopaque, slot: u32, out: *Handle) callconv(.c) Verdict,
    restore_at: *const fn (m: *anyopaque, slot: u32, snap: Handle, pos: u64) callconv(.c) Verdict,
    slice: *const fn (m: *anyopaque, snap: Handle, len: u64, out: *Handle) callconv(.c) Verdict,
    handoff: *const fn (m: *anyopaque, from_slot: u32, to_slot: u32) callconv(.c) Verdict,
    free: *const fn (m: *anyopaque, snap: Handle) callconv(.c) void,
    bytes: *const fn (m: *anyopaque, snap: Handle) callconv(.c) u64,
    /// Writes `snap` into the host's `buf` (at least `bytes(snap)` long), setting `len`.
    serialize: *const fn (m: *anyopaque, snap: Handle, buf: [*]u8, cap: usize, len: *usize) callconv(.c) Verdict,
    deserialize: *const fn (m: *anyopaque, bytes: Str, out: *Handle) callconv(.c) Verdict,
};

pub const CacheHooks = extern struct {
    size: u32 = @sizeOf(CacheHooks),
    /// Writes one entry per served layer into `out` (`cap` entries) and returns the count; null = every layer full
    /// attention at the shell's geometry.
    layout: ?*const fn (p: *const Parsed, span: ?*const LayerSpan, out: [*]LayerCache, cap: u32) callconv(.c) u32 = null,
    aux: ?*const AuxHooks = null,
};

/// An MTP head's calls; `forward` takes the head's rows in `Forward.tokens` and their positions in `positions`.
pub const HeadHooks = extern struct {
    size: u32 = @sizeOf(HeadHooks),
    load: *const fn (m: *anyopaque, ctx: *const LoadCtx) callconv(.c) Verdict,
    activate: *const fn (m: *anyopaque, slot: u32) callconv(.c) Verdict,
    release: *const fn (m: *anyopaque, slot: u32) callconv(.c) void,
    forward: *const fn (m: *anyopaque, f: *const Forward, out: *ForwardOut) callconv(.c) Verdict,
    truncate: *const fn (m: *anyopaque, slot: u32, len: u64) callconv(.c) void,
    adopt: *const fn (m: *anyopaque, slot: u32, snap: Handle) callconv(.c) Verdict,
    mark: *const fn (m: *anyopaque, slot: u32) callconv(.c) void,
    draft_select: *const fn (m: *anyopaque, logits: Array, out: *Array) callconv(.c) Verdict,
    shortlist: ?*const fn (m: *anyopaque, hidden: Array, k: u32, out: *Array) callconv(.c) Verdict = null,
    rerank: ?*const fn (m: *anyopaque, hidden: Array, shortlist: Array, out: *Array) callconv(.c) Verdict = null,
    bill: *const fn (p: *const Parsed) callconv(.c) u64,
};

/// A drafter the model carries in its own forward (DSpark); stages and block are `SpecFacts`.
pub const NativeDrafter = extern struct {
    size: u32 = @sizeOf(NativeDrafter),
    step: *const fn (m: *anyopaque, f: *const Forward, out: *ForwardOut) callconv(.c) Verdict,
    verify: *const fn (m: *anyopaque, f: *const Forward, out: *ForwardOut) callconv(.c) Verdict,
};

pub const SpecHooks = extern struct {
    size: u32 = @sizeOf(SpecHooks),
    head: ?*const HeadHooks = null,
    native_drafter: ?*const NativeDrafter = null,
    /// Writes the verify widths to warm into `out` (`cap` entries); returns the count.
    warmup_shapes: ?*const fn (p: *const Parsed, out: [*]u32, cap: u32) callconv(.c) u32 = null,
    tile_lanes: ?*const fn (m: *anyopaque, lane: u32) callconv(.c) Verdict = null,
};

pub const MediaKind = enum(u8) { image, video, audio };

/// One prepared media item: its pixels (or samples) as a new reference the host owns, and its row plan.
pub const MediaItem = extern struct {
    size: u32 = @sizeOf(MediaItem),
    kind: MediaKind = .image,
    /// Rows the item adds to the prompt.
    rows: u32 = 0,
    /// Grid in patches: time, height, width.
    grid: [3]u32 = .{ 0, 0, 0 },
    /// Encoder calls the item takes (crops plus a thumbnail); 1 = a single pass.
    tiles: u32 = 1,
    pixels: Array = .{},
};

pub const MediaHooks = extern struct {
    size: u32 = @sizeOf(MediaHooks),
    /// The tower forward: the item's rows.
    tower: *const fn (m: *anyopaque, item: *const MediaItem, out: *Array) callconv(.c) Verdict,
    /// Decodes and resizes one item's `bytes` within the pixel `budget`.
    prepare: *const fn (p: *const Parsed, kind: MediaKind, bytes: Str, budget: u64, out: *MediaItem) callconv(.c) Verdict,
    /// Writes the item's placeholder token run into `out` (`cap` entries); returns its length.
    placeholder_rows: ?*const fn (p: *const Parsed, item: *const MediaItem, out: [*]u32, cap: u32) callconv(.c) u32 = null,
    /// The M-RoPE table over the expanded prompt `ids`.
    positions: ?*const fn (p: *const Parsed, ids: Span(u32), items: Span(MediaItem), out: *Array) callconv(.c) Verdict = null,
    video_prune: ?*const fn (m: *anyopaque, item: *const MediaItem, rows: Array, out: *Array) callconv(.c) Verdict = null,
};

pub const EmbedHooks = extern struct {
    size: u32 = @sizeOf(EmbedHooks),
    /// The projection head over pooled rows; pooling itself is `EmbedFacts.pooling`.
    project: ?*const fn (m: *anyopaque, pooled: Array, out: *Array) callconv(.c) Verdict = null,
};

pub const DistHooks = extern struct {
    size: u32 = @sizeOf(DistHooks),
    /// Splits the model across `hosts` machines: one span per host into `out`; returns the count.
    plan: *const fn (p: *const Parsed, hosts: u32, out: [*]LayerSpan, cap: u32) callconv(.c) u32,
    /// This machine's part of a forward.
    serve: *const fn (m: *anyopaque, f: *const Forward, out: *ForwardOut) callconv(.c) Verdict,
};

pub const WeightHooks = extern struct {
    size: u32 = @sizeOf(WeightHooks),
    /// After the weights load: folds, narrows, residents.
    transform: ?*const fn (m: *anyopaque) callconv(.c) Verdict = null,
    /// Whether the loader keeps tensor `key`; null = all.
    keep_key: ?*const fn (p: *const Parsed, key: Str) callconv(.c) bool = null,
};

// ── The table ──

/// A model plugin's table, exported as `exportName(name, version)`. `claims` and `parse` never reach MLX.
pub const ModelTable = extern struct {
    size: u32 = @sizeOf(ModelTable),
    version: u32 = version,
    name: Str,
    claims: *const fn (model_type: Str, config_json: Str, dir: *const PackDir) callconv(.c) Claim,
    /// `stub`: discovery's parse (no sidecar beyond what the shell needs). A refusal leaves nothing to free.
    parse: *const fn (dir: *const PackDir, config_json: Str, settings_json: Str, span: ?*const LayerSpan, stub: bool, out: *Parsed) callconv(.c) Verdict,
    /// Frees what `parse` allocated (`Parsed.private` and the text it points at).
    free_parsed: *const fn (p: *Parsed) callconv(.c) void,
    bill: *const fn (p: *const Parsed, shape: *const Shape, out: *Bill) callconv(.c) void,
    load: *const fn (ctx: *const LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque,
    unload: *const fn (m: *anyopaque) callconv(.c) void,
    /// A mode the model does not declare is refused `unsupported`.
    forward: *const fn (m: *anyopaque, f: *const Forward, out: *ForwardOut) callconv(.c) Verdict,
    /// Deferred post-sample work for `slot`.
    forward_settle: ?*const fn (m: *anyopaque, slot: u32) callconv(.c) Verdict = null,
    cache: *const CacheHooks,
    spec: ?*const SpecHooks = null,
    media: ?*const MediaHooks = null,
    embed: ?*const EmbedHooks = null,
    dist: ?*const DistHooks = null,
    weights: *const WeightHooks,
    /// Per-request arm counters for `[spec-stats]`, in memory the plugin owns until the next call on `m`.
    meters: ?*const fn (m: *anyopaque, out: *Str) callconv(.c) void = null,
};

/// The modes a model serves: prefill, decode and verify always; the rest as its caps declare.
pub fn declaresMode(caps: Caps, mode: Mode) bool {
    return switch (mode) {
        .prefill, .decode, .verify => true,
        .batched_rows => caps.batches_decode,
        .canvas => caps.block_diffusion,
    };
}

// ── tests ──

const testing = std.testing;

test "abi plugin: caps fit one word with the generic defaults" {
    try testing.expectEqual(@as(usize, 8), @sizeOf(Caps));
    const c: Caps = .{};
    try testing.expect(c.prefix_cache and c.batch_per_slot_attn and c.batch_pads_kv and !c.batches_decode);
}

test "abi plugin: every struct opens with its size, set by default" {
    inline for (.{ LayerSpan, Quant, GenDefaults, Shell, NoThink, SamplingDefaults, ChatFacts, MarkerAlias, TokenizerFacts, SsmState, CacheFacts, SpecPolicy, SpecFacts, PlaceholderIds, EmbedFacts, WeightFacts, Parsed, Shape, Bill, ForwardOut, KvView, LayerCache, CacheHooks, SpecHooks, EmbedHooks, WeightHooks, MediaItem }) |T| {
        try testing.expectEqual(@as(usize, 0), @offsetOf(T, "size"));
        const v: T = .{};
        try testing.expectEqual(@as(u32, @sizeOf(T)), v.size);
    }
    inline for (.{ LoadCtx, Forward, CacheOps, AuxHooks, HeadHooks, NativeDrafter, MediaHooks, DistHooks, ModelTable }) |T|
        try testing.expectEqual(@as(usize, 0), @offsetOf(T, "size"));
}

test "abi plugin: unset generation defaults read as unset" {
    const g: GenDefaults = .{};
    try testing.expect(std.math.isNan(g.temperature) and std.math.isNan(g.top_p) and g.top_k == -1);
}

test "abi plugin: prefill, decode and verify are always declared; batched rows and canvas follow the caps" {
    try testing.expect(declaresMode(.{}, .verify) and !declaresMode(.{}, .batched_rows) and !declaresMode(.{}, .canvas));
    try testing.expect(declaresMode(.{ .batches_decode = true }, .batched_rows));
    try testing.expect(declaresMode(.{ .block_diffusion = true }, .canvas));
}
