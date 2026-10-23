# glm5_next plugin

Owns `glm5_next`, `glm5_next_text` (GLM-5.3-Flash: mlx-community, NVIDIA ModelOpt FP4, Sushi EXL3 packs) and claims
`glm_moe_dsa` to refuse it (a streamed pack, mlx-stream's). Ling 3.0 (`plugins/bailing_hybrid`) shares this plugin's
config readers, load (`host.zig`), slot access (`slots.zig`) and fixture writer.

- Parse: `config.zig` (MLX-free) reads every key the engine's GLM arm reads into `Config`; `model_config.zig` maps it
  to the engine's `ModelConfig` field for field (`expectEngineParse`: equal on the three GLM fixtures, a flat pack, the
  refusals, and the file parse with processor + generation config). Refuses: `expert_streaming`, `qk_rope_head_dim`,
  any Gemma/BERT/sliding key (`foreign_keys`), `glm_moe_dsa`.
- Load (`host.zig`): the engine's layered `Transformer` built as `engines/mlx/load.zig` builds it (prefix probe,
  `Transformer.init`, compiled MoE routing + GDN gate on a GPU), adopting the driver's loaded map (`intree.takeWeights`)
  or reading the pack from disk. Its helpers (projections, KDA, MoE, embedding, head) serve the forward.
- Forward (`table_forward.zig`, a copy of `forward.zig`'s): the [B,S,4,D] stream through KDA / DSA sublayers inside
  mHC; KV and recurrent state only through `CacheOps` (`slots.zig` hands `gatedDeltaNet` a temporary entry and writes
  conv, ssm and a verify's per-row capture back); positions from `Forward.positions`; captures = the stream MEAN;
  batched rows in groups of 8 (`MOE_DECODE_BATCH_MAX`), each slot's KDA and DSA by its own call.
- Spec: `trunk_weight` (embedding table and head, both KDA hybrids); `SpecHooks.head` = `head.zig`, the `mtp.0` layer
  loaded as `mtp.zig` loads it, its block over the head's one-layer cache at the cache's OWN length (`CacheOps.len`),
  never the trunk position. DFlash: captures through `Forward.capture_layers`; the policy stays engine-side.
- Proof (`proof_test.zig`, tiny random pack at the release's DSA geometry, `fixture.zig`): every row's logits, both
  captured layers, every KDA state and verify capture bit-equal to the legacy `forwardWith` over prefill, decodes,
  verify windows past the indexer budget, a later chunk; batched rows (two row groups) vs `forwardGlmBatchedDecode`; a
  device token; media rows over two chunks; the `mtp.0` head vs the legacy head; a Sushi pack (EXL3 experts through
  `sdk/quant/exl3/format.zig`). Served (`served_test.zig`, the driver on the legacy config via `both.zig`): steps,
  pipelined greedy, media; the head through `MtpHeadRef.plugin` vs the `.glm` arm incl. a pinned-depth round loop.
- Bill: `kv_per_token` (latent + indexer key and gate per DSA layer), `ssm_checkpoint` (f32). The prefill bill stays
  `bill.zig`'s `glm5PrefillMemoryNeeded` (the served config is the legacy parse, so `isGlm5()` arms still read it).
- Discovery: caps advertise what legacy discovery did (no images for a `_text` pack's vestigial tower; video beside a
  `video_token_id`, which the engine never reads). One change: a ModelOpt pack's unloaded row names 4 bits, as its
  loaded row always did (legacy discovery read only `quantization.bits`).

Served (`engines/mlx/plugin/served.zig`). Streamed packs (`expert_streaming`, `glm_moe_dsa`) stay mlx-stream's.
Batched verify rows (`batched_rows`, rows > 1) are refused by name: the engine keeps GLM's MTP rounds solo
(`mtpRoundsStaySolo`), so nothing calls it.

## Rules (from root AGENTS.md)
- A norm's epsilon sits where the REFERENCE puts it: fla's l2norm adds it to the SUM (`gdnQkNormEps`: 1e-6/dk).
- A capture gate is a promise the forward keeps: GLM taps the MEAN of its hyper-connection streams.
- A sparse target's DFlash is policy-driven (`dflash_policy.zig`, `MLX_SERVE_DFLASH_POLICY=0`).
- Generation growth is billed to the KV reserve's HEADROOM, never the auto budget (`glm5PrefillMemoryNeeded`).
- A cache row no op reads is never computed: the indexer's rows below its budget ride `mlx_depends`.
- Partials crossing threadgroups in ONE dispatch go by atomic store/load (`hcPre`).
- jinja.cpp reads `x.0` as an INTEGER index (GLM-5 tool history).
- A prefix-cache HIT is not bit-identical on a hybrid: low-margin tokens move >1 nat with chunk width (numerics).

## Known issues
- With a vocab under 16384 rows the engine's served-head readout builds a 3-bit
  draft head the legacy GLM head never built (drafts only, never output): needs a `SpecFacts` fact.
- Kernel contracts for the 20 `kernels/*.metal` rows are not registered.
- No `[dtype-trace]`, `MLX_SERVE_GLM_PROFILE` or layer-cap diagnostics on the table forward.
