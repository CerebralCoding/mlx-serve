# qwen4_exp plugin (Qwen3.8-Flash-Next)

Owns `qwen4_exp`, `qwen4_exp_text`: a qwen3_5 GDN+MoE trunk in 4 hyper-connection streams, n-gram PLE
(`ngram_table.bin` or the n-gram rows embedded in the pack's shards), QSA sparse attention past `indexer_budget`,
an in-checkpoint MTP layer, a Qwen3-VL tower. Packs, oracle: `docs/reference.md`; fixtures: `fixtures/MANIFEST.md`.

- Table: `config.zig` (MLX-free claims + parse; `Config` carries `ModelConfig`'s names, `model_config.zig` maps them
  one to one, pinned against `parseConfigLegacy` on both shipped packs), `bill.zig`, `plugin.zig` (load + forward),
  `contract.zig` (kernel contract).
- SERVED by the driver (`served.zig`). The plugin wraps the engine's `Transformer` (`forwardQwen4With`, the batched
  GDN tick, `forwardMoeBatchedVerify`) on the driver's slot caches; it adopts the driver's map (`Resident.take`, no
  second load). Hooks: `forward_settle` (deferred PLE: `Owed.input` on the host gather, `.history` on `--ple-gpu`),
  `release_slot` (`ssmGroupDrop`, QSA marks), `meters` (`[qsa-arms]`). Proofs: `proof_test.zig` (the table against
  legacy), `served_test.zig` (load as the scheduler loads, then captured steps, pipelined greedy, batched rows and
  verify, MTP rounds at auto and pinned depth with equal tokens AND accepted counts, host and GPU PLE, an image prompt
  on the M-RoPE table), all bit-equal.
- P8 DEBT, `engines/mlx/plugin/intree.zig` (in-tree only, delete when the ABI carries these): the MTP head, group
  lanes and the grouped verify run on the plugin's inner `Transformer` through the legacy `.qwen4` arm
  (`MtpHeadRef{ .qwen4 = inner }`, `Transformer.qwen4Host()` at every engine site that touches `qwen4`/`qwen4_mtp`);
  the plugin builds its inner trunk's `ModelConfig` from its facts (`toModelConfig`, field-equal to the legacy parse
  the engine serves it with). An ABI-clean head needs: (1) `ForwardOut.next`, the chained hidden
  apart from the projected one; (2) a fact that the trunk's hidden is the pre-mixer stream; (3) a placed head forward
  (dead history rows); (4) `HeadHooks.mark` with a length; (5) head snapshot / bytes / serialize for the prefix cache
  and SSD; (6) group lanes and the grouped verify on the plugin arm; (7) the rerank on the head's mixer output. Beside
  the head: (8) the slot's whole M-RoPE table (pooled QSA keys rope at block starts the chunk does not carry;
  `intree.enter` hands the engine's ctx); (9) a `dist` hook (a Thunderbolt pair keeps the legacy path,
  `driver.declinePlanned`). The vision tower loads into the engine's map like any served pack's (`driver.loadWith`).
- Unserved hooks: `cache.aux` (QSA history + PLE prev in checkpoints/SSD ride the engine's SSM entries),
  `capture_stream_all` (no caller on this arch).
- Kernel contract: every Metal kernel registered (`contract*.zig`); the multi-launch ones (qsa_decode, qsa_select's
  split pair, qsa_attn) chained against their wrappers. Unregistered: qsa_decode's CUDA pair (no CUDA rig).

## Rules (from the root AGENTS.md)

- NOT a qwen3_5 pack: hyper-connections, n-gram PLE, QSA around the trunk; HF `hidden_states[i]` is the INPUT of layer
  i. Vision rows splice BEFORE the hc tile. `isLinearLayer` reads only `full_attention_interval`.
- Module state READ-ONLY: text slots batch-decode; only the MTP slot is exclusive (`slotExclusiveDecode`).
- The spec "hidden" IS the pre-mixer stream (`[B,L,hc*hidden]`); head row r = (stream r, token r+1) at r+1.
- Per-request state outside conv/ssm rides `SSMCacheEntry.aux_state` + `ple_prev`; every reset/free via
  `ssmFreeQsaState`; restore-parity bar = the chunking class (~0.3 nats top-5).
- QSA GATHERS selected blocks, never a dense `[S, kv]` mask; exact select `msv_qsa_select` (16-way split at decode);
  the score sheet stays ONE NAX kernel at every width (`msv_qsa_score`): decode, verify, prefill pick the same blocks.
- QSA's visible tail is PER QUERY; scores in f32; `torch.topk` keeps the LOWER index on ties; n-gram eos is the TEXT
  config's (`ngram_eos`). One YaRN mscale on every indexer arm; pooled keys rope with attention's M-RoPE table.
- Past the budget a solo dense-KV decode row skips mask, gather and SDPA (`qsa_decode.attend`): reduce BEFORE the
  select (a shuffle in divergent code returns garbage).
- A host read inside the layer loop is a GPU barrier: PLE defers (`ple_defer` + `flushDeferredPle`) and claims its
  spec slot at build time; the GPU arm settles its n-gram history at the NEXT forward.
- PLE read path is calibrated on the inference thread before and after warming, median of 5 (`calibrateArm`); a cold
  32 GB mmap is a serial SSD fault (`PrefetchPool`, `startWarm`); `--ple-gpu` gates the no-copy wrap BEFORE wrapping.
- Residual stream is bf16; the f32 fixture is the MATH oracle (`QWEN4_STREAM_F32=1`); `qwen4 fixture bf16` pins
  streams at 0.999. GDN state + gate in f32 (`MLX_SERVE_GDN_STATE_F32=0` = bf16), its own SSD namespace.
- Tiny MoE oracles tie everywhere: fixtures dump the reference's own margins; `Qwen4Ties` acquits by those.
- MTP head state is a `Qwen4MtpState` swapped onto the module (`qwen4MtpActivate` before EVERY head touch); the head
  projects only the consumed row; batched verify is opt-in (`MLX_SERVE_MTP_BATCHED_QWEN4`).
- A fast path keyed on one pack's quant layout is silent on another's: new pack layout => `tests/qwen4_engagement.sh`.
- Decode is kernel-time and dispatch bound: a kernel that replaces ONE dispatch loses to MLX's qmv; rows per simdgroup
  is an occupancy decision (sweep in situ). Meters: `fwd-ubench`, `MLX_SERVE_STEP_TRACE=1`.

## Nulls (measured, reverted)

One-expert gather kernels for the shared expert, tail ops folded into down+reduce, the uv hc kernel at one row,
grouped MTP rounds (#685: win only at 4 streams code), `mlx_depends` ordering of the q projection, top-k inside gate+up.
