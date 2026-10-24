# mlx-serve — project context for AI

Native Zig server running MLX-format LMs on Apple Silicon; OpenAI/Anthropic/Ollama-compatible HTTP APIs + native media generation. No Python. This file is the COMPRESSED layer; detail lives one hop away.

## Detail map (read on demand)

- `docs/reference.md` — deep detail: per-file contracts, media-gen schemas, API surfaces, LAN design, observability, embedded engines, arch numerics (dsv4/inkling/H3, Kokoro, runtime LoRA), website/tier list, licensing. Read its section BEFORE deep work on a subsystem.
- `docs/gotchas/{tool-calling,server-http,engine-mlx,models-media,app}.md` — full war stories behind every rule in `## Rules`.
- `src/plugins/<name>/AGENTS.md` — each model's rules, quirks, packs and oracles. `docs/known-issues.md` — what is broken now.
- `docs/CUDA.md` — NVIDIA (Linux, Windows): builds, our own CUDA kernels, per-device defaults, memory, measuring, MLX patches.
- `tests/CLAUDE.md` — integration-test matrix. `app/CLAUDE.md` — Swift app layout + rules; `app-web/CLAUDE.md` — console source (auto-load in their dirs).
- Skills: `/release` (pre-release checklist, CalVer, CHANGELOG), `/bench` (llmprobe methodology + comparison traps).
- `containers/{agent-shell-mlxserve,guest-kernel}/` — Agent Sandbox guest image + kernel, two pinned artifacts that BUMP TOGETHER. Detail: `docs/reference.md`.
- `website/` — GitHub Pages site + `llm-tier-list/`. Design: `docs/reference.md`. Guards: `tests/test_website_pages.sh`, `tests/website_tier_list_logic.mjs`.
- **Growth policy (ENFORCED)**: this file stays under 100k bytes, short-form and minimal; EVERY rule bullet is ≤ 3 lines. Be PICKY: a rule earns a line only for a real bug or a strong gotcha we actually hit and that someone would repeat. Not a diary: no log of what happened, no measurements, no round-by-round notes, no restating what the code or git history says. Same bar for `docs/gotchas/*.md` (one short story per gotcha) and `docs/reference.md` (one section per subsystem). When in doubt, leave it out; App content → `app/CLAUDE.md`.

## Stack

Zig 0.17.0 (pinned via `scripts/fetch-zig.sh`); mlx + mlx-c PINNED SUBMODULES (`lib/mlx-src` v0.32.3, `lib/mlxc-src` 56b2d39 = PR #127 + `patches/mlxc-gather-qmm-global-scale.patch`, applied by the build scripts) self-built NAX-enabled by `scripts/build-mlx.sh` into `lib/mlx/` (FFI `src/mlx/ffi.zig`); jinja.cpp (wangzhaode, Apache-2.0, NOT llama.cpp's) as `lib/jinja_cpp/libjinja.a`; stb_image + libwebp; safetensors; BPE. Embedded engines: ds4 (`lib/ds4`, DSV4-Flash GGUF), libllama (`lib/llama`, generic GGUF), the Neural Engine runtime (`lib/ane`). Absorbed in-tree: sushi's EXL3 + STQ (`src/sdk/quant/exl3/sushi/`, module `sushi_exl3`, `zig build exl3-test`) and davidtai's mlx-stream (`src/mlx_stream.zig`).

## Layout (`src/`)

Two seams: the ENGINE interface (`abi/engine.zig`, host ↔ an engine, per request) and the PLUGIN tables
(`abi/plugin.zig` model, `abi/media.zig` media, `abi/quant.zig` quant; per load and per forward, never per layer).

| Area | Role |
|---|---|
| `main.zig`, `env.zig` | Entry, CLI flags + subcommands; every environment variable registered in `env.zig` (kind, owner, since) |
| `exe.zig`, `host_unit.zig` | `-Dunits` roots: the host and each plugin group (`unit_<g>.zig`, generated) as their own objects (below) |
| `host/` | Everything model-agnostic: `http/` (router, conn, SSE, WS, multipart, auth), `api/` (OpenAI, Anthropic, Responses, Ollama, embeddings, `/v1/models`, providers, transcription), `chat/` (templates, `tools/` one file per dialect, `reasoning/`, stops, loop-stop, format corpus + traffic replay), `registry/` (discovery, modality map, model settings, preflight, load progress), `gpu_thread/` (the inference thread: sole MLX caller), `media/` (request media), `media_jobs/` (media-gen jobs, previews), `lan/`, `launch/`, `cli/`, `metrics/`, `plugins/` (outside dylibs, `mlx-serve plugin new\|check\|install`), `sandbox/` |
| `abi/` | Extern tables only, no globals: engine, model plugin, media, quant, `Host`, dylib manifest; `contract_test.zig` + `exactness_test.zig` run every engine |
| `engines/mlx/` | Our engine: `scheduler.zig` + `step.zig` (slots, admission in `admission/`, batching in `batching/`/`batched/`), `generate/`, `sampling/`, `grammar/`, `spec/` (PLD, drafter, DFlash/DSpark, MTP rounds, round cost), `kv/`, `cache/` (prefix cache, SSD tier `cache/ssd/`), `ssm/`, `cluster/` (Thunderbolt RDMA), `plugin/` (the DRIVER that serves a plugin table, its proofs and tapes), `load/` + `forward/` + `transformer.zig` (the legacy `Transformer` the unflipped archs still run) |
| `engines/{ds4,llama,ane}/` | One adapter each over its runtime (`lib/ds4`, `lib/llama` + `lib/llama_shim`, `lib/ane`); `engines/own_thread/` is what the three share |
| `sdk/` | Everything a plugin imports: `bases/` (decoder, hybrid, encoder, DiT skeletons + `bases/blocks/` the blocks they compose), `blocks/` (attention, mlp, moe, gdn, kda, ssm, conv, norms, rope, hc, linear, qmm, dit, schedulers), `quant/` (affine/fp8/mxfp8/nvfp4, `hadamard/`, `exl3/`, `gguf/`, `fp8_block128`), `weights/` (resident, streamed SSD banks, LoRA), `spec/` (MTP head), `vision/` (SigLIP, Qwen3-VL, DINOv3, M-RoPE), `media/` (job progress, previews, JPEG) + `mesh/` (pure-Zig mesh/GLB/xatlas), `tokenizer/`, `testing/` (fixtures, conformance, fake host, `kernel_contract/`), `util/` (`shared.zig`, `log.zig`, `rtfmt.zig`) |
| `plugins/<name>/` | One model family or media backend each: `plugin.zig` (the table), `config.zig` (MLX-free claim + parse + bill), its own blocks, `kernels/`, `fixtures/`, proofs, and its `AGENTS.md` (its rules and quirks: read it before touching the model). `_template*` = the barebones plugins, `_example*` = the kit's |
| `mlx/` | `ffi.zig` (mlx-c), `ops.zig` (the op layer: a `Graph` owns what it makes), `scope.zig`, `kernels.zig` (the kernel loader), `state.zig` |
| `kernels/<area>/` | Metal/CUDA kernels two or more plugins use (one model's own: `plugins/<name>/kernels/`); every file listed with its sha256 in `kernels/digests.txt` |
| `mlx_stream.zig` | The absorbed mlx-stream plugin (MiMo + GLM stream packs; the bank sources a resident trunk streams from): archs in `plugins/<model>/stream/`, EXL3 lanes in `sdk/quant/exl3/stream/`, the expert streamer in `sdk/weights/streamed/`; `zig build mlx-stream-test` |
| `kit.zig`, `kit_header.zig`, `plugin_host.zig` | The plugin kit's module, its generated C header, and the host surface a plugin's own test build imports |
| `testing/fake_engine.zig`, `tests.zig`, `preview_tests.zig`, `ios_lib.zig` | Test roots, the no-model engine, the iOS entry |
| `html/index.html` | The console, BUILT from `app-web/` (never hand-edited) |

Build glue: `build.zig` lists the steps; `build/` holds each graph (`macos`, `linux`, `ios`, `standalone`, `lint`), what
they link (`libs`), the configure-time reads and checks (`config`), `plugins.zig` (scans `src/plugins/*/plugin.zig` into
the generated registry, refuses a `config.zig` that reaches MLX), `test_runner.zig` (shards the tests) and `kit/`.

CLI flags: `mlx-serve --help` (`printUsage` in `src/main.zig` is the source of truth; `docs/cli.md` for users).

Generation defaults for omitted fields: body > model `generation_defaults` > global `~/.mlx-serve/generation-settings.json` > launch flags > model `generation_config.json` > hardcoded (1.0/1.0/off); a rule with `ignore_client` outranks the body. Read per request. Missing generation_config = wild-sampling signature.

## Building

- First-time: `./scripts/fetch-zig.sh` stages the pinned Zig at `.zig-toolchain/`.
- Zig caches configure-time subprocess output: after a toolchain/SDK change `rm -rf .zig-cache` or the link references a ghost path.
- **ALWAYS `zig build -Doptimize=ReleaseFast`, never bare `zig build`** (Debug 2–4× slower ⇒ fake regressions). `zig build test` does NOT refresh `zig-out/bin/mlx-serve` — rebuild before any live A/B.
- **Edit loop: `zig build check`** (35 s, Sema only) → `zig build test` (8 parallel shards, `-Dtest-shards=N`; a `sharded:` test splits its own work) → ReleaseFast only to RUN or MEASURE. ReleaseFast is single-threaded LLVM `-O3` per object (no flag tunes it): a plugin edit rebuilds in 0:30 with `-Dunits`, 3:30 without.
- **`-Dunits=true`** (off by default; `test`/`check` run without it) builds the host and 8 plugin groups as parallel objects; a plugin edit rebuilds its group only while `zig build unit-closure` holds (the plugin's imports stay in `sdk/`, `mlx/`, `abi/`, `kernels/`).
  A unit copies every global it reaches: host state is `Shared` / `SharedThreadLocal` (`sdk/util/shared.zig`), checked by `zig build units-check -Dunits=true` (data AND thread-local sections).
- **Every kernel registers with the kernel contract** (`sdk/testing/kernel_contract/REGISTERING.md`: one `contract.zig`
  per area or plugin, five gates against a float64 reference, planted controls must fail); it runs as ONE `sharded:`
  test split across the shards. Lint: `python3 scripts/lint.py --strict` (`zig build lint`) stays 0.
- Swift app: `bash app/build.sh`. The two bundle binaries move together.
- mlx + mlx-c: submodules built by `scripts/build-mlx.sh` (deployment target 26.2 → NAX kernels; script + `tests/test_mlx_staged_nax.sh` ASSERT `*_nax` in the metallib). Min macOS 26.2. Bump = checkout tag → rerun → re-diff `src/mlx/ffi.zig` externs. Brew: webp ≥ 1.6.0.
- **Linux/Windows CUDA**: build with `./scripts/build-linux.sh` / `scripts\build-windows.ps1`; MLX's JIT needs `CUDA_HOME`; custom MSL runs on Metal only. Kernels, per-device defaults, memory, measuring, and its rules (GPU-pool loads, graphs, kernel cache, matching MLX's bits): `docs/CUDA.md`.
- Rebuild Jinja after `lib/jinja_cpp/*.cpp` changes: compile the 7 `.cpp` (`clang++ -std=c++17 -O2 -DNDEBUG -I .`) into `obj/` and `ar rcs libjinja.a obj/*.o`.

## Testing — TDD is mandatory

Order: (1) failing test FIRST, for the right reason; (2) minimum code to green; (3) full suite (`zig build test` all steps, 0 fail + `bash app/test.sh`/`swift build` + relevant `tests/*.sh`); (4) refactor. A live curl is a sanity check, NOT a test.

Feature = unit test that fails without it (+ integration script if HTTP-observable). Bug fix = regression test red→fix→green, red-on-revert. Cross-arch = cover every touched arch. Refactor = characterization test first. UI/build scripts = factor a pure helper and test that.

**Class bugs get class guards.** A live failure revealing a CLASS ships: the instance regression test; a corpus entry or universal invariant in `src/host/chat/format_corpus_test.zig`; a rule here + story in `docs/gotchas/`.

Hermetic suites: `zig build test -Dtest-filter="format corpus"`, `-Dslow-tests -Dtest-filter="tool traffic"` (the replay is gated behind `-Dslow-tests`; the bare filter compiles and reports success without running it). Full suite: `zig build test -Dslow-tests`. Full matrix: `tests/CLAUDE.md`.

## Releases & benchmarking

- Process/CalVer/CHANGELOG: `/release`. Per-release perf gate = `./tests/bench.sh` on the FINAL tree vs the previous column in `benchmarks.md` (ONE new column per release).
- Methodology: `/bench` (llmprobe measures; bench.sh drives mlx-serve, `--url` probes another engine). Only diff same-methodology cells; spec cells are variance: same-session ratios only, name the engine in every win, an A/B arm is proven by ENGAGEMENT lines. Record = `benchmarks.md` + `~/claude-tmp/bench-<tag>/`.

## Conventions

- KISS, DRY, YAGNI; simple is not easy: the elegant solution, minimal and fast code, no speculative abstractions or knobs.
- Tests at the bottom of each source file; shell integration tests in `tests/`. Env levers only for paths with two arms worth comparing (lossy/tradeoff), never for an obvious win/fix; every one is registered in `src/env.zig` (kind, owner, since) and read through it.
- Commit messages and PR bodies carry no `Co-Authored-By` or generated-by line.
- Inference thread is the SOLE mlx caller (even frees) — media gen posts to `gen_queue`, never a gpu mutex. Chat runs between gen steps (`Progress.boundary`), CPU-only stages on a worker (`gen_sse.offload`), the job's peak billed to admission (`gen_reserve_bytes`; per request via `GenRequest.peak_bytes` where the request moves it: a 1536 3D cascade vs 1024).
- Concurrent requests batch-decode on pure-attention archs, the qwen3_5 family incl. `qwen3_5_moe`/`qwen4_exp` (`configBatchesDecode`) and llama.cpp GGUFs; `--max-concurrent` sizes the submit queue, not a decode gate. A slot's verdict is a `BatchVerdict` reason (`[batched] slot serial: <reason>`, `/props` `batching`, `/v1/models` `batched_decode`, `decode_serial_total{reason}`). Slots joining mid-generation drain lazy pipeline state first.
- A batched group past 1024 KV tokens attends PER SLOT (`perSlotBatchedAttn`: own view, no pad/stack/array mask; causal + quantized-KV kernels eligible). Below that, and on qwen4's QSA reads, the STACKED arm is capped by PADDING WASTE (`groupKeepCount`, `MAX_PAD_WASTE` 1.5, must stay < 2.0); longest slots fall to serial.
- A batched-decode guard that only runs at N=1 pins a shape that never ships: `MLX_SERVE_FORCE_BATCHED=1` at one slot still has `batch == 1` inside the forward (decode-attn-quant + fused QK-norm gates key on it). `tests/test_batched_equivalence.sh` runs a real two-stream arm; both kernels log `[batched] ... engaged (slots=N)`.
- A cold prefill YIELDS to decode ticks at chunk boundaries (`scheduler.interleaveDecodeTick`, up to 8 ticks for a quarter of wall time, `interleaveTicksFor`) and narrows to 2048 while anyone decodes (`companyPrefillChunk`: a chunk boundary is the only yield point); bounds the GAP, not 4-way aggregate. Idle cost zero. Guard: `tests/test_prefill_interleave.sh`. The decode share DEFAULTS to 0.25 (`DEFAULT_DECODE_SHARE`; `--prefill-decode-share 0` = the old schedule): without one a quick question waited 90-120 s behind a 90k prefill. With a decode share, the same boundary also admits small pending requests (`admitSmallAtBoundary`: uncached tail at most one chunk, up to 4 per boundary within one decode share of wall time, never from a nested prefill) so a newcomer waits one chunk, not the whole prefill. Guard: `tests/test_boundary_admission.sh`.
- KV reuse via prompt-prefix matching; invalidated after tool calls + pad-only gens; hot cache spills to SSD; RAM invalidation propagates to disk.
- Chat templates live in model dirs; Jinja renders with fallback formatting.
- **Diffs are read by a human. Keep them small.** A comment says what the code cannot (a non-obvious WHY, a contract, a unit), in one to three lines. Never: bug history, measurements, audit trails, review item numbers, dates, "PR #NNN", what an older commit did, or a restatement of the code. That belongs in the commit message and, if it is a rule, in `docs/gotchas/*.md`.
- **A dependency bump moves the pin and nothing else**: docs never name a submodule commit (git records it), a doc gets a line only for a user-facing change, and an app catalog entry lands only where a surface shows it (the tray lists `gemmaModelOptions` ids with `4bit`).
- **No source-scan tests** (`@embedFile` + "this string appears in that function"): they pin text, not behaviour, and pass against their own needles (lint enforces it). Test the behaviour; a class guard that can only be text is one line in `scripts/lint_pins.txt` (`!needle` forbids). A test comment is one line saying what the bar is.
- **One story per gotcha, one line per rule.** A gotcha entry is the defect, the cause, the fix and the guard in under 20 lines, written once after the work lands. No round-by-round logs, no ledgers, no "what the reviewer said". CHANGELOG: one user-facing sentence per change, no provisional numbers.
- **New model support starts from mlx-lm's layout**: before writing a converter or loader, read mlx-lm's model file (merged or open PR) and the MLX packs already on Hugging Face, and match their weight names, config keys and quant layout so packs load across engines. Invent only what mlx-lm has no answer for.
- **Code two plugins need belongs in the SDK** (recommendation): when a second plugin wants a block, kernel or helper, promote it to `sdk/blocks/` or `sdk/bases/blocks/` with both plugins' tests rather than copying it, and land a speed-up in `sdk/` first. A deliberate one-model change copies the block into the plugin, marked `//! copied_from: <sdk path>`: a debt to fold back.
- **One feature, one PR**: UI + Zig in the same PR; split only genuinely unrelated features.

## Supported architectures

Dispatch on `config.json` `model_type`: the first plugin whose `config.zig` claims it serves it (`build/plugins.zig`
registry). Each plugin's `AGENTS.md` holds its rules, quirks, packs and oracles (`src/plugins/<name>/AGENTS.md`; the
name column below is the folder). With `--mlx-gguf` (opt-in, experimental) a GGUF that `sdk/quant/gguf/claim.zig`
`servablePath` accepts (arch `qwen35`/`qwen35moe`/`gemma4`, every tensor type known) runs on the MLX path as the
matching arch; `--engine ds4|llama` still wins. Every other GGUF bypasses MLX → embedded engine by header
(`engine_pick.preferredEngine`: antirez DSV4-Flash + the ds4-only archs `deepseek41`/`qwen4exp`/`glm-dsa`/`glm5-next`
→ ds4, else llama.cpp); a SPLIT GGUF (`split.count` > 1) always → llama.cpp (ds4 `exit()`s on a missing shard). ds4
DSpark arms with `--dspark` when a `-DSpark-` GGUF sits beside the model (`mtpDraftTokens() > 1`, not `hasMtp()`).

| model_type | Plugin | Notes |
|---|---|---|
| `gemma4`, `gemma4_text`, `gemma4_unified(_text)` | `gemma4` | SigLIP vision, clipped linears, PLE; Gemma 4 assistant drafter |
| `gemma3`, `gemma3_text` | `gemma3` | + EmbeddingGemma encoder when `use_bidirectional_attention` (`embedding_gemma`) |
| `embedding_gemma2` | `embedding_gemma2` | Gemma 4 trunk trained bidirectional; images + video frames on `/v1/embeddings` `messages` |
| `bert` | `bert` | Embedding encoder, per-checkpoint pooling |
| `llama`, `mistral`, `qwen2`, `qwen3`, `qwen3_moe(_text)`, `qwen3_next`, `hy_v3` | `llama`, `qwen3`, `qwen3_next`, `hy_v3` | Standard dense / DeltaNet / Hunyuan 3 MoE |
| `qwen3_5(_text)`, `qwen3_5_moe(_text)`, `minicpmv4_7` | `qwen3_5` | GDN + optional MoE; Qwen3-VL vision; Qwen3.8 packs; MiniCPM-V 4.7 tower + canvas M-RoPE; native MTP |
| `prism_hadamard_qwen35` | `prism_hadamard_qwen35` | Bonsai 2: qwen3_5 behind block Hadamard rotations, f16 activations, exact 2-bit GEMVs |
| `qwen4_exp(_text)` | `qwen4_exp` | Qwen3.8-Flash-Next: hyper-connections, n-gram PLE, QSA; batched text, exclusive MTP slot |
| `nemotron_h`, `nemotron_h_puzzle`, `nemotron_h_omni` | `nemotron_h` | Transformer + Mamba2 hybrid, NoPE attention, RADIO vision, LiLiCorr DFlash, MTP sidecar |
| `lfm2`, `lfm2_moe`, `lfm2_vl` (`lfm2-vl`) | `lfm2` | Gated conv + attention hybrid; SigLIP2 NaFlex tower with tiling |
| `laguna`, `kolibri1`, `flint` | same | Sliding + NoPE attention arms with sigmoid MoE and an UNGATED shared expert |
| `gpt_oss`, `spark2_5`, `k2_horizon`, `muse_glimmer(_text)` | same | Sink attention / sliding dense / grouped-RMS Llama / Muse VL (Harmony channels, ATEM tools) |
| `mimo_v2`, `mimo_v2_flash` | `mimo_v2` | MiMo-V2.6-Flash, mlx-lm's layout (release refused by name); legacy forward deleted; MTP = 3 heads (`mtp_head.zig`); stream packs (`expert_streaming`) stream their `experts.bin` through the same table |
| `inkling_mm_model` | `inkling` | NO RoPE (relative-logit bias + short convs); serial, spec off |
| `deepseek_v4`, `deepseek_v41` | `deepseek_v4`, `deepseek_v41` | 1M ctx, module-owned state, serial + single-flight, DSML tools on our byte-pinned templates, DSpark; V4.1 images + EXL3 stream packs |
| `bailing_hybrid` | `bailing_hybrid` | Ling 3.0: KDA + MLA hybrid MoE |
| `glm5_next(_text)`, `glm_moe_dsa` | `glm5_next` | GLM-5.3-Flash: KDA + DSA inside mHC, 288-expert MoE, batched decode, `mtp.0` head, vision |
| `kev`, `laya`, `d1`, `clef` | same | Typed decisions (`POST /v1/decisions`) |
| `qwen3_asr` | `qwen3_asr` | `/v1/audio/transcriptions` only, + the forced aligner (`qwen3_forced_aligner`) |
| `nemotron_diarization` | `nemotron_diarization` | `diarized_json` speakers, run once over the whole clip |
| `*.gguf` | engines | ds4 / llama.cpp (above); GGUF presence WINS over a stray config.json |
| media | `flux2`, `krea`, `mage_flow`, `qwen_image`, `moss_image`, `ltx`, `minimax_h3`, `hunyuan3d`, `trellis2`, `pixal3d`, `acestep`, `music3`, `stable_audio3`, `yue2`, `qwen3_tts`, `kokoro` | Media plugins (`abi/media.zig`); configless packs are claimed from `model_index.json` / `pipeline.json` / `model_config.json` (`sdk/media/pack_claims.zig`) |

Models with `vision_config` but no vision weights disable vision. Embedded-engine detail: `docs/reference.md`.

## Unified media generation

One server, one registry — image/audio/video/3D coexist with chat. Engine slots are MODALITY-named unions on `LoadedModel`; new backend = one union arm + impl file. App flow load→generate→unload; headless starts idle. Which pack or `model_type` is which modality is its media plugin's answer (`MediaTable.claims` + `modality`, read through `host/media_jobs/plugin/served.zig`): discovery, `list`, the load arms and each `/v1/models` row's `modality` read it. Downloads → `~/.mlx-serve/models` (Swift + Zig).

- `POST /v1/images/edits` = OpenAI multipart translated by `gen.openaiEditFormToJson` into the `mode:"edit"` JSON body; unhonored fields = NAMED 400.
- LoRAs are STACKED, ONE grammar across image/LTX/H3: `lora_paths`+`lora_scales` (cap 8, `gen.parseLoraFields`), summed at forward — never merged. Resident backends reconcile via `setLoras`; H3 pre-validates (`lora.validatePath`), Turbo = file 0.
- **FLUX real CFG** (`guidance_scale`+`negative_prompt`, `gen.ImageEngine.supportsGuidance`): distilled klein takes NEITHER field (one forward per step); the undistilled base 9B (`flux2-klein-9b-base`) needs it — TWO forwards per step, `uncond + scale·(cond−uncond)`.
- Endpoint/field/backend detail: `docs/reference.md`. Guards: `tests/test_unified_gen.sh` + per-modality scripts; parity via env-gated cos oracles (`tests/dump_*_fixtures.py`).
- **ANE media offload** (opt-in, LOSSY: `--ane-image` Krea, `--ane-video` H3, `--ane-audio` ACE-Step; ONE value `ane.media_offload`): calibrated share that sticks, free-disk-capped lineage cache, fp16 seam SCALE, LoRA blocks decline. Rules: `docs/reference.md` "ANE prefill rules" → media offload.
- **Media DiT linears run int8 on NAX (LOSSY; default ON only with NAX + 64 GiB RAM, `--w8a8` / `--no-w8a8`, app toggle) and the int8 copy REPLACES the packed weights**: each loader frees a block's packed arrays as it converts it (`Weights.dropBlock`), so the load peak is the converted size; the bill is `w8a8.residentBytes` (a 4-bit pack GROWS ~1.8x). Kernels need NAX (`w8a8.nax()`, also gates int8 attention); name-keyed maps hold the copy in a `w8a8.Overlay`. Bar per engine: `w8a8.agreement` cos AND rms vs the packed forward, resident bytes vs the bill (`expectBilled`); `zig build test` runs with it OFF.
- **"Free disk" is what the OS will GRANT, not statfs** (`msv_volume_free_for_use` = `volumeAvailableCapacityForImportantUsage`; `kv_disk_cache.volumeSpace` + the ANE cap read it, statfs is the fallback): purgeable space is released on demand.

## HTTP APIs

- **OpenAI chat/completions + Responses**: usage ALWAYS carries `prompt_tokens_details.cached_tokens` (ONE `formatChatUsage`); thinking opt-ins = `reasoning_effort` OR `enable_thinking`, top level or in `chat_template_kwargs` (`reasoning_budget_tokens` outranks); `n>1` 400s. `/v1/responses`: every SSE event has `sequence_number`, `background:true` → 400, stateful via `ResponseStore`, WS via Upgrade, NO `[DONE]` on WS.
- **Continuing a partial reply** (`chat.continuationRequested`/`continuationPrefill`): trailing assistant message becomes a PREFILL (whitespace trimmed). `/v1/chat/completions` takes `continue_final_message` EXPLICITLY; `/v1/messages` INFERS; `/v1/responses` neither. Tool-call replies never continue; in the tokenize-cache key; ds4 can't (`continuationRejectReason`).
- Context-overflow 400s name BOTH counts (`contextOverflowMessage`); the app renders it as a card.
- **Anthropic `/v1/messages`** (Claude Code): typed blocks, `input_schema`→`parameters`, stop-reason map incl. `stop_sequence` echo, full SSE block lifecycle. Launcher env: `ANTHROPIC_BASE_URL` + dummy keys + `ANTHROPIC_DEFAULT_*_MODEL=mlx-serve`.
- **Ollama `/api/*`**: pure translation, no-model endpoints answered pre-scheduler; `resolveName` handles `name:tag`. One path must never register under TWO ids (`registry.peekByPath` — double-residency OOM).
- `/v1/models` rows carry `context_length`+`max_model_len` at TOP level twinning `meta.context_length` (#188): discovery clients never read `meta.*`; an outside plugin's pack adds `plugin`. Both emitters. Guard: `tests/test_models_capabilities.sh` [4b].
- **Providers**: `<model>@<name>` from `~/.mlx-serve/providers.json` proxies `/v1/chat/completions` ONLY (other surfaces = named 400) through curl with the provider's key; rows ride `/v1/models` with a `provider` badge while the probe answers, a non-empty `models` list FILTERS the provider's list (and IS the list when it has none), none when unreachable. Never shared to the LAN. Design: `docs/reference.md`.
- **LAN sharing**: proxy is a TRANSPORT; keyless gate = `routeClass` × `SharedSet`; `<id>@<peer>` mirroring; loops impossible by construction (self-token + tunnel marker, one hop). Design: `docs/reference.md`.
- **Observability** (`--metrics`): zero cost off; TTFT at prefill completion; live tok/s via ONE atomic per tick. `--api-key`: loopback exempt; `/health`+OPTIONS open; `constTimeEql`. No admin surface.

## Tool calling (server pipeline)

With `tools`, tokens buffer for detection (all tag families + raw JSON); thinking buffers separately. Parse chain: strict → tolerant repairs → truncation salvage, then the ONE chokepoint `server.parseToolCallsForRequest` = parse → inferred-name filter → parallel clamp → buried-param hoist → schema coercion (the last two gated by `--no-tool-autocorrect`; emitted `arguments` ALWAYS valid JSON). Serialization `chat.serializeMessagesJson`: role "tool" native, args as JSON STRINGS, every string via `appendJsonString`. Streaming: full args in ONE SSE delta, thinking → `reasoning_content`. Fallback `fallbackFormatChat`. Client: `app/CLAUDE.md`.

## Prompt-based skills / downloads / debugging

- Product skills: `~/.mlx-serve/skills/*.md` (frontmatter trigger substring → body into system prompt; `SkillManager` rescans on mtime).
- `DownloadManager`: streams to `.partial`, Range resume, 3 retries, cancel preserves partial, size-matching files skipped.
- Server log `~/.mlx-serve/logs/mlx-serve-<port>.log` is THE post-mortem file (`--log-level debug`). Grep: `jinja error:`, `[cache]`, `<- N+M tokens`, `tool_msgs=`, `[spec-stats]`, `spec-gate:`, `[loop-stop]`, `[lan] proxy`, `[disk-cache]`, `[hot-cache]`, `[admission]`, `[providers]`. Capture: `MLX_SERVE_RAW_DUMP_FILE=<abs>` → `tests/harvest_tool_traffic.py`. Reproduce tool bugs `stream:false` first; `pkill -f mlx-serve` between KV-poison tests.

## Rules (distilled gotchas — stories in docs/gotchas/; every bullet ≤ 3 lines)

### Tool calling & formats (→ docs/gotchas/tool-calling.md)

- **Control bytes**: ONE raw byte <0x20 in history kills the strict render → SILENT `fallbackFormatChat` (model loses its stop token). Everything through `appendJsonString`; wrong-family tags out ⇒ suspect silent fallback first.
- **A NUL byte in any message TRUNCATED the rendered prompt**: jinja decodes `\u0000` to a real NUL and a `std.mem.span` read cut the turn. `jinja_render_chat` returns its LENGTH. Tell: the same `prompt=` count on consecutive turns.
- **A late `system` turn stays in place only where the template renders it there** (`templatePreservesMidSystem`, probed with the request's tools/kwargs); else `chat.foldSystemMessages` folds it into the leading system at RENDER. Parse sites (`/v1/messages`, `/v1/responses`) join only the leading run: every other served template raises on a non-first system turn, and the raise is the SILENT generic fallback.
- **`developer` is OpenAI's spelling of the system turn and every template we serve reads it as `system` or raises** (`chat.canonicalRole` at both parse sites, chat + Responses; #348): a raise is the SILENT generic fallback.
- **A template that renders tool CALLS but never names a `'tool'` role is tool-aware** (MiMo's generic `{{ role }}` header): `chat.templateRendersToolTurn` probe-renders one tool turn and keeps it native.
- **jinja.cpp reads `x.0` as an INTEGER index** (GLM-5 tool history): a digit parsed as a name raised into the SILENT fallback.
- **jinja.cpp lexes `\xHH`** (Flint's `replace("\x00", "")`): an unknown escape raised at LEX time, so every Flint turn rendered the generic fallback. Pin a new template with a byte-exact jinja2 render test.
- **A `chat_template` value can be a POINTER** (`{% include 'chat_template.jinja' %}`, transformers ≥5, issue #169): `chat.isIncludeStub` reads an include-only value as "no inline template" so the sidecar loads. Grep the log for `jinja` first.
- **A JSON-dialect call cut INSIDE the object still names its tool** (`chat.truncatedJsonCallName`, depth-1 `"name"` only): the Hermes JSON arm had no NAME + `{}` salvage, so a 4 KB `edit` that hit EOS mid-string vanished as an empty `stop`. Shipping the raw markup as content is NOT the fix.
- **A template that iterates any non-string `content` raises on `null`** (LFM2-VL 1.6B): every tool-call turn fell to the generic fallback, and with an image it became a 400. `EmptyContent.empty_string` is sniffed for it like Harmony (`if content is not string`).
- **Model-mangled arg JSON**: strict parse fails → `looseRepairToolCallJson`; never drop the whole call. **Truncated opener**: recover NAME + `{}` — NEVER ship partial values.
- **Delimiter-drop tolerance** (hy3): a tag parser never bails on ONE missing delimiter; plural-wrapper recovery keys strictly on the SUFFIXED `<tool_calls:` form. A DOUBLED opener keeps what follows the last one (Flint `<arg_key><arg_key>K`); an argument KEY never carries markup (corpus invariant).
- **A `<tool_call>` body carrying `<function=` is the XML dialect and is read FIRST** (qwen 3.5+ template mandates it): a parameter VALUE never decides the call.
- **A model can call in ANOTHER family's dialect** (qwen4_exp writes Claude's `<function_calls><invoke name=…>`, #928): `BARE_INVOKE` rides ATEM's parser as `.inferred` (a declared name only), AFTER every native dialect (inside their values it is payload); the stream gate and the scrubber know its openers.
- **A `<parameter>` VALUE may spell the dialect's own close tags** (`hermesValueEnd` = LAST `</parameter>` before the next opener; `hermesParamSpanEnclosing` skips values when scanning for `</function>`/`</tool_call>`). The TRUNCATED-call branch reads `<function=` first too; `isJsonNumber` (JSON grammar, not `parseFloat`) decides literal vs string.
- **A hand-written partial-prefix ladder DERIVES its test from the marker** (MiniCPM5 `<function`): an enumerating test copies the array's gaps. A gate check is a strict SUPERSET of its parser, else a real opener flushes AND duplicates.
- **A `</think>` inside a tool ARGUMENT is payload** (`chat.thinkCloseIsToolCallPayload`): decline a close whose nearest preceding tool opener is still OPEN AND whose block closes afterwards — both halves load-bearing.
- **A short exact cycle convicts on SPAN, not reps** (`degenerate_loop_min_span` 128, `exactCyclePeriod`): 16 reps at period 1 is a 24-wide map wall row `"1111…"` on a per-digit tokenizer, or a zeroed array at period 2. Legit code repeats short cycles; a loop runs on.
- **Near-repeat loop tier** (`generate.isNearRepeatTailLoop`): 1024-token window, THREE ratios must ALL be low (distinct ≤0.12, 4-grams ≤0.35, novelty ≤0.10); the third (PROGRESS) tells a loop from procedural code. Bar errs toward acquittal.
- **A loop tier convicts on SPAN, because a low-entropy FILE ends and a loop does not** (`near_repeat_min_span` 4096, `degenerate_loop_long_min_span` 1024): a tile map of identical rows is a cycle AND a zero-novelty window by content, and no content measure tells it from a loop.
- **Loop-stop cuts are intentional stops**: `finish_reason "stop"` (`scheduler.loopStopReason`), `[loop-stop]` logged, tool parsing suppressed; `finish_details:{"type":"repetition_loop"}` on chat+completions; non-streaming trimmed to the span start (`loopTrimmedIds`). Guard: `tests/test_loop_stop_signal.sh`.
- **A loop convicted INSIDE an open think block closes the thought instead of ending the turn** (`scheduler.loopRecoveryAction`, once per request, `[loop-recover]`): the cut left agents a thinking-only `stop` with no tool call. Every thinking request arms the bound (`thinkBoundBudget`); answer-side loops still cut.
- **A container param string with a key repeated at the SAME value still coerces** (#402, `parseContainerAllowingRepeats`: first-wins and last-wins parses must agree, a CONFLICTING repeat stays a string).
- **`tool_choice` that obliges a call is ENFORCED, not asked** (`ToolForce`, `toolForceTick`): after `</think>` (or closing a thought that reaches 3/4 of `max_tokens`) the dialect's opener (`chat.forcedToolOpener`, XML only) is committed, decoding plain, ending before `=` (a lone `=` steers the name) and never repeating the model's pending token; every template gets the instruction line too. Guard: `tests/test_tool_choice_required.sh`.
- **Types come from the SCHEMA, never the value's spelling** (`coerceToolArgsToSchema`; undecidable → untouched; a `oneOf`/`anyOf` declares a type only when its branches agree, fx `shell.request`). ONE chokepoint `server.parseToolCallsForRequest` — never call `chat.parseToolCalls` from a handler.
- **Buried required params**: `hoistMisplacedRequiredParams` lifts only on all-schema-read unanimity. Pristine args = the parse layer is innocent.
- **Heuristic raw-JSON inference must name a DECLARED tool** (`filterInferredBySchema`); explicit tag calls never filtered; new heuristics set `.inferred`; not autocorrect-gated.
- **Hard invariants (replay-pinned)**: emitted args ALWAYS valid JSON; every converter escapes + dedups (the NAME too, every emitter, #748); a parsed name never spans a line or tag (corpus invariant); coercion never worsens conformance; broken output stays honest. Harness: `src/host/chat/tool_traffic_replay_test.zig`.
- **A Hermes param value keeps its own whitespace** (`chat.stripHermesValueFraming`): the template frames it with EXACTLY one newline per side and that is all the parser strips. Padded SCALARS still type from their spelling (`isJsonLiteral` probe).
- **Gemma dropped `<|"|>`**: rich bare values run to the CONFIRMED closing delimiter or top-level final `}`, never the first `,`/`}` inside markup.
- **Think-tag leaks**: strip pos-0 unclosed openers; `trimTrailingThinkClosers`; universal no-tag-leak corpus invariant covers new entries.
- **Unparsed tool markup never rides out on a TOOLS request** (`chat.trimLeakedToolMarkup`): one cut at the first wrapper opener, applied ONCE; `/v1/messages` cuts at emission; streaming end-flush concatenates BEFORE cutting. No `tools` = raw text on both paths, like vLLM/llama.cpp. Guard: `tests/test_no_tools_markup_passthrough.sh`.
- **Streaming + tools + thinking**: buffer until pattern resolution; reasoning streams INCREMENTALLY on the tools path (`.hold_thinking` + `chat.unstreamedReasoning`, never a resend; a BUDGET keeps buffering). Pinned by corpus streaming replay + `tests/test_messages_stream_thinking_tools.sh`.
- **The streaming think gate scans with a CURSOR** (`chat.ThinkScan`, O(n)); close tag cannot latch once a `tool_call` substring is seen — falls back to the full scan.
- **Pythonic tool calls (lfm2/LFM2.5)**: call-expression grammar; TYPES live in the SPELLING; marker-gated ⇒ additive; truncation ships NAME + `{}`.
- **A template can open `<think>` unconditionally** (LFM2.5): whether a prompt ends inside a think block is a property of the RENDERED BYTES (`server.promptOpensThink`), never ANDed with `enable_thinking`; thinking-off + prompt-opened ⇒ a stream arm DROPS the block.
- **A model can open its OWN think block when the template did not** (Gemma 4 `<|channel>thought`, a bare `<think>` on LFM2.5-8B-A1B, muse's ` to=self<|message|>`): the plain arm latches via `chat.modelThinkOpener` (BEFORE the marker skip) and `promptOpensMuseHeader`. Bar = the stream/non-stream split invariant per arch (`tests/test_smoke_matrix.sh`).
- **`in_think_block` seeds from `prompt_opened_think` ALONE at every stream site, never the request flag**. Class guard = lint pin; bar = stream-vs-non-stream byte invariant (`tests/test_lfm2_vision.sh` [6/7]).
- **An integration assertion that a MODEL must think/answer/call is a checkpoint expectation**: assert the INVARIANT, branch on the model's choice.
- **A contract COMMENT is read as a spec** — pin it with a test or it gets filed as a bug (#94).
- **Assistant-history reasoning round-trips** (`Message.reasoning_content`, OMITTED when absent): reasoning-persisting templates (laguna, inkling) otherwise render nothink signatures from turn 2.
- **A template can raise_exception on OUR extra-context values** (Inkling, Qwen3.8): `serializeExtraContext` sniffs the family; tool-call `arguments` stay OBJECTS; history tool_calls carry `"id"`. Qwen3.8 = THIRD effort vocabulary (`xhigh|medium|low`); the thinking-off refusal is sniffed on the refusal string and only a refusing template gets `noThinkTailSuffix`.
- **A transcribed chat template is worth what it is PINNED against; whitespace is a token-level contract** (`tests/dsv4_template_ab.py`, byte equality over the shapes the server emits).
- **Inkling channel markers are single special tokens, but tool turns need more**: `streamShouldBufferForTools` buffers on the invoke marker and HOLDS bare-identifier segments after boundary markers.
- **A muse channel HEADER is ordinary text between single-token markers**: every stream surface HOLDS an unresolved header (incl. sub-`to=` prefixes) and drops header tokens (`museHeaderSkipNext`); `to=self`→reasoning, `to=user`/bare→content, else→ATEM tool (`splitMuseChannels`/`museStreamVerdict`). ATEM strings RAW, JSON keeps type.
- **An error echo teaches the model the error** (Inkling name salvage): NAME = trailing identifier run; body = BALANCED JSON; a parsed NAME never contains `<|` (corpus invariant); bare-JSON inference never runs on invoke-marker text; `applyFamilySamplingDefaults` fills top_p 0.95 when no generation_config.
- **Thinking-off is enforced in the PROMPT; generated reasoning is ALWAYS delivered** (`chat.noThinkTailSuffix`: muse commits ` to=user<|message|>` without tools, LFM2.5 gets `</think>` appended). Every delivery site splits via `splitThinkBlock(text, true, …)` — no strip-and-discard path.
- **Muse renders round-tripped reasoning as `to=self` HISTORY, so PRIOR-turn reasoning is dropped from the prompt** (`chat.dropPriorTurnReasoning`): only assistant messages AFTER the last user message keep `reasoning_content`. laguna/inkling untouched — per-family.
- **`preserve_thinking` keeps the TEMPLATE's default** (Qwen3.8: every turn's `<think>` stays): forcing it false re-prefilled the history every turn; `chat_template_kwargs` `preserve_thinking:false` turns it off. Kwargs layer request > model settings > generation_config > arch (`resolveChatThinking`, `mergeTemplateKwargs`); `messages`/`tools` are never kwargs.

### Server, HTTP, lifecycle (→ docs/gotchas/server-http.md)

Reasoning, budgets, agents:
- **A reasoning budget is enforced at DECODE** (`armThinkBound` → `think_bound`, `thinkBoundTick`): early-stop line + atomic closer committed as ONE forward (`commitForcedTokens`); bare-`<think>` family only; all three surfaces (`reasoning_budget_tokens`). Guard: `tests/test_reasoning_budget_stream.sh`.
- **Effort budgets are pi's ladder** (`responses.effortBudget`): minimal 1024, low 2048, medium 8192, high/xhigh uncapped; the word maps to a budget only where the bound can arm (`effortWordOnly`).
- **Agent output share is ctx/2; compaction reserve ctx/4 capped 20000** (`host/launch/budget.zig`, served as each chat row's `agent_budget`; the app and the pi extension read it, never derive it); pi `settings.json` is MERGED, opencode2 carries `compaction` + `limit.output`.
- **A launch below the agent's context floor WARNS, never blocks** (`budget.contextFloor`: claude 64k, opencode 32k, others 16k; the row's `agent_budget.too_small_for` drives the app's NSAlert BEFORE launch).
- **A constrained JSON payload offset is AUTHORITATIVE** (`reasoning_protocol.Delivery`): nothing after it is re-parsed or cleaned. Guard: `tests/test_json_schema_protocol_routing.py`.
- **Schema-mask surfaces share one policy** (`schemaMasksThinking`, #331): defer only across bare `<think>` + atomic closer with no finite budget; tools present = no mask. Guard: `tests/test_json_schema_thinking.sh`.

Request parsing + media:
- **`messages.deinit` frees the Message array and NOTHING it points at**: media is owned by ONE `server.RequestMedia`; `Message` BORROWS; slots are INDICES.
- **Undecodable media anywhere in the conversation is a NAMED 400** (`IMAGE_DECODE_REJECT`; remote URLs never fetched); `stop: ""` skipped; schema-less `json_schema` 400; empty embedding input 400; Ollama promptless generate = load handshake. Guard: `tests/test_api_edges.sh`.
- **Latest-turn user media on a tower-less model is refused by NAME** (`mediaRejectReason`), older/tool media replaced by a note the agent can read (`dropMedia`); non-text model on a text surface 400s BEFORE prefill (`textGenRejectReason`; new surface → `isTextGenRoute`, new modality → `modalityFromType`).
- **Media atoms come from media parts, never rewriting user/tool text** (`chat.appendMediaContentParts`); item k expands to run k (`expandMediaPlaceholders`), count/kind mismatch = named 400. Encoder cache keys pixels; text parts JOIN in order. Guards: format corpus, `tests/test_nemotron_bolt.py`.
- **A dispatch field must be readable from every body SHAPE** (`parseModelFromRequest(body, content_type)`); header parameter lookups key at a boundary (`name=` vs `filename=`); binary bodies are not logged (`bodyIsText`); request ints clamp (`parseRequestSeed`, `clampJsonI32`).
- **A client-supplied PATH is proven on OUR side of the mlx boundary** (`lora.loadFile` stat → 400) and accepted only from LOOPBACK (`parseLoraFields(…, local)` → 403, #540: missing vs not-a-LoRA is a host-file oracle). `/v1/images/edits` forwards the LoRA fields (#268).
- **Hand-written error text is not JSON**: escape at the SINK (`jsonEscapeMessage`), truncate on a UTF-8 boundary. NO model-byte string is guaranteed UTF-8 — sanitizing lives INSIDE every escaper (`chat.utf8Next`); logprobs `bytes` keeps exact bytes.
- **A list of owned buffers is freed in ONE place**: a per-item `defer free` in a send loop plus the handler's defer over the same list double-freed when a send failed (client gone mid-flush; `flush_text` on chat + messages).
- **A body is exactly its `Content-Length`, read within `REQUEST_READ_LIMIT_MS` of silence** (`readRequest`, `awaitReadable`: a socket timeout panics std's reader; extra head bytes overran the heap, a 2^64 length wrapped) and nests at most `MAX_JSON_DEPTH` (recursive schema walkers overflowed a thread stack); zero prompt tokens are refused at `Scheduler.submit`; a client string spliced raw into a template is vetted first (`echoModelName`).

Sampling + logprobs + streams:
- **A `seed` binds EVERY sampler with a fresh key PER DRAW** (`generate.seedKey` + `SamplingParams.draw`), the verify round's bonus/residual draw included (`correctionDraw`: keyed at the output position; an unkeyed `sampleFromProbs` made every seeded drafter run differ).
- **Logprobs are the MODEL's distribution**: pre-temperature, `logits - logsumexp` in f32 (`computeLogprobs`), ids travel with values, entry belongs to the RETURNED token (`pending_logprob`), pre-GRAMMAR-mask under `response_format` (`nextConstrained`). Bar: temp-0 rank 1 == chosen.
- **Streaming logprobs**: a SIBLING of `delta`/`text` (`ChunkExtras`), ONE collector `StreamLogprobs` with a high-water mark, shipped once; `logprobs.content` describes `message.content` (`contentTokenRange`, `skipToContent`, `dropPending`). `/v1/completions` logprobs is an INTEGER + four arrays. Guard: `tests/test_logprobs.sh`.
- **A client stop string ends the ANSWER, never the reasoning** (#549, `chat.answerStopIndex`): a match counts only where the split delivers it as content, judged on the text UP TO the match, so stream and non-stream cut the same byte. Guard: corpus `a stop string ends the answer`.
- **Stream and non-stream are the SAME BYTES**; only an all-whitespace lead chunk may be withheld (`streamContentLead`). Also agree: spent budget WITHHOLDS the rest; tool replies carry `visibleToolPreamble`; disconnect = `client_disconnect`; stop sequences cut at INDEX, a tail that may still begin one is HELD (`StopStream`).
- **A client cannot time our stream**: final-chunk server `timings`; the `include_usage` chunk ships `"choices": []`. Guard: `tests/test_loop_stop_signal.sh`.
- **Liveness is a property of the SOCKET** (`beatStreamKeepalive`, 5 s byte-silence; `StreamHeartbeat` mirrors `StallClock`; a media job's CPU stage beats via `gen_sse.offload`). GAP: Ollama sink drops SSE comments. `--timeout` is a STALL timeout; `toolCallFinishReason` preserves "length".
- **A media handler's LAZY result evaluates at the encode** (`gen.imageJobFailed`): the denoise ran inside `toPng`, whose bare `try` escaped the failure arm and closed the socket on a Metal OOM; every failure after the guarded call goes through the same arm (503 for memory). The 3D job READS its stages on the host, so a Metal failure returned a zeroed GLB as a 200: `latchedJobFailure` checks the latch after the shape stage and after the job.
- **A long job's abort must not depend on the RESPONSE SHAPE** (`gen_sse.StreamCtx.stream`, `Conn.peerClosed`).
- **`Conn.close` sends FIN, then waits for the PEER to hang up (≤ 5 min) when the body is close-delimited** (SSE, NDJSON: no `Content-Length`): closed under a lagging reader, the kernel's `fin_timeout` RSTs it and the client gets ECONNRESET after every byte. A response written with a `Content-Length` sets `Conn.length_framed` and closes at once; an unmarked head keeps the wait. Guard: `tests/test_responses_streaming.sh` [G].
- **Grammar**: every state has a legal byte; the mask never walks the vocabulary (#380, `token_mask.buildMask`, `nextConstrained` lazy); NO whitespace OUTSIDE the root, the model's own layout inside (`MAX_FREE_WS` 16); token→bytes tables are per-MODEL (`grammarTokenBytes`). Guard: `tests/test_json_mode_multi_model.sh`.

Loading + residency:
- **A missing tensor is a load ERROR, never `unreachable`** (#217, `error.MissingWeight` → named 503); load failures cross the inference thread by NAME (`req.error_name` → `loadErrorFromName`, #144).
- **The load preflight's "available" is capped by Metal's working-set limit** (`effectiveAvailableBytes(…, mlx.maxRecommendedWorkingSet())`): a lowered `iogpu.wired_limit_mb` binds below free RAM and weights past it OOM in warmup, leaving a server that refuses every request.
- **`modelDiskBytes` bills the shards the INDEX names** (#274); an index naming NO shard on disk is STALE, not a filter (`indexShardSet` → null).
- **Tokenizer-derived config is set in ONE place for both load paths** (`ModelConfig.applyTokenizer`: EOS merges, LFM2 image tokens).
- **A launch flag that shapes a LOAD is retained on the Scheduler** (`ensureLoaded`'s cold-load `LoadRequest` is a second site; `--drafter` via `coldLoadDrafterDir`). Guard: `tests/test_cold_load_launch_flags.sh`. An arg loop needs an else branch (`classifyUnparsedArg`); guard `tests/test_headless_spec_flags.sh`.
- **Per-model settings** (`model-settings.json`, #269) are stamped at BOTH load sites (`applyModelSettings`); read `manualContext(config)` / `configuredKvQuantFor(config)`, never the raw flags. Guard: `tests/test_model_settings.sh`.
- **Endpoint EXISTENCE never depends on model state**: 404 BEFORE resolution (`ROUTE_PATHS`); a status route never reaches `ensureLoaded` (`handlePropsNoModel`); `GET /` renders with no model. Guards: `tests/test_route_404_no_load.sh`, `tests/test_index_page.sh`.
- **Only an unknown NAME falls back to the default model; a PATH names its own entry** (#585, `resolveRequestModelId`, shared with the LAN gate); unregistered path = 404. Guard: `tests/test_load_failure_no_fallback.sh`.
- **A request's model NAME resolves in ONE place** (`resolveRequestModelId`: id, path, Model Settings `alias`, Ollama tag): dispatch, the keyless LAN gate, load, unload and `/api/show` all read it; a gate that resolves less than dispatch lets an alias reach an unshared model. Guard: `tests/test_model_alias.sh` [2].
- **`--model-dir` is REPEATABLE**, merged FIRST-WINS; app reads via `readRoots`, writes via `ownedRoots`. Guard: `tests/test_multi_model_dir.sh`.
- **A streaming pack that sets no `ctx_size` takes an AUTO context** (`mlx_stream.resolveContext`: the largest rung of 256k..32k whose load bill fits the ceiling, cached per pack for `contextLength`), not the plugin's 16k standard request; a manual `ctx_size` wins.
- **A reload FREES retained CPU state off-mutex while `.loading`** (`releaseRetainedCpuState`): a refcount-less reader takes the mutex AND skips `.loading` entries.
- **Embedded engines** (ds4, llama.cpp, the Neural Engine) are reached only through their `abi/engine.zig` tables (`src/host/engines/`), each on its own thread: one ds4 session per model, one llama.cpp sequence per request (`session_claims` ≤ `Info.max_concurrent`); ds4 in-checkpoint MTP only when the GGUF declares `nextn_predict_layers` on an arch ds4 embeds it for (`ds4_embedded_mtp_archs`; DSV4 ships its head apart); the `Tokenizer` is a STUB — count via `server.encodeText`, stop on the ENGINE's set (`Info.eos`, #667); embeddings refuse by NAME; a llama trim is FALLIBLE (cold-prefill on refusal). Guard: `tests/test_ds4_serve.sh`.
- **An embedding SUB-BATCH is its own forward** (cache reset per sub-batch). Guard: `tests/test_embeddings.sh` [4c].
- **A READY model never advertises LESS than its stub** (`readyHasChat`). Default bind 0.0.0.0 WARNS (`shouldWarnOpenBind`).
- **One port, one server** (`listenExclusive`): std's `reuse_address` also sets SO_REUSEPORT (a second instance stole the port); macOS clears it after bind, Linux (which joins the reuseport group AT bind) never sets it (`listenExclusiveLinux`).

Memory bills + admission:
- **KV is billed per CACHING LAYER at the arch's OWN K/V widths** (`kvBytesPerToken` ← `attnCacheLayerCount`; only `.attention` blocks of a `layer_block_types` hybrid; sliding rings in `slotFixedKvBytes`); STORED and SCORED widths are two parameters (`prefillScoreHeadDim`).
- **The prefill working set includes the arch's OWN streams** (`prefillStreamBytesPerToken`: GDN q/k/v across cadence layers, MoE `top_k` replication) plus a RUNTIME floor (`PREFILL_RUNTIME_FLOOR_BYTES`); a lazily copied side-channel state must be NAMED in the cadence eval (`evalCadencePoint`, #366).
- **The guard bills what the ARCH reads** (`prefillAttnKeys`, `prefillFfnWidth`, dsv4 own term); a bill follows its buffer (QSA ring once per slot, `qsaRingBytes`); qwen4 MTP head KV billed at its effective width (`--mtp-head-kv-quant` opt-in).
- **The prefill CHUNK is a machine decision** (`resolvePrefillChunk`: widest rung whose reserve ≤ a quarter of the budget; `--prefill-chunk` wins); on qwen4 per-REQUEST and re-chosen per chunk (`chooseRequestPrefillChunk`, `adaptivePrefillWidth`). Vision chunks like text (#197).
- **Auto-context**: KV at the CONFIGURED width, activations ONCE (`prefillTransientReserve`), PINNED at load (`pinAutoContext`); ask `getEffectiveContextLength`. Load-time bills run inside `Scheduler.init` — use `configuredKvQuant()` + `CTX_SIZING_CACHE_RESERVE`, capped at half the room beside the weights (`ctxCacheShare`).
- **The registry gate bills what is RESIDENT, not the `.safetensors` sum** (`planEvictionsLocked`: weights or live MLX bytes, + `pleTableBill`; a victim frees its hot prefix cache from the live term, #927); its cap is the GPU's, so a load also unloads idle models to fit free RAM (`makeRoomInRam`).
- **The GPU ceiling sees EXTERNAL pressure** (`currentGpuMemoryCeiling`); Metal OOM is UNCATCHABLE; embedded-engine requests are exempt (`mlxMemoryGuardApplies`). What the OS grants is free + file-backed + PURGEABLE pages (`computeAvailableBytes`, Swift twin `computeAvailableForModel`; #932).
- **Every request is re-billed against LIVE memory before prefill** (`slotHoldsForMemory`): no fit + company = back to `pending`; alone it proceeds. Guard: `tests/test_memory_pressure_4way.sh`.
- **A generation's growth is billed to the KV reserve's HEADROOM, never the auto budget** (`glm5PrefillMemoryNeeded`): an omitted `max_tokens` is the rest of a 1M window, and pricing that answer refused a 431-token prompt at 43 GB beside 36 GB free.
- **Siblings are billed against un-allocated promises and the plan leaves the OS an eighth of RAM** (`PromiseLedger`, 2 GB `SIBLING_MARGIN_BYTES`; `osReserveBytes` 2..8 GB, `--os-reserve-gib 0` / app toggle).
- **A long prefill EVICTS the hot cache to be admitted, on the inference thread** (#353, `evictLruToAdmit`); refusals by NAME (`PrefillDoesNotFit` → 400); warm prompts bill via `WarmPrefix`; ONE `[admission]` line.
- **A prompt that does not fit UNLOADS idle resident models before it is refused** (`fitAfterUnloadingIdle`, LRU, never its own model): an image model idle beside GLM-5.3 held the memory an agent turn needed, and the 400 ended the agent's session.
- **A multi-chunk prompt RESERVES its KV up front** (`reservedPrefillTokens`: a grow copies the cache); qwen4_exp past 32k pre-buys generation headroom (`KVCache.reservedTokens`); `max_tokens` clamped FIRST.
- **Long-context mechanisms of #363 gate on ONE predicate** (`ModelConfig.longCtxGated()`, qwen4_exp).
- **Gates, preflight and COMMIT read ONE estimator** (#126, `estimatePeakResidentBytes` → `gateEstimateBytes`); a refusal quotes the number it COMPARED (`loadRequirementBytes`); staged bills are per-STAGE and a residency bill is a PLAN (`stagedPeakBytes`, `ltxPeakBytes`).
- **An H3 request is billed on its OWN sequence** (`h3ActivationBytes`: `h3RequestRows` x `H3_ROW_BYTES` + the fast recipe's per-block attention cache; it prices the resident plan too, and `handleVideoH3` refuses the staged one by NAME before any stage loads, #764): the per-model peak bills one canvas, and a 1536x672 x 124f run died in an uncatchable Metal OOM on its first step.
- **A video response is ONE JSON body**: the transport cap bills DELIVERED frames incl. chain windows (#283, `videoRgbTransportReason`, 768 MB).
- **A resident H3 DiT must skip the AdaLN precompute and keep its adapters in the cache** (`minimax_h3.Resident`): precompute frees the modulation weights and bakes tables for ONE schedule, and a LoRA stack's refs point into the stack that bound them. Residency is all-or-nothing against LIVE memory and every other load releases it before refusing (`availForLoad`).
- **Serial ≠ exclusive**: module-owned decode state needs admission single-flight (`admitPendingTick` on `modelExclusiveDecode`); held slots stay in `pending`.

Prefix cache (RAM + SSD):
- Disabled prefix retention reserves zero bytes in context sizing, chunk sizing and reported cache budgets; the configured byte cap is not proof that the cache exists.
- **The hot-cache budget is CLAMPED at load, a HARD cap, and FOLLOWS residency** (#364, `clampedPrefixCacheMem`, `reviseHotCacheBudgets`). Guard: `tests/test_prefix_cache_budget_revisit.sh`.
- **An oversized candidate is TRIMMED to the longest restorable prefix** (#330, `trimLenForBudget`, `trimmedCopy` a real copy; QSA bank priced via `trimmedCheckpointBytes`); the replace path sheds inherited checkpoints first; commit owns `ssm_cps` on EVERY outcome.
- **A disk restore leaves the first grow's ROOM** (`DiskTier.restoreKvInto` + `KVCache.nextCapacity`): an exact-size restore made the re-forwarded token grow every layer (a grow COPIES), and the honest bill refused a restored 38k prompt on 16 GB.
- **A restored cache shares its donor's buffer; a SHORT restore regrows from the prefix** (`KVCache.restoredOversized`, #492), never copies the donor's capacity.
- **A share that does not fit is taken over, not refused** (PR #518, `checkoutRestored`): the admission pass checks out a full-entry hit on demand and credits its rows on every arch. A restore always leaves a token to forward (a 1-token full hit segfaulted).
- **Checkpoint retention thins the INTERIOR, dense newest quarter** (`spanPreservingDropIndex`, `ThinPolicy`); a decline is observable (`CommitStatus`, `TrimDecline`).
- **Eviction is WORKLOAD-fair** (#378, `cache_key` via `requestCacheKey`, `lruIndexExcluding`). Guard: `tests/test_prefix_cache_workloads.sh`.
- **State AT/AFTER media is keyed per ITEM on its PIXELS** (`Entry.media: []MediaSpan`; a match stops at the first item differing in position or pixels, `mediaSharedBound`; an entry keeps only the items it covers); inheritance + thinning obey the first item (`bestCheckpointDonor`, `boundaryCheckpointIndex`). Guard: `tests/test_vision_prefix_cache.sh`.
- **SSD tier**: persists the FULL media-keyed record — image rows included, under the v9 (tokens, spans) identity — and clamps what a lookup serves AT MATCH (`mediaSharedBound` per entry: agreeing spans read the media rows back, a divergent list stops below the first differing item, a text turn at the entry's first item; #494 → #782); spec payloads stay text-only; checkpoints come off the TOP of the flush budget; hybrid arm ranks by restorable checkpoint (`bestHybridMatch`); a RAM decline spills (`spillDeclinedToDisk`, 4 GB floor).
- **The idle loop lands EVERY owed session, oldest first** (`Entry.disk_dirty`, `flushOwedSlice`, one bounded slice per call): flushing only the newest left an older session partial until it was newest again. A failed write or a slice that lands nothing gives the session up until its next commit; a checked-out entry is never owed.
- **SIGTERM finishes the owed slices, under a deadline** (`--shutdown-persist-secs`, default 45, `shutdownPersist`): never a whole-cache dump; staged bytes still queued at the deadline are fenced, not written. launchd's default `ExitTimeOut` is 5 s and SIGKILLs the flush, so a plist sets it above the deadline.
- **A change to what stored K/V MEAN gets a fresh SSD root** (`ModelConfig.cacheLayoutNamespace` → `modelFingerprintWithLayout`; null keeps the old root): the fingerprint hashes only the dir, the config's stat and the overrides, so a fixed model would restore its buggy keys (Nemotron-H NoPE).
- **A disk restore evals each chunk before loading the next** (a lazy `mlx_load_safetensors` holds its fd until eval: 256 files = ~250k tokens); restore entry points drop their own latch, or the cold fallback fails.
- **SSD-first** (qwen4 + disk tier, or `--no-prefix-cache-ram` + disk): SSD-only retains no idle RAM KV; writes ride `engines/mlx/cache/ssd/writer.zig`; a checkout is a PROMISE until append donates (`donateCheckout`/`releaseCheckout`).
- **The batched pad-waste cap reads `KVCache.kvLenForBatching`**, never `cache.step`.
- **An in-place SSD commit bills by MEASURE** (`nonChunkBytes` after − before, #573): the per-term delta in `appendSsmOnly` under-billed whole checkpoint lists in ReleaseFast builds and the tier outgrew its cap. Its guard is red only under `zig build test -Doptimize=ReleaseFast`.

MLX errors + threads:
- **Chat does not interleave with a media job on CUDA** (`scheduler.genYieldEnabled`; Metal unchanged): a decode run inside one image job left state the next job's first step faulted on once that model was evicted (3 of 4 stress runs; 0 with the hook off). Guard: `genYieldEnabled` test + the stress shape.
- **A sticky CUDA fault poisons the process** (`mlx.gpuContextLost`, set in the latch on `illegal memory access`/`unspecified launch failure`...): every surface refuses by name (503 `gpu_context_lost` via `ensureLoaded`), `/health` says `cuda context lost`, never a crash on the empty arrays that follow (Kokoro SEGV'd). Guard: `test_mlx_error_recovery.sh` [7].
- **An MLX failure is CATCHABLE** (#353, `installErrorHandler`): `checkError` per chunk, `checkErrorDecode` per tick, a latched error never 200s; a swallowed failure DROPS its latch (`dropLatchedErrorUnless(had_error)`, media jobs in `runGenJobs`); never hand a null `mlx_array` to the tensor-map insert. Guard: `tests/test_mlx_error_recovery.sh`.
- **Threads**: detach every conn thread; drain (`active_conn_threads` + `cancelAllInFlight`) BEFORE `scheduler.deinit`; handler sampling state outlives every pass (`Slot.in_pass`); an adopted spec cache has ONE owner (#266); sleep inhibition follows the inference-thread wait (#251, `tests/test_sleep_inhibit.sh`).
- **On Linux a thread's static TLS comes out of its stack**: std's 256 KiB thread-local signal stack made a 64 KiB `stack_size` an EINVAL that Zig treats as `unreachable`; explicit stacks are 1 MiB.
- **A weight is LAZY until evaluated**: force-eval at init; `mlx_compile` re-runs a CAPTURED lazy load per call.
- **Ownership by PROVENANCE, never content** (`{slice, owned}`).

LAN + console:
- **LAN**: per-INTERFACE dns_sd callbacks, loopback-first fetches, eviction via `attemptKnown`; proxying bounded by the TUNNEL MARKER (`isTunneledRequest` at gate AND dispatch, ONE hop, `error.SelfFetch`).
- **Console** (`app-web/`, Svelte): `src/html/index.html` is BUILT, never hand-edited: change the source, `npm run build`, commit both. Rules + guards: `app-web/CLAUDE.md`; `tests/test_index_page.sh` serves it headless. Metrics are on when `GET /metrics.json` is not 503.

### Engine: KV, spec-decode, kernels, MLX (→ docs/gotchas/engine-mlx.md)

Attention + KV:
- **Prefill-chunk cap AND eval cadence read the SCORE width** (`prefillScoreHeadDim`, 0 = none, `prefillEvalCadenceFor`); hd-256 branches key on 256 EXACTLY. Every qwen3.5/3.6/3.8 checkpoint is hd 256 — read the CHECKPOINT before porting an hd-128 kernel.
- **On NAX the stock sdpa is the hd-256 kernel** (`naxSdpaPreferred`); `force_fused` only where mlx has one (`sdpaForceFusedFor`). Band arm stays ours; hd 512 declined.
- **MLX sdpa has a WIDTH WALL** (`q_len*gqa <= 32`): at hd 256 only, causal qL 2..15 ride `splitCausalSdpa` in groups of min(8, 32/gqa) rows, yielding to NAX force-fuse past 8 rows; array masks ride `splitMaskedSdpa256`. `--mtp-max-depth` is a CAP — force verify width via PLD draft-len in A/Bs.
- **hd-256 prefill kernel (`msv_attn_p256`)**: band always fused; causal via kv-chunk budget; q_len < 16 ALWAYS declined; its mask arm (`fusedSdpa256Masked`) is the QSA prefill FALLBACK. Guards bill through `prefillHeadDimFused`.
- **An array-masked prefill SDPA computes EVERY key block, so a sliding layer pays the whole chunk**: sliding layers read a band kernel (`msv_attn_p256` at hd 256, `nax_attention` at q/k 192/128 + v 128); without one the wide prefill chunk loses.
- **Fused decode QK-norm+RoPE** (`fusedQkNormRope`, laguna): bit-identical; live paired A/B is the bar.
- **Decode-only dense-attention requant** (`--decode-attn-quant`, default ON, LOSSY): side copies at decode AND verify; prefill dense; tail layers nvfp4-g16 (`attnDqFor`). A/B per newly-adopted dense arch. dsv4 comp_in requant is EXPLICIT opt-in (`decodeAttnQuantExplicit`).
- **KV invalidation** after tool calls + pad-only gens. Sliding layers keep the full BUFFER (MiMo: a RING, below); the VIEW is sized by the whole BLOCK (`slidingViewFor`, `window + q_len - 1`).
- **A full-buffer sliding layer caches every token it never reads**: `slidingRing` keeps window + slack rows, billed per slot (`slotFixedKvBytes`), restores above `ringFloor`, never spills.
- **Hot-cache restore ALWAYS clamps** (`truncate(final_len)`); a failed restore hands back an EMPTY cache (`errdefer`); every eviction loop needs a no-progress exit (`evictOneLruProgress`). The hybrid DISK arm adopts the spec sidecar of the entry it RESTORED, honours `poisoned`.
- **kv-quant contract**: attention reads `KVCache.denseView` on EVERY path; schemes extend via enum + two switch arms; `HotEntry` records its scheme; quant snapshots carry 6 handles.
- **kv8/kv4 attention at hd 256 rides `matmul2d` over a threadgroup tile** (`qkvAttnMppKernel`, window `qkvMppWins`: t_q>=4 from 2K KV, 2-3 from 8K, 1 from 16K).
- **Quantized-KV packed reads are kernel-or-DENSE, per WIDTH** (`kvAttnFusedEligible` t_q==1; `kvAttnVerifyEligible` t_q 2..8 only while gqa x t_q <= 12 rows, `qkvVerifyRowsPay`); verify kernel OFF on G17; floor 2048. Guard: `tests/test_kv_quant_fused_equivalence.sh`.
- **A parity loop asserts FINITENESS before it diffs**; additive masks via `mlx_where`. Threadgroup memory is an OCCUPANCY decision (≤ ~10 KiB, `qkvDecBlockFor`).
- **Every KV buffer is sized from the OPERAND it stores** (K/V widths differ on MLA; an MLA cache is billed per ATTENTION head); a scheme with a shape constraint refuses at LOAD.
- **A fallible re-init BEHIND a `deinit` leaves a freed object** — build first, then swap (`KVCache.reinit`). A handle freed before a fallible op is reset AT the free (`updateDense`). Guard: `KVCache dense update` fault sweep.
- **KV growth is PROPORTIONAL** (`nextCapacity` +25%, floor one chunk, cap 8192). `active` flat while phys climbs = the POOL (`/props` `memory.cache_bytes`).
- **Allocator cache**: `mlx_clear_cache()` per CHUNK and per emitted block, un-skippable (#110): `applyMlxCacheLimit` once in `main()`; cadence interval-based; `Generator.advanceStep` is the one step mover; `finishSlot` clears the tail.
- **Hybrid-SSM prefix cache retains ~3.4× what it reports** (lever = ENTRY COUNT, `ramCappedPrefixCacheEntries`). SSM-checkpoint stride never sub-divides the chunk; dense hd-256 chunk cap 8192, MoE 4096.
- **Hybrid prefix candidates rank by RESTORABLE SSM position** (`findBestRestorableMatch`). The always-on SSM snapshot sits `SSM_SNAPSHOT_BACKOFF` (30) tokens BEFORE prompt end.
- **A committed turn also snapshots where decode stopped** (`appendDecodeEndCheckpoint`, image turns too), so the next round resumes after the generated tail; a plain pipelined step forwarded its stop and a lookahead, so it rolls back to its lagged state first (`rollbackSsmTo`). Guards: `tests/test_cache_reuses_generated_tokens.sh`, `tests/test_decode_end_media_reuse.sh`.
- **A restored tail inside the backoff window forwards as ONE span** (`ssmSnapshotBackoff(…, restored)`): same prompt warm == cold BYTES. Guard: `tests/test_hybrid_reuse_equivalence.sh`.
- **INT4 long-greedy divergence is legit**; byte-stable greedy ⇒ no spec + `--kv-quant off/8`. A prefix-cache HIT is not bit-identical on a HYBRID (≤0.047 nats; GLM-5.3 low-margin tokens move >1 nat with chunk width, numerics not state) ⇒ `--prefix-cache-entries 0`.
- **SSM/hybrid**: init checks `ssm_state.ctx == null`; param-free RMS norm uses `ones()`; Nemotron dt clip accepts numeric/positive-infinity `time_step_limit` only; restored LiLiCorr without an anchor takes one plain round; PLD snapshots guard each field. Guards: format corpus, LiLiCorr tests.
- **A GDN trunk's `KVCache.step` is 0 forever**: batched rope offsets read the slot's `moe_seq_offset`, which the batched tick ADVANCES. Bar: `test_batched_equivalence.sh`.
- **A model whose per-request state lives OUTSIDE the KVCache is excluded from prefix-cache reuse AND owed a `Transformer.deinit` arm.** `openDirAbsolute` on an empty/relative path is ReleaseFast UB — guard every site.

Spec decode:
- **Verify invariant** (all drafters): `cache.step = prompt_len + emitted`, t1 NOT in cache on entry, verify input `[t1, draft…]`, partial-accept correction from ORIGINAL `verify_logits[accepted]`.
- **A capture gate is a promise the forward must keep**: `supportsLayerCapture` said yes while `forwardGlm5With` ignored `capture_layers`, and the first DFlash prefill crashed on an unfilled slot. An arch that passes the gate fills every slot (GLM taps the MEAN of its hyper-connection streams); `dflash.encodeContext` names an unfilled one (`DflashCaptureMissing`). Guard: `glm5_next DFlash capture`.
- **A sparse target's DFlash is policy-driven** (`engines/mlx/spec/dflash/policy.zig`, GLM): rows follow the selector's calibrated confidence, plain ticks ride the serial pipeline with their taps parked (`dflash_pending`), and the plain cost is the REQUEST's own tick interval — the shared round-cost table's plain cell ran stale-high and the policy never stepped back.
- **A block decoder checks its ENTRY token first** (`generate.tokenStops`, all five); only an ALL-pad generation declines commit (`commitDeclinesPadOnly`); a cancel mid batched tick still RECORDS the row (`batchedTickAction`).
- **The token budget is a PRE-COMMIT invariant in every block decoder**; blocks publish through ONE `+= 1` loop.
- **A spec path that refuses a KV scheme must be gated at LOAD, or implemented**: `compactRows` refused quantized KV mid-decode, so a draft tree under `--kv-quant 8` 500'd the second request. Guard: smoke `drafter_kv8`.
- **A per-round buffer takes the max over EVERY round producer** (`generate.MAX_ROUND_DRAFTS`: head chain AND prompt lookup; a 14-draft lookup overflowed a `MAX_DEPTH` stash). A segfault on a field nothing writes ⇒ replay on ReleaseSafe.
- **A committed argmax is a `CommittedArgmax`** from `verifyArgmax` only (masks reserved ids); acceptance-test argmaxes are exempt.
- **No KV snapshots across verify**: rollback = scalar anchors + per-position SSM capture + offset-only `truncate`. A multi-token forward is not a prefill (`prefillEvalCadenceApplies`, seq ≥ 32).
- **Dispatch discipline**: all four surfaces default/admit drafters via `server.requestEnableDrafter`, then stream/non-stream wire `use_*` via ONE `server.requestSpecModes`; priority DFlash > MTP > gemma drafter > PLD; MTP default = `defaultEnableMtp` (any loaded head, MoE included); logprobs>0 + grammar disable spec; tools disable NOTHING; engagement COUNTS in tests.
- **Gates read DISPATCH, not ARMED flags** (`slotTicksRegular` → `specTickMode` + `spec_disabled_runtime`); a guard shaping INIT options does not bind dispatch; `nextPld` self-declines on `!pld_enabled`.
- **Batched MTP verify is ONE `[N, S]` trunk forward sized by the verify lane** (`mtpSubGroupPlan`; 7 rows split-K, 16-row M4 shader tile for 4..8 lanes, `mtp_group_fill`); past the crowd threshold slots decode PLAIN (`mtp_plain_tick`).
- **A grouped verify row equals its solo verify only while both run the SAME arm** (#727): a group's `projected` GDN slices take the fused verify arm a solo verify takes; the MoE rows arm stops at 8 rows, so a joined group past it runs the sorted verify kernels and its bits differ from solo (accepted: joining is faster).
- **Group lanes draft as rows of ONE head forward** (`mtp.forwardLanes`), greedy off the exact trunk head, sampled off its top-32; next chain pre-dispatched (`mtpGroupPreDraft`). M-RoPE/confidence-gated lanes keep `mtpChainBuild`.
- **The group planner prices each width per row** (`mtp_group_planner.zig`, per-request `enable_batch_mtp:false`) and falls to plain batched decode at width 0; `[mtp-planner]` names the choice.
- **A group's sampled accept is ONE filtered block on the group's own eval** (`mtpGroupSampledAccept`; one shared sampler, no seeded row). A padded verify row reads `1+m` rows (`verifyRows2d`, #446).
- **A DFlash sidecar yields to the MTP head when the request has company** (`requestSpecModes(..., has_company)`); a burst's first request yields after 2 ticks with company (`specYieldTick`, one-way): DSpark, sparse targets and PLD on CUDA always, a dense DFlash slot only once the others reach `mtpCrowdThresholdFor`.
- **DFlash is a METHOD keyed on the CONFIG CONTRACT** (`block_size` + `mask_token_id` + `target_layer_ids`; `--drafter` the one flag): drafts from ONE assistant forward, anchor row DROPPED, context via `capture_layers`, block K/V never cached.
- **DFlash2** = v1 + selector + convs on NESTED `dflash_config`; selector codebooks bf16 gather tables NEVER quantized; discovery classifies by contract.
- **DSpark** = DFlash + Markov head (`dflash.Contract`, `MarkovHead`): `block_size` at ROOT; `rope_is_neox_style:false` = MLX `traditional=true`; `w1` DENSE. Serves SAMPLED requests via one-hot acceptance.
- **Hybrid forward is capture-capable; partial accept rolls the CONV state back** (`ssmRollbackFromCapture`); `hybrid_path` separate from `moe_path`. The hybrid veto belongs to the GEMMA drafter only (`archBlocksAssistantSidecar`).
- **DFlash yield gate** = round cost / serial step, calibrated on DENSE trunks; MoE takes an absolute floor (1.8 accepted/round). The ngram spec-gate scores the PROMPT; DFlash is EXEMPT. Acceptance is a THINKING-MODE property.
- **The measured round-cost table OUTRANKS the calibrated gate** (`Table.roundBeatsSerial`): the bar is a constant from an older round cost, and a sticky disable lands on PLAIN decode. Serial is only measured once something decodes plain.
- **A shipped spec sidecar is a load-time dependency** (`resolveInDirDrafter`: `<model_dir>/drafter`, else the pack's one servable drafter subfolder, never standalone sushi's `dflash2/` cache); `--drafter` wins, `--no-drafter` opts out; discovery refuses `*_assistant` standalone.
- **A declared DFlash own head is mandatory** (`has_own_lm_head`, `draft_vocab_size`): load and validate its token space; never silently borrow the target head. Every API surface admits a loaded DFlash independently of the Gemma drafter slot.
- **Row-exact archs make drafted output byte-identical to serial ONLY while a DFlash drafter is bound** (`rowExactDecode`: nemotron_h + dense qwen3_5; MTP/serial stay stock): a sampled verify row draws `keyed_sample` at the ABSOLUTE position. Serial reference = the SAME server with `"enable_drafter": false`, never a `--no-drafter` boot.
- **The spec BLOCK is a MACHINE property** (`resolveBlockSize`, `blockCapForMachine`: M3 Ultra 8, else 5; wide lane: NAX, M4 ternary); a wider block loses off a wide lane.
- **Trunk-derived draft state rides the prefix cache** (`DflashSnap`, `Entry.mtp`, `restoreSpecSnap`): adopt only on `base + step == matched`; survives the SSD tier (`spec.safetensors`).
- **A sidecar's weights are a per-round READ**: dense bf16 assistants quantized at load (`defaultQuantBits`: 8, 4 on NAX); MTP head trunk requantized (`DEFAULT_HEAD_QUANT_BITS` 4/g64, fc bf16 for m5Nax); head `fc` may ship quantized (`QLinear`).
- **MTP head load**: delta norms AUTO-FOLD (`mtpNormNeedsRepair` also reads the norm's own negative fraction); quant mode solved PER WEIGHT from GEOMETRY (`quantParamsFromGeometry`, `.biases` optional); verification is ALWAYS the trunk head; every lever kill-switched.
- **`--no-mtp` gates an IN-CHECKPOINT head too** (`entry.mtp` reads `params.mtp_enabled`).
- **Per-arch exclusions read ONE predicate** (`ownsModuleDecodeState()` + `specInitWiring`); "module-owned" is a property of the STATE (`module_owned_state_fields` vs `module_shared_readonly_fields`); only the slot driving a module-owned head is exclusive (`slotExclusiveDecode`).
- **A module-owned forward must honour `ctx.capture_hidden`** (publish last-row + all-rows via `skip_lm_head`).
- **Round cost is MEASURED** (`round_cost.zig`): per model/width/KV bucket from solo SINGLE-CHUNK rounds; `tok` is a workload MIXTURE (plan on `planTok`); per-silicon rows are COLD-START caps (`mtp_depth_free`), never deleted.
- **Table hygiene**: interleaved prefill drops the round clock; refuse past `IMPLAUSIBLE_STEP` 1.5x / `SELF_SPIKE` 3x at fold AND load; load sweep also `IMPLAUSIBLE_WIDER` 1.25x (#382). Persistence OPT-IN (`MLX_SERVE_ROUND_COST_PERSIST=1`), keyed on chip/model/quant/OS/engine build.
- **Width trials** (`mtpWidthTrialTarget`): m_lo then m_lo+1, NEVER m_lo−1; `[spec-stats]` carries `width_trials= table=`.
- **Auto MTP depth**: below 8192 KV plan tokens from acceptance EMAs + table round TIME in ONE chunk (`MtpDepthPolicy.accept`); past 8k legacy. Rank policies in `src/engines/mlx/spec/mtp_replay_test.zig`, never live tok/s.
- **The adaptive serial switch serves sidecar heads too** (`mtpSerialProbeEarned`: 16 rounds, EMA < 2.0, floor 32k KV).
- **Auto-mode MTP output is NOT byte-reproducible**; byte bar = `--mtp-min-depth n --mtp-max-depth n`. Acceptance is a PROMPT-TYPE property — measure per index (`acc_idx=`) before touching round cost.
- **EV cost tables are refit whenever the verify forward changes** (`MTP_EV_DEFAULT_COSTS`); `MtpCostProfile` comes from the runtime fingerprint, never the sidecar; qwen4 has its own G17 surface. A/B profiles with persistence OFF on both arms.
- **Drafts shortlist on a coarse lm_head and re-score exactly** (`buildRerankCoarse`/`rerankShortlist`, from the MIXER output). Proposal is per REQUEST (`mtpDraftStepPath`): greedy = argmax, sampled = q over the exact top-32; sampled group rows stay batched (`shortlistProposalRows`).
- **A draft depth's proposal is resolved in ONE place, `mtpDraftSampling(step)`**: the greedy tail (`--mtp-greedy-tail`, per-model `mtp_greedy_tail`) samples depth 0 and argmaxes later depths with a one-hot q. A chain carrying q drafts `[1]` ids at every depth (`argmaxStepProposal`): the accept graphs concat them, and `draftSelect` returns `[1,1]`.

Sampling:
- **Penalties apply in `Generator.sampleLazy`** (#564); `SamplingParams.penalized()` keeps a request off `next`'s fast path, spec and batching. `frequency_penalty` is OpenAI's, subtracted per occurrence, never a `repeat_penalty` multiplier (#912). Guard: `tests/test_repeat_penalty.sh`.
- **`top_p` 0 is GREEDY** (`applyTopP` floors at `floatMin`). Filters cut by RANK with lowest-id tie break (`ranksDescending`, `topRanksDescending`); cumsum in f32; top-k + top-p are ONE pass (`filterTopKTopP`). Block helpers use `_axis` ops.

Kernels + numerics:
- **A weight tiled in place is read ONLY through `lane_qmm`** (`tileInPlace` at DFlash bind; `PREFILL` past 128 rows gives MLX's qmm bits): MLX would misread its buffer. The caller passes bits and group size, since 8-bit g32 has 4-bit g64's shapes.
- **Slice-born weights into gather_qmm/quantized_matmul are `mlx_contiguous`-materialized at load**. mlx `Copy`/`contiguous` are VIEW ops: a slice outliving its parent goes through `materializedOwnedCopy` + eval; a raw data-pointer read PROVES row-major contiguity. A view-materializing helper owns the view (`sliceContig`).
- **`mx.quantize` packs DENSELY** (element i at bit `i*bits`, straddling words at 3/5/6 bits, #305): every hand-rolled unpack is tested at EVERY shipped width. Fused MoE kernels take 3-bit as a BYTE TRIPLE (`mlxserve_qpack`).
- **A fast path keyed on one pack's quant layout is SILENT on another's** (oMLX Flash Next oQ4e ran half speed, no error): tiny dense-only tensors dequantize at load (`dequantizeSmall`), kernels unpack every affine width, config caches key per width, gates are measured per shape. New pack layout ⇒ `tests/qwen4_engagement.sh` first.
- **An f32 SCALAR promotes every bf16 operand** — scalars via `scalarOf(v, dtype)`, const tables via `constTableAs`; a chain that returns f32 by design makes the CALLER cast back (Mamba2's SSM `y`), and a quantized KV cache returns the dtype it was FED. Tell: `[dtype-trace] residual widened`.
- **Every forward path carries a `[dtype-trace]`**; 1-D f16 tables narrowed at load (`narrowsLoadedF16`); a kernel's dtype and threadgroup BLOCK SIZE are ONE decision (`gdnBlockTFor`); signatures come from each input's ACTUAL dtype, <8-element arrays land in `constant`.
- **A decode kernel can be LATENCY-bound** (qwen4 `hcReadFused`): split-K + unrolled loads beat the op chain; redistributing a reduction into every threadgroup LOSES. Meter: `MLX_SERVE_DIAG=decode-fwd-ubench=N`.
- **A prefill-tiled kernel at decode width is ONE threadgroup's work**: gate by rows (`MIXES_KERNEL_MIN_ROWS`, `decodeSelect`).
- **Partials crossing threadgroups in ONE dispatch go by atomic store/load** (`hcPre`): plain loads read a stale L1 even after a device fence.
- **A cache row no op in this forward reads is never computed** (decode evals token + logits only): tie it to the output (`mlx_depends`; GLM's indexer below its budget).
- **An ablation that leaves an output unwritten measures NaN ROUTING, not cost** (garbage activations send every MoE layer to the same cached experts): ablate with sane values.
- **A decode kernel keyed on `batch*seq == 1` declines verify rows AND batched slots** — the grid carries the rows (`HC_FUSED_MAX_ROWS`/`GDN_FUSED_MAX_ROWS` 16).
- **A grid narrower than the rows it commits writes none of the rest** (the GDN tree prework copies the conv window's 3 state rows striding by `TL`): a 1- or 2-node tree committed an unwritten row, drafts != off. Guard: `gdnDecodeTreeCase` at 1, 2 and 8 nodes.
- **Prefill fusions take the chunk WIDTH as a scalar INPUT, never a template** (`hc_prefill.zig`); a per-token-varying template value is a fresh JIT per value. Every `metal_kernel` helper owes a `streamIsGpu` guard.
- **GDN decode is three fused dispatches per layer** (`gdnPreworkFused`, `gdnNormGateFused`); a greedy byte flip there is legit.
- **A 30 GB no-copy table is made resident by the FIRST GPU forward** (`--ple-gpu`, opt-in); the host gather faults in only the rows it reads. Embedded packs repack into one buffer at load.
- **A no-copy MLX wrap COPIES silently when Metal refuses it** (`mlx_array_new_data_managed*`: base not page-aligned or length past `maxBufferLength`, then malloc + copy + deleter NOW). Gate BEFORE wrapping (`ple_gpu.chooseArm`); the buffer counts in MLX active memory.
- **Mamba2 decode is ONE dispatch per layer** (`mamba2_decode.zig`: conv + dt + SSM + gate, the group norm a second; the op chain serves prefill and any geometry it declines). Nemotron MoE decode reads its experts in place (`NemotronMoe.inPlace`: `gatherQmv` fc1 with ReLU² fused → `gatherQmv` fc2), never the sort machinery. Bar: `nemotron_h served: the fused Mamba2 step matches the op chain`.
- **A decode-width residual add and the NEXT block's norm are one dispatch** (`add_norm.zig`, BIT-EQUAL to `add -> fast::rms_norm`; the MoE arm folds the K-sum AND the shared expert): a fused path that replaces a chain must carry EVERY term of it. Bar: `nemotron_h served: the fused add+norm decode path equals the unfused blocks`.
- **A single routing group is the UNGROUPED fused router** (`groupLimitedRouting` → `.sigmoid_bias` when `n_group <= 1`): the grouped kernel declines `n_group 1` and Nemotron-H ran the ~15-op chain per MoE layer.
- **A host read inside a graph build or layer loop is a GPU BARRIER**: qwen4 PLE defers (`ple_defer` + `flushDeferredPle`, claims its spec slot at BUILD time via `pleClaimSpecCapture`); batch non-consumed reads into ONE eval. A rollback that re-forwards is a SECOND forward (`DsparkAnchors`).
- **Every array of one `async_eval` carries that batch's END event** (`lazyForward` ships the sampled token as its own batch; the GPU PLE arm settles its n-gram history at the NEXT forward/drain/finish): waiting on any array of the forward's batch waits for the whole forward.
- **A cold 32 GB mmap read is a serial SSD fault** (`NgramTable.gather` → `PrefetchPool`; `startWarm`).
- **A chain the GPU already OVERLAPS is not a dispatch to fuse**; a fusion pays only if it shortens the DEPENDENCY CHAIN; a weight-layout fusion is not output-preserving. Measure the marginal in-situ; "duplicate and read the marginal" is unsound for anything that MUTATES state.
- **Flash Next decode is kernel-time and dispatch bound**: a custom GEMV that replaces ONE dispatch loses to MLX's qmv, only a kernel replacing SEVERAL and no slower itself wins; rows per simdgroup is an OCCUPANCY decision (sweep in situ before deepening a multi-row kernel). Meters: `tests/fwd_ubench.sh`, `MLX_SERVE_DIAG=step-trace`.
- **Reproducing an MLX op means its REDUCTION TREE and ACCUMULATOR**; `mlx_compile` is not output-preserving; JIT vs metallib transcendentals disagree — sweep the 16-bit domain (`swigluSigTable`).
- **A table that stands in for an MLX op is cached PER DEVICE** (`swigluSigTable`): CPU and GPU sigmoids round apart, and
  Qwen3-TTS's CPU stream cached CPU bits that every later fused SwiGLU on the GPU read.
- **MoE decode**: gate+up fused `gatherQmv` default by the kernel's OWN conditions (`useGatherQmvDecode`); down+reduce ONE dispatch over 8 lanes (`gatherQmvDownReduce`, `DOWNRED_LPR`; bar = fp32-truth RMS). MoE PREFILL uses `_gather_sort`.
- **A streamed bank feeds the shared MoE block through `Moe.Weights.streamed`** (`sdk/bases/blocks/moe.zig`, contract `sdk/weights/routed.zig`): every arm reads slot banks in the in-memory arm's kernel class (fp4 decode over up to 3 banks; sorted calls at the row-block B/E); an arm whose kernel reads one bank is refused by name.
- **Affine-4 banks at ONE decode token take `moe_affine4`**; a one-expert bank VIEW is what the kernels bind, because MLX bills a command buffer per bound array's `data_size`.
- **MLX's concurrent encoder puts a GLOBAL barrier before any kernel that reads an outstanding write**: parallel branches overlap one stage only, and ordering the q projection behind the indexer with `mlx_depends` made the layer slower; judge a chained kernel by its stage, not its alone-time.
- **The indexer score stays on the NAX matrix kernel at every width** (`msv_qsa_score`): no scalar accumulation order reproduces it, and decode, verify and prefill must pick the same blocks.
- **A fused kernel gated on ONE model's shape leaves the others on the composed chain**: gate on the kernel's arithmetic, test each served shape; re-sweep planners after an mlx bump.
- **verifyQmm lanes** (`vqmmLaneForTile`): split-K M 2–7 / wide tile / NAX m16 M 8–16 / shader matmul2d on G16 (4-bit, M 8–24); plain-SIMD tiles BITS-templated and SHAPE-gated (`mixedPlainShapeEnabled`; NAX by width, shape AND rows, `mixedNaxShapeEnabled`). M=8 is the plain-SIMD cliff; with neither tile, 8–16 rows go to `simd_qmm` (`simdQmmDecodeMinRows`).
- **A verify lane is never byte-identical to stock**; parity = fp32-dequant truth per width, RMS ratio vs stock ≤ 3.0x, never cosine (`VerifyQmmParity`). 2-bit GEMV accumulates in f32 from exact products (`qmv2.zig`, M 1..8; unmeasured GPU generations keep the old M 1..3 dispatch).
- **A `metal_kernel` config cache is keyed by FULL SHAPE** (`ShapeKey`); a borrowed-handle cache evicts LRU (`vqmmScalarEvictIndex`). A bandwidth bench smaller than a real step measures CACHE.
- **Metal caps threads per threadgroup PER COMPILED KERNEL, by register use** (M1/M2 can be below 1024): a kernel above 256 threads probes on its first eval and shrinks simdgroups in the same summation order (`mma_sg`); under test a latched MLX error is invisible and poisons the next tests, so a kernel test drops its own latch.
- **A diagnostic is a token of `MLX_SERVE_DIAG`, never a variable of its own** (`env.zig` kind `diag`; absent or `0` = off). A default belongs to the engine that MEASURED it.
- **Kernel testing**: parity = no-worse-than fp32 truth, never kernel-vs-kernel; same-boot A/Bs at per-cell MEDIANS, interleaved in one process; every adopted shape gets its own A/B; content-forking arms need 3+ reps with a NONCE per rep.
- **A fusion on a small dense model pays only by shortening the DEPENDENT chain** (`fuseRowGroup` at decode width only, `fusedResidualNorm`, `tableGateMul`, `fusedGatedConvStep`, `fusedGeluPleProj`).
- **Gates**: a lower bound may REPLACE the formula (KDA `kdaUsesBoundedGate`); per-head vs per-channel gating is an INDEXING contract (`gdnKernelSource(vectorized, …)`). A struct in a generic fn must capture its comptime params.
- **A norm's epsilon sits where the REFERENCE puts it**: fla's l2norm adds it to the SUM, `rms_norm` to the MEAN (`gdnQkNormEps`: GLM 1e-6/dk); GLM's small post-conv keys came out 1.8% short per KDA layer at 1e-6.
- **Prefill perf kernels A/B PER ARCH before default-on** (`gdnBlockedEligible`, `prefillDqGemm`). dsv4 decode glue is FUSED (`dsv4_emit_win`, `dsv4_dec_chain`). A sparse-attention PATTERN cannot be tuned into correctness (H3, DEAD).
- **A per-block temporary of GBs is a FRESH allocation in the server** (8 GB pool cap, `applyMlxCacheLimit`; the test binary has none): FastH3 VSA lost to dense in situ while its ubench won, until pooling and assembly became kernels (`h3_vsa`). Price media-DiT changes by the server's step log.
- **Memory**: RSS is blind to Metal (`/props` `active_bytes`); a live handle pins its buffer through the eval, so free a step's intermediates BEFORE evaluating it; a weights MAP outliving the model pins every buffer; `iterator_next` hands a +1; `mlx_array_new_data` COPIES shape-worth of bytes; Metal at the working-set edge returns ZEROS before it aborts.
- **Tokenizer**: special-token splitting only in `Tokenizer.encode` (first-byte buckets); any per-position loop over a vocab-derived collection needs an index.
- **Upgrades**: mlx-c → diff `lib/mlxc-src/mlx/c/*.h` vs `src/mlx/ffi.zig`; ds4 → re-sync `engines/ds4/ffi.zig` EngineOptions + CORE_OBJS; NAX presence asserted, never assumed.
- **ANE prefill (opt-in, M4-and-below, LOSSY)**: rules in `docs/reference.md` "ANE prefill rules"; parity = per-program cos/rms + perceived-content equivalence; `ready: N/M` under-count = silent partial coverage. Guard: `tests/test_ane_prefill.sh`.

### Model loading, configs, converters, media parity (→ docs/gotchas/models-media.md)

Configs, templates, tokenizers:
- **Config JSON is untrusted**: use the checked readers in `model.zig`; optional nulls retain defaults, explicit disable sentinels and guarded skips remain valid, and unsafe narrowing or dimensions return named errors.
- **A marker family that is GLM/`<think>` under another spelling is ALIASED at decode, never re-parsed** (K2 `<ifm|…>` → `Tokenizer.marker_aliases`); only the rendered prompt keeps the pack's spelling (`k2ThinkOpenerAt`). JSON-schema + thinking on K2 keeps thinking off (known).
- **`generation_config.json` `eos_token_id` joins the stop set** (`mergeEosTokens`, additive); Gemma terminators merge additively (`ensureGemmaTerminators`). jinja.cpp: `is sameas true` / `is divisibleby 3` parse a BARE test argument.
- **A struct default is ONE family's**: Gemma's 1024 sliding default hit every Llama 3.x (no `sliding_window` key) past 1k tokens; `rope_type: llama3` is read. Guard: smoke `long prompt`.
- **Config reads**: when the reference IGNORES a field, the field is not the truth (laguna YaRN mscale); a field HF allows in two SHAPES is read as both (`chat_template`; Nemotron-H's `hybrid_override_pattern` string vs `layers_block_type` list); `text_config` FIRST, then root, PER FIELD; a default only ONE family wants is pinned PER LAYER TYPE (muse `rope_local_base_freq`, `tests/test_muse_repetition.sh`).
- **A shared attention helper follows EACH arch's position scheme**: Nemotron-H is NoPE whatever `rope_theta` its config carries (`layer_no_rope` set at parse, read by `HybridAttention` AND the MTP head); LFM2 ropes. Guard: `nemotron_h served: a NoPE attention layer ignores the order of earlier tokens`.
- **`*_text` siblings**: accept the tag, collapse to base type, prefix by `text_config` presence, force `tie_word_embeddings` for Gemma, add to BOTH visibility allowlists.
- **A sampler never draws a RESERVED special or a PADDING row** (`reservedOutputIds` + `definedVocabSize` → `installSuppressMask`; `unpadded_vocab_size` = ONE trim); logprobs stay RAW.
- **WordPiece honours `max_input_chars_per_word`** (`Tokenizer.wordpiece_max_chars`, HF: a longer word is one `[UNK]`): the greedy longest-match is cubic in the word, and a 200k-char word pinned an embeddings connection thread for good.
- **Metaspace `prepend_scheme` is THREE-valued** (`MetaspacePrepend`): `first` prepends ▁ only at offset 0, never after a special token (Mistral `[INST]Use`); `always` prepends per segment (laya). Diff `/tokenize` vs HF on a prompt WITH specials.
- **Digit GROUPING is per-model** (`Tokenizer.digit_group`; a COMBINED Split regex hides it); the CASED grammar (`\p{Lu}` + attached contractions) is `.llama3` (Muse, `[\r\n/]*` tail) or `.cased` (Flint: `[\p{P}\p{S}]` punct, newline-only tail, an unmatched `\p{C}` run kept whole), the plain Llama-3 regex is `.gpt2` + 3-digit groups; `ignore_merges` emits a vocab word whole; `\s` is Unicode White_Space (U+3000, NBSP), `\s*[\r\n]+` BACKTRACKS to a run's last newline, and a Split entry of its OWN (DeepSeek's and Flint's digits, CJK) cuts the text first under EITHER grammar (`isolated_runs`). Guard: `tests/test_tokenizer_hf_parity.sh`. The degenerate-tail guard has a LONG-period tier (`isDegenerateTailLoopRange`).

Weights, quant, loading:
- **Tree, prefix and axis order are CONVERTER choices — probe** (`resolveWeightPrefix`, `lagunaRouterBase`, `hy3ExpertContainer`, `resolveVisionPrefix` + `patchProjLayout`). A family's geometry comes from the CHECKPOINT once a second size exists (FLUX klein 4B/9B).
- **Quant resolves PER WEIGHT** (`computeQuantParams`; scales dtype decides fp8 vs affine; overrides can hide inside the fp family, `fpParamsFromGeometry`); affine bits outside {2,3,4,5,6,8} rejected at PARSE; no engine hardcodes a width (`affineParamsFromGeometry`, scan-pinned).
- **Dense checkpoints**: scales absence PER-TENSOR (`getLayerScaleOpt`); dense contracted weights owe `maybeTransposeForBf16` — never depthwise conv or SSM state.
- **ModelOpt's NVFP4 `weight_scale_2` is amax / (448 x 6); MLX's nvfp4 `global_scale` is the amax** (`modelopt_pack.dequantDense`): passed through raw, every weight shrank 2688x and the vision tower saw a blank image.
- **A pack runs in its reference's numerics** (`actDtype`, `LoadOpts.keep_f16`; Bonsai 2 f16 + f32 GDN state; qwen4 f32 GDN state + gate, `MLX_SERVE_GDN_STATE_F32=0` = bf16); constants take the activation dtype. Guard: `test_hadamard_fidelity.sh`.
- **A plugin's fork of an SDK function keeps every arm its loader can bind**, else refuses that layout at load BY NAME (`qwen4_exp` `unservedLayout`): the trunk copy dropped `per_expert_scale`; ModelOpt FP4 decoded garbage, tapes stayed green.
- **Every new LM or media port supports a bf16 pack AND the quantized ones**: loader, converter presets and app catalog carry both; quantized is the default download, bf16 the quality reference.
- **A gather-read table is quantized only where the READER has a quantized-gather path** (media `NEVER_QUANTIZE`; LM `embed_tokens` via `gatherQuantizedRows`).
- **Calibrated quant**: weights per-input-channel and per-expert; bit width beats group granularity ≤3 bits; round (s,b) to the STORED dtype first; an imatrix is valid only for the WEIGHTS it was collected on; uniform ≤2-bit experts to the LAST layer cause agent loops (4-bit tail fixes it).
- **Publish MTP head norms FOLDED** (`--fold-mtp-norms`).
- **Discovery**: a configless repo SHAPE is its media plugin's claim (`served.claimIn`; discovery, `list` and `gen.peekModelType` read it); size sums stat THROUGH symlinks; an incomplete media pack is invisible until its marker lands (`requiredMediaMarker`, `error.IncompleteMediaPack`); a GGUF folder is a SHELF (`id#<file>`). Guard: `tests/test_model_rescan.sh`.
- **The `--no-vision` drop list is a GLOBAL prefix filter**: a tower loads through `loadWeightsWithVision`, never `loadWeights`.

Vision:
- **`x-mlx-pixels` is a GEMMA format** (`wantsServerPreprocess`; server refuses when `vp.mode != .gemma`); `VisionEncoder.forward` returns named errors.
- **A MiniCPM-V image is SEVERAL encoder calls** (thumbnail + crops, `tile_*` on `ImageData`) laid out as `<image_id>N</image_id><image>…</image><slice>…</slice>` rows; the canvas M-RoPE table is built from the EXPANDED ids (`minicpm_canvas.canvasPositions`), so a changed placeholder run shifts every crop silently. Guard: the `MiniCPM-V 4.7` test in server.zig.
- **A placeholder id the splice mask does not name never reaches the prompt** (`spliceVisionRows` takes `video_token_id`; an id of 0 is UNSET, never token 0: audio-only Qwen3-ASR heard silence); a chunked splice resumes its ROW INDEX (#197, `vision_splice_offset`, `tests/test_vision_chunked_prefill.sh`); every generative forward arm splices vision (scan-pinned).
- **Per-arch preprocessing**: patch feature order (LFM2-VL channel INNERMOST); two GELUs in one checkpoint (LFM2-VL); NaFlex resample is `bilinear` WITH antialias and TILING is the resolution; muse = aspect-closest grid, Lanczos, CHANNEL-major shuffle (`VisionPreproc.mode`). Guards: `tests/test_lfm2_vision.sh`, `tests/test_muse_vision.sh`.
- **A processor pixel bound is not the ENGINE's** (`effectivePixelBounds`, `ENGINE_MAX_PIXELS` 1536²).
- **Nemotron omni's native MTP config requires a final vision LayerNorm BEFORE pixel shuffle**, even if a conversion omitted MTP weights; projector quantization is per linear. Guards: Super RADIO reference fixtures and format corpus.

Media backends:
- **A condition works only as TRAINING fed it**: `instrumental` as TEXT (`[Instrumental]`; + lyrics = NAMED 400, `instrumentalConflicts`); Stable Audio 3 `seconds_total` as whole seconds >= 1 (`trainedSeconds`; a fraction is noise, reference too).
- **xatlas segments one connected surface on ONE thread** (`uvwrap.parametrizeParallel`: median-split regions as `faceMaterialData` = one chart group per core; 1M faces 424 s → 3.4 s). Hunyuan paint keeps `parametrize`.
- **A flow-DiT forward over tens of thousands of tokens needs a scope PER BLOCK** (`Dit.forward`, `Decoder.run`): one scope around the whole forward keeps every block's intermediates alive together; fixture-sized oracles never show it.
- **A full-resolution f32 VAE stage is BANDED, exact because everything but its 3x3 convs is per-pixel** (`qwen_image.Stage.banded` + `Conv.forwardStrips`); MLX's peak counter does not see the whole-stage peak. Bar = the fixture oracle with bands forced.
- **A pinned library's op TRANSIENT is invisible to residency bills** (#321, #424): `conv3dDepthChunked` windows decoder convs; H3 `encodeMoments` EVALS per tile pass. Diff conv DISPATCH per mlx bump.
- **Canvas**: `recommendedResolution(totalGB:)` per Mac, capped by `autoCanvasCapPixels`; two-stage denoises at HALF; `maxFramePayloadBytes` is a TRANSPORT limit. A tiled decoder whose positions normalize over the EXTENT refuses (`error.TilingUnsupported`).
- **LoRA alpha lives wherever the exporter put it** (`lora.fileAlphaScale`; per-module tensor WINS); test with `tests/lora_noise.py` + `tests/test_real_loras.sh`.
- **A reference-image editor keeps its input's geometry** (`resolveEditTargetSize`); a few-step distilled model can REQUIRE bf16 (MageFlow Turbo: `roundBf16`, `scalarLike`).
- **Latent→RGB previews are GENERATED fits per latent SPACE** (#208, `tests/dump_latent_rgb_factors.py`; `LTXAV` ≠ `LTXV`).

Parity method:
- **Cosine cannot see SCALE** — concatenated tensors assert `rms_ratio`; a permutation-invariant checksum cannot see a permutation (`pos_weighted`); an oracle that cannot execute the reference must say so; a faithful port can make the checkpoint WORSE (H3 splice default-off).
- **Deep ViT/DiT fixtures dump fp32 on CPU**; transformers ≥5 zeroes custom rotary buffers (assert non-zero), `.float()` drops a complex buffer's sine half; an oracle that IMPROVES when a transform is removed had it disabled.

### Licensing & third-party code

Ported kernels + vendored code are enumerated in `NOTICE` (the ONE place); `LICENSE`/`LICENSE-APACHE-2.0`/`NOTICE` ride every packaging path, pinned by `tests/test_release_workflow_gates.sh`. To enumerate ports, grep comments for `mlxfast|oMLX|MTPLX|mlx-lm|port`. Full story: `docs/reference.md` "Licensing".

### App (Swift)

All app-side rules: `app/CLAUDE.md`; stories: `docs/gotchas/app.md`. Crossovers: agent budgets derive from advertised `context_length`; the app ALWAYS emits memory-critical launch flags (`ServerOptions` defaults mirror Zig defaults — change both together); the two bundle binaries move together.
