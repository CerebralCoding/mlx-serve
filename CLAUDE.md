# mlx-serve – project context for AI

Native Zig server that runs MLX-format LMs on Apple Silicon and exposes OpenAI-compatible and Anthropic-compatible HTTP APIs. No Python.

## Stack

- **Zig** 0.15+; **mlx-c** (Apple) via Homebrew, FFI in `src/mlx.zig`
- **Jinja engine** (`lib/jinja_cpp`): llama.cpp's C++17 Jinja2 + nlohmann/json, pre-compiled as `libjinja.a`
- **stb_image** (JPEG/PNG) + **libwebp** (WebP) for vision; **safetensors** weights; BPE tokenizers

## Layout

### Zig server (`src/`)

| Path | Role |
|------|------|
| `main.zig` | Entry, CLI flags |
| `mlx.zig` | mlx-c FFI |
| `model.zig` | Config + safetensors loading |
| `tokenizer.zig` | BPE tokenizer |
| `transformer.zig` | Forward pass; arch dispatch (attention, MLP, MoE, GatedDeltaNet) |
| `generate.zig` | Autoregressive generation, sampling, PLD/drafter orchestration |
| `chat.zig` | Chat templates (ChatML, Gemma, Llama-3, Jinja2); thinking tags; tool call parsing |
| `vision.zig` | Gemma 4 SigLIP vision encoder |
| `server.zig` | HTTP: `/health`, `/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/v1/messages`, `/v1/responses`, WebSocket on `/v1/responses` |
| `responses.zig` | OpenAI Responses API: parser, envelope, in-memory `ResponseStore`, compaction blob |
| `ws.zig` | RFC 6455 WebSocket framing (server-side, generic over `Conn`) |
| `pld_index.zig` | PLD n-gram index (`PldLookup.findMatch`, `ngramRepeatScore`) |
| `drafter.zig` | Gemma 4 assistant drafter (cross-attention spec-decode) |
| `status.zig` | TUI status bar |
| `log.zig` | Leveled logging |
| `build.zig` | Links mlx-c, libjinja, libwebp, stb_image |

CLI flags: `--model --serve --host --port --prompt --max-tokens --temp --ctx-size --timeout --reasoning-budget --no-vision --pld --pld-draft-len --pld-key-len --drafter --draft-block-size --log-level --version --help`

### Swift macOS app (`app/Sources/MLXServe/`)

| Path | Role |
|------|------|
| `MLXServeApp.swift` | App entry, menu bar + Chat/Browser windows |
| `AppState.swift` | Global state, chat session persistence |
| `Models/{ChatModels,AgentModels}.swift` | `ChatMessage`, `ChatImage`, `SerializedToolCall`, `AgentPlan` |
| `Services/APIClient.swift` | HTTP + SSE streaming client |
| `Services/AgentPrompt.swift` | System prompt, 10 tools, `SkillManager` |
| `Services/AgentEngine.swift` | Shared agent logic: history, tool exec, repetition tracking, overflow |
| `Services/ToolExecutor.swift` | Tool handlers (shell, file, search, browse, webSearch, saveMemory) |
| `Services/{ImagePreprocessor,BrowserManager,ServerManager,TestServer,AgentMemory}.swift` | Image prep, WKWebView, server lifecycle, embedded test server (port 8090), agent context |
| `Views/{ChatView,StatusMenuView,BrowserView}.swift` | Chat UI + `runAgentLoop()`, menu bar, browser |

## Building

- **Always use `./app/build.sh` for the Swift app, not direct `swift build`** — the script knows the right Swift-version flags, links Zig artifacts, signs the bundle, and keeps `MLXCore` + `mlx-serve` in lockstep. Skip notarization in dev with `SKIP_NOTARIZE=1 bash app/build.sh`.
- Zig only: `zig build -Doptimize=ReleaseFast` (needs `brew install webp`)
- Direct `swift build` (escape hatch only — fast iteration on a Swift-only change): `cd app && swift build -c release -Xswiftc -swift-version -Xswiftc 5`. Don't ship a build that didn't go through `build.sh`.
- **Rebuild Jinja** (after `lib/jinja_cpp/*.cpp` changes): `cd lib/jinja_cpp && for f in jinja_wrapper caps lexer parser runtime jinja_string value; do clang++ -std=c++17 -O2 -DNDEBUG -I . -c $f.cpp -o obj/$f.o; done && ar rcs libjinja.a obj/*.o`

The `-Xswiftc -swift-version -Xswiftc 5` flag forces Swift 5 mode under Swift 6.3 (Xcode 26+) — required because the pinned `swift-sdk` 0.10.x emits `[#SendingRisksDataRace]` errors otherwise. Pin held at 0.10.x for macos-14 / Swift 6.1 CI compat. `app/build.sh` already passes the flag; only direct `swift build`/`swift test` need it.

## Testing

- Always add tests for changes; unit tests OK but integration tests with real models are the real tests
- Do things in a TDD style, write failing test first, then implement it.
- Cover all supported architectures, not just one
- After big features: build mlx-serve + app bundle, run `.app` with TestServer.swift enabled, test agentic harness
- Always run `zig build test` and `swift test` before submitting

| Command | Purpose |
|---|---|
| `zig build test` | Zig unit tests |
| `cd app && swift test` | Swift unit tests |
| `./tests/integration_test.sh [model_dir] [port]` | 36 end-to-end API tests |
| `./tests/test_tool_response.sh [port]` | Tool calling round-trip |
| `./tests/test_kv_cache_poison.sh [port]` | KV cache poisoning regression |
| `./tests/test_anthropic_api.sh [port]` | Anthropic Messages API |
| `PLD_TEST_MODEL=<dir> ./tests/test_pld_equivalence.sh` | PLD byte-equivalence (default gemma-4-e4b-it-8bit) |
| `./tests/test_streaming_pld.sh [port]` | Streaming PLD byte-identical to non-streaming |
| `./tests/test_drafter_equivalence.sh [port]` | Gemma 4 drafter byte-equivalence |
| `UD_MOE_MODEL=<dir> ./tests/test_ud_moe.sh` | Unsloth UD MoE load + generate (default Qwen3.6-27B-UD-MLX-4bit) |
| `./tests/test_long_agent_memory.sh [port]` | 10-turn Claude-Code-style agent: plants 3 facts in turn 1, recalls them across mode transitions (tools on/off, thinking on/off). Catches "model acts like first-time-seen" regressions. |
| `./tests/bench_spec.sh [runs]` / `--corpus` | Spec-decode benchmark + threshold-tuning corpus |

## Versioning & Releases

CalVer `YY.M.N` (e.g., `v26.4.25` = 2026, April, 25th release). `N` auto-increments from the last GitHub release for that `YY.M` prefix; `build.sh` computes via `gh release list`.

**Version sources**: `app/Info.plist` (`CFBundleVersion`/`CFBundleShortVersionString`), Zig `-Dversion` build option (`build_options.version`), git tag (`gh release create v{version}`).

**Release**:
1. Update `CHANGELOG.md` with NEXT version (check `gh release list --limit 1` first — never reuse an existing tag)
2. Commit + push
3. `cd app && SKIP_NOTARIZE=1 bash build.sh` — prints the `gh release create` command
4. Run that command

### CHANGELOG style

**One entry per shipped release. No new entries for unshipped work — fold it into the next pending entry.** Always run `gh release list --limit 1` first; if the topmost CHANGELOG entry is newer than the latest GitHub release, that entry is unshipped and any new bullets get merged into it. Never bump version numbers ahead of an actual release.

Tone: high-level executive bullets, marketing-style. The audience is users/integrators, not contributors reading the diff.

- Lead each bullet with **what changed for the user** (capability, speed, model support), not the implementation.
- Quantify where impressive — concrete tok/s percentages, model names, the workload it applies to.
- Avoid: file paths, function names, internal symbol renames, line-count diffs, "we discovered that…", PR/issue numbers.
- 4–7 bullets per release. If you need more, the release is too big and should ship sooner.

Template:

```markdown
## vYY.M.N — Two-to-five-word headline

- **<User-visible thing>**: one or two sentences on the impact. Numbers if you have them.
- **<New model / API / behavior>**: what unlocks, when it kicks in, what stays the same.
- **<Speed or reliability win>**: workload + measured gain.
- **<Removed / deprecated thing, if any>**: why, and what users should do instead.

---
```

When in doubt, look at the existing entries (v26.5.4 and earlier) — keep the same density and tone.

## Benchmarking

Run `./bench.sh` after major features/optimizations; results go in `BenchmarkLog.md`. Flags: `--model gemma`, `--no-mlx-lm`, `--runs 5`.

## Conventions

- Minimal, DRY Zig; avoid unnecessary abstraction
- Tests at the bottom of each source file (Zig convention)
- Shell integration tests in `tests/`, need a running server
- Chat templates live in model dirs; Jinja renders them with fallback formatting
- Concurrent health checks via threaded connections; single-slot generation
- KV cache reuse via prompt-prefix matching; invalidated after tool calls and pad-only generations

## Supported Architectures

Dispatched on `model_type` in `config.json` via `model.zig` (config/weights) and `transformer.zig` (forward).

| `model_type` | Family | Weight prefix | Vision | MoE | Notes |
|---|---|---|---|---|---|
| `gemma4`, `gemma4_text` | Gemma 4 | `language_model.model` | SigLIP | -- | Full vision, clipped linears, PLE |
| `gemma3` | Gemma 3 | `language_model.model` | -- | -- | |
| `qwen3` | Qwen 3 | `model` | -- | -- | QK norm |
| `qwen3_5`, `qwen3_5_moe(_text)` | Qwen 3.5/3.6 | `language_model.model` | -- | Optional | GatedDeltaNet + MoE/dense, shared expert routing |
| `qwen3_next` | Qwen 3-next | `model` | -- | Optional | DeltaNet |
| `nemotron_h` | Nemotron-H | `backbone` | -- | -- | Hybrid transformer + Mamba2 SSM |
| `lfm2` | Liquid LFM2.5 | `model` | -- | -- | Hybrid gated conv + full attention |
| `llama`, `mistral` | Llama/Mistral | `model` | -- | -- | |

**TODO**: `lfm2-vl` (vision encoder), `phi`/`phi3` (different layout), `command-r` (different arch).

Models with `vision_config` but no vision weights (e.g., text-only quantized Qwen 3.5) gracefully disable vision at init. Swift app flags unsupported archs via `supportedModelTypes` in `HFModels.swift`.

## OpenAI Responses API (`/v1/responses`)

Pure data in `responses.zig`; HTTP/orchestration in `server.zig`. Supports `POST`/`GET`/`DELETE /{id}`.

- **Envelope** (`buildResponsesEnvelope` + `ResponseEcho`): echoes `tools`, `tool_choice`, `text`, `reasoning`, `usage` (with `cached_tokens` + `reasoning_tokens`), `truncation`, `parallel_tool_calls`, sampling params, `metadata`, etc. Renderers reshape into the strict ResponseResource schema (flat tool form, not nested chat-completions).
- **Streaming SSE**: `response.created`, `response.in_progress`, `response.output_item.added`, per-type deltas/`.done`, `response.completed`. **Every event needs `sequence_number`** — `sendResponsesEvent` injects it; the POST handler threads a per-request `seq_num`.
- **Stateful chains**: `ResponseStore` keyed by id. `previous_response_id` replays history; missing → 404. `inputContainsFunctionCallOutput` triggers final-answer mode (tools disabled).
- **Compliance**: `experiments/openresponses` validates strict schema; currently 17/17. `top_level response_format` accepted as alias for `text.format`.
- **Compaction (`POST /v1/responses/compact`)**: pure data, no LLM call. Synthesizes opaque base64 `encrypted_content` over `{"v":1,"msgs":[...]}`. `appendCompactionInputItem` reconstitutes on round-trip. `model` required (422 on missing). Drops tool calls + images.
- **WebSocket transport (`ws[s]://host/v1/responses`)**: same endpoint, opt-in via `Upgrade: websocket`. Each text frame is a `response.create`-shaped JSON; SSE events become single WS text frames via `WsBridge` on `Conn.ws_mode`. **No `[DONE]` on success** (`response.completed`/`.failed`/`.incomplete` is the terminator). Sequence numbers reset per response. `WsLocalCache` holds `store: false` responses for the connection lifetime; failed continuations evict the chain root.

## Anthropic Messages API (`/v1/messages`)

For Claude Code and Anthropic SDK clients with local models.

- `system` (top-level) → internal system message
- Typed content blocks (`text`, `tool_use`, `tool_result`, `thinking`) → internal `Message` structs
- `input_schema` → OpenAI `parameters` for chat templates
- `tool_result` in user messages → internal `role: "tool"`
- `thinking` → `enable_thinking` + `reasoning_budget`; thinking blocks emit fake `signature`
- Stop reasons: `stop`→`end_turn`, `length`→`max_tokens`, `tool_calls`→`tool_use`
- SSE events: `message_start`, `content_block_{start,delta,stop}` (with `text_delta`/`thinking_delta`/`signature_delta`/`input_json_delta`), `message_{delta,stop}` — explicit start/stop lifecycle per indexed block

**Claude Code launcher**: app sets `ANTHROPIC_BASE_URL`, dummy `ANTHROPIC_API_KEY`/`ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_DEFAULT_*_MODEL=mlx-serve`, `CLAUDE_CODE_SUBAGENT_MODEL=mlx-serve`.

## Tool Calling

### Server (Zig)
- **Detection**: when `tools` present, server buffers tokens, checks `<tool_call>`, Hermes XML, Gemma 4 `<|tool_call>`, raw JSON. Gemma 4 double-brace args (`{{...}}`) unwrapped before parse. Thinking tokens buffered separately if enabled.
- **Serialization** (`chat.serializeMessagesJson`): `role: "tool"` passed natively (Gemma 4 templates render `<|turn>tool` directly). Tool call `arguments` serialized as JSON strings, not objects.
- **Streaming**: full args sent in one SSE delta (avoids client accumulation bugs). Thinking buffered until close tag, emitted as `reasoning_content`.
- **Fallback** (`chat.fallbackFormatChat`): ChatML, Llama (`ipython` role), Gemma (`Tool result:` in user turn).
- **KV cache**: `reuseKVCache()` token-by-token prefix compare; auto-invalidated after tool calls and pad-only gens. Sliding-window layers keep full buffer; views slice to last `sw` during decode.

### Client (Swift)
- **Agent loop** (`ChatView.runAgentLoop`): up to 150 iters; tools → parse → exec → feed back → repeat. Synthetic user nudge after tool results for some models.
- **History builder**: filters errors/pad-only/summaries, truncates assistant at 500 chars, budget-aware (walks backward, fits within `context_length - max_tokens - system_prompt`), pins first user + first assistant, auto-compacts tool results when tight.
- **SSE parsing**: accumulates tool-call deltas; emits `.toolCalls` on `finish_reason: "tool_calls"`; fallback if stream drops without `finish_reason`.
- **Storage**: `SerializedToolCall` (id, name, args as JSON string) on `ChatMessage.toolCalls`; Codable-persisted; backwards-compat (optional field).
- **Error recovery**: tool errors include sent args + retry hint → enables self-correction.

## Prompt-based Skills

`.md` files in `~/.mlx-serve/skills/` with YAML frontmatter:
```markdown
---
name: deploy
description: Deploy to production
trigger: deploy, release, ship it
---
Steps...
```
- `trigger`: comma-separated, case-insensitive substring match in user message → body injected into system prompt
- Skill index (name + description) always included
- `SkillManager` re-scans on dir mtime change
- UI: folder icon opens `~/.mlx-serve/` in Finder

## Resumable Downloads

`DownloadManager` streams to `<file>.partial` via `URLSessionDataTask`. Resume sends `Range: bytes=<existing>-`; 206 → continue, 200 → truncate+restart. 3 retries with 2s/4s backoff. Cancel preserves `.partial`. UI shows "Resume" when `hasPartialDownload()`. Already-complete files (size matches HF metadata) are skipped.

## Debugging

### Server logs
- `--log-level debug` for verbose output
- App captures stderr in `ServerManager.serverLog` (64KB rolling buffer); view via log button in menu bar
- Manual: `./zig-out/bin/mlx-serve --model <path> --serve --port 8080 --log-level debug 2>&1`
- Patterns: `jinja error:`, `[cache] reusing N/M tokens`, `[cache] invalidated`, `<- N+M tokens (Xms) [reason]`, `tool_msgs=N`, `[spec-stats] mode=...`, `spec-gate: ngram-score=...`

### Swift logs
- `print()` invisible when launched via `open` — run binary directly or write to file
- Every agent request dumped to `~/.mlx-serve/last-agent-request.json`; replay via curl
- Chat history at `~/.mlx-serve/chat-history.json`

### Reproducing
- Tool calling: curl with `stream: false` first, then `stream: true`
- Jinja offline: `pip3 install jinja2`, render `chat_template.jinja` with dumped request
- KV cache: `pkill -f mlx-serve` between tests — one bad request can poison the cache

## Gotchas

### KV cache after tool calls
Generated tool-call tokens are in the cache but not in `cached_prompt_ids` → reusing for the next request (with tool results) corrupts attention. Auto-invalidated. Pad-only generations also trigger invalidation.

### Sliding window KV cache
Gemma 4 E4B (512-token window) keeps full buffer — no trimming. Prefill returns all entries (Q/K dim match); decode views return last `sw`. Mask handles attention scope. Matches mlx-lm `RotatingKVCache`.

### Gemma 4 tool calling
Templates render `role: "tool"` natively as `<|turn>tool` — no transformation. Don't add `tool_responses` field (causes duplicate content). Args serialized as JSON strings.

### Streaming with tools + thinking
Server buffers tokens to detect tool patterns. With thinking enabled, `<|channel>thought` is buffered (not flushed) until closing `<channel|>`. After generation, thinking is split into `reasoning_content`; channel tags stripped from visible content.

### SSM/GatedDeltaNet state init
`conv1dWithCache` sets `ssm.initialized = true` after conv update but BEFORE SSM recurrence state exists. Init code must check `ssm.ssm_state.ctx == null`, NOT `!ssm.initialized`. Used by both `mamba2Mixer` and `gatedDeltaNet`.

### Parameter-free RMS norm
mlx-c crashes on null/empty weight for `mlx_fast_rms_norm`. Pass `ones([dim], bfloat16)` for parameter-free norm. Affects GatedDeltaNet Q/K norm and Mamba2 group norm.

### Nemotron-H time_step_limit
Python defaults to `(0.0, inf)` (no dt clipping). `time_step_min`/`time_step_max` in config.json are NOT used by Python for SSM clipping. Only `time_step_limit` JSON array overrides.

### Speculative decoding (PLD + drafter) — overview

Two paths share a verify invariant: `cache.step = prompt_len + tokens_emitted`, t1 NOT in cache on entry, no pending state. Verify input is `[t1, draft[0..m-1]]` length `1+m`; full accept samples `new_t1` from `verify_logits[m]` (bonus prediction); partial accept rolls back via `KVCache.snapshot/restore` + `ssmSnapshot/Restore` and re-forwards `[t1, draft[0..accepted-1]]`. `accepted=0` still re-forwards `[t1]`. Pending correction sampled from *original* `verify_logits[accepted]` (NOT re-forward); index is `accepted` not `accepted-1` — off-by-one silently corrupts output, guarded by `tests/test_pld_equivalence.sh`.

**PLD** (`src/pld_index.zig`, `Generator.nextPld`): model-agnostic n-gram match in `prompt + generated`. CLI `--pld --pld-draft-len 5 --pld-key-len 3`; per-request `enable_pld`. `Generator.initWithOptions` clones prompt to `prompt_ids_owned` (caller-supplied freed before `nextPld`). Stochastic verify: draft as one-hot; `accept_prob = min(1, target_p[draft[i]])`, residual `max(target_p − one_hot, 0)` renormalized — preserves marginal per Leviathan. One-hot built via `pldOneHotRow` (no scatter).

**Drafter** (`src/drafter.zig`, `Generator.nextDrafter`): Gemma 4 only. 4-layer, hidden 256, no K/V projections — cross-attends into target's K/V via layer-type mapping (drafter sliding → target last sliding; drafter full → target last full). Loaded via `--drafter <dir>`. `block_size` auto-detected per target (E2B=2, E4B=4, 26B-A4B=4, 31B=8 — matches vLLM PR #41745) via `recommendedBlockSize`; override with `--draft-block-size`. Input: `concat([target.embed(prev) * sqrt(target.hidden), h_prev], -1)` → drafter hidden 256. Autoregressive within round (`block_size − 1` drafts), constant RoPE offset. Sparse `MaskedEmbedding` LM head (~2048 centroids, top-32 → ~4096 token logits of 262144). Linear weights pre-transposed at load.

**Validation**: `error.UnsupportedDrafterArch` (model_type mismatch), `error.DrafterTargetMismatch` (hidden_size or layer_types incompatible).

**`forwardCaptureHidden`**: `forwardStandard` and `forwardMoe` honor `capture_hidden`, slicing post-final-norm hidden at LAST position. Drafter seeds first `h_prev`; PLD uses during partial-accept rollback. Other forward paths (BERT, hybrid) leave it empty — drafter/PLD not wired there.

**Auto-disable**: tools, `logprobs > 0`, grammar-constrained sampling. PLD works on hybrid SSM (LFM2.5, Nemotron-H) — see snapshot null-state guard below. Drafter is non-streaming-only in v1; streaming drafter requests fall through to regular streaming with a log line. Drafter > PLD > regular priority when both enabled.

**Adaptive prompt-time gate** (`spec_gate_threshold = 0.01` in `server.zig`): n-gram repetition score on tokenized prompt (`pld_index.ngramRepeatScore`, 3-grams). If `score < threshold` AND user didn't set `enable_pld:true`/`enable_drafter:true`, the flag is silently disabled. Runs in all three request paths; chat-completions logs `spec-gate: ngram-score=X.XXX` once per request. v4 corpus validation: 9/9 correct decisions; threshold 0.01 cleanly separates "any 3-gram repeats" from pure-novel prompts.

**Runtime acceptance gate** (`RUNTIME_GATE_MIN_PER_DRAFT_RATE = 0.50`, warmup 5): when per-draft acceptance falls below 50% mid-decode, `Generator.spec_disabled_runtime` flips on (sticky). Subsequent calls short-circuit to `Generator.next`, which has a transition shim: when no pending logits/token, sync `forward([next_token_id])` to seed pending_logits. Pre-v26.5.6 the gate compared per-round against 0.30 → almost never fired; 0.50 cleanly cuts creative-content tail (22-47%) while leaving heavy-echo (84-97%) untouched. Does NOT save MoE+drafter regressions where per-draft is high but verify cost dominates — handled by MoE default-off in `serve()`.

**Default-on policy**: PLD/drafter not flipped on at CLI; users still pass `--pld`/`--drafter`. While loaded:
- Dense Gemma 4 (E2B/E4B/31B) drafter: `enable_drafter` defaults TRUE per-request; gates handle creative content
- MoE Gemma 4 (26B-A4B) drafter: `enable_drafter` defaults FALSE — verify forward MoE expert-routing penalty makes drafter regress at batch=1 even at 97.8% per-draft (every block_size tested). PLD remains default-on (1.43× echo). Per-request override still works.

### PLD/drafter long-greedy byte-divergence at INT4
AR (`next`) forwards `[1,1,d]` qmv; verify forwards `[1,K+1,d]` qmm. INT4 float reductions in slightly different orders → near-tie argmax can flip → divergence cascades. First ~30–80 generated tokens at temp=0 are byte-identical (equivalence tests live here); beyond that, paths may diverge char-by-char while both being mathematically valid greedy outputs. At temp ≥ 0.01 the Leviathan sampler preserves the target distribution → exact past 30 tokens. **For byte-stable long-greedy at temp=0 on INT4: `--no-pld`, no `--drafter`.** For chat/agent (temp>0) spec-decode is exact and free.

### PLD on hybrid SSM (snapshot null-state guard)
`SSMCacheEntry` has two slots (`conv_state`, `ssm_state`) populated by different layer types: LFM2's `gated_conv` writes only `conv_state` (sets `initialized=true`) and never touches `ssm_state`. `mlx_array_set` with null source aborts via mlx-c's default handler (`exit(-1)`). `ssmSnapshot`/`ssmRestore` and `PrefillCache` save/restore must check each field's `.ctx != null` independently — `initialized` alone insufficient. This was the previous "off on hybrid SSM" auto-disable; lifted once per-field guard landed.

### mlx-c API changes
mlx-c 0.6.0 added a `global_scale` param (may be null) to `mlx_dequantize` between `mode` and `dtype`. FFI in `mlx.zig` must match installed header. When upgrading, diff `/opt/homebrew/include/mlx/c/ops.h` against `extern "c"` decls.

### Two binaries in the app bundle
`.app` contains `MLXCore` (Swift UI) AND `mlx-serve` (Zig server). Both must be updated together. Forgetting one after rebuild is a common "still doesn't work" cause.

### WebSearch + Browse
`webSearch` navigates DuckDuckGo HTML, extracts results via JS. `browse.readText` navigates first then extracts — ensures correct page (not previous).

### WKWebView main thread
`BrowserManager` is `@MainActor`. All WKWebView ops (navigate, readText, evaluateJS) on main thread. Created eagerly at app launch so tools work without Browser window open.

### Swift JSONSerialization quirks
- `[String: Any]` non-deterministic key order
- `""` stays `""` in JSON (not `null`); server treats both as empty
- `Double` like `0.7` → `0.69999999999999996` — fine
- `arguments` in tool_calls must be a JSON String (e.g., `"{\"command\":\"ls\"}"`), not nested dict; server checks `if (v == .string)`
