# Plan: close the ~4× Laguna decode gap vs mlx-lm (match or beat 68.9 tok/s)

## TL;DR for the next agent

Laguna support is **implemented, correct, and GPU-validated** (`feature/laguna` worktree; see
`laguna-plan.md` + memory `project_laguna_port`). The open problem is **pure decode speed**: on
this **M4 Max**, our engine decodes the pipenetwork 2-bit at **16.7 tok/s** while stock **mlx-lm
does 68.9 tok/s on the exact same weights, machine, and prompt** — a **~4.1× gap**. Your job is to
find and close it.

**Do NOT re-derive the following — already established this session:**
- **Not hardware.** mlx-lm hits 68.9 on our M4 Max ≈ the public M5-Max benchmark's 68.49. M4-vs-M5
  barely matters for this model; the whole gap is our engine.
- **Not DFlash / spec decode.** mlx-lm's 68.9 is raw autoregressive decode (no draft/MTP). DFlash
  would sit *on top*. It is not the answer.
- **Not CPU-bound.** GPU "Device Utilization %" (ioreg IOAccelerator) sits at **100%** during our
  decode → we are **GPU-bound**.
- **Not a missing async pipeline.** Our `Generator.next()` already async-evals the next token's
  graph while reading the current one (`src/generate.zig:1360-1374`, `:3864`), same as mlx-lm's
  `generate_step._step` (`mx.async_eval(y); mx.async_eval(next_y); mx.eval(y)`).
- **Not `mx.compile`.** mlx-lm does **not** compile the model forward (grep-verified: no `mx.compile`
  in its `generate.py` or `laguna.py`). Both build lazy graphs and let MLX auto-fuse. So "we should
  compile" is not obviously the fix (though compiling *our* per-token graph is a candidate lever —
  see H5).
- **Not a different MoE algorithm.** mlx-lm's `SwitchGLU` is nearly identical to our `moeMLP2`
  (same `do_sort = indices >= 64` threshold, same expand→gate/up/down gather → squeeze).

So: **same algorithm, same primitives, same pipeline, GPU-bound, 4× slower.** That points at a
**specific kernel/dispatch inefficiency** (wrong kernel variant, non-contiguous inputs forcing
copies, redundant per-token ops, or far more/smaller dispatches), findable only with a **GPU kernel
profile**. Profile first; do not guess-and-optimize.

## Success criteria

1. **Match or beat mlx-lm decode on this M4 Max**: ≥ ~68 tok/s on the pipenetwork 2-bit at the
   matched prompt below (256-token, temp 0, warm, PLD off). Stretch: beat it.
2. Re-check the 4-bit and (if re-downloaded) nvfp4 variants don't regress and ideally see the same
   speedup — the fix should be systemic (all three sit at ~60-85 ms/token today).
3. **No correctness regression**: chat/thinking/tools/long-ctx smoke still pass (see
   `laguna-plan.md` Verification), `zig build test` stays 9/9 / 0-fail, and every optimization is
   **kill-switched + bit-identical to its off state** (engine convention — see the MTP/prefill
   kernel rules in root `CLAUDE.md`).
4. Also measure **prefill** on a matched ≥1K-token prompt (the public benchmark shows mlx-lm ~1411
   prefill tok/s @ 1K on M5; we haven't measured a clean matched prefill number — do it, it may be
   a second gap).

## The reproducible same-hardware baseline (build this first)

**mlx-lm reference** (the number to beat). Stock mlx-lm has no `laguna` arch — the model repo bundles
an mlx-lm loader (`laguna.py`) at a pinned HF revision:

```sh
# pinned versions match the public benchmark (mlx-lm 0.31.3, mlx 0.32.0)
mkdir -p /tmp/mlxlm-baseline && cd /tmp/mlxlm-baseline
REV=5a67ae47cdc38ec7d16a09f9efb7add1bb631131
curl -sL "https://huggingface.co/pipenetwork/Laguna-S-2.1-MLX-2bit/resolve/$REV/laguna.py" -o laguna.py
uv venv --python 3.13
uv pip install "mlx-lm==0.31.3" "mlx==0.32.0"
MODELS=$(.venv/bin/python -c "import os,mlx_lm;print(os.path.join(os.path.dirname(mlx_lm.__file__),'models'))")
cp laguna.py "$MODELS/laguna.py"          # relative imports require it to live in mlx_lm/models/
export HF_HUB_OFFLINE=1
M=~/.mlx-serve/models/pipenetwork/Laguna-S-2.1-MLX-2bit
.venv/bin/python -m mlx_lm generate --model "$M" \
  --prompt "Explain how a hash map works in detail." --max-tokens 256 --temp 0.0
# → "Generation: 256 tokens, 68.939 tokens-per-sec", Peak memory ~37.9 GB
```

**Our engine** (the number to move). Kill any mlx-serve first (never load two big MLX models at
once — OOMs the GPU):

```sh
cd <this worktree>; export PATH="$PWD/.zig-toolchain:$PATH"
zig build -Doptimize=ReleaseFast            # ALWAYS ReleaseFast — Debug is 2-4× slower (fake numbers)
./zig-out/bin/mlx-serve --model ~/.mlx-serve/models/pipenetwork/Laguna-S-2.1-MLX-2bit \
  --serve --host 127.0.0.1 --port 11500 --metrics --no-pld &
# warm it (run the request twice), then read the 2nd run's decode rate from the server log:
curl -s http://127.0.0.1:11500/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"Laguna-S-2.1-MLX-2bit","messages":[{"role":"user","content":"Explain how a hash map works in detail."}],
  "max_tokens":256,"temperature":0}' >/dev/null
grep -oE "decode: [0-9.]+ tok/s" ~/.mlx-serve/logs/mlx-serve-11500.log | tail -1   # → 16.7 tok/s today
```

GPU-util sanity (no sudo): `ioreg -r -d 1 -c IOAccelerator | grep -oE '"Device Utilization %"=[0-9]+'`
while a long generation runs → 100% today.

**Environment**: this machine is **Apple M4 Max, 128 GB, macOS 26.5, no NAX** (`applegpu_g16s`).
The public benchmark repo (`github.com/tanishq-dubey/macos-laguna-s2.1`) is M5 Max / macOS 27 — do
NOT compare our M4 numbers to their M5 numbers; use the mlx-lm-on-this-M4 baseline above.

## Investigation (profile-first — this is the core of the work)

### Step 1 — GPU kernel timeline profile (find WHERE the 60 ms/token goes)
The gap is GPU-bound, so get a per-kernel GPU timeline for one decode step and compare op-for-op
against mlx-lm.

Options (any that works):
- **Instruments → Metal System Trace**, attach to the running `mlx-serve` process during a
  sustained decode. Look at: total kernels dispatched per token, per-kernel GPU time, and **gaps
  between kernels** (launch/serialization overhead vs useful compute).
- **mlx metal capture on the mlx-lm side** (easiest reference): in a scratch Python script call
  `mx.metal.start_capture("mlxlm.gputrace")` around a `_step`, open in Xcode. Gives you mlx-lm's
  kernel count + timing per token as the target to match.
- **Coarse per-sub-block timing in our decode** (quick first cut, invasive): env-gate a block that
  wraps attention / MoE / lm_head in the decode forward with `mlx_array_eval` + a stopwatch (this
  breaks the async pipeline, so use it only to get a rough %-split, not absolute numbers). Also
  instrument mlx-lm's `laguna.py` forward the same way (add `mx.eval` + `time.perf_counter` around
  its `self_attn`, `mlp`, and `lm_head`) and compare the split. **This single comparison — which
  sub-block dominates and by how much vs mlx-lm — likely names the culprit.**

Expected shape of the answer: either (a) one op class (MoE expert gather, attention matmuls, or
lm_head) is far slower than mlx-lm's equivalent, or (b) we dispatch many more small kernels (gaps
dominate). (a) → fix that op; (b) → fusion / fewer dispatches.

### Step 2 — ranked hypotheses (test against the profile, cheapest first)

- **H1 — non-contiguous inputs / wrong quant-kernel variant (most likely).** MLX picks a fast qmv
  kernel only for the right input layout/contiguity; a strided/transposed input can force a copy or
  a slower general kernel. Our attention feeds strided views (transposes), and `moeMLP2` reshapes
  `expert_x` to a 5-D `[B,S,1,1,D]` before `gather_qmm` (`src/transformer.zig:9330+`) — verify these
  land the same kernel mlx-lm's `SwitchLinear`/`QuantizedLinear` hit. Check `ensure_row_contiguous`
  semantics and whether we should materialize contiguity before the qmatmul. Compare our
  `qmatmulBits` / `gatherExpertMm` (`:10814`, `:10903`) call shapes vs mlx-lm `SwitchLinear.__call__`.
- **H2 — redundant per-token ops.** Per token × 48 layers we may rebuild things mlx-lm reuses or
  skips: sliding masks, RoPE, the YaRN mscale multiply, the softplus gate (astype→logaddexp→astype→
  reshape→multiply), extra reshape/squeeze in the MoE (5-D expand+squeeze vs SwitchGLU's
  `expand_dims(x,(-2,-3))` + `squeeze(-2)`). Each is a separate kernel; at 48 layers they add up.
  Diff `lagunaAttnWith` (`:8448`) and `moeMLP2` (`:9159`) op-by-op against `laguna.py`'s `Attention`
  and `LagunaSparseMoeBlock`.
- **H3 — KV-cache update cost.** `updateDense` (`:2338`) does `slice_update` + view slicing per
  layer per token. Confirm it's O(1) amortized (chunked growth, buffer donation), not copying the
  whole KV each token. mlx-lm uses `KVCache`/`RotatingKVCache` — compare.
- **H4 — 2-bit expert kernel path.** The 2-bit is mixed 2/4/8-bit; confirm `computeQuantParams`
  resolves each expert weight to the fast affine gather kernel and not a fallback. Try the 4-bit
  (uniform) variant — if the gap shrinks at 4-bit, the 2-bit kernel path is implicated.
- **H5 — fuse our per-token graph.** Even though mlx-lm doesn't compile, our graph may have more
  dispatch overhead; wrapping the decode forward (or the hot sub-block) in `mx.compile` /
  `mlx_compile` could collapse the elementwise chains (gate/norm/rope/mscale). Only pursue if the
  profile shows dispatch-gap domination (Step 1 case (b)).
- **H6 — something computes more than it should.** Rule out: are we gathering only top-10 experts
  (not all 256)? Is attention S=1 (not re-reading full context as a matmul)? Is the lm_head single
  matmul? A profile makes this obvious.

### Step 3 — fix, A/B, verify
For each change: implement behind an env kill-switch, prove the off-state is byte-identical
(temp-0 greedy output unchanged), then A/B **on this M4 Max vs the mlx-lm baseline** (warm, matched
prompt, PLD off). Keep `zig build test` 9/9. Re-run the correctness smoke (chat/thinking/tools/
long-ctx) — a speed win that corrupts output is not a win. Log any coverage/kill-switch per the
"no silent caps" rule.

## Code landmarks

**Ours (`src/`):**
- `generate.zig:3864` `Generator.next()` — the pipelined single-token step; `:1360-1374` the initial
  async-eval; `lazyForward` `:4171`.
- `scheduler.zig:4037` — the server's regular decode path calls `gen.next()` (this is what the HTTP
  benchmark exercises; laguna is MoE → non-batchable → serial, `modelBatchable` false).
- `transformer.zig:8448` `lagunaAttnWith` — per-token attention (q/k/v/o + g_proj matmuls, QK norm,
  YaRN/default rope, mscale, sdpa, softplus gate).
- `transformer.zig:9159` `moeMLP2`; decode branch `:9330+`; `gatherExpertMm` `:10814`; `qmatmulBits`
  `:10903`. `forwardMoeWith` is the 48-layer loop.
- `transformer.zig:2338` `updateDense` (KV cache write/view).

**mlx-lm reference** (fetch `laguna.py` from HF rev `5a67ae47` as above; the rest is in the venv's
`mlx_lm/`): `models/laguna.py` (`Attention`, `LagunaSparseMoeBlock`, `Model`), `models/switch_layers.py`
(`SwitchGLU`, `SwitchLinear`), `generate.py` (`generate_step`, `_step` — the pipeline).

## Notes / gotchas

- Measure with `--no-pld` for a clean AR comparison; PLD gates off on non-repetitive prompts anyway
  but be explicit. Never quote a spec-decode cell as a raw-decode number.
- The pipenetwork 2-bit **dropped `mlp.gate.e_score_correction_bias`** (our loader zero-fills it →
  plain sigmoid top-k). Fine for a speed comparison (mlx-lm's loader does the same); just don't
  chase a routing-quality ghost.
- Doc discrepancy to cross-check only if you touch numerics: mlx-lm's `laguna.py` header comment
  says full-attn YaRN "factor 128", but `config.json` (and our port) use **factor 32**. The config
  is authoritative; the comment is likely stale. Irrelevant to the speed gap.
- ReleaseFast only; rebuild `zig-out/bin/mlx-serve` before every A/B (`zig build test` does NOT
  refresh the exe).
- Full context on the implementation + prior validation: `laguna-plan.md`, memory
  `project_laguna_port`; the general "our decode has per-token overhead" theme also appears in
  `project_beat_omlx_prefill_stack` (that was Qwen-MTP round composition — related mindset, different
  model).
```

## One-line problem statement to re-orient fast

> On M4 Max, our engine decodes Laguna-2bit at 16.7 tok/s vs mlx-lm's 68.9 (same weights/machine/
> prompt); GPU is 100% util and the algorithm/pipeline match mlx-lm, so a GPU kernel profile must
> localize the ~45 ms/token of extra per-token GPU cost. Match or beat 68.9.
