# Benchmark Log

Current benchmark state, one page: the final numbers, how to reproduce them, and the
rules that keep them honest. The full historical narrative (2026-04 → 2026-07 session
entries, bisects, retractions) lives in this file's **git history**.

Keep this document lightweight, data in table format only, no paragraphs, no fluff, no details.

Hardware: Apple M4 Max, 128 GB, AC power, idle machine. Engines: LM Studio 0.4.15+2,
MTPLX 2.0.2, mlx-lm 0.31.3. Identical weights within every row. Last refreshed
**2026-07-16** (feature/more-hy3-fixes soak; v26.7.7 baseline).

## Native-MTP context ladder — Qwen3.6-27B, identical checkpoint on all 3 engines

Model `Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed` (4-bit trunk + calibrated MTP
adapter; LM Studio has no MTP support and decodes plain AR). Coding-agent prompts,
temp 0.6, max_tokens 128, fresh loads, **cold prompts**, best-of-N per cell.
CSV: `docs/perf-csvs/mtp-ladder-26.7.6.csv` · chart: `docs/perf-pngs/perf-mtp-ladder-26.7.6.png`.

| prefill tok/s | 0.5k | 1k | 2k | 4k | 8k | 16k | 32k | 64k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **mlx-serve** | 243.8 | 248.1 | 239.3 | 242.4 | 235.0 | 224.4 | 209.3 | 186.0 |
| LM Studio (cold) | 229.2 | 243.5 | 245.2 | 239.7 | 239.3 | 226.5 | 214.7 | 189.5 |
| MTPLX | 237.9 | 244.7 | 235.1 | 236.8 | 231.3 | 216.4 | 205.1 | 180.3 |

| decode tok/s | 0.5k | 1k | 2k | 4k | 8k | 16k | 32k | 64k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **mlx-serve** (MTP d3) | 55.5 | 50.8 | 50.6 | 50.2 | 47.5 | 45.9 | 41.3 | 33.5 |
| LM Studio (AR) | 29.4 | 29.1 | 29.4 | 28.0 | 27.6 | 26.3 | 25.0 | 21.7 |
| MTPLX (MTP) | 46.6 | 40.4 | 43.4 | 38.5 | 40.3 | 37.6 | 37.2 | 29.9 |

Verdict: decode +11–30% vs MTPLX and +54–89% vs LM Studio at every rung; prefill ahead
of MTPLX at all 8 (+1.4–3.7%) and within ±2.5% of LM Studio everywhere.

## MTP — Qwen3.6-27B, checkpoint (2026-07-13)

Three shipped levers: deferred history append (one merged head forward per
round), the verify-width split-K qmm kernel family (`transformer.verifyQmm`,
ported from MTPLX turbo, Apache-2.0), and cross-request EV seeding
(`MLX_SERVE_MTP_EV_SEED`). EV costs refit twice along the way.

| protocol (mtp_cold/warm.sh, Youssofal artifact) | mlx-serve | MTPLX 2.0.2 |
|---|---:|---:|
| WARM before (same-session pair) | 59.6 / 63.1 | 75.5 / 73.5 |
| WARM after, pair 1 | 73.3 / 77.2 | 69.0 / 71.7 |
| WARM after, pair 2 | 75.6 / **79.5** | 73.8 / 71.1 |
| COLD after (first request, fresh boot) | **70.2** | 34.7 |

| round T(m) ms (saturated fixed depths, same-session) | T(1) | T(3) | T(6) |
|---|---:|---:|---:|
| Phase-0 baseline | 43.4 | 76.3 | 119.2 |
| + deferred history append | 42.0 | 62.6 | 111.6 |
| + verify-qmm kernels (ksplit + msg lm_head tile) | 44.6 | 54.4 | 89.9 |

| attribution (MLX_SERVE_GDN_UBENCH=1 µbench, 27B geometry) | T=2 | T=64 | T=2048 |
|---|---:|---:|---:|
| GDN recurrence kernel ×48 layers (ms) | 6.7 | 12.6 | 214 |
| bare 4-bit qmm ×48×4 (ms) | 42 | 213 | 5317 |
| live forward ladder (ms) | 44 | 287 | ~8700 |

Verdict: the forward ladder is MLX qmm row-count cost, NOT GDN — chunked-GDN
and compiled-draft-chain levers measured ≤2.5% prefill / dead-even decode and
were not shipped; the verify-SHAPED qmm kernel is what closed (then flipped)
the warm gap. Dead ends kept honest: sampled drafts lose to greedy on our
stack (65.8/64.1 vs 67.6/69.4); M=8 kernel rows are a spill cliff (636 ms
rounds) — cap M=7, adaptive depth cap 6. Creative temp-0.8 after a code-hot
seed: 30.4–37.7 vs AR ~28 (demotes + sticky-disables cleanly; disabled runs
never publish the seed). Guards: verifyQmm parity test, MTP equivalence 9/9,
gemma-4bit PLD byte-equivalence (kernel engages there too), full suite green.

## vs LM Studio — family matrix (ctx 4096, temp 0, best cell vs best cell)

CSVs: `docs/perf-csvs/{gemma,qwen36}-26.7.6.csv` · chart:
`website/perf-vs-lmstudio-qwen36-26.7.6.png`. Geomean over 18 cells: **+47.7%**
(**+38.0%** with native-MTP cells excluded).

| Model | Echo | Code | Free-form |
|---|---:|---:|---:|
| Gemma 4 E2B | +111% | +25% | +2% |
| Gemma 4 E4B | +113% | +64% | 0% |
| Gemma 4 31B | +98% | +24% | −1% |
| Gemma 4 26B-A4B-MoE (QAT) | +62% | +5% | 0% |
| Qwen 3.6 27B | +112% | +98% | +32% |
| Qwen 3.6 35B-A3B-MoE | +132% | +70% | +32% |

### Refresh 2026-07-16 (`feature/more-hy3-fixes` soak) — all 4 engines, code cell (echo/free-form opt-in)

CSV: `docs/perf-csvs/all-26.7.9.csv` · chart:
`docs/perf-pngs/perf-vs-lmstudio-omlx-all-26.7.9.png`. vs `all-26.7.7.csv` (same
methodology): **no regression** — every mlx-serve decode cell within ±5%.
**oMLX — not LM Studio — is the competitor to watch.**

| Model | LM-GGUF | LM-MLX | oMLX | MTPLX | mlx-serve (best) | vs GGUF | **vs oMLX** |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gemma 4 E4B 4bit | 93 | 117 | 125 | — | **174** (drafter) | +87% | **+39%** |
| Gemma 4 31B 4bit | 22 | 26 | 25 | — | **32** (drafter) | +45% | +28% |
| Gemma 4 26B-A4B-MoE (QAT) | 96 | 117 | 125 | — | **127** (PLD) | +33% | **+1.6%** |
| Qwen 3.6 27B 4bit | 23 | 23 | 30 | — | **76** (MTP) | +234% | +153% |
| Qwen 3.6 35B-A3B-MoE | 89 | 88 | 149 | 52 | **215** (MTP) | +143% | **+44%** |
| Qwen 3.6 27B MTPLX-opt | — | 30 | 30 | 75 | **78** (MTP) | +162% | **+4% vs MTPLX** |

| Side finding | Result |
|---|---|
| `verifyQmm` on the gemma-E4B drafter verify shape (M=5) | **net loss**: 173.4 ON vs 177.7 OFF (`MLX_SERVE_VERIFY_QMM=0`), same engagement ⇒ kernel cost. Small; not a blocker |
| hy_v3 manual reading (no bench family) | Hy3-oQ2e (295B-A21B 2-bit) decode 25–28 tok/s · Hy3-REAP62 4-bit ~26 tok/s |
| **A GGUF-only chart flatters us** | vs LM-GGUF the 26B-A4B row reads +33%; vs oMLX it is **+1.6%**. Never quote a win without naming the engine it's over. |

### Perf gate 2026-07-16 (`feature/m5-neural-accel` — NAX verify-qmm lane, dormant on this M4)

CSV: `docs/perf-csvs/bench-all-20260716-233545.csv` vs `all-26.7.9.csv` (same
methodology): **no regression** — every decode cell within ±3.5%, signs mixed
(35B mtp +3.5%, 31B pld +11.3% = PLD spread); worst deltas are prefill cells
(−4.7/−4.5%) with equal-sized prefill gains elsewhere — prefill never reaches
the touched dispatch. MTP cells 76.4/78.3/222.7 vs 76.1/77.8/215.1 baseline.
NAX lane never engages here (probe false: `applegpu_g16s`); M5 numbers TBD.

## Long-context prefill (hd-256)

| Case | Result |
|---|---|
| Gemma 26B-A4B, 99K prompt (band kernel v2) | 317 (composed) → 652.6 (v1) → **715.4 tok/s** (2.4×), MLX peak 39.2 GB unchanged, identical output |
| Gemma E4B, 851-token bench cell | 2045.7 → **2135.7 tok/s** (kernel v2), decode flat |
| Qwen 27B, 8K prompt (2048-chunk cap) | 225.0 → **235.8 tok/s**, peak phys 28.9 → **19.8 GB** |
| Qwen 27B, 32K prompt (2048-chunk cap) | 205.4 → **209.3 tok/s** |

## Restart TTFT — SSD prefix-cache tier (~11.2K-token prompt, temp 0)

CSVs: `docs/perf-csvs/ttft-ssd-kv-cache-{baseline-20260706,final-20260707}.csv`.

| Model | Restart TTFT before → after |
|---|---|
| gemma4-e4b-4bit | 5.9 s → 0.66 s (9×) |
| gemma3-12b-4bit | 40.6 s → 2.7 s (15×) |
| gemma4-26B-A4B-moe-4bit | 10.5 s → 2.3 s (4.6×) |
| qwen36-27b (hybrid SSM) | 47.8 s → 0.55 s (87×) |

## Methodology rules (hard-won — violate these and the numbers lie)

- **Cold prompts only** for prefill: engines prompt-cache across requests; nested or
  repeated prompts inflate client-side pp. `bench.sh` salts every run's prompt at
  position 0 for exactly this reason, and its prompts (<2048 tokens) sit below LM
  Studio's cache-chunk granularity — the family matrix was never contaminated.
- **Same-boot A/Bs only** for kernel/dispatch decisions: warm-GPU rungs read 2–4% below
  cold single-cell boots (thermal), and an isolated-kernel µbench win can still be a
  live loss. Treat sub-2% single-run deltas as noise.
- **Diff only against a CSV from the SAME bench methodology.** Shared-boot landed in
  v26.7.7; `{gemma,qwen36}-26.7.6.csv` are per-spec-boot AND pre-`verifyQmm`, so a
  `--family all` run diffed against them mixes a methodology change with a code epoch.
  Use `all-<version>.csv`. (2026-07-16: this produced a bogus "−10.6% drafter
  regression" that evaporated against `all-26.7.7.csv` at −3.8%.)
- **"Reproducible ⇒ not variance" is FALSE for spec-decode cells.** N samples inside one
  condition (same boot style, same warm state, back-to-back) agree tightly and still
  miss a ~5% cross-run spread: `gemma4-e4b/drafter/code` read 173.4±0.5 four times, then
  182.1 on the same binary in a `--family all` run. Sample across runs/boot-orders
  before any regression claim.
- **LM Studio's own cells drift too.** In the 2026-07-16 refresh the 4 worst cells vs
  baseline were all `lmstudio-baseline` (−9…−23%) — their engine/version, not ours.
  Check the engine column before attributing a delta to mlx-serve.
- **Thermal soak >> thermal drift**: same-config warm readings fell 63.8 → 51 tok/s
  over 90 min of continuous GPU load, zero code change. Cross-session absolute
  comparisons need a cooled machine; only same-session ratios are trustworthy.
- **The first boot of a cold model FILE pollutes its whole bench cell** (disk
  page-in + first-touch riding under prefill AND decode): a gemma sweep whose
  weights hadn't been touched in weeks read −13…−55% on each model's FIRST
  cell while later same-model cells were flat; a warm manual re-run matched
  the reference exactly. Pre-touch weights (or discard cell 1) before
  comparing CSVs.
- **Always `zig build -Doptimize=ReleaseFast`** — a Debug binary is 2–4× slower and
  reads as a fake regression.
- One 27B in memory at a time (`pkill -f mlx-serve` before any Python mlx run;
  `lms unload --all` after LM Studio measurements).

## How to reproduce

- Family matrix: `./tests/bench.sh --family gemma|qwen36 [--lmstudio --omlx]`
- MTP ladder: manual harness (`~/mlx-bench-assets/harness.py` + nested prompts;
  regeneration recipe in the memory notes) — server flags
  `--no-pld --mtp-depth 3 --prefix-cache-entries 0 --ctx-size 140000`.
- Chart re-render: `python3 tests/plot_mtp_ladder.py docs/perf-csvs/mtp-ladder-26.7.6.csv out.png`
  and `python3 tests/plot_vs_lmstudio_omlx.py docs/perf-csvs/<family>-26.7.6.csv out.png --family <family>`.
