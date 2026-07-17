# TODO: M5 NAX verify-qmm lane (needs M5 hardware to validate)

Need someone with a M5 chip to test this please.
Download ddalcu/qwen 27b mtp

Run this benchmark on current release dmg from GH, to get a baseline:
npx llmprobe http://127.0.0.1:11234 --bench --quick

Compile this code and run again the same, and compare prefill and decode.

Then do the section "M5-day runbook"


Status: **IMPLEMENTED 2026-07-16, live-unvalidated** — everything below §9.2
is in the tree and hermetically tested on the M4 Max; what remains is the
M5-hardware half of §9 (items 3–7). Original design notes kept intact below
as the M5-day reference.

## 0. What landed (2026-07-16, no M5 available) and the M5-day runbook

Landed in `src/transformer.zig` (+ `src/mlx.zig` FFI, `src/generate.zig` cap):

- **Probe**: `verifyQmmNaxAvailable()` (cached; `[nax-probe]` diagnostic via
  `zig build test -Dtest-filter=verifyQmmNaxAvailable` — prints
  `arch=applegpu_g16s macos=26.5 available=false` on this M4). Pure parts
  (`naxArchIsG17`, `macosVersionAtLeast`, `naxAvailableFrom`) unit-tested;
  `MLX_SERVE_FORCE_GPU_FAMILY_FALLBACK=1` rehearsal honored.
  `mlx_device_info_get_string` added to mlx.zig (signature checked against
  the installed mlx-c 0.6 header).
- **Kernel**: `VQMM_NAX_SOURCE`/`VQMM_NAX_HEADER` — MTPLX's m16 tile
  VERBATIM, T/GS/KCONST as template args, ONE cached kernel object
  (`mlxserve_vqmm_nax_m16`), construction strictly behind the probe.
  The EXACT emitted bytes (extracted from the Zig literals) compile
  warning-free with `xcrun metal -Wall -Wextra` against the macOS 26.4 SDK
  at gs {32,64,128} × {bfloat,half} × K {2048,5120,17408} — syntax/type/API
  risk is retired; only pipeline creation + numerics need the M5.
- **Dispatch**: pure `vqmmLaneFor(m, K, N, nax_on, nax_min_m)` (hermetic
  table test) — M 2..7 keep split-K/msg byte-identically, M 8..16 → NAX when
  eligible, `runVerifyQmmNax` pads to [16,K]/slices back via `naxPadTo16`/
  `naxSliceRows`, which ARE executed on this M4 by the "NAX host
  scaffolding" test (production pad/slice around stock qmm: pad rows exact
  zeros, real rows byte-preserved, slice bounds exact) — so the only code
  that has never run is the kernel dispatch itself. New env:
  `MLX_SERVE_VERIFY_QMM_NAX=0` (lane kill), `MLX_SERVE_VERIFY_QMM_NAX_MIN_M`
  (default 8; set 5 for the §7-step-2 A/B — also re-points the µbench's 4/6
  rows at NAX when set to 4).
- **Controller**: `MTP_ADAPTIVE_NAX_CAP = 8`; `resolveMtpDepthCap` keys on
  `verifyQmmNaxEnabled()` (ONE predicate for cap + dispatch, so every kill
  switch reverts both — pinned by the mtpDepthCapFor test).
- **Self-gating M5 tests already in the tree**: the `verifyQmm` parity test
  runs NAX rows M {8,9,12,16} × all three gs arms vs the fp32 dequant truth
  automatically wherever the probe is live; the
  `MLX_SERVE_VQMM_UBENCH=1` µbench now carries M {8,12,16} rows (print
  `fallback` on non-M5). Probe-forced fall-through (M 1/8/16/17 → stock)
  pinned on every machine via `vqmm_nax_probe_override` (only ever force
  FALSE — never true off-M5, see §10 never-build rule).

M5-day runbook (the part that still needs hardware — §9 items 3–7):

1. `zig build test -Doptimize=ReleaseFast` — the NAX parity rows light up
   automatically; `-Dtest-filter=verifyQmmNaxAvailable` must print
   `available=true`. Any parity failure here stops the day.
2. `MLX_SERVE_VQMM_UBENCH=1 zig build test -Doptimize=ReleaseFast
   -Dtest-filter="verifyQmm µbench"` — expect the M 8..16 rows to beat the
   stock ladder (their ledger: lm_head ~2.5× stock at verify widths).
3. Live trace sweep (§9.5): T(8) gate — if not well under ~110 ms, STOP and
   investigate before trusting the depth-8 cap (`--mtp-depth 7`/`6` are the
   same-boot A/Bs; explicit flag wins over the auto cap).
4. §7 step 2: re-run bench with `MLX_SERVE_VERIFY_QMM_NAX_MIN_M=5` — keep
   whichever wins the same-boot trace.
5. Refit `MTP_EV_DEFAULT_COSTS` on the M5 (§8.2; check realized m_avg).
6. `tests/test_mtp_equivalence.sh` (9/9), gemma-4bit
   `tests/test_pld_equivalence.sh`, creative temp-0.8 demote check.
7. MTPLX comparison only with their `qmm_m16_nax: "ok"` selfcheck confirmed
   (§9.6); rebuild the EXE before every live A/B; same-boot numbers only.
8. Re-dissect their `nax_verify.py`/CHANGELOG for drift first (§10) — this
   port matches MTPLX 2.0.2 (checkout `510ac8c`).
