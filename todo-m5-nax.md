# M5 NAX verify-qmm lane: validation and adaptive-depth fix

Status: **validated on an M5 Max (`applegpu_g17s`) on 2026-07-17; the
adaptive-depth follow-up is implemented.** The original hardware-validation
blocker is closed. This note records the evidence and the guardrails behind
enabling depth 8 automatically.

## What the M5 Max run established

The important comparison was same-session, with the same ddalcu Qwen 27B MTP
checkpoint and workload:

| Configuration | Decode | Relevant round time |
|---|---:|---:|
| Fixed depth 8, NAX enabled | ~115.6 tok/s | M=8 ~68.5 ms |
| Fixed depth 6, NAX enabled | ~101.6 tok/s | — |
| Fixed depth 8, NAX disabled | ~72.3 tok/s | M=8 ~124–132 ms |

This proves that the G17 NAX lane is profitable for this M5 Max/model pair and
that a depth-8 controller ceiling can be useful. It is not a universal M5
performance guarantee: variant, power state, workload, model acceptance, and
thermal history can move the absolute numbers. Keep public comparisons
same-session and report realized `m_avg`, not just the configured maximum.

The completed post-fix controller A/B (same rebuilt MLX 0.32.0 binary and
integer-stream request) reached `m_avg=8.00` after warmup and produced 117.6
tok/s, versus 104.2 tok/s with an explicit cap of 6 (+12.9%). Steady trace
windows were about 69.2 ms at depth 8 and 62.2 ms at depth 6; the extra two
drafts more than paid for their round-time increase.

The initially confusing result was controller behavior, not a failed NAX
kernel. Auto was conservatively capped at 6. Giving the adaptive controller an
explicit `--mtp-depth 8` raised its ceiling, but its M4-fitted cost model made
the existing marginal horizon stop at shallow widths (the observed maximum was
depth 4), so that run behaved almost identically to auto 6. Saturating the fixed
controller at depths 6 and 8 exposed the hardware win above. The missing input
was the M5 width surface: its intermediate rows are cheaper than the M4 fit and
the marginal drops again when verify M=8 enters NAX.

## Implemented controller policy

Auto depth 8 and the M5 cost profile are selected together, per resident model,
only when all of the following are true:

- The target is the calibrated dense Qwen3.6-27B geometry (5120 hidden, 64
  layers, 248320-vocab head) with global 4-bit-affine quantization, and its
  actual lm_head is affine-4/gs-64 with NAX-compatible K/N geometry. Its token
  embedding and every resident trunk projection must also be homogeneous
  affine-4/gs-64; same-geometry Unsloth Dynamic/mixed checkpoints stay at cap
  6. Other model sizes/families stay at cap 6 until measured.
- Verify-qmm and its NAX lane are enabled, the runtime probe identifies G17 on
  a supported macOS release, and the dispatch predicate selects NAX for both
  M=8 and M=9. Those are the verify widths reached by draft depths 7 and 8.
- The bound MTP head matches the measured native dense sidecar surface
  (affine-8/gs-32 linears), and its draft-only lm_head was successfully built
  at affine-3/gs-64. Disabling or failing that requantization, or binding a
  differently quantized compatible sidecar, retains cap 6.
- The adaptive controller is enabled and `--mtp-depth` is left at auto.

The same eligibility decision chooses both the cost profile and cap, so the
planner cannot assume a lane the forward pass will not serve. The family-wide
`MLX_SERVE_VERIFY_QMM=0` switch, `MLX_SERVE_VERIFY_QMM_NAX=0`, forced GPU-family
fallback, incompatible lm_head storage/geometry, or an
`MLX_SERVE_VERIFY_QMM_NAX_MIN_M` value that excludes M=8 or M=9 all retain the
ordinary adaptive cap of 6. So do `MLX_SERVE_MTP_DRAFT_HEAD_BITS=0`, a failed
draft-head build, and sidecar storage/geometry outside the measured profile.
Fixed mode retains its default depth of 3. An explicit `--mtp-depth` always
wins and remains clamped to `MAX_DEPTH`.

## Cost-model change

The established M1–M4 profile is unchanged:

```text
draft .06 / low-width marginal .06 / high-width marginal .24 / sync .02
```

The qualifying M5/G17 Qwen3.6-27B target gets the measured NAX-aware profile:

```text
draft .06 / low-width marginal .06 / high-width marginal .15
NAX marginal .04 beginning at draft index 7 (verify M=8) / sync .02
```

The falling marginal at the NAX boundary is hardware-specific, so it belongs in
the selected cost profile rather than in a global planner rewrite. With the M5
fit, realistic post-warmup EMAs clear the existing confidence-gated horizon
through depth 8. The planner algorithm, ordinary profile, and M1–M4 outputs stay
unchanged. `MLX_SERVE_MTP_EV_COSTS` remains the explicit scalar tuning override
and deliberately selects the generic two-region form.

## Validation guardrails

- The self-gating parity suite exercises NAX M {8,9,12,16} across group sizes
  {32,64,128} wherever the hardware probe is live, against the fp32 dequantized
  truth.
- Pure dispatch/eligibility tests pin every kill switch, minimum-M boundary,
  actual-lm_head, full architecture, homogeneous-trunk, sidecar, and draft-head
  requirements, cap fallback, and the rule that explicit depths win.
- Planner tests pin that realistic warmup EMAs stop shallow on the ordinary
  surface but reach depth 8 on the selected M5 profile; cold EMAs stay shallow.
- Live regression checks must verify both output equivalence and engagement:
  inspect `drafted=`, `ext_rounds=`, the realized-depth histogram/`m_avg`, and
  `MLX_SERVE_MTP_TRACE=1` round timings.

Useful M5 commands remain:

```sh
zig build test -Doptimize=ReleaseFast
MLX_SERVE_VQMM_UBENCH=1 zig build test -Doptimize=ReleaseFast \
  -Dtest-filter="verifyQmm µbench"
./tests/test_mtp_equivalence.sh
npx llmprobe http://127.0.0.1:11234 --bench --quick
```

For any additional M5 variant, first confirm the probe, parity rows, NAX
dispatch, and same-session fixed 6/fixed 8 trace before generalizing this
profile. Rebuild the executable before each live A/B; a test build alone does
not refresh `zig-out/bin/mlx-serve`.
