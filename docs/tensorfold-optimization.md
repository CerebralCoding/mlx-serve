# TensorFold optimization experiments

Branch `feat/tensorfold-optimization`, based on main `2496d20`. This work starts
from main and does not incorporate the earlier feature/tensorfold changes.
The state/tree results below are the initial baseline. The later native NAX
experiment is documented at the end.

## State storage

The opt-in DFlash tree verifier keeps recurrent branch states inside the GPU kernel
and retains only the initial state plus row-local prework. Commit replays the
accepted path into a compact state allocation. The existing serial and MTP
recurrence paths are unchanged. Replay rounds to the original state dtype after
every step; ordinary Qwen packs therefore retain main's bf16 semantics.

For the 27B's 48 recurrent layers, the old per-node state payload is 72 MiB per
verification row. The new retained prework is 386 bf16 elements per head per row,
or about 1.70 MiB per row across those layers. Initial/final state, convolution
inputs, activations, allocator caching, and possible kernel-private spills are
additional costs. These shape counts are not a claim about whole-server peak RSS.

The GPU regression test checks every node's output and every accepted path's
state against independent serial steps, covering chains and branches at widths
1, 5, 8, and 16, including the actual 16-key-head/48-value-head geometry. It also
checks active MLX allocation against the compact-prework bound.

## Proposal and verification sizes

Supported DFlash2 tree targets can use a 16-position proposal even when the
checkpoint's default block is 8, through an explicit `--draft-block-size 16`.
The default proposal remains the checkpoint's block. Probability allocation is
opt-in while the wide verification path is being optimized. Proposal depth and the draft-node budget are
separate. The maximum verifier size is 16 rows: one pending token plus 15 draft
nodes. Non-tree drafting retains its checkpoint block limit.

Tree nodes retain their path scores. Probability ordering caps each node's
estimated chance by its ancestors and preserves parent-before-child order, so
every selected prefix is a valid tree. Qwen 27B uses TensorFold's checked-in
greedy/sample calibration for the corresponding sampling settings; other
settings use exponentiated path scores. Calibration changes proposal allocation,
not target sampling or acceptance. The imported fit is a prior from TensorFold's
runtime, not a new calibration measured on this implementation.

The allocator maximizes expected emitted tokens divided by measured complete
round cost among measured prefix sizes. It bootstraps wide and seven-node
prefixes, then occasionally explores smaller prefixes. Costs are model-local,
separated by greedy/sample regime and context bucket, and never persisted.
The first observation at a new width is excluded from its steady-state price.

## Controls and evidence

- `--draft-block-size 8|16`: proposal positions on supported tree targets.
- `MLX_SERVE_DFLASH_TREE_NODES=0..15`: independent draft-node budget; defaults
  to proposal positions minus one. Zero verifies only the pending root token
  with the same target kernels: a correctness control, not the ordinary serial
  engine configuration.
- `MLX_SERVE_DFLASH_TREE_ALLOCATE=1`: enable probability/cost allocation;
  default 0 preserves the original preorder tree arm.
- `MLX_SERVE_DFLASH_TREE_REPLAY=1`: enable compact replay and support up to sixteen
  verification rows. Default 0 retains original capture, limited to eight rows.

To reproduce the original block-8 geometry, combine block 8, node budget 7,
allocation off, and replay off. To isolate state storage, enable only replay.
To exercise the full experiment, enable both `MLX_SERVE_DFLASH_TREE_REPLAY=1`
and `MLX_SERVE_DFLASH_TREE_ALLOCATE=1`, with `--draft-block-size 16`.
`[dflash-tree]` lines identify proposal size, selected/budgeted nodes, probability
allocation, calibration choice, and state mode. `[spec-stats]` counts actual
verified draft nodes when allocation varies the width.

Correctness checks use the repository's Zig toolchain:

```
.zig-toolchain/zig build test -Doptimize=ReleaseFast '-Dtest-filter=tree'
.zig-toolchain/zig build test -Doptimize=ReleaseFast '-Dtest-filter=dflash:'
.zig-toolchain/zig build test -Doptimize=ReleaseFast '-Dtest-filter=bestFirstTree'
./tests/test_mlx_staged_nax.sh
```

Benchmark artifacts use the requested TensorFold chat prompts and harness,
64 output tokens, thinking off, seeds 1234–1238, five measured repetitions and
one warmup per cell. Raw replies, timings, commands and logs remain under
`~/claude-tmp/tfo/`. One server runs at a time.

## Initial full-model validation

All 24 replies (warmups included) were byte-identical across capture/replay,
block-8/block-16, fixed/allocation modes, and the reversed-order capture/replay
repeats. Every response had 64 tokens and met the 15-second first-reply bound.
The comparison pack was `ddalcu/Qwen3.8-27B-MLX-Serve-4bit`, with the shared
`z-lab/Qwen3.8-27B-DFlash2` drafter, KV quantization off.

| Mode | Code T=1 | Chat T=1 | Code T=0 | Chat T=0 |
|---|---:|---:|---:|---:|
| Original capture, block 8 | 134.76 | 73.75 | 166.78 | 74.07 |
| Replay, block 8 | 132.49 | 73.55 | 164.59 | 73.17 |
| Replay, block 16, allocation | 121.19 | 63.94 | 146.96 | 72.28 |
| Replay, block 16, fixed 15 nodes | 108.03 | 47.29 | 151.90 | 54.51 |
| Replay, block 8, repeat | 131.35 | 72.54 | 164.66 | 73.20 |
| Original capture, block 8, repeat | 132.94 | 73.63 | 167.38 | 74.27 |

Median decode tokens/s. These are controlled same-binary comparisons against the
original main capture path, not a new TensorFold engine comparison. The initial
allocator used cold wide/half-width exploration and then smaller-prefix probes;
the current implementation additionally revisits already measured widths to
avoid sticking to startup-contaminated estimates. These results predate dispatch
configuration caching in the replay kernels.

Wider fixed trees increase accepted tokens but are slower overall with the
current target kernels. Therefore neither wider proposals nor probability
allocation is enabled by default. The initial replay-only cost was about 1–2%,
so memory reduction alone does not establish a trade-off-free speed win.

Example engagement lines from the saved runs:

```
[dflash-tree] proposal=8 nodes=7/7 allocation=fixed calibration=qwen27-dflash2 state=capture
[dflash-tree] proposal=8 nodes=7/7 allocation=fixed calibration=qwen27-dflash2 state=replay
[dflash-tree] proposal=16 nodes=15/15 allocation=probability/cost calibration=qwen27-dflash2 state=replay
[dflash-tree] proposal=16 nodes=7/15 allocation=probability/cost calibration=qwen27-dflash2 state=replay
```

After caching dispatch configuration and revisiting measured widths:

| Mode | Code T=1 | Chat T=1 | Code T=0 | Chat T=0 |
|---|---:|---:|---:|---:|
| Original capture, block 8 | 135.11 | 73.51 | 166.83 | 74.17 |
| Replay, block 8 | 131.61 | 72.74 | 165.83 | 73.35 |
| Replay, block 16, allocation | 135.61 | 63.98 | 168.54 | 72.21 |

All 24 replies again match the original capture arm byte for byte. The saved
names are `opt-capture-b8-cached-control`, `opt-replay-b8-cached`, and
`opt-replay-b16-allocation-cached`. The runs used the same weights and settings
as above; first replies were all under 0.25 seconds in this set.

Replay remains approximately 0.6–2.6% slower in this final pass. Larger allocated
trees are close on code but about 13% slower on sampled chat. Accordingly **all
new execution policies remain opt-in**, including replay. A future default change
needs a demonstrated speed result as well as the established state correctness
and compact-allocation result. No TensorFold speed victory is claimed here.

The next bottleneck to investigate is the target's verification cost by width:
the larger tree already improves acceptance, but current kernels do not have
TensorFold's nearly flat cost through sixteen rows. Changes to weights/layouts
or state precision are outside this implementation.

## Native NAX verification

The next pass addresses the expensive target verification rather than increasing
the tree budget further. `src/dflash_nax.zig` adapts TensorFold's original Python
`lane_qmm.py` native uint4 tensor-operation kernel. It reads the existing affine
weights, scales and biases directly, with no second weight layout or resident
weight copy. Only bf16 activations, affine 4-bit/group-32 or group-64 weights,
aligned matrices, and one through sixteen rows qualify.

Target dispatch is restricted to the DFlash-bound dense Qwen 27B architecture
on a NAX-capable GPU. Other formats retain their existing kernels. The existing
joined gate/up weights are used consistently across one-row and tree forwards;
their split-K count is derived from each member's original width, following
Python `lane_fuse.py`. Changing the split count with the combined width would
change floating-point arithmetic between serial and verification rows.

The 16-column and 32-column variants preserve that same split and are tested
bit-for-bit against each other. Each verification row is also checked against
an independent one-row call, and joined gate/up halves against independent
projections. Accuracy is checked against an fp32 dequantized reference using
the existing no-worse-than-stock test. These checks are not a broad model-quality
evaluation, nor a claim that different kernels reproduce main's floating-point
outputs bit-for-bit.

The verification submission interval can be shortened, including submitting
after the first layer, following Python `row_forward.py`. This changes when
work starts, not its arithmetic. The original default interval remains four.

Opt-in settings used for the final candidate:

```text
MLX_SERVE_DFLASH_NAX=1
MLX_SERVE_DFLASH_NAX_DRAFTER=1
MLX_SERVE_DFLASH_NAX_TILE=16
MLX_SERVE_DFLASH_TREE_REPLAY=1
MLX_SERVE_DFLASH_TREE_ALLOCATE=0
MLX_SERVE_DFLASH_PIPELINE=1
MLX_SERVE_ROUND_COST_PERSIST=0
```

Server arguments include `--draft-block-size 16 --kv-quant off --no-mtp
--no-pld --drafter <shared z-lab directory>`. NAX defaults off; the tile defaults
to 32 when enabled without a tile override. Probability/cost allocation remains
off in the speed candidate: its serving-time exploration still costs more than
it saves on these short requests.

Rejected experiments were removed from the runtime:

- A cooperative 64-column tile and standalone narrow gate projections were
  slower.
- Combined Z/B/A projection saved launches but did not improve the four-cell
  result.
- Drafter submissions after layers zero and two did not improve throughput.
- A 5120-wide fused residual/norm kernel passed exact kernel and full-reply
  parity, but its gain was inconclusive.
- Reusing private state slots after their last child reduced one sixteen-row
  tree to two live slots and passed branch/state parity. It did not improve
  steady-state throughput and introduced JIT stalls for new slot counts.

The retained implementation does not import the halted Zig refactor or the
discarded feature branch, change state precision, or introduce another weight
representation. The reference checkout is original Python TensorFold
`beddbb7bc818b432163c30500aea256e2b46ff8a`; its tracked files are unchanged.

## Final same-day engine comparison (2026-09-28)

**The overall goal is not met.** The retained block-16 candidate improves on the
main-path control, but does not beat Python TensorFold across both checkpoints
and all four cells. In particular, the primary Vontra checkpoint is slower than
TensorFold in all four cells with this configuration. Do not promote these
settings as an unconditional default or quote the ddalcu wins as a general win.

Hardware was Apple M5 Max on AC power. Native MLX was 0.32.3, staged commit
`d73eb752ef2e`, with mlx-c `56b2d39fc831`; Python TensorFold used MLX 0.31.2
and MLX-lm 0.31.3. Main base:
`2496d200148c2526d0fc6c618814f7271a89543b`.
The tested release executable SHA-256 was
`5a98c543f9c2e7b4c41a37aadaf9850133e273c2cb05eb6cd460261c12de27cf`.

Each engine read the same canonical checkpoint directory for its comparison
and the shared `~/.mlx-serve/models/z-lab/Qwen3.8-27B-DFlash2` drafter.
The table cells are five-run medians from TensorFold's own
`tools/bench_openai.py`, adapted to send both prompts through chat and save
complete replies. Each cell has an additional warmup. Output length is 64,
thinking is off, and seeds are 1234–1238. T=1 uses top-k 20/top-p 0.95.
Context is 4096, KV quantization and prefix caches are off, and persistent
round-cost learning is off. One server ran at a time.

The main control uses the same executable with the new paths disabled and
block 8. It measures the main execution path; it is not a separately rebuilt
main executable. Ordinary serial runs disable MTP, PLD and the drafter.
They differ from the pending-root-only correctness control, which deliberately
retains the speculative target's numerical kernels.

### ddalcu/Qwen3.8-27B-MLX-Serve-4bit

Median decode tokens/s; rows are in execution order.

| Arm / artifact prefix | Code T=1 | Chat T=1 | Code T=0 | Chat T=0 |
|---|---:|---:|---:|---:|
| TF, `tf-final-before` | 199.96 | 70.26 | 204.25 | 93.42 |
| Optimized, `opt-final-before` | 211.83 | 87.51 | 250.11 | 89.19 |
| Main control, `main-final-control` | 132.98 | 73.03 | 167.04 | 74.03 |
| Optimized, `opt-final-after` | 202.25 | 87.98 | 245.97 | 89.17 |
| TF, `tf-final-after` | 201.07 | 71.14 | 204.95 | 93.71 |
| Our ordinary serial, `opt-serial-final` | 35.66 | 35.57 | 35.83 | 35.88 |
| TF ordinary serial, `tf-serial-final` | 27.63 | 27.60 | 27.84 | 27.81 |

Both engine orders show three winning cells and a greedy-chat loss of about
5%. The sampled-code margin varies from about 6% to less than 1%, so it is
not a robust large win. Against the main-path control, the retained candidate
gains roughly 52–59% in sampled code, 20% in sampled chat, 47–50% in greedy
code and 20% in greedy chat.

### Vontra/Qwen3.8-27B-MLX-4bit

This primary-checkpoint pass used one server launch per arm, in table order.
It does not have the reversed-order confirmation of the ddalcu comparison.

| Arm / artifact prefix | Code T=1 | Chat T=1 | Code T=0 | Chat T=0 |
|---|---:|---:|---:|---:|
| TF, `tf-vontra-final` | 201.06 | 73.07 | 238.38 | 93.78 |
| Optimized, `opt-vontra-final` | 181.40 | 72.09 | 205.74 | 90.50 |
| Main control, `main-vontra-final` | 132.76 | 72.93 | 148.15 | 73.70 |
| Our ordinary serial, `opt-vontra-serial-final` | 35.62 | 35.56 | 35.84 | 35.83 |
| TF ordinary serial, `tf-vontra-serial-final` | 27.67 | 27.64 | 27.89 | 27.83 |

The optimized path is about 37% faster than main on code and 23% faster on
greedy chat, but sampled chat is about 1% slower. Against TF it is approximately
10%, 1%, 14% and 4% slower, respectively. This fails the requested performance
bar and the no-other-trade-offs bar for default adoption.

Further isolated probes, preserving the candidate target kernels:

| Probe / artifact prefix | Code T=1 | Chat T=1 | Code T=0 | Chat T=0 |
|---|---:|---:|---:|---:|
| Stock drafter kernels, `opt-vontra-stock-drafter` | 186.42 | 64.95 | 228.82 | 82.41 |
| Proposal 8, node budget 15, `opt-vontra-proposal8` | 155.15 | 74.94 | 156.19 | 96.82 |
| Probability/cost allocation, `opt-vontra-allocation` | 181.55 | 71.22 | 213.05 | 86.99 |

The shorter proposal wins both chat cells in this exploratory pass but loses
too much code throughput. Stock drafter kernels improve greedy-code acceptance
but slow chat. The allocator does not resolve the trade-off. None is selected
per benchmark prompt, and none replaces the fixed candidate in the main tables.

### Remaining bottlenecks and interpretation

On Vontra greedy code, the candidate requires seven rounds versus TF's six;
on greedy chat, both require fifteen. Acceptance therefore matters for code,
while the chat gap remains even at equal round counts. The chat timings imply
roughly 46–47 ms per round here versus TF's logged 44–45 ms, though the external
harness and server round timers do not have identical boundaries.

Ordinary serial is faster here than in TF, but this uses different kernels from
our DFlash target. It does not prove speculative verification is faster. Larger
trees need both inexpensive verification and good drafter acceptance; changing
one alone has repeatedly moved the loss into another cell.

First-token latency also remains worse than TF: approximately 0.18–0.20 seconds
here versus 0.07 seconds for TF, with main control around 0.19 seconds. The
candidate is not a latency victory. Every measured request in the final passes
returned its first token within 15 seconds and produced exactly 64 tokens.

### Correctness, memory and engagement

Both final ddalcu optimized runs match all 24 complete replies from
`opt-nax-root-only-split`. On Vontra, `opt-vontra-final` and all three probes
match all 24 replies from `opt-vontra-root-control`. These comparisons include
the warmups and both sampling regimes. They validate tree execution against
independent one-row forwards using the same target kernels.

This is not bitwise equivalence to different engine kernels: only 15/24 Vontra
candidate replies match TF, and 16/24 match the main-path control. Ordinary
serial also uses a different sampling/kernel path. A broad model-quality
evaluation has not been performed; the kernel accuracy and row-invariance tests
are the evidence available here.

MLX allocator measurements, in bytes:

| Checkpoint | Path | Final active | Peak since load | Final cache |
|---|---|---:|---:|---:|
| ddalcu | Main control | 18,227,507,000 | 21,821,357,596 | 638,245,544 |
| ddalcu | Optimized | 18,227,507,004 | 21,821,357,596 | 147,990,128 |
| Vontra | Main control | 16,399,871,800 | 19,993,722,396 | 638,245,544 |
| Vontra | Optimized | 16,399,871,804 | 19,993,722,396 | 148,055,792 |

The optimized path retains about 490 MB less allocator cache; active memory and
the loading-dominated peak are effectively unchanged. These are MLX allocations,
not whole-process RSS or a decode-only peak measurement. No second weight
representation is retained.

Representative ddalcu engagement lines:

```text
[dflash-nax] uint4 engaged: rows=16 K=5120 N=1280 gs=64 tile=16 layout=original
[dflash-tree] proposal=16 nodes=15/15 allocation=fixed calibration=qwen27-dflash2 state=replay
[dflash-tree] pipeline engaged: layers=1 first-alone=true
[spec-stats] mode=dflash attempts=6 accepts=58 avg_per_round=9.67 gate_min=2.00 per_draft_pct=64.4% block_size=16 partial_rounds=6 runtime_disabled=false
```

Vontra also logs the same NAX/tree/pipeline engagement; its greedy-code records
show `attempts=7 accepts=57`, and greedy-chat records
`attempts=15 accepts=49`, always `runtime_disabled=false`.
Full lines are saved in each arm's `.engagement.txt` and `.log`.

Validation passed: ReleaseFast build (7/7 steps), tree tests (10/10),
DFlash tests (42/42), native NAX tests (3/3), and staged NAX checks (9/9,
5,772 NAX symbol hits). The staged binary links the local MLX build.
Kernel tests cover row invariance, joined/separate projection equality,
16/32-column tile equality, fp32-reference accuracy, branch-state replay and
compact allocation. These runs required actual Metal access.

Raw timings, full replies, engagement logs, commands, environment, source and
binary fingerprints remain in `~/claude-tmp/tfo/`, under the artifact prefixes
above. The local sequential harness is `zig-out/tfo/matrix.py`; final configs
are `state-tree-fixed.json`, `vontra-final.json`, and `vontra-probes.json`
in that directory. Generated benchmark artifacts are ignored. All benchmark
servers were stopped after validation.

## Deeper validation and refreshed TensorFold reference (2026-09-28)

The subsequent reference is the read-only mirror at `~/OpenSource/TensorFold`,
commit `34bae79ac97da6c3ab3fe10159cf49633ce8112a` (0.3.6.1). Its current
`pyproject.toml` requires MLX >=0.32.2. The isolated environment in
`zig-out/tfo/tensorfold-venv` uses MLX/MLX-Metal 0.32.2 and MLX-lm 0.31.3;
the mirror is imported through `PYTHONPATH` with bytecode writes disabled.
The server working directory is this repository. The mirror remains unchanged.
The preceding 0.31.2 measurements remain historical results, not current TF claims.

The live Qwen serving head (`families/qwen3_5/dflash_head.py`, `DraftSlot.get`)
explicitly disables the proposer's n-gram prior and supplies `copy=None`.
Those features in the generic proposer do not explain the serving gap and have
not been added here. Qwen dense kernels and the DFlash proposer are unchanged
between the two reference commits; the supported MLX version has changed.

`tests/dflash_validation.py` exercises ten prompts at T=0 and T=1, repeated
twice with seed 731, normal EOS handling, top-k 20/top-p 0.95 and thinking off.
The corpus covers arithmetic, sorting, typed extraction, a multi-turn update,
longer code/explanation responses (up to 384 tokens), and retrieval from 96
records. It records complete responses and can enforce equality against a
one-row reference. Sixteen unique cells have exact JSON answers; the four
free-form cells are checked for determinism and reference parity, not graded
for semantic quality. This is a small correctness corpus, not a broad quality
benchmark.

Before the shared-input experiment, the candidate, its pending-root-only
control, the main-path control, and latest TensorFold each passed all 16 scored
cells and had zero repeat mismatches across 40 requests. All 40 candidate
responses exactly matched the pending-root-only control. Artifacts:
`deep-validation-{opt,root,main}` and `deep-latest-tf-validation`.

A diagnostic 256-token candidate run (`deep-vontra-trace256`) found median
target verification/sampling phases of approximately 45–46 ms, compared with
3.8–4.8 ms for the assistant, 1.4–1.6 ms for tree/head work, and 0.3–1.0 ms
for commit construction. The trace adds an evaluation barrier and is not used
as a speed result. It identifies verification as the dominant round cost.

Fresh latest-TF medians, with five seeds 1234–1238 and one warmup per cell,
same Vontra checkpoint and shared drafter, both prompts through chat:

| Artifact | Output tokens | Code T=1 | Chat T=1 | Code T=0 | Chat T=0 |
|---|---:|---:|---:|---:|---:|
| `deep-latest-tf-vontra64` | 64 | 201.06 | 77.24 | 202.29 | 87.35 |
| `deep-latest-tf-vontra256` | 256 | 174.73 | 80.30 | 206.93 | 85.85 |

These are decode tokens/s. A sampled-code 256-token run had a 64.1 ms round
average versus about 45–46 ms normally; it remains in the recorded sample.
Different output lengths and seed counts must not be mixed when comparing
the earlier three-seed exploratory runs with this baseline.

### Shared activation sums experiment

Python `lane_qmm.lane_matmul` shares group sums between projections of the same
activation. The local `MLX_SERVE_DFLASH_NAX_SHARED_INPUT=1` experiment does this
for Q/K/V and GDN QKV/Z on the eligible Qwen target. `dflash_nax.Input` owns its
activation view and separate group-32/group-64 sums for one projection scope;
it does not cache C handle addresses or retain another weight representation.
The flag defaults off. The kernel arithmetic and split-K choice are unchanged.

Five-seed 256-token off/on/on/off runs used release binary SHA-256
`183ce44dbe96be53d44d7d7e3161a5b128c0a847ff95480e7aa2a638edec964b`:

| Artifact | Code T=1 | Chat T=1 | Code T=0 | Chat T=0 |
|---|---:|---:|---:|---:|
| `deep-shared-off256` | 172.16 | 77.36 | 194.71 | 80.67 |
| `deep-shared-on256` | 179.13 | 77.34 | 201.64 | 80.56 |
| `deep-shared-on256-repeat` | 178.54 | 74.68 | 194.03 | 77.64 |
| `deep-shared-off256-repeat` | 173.28 | 76.74 | 194.48 | 77.72 |

Sampled code improves 3.0–4.1% in the two comparisons; the other cells do not
establish a consistent gain. Sampled chat falls 2.7% in the reverse comparison.
This is **not** evidence of a trade-off-free improvement or an overall TF win.
All four arms produced the same 24 complete responses, and every request logged
engaged DFlash with `runtime_disabled=false`. Enabled runs additionally logged:

```text
[dflash-nax] shared input sums engaged: rows=1 K=5120 gs=64
```

Final active MLX allocation was 16,399,871,804 bytes and loading-dominated peak
was 19,993,722,396 bytes for every arm. Cached allocation was 168,264,448 bytes
off versus 168,248,064 on (16 KiB less). These are allocator metrics, not RSS.
The enabled path also matched all 40 expanded-corpus replies from the previous
candidate (`deep-shared-validation`), with no repeat mismatches and 16/16 scored
cells passing.

Validation: ReleaseFast build passed (7/7 steps); native NAX tests passed (3/3);
the complete Zig suite passed 2,807 tests, skipped 210, and failed none.
The new kernel checks cover shared-input lifetime before lazy evaluation,
alternating quantization group sizes, tile equality, row invariance, and
unsupported quantization rejection. Swift was outside this task's scope.

At 64 tokens the new sum-sharing flag is effectively neutral:

| Artifact | Code T=1 | Chat T=1 | Code T=0 | Chat T=0 |
|---|---:|---:|---:|---:|
| `deep-shared-off64` | 176.01 | 71.63 | 203.73 | 88.35 |
| `deep-shared-on64` | 175.43 | 71.36 | 203.66 | 88.43 |

All 24 replies match; maximum first-token latency was below 0.21 seconds.
The loading-dominated peak and final active allocation are identical to the
256-token experiment. Cache is again 16 KiB smaller with sharing.

### Fresh main and TensorFold controls

Five seeds and a warmup per cell; the same primary Vontra checkpoint. Main
means the same executable with experiment flags disabled and proposal block 8,
not a separate main-branch build. Optimized here retains block 16, compact
replay, native NAX and pipeline 1; shared sums are separately identified above.

| Arm / artifact | Output tokens | Code T=1 | Chat T=1 | Code T=0 | Chat T=0 |
|---|---:|---:|---:|---:|---:|
| Main, `deep-main64-five` | 64 | 132.02 | 72.93 | 147.54 | 73.66 |
| Optimized, sharing off, `deep-shared-off64` | 64 | 176.01 | 71.63 | 203.73 | 88.35 |
| Latest TF, `deep-latest-tf-vontra64` | 64 | 201.06 | 77.24 | 202.29 | 87.35 |
| Main, `deep-main256-five` | 256 | 138.27 | 72.48 | 135.59 | 70.57 |
| Optimized, sharing off, `deep-shared-off256-repeat` | 256 | 173.28 | 76.74 | 194.48 | 77.72 |
| Latest TF, `deep-latest-tf-vontra256-repeat` | 256 | 191.71 | 79.69 | 206.18 | 85.60 |

The TF repeat produced the same 24 complete responses as its initial 256-token
run. Its sampled-code median recovered from 174.73 to 191.71; the slow initial
observation remains in the report. This confirms why the first comparison
alone must not be called a sampled-code win over TF.

The overall goal remains unmet: the branch is much faster than main on code
but loses sampled chat at 64 tokens, and latest TF wins every 256-token cell
in the final control comparison. At 64 tokens, the branch's small greedy wins
over TF do not offset its sampled losses. First-token latency remains around
0.19 seconds here versus roughly 0.07 seconds for warmed TF requests.

Ordinary serial decode, 64 tokens, five seeds (MTP, PLD/copy and neural drafts
disabled; TF logs `drafts: off` and `accepted=0/0`):

| Artifact | Code T=1 | Chat T=1 | Code T=0 | Chat T=0 |
|---|---:|---:|---:|---:|
| `deep-main-serial64` | 35.53 | 35.54 | 35.77 | 35.76 |
| `deep-latest-tf-serial64` | 27.54 | 27.51 | 27.81 | 27.74 |

Our ordinary serial is about 29% faster. This does not transfer automatically
to the speculative target kernels. At 256 tokens, greedy code takes 26 rounds
here versus 27 in TF, while greedy chat takes 64 versus 65. Despite requiring
fewer rounds, we finish slower. Alongside the phase profile, this points to
target round cost as a priority over adding more proposals. The engines do not
produce identical transcripts: only 3/24 full replies match in this latest
256-token comparison, all in sampled code. Round-count comparisons therefore
describe the observed workloads, not a forced identical-token microbenchmark.

An additional 256-token build control ran the exact pre-experiment cached
executable (`5a98c543...`, HEAD `71cd8d5`) followed by the new executable with
sharing disabled. Medians were 179.34/74.59/194.77/77.61 versus
172.34/77.33/200.91/77.59 in the table's cell order. All 24 full replies matched;
active and peak allocation were identical. The old executable also reproduced
the later-session lower greedy throughput, so that shift is not unique to the
refactor. The mixed direction of the per-cell changes does not establish a
performance improvement from the refactor itself.

All servers were stopped. TensorFold's mirror still has the same HEAD and a
clean status. Runtime changes remain local and opt-in; no commits, pushes or
PR updates were made during this pass. Reproduction configs are
`zig-out/tfo/deep-{latest-tf,validation,shared,controls,binary-control}.json`;
the full logs, commands, dependency versions, replies and measurements use
the corresponding artifact names under `~/claude-tmp/tfo/`.

## Producer fusion investigation (2026-09-28)

Tracing the current Python M5 serving path matters: Qwen's tensor-unit backend
uses `lane_multi.multi_tree_forward`, not the non-tensor-unit `row_forward`
backend. Its `lane_glue.norm_xs`, `lane_fuse.mlp_act`, and `lane_fuse.gdn_post`
produce activation group sums alongside their normal outputs. Sharing an
independently computed sum, as above, leaves those producer dispatch savings
unrealized. Reference remains the read-only mirror at `34bae79ac97da6c3ab3fe10159cf49633ce8112a`.

Two local, default-off probes tested this design while preserving native
arithmetic. Both were subsequently removed after their mixed results; their
tested source is saved in `zig-out/tfo/producer-and-metadata-prototypes.patch`:

- `MLX_SERVE_DFLASH_NAX_PRODUCER=1`: joined gate/up input directly produces
  SwiGLU and 64-element affine sums. Uses our existing sigmoid lookup and both
  bf16 multiplication rounding points; no strided gate/up half copies.
- `MLX_SERVE_DFLASH_NAX_NORM_PRODUCER=1`: the target MLP's residual seam
  produces residual, RMS-normalized activation, and affine sums together.
  Restricted to bf16 `[1, M, 5120]`, `M <= 16`, single-token or tree calls.
  Uses MLX's 1024-thread looped RMS reduction and retains both normalization
  rounding points. The following joined gate/up projection consumes the sums
  through an explicit scope-owned `Input`.

Both require the experimental target NAX path. No second weight layout or
global array-pointer cache is introduced. Tests compare activations, every
group sum including padding, and projected values against the unfused path
at 1, 2, 5, 8, 12 and 16 rows.

The first activation prototype launched threadgroups even for padded rows.
Its off/on/on/off 256-token, five-seed measurements were:

| Arm | Code T=1 | Chat T=1 | Code T=0 | Chat T=0 |
|---|---:|---:|---:|---:|
| off | 179.32 | 77.41 | 202.07 | 80.64 |
| on | 176.36 | 78.08 | 196.09 | 81.36 |
| on repeat | 180.63 | 75.42 | 197.03 | 81.21 |
| off repeat | 177.72 | 77.52 | 201.97 | 80.53 |

Units are median decode tok/s with the same TF chat harness. This does **not**
justify enabling it: greedy code regressed by about 3% in both orders, despite
a small greedy-chat gain. All 24 full replies per arm matched, all requests
engaged DFlash with zero runtime disables, and peak MLX allocation remained
19,993,722,396 bytes. Binary SHA256 was
`e1fdd6b9d031e3b31f496a9a5e040079443b0b5b8da2809436918515532f6d73`;
artifacts are `producer-{off,on}256[-repeat]`. The revised activation kernel
launches only actual rows and fills padded sum slots from row zero.

The revised activation and new normalization producer were then isolated in
one binary (`7ce7c0dcaabc6f15e7ef4467b50b7942f6932041f2465b491278bb66cb3c2c4a`):

| Arm | Code T=1 | Chat T=1 | Code T=0 | Chat T=0 |
|---|---:|---:|---:|---:|
| neither | 179.26 | 77.30 | 194.10 | 80.53 |
| activation | 181.15 | 75.44 | 197.04 | 79.69 |
| normalization | 180.18 | 76.39 | 195.65 | 78.19 |
| both | 174.05 | 76.42 | 196.29 | 81.30 |

These are exploratory, single-order five-seed runs, not evidence of an
across-the-board gain. Every arm matched all 24 control replies; DFlash stayed
engaged in every request. Peak allocation remained 19,993,722,396 bytes and
active allocation was unchanged except 12 fewer bytes in activation arms.
Neither experiment was promoted; both were removed from runtime code after
this comparison. Artifacts: `producer-v2-{00,10,01,11}-256`.
Release build and the full suite passed: 2,809 tests, 210 skipped, zero failed.

### TensorFold's actual M5 layout path

Current `families/qwen3_5/__init__.py` calls `lane_qmm.install(..., tile=True,
wide=True)` by default. That packs scales/biases into group-major pairs and
replaces eligible weights with a tiled layout; eligible wide projections then
use a cooperative 64-column kernel. This differs materially from our original
layout kernel. The previously rejected native cooperative experiment therefore
did not reproduce the whole current TensorFold layout/kernel combination.

A read-only TensorFold diagnostic toggled its existing `TF_LANE_TILE` switch,
using 256 tokens and five seeds, normal/disabled/normal order:

| TensorFold arm | Code T=1 | Chat T=1 | Code T=0 | Chat T=0 |
|---|---:|---:|---:|---:|
| normal tiled | 197.81 | 80.25 | 207.15 | 85.66 |
| tiling disabled | 178.40 | 72.10 | 185.11 | 77.13 |
| normal tiled repeat | 193.06 | 80.53 | 207.68 | 86.23 |

All 24 full replies matched exactly across all three arms. Greedy throughput
fell about 10–11% without tiling and recovered on the repeat. TF's startup
forward diagnostic likewise measured sixteen rows at 37.8/42.6/37.7 ms.
This is evidence about the **combined tiling and tile-width choice**: disabling
tiling also selects its 32-column original-layout kernel instead of the wide
tiled kernel. Scale/bias packing and producer fusions remain enabled in both
arms. It is not an isolated estimate of weight reordering alone and must not
be presented as an mlx-serve win against normal TensorFold.

Artifacts: `tf-layout-{tiled,original}256` and `tf-layout-tiled256-repeat`.
The harness now clears and records `TF_*` environment switches explicitly.
TF still reports 18.3 GiB resident weights with tiling versus 18.0 GiB without;
these engine-reported weight figures are not comparable to our MLX active/peak
allocator figures. The mirror remains unchanged.

### Affine metadata scheduling without another weight layout

`MLX_SERVE_DFLASH_NAX_METADATA=1` moves scale/bias and input-sum reads before
the tensor operation, following current Python `lane_qmm._MAIN` ordering.
Mode `2` additionally reads the original scale/bias rows cooperatively into
group-major pairs in threadgroup memory. It retains the original packed
weights and persistent metadata arrays. Staging plus worst-case split partials
is bounded to 32 KiB; unsupported tile/size combinations use mode 1. Mode 0
remains the default. Neither mode changes FMA order or split counts.
Mode `3` is a further diagnostic: mode 2 for multiple rows, mode 0 for a
single row, separating verification throughput from first-token work.

Kernel checks exercise both modes at the target shapes and all six row counts,
including group sizes 32/64, against the existing exact-output checks. The
first five-seed 256-token comparison and reverse-order confirmation used
binary `47460ebda45defe9adbd728adf05d747023a6e38b6c7a2ca334224e63085e5e1`:

| Arm | Code T=1 | Chat T=1 | Code T=0 | Chat T=0 |
|---|---:|---:|---:|---:|
| mode 0 | 172.30 | 76.05 | 195.05 | 80.68 |
| mode 1 | 180.59 | 78.10 | 196.98 | 78.46 |
| mode 2 | 181.90 | 78.40 | 204.86 | 81.62 |
| mode 2 repeat | 182.17 | 78.46 | 197.50 | 81.74 |
| mode 0 repeat | 179.60 | 75.77 | 202.49 | 80.68 |

Mode 2 improved sampled code, sampled chat and greedy chat in both comparisons.
Greedy code was mixed (+5.0%, then -2.5%); the larger first-run gain is not a
stable four-cell speedup. Every arm matched all 24 full replies, with every
request engaging DFlash and no runtime disables. Active MLX allocation stayed
16,399,871,804 bytes and peak stayed 19,993,722,396 bytes. Cache differed by at
most 32 KiB. The expanded validation passed 16/16 scored cells and matched all
40 reference replies, with zero repeat mismatches. Artifacts: `metadata-{0,1,2}-256`,
`metadata-{0,2}-256-repeat`, and `metadata-2-validation`.

Changing the submission interval to two or four layers did not improve the
overall result. At metadata mode 2, pipeline 2/4/1 measured respectively
175.67/75.65/197.31/78.70, 174.91/75.70/197.84/78.69, and
175.62/75.48/197.85/81.60 tok/s. All use an early first-layer submission.
The candidate retains pipeline 1. Artifacts: `metadata-pipeline{1,2,4}-256`.

After removing the producer prototypes, binary
`499f96db5a304156208f6362ed0c442b183ed5246ba5e48db210c59d31fd59eb`
passed the full suite again (2,807 passed, 210 skipped), then ran the requested
64-token protocol with fresh serial controls and normal TensorFold:

| Arm | Code T=1 | Chat T=1 | Code T=0 | Chat T=0 |
|---|---:|---:|---:|---:|
| ours, metadata 0 | 175.25 | 74.05 | 203.99 | 91.18 |
| ours, metadata 2 | 180.14 | 74.80 | 208.41 | 89.69 |
| normal TensorFold | 199.86 | 77.41 | 204.18 | 88.32 |
| ours, default serial | 35.56 | 35.56 | 35.75 | 35.76 |
| ours, serial, metadata 0 env | 35.54 | 35.55 | 35.78 | 35.78 |
| ours, serial, metadata 2 env | 35.58 | 35.57 | 35.80 | 35.81 |
| TensorFold, serial | 27.60 | 27.59 | 27.88 | 27.82 |

Both native speculative arms matched all 24 replies, and all three native
serial arms also matched each other's 24 replies. The serial controls do not
engage the experimental NAX path: these environment switches alone do not
enable the speculative target's row-exact dispatch. Thus unchanged serial
speed demonstrates isolation, not the speed of single-row NAX staging.

Metadata 2 still does not establish a trade-off-free win: greedy chat declined
1.6% in this short-request run, and median first-token times were higher in
all four cells (190/183/187/186 ms versus 195/197/194/195 ms). This motivated
testing mode 3 independently. Active and peak MLX allocation were identical
between speculative arms, with a 16 KiB cache difference. Serial active
allocation was identical across its native arms; peak differed by at most
16 KiB. All native requests returned their first event within 0.207 s in this
pass. Artifacts: `metadata-{off,on,tf}64`, `metadata-default-serial64`,
`metadata-serial-{off,on}64`, and `metadata-tf-serial64`.

The subsequent same-binary 64-token comparison tested the row-specific mode 3:

| Arm | Code T=1 | Chat T=1 | Code T=0 | Chat T=0 |
|---|---:|---:|---:|---:|
| mode 0 | 180.65 | 74.28 | 210.06 | 90.09 |
| mode 2 | 182.07 | 75.42 | 215.01 | 91.89 |
| mode 3 | 182.29 | 72.72 | 209.87 | 90.99 |

Mode 3 did not improve the overall result. Mode 2 improved all four cells in
this pass by 0.8–2.4%, with lower median first-token latency too:
189/187/189/198 ms for mode 0 versus 186/184/182/183 ms for mode 2.
The earlier latency increase therefore was not stable evidence of a
single-row staging penalty. All three arms matched all 24 full replies,
engaged DFlash for every request, and had zero runtime disables. Active and
peak allocation remained 16,399,871,804 and 19,993,722,396 bytes; cache varied
by at most 80 KiB. Artifacts: `metadata-scoped-{0,2,3}-64`.

These arms used the final binary
`4d26e2aa828d5e2c102d82d2b4e040d016231cf82fa26e4339a573061e8ac4e3`.
Exact kernel checks now cover all three experimental dispatch modes and both
16- and 32-column tiles. The release build and full suite passed again:
2,807 passed, 210 skipped, zero failures. Mode 0 remains the default;
mode 2 is the candidate for the final long-request comparison.

### Final same-day comparison and validation

The final binary above ran mode 0, mode 2, then normal TensorFold at 256 tokens,
five seeds per cell, one server at a time, using the same model directories:

| Arm | Code T=1 | Chat T=1 | Code T=0 | Chat T=0 |
|---|---:|---:|---:|---:|
| ours, metadata 0 | 174.54 | 75.99 | 199.33 | 79.90 |
| ours, metadata 2 | 180.33 | 77.86 | 203.42 | 80.87 |
| normal TensorFold | 191.28 | 81.05 | 201.94 | 86.75 |
| native change | +3.3% | +2.5% | +2.1% | +1.2% |

This final pass improves all four native cells, but the earlier reverse-order
greedy-code regression remains part of the evidence. We have **not** established
a consistent four-cell win over TensorFold. Native mode 2 trails it by 5.7%,
3.9%, and 6.8% in sampled code, sampled chat, and greedy chat; greedy code is
0.7% ahead in this run. TensorFold also retains substantially lower median
first-token latency: 74/71/67/69 ms versus native 184/183/183/183 ms.
Native mode 0 measured 185/188/186/184 ms.

All 24 native mode-2 replies exactly matched mode 0, with 24 speculative
requests and zero runtime disables. Cross-engine replies are not identical;
this comparison does not claim token-for-token cross-engine parity. Active
and peak native allocation again matched exactly at 16,399,871,804 and
19,993,722,396 bytes; mode 2's cache was 16 KiB smaller. The final expanded
validation matched all 40 reference replies, had zero repeat mismatches, and
passed all 16 scored cells. Default behavior remains unchanged.

Engagement evidence from `metadata-clean-2-256.log`:

```text
[dflash-nax] affine metadata mode=2 engaged: rows=1 K=5120 N=10240 tile=16 layout=original
[dflash-nax] uint4 engaged: rows=16 K=5120 N=1280 gs=64 tile=16 layout=original
[spec-stats] mode=dflash attempts=29 accepts=227 avg_per_round=7.83 gate_min=2.00 per_draft_pct=52.2% block_size=16 partial_rounds=28 runtime_disabled=false
```

The first TensorFold warmup likewise reported `tokens=256`, `rounds=29`,
`accepted=226/418`, `rows=15.4`, and `checkpoints=off`. Its source remained
clean at `34bae79ac97da6c3ab3fe10159cf49633ce8112a`. All test servers stopped.
Artifacts under `~/claude-tmp/tfo/`: `metadata-clean-{0,2}-256`,
`metadata-clean-tf256`, and `metadata-clean-validation`. Raw logs, request
records, environment settings, and source/binary hashes accompany the results.

The remaining high-value target is reproducing the current Python tiled
weight/cooperative-kernel combination, while preserving the native serial
path and avoiding a persistent second weight copy. The diagnostic above
establishes its importance; the metadata-only change is not an implementation
of that full path. Removed producer-fusion prototypes remain available for
local inspection in ignored `zig-out/tfo/producer-and-metadata-prototypes.patch`.

### Single-storage tiled MLP experiment

The next local experiment is `MLX_SERVE_DFLASH_TILED_MLP=1`, scoped to the
M5 dense Qwen 27B target with native NAX enabled. It replaces joined gate/up
and down projection weights with explicit `[N/64, K/GS, 64, GS/8]` uint32
storage, following current Python `lane_qmm.tile_weight` and `_COOP`.
Scales and biases retain their existing storage. Attention, GDN, embeddings,
the output head, and drafter weights are not converted by this switch.

The name map's gate/up views and the fused allocation's owner are replaced
together. Gate/up remain physical views of one allocation. Conversion is
evaluated and synchronized one projection at a time, with the old buffers
released and allocator cache cleared before proceeding. A load-time active
memory guard fails if a layer retains more than 64 KiB beyond its pre-conversion
residency; this allowance is not enough to hide a retained target projection.
For the tested 34,816-by-5,120 joined matrix the temporary replacement buffer
is 89,128,960 bytes (85 MiB), not another model-sized allocation.

All input widths read the same tiled storage directly. This differs from
TensorFold's `_call`: beyond its 128-row limit it temporarily calls
`untile_weight` before stock MLX prefill. The native prototype does not restore
an original-layout weight during forward execution. Prefill performance must
therefore be validated explicitly; memory equivalence alone is insufficient.

The ownership test checks actual gate/up data pointers, unchanged residency,
and exact results at 1/5/16/17/33/129 rows. Target-shape tests also compare the
cooperative kernel exactly against the original NAX kernel at all draft widths
and both supported group sizes, retaining the existing fp32-reference accuracy
checks. The first full suite passed 2,808 tests, with 210 skipped.

The first model pair, `tiled-mlp-{0,1}-64`, did **not** engage conversion: a
load-time gate incorrectly depended on `dflash_bound`, which is set later.
Those measurements are A/A controls only. The corrected gate checks model and
hardware eligibility before binding; the row-exact coverage scan explicitly
recognizes tiled projections. ANE prefill declines these experimental weights
rather than misinterpreting them through its original-layout reader.

The first engaged 64-token pair (`tiled-live-{1,0}-64`, binary
`f5c0d85f562db9ba7dbb606991f94534b7b70d1c7ed94dd76124caed57d933e8`)
measured:

| Arm | Code T=1 | Chat T=1 | Code T=0 | Chat T=0 |
|---|---:|---:|---:|---:|
| existing path | 180.37 | 74.10 | 212.49 | 89.55 |
| tiled MLP | 206.88 | 73.79 | 259.40 | 76.00 |

These are not an isolated layout speedup: only 9/24 replies matched the
existing path, because the new direct prefill reader uses row-exact NAX
arithmetic rather than stock MLX prefill. Median first-token latency improved
from 194/182/185/188 ms to 159/166/160/158 ms, but greedy chat regressed.
All requests engaged DFlash with no runtime disables. Active allocation was
16,399,871,292 bytes versus 16,399,871,804 for the existing path; peak was
identical at 19,993,722,396 bytes. The per-layer load guard also confirmed
unchanged residency after replacing original allocations.

To isolate the layout, binary
`f271ebe04ebab7b4e48f932cda08df326bab0370ef16354627e7e070cf743e97`
then used the **same 64-column cooperative kernel** on original versus tiled
storage, including identical prefill arithmetic, at 256 tokens:

| Arm | Code T=1 | Chat T=1 | Code T=0 | Chat T=0 |
|---|---:|---:|---:|---:|
| original-layout cooperative | 172.17 | 67.62 | 177.87 | 70.42 |
| tiled cooperative | 186.53 | 74.52 | 195.35 | 77.22 |
| original-layout cooperative repeat | 171.90 | 68.09 | 178.86 | 70.31 |

All 24 full replies matched across all three arms. Tiling improves these cells
by about 8–10%, recovering on the reverse-order control. Active allocation
is exactly 16,399,871,292 bytes in every arm; peak is exactly 19,993,722,396.
This isolates a layout gain **against the original-layout cooperative kernel**,
not against the best existing 16-column native kernel or normal TensorFold.
Artifacts: `tiled-equal-{0,1,2}-256`.

`tiled-validation` passed all 16 scored cells with 40 repeat-stable replies.
Compared with `metadata-clean-validation`, 32/40 texts match; the eight
differences are the long code/chat cases at both temperatures and repeats.
All scored structured tasks, including the 1,866-token context lookup, match.
This is a small correctness corpus, not proof of broad quality equivalence.

A spot-check exposed the first direct reader's prefill cost: the long-context
validation logged about 799–809 prompt tok/s after warmup, versus about 968–971
on the prior native path. The next revision uses TensorFold's 32-row cooperative
tile above sixteen input rows, with bounded 16-output reduction chunks and
explicit bounds on the ragged final group. It still reads the single tiled
allocation directly. The final `reference` arm also uses the best existing
native kernel at decode widths, reserving the original-layout cooperative
reader for wider prefill; it therefore tests whether tiling improves on our
best decode path with identical prefill arithmetic. Earlier `tiled-equal-*`
records used the cooperative reader at every width, as their binary hash pins.

The 32-row prefill revision, binary
`732db10b7a05b5569d24830e786aa2cb1696b902f4d289ed2871ed76615ba9bf`,
passed the full suite (2,808 passed, 210 skipped). Both expanded validation
arms matched all 40 replies of the 16-row tiled version, with zero repeat
mismatches and all 16 scored cells passing. However, the 1,866-token prefill
spot-check remained around 808–809 tok/s; increasing the row tile did not
remove that regression.

Its fresh 64-token, five-seed comparison was:

| Arm | Code T=1 | Chat T=1 | Code T=0 | Chat T=0 |
|---|---:|---:|---:|---:|
| previous native path | 180.65 | 73.02 | 210.91 | 90.49 |
| best native decode, matching prefill arithmetic | 205.21 | 73.99 | 251.31 | 75.88 |
| tiled cooperative | 216.43 | 74.65 | 255.65 | 76.60 |
| normal TensorFold | 202.55 | 77.91 | 205.59 | 88.78 |
| ordinary native serial | 35.58 | 35.59 | 35.79 | 35.81 |
| tiled MLP, serial | 31.53 | 31.51 | 31.70 | 31.70 |

The matching-arithmetic native comparison is exact on all 24 replies:
the tiled arm improves by 5.5% / 0.9% / 1.7% / 0.9% in this single-order pass.
It is a much smaller gain than the 8–10% improvement over untiled cooperative
dispatch. Both have exactly 16,399,871,292 active bytes and 19,993,722,396 peak
bytes. The tiled arm beats TensorFold's two code cells here, but loses both
chat cells and still has higher first-token latency. Cross-engine replies
are not generally identical. It is not an overall TensorFold win.

Ordinary serial reveals a separate 11–12% slowdown when the cooperative
reader replaces stock one-token MLP decoding. Serial active memory is exactly
15,133,982,284 bytes in both arms; peak falls slightly from 15,392,998,296 to
15,387,057,048. Twenty of 24 serial replies match (all greedy replies match).
The memory constraint holds, but the speed trade-offs prohibit default adoption.
Artifacts: `tiled-final-{0,reference,1}-64`, `tiled-final-tf64`,
`tiled-final-serial-{0,1}-64`, and `tiled-final-validation-{0,1}`.

The next reader experiment adds 16- and 32-column NAX reads of the **same**
64-column tiled storage (`MLX_SERVE_DFLASH_TILED_TILE=16|32`; default 64).
Each narrow tile addresses its column range directly inside a stored group.
No untiled matrix is reconstructed, and no weight cache is created. Wider
prefill retains the cooperative reader. Target-shape checks compare every
reader bit-for-bit, including joined split counts, both group sizes, and all
six draft row counts. This tests whether kernel selection can repair the
one-row regression without compromising the memory constraint.

The reader comparison used final binary
`90a046a78faed36bb405b44995c5ba5512206440670052d75f0a991d1d57e96a`.
Release and the full suite passed (2,808 passed, 210 skipped); the suite
includes exact target-shape parity for both new narrow tiled readers.
With 64 output tokens and five measured seeds per cell:

| Reader / mode | Code T=1 | Chat T=1 | Code T=0 | Chat T=0 |
|---|---:|---:|---:|---:|
| tiled 64-column, serial | 31.50 | 31.51 | 31.67 | 31.67 |
| tiled 16-column, serial | 34.14 | 34.15 | 34.35 | 34.34 |
| tiled 32-column, serial | 34.43 | 34.43 | 34.63 | 34.63 |
| tiled 64-column, DFlash | 210.18 | 73.85 | 255.39 | 75.81 |
| tiled 16-column, DFlash | 216.16 | 78.33 | 270.94 | 78.32 |
| tiled 32-column, DFlash | 230.44 | 79.35 | 278.77 | 81.12 |

All 24 replies match exactly within each three-reader comparison, including
sampled responses. Each DFlash arm logs 24 engaged requests and zero runtime
disables. The 32-column reader improves speculative throughput by 7.0–9.6%
over the tiled cooperative reader and recovers most of its serial penalty.
This is a kernel-choice gain over **the same tiled allocation**, not another
weight layout. Artifacts: `tiled-readers-{serial,spec}{64,16,32}`.

All three serial readers have exactly 15,133,982,284 active bytes and
15,387,040,664 peak bytes. All three DFlash readers have exactly
16,399,871,292 active bytes and 19,993,722,396 peak bytes. Allocator cache
varies slightly; it is not a retained original-layout projection. Narrow
reader engagement is explicit, for example:

```text
[dflash-tiled] MLP layer 0 replaced original storage: active_before=144834568 active_after=144834568 duplicate_weight_bytes=0
[dflash-tiled] MLP layer 63 replaced original storage: active_before=9269411848 active_after=9269411848 duplicate_weight_bytes=0
[dflash-tiled] narrow reader engaged: tile=32 storage=single layout=tiled
```

Confirmation then ran tiled 32 first and the best original-layout decode
reference second, using the same final binary and matching prefill arithmetic:

| Arm | Code T=1 | Chat T=1 | Code T=0 | Chat T=0 |
|---|---:|---:|---:|---:|
| best original-layout decode, matching prefill | 205.05 | 75.37 | 255.97 | 74.34 |
| tiled 32-column reader | 229.58 | 79.75 | 277.90 | 80.03 |
| relative improvement | 12.0% | 5.8% | 8.6% | 7.7% |

All 24 replies match exactly between both confirmation arms and the earlier
32-column run. Both confirmation arms retain exactly 16,399,871,292 active
bytes and 19,993,722,396 peak bytes. All requests engage DFlash, with zero
runtime disables. Example from `tiled-confirm-spec32.log`:

```text
[spec-stats] mode=dflash attempts=8 accepts=56 avg_per_round=7.00 gate_min=2.00 per_draft_pct=46.7% block_size=16 partial_rounds=8 runtime_disabled=false table=<2k: table_drops=t0/c0/b0/i0 block_avg=15.00 block_hist= chooser_trials=0
```

Artifacts: `tiled-confirm-spec32` and `tiled-confirm-reference`. This comparison
isolates the decode reader/layout improvement; it is not a comparison against
unmodified upstream main. Against the same-session normal TensorFold run
above, the selected reader wins both code cells and narrowly leads sampled
chat, but remains behind greedy chat (80.03 versus 88.78 tok/s), with higher
first-token latency. It still does not meet the overall no-tradeoff goal.

The final serial control (`tiled-confirm-serial0`, same binary) measured
35.61 / 35.56 / 35.84 / 35.85 tok/s. The tiled 32-column reader remains
3.2–3.4% slower. All greedy replies match; 20/24 replies match overall,
consistent with the earlier stock-versus-row-exact prefill distinction.
Active allocation is identical at 15,133,982,284 bytes; the control's peak
is 15,392,981,912 bytes versus 15,387,040,664 for tiled 32.

Expanded validation (`tiled-confirm-validation32`) passes all 16 scored
cells, produces 40/40 exact matches against `tiled-final-validation-0`,
and has zero repeat mismatches. Active memory is 16,399,874,904 bytes,
matching the earlier expanded validation; peak remains 19,993,722,396.
The 1,866-token prompt still prefills at 805–808 tok/s, versus the earlier
stock path's roughly 968–971. The narrow decode reader does not address
this wider-prefill cost. No benchmark servers remain running.

The implementation therefore remains opt-in. The memory constraint is
satisfied for the tested model: one resident packed allocation per converted
projection, shared gate/up views, no forward-time untile or second weight
cache, and unchanged measured process peak. Conversion does temporarily need
one replacement projection (up to 85 MiB). A no-regression default still
requires faster one-row and wide-prefill consumers of this same storage;
retaining the original weights beside it is not an acceptable workaround.
These measurements were collected from the uncommitted working tree; the
binary hashes above identify the exact tested builds.
