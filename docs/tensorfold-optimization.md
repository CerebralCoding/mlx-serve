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
