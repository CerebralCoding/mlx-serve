# TensorFold native projection port

Source: the sibling TensorFold `feat/zig` refactor at `2fbd46c`. Its working
tree is not modified. Attribution is in `NOTICE`.

## Implemented

`src/tensor_qmm.zig` and `src/metal/tensor_qmm.metal` adapt the native
`lanes.Linear` projection for mlx-c. The path supports M5 tensor units,
bf16 activations/scales/biases, affine Q4 with group size 64, and 1–16 rows.
Unsupported hardware, CPU streams, dtypes, quantization, or geometry retain
the existing dispatch.

Weights are tiled once; scales and biases are transposed and interleaved.
The model owns the prepared copies and retained source descriptors, so the
cache cannot confuse a recycled wrapper address with an old weight. Unloading
the model releases them. Preparation is lazy on the first eligible forward,
which adds first-use latency and memory residency.

Every row uses the same 16-row tensor operation and K reduction order. The
MLP must also use the same joined gate/up width in serial and verification
calls: the tensor split-K policy depends on that width. The integration test
caught a late sampled-output mismatch before this dispatch rule was fixed.

Enable explicitly before starting the server:

```sh
env MLX_SERVE_ROW_TENSOR_QMM=1 ./zig-out/bin/mlx-serve \
  --model "$HOME/.mlx-serve/models/ddalcu/Qwen3.8-27B-MLX-Serve-4bit" --mtp --serve
```

The default is off. This port trades a second packed copy of eligible weights
for faster projection; it is not a free memory optimization. The measured
configuration is an M5 Max with 128 GB. Other M5 configurations and other
dense Qwen checkpoint geometries have not been performance-qualified.

## Correctness checks

```sh
.zig-toolchain/zig build test -Doptimize=ReleaseFast -Dtest-filter=tensor_qmm
python3 tests/test_tensor_qmm.py --url http://127.0.0.1:19288 \
  --log zig-out/tensorfold-port/tiled-fixed-a.log
```

The HTTP test expects a tensor-enabled server with `--mtp --no-pld
--no-drafter --prefix-cache-entries 0` and the supplied `--log-file`.
It checks full output, finish reason, token count, MTP engagement, and serial
opt-out for greedy and seeded sampled requests, including 1-/3-token limits.
All six cases passed on the ddalcu 27B pack.

Kernel tests cover every row of a 16-row window, slices at different offsets,
batched reshaping, partial column tiles, split-K variants, cached reuse, CPU
decline and invalid geometry. Accuracy is checked against independently
dequantized float32 matmul: relative RMS error below 1%, and RMS error no
more than 3× stock quantized matmul's error against the same reference.

After the live correctness cases, MLX active memory was 32,232,786,756 bytes.
After `/v1/unload-model`, it was 131,112 bytes. This establishes release of
prepared weights on the actual model-unload path, not merely process exit.

### Flash, 6-bit and 8-bit validation

On 2026-09-27, each additional ddalcu pack was run sequentially on both the
pre-port executable (`2b2155f`) and the current executable (`fd7bad1`), with
`MLX_SERVE_ROW_TENSOR_QMM=1` on the current executable. Six prompt/settings
cases were each requested with MTP off and on: two greedy, two seeded sampled,
and sampled 1-/3-token limits. Complete messages, finish reasons and completion
counts were compared; MTP engagement and serial opt-out were checked in logs.

| Model | Serial/MTP exact matches | Current vs baseline responses | Active MLX bytes, both builds |
| --- | ---: | --- | ---: |
| Flash Next mixed 4/8-bit | 3/6 | All 12 identical | 74,733,650,580 |
| 27B 6-bit | 5/6 | All 12 identical | 22,821,391,188 |
| 27B 8-bit | 6/6 | All 12 identical | 30,828,547,156 |

All requests succeeded and respected their token limits. The tensor port
correctly remained inactive in all three models, and the active-memory delta
against baseline was exactly zero. Flash has a different architecture; Q6/Q8
decline the affine-Q4-only kernel. The existing `simd_qmm` and `rowqmv` paths
also accept only 4-bit weights, so the row-exact projection guarantee does not
extend to Q6/Q8 simply because their model family is eligible.

Flash differed between serial and MTP on the greedy sky explanation and both
long sampled cases. Q6 differed on the temperature-0.7, seed-42 sky explanation
(91 versus 89 tokens). These same differences and complete responses appeared
on the baseline: they are existing parity gaps, not regressions introduced by
this port. The Q8 cases passed, but six cases do not establish universal parity.
No general quality, vision, long-context, concurrency or performance claim is
made from these short HTTP validation runs.

The new 6-/8-bit downloads live under `~/.mlx-serve/models/ddalcu/`, with direct
symlinks in `models/`. Both passed `hf cache verify --fail-on-missing-files`
for all 19 remote files at these revisions:

* 6-bit: `0fb56b5c9579b24d3d62c4d70f37ade7c4dcc0c0`.
* 8-bit: `011e38296b3d2aa99245ed49a700459c4ac246b6`.
* Existing Flash: `7eaef0fa82b4c3bf5c64cec60ace4bf48fd271e3`.

Raw responses, props snapshots and logs are in
`zig-out/tensorfold-port/{flash,6bit,8bit}-{candidate,baseline}*`.
All validation servers were stopped afterward.

## Other native work assessed

* KV capacity management, queued MTP chains, and adaptive draft scheduling
  already have implementations here. Replacing them wholesale would duplicate
  machinery and discard this server's batching and multi-model contracts.
* The native mapped draft vocabularies concern Nemotron and Flash. This server
  already limits dense Qwen drafts to 98,304 vocabulary rows. Porting the other
  ID maps requires separate checkpoint/tokenizer and sampling-key validation;
  it is not part of this projection change.
* Native resident Flash PLE enables GPU token handoff at the cost of keeping
  the large table resident. The selected ddalcu Flash pack is already about
  100 GiB on disk. That memory tradeoff needs separate qualification; the
  existing bounded PLE policy is retained.
* Tensor attention is a separate kernel port and needs its own comparison
  against the server's existing attention paths. Projection measurements
  cannot establish an attention benefit.

## Measurement artifacts

Local llmprobe reports and server logs are under `zig-out/tensorfold-port/`.
Measurements use llmprobe 0.6.12, MTP enabled, reasoning off, 16K context,
512/4K rungs, and warmup plus three samples per scenario. They are same-machine
comparisons with the rebased mlx-serve branch, not comparisons against the
TensorFold Python or native engines.

The baseline executable was saved from `feature/tensorfold` at `2b2155f`,
after rebasing onto upstream `main` at `d36b3e0`. The checkpoint is
`ddalcu/Qwen3.8-27B-MLX-Serve-4bit` revision
`b543ed7cbafbf4984266e784c2ccb1f0730847b6`.

Final measurements on 2026-09-27 ran in this order: tensor `final-a`,
baseline `baseline-b`, tensor `final-b`, each in a fresh server process.
The tensor engagement line is present in both tensor logs and absent from
the baseline. No builds or other model work ran alongside these measurements.

| llmprobe workload | Baseline | Tensor A | Tensor B |
| --- | ---: | ---: | ---: |
| Decode, tok/s | 66.3 | 70.0 | 70.1 |
| Predictable MTP, tok/s | 184.9 | 283.1 | 281.9 |
| Novel MTP, tok/s | 46.3 | 51.0 | 50.8 |
| Prefill, tok/s | 976.6 | 969.6 | 973.8 |
| ~513-token context decode, tok/s | 66.8 | 71.6 | 71.5 |
| ~4,254-token context decode, tok/s | 71.5 | 77.4 | 77.2 |
| Four-stream aggregate, tok/s | 87.9 | 85.3 | 106.2 |
| First-token latency, ms | 233 | 237 | 234 |

The repeated single-stream gains were about 6% on decode, 53% on predictable
MTP, and 10% on novel MTP. Prefill stayed within 1%. Concurrency varied enough
that these runs do not establish a concurrency win. Speculative content and
acceptance can vary across run nonces; these are workload-specific observations.
The final three sustained-load checks were steady (0.3%, 0%, and -0.1% drift).

After the full benchmark, active MLX memory was 17.83 GB for the baseline
and 32.23 GB for Tensor B, a 14.40 GB increase (decimal units). Peak allocation
was 22.06 GB and 36.43 GB respectively. This is why the path remains opt-in.

### Memory accounting

Reading the 27B checkpoint's safetensors headers accounts for the observed
increase without loading another model:

| Additional representation | Bytes |
| --- | ---: |
| Tiled Q4 trunk projection weights | 12,799,180,800 |
| Interleaved bf16 scales and biases | 1,601,372,160 |
| Predicted total | 14,400,552,960 |
| Measured active-memory increase | 14,400,552,956 |

The four-byte residual is immaterial. For an N-by-K affine Q4 matrix with
group size 64, the new tiled weight costs N*K/2 bytes and its interleaved
scale/bias buffer costs N*K/16 bytes. The 48-row GDN a/b projections reuse
their original weight layout and add only scale/bias buffers. Embedding
gathers and the separately implemented MTP head are not included. The three
MLP projections alone account for 9,625,927,680 bytes of prepared storage.

`Packed.init` materializes these representations once. Its retained source
descriptors share existing storage; they do not create a third weight copy.
The original row-major arrays remain necessary for stock prefill (>16 rows)
and fallback dispatch. Joined gate/up arrays replace their separate storage
with views, so they are not an additional full copy either. The allocator
pool was actually smaller in Tensor B, and KV residency was zero after both
runs. Model unload released the prepared buffers as described above.

The principal fix would be a single canonical tiled representation consumed
by both decode and prefill kernels, with corresponding loader/ownership
changes. Simply dropping the original handles would break existing consumers.
A smaller interim change could prepare only selected projections under a
memory budget, but its speed/memory tradeoff must be measured. Dropping just
the scale/bias copy can recover at most 1.60 GB and would require changing the
kernel's reads. The untiled experiment already regressed decode, so removing
the tiled weight copy while keeping the same kernel is not a proven solution.

### Investigation: one layout for prefill and decode

TensorFold already demonstrates the ownership half of this design:
`native/weights.zig::releaseLinearSources` removes the original projection
arrays after constructing its `lanes.Linear` objects. Its tensor-mode Linear
owns just the tiled weight and interleaved scale/bias buffer. Its `apply`
accepts 1–128 rows, using 16- or 32-row work blocks. That is useful source
material, but does not qualify large mlx-serve prefills for speed.

The layout itself is usable at any activation-row count. In uint32 units,
logical packed coordinate `(n, j)` maps to:

```
(((n / 32) * (K / 64) + j / 8) * 32 + n % 32) * 8 + j % 8
```

The scale/bias pair for `(n, group)` is at `2 * (group * N + n)`. A prefill
kernel can read these same tiles directly. This is not a normal 2D strided
view: the divisions/remainders mean stock quantized matmul cannot consume
the buffer simply by changing array strides. MLX's pinned
`QuantizedBlockLoader` in `quantized_nax.h` assumes row-major packed weights
and separate row-major scales/biases. Its prefill dispatcher uses 32/64-row
by 64-column tiles; a tiled loader would need to assemble adjacent 32-column
storage tiles for that computation.

Two kernel approaches merit measurement:

1. Extend the current MPP kernel across many row blocks, initially following
   TensorFold's 32-row block. This preserves its quantized-dot plus affine
   correction arithmetic, but its split-K and repeated scale loads were
   selected for small row counts and may underperform large GEMM.
2. Write a prefill tile loader for the same packed representation, with
   larger row/column work tiles. Dequantizing each tile into threadgroup
   memory and using bf16 matrix multiplication follows stock MLX's prefill
   structure. It avoids a full resident dequantized matrix, but its rounding
   differs from the current quantized-dot kernel and needs accuracy checks.

Start with the 64 dense MLPs: they contain 9.63 GB of the duplication and
have a compact set of shapes. Keep gate/up's joined output width consistent
through decode and verification; a changed width changes split-K arithmetic.
Adopt an explicit projection object carrying layout, quantization and logical
shape, rather than passing a tiled buffer disguised as a normal weight.
Convert at load and release original arrays only after all consumers of that
projection use the new object. Replace descriptor-pointer cache identity
with model-owned projection identity as part of that ownership change.

Consumers to audit before releasing originals include batched verification
paths that bypass `Transformer.qmatmul`, ANE/dequantized-prefill paths, and
the MTP draft-head builders, full-head fallback and candidate-row reranker
in `src/mtp.zig`. The latter directly gather/dequantize the trunk lm_head;
leave the head in its original representation in the first MLP-only phase.
Unsupported hardware and quantizations can choose original storage at model
load. Runtime fallback from an already converted projection needs a tiled
reader or a bounded temporary conversion; retaining every original "just
in case" recreates the memory problem.

Qualification should cover rows 1–16 and prefill boundaries 17/31/32/33,
127/128/129, 512 and 4096/8192; independent float32 dequantized references;
decode/verification row equality; full-model serial/MTP completion equality;
mixed prefill/decode scheduling; unload/reload; and measured resident/peak
memory. Compare llmprobe prefill, TTFT, decode and concurrency against both
stock and the current duplicated-layout implementation. Do not assume that
shrinking the server's prefill chunk to 128 is performance-neutral.

This proposed first phase remains affine Q4/gs64 on M5. Q6 and Q8 need their
own packed readers and arithmetic qualification; Flash's expert gathers and
mixed quantization need a separate design. Their current validation checks
the existing fallback paths, not an unimplemented shared-layout optimization.

### Experiment: shared dense-MLP storage

The first approach above is implemented behind a second, default-off flag:

```sh
env MLX_SERVE_ROW_TENSOR_QMM=1 MLX_SERVE_SHARED_TENSOR_MLP=1 \
  ./zig-out/bin/mlx-serve \
  --model "$HOME/.mlx-serve/models/ddalcu/Qwen3.8-27B-MLX-Serve-4bit" \
  --mtp --serve
```

All 64 dense MLPs use explicit `tensor_qmm.Projection` objects for joined
gate/up and down. Each object holds only tiled Q4 weights and interleaved
bf16 scale/bias pairs. Conversion materializes these buffers before releasing
the original map entries and joined-array owners. Source descriptors are no
longer needed as cache keys. The existing model-owned array list owns the new
buffers through unload. Other projections retain the existing cache design.

Decode and verification retain the same joined output width and split-K
policy. Prefills below 1024 rows use a dedicated reader: vectorized Q4 loads,
tile-local bf16 dequantization, and Metal matrix multiplication read that same
storage. Output tiles cover 64 columns and 64/128 rows, with only a 9 KiB
weight tile in threadgroup memory. Larger prefills restore temporary row-major
Q4 weights and scales/biases for stock MLX QMM. These
arrays have no model or cache owner and are released after evaluation; no
second packed representation is retained between requests. Joined
SwiGLU reads gate/up in place, avoiding the contiguous copies that slicing
the joined projection would otherwise require. Its sigmoid lookup and two
bf16 multiplies preserve the existing activation's rounding. This does not
reduce the server's 8192-token prefill chunk. The MTP head and trunk lm_head are unchanged, and
the `g17_nax_q4_gs64` MTP profile remains available. ANE prefill explicitly
declines when shared MLP storage is installed. Q6, Q8, Flash and unsupported
geometries retain their existing representation; this experiment supplies no
new reader for their quantization or expert layouts.

The load log must contain:

```
[shared-tensor-mlp] 64 layers prepared; original MLP arrays released; tiled layout serves all row counts
[shared-tensor-mlp] dedicated tiled prefill reader engaged
[shared-tensor-mlp] joined SwiGLU reads gate/up in place
[shared-tensor-mlp] transient stock prefill engaged (no retained row-major weights)
```

Validation includes split-K 1/2/4/8, first/last row equality for the row-exact
reader, and independent float32 dequantized-matmul references for every
prefill output. Row boundaries include 1/16/17/31/32/33/63/64/65/127/128/129/
255/256/257/511/512/513/1023/1024/1025/4096/8192; column cases include partial tiles and the
actual 27B gate/up and down projection dimensions. The dedicated reader is
also checked bit-for-bit against stock unsplit MLX NAX prefill, including
the model's real projection shapes through 1025 rows. A synchronized allocation
check at both real projection sizes permits only the result plus 64 KiB of
bookkeeping after evaluation, catching retained restored weights. The
accuracy limits remain relative RMS below 1% and RMS at most 3x stock QMM's
error. Ownership tests check that original handles are cleared, exactly four
new arrays remain owned per MLP, and decode/prefill still work. CPU streams
and six-/eight-bit configurations decline conversion without changing ownership.

Bitwise stock-prefill equality is deliberately scoped to the checked unsplit
NAX shapes. Stock MLX can choose another split-K reduction for short prompts.
In the six short HTTP cases, five complete responses matched the duplicated
control, while the temperature-0.7/seed-42 sky explanation differed (84 versus
70 tokens). Both builds independently passed all six serial/MTP comparisons.
These tests do not establish universal cross-build sampling or model-quality
equivalence.

The full Zig suite passed (2787 passed, 210 skipped), as did the release build.
The live test is `tests/test_tensor_qmm.py --shared --log <server-log>
--report <responses.json>`, using MTP, no PLD/drafter, and no prefix-cache
entries. All nine cases passed, including the 8852-token prompt crossing the
8192-token chunk boundary. An actual unload returned active MLX memory to
131,100 bytes; reloading installed all 64 shared MLPs again and reproduced
the checked seeded response. The final response report, unload snapshot and
reload response use the `shared-transient-b-` prefix under
`zig-out/tensorfold-port/`. All validation servers were stopped afterward.

The comparison uses the same llmprobe 0.6.12 protocol above, AC power,
`--ctx-size 16384 --mtp --no-pld --no-drafter --prefix-cache-entries 0
--no-vision --no-warmup-eager`, and port 19288. No builds or other model jobs
ran alongside benchmarks. `mlx-serve-duplicated` is the saved `fd7bad1`
executable with only `MLX_SERVE_ROW_TENSOR_QMM=1`; shared runs add the second
flag. Each retained arm starts in a fresh process and runs the six short
correctness cases before llmprobe's own warmup and median-of-three protocol.
Engagement is confirmed in each server log. An earlier cold duplicated run
had 31.3% decode drift while MTP changed widths; its 53.3 tok/s result is
excluded. The retained arms ran in A/control-D/B order, with the discarded
fused-copy trial and final build/checks between A and D. D has 1% decode
drift; both shared runs have 0.1% drift. Reports are `duplicated-d.json` and
`shared-transient-{a,b}.json`. The final B executable also includes the
CPU-installation guard; its GPU implementation is unchanged from A.

| llmprobe workload | Duplicated D | Shared transient A | Shared transient B |
| --- | ---: | ---: | ---: |
| Decode, tok/s | 69.3 | 70.5 | 70.5 |
| Predictable MTP, tok/s | 282.3 | 283.9 | 286.5 |
| Novel MTP, tok/s | 54.1 | 51.9 | 51.6 |
| Prefill (~2K), tok/s | 974.7 | 960.5 | 953.3 |
| ~512-token context decode, tok/s | 70.5 | 72.1 | 73.0 |
| ~4.3K-token context decode, tok/s | 73.7 | 75.6 | 75.4 |
| ~4.3K-token context prefill, tok/s | 1012 | 996 | 994 |
| ~4.3K-token first-token latency, seconds | 4.2 | 4.3 | 4.3 |
| Four-stream aggregate, tok/s | 104.6 | 109.8 | 106.1 |
| Short-prompt first-token latency, ms | 238 | 188 | 224 |

Decode and predictable MTP are retained; prefill costs 1.5–2.2% at ~2K
and 1.6–1.8% at ~4.3K. This is a substantial reduction from the first shared
reader's 28–42% prefill regression, but it is not zero overhead. Prompt
nonces change token counts, and small prompts can cross kernel tile
boundaries; these runs establish no universal TTFT, novel-MTP or concurrency
improvement. Four streams do not qualify arbitrary continuous batching or
mixed prefill/decode scheduling.

| MLX allocation after the benchmark | Duplicated D | Shared transient A | Shared transient B |
| --- | ---: | ---: | ---: |
| Active bytes | 32,232,787,044 | 22,606,859,320 | 22,606,859,320 |
| Peak bytes | 36,471,835,744 | 27,767,837,240 | 27,724,236,204 |
| Allocator cache bytes | 119,785,500 | 102,496,300 | 119,785,500 |
| KV cache bytes | 0 | 0 | 0 |

The MLP storage saving is **9,625,927,680 bytes** (9.63 GB / 8.96 GiB):
29.9% of duplicated-layout residency and 66.8% of its extra weight storage.
The measured active delta includes another 44 bytes of runtime bookkeeping.
Other projection copies still account for about 4.77 GB above the original
representation. Peak allocation falls by 8.70–8.75 GB here. Temporary
restoration costs roughly 1 GB of peak allocation compared with the direct
shared reader, while preserving the full resident-weight saving.

Keep the experiment opt-in. Its resident representation is shared, while
large-prefill execution uses temporary stock-layout buffers. It does not
establish that a direct tiled reader can match stock prefill at every size.
Wider tiles, explicit register fragments and a fused restore kernel were
tested and discarded because they did not improve the retained tradeoff.
The `shared-joined-*`, `shared-square-*`, `shared-aligned-*`, `shared-register*`
and `shared-fused-*` artifacts are development variants, not final results.
