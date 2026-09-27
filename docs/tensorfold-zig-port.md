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

Earlier `candidate-a` and `tiled-a` reports are development experiments,
not results for the final implementation. The untiled experiment regressed
decode, and the first tiled version preceded the joined-width correctness fix.
