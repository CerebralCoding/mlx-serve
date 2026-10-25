# Add a kernel

1. Put the source in a file: `src/kernels/<area>/` when two or more plugins use it, `src/plugins/<name>/kernels/`
   when one does. List it with its sha256 in `src/kernels/digests.txt` (`shasum -a 256`, run from `src/`).
2. Load it through `src/mlx/kernels.zig` (`source` / `resolve`; `MLX_SERVE_KERNEL_DIR` swaps the text at runtime).
3. Register it with the kernel contract: `src/sdk/testing/kernel_contract/REGISTERING.md` (five gates against a
   float64 reference; the planted controls must fail).
4. Gate it on the kernel's own arithmetic (shape, dtype, threadgroup memory), never on one model's shape, and give
   it a kill switch in `src/env.zig` only if the old path is worth comparing.
5. A/B it in situ at per-cell medians before default-on.
