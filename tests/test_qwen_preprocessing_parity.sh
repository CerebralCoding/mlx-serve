#!/usr/bin/env bash
# CPU-only Qwen preprocessing parity against the pinned mlx-vlm processor.
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="${1:-tests/fixtures/house.jpeg}"
FIXTURE="${QWEN_PREPROCESS_FIXTURE:-/tmp/qwen_preprocess_fixture}"
PYTHON="${PYTHON:-python3}"
MIN_PIXELS="${QWEN_PREPROCESS_MIN_PIXELS:-3136}"
MAX_PIXELS="${QWEN_PREPROCESS_MAX_PIXELS:-1003520}"

"$PYTHON" tests/build_qwen_preprocess_fixture.py \
  --image "$IMAGE" \
  --out "$FIXTURE" \
  --min-pixels "$MIN_PIXELS" \
  --max-pixels "$MAX_PIXELS"

./.zig-toolchain/zig build test \
  -Dtest-filter="qwen preprocessing parity" \
  -Dqwen-preprocess-fixture="$FIXTURE" \
  --summary all
