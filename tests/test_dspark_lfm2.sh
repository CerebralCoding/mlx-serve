#!/bin/bash
# test_dspark_lfm2.sh — DSpark (LiquidAI LFM2.5, 2.6B dense hybrid and 8B-A1B lfm2_moe) on a target whose drafter/
# holds the sidecar. [1] the sidecar is CLASSIFIED as DSpark (`dspark: markov head rank=`) and the hybrid trunk no
# longer vetoes it (`[spec-wiring] ... dflash=true`); [2] rounds ENGAGE with accepts > 0; [3] greedy bytes equal a
# --no-drafter boot on the RAW /v1/completions surface (a chat model's reasoning amplifies a near-tie flip), on an echo
# prompt so partial-accept rounds (the conv-state rollback) still happen; [4] the Markov chain is LOAD-BEARING:
# MLX_SERVE_DFLASH_MARKOV=0 roughly halves acceptance on a NOVEL prompt.
#
#   DSPARK_TEST_MODEL=<target dir> ./tests/test_dspark_lfm2.sh   # DSPARK_TEST_PORT (11357); SKIPs without drafter/
#
# Hermetic twins: the split-contract, anchor_row_drafts and markov-head parse tests, dflashGateMinimum.

set -euo pipefail

MODEL="${DSPARK_TEST_MODEL:-}"
if [ -z "$MODEL" ]; then echo "SKIP: DSPARK_TEST_MODEL not set"; exit 0; fi
if [ ! -f "$MODEL/drafter/config.json" ]; then
    echo "SKIP: $MODEL/drafter/config.json not found"; exit 0
fi

PORT="${DSPARK_TEST_PORT:-11357}"
BASE="http://127.0.0.1:$PORT"
BIN="$(dirname "$0")/../zig-out/bin/mlx-serve"
SERVER_PID=""
cleanup() { [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

boot() { # $1 = log file, $2... = extra args (env via DSPARK_ENV)
    local log="$1"; shift
    cleanup
    # shellcheck disable=SC2086
    env ${DSPARK_ENV:-} "$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" \
        --ctx-size 8192 --prefix-cache-entries 0 --log-level debug "$@" > "$log" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 120); do curl -s -m 2 "$BASE/health" > /dev/null 2>&1 && return 0; sleep 1; done
    echo "FAIL: server did not come up"; cat "$log"; exit 1
}

# An ECHO prompt on the RAW completions surface: no chat template, so no
# reasoning block to amplify one near-tie argmax flip into two different
# plans, and the repetition keeps acceptance high so partial-accept rounds
# (the conv-state rollback) actually happen.
ECHO_TEXT=$(python3 -c "print(' '.join(['The quick brown fox jumps over the lazy dog number %d.' % i for i in range(40)]))")
BODY=$(python3 - "$ECHO_TEXT" <<'PY'
import json, sys
print(json.dumps({"model": "m", "max_tokens": 300, "temperature": 0, "prompt": sys.argv[1]}))
PY
)
# A NOVEL prompt: nothing in it is draftable from context, so its acceptance
# is the Markov chain's own contribution and nothing else's.
NOVEL='{"model":"m","max_tokens":220,"temperature":0,"messages":[{"role":"user","content":"Explain in detail how a bicycle derailleur shifts gears, step by step."}]}'
asknovel() { curl -s -m 300 "$BASE/v1/chat/completions" -H 'content-type: application/json' -d "$NOVEL" > /dev/null; }
pdpct() { grep -o "mode=dflash.*per_draft_pct=[0-9.]*%" "$1" | tail -1 | sed -n 's/.*per_draft_pct=\([0-9]*\)\..*/\1/p'; }

ask() { curl -s -m 300 "$BASE/v1/completions" -H 'content-type: application/json' -d "$BODY" \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['text'])"; }

L1=$(mktemp /tmp/dspark_serial.XXXXXX); L2=$(mktemp /tmp/dspark_on.XXXXXX); L3=$(mktemp /tmp/dspark_nomarkov.XXXXXX)

echo "[1] serial reference"
boot "$L1" --no-drafter
SERIAL=$(ask)

echo "[2] DSpark engaged"
boot "$L2"
grep -q "dspark: markov head rank=" "$L2" || { echo "FAIL: sidecar not classified as DSpark"; exit 1; }
DS=$(ask)
grep -q "\[spec-wiring\].*dflash=true" "$L2" || { echo "FAIL: dflash not wired (hybrid veto?)"; exit 1; }
STATS=$(grep -o "mode=dflash.*per_draft_pct=[0-9.]*%" "$L2" | tail -1)
[ -n "$STATS" ] || { echo "FAIL: no dflash rounds"; exit 1; }
echo "    $STATS"
ACC=$(echo "$STATS" | sed -n 's/.*accepts=\([0-9]*\).*/\1/p')
[ "${ACC:-0}" -gt 0 ] || { echo "FAIL: zero accepted drafts"; exit 1; }

echo "[3] greedy output identical to serial"
[ "$SERIAL" = "$DS" ] || { echo "FAIL: DSpark greedy output differs from serial"; diff <(echo "$SERIAL") <(echo "$DS") | head; exit 1; }

echo "[4] the Markov chain is load-bearing (novel prompt)"
# An ECHO prompt drafts fine from the base logits alone, so the comparison
# has to run where the chain is the only thing carrying it.
asknovel
ON_PD=$(pdpct "$L2")
DSPARK_ENV="MLX_SERVE_DFLASH_MARKOV=0" boot "$L3"
asknovel
OFF_PD=$(pdpct "$L3")
echo "    novel per-draft: markov on=${ON_PD}% off=${OFF_PD}%"
[ -n "$ON_PD" ] && [ -n "$OFF_PD" ] || { echo "FAIL: no novel-prompt dflash rounds"; exit 1; }
[ "$OFF_PD" -lt "$((ON_PD / 2))" ] || { echo "FAIL: base logits alone draft as well as the chain — markov head may be unused"; exit 1; }

rm -f "$L1" "$L2" "$L3"
echo "PASS: DSpark on LFM2.5"
