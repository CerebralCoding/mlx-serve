#!/bin/bash
# MTP byte-equivalence test.
#
# Verifies that running the same temp=0 chat completion request against the
# server with --mtp produces *identical* output text to running it without
# --mtp. This is the correctness gate for the MTP draft+verify implementation:
# greedy decode must yield identical token streams whether or not MTP is on.
#
# Requires:
#   - A built mlx-serve binary (run `zig build -Doptimize=ReleaseFast` first)
#   - A model directory with MTP weights present in safetensors
#     (config field `mtp_num_hidden_layers > 0` AND actual `*.mtp.*` tensors)
#
# Most MLX-converted Qwen3.5/Qwen3.6 checkpoints from Hugging Face have the
# config metadata but the conversion strips the MTP weight tensors. Verify
# with: python3 -c "from safetensors import safe_open; \
#   import sys; \
#   [print(k) for k in safe_open(sys.argv[1], framework='pt').keys() if 'mtp' in k.lower()][:5]" \
#   path/to/model.safetensors
#
# Usage:
#   MTP_TEST_MODEL=/path/to/qwen3.5-with-mtp ./tests/test_mtp_equivalence.sh [port]
#
# Without MTP_TEST_MODEL set, the test exits 0 with a skip message — keeps it
# safe for CI matrices that don't have an MTP-bearing checkpoint available.

set -e

PORT=${1:-8090}
BASE="http://127.0.0.1:$PORT"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

if [ -z "$MTP_TEST_MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} test_mtp_equivalence: \$MTP_TEST_MODEL is not set."
    echo
    echo "  Set MTP_TEST_MODEL to a model directory whose safetensors include"
    echo "  MTP head weights (keys matching '*.mtp.*'). Most MLX-converted"
    echo "  Qwen3.5/3.6 checkpoints have the config field but no weights;"
    echo "  see the comment block in this script for verification commands."
    exit 0
fi

if [ ! -d "$MTP_TEST_MODEL" ]; then
    echo -e "${RED}FAIL${NC} \$MTP_TEST_MODEL=$MTP_TEST_MODEL is not a directory."
    exit 1
fi

if [ ! -f "$MTP_TEST_MODEL/config.json" ]; then
    echo -e "${RED}FAIL${NC} $MTP_TEST_MODEL/config.json missing."
    exit 1
fi

# Confirm MTP weights are actually present (not just config metadata).
HAS_MTP_WEIGHTS=$(python3 -c "
import os, sys
from safetensors import safe_open
d = '$MTP_TEST_MODEL'
n = 0
for f in os.listdir(d):
    if f.endswith('.safetensors'):
        try:
            with safe_open(os.path.join(d, f), framework='pt') as st:
                for k in st.keys():
                    if '.mtp.' in k or k.startswith('mtp.'):
                        n += 1
                        break
            if n: break
        except: pass
print('1' if n else '0')
" 2>/dev/null)

if [ "$HAS_MTP_WEIGHTS" != "1" ]; then
    echo -e "${YELLOW}SKIP${NC} test_mtp_equivalence: $MTP_TEST_MODEL config.json may declare MTP layers, but no '*.mtp.*' tensors were found in the safetensors. The MLX conversion likely stripped them."
    exit 0
fi

# Find binary
BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
if [ ! -x "$BINARY" ]; then
    echo -e "${RED}FAIL${NC} $BINARY not found or not executable. Build first with 'zig build -Doptimize=ReleaseFast'."
    exit 1
fi

PROMPT="Write the first line of the Linux kernel boot message."
JSON_PAYLOAD=$(cat <<EOF
{
  "model": "mlx-serve",
  "messages": [{"role": "user", "content": "$PROMPT"}],
  "max_tokens": 64,
  "temperature": 0.0,
  "stream": false
}
EOF
)

# Long-greedy memorized prompt: at INT4, AR vs verify use different MLX
# quantized matmul kernels and near-tie argmax tokens can flip past ~30–80
# tokens. We assert byte-equivalence on the first 30 tokens only — that catches
# real logic regressions while accepting the float-noise cascade. See CLAUDE.md
# "MTP/PLD/drafter long-greedy byte-divergence at INT4".
LONG_PROMPT='Recite the first paragraph of "A Tale of Two Cities" by Charles Dickens.'
LONG_JSON_PAYLOAD=$(python3 -c "
import json, sys
print(json.dumps({
    'model': 'mlx-serve',
    'messages': [{'role': 'user', 'content': sys.argv[1]}],
    'max_tokens': 200,
    'temperature': 0.0,
    'stream': False,
}))
" "$LONG_PROMPT")
FIRST_N_TOKENS=30

run_request() {
    # All status messages go to stderr so the captured stdout is JUST the
    # final completion text from the model. Optional 3rd arg: payload override.
    local label="$1" mtp_flag="$2" payload="${3:-$JSON_PAYLOAD}"
    echo -e "  starting server ($label)..." >&2
    local logfile=$(mktemp)
    "$BINARY" --model "$MTP_TEST_MODEL" --serve --port "$PORT" $mtp_flag > "$logfile" 2>&1 &
    local pid=$!
    # Wait for server up (max 60s — model load can be slow)
    local up=0
    for i in $(seq 1 60); do
        if curl -s -f "$BASE/health" > /dev/null 2>&1; then
            up=1
            break
        fi
        sleep 1
    done
    if [ "$up" != "1" ]; then
        echo -e "  ${RED}FAIL${NC} server did not become healthy in 60s" >&2
        cat "$logfile" | tail -20 >&2
        kill $pid 2>/dev/null || true
        rm -f "$logfile"
        return 1
    fi
    local body
    body=$(echo "$payload" | curl -s -X POST -H "Content-Type: application/json" -d @- "$BASE/v1/chat/completions")
    kill $pid 2>/dev/null || true
    wait $pid 2>/dev/null || true
    rm -f "$logfile"
    # Extract content
    echo "$body" | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
}

# Run the request AND tokenize the completion before stopping the server so we
# can do token-level comparison. Stores results into the named variables.
run_and_tokenize() {
    local label="$1" mtp_flag="$2" payload="$3" out_completion_var="$4" out_tokens_var="$5"
    echo -e "  starting server ($label)..." >&2
    local logfile=$(mktemp)
    "$BINARY" --model "$MTP_TEST_MODEL" --serve --port "$PORT" $mtp_flag > "$logfile" 2>&1 &
    local pid=$!
    local up=0
    for i in $(seq 1 60); do
        if curl -s -f "$BASE/health" > /dev/null 2>&1; then
            up=1
            break
        fi
        sleep 1
    done
    if [ "$up" != "1" ]; then
        echo -e "  ${RED}FAIL${NC} server did not become healthy in 60s" >&2
        cat "$logfile" | tail -20 >&2
        kill $pid 2>/dev/null || true
        rm -f "$logfile"
        return 1
    fi
    local body
    body=$(echo "$payload" | curl -s -X POST -H "Content-Type: application/json" -d @- "$BASE/v1/chat/completions")
    local completion
    completion=$(echo "$body" | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])")
    local tok_payload
    tok_payload=$(python3 -c "import json,sys; print(json.dumps({'content': sys.argv[1]}))" "$completion")
    local tokens
    tokens=$(echo "$tok_payload" | curl -s -X POST -H "Content-Type: application/json" -d @- "$BASE/tokenize" | python3 -c "import sys,json; print(','.join(str(t) for t in json.load(sys.stdin)['tokens']))")
    kill $pid 2>/dev/null || true
    wait $pid 2>/dev/null || true
    rm -f "$logfile"
    printf -v "$out_completion_var" '%s' "$completion"
    printf -v "$out_tokens_var" '%s' "$tokens"
}

echo "== MTP byte-equivalence test =="
echo "  model: $MTP_TEST_MODEL"
echo "  prompt: $PROMPT"
echo

# Pre-emptively kill any stale server on the test port.
pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
sleep 1

OUT_NOMTP=$(run_request "without --mtp" "--no-mtp") || exit 1
echo "  no-mtp output captured ($(echo "$OUT_NOMTP" | wc -c) bytes)"

sleep 2
OUT_MTP=$(run_request "with --mtp" "--mtp") || exit 1
echo "  with-mtp output captured ($(echo "$OUT_MTP" | wc -c) bytes)"

if [ "$OUT_NOMTP" = "$OUT_MTP" ]; then
    echo -e "${GREEN}PASS${NC} short-prompt byte-identical output with vs without --mtp"
else
    echo -e "${RED}FAIL${NC} outputs differ:"
    echo "  --no-mtp:"
    echo "$OUT_NOMTP" | sed 's/^/    /'
    echo "  --mtp:"
    echo "$OUT_MTP" | sed 's/^/    /'
    diff <(echo "$OUT_NOMTP") <(echo "$OUT_MTP") | sed 's/^/    /'
    exit 1
fi

echo
echo "== MTP long-greedy first-${FIRST_N_TOKENS}-tokens equivalence =="
echo "  prompt: <memorized recital, max_tokens=200>"
echo "  rationale: see CLAUDE.md 'MTP/PLD/drafter long-greedy byte-divergence at INT4'"
echo

sleep 2
LONG_COMPLETION_NOMTP=""
LONG_TOKENS_NOMTP=""
run_and_tokenize "without --mtp (long)" "--no-mtp" "$LONG_JSON_PAYLOAD" LONG_COMPLETION_NOMTP LONG_TOKENS_NOMTP || exit 1
echo "  no-mtp long completion ($(echo "$LONG_COMPLETION_NOMTP" | wc -c) bytes, $(echo "$LONG_TOKENS_NOMTP" | tr ',' '\n' | wc -l | tr -d ' ') tokens)"

sleep 2
LONG_COMPLETION_MTP=""
LONG_TOKENS_MTP=""
run_and_tokenize "with --mtp (long)" "--mtp" "$LONG_JSON_PAYLOAD" LONG_COMPLETION_MTP LONG_TOKENS_MTP || exit 1
echo "  with-mtp long completion ($(echo "$LONG_COMPLETION_MTP" | wc -c) bytes, $(echo "$LONG_TOKENS_MTP" | tr ',' '\n' | wc -l | tr -d ' ') tokens)"

DIVERGENCE=$(python3 - <<PY
nomtp = "$LONG_TOKENS_NOMTP".split(",") if "$LONG_TOKENS_NOMTP" else []
mtp   = "$LONG_TOKENS_MTP".split(",") if "$LONG_TOKENS_MTP" else []
n = $FIRST_N_TOKENS
a = nomtp[:n]
b = mtp[:n]
if len(a) < n or len(b) < n:
    print(f"SHORT len(no-mtp)={len(nomtp)} len(mtp)={len(mtp)} need>={n}")
else:
    diverge = -1
    for i,(x,y) in enumerate(zip(a,b)):
        if x != y:
            diverge = i
            break
    if diverge < 0:
        print("OK")
    else:
        print(f"DIFF at index {diverge}: no-mtp={a[diverge]} mtp={b[diverge]}")
PY
)

if [ "$DIVERGENCE" = "OK" ]; then
    echo -e "${GREEN}PASS${NC} first ${FIRST_N_TOKENS} tokens byte-identical with vs without --mtp"
    exit 0
else
    echo -e "${RED}FAIL${NC} first-${FIRST_N_TOKENS}-tokens divergence: $DIVERGENCE"
    echo "  no-mtp first ${FIRST_N_TOKENS} tokens: $(echo "$LONG_TOKENS_NOMTP" | cut -d',' -f1-${FIRST_N_TOKENS})"
    echo "  with-mtp first ${FIRST_N_TOKENS} tokens: $(echo "$LONG_TOKENS_MTP" | cut -d',' -f1-${FIRST_N_TOKENS})"
    exit 1
fi
