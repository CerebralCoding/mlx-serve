#!/bin/bash
# LAN model sharing (src/lan.zig): two servers on ONE machine.
#
#   A  --model <dir> --lan-share all --lan-name lantest-a --lan-discover
#   B  headless (--model-dir <empty>) --lan-discover
#
#   1. B discovers A over Bonjour: /v1/models on B gains `<id>@lantest-a`.
#   2. Chat through B naming the remote id → proxied to A, real completion
#      back (non-stream + SSE stream with [DONE]).
#   3. /v1/load-model + /v1/unload-model on B with the remote id → 200 no-ops.
#   4. Share gate on A via the machine's LAN IP (non-loopback): /metrics,
#      GET /, /v1/load-model and `@peer` model ids are all 403 host-local,
#      while the same requests from loopback keep today's behavior.
#   5. Self-detection: A (also discovering) never lists its own shared
#      models as remote entries.
#
# SKIPs cleanly when the model dir is missing or the Mac has no non-loopback
# IP (discovery + gate need a routable interface).
#
# Usage: ./tests/test_lan_share.sh [/path/to/model] [portA]

MODEL="${1:-$HOME/.mlx-serve/models/mlx-community/gemma-4-e4b-it-4bit}"
PORT_A="${2:-11311}"
PORT_B=$((PORT_A + 1))
PEER_NAME="lantest-a-$$"
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'

if [ ! -d "$MODEL" ] && [ -d "$HOME/.mlx-serve/models/gemma-4-e4b-it-4bit" ]; then
    MODEL="$HOME/.mlx-serve/models/gemma-4-e4b-it-4bit"
fi
if [ ! -d "$MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} test_lan_share: $MODEL not found."
    exit 0
fi
BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
if [ ! -x "$BINARY" ]; then
    echo -e "${RED}FAIL${NC} $BINARY not found. Build first."
    exit 1
fi
LAN_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true)
if [ -z "$LAN_IP" ]; then
    echo -e "${YELLOW}SKIP${NC} test_lan_share: no non-loopback IP (offline Mac)."
    exit 0
fi

pkill -f "mlx-serve.*--port $PORT_A" 2>/dev/null || true
pkill -f "mlx-serve.*--port $PORT_B" 2>/dev/null || true
sleep 1

EMPTY_DIR=$(mktemp -d)
LOG_A=$(mktemp); LOG_B=$(mktemp)
PID_A=""; PID_B=""
cleanup() {
    [ -n "$PID_A" ] && kill "$PID_A" 2>/dev/null || true
    [ -n "$PID_B" ] && kill "$PID_B" 2>/dev/null || true
    wait 2>/dev/null || true
    rm -rf "$EMPTY_DIR" "$LOG_A" "$LOG_B"
}
trap cleanup EXIT

PASS=0; FAIL=0
ok()   { PASS=$((PASS+1)); echo -e "${GREEN}PASS${NC} $1"; }
bad()  { FAIL=$((FAIL+1)); echo -e "${RED}FAIL${NC} $1"; }
check() { # <desc> <cond-exit-code>
    if [ "$2" -eq 0 ]; then ok "$1"; else bad "$1"; fi
}

wait_health() { # <port>
    for _ in $(seq 1 120); do
        curl -s -m 2 "http://127.0.0.1:$1/health" | grep -q '"ok"' && return 0
        sleep 1
    done
    return 1
}

echo "── Booting A (shares '$(basename "$MODEL")' as \"$PEER_NAME\") + B (headless, discovering)"
"$BINARY" --model "$MODEL" --serve --port "$PORT_A" --log-level debug \
    --lan-share all --lan-name "$PEER_NAME" --lan-discover --log-file off >"$LOG_A" 2>&1 &
PID_A=$!
"$BINARY" --serve --port "$PORT_B" --model-dir "$EMPTY_DIR" --log-level debug \
    --lan-discover --log-file off >"$LOG_B" 2>&1 &
PID_B=$!
wait_health "$PORT_A" || { bad "server A never became healthy"; cat "$LOG_A" | tail -20; exit 1; }
wait_health "$PORT_B" || { bad "server B never became healthy"; cat "$LOG_B" | tail -20; exit 1; }
ok "both servers healthy"

# ── 1. Discovery: B lists A's model as <id>@peer ──
RID=""
for _ in $(seq 1 45); do
    RID=$(curl -s "http://127.0.0.1:$PORT_B/v1/models" | python3 -c '
import json,sys
try:
    d = json.load(sys.stdin)
    ids = [m["id"] for m in d.get("data", []) if m.get("id","").endswith("@'"$PEER_NAME"'")]
    print(ids[0] if ids else "")
except Exception:
    print("")')
    [ -n "$RID" ] && break
    sleep 1
done
if [ -n "$RID" ]; then
    ok "B discovered remote model: $RID"
else
    bad "B never discovered A's model (45s)"; tail -20 "$LOG_B"
fi

if [ -n "$RID" ]; then
    # Remote entry carries the lan_peer badge + peer metadata.
    ENTRY=$(curl -s "http://127.0.0.1:$PORT_B/v1/models")
    echo "$ENTRY" | grep -q "\"lan_peer\":\"$PEER_NAME\""
    check "remote entry carries lan_peer badge" $?

    # ── 2. Chat through B → proxied to A ──
    RESP=$(curl -s -m 120 "http://127.0.0.1:$PORT_B/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$RID\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly one word: pong\"}],\"max_tokens\":30,\"temperature\":0}")
    echo "$RESP" | python3 -c '
import json,sys
d = json.load(sys.stdin)
c = d["choices"][0]["message"]["content"]
assert c and c.strip(), "empty content"
assert d.get("model"), "no model echoed"' 2>/dev/null
    check "non-stream chat through B returns a completion" $?
    grep -q "\[lan\] proxy POST /v1/chat/completions" "$LOG_B"
    check "B logged the tunnel engagement" $?

    STREAM=$(curl -s -m 120 -N "http://127.0.0.1:$PORT_B/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$RID\",\"messages\":[{\"role\":\"user\",\"content\":\"Count to three.\"}],\"max_tokens\":40,\"temperature\":0,\"stream\":true}")
    echo "$STREAM" | grep -q "data: \[DONE\]"
    check "streaming chat through B passes SSE + [DONE] through" $?

    # ── 3. load/unload no-ops on the remote id ──
    LOAD=$(curl -s -m 10 "http://127.0.0.1:$PORT_B/v1/load-model" \
        -H 'Content-Type: application/json' -d "{\"model\":\"$RID\"}")
    echo "$LOAD" | grep -q "\"model\""
    check "load-model on a remote id is a 200 no-op with the entry" $?
    UNLOAD_CODE=$(curl -s -o /dev/null -w "%{http_code}" -m 10 "http://127.0.0.1:$PORT_B/v1/unload-model" \
        -H 'Content-Type: application/json' -d "{\"model\":\"$RID\"}")
    [ "$UNLOAD_CODE" = "200" ]
    check "unload-model on a remote id is a 200 no-op" $?
fi

# ── 4. Share gate on A (requests via the LAN IP are non-loopback) ──
CODE=$(curl -s -o /dev/null -w "%{http_code}" -m 5 "http://$LAN_IP:$PORT_A/metrics")
[ "$CODE" = "403" ]; check "LAN client: /metrics is host-local (403, got $CODE)" $?
CODE=$(curl -s -o /dev/null -w "%{http_code}" -m 5 "http://$LAN_IP:$PORT_A/")
[ "$CODE" = "403" ]; check "LAN client: status page is host-local (403, got $CODE)" $?
CODE=$(curl -s -o /dev/null -w "%{http_code}" -m 5 "http://$LAN_IP:$PORT_A/v1/load-model" \
    -H 'Content-Type: application/json' -d '{"model":"x"}')
[ "$CODE" = "403" ]; check "LAN client: /v1/load-model is host-local (403, got $CODE)" $?
CODE=$(curl -s -o /dev/null -w "%{http_code}" -m 5 "http://$LAN_IP:$PORT_A/v1/chat/completions" \
    -H 'Content-Type: application/json' -d '{"model":"x@nowhere","messages":[]}')
[ "$CODE" = "403" ]; check "LAN client: @peer ids are denied (no multi-hop; 403, got $CODE)" $?
CODE=$(curl -s -o /dev/null -w "%{http_code}" -m 5 "http://$LAN_IP:$PORT_A/health")
[ "$CODE" = "200" ]; check "LAN client: /health stays open (200, got $CODE)" $?
curl -s -m 5 "http://$LAN_IP:$PORT_A/v1/models" | grep -q '"data"'
check "LAN client: /v1/models serves the shared list" $?
# Loopback keeps full local behavior (metrics off => 503, never 403).
CODE=$(curl -s -o /dev/null -w "%{http_code}" -m 5 "http://127.0.0.1:$PORT_A/metrics")
[ "$CODE" = "503" ]; check "loopback client: /metrics untouched by the gate (503, got $CODE)" $?

# ── 5. Self-detection: A never mirrors its own advertisement ──
curl -s "http://127.0.0.1:$PORT_A/v1/models" | grep -q "@$PEER_NAME"
if [ $? -ne 0 ]; then ok "A does not list its own shared models as remote"; else bad "A mirrored itself (@$PEER_NAME in its own list)"; fi

echo ""
echo "── test_lan_share: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
