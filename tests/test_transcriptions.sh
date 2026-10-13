#!/usr/bin/env bash
# POST /v1/audio/transcriptions on Qwen3-ASR (mlx-community's mlx-audio packs):
# discovered under a private --model-dir as a stub advertising "transcription",
# load by id, then every response_format, a forced language, a 48 kHz stereo
# WAV, two clips at once (each keeps its own words), the named 400s, and a
# --no-vision boot that refuses by name. Transcripts are checked by WORD RECALL
# against the script `say` spoke: an audio tower whose rows never reach the
# trunk answers "" (the model hears silence), which no other check can see.
#
# Skips without the pack or without macOS `say`. Download with:
#   hf download mlx-community/Qwen3-ASR-0.6B-8bit --local-dir ~/.mlx-serve/models/mlx-community/Qwen3-ASR-0.6B-8bit
#
# Usage: ASR_MODEL=<dir> BINARY=<mlx-serve> ./tests/test_transcriptions.sh [port]
set -uo pipefail
PORT="${1:-11443}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BINARY:-$ROOT/zig-out/bin/mlx-serve}"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }
ASR="${ASR_MODEL:-$HOME/.mlx-serve/models/mlx-community/Qwen3-ASR-0.6B-8bit}"
[ -f "$ASR/config.json" ] || { echo "SKIP: no Qwen3-ASR pack at $ASR (set ASR_MODEL)"; exit 0; }
command -v say >/dev/null || { echo "SKIP: needs macOS say to speak the test clips"; exit 0; }

TMP="$(mktemp -d)"
mkdir -p "$TMP/models/mlx-community" "$TMP/home"
ln -s "$ASR" "$TMP/models/mlx-community/$(basename "$ASR")"
ID="mlx-community/$(basename "$ASR")"
LOG="$TMP/server.log"
SRV=""
trap '[ -n "$SRV" ] && kill $SRV 2>/dev/null; rm -rf "$TMP"' EXIT
boot() { # extra flags...
  HOME="$TMP/home" "$BIN" --serve --model-dir "$TMP/models" --port "$PORT" --host 127.0.0.1 "$@" >"$LOG" 2>&1 &
  SRV=$!
  for _ in $(seq 1 60); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    kill -0 $SRV 2>/dev/null || { echo "FAIL: server did not start"; tail -5 "$LOG"; exit 1; }
    sleep 1
  done
  echo "FAIL: server not healthy"; exit 1
}
T=/v1/audio/transcriptions
url() { echo "http://127.0.0.1:$PORT$1"; }

SHORT="The quick brown fox jumps over the lazy dog. Speech recognition on Apple Silicon, served natively from Zig."
LONG="Speech recognition has come a long way. Twenty years ago, dictation software needed hours of training on a single voice, and still made mistakes on every other sentence. Today a model with less than one billion parameters can transcribe dozens of languages on a laptop, without sending a single byte of audio to the cloud. That is the promise of running models locally."
say -o "$TMP/short.wav" --data-format=LEI16@16000 "$SHORT"
say -o "$TMP/long48.wav" --data-format=LEI16@48000 --channels=2 "$LONG"

recall() { # spoken heard -> fraction of spoken words heard (case, punctuation and digits-vs-words aside)
  python3 - "$1" "$2" <<'EOF'
import re, sys
norm = lambda s: re.findall(r"[a-z]+", s.lower())
spoken, heard = norm(sys.argv[1]), set(norm(sys.argv[2]))
print(sum(w in heard for w in spoken) / len(spoken))
EOF
}
check_recall() { # label spoken heard
  local r; r=$(recall "$2" "$3")
  python3 -c "import sys; sys.exit(0 if $r >= 0.9 else 1)" || { echo "FAIL: $1 recall $r: '$3'"; exit 1; }
  echo "PASS: $1 (recall $r)"
}
json_field() { python3 -c "import sys,json; print(json.load(sys.stdin)$1)"; }

boot
# 1. Discovery: the stub already says what the pack does.
caps() { # state
  curl -s "$(url /v1/models)" | python3 -c "
import sys,json
m=[x for x in json.load(sys.stdin)['data'] if x['id']=='$ID']
assert m, 'not discovered'
assert m[0]['state']=='$1', m[0]['state']
c=m[0].get('capabilities',[])
assert 'transcription' in c and 'vision' not in c and 'chat' not in c, c
print(c)"
}
c=$(caps unloaded) || { echo "FAIL: stub not discovered with transcription"; exit 1; }
echo "PASS: stub advertises $c"

# 2. Every format; the first request cold-loads the model by id.
J=$(curl -s "$(url $T)" -F file=@"$TMP/short.wav" -F model="$ID")
TEXT=$(echo "$J" | json_field "['text']") || { echo "FAIL: json format: $J"; tail -20 "$LOG"; exit 1; }
check_recall "json" "$SHORT" "$TEXT"
c=$(caps ready) || { echo "FAIL: ready model lacks transcription"; exit 1; }
echo "PASS: ready model advertises $c"
P=$(curl -s "$(url $T)" -F file=@"$TMP/short.wav" -F model="$ID" -F response_format=text)
[ "$P" = "$TEXT" ] || { echo "FAIL: text format '$P' != json text '$TEXT'"; exit 1; }
echo "PASS: text format is the json text"
V=$(curl -s "$(url $T)" -F file=@"$TMP/short.wav" -F model="$ID" -F response_format=verbose_json -F 'timestamp_granularities[]=segment')
echo "$V" | python3 -c "
import sys,json
v=json.load(sys.stdin)
assert v['task']=='transcribe' and v['language']=='english', v
assert v['text']=='''$TEXT''' and 6.0 < v['duration'] < 8.0, v
s=v['segments']; assert len(s)==1 and s[0]['start']==0 and s[0]['end']==v['duration'], s
" || { echo "FAIL: verbose_json: $V"; exit 1; }
echo "PASS: verbose_json names the detected language and one segment"
F=$(curl -s "$(url $T)" -F file=@"$TMP/short.wav" -F model="$ID" -F language=en | json_field "['text']")
check_recall "language=en" "$SHORT" "$F"

# 3. 48 kHz stereo is mixed and resampled.
L=$(curl -s "$(url $T)" -F file=@"$TMP/long48.wav" -F model="$ID" | json_field "['text']")
check_recall "48 kHz stereo, 21 s" "$LONG" "$L"

# 4. Two clips at once: each answer holds its own words, not the other's.
curl -s "$(url $T)" -F file=@"$TMP/short.wav" -F model="$ID" -o "$TMP/a.json" &
A=$!
curl -s "$(url $T)" -F file=@"$TMP/long48.wav" -F model="$ID" -o "$TMP/b.json" &
B=$!
wait $A $B
check_recall "concurrent short" "$SHORT" "$(json_field "['text']" <"$TMP/a.json")"
check_recall "concurrent long" "$LONG" "$(json_field "['text']" <"$TMP/b.json")"
grep -q "fox" "$TMP/b.json" && { echo "FAIL: the long clip's answer holds the short clip's words"; exit 1; }
echo "PASS: concurrent answers stay apart"

# 5. Named refusals.
b400() { # label needle curl-args...
  local code
  code=$(curl -s -o "$TMP/err.txt" -w "%{http_code}" "$(url $T)" "${@:3}")
  [ "$code" = "400" ] || { echo "FAIL: $1 returned $code (want 400)"; cat "$TMP/err.txt"; exit 1; }
  grep -q "$2" "$TMP/err.txt" || { echo "FAIL: $1 400 does not name '$2'"; cat "$TMP/err.txt"; exit 1; }
  echo "PASS: $1 -> 400"
}
printf 'ID3\x04\x00 not a wav' >"$TMP/fake.mp3"
b400 "no file" "file" -F model="$ID"
b400 "not a WAV" "WAV" -F file=@"$TMP/fake.mp3" -F model="$ID"
b400 "srt" "srt" -F file=@"$TMP/short.wav" -F model="$ID" -F response_format=srt
b400 "stream" "stream" -F file=@"$TMP/short.wav" -F model="$ID" -F stream=true
b400 "unknown language" "language" -F file=@"$TMP/short.wav" -F model="$ID" -F language=xx
b400 "JSON body" "multipart" -H 'Content-Type: application/json' -d "{\"model\":\"$ID\"}"
grep -q "\[mlx\]" "$LOG" && { echo "FAIL: MLX error in the log"; grep "\[mlx\]" "$LOG" | head -3; exit 1; }

# 6. Without its tower the model cannot transcribe, and says why.
kill $SRV; wait $SRV 2>/dev/null
boot --no-vision
b400 "--no-vision" "Qwen3-ASR" -F file=@"$TMP/short.wav" -F model="$ID"
echo "ALL PASS"
