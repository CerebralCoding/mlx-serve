#!/bin/bash
# Focused spec-decode benchmark: measures decode tok/s for (none/PLD/drafter) ×
# (heavy-echo / creative-novel) on the models we ship support for. Drives the
# default-on flip decision for `--pld` and `--drafter`.
#
# Usage:
#   ./tests/bench_spec.sh [runs]            # ungated (explicit enable_*:true in body — bypasses adaptive gate)
#   ./tests/bench_spec.sh [runs] --gated    # gated (no explicit override — adaptive gate decides per-prompt)
#   ./tests/bench_spec.sh --corpus          # threshold-tuning corpus (8-10 prompts × pld on/off, prints gate analysis)
#
# The first two modes together quantify the gate's effect: gated should match
# ungated on heavy-echo (gate keeps spec on) and match `none` on creative
# (gate disables spec). Run 1 is warmup, dropped from stats.
#
# Corpus mode runs a wider prompt set (code review, RAG QA, agent multi-turn,
# Q&A, JSON transform, code translation, summarization, creative) on a single
# model — it prints per-prompt n-gram score, decode tok/s with PLD on (gated)
# vs PLD off, and a recommendation about whether the current
# `spec_gate_threshold` is well-calibrated.
#
# Output: pipe-separated rows to stdout; per-run debug to stderr.
set -uo pipefail

# Default-init for clean -u behavior in subshells / sourced contexts.
RUNS="${1:-5}"
GATED=0
CORPUS=0
for arg in "$@"; do
    if [ "$arg" = "--gated" ]; then GATED=1; fi
    if [ "$arg" = "--corpus" ]; then CORPUS=1; fi
done

BIN="./zig-out/bin/mlx-serve"
PORT=8091
MODELS_DIR="$HOME/.mlx-serve/models"

HEAVY_ECHO='Repeat the following Python code verbatim, but rename the function `compute_total` to `total`:
```python
def compute_total(items):
    total = 0
    for item in items:
        total += item.price * item.quantity
    if total > 100:
        total *= 0.9
    return total
```
Output ONLY the renamed code, nothing else.'

CREATIVE='Write a 30-line poem about a lighthouse keeper at the end of the world. Use vivid imagery.'

start_server() {
    local model="$1" extra="$2"
    pkill -f "mlx-serve" 2>/dev/null; sleep 0.5
    eval "$BIN --model '$model' --serve --port $PORT --log-level info $extra" >/tmp/bench-srv.log 2>&1 &
    SRV=$!
    for i in $(seq 1 60); do
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then return 0; fi
        sleep 0.5
    done
    echo "  ERROR: server failed to start" >&2; return 1
}

stop_server() { kill ${SRV:-0} 2>/dev/null; wait ${SRV:-0} 2>/dev/null; }

run_cell() {
    local label="$1" model_path="$2" spec_args="$3" prompt_label="$4" prompt="$5"
    if [[ ! -d "$model_path" ]]; then echo "$label|$prompt_label|SKIP" ; return; fi
    if ! start_server "$model_path" "$spec_args"; then return 1; fi
    local toks_per_s_vals=()
    for r in $(seq 1 $RUNS); do
        : > /tmp/bench-srv.log
        # Build the request body. Default mode injects explicit `enable_pld:true`
        # / `enable_drafter:true` to bypass the adaptive gate (= measure raw
        # spec-decode speedup). `--gated` mode omits the explicit override so
        # the gate's per-prompt decision is what's measured.
        local body
        if [ "$GATED" = "1" ] || [[ "$label" == */none ]]; then
            body=$(jq -nc --arg p "$prompt" '{model:"x",messages:[{role:"user",content:$p}],max_tokens:120,temperature:0,stream:false}')
        else
            local opt='{}'
            if [[ "$label" == */pld ]]; then opt='{"enable_pld":true}'; fi
            if [[ "$label" == */drafter ]]; then opt='{"enable_drafter":true}'; fi
            body=$(jq -nc --arg p "$prompt" --argjson opt "$opt" '{model:"x",messages:[{role:"user",content:$p}],max_tokens:120,temperature:0,stream:false} + $opt')
        fi
        curl -sf -m 90 "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" -d "$body" -o /dev/null
        # Server log line:  "<- N+M tokens (Xms) [prefill: P tok/s, decode: D tok/s] [stop]"
        local tps=$(LC_ALL=C grep -aoE 'decode: [0-9.]+ tok/s' /tmp/bench-srv.log | tail -1 | LC_ALL=C grep -aoE '[0-9]+\.[0-9]+' | head -1)
        if [[ -z "$tps" ]]; then tps="0"; fi
        toks_per_s_vals+=("$tps")
        echo "  $label | $prompt_label | run=$r | tps=$tps" >&2
    done
    stop_server
    local timed=("${toks_per_s_vals[@]:1}")
    local timed_csv=$(IFS=,; echo "${timed[*]}")
    local stats=$(python3 -c "import statistics as s; v=[float(x) for x in '$timed_csv'.split(',') if x]; print(f'{s.mean(v):.2f}|{min(v):.2f}|{max(v):.2f}')" 2>/dev/null)
    if [[ -z "$stats" ]]; then stats="0|0|0"; fi
    echo "$label|$prompt_label|$stats"
}

# ── Corpus mode: threshold tuning ──
# 9 prompts spanning typical real-world workloads. For each prompt we run
# PLD off (baseline) and PLD on (gated — adaptive gate decides). The server's
# spec-gate log line gives us the n-gram score per prompt. We then compute
# per-prompt speedup/slowdown and a recommendation about whether the current
# threshold is well-calibrated.
if [ "$CORPUS" = "1" ]; then
    GEMMA="$MODELS_DIR/gemma-4-e4b-it-4bit"
    if [ ! -d "$GEMMA" ]; then
        echo "ERROR: corpus mode needs Gemma 4 E4B at $GEMMA" >&2
        exit 1
    fi

    # Prompts (label|text). Pick prompts that span repetition density: heavy
    # echo, structured-output, code translation, RAG-style retrieval-grounded
    # answer, conversational agent turn, JSON transform, summarization,
    # creative writing, plain Q&A.
    declare -a CORPUS_LABELS=(
        heavy-echo
        code-rename
        json-transform
        rag-qa
        agent-turn
        plain-qa
        code-translate
        summarize
        creative
    )
    declare -a CORPUS_PROMPTS=(
        "$HEAVY_ECHO"
        'Rename `getUserById` to `findUser` in this code, output only the renamed code:
function getUserById(id) {
    const user = database.users.find(u => u.id === id);
    if (!user) return null;
    return user;
}'
        'Convert this list of users to a JSON array, one object per user with keys "name" and "age":
- Alice, 30
- Bob, 25
- Charlie, 42
- Dana, 18
Output only the JSON, no explanation.'
        'Context: Apple announced the M5 chip in October 2026. The chip features 12 performance cores and 8 efficiency cores, fabricated on a 2nm process. It has a unified memory bandwidth of 800 GB/s and supports up to 256GB of LPDDR6 memory.

Question: How much memory bandwidth does the M5 chip have? Answer in one short sentence.'
        'You are a coding agent. The user said: "fix the typo in src/auth.zig". Reply with a one-line plan: which tool you would call first and what arguments you would pass to it.'
        'What is the capital of Australia? Answer in one sentence.'
        'Translate this Python function to TypeScript. Output only the TypeScript:
def is_palindrome(s: str) -> bool:
    s = s.lower().replace(" ", "")
    return s == s[::-1]'
        'Summarize the following text in exactly two sentences:

The 2026 Antarctic Treaty conference reaffirmed the moratorium on commercial mineral extraction in the polar region. Delegates from 54 nations agreed to extend the treaty for an additional 50 years, citing accelerating ice loss measurements gathered by the joint US-EU SCOTT-II satellite array launched the prior year. Several signatories pushed for a stricter biosphere protection clause, but consensus stopped short of a binding limit on tourist vessel traffic, which has tripled since 2020.'
        'Write a 30-line poem about a lighthouse keeper at the end of the world. Use vivid imagery.'
    )

    SRV=0
    pkill -f "mlx-serve" 2>/dev/null; sleep 0.5
    eval "$BIN --model '$GEMMA' --serve --port $PORT --pld --log-level info" >/tmp/bench-srv.log 2>&1 &
    SRV=$!
    for i in $(seq 1 60); do
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
        sleep 0.5
    done
    if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "ERROR: server failed to start" >&2; exit 1
    fi

    echo "# corpus mode: Gemma-4-E4B + --pld; gated (adaptive gate decides per-prompt)"
    echo "# threshold currently 0.01 (in src/server.zig spec_gate_threshold)"
    echo "prompt|ngram_score|gate_decision|baseline_tps|pld_tps|ratio"

    n_prompts=${#CORPUS_LABELS[@]}
    declare -a OUT_LINES=()
    for i in $(seq 0 $((n_prompts - 1))); do
        label="${CORPUS_LABELS[$i]}"
        prompt="${CORPUS_PROMPTS[$i]}"

        # Baseline: explicit enable_pld:false to bypass any gate decision and force off.
        : > /tmp/bench-srv.log
        body=$(jq -nc --arg p "$prompt" '{model:"x",messages:[{role:"user",content:$p}],max_tokens:120,temperature:0,stream:false,enable_pld:false}')
        curl -sf -m 90 "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" -d "$body" -o /dev/null
        baseline_tps=$(LC_ALL=C grep -aoE 'decode: [0-9.]+ tok/s' /tmp/bench-srv.log | tail -1 | LC_ALL=C grep -aoE '[0-9]+\.[0-9]+' | head -1)
        baseline_tps=${baseline_tps:-0}

        # PLD on (gated — no explicit override): the prompt-time gate decides.
        : > /tmp/bench-srv.log
        body=$(jq -nc --arg p "$prompt" '{model:"x",messages:[{role:"user",content:$p}],max_tokens:120,temperature:0,stream:false}')
        curl -sf -m 90 "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" -d "$body" -o /dev/null
        pld_tps=$(LC_ALL=C grep -aoE 'decode: [0-9.]+ tok/s' /tmp/bench-srv.log | tail -1 | LC_ALL=C grep -aoE '[0-9]+\.[0-9]+' | head -1)
        pld_tps=${pld_tps:-0}
        ngram=$(LC_ALL=C grep -aoE 'spec-gate: ngram-score=[0-9.]+' /tmp/bench-srv.log | tail -1 | LC_ALL=C grep -aoE '[0-9]+\.[0-9]+' | head -1)
        ngram=${ngram:-0}
        if LC_ALL=C grep -aq 'pld=disabled (ngram-score' /tmp/bench-srv.log; then
            gate=disabled
        else
            gate=enabled
        fi

        ratio=$(python3 -c "p=float('$pld_tps'); b=float('$baseline_tps'); print(f'{p/b:.2f}' if b>0 else 'N/A')")
        line=$(printf "%-15s|%6s|%-9s|%7s|%7s|%5s" "$label" "$ngram" "$gate" "$baseline_tps" "$pld_tps" "$ratio")
        echo "$line"
        OUT_LINES+=("${label}|${ngram}|${gate}|${baseline_tps}|${pld_tps}|${ratio}")
    done

    kill $SRV 2>/dev/null; wait $SRV 2>/dev/null || true

    # Threshold-recommendation analysis: identify which decisions were
    # "correct" (= speedup vs slowdown when gate left PLD on; parity when gate
    # disabled) and recommend a threshold. We have only 9 data points so this
    # is descriptive, not statistical.
    echo
    echo "# Analysis"
    csv=$(IFS=$'\n'; echo "${OUT_LINES[*]}")
    python3 - <<PY
import sys
csv = """$csv"""
rows=[]
for line in csv.strip().split("\\n"):
    p,sc,g,b,pld,r = [x.strip() for x in line.split("|")]
    sc = float(sc); b=float(b); pld=float(pld)
    r = pld/b if b>0 else 0.0
    rows.append((p, sc, g, b, pld, r))

THRESH = 0.01

# Classify: gate decision was "correct" if (gate=enabled AND ratio>=0.98) OR (gate=disabled AND ratio<=1.02 baseline parity).
correct, wrong = [], []
for p,sc,g,b,pld,r in rows:
    if g == 'enabled':
        if r >= 0.98:
            correct.append(p)
        else:
            wrong.append((p, 'kept-on but slow', r, sc))
    else:
        # Gate disabled — the "correct" check is harder: we'd need to also
        # know what PLD ratio WOULD have been with override on. We don't have
        # that here, so we just note it.
        correct.append(p + ' (disabled by gate)')

print(f"correct decisions: {len(correct)}/{len(rows)}")
for p in correct: print(f"  ✓ {p}")
print(f"wrong decisions:  {len(wrong)}/{len(rows)}")
for p, why, r, sc in wrong:
    print(f"  ✗ {p}: {why} (ratio={r:.2f}, score={sc:.3f})")

# Threshold sensitivity: what scores DO the wrong-decisions cluster around?
if wrong:
    bad_scores = [sc for _,_,_,sc in wrong]
    print(f"  wrong-decisions span scores [{min(bad_scores):.3f}, {max(bad_scores):.3f}]")
    print(f"  raising threshold to {max(bad_scores)+0.001:.3f} would gate them off")
else:
    print("  no wrong decisions — current threshold is well-calibrated for this corpus")
PY

    exit 0
fi

if [ "$GATED" = "1" ]; then
    echo "# bench mode: GATED (no explicit enable_*; adaptive gate decides per-prompt)"
else
    echo "# bench mode: ungated (explicit enable_*=true; adaptive gate bypassed)"
fi
echo "label|prompt|mean_tps|min_tps|max_tps"

QWEN="$MODELS_DIR/Qwen3.5-4B-MLX-4bit"
run_cell "Qwen3.5-4B/none"    "$QWEN" ""        heavy-echo  "$HEAVY_ECHO"
run_cell "Qwen3.5-4B/pld"     "$QWEN" "--pld"   heavy-echo  "$HEAVY_ECHO"
run_cell "Qwen3.5-4B/none"    "$QWEN" ""        creative    "$CREATIVE"
run_cell "Qwen3.5-4B/pld"     "$QWEN" "--pld"   creative    "$CREATIVE"

GEMMA="$MODELS_DIR/gemma-4-e4b-it-4bit"
DRAFTER="$MODELS_DIR/gemma-4-E4B-it-assistant-bf16"
run_cell "Gemma-4-E4B/none"     "$GEMMA" ""                        heavy-echo  "$HEAVY_ECHO"
run_cell "Gemma-4-E4B/pld"      "$GEMMA" "--pld"                   heavy-echo  "$HEAVY_ECHO"
run_cell "Gemma-4-E4B/drafter"  "$GEMMA" "--drafter $DRAFTER"      heavy-echo  "$HEAVY_ECHO"
run_cell "Gemma-4-E4B/none"     "$GEMMA" ""                        creative    "$CREATIVE"
run_cell "Gemma-4-E4B/pld"      "$GEMMA" "--pld"                   creative    "$CREATIVE"
run_cell "Gemma-4-E4B/drafter"  "$GEMMA" "--drafter $DRAFTER"      creative    "$CREATIVE"

LFM="$MODELS_DIR/LFM2.5-350M-MLX-8bit"
run_cell "LFM2.5-350M/none"     "$LFM" ""        heavy-echo  "$HEAVY_ECHO"
run_cell "LFM2.5-350M/pld"      "$LFM" "--pld"   heavy-echo  "$HEAVY_ECHO"
run_cell "LFM2.5-350M/none"     "$LFM" ""        creative    "$CREATIVE"
run_cell "LFM2.5-350M/pld"      "$LFM" "--pld"   creative    "$CREATIVE"
