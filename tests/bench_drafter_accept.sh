#!/bin/bash
# bench_drafter_accept.sh — Phase 1 instrumentation harness for the Gemma 4
# assistant-drafter speculative-decode investigation. Measures per-request
# acceptance rate alongside decode tok/s so we can decide between the three
# Phase 2/3 outcomes outlined in the plan:
#
#   A) accept rate is OK (~50-60%) but per-step cost too high
#   B) accept rate is low (< 30%) — drafter is producing wrong drafts on MoE
#   C) runtime gate is firing late
#
# Reference: vLLM PR #41745 (Google's Gemma 4 assistant drafter), reported
# acceptance rates around 60% on E2B/26B-A4B/31B at single-stream batch.
# Per-target block_size defaults: E2B=2, E4B=4, 26B-A4B=4, 31B=8.
#
# The harness:
#   - boots mlx-serve once per (target, spec) cell
#   - sends echo-heavy + creative (novel) prompts
#   - greps server log for `[spec-stats]` lines (added in Generator.logSpecStats)
#   - parses `decode: X tok/s` from the matching turn
#   - emits CSV on stdout: target,spec,prompt,block_size,attempts,accepts,avg_per_round,per_draft_pct,decode_tps,runtime_disabled
#
# Implementation note on log parsing: we don't try to write turn markers into
# the server's stderr file. The server holds its own fd at a non-O_APPEND
# offset, so external `>>` writes get overwritten by the next server log
# emission. Instead, we snapshot `wc -l` before each curl and only consider
# lines AFTER that snapshot when parsing the per-turn `[spec-stats]`/`decode`
# pair.
set -uo pipefail

MODELS_DIR="$HOME/.mlx-serve/models"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
PORT="${PORT:-11247}"
MAX_TOKENS="${MAX_TOKENS:-96}"
RUNS="${RUNS:-2}"  # min 2 — first run is warmup, dropped from accept-rate too
LOG=/tmp/bench_drafter_accept.log

# logical|target_path|drafter_path|block_size_override(empty=server default)
TARGETS=(
    "gemma4-e2b-8bit|$MODELS_DIR/gemma-4-e2b-it-8bit|$MODELS_DIR/mlx-community/gemma-4-E2B-it-assistant-bf16|"
    "gemma4-e4b-8bit|$MODELS_DIR/gemma-4-e4b-it-8bit|$MODELS_DIR/mlx-community/gemma-4-E4B-it-assistant-bf16|"
    "gemma4-26b-a4b-moe-4bit|$MODELS_DIR/gemma-4-26b-a4b-it-4bit|$MODELS_DIR/mlx-community/gemma-4-26B-A4B-it-assistant-bf16|"
    "gemma4-31b-8bit|$MODELS_DIR/gemma-4-31b-it-8bit|$MODELS_DIR/mlx-community/gemma-4-31B-it-assistant-bf16|"
    # Phase 1: also measure 31B at vLLM's recommended block_size=8 to confirm
    # it moves the needle in the direction the PR reports.
    "gemma4-31b-8bit-bs8|$MODELS_DIR/gemma-4-31b-it-8bit|$MODELS_DIR/mlx-community/gemma-4-31B-it-assistant-bf16|8"
    # Phase 1 sanity: also try 26B-A4B at smaller block_size in case
    # per-round overhead dominates on MoE.
    "gemma4-26b-a4b-moe-4bit-bs2|$MODELS_DIR/gemma-4-26b-a4b-it-4bit|$MODELS_DIR/mlx-community/gemma-4-26B-A4B-it-assistant-bf16|2"
)

ECHO_PROMPT='Repeat the following paragraph back to me word for word, exactly as written, with no additional commentary. Then add the single sentence "End of recitation." on a new line. PARAGRAPH: The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. How vexingly quick daft zebras jump. The five boxing wizards jump quickly. Jackdaws love my big sphinx of quartz. Bright vixens jump; dozy fowl quack. Sphinx of black quartz, judge my vow. Two driven jocks help fax my big quiz. Now repeat the paragraph above exactly:'
CREATIVE_PROMPT='Write a 30-line poem about a lighthouse keeper at the end of the world. Use vivid imagery.'

start_server() {
    local target="$1" drafter="$2" block_size="$3" spec="$4"
    pkill -f mlx-serve 2>/dev/null
    sleep 2
    local args=(--model "$target" --serve --port "$PORT" --log-level info)
    case "$spec" in
        none)    args+=(--no-pld) ;;
        drafter) args+=(--no-pld --drafter "$drafter")
                 [[ -n "$block_size" ]] && args+=(--draft-block-size "$block_size") ;;
        *) echo "bad spec: $spec" >&2; return 1 ;;
    esac
    : > "$LOG"
    "$BINARY" "${args[@]}" >>"$LOG" 2>&1 &
    local pid=$!
    for i in $(seq 1 240); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
        sleep 0.5
        kill -0 "$pid" 2>/dev/null || { echo "server died (see $LOG)" >&2; return 1; }
    done
    echo "server didn't start in 120s (see $LOG)" >&2
    return 1
}

stop_server() {
    pkill -f mlx-serve 2>/dev/null
    sleep 1
}

# Send one chat-completions request. Returns 0 on success, nonzero on curl error.
send_one() {
    local prompt="$1" enable_drafter="$2"
    # Salt the prompt so the kv-cache reuse doesn't inflate prefill.
    local salt
    salt="[run-$$-$RANDOM] "
    local body
    body=$(jq -nc --arg p "${salt}${prompt}" --argjson mt "$MAX_TOKENS" --argjson ed "$enable_drafter" \
        '{model:"x",messages:[{role:"user",content:$p}],max_tokens:$mt,temperature:0,stream:false,enable_drafter:$ed,enable_pld:false,enable_thinking:false}')
    curl -sf -m 240 -H "Content-Type: application/json" \
        -d "$body" "http://127.0.0.1:$PORT/v1/chat/completions" -o /dev/null
}

# Parse the LAST [spec-stats] line and the LAST `decode: X tok/s` from $1
# (where $1 is the slice of server log produced by THIS turn, written to a
# scratch file by the caller). Echo:
# "attempts|accepts|avg_per_round|per_draft_pct|decode_tps|runtime_disabled"
parse_turn() {
    local turn_file="$1"
    local stats decode
    stats=$(LC_ALL=C grep -aE '\[spec-stats\] mode=' "$turn_file" | tail -1)
    decode=$(LC_ALL=C grep -aoE 'decode: [0-9.]+ tok/s' "$turn_file" | tail -1 | grep -oE '[0-9.]+' | head -1)
    if [[ -z "$stats" ]]; then
        echo "0|0|0.00||${decode:-0}|n/a"
        return
    fi
    local attempts accepts avg pdp rd
    attempts=$(echo "$stats" | grep -oE 'attempts=[0-9]+' | grep -oE '[0-9]+')
    accepts=$(echo "$stats" | grep -oE 'accepts=[0-9]+' | grep -oE '[0-9]+')
    avg=$(echo "$stats" | grep -oE 'avg_per_round=[0-9.]+' | grep -oE '[0-9.]+')
    pdp=$(echo "$stats" | grep -oE 'per_draft_pct=[0-9.]+' | grep -oE '[0-9.]+')
    rd=$(echo "$stats" | grep -oE 'runtime_disabled=(true|false)' | cut -d= -f2)
    echo "${attempts:-0}|${accepts:-0}|${avg:-0.00}|${pdp:-}|${decode:-0}|${rd:-n/a}"
}

run_cell() {
    local logical="$1" target="$2" drafter="$3" block_size="$4" spec="$5"
    echo "  -> starting $logical/$spec" >&2
    if ! start_server "$target" "$drafter" "$block_size" "$spec"; then
        echo "  SKIP $logical/$spec — server failed" >&2
        return
    fi

    # Two prompts × $RUNS runs each. Run 1 is warmup; we only parse run $RUNS.
    for prompt_kind in echo creative; do
        local prompt
        case "$prompt_kind" in
            echo)     prompt="$ECHO_PROMPT" ;;
            creative) prompt="$CREATIVE_PROMPT" ;;
        esac
        for r in $(seq 1 "$RUNS"); do
            local enable_drafter
            [[ "$spec" == "drafter" ]] && enable_drafter=true || enable_drafter=false

            # Snapshot log size BEFORE the curl; sleep briefly after the
            # request so the server's log line for this request has flushed.
            local pre_lines
            pre_lines=$(wc -l < "$LOG" 2>/dev/null || echo 0)
            send_one "$prompt" "$enable_drafter" || {
                echo "  WARN $logical/$spec/$prompt_kind/run=$r curl failed" >&2
                continue
            }
            sleep 0.4

            # Last run only — emit the row.
            if [[ "$r" -lt "$RUNS" ]]; then continue; fi

            local turn_log="$LOG.turn"
            tail -n +"$((pre_lines + 1))" "$LOG" > "$turn_log" 2>/dev/null || : > "$turn_log"
            local parsed
            parsed=$(parse_turn "$turn_log")
            local effective_bs="$block_size"
            [[ -z "$effective_bs" ]] && effective_bs="server-default"
            echo "$logical|$spec|$prompt_kind|$effective_bs|$parsed"
        done
    done

    stop_server
}

trap 'stop_server; rm -f "$LOG.turn"' EXIT INT TERM

[[ -x "$BINARY" ]] || { echo "ERROR: binary not built: $BINARY" >&2; echo "Run: zig build -Doptimize=ReleaseFast" >&2; exit 1; }

echo "target|spec|prompt|block_size|attempts|accepts|avg_per_round|per_draft_pct|decode_tps|runtime_disabled"

for row in "${TARGETS[@]}"; do
    IFS='|' read -r logical target drafter block_size <<<"$row"
    [[ -d "$target"  ]] || { echo "SKIP missing target $target" >&2; continue; }
    [[ -d "$drafter" ]] || { echo "SKIP missing drafter $drafter" >&2; continue; }
    # baseline (none) — needed to compute speedup ratios when reading the CSV
    run_cell "$logical" "$target" "$drafter" "$block_size" none
    run_cell "$logical" "$target" "$drafter" "$block_size" drafter
done
