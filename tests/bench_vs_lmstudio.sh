#!/bin/bash
# bench_vs_lmstudio.sh — apples-to-apples MLX-serve vs LM Studio benchmark
# matrix that produced the charts in docs/perf-vs-lmstudio*.png.
#
# Two model families are shipped:
#   --family gemma   Gemma 4 E2B (8bit), E4B (8bit), 31B (8bit), 26B-A4B-MoE
#                    (4bit). Compares LM Studio (MLX baseline + GGUF where
#                    available) vs MLX-serve {none, pld, drafter} where drafter
#                    uses the matching gemma-4-*-it-assistant-bf16 checkpoint.
#   --family qwen36  Qwen 3.6 27B, 35B-A3B. Both engines load the same
#                    standard mlx-community 4-bit MLX checkpoints (not the
#                    unsloth UD variants — UD weights are bigger on disk and
#                    less representative of what most users actually run).
#                    Compares LM Studio MLX vs MLX-serve {none, pld}, plus a
#                    GGUF baseline on the LMS side. Qwen has no Gemma-4-style
#                    drafter.
#
# Apples-to-apples controls (the same for every cell in a row):
#   - Context size: 4096 (--ctx-size on MLX-serve, --context-length on LMS).
#   - Sampling: temperature=0, top_p=1, max_tokens=128, stream=false.
#   - System prompt: none.
#   - Thinking: disabled. MLX-serve uses the `enable_thinking:false` body
#     field, which the Jinja chat template honors to render
#     `<think>\n\n</think>\n\n` before content. LM Studio silently ignores
#     `chat_template_kwargs.enable_thinking:false` on Qwen 3.6, so we use the
#     assistant-prefill workaround: send messages ending in an assistant
#     message containing the closed `<think></think>` block plus
#     `add_generation_prompt:false, continue_final_message:true`. Both paths
#     deliver identical pre-decode tokens to the model. Gemma 4 doesn't have
#     a default thinking mode, so the workaround is a no-op there.
#
# Output:
#   - CSV at $OUT (default docs/perf-vs-lmstudio-<family>.csv) with rows:
#     label|engine|model|spec|prompt|prefill_tps|decode_tps|prompt_toks|completion_toks|notes
#   - To generate the chart: python3 tests/plot_vs_lmstudio.py <csv> <png> [--family <family>]
#
# Requirements:
#   - LM Studio CLI (`lms`) installed; models pre-downloaded for the chosen family.
#   - mlx-serve binary built (default ./zig-out/bin/mlx-serve).
#   - jq, python3, curl on PATH.

set -uo pipefail

# ── Defaults ──
FAMILY=""
RUNS=2
MAX_TOKENS=128
CTX=4096
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
PNG_OUT=""
KEEP_CSV=""
SERVER_PORT=11250
LMS_PORT=1234

usage() {
    cat <<EOF
Usage: $0 --family <gemma|qwen36> [options]

Options:
  --family NAME        Model family: 'gemma' or 'qwen36' (required)
  --out PATH           Chart PNG output path. Default is timestamped:
                       docs/perf-vs-lmstudio-<family>-YYYYMMDD-HHMMSS.png
  --keep-csv PATH      Also retain the raw CSV at this path. By default the
                       CSV is written to a temp file and deleted on exit.
  --runs N             Repeats per cell (run 1 dropped as warmup; default: 2)
  --max-tokens N       Decode budget (default: 128)
  --ctx-size N         Context size for both engines (default: 4096)
  --binary PATH        mlx-serve binary (default: ./zig-out/bin/mlx-serve)
  -h, --help           This message

Examples:
  $0 --family gemma                # writes docs/perf-vs-lmstudio-gemma.png
  $0 --family qwen36 --runs 3      # Qwen 3.6 matrix with one extra repeat
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --family)     FAMILY="$2"; shift 2 ;;
        --out)        PNG_OUT="$2"; shift 2 ;;
        --keep-csv)   KEEP_CSV="$2"; shift 2 ;;
        --runs)       RUNS="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --ctx-size)   CTX="$2"; shift 2 ;;
        --binary)     BINARY="$2"; shift 2 ;;
        -h|--help)    usage; exit 0 ;;
        *)            echo "Unknown arg: $1" >&2; usage; exit 1 ;;
    esac
done

[[ -z "$FAMILY" ]] && { echo "Missing --family" >&2; usage; exit 1; }
[[ -x "$BINARY" ]] || { echo "Build mlx-serve first: zig build -Doptimize=ReleaseFast" >&2; exit 1; }

# Resolve repo root (this script lives in tests/, run from anywhere).
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Default output gets a timestamp suffix so consecutive runs don't clobber
# each other (handy when sweeping over runs / comparing tweaks). Override
# with --out PATH to pick an exact filename (e.g. one referenced by README).
TS="$(date +%Y%m%d-%H%M%S)"
[[ -z "$PNG_OUT" ]] && PNG_OUT="docs/perf-vs-lmstudio-${FAMILY}-${TS}.png"

# Raw CSV is an internal artifact of the run. Use a temp file unless the
# caller explicitly asked for it via --keep-csv.
OUT="$(mktemp -t bench_vs_lms.XXXXXX).csv"

# ── Family-specific cell definitions ──
#
# Each row: label_prefix|mlxserve_path|lms_baseline_key|lms_alt_key|drafter_dir
# - lms_baseline_key  → primary LMS baseline (UD MLX where applicable)
# - lms_alt_key       → secondary LMS variant (GGUF for Qwen, empty for Gemma)
# - drafter_dir       → Gemma 4 assistant drafter checkpoint, empty otherwise
declare -a TARGETS
case "$FAMILY" in
    gemma)
        MD="$HOME/.mlx-serve/models"
        DM="$MD/mlx-community"
        # 4-bit across the board (apples-to-apples MLX 4-bit vs GGUF Q4 vs
        # mlx-serve 4-bit). All four targets get both MLX baseline and GGUF alt.
        LMS_DIR="$HOME/.lmstudio/models"
        TARGETS=(
            "gemma4-e2b-4bit|$LMS_DIR/mlx-community/gemma-4-e2b-it-4bit|gemma-4-e2b-it@4bit|google/gemma-4-e2b|$DM/gemma-4-E2B-it-assistant-bf16"
            "gemma4-e4b-4bit|$LMS_DIR/mlx-community/gemma-4-e4b-it-4bit|gemma-4-e4b-it@4bit|google/gemma-4-e4b|$DM/gemma-4-E4B-it-assistant-bf16"
            "gemma4-31b-4bit|$LMS_DIR/mlx-community/gemma-4-31b-it-4bit|gemma-4-31b-it@4bit|google/gemma-4-31b|$DM/gemma-4-31B-it-assistant-bf16"
            "gemma4-26b-a4b-moe-4bit|$MD/gemma-4-26b-a4b-it-4bit|gemma-4-26b-a4b-it|google/gemma-4-26b-a4b|$DM/gemma-4-26B-A4B-it-assistant-bf16"
        )
        # Specs measured (per row): mlx-serve {none,pld,drafter} + lms_baseline + lms_alt (GGUF).
        # Order matters: mlx-serve runs first while the machine is coolest, LMS
        # runs last so any thermal throttling that builds up during the row
        # falls on the comparison engine, not on us. lms_alt rows skip silently
        # when the row has no GGUF key configured (31B, 26B-A4B currently).
        SPECS=("mlx-serve::none" "mlx-serve::pld" "mlx-serve::drafter" "lmstudio:lms_baseline:none" "lmstudio:lms_alt:none")
        # Workaround needed? (assistant <think></think> prefill on LMS)
        LMS_THINKING_WORKAROUND=0
        ;;
    qwen36)
        LMS_DIR="$HOME/.lmstudio/models"
        # Both engines load the same standard mlx-community 4-bit MLX weights
        # (not the unsloth UD variants — those are bigger on disk and we want
        # the more representative production checkpoint here).
        TARGETS=(
            "qwen36-27b|$LMS_DIR/mlx-community/Qwen3.6-27B-4bit|mlx-community/qwen3.6-27b|qwen/qwen3.6-27b|"
            "qwen36-35b-a3b|$LMS_DIR/mlx-community/Qwen3.6-35B-A3B-4bit|qwen3.6-35b-a3b@4bit|qwen/qwen3.6-35b-a3b|"
        )
        # Same ordering rule as gemma: mlx-serve first (cool machine),
        # LMS specs last so thermal throttling penalises the comparison.
        SPECS=("mlx-serve::none" "mlx-serve::pld" "lmstudio:lms_baseline:none" "lmstudio:lms_alt:none")
        LMS_THINKING_WORKAROUND=1
        ;;
    *)
        echo "Unknown family '$FAMILY' (try gemma or qwen36)" >&2
        exit 1
        ;;
esac

# ── Test prompts (kept identical to bench_run.sh's so cross-bench numbers compare) ──
PREFILL_PROMPT="Explain the following topics in extreme detail: $(python3 -c "print(', '.join([f'topic {i} about science and technology and its impact on human civilization throughout history' for i in range(1,50)]))")"
DECODE_PROMPT="Write a detailed essay about quantum computing"
ECHO_PROMPT='Repeat the following paragraph back to me word for word, exactly as written, with no additional commentary. Then add the single sentence "End of recitation." on a new line. PARAGRAPH: The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. How vexingly quick daft zebras jump. The five boxing wizards jump quickly. Jackdaws love my big sphinx of quartz. Bright vixens jump; dozy fowl quack. Sphinx of black quartz, judge my vow. Two driven jocks help fax my big quiz. Now repeat the paragraph above exactly:'
# Code-completion prompt: tests speculative-decode value on the workload the
# drafter was actually trained for. Output is fresh code so PLD's prompt-n-gram
# lookup mostly misses; whether drafter wins here is the question.
CODE_PROMPT='Implement the following Python functions. Output only the complete code in a single Python code block, no commentary.

def fibonacci(n: int) -> int:
    """Return the nth Fibonacci number using memoization."""

def gcd(a: int, b: int) -> int:
    """Return the greatest common divisor of a and b using Euclid'\''s algorithm."""

def reverse_string(s: str) -> str:
    """Return s reversed without using slicing."""

def count_vowels(s: str) -> int:
    """Return the number of vowels (a, e, i, o, u, case-insensitive) in s."""

def is_palindrome(s: str) -> bool:
    """Return True if s is a palindrome ignoring case and non-alphanumerics."""'

# ── Body builders ──
build_body_mlx() {
    local prompt="$1" spec="$2" mt="$3"
    local epld=false edrft=false
    [[ "$spec" == "pld" ]]     && epld=true
    [[ "$spec" == "drafter" ]] && edrft=true
    jq -nc --arg p "$prompt" --argjson mt "$mt" --argjson epld "$epld" --argjson edrft "$edrft" \
        '{model:"x", messages:[{role:"user",content:$p}], max_tokens:$mt, temperature:0.0, top_p:1.0, stream:false, enable_thinking:false, enable_pld:$epld, enable_drafter:$edrft}'
}

build_body_lms() {
    local prompt="$1" model="$2" mt="$3"
    if [[ "$LMS_THINKING_WORKAROUND" == "1" ]]; then
        # Assistant-prefill `<think>\n\n</think>\n\n` so model continues in
        # content mode. Required for Qwen 3.6 since LM Studio ignores
        # chat_template_kwargs.enable_thinking on this family.
        jq -nc --arg p "$prompt" --arg model "$model" --argjson mt "$mt" \
            '{model:$model, messages:[{role:"user",content:$p},{role:"assistant",content:"<think>\n\n</think>\n\n"}], max_tokens:$mt, temperature:0.0, top_p:1.0, stream:false, add_generation_prompt:false, continue_final_message:true}'
    else
        jq -nc --arg p "$prompt" --arg model "$model" --argjson mt "$mt" \
            '{model:$model, messages:[{role:"user",content:$p}], max_tokens:$mt, temperature:0.0, top_p:1.0, stream:false, chat_template_kwargs:{enable_thinking:false}}'
    fi
}

# ── HTTP helpers ──
salted() { echo "[run-$1-$RANDOM] $2"; }

send_one() {
    local engine="$1" body="$2"
    local port=$([[ "$engine" == "lmstudio" ]] && echo "$LMS_PORT" || echo "$SERVER_PORT")
    local t0 t1 resp
    t0=$(python3 -c 'import time;print(int(time.time()*1000))')
    resp=$(curl -sf -m 240 -X POST "http://127.0.0.1:$port/v1/chat/completions" \
        -H "Content-Type: application/json" -d "$body")
    t1=$(python3 -c 'import time;print(int(time.time()*1000))')
    if [[ -z "$resp" ]]; then echo "ERR|0|0|0"; return; fi
    python3 -c "
import json,sys
r=json.loads(sys.argv[1])
u=r.get('usage',{}) or {}
ctd=u.get('completion_tokens_details',{}) or {}
print(f\"{int(sys.argv[2])-int(sys.argv[3])}|{u.get('prompt_tokens',0)}|{u.get('completion_tokens',0)}|{ctd.get('reasoning_tokens',0)}\")
" "$resp" "$t1" "$t0"
}

bench_decode() {
    local engine="$1" model="$2" spec="$3" prompt="$4" mt="$5"
    local elapsed_csv="" last_pt=0 last_ct=0 leaked=0
    for i in $(seq 1 "$RUNS"); do
        local body
        if [[ "$engine" == "mlx-serve" ]]; then
            body=$(build_body_mlx "$(salted "$i" "$prompt")" "$spec" "$mt")
        else
            body=$(build_body_lms "$(salted "$i" "$prompt")" "$model" "$mt")
        fi
        IFS='|' read -r elapsed pt ct rt < <(send_one "$engine" "$body")
        last_pt="$pt"; last_ct="$ct"
        [[ "$rt" -gt 0 ]] && leaked=$((leaked + rt))
        if [[ "$i" -gt 1 && "$ct" -gt 0 ]]; then
            elapsed_csv+="${elapsed},"
        fi
    done
    elapsed_csv="${elapsed_csv%,}"
    if [[ -z "$elapsed_csv" ]]; then echo "0|$last_pt|$last_ct|$leaked"; return; fi
    python3 -c "
e=[float(x) for x in '$elapsed_csv'.split(',') if x]
ct=$last_ct
avg=sum(e)/len(e)
tps=ct/(avg/1000.0) if avg>0 and ct>0 else 0
print(f'{tps:.1f}|$last_pt|$last_ct|$leaked')"
}

bench_prefill() {
    local engine="$1" model="$2" spec="$3"
    local elapsed_csv="" last_pt=0
    for i in $(seq 1 "$RUNS"); do
        local body
        if [[ "$engine" == "mlx-serve" ]]; then
            body=$(build_body_mlx "$(salted "$i" "$PREFILL_PROMPT")" "$spec" 1)
        else
            body=$(build_body_lms "$(salted "$i" "$PREFILL_PROMPT")" "$model" 1)
        fi
        IFS='|' read -r elapsed pt ct rt < <(send_one "$engine" "$body")
        last_pt="$pt"
        if [[ "$i" -gt 1 && "$pt" -gt 0 ]]; then
            elapsed_csv+="${elapsed},"
        fi
    done
    elapsed_csv="${elapsed_csv%,}"
    if [[ -z "$elapsed_csv" ]]; then echo "0|$last_pt"; return; fi
    python3 -c "
e=[float(x) for x in '$elapsed_csv'.split(',') if x]
pt=$last_pt
avg=sum(e)/len(e)
tps=pt/(avg/1000.0) if avg>0 and pt>0 else 0
print(f'{tps:.1f}|$last_pt')"
}

# ── Engine lifecycle ──
start_engine() {
    local engine="$1" model_or_path="$2" spec="$3" drafter="$4"
    pkill -9 -x mlx-serve 2>/dev/null; sleep 2
    case "$engine" in
        mlx-serve)
            local extra=""
            case "$spec" in
                none)    extra="--no-pld" ;;
                pld)     extra="" ;;
                drafter) [[ -z "$drafter" ]] && { echo "  drafter spec missing --drafter dir" >&2; return 1; }
                         extra="--no-pld --drafter $drafter" ;;
            esac
            "$BINARY" --model "$model_or_path" --serve --port "$SERVER_PORT" \
                --ctx-size "$CTX" --log-level info $extra >/tmp/bench_vs_lms_engine.log 2>&1 &
            local pid=$!
            for i in $(seq 1 240); do
                curl -sf "http://127.0.0.1:$SERVER_PORT/health" >/dev/null 2>&1 && return 0
                sleep 0.5
                kill -0 "$pid" 2>/dev/null || { echo "  mlx-serve died" >&2; return 1; }
            done
            return 1
            ;;
        lmstudio)
            lms server start --port "$LMS_PORT" >/dev/null 2>&1
            for i in $(seq 1 60); do
                curl -sf "http://127.0.0.1:$LMS_PORT/v1/models" >/dev/null 2>&1 && break
                sleep 0.5
            done
            lms unload --all >/dev/null 2>&1
            # `lms load` is unreliable on some LM Studio releases (silently hangs
            # for many minutes). HTTP JIT-load via /v1/chat/completions is the
            # supported path: LM Studio loads the model on first request when
            # `Just-In-Time Model Loading` is enabled (default in 0.4.x). The
            # warmup curl with a long timeout serves as both load-trigger and
            # health probe.
            local warmup_body
            warmup_body=$(jq -nc --arg model "$model_or_path" '{model:$model,messages:[{role:"user",content:"hi"}],max_tokens:1,stream:false}')
            if ! curl -sf -m 600 -X POST "http://127.0.0.1:$LMS_PORT/v1/chat/completions" \
                -H "Content-Type: application/json" -d "$warmup_body" >/dev/null 2>&1; then
                echo "  lms HTTP JIT-load failed for $model_or_path (timed out at 600s)" >&2
                return 1
            fi
            ;;
    esac
}

# ── Output emission ──
echo "label|engine|model|spec|prompt|prefill_tps|decode_tps|prompt_toks|completion_toks|notes" > "$OUT"
emit() {
    local label="$1" engine="$2" model="$3" spec="$4" prompt_kind="$5" pf="$6" dc="$7" pt="$8" ct="$9" notes="${10}"
    echo "${label}|${engine}|${model}|${spec}|${prompt_kind}|${pf}|${dc}|${pt}|${ct}|${notes}" | tee -a "$OUT"
}

run_cell() {
    local label_prefix="$1" engine="$2" model_or_path="$3" spec="$4" drafter="$5" notes="$6"
    local label="${label_prefix}"
    echo "  >> $label / $engine / $spec" >&2
    if ! start_engine "$engine" "$model_or_path" "$spec" "$drafter"; then
        echo "  SKIP $label" >&2
        return
    fi

    local out tps pt ct rt cell_notes

    out=$(bench_prefill "$engine" "$model_or_path" "$spec")
    IFS='|' read -r tps pt <<<"$out"
    emit "$label" "$engine" "$model_or_path" "$spec" "prefill" "$tps" "0" "$pt" "1" "$notes"

    out=$(bench_decode "$engine" "$model_or_path" "$spec" "$DECODE_PROMPT" "$MAX_TOKENS")
    IFS='|' read -r tps pt ct rt <<<"$out"
    cell_notes="$notes"
    [[ "$rt" -gt 0 ]] && cell_notes="${cell_notes:+$cell_notes,}thinking_leaked=$rt"
    emit "$label" "$engine" "$model_or_path" "$spec" "decode" "0" "$tps" "$pt" "$ct" "$cell_notes"

    out=$(bench_decode "$engine" "$model_or_path" "$spec" "$ECHO_PROMPT" "$MAX_TOKENS")
    IFS='|' read -r tps pt ct rt <<<"$out"
    cell_notes="$notes"
    [[ "$rt" -gt 0 ]] && cell_notes="${cell_notes:+$cell_notes,}thinking_leaked=$rt"
    emit "$label" "$engine" "$model_or_path" "$spec" "echo" "0" "$tps" "$pt" "$ct" "$cell_notes"

    out=$(bench_decode "$engine" "$model_or_path" "$spec" "$CODE_PROMPT" "$MAX_TOKENS")
    IFS='|' read -r tps pt ct rt <<<"$out"
    cell_notes="$notes"
    [[ "$rt" -gt 0 ]] && cell_notes="${cell_notes:+$cell_notes,}thinking_leaked=$rt"
    emit "$label" "$engine" "$model_or_path" "$spec" "code" "0" "$tps" "$pt" "$ct" "$cell_notes"
}

cleanup() {
    pkill -9 -x mlx-serve 2>/dev/null
    lms unload --all >/dev/null 2>&1 || true
    # Remove the temp CSV unless the caller asked to keep it.
    if [[ -z "$KEEP_CSV" && -n "$OUT" && -f "$OUT" ]]; then
        rm -f "$OUT"
    fi
}
trap cleanup EXIT INT TERM

for row in "${TARGETS[@]}"; do
    IFS='|' read -r logical mlxserve_path lms_baseline lms_alt drafter <<<"$row"
    [[ -d "$mlxserve_path" ]] || { echo "SKIP missing $mlxserve_path" >&2; continue; }

    for spec_entry in "${SPECS[@]}"; do
        IFS=':' read -r engine variant spec <<<"$spec_entry"
        case "$engine|$variant" in
            "lmstudio|lms_baseline")
                run_cell "${logical}/lmstudio-baseline/${spec}"  "lmstudio"  "$lms_baseline"  "$spec" "" ""
                ;;
            "lmstudio|lms_alt")
                [[ -z "$lms_alt" ]] && continue
                run_cell "${logical}/lmstudio-alt/${spec}"       "lmstudio"  "$lms_alt"       "$spec" "" ""
                ;;
            "mlx-serve|"|"mlx-serve|*")
                # Skip drafter cell when no drafter dir is present (Qwen).
                if [[ "$spec" == "drafter" && ( -z "$drafter" || ! -d "$drafter" ) ]]; then
                    continue
                fi
                run_cell "${logical}/mlx-serve/${spec}"          "mlx-serve" "$mlxserve_path" "$spec" "$drafter" ""
                ;;
        esac
    done
done

pkill -9 -x mlx-serve 2>/dev/null
lms unload --all >/dev/null 2>&1 || true

# Render the chart (only artifact most users want).
mkdir -p "$(dirname "$PNG_OUT")"
python3 "$SCRIPT_DIR/plot_vs_lmstudio.py" "$OUT" "$PNG_OUT" --family "$FAMILY"

# Optionally retain the raw CSV; otherwise it's a tempfile and gets cleaned
# up by the EXIT trap below.
if [[ -n "$KEEP_CSV" ]]; then
    mkdir -p "$(dirname "$KEEP_CSV")"
    cp "$OUT" "$KEEP_CSV"
    echo "CSV retained at $KEEP_CSV"
fi

echo
echo "=== chart written to $PNG_OUT ==="
