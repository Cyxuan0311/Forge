#!/usr/bin/env bash
#
# Unified CPU/GPU inference verification for Forge example scripts.
#
# Drives each interactive inference script non-interactively by feeding a
# simple "hi" prompt followed by /quit, then checks whether the model
# produced any output. Uses greedy decoding (temperature=0) so results are
# reproducible and easy to eyeball for correctness.
#
# Usage:
#   examples/verify_inference.sh                 # all scripts, CPU + GPU
#   examples/verify_inference.sh --cpu-only      # CPU only
#   examples/verify_inference.sh --gpu-only      # GPU (cuda) only
#   examples/verify_inference.sh qwen tinyllama  # only selected scripts
#   examples/verify_inference.sh --prompt "hello" --max-new-tokens 32
#
# Environment overrides:
#   PYTHON           Python interpreter        (default: python3)
#   PROMPT           Prompt text               (default: hi)
#   MAX_NEW_TOKENS   Tokens to generate        (default: 64)
#   TEMPERATURE      Sampling temperature      (default: 0 = greedy)
#   TIMEOUT          Per-run timeout seconds   (default: 600)
#
# Per-script model path overrides (optional; otherwise script defaults are used):
#   TINYLLAMA_MODEL, QWEN_MODEL, LLAMA3_MODEL, DEEPSEEK_MODEL,
#   MIMO_MODEL, GEMMA4_MODEL, MINICPMV_MODEL, QWEN3VL_MODEL, PHIMOE_MODEL

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)" || exit 1
cd "$PROJECT_DIR" || exit 1

# --- Configuration -----------------------------------------------------------
PYTHON="${PYTHON:-python3}"
PROMPT="${PROMPT:-hi}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
TEMPERATURE="${TEMPERATURE:-0}"
TIMEOUT="${TIMEOUT:-600}"

RUN_CPU=1
RUN_GPU=1
SELECTED=()

# Colors (disabled when not a TTY)
if [ -t 1 ]; then
    C_RESET=$'\033[0m'; C_GREEN=$'\033[32m'; C_RED=$'\033[31m'
    C_YELLOW=$'\033[33m'; C_BLUE=$'\033[34m'; C_BOLD=$'\033[1m'
else
    C_RESET=""; C_GREEN=""; C_RED=""; C_YELLOW=""; C_BLUE=""; C_BOLD=""
fi

# Test registry: "key|script_file|model_env_var|extra_args"
# key           - short name used for CLI selection
# script_file   - filename under examples/
# model_env_var - env var name that (if set) supplies --model-path
# extra_args    - additional args always passed to the script
TESTS=(
    "tinyllama|tinyllama_inference.py|TINYLLAMA_MODEL|"
    "qwen|qwen_inference.py|QWEN_MODEL|"
    "llama3|llama3_inference.py|LLAMA3_MODEL|"
    "deepseek|deepseek_r1_inference.py|DEEPSEEK_MODEL|"
    "mimo|mimo_inference.py|MIMO_MODEL|"
    "gemma4|gemma4_inference.py|GEMMA4_MODEL|"
    "minicpmv|minicpmv_cli_inference.py|MINICPMV_MODEL|"
    "qwen3vl|qwen3vl_inference.py|QWEN3VL_MODEL|"
    "phimoe|phi_mini_moe_inference.py|PHIMOE_MODEL|"
)

usage() {
    sed -n '3,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

# --- Argument parsing --------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        --cpu-only) RUN_GPU=0; shift ;;
        --gpu-only) RUN_CPU=0; shift ;;
        --prompt) PROMPT="$2"; shift 2 ;;
        --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        -h|--help) usage 0 ;;
        -*) echo "Unknown option: $1" >&2; usage 1 ;;
        *) SELECTED+=("$1"); shift ;;
    esac
done

# Build the ordered device list
DEVICES=()
[ "$RUN_CPU" -eq 1 ] && DEVICES+=("cpu")
[ "$RUN_GPU" -eq 1 ] && DEVICES+=("cuda")

# --- Helpers -----------------------------------------------------------------
# Returns 0 if $1 is in the SELECTED list (or SELECTED is empty = all)
is_selected() {
    [ "${#SELECTED[@]}" -eq 0 ] && return 0
    local key="$1" sel
    for sel in "${SELECTED[@]}"; do
        [ "$sel" = "$key" ] && return 0
    done
    return 1
}

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
declare -a SUMMARY=()

# run_one <key> <script> <device> <model_path> <extra_args>
run_one() {
    local key="$1" script="$2" device="$3" model_path="$4" extra="$5"
    local label="${key} [${device}]"

    # Device-specific args
    local dev_args="--device ${device}"
    if [ "$device" = "cpu" ]; then
        dev_args="${dev_args} --gpu-layers 0"
    fi

    local model_args=""
    if [ -n "$model_path" ]; then
        model_args="--model-path ${model_path}"
    fi

    printf '%s\n' "${C_BLUE}▶ ${label}${C_RESET}"

    local out
    # shellcheck disable=SC2086
    out="$(printf '%s\n/quit\n' "$PROMPT" | timeout "$TIMEOUT" \
        "$PYTHON" "examples/${script}" \
        $dev_args $model_args \
        --temperature "$TEMPERATURE" \
        --max-new-tokens "$MAX_NEW_TOKENS" $extra 2>&1)"
    local status=$?

    # Show the assistant portion of the output for quick eyeballing
    printf '%s\n' "$out" | sed 's/^/    /'

    # Classify result
    if [ "$status" -eq 124 ]; then
        printf '%s\n\n' "${C_RED}✗ FAIL (timeout after ${TIMEOUT}s)${C_RESET}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        SUMMARY+=("${C_RED}FAIL${C_RESET}  ${label} (timeout)")
        return
    fi

    if printf '%s' "$out" | grep -qiE "not found|no model file found"; then
        printf '%s\n\n' "${C_YELLOW}– SKIP (model file not found)${C_RESET}"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        SUMMARY+=("${C_YELLOW}SKIP${C_RESET}  ${label} (model not found)")
        return
    fi

    # A successful run prints a stats line like "[12 tokens, 0.34s, ...]"
    local ntokens
    ntokens="$(printf '%s' "$out" | grep -oE '\[[0-9]+ tokens,' | grep -oE '[0-9]+' | tail -n1)"

    if [ "$status" -eq 0 ] && [ -n "$ntokens" ] && [ "$ntokens" -gt 0 ]; then
        printf '%s\n\n' "${C_GREEN}✓ PASS (${ntokens} tokens)${C_RESET}"
        PASS_COUNT=$((PASS_COUNT + 1))
        SUMMARY+=("${C_GREEN}PASS${C_RESET}  ${label} (${ntokens} tokens)")
    else
        printf '%s\n\n' "${C_RED}✗ FAIL (exit=${status}, no output tokens)${C_RESET}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        SUMMARY+=("${C_RED}FAIL${C_RESET}  ${label} (exit=${status})")
    fi
}

# --- Main --------------------------------------------------------------------
printf '%s\n' "${C_BOLD}============================================================${C_RESET}"
printf '%s\n' "${C_BOLD}  Forge Inference Verification${C_RESET}"
printf '  prompt=%q  max_new_tokens=%s  temperature=%s\n' "$PROMPT" "$MAX_NEW_TOKENS" "$TEMPERATURE"
printf '  devices=%s\n' "${DEVICES[*]}"
printf '%s\n\n' "${C_BOLD}============================================================${C_RESET}"

for entry in "${TESTS[@]}"; do
    IFS='|' read -r key script model_env extra <<< "$entry"

    is_selected "$key" || continue

    # Resolve optional model path from env var
    model_path=""
    if [ -n "$model_env" ]; then
        model_path="${!model_env:-}"
    fi

    for device in "${DEVICES[@]}"; do
        run_one "$key" "$script" "$device" "$model_path" "$extra"
    done
done

# --- Summary -----------------------------------------------------------------
printf '%s\n' "${C_BOLD}============================================================${C_RESET}"
printf '%s\n' "${C_BOLD}  Summary${C_RESET}"
printf '%s\n' "${C_BOLD}============================================================${C_RESET}"
for line in "${SUMMARY[@]}"; do
    printf '  %s\n' "$line"
done
printf '\n  %sPASS: %d%s   %sFAIL: %d%s   %sSKIP: %d%s\n' \
    "$C_GREEN" "$PASS_COUNT" "$C_RESET" \
    "$C_RED" "$FAIL_COUNT" "$C_RESET" \
    "$C_YELLOW" "$SKIP_COUNT" "$C_RESET"

# Non-zero exit if any real failure occurred
[ "$FAIL_COUNT" -eq 0 ]