#!/usr/bin/env bash
# Run every self-supervised ablation config in sequence.
#
# Each run writes its own history / metrics / finetuned checkpoint under
# /workspace/cv-midterm/out/<ablation>/ (see each config's `output:` block).
# A run that crashes does NOT stop the rest — failures are collected and
# printed at the end, and the script exits non-zero if any run failed.
#
# Usage:
#   ./run_ssl_ablation.sh                    # run all 6 configs
#   ./run_ssl_ablation.sh ssl-r50-*          # run only configs matching a glob
#
# Logs per config: /workspace/cv-midterm/out/<ablation>/run.log

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/src"

ALL_CONFIGS=(
    # ssl-r50-eval-only
    # ssl-r50-projector-only
    # ssl-r50-full-finetune
    # ssl-r50-temp-low
    ssl-swin-projector-only
    # ssl-swin-full-finetune
)

# If the user passed glob(s), filter ALL_CONFIGS by them; otherwise run all.
if [[ $# -gt 0 ]]; then
    CONFIGS=()
    for arg in "$@"; do
        for cfg in "${ALL_CONFIGS[@]}"; do
            # shellcheck disable=SC2053
            [[ $cfg == $arg ]] && CONFIGS+=("$cfg")
        done
    done
    if [[ ${#CONFIGS[@]} -eq 0 ]]; then
        echo "no configs matched: $*" >&2
        echo "available: ${ALL_CONFIGS[*]}" >&2
        exit 2
    fi
else
    CONFIGS=("${ALL_CONFIGS[@]}")
fi

OUT_ROOT=/workspace/cv-midterm/out
mkdir -p "$OUT_ROOT"

FAILED=()
START_ALL=$SECONDS

for cfg in "${CONFIGS[@]}"; do
    log_dir="$OUT_ROOT/$cfg"
    mkdir -p "$log_dir"
    log_file="$log_dir/run.log"

    echo
    echo "==========================================================="
    echo "[ablation] $cfg  →  $log_file"
    echo "==========================================================="
    start=$SECONDS

    if python eval.py --config-name="self-supervised/$cfg" mode=test 2>&1 | tee "$log_file"; then
        dur=$((SECONDS - start))
        echo "[ablation] done: $cfg (${dur}s)"
    else
        dur=$((SECONDS - start))
        echo "[ablation] FAILED: $cfg (${dur}s) — see $log_file"
        FAILED+=("$cfg")
    fi
done

total=$((SECONDS - START_ALL))
echo
echo "==========================================================="
echo "[ablation] total time: ${total}s, ran ${#CONFIGS[@]} configs"
echo "==========================================================="

if (( ${#FAILED[@]} )); then
    echo "[ablation] failed runs:"
    printf '  - %s\n' "${FAILED[@]}"
    exit 1
fi

echo "[ablation] all runs OK"
