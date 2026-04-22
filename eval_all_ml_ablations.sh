#!/usr/bin/env bash
# Evaluate all metric learning ablation checkpoints and compare HOTA scores
#
# This script runs evaluation on all 4 metric learning configs:
# - ml_r50_full: ResNet50 with full backbone finetune
# - ml_r50_projector: ResNet50 with frozen backbone (projector-only)
# - ml_swin_full: Swin-T with full backbone finetune
# - ml_swin_projector: Swin-T with frozen backbone (projector-only)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_ROOT="$SCRIPT_DIR/out"

ALL_ABLATIONS=(
    "ml_r50_full"
    "ml_r50_projector"
    "ml_swin_full"
    "ml_swin_projector"
)

echo "=========================================================="
echo "Metric Learning Ablation Study - HOTA Evaluation"
echo "=========================================================="
echo
echo "This will evaluate all 4 metric learning variants."
echo "Total sequences: 16 per variant"
echo "Estimated time: ~30-40 minutes"
echo
echo "=========================================================="

FAILED=()
RESULTS=()

START_TIME=$SECONDS

for ablation in "${ALL_ABLATIONS[@]}"; do
    ckpt_path="$OUT_ROOT/$ablation/finetuned.pth"

    if [[ ! -f "$ckpt_path" ]]; then
        echo "[$(date '+%H:%M:%S')] ⚠ Checkpoint not found: $ablation"
        FAILED+=("$ablation (checkpoint not found)")
        continue
    fi

    echo
    echo "=========================================================="
    echo "[$(date '+%H:%M:%S')] Starting evaluation: $ablation"
    echo "=========================================================="

    start=$SECONDS

    if $SCRIPT_DIR/run_ml_eval.sh "$ablation" > "$OUT_ROOT/$ablation/eval.log" 2>&1; then
        dur=$((SECONDS - start))
        echo "[$(date '+%H:%M:%S')] ✓ Completed: $ablation (${dur}s)"

        # Extract HOTA score from summary
        if [[ -f "$OUT_ROOT/$ablation/metrics.txt.summary.csv" ]]; then
            overall_hota=$(tail -1 "$OUT_ROOT/$ablation/metrics.txt.summary.csv" | cut -d',' -f7)
            RESULTS+=("$ablation: HOTA=$overall_hota")
        else
            RESULTS+=("$ablation: metrics not found")
        fi
    else
        dur=$((SECONDS - start))
        echo "[$(date '+%H:%M:%S')] ✗ Failed: $ablation (${dur}s)"
        FAILED+=("$ablation")
    fi
done

total=$((SECONDS - START_TIME))

echo
echo "=========================================================="
echo "EVALUATION SUMMARY"
echo "=========================================================="
echo "Total time: ${total}s ($(printf '%d:%02d' $((total / 60)) $((total % 60))))"
echo

echo "Results:"
for result in "${RESULTS[@]}"; do
    echo "  $result"
done

if (( ${#FAILED[@]} )); then
    echo
    echo "⚠ Failed evaluations:"
    printf '  - %s\n' "${FAILED[@]}"
    echo
    echo "Check logs at:"
    for failed in "${FAILED[@]}"; do
        ablation=$(echo "$failed" | cut -d' ' -f1)
        echo "  $OUT_ROOT/$ablation/eval.log"
    done
    exit 1
else
    echo
    echo "✓ All evaluations completed successfully!"
    echo
    echo "Summary files available at:"
    for ablation in "${ALL_ABLATIONS[@]}"; do
        echo "  $OUT_ROOT/$ablation/metrics.txt.summary.csv"
    done
fi
