#!/usr/bin/env bash
# Run evaluation on metric learning checkpoints and report all metrics including HOTA
#
# Usage:
#   ./run_ml_eval.sh               # evaluate ml_r50_full (full-finetune)
#   ./run_ml_eval.sh ml_r50_full   # explicit checkpoint dir name
#   ./run_ml_eval.sh ml_swin_full  # evaluate swin variant

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/src"

# Determine which checkpoint to evaluate
if [[ $# -eq 0 ]]; then
    CKPT_DIR="ml_r50_full"
else
    CKPT_DIR="$1"
fi

OUT_ROOT="/home/astasy/cv_mid/fixes/cv-midterm/out"
CKPT_PATH="$OUT_ROOT/$CKPT_DIR/finetuned.pth"
METRICS_PATH="$OUT_ROOT/$CKPT_DIR/metrics.txt"

if [[ ! -f "$CKPT_PATH" ]]; then
    echo "Error: checkpoint not found at $CKPT_PATH" >&2
    exit 1
fi

echo "=========================================================="
echo "Metric Learning Evaluation with HOTA"
echo "=========================================================="
echo "Checkpoint: $CKPT_PATH"
echo "Metrics output: $METRICS_PATH"
echo

# Map checkpoint dir to config name
case "$CKPT_DIR" in
    ml_r50_full|ml-r50-full-finetune)
        CONFIG_NAME="metric-learning/ml-r50-full-finetune"
        ;;
    ml_r50_projector|ml-r50-projector-only)
        CONFIG_NAME="metric-learning/ml-r50-projector-only"
        ;;
    ml_swin_full|ml-swin-full-finetune)
        CONFIG_NAME="metric-learning/ml-swin-full-finetune"
        ;;
    ml_swin_projector|ml-swin-projector-only)
        CONFIG_NAME="metric-learning/ml-swin-projector-only"
        ;;
    *)
        echo "Unknown checkpoint dir: $CKPT_DIR" >&2
        exit 1
        ;;
esac

echo "Config: $CONFIG_NAME"
echo "=========================================================="
echo

# Run evaluation with steps=0 to skip finetuning, just eval the checkpoint
python eval.py --config-name="$CONFIG_NAME" \
    mode=test \
    testing.steps=0 \
    evaluation.run_mot=true \
    evaluation.verbose=true

echo
echo "=========================================================="
echo "METRICS SUMMARY"
echo "=========================================================="

if [[ -f "$METRICS_PATH.summary.csv" ]]; then
    echo
    echo "Per-sequence metrics:"
    cat "$METRICS_PATH.summary.csv"
else
    echo "Warning: summary metrics file not found at $METRICS_PATH.summary.csv"
fi

echo
echo "=========================================================="
echo "Evaluation complete!"
echo "Metrics available at: $OUT_ROOT/$CKPT_DIR/"
echo "=========================================================="
