#!/usr/bin/env bash
# Evaluate metric learning ResNet50 full finetune checkpoint
# This runs evaluation on the test-dev set and reports all metrics including HOTA

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/src"

CHECKPOINT_DIR="/home/astasy/cv_mid/fixes/cv-midterm/out/ml_r50_full"
FINETUNED_CKPT="$CHECKPOINT_DIR/finetuned.pth"

if [[ ! -f "$FINETUNED_CKPT" ]]; then
    echo "Error: checkpoint not found: $FINETUNED_CKPT" >&2
    exit 1
fi

echo "=========================================================="
echo "Evaluating metric learning ResNet50 full-finetune"
echo "Checkpoint: $FINETUNED_CKPT"
echo "=========================================================="

python eval.py --config-name="metric-learning/ml-r50-full-finetune" \
    mode=test \
    reid.checkpoint_finetuned="$FINETUNED_CKPT" \
    evaluation.run_mot=true \
    testing.steps=0

echo
echo "=========================================================="
echo "Evaluation complete"
echo "Metrics summary: $CHECKPOINT_DIR/metrics.txt.summary.csv"
echo "=========================================================="
