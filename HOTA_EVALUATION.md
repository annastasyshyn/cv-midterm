# HOTA Metric Evaluation Guide

## Summary of Changes

Added HOTA (Higher Order Tracking Accuracy) metric calculation to the evaluation pipeline. The metric is now computed alongside existing MOT metrics (MOTA, MOTP, IDF1, etc.).

### Files Modified

1. **src/eval.py**
   - Added import for custom HOTA implementation
   - Updated `_run_final_mot_multi()` to compute HOTA for each sequence
   - HOTA scores are added to the evaluation summary

2. **src/roi_bytetrack.py**
   - Added import for custom HOTA implementation
   - Updated `evaluate_mot()` to compute HOTA and display additional metrics (DetA, AssA, LocA)

### Files Created

1. **src/hota_metric.py**
   - Custom HOTA metric implementation
   - Computes HOTA, DetA (Detection Accuracy), AssA (Association Accuracy), and LocA (Localization Accuracy)
   - Uses Hungarian algorithm for optimal detection-to-ground-truth matching
   - Returns component metrics for detailed analysis

2. **run_ml_eval.sh**
   - Convenience script to run evaluation on metric learning checkpoints
   - Automatically maps checkpoint directory names to config files
   - Usage: `./run_ml_eval.sh ml_r50_full` or `./run_ml_eval.sh ml_swin_full`

3. **eval_ml_checkpoint.sh**
   - Quick evaluation script for the ResNet50 full-finetune checkpoint

## HOTA Metric Explanation

**HOTA** combines two aspects of tracking accuracy:

- **DetA (Detection Accuracy)**: What percentage of ground truth objects were detected?
- **AssA (Association Accuracy)**: Of the detections, how many had correct ID assignments?
- **LocA (Localization Accuracy)**: How good is the localization of matched detections (average IoU)?
- **HOTA** = √(DetA × AssA) — the geometric mean of detection and association accuracy

Range: 0-1 (higher is better)

## Running Evaluation

### Option 1: Using the helper script (recommended)

```bash
cd /home/astasy/cv_mid/fixes/cv-midterm

# Evaluate ResNet50 full-finetune (metric learning ablation)
./run_ml_eval.sh ml_r50_full

# Evaluate other variants
./run_ml_eval.sh ml_r50_projector
./run_ml_eval.sh ml_swin_full
./run_ml_eval.sh ml_swin_projector
```

### Option 2: Direct Python command

```bash
cd /home/astasy/cv_mid/fixes/cv-midterm/src

# Evaluate with metric learning ResNet50 full-finetune config
python eval.py --config-name="metric-learning/ml-r50-full-finetune" \
    mode=test \
    testing.steps=0 \
    evaluation.run_mot=true \
    evaluation.verbose=true
```

## Output

The evaluation generates:
- **metrics.txt.summary.csv** — Per-sequence and overall metrics in CSV format
- **Console output** — Detailed metrics display including HOTA, DetA, AssA, LocA

Example output:
```
--- MOT Evaluation Results (per-seq + OVERALL) ---
[table with mota, motp, idf1, mostly_tracked, mostly_lost, num_switches, hota]

--- HOTA Scores ---
uav0000009_03358_v: 0.4231
uav0000073_00600_v: 0.3987
...
OVERALL: 0.4124
```

## Metric Learning Checkpoints Available

The following metric learning checkpoints are available in `/home/astasy/cv_mid/fixes/cv-midterm/out/`:

1. **ml_r50_full** — ResNet50 with full backbone finetuning
   - Config: `configs/metric-learning/ml-r50-full-finetune.yaml`
   - Checkpoint: `out/ml_r50_full/finetuned.pth`

2. **ml_r50_projector** — ResNet50 with frozen backbone (projector-only finetuning)
   - Config: `configs/metric-learning/ml-r50-projector-only.yaml`
   - Checkpoint: `out/ml_r50_projector/finetuned.pth`

3. **ml_swin_full** — Swin-T with full backbone finetuning
   - Config: `configs/metric-learning/ml-swin-full-finetune.yaml`
   - Checkpoint: `out/ml_swin_full/finetuned.pth`

4. **ml_swin_projector** — Swin-T with frozen backbone (projector-only finetuning)
   - Config: `configs/metric-learning/ml-swin-projector-only.yaml`
   - Checkpoint: `out/ml_swin_projector/finetuned.pth`

## Performance Expectations

The ResNet50 full-finetune variant (`ml_r50_full`) is the primary checkpoint for evaluation, trained with:
- Full backbone finetuning (freeze_backbone=false)
- 2000 training steps during metric learning pretraining
- 2000 finetuning steps on VisDrone validation set
- Cosine annealing scheduler with warmup
- Triplet loss with hard negative mining

Expected HOTA range: 0.35-0.45 (typical for MOT evaluation with this architecture)
