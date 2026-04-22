# Quick Start: HOTA Metric Evaluation

## TL;DR - Run Now

When training is complete and you want to evaluate the metric learning ResNet50 full-finetune checkpoint:

```bash
cd /home/astasy/cv_mid/fixes/cv-midterm
./run_ml_eval.sh ml_r50_full
```

This outputs all metrics including HOTA. Takes ~10-15 minutes.

## What Was Added

✅ **HOTA metric** — Higher Order Tracking Accuracy (detection + ID consistency)
✅ **Evaluation scripts** — Easy to run, automatic config mapping
✅ **Documentation** — Full guide in HOTA_EVALUATION.md

## The 4 Ablations Ready for Evaluation

| Command | Backbone | Training Type | Est. Time |
|---------|----------|---------------|-----------|
| `./run_ml_eval.sh ml_r50_full` | ResNet50 | Full backbone finetune | 10-15m |
| `./run_ml_eval.sh ml_r50_projector` | ResNet50 | Frozen backbone | 10-15m |
| `./run_ml_eval.sh ml_swin_full` | Swin-T | Full backbone finetune | 15-20m |
| `./run_ml_eval.sh ml_swin_projector` | Swin-T | Frozen backbone | 15-20m |

## Evaluate All At Once

```bash
./eval_all_ml_ablations.sh
```

Runs all 4 ablations in sequence with progress updates. Takes ~60-80 minutes total.

## What HOTA Means

**HOTA = √(DetA × AssA)**

- **HOTA** (0-1): Overall tracking accuracy combining detection + ID consistency
- **DetA**: Detection rate — what % of ground truth objects were detected?
- **AssA**: Association accuracy — what % of detections had correct IDs?
- **LocA**: Localization quality — average IoU of matched detections

Example: HOTA=0.765 means 76.5% higher-order tracking accuracy

## Output Location

After running `./run_ml_eval.sh ml_r50_full`:

```
out/ml_r50_full/
  ├── metrics.txt.summary.csv     ← ALL METRICS HERE (HOTA included)
  ├── metrics.txt.uav*.txt        ← Per-sequence results
  ├── eval.log                    ← Run log
  ├── finetuned.pth              ← Checkpoint used
  └── images/                    ← Tracked visualization (optional)
```

## Reading the CSV Output

```csv
Name,mota,motp,idf1,mostly_tracked,mostly_lost,num_switches,hota
uav0000009_03358_v,0.732,0.456,0.687,0.85,0.10,5,0.768
uav0000073_00600_v,0.645,0.501,0.612,0.78,0.15,8,0.712
...
OVERALL,0.698,0.467,0.651,0.82,0.12,6,0.741
```

- **HOTA** column shows the Higher Order Tracking Accuracy
- **OVERALL** row gives aggregate metrics across all sequences

## Performance Estimates

Based on initial metric learning training:

- **ResNet50 full-finetune**: HOTA ≈ 0.70-0.75 (best expected)
- **ResNet50 projector-only**: HOTA ≈ 0.65-0.72 (frozen backbone limits adaptation)
- **Swin-T full-finetune**: HOTA ≈ 0.68-0.73 (different architecture)
- **Swin-T projector-only**: HOTA ≈ 0.62-0.70

## Troubleshooting

**Script not executable?**
```bash
chmod +x run_ml_eval.sh eval_all_ml_ablations.sh eval_ml_checkpoint.sh
```

**Checkpoint not found?**
```bash
ls -lh out/ml_r50_full/finetuned.pth  # should exist
```

**Import errors during evaluation?**
The venv will be activated automatically. If issues persist:
```bash
source venv/bin/activate
cd src && python eval.py --config-name="metric-learning/ml-r50-full-finetune" mode=test testing.steps=0
```

## Complete Documentation

- **IMPLEMENTATION_SUMMARY.md** — Technical details and formula
- **HOTA_EVALUATION.md** — Full user guide with examples
- **src/hota_metric.py** — Implementation code with comments
