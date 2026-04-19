# cv-midterm

DL CV course mid-term on Multi-Object Tracking: ByteTrack + ROI appearance
embeddings + supervised (triplet) / self-supervised (InfoNCE) metric learning,
evaluated with HOTA-family MOT metrics.

---

## 1. Data layout

Download VisDrone MOT (train, val, test-dev):
https://github.com/VisDrone/VisDrone-Dataset.git

The pipeline expects each split as:

```
<split-root>/
  sequences/<seq_name>/0000001.jpg, 0000002.jpg, ...
  annotations/<seq_name>.txt     # MOTChallenge format
```

All paths in the configs point at `/workspace/cv-midterm/...` (vast.ai default
mount). Edit the three `data:` roots in the YAMLs, or override on the CLI.

---

## 2. Running

Both pipelines have **two modes** controlled by top-level `mode:` in the config
(or overridden on the CLI):

| Mode    | What it does                                                           | Writes                                 |
| ------- | ---------------------------------------------------------------------- | -------------------------------------- |
| `train` | Pretrain the embedder on `data.train.root`                             | `reid.checkpoint_pretrain`             |
| `test`  | Load pretrain → finetune on `data.val.root` → MOT eval on `data.test.*`| `reid.checkpoint_finetuned`, plots    |

Runs produce a history JSON (`output.history_path`), which the visualizer
consumes to produce PNGs.

### Self-supervised (InfoNCE)

```bash
cd src

# 1. Pretrain the SSL embedder on VisDrone-train
python eval.py --config-name=self-supervised/self-supervised-starting mode=train

# 2. Finetune the projector on VisDrone-val, then final MOT eval on test-dev
python eval.py --config-name=self-supervised/self-supervised-starting mode=test

# 3. Plot the curves (one combined PNG)
python ../visualizations/training_visualizer.py \
    /workspace/cv-midterm/ssl_history_test.json \
    --savefig /workspace/cv-midterm/plots/ssl_test.png
```

### Metric learning (triplet)

```bash
cd src

python eval.py --config-name=metric-learning/metric-learning-starting mode=train
python eval.py --config-name=metric-learning/metric-learning-starting mode=test

python ../visualizations/training_visualizer.py \
    /workspace/cv-midterm/ml_history_test.json \
    --savefig /workspace/cv-midterm/plots/ml_test.png
```

### Baseline (pretrained ImageNet features, no training)

```bash
python eval.py --config-name=baseline    # flat-layout config, evaluates data.dataset
python eval.py --config-name=roi         # same + ROI appearance fusion
```

### CLI overrides (Hydra)

Every config field is overridable:

```bash
python eval.py --config-name=self-supervised/self-supervised-starting \
    mode=test \
    testing.steps=2000 \
    testing.lr=5e-5 \
    training.scheduler.type=step \
    training.scheduler.step_size=1500
```

---

## 3. Config — the non-trivial knobs

Everything here lives under the self-supervised / metric-learning configs.
Fields not mentioned (LRs, num_workers, batch sizes, tracking thresholds) are
standard and documented inline.

### 3.1 Data splits — `data.{train,val,test}`

The two modes use **three directories**, each with its own purpose:

```yaml
data:
  train:
    root: .../VisDrone2019-MOT-train      # SSL pretraining crops       (mode=train)
  val:
    root: .../VisDrone2019-MOT-val        # finetuning crops            (mode=test)
  test:
    mot_sequence:    .../test-dev/sequences/uavXXXXXXXX
    mot_annotations: .../test-dev/annotations/uavXXXXXXXX.txt
    mot_root:        null  # or .../test-dev
```

`test` has **two switchable options** for the final MOT eval:

- **Option A — full test-dev sweep (for the report)**
  Set `data.test.mot_root` to the test-dev root. Every sequence under
  `mot_root/sequences/` gets tracked; `motmetrics.compute_many` produces
  per-sequence + OVERALL metrics.
  → `mot_sequence` / `mot_annotations` are **ignored for the final eval** when
  `mot_root` is set (they're still used for cheap periodic checkpoints during
  finetuning).

- **Option B — single sequence (quick iteration / debugging)**
  Set `mot_sequence` + `mot_annotations`, leave **`mot_root: null`**.
  The final eval runs on that one sequence only.

**To run over all of test-dev: set `mot_root` to the test-dev directory.**
**To run on one sequence: set `mot_root: null`.**

### 3.2 Modes — `mode: train | test`

- `mode=train`: load ImageNet-pretrained ResNet50, train contrastive/triplet
  loss on `data.train.root` for `training.steps` iterations, save to
  `reid.checkpoint_pretrain`. No MOT eval.
- `mode=test`: load `reid.checkpoint_pretrain`, finetune on `data.val.root`
  for `testing.steps` iterations (set `testing.steps=0` to skip finetuning and
  just evaluate the pretrain weights), save to `reid.checkpoint_finetuned`,
  then run the final MOT eval on `data.test.*`.

`output.history_path` uses `${mode}` interpolation so the two runs don't
overwrite each other's curves:

```yaml
output:
  history_path: /workspace/cv-midterm/ssl_history_${mode}.json
  # → produces .../ssl_history_train.json  and  .../ssl_history_test.json
```

### 3.3 What "steps" means

A **step = one optimizer update on one batch**, NOT one epoch.

- **SSL**: each step pulls `batch_size` bboxes from the dataset, runs two
  augmented views through the embedder, and steps InfoNCE. With
  `batch_size=64` and 50k VisDrone-train bboxes, one epoch ≈ 780 steps.
- **Metric learning**: each step samples a **PK batch** (`n_ids` identities ×
  `k_per_id` crops) via `TripletDataset.sample_nk_batch`. `batch_size` in the
  config is kept for parity but **is not used** by the triplet sampler — the
  effective batch is `n_ids × k_per_id`.

`log_every` controls how often we log the running mean loss to the history.
`val_every` controls held-out loss cadence (see 3.6). `mot_eval_every`
triggers a full single-sequence MOT pass during training — **very expensive**,
default 0 (disabled).

### 3.4 Crops — `training.crop_size` / `testing.crop_size`

Every training sample is **one bounding box** from the annotations, cropped
out of its source frame. The raw crop can be any pixel size (20×10 for a far
pedestrian, 300×150 for a close car). We then:

1. Resize to `1.15 × crop_size`
2. `RandomResizedCrop` down to exactly `crop_size`
3. Flip / colour jitter / grayscale / blur / coarse dropout
4. Two independent augmentation passes → the two views for InfoNCE

So `crop_size: [256, 128]` (H×W) means every bbox ends up as a 256×128 tensor
before hitting the ResNet. The default 256×128 is the standard person-ReID
shape (Market-1501, MSMT17, MASA). For VisDrone's mix of cars/bikes/pedestrians
a squarer `[192, 192]` is worth an ablation.

Why do this at all? The reid branch's job is **appearance, not geometry**.
ByteTrack handles position/scale/motion (IoU + Kalman). The embedder must
produce the same descriptor for an object seen large and small, so we
normalise the input shape and let the contrastive loss force scale-invariance.

**What we throw away**: aspect ratio, absolute scale, and surrounding context.
Tiny bboxes (e.g. 10×10) become mostly interpolated noise — consider filtering
them out of `AugmentedInstanceDataset.build_index` if they hurt signal.

### 3.5 Finetuning — `testing.freeze_backbone`

Default `true`: only the projector head (SSL) or the embedding head (ML) gets
gradient updates during finetuning; the ResNet50 backbone is frozen. Flipping
this to `false` finetunes everything (≈ 25M params vs ≈ 4.5M), takes longer,
and tends to forget ImageNet features unless you also drop the LR further.

### 3.6 Held-out loss during finetuning — `testing.val_every`

In `mode=test` we also compute the InfoNCE/triplet loss on **held-out test
data** every `val_every` steps — the second curve on the overview plot. It
reuses the same dispatch as the final MOT eval:

- `data.test.mot_root` set → held-out loss on every test-dev sequence.
- `mot_root: null` + `mot_sequence` set → held-out loss on that one sequence.
- neither set → skipped.

`val_max_batches` caps how many batches are drawn per val pass (the full
test-dev crop set can be huge).

**Caveat**: using test-dev as the held-out signal during finetuning is fine
for observability (watching whether finetuning generalises) but it peeks at
the final test set. Don't use this signal for early stopping / checkpoint
selection unless you explicitly report that in the paper.

### 3.7 Scheduler — `training.scheduler` / `testing.scheduler`

```yaml
scheduler:
  type: cosine       # none | cosine | step
  warmup_steps: 500  # linear warmup from lr·0.01 → lr  (0 disables warmup)
  eta_min: 0.0       # cosine: lower bound LR
  step_size: 1000    # step: decay every N steps
  gamma: 0.1         # step: decay factor
```

Stepped **once per optimizer step** (per-iteration, not per-epoch). With
`warmup_steps > 0` the scheduler is a `SequentialLR(LinearLR, main)`. The
tqdm postfix shows the live LR so you can sanity-check the schedule.

### 3.8 Visualisation — `output.save_images`

When `true`, the tracker renders bboxes + track IDs onto every frame of the
eval sequence and writes annotated JPEGs to `output.images_tracked/<seq>/`.
This can produce **tens of GB** during a full test-dev sweep — default is
`false`. Periodic mid-training MOT eval ignores this flag and never writes
images.

```bash
# quick visual debugging on one sequence
python eval.py --config-name=self-supervised/self-supervised-starting \
    mode=test data.test.mot_root=null output.save_images=true
```

---

## 4. The plot

`TrainingVisualizer.savefig(path)` produces a **single combined figure**:

- Top panel: loss curve (pretraining or finetuning) with the held-out
  `val_loss` overlaid in orange-with-markers (when populated).
- Lower panels: MOT metric grid (MOTA, IDF1, MOTP, IDSW, MT, ML) — only shown
  if `testing.mot_eval_every > 0` produced checkpoints during training.

```bash
python visualizations/training_visualizer.py <history.json> \
    --savefig /workspace/cv-midterm/plots/overview.png          # one combined PNG
python visualizations/training_visualizer.py <history.json> \
    --save-dir /workspace/cv-midterm/plots                      # losses.png + mot_metrics.png
```

---

