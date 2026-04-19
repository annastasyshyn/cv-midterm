import os

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    StepLR,
)
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from ultralyticsplus import YOLO

from roi_bytetrack import ROIByteTrack


# ---------- path helpers ----------------------------------------------------

def _resolve_test_paths(cfg: DictConfig):
    """
    Final MOT eval sequence. Prefers nested `data.test.mot_*` fields (used by
    the self-supervised / metric-learning configs) and falls back to the flat
    `data.dataset` / `data.annotations` layout for legacy baseline / roi configs.
    """
    seq = OmegaConf.select(cfg, "data.test.mot_sequence")
    ann = OmegaConf.select(cfg, "data.test.mot_annotations")
    if seq and ann:
        return seq, ann
    return cfg.data.dataset, cfg.data.annotations


def _coerce_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _load_state_dict(path, device):
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "model_state" in obj:
        return obj["model_state"]
    return obj


# ---------- scheduler -------------------------------------------------------

def _build_scheduler(optimizer, sched_cfg, total_steps):
    """
    Build an LR scheduler from a config block. Returns None if disabled.

    Supported `type`:
      - "none" (or missing)     → no scheduler
      - "cosine"                → CosineAnnealingLR(T_max=steps-warmup, eta_min=...)
      - "step"                  → StepLR(step_size=..., gamma=...)

    `warmup_steps > 0` prepends a LinearLR warmup (0.01*lr → lr) via SequentialLR.
    Schedulers are stepped once per optimizer step (per-iteration).
    """
    if sched_cfg is None:
        return None
    sched_type = str(OmegaConf.select(sched_cfg, "type") or "none").lower()
    if sched_type in ("none", "null", ""):
        return None

    warmup_steps = int(OmegaConf.select(sched_cfg, "warmup_steps") or 0)
    main_steps = max(1, total_steps - warmup_steps)

    if sched_type == "cosine":
        main = CosineAnnealingLR(
            optimizer,
            T_max=main_steps,
            eta_min=float(OmegaConf.select(sched_cfg, "eta_min") or 0.0),
        )
    elif sched_type == "step":
        main = StepLR(
            optimizer,
            step_size=int(OmegaConf.select(sched_cfg, "step_size") or 1000),
            gamma=float(OmegaConf.select(sched_cfg, "gamma") or 0.1),
        )
    else:
        raise ValueError(f"unknown scheduler.type: {sched_type!r}")

    if warmup_steps > 0:
        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(optimizer, [warmup, main], milestones=[warmup_steps])
    return main


# ---------- shared MOT ------------------------------------------------------

def _run_final_mot(cfg: DictConfig, det_model, reid_model, device):
    """
    Final MOT eval. If `data.test.mot_root` is set, iterates every sequence
    under that root and reports per-seq + overall metrics. Otherwise runs on
    the single `data.test.mot_*` sequence.
    """
    mot_root = OmegaConf.select(cfg, "data.test.mot_root")
    if mot_root:
        _run_final_mot_multi(cfg, det_model, reid_model, device, mot_root)
        return

    test_sequence, test_annotations = _resolve_test_paths(cfg)
    print(f"[main] test MOT seq (single): {test_sequence}")

    reid_model.eval()
    tracker = ROIByteTrack(model=det_model, reid_model=reid_model, device=device)
    tracker.process_tracking(
        test_sequence,
        cfg.output.images_tracked,
        cfg.output.metrics_path,
        use_roi=cfg.tracking.use_roi,
        roi_coef=cfg.tracking.roi_coef,
    )
    tracker.evaluate_mot(
        test_annotations,
        cfg.output.metrics_path,
        verbose=cfg.evaluation.verbose,
    )


def _build_mot_accumulator(gt_file, res_file):
    """Replicates ROIByteTrack.evaluate_mot's accumulator construction."""
    import motmetrics as mm

    gt_cols = ["frame", "id", "x", "y", "w", "h", "conf", "class", "trunc", "occ"]
    gt_df = pd.read_csv(gt_file, header=None, names=gt_cols)
    res_cols = ["frame", "id", "x", "y", "w", "h", "conf", "v1", "v2", "v3"]
    try:
        res_df = pd.read_csv(res_file, header=None, names=res_cols)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        res_df = pd.DataFrame(columns=res_cols)

    acc = mm.MOTAccumulator(auto_id=True)
    for frame_id in sorted(gt_df["frame"].unique()):
        gt_frame = gt_df[gt_df["frame"] == frame_id]
        res_frame = res_df[res_df["frame"] == frame_id]

        gt_boxes = gt_frame[["x", "y", "w", "h"]].values
        res_boxes = (
            res_frame[["x", "y", "w", "h"]].values
            if not res_frame.empty else np.zeros((0, 4))
        )

        distances = mm.distances.iou_matrix(gt_boxes, res_boxes, max_iou=0.5)
        acc.update(
            gt_frame["id"].values,
            res_frame["id"].values if not res_frame.empty else [],
            distances,
        )
    return acc


def _run_final_mot_multi(cfg, det_model, reid_model, device, mot_root):
    """
    Iterates every sequence under `<mot_root>/sequences/`, runs tracking, and
    aggregates MOT metrics across sequences via motmetrics.compute_many.
    """
    import motmetrics as mm

    sequences_dir = os.path.join(mot_root, "sequences")
    annotations_dir = os.path.join(mot_root, "annotations")
    if not os.path.isdir(sequences_dir):
        raise FileNotFoundError(f"sequences dir not found: {sequences_dir}")

    images_root = cfg.output.images_tracked
    metrics_prefix = cfg.output.metrics_path  # we'll append ".<seq>.txt"
    os.makedirs(images_root, exist_ok=True)

    reid_model.eval()
    tracker = ROIByteTrack(model=det_model, reid_model=reid_model, device=device)

    accs = []
    names = []

    seqs = sorted(s for s in os.listdir(sequences_dir)
                  if os.path.isdir(os.path.join(sequences_dir, s)))
    print(f"[main] test MOT root: {mot_root} ({len(seqs)} sequences)")

    for seq_name in seqs:
        seq_path = os.path.join(sequences_dir, seq_name)
        ann_file = os.path.join(annotations_dir, seq_name + ".txt")
        if not os.path.exists(ann_file):
            print(f"[main]  skip {seq_name}: no annotations")
            continue

        out_images = os.path.join(images_root, seq_name)
        out_metrics = f"{metrics_prefix}.{seq_name}.txt"

        print(f"[main]  >> {seq_name}")
        tracker.process_tracking(
            seq_path,
            out_images,
            out_metrics,
            use_roi=cfg.tracking.use_roi,
            roi_coef=cfg.tracking.roi_coef,
        )
        accs.append(_build_mot_accumulator(ann_file, out_metrics))
        names.append(seq_name)

    if not accs:
        print("[main] no sequences evaluated.")
        return

    mh = mm.metrics.create()
    metrics_list = ["mota", "motp", "idf1", "mostly_tracked", "mostly_lost", "num_switches"]
    summary = mh.compute_many(
        accs, metrics=metrics_list, names=names, generate_overall=True
    )

    if cfg.evaluation.verbose:
        str_summary = mm.io.render_summary(
            summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names
        )
        print("\n--- MOT Evaluation Results (per-seq + OVERALL) ---")
        print(str_summary)

    summary_path = f"{metrics_prefix}.summary.csv"
    summary.to_csv(summary_path)
    print(f"[main] summary → {summary_path}")


def _make_mot_eval_fn(cfg, det_model, reid_model, device, sequence, annotations):
    """
    `(step) -> metrics_dict`. Runs a fresh tracker over the given sequence
    using the live reid weights. Heavy on disk I/O — keep the caller's
    `mot_eval_every` large.
    """
    ckpt_images_dir = cfg.output.images_tracked + "_ckpt"
    ckpt_metrics_path = cfg.output.metrics_path + ".ckpt"

    def _run(step):
        was_training = reid_model.training
        reid_model.eval()
        tracker = ROIByteTrack(model=det_model, reid_model=reid_model, device=device)
        tracker.process_tracking(
            sequence,
            ckpt_images_dir,
            ckpt_metrics_path,
            use_roi=cfg.tracking.use_roi,
            roi_coef=cfg.tracking.roi_coef,
        )
        summary = tracker.evaluate_mot(annotations, ckpt_metrics_path, verbose=False)
        if was_training:
            reid_model.train()

        row = summary.iloc[0]
        metrics = {k: _coerce_float(row[k]) for k in row.index}
        print(
            f"[ckpt] mot@step={step}: "
            + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items() if v is not None)
        )
        return metrics

    return _run


# ---------- self-supervised flows -------------------------------------------

def _build_ssl_backbone_model(cfg, device):
    from self_supervised import SelfSupervisedInstanceEmbedder

    backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    return SelfSupervisedInstanceEmbedder(
        backbone=backbone,
        feat_dim=cfg.reid.feat_dim,
        embed_dim=cfg.reid.embed_dim,
        temperature=cfg.reid.temperature,
    ).to(device)


def _ssl_pretrain(cfg, device):
    """mode=train: pretrain the SSL embedder on data.train.root."""
    from self_supervised import AugmentedInstanceDataset, train_self_supervised

    train_root = OmegaConf.select(cfg, "data.train.root")
    if not train_root:
        raise ValueError("mode=train requires `data.train.root`.")

    model = _build_ssl_backbone_model(cfg, device)

    crop_size = tuple(cfg.training.crop_size)
    train_ds = AugmentedInstanceDataset(dataset_dir=train_root, crop_size=crop_size)
    print(f"[ssl/train] crops: {len(train_ds)}  (root={train_root})")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        drop_last=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    scheduler = _build_scheduler(
        optimizer, OmegaConf.select(cfg, "training.scheduler"), int(cfg.training.steps)
    )

    train_self_supervised(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        steps=int(cfg.training.steps),
        device=device,
        scheduler=scheduler,
        history_path=cfg.output.history_path,
        log_every=int(cfg.training.log_every),
        desc="ssl pretrain",
        mode="train",
    )

    torch.save(model.state_dict(), cfg.reid.checkpoint_pretrain)
    print(f"[ssl/train] pretrain checkpoint → {cfg.reid.checkpoint_pretrain}")


def _ssl_test(cfg, det_model, device):
    """
    mode=test: load pretrain → finetune projector on data.val.root → save
    finetuned weights → final MOT eval on data.test.mot_*.
    """
    from self_supervised import AugmentedInstanceDataset, train_self_supervised

    val_root = OmegaConf.select(cfg, "data.val.root")
    if not val_root:
        raise ValueError("mode=test requires `data.val.root`.")

    model = _build_ssl_backbone_model(cfg, device)
    state = _load_state_dict(cfg.reid.checkpoint_pretrain, device)
    model.load_state_dict(state)
    print(f"[ssl/test] loaded pretrain: {cfg.reid.checkpoint_pretrain}")

    steps = int(cfg.testing.steps)
    if steps > 0:
        if bool(cfg.testing.freeze_backbone):
            for p in model.backbone.parameters():
                p.requires_grad_(False)
            trainable = [p for p in model.parameters() if p.requires_grad]
            print(f"[ssl/test] frozen backbone; finetuning {sum(p.numel() for p in trainable)} params")
        else:
            trainable = list(model.parameters())

        crop_size = tuple(cfg.testing.crop_size)
        val_ds = AugmentedInstanceDataset(dataset_dir=val_root, crop_size=crop_size)
        print(f"[ssl/test] finetune crops: {len(val_ds)}  (root={val_root})")

        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.testing.batch_size,
            shuffle=True,
            num_workers=cfg.testing.num_workers,
            pin_memory=cfg.testing.pin_memory,
            drop_last=True,
        )

        optimizer = torch.optim.Adam(trainable, lr=cfg.testing.lr)
        scheduler = _build_scheduler(
            optimizer, OmegaConf.select(cfg, "testing.scheduler"), steps
        )

        mot_eval_fn = None
        if int(cfg.testing.mot_eval_every) > 0:
            seq, ann = _resolve_test_paths(cfg)
            mot_eval_fn = _make_mot_eval_fn(cfg, det_model, model, device, seq, ann)

        train_self_supervised(
            model=model,
            train_loader=val_loader,
            optimizer=optimizer,
            steps=steps,
            device=device,
            scheduler=scheduler,
            mot_eval_fn=mot_eval_fn,
            mot_eval_every=int(cfg.testing.mot_eval_every),
            history_path=cfg.output.history_path,
            log_every=int(cfg.testing.log_every),
            desc="ssl finetune",
            mode="test",
        )

        torch.save(model.state_dict(), cfg.reid.checkpoint_finetuned)
        print(f"[ssl/test] finetuned checkpoint → {cfg.reid.checkpoint_finetuned}")
    else:
        print("[ssl/test] steps=0 — skipping finetuning, evaluating pretrain weights")

    _run_final_mot(cfg, det_model, model, device)


# ---------- metric-learning flows -------------------------------------------

def _build_ml_model(cfg, device, num_classes=0):
    from metric import EmbeddingModel
    base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    return EmbeddingModel(
        base=base,
        num_classes=num_classes,
        out_dim=cfg.reid.out_dim,
    ).to(device)


def _ml_pretrain(cfg, device):
    from metric import TripletDataset, train_triplet

    train_root = OmegaConf.select(cfg, "data.train.root")
    if not train_root:
        raise ValueError("mode=train requires `data.train.root`.")

    dataset = TripletDataset(dataset_dir=train_root, max_frame_delta=cfg.training.max_k)
    model = _build_ml_model(cfg, device, num_classes=len(dataset.identity_keys))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    scheduler = _build_scheduler(
        optimizer, OmegaConf.select(cfg, "training.scheduler"), int(cfg.training.steps)
    )

    train_triplet(
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        steps=int(cfg.training.steps),
        n_ids=cfg.training.n_ids,
        k_per_id=cfg.training.k_per_id,
        margin=cfg.training.margin,
        max_k=cfg.training.max_k,
        ce_weight=cfg.training.ce_weight,
        hard_negatives=cfg.training.hard_negatives,
        freeze_backbone=cfg.training.freeze_backbone,
        device=device,
        scheduler=scheduler,
        history_path=cfg.output.history_path,
        log_every=int(cfg.training.log_every),
        desc="ml pretrain",
        mode="train",
    )

    torch.save(model.state_dict(), cfg.reid.checkpoint_pretrain)
    print(f"[ml/train] pretrain checkpoint → {cfg.reid.checkpoint_pretrain}")


def _ml_test(cfg, det_model, device):
    from metric import TripletDataset, train_triplet

    val_root = OmegaConf.select(cfg, "data.val.root")
    if not val_root:
        raise ValueError("mode=test requires `data.val.root`.")

    state = _load_state_dict(cfg.reid.checkpoint_pretrain, device)
    num_classes = 0
    if "classifier.weight" in state:
        num_classes = state["classifier.weight"].shape[0]
    model = _build_ml_model(cfg, device, num_classes=num_classes)
    model.load_state_dict(state)
    print(f"[ml/test] loaded pretrain: {cfg.reid.checkpoint_pretrain}")

    steps = int(cfg.testing.steps)
    if steps > 0:
        dataset = TripletDataset(dataset_dir=val_root, max_frame_delta=cfg.testing.max_k)

        if bool(cfg.testing.freeze_backbone):
            for p in model.backbone.parameters():
                p.requires_grad_(False)
            trainable = [p for p in model.parameters() if p.requires_grad]
            print(f"[ml/test] frozen backbone; finetuning {sum(p.numel() for p in trainable)} params")
        else:
            trainable = list(model.parameters())

        optimizer = torch.optim.Adam(trainable, lr=cfg.testing.lr)
        scheduler = _build_scheduler(
            optimizer, OmegaConf.select(cfg, "testing.scheduler"), steps
        )

        mot_eval_fn = None
        if int(cfg.testing.mot_eval_every) > 0:
            seq, ann = _resolve_test_paths(cfg)
            mot_eval_fn = _make_mot_eval_fn(cfg, det_model, model, device, seq, ann)

        train_triplet(
            model=model,
            dataset=dataset,
            optimizer=optimizer,
            steps=steps,
            n_ids=cfg.testing.n_ids,
            k_per_id=cfg.testing.k_per_id,
            margin=cfg.testing.margin,
            max_k=cfg.testing.max_k,
            ce_weight=cfg.testing.ce_weight,
            hard_negatives=cfg.testing.hard_negatives,
            freeze_backbone=False,  # already applied above
            device=device,
            scheduler=scheduler,
            mot_eval_fn=mot_eval_fn,
            mot_eval_every=int(cfg.testing.mot_eval_every),
            history_path=cfg.output.history_path,
            log_every=int(cfg.testing.log_every),
            desc="ml finetune",
            mode="test",
        )

        torch.save(model.state_dict(), cfg.reid.checkpoint_finetuned)
        print(f"[ml/test] finetuned checkpoint → {cfg.reid.checkpoint_finetuned}")
    else:
        print("[ml/test] steps=0 — skipping finetuning, evaluating pretrain weights")

    _run_final_mot(cfg, det_model, model, device)


# ---------- pretrained-only (no training) -----------------------------------

def _build_pretrained(device):
    base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    base.fc = torch.nn.Identity()
    base.to(device).eval()
    return base


# ---------- entrypoint ------------------------------------------------------

@hydra.main(version_base=None, config_path="../configs", config_name="baseline")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = OmegaConf.select(cfg, "mode") or "test"

    det_model = YOLO(cfg.models["detection-model"])
    det_model.overrides["conf"] = cfg.detection.conf
    det_model.overrides["iou"] = cfg.detection.iou
    det_model.overrides["agnostic_nms"] = cfg.detection.agnostic_nms
    det_model.overrides["max_det"] = cfg.detection.max_det

    reid_type = cfg.reid.type

    if reid_type == "pretrained":
        reid_model = _build_pretrained(device)
        _run_final_mot(cfg, det_model, reid_model, device)
        return

    if reid_type == "self_supervised":
        if mode == "train":
            _ssl_pretrain(cfg, device)
            return
        if mode == "test":
            _ssl_test(cfg, det_model, device)
            return
        raise ValueError(f"unknown mode={mode!r} for self_supervised.")

    if reid_type == "metric_learning":
        if mode == "train":
            _ml_pretrain(cfg, device)
            return
        if mode == "test":
            _ml_test(cfg, det_model, device)
            return
        raise ValueError(f"unknown mode={mode!r} for metric_learning.")

    raise ValueError(f"unknown reid.type: {reid_type!r}.")


if __name__ == "__main__":
    main()
