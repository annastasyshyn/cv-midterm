import json
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
from torchvision.models import (
    ResNet50_Weights,
    Swin_T_Weights,
    resnet50,
    swin_t,
)
from ultralyticsplus import YOLO

from roi_bytetrack import ROIByteTrack
from hota_metric import compute_hota_metrics


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
    if path is None:
        return None
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

    save_images = bool(OmegaConf.select(cfg, "output.save_images"))
    test_sequence, test_annotations = _resolve_test_paths(cfg)
    print(f"[main] test MOT seq (single): {test_sequence}  save_images={save_images}")

    os.makedirs(os.path.dirname(cfg.output.metrics_path) or ".", exist_ok=True)
    reid_model.eval()
    tracker = ROIByteTrack(model=det_model, reid_model=reid_model, device=device)
    tracker.process_tracking(
        test_sequence,
        cfg.output.images_tracked,
        cfg.output.metrics_path,
        use_roi=cfg.tracking.use_roi,
        roi_coef=cfg.tracking.roi_coef,
        save_images=save_images,
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

    save_images = bool(OmegaConf.select(cfg, "output.save_images"))
    images_root = cfg.output.images_tracked
    metrics_prefix = cfg.output.metrics_path  # we'll append ".<seq>.txt"
    os.makedirs(os.path.dirname(metrics_prefix) or ".", exist_ok=True)
    if save_images:
        os.makedirs(images_root, exist_ok=True)

    reid_model.eval()
    tracker = ROIByteTrack(model=det_model, reid_model=reid_model, device=device)

    accs = []
    names = []

    seqs = sorted(s for s in os.listdir(sequences_dir)
                  if os.path.isdir(os.path.join(sequences_dir, s)))
    print(f"[main] test MOT root: {mot_root} ({len(seqs)} sequences)  save_images={save_images}")

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
            save_images=save_images,
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

    # Compute HOTA for each sequence and overall
    hota_scores = {}
    annotations_dir = os.path.join(mot_root, "annotations")
    for seq_name in names:
        ann_file = os.path.join(annotations_dir, seq_name + ".txt")
        out_metrics = f"{metrics_prefix}.{seq_name}.txt"
        hota_dict = compute_hota_metrics(ann_file, out_metrics)
        hota_scores[seq_name] = hota_dict['hota']

    # Add HOTA column to summary
    hota_values = [hota_scores.get(name, 0.0) for name in names]
    # Compute overall HOTA as mean
    overall_hota = np.mean(hota_values) if hota_values else 0.0
    summary['hota'] = hota_values + [overall_hota]

    if cfg.evaluation.verbose:
        str_summary = mm.io.render_summary(
            summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names
        )
        print("\n--- MOT Evaluation Results (per-seq + OVERALL) ---")
        print(str_summary)
        print("\n--- HOTA Scores ---")
        for seq_name, hota_score in hota_scores.items():
            print(f"{seq_name}: {hota_score:.4f}")
        print(f"OVERALL: {overall_hota:.4f}")

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
            save_images=False,  # always skip for periodic eval — disk-prohibitive otherwise
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

def _detect_backbone_from_path(path: str) -> str:
    """
    Infer the backbone architecture from the checkpoint filename.

    The dispatch is filename-driven (not cfg.reid.backbone) so a stray config
    like `backbone: resnet50` paired with `ssl_pretrain_swin_t.pth` fails
    loudly at build time rather than at state_dict load.
    """
    name = os.path.basename(str(path)).lower()
    if "resnet50" in name:
        return "resnet50"
    if "swin_t" in name:
        return "swin_t"
    raise ValueError(
        f"Cannot infer backbone from checkpoint path {path!r}: "
        "filename must contain 'resnet50' or 'swin_t'."
    )


def _is_wrapped_embedder_state(state: dict) -> bool:
    """
    True when `state` looks like a saved SSL or metric-learning embedder
    (has the `backbone.` prefix our wrappers introduce).
    False when it looks like a bare torchvision backbone state_dict
    (e.g. `conv1.weight` / `layer1.*` for ResNet, `features.*` / `norm.*`
    / `head.*` for Swin-T) — a classifier pretrained directly on the raw
    architecture, with no SSL projector / ML head baked in.
    """
    return any(k.startswith("backbone.") for k in state.keys())


def _build_ssl_backbone_model(cfg, device, raw_backbone_state=None, imagenet_init=True):
    """
    Build the SSL embedder. When `raw_backbone_state` is provided (a bare
    torchvision backbone state_dict), it is loaded into the base model
    BEFORE the embedder wraps it — so the wrapped `backbone` sub-module
    inherits the pretrained weights while the `projector` stays randomly
    initialised and is later learned during finetuning.

    `imagenet_init`: when False, skip torchvision's ImageNet weight download
      — useful in `mode=test` where we're about to overwrite every weight
      with a user-supplied checkpoint anyway. Default True preserves the
      `mode=train` behaviour (pretrain starts from ImageNet).
    """
    from self_supervised import SelfSupervisedInstanceEmbedder

    backbone_type = _detect_backbone_from_path(cfg.reid.checkpoint_pretrain)
    if backbone_type == "resnet50":
        base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if imagenet_init else None)
    elif backbone_type == "swin_t":
        base = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1 if imagenet_init else None)
    else:
        raise ValueError(f"Unsupported backbone_type={backbone_type!r}")

    if raw_backbone_state is not None:
        # Drop classifier-head keys: supervised-pretrain checkpoints carry an
        # n_classes-specific head (e.g. VisDrone → fc.weight is [10, 2048]),
        # which shape-mismatches the ImageNet head [1000, 2048]. We never use
        # the classifier — the embedder taps features before it — so strip
        # the keys instead of resizing.
        head_keys = {
            "resnet50": ("fc.weight", "fc.bias"),
            "swin_t":   ("head.weight", "head.bias"),
        }[backbone_type]
        filtered_state = {k: v for k, v in raw_backbone_state.items() if k not in head_keys}
        dropped = [k for k in head_keys if k in raw_backbone_state]

        missing, unexpected = base.load_state_dict(filtered_state, strict=False)
        print(
            f"[ssl] loaded raw {backbone_type} weights "
            f"(missing={len(missing)}, unexpected={len(unexpected)}, "
            f"dropped_head={dropped})"
        )

    print(f"[ssl] backbone={backbone_type} (inferred from {cfg.reid.checkpoint_pretrain})")
    return SelfSupervisedInstanceEmbedder(
        backbone=base,
        backbone_type=backbone_type,
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


def _build_heldout_ssl_dataset(cfg, crop_size):
    """
    Build an AugmentedInstanceDataset over whatever test-dev data is configured:
      - data.test.mot_root set → all sequences under it (full test-dev).
      - else data.test.mot_sequence set → a single sequence via sequence_filter.
      - else → None (caller skips val-loss).
    """
    from self_supervised import AugmentedInstanceDataset

    mot_root = OmegaConf.select(cfg, "data.test.mot_root")
    if mot_root:
        ds = AugmentedInstanceDataset(dataset_dir=mot_root, crop_size=crop_size)
        print(f"[ssl/test] held-out crops (full test-dev): {len(ds)}  (root={mot_root})")
        return ds

    seq_path = OmegaConf.select(cfg, "data.test.mot_sequence")
    if not seq_path:
        print("[ssl/test] no data.test.mot_root or mot_sequence — skipping held-out loss")
        return None

    # Derive the test-dev root + sequence name from the single-seq path.
    # Path layout: <root>/sequences/<seq_name>
    seq_path = seq_path.rstrip("/")
    seq_name = os.path.basename(seq_path)
    seq_root = os.path.dirname(os.path.dirname(seq_path))
    ds = AugmentedInstanceDataset(
        dataset_dir=seq_root,
        crop_size=crop_size,
        sequence_filter=[seq_name],
    )
    print(f"[ssl/test] held-out crops (single seq): {len(ds)}  (seq={seq_name})")
    return ds


def _ssl_test(cfg, det_model, device):
    """
    mode=test: load pretrain → finetune projector on data.val.root → save
    finetuned weights → final MOT eval on data.test.mot_*.
    """
    from self_supervised import AugmentedInstanceDataset, train_self_supervised

    val_root = OmegaConf.select(cfg, "data.val.root")
    if not val_root:
        raise ValueError("mode=test requires `data.val.root`.")

    state = _load_state_dict(cfg.reid.checkpoint_pretrain, device)
    if _is_wrapped_embedder_state(state):
        # Full SSL embedder checkpoint (backbone + projector, produced by
        # a prior mode=train run). Build an empty embedder and load all of it.
        model = _build_ssl_backbone_model(cfg, device, imagenet_init=False)
        model.load_state_dict(state)
        print(f"[ssl/test] loaded SSL embedder: {cfg.reid.checkpoint_pretrain}")
    else:
        # Bare torchvision backbone checkpoint (supervised pretrain on
        # VisDrone, no SSL projector baked in). Load it into the base
        # before wrapping; projector stays random-init and must be trained
        # during finetune — so testing.steps=0 with this flavor yields
        # random-projection embeddings.
        model = _build_ssl_backbone_model(cfg, device, raw_backbone_state=state, imagenet_init=False)
        print(f"[ssl/test] loaded raw backbone: {cfg.reid.checkpoint_pretrain} "
              "(projector random-init, will be trained during finetune)")
        if int(cfg.testing.steps) == 0:
            print("[ssl/test] WARNING: raw backbone + testing.steps=0 → "
                  "embeddings are a random projection of backbone features; "
                  "MOT metrics will be essentially random.")

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

        finetune_loader = DataLoader(
            val_ds,
            batch_size=cfg.testing.batch_size,
            shuffle=True,
            num_workers=cfg.testing.num_workers,
            pin_memory=cfg.testing.pin_memory,
            drop_last=True,
        )

        # Held-out validation loss on the test-dev crops. Works in both modes:
        #   Option A — mot_root set: index every sequence under the root
        #   Option B — mot_root null, mot_sequence set: index just that one sequence
        # This mirrors the final MOT eval dispatch so the "val loss" curve on
        # the overview plot always reflects the same test data the final
        # metrics are reported on.
        heldout_loader = None
        val_every = int(OmegaConf.select(cfg, "testing.val_every") or 0)
        if val_every > 0:
            heldout_ds = _build_heldout_ssl_dataset(cfg, crop_size)
            if heldout_ds is not None:
                heldout_loader = DataLoader(
                    heldout_ds,
                    batch_size=cfg.testing.batch_size,
                    shuffle=False,
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

        # Periodic stepped checkpoints under the finetuned path, e.g.
        # `out/ssl_r50_full/finetuned.step02000.pth`. The final post-loop
        # save still lands at `cfg.reid.checkpoint_finetuned` so downstream
        # eval picks up the last weights regardless of save_every.
        save_every = int(OmegaConf.select(cfg, "testing.save_every") or 0)
        checkpoint_fn = None
        if save_every > 0:
            ckpt_base = cfg.reid.checkpoint_finetuned
            os.makedirs(os.path.dirname(ckpt_base) or ".", exist_ok=True)

            def checkpoint_fn(step, _base=ckpt_base):
                stem, ext = os.path.splitext(_base)
                stepped = f"{stem}.step{step:06d}{ext or '.pth'}"
                torch.save(model.state_dict(), stepped)
                print(f"[ssl/test] ckpt@step={step} → {stepped}")

        train_self_supervised(
            model=model,
            train_loader=finetune_loader,
            optimizer=optimizer,
            steps=steps,
            device=device,
            scheduler=scheduler,
            val_loader=heldout_loader,
            val_every=val_every,
            val_max_batches=OmegaConf.select(cfg, "testing.val_max_batches"),
            mot_eval_fn=mot_eval_fn,
            mot_eval_every=int(cfg.testing.mot_eval_every),
            checkpoint_fn=checkpoint_fn,
            checkpoint_every=save_every,
            history_path=cfg.output.history_path,
            log_every=int(cfg.testing.log_every),
            desc="ssl finetune",
            mode="test",
        )

        os.makedirs(os.path.dirname(cfg.reid.checkpoint_finetuned) or ".", exist_ok=True)
        torch.save(model.state_dict(), cfg.reid.checkpoint_finetuned)
        print(f"[ssl/test] finetuned checkpoint → {cfg.reid.checkpoint_finetuned}")
    else:
        print("[ssl/test] steps=0 — skipping finetuning, evaluating pretrain weights")

    _run_final_mot(cfg, det_model, model, device)


# ---------- metric-learning flows -------------------------------------------

_BACKBONE_FEAT_DIM = {"resnet50": 2048, "swin_t": 768}


def _build_ml_model(cfg, device, num_classes=0, raw_backbone_state=None, imagenet_init=True):
    """
    Build the metric-learning EmbeddingModel. Mirrors `_build_ssl_backbone_model`:
      - backbone type is inferred from `cfg.reid.checkpoint_pretrain` filename
      - `imagenet_init=False` skips the torchvision ImageNet download when the
        caller is about to overwrite every weight anyway (`mode=test`)
      - `raw_backbone_state` (bare torchvision state_dict) is stripped of its
        classifier head and loaded into `base` before wrapping, so the
        EmbeddingModel's `head` stays random-init and gets trained during
        finetune.
    """
    from metric import EmbeddingModel

    backbone_type = _detect_backbone_from_path(cfg.reid.checkpoint_pretrain)
    if backbone_type == "resnet50":
        base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if imagenet_init else None)
    elif backbone_type == "swin_t":
        base = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1 if imagenet_init else None)
    else:
        raise ValueError(f"Unsupported backbone_type={backbone_type!r}")

    if raw_backbone_state is not None:
        head_keys = {
            "resnet50": ("fc.weight", "fc.bias"),
            "swin_t":   ("head.weight", "head.bias"),
        }[backbone_type]
        filtered_state = {k: v for k, v in raw_backbone_state.items() if k not in head_keys}
        dropped = [k for k in head_keys if k in raw_backbone_state]
        missing, unexpected = base.load_state_dict(filtered_state, strict=False)
        print(
            f"[ml] loaded raw {backbone_type} weights "
            f"(missing={len(missing)}, unexpected={len(unexpected)}, "
            f"dropped_head={dropped})"
        )

    print(f"[ml] backbone={backbone_type} (inferred from {cfg.reid.checkpoint_pretrain})")
    return EmbeddingModel(
        base=base,
        num_classes=num_classes,
        out_dim=cfg.reid.out_dim,
        backbone_type=backbone_type,
        feat_dim=_BACKBONE_FEAT_DIM[backbone_type],
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

    _max_k_max = OmegaConf.select(cfg, "training.max_k_max")
    _max_k_max = int(_max_k_max) if _max_k_max is not None else None
    _max_k_warmup = OmegaConf.select(cfg, "training.max_k_warmup")
    _max_k_warmup = int(_max_k_warmup) if _max_k_warmup is not None else None

    train_triplet(
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        steps=int(cfg.training.steps),
        n_ids=cfg.training.n_ids,
        k_per_id=cfg.training.k_per_id,
        margin=cfg.training.margin,
        max_k=cfg.training.max_k,
        max_k_max=_max_k_max,
        max_k_warmup=_max_k_warmup,
        ce_weight=cfg.training.ce_weight,
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


def _build_heldout_ml_dataset(cfg, max_frame_delta):
    """
    Build a TripletDataset over whatever test-dev data is configured, mirroring
    `_build_heldout_ssl_dataset`:
      - data.test.mot_root set → all test-dev sequences.
      - else data.test.mot_sequence set → a single sequence via sequence_filter.
      - else → None (caller skips val-loss).
    """
    from metric import TripletDataset

    mot_root = OmegaConf.select(cfg, "data.test.mot_root")
    if mot_root:
        ds = TripletDataset(dataset_dir=mot_root, max_frame_delta=max_frame_delta)
        print(f"[ml/test] held-out identities (full test-dev): {len(ds)}  (root={mot_root})")
        return ds

    seq_path = OmegaConf.select(cfg, "data.test.mot_sequence")
    if not seq_path:
        print("[ml/test] no data.test.mot_root or mot_sequence — skipping held-out loss")
        return None

    seq_path = seq_path.rstrip("/")
    seq_name = os.path.basename(seq_path)
    seq_root = os.path.dirname(os.path.dirname(seq_path))
    ds = TripletDataset(
        dataset_dir=seq_root,
        max_frame_delta=max_frame_delta,
        sequence_filter=[seq_name],
    )
    print(f"[ml/test] held-out identities (single seq): {len(ds)}  (seq={seq_name})")
    if len(ds) < 2:
        print("[ml/test] single seq has <2 identities with ≥2 entries — triplet val loss undefined, skipping")
        return None
    return ds


def _ml_test(cfg, det_model, device):
    from metric import TripletDataset, train_triplet

    val_root = OmegaConf.select(cfg, "data.val.root")
    if not val_root:
        raise ValueError("mode=test requires `data.val.root`.")

    state = _load_state_dict(cfg.reid.checkpoint_pretrain, device)
    if state is None:
        # No pretrained checkpoint; use ImageNet weights only.
        model = _build_ml_model(cfg, device, imagenet_init=True)
        print("[ml/test] using ImageNet weights, no pretrained checkpoint")
    elif _is_wrapped_embedder_state(state):
        # Full EmbeddingModel checkpoint (backbone + head, from a prior ML
        # mode=train run). Build an empty wrapper and load everything.
        num_classes = 0
        if "classifier.weight" in state:
            num_classes = state["classifier.weight"].shape[0]
        model = _build_ml_model(cfg, device, num_classes=num_classes, imagenet_init=False)
        model.load_state_dict(state)
        print(f"[ml/test] loaded wrapped embedder: {cfg.reid.checkpoint_pretrain}")
    else:
        # Bare torchvision backbone (supervised pretrain on VisDrone, no
        # EmbeddingModel head). Head stays random-init — must be trained.
        model = _build_ml_model(cfg, device, raw_backbone_state=state, imagenet_init=False)
        print(f"[ml/test] loaded raw backbone: {cfg.reid.checkpoint_pretrain} "
              "(head random-init, will be trained during finetune)")
        if int(cfg.testing.steps) == 0:
            print("[ml/test] WARNING: raw backbone + testing.steps=0 → "
                  "embeddings are a random projection of backbone features; "
                  "MOT metrics will be essentially random.")

    # Resume: load finetuned checkpoint + existing history when testing.resume=true.
    resume = bool(OmegaConf.select(cfg, "testing.resume"))
    prior_history = None
    step_offset = 0
    if resume:
        ckpt_path = str(cfg.reid.checkpoint_finetuned)
        if os.path.exists(ckpt_path):
            resume_state = _load_state_dict(ckpt_path, device)
            model.load_state_dict(resume_state)
            print(f"[ml/test] resumed weights from {ckpt_path}")
        else:
            print(f"[ml/test] resume=true but {ckpt_path} not found — starting fresh")
        hist_path = str(cfg.output.history_path)
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                prior_history = json.load(f)
            loss_steps = prior_history.get("loss", {}).get("steps", [])
            step_offset = loss_steps[-1] if loss_steps else 0
            print(f"[ml/test] appending to existing history (step_offset={step_offset})")
        else:
            print(f"[ml/test] no history at {hist_path} — starting fresh history")

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

        val_every = int(OmegaConf.select(cfg, "testing.val_every") or 0)
        heldout_ds = None
        if val_every > 0:
            heldout_ds = _build_heldout_ml_dataset(cfg, max_frame_delta=cfg.testing.max_k)

        mot_eval_fn = None
        if int(cfg.testing.mot_eval_every) > 0:
            seq, ann = _resolve_test_paths(cfg)
            mot_eval_fn = _make_mot_eval_fn(cfg, det_model, model, device, seq, ann)

        # Periodic stepped checkpoints under the finetuned path — same
        # pattern as _ssl_test. Final post-loop save still lands at
        # `cfg.reid.checkpoint_finetuned`.
        save_every = int(OmegaConf.select(cfg, "testing.save_every") or 0)
        checkpoint_fn = None
        if save_every > 0:
            ckpt_base = cfg.reid.checkpoint_finetuned
            os.makedirs(os.path.dirname(ckpt_base) or ".", exist_ok=True)

            def checkpoint_fn(step, _base=ckpt_base):
                stem, ext = os.path.splitext(_base)
                stepped = f"{stem}.step{step:06d}{ext or '.pth'}"
                torch.save(model.state_dict(), stepped)
                print(f"[ml/test] ckpt@step={step} → {stepped}")

        _max_k_max = OmegaConf.select(cfg, "testing.max_k_max")
        _max_k_max = int(_max_k_max) if _max_k_max is not None else None
        _max_k_warmup = OmegaConf.select(cfg, "testing.max_k_warmup")
        _max_k_warmup = int(_max_k_warmup) if _max_k_warmup is not None else None
        _freeze_bn = bool(OmegaConf.select(cfg, "testing.freeze_bn") or False)

        train_triplet(
            model=model,
            dataset=dataset,
            optimizer=optimizer,
            steps=steps,
            n_ids=cfg.testing.n_ids,
            k_per_id=cfg.testing.k_per_id,
            margin=cfg.testing.margin,
            max_k=cfg.testing.max_k,
            max_k_max=_max_k_max,
            max_k_warmup=_max_k_warmup,
            ce_weight=cfg.testing.ce_weight,
            freeze_backbone=False,  # already applied above
            device=device,
            scheduler=scheduler,
            val_dataset=heldout_ds,
            val_every=val_every,
            val_max_batches=OmegaConf.select(cfg, "testing.val_max_batches"),
            mot_eval_fn=mot_eval_fn,
            mot_eval_every=int(cfg.testing.mot_eval_every),
            checkpoint_fn=checkpoint_fn,
            checkpoint_every=save_every,
            history_path=cfg.output.history_path,
            log_every=int(cfg.testing.log_every),
            desc="ml finetune",
            mode="test",
            step_offset=step_offset,
            prior_history=prior_history,
            freeze_bn=_freeze_bn,
        )

        os.makedirs(os.path.dirname(cfg.reid.checkpoint_finetuned) or ".", exist_ok=True)
        torch.save(model.state_dict(), cfg.reid.checkpoint_finetuned)
        print(f"[ml/test] finetuned checkpoint → {cfg.reid.checkpoint_finetuned}")
    else:
        print("[ml/test] steps=0 — skipping finetuning, evaluating pretrain weights")

    run_mot = OmegaConf.select(cfg, "evaluation.run_mot")
    if run_mot is None or bool(run_mot):
        _run_final_mot(cfg, det_model, model, device)
    else:
        print("[ml/test] evaluation.run_mot=false — skipping final MOT eval")


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
