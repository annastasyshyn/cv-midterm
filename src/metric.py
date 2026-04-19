import json
import os
import cv2
import numpy as np
import pandas as pd
import motmetrics as mm
import supervision as sv
# from ultralytics import YOLO
from ultralyticsplus import YOLO

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

from roi_bytetrack import ROIByteTrack


class TripletDataset(Dataset):
    def __init__(self, dataset_dir, max_frame_delta=5, transform=None,
                 crop_size=(256, 128)):
        self.dataset_dir = dataset_dir
        self.max_frame_delta = max_frame_delta
        self.crop_size = crop_size

        self.crop_transform = transform or T.Compose(
            [
                T.ToPILImage(),
                T.Resize(crop_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.identity_index = {}
        self.identity_keys = []
        self.build_index()

    def __len__(self):
        return len(self.identity_keys)

    def __getitem__(self, idx):
        key = self.identity_keys[idx]
        seq_path = os.path.join(self.dataset_dir, "sequences", key[0])
        entries = self.identity_index[key]
        frame_id, bbox = entries[np.random.randint(len(entries))]
        frame = self.load_frame(seq_path, frame_id)
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        raw = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if raw.size == 0:
            raw = np.zeros((self.crop_size[0], self.crop_size[1], 3), dtype=np.uint8)
        crop = self.crop_transform(raw)
        return (
            crop,
            torch.tensor(idx, dtype=torch.long),
            torch.tensor(frame_id, dtype=torch.long),
        )

    def build_index(self):
        sequences_dir = os.path.join(self.dataset_dir, "sequences")
        annotations_dir = os.path.join(self.dataset_dir, "annotations")

        for seq in sorted(os.listdir(sequences_dir)):
            seq_path = os.path.join(sequences_dir, seq)
            if not os.path.isdir(seq_path):
                continue

            anno_file = os.path.join(annotations_dir, seq + ".txt")
            if not os.path.exists(anno_file):
                continue

            for track_id, frames_data in self.parse_annotations(anno_file).items():
                entries = sorted(frames_data.items()) 
                if len(entries) >= 2:
                    self.identity_index[(seq, track_id)] = entries

        self.identity_keys = list(self.identity_index.keys())

    def sample_nk_batch(self, n_ids: int, k_per_id: int, max_k: int):
        n_avail = len(self.identity_keys)
        chosen = np.random.choice(n_avail, size=min(n_ids, n_avail), replace=False)

        all_crops, all_labels, all_frame_ids = [], [], []

        for local_label, key_idx in enumerate(chosen):
            key = self.identity_keys[key_idx]
            entries = self.identity_index[key]
            seq_path = os.path.join(self.dataset_dir, "sequences", key[0])

            anchor_i = np.random.randint(len(entries))
            lo = max(0, anchor_i - max_k)
            hi = min(len(entries), anchor_i + max_k + 1)
            window = [i for i in range(lo, hi) if i != anchor_i]

            if not window:
                window = [i for i in range(len(entries)) if i != anchor_i]

            n_extra = min(k_per_id - 1, len(window))
            picked_indices = [anchor_i] + list(
                np.random.choice(window, size=n_extra, replace=False)
            )

            for sidx in picked_indices:
                frame_id, bbox = entries[sidx]
                frame = self.load_frame(seq_path, frame_id)
                x1, y1, x2, y2 = map(int, bbox)
                h, w = frame.shape[:2]
                raw = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                if raw.size == 0:
                    raw = np.zeros((self.crop_size[0], self.crop_size[1], 3), dtype=np.uint8)
                crop = self.crop_transform(raw)
                all_crops.append(crop)
                all_labels.append(local_label)
                all_frame_ids.append(frame_id)

        return (
            torch.stack(all_crops),
            torch.tensor(all_labels, dtype=torch.long),
            torch.tensor(all_frame_ids, dtype=torch.long),
        )

    def parse_annotations(self, anno_file):
        annotations = {}
        with open(anno_file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                frame_id = int(parts[0])
                track_id = int(parts[1])
                x, y, w, h = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
                if track_id not in annotations:
                    annotations[track_id] = {}
                annotations[track_id][frame_id] = (x, y, x + w, y + h)
        return annotations

    def load_frame(self, seq_path, frame_id):
        img_path = os.path.join(seq_path, f"{frame_id:07d}.jpg")
        img = cv2.imread(img_path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class EmbeddingModel(nn.Module):
    def __init__(
        self,
        base,
        num_classes,
        out_dim=128,
    ):
        super().__init__()

        # base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        self.backbone = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim),
        )

        # self.classifier = nn.Linear(out_dim, num_classes) if num_classes > 0 else None
        self.classifier = None

    def forward(self, images):
        feat = self.backbone(images)
        emb = self.head(feat) 
        emb = nn.functional.normalize(emb, p=2, dim=1)
        return emb


def save_checkpoint(path, model, optimizer, step, config):
    torch.save(
        {
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


def triplet_loss(
    emb: torch.Tensor,
    labels: torch.Tensor,
    frame_ids: torch.Tensor,
    margin: float,
    max_k: int,
    hard_negatives: bool = True,
) -> torch.Tensor:

    B = emb.size(0)
    idx = torch.arange(B, device=emb.device)
    losses = []

    for i in range(B):
        pos_mask = (labels == labels[i]) & (idx != i)
        neg_mask = labels != labels[i]

        if not pos_mask.any() or not neg_mask.any():
            continue

        frame_deltas = (frame_ids - frame_ids[i]).abs().float()

        within_k = pos_mask & (frame_deltas <= max_k)
        candidates = within_k if within_k.any() else pos_mask
        pos_idxs = candidates.nonzero(as_tuple=True)[0]

        j = pos_idxs[frame_deltas[pos_idxs].argmin()].item()

        same_frame = frame_ids == frame_ids[i]
        same_frame_neg_mask = neg_mask & same_frame
        neg_mask_to_use = same_frame_neg_mask if same_frame_neg_mask.any() else neg_mask

        if hard_negatives:
            neg_dists = nn.functional.pairwise_distance(emb[i : i + 1], emb)
            neg_dists[~neg_mask_to_use] = float("inf")
            k = neg_dists.argmin().item()
        else:
            neg_idxs = neg_mask_to_use.nonzero(as_tuple=True)[0]
            if len(neg_idxs) == 0:
                continue
            k = neg_idxs[torch.randint(len(neg_idxs), (1,), device=emb.device)].item()

        d_pos = nn.functional.pairwise_distance(emb[i : i + 1], emb[j : j + 1])
        d_neg = nn.functional.pairwise_distance(emb[i : i + 1], emb[k : k + 1])
        dynamic_margin = margin * (1.0 + frame_deltas[j] / max_k)

        losses.append(nn.functional.relu(d_pos - d_neg + dynamic_margin))

    if not losses:
        return emb.sum() * 0.0

    return torch.cat(losses).mean()


def _save_history(history, history_path):
    if not history_path:
        return
    parent = os.path.dirname(history_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)


def train_triplet(
    model,
    dataset,
    optimizer,
    steps=1000,
    n_ids=8,
    k_per_id=4,
    margin=1.0,
    max_k=5,
    ce_weight=0.5,
    hard_negatives=True,
    freeze_backbone=False,
    device="cuda",
    scheduler=None,
    mot_eval_fn=None,
    mot_eval_every=0,
    history_path=None,
    log_every=10,
    desc="reid",
    mode=None,
):
    """
    Triplet training loop with an optional history dict written to `history_path`.

    history schema:
      {"mode": str, "loss": {"steps": [...], "values": [...]},
       "mot": {"steps": [...], "metrics": [{...}, ...]}}
    """
    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(False)

    history = {
        "mode": mode,
        "loss": {"steps": [], "values": []},
        "mot":  {"steps": [], "metrics": []},
    }

    model.train()
    ce_fn = (
        nn.CrossEntropyLoss()
        if (model.classifier is not None and ce_weight > 0)
        else None
    )

    running = []
    for step in tqdm(range(1, steps + 1), desc=desc):
        crops, labels, frame_ids = dataset.sample_nk_batch(n_ids, k_per_id, max_k)

        crops = crops.to(device)
        labels = labels.to(device)
        frame_ids = frame_ids.to(device)

        optimizer.zero_grad()
        emb = model(crops)

        loss = triplet_loss(emb, labels, frame_ids, margin, max_k, hard_negatives)

        if ce_fn is not None:
            unique_labels, remapped = torch.unique(labels, return_inverse=True)
            logits = model.classifier(emb)
            loss = loss + ce_weight * ce_fn(logits[:, : len(unique_labels)], remapped)

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        running.append(loss.item())

        if step % log_every == 0:
            history["loss"]["steps"].append(step)
            history["loss"]["values"].append(float(np.mean(running)))
            running = []
            _save_history(history, history_path)

        if mot_eval_fn is not None and mot_eval_every and step % mot_eval_every == 0:
            metrics = mot_eval_fn(step)
            history["mot"]["steps"].append(step)
            history["mot"]["metrics"].append(metrics)
            model.train()
            _save_history(history, history_path)

    _save_history(history, history_path)
    return history


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = YOLO("mshamrai/yolov8n-visdrone")  # model for detection
    model.overrides["conf"] = 0.25
    model.overrides["iou"] = 0.45
    model.overrides["agnostic_nms"] = False
    model.overrides["max_det"] = 1000

    max_k = 5
    dataset = TripletDataset(dataset_dir="VisDrone2019-MOT-val", max_frame_delta=max_k)
    n_ids = len(dataset.identity_keys)

    embedding_model = EmbeddingModel(
        resnet50(weights=ResNet50_Weights.IMAGENET1K_V1),
        num_classes=n_ids,
    )
    embedding_model.to(device)
    optimizer = torch.optim.Adam(embedding_model.parameters(), lr=1e-4)

    train_triplet(
        embedding_model,
        dataset,
        optimizer,
        steps=1000,
        n_ids=8,
        k_per_id=4,
        margin=1.0,
        max_k=max_k,
        ce_weight=0.5,
        freeze_backbone=False,
        device=device,
    )

    save_checkpoint(
        "triplet_embedding_model.pth",
        embedding_model,
        optimizer,
        step=1000,
        config=dict(
            steps=1000, n_ids=8, k_per_id=4, margin=1.0,
            max_k=max_k, ce_weight=0.5, hard_negatives=True,
            freeze_backbone=False,
        ),
    )

    track = ROIByteTrack(
        model=model,
        reid_model=embedding_model,
        device=device,
    )

    mot_metricks_path = "detection_metrics_sml.txt"

    track.process_tracking(
        "VisDrone2019-MOT-test-dev/sequences/uav0000009_03358_v",
        "output_images_tracked",
        mot_metricks_path,
        use_roi=True,
        roi_coef=0.25,
    )

    track.evaluate_mot(
        "VisDrone2019-MOT-test-dev/annotations/uav0000009_03358_v.txt",
        mot_metricks_path,
        verbose=True,
    )