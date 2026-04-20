import json
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from ultralyticsplus import YOLO
from roi_bytetrack import ROIByteTrack


def build_contrastive_augmentations(height: int = 256, width: int = 128):
    """
    Strong, stochastic augmentations for instance contrastive learning.
    Two independent pipelines are returned -- same recipe but independently
    sampled per call
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def make():
        return A.Compose([
            A.Resize(int(height * 1.15), int(width * 1.15)),
            A.RandomResizedCrop(size=(height, width), scale=(0.6, 1.0), ratio=(0.8, 1.25), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
            A.ToGray(p=0.2),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.05, 0.125), hole_width_range=(0.05, 0.125), p=0.3),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    return make(), make()


class AugmentedInstanceDataset(Dataset):
    def __init__(self, dataset_dir, crop_size=(256, 128), aug_strength='strong',
                 sequence_filter=None):
        """
        dataset_dir: root with `sequences/` + `annotations/` subdirs.
        sequence_filter: optional iterable of sequence names — when provided,
          only those sequences are indexed (lets the caller reuse the same
          root for Option-B single-seq eval by passing `[seq_name]`).
        """
        self.dataset_dir = dataset_dir
        self.crop_size = crop_size
        self.sequence_filter = set(sequence_filter) if sequence_filter else None

        self.aug_pipeline1, self.aug_pipeline2 = build_contrastive_augmentations(
            height=crop_size[0], width=crop_size[1]
        )

        self.bbox_list = []
        self.build_index()

    def __len__(self):
        return len(self.bbox_list)

    def __getitem__(self, idx):

        seq_name, frame_id, bbox = self.bbox_list[idx]
        seq_path = os.path.join(self.dataset_dir, "sequences", seq_name)

        frame = self.load_frame(seq_path, frame_id)
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

        if crop.size == 0:
            crop = np.zeros((self.crop_size[0], self.crop_size[1], 3), dtype=np.uint8)

        view1 = self.aug_pipeline1(image=crop)['image']
        view2 = self.aug_pipeline2(image=crop)['image']

        return view1, view2

    def build_index(self):
        sequences_dir = os.path.join(self.dataset_dir, "sequences")
        annotations_dir = os.path.join(self.dataset_dir, "annotations")

        for seq in sorted(os.listdir(sequences_dir)):
            if self.sequence_filter is not None and seq not in self.sequence_filter:
                continue
            seq_path = os.path.join(sequences_dir, seq)
            if not os.path.isdir(seq_path):
                continue

            anno_file = os.path.join(annotations_dir, seq + ".txt")
            if not os.path.exists(anno_file):
                continue

            with open(anno_file, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    frame_id = int(parts[0])
                    track_id = int(parts[1])
                    x, y, w, h = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])

                    self.bbox_list.append((seq, frame_id, (x, y, x + w, y + h)))

    def load_frame(self, seq_path, frame_id):
        img_path = os.path.join(seq_path, f"{frame_id:07d}.jpg")
        img = cv2.imread(img_path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def get_batch_loader(self, batch_size=32, num_workers=4, shuffle=True, pin_memory=True):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )


class _SwinFeatureExtractor(nn.Module):
    """
    Trim torchvision's `swin_t` down to the pre-head global feature vector.
    `features` outputs (N, H, W, C); `norm` acts on the channel dim; we then
    permute to NCHW so AdaptiveAvgPool2d + flatten can reuse the same
    `encode()` code path as the ResNet branch.
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.features = backbone.features
        self.norm = backbone.norm
        self.avgpool = backbone.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return self.avgpool(x)


class SelfSupervisedInstanceEmbedder(nn.Module):
    """
    MASA-style instance embedder trained with InfoNCE contrastive loss.

    Pipeline per batch:
      1. Two augmented views (v1, v2) of the same crop arrive from the loader.
      2. Backbone + projection head map each view to an L2-normalised embedding.
      3. Positives = (v1_i, v2_i); negatives = (v1_i, v2_j) for j != i.
        This relies on the assumption:
        **different bounding boxes within the same frame (and across frames)
        correspond to different object instances**

    Reference: "MASA: Matching Anything by Segmenting Anything"
    (https://arxiv.org/abs/2406.04221)
    """

    def __init__(
        self,
        backbone: nn.Module,
        backbone_type: str = "resnet50",
        feat_dim: int = 2048,
        embed_dim: int = 128,
        temperature: float = 0.2,
    ) -> None:
        super().__init__()
        if backbone_type == "resnet50":
            self.backbone = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
                backbone.layer4,
                nn.AdaptiveAvgPool2d((1, 1)),
            )
        elif backbone_type == "swin_t":
            self.backbone = _SwinFeatureExtractor(backbone)
        else:
            raise ValueError(
                f"Unsupported backbone_type={backbone_type!r}. "
                "Expected 'resnet50' or 'swin_t'."
            )
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, embed_dim),
        )
        self.temperature = temperature

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)
        f = torch.flatten(f, 1)
        z = self.projector(f)
        return F.normalize(z, dim=-1)

    def forward(self, view1: torch.Tensor, view2: torch.Tensor = None) -> torch.Tensor:
        if view2 is None:
            return self.encode(view1)
        else:
            z1 = self.encode(view1)
            z2 = self.encode(view2)
            return self.contrastive_loss(z1, z2)

    def contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Symmetric InfoNCE over a batch of paired embeddings
        """
        N = z1.size(0)
        logits = (z1 @ z2.t()) / self.temperature
        targets = torch.arange(N, device=z1.device)

        ## the paper uses just crossentropy over a dirac function, so turgets are like that
        loss_a = F.cross_entropy(logits, targets)
        loss_b = F.cross_entropy(logits.t(), targets)
        return 0.5 * (loss_a + loss_b)


@torch.no_grad()
def evaluate_ssl_loss(model, val_loader, device, max_batches=None):
    """
    Average InfoNCE loss on a held-out loader. Restores the caller's train/eval mode.
    Used in mode=test to track a validation signal on the test-dev crops
    while finetuning on data.val.root.
    """
    was_training = model.training
    model.eval()
    losses = []
    for bi, (v1, v2) in enumerate(val_loader):
        if max_batches is not None and bi >= max_batches:
            break
        v1 = v1.to(device, non_blocking=True)
        v2 = v2.to(device, non_blocking=True)
        loss = model(v1, v2)
        losses.append(loss.item())
    if was_training:
        model.train()
    return float(np.mean(losses)) if losses else float("nan")


def _save_history(history, history_path):
    if not history_path:
        return
    parent = os.path.dirname(history_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)


def train_self_supervised(
    model,
    train_loader,
    optimizer,
    steps=1000,
    device="cuda",
    scheduler=None,
    val_loader=None,
    val_every=0,
    val_max_batches=None,
    mot_eval_fn=None,
    mot_eval_every=0,
    checkpoint_fn=None,
    checkpoint_every=0,
    history_path=None,
    log_every=10,
    desc="ssl training",
    mode=None,
):
    """
    Train the SSL embedder. Logs a running loss and optional periodic MOT
    metrics into a history dict that the visualizer consumes.

    The same function is used for both pretraining (mode=train, loader over
    data.train.root) and finetuning (mode=test, loader over data.val.root) —
    the loss curve is always labeled against the loader that drove it.

    mot_eval_fn: optional callable `(step: int) -> dict` returning MOT metrics
      for the current model state (expensive — keep mot_eval_every large).
    checkpoint_fn: optional callable `(step: int) -> None` — invoked every
      `checkpoint_every` steps to persist a stepped snapshot of the model.
      Cheap (state_dict write), so a 1-2k step cadence is reasonable.
    """
    history = {
        "mode": mode,
        "loss":     {"steps": [], "values": []},
        "val_loss": {"steps": [], "values": []},
        "mot":      {"steps": [], "metrics": []},
    }

    model.train()
    pbar = tqdm(total=steps, desc=desc)
    step = 0
    running = []

    while step < steps:
        for v1, v2 in train_loader:
            if step >= steps:
                break

            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)

            optimizer.zero_grad()
            loss = model(v1, v2)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            running.append(loss.item())
            step += 1
            pbar.update(1)
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            })

            if step % log_every == 0:
                history["loss"]["steps"].append(step)
                history["loss"]["values"].append(float(np.mean(running)))
                running = []
                _save_history(history, history_path)

            if val_loader is not None and val_every and step % val_every == 0:
                vl = evaluate_ssl_loss(model, val_loader, device, val_max_batches)
                history["val_loss"]["steps"].append(step)
                history["val_loss"]["values"].append(vl)
                _save_history(history, history_path)

            if mot_eval_fn is not None and mot_eval_every and step % mot_eval_every == 0:
                metrics = mot_eval_fn(step)
                history["mot"]["steps"].append(step)
                history["mot"]["metrics"].append(metrics)
                model.train()
                _save_history(history, history_path)

            if checkpoint_fn is not None and checkpoint_every and step % checkpoint_every == 0:
                checkpoint_fn(step)

    pbar.close()
    _save_history(history, history_path)
    return history


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    dataset = AugmentedInstanceDataset(dataset_dir="VisDrone2019-MOT-train")
    print(f"Dataset size: {len(dataset)} instances")

    backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model = SelfSupervisedInstanceEmbedder(backbone).to(device)

    optimizer = Adam(model.parameters(), lr=1e-3)

    train_loader = dataset.get_batch_loader(batch_size=64, num_workers=0)
    train_self_supervised(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        steps=1000,
        device=device,
    )

    checkpoint_path = "ssl_embedding_model.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

    detection_model = YOLO('mshamrai/yolov8n-visdrone')
    detection_model.overrides['conf'] = 0.25
    detection_model.overrides['iou'] = 0.45
    detection_model.overrides['agnostic_nms'] = False
    detection_model.overrides['max_det'] = 1000

    track = ROIByteTrack(model=detection_model,
                         reid_model=model,
                         device=device)

    mot_metricks_path = "detection_metrics.txt"

    track.process_tracking("VisDrone2019-MOT-test-dev/sequences/uav0000009_03358_v",
                           "output_images_tracked",
                           mot_metricks_path,
                           use_roi=True,
                           roi_coef=0.5)

    track.evaluate_mot("VisDrone2019-MOT-test-dev/annotations/uav0000009_03358_v.txt",
                       mot_metricks_path,
                       verbose=True)
