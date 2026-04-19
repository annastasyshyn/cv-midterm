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
            A.CoarseDropout(max_holes=4, max_height=0.125, max_width=0.125, p=0.3),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    return make(), make()


class AugmentedInstanceDataset(Dataset):
    def __init__(self, dataset_dir, crop_size=(256, 128), aug_strength='strong'):
        self.dataset_dir = dataset_dir
        self.crop_size = crop_size
        
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
    
    def get_batch_loader(self, batch_size=32, num_workers=4):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )


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
        feat_dim: int = 2048,
        embed_dim: int = 128,
        temperature: float = 0.2,
    ) -> None:
        super().__init__()
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


def train_self_supervised(
    model,
    dataset,
    optimizer,
    steps=1000,
    batch_size=32,
    device="cuda",
):
    model.train()
    dataloader = dataset.get_batch_loader(batch_size=batch_size, num_workers=0)
    
    pbar = tqdm(total=steps, desc="self-supervised training")
    step = 0
    
    while step < steps:
        for view1, view2 in dataloader:
            if step >= steps:
                break
            
            view1 = view1.to(device)
            view2 = view2.to(device)
            
            optimizer.zero_grad()
            loss = model(view1, view2)
            loss.backward()
            optimizer.step()
            
            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            step += 1
    
    pbar.close()


if __name__ == "__main__":
    ## tests in Bohdan, ask if needed

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    

    dataset = AugmentedInstanceDataset(dataset_dir="VisDrone2019-MOT-train")
    print(f"Dataset size: {len(dataset)} instances")
    

    backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model = SelfSupervisedInstanceEmbedder(backbone).to(device)
    
    optimizer = Adam(model.parameters(), lr=1e-3)
    
    train_self_supervised(
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        steps=200,
        batch_size=64,
        device=device,
    )
    
    checkpoint_path = "triplet_embedding_model.pth"
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
