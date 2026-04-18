import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2


# def build_contrastive_augmentations(height: int = 256, width: int = 128): ## TODO: Find better ones
#     """
#     Strong, stochastic augmentations for instance contrastive learning.
#     Two independent pipelines are returned -- same recipe but independently
#     sampled per call
#     """
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]

#     def make():
#         return A.Compose([
#             A.Resize(int(height * 1.15), int(width * 1.15)),
#             A.RandomResizedCrop(size=(height, width), scale=(0.6, 1.0), ratio=(0.8, 1.25), p=1.0),
#             A.HorizontalFlip(p=0.5),
#             A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
#             A.ToGray(p=0.2),
#             A.GaussianBlur(blur_limit=(3, 7), p=0.5),
#             A.CoarseDropout(max_holes=4, max_height=32, max_width=16, p=0.3),
#             A.Normalize(mean=mean, std=std),
#             ToTensorV2(),
#         ])

#     return make(), make()


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

    def __init__( ## TODO: generated constants, delete
        self,
        backbone: nn.Module,
        feat_dim: int = 2048,
        embed_dim: int = 128,
        temperature: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, embed_dim),
        )
        self.temperature = temperature

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)
        if f.ndim > 2: ## crutch, delete
            f = torch.flatten(f, 1)
        z = self.projector(f)
        return F.normalize(z, dim=-1)

    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        z1 = self.encode(view1)
        z2 = self.encode(view2)
        return self.contrastive_loss(z1, z2)

    def contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Symmetric InfoNCE over a batch of paired embeddings
        """
        N = z1.size(0)
        logits = (z1 @ z2.t()) / self.temperature ## similarities via dot products, ask Bohdan if you will
        targets = torch.arange(N, device=z1.device)

        ## the paper uses just crossentropy over a dirac function, so turgets are like that
        loss_a = F.cross_entropy(logits, targets)
        loss_b = F.cross_entropy(logits.t(), targets)
        return 0.5 * (loss_a + loss_b)


if __name__ == "__main__":
    ## tests in Bohdan, ask if needed
    ...
