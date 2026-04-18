import os
import cv2
import numpy as np
import pandas as pd
import motmetrics as mm
import supervision as sv
from torchvision.ops import roi_align

# from ultralytics import YOLO
from ultralyticsplus import YOLO

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

from roi_bytetrack import ROIByteTrack


def build_rois_xyxy(boxes_xyxy: torch.Tensor) -> torch.Tensor:
    b = boxes_xyxy.shape[0]
    batch_idx = torch.arange(
        b, device=boxes_xyxy.device, dtype=boxes_xyxy.dtype
    ).unsqueeze(1)
    return torch.cat([batch_idx, boxes_xyxy], dim=1)


class TripletDataset(Dataset):
    def __init__(self, dataset_dir, max_frame_delta=5, transform=None):
        self.dataset_dir = dataset_dir
        self.max_frame_delta = max_frame_delta
        self.triplets = []

        self.frame_transform = transform or T.Compose(
            [
                T.ToPILImage(),
                T.Resize((720, 1280)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.build_triplets()

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        seq = triplet["seq"]
        seq_path = os.path.join(self.dataset_dir, "sequences", seq)

        anchor_img = self.load_frame(seq_path, triplet["anchor_frame"])
        positive_img = self.load_frame(seq_path, triplet["positive_frame"])
        negative_img = self.load_frame(seq_path, triplet["negative_frame"])

        anchor = self.frame_transform(anchor_img)
        positive = self.frame_transform(positive_img)
        negative = self.frame_transform(negative_img)

        anchor_box = torch.tensor(triplet["anchor_bbox"], dtype=torch.float32)
        positive_box = torch.tensor(triplet["positive_bbox"], dtype=torch.float32)
        negative_box = torch.tensor(triplet["negative_bbox"], dtype=torch.float32)

        return anchor, anchor_box, positive, positive_box, negative, negative_box

    def build_triplets(self):
        sequences_dir = os.path.join(self.dataset_dir, "sequences")
        annotations_dir = os.path.join(self.dataset_dir, "annotations")

        for seq in sorted(os.listdir(sequences_dir)):
            seq_path = os.path.join(sequences_dir, seq)
            if not os.path.isdir(seq_path):
                continue

            anno_file = os.path.join(annotations_dir, seq + ".txt")
            if not os.path.exists(anno_file):
                continue

            annotations = self.parse_annotations(anno_file)

            for track_id, frames_data in annotations.items():
                sorted_frames = sorted(frames_data.keys())

                for i, frame_idx in enumerate(sorted_frames):
                    for j in range( # TODO: sample because too namy triplets n^2
                        i + 1, min(i + self.max_frame_delta + 1, len(sorted_frames))
                    ):
                        pos_frame = sorted_frames[j]
                        other_tracks = [
                            tid
                            for tid in annotations.keys()
                            if tid != track_id and frame_idx in annotations[tid]
                        ]

                        if other_tracks:
                            neg_track = np.random.choice(other_tracks)
                            if frame_idx in annotations[neg_track]:
                                neg_frame = frame_idx
                            else:
                                neg_frame = np.random.choice(list(annotations[neg_track].keys()))

                            # TODO: hard negatives mining 

                            self.triplets.append(
                                {
                                    "seq": seq,
                                    "anchor_frame": frame_idx,
                                    "anchor_track": track_id,
                                    "anchor_bbox": frames_data[frame_idx],
                                    "positive_frame": pos_frame,
                                    "positive_track": track_id,
                                    "positive_bbox": frames_data[pos_frame],
                                    "negative_frame": neg_frame,
                                    "negative_track": neg_track,
                                    "negative_bbox": annotations[neg_track][neg_frame],
                                }
                            )

    def parse_annotations(self, anno_file):
        annotations = {}
        with open(anno_file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                frame_id = int(parts[0])
                track_id = int(parts[1])
                x, y, w, h = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
                x1, y1 = x, y
                x2, y2 = x + w, y + h

                if track_id not in annotations:
                    annotations[track_id] = {}
                annotations[track_id][frame_id] = (x1, y1, x2, y2)

        return annotations

    def load_frame(self, seq_path, frame_id):
        img_name = f"{frame_id:07d}.jpg"
        img_path = os.path.join(seq_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


class EmbeddingModel(nn.Module):
    def __init__(self, base, out_dim=128, roi_out=(7, 7), roi_spatial_scale=1.0 / 32.0):
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

        self.roi_output_size = roi_out
        self.roi_spatial_scale = roi_spatial_scale

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048 * roi_out[0] * roi_out[1], 512),
            nn.ReLU(),
            nn.Linear(512, out_dim),
        )

    def forward(self, images, rois):
        feat = self.backbone(images)  # [B, 2048, H/32, W/32]

        roi_feat = roi_align(
            input=feat,
            boxes=rois,
            output_size=self.roi_output_size,
            spatial_scale=self.roi_spatial_scale,
            sampling_ratio=2,
            aligned=True,
        )  # [N, 2048, 7, 7]

        emb = self.head(roi_feat)  # [N, out_dim]
        emb = nn.functional.normalize(emb, p=2, dim=1)
        return emb


def train_triplet(model, dataloader, optimizer, margin=1.0, device="cuda"):
    model.train()
    criterion = nn.TripletMarginLoss(margin=margin)
    for batch in tqdm(dataloader, desc="triplet"):
        anchor_img, anchor_box, pos_img, pos_box, neg_img, neg_box = batch

        anchor_img = anchor_img.to(device)
        pos_img = pos_img.to(device)
        neg_img = neg_img.to(device)

        anchor_box = anchor_box.to(device)
        pos_box = pos_box.to(device)
        neg_box = neg_box.to(device)

        anchor_rois = build_rois_xyxy(anchor_box)
        pos_rois = build_rois_xyxy(pos_box)
        neg_rois = build_rois_xyxy(neg_box)

        optimizer.zero_grad()

        anchor_emb = model(anchor_img, anchor_rois)
        pos_emb = model(pos_img, pos_rois)
        neg_emb = model(neg_img, neg_rois)

        loss = criterion(anchor_emb, pos_emb, neg_emb)
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = YOLO("mshamrai/yolov8n-visdrone")  # model for detection
    model.overrides["conf"] = 0.25
    model.overrides["iou"] = 0.45
    model.overrides["agnostic_nms"] = False
    model.overrides["max_det"] = 1000

    # training embedding model
    embedding_model = EmbeddingModel(resnet50(weights=ResNet50_Weights.IMAGENET1K_V1))
    embedding_model.to(device)
    optimizer = torch.optim.Adam(embedding_model.parameters(), lr=1e-4)
    dataset = TripletDataset(dataset_dir="VisDrone2019-MOT-val", max_frame_delta=5)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    train_triplet(embedding_model, dataloader, optimizer, margin=1.0, device=device)

    triplet_model_path = "triplet_embedding_model.pth"
    torch.save(embedding_model.state_dict(), triplet_model_path)

    reid_model = resnet50(
        weights=ResNet50_Weights.IMAGENET1K_V1
    )  # model for feature extraction
    reid_model.fc = torch.nn.Identity()
    reid_model.to(device).eval()

    track = ROIByteTrack(
        model=model,
        #  reid_model=reid_model,
        reid_model=embedding_model,
        device=device,
    )

    mot_metricks_path = "detection_metrics_sml.txt"  # path where mot metricks will be stored, maybe will rework this

    track.process_tracking(
        # "../task1_2/VisDrone2019-MOT-test-dev/sequences/uav0000009_03358_v", # path to dataset image sequence
        "VisDrone2019-MOT-val/sequences/uav0000086_00000_v",  # path to dataset image sequence
        "output_images_tracked",  # path to output tracked objects on the image
        mot_metricks_path,
        use_roi=True,
        roi_coef=0.25,
    )

    track.evaluate_mot(
        # "../task1_2/VisDrone2019-MOT-test-dev/annotations/uav0000009_03358_v.txt", #path to annotations
        "VisDrone2019-MOT-val/annotations/uav0000086_00000_v.txt",  # path to annotations
        mot_metricks_path,
        verbose=True,
    )
