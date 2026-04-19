import os
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
from src.metric import TripletDataset
import numpy as np

dataset_dir = "VisDrone2019-MOT-val"

triplet_dataset = TripletDataset(dataset_dir)

n_ids = 4
k_per_id = 3
max_k = 10


def sample_triplets(dataset, n_ids=4, max_k=10):
    n_avail = len(dataset.identity_keys)
    chosen = torch.randperm(n_avail)[:n_ids]
    triplets = []
    for key_idx in chosen:
        key = dataset.identity_keys[key_idx]
        entries = dataset.identity_index[key]
        if len(entries) < 2:
            continue
        anchor_i = torch.randint(len(entries), (1,)).item()
        pos_candidates = [i for i in range(len(entries)) if i != anchor_i]
        positive_i = pos_candidates[torch.randint(len(pos_candidates), (1,)).item()]
        neg_keys = [k for k in dataset.identity_keys if k != key and len(dataset.identity_index[k]) > 0]
        neg_key = neg_keys[torch.randint(len(neg_keys), (1,)).item()]
        neg_entries = dataset.identity_index[neg_key]
        negative_i = torch.randint(len(neg_entries), (1,)).item()
        def get_crop(entries, idx, seq, crop_transform):
            frame_id, bbox = entries[idx]
            seq_path = os.path.join(dataset.dataset_dir, "sequences", seq)
            frame = dataset.load_frame(seq_path, frame_id)
            x1, y1, x2, y2 = map(int, bbox)
            h, w = frame.shape[:2]
            raw = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if raw.size == 0:
                raw = np.zeros((dataset.crop_size[0], dataset.crop_size[1], 3), dtype=np.uint8)
            return dataset.crop_transform(raw)
        anchor = get_crop(entries, anchor_i, key[0], dataset.crop_transform)
        positive = get_crop(entries, positive_i, key[0], dataset.crop_transform)
        negative = get_crop(neg_entries, negative_i, neg_key[0], dataset.crop_transform)
        triplets.append((anchor, positive, negative))
    return triplets

triplets = sample_triplets(triplet_dataset, n_ids=n_ids, max_k=max_k)

def visualize_triplets(triplets):
    n = len(triplets)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes.reshape(1, 3)
    for i, (anchor, positive, negative) in enumerate(triplets):
        for j, (img_tensor, title) in enumerate(zip([anchor, positive, negative], ["Anchor", "Positive", "Negative"])):
            img = img_tensor.cpu().permute(1, 2, 0).numpy()
            img = (img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
            img = img.clip(0, 1)
            ax = axes[i, j]
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(title)
    plt.tight_layout()
    plt.savefig("triplet_visualization.png")

if __name__ == "__main__":
    visualize_triplets(triplets)
