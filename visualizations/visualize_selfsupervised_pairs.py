import os
import matplotlib.pyplot as plt
import torch
from src.self_supervised import AugmentedInstanceDataset

dataset_dir = "VisDrone2019-MOT-val"

dataset = AugmentedInstanceDataset(dataset_dir)

n_pairs = 8
indices = torch.randperm(len(dataset))[:n_pairs]
pairs = [dataset[i] for i in indices]

def visualize_pairs(pairs):
    n = len(pairs)
    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    if n == 1:
        axes = axes.reshape(1, 2)
    for i, (view1, view2) in enumerate(pairs):
        for j, (img_tensor, title) in enumerate(zip([view1, view2], ["View 1", "View 2"])):
            img = img_tensor.cpu().permute(1, 2, 0).numpy()
            img = (img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
            img = img.clip(0, 1)
            ax = axes[i, j]
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(title)
    plt.tight_layout()
    plt.savefig("selfsupervised_pairs.png")

if __name__ == "__main__":
    visualize_pairs(pairs)
