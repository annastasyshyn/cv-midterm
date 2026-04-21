import os
import logging
import torch
import motmetrics as mm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
from ultralyticsplus import YOLO
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights,
    resnet101, ResNet101_Weights,
)

from roi_bytetrack import ROIByteTrack

logging.basicConfig(level=logging.INFO)

BACKBONES = {
    "resnet18":  lambda: resnet18(weights=ResNet18_Weights.IMAGENET1K_V1),
    "resnet50":  lambda: resnet50(weights=ResNet50_Weights.IMAGENET1K_V1),
    "resnet101": lambda: resnet101(weights=ResNet101_Weights.IMAGENET1K_V1),
}

ROI_COEFS = [0.1, 0.2, 0.4]

YOLO_MODEL = "mshamrai/yolov8n-visdrone"

SEQUENCES_ROOT = "../task1_2/VisDrone2019-MOT-test-dev/sequences"
ANNOTATIONS_ROOT = "../task1_2/VisDrone2019-MOT-test-dev/annotations"
RESULTS_DIR = "grid_search_roi_results"
PLOTS_DIR = "grid_search_roi_plots"


def run_experiment(backbone_name, roi_coef, device):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model = YOLO(YOLO_MODEL)
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000

    reid_model = BACKBONES[backbone_name]()
    reid_model.fc = torch.nn.Identity()
    reid_model.to(device).eval()

    sequences = sorted(os.listdir(SEQUENCES_ROOT))
    all_accs = []
    all_names = []

    for seq_name in sequences:
        track = ROIByteTrack(model=model, reid_model=reid_model, device=device)

        seq_path = os.path.join(SEQUENCES_ROOT, seq_name)
        ann_path = os.path.join(ANNOTATIONS_ROOT, seq_name + ".txt")
        mot_path = os.path.join(RESULTS_DIR, f"det_{backbone_name}_roi{roi_coef}_{seq_name}.txt")

        track.process_traching(
            seq_path, None, mot_path,
            use_roi=True,
            roi_coef=roi_coef,
            ema_coef=None,
            verbose=False,
        )

        if os.path.exists(ann_path):
            _, acc = track.evaluate_mot(ann_path, mot_path, verbose=False)
            all_accs.append(acc)
            all_names.append(seq_name)

    return all_accs, all_names


def save_plots(results: dict):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    backbone_names = list(BACKBONES.keys())
    roi_labels = [f"roi={c}" for c in ROI_COEFS]

    for metric, label, fmt in [
        ("mota",         "MOTA",        "{:.1%}"),
        ("idf1",         "IDF1",        "{:.1%}"),
        ("num_switches", "ID Switches", "{:.0f}"),
    ]:
        data = np.zeros((len(backbone_names), len(ROI_COEFS)))
        for i, bb in enumerate(backbone_names):
            for j, coef in enumerate(ROI_COEFS):
                key = f"{bb} | roi={coef}"
                data[i, j] = results[key][metric] if key in results else np.nan

        fig, ax = plt.subplots(figsize=(len(ROI_COEFS) * 3, len(backbone_names) * 1.5 + 1))
        im = ax.imshow(data, aspect="auto", cmap="RdYlGn" if metric != "num_switches" else "RdYlGn_r")
        ax.set_xticks(range(len(ROI_COEFS)))
        ax.set_xticklabels(roi_labels, fontsize=10)
        ax.set_yticks(range(len(backbone_names)))
        ax.set_yticklabels(backbone_names)
        for i in range(len(backbone_names)):
            for j in range(len(ROI_COEFS)):
                ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center", fontsize=10)
        plt.colorbar(im, ax=ax)
        ax.set_title(f"{label} — backbone × ROI coefficient")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, f"heatmap_{metric}.png"), dpi=150)
        plt.close(fig)

    exp_keys = [f"{bb} | roi={c}" for bb in backbone_names for c in ROI_COEFS]
    motas = [results[k]["mota"] for k in exp_keys if k in results]
    idf1s = [results[k]["idf1"] for k in exp_keys if k in results]
    x = np.arange(len(exp_keys))
    width = 0.4

    fig, ax = plt.subplots(figsize=(max(10, len(exp_keys) * 1.4), 5))
    ax.bar(x - width / 2, motas, width, label="MOTA")
    ax.bar(x + width / 2, idf1s, width, label="IDF1")
    ax.set_xticks(x)
    ax.set_xticklabels(exp_keys, rotation=30, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("MOTA vs IDF1 per experiment")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "bar_mota_idf1.png"), dpi=150)
    plt.close(fig)

    ids_vals = [results[k]["num_switches"] for k in exp_keys if k in results]
    fig, ax = plt.subplots(figsize=(max(10, len(exp_keys) * 1.4), 4))
    ax.bar(x, ids_vals)
    ax.set_xticks(x)
    ax.set_xticklabels(exp_keys, rotation=30, ha="right", fontsize=8)
    ax.set_title("ID Switches per experiment")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "bar_id_switches.png"), dpi=150)
    plt.close(fig)

    print(f"\nPlots saved to '{PLOTS_DIR}/'")


def _print_summary(results: dict):
    backbone_names = list(BACKBONES.keys())
    col = 7

    def _row(key):
        r = results[key]
        return (f"{r['mota']:>{col}.1%}  {r['motp']:>{col}.3f}  {r['idf1']:>{col}.1%}"
                f"  {int(r['mostly_tracked']):>4}  {int(r['mostly_lost']):>4}  {int(r['num_switches']):>5}")

    header = f"{'':>20}  {'MOTA':>{col}}  {'MOTP':>{col}}  {'IDF1':>{col}}  {'MT':>4}  {'ML':>4}  {'IDs':>5}"
    sep = "-" * len(header)

    # --- per backbone ---
    print("\n" + "=" * len(header))
    print("RESULTS BY BACKBONE")
    print("=" * len(header))
    for bb in backbone_names:
        print(f"\n  Backbone: {bb}")
        print(f"  {header}")
        print(f"  {sep}")
        for coef in ROI_COEFS:
            key = f"{bb} | roi={coef}"
            if key in results:
                print(f"  {'ROI coef=' + str(coef):<20}  {_row(key)}")

    # --- per ROI coef ---
    print("\n" + "=" * len(header))
    print("RESULTS BY ROI COEFFICIENT")
    print("=" * len(header))
    for coef in ROI_COEFS:
        print(f"\n  ROI coefficient: {coef}")
        print(f"  {header}")
        print(f"  {sep}")
        for bb in backbone_names:
            key = f"{bb} | roi={coef}"
            if key in results:
                print(f"  {bb:<20}  {_row(key)}")

    # --- full ranking ---
    print("\n" + "=" * len(header))
    print("FULL RANKING (sorted by MOTA)")
    print("=" * len(header))
    print(f"  {'Experiment':<40}  {header.strip()}")
    print(f"  {sep}")
    for key, _ in sorted(results.items(), key=lambda x: x[1]['mota'], reverse=True):
        print(f"  {key:<40}  {_row(key)}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics_list = ['mota', 'motp', 'idf1', 'mostly_tracked', 'mostly_lost', 'num_switches']
    mh = mm.metrics.create()

    results = {}

    for backbone_name in BACKBONES:
        for roi_coef in ROI_COEFS:
            exp_key = f"{backbone_name} | roi={roi_coef}"
            print(f"\n{'='*60}")
            print(f"Running: {exp_key}")
            print('='*60)

            all_accs, all_names = run_experiment(backbone_name, roi_coef, device)

            if all_accs:
                overall = mh.compute_many(
                    all_accs, names=all_names,
                    metrics=metrics_list, generate_overall=True,
                )
                str_overall = mm.io.render_summary(
                    overall,
                    formatters=mh.formatters,
                    namemap=mm.io.motchallenge_metric_names,
                )
                print(str_overall)
                results[exp_key] = overall.loc["OVERALL"]

    _print_summary(results)

    if results:
        save_plots(results)


if __name__ == "__main__":
    main()
