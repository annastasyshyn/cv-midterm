import os
import logging
import torch
import motmetrics as mm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from ultralyticsplus import YOLO
from torchvision.models import resnet50, ResNet50_Weights

from roi_bytetrack import ROIByteTrack

logging.basicConfig(level=logging.INFO)

TRACKER_CONFIGS = [
    {"name": "cfg1_act0.40_buf25_match0.60", "track_activation_threshold": 0.40, "lost_track_buffer": 25, "minimum_matching_threshold": 0.60},
    {"name": "cfg2_act0.35_buf30_match0.72", "track_activation_threshold": 0.35, "lost_track_buffer": 30, "minimum_matching_threshold": 0.72},
    {"name": "cfg3_act0.30_buf30_match0.77", "track_activation_threshold": 0.30, "lost_track_buffer": 30, "minimum_matching_threshold": 0.77},
]

YOLO_MODELS = [
    "mshamrai/yolov8n-visdrone",
    "mshamrai/yolov8s-visdrone",
    "mshamrai/yolov8m-visdrone",
    "mshamrai/yolov8l-visdrone",
]

SEQUENCES_ROOT = "../task1_2/VisDrone2019-MOT-test-dev/sequences"
ANNOTATIONS_ROOT = "../task1_2/VisDrone2019-MOT-test-dev/annotations"
RESULTS_DIR = "grid_search_results"
PLOTS_DIR = "grid_search_plots"


def run_experiment(model_name, tracker_cfg, device):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model = YOLO(model_name)
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000

    reid_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    reid_model.fc = torch.nn.Identity()
    reid_model.to(device).eval()

    tracker_params = {k: v for k, v in tracker_cfg.items() if k != "name"}
    model_short = model_name.split("/")[-1]

    sequences = sorted(os.listdir(SEQUENCES_ROOT))
    all_accs = []
    all_names = []

    for seq_name in sequences:
        track = ROIByteTrack(model=model, reid_model=reid_model, device=device, tracker_params=tracker_params)

        seq_path = os.path.join(SEQUENCES_ROOT, seq_name)
        ann_path = os.path.join(ANNOTATIONS_ROOT, seq_name + ".txt")
        mot_path = os.path.join(RESULTS_DIR, f"det_{model_short}_{tracker_cfg['name']}_{seq_name}.txt")

        track.process_traching(seq_path, None, mot_path, use_roi=False, verbose=False)

        if os.path.exists(ann_path):
            _, acc = track.evaluate_mot(ann_path, mot_path, verbose=False)
            all_accs.append(acc)
            all_names.append(seq_name)

    return all_accs, all_names


def save_plots(results: dict):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    model_names = [m.split("/")[-1] for m in YOLO_MODELS]
    cfg_names = [c["name"] for c in TRACKER_CONFIGS]

    # --- 1. Heatmaps: MOTA / IDF1 / IDs ---
    for metric, label, fmt in [
        ("mota", "MOTA", "{:.1%}"),
        ("idf1", "IDF1", "{:.1%}"),
        ("num_switches", "ID Switches", "{:.0f}"),
    ]:
        data = np.zeros((len(model_names), len(cfg_names)))
        for i, m in enumerate(model_names):
            for j, c in enumerate(cfg_names):
                key = f"{m} | {c}"
                data[i, j] = results[key][metric] if key in results else np.nan

        fig, ax = plt.subplots(figsize=(len(cfg_names) * 3, len(model_names) * 1.5 + 1))
        im = ax.imshow(data, aspect="auto", cmap="RdYlGn" if metric != "num_switches" else "RdYlGn_r")
        ax.set_xticks(range(len(cfg_names)))
        ax.set_xticklabels(cfg_names, rotation=20, ha="right", fontsize=8)
        ax.set_yticks(range(len(model_names)))
        ax.set_yticklabels(model_names)
        for i in range(len(model_names)):
            for j in range(len(cfg_names)):
                ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center", fontsize=9)
        plt.colorbar(im, ax=ax)
        ax.set_title(f"{label} — models × tracker configs")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, f"heatmap_{metric}.png"), dpi=150)
        plt.close(fig)

    # --- 2. Grouped bar chart: MOTA & IDF1 per experiment ---
    exp_keys = [f"{m} | {c}" for m in model_names for c in cfg_names]
    motas = [results[k]["mota"] for k in exp_keys if k in results]
    idf1s = [results[k]["idf1"] for k in exp_keys if k in results]
    x = np.arange(len(exp_keys))
    width = 0.4

    fig, ax = plt.subplots(figsize=(max(12, len(exp_keys) * 1.2), 5))
    ax.bar(x - width / 2, motas, width, label="MOTA")
    ax.bar(x + width / 2, idf1s, width, label="IDF1")
    ax.set_xticks(x)
    ax.set_xticklabels(exp_keys, rotation=35, ha="right", fontsize=7)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("MOTA vs IDF1 per experiment")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "bar_mota_idf1.png"), dpi=150)
    plt.close(fig)

    # --- 3. ID Switches bar chart ---
    ids_vals = [results[k]["num_switches"] for k in exp_keys if k in results]
    fig, ax = plt.subplots(figsize=(max(12, len(exp_keys) * 1.2), 4))
    ax.bar(x, ids_vals)
    ax.set_xticks(x)
    ax.set_xticklabels(exp_keys, rotation=35, ha="right", fontsize=7)
    ax.set_title("ID Switches per experiment")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "bar_id_switches.png"), dpi=150)
    plt.close(fig)

    print(f"\nPlots saved to '{PLOTS_DIR}/'")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics_list = ['mota', 'motp', 'idf1', 'mostly_tracked', 'mostly_lost', 'num_switches']
    mh = mm.metrics.create()

    results = {}

    for model_name in YOLO_MODELS:
        model_short = model_name.split("/")[-1]
        for tracker_cfg in TRACKER_CONFIGS:
            exp_key = f"{model_short} | {tracker_cfg['name']}"
            print(f"\n{'='*60}")
            print(f"Running: {exp_key}")
            print('='*60)

            all_accs, all_names = run_experiment(model_name, tracker_cfg, device)

            if all_accs:
                overall = mh.compute_many(
                    all_accs, names=all_names,
                    metrics=metrics_list, generate_overall=True
                )
                str_overall = mm.io.render_summary(
                    overall,
                    formatters=mh.formatters,
                    namemap=mm.io.motchallenge_metric_names
                )
                print(str_overall)
                results[exp_key] = overall.loc["OVERALL"]

    _print_summary(results)

    if results:
        save_plots(results)


def _print_summary(results: dict):
    model_names = [m.split("/")[-1] for m in YOLO_MODELS]
    cfg_names   = [c["name"] for c in TRACKER_CONFIGS]
    col = 7

    def _row(key):
        r = results[key]
        return (f"{r['mota']:>{col}.1%}  {r['motp']:>{col}.3f}  {r['idf1']:>{col}.1%}"
                f"  {int(r['mostly_tracked']):>4}  {int(r['mostly_lost']):>4}  {int(r['num_switches']):>5}")

    header = f"{'':>30}  {'MOTA':>{col}}  {'MOTP':>{col}}  {'IDF1':>{col}}  {'MT':>4}  {'ML':>4}  {'IDs':>5}"
    sep    = "-" * len(header)

    # --- per model (all configs) ---
    print("\n" + "=" * len(header))
    print("RESULTS BY MODEL")
    print("=" * len(header))
    for model in model_names:
        print(f"\n  Model: {model}")
        print(f"  {header}")
        print(f"  {sep}")
        for cfg in cfg_names:
            key = f"{model} | {cfg}"
            if key in results:
                print(f"  {cfg:<30}  {_row(key)}")

    # --- per config (all models) ---
    print("\n" + "=" * len(header))
    print("RESULTS BY TRACKER CONFIG")
    print("=" * len(header))
    for cfg in cfg_names:
        print(f"\n  Config: {cfg}")
        print(f"  {header}")
        print(f"  {sep}")
        for model in model_names:
            key = f"{model} | {cfg}"
            if key in results:
                print(f"  {model:<30}  {_row(key)}")

    # --- full sorted ranking ---
    print("\n" + "=" * len(header))
    print("FULL RANKING (sorted by MOTA)")
    print("=" * len(header))
    print(f"  {'Experiment':<55}  {header.strip()}")
    print(f"  {sep}")
    for key, _ in sorted(results.items(), key=lambda x: x[1]['mota'], reverse=True):
        print(f"  {key:<55}  {_row(key)}")


if __name__ == "__main__":
    main()
