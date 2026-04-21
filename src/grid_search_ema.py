import os
import logging
import torch
import motmetrics as mm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
from ultralyticsplus import YOLO
from torchvision.models import resnet18, ResNet18_Weights

from roi_bytetrack import ROIByteTrack

logging.basicConfig(level=logging.INFO)

EMA_COEFS = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95]

BACKBONE_NAME = "resnet18"
ROI_COEF = 0.4
YOLO_MODEL = "mshamrai/yolov8n-visdrone"

SEQUENCES_ROOT = "../task1_2/VisDrone2019-MOT-test-dev/sequences"
ANNOTATIONS_ROOT = "../task1_2/VisDrone2019-MOT-test-dev/annotations"
RESULTS_DIR = "grid_search_ema_results"
PLOTS_DIR = "grid_search_ema_plots"


def run_experiment(ema_coef, device):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model = YOLO(YOLO_MODEL)
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000

    reid_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    reid_model.fc = torch.nn.Identity()
    reid_model.to(device).eval()

    sequences = sorted(os.listdir(SEQUENCES_ROOT))
    all_accs = []
    all_names = []

    for seq_name in sequences:
        track = ROIByteTrack(model=model, reid_model=reid_model, device=device)

        seq_path = os.path.join(SEQUENCES_ROOT, seq_name)
        ann_path = os.path.join(ANNOTATIONS_ROOT, seq_name + ".txt")
        mot_path = os.path.join(RESULTS_DIR, f"det_{BACKBONE_NAME}_ema{ema_coef}_{seq_name}.txt")

        track.process_traching(
            seq_path, None, mot_path,
            use_roi=True,
            roi_coef=ROI_COEF,
            ema_coef=ema_coef,
            verbose=False,
        )

        if os.path.exists(ann_path):
            _, acc = track.evaluate_mot(ann_path, mot_path, verbose=False)
            all_accs.append(acc)
            all_names.append(seq_name)

    return all_accs, all_names


def save_plots(results: dict):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    keys = [f"ema={c}" for c in EMA_COEFS]
    x = np.arange(len(keys))

    # --- bar: MOTA & IDF1 ---
    motas = [results[k]["mota"] for k in keys if k in results]
    idf1s = [results[k]["idf1"] for k in keys if k in results]
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 1.2), 5))
    ax.bar(x - width / 2, motas, width, label="MOTA")
    ax.bar(x + width / 2, idf1s, width, label="IDF1")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, fontsize=10)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title(f"MOTA vs IDF1 — {BACKBONE_NAME}, roi={ROI_COEF}, EMA sweep")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "bar_mota_idf1.png"), dpi=150)
    plt.close(fig)

    # --- line: MOTA, IDF1, IDs vs EMA ---
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    metrics = [
        ("mota", "MOTA", True),
        ("idf1", "IDF1", True),
        ("num_switches", "ID Switches", False),
    ]
    for ax, (metric, label, as_pct) in zip(axes, metrics):
        vals = [results[k][metric] for k in keys if k in results]
        ax.plot(EMA_COEFS, vals, marker="o")
        ax.set_ylabel(label)
        if as_pct:
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(True, linestyle="--", alpha=0.5)
        for xi, yi in zip(EMA_COEFS, vals):
            lbl = f"{yi:.1%}" if as_pct else f"{int(yi)}"
            ax.annotate(lbl, (xi, yi), textcoords="offset points", xytext=(0, 6), fontsize=8, ha="center")
    axes[-1].set_xlabel("EMA coefficient")
    fig.suptitle(f"Metrics vs EMA — {BACKBONE_NAME}, roi={ROI_COEF}", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "line_metrics_vs_ema.png"), dpi=150)
    plt.close(fig)

    # --- bar: ID Switches ---
    ids_vals = [results[k]["num_switches"] for k in keys if k in results]
    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 1.2), 4))
    ax.bar(x, ids_vals)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, fontsize=10)
    ax.set_title("ID Switches per EMA coefficient")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "bar_id_switches.png"), dpi=150)
    plt.close(fig)

    print(f"\nPlots saved to '{PLOTS_DIR}/'")


def _print_summary(results: dict):
    col = 7
    keys = [f"ema={c}" for c in EMA_COEFS]

    def _row(key):
        r = results[key]
        return (f"{r['mota']:>{col}.1%}  {r['motp']:>{col}.3f}  {r['idf1']:>{col}.1%}"
                f"  {int(r['mostly_tracked']):>4}  {int(r['mostly_lost']):>4}  {int(r['num_switches']):>5}")

    header = f"{'':>12}  {'MOTA':>{col}}  {'MOTP':>{col}}  {'IDF1':>{col}}  {'MT':>4}  {'ML':>4}  {'IDs':>5}"
    sep = "-" * len(header)

    print("\n" + "=" * len(header))
    print(f"EMA GRID SEARCH — backbone={BACKBONE_NAME}  roi_coef={ROI_COEF}  model={YOLO_MODEL}")
    print("=" * len(header))
    print(f"  {header}")
    print(f"  {sep}")
    for key in keys:
        if key in results:
            print(f"  {key:<12}  {_row(key)}")

    print("\n" + "=" * len(header))
    print("FULL RANKING (sorted by MOTA)")
    print("=" * len(header))
    print(f"  {header}")
    print(f"  {sep}")
    for key, _ in sorted(results.items(), key=lambda x: x[1]['mota'], reverse=True):
        print(f"  {key:<12}  {_row(key)}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics_list = ['mota', 'motp', 'idf1', 'mostly_tracked', 'mostly_lost', 'num_switches']
    mh = mm.metrics.create()

    results = {}

    for ema_coef in EMA_COEFS:
        exp_key = f"ema={ema_coef}"
        print(f"\n{'='*60}")
        print(f"Running: {BACKBONE_NAME} | roi={ROI_COEF} | {exp_key}")
        print('='*60)

        all_accs, all_names = run_experiment(ema_coef, device)

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
