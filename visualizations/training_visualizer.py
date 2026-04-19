"""
Renders training history produced by `train_self_supervised` or `train_triplet`.

History schema (JSON on disk):
  {
    "mode": "train" | "test" | null,
    "loss": {"steps": [...], "values": [...]},
    "mot":  {"steps": [...], "metrics": [{"mota": ..., "idf1": ..., ...}, ...]}
  }

Two data sources per run:
  - loss curve (training on data.train.root if mode=train, finetuning on
    data.val.root if mode=test)
  - MOT metrics on data.test.mot_* (only populated in mode=test when
    testing.mot_eval_every > 0)

Usage:
  python visualizations/training_visualizer.py /working/cv-midterm/ssl_history_train.json \
      --save-dir /working/cv-midterm/plots
"""

import argparse
import json
import os
from typing import Iterable, Optional, Union

import matplotlib.pyplot as plt


class TrainingVisualizer:
    DEFAULT_MOT_KEYS = (
        "mota",
        "idf1",
        "motp",
        "num_switches",
        "mostly_tracked",
        "mostly_lost",
    )

    def __init__(self, history: Union[str, os.PathLike, dict]):
        if isinstance(history, (str, os.PathLike)):
            with open(history, "r") as f:
                history = json.load(f)
        self.history = history

    @classmethod
    def from_file(cls, path: Union[str, os.PathLike]) -> "TrainingVisualizer":
        return cls(path)

    def plot_losses(self, ax=None, save_path: Optional[str] = None, show: bool = False):
        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure

        loss = self.history.get("loss", {})
        mode = self.history.get("mode")
        label, title = self._loss_labels(mode)

        if loss.get("steps"):
            ax.plot(loss["steps"], loss["values"], label=label, color="#1f77b4", linewidth=1.5)
            ax.legend()

        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        if save_path and created:
            fig.tight_layout()
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"saved {save_path}")
        if show and created:
            plt.show()
        return ax

    @staticmethod
    def _loss_labels(mode):
        if mode == "train":
            return "pretrain (train set)", "Pretraining loss"
        if mode == "test":
            return "finetune (val set)", "Finetuning loss"
        return "loss", "Training loss"

    def plot_mot_metrics(
        self,
        metrics: Optional[Iterable[str]] = None,
        save_path: Optional[str] = None,
        show: bool = False,
    ):
        keys = list(metrics or self.DEFAULT_MOT_KEYS)
        mot = self.history.get("mot", {})
        steps = mot.get("steps", [])
        series = mot.get("metrics", [])

        if not steps:
            print("no MOT checkpoints in history — skipping MOT plot")
            return None

        present = [k for k in keys if any(s.get(k) is not None for s in series)]
        if not present:
            print("no recognized MOT metrics in history — skipping MOT plot")
            return None

        cols = 2 if len(present) > 1 else 1
        rows = (len(present) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.5 * rows), squeeze=False)

        for i, m in enumerate(present):
            ax = axes[i // cols][i % cols]
            values = [s.get(m) for s in series]
            xs = [st for st, v in zip(steps, values) if v is not None]
            ys = [v for v in values if v is not None]
            ax.plot(xs, ys, marker="o", color=f"C{i % 10}", linewidth=1.8)
            ax.set_xlabel("step")
            ax.set_ylabel(m)
            ax.set_title(m)
            ax.grid(True, alpha=0.3)

        for j in range(len(present), rows * cols):
            axes[j // cols][j % cols].set_visible(False)

        fig.suptitle("MOT metrics at validation checkpoints", y=1.02, fontsize=13)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"saved {save_path}")
        if show:
            plt.show()
        return fig

    def plot_all(self, save_dir: Optional[str] = None, show: bool = False):
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            loss_path = os.path.join(save_dir, "losses.png")
            mot_path = os.path.join(save_dir, "mot_metrics.png")
        else:
            loss_path = mot_path = None

        self.plot_losses(save_path=loss_path, show=show)
        self.plot_mot_metrics(save_path=mot_path, show=show)

    def summary(self) -> dict:
        """Compact text summary — useful for logging."""
        out = {"mode": self.history.get("mode")}
        loss = self.history.get("loss", {})
        if loss.get("values"):
            out["loss_final"] = loss["values"][-1]
            out["loss_min"] = min(loss["values"])
        mot = self.history.get("mot", {})
        if mot.get("metrics"):
            last = mot["metrics"][-1]
            for k in ("mota", "idf1", "motp"):
                if k in last:
                    out[f"{k}_last"] = last[k]
        return out


def main():
    parser = argparse.ArgumentParser(description="Plot SSL training history.")
    parser.add_argument("history", type=str, help="path to ssl_history.json")
    parser.add_argument("--save-dir", type=str, default=None, help="write figures here")
    parser.add_argument("--show", action="store_true", help="open an interactive window")
    args = parser.parse_args()

    viz = TrainingVisualizer.from_file(args.history)
    viz.plot_all(save_dir=args.save_dir, show=args.show)

    s = viz.summary()
    if s:
        print("summary:", json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
