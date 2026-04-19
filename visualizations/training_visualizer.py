"""
Renders training history produced by `train_self_supervised` or `train_triplet`.

History schema (JSON on disk):
  {
    "mode":     "train" | "test" | null,
    "loss":     {"steps": [...], "values": [...]},
    "val_loss": {"steps": [...], "values": [...]},
    "mot":      {"steps": [...], "metrics": [{"mota": ..., "idf1": ..., ...}, ...]}
  }

Data sources per run:
  - loss curve (pretraining on data.train.root in mode=train, finetuning on
    data.val.root in mode=test)
  - val_loss curve (mode=test only, held-out InfoNCE on data.test.mot_root)
  - MOT metrics on data.test.mot_* (only populated in mode=test when
    testing.mot_eval_every > 0)

Usage:
  # grid of separate PNGs under save-dir
  python visualizations/training_visualizer.py /workspace/cv-midterm/ssl_history_test.json \
      --save-dir /workspace/cv-midterm/plots

  # one combined PNG
  python visualizations/training_visualizer.py /workspace/cv-midterm/ssl_history_test.json \
      --savefig /workspace/cv-midterm/plots/overview.png
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
        val_loss = self.history.get("val_loss", {})
        mode = self.history.get("mode")
        label, title = self._loss_labels(mode)

        plotted = False
        if loss.get("steps"):
            ax.plot(loss["steps"], loss["values"], label=label, color="#1f77b4", linewidth=1.5)
            plotted = True
        if val_loss.get("steps"):
            ax.plot(
                val_loss["steps"], val_loss["values"],
                label="held-out (test-dev)",
                color="#ff7f0e", marker="o", linewidth=1.8,
            )
            plotted = True

        if plotted:
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

    def savefig(self, path: str, show: bool = False):
        """
        Produce a single combined figure (loss curve on top, MOT metrics grid
        below if present) and save it to `path`. Layout collapses gracefully
        when MOT metrics are missing.
        """
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        mot = self.history.get("mot", {})
        mot_keys = [
            k for k in self.DEFAULT_MOT_KEYS
            if any(s.get(k) is not None for s in mot.get("metrics", []))
        ]
        has_mot = bool(mot_keys) and bool(mot.get("steps"))

        if not has_mot:
            fig, ax_loss = plt.subplots(figsize=(9, 5))
            self.plot_losses(ax=ax_loss)
            fig.tight_layout()
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"saved {path}")
            if show:
                plt.show()
            return fig

        mot_cols = 2 if len(mot_keys) > 1 else 1
        mot_rows = (len(mot_keys) + mot_cols - 1) // mot_cols
        n_rows = 1 + mot_rows

        fig = plt.figure(figsize=(6 * mot_cols, 4 + 3.2 * mot_rows))
        gs = fig.add_gridspec(n_rows, mot_cols)

        ax_loss = fig.add_subplot(gs[0, :])
        self.plot_losses(ax=ax_loss)

        steps = mot["steps"]
        series = mot["metrics"]
        for i, m in enumerate(mot_keys):
            r = 1 + i // mot_cols
            c = i % mot_cols
            ax = fig.add_subplot(gs[r, c])
            values = [s.get(m) for s in series]
            xs = [st for st, v in zip(steps, values) if v is not None]
            ys = [v for v in values if v is not None]
            ax.plot(xs, ys, marker="o", color=f"C{i % 10}", linewidth=1.8)
            ax.set_xlabel("step")
            ax.set_ylabel(m)
            ax.set_title(m)
            ax.grid(True, alpha=0.3)

        mode = self.history.get("mode") or "?"
        fig.suptitle(f"SSL training overview (mode={mode})", y=1.00, fontsize=14)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"saved {path}")
        if show:
            plt.show()
        return fig

    def summary(self) -> dict:
        """Compact text summary — useful for logging."""
        out = {"mode": self.history.get("mode")}
        loss = self.history.get("loss", {})
        if loss.get("values"):
            out["loss_final"] = loss["values"][-1]
            out["loss_min"] = min(loss["values"])
        val = self.history.get("val_loss", {})
        if val.get("values"):
            out["val_loss_final"] = val["values"][-1]
            out["val_loss_min"] = min(val["values"])
        mot = self.history.get("mot", {})
        if mot.get("metrics"):
            last = mot["metrics"][-1]
            for k in ("mota", "idf1", "motp"):
                if k in last:
                    out[f"{k}_last"] = last[k]
        return out


def main():
    parser = argparse.ArgumentParser(description="Plot SSL / metric-learning training history.")
    parser.add_argument("history", type=str, help="path to *_history_*.json")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="write separate losses.png / mot_metrics.png here")
    parser.add_argument("--savefig", type=str, default=None,
                        help="write ONE combined overview figure to this path (.png)")
    parser.add_argument("--show", action="store_true", help="open an interactive window")
    args = parser.parse_args()

    viz = TrainingVisualizer.from_file(args.history)

    if args.savefig:
        viz.savefig(args.savefig, show=args.show)
    if args.save_dir:
        viz.plot_all(save_dir=args.save_dir, show=args.show)
    if not args.savefig and not args.save_dir:
        viz.plot_all(show=args.show or True)

    s = viz.summary()
    if s:
        print("summary:", json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
