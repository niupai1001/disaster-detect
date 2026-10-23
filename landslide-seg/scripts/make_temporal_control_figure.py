"""时间控制四组的曲线与指标图；读取 summary_temporal_control.json。"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/segformer/v4r1_temporal_control"
ASSETS = ROOT / "docs/assets/research"
ORDER = ["single", "mean", "content", "date", "history"]
LABEL = {
    "single": "single frame (T=1)",
    "mean": "15 frames, mean",
    "content": "15 frames, content weights",
    "date": "15 frames, content+date",
    "history": "up to ref. date, content+date",
}
COLOR = {"single": "#333333", "mean": "#1f77b4", "content": "#2ca02c", "date": "#d62728",
         "history": "#9467bd"}
BASELINE = 0.30813804525753863


def curves(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key in ORDER:
        run = data["runs"].get(key, {})
        pts = [(p["step"], p["event_mean_iou"]) for p in run.get("curve", [])
               if p.get("event_mean_iou") is not None]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, label=LABEL[key], color=COLOR[key], lw=1.6)
    ax.axhline(BASELINE, color="#888888", ls="--", lw=1, label="reference single-date run (batch64)")
    ax.set_xlabel("successful updates")
    ax.set_ylabel("DEV event-mean IoU")
    ax.set_title("Our data split (event-held-out): 2223 train / 669 dev, seed 1234")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "temporal_control_curves.png", dpi=170)
    plt.close(fig)


def metrics(data: dict) -> None:
    ev = {k: data["runs"].get(k, {}).get("evals", {}).get("best") for k in ORDER}
    have = [k for k in ORDER if ev.get(k)]
    if not have:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4))
    x = np.arange(len(have))
    width = 0.38
    for i, event in enumerate(("hiroshima", "thrissur")):
        vals = [ev[k]["by_event_iou"].get(event, 0) for k in have]
        axes[0].bar(x + (i - 0.5) * width, vals, width, label=event)
    axes[0].set_xticks(x, [LABEL[k] for k in have], rotation=20, fontsize=7)
    axes[0].set_ylabel("IoU")
    axes[0].set_title("Per-event IoU at best checkpoint")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    bins = ["1_4", "5_9", "10_19", "20_49", "50_199", "ge200"]
    for key in have:
        vals = []
        for b in bins:
            rec = ev[key].get("fine_bins", {}).get(b) or {}
            vals.append(rec.get("object_recall_evaluable"))
        axes[1].plot(range(len(bins)), [np.nan if v is None else v for v in vals],
                     marker="o", ms=3, label=LABEL[key], color=COLOR[key])
    axes[1].set_xticks(range(len(bins)), ["1-4", "5-9", "10-19", "20-49", "50-199", ">=200"])
    axes[1].set_xlabel("object area (pixels)")
    axes[1].set_ylabel("object recall (IoU>=0.25)")
    axes[1].set_title("Object recall by size at best checkpoint")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "temporal_control_metrics.png", dpi=170)
    plt.close(fig)


def main() -> None:
    data = json.loads((OUT / "summary_temporal_control.json").read_text(encoding="utf-8"))
    ASSETS.mkdir(parents=True, exist_ok=True)
    curves(data)
    metrics(data)
    print("wrote temporal_control_curves.png / temporal_control_metrics.png")


if __name__ == "__main__":
    main()
