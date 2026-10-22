"""汇总 v6 批次的 DOFA 修正探针、U-TAE 真实日期配对与主模型四组结果。

只读取已保存的 eval JSON / train_summary / 曲线，不重新推理。
生成 ``reports/v6_execution/summary.json`` 与报告用的图。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/v6_execution"
FIG = ROOT / "docs/assets/research"
FOLD_LABEL = {1: "Italy", 2: "Kyrgyzstan", 3: "New Zealand", 4: "Africa/Caribbean"}
MAIN = {
    "single_baseline": ("SegFormer-Temporal-Single-Baseline", "单期参照"),
    "multi_baseline": ("SegFormer-Temporal-Multi15-Baseline", "多期时间聚合"),
    "single_spectral": ("SegFormer-Temporal-Single-Spectral", "单期光谱关系编码"),
    "multi_spectral": ("SegFormer-Temporal-Multi15-Spectral", "多期+光谱关系编码"),
}
# 图内用英文，避免 Windows 上缺少 CJK 字体产生方块；中文解释放在图下。
FIG_LABEL = {
    "single_baseline": "Single frame",
    "multi_baseline": "15 frames",
    "single_spectral": "Single + spectral",
    "multi_spectral": "15 frames + spectral",
}


def load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def pixel(ev: dict) -> dict:
    p = ev["overall"]["pixel"]
    return {"iou": p.get("iou"), "dice": p.get("dice"), "precision": p.get("precision"),
            "recall": p.get("recall"), "ap": p.get("average_precision"),
            "obj_recall": ev["overall"]["objects"].get("object_recall_evaluable"),
            "obj_recall_all": ev["overall"]["objects"].get("object_recall_all"),
            "obj_strict": ev["overall"]["objects"].get("object_recall_strict_evaluable"),
            "boundary_f1": ev["overall"]["boundary"].get("f1"),
            "boundary_precision": ev["overall"]["boundary"].get("precision"),
            "boundary_recall": ev["overall"]["boundary"].get("recall")}


def dofa() -> dict:
    out = {}
    for f in (1, 2, 3, 4):
        corr_dir = ROOT / f"outputs/segformer/event_cv_v6_wave_repair/DOFA-CorrectWavelengths-FrozenProbe-EventFold{f}/1234/eval"
        rgb_dir = ROOT / f"outputs/segformer/event_cv_v5_probe/Probe-dofa_rgb-F{f}/1234/eval"
        row = {"fold": f, "label": FOLD_LABEL[f]}
        for ck in ("best", "last"):
            for split in ("val", "test"):
                c = load(corr_dir / f"eval_{ck}_{split}.json")
                r = load(rgb_dir / f"eval_{ck}_{split}.json")
                row[f"{ck}_{split}"] = {
                    "corrected": pixel(c) if c else None,
                    "rgb": pixel(r) if r else None,
                }
        out[str(f)] = row
    return out


def utae() -> dict:
    out = {}
    for mode in ("single", "multi", "history"):
        d = ROOT / f"outputs/utae_v6/utae_{mode}_realdates/1234"
        summ = load(d / "train_summary.json")
        meta = load(d / "run_meta.json")
        row = {"mode": mode, "seconds": (summ or {}).get("seconds"),
               "best_epoch": (summ or {}).get("best_epoch"),
               "peak_mib": (summ or {}).get("peak_alloc_mib"),
               "history": (summ or {}).get("history"),
               "init_digest": (meta or {}).get("init_digest")}
        for ck in ("best", "last"):
            for split in ("val", "test"):
                j = load(d / "eval" / f"eval_{ck}_{split}.json")
                if j:
                    row[f"{ck}_{split}"] = {"overall": j.get("overall", j),
                                            "by_event": j.get("by_event")}
        out[mode] = row
    return out


def main_groups() -> dict:
    out = {}
    for key, (exp, label) in MAIN.items():
        d = ROOT / f"outputs/segformer/v6_main/{exp}/1234"
        summ = load(d / "train_summary.json")
        meta = load(d / "run_meta.json")
        row = {"experiment": exp, "label": label,
               "seconds": (summ or {}).get("seconds"),
               "best_step": (summ or {}).get("best_step"),
               "peak_mib": (summ or {}).get("peak_alloc_mib"),
               "params": (summ or {}).get("params"),
               "module_activity": (summ or {}).get("module_activity"),
               "history": (summ or {}).get("history")}
        for ck in ("best", "last"):
            j = load(d / "eval" / f"eval_{ck}.json")
            if j:
                row[ck] = {
                    "val": pixel(j["val"]) if "val" in j else None,
                    "test": pixel(j["test"]) if "test" in j else None,
                    "val_by_event": {k: v["pixel"]["iou"] for k, v in j.get("val", {}).get("by_event", {}).items()},
                    "test_by_event": {k: v["pixel"]["iou"] for k, v in j.get("test", {}).get("by_event", {}).items()},
                    "efficiency": j.get("efficiency"),
                }
        out[key] = row
    return out


def deltas(summary: dict) -> dict:
    m = summary["main"]
    g = lambda k, split: (m.get(k, {}).get("best", {}) or {}).get(split, {})
    out = {}
    for split in ("val", "test"):
        s, mu = g("single_baseline", split), g("multi_baseline", split)
        ss, ms = g("single_spectral", split), g("multi_spectral", split)
        def d(a, b, metric="iou"):
            if not a or not b or a.get(metric) is None or b.get(metric) is None:
                return None
            return b[metric] - a[metric]
        out[split] = {
            "temporal_iou": d(s, mu), "single_spectral_iou": d(s, ss),
            "multi_spectral_iou": d(mu, ms), "interaction_iou": (d(mu, ms) - d(s, ss))
            if None not in (d(mu, ms), d(s, ss)) else None,
            "temporal_ap": d(s, mu, "ap"), "single_spectral_ap": d(s, ss, "ap"),
            "multi_spectral_ap": d(mu, ms, "ap"),
            "temporal_obj": d(s, mu, "obj_recall"), "single_spectral_obj": d(s, ss, "obj_recall"),
            "multi_spectral_obj": d(mu, ms, "obj_recall"),
            "temporal_bf1": d(s, mu, "boundary_f1"), "single_spectral_bf1": d(s, ss, "boundary_f1"),
            "multi_spectral_bf1": d(mu, ms, "boundary_f1"),
            "temporal_recall": d(s, mu, "recall"), "temporal_precision": d(s, mu, "precision"),
        }
    return out


def figures(summary: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)

    # DOFA corrected vs RGB
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, split in zip(axes, ("val", "test")):
        folds = [1, 2, 3, 4]
        x = np.arange(len(folds)); w = 0.36
        corr = [(summary["dofa"][str(f)]["best_" + split]["corrected"] or {}).get("iou")
                for f in folds]
        rgb = [(summary["dofa"][str(f)]["best_" + split]["rgb"] or {}).get("iou")
               for f in folds]
        corr = [np.nan if v is None else v for v in corr]
        rgb = [np.nan if v is None else v for v in rgb]
        ax.bar(x - w / 2, corr, w, label="Corrected 10-band", color="tab:green")
        ax.bar(x + w / 2, rgb, w, label="RGB", color="tab:gray")
        ax.set_xticks(x); ax.set_xticklabels([FOLD_LABEL[f] for f in folds], fontsize=8)
        ax.set_ylabel(f"{split.upper()} IoU@0.5"); ax.set_title(f"DOFA frozen probe, {split}")
        ax.legend(fontsize=8); ax.grid(alpha=.25, axis="y")
    fig.tight_layout(); fig.savefig(FIG / "temporal_dofa_corrected.png", dpi=150); plt.close(fig)

    # U-TAE curves
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for mode, color in (("single", "tab:blue"), ("multi", "tab:red"), ("history", "tab:green")):
        h = summary["utae"].get(mode, {}).get("history") or []
        if not h:
            continue
        ax.plot([r["epoch"] for r in h], [r["val"]["pooled_iou"] for r in h], color=color, label=mode)
    ax.set_xlabel("epoch"); ax.set_ylabel("VAL pooled IoU@0.5")
    ax.set_title("U-TAE real-date pair (3012/758/945, DEM)")
    ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(FIG / "temporal_utae_pair.png", dpi=150); plt.close(fig)

    # Main curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, keys, title in (
        (axes[0], ("single_baseline", "multi_baseline"), "No spectral module"),
        (axes[1], ("single_spectral", "multi_spectral"), "With spectral relation module"),
    ):
        for k in keys:
            h = summary["main"].get(k, {}).get("history") or []
            if not h:
                continue
            ax.plot([r["global_step"] for r in h], [r["val_iou"] for r in h],
                    label=FIG_LABEL[k])
        ax.set_title(title); ax.set_xlabel("successful updates")
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    axes[0].set_ylabel("VAL pooled IoU@0.5")
    fig.tight_layout(); fig.savefig(FIG / "temporal_main_curves.png", dpi=150); plt.close(fig)

    # Main metrics
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    metrics = [("iou", "IoU@0.5"), ("obj_recall", "Object recall (IoU>=.25)"),
               ("boundary_f1", "Boundary F1 (1 px)")]
    keys = list(MAIN)
    for ax, (metric, name) in zip(axes, metrics):
        vals = [((summary["main"].get(k, {}).get("best", {}) or {}).get("test", {}) or {}).get(metric)
                for k in keys]
        ax.bar(range(len(keys)), [v if v is not None else 0 for v in vals],
               color=["tab:gray", "tab:red", "tab:green", "tab:olive"])
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([FIG_LABEL[k] for k in keys], rotation=20, ha="right", fontsize=7)
        ax.set_title("TEST " + name); ax.grid(alpha=.25, axis="y")
    fig.tight_layout(); fig.savefig(FIG / "temporal_main_metrics.png", dpi=150); plt.close(fig)

    # Diagnostics: activity
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for k in ("single_spectral", "multi_spectral"):
        act = summary["main"].get(k, {}).get("module_activity") or {}
        sp = act.get("spectral") or {}
        if sp:
            ax.bar(np.arange(len(sp)) + (0.2 if k == "multi_spectral" else -0.2),
                   list(sp.values()), 0.38, label=FIG_LABEL[k])
            ax.set_xticks(range(len(sp))); ax.set_xticklabels(list(sp), rotation=20, ha="right", fontsize=7)
    ax.set_yscale("log"); ax.set_ylabel("sum |w| (log)")
    ax.set_title("Spectral module activity after training"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "temporal_main_diagnostics.png", dpi=150); plt.close(fig)


def main() -> None:
    summary = {"dofa": dofa(), "utae": utae(), "main": main_groups()}
    summary["deltas"] = deltas(summary)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    figures(summary)
    print(json.dumps(summary["deltas"], indent=1, ensure_ascii=False))
    print("[saved]", OUT / "summary.json", FIG)


if __name__ == "__main__":
    main()
