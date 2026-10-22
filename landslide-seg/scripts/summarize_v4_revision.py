"""Summarize completed v4r1 runs and render reproducible DEV error cases."""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/segformer/plot_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from landslide_seg.config import load_segformer_config
from landslide_seg.training.segformer_metrics import label_objects, object_table


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rgb(x, channels=(2, 1, 0)):
    arr = np.moveaxis(x[list(channels)].astype(np.float32), 0, -1)
    lo, hi = np.percentile(arr, [2, 98], axis=(0, 1))
    return np.clip((arr - lo) / np.maximum(hi - lo, 1), 0, 1)


def main():
    registry = load(ROOT / "configs/segformer/v4r1/experiments.json")
    out = ROOT / "reports/v4r1_results"
    out.mkdir(parents=True, exist_ok=True)
    rows, payloads, curves = [], [], []
    for item in registry:
        cfg = load_segformer_config(ROOT / item["config"])["segformer"]
        run = ROOT / cfg["output_dir"] / cfg["data"]["version"] / item["experiment"] / str(cfg["seed"])
        summary = load(run / "train_summary.json")
        if not summary["complete"] or summary["global_step"] != 3600:
            raise RuntimeError(f"Run incomplete: {run}")
        evaluations = {tag: load(run / "eval" / f"eval_{tag}.json") for tag in
                       ("best_val", "last_val", "best_train", "last_train", "best_common_train_reference")}
        best, last = evaluations["best_val"], evaluations["last_val"]
        px, obj, boundary = best["overall"]["pixel"], best["overall"]["objects"], best["overall"]["boundary"]
        row = {"experiment": item["experiment"], "stem": item["stem"], "pool_percent": item["pool"],
               "n_train": summary["n_train"], "seed": cfg["seed"], "updates": summary["global_step"],
               "samples_exposed": summary["samples_exposed"], "best_step": summary["best_step"],
               "dev_event_mean_iou": best["event_mean_iou"], "dev_pooled_iou": px["iou"],
               "hiroshima_iou": best["by_event"]["hiroshima"]["pixel"]["iou"],
               "thrissur_iou": best["by_event"]["thrissur"]["pixel"]["iou"],
               "dice": px["dice"], "precision": px["precision"], "recall": px["recall"], "ap": px["average_precision"],
               "object_recall": obj["object_recall_evaluable"], "tiny_recall": obj["tiny"]["object_recall_evaluable"],
               "boundary_f1": boundary["f1"], "background_fpr": px["false_positive_rate"],
               "unmatched_prediction_objects_per_tile": obj["unmatched_prediction_objects_per_tile"],
               "best_train_iou": evaluations["best_train"]["overall"]["pixel"]["iou"],
               "last_train_iou": evaluations["last_train"]["overall"]["pixel"]["iou"],
               "common_reference_iou": evaluations["best_common_train_reference"]["overall"]["pixel"]["iou"],
               "last_dev_mean_iou": last["event_mean_iou"], "params": summary["params"]["total"],
               "training_seconds": summary["seconds"], "peak_alloc_mib": summary["peak_alloc_mib"],
               "latency_b1_ms": best["efficiency"]["latency_b1"]["mean_ms"],
               "negative_tile_fpr": None, "final_evaluated": False, "run_dir": str(run)}
        for name, values in obj["fine_bins"].items():
            row[f"recall_{name}"] = values["object_recall_evaluable"]
            row[f"n_objects_{name}"] = values["n_objects_evaluable"]
            row[f"n_matched_{name}"] = values["n_matched_evaluable"]
            row[f"recall_all_{name}"] = values["object_recall_all"]
            row[f"n_objects_all_{name}"] = values["n_objects"]
            row[f"n_matched_all_{name}"] = values["n_matched_all"]
        rows.append(row)
        payloads.append({"experiment": item["experiment"], "evaluations": evaluations})
        curves.extend(dict(experiment=item["experiment"], **h) for h in summary["history"])
    csv_path = out / "learning_curve_and_controls.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "full_evaluations.json").write_text(json.dumps(payloads, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "update_curves.json").write_text(json.dumps(curves, ensure_ascii=False, indent=1), encoding="utf-8")
    frozen = load(ROOT / "manifests/v4r1/protocol.json")
    membership = {k: set(v) for k, v in frozen["pools"].items()}
    bin_names = ("1_4", "5_9", "10_19", "20_49", "50_199", "ge200")
    population = {k: {"pool_percent": int(k), "n_tiles": 0, "valid_pixels": 0, "valid_foreground_pixels": 0,
                      "raw_foreground_pixels": 0, "events": set(), **{f"objects_{b}": 0 for b in bin_names},
                      **{f"evaluable_objects_{b}": 0 for b in bin_names}} for k in membership}
    source_rows = [json.loads(line) for line in (ROOT / "manifests/v4r1/pool_100.jsonl").read_text(encoding="utf-8").splitlines()]
    # Count the real supervised population without changing or filtering it.
    for rec in source_rows:
        if rec["split"] != "train":
            continue
        with np.load(rec["cache"]) as cache:
            gt, valid = cache["mask"].astype(bool), cache["valid"].astype(bool)
        labels, n = label_objects(gt)
        objects = object_table(labels, n, valid)
        for k, members in membership.items():
            if rec["sample_id"] not in members:
                continue
            target = population[k]
            target["n_tiles"] += 1
            target["events"].add(rec["event_id"])
            target["valid_pixels"] += int(valid.sum())
            target["valid_foreground_pixels"] += int((gt & valid).sum())
            target["raw_foreground_pixels"] += int(gt.sum())
            for obj in objects:
                area = obj["area"]
                b = "1_4" if area < 5 else "5_9" if area < 10 else "10_19" if area < 20 else "20_49" if area < 50 else "50_199" if area < 200 else "ge200"
                target[f"objects_{b}"] += 1
                target[f"evaluable_objects_{b}"] += int(not obj["truncated"])
    for stats in population.values():
        stats["events"] = len(stats["events"])
        stats["valid_foreground_fraction"] = stats["valid_foreground_pixels"] / stats["valid_pixels"]
    with (out / "training_pool_composition.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(population["25"]))
        writer.writeheader()
        writer.writerows(population.values())
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for row in rows:
        history = [h for h in curves if h["experiment"] == row["experiment"]]
        label = f"{'detail residual' if row['stem'] else 'standard'} N={row['n_train']}"
        axes[0].plot([h["global_step"] for h in history], [h["train_loss"] for h in history], label=label)
        axes[1].plot([h["global_step"] for h in history], [h["train_clean"]["pooled"]["iou"] for h in history], label=label)
        axes[2].plot([h["global_step"] for h in history], [h["dev"]["event_mean_iou"] for h in history], label=label)
    for ax, title in zip(axes, ("Training loss", "Clean training IoU", "DEV event mean IoU")):
        ax.set(title=title, xlabel="Optimizer updates")
        ax.grid(alpha=.2)
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "update_curves.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for stem in (True, False):
        series = sorted([r for r in rows if r["stem"] == stem], key=lambda r: r["n_train"])
        for ax, key, title in zip(axes, ("dev_event_mean_iou", "hiroshima_iou", "thrissur_iou"), ("DEV event mean", "Hiroshima", "Thrissur")):
            ax.plot([r["n_train"] for r in series], [r[key] for r in series], "o-", label="detail residual" if stem else "standard")
            ax.set(title=title, xlabel="Training tiles", ylabel="Foreground IoU")
            ax.grid(alpha=.2)
            ax.legend()
    fig.tight_layout()
    fig.savefig(out / "data_learning_curve.png", dpi=150)
    plt.close(fig)
    # Select cases by numerical error types, never by presumed label correctness.
    records = {}
    for line in (ROOT / "manifests/v4r1/pool_100.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["split"] == "val":
            records[r["sample_id"]] = r
    atlas = out / "error_atlas"
    atlas.mkdir(exist_ok=True)
    selection, threshold_rows = [], []
    for row in rows:
        run = Path(row["run_dir"])
        with (run / "eval/per_tile_best_val.csv").open(encoding="utf-8") as f:
            tiles = list(csv.DictReader(f))
        for ev in ("hiroshima", "thrissur"):
            candidates = [r for r in tiles if r["event_id"] == ev]
            conditions = {
                "complete_miss": [r for r in candidates if int(r["gt_px"]) > 0 and int(r["tp"]) == 0],
                "partial_detection": [r for r in candidates if int(r["tp"]) > 0 and int(r["fn"]) > int(r["tp"])],
                "false_positive": [r for r in candidates if int(r["fp"]) > 0],
                "boundary_error": [r for r in candidates if int(r["gt_px"]) >= 20
                                   and int(r["tp"]) >= .5 * int(r["gt_px"])
                                   and int(r["tp"]) >= .5 * (int(r["tp"]) + int(r["fp"]))
                                   and r["boundary_f1"]],
            }
            for category, options in conditions.items():
                key = (lambda r: int(r["fp"])) if category == "false_positive" else ((lambda r: -float(r["boundary_f1"])) if category == "boundary_error" else (lambda r: int(r["fn"])))
                if not options:
                    selection.append({"experiment": row["experiment"], "event": ev, "category": category, "available": False})
                    continue
                selected = max(options, key=key)
                selection.append({"experiment": row["experiment"], "event": ev, "category": category,
                                  "sample_id": selected["sample_id"], "available": True,
                                  "tp": int(selected["tp"]), "fp": int(selected["fp"]), "fn": int(selected["fn"]),
                                  "criterion": ("minimum boundary F1 with GT>=20 pixels, pixel recall>=0.5 and precision>=0.5"
                                                if category == "boundary_error" else "maximum FP or FN among qualifying DEV tiles")})
        with np.load(run / "eval/predictions_best_val.npz") as pred:
            lookup = {str(sid): i for i, sid in enumerate(pred["sample_ids"])}
            for ev in ("hiroshima", "thrissur"):
                ids = [i for sid, i in lookup.items() if records[sid]["event_id"] == ev]
                probabilities, truths, valid_pixels = pred["probability"][ids], pred["gt"][ids], pred["valid"][ids]
                probabilities, truths = probabilities[valid_pixels], truths[valid_pixels]
                for threshold in np.arange(.1, 1., .1):
                    mask = probabilities >= threshold
                    tp, fp, fn = int((mask & truths).sum()), int((mask & ~truths).sum()), int((~mask & truths).sum())
                    threshold_rows.append({"experiment": row["experiment"], "event": ev, "threshold": round(float(threshold), 1),
                                           "tp": tp, "fp": fp, "fn": fn, "precision": tp / (tp + fp) if tp + fp else None,
                                           "recall": tp / (tp + fn) if tp + fn else None, "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
                                           "used_for_threshold_selection": False})
            picked = [s for s in selection if s["experiment"] == row["experiment"] and s["available"]]
            fig, axes = plt.subplots(len(picked), 7, figsize=(21, 3 * len(picked)), squeeze=False)
            for i, case in enumerate(picked):
                sid = case["sample_id"]
                idx, rec = lookup[sid], records[sid]
                with np.load(rec["cache"]) as raw:
                    color = rgb(raw["x10"])
                    false_color = rgb(raw["x10"], (9, 7, 2))
                p, gt, v = pred["probability"][idx], pred["gt"][idx], pred["valid"][idx]
                mask = (p >= .5) & v
                error = np.stack(((~gt) & mask & v, gt & mask & v, gt & (~mask) & v), axis=-1).astype(float)
                panels = [(color, f"{case['event']} {case['category']}\n{sid} {rec['date']}"),
                          (false_color, "B12 / B8A / B04"), (gt, "GT (unreviewed visibility)"),
                          (v, "Valid mask (not raw SCL)"), (p, "Probability"),
                          (mask, "Prediction >=0.5"), (error, "Red FP / green TP / blue FN")]
                for ax, (arr, title) in zip(axes[i], panels):
                    ax.imshow(arr, **({} if arr.ndim == 3 else {"cmap": "gray", "vmin": 0, "vmax": 1}))
                    ax.set_title(title, fontsize=8)
                    ax.axis("off")
            fig.tight_layout()
            fig.savefig(atlas / f"{row['experiment']}.png", dpi=100)
            plt.close(fig)
    (atlas / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=1), encoding="utf-8")
    with (out / "dev_threshold_diagnostics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(threshold_rows[0]))
        writer.writeheader()
        writer.writerows(threshold_rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, event in zip(axes, ("hiroshima", "thrissur")):
        for row in rows:
            points = [p for p in threshold_rows if p["event"] == event and p["experiment"] == row["experiment"] and p["precision"] is not None]
            ax.plot([p["recall"] for p in points], [p["precision"] for p in points], "o-", ms=3,
                    label=f"{'detail' if row['stem'] else 'standard'} N={row['n_train']}")
        ax.set(title=event + " DEV threshold diagnostics", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "dev_threshold_diagnostics.png", dpi=150)
    plt.close(fig)
    protocol = load(ROOT / "manifests/v4r1/protocol.json")
    lines = ["# v4r1 六组单seed实验执行报告", "", "所有结果来自真实训练与TRAIN/DEV评测。未评New Zealand或Italy；不作历史盲测或跨seed稳定性声明。", "",
             "## 已修复并验证", "", "- 公共559个训练样本拟合归一化；四档共享，DEV/FINAL不参与。",
             "- 精确3600次更新、batch64跨epoch接续、每100次更新验证；每组230400次样本曝光。",
             "- Hiroshima448与Thrissur221，按事件等权平均IoU选best；另存固定末态。",
             "- 中断恢复CPU集成测试验证权重/RNG/采样游标一致；真实SegFormer13步验证梯度、尺寸和重载。",
             "- 前景AP、对象细分召回、边界F1、背景像素误报率和成本均保存；未独立核实负影像FPR保持N/A。", "",
             f"协议哈希：`{protocol['sha256']}`。正式配置：`configs/segformer/v4r1/experiments.json`。", "",
             "## 最佳checkpoint的DEV结果", "", "| 架构 | 训练量 | 最佳步数 | 事件均值IoU | Hiroshima | Thrissur | Dice pooled | AP pooled | tiny召回 | 边界F1 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(v):
        return "N/A" if v is None else f"{v:.4f}"
    for r in rows:
        lines.append(f"| SegFormer-B0{'原网格细节残差' if r['stem'] else '直接多光谱'} | {r['n_train']} | {r['best_step']} | " + " | ".join(fmt(r[k]) for k in ("dev_event_mean_iou", "hiroshima_iou", "thrissur_iou", "dice", "ap", "tiny_recall", "boundary_f1")) + " |")
    lines += ["", "## 拟合与末态", "", "| 架构 | 训练量 | best训练IoU | 末态训练IoU | 末态DEV均值IoU | 训练秒数 | 参数量 | 峰值MiB | batch1毫秒 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {'原网格细节残差' if r['stem'] else '直接多光谱'} | {r['n_train']} | {fmt(r['best_train_iou'])} | {fmt(r['last_train_iou'])} | {fmt(r['last_dev_mean_iou'])} | {r['training_seconds']:.1f} | {r['params']} | {r['peak_alloc_mib']:.1f} | {r['latency_b1_ms']:.2f} |")
    lines += ["", "## 解释边界与下一步", "", "- 这些组只改变固定九个训练事件内部的监督样本量、或独立开关细节分支；不估计增加新事件的效应。",
              "- fixed-update比较不是收敛能力的完整比较；大档的每样本重复曝光更少。平台不能单独证明数据量无效。",
              "- 共同参考训练池对小档含未训练样本，CSV明确标注；实际训练拟合另列，不混用。",
              "- 历史v3使用不同数据/统计/预算选择，不能把本轮与历史差值单因子归因。",
              "- 错误图谱依据GT和数值误差选例；可见性未经人工审核，未据模型输出判标签错误。",
              "- 1–4像素主评测分母很小，须以CSV中的n_objects_1_4解读召回变化；几个对象的波动不能作为稳定改善证据。",
              "- 训练秒数包含周期性TRAIN/DEV推理与checkpoint保存，不等于纯GPU反传时间；不同N的诊断成本不同。",
              "- 阈值图仅作DEV诊断，主结果始终使用0.5；图上九个采样阈值不用于近似替代精确AP。",
              "- 先根据两DEV与完整训练拟合判断事件多样性或编码器对照；不依据FINAL决定研究路线。", "",
              "## 产物", "", "- `learning_curve_and_controls.csv`：六组完整指标、对象数、成本与曝光量。",
              "- `full_evaluations.json`：best/last、TRAIN/DEV、共同参考池原始评测汇总。",
              "- `update_curves.json`、`update_curves.png`、`data_learning_curve.png`：统一更新数与数据量曲线。",
              "- `error_atlas/selection.json`、各架构PNG：DEV漏检、部分检出、误报、边界错误案例。",
              "- 各run保存best.pt、last.pt、resolved_config.json、run_meta.json、metrics.jsonl、sample_exposures.json。"]
    report = ROOT / "reports/v4r1_execution_report_2026-10-19.md"
    interpretation = out / "interpretation.md"
    if interpretation.exists():
        lines += ["", interpretation.read_text(encoding="utf-8")]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report)
    for r in rows:
        print(r["experiment"], "dev_mean", r["dev_event_mean_iou"], "H", r["hiroshima_iou"], "T", r["thrissur_iou"], "train", r["best_train_iou"])


if __name__ == "__main__":
    main()
