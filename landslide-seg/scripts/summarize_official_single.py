"""BENCH_OFFICIAL 单期主实验结果汇总。

读 run 目录下的 train_summary.json 与 eval/eval_{best,last}_{val,test}.json，
输出官方 VAL / TEST 两个 split 的完整表：总体、逐事件、尺度召回、成本。

注意：TEST 只评测不参与选模；主结果为 VAL 选出的 best。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

RUN = "outputs/segformer/official_single/SegFormer-LateDetail-OfficialSplit-SingleDate/1234"
BINS = ["1_4", "5_9", "10_19", "20_49", "50_199", "ge200"]
FIXED = [500, 1200, 2400, 3600]


def collect(d: Path) -> dict:
    s = json.loads((d / "train_summary.json").read_text(encoding="utf-8"))
    out = {"run_dir": str(d.relative_to(ROOT)).replace("\\", "/"),
           "n_train": s.get("n_train"), "n_val": s.get("n_val"),
           "best_val_iou": s.get("best_val_iou"), "best_step": s.get("best_step"),
           "selection_metric": s.get("selection_metric"),
           "seconds": s.get("seconds"), "params": s.get("params"),
           "peak_alloc_mib": s.get("peak_alloc_mib"),
           "samples_exposed": s.get("samples_exposed"),
           "dev_steps": {}, "train_iou_steps": {}, "dev_pooled_steps": {}}
    for r in s.get("history", []):
        st = int(r["global_step"])
        if st in FIXED:
            out["dev_steps"][st] = r["dev"]["event_mean_iou"]
            out["dev_pooled_steps"][st] = r["dev"]["pooled"]["iou"]
            out["train_iou_steps"][st] = r["train_clean"]["pooled"]["iou"]
    for tag in ("best_val", "last_val", "best_test", "last_test", "best_train", "last_train"):
        p = d / "eval" / f"eval_{tag}.json"
        if not p.is_file():
            continue
        e = json.loads(p.read_text(encoding="utf-8"))
        o = e["overall"]
        px = o["pixel"]
        out[tag] = {
            "n": e["n_samples"], "event_mean_iou": e.get("event_mean_iou"),
            "by_event": {k: {"iou": v["pixel"]["iou"], "ap": v["pixel"].get("average_precision")}
                         for k, v in e["by_event"].items()},
            "iou": px.get("iou"), "dice": px.get("dice"), "precision": px.get("precision"),
            "recall": px.get("recall"), "ap": px.get("average_precision"),
            "fpr": px.get("false_positive_rate"),
            "boundary_f1": o["boundary"].get("f1"), "objects": o.get("objects"),
            "latency": (e.get("efficiency") or {}).get("latency_b1"),
        }
        ob = d / "eval" / f"objects_{tag}.json"
        if ob.is_file():
            agg = {b: {"n": 0, "hit": 0} for b in BINS}
            for row in json.loads(ob.read_text(encoding="utf-8")):
                for b in BINS:
                    x = (row.get("fine_bins") or {}).get(b)
                    if x:
                        agg[b]["n"] += x["n_objects_evaluable"]
                        agg[b]["hit"] += x["n_matched_evaluable"]
            out[tag]["scale_recall"] = {b: {"n": v["n"], "hit": v["hit"],
                                            "recall": (v["hit"] / v["n"]) if v["n"] else None}
                                        for b, v in agg.items()}
    return out


def f(v, n=4):
    return "-" if v is None else (f"{v:.{n}f}" if isinstance(v, float) else str(v))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--out", default="reports/official_single")
    args = ap.parse_args()
    d = ROOT / args.run
    r = collect(d)
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")

    L = ["# BENCH_OFFICIAL 单期主实验（SegFormer-LateDetail-OfficialSplit-SingleDate）\n",
         "官方 S12LS-LD split 的**单期**变体（十波段、无 DEM、无多时相）。"
         "选模用官方 VAL 前景 pooled IoU@0.5；TEST 只评测、不参与任何选择。\n",
         "## 1. 训练\n",
         "| 项 | 值 |", "|---|---|",
         f"| n_train / n_val | {r['n_train']} / {r['n_val']} |",
         f"| 选模口径 | `{r['selection_metric']}` |",
         f"| best 步 | {r['best_step']} |",
         f"| best VAL pooled IoU | {f(r['best_val_iou'])} |",
         f"| 训练耗时 | {f(r['seconds'],1)} s |",
         f"| 参数 | {(r.get('params') or {}).get('total', 0):,} |",
         f"| 训练峰值显存 | {f(r['peak_alloc_mib'],0)} MiB |",
         f"| 曝光 | {r['samples_exposed']} |",
         "\n## 2. 曲线（每 100 更新）\n",
         "| 步 | VAL 事件均值 | VAL pooled IoU | TRAIN pooled IoU |", "|---:|---:|---:|---:|"]
    for st in FIXED:
        L.append(f"| {st} | {f(r['dev_steps'].get(st))} | {f(r['dev_pooled_steps'].get(st))} | "
                 f"{f(r['train_iou_steps'].get(st))} |")

    for split in ("val", "test"):
        L.append(f"\n## 3.{'1' if split == 'val' else '2'} {split.upper()}：总体\n")
        L.append("| 检查点 | 事件均值 | 合并 IoU | Dice | Precision | Recall | AP | 边界 F1 | 对象召回 | FPR |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for ck in ("best", "last"):
            b = r.get(f"{ck}_{split}")
            if not b:
                continue
            o = b.get("objects") or {}
            L.append(f"| {ck} | {f(b['event_mean_iou'])} | {f(b['iou'])} | {f(b['dice'])} | "
                     f"{f(b['precision'])} | {f(b['recall'])} | {f(b['ap'])} | {f(b['boundary_f1'])} | "
                     f"{f(o.get('object_recall_evaluable'))} | {f(b['fpr'],5)} |")
        L.append(f"\n### 逐事件（{split}）\n")
        evs = sorted({e for ck in ("best", "last") for e in ((r.get(f"{ck}_{split}") or {}).get("by_event") or {})})
        L.append("| 检查点 | 事件 | IoU | AP |")
        L.append("|---|---|---:|---:|")
        for ck in ("best", "last"):
            b = r.get(f"{ck}_{split}") or {}
            for e in evs:
                x = (b.get("by_event") or {}).get(e)
                if x:
                    L.append(f"| {ck} | {e} | {f(x['iou'])} | {f(x['ap'])} |")
        L.append(f"\n### 尺度召回（{split}，对象级）\n")
        L.append("| 检查点 | " + " | ".join(f"{b} px" for b in BINS) + " |")
        L.append("|---|" + "---:|" * len(BINS))
        for ck in ("best", "last"):
            b = r.get(f"{ck}_{split}") or {}
            sr = b.get("scale_recall")
            if not sr:
                continue
            L.append(f"| {ck} | " + " | ".join(
                f"{f(sr[b]['recall'],3)} ({sr[b]['hit']}/{sr[b]['n']})" for b in BINS) + " |")

    L.append("\n## 4. 口径说明\n")
    L.append("- 本表是**单期**变体，官方基线是多时相 + DEM，二者不是同一输入条件，差值不等于时间信息或 DEM 的贡献。")
    L.append("- 本项目主口径只统计 valid 像素；官方 torchmetrics 用全像素（见 `interface_validation.md` 的手算例）。"
             "与官方表值并排前需按同一口径复算。")
    L.append("- 训练预算是 3,600 成功更新，不等于官方的 75 epochs × 3 seeds。")
    L.append("- 官方 split 是随机划分，同事件跨 train/val/test，故此处的 VAL/TEST 不是跨事件外推。")

    md = out_dir / "results_summary.md"
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n[saved] {md}")


if __name__ == "__main__":
    main()
