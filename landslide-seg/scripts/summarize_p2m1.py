"""P2-M1 结果汇总：B2 与同版本 B0 控制组（同数据/同预算/同 seed），旧 2223 组作背景。

只做数值汇总，不做解释。输出：
- ``reports/p13/results_summary.json``：全部原始数值
- ``reports/p13/results_summary.md``：逐事件 IoU/AP、尺度召回、对象、边界、成本

2223 组（v4r1 detailres_n100）仅作背景参照：它的训练集、统计文件和本对照都不同，
不能当作本对照的控制组。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ARMS = {
    "B2-AllTrain": "outputs/segformer/p2m1/segformer_b2_latedetail_alltrain_s1234/1234",
    "B0-AllTrain-ctl": "outputs/segformer/p2m1/segformer_b0_latedetail_alltrain_s1234_ctl/1234",
    "REF-2223(v4r1)": "outputs/segformer/s2_10band_v4r1_p20/segformer_b0_direct10_detailres_n100_u3600_s1234/1234",
}
BINS = ["1_4", "5_9", "10_19", "20_49", "50_199", "ge200"]
FIXED = [500, 1200, 2400, 3600]


def load(d: Path) -> dict:
    s = json.loads((d / "train_summary.json").read_text(encoding="utf-8"))
    out = {
        "run_dir": str(d.relative_to(ROOT)).replace("\\", "/"),
        "n_train": s.get("n_train"), "n_val": s.get("n_val"),
        "params": s.get("params"), "best_val_iou": s.get("best_val_iou"),
        "best_step": s.get("best_step"), "seconds": s.get("seconds"),
        "samples_exposed": s.get("samples_exposed"),
        "unique_samples_exposed": s.get("unique_samples_exposed"),
        "peak_alloc_mib": s.get("peak_alloc_mib"),
        "dev_steps": {}, "train_iou_steps": {},
    }
    for r in s.get("history", []):
        st = int(r["global_step"])
        if st in FIXED:
            out["dev_steps"][st] = r["dev"]["event_mean_iou"]
            out["train_iou_steps"][st] = r["train_clean"]["pooled"]["iou"]
    for tag in ("best_val", "last_val", "best_train", "last_train"):
        p = d / "eval" / f"eval_{tag}.json"
        if not p.is_file():
            continue
        e = json.loads(p.read_text(encoding="utf-8"))
        o = e["overall"]
        px = o["pixel"]
        out[tag] = {
            "split": e["split"], "n": e["n_samples"],
            "event_mean_iou": e.get("event_mean_iou"),
            "by_event": {k: {"iou": v["pixel"]["iou"],
                             "ap": v["pixel"].get("average_precision"),
                             "dice": v["pixel"].get("dice"),
                             "precision": v["pixel"].get("precision"),
                             "recall": v["pixel"].get("recall")}
                         for k, v in e["by_event"].items()},
            "iou": px.get("iou"), "dice": px.get("dice"),
            "precision": px.get("precision"), "recall": px.get("recall"),
            "ap": px.get("average_precision"), "fpr": px.get("false_positive_rate"),
            "boundary_f1": o["boundary"].get("f1"),
            "objects": o.get("objects"),
            "latency": (e.get("efficiency") or {}).get("latency_b1"),
        }
        ob = d / "eval" / f"objects_{tag}.json"
        if ob.is_file():
            agg = {b: {"n": 0, "hit": 0, "gt_px": 0, "hit_px": 0} for b in BINS}
            for row in json.loads(ob.read_text(encoding="utf-8")):
                for b in BINS:
                    x = (row.get("fine_bins") or {}).get(b)
                    if not x:
                        continue
                    agg[b]["n"] += x["n_objects_evaluable"]
                    agg[b]["hit"] += x["n_matched_evaluable"]
                    agg[b]["gt_px"] += x["gt_px"]
                    agg[b]["hit_px"] += x["hit_px"]
            out[tag]["scale_recall"] = {
                b: {"n": v["n"], "hit": v["hit"],
                    "object_recall": (v["hit"] / v["n"]) if v["n"] else None,
                    "pixel_recall": (v["hit_px"] / v["gt_px"]) if v["gt_px"] else None}
                for b, v in agg.items()}
    return out


def f(v, n=4):
    return "-" if v is None else (f"{v:.{n}f}" if isinstance(v, float) else str(v))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="reports/p2m1")
    args = ap.parse_args()

    data = {}
    for k, rel in ARMS.items():
        d = ROOT / rel
        if (d / "train_summary.json").is_file():
            data[k] = load(d)
        else:
            print(f"[skip] {k}: {rel}")
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "capacity_summary.json").write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")

    L = ["# P2-M1 同族规格对照（客观数值）\n",
         "`B2-AllTrain` 与 `B0-AllTrain-ctl` 同数据（2223/669）、同 stats、同损失、同增强、同预算、同 seed，"
         "同代码不同主干规格；`REF-2223` 为 v4r1 旧代码组，仅作背景参照。\n"]

    L.append("## 1. 训练与选模\n")
    L.append("| 组 | n_train | n_val | best DEV | best@步 | DEV@500 | @1200 | @2400 | @3600 | TRAIN IoU@3600 | 秒 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k, v in data.items():
        d = v["dev_steps"]
        L.append(f"| {k} | {v['n_train']} | {v['n_val']} | {f(v['best_val_iou'])} | {v['best_step']} | "
                 f"{f(d.get(500))} | {f(d.get(1200))} | {f(d.get(2400))} | {f(d.get(3600))} | "
                 f"{f(v['train_iou_steps'].get(3600))} | {f(v['seconds'],1)} |")

    L.append("\n## 2. DEV：best 检查点总体\n")
    L.append("| 组 | 事件均值 | 合并 IoU | Dice | Precision | Recall | AP | 边界 F1 | 对象召回(可评估) | 负瓦片 FPR |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k, v in data.items():
        b = v.get("best_val")
        if not b:
            continue
        o = b.get("objects") or {}
        L.append(f"| {k} | {f(b['event_mean_iou'])} | {f(b['iou'])} | {f(b['dice'])} | {f(b['precision'])} | "
                 f"{f(b['recall'])} | {f(b['ap'])} | {f(b['boundary_f1'])} | "
                 f"{f(o.get('object_recall_evaluable'))} | {f(b['fpr'],5)} |")

    L.append("\n## 3. DEV：逐事件 IoU / AP（best 检查点）\n")
    evs = sorted({e for v in data.values() for e in (v.get("best_val") or {}).get("by_event", {})})
    L.append("| 组 | 事件 | IoU | AP | Dice | Precision | Recall |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for k, v in data.items():
        b = v.get("best_val")
        if not b:
            continue
        for e in evs:
            x = b["by_event"].get(e)
            if not x:
                continue
            L.append(f"| {k} | {e} | {f(x['iou'])} | {f(x['ap'])} | {f(x['dice'])} | "
                     f"{f(x['precision'])} | {f(x['recall'])} |")

    L.append("\n## 4. DEV：尺度召回（对象级，best 检查点）\n")
    L.append("| 组 | " + " | ".join(f"{b} px" for b in BINS) + " |")
    L.append("|---|" + "---:|" * len(BINS))
    for k, v in data.items():
        b = v.get("best_val") or {}
        sr = b.get("scale_recall")
        if not sr:
            continue
        cells = []
        for bn in BINS:
            x = sr.get(bn, {})
            cells.append(f"{f(x.get('object_recall'),3)} ({x.get('hit')}/{x.get('n')})")
        L.append(f"| {k} | " + " | ".join(cells) + " |")

    L.append("\n## 5. DEV：尺度像素召回（best 检查点）\n")
    L.append("| 组 | " + " | ".join(f"{b} px" for b in BINS) + " |")
    L.append("|---|" + "---:|" * len(BINS))
    for k, v in data.items():
        b = v.get("best_val") or {}
        sr = b.get("scale_recall")
        if not sr:
            continue
        L.append(f"| {k} | " + " | ".join(f"{f(sr.get(bn,{}).get('pixel_recall'),3)}" for bn in BINS) + " |")

    L.append("\n## 6. last 检查点（第 3600 步）\n")
    L.append("| 组 | 事件均值 | 合并 IoU | AP | 对象召回(可评估) |")
    L.append("|---|---:|---:|---:|---:|")
    for k, v in data.items():
        b = v.get("last_val")
        if not b:
            continue
        o = b.get("objects") or {}
        L.append(f"| {k} | {f(b['event_mean_iou'])} | {f(b['iou'])} | {f(b['ap'])} | "
                 f"{f(o.get('object_recall_evaluable'))} |")

    L.append("\n## 7. TRAIN 侧（best 检查点）\n")
    L.append("| 组 | 事件均值 | 合并 IoU | AP | 对象召回(可评估) |")
    L.append("|---|---:|---:|---:|---:|")
    for k, v in data.items():
        b = v.get("best_train")
        if not b:
            continue
        o = b.get("objects") or {}
        L.append(f"| {k} | {f(b['event_mean_iou'])} | {f(b['iou'])} | {f(b['ap'])} | "
                 f"{f(o.get('object_recall_evaluable'))} |")

    L.append("\n## 8. 成本\n")
    L.append("| 组 | 总参数 | encoder | decoder | new | 训练峰值 MiB | 推理 p50 ms | 更新/曝光 | 唯一曝光样本 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k, v in data.items():
        p = v.get("params") or {}
        lat = ((v.get("best_val") or {}).get("latency") or {}).get("p50_ms")
        L.append(f"| {k} | {p.get('total',0):,} | {p.get('encoder',0):,} | {p.get('decoder',0):,} | "
                 f"{p.get('new',0):,} | {f(v.get('peak_alloc_mib'),0)} | {f(lat,2)} | "
                 f"{v.get('samples_exposed')} | {v.get('unique_samples_exposed')} |")

    md = out_dir / "capacity_summary.md"
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n[saved] {md}")


if __name__ == "__main__":
    main()
