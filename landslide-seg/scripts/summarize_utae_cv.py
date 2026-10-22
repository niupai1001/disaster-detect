"""汇总四折 U-TAE 时间对照（single / history / multi）的已完成结果。

读 ``outputs/utae_cv/<F>/<mode>/1234`` 下的 train_summary.json 与 eval_summary.json，
输出 ``summary.json``、``summary.md`` 与终端表格。未完成的 run 标记为 running，
不参与均值，避免用部分折冒充全量。
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "utae_cv"
FOLDS = ("F1", "F2", "F3", "F4")
MODES = ("single", "history", "multi")
METRICS = ("iou", "average_precision", "dice", "precision", "recall")


def load(fold: str, mode: str) -> dict:
    d = OUT / fold / mode / "1234"
    rec = {"fold": fold, "mode": mode, "status": "missing"}
    s, e = d / "train_summary.json", d / "eval_summary.json"
    if s.is_file():
        sj = json.loads(s.read_text(encoding="utf-8"))
        rec.update({"status": "running", "epochs": sj["epochs"], "complete": sj["complete"],
                    "best_epoch": sj["best_epoch"], "inner_sel": sj["best_inner_selection"],
                    "seconds": sj["seconds"], "peak_mib": sj.get("peak_alloc_mib")})
    if e.is_file():
        ej = json.loads(e.read_text(encoding="utf-8"))
        rec["status"] = "done"
        rec["ckpt_epoch"] = ej["best"]["outer"]["checkpoint_epoch"]
        for split in ("inner", "outer"):
            ov = ej["best"][split]["overall"]
            rec[split] = {m: ov["pixel"].get(m) for m in METRICS}
            rec[split]["object_recall"] = ov["objects"]["object_recall_evaluable"]
            rec[split]["boundary_f1"] = ov["boundary"]["f1"]
            rec[split]["n_tiles"] = ov["n_tiles"]
        rec["by_event"] = {ev: {"iou": v["pixel"]["iou"],
                                "average_precision": v["pixel"]["average_precision"],
                                "n_tiles": v["n_tiles"]}
                           for ev, v in ej["best"]["outer"]["by_event"].items()}
        rec["last_outer_iou"] = ej["last"]["outer"]["overall"]["pixel"]["iou"]
    return rec


def fmt(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}"


def main() -> None:
    data = {f: {m: load(f, m) for m in MODES} for f in FOLDS}
    lines: list[str] = []
    lines.append("# 四折 U-TAE 时间对照汇总\n")

    lines.append("## 表 A：逐折逐模式（best checkpoint，OUTER 外部留出）\n")
    lines.append("| 折 | 模式 | 状态 | best ep | inner IoU（选模） | OUTER IoU | OUTER AP | Dice | P | R | 对象召回 | 边界 F1 | 训练 |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for f in FOLDS:
        for m in MODES:
            r = data[f][m]
            if r["status"] == "done":
                o = r["outer"]
                lines.append(f"| {f} | {m} | done | {r['ckpt_epoch']} | {fmt(r['inner']['iou'])} | "
                             f"{fmt(o['iou'])} | {fmt(o['average_precision'])} | {fmt(o['dice'])} | "
                             f"{fmt(o['precision'])} | {fmt(o['recall'])} | {fmt(o['object_recall'])} | "
                             f"{fmt(o['boundary_f1'])} | {r['seconds']/60:.0f} min |")
            else:
                lines.append(f"| {f} | {m} | {r['status']} | {r.get('best_epoch','')} | "
                             f"{fmt(r.get('inner_sel'))} | — | — | — | — | — | — | — | "
                             f"{(r.get('seconds') or 0)/60:.0f} min |")

    lines.append("\n## 表 B：汇总（只计已完成折）\n")
    lines.append("| 模式 | 完成折 | OUTER IoU 四折均值 | 事件宏平均 | 最差事件 | AP 四折均值 | AP 事件宏平均 |")
    lines.append("|---|---:|---:|---:|---|---:|---:|")
    agg = {}
    for m in MODES:
        done = [data[f][m] for f in FOLDS if data[f][m]["status"] == "done"]
        if not done:
            lines.append(f"| {m} | 0 | — | — | — | — | — |")
            continue
        ious = [r["outer"]["iou"] for r in done]
        aps = [r["outer"]["average_precision"] for r in done]
        ev = {}
        for r in done:
            for e, v in r["by_event"].items():
                ev[e] = v["iou"]
        ev_ap = {}
        for r in done:
            for e, v in r["by_event"].items():
                ev_ap[e] = v["average_precision"]
        worst = min(ev.items(), key=lambda kv: (kv[1] is None, kv[1]))
        agg[m] = {"n_folds": len(done), "iou_mean": sum(ious) / len(ious),
                  "ap_mean": sum(aps) / len(aps),
                  "event_macro_iou": sum(ev.values()) / len(ev),
                  "event_macro_ap": sum(ev_ap.values()) / len(ev_ap),
                  "worst_event": worst[0], "worst_iou": worst[1],
                  "per_fold_iou": {r["fold"]: r["outer"]["iou"] for r in done},
                  "per_event_iou": ev}
        lines.append(f"| {m} | {len(done)} | {fmt(agg[m]['iou_mean'])} | {fmt(agg[m]['event_macro_iou'])} | "
                     f"{worst[0]} {fmt(worst[1])} | {fmt(agg[m]['ap_mean'])} | {fmt(agg[m]['event_macro_ap'])} |")

    lines.append("\n## 表 C：逐事件 OUTER IoU（每个事件只在一个折里是外部）\n")
    events = sorted({e for m in MODES for r in data.values() for e, v in (r[m].get("by_event") or {}).items()})
    lines.append("| 外部事件 | 折 | single | history | multi |")
    lines.append("|---|---|---:|---:|---:|")
    for e in events:
        fold = next((f for f in FOLDS for m in MODES if e in (data[f][m].get("by_event") or {})), "?")
        cells = [fmt((data[fold][m].get("by_event") or {}).get(e, {}).get("iou")) for m in MODES]
        lines.append(f"| {e} | {fold} | " + " | ".join(cells) + " |")

    lines.append("\n## 表 D：相对单期的差值（同一折内）\n")
    lines.append("| 折 | history − single | multi − single | single IoU |")
    lines.append("|---|---:|---:|---:|")
    for f in FOLDS:
        s = data[f]["single"]
        if s["status"] != "done":
            continue
        base = s["outer"]["iou"]
        for m in ("history", "multi"):
            pass
        h = data[f]["history"]["outer"]["iou"] if data[f]["history"]["status"] == "done" else None
        mu = data[f]["multi"]["outer"]["iou"] if data[f]["multi"]["status"] == "done" else None
        lines.append(f"| {f} | {'—' if h is None else f'{h-base:+.4f}'} | "
                     f"{'—' if mu is None else f'{mu-base:+.4f}'} | {fmt(base)} |")

    md = "\n".join(lines) + "\n"
    (OUT / "summary.md").write_text(md, encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps({"cells": data, "agg": agg}, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
