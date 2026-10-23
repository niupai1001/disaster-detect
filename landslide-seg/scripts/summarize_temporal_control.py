"""汇总时间控制四组的训练曲线、best/last/预定步指标与成本。

用法::

    python scripts/summarize_temporal_control.py
    python scripts/summarize_temporal_control.py --version-dir outputs/segformer/v4r1_temporal_control

只读取 run 目录，不训练、不改 checkpoint。输出 ``summary_temporal_control.json``
与同目录的 markdown 表；图由 ``scripts/make_temporal_control_figure.py`` 生成。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["single", "mean", "content", "date", "history"]
DISPLAY = {
    "single": "SegFormer B0｜单期十波段｜参考影像细节",
    "mean": "SegFormer B0｜15期十波段｜均值汇聚＋参考影像细节",
    "content": "SegFormer B0｜15期十波段｜内容权重＋参考影像细节",
    "date": "SegFormer B0｜15期十波段｜内容与日期权重＋参考影像细节",
    "history": "SegFormer B0｜截止参考日期的观测｜内容与日期权重（回顾性之外的可用输入）",
}
TAGS = ["best", "last", "step1200", "step2400", "step3600"]


def _key(run_dir: Path) -> str | None:
    name = run_dir.name.lower()
    if "history" in name:
        return "history"
    for key in ORDER:
        if name.endswith(f"-{key}15") or name.endswith(f"-{key}"):
            return key
    return None


def _read_eval(path: Path) -> dict | None:
    if not path.is_file():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    val = d["val"]
    obj = val["overall"]["objects"]
    return {
        "checkpoint": path.stem.replace("eval_", ""),
        "global_step": d.get("global_step"),
        "event_mean_iou": d.get("val_event_mean_iou"),
        "pooled_iou": val["overall"]["pixel"]["iou"],
        "dice": val["overall"]["pixel"]["dice"],
        "precision": val["overall"]["pixel"]["precision"],
        "recall": val["overall"]["pixel"]["recall"],
        "average_precision": val["overall"]["pixel"].get("average_precision"),
        "object_recall_evaluable": obj["object_recall_evaluable"],
        "boundary_f1": val["overall"]["boundary"]["f1"],
        "tp": val["overall"]["pixel"].get("tp"),
        "fp": val["overall"]["pixel"].get("fp"),
        "fn": val["overall"]["pixel"].get("fn"),
        "by_event_iou": {k: v["pixel"]["iou"] for k, v in val["by_event"].items()},
        "fine_bins": obj.get("fine_bins", {}),
        "n_samples": val["n_samples"],
        "eval_seconds": val["seconds"],
        "latency": d.get("efficiency", {}).get("latency_b1"),
        "checkpoint_sha256": d.get("checkpoint_sha256"),
    }


def summarize(version_dir: Path) -> dict:
    runs = {}
    for exp_dir in sorted(version_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        key = _key(exp_dir)
        if key is None:
            continue
        run_dir = exp_dir / "1234"
        curve = []
        mpath = run_dir / "metrics.jsonl"
        if mpath.is_file():
            for line in mpath.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    curve.append({
                        "step": row["global_step"],
                        "event_mean_iou": row.get("dev_event_mean_iou"),
                        "pooled_iou": row.get("dev_pooled_iou"),
                        "train_loss": row.get("train_loss"),
                    })
        evals = {tag: _read_eval(run_dir / "eval" / f"eval_{tag}.json") for tag in TAGS}
        summary_path = run_dir / "train_summary.json"
        summary = (json.loads(summary_path.read_text(encoding="utf-8"))
                   if summary_path.is_file() else {})
        values = [p["event_mean_iou"] for p in curve if p.get("event_mean_iou") is not None]
        runs[key] = {
            "display": DISPLAY[key],
            "run_dir": str(run_dir),
            "status": ("未完成" if summary and not summary.get("complete")
                       else ("完成" if summary else "运行中")),
            "global_step": summary.get("global_step") or (curve[-1]["step"] if curve else None),
            "selection_metric": summary.get("selection_metric", "dev_event_mean_iou"),
            "best_step": summary.get("best_step"),
            "best_val_iou": summary.get("best_val_iou") or (max(values) if values else None),
            "best_dev_event_mean_iou": summary.get("best_dev_event_mean_iou"),
            "seconds": summary.get("seconds") or (curve[-1].get("seconds") if curve else None),
            "peak_alloc_mib": summary.get("peak_alloc_mib"),
            "params": summary.get("params"),
            "curve": curve,
            "evals": evals,
        }
    return {"version_dir": str(version_dir), "runs": runs}


def _fmt(v, digits=4):
    if v is None:
        return "—"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def to_markdown(data: dict) -> str:
    runs = data["runs"]
    base = runs.get("single", {}).get("evals", {}).get("best") or {}
    lines = [
        "| 模型 | 事件平均 IoU | 相对单期 | 像素汇总 IoU | AP | 广岛 IoU | 特里苏尔 IoU | 对象召回 | 边界 F1 | 参数量 | 训练耗时 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ORDER:
        run = runs.get(key)
        if not run:
            continue
        ev = run.get("evals", {}).get("best")
        if not ev:
            lines.append(f"| {DISPLAY[key]} | — | — | — | — | — | — | — | — | — | — |")
            continue
        delta = (ev["event_mean_iou"] - base["event_mean_iou"]) if base.get("event_mean_iou") is not None else None
        ious = ev.get("by_event_iou", {})
        lines.append("| " + " | ".join([
            DISPLAY[key], _fmt(ev["event_mean_iou"]),
            "—" if delta is None else f"{delta:+.4f}",
            _fmt(ev["pooled_iou"]), _fmt(ev["average_precision"]),
            _fmt(ious.get("hiroshima")), _fmt(ious.get("thrissur")),
            _fmt(ev["object_recall_evaluable"]), _fmt(ev["boundary_f1"]),
            f"{run.get('params', {}).get('total', 0):,}" if run.get("params") else "—",
            f"{run.get('seconds', 0) / 60:.0f} min" if run.get("seconds") else "—",
        ]) + " |")
    lines += ["", "口径：事件平均 IoU 为广岛与特里苏尔两个事件 IoU 的算术平均；"
                  "像素汇总 IoU 先累计两事件混淆计数再计算；best 由每 100 次更新按事件平均 IoU 选出。"]

    lines += ["", "| 模型 | 检查点 | 步数 | 事件平均 IoU | 像素汇总 IoU | 对象召回 | 边界 F1 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for key in ORDER:
        run = runs.get(key, {})
        for tag in TAGS:
            ev = run.get("evals", {}).get(tag)
            if not ev:
                continue
            step = ev.get("global_step")
            if step is None:
                step = (int(tag[4:]) if tag.startswith("step") else
                        run.get("best_step") if tag == "best" else run.get("global_step"))
            lines.append(f"| {DISPLAY[key]} | {tag} | {step} | "
                         f"{_fmt(ev['event_mean_iou'])} | {_fmt(ev['pooled_iou'])} | "
                         f"{_fmt(ev['object_recall_evaluable'])} | {_fmt(ev['boundary_f1'])} |")

    lines += ["", "| 模型 | 1–4 px | 5–9 px | 10–19 px | 20–49 px | 50–199 px | ≥200 px |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for key in ORDER:
        ev = runs.get(key, {}).get("evals", {}).get("best")
        if not ev:
            continue
        fb = ev.get("fine_bins", {})
        cells = [f"{_fmt((fb.get(k) or {}).get('object_recall_evaluable'))}"
                 f"（{(fb.get(k) or {}).get('n_objects_evaluable', 0)}）"
                 for k in ("1_4", "5_9", "10_19", "20_49", "50_199", "ge200")]
        lines.append(f"| {DISPLAY[key]} | " + " | ".join(cells) + " |")
    lines += ["", "尺度对象召回括号内为可评价对象数（IoU≥0.25 匹配，1 px 边界容差）。"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version-dir", default="outputs/segformer/v4r1_temporal_control")
    args = ap.parse_args()
    out_dir = ROOT / args.version_dir
    data = summarize(out_dir)
    (out_dir / "summary_temporal_control.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    md = to_markdown(data)
    (out_dir / "summary_temporal_control.md").write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
