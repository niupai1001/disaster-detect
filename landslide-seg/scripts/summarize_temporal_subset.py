"""汇总一个派生共同子集批次的配对组；只用已验收的正式结果。

批次由队列状态文件（docs/assets/research/v4r2_*_queue.json）描述：它列出实验名与
配置，本脚本从每个 run 自己的 resolved_config 读期数/窗口，从 execution_audit.json
读验收状态，不硬编码步数。

用法::

    python scripts/summarize_temporal_subset.py --queue docs/assets/research/v4r2_temporal_cutoff_queue.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from summarize_temporal_control import _read_eval

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "docs/assets/research/execution_audit.json"


PLAN_LABELS = {"post5": "5张灾后", "post3": "3张灾后", "pre2_post3": "2张灾前＋3张灾后"}


def label(temporal: dict) -> str:
    if temporal.get("plan"):
        return PLAN_LABELS.get(Path(temporal["plan"]).stem, Path(temporal["plan"]).stem)
    if temporal.get("window_days"):
        return f"{temporal['window_days']}天窗口"
    return f"{temporal.get('n_frames')}期"


def collect(queue_path: Path) -> dict:
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    accepted = {r["run_dir"]: r for r in
                json.loads(AUDIT.read_text(encoding="utf-8"))["runs"]}
    lock_path = ROOT / queue["subset"]["protocol"]
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    plan_meta = {Path(v["path"]).stem: v for v in lock.get("plans", {}).values()}
    records = []
    for item in queue["runs"]:
        run = ROOT / "outputs/segformer" / queue["batch"] / item["experiment"] / "1234"
        relative = run.relative_to(ROOT).as_posix()
        config = run / "resolved_config.json"
        temporal = json.loads(config.read_text(encoding="utf-8"))["segformer"]["temporal"] \
            if config.is_file() else {}
        steps = (json.loads(config.read_text(encoding="utf-8"))["segformer"]["train"]
                 .get("report_steps", []) if config.is_file() else [])
        row = {"experiment": item["experiment"], "label": label(temporal),
               "n_frames": temporal.get("n_frames"), "window_days": temporal.get("window_days"),
               "run_dir": relative,
               "status": accepted.get(relative, {}).get("status",
                                                        "未启动" if not run.exists() else "运行或验收中")}
        summary = run / "train_summary.json"
        if summary.is_file():
            d = json.loads(summary.read_text(encoding="utf-8"))
            row.update(global_step=d["global_step"], best_step=d.get("best_step"),
                       seconds=d.get("seconds"), peak_alloc_mib=d.get("peak_alloc_mib"))
        tags = ("best", "last", *(f"step{s}" for s in steps))
        row["evals"] = {tag: _read_eval(run / "eval" / f"eval_{tag}.json") for tag in tags}
        plan_key = Path(temporal["plan"]).stem if temporal.get("plan") else None
        expect_future = (int(plan_meta.get(plan_key, {}).get("n_future_frames", -1))
                         if plan_key else 0)
        plan = run / "frame_selection.jsonl"
        if plan.is_file():
            rows = [json.loads(x) for x in plan.read_text(encoding="utf-8").splitlines() if x.strip()]
            row["frame_plan"] = {
                "rows": len(rows),
                "post_reference_frames": sum(1 for r in rows for d in r["frame_dates"]
                                             if d > r["anchor_date"]),
                "wrong_frame_count": sum(1 for r in rows
                                         if int(r["n_frames"]) != temporal.get("n_frames")),
            }
        plan_ok = (row.get("frame_plan", {}).get("wrong_frame_count") == 0
                   and row.get("frame_plan", {}).get("post_reference_frames") == expect_future)
        row["plan_registered_future_frames"] = expect_future
        checks = accepted.get(relative, {}).get("checkpoints", {})
        row["formal_result"] = (row["status"] == "COMPLETED_VALID_FOR_AUDITED_SCOPE"
                                and plan_ok
                                and all(
            ev is not None and checks.get(tag, {}).get("evaluation_sha256") == hashlib.sha256(
                (run / "eval" / f"eval_{tag}.json").read_bytes()).hexdigest()
            for tag, ev in row["evals"].items()))
        if row["status"] == "COMPLETED_VALID_FOR_AUDITED_SCOPE" and not row["formal_result"]:
            row["status"] = "评价缺失或变动，待重新验收"
        records.append(row)
    retrospective = any(v.get("n_future_frames", 0) > 0 for v in plan_meta.values())
    return {"batch": queue["batch"], "purpose": queue["purpose"], "protocol": queue["protocol"],
            "subset": {**queue["subset"],
                       "sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest()},
            "budget": queue["budget"], "seed": 1234, "selection": "dev_event_mean_iou",
            "retrospective": retrospective,
            "observation": "同成员、同2000步与同日程前缀；只改观测数量、历史窗口或灾前灾后组成"
                           + ("；组成批次为回顾性条件，灾后帧可晚于参考日期" if retrospective else ""),
            "comparison_limit": f"派生共同子集（TRAIN {lock['counts']['train']}），"
                                "不与原2223块全体混排；单seed、两个反复用于研发的验证事件。",
            "runs": records}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queue", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    queue_path = ROOT / args.queue
    data = collect(queue_path)
    out = ROOT / (args.out or f"outputs/segformer/{data['batch']}")
    out.mkdir(parents=True, exist_ok=True)
    name = "summary_" + re.sub(r"^v4r\d+_", "", data["batch"])
    (out / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    lines = ["| 输入条件 | 状态 | best步 | 事件平均IoU(best) | 广岛 | 特里苏尔 | AP | 1–4像素对象召回 | 末态事件平均IoU |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in data["runs"]:
        best = r["evals"]["best"] if r["formal_result"] else None
        last = r["evals"]["last"] if r["formal_result"] else None
        cells = [r["label"], r["status"], str(r.get("best_step", "—"))]
        if best:
            fine = best["fine_bins"]["1_4"]
            cells += [f"{best['event_mean_iou']:.4f}",
                      f"{best['by_event_iou']['hiroshima']:.4f}",
                      f"{best['by_event_iou']['thrissur']:.4f}",
                      f"{best['average_precision']:.4f}",
                      f"{fine['object_recall_evaluable']:.4f}（{fine['n_objects_evaluable']}个）",
                      f"{last['event_mean_iou']:.4f}" if last else "—"]
        else:
            cells += ["—"] * 6
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "IoU为广岛与特里苏尔等权平均；只展示训练、权重和全部预定评价已验收的正式结果。"
                  "缺失组不计零分、不插值；2000步与历史3600步全程运行不是等预算比较。",
              ("逐样本帧计划按登记核对：未来帧数与登记一致（回顾性条件，灾后观测可晚于参考日期），"
               "且每块帧数等于该组 N；详见 JSON 的 frame_plan。"
               if data["retrospective"] else
               "逐样本帧计划另核对两项：不含参考日期之后的观测（post_reference_frames=0），"
               "且每块帧数等于该组N（wrong_frame_count=0）；详见JSON的frame_plan。")]
    (out / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
