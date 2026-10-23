"""汇总固定事件、相同日期汇聚下的1/3/5/7/15期；缺失结果不插值。"""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
from summarize_temporal_control import _read_eval

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/segformer/v4r1_temporal_frame_count"


def collect():
    audit = json.loads((ROOT / "docs/assets/research/execution_audit.json").read_text(encoding="utf-8"))
    accepted = {r["run_dir"]: r for r in audit["runs"]}
    records = []
    for n in (1, 3, 5, 7, 15):
        if n in (1, 15):
            run = ROOT / "outputs/segformer/v4r1_temporal_control" / (
                "SegFormer-B0-Temporal-Single" if n == 1 else "SegFormer-B0-Temporal-Date15") / "1234"
        else:
            run = OUT / f"SegFormer-B0-Temporal-Date{n}" / "1234"
        relative = run.relative_to(ROOT).as_posix()
        row = {"n_frames": n, "run_dir": relative,
               "status": accepted.get(relative, {}).get("status", "未启动" if not run.exists() else "运行或验收中")}
        summary = run / "train_summary.json"
        if summary.exists():
            d = json.loads(summary.read_text(encoding="utf-8"))
            row.update(global_step=d["global_step"], best_step=d.get("best_step"),
                       seconds=d.get("seconds"), peak_alloc_mib=d.get("peak_alloc_mib"))
        else:
            metrics = run / "metrics.jsonl"
            if metrics.exists():
                lines = metrics.read_text(encoding="utf-8").splitlines()
                # 正在追加的最后一行可能尚未写完，不把残行当实验损坏。
                for line in reversed(lines):
                    try:
                        row["global_step"] = json.loads(line)["global_step"]
                        break
                    except (ValueError, KeyError):
                        continue
        row["evals"] = {tag: _read_eval(run / "eval" / f"eval_{tag}.json")
                        for tag in ("best", "last", "step1200", "step2400", "step3600")}
        checks = accepted.get(relative, {}).get("checkpoints", {})
        row["formal_result"] = row["status"] == "COMPLETED_VALID_FOR_AUDITED_SCOPE" and all(
            ev is not None and checks.get(tag, {}).get("evaluation_sha256") == hashlib.sha256(
                (run / "eval" / f"eval_{tag}.json").read_bytes()).hexdigest()
            for tag, ev in row["evals"].items())
        if row["status"] == "COMPLETED_VALID_FOR_AUDITED_SCOPE" and not row["formal_result"]:
            row["status"] = "评价缺失或变动，待重新验收"
        records.append(row)
    return {"protocol": "event_ood_v4r1", "train": 2223, "dev": 669, "seed": 1234,
            "selection": "dev_event_mean_iou", "updates": 3600, "microbatch": 8, "accumulation": 8,
            "observation": "回顾性，含参考日期以后；T=1等价恒等，其余内容与相对日期加权",
            "frame_selection": "参考帧必保留，其他帧在日期排序后等秩取样；不使用标签或预测",
            "comparison_limit": "N≥3共享完整日期端点跨度；T=1没有多期跨度。单seed、两个已用于研发的验证事件。",
            "runs": records}


def main():
    data = collect()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary_frame_count.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["| SegFormer B0观测与汇聚（均保留参考影像细节） | 状态 | 更新数 | 最佳事件平均IoU | AP | 1–4像素对象召回 |",
             "|---|---|---:|---:|---:|---:|"]
    for r in data["runs"]:
        ev = r["evals"]["best"] if r["formal_result"] else None
        cells = [f"{r['n_frames']}期十波段，" + ("恒等" if r["n_frames"] == 1 else "内容与日期加权"),
                 r["status"], str(r.get("global_step", "—"))]
        if ev:
            fine = ev["fine_bins"]["1_4"]
            cells += [f"{ev['event_mean_iou']:.4f}", f"{ev['average_precision']:.4f}",
                      f"{fine['object_recall_evaluable']:.4f}（支持数{fine['n_objects_evaluable']}）"]
        else:
            cells += ["—", "—", "—"]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "IoU为广岛与特里苏尔等权平均；只展示训练、权重和全部预定评价已验收的正式结果。缺失组不计零分、不插值。"]
    (OUT / "summary_frame_count.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
