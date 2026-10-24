"""汇总并核验 U-TAE 的时间对照：开关组（均值/内容/日期）与期数组（1/3/5/15）。

只读已有产物：要求 run_meta 的成员/统计哈希、`model_knobs` 与配置一致，
best/last 评价文件的权重哈希与磁盘一致、验证成员为 669；缺一项就不标正式结果。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from landslide_seg.config import file_sha256, resolve_path  # noqa: E402

EXPECTED_DEV = 669


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cfg(path: Path) -> dict:
    sys.path.insert(0, str(ROOT / "scripts"))
    from train_utae_cv import load_cv_config
    return load_cv_config(path)


def check_run(run: Path, cfg: dict, n_frames=None) -> dict:
    record = {"run_dir": run.relative_to(ROOT).as_posix(), "errors": [], "formal_result": False}
    for name in ("run_meta.json", "train_summary.json"):
        if not (run / name).is_file():
            record["errors"].append(f"missing {name}")
    if record["errors"]:
        return record
    meta = json.loads((run / "run_meta.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "train_summary.json").read_text(encoding="utf-8"))
    record.update(best_epoch=summary.get("best_epoch"), best_val_iou=summary.get("best_val_iou"),
                  epochs_run=summary.get("epochs_run"), seconds=summary.get("seconds"),
                  n_train=meta.get("n_train"), n_inner=meta.get("n_inner"),
                  knobs=meta.get("model_knobs"))
    expected_knobs = {"positional": str(cfg.get("model", {}).get("positional", "dates")),
                      "aggregation": str(cfg.get("model", {}).get("aggregation", "attention"))}
    if meta.get("model_knobs") is None:
        record["warnings"] = ["历史运行未记录 model_knobs"]
    elif meta["model_knobs"] != expected_knobs:
        record["errors"].append("model_knobs 与配置不一致")
    if summary.get("epochs_run") is not None and int(summary["epochs_run"]) != int(cfg["train"]["epochs"]):
        record["errors"].append("训练轮数与配置不一致")
    if int(meta.get("n_inner") or 0) != EXPECTED_DEV:
        record["errors"].append("验证成员数不是 669")
    if n_frames is not None and meta.get("temporal", {}).get("n_frames") != n_frames:
        record["errors"].append("帧数与登记不一致")
    evals = run / "eval" / "eval_dev_summary.json"
    if not evals.is_file():
        evals = run / "eval_dev_summary.json"
    if not evals.is_file():
        record["errors"].append("缺少 eval_dev_summary")
        return record
    data = json.loads(evals.read_text(encoding="utf-8"))
    record["evals"] = {}
    for tag in ("best", "last"):
        ev = data.get(tag)
        ckpt = run / f"{tag}.pt"
        if ev is None or not ckpt.is_file():
            record["errors"].append(f"缺少 {tag} 权重或评价")
            continue
        if ev.get("checkpoint_sha256") != file_sha256(ckpt):
            record["errors"].append(f"{tag} 评价权重哈希不一致")
        if int(ev.get("n_samples") or 0) != EXPECTED_DEV:
            record["errors"].append(f"{tag} 评价成员数不是 669")
        if set(ev.get("by_event", {})) != {"hiroshima", "thrissur"}:
            record["errors"].append(f"{tag} 评价事件不一致")
        by_event = {k: v["pixel"]["iou"] for k, v in ev["by_event"].items()}
        record["evals"][tag] = {
            "epoch": ev.get("epoch"),
            "pooled_iou": ev["overall"]["pixel"]["iou"],
            "event_mean_iou": float(sum(by_event.values()) / len(by_event)),
            "average_precision": ev["overall"]["pixel"].get("average_precision"),
            "object_recall_evaluable": ev["overall"]["objects"]["object_recall_evaluable"],
            "by_event": by_event,
        }
    record["formal_result"] = not record["errors"]
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queue", action="append", default=[],
                    help="队列 JSON（相对项目根）；可重复")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    records = []
    for qpath in args.queue:
        queue = json.loads((ROOT / qpath).read_text(encoding="utf-8"))
        batch_dir = ROOT / "outputs" / ("utae_v4r1_controls" if queue["batch"] == "utae_v4r1_controls"
                                        else "utae_v4r1_frame_count")
        for item in queue["runs"]:
            cfg_path = ROOT / item["config"]
            run = batch_dir / item["experiment"] / "v4r1" / "multi" / "1234"
            record = (check_run(run, load_cfg(cfg_path), item.get("n_frames")) if run.is_dir()
                      else {"run_dir": run.relative_to(ROOT).as_posix(), "errors": ["未启动"]})
            record.update(label=item["experiment"], batch=queue["batch"])
            records.append(record)
    reference = ROOT / "outputs/utae_v4r1/v4r1/multi/1234"
    ref_cfg = ROOT / "configs/utae/v4r1/multi.yaml"
    if reference.is_dir():
        rec = check_run(reference, load_cfg(ref_cfg))
        rec.update(label="date15", batch="reference")
        records.append(rec)
    out = {"scope": "U-TAE 时间对照（开关组与期数组）", "dev": EXPECTED_DEV,
           "note": "reference(date15) 为 75 轮、best@17；新组 30 轮，不是等轮数比较。",
           "runs": records}
    path = resolve_path(args.out or "docs/assets/research/utae_time_summary.json")
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("| 组 | 正式结果 | best轮 | 事件平均IoU | 广岛 | 特里苏尔 | AP | 差异原因 |")
    print("|---|---|---|---:|---:|---:|---:|---|")
    for r in records:
        ev = (r.get("evals") or {}).get("best")
        cells = [r["label"], "是" if r.get("formal_result") else "否", str(r.get("best_epoch", "—"))]
        if ev:
            cells += [f"{ev['event_mean_iou']:.4f}", f"{ev['by_event']['hiroshima']:.4f}",
                      f"{ev['by_event']['thrissur']:.4f}", f"{ev['average_precision']:.4f}"]
        else:
            cells += ["—"] * 4
        cells.append("; ".join(r.get("errors", [])) or "")
        print("| " + " | ".join(cells) + " |")
    print(path)


if __name__ == "__main__":
    main()
