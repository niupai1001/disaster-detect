"""生成"参考日期当日及以前至少有 N 帧"的共同子集清单；不改原清单。

用途：截止参考日期的时间数量组（1/3/5/7 期）必须四组同成员，否则期数效应与
成员变化混在一起。TRAIN 只保留可用帧数达标的样本，DEV 必须全部达标；不达标就
报 NOT_ELIGIBLE，不补帧、不改事件日期、不引入 DEV 训练。

用法::

    python scripts/build_temporal_subset.py --min-frames 7
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from landslide_seg.config import file_sha256  # noqa: E402


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT)).replace("\\", "/")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default="manifests/v4r1/pool_100.jsonl")
    ap.add_argument("--protocol", default="manifests/v4r1/protocol.json")
    ap.add_argument("--temporal-root", default="data/s2_temporal_v4r1")
    ap.add_argument("--min-frames", type=int, default=7)
    ap.add_argument("--window-days", type=int, default=None,
                    help="只统计参考日期前 N 天内的帧；用于固定跨度的共同成员")
    ap.add_argument("--out-dir", default="manifests/v4r2")
    ap.add_argument("--name", default="cutoff7")
    args = ap.parse_args()

    pool = ROOT / args.pool
    proto = ROOT / args.protocol
    index_path = ROOT / args.temporal_root / "index.json"
    dates = {r["sample_id"]: r["dates"]
             for r in json.loads(index_path.read_text(encoding="utf-8"))["rows"]}
    rows = [json.loads(x) for x in pool.read_text(encoding="utf-8-sig").splitlines() if x.strip()]

    kept, dropped, available = [], Counter(), {"TRAIN": [], "DEV": []}
    for row in rows:
        role = row["protocol_role"]
        anchor = np.datetime64(row["date"], "D").astype(int)
        days = np.asarray([np.datetime64(d, "D").astype(int) for d in dates[row["sample_id"]]])
        used = int(((days <= anchor) & (days >= anchor - (args.window_days or 0))).sum())
        available[role].append(used)
        if used >= args.min_frames:
            kept.append(row)
        elif role == "TRAIN":
            dropped[row["event_id"]] += 1
        else:
            raise SystemExit(f"NOT_ELIGIBLE: DEV {row['sample_id']} 只有 {used} 帧，"
                             f"不足 {args.min_frames}；不补齐、不改事件日期")

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / f"pool_{args.name}.jsonl"
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept),
                        encoding="utf-8")
    train = [r for r in kept if r["protocol_role"] == "TRAIN"]
    dev = [r for r in kept if r["protocol_role"] == "DEV"]
    lock = {
        "schema_version": 1,
        "created": "2026-10-23",
        "purpose": "截止参考日期的时间数量组共同子集；只与同成员配对组比较",
        "rule": f"TRAIN 只保留参考日期当日及以前至少 {args.min_frames} 个真实观测的样本"
                + (f"（且这些观测都在参考日期前 {args.window_days} 天内）" if args.window_days else "")
                + "；DEV 原样全保留",
        "min_frames_at_or_before_reference": args.min_frames,
        "window_days": args.window_days,
        "base_manifest": rel(pool),
        "base_manifest_sha256": file_sha256(pool),
        "base_protocol": rel(proto),
        "base_protocol_sha256": file_sha256(proto),
        "temporal_index": rel(index_path),
        "temporal_index_sha256": file_sha256(index_path),
        "manifest": rel(manifest),
        "manifest_sha256": file_sha256(manifest),
        "train_ids": [r["sample_id"] for r in train],
        "dev_ids": [r["sample_id"] for r in dev],
        "counts": {"train": len(train), "dev": len(dev)},
        "train_counts_by_event": dict(sorted(Counter(r["event_id"] for r in train).items())),
        "dev_counts_by_event": dict(sorted(Counter(r["event_id"] for r in dev).items())),
        "dropped_train_by_event": dict(sorted(dropped.items())),
        "available_frames": {role: {"min": min(v), "median": statistics.median(v), "max": max(v),
                                    "n": len(v)} for role, v in available.items()},
    }
    lock_path = out_dir / f"subset_protocol_{args.name}.json"
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in lock.items() if not k.endswith("_ids")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
