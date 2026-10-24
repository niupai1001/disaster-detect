"""生成灾前灾后组成的共同清单与三份逐样本帧计划；不改原清单。

合格集来自 ``docs/assets/research/temporal_composition_eligibility.json`` 的 retro 政策：
灾害日期唯一且高置信，至少 2 张灾前、5 张灾后，灾后含参考帧。三组共用同一批成员：
5 张灾后、3 张灾后、2 张灾前＋同 3 张灾后。帧计划显式给出每块送哪几帧，
训练与评价都按它取帧，不再用期数/窗口规则。

用法::

    python scripts/build_composition_plans.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from landslide_seg.config import file_sha256  # noqa: E402

GROUPS = ("post5", "post3", "pre2_post3")


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT)).replace("\\", "/")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eligibility", default="docs/assets/research/temporal_composition_eligibility.json")
    ap.add_argument("--pool", default="manifests/v4r1/pool_100.jsonl")
    ap.add_argument("--temporal-root", default="data/s2_temporal_v4r1")
    ap.add_argument("--out-dir", default="manifests/v4r3")
    ap.add_argument("--name", default="composition")
    args = ap.parse_args()

    eligibility_path = ROOT / args.eligibility
    pool_path = ROOT / args.pool
    index_path = ROOT / args.temporal_root / "index.json"
    eligible = json.loads(eligibility_path.read_text(encoding="utf-8"))["policies"]["retro"]["samples"]
    dates = {r["sample_id"]: r["dates"]
             for r in json.loads(index_path.read_text(encoding="utf-8"))["rows"]}
    pool = [json.loads(x) for x in pool_path.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
    rows = {r["sample_id"]: r for r in pool}

    plans = {g: {} for g in GROUPS}
    for sample in eligible:
        sid = sample["sample_id"]
        if sid not in rows:
            raise SystemExit(f"合格样本 {sid} 不在原清单中")
        expected = "train" if rows[sid]["protocol_role"] == "TRAIN" else "val"
        if sample["split"] != expected:
            raise SystemExit(f"{sid} 的角色 {sample['split']} 与原清单 {expected} 不一致")
        anchor = int(rows[sid]["time_index"])
        days = dates[sid]
        for group in GROUPS:
            frames = list(sample[group])
            if len(set(frames)) != len(frames):
                raise SystemExit(f"{sid}/{group} 含重复帧")
            if anchor not in frames:
                raise SystemExit(f"{sid}/{group} 未包含参考帧")
            plans[group][sid] = frames
    # 三组的固定对照关系必须由数据本身满足，而不是只写在文档里
    for sid in plans["post5"]:
        assert set(plans["post3"][sid]) <= set(plans["post5"][sid]), sid
        assert set(plans["pre2_post3"][sid]) & set(plans["post5"][sid]) == set(plans["post3"][sid]), sid
        assert len(set(plans["pre2_post3"][sid]) - set(plans["post3"][sid])) == 2, sid

    out_dir = ROOT / args.out_dir
    plan_dir = out_dir / "plans"
    plan_dir.mkdir(parents=True, exist_ok=True)
    kept = [rows[s["sample_id"]] for s in eligible]
    manifest = out_dir / f"pool_{args.name}.jsonl"
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept),
                        encoding="utf-8")
    lock = {
        "schema_version": 1,
        "created": "2026-10-23",
        "purpose": "灾前灾后组成的共同合格集：同成员，只改帧的组成",
        "rule": ("灾害日期唯一且高置信、至少2张灾前与5张灾后、灾后含参考帧；"
                 "三组共用同一批成员，混合组与纯灾后组共享同3张灾后帧"),
        "min_frames_at_or_before_reference": 5,
        "base_manifest": rel(pool_path),
        "base_manifest_sha256": file_sha256(pool_path),
        "base_protocol": "manifests/v4r1/protocol.json",
        "base_protocol_sha256": file_sha256(ROOT / "manifests/v4r1/protocol.json"),
        "temporal_index": rel(index_path),
        "temporal_index_sha256": file_sha256(index_path),
        "eligibility": rel(eligibility_path),
        "eligibility_sha256": file_sha256(eligibility_path),
        "manifest": rel(manifest),
        "manifest_sha256": file_sha256(manifest),
        "train_ids": [r["sample_id"] for r in kept if r["protocol_role"] == "TRAIN"],
        "dev_ids": [r["sample_id"] for r in kept if r["protocol_role"] == "DEV"],
        "counts": {"train": sum(1 for r in kept if r["protocol_role"] == "TRAIN"),
                   "dev": sum(1 for r in kept if r["protocol_role"] == "DEV")},
        "train_counts_by_event": dict(sorted(Counter(
            r["event_id"] for r in kept if r["protocol_role"] == "TRAIN").items())),
        "dev_counts_by_event": dict(sorted(Counter(
            r["event_id"] for r in kept if r["protocol_role"] == "DEV").items())),
        "plans": {},
        "notes": ["日期来自发布元数据，未独立核实灾害记录", "回顾性条件：灾后帧可能晚于参考日期"],
    }
    for group in GROUPS:
        path = plan_dir / f"{group}.json"
        path.write_text(json.dumps(plans[group], ensure_ascii=False) + "\n", encoding="utf-8")
        future_frames, max_after = 0, 0
        for sid, frames in plans[group].items():
            anchor = int(rows[sid]["time_index"])
            days = np.asarray([np.datetime64(d, "D").astype(int) for d in dates[sid]])
            delta = days[frames] - days[anchor]
            future_frames += int((delta > 0).sum())
            max_after = max(max_after, int(max(0, delta.max())))
        lock["plans"][group] = {"path": rel(path), "sha256": file_sha256(path),
                               "n_frames": sorted({len(v) for v in plans[group].values()}),
                               "n_samples": len(plans[group]),
                               "n_future_frames": future_frames,
                               "max_days_after_anchor": max_after,
                               "allows_future": future_frames > 0}
    lock_path = out_dir / f"subset_protocol_{args.name}.json"
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in lock.items() if not k.endswith("_ids")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
