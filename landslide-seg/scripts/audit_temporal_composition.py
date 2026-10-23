"""仅审计日期组成共同合格集；不删改原清单，不自动训练新子集。"""
from __future__ import annotations
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from landslide_seg.data.temporal_v6 import select_frame_indices
from landslide_seg.config import file_sha256


def select_post(frames, count):
    frames = sorted(frames, key=lambda r: r["date"])
    anchor = [i for i, r in enumerate(frames) if r["anchor"] == "True"]
    if len(anchor) != 1:
        raise ValueError("灾后合法序列必须包含唯一参考帧")
    days = np.asarray([r["date"] for r in frames], dtype="datetime64[D]").astype(np.int64)
    idx, _ = select_frame_indices(days, anchor[0], "multi", count)
    return [int(frames[i]["frame_id"]) for i in idx]


def main():
    source = ROOT / "docs/assets/research/temporal_observations.csv"
    groups = defaultdict(list)
    with source.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            groups[row["sample_id"]].append(row)
    result = {"source_sha256": file_sha256(source),
              "eligibility_rule": "灾害日期字段唯一、date_confidence=1、至少2PRE和5POST且POST含参考帧；日期来源为发布元数据，未独立核实灾害记录",
              "claim_limit": "仅共同合格子集审计；原2223/669全体上的数量实验继续，不把子集与全体直接比较",
              "policies": {}}
    for policy in ("retro", "as_of_anchor"):
        samples, counts = [], defaultdict(int)
        for sid, frames in groups.items():
            if any(r["pre_post"] == "AMBIGUOUS_EVENT_DATE" for r in frames):
                continue
            if any(r["date_confidence"] != "1.0" and r["date_confidence"] != "1" for r in frames):
                continue
            legal = [r for r in frames if policy == "retro" or r["after_anchor"] == "False"]
            pre = sorted([r for r in legal if r["pre_post"] == "PRE"], key=lambda r: r["date"])
            post = [r for r in legal if r["pre_post"] == "POST"]
            if len(pre) < 2 or len(post) < 5:
                continue
            post5 = select_post(post, 5)
            post3 = select_post(post, 3)
            pre2 = [int(r["frame_id"]) for r in pre[-2:]]
            assert len(set(post5)) == 5 and len(set(post3)) == 3 and len(set(pre2 + post3)) == 5
            assert set(post3) <= set(post5), "两种比较必须共享同一组三个灾后帧"
            samples.append({"sample_id": sid, "event_id": frames[0]["event_id"], "split": frames[0]["split"],
                            "post5": post5, "post3": post3, "pre2_post3": pre2 + post3})
            counts[f"{frames[0]['split']}/{frames[0]['event_id']}"] += 1
        result["policies"][policy] = {"n_samples": len(samples), "counts": dict(counts), "samples": samples}
    out = ROOT / "docs/assets/research/temporal_composition_eligibility.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({p: {k: v for k, v in d.items() if k != "samples"} for p, d in result["policies"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
