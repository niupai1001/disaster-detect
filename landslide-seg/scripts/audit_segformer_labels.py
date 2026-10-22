"""P0b 标签审计：对 GT 做连通域，输出**对象级**面积分布与截断比例。

执行计划 v2 §6 P0b 要求：尺度档位必须由对象级分布支撑，而不是"整块正像素 <50"。
同时给出 1–4 px 极小对象、触边/邻无效区截断对象、负瓦片的真实数量。

用法::

    python scripts/audit_segformer_labels.py --split train
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:  # Windows 控制台/管道默认 cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np  # noqa: E402

from landslide_seg.config import load_segformer_config, resolve_path  # noqa: E402
from landslide_seg.data.segformer_dataset import filter_split, read_manifest  # noqa: E402
from landslide_seg.training.segformer_metrics import (  # noqa: E402
    assign_bin,
    label_objects,
    object_table,
)


def _quantiles(vals: list[float], qs=(0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)) -> dict:
    if not vals:
        return {}
    a = np.asarray(vals, dtype=np.float64)
    return {f"p{int(q*100)}": round(float(np.quantile(a, q)), 1) for q in qs} | {
        "min": round(float(a.min()), 1),
        "max": round(float(a.max()), 1),
        "mean": round(float(a.mean()), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "segformer" / "base.yaml"))
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()

    cfg = load_segformer_config(args.config, args.set)["segformer"]
    manifest = resolve_path(cfg["data"]["manifest"])
    records = filter_split(read_manifest(manifest), args.split)
    if not records:
        raise RuntimeError(f"split={args.split} 为空")

    areas: list[int] = []
    truncated = 0
    tiny = 0
    n_objects = 0
    per_event = defaultdict(lambda: {"n_tiles": 0, "n_pos_tiles": 0, "n_objects": 0,
                                     "n_truncated": 0, "n_tiny": 0, "areas": []})
    bins = Counter()
    bins_eval = Counter()
    negative_tiles = 0
    positive_tiles = 0
    gt_px_total = 0
    valid_px_total = 0

    for rec in records:
        d = np.load(rec["cache"])
        mask = d["mask"].astype(bool)
        valid = d["valid"].astype(bool)
        gt_px_total += int(mask.sum())
        valid_px_total += int(valid.sum())
        ev = per_event[rec["event_id"]]
        ev["n_tiles"] += 1
        if not mask.any():
            negative_tiles += 1
            continue
        positive_tiles += 1
        ev["n_pos_tiles"] += 1
        labels, n = label_objects(mask)
        objects = object_table(labels, n, valid)
        n_objects += len(objects)
        ev["n_objects"] += len(objects)
        for obj in objects:
            areas.append(obj["area"])
            ev["areas"].append(obj["area"])
            bins[assign_bin(obj["area"])] += 1
            if obj["truncated"]:
                truncated += 1
                ev["n_truncated"] += 1
            else:
                bins_eval[assign_bin(obj["area"])] += 1
            if 1 <= obj["area"] <= 4:
                tiny += 1
                ev["n_tiny"] += 1

    per_event_out = {}
    for ev, v in sorted(per_event.items()):
        per_event_out[ev] = {
            "n_tiles": v["n_tiles"],
            "n_positive_tiles": v["n_pos_tiles"],
            "n_objects": v["n_objects"],
            "n_truncated": v["n_truncated"],
            "truncated_frac": round(v["n_truncated"] / max(v["n_objects"], 1), 4),
            "n_tiny_1_4px": v["n_tiny"],
            "object_area": _quantiles(v.pop("areas")),
        }

    report = {
        "split": args.split,
        "n_tiles": len(records),
        "n_positive_tiles": positive_tiles,
        "n_negative_tiles": negative_tiles,
        "gt_px": gt_px_total,
        "valid_px": valid_px_total,
        "gt_px_over_valid": round(gt_px_total / max(valid_px_total, 1), 5),
        "n_objects": n_objects,
        "objects_per_positive_tile": round(n_objects / max(positive_tiles, 1), 3),
        "n_truncated": truncated,
        "truncated_frac": round(truncated / max(n_objects, 1), 4),
        "n_tiny_1_4px": tiny,
        "tiny_frac": round(tiny / max(n_objects, 1), 4),
        "object_area_px": _quantiles(areas),
        "object_area_ha": {k: round(v / 100.0, 4) for k, v in _quantiles(areas).items()},
        "bins_all_objects": dict(bins),
        "bins_evaluable_objects": dict(bins_eval),
        "thresholds_px": [50, 200],
        "by_event": per_event_out,
    }
    out_dir = resolve_path("outputs/segformer/_audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"labels_{args.split}.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "by_event"},
                     indent=1, ensure_ascii=False))
    print(f"[saved] {out_dir / f'labels_{args.split}.json'}")


if __name__ == "__main__":
    main()
