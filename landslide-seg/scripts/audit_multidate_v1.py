"""多日期/时相契约审计（只生成审阅清单，不改标签、不删样本）。

背景：Sen12Landslides 的 ``event_date`` 对 China/Indonesia 可能是**日期列表**
（实测 215/1,277 个训练样本），而当前 P0 只取第一个日期做 `days_after` 基准，
``pre_post_dates`` 也只保留最后一对。这会让"灾后天数"和"选图是否灾后"的解释
不成立，必须单列审阅，而不是直接当标签错误。

输出：
- ``manifests/s2_10band_v1_multidate_review.csv``：215 行多日期样本，含各项标志；
- ``outputs/segformer/<version>/multidate_review_summary.json``：计数与交叉表。

用法::

    python scripts/audit_multidate_v1.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:  # Windows 控制台/管道默认 cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from landslide_seg.config import load_segformer_config, resolve_path  # noqa: E402
from landslide_seg.data.segformer_dataset import read_manifest  # noqa: E402

DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "segformer" / "base.yaml"))
    args = ap.parse_args()
    cfg = load_segformer_config(args.config)["segformer"]
    version = cfg["data"]["version"]
    rows = read_manifest(resolve_path(cfg["data"]["manifest"]))

    out_rows = []
    for r in rows:
        dates = DATE_RE.findall(str(r.get("event_date", "")))
        first = min(dates) if dates else None
        last = max(dates) if dates else None
        chosen = r.get("date")
        days_at_post = r.get("days_at_post")
        flags = {
            "n_event_dates": len(dates),
            "first_event_date": first,
            "last_event_date": last,
            "chosen_date": chosen,
            "days_after_first": r.get("days_after_event"),
            "chosen_before_latest_event": bool(last and chosen and chosen < last),
            "post_index": r.get("post_index"),
            "days_at_post": days_at_post,
            "post_before_first_event": bool(days_at_post is not None and days_at_post < 0),
            "temporal_fallback": bool(r.get("temporal_fallback")),
            "days_after_event_gt_365": bool((r.get("days_after_event") or 0) > 365),
            "one_way_ambiguous": bool(
                (last and chosen and chosen < last)
                or (days_at_post is not None and days_at_post < 0)
            ),
        }
        flags.update(
            {
                "sample_id": r["sample_id"],
                "event_id": r["event_id"],
                "split": r.get("split"),
                "event_date_raw": r.get("event_date"),
                "cache": r.get("cache"),
            }
        )
        out_rows.append(flags)

    multi = [r for r in out_rows if r["n_event_dates"] > 1]
    early = [r for r in out_rows if r["chosen_before_latest_event"]]
    post_before = [r for r in out_rows if r["post_before_first_event"]]
    one_way = [r for r in out_rows if r["one_way_ambiguous"]]

    manifest_dir = resolve_path("manifests")
    csv_path = manifest_dir / f"{version}_multidate_review.csv"
    fieldnames = [
        "sample_id", "event_id", "split", "event_date_raw", "n_event_dates",
        "first_event_date", "last_event_date", "chosen_date", "days_after_first",
        "chosen_before_latest_event", "post_index", "days_at_post",
        "post_before_first_event", "temporal_fallback", "days_after_event_gt_365",
        "one_way_ambiguous", "cache",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(multi, key=lambda r: (r["event_id"], r["sample_id"])))

    summary = {
        "version": version,
        "n_samples_total": len(out_rows),
        "multi_date_total": len(multi),
        "multi_date_by_region": dict(Counter(r["event_id"] for r in multi)),
        "chosen_before_latest_total": len(early),
        "chosen_before_latest_by_region": dict(Counter(r["event_id"] for r in early)),
        "post_before_first_event_total": len(post_before),
        "post_before_first_event_by_region": dict(Counter(r["event_id"] for r in post_before)),
        "one_way_ambiguous_total": len(one_way),
        "one_way_ambiguous_by_split": dict(Counter(r["split"] for r in one_way)),
        "temporal_fallback_total": sum(1 for r in out_rows if r["temporal_fallback"]),
        "days_after_gt_365_total": sum(1 for r in out_rows if r["days_after_event_gt_365"]),
        "review_csv": str(csv_path).replace("\\", "/"),
        "notes": [
            "多日期样本只列入审阅，不据此改日期、不改标签、不删样本。",
            "days_after_first 以 event_date 的第一个日期为基准，跨事件时无物理含义。",
            "post_before_first_event 说明 pre_post 的 post 索引与首个事件日期不自洽，不能解释为同一事件的选图延迟。",
            "是否真的标签错配，需要回到原始 inventory 的 ann_id、几何与事件日期逐对象核实。",
        ],
    }
    out_dir = resolve_path(cfg["output_dir"]) / version
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "multidate_review_summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"[saved] {csv_path}")


if __name__ == "__main__":
    main()
