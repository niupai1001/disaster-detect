"""裁决多日期样本并生成 ``s2_10band_v2_clean``（不改标签、保留旧版本）。

裁决规则（只用时间与观测质量，不看模型输出）：

1. ``event_date`` 只有一个日期 → 原样保留；
2. 多个日期且**所选影像日期 >= 最晚事件日期** → 保留（时间安全，只把
   ``days_after_event`` 的基准改为最晚事件日期）；
3. 多个日期且所选影像早于最晚日期 → 在**最晚日期之后 120 天**内找可用时相
   （cloud<=0.5、valid_ratio>=0.5、滑坡像素不可观测<=60%），按日期取最早；
   找到则重选该时相并重算缓存；
4. 找不到 → 隔离，理由 ``multi_event_unresolvable``。

产物：
- ``data/s2_10band_v2_clean/*.npz``（全部保留样本的缓存副本或重算缓存）；
- ``manifests/s2_10band_v2_clean.jsonl`` / ``_quarantine.jsonl`` / ``_summary.json``；
- ``data/s2_10band_v2_clean/stats.json``（新 train 拟合）。

用法::

    python scripts/build_s2_10band_v2_clean.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

try:  # Windows 控制台/管道默认 cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np  # noqa: E402

from landslide_seg.config import load_segformer_config, resolve_path  # noqa: E402
from landslide_seg.data.segformer_dataset import S2_10_BANDS, read_manifest  # noqa: E402
from prepare_segformer_data import CLOUD_CODES, SCL_VALID, UNKNOWN_MASK_VALUE, compute_stats  # noqa: E402

DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
WINDOW_DAYS = 120
MIN_VALID_RATIO = 0.5
MAX_LABEL_INVALID = 0.6
MAX_CLOUD = 0.5


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_time_series(path: Path) -> tuple[date, np.ndarray, np.ndarray, np.ndarray]:
    import h5py

    with h5py.File(path, "r") as f:
        units = f["time"].attrs["units"]
        units = units.decode() if isinstance(units, bytes) else str(units)
        ref = date.fromisoformat(units.split("since")[-1].strip())
        t = f["time"][:]
        scl = f["SCL"][:]
        mask = f["MASK"][:]
    return ref, t, scl, mask


def _build_sample(path: Path, idx: int, scl: np.ndarray, mask_raw: np.ndarray) -> dict:
    import h5py

    with h5py.File(path, "r") as f:
        x10 = np.stack([f[b][idx] for b in S2_10_BANDS]).astype(np.float32)
    unknown = mask_raw == UNKNOWN_MASK_VALUE
    mask = np.where(unknown, 0, mask_raw).astype(np.uint8)
    scl_ok = np.isin(scl, SCL_VALID)
    finite_ok = np.isfinite(x10).all(axis=0)
    nodata = (x10 <= 0).any(axis=0)
    valid = scl_ok & finite_ok & (~nodata) & (~unknown)

    mask_px = int((mask > 0).sum())
    labelled_under_invalid = float((mask[~valid] > 0).sum() / max(mask_px, 1))
    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    trunc = float((mask[border] > 0).sum() / max(mask_px, 1))
    return {
        "x10": x10.astype(np.int16),
        "mask": mask,
        "valid": valid.astype(bool),
        "valid_ratio": float(valid.mean()),
        "mask_px": mask_px,
        "labels_under_invalid_ratio": labelled_under_invalid,
        "label_border_truncation_ratio": trunc,
        "unknown_px": int(unknown.sum()),
    }


def _usable_candidates(ref: date, t: np.ndarray, scl: np.ndarray, latest: date) -> list[dict]:
    out = []
    for i in range(scl.shape[0]):
        d = ref + timedelta(days=int(t[i]))
        if d < latest:
            continue
        cloud = float(np.isin(scl[i], CLOUD_CODES).mean())
        out.append(
            {
                "index": int(i),
                "date": d.isoformat(),
                "cloud": round(cloud, 4),
                "days_after_latest": int((d - latest).days),
            }
        )
    return sorted(out, key=lambda c: c["date"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-config", default=str(ROOT / "configs" / "segformer" / "base.yaml"))
    ap.add_argument("--version", default="s2_10band_v2_clean")
    args = ap.parse_args()
    cfg = load_segformer_config(args.parent_config)["segformer"]
    v1 = read_manifest(resolve_path(cfg["data"]["manifest"]))
    v2_name = args.version
    cache_v1 = resolve_path(cfg["data"]["cache_dir"])
    cache_v2 = resolve_path("data") / v2_name
    cache_v2.mkdir(parents=True, exist_ok=True)
    manifest_v2 = resolve_path("manifests") / f"{v2_name}.jsonl"
    quarantine_v2 = resolve_path("manifests") / f"{v2_name}_quarantine.jsonl"
    stats_v2 = cache_v2 / "stats.json"

    kept, quarantined = [], []
    rule_counter = Counter()
    reselect_days = []

    for rec in v1:
        dates = sorted(DATE_RE.findall(str(rec.get("event_date", ""))))
        rec = dict(rec)
        rec["parent_version"] = cfg["data"]["version"]
        rec["event_date_list"] = dates
        rec["n_event_dates"] = len(dates)

        if len(dates) <= 1:
            latest = dates[0] if dates else None
            rec["event_date_ref"] = latest
            rec["timing_rule"] = "single_date"
            src = Path(rec["cache"])
            dst = cache_v2 / src.name
            if not dst.is_file():
                shutil.copy2(src, dst)
            rec["cache"] = str(dst).replace("\\", "/")
            rec["cache_sha256"] = _sha256(dst)
            kept.append(rec)
            rule_counter["single_date"] += 1
            continue

        latest = dates[-1]
        chosen = date.fromisoformat(rec["date"])
        rec["event_date_ref"] = latest
        rec["days_after_event_first"] = rec.get("days_after_event")

        if chosen >= date.fromisoformat(latest):
            rec["timing_rule"] = "chosen_after_latest"
            rec["days_after_event_ref"] = int((chosen - date.fromisoformat(latest)).days)
            rec["days_after_event"] = rec["days_after_event_ref"]
            src = Path(rec["cache"])
            dst = cache_v2 / src.name
            if not dst.is_file():
                shutil.copy2(src, dst)
            rec["cache"] = str(dst).replace("\\", "/")
            rec["cache_sha256"] = _sha256(dst)
            kept.append(rec)
            rule_counter["chosen_after_latest"] += 1
            continue

        # 需要重选：最晚日期之后 120 天内的最早可用时相
        path = Path(rec["source_uri"])
        ref, t, scl, mask = _read_time_series(path)
        cands = [
            c
            for c in _usable_candidates(ref, t, scl, date.fromisoformat(latest))
            if c["days_after_latest"] <= WINDOW_DAYS and c["cloud"] <= MAX_CLOUD
        ]
        selected = None
        for cand in cands:
            built = _build_sample(path, cand["index"], scl[cand["index"]], mask[cand["index"]])
            if (
                built["valid_ratio"] >= MIN_VALID_RATIO
                and built["labels_under_invalid_ratio"] <= MAX_LABEL_INVALID
            ):
                selected = (cand, built)
                break
        if selected is None:
            quarantined.append(
                {
                    "sample_id": rec["sample_id"],
                    "event_id": rec["event_id"],
                    "reasons": ["multi_event_unresolvable: 最晚事件日期之后 120 天内无可用时相"],
                    "category": "multi_event_unresolvable",
                    "event_date_list": dates,
                    "latest_event_date": latest,
                    "v1_chosen_date": rec["date"],
                    "n_candidates": len(cands),
                }
            )
            rule_counter["quarantined"] += 1
            continue

        cand, built = selected
        v1_chosen = rec["date"]
        dst = cache_v2 / f"{rec['sample_id']}.npz"
        np.savez_compressed(dst, x10=built["x10"], mask=built["mask"], valid=built["valid"])
        rec.update(
            {
                "date": cand["date"],
                "time_index": cand["index"],
                "days_after_event": cand["days_after_latest"],
                "days_after_event_ref": cand["days_after_latest"],
                "cloud_fraction": cand["cloud"],
                "valid_ratio": round(built["valid_ratio"], 4),
                "mask_px": built["mask_px"],
                "mask_coverage": round(built["mask_px"] / 16384, 6),
                "label_status": "positive" if built["mask_px"] > 0 else "empty_annotated",
                "labels_under_invalid_ratio": round(built["labels_under_invalid_ratio"], 4),
                "label_border_truncation_ratio": round(built["label_border_truncation_ratio"], 4),
                "unknown_px": built["unknown_px"],
                "timing_rule": "reselect_within_window",
                "temporal_fallback": False,
                "cache": str(dst).replace("\\", "/"),
                "cache_sha256": _sha256(dst),
            }
        )
        rec["observation_selection"] = {
            "rule": "earliest usable timestep within 120 days after latest event date",
            "candidates": cands[:8],
            "chosen": cand,
            "v1_chosen_date": v1_chosen,
        }
        reselect_days.append(cand["days_after_latest"])
        kept.append(rec)
        rule_counter["reselect_within_window"] += 1

    # 统计只用新 train
    train = [r for r in kept if r.get("split") == "train"]
    stats = compute_stats(train, stats_v2)
    stats["data_version"] = v2_name
    stats["parent_version"] = cfg["data"]["version"]
    stats["timing_rule"] = "see manifests/s2_10band_v2_clean_summary.json"
    stats_v2.write_text(json.dumps(stats, indent=1, ensure_ascii=False), encoding="utf-8")

    kept.sort(key=lambda r: (r["split"], r["event_id"], r["sample_id"]))
    quarantined.sort(key=lambda r: (r["category"], r["event_id"], r["sample_id"]))
    manifest_v2.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in kept), encoding="utf-8"
    )
    quarantine_v2.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in quarantined), encoding="utf-8"
    )

    splits = {}
    for split in ("train", "val", "test"):
        rows = [r for r in kept if r.get("split") == split]
        splits[split] = {
            "n_samples": len(rows),
            "events": dict(Counter(r["event_id"] for r in rows)),
            "n_positive_tiles": sum(1 for r in rows if r["mask_px"] > 0),
            "n_negative_tiles": sum(1 for r in rows if r["mask_px"] == 0),
            "positive_px": int(sum(r["mask_px"] for r in rows)),
            "n_reselected": sum(1 for r in rows if r["timing_rule"] == "reselect_within_window"),
        }

    summary = {
        "data_version": v2_name,
        "parent_version": cfg["data"]["version"],
        "n_v1": len(v1),
        "n_kept": len(kept),
        "n_quarantined": len(quarantined),
        "rules": {
            "window_days": WINDOW_DAYS,
            "keep": "chosen_date >= latest event date",
            "reselect": "earliest usable timestep within window after latest event date",
            "quarantine": "no usable timestep in window",
            "note": "只裁决时间契约；未改标签、未用模型输出、未把日期简单改成 max(event_date)。",
        },
        "timing_rule_counts": dict(rule_counter),
        "reselect_days_after_latest": (
            {
                "n": len(reselect_days),
                "min": min(reselect_days),
                "p50": float(np.median(reselect_days)),
                "max": max(reselect_days),
            }
            if reselect_days
            else {}
        ),
        "quarantine_by_region": dict(Counter(r["event_id"] for r in quarantined)),
        "splits": splits,
        "stats": {"mean": stats["mean"], "std": stats["std"], "fit_split": "train"},
        "identity": {
            "parent_manifest": _sha256(resolve_path(cfg["data"]["manifest"])),
            "parent_stats": _sha256(resolve_path(cfg["data"]["stats"])),
            "script_sha256": _sha256(Path(__file__)),
            "manifest_sha256": _sha256(manifest_v2),
            "stats_sha256": _sha256(stats_v2),
            "quarantine_sha256": _sha256(quarantine_v2),
        },
    }
    summary_path = resolve_path("manifests") / f"{v2_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")

    # 验证 val/test 与 v1 完全一致（决策只影响 train）
    v1_by_split = {s: {r["sample_id"] for r in v1 if r["split"] == s} for s in ("val", "test")}
    v2_by_split = {s: {r["sample_id"] for r in kept if r["split"] == s} for s in ("val", "test")}
    for s in ("val", "test"):
        assert v1_by_split[s] == v2_by_split[s], f"{s} split 不应被多日期裁决改变"

    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"[saved] {manifest_v2}")


if __name__ == "__main__":
    main()
