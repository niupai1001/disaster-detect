"""固定事件清单逐帧日期、云和缺测侧表；不改变缓存、成员和评价有效区。"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from landslide_seg.config import file_sha256, resolve_path
from landslide_seg.data.segformer_dataset import S2_10_BANDS, read_manifest


def text_attr(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def main():
    manifest = ROOT / "manifests/v4r1/pool_100.jsonl"
    index = ROOT / "data/s2_temporal_v4r1/index.json"
    members = {r["sample_id"]: r for r in json.loads(index.read_text())["rows"]}
    out = ROOT / "docs/assets/research/temporal_observations.csv"
    summary = {"manifest_sha256": file_sha256(manifest), "index_sha256": file_sha256(index),
               "n_samples": 0, "n_frames": 0, "nodata_pixels": 0, "ambiguous_event_date_samples": 0, "errors": [],
               "date_note": "灾害日期来自原文件元数据；未独立核实灾害记录。当天归为POST，之前为PRE。",
               "radiometry": {}, "eligibility": {}}
    fields = ["sample_id", "event_id", "split", "frame_id", "source_frame_id", "date", "event_date", "date_confidence",
              "anchor_date", "anchor", "after_anchor", "pre_post", "days_from_event", "source_sha256",
              "nodata_fraction", "cloud_shadow_fraction", "unknown255_fraction"]
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in read_manifest(manifest):
            sid = row["sample_id"]
            try:
                path = ROOT / "data/sen12" / f"{sid}.nc"
                digest = file_sha256(path)
                dates = np.asarray(members[sid]["dates"], dtype="datetime64[D]")
                try:
                    event = np.datetime64(row["event_date"])
                    if np.isnat(event):
                        event = None
                except ValueError:
                    event = None
                if event is None:
                    summary["ambiguous_event_date_samples"] += 1
                with h5py.File(path) as nc:
                    source_event = text_attr(nc.attrs.get("event_date", ""))
                    if source_event != row["event_date"]:
                        raise ValueError(f"source event_date {source_event} != manifest")
                    confidence = text_attr(nc.attrs.get("date_confidence", "unknown"))
                    units = text_attr(nc["time"].attrs["units"])
                    if not units.startswith("days since "):
                        raise ValueError(f"未适配的源日期单位: {units}")
                    times = nc["time"][:]
                    if not np.isfinite(times).all() or not np.equal(times, np.floor(times)).all():
                        raise ValueError("源日期不是有限整数天")
                    source_dates = np.datetime64(units[len("days since "):].strip()) + times.astype("timedelta64[D]")
                    order = np.argsort(source_dates, kind="stable")
                    if not np.array_equal(source_dates[order], dates):
                        raise ValueError("源观测日期与缓存索引不一致")
                    scl = nc["SCL"][:]
                    nodata = scl == 0
                    for band in S2_10_BANDS:
                        v = nc[band][:]
                        nodata |= ~np.isfinite(v)
                        fill = nc[band].attrs.get("_FillValue")
                        if fill is not None and np.isfinite(fill).all():
                            nodata |= v == np.asarray(fill).item()
                        attrs = {k: text_attr(nc[band].attrs[k]) for k in ("scale_factor", "add_offset", "units", "_FillValue") if k in nc[band].attrs}
                        summary["radiometry"].setdefault(band, [])
                        if attrs not in summary["radiometry"][band]:
                            summary["radiometry"][band].append(attrs)
                    scl, nodata = scl[order], nodata[order]
                    for i, date in enumerate(dates):
                        writer.writerow(dict(sample_id=sid, event_id=row["event_id"], split=row["split"],
                            frame_id=i, source_frame_id=int(order[i]), date=str(date), event_date=row["event_date"], date_confidence=confidence,
                            anchor_date=row["date"], anchor=i == row["time_index"],
                            after_anchor=bool(date > np.datetime64(row["date"])),
                            pre_post=("AMBIGUOUS_EVENT_DATE" if event is None else "PRE" if date < event else "POST"),
                            days_from_event="" if event is None else int((date-event).astype(int)),
                            source_sha256=digest, nodata_fraction=float(nodata[i].mean()),
                            cloud_shadow_fraction=float(np.isin(scl[i], [3, 8, 9, 10]).mean()),
                            unknown255_fraction=float((scl[i] == 255).mean())))
                    summary["nodata_pixels"] += int(nodata.sum())
                summary["n_samples"] += 1
                summary["n_frames"] += len(dates)
                for mode, available in (("retro", len(dates)), ("as_of_anchor", int((dates <= np.datetime64(row["date"])).sum()))):
                    for n in (1, 3, 5, 7, 15):
                        key = f"{row['split']}/{mode}/{n}"
                        summary["eligibility"][key] = summary["eligibility"].get(key, 0) + int(available >= n)
                if summary["n_samples"] % 200 == 0:
                    print(summary["n_samples"], flush=True)
            except (OSError, KeyError, ValueError) as exc:
                summary["errors"].append({"sample_id": sid, "error": str(exc)})
    summary["csv_sha256"] = file_sha256(out)
    out.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: (len(v) if k == "errors" else v) for k, v in summary.items() if k != "radiometry"}, ensure_ascii=False))
    raise SystemExit(bool(summary["errors"]))


if __name__ == "__main__":
    main()
