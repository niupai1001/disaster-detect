"""数据认证：把"能用"的样本固化成清单，把无法核实的隔离。

认证项（每一条都有可复现的证据字段）：
  C1 源产品   —— STAC 同一天、同一点命中同一景 L2A；记录 item id / datetime / 云量
  C2 波段     —— 10 波段 + 回补 B01/B09，无 NaN
  C3 空间网格 —— 128×128 @10 m，CRS 一致，坐标与范围自洽
  C4 辐射     —— DN ∈ [0, 10000]（±5% 容差），记录各波段中位数
  C5 配准     —— STAC 的 B08 与 Sen12 的 B08 相关系数 ≥ 0.90（跨源交叉验证）
  C6 标签     —— {0,1}；正样本掩膜像素 ≥ 20；掩膜落在云/无效区的比例 ≤ 20%
  C7 有效像素 —— SCL 派生 valid 比例 ≥ 0.5

输出：
  data/certified/<sample_id>.npz        缓存（x12, x10, mask, valid）
  manifests/certified.jsonl             通过认证的清单
  manifests/quarantine.jsonl            隔离清单（含原因）

用法：
  python scripts/certify_data.py --per-event 40 --neg-per-event 25
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from landslide_seg.data.band_mapping import S2_L2A_BANDS  # noqa: E402
from landslide_seg.data.s2_backfill import (  # noqa: E402
    backfill_b01_b09,
    find_same_day_item,
    read_asset_window,
)
from landslide_seg.data.sen12 import SEN12_S2_BANDS, load_sen12  # noqa: E402

CLOUD_CODES = (2, 3, 8, 9, 10)


def event_of(path: Path) -> str:
    return re.sub(r"_s2_\d+$", "", path.stem)


def cheap_scan(path: Path) -> dict | None:
    """不联网的初筛：读元数据、掩膜、SCL 与 time，直接做"灾后最清晰时相"选择。

    只用 ``pre_post['post']`` 会误杀大量样本：实测很多事件的 post 时相整幅被云覆盖
    （italy 某样本 SCL=9 占 100%），而同一景更晚的灾后时相是晴空。
    """
    import ast
    import datetime as _dt

    import h5py

    try:
        with h5py.File(path, "r") as f:
            mask = f["MASK"][:]
            scl = f["SCL"][:] if "SCL" in f else None
            time = f["time"][:]
            ed = f.attrs.get("event_date", b"")
            crs = f.attrs.get("crs", b"")
            units = f["time"].attrs.get("units", b"")
    except Exception:
        return None

    ed = ed.decode("utf-8", "ignore") if isinstance(ed, bytes) else str(ed)
    crs = crs.decode("utf-8", "ignore") if isinstance(crs, bytes) else str(crs)
    units = units.decode("utf-8", "ignore") if isinstance(units, bytes) else str(units)
    rec = {
        "path": str(path), "event": event_of(path), "post": None,
        "mask_px": 0, "cloud": 1.0, "valid": 0.0,
        "event_date": ed, "crs": crs, "select_reason": "",
    }
    if not ed:
        rec["select_reason"] = "missing event_date"
        return rec
    try:
        event_dt = _dt.date.fromisoformat(ed[:10])
        ref = _dt.date.fromisoformat(units.split("since")[-1].strip())
    except Exception:
        rec["select_reason"] = "unparsable dates"
        return rec

    best = None
    for t in range(mask.shape[0]):
        d = ref + _dt.timedelta(days=int(time[t]))
        if d < event_dt:
            continue
        c = float(np.isin(scl[t], CLOUD_CODES).mean()) if scl is not None else 1.0
        cand = {"index": t, "date": d.isoformat(), "cloud": c}
        if best is None or (c, d) < (best["cloud"], _dt.date.fromisoformat(best["date"])):
            best = cand
    if best is None:
        rec["select_reason"] = "no post-event timestep"
        return rec
    rec["post"] = best["index"]
    rec["cloud"] = best["cloud"]
    rec["valid"] = 1.0 - best["cloud"]
    rec["date"] = best["date"]
    rec["mask_px"] = int((mask[best["index"]] > 0).sum())
    return rec


_ITEM_CACHE: dict = {}


def _cached_item(lon: float, lat: float, date):
    key = (round(lon, 2), round(lat, 2), date.isoformat())
    if key not in _ITEM_CACHE:
        _ITEM_CACHE[key] = find_same_day_item(lon, lat, date)
    return _ITEM_CACHE[key]


def certify_one(cand: dict, out_dir: Path) -> tuple[dict | None, dict | None]:
    """完整认证单个样本，返回 (通过记录, 隔离记录)。"""
    path = Path(cand["path"])
    sid = path.stem
    s = load_sen12(path)
    post, sel = s.select_observation(max_cloud=0.5)
    if post is None:
        return None, {"sample_id": sid, "event": cand["event"],
                      "reasons": [f"C1 {sel['reason']}"], "qc": {}}

    reasons: list[str] = []
    qc: dict = {}

    # C3 网格
    shape_ok = s.mask.shape[1:] == (128, 128) and s.bands["B02"].shape[1:] == (128, 128)
    if not shape_ok:
        reasons.append(f"C3 网格非 128x128: {s.mask.shape[1:]}")

    # C2 波段 / C4 辐射
    band_stats = {}
    for b in SEN12_S2_BANDS:
        a = s.bands[b][post].astype(np.float32)
        band_stats[b] = {"min": float(a.min()), "max": float(a.max()),
                         "median": float(np.median(a))}
        if not np.isfinite(a).all():
            reasons.append(f"C2 {b} 含 NaN/Inf")
        if a.max() > 10500 or a.min() < -10:
            reasons.append(f"C4 {b} 超出 DN 范围 [{a.min():.0f},{a.max():.0f}]")

    # C7 有效像素
    valid = s.valid_mask(post)
    if valid.mean() < 0.5:
        reasons.append(f"C7 有效像素仅 {valid.mean():.1%}")

    # C1/C5 回补 + 配准交叉验证
    bf = backfill_b01_b09(s, post)
    align = {"available": False}
    if bf is None:
        reasons.append("C1 同一天无 L2A 产品（拒绝跨日替代）")
    else:
        import datetime as _dt

        from pyproj import Transformer

        date = s.attrs["_time_ref"] + _dt.timedelta(days=int(s.time[post]))
        tr = Transformer.from_crs(s.attrs["crs"], "EPSG:4326", always_xy=True)
        lon, lat = tr.transform(s.attrs["center_x"], s.attrs["center_y"])
        item = _cached_item(lon, lat, date)
        cx, cy = s.attrs["center_x"], s.attrs["center_y"]
        bounds = (cx - 640, cy - 640, cx + 640, cy + 640)
        ref = read_asset_window(item["assets"]["nir"]["href"], bounds,
                               s.attrs["crs"], (128, 128))
        ours = s.bands["B08"][post].astype(np.float32)
        ok = (ref > 0) & (ours > 0)
        corr = float(np.corrcoef(ref[ok], ours[ok])[0, 1]) if ok.sum() > 100 else float("nan")
        align = {"available": True, "corr_B08": round(corr, 4), "n_common": int(ok.sum())}
        if not (corr >= 0.90):
            reasons.append(f"C5 跨源配准相关仅 {corr:.3f} < 0.90")

        gap = (bf.bands["B01"] <= 0) | (bf.bands["B09"] <= 0)
        valid = valid & (~gap)
        qc["backfill_gap"] = float(gap.mean())

    # C6 标签
    gt = s.mask[post]
    uniq = np.unique(gt)
    if not set(uniq.tolist()).issubset({0, 1}):
        reasons.append(f"C6 掩膜取值异常 {uniq[:5].tolist()}")
    mask_px = int((gt > 0).sum())
    labelled_under_invalid = float((gt[~valid] > 0).sum() / max(mask_px, 1))
    if labelled_under_invalid > 0.2:
        reasons.append(f"C6 {labelled_under_invalid:.0%} 的滑坡像素落在云/无效区，标签不可核实")

    if reasons:
        return None, {"sample_id": sid, "event": cand["event"], "reasons": reasons,
                      "qc": {**qc, "align": align}}

    # 组装并缓存
    x12 = np.zeros((12, 128, 128), dtype=np.float32)
    for i, name in enumerate(S2_L2A_BANDS):
        if name in bf.bands:
            x12[i] = bf.bands[name]
        else:
            x12[i] = s.bands[name][post].astype(np.float32)
    x10 = np.stack([s.bands[b][post] for b in SEN12_S2_BANDS]).astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f"{sid}.npz"
    np.savez_compressed(
        npz, x12=x12, x10=x10, mask=gt.astype(np.int64), valid=valid,
        doy=np.int64(s.doy(post)),
    )

    rec = {
        "sample_id": sid,
        "event_id": cand["event"],
        "date": (s.attrs["_time_ref"] + __import__("datetime").timedelta(days=int(s.time[post]))).isoformat(),
        "event_date": s.attrs.get("event_date"),
        "time_index": post,
        "observation_selection": sel,
        "crs": s.attrs["crs"],
        "bbox": [cx - 640, cy - 640, cx + 640, cy + 640],
        "center_xy": [cx, cy],
        "bands_present": list(S2_L2A_BANDS),
        "backfilled_bands": ["B01", "B09"],
        "backfill_source": {
            "item_id": bf.item_id,
            "datetime": bf.datetime,
            "cloud_cover": bf.cloud_cover,
            "method": bf.method,
            "gap_ratio": float(qc.get("backfill_gap", 0.0)),
        },
        "radiometry": {"scale": "DN 0-10000", "band_stats": band_stats},
        "grid": {"size": [128, 128], "resolution_m": 10.0},
        "cloud_fraction": round(float(np.isin(s.scl[post], CLOUD_CODES).mean()), 4),
        "valid_ratio": round(float(valid.mean()), 4),
        "mask_coverage": round(float((gt > 0).mean()), 6),
        "mask_px": mask_px,
        "label_status": "positive" if mask_px > 0 else "unlabeled",
        "labels_under_invalid_ratio": round(labelled_under_invalid, 4),
        "alignment": align,
        "doy": int(s.doy(post)),
        "source_uri": str(path).replace("\\", "/"),
        "cache": str(npz).replace("\\", "/"),
        "cache_sha256": hashlib.sha256(npz.read_bytes()).hexdigest()[:16],
        "split": None,
    }
    return rec, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "data" / "sen12"))
    ap.add_argument("--per-event", type=int, default=40)
    ap.add_argument("--neg-per-event", type=int, default=25)
    ap.add_argument("--max-cloud", type=float, default=0.30)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "manifests"))
    ap.add_argument("--csv-out", default=str(ROOT / "data" / "certified"))
    args = ap.parse_args()

    files = sorted(Path(args.root).glob("*_s2_*.nc"))
    print(f"[scan] {len(files)} 个文件")
    cands = []
    for p in files:
        c = cheap_scan(p)
        if c:
            cands.append(c)
    by_event: dict[str, list[dict]] = {}
    for c in cands:
        by_event.setdefault(c["event"], []).append(c)
    print("[scan] 事件:", {k: len(v) for k, v in by_event.items()})

    selected: list[dict] = []
    for ev, rows in sorted(by_event.items()):
        pos = [r for r in rows if r["mask_px"] >= 20 and r["cloud"] <= args.max_cloud]
        neg = [r for r in rows if r["mask_px"] == 0 and r["cloud"] <= 0.15]
        pos.sort(key=lambda r: (-r["mask_px"], r["cloud"]))
        neg.sort(key=lambda r: r["cloud"])
        # 正样本按掩膜像素分档抽样，避免全是大目标
        chunk = max(1, len(pos) // max(args.per_event, 1))
        pick_pos = pos[::chunk][: args.per_event] if pos else []
        pick_neg = neg[: args.neg_per_event]
        selected.extend(pick_pos + pick_neg)
        print(f"[select] {ev}: 正 {len(pick_pos)} / 负 {len(pick_neg)}")

    out_dir = Path(args.out)
    csv_dir = Path(args.csv_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    certified, quarantined = [], []
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(certify_one, c, csv_dir): c for c in selected}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                rec, bad = fut.result()
            except Exception as exc:
                c = futs[fut]
                quarantined.append({"sample_id": Path(c["path"]).stem, "event": c["event"],
                                    "reasons": [f"EXC {type(exc).__name__}: {exc}"], "qc": {}})
                continue
            if rec:
                certified.append(rec)
            if bad:
                quarantined.append(bad)
            if done % 10 == 0:
                print(f"  ... {done}/{len(selected)}  通过 {len(certified)} / 隔离 {len(quarantined)}")

    (out_dir / "certified.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in certified),
        encoding="utf-8",
    )
    (out_dir / "quarantine.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in quarantined),
        encoding="utf-8",
    )

    # 未标注图块（无掩膜、无事件日期）不能当负样本，单独记入隔离清单
    unannot = [c for c in cands if c["mask_px"] == 0 and c["select_reason"] == "missing event_date"]
    per_ev = {}
    for c in unannot:
        if per_ev.get(c["event"], 0) >= 5:
            continue
        per_ev[c["event"]] = per_ev.get(c["event"], 0) + 1
        quarantined.append({
            "sample_id": Path(c["path"]).stem,
            "event": c["event"],
            "reasons": ["unannotated: 无掩膜且无 event_date —— 属于未标注，"
                        "不能当作已验证的无滑坡负样本"],
            "qc": {},
        })
    print(f"[negatives] 未标注图块 {len(unannot)} 个（已隔离，未作为负样本使用）")

    pos = sum(1 for r in certified if r["label_status"] == "positive")
    print(f"\n[certified] {len(certified)} 个（正 {pos} / 负 {len(certified)-pos}）")
    print(f"[quarantine] {len(quarantined)} 个")
    from collections import Counter

    print("[events]", dict(Counter(r["event_id"] for r in certified)))
    print(f"[saved] {out_dir/'certified.jsonl'}")


if __name__ == "__main__":
    main()
