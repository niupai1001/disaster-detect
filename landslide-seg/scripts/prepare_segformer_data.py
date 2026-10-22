"""P0：生成 SegFormer 用的 ``s2_10band_v1`` 清单与缓存。

与旧的 12 波段认证流程的区别（对应执行计划 §5）：
1. **不需要** B01/B09 回补，因此原 C1/C5（同日 STAC 产品、跨源配准）不再是准入条件；
   10 波段完全来自 Sen12 单文件，标签网格与影像同源。
2. 取消"正像素 ≥ 20"的自动筛除；极小目标保留并单独统计（否则在数据入口就丢了研究重点）。
3. ``valid10`` 由 SCL 允许清单 ∩ 各波段有限值/非 nodata ∩ 标签已知 三者的交集生成。
4. 时相选择记录采集日期、time_index、距事件天数、云量、候选列表，供事后复核。
5. 划分按事件整体互斥（默认 test=italy / val=hiroshima），不再做内部 val-block 切分。
6. 标准化统计只用训练集有效像素，val/test 只应用。

v2 修正（2026-10-18，见 ``docs/segformer_execution_plan_2026-10-18_v2.md`` §4）：
- **D1** ``center_lon/center_lat`` 是数值数组，被 ``_decode`` 转成 ``'[480640.]'`` 后不能直接
  ``float()``；旧版因此把 2,161 个已通过筛选的样本全部记成 ``EXC`` 隔离、清单为 0 行。
- **D2** SCL 改用**允许清单** ``{4,5,6,7}``；255 等未定义码以前会被当成有效（实测影响 131 瓦片）。
- **D3** 时相不再只看"灾后最低云量"（实测与标注给出的 post 索引 91% 不一致、p90 滞后 193 天），
  改为"灾后窗口内最早的可用时相"，窗口内无解才退化为全灾后最低云量并落 ``temporal_fallback``。
- **D4** 统计文件增加逐波段分位数；summary 增加脚本/规则/清单/统计哈希，便于事后核对。

未标注图块（``annotated=False``、无 mask、无 event_date）一律隔离，不当负样本。

用法::

    python scripts/prepare_segformer_data.py --config configs/segformer/base.yaml --limit 30
    python scripts/prepare_segformer_data.py --config configs/segformer/base.yaml --workers 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:  # Windows 控制台/管道默认 cp1252，直接 print 中文会 UnicodeEncodeError
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np  # noqa: E402

from landslide_seg.config import load_segformer_config  # noqa: E402
from landslide_seg.data.segformer_dataset import S2_10_BANDS  # noqa: E402

#: 云 + 云影（SCL 2=暗区/3=云影/8=中概率云/9=高概率云/10=薄卷云）
CLOUD_CODES = (2, 3, 8, 9, 10)
#: 允许的 SCL 类别：4 植被 / 5 裸地 / 6 水 / 7 未分类。
#: 用**允许清单**而不是排除清单，任何未定义码（0/1/2/3/8/9/10/11/255/其它）都算无效。
SCL_VALID = (4, 5, 6, 7)
#: 掩膜里的"未知"标记。Sen12 实测只有 {0,1}，但保留显式处理以防其它分卷出现。
UNKNOWN_MASK_VALUE = 255


def event_of(path: Path) -> str:
    return re.sub(r"_s2_\d+$", "", path.stem)


def _decode(v) -> str:
    return v.decode("utf-8", "ignore") if isinstance(v, bytes) else str(v)


def _num(v) -> float | None:
    """把 nc 属性解析成标量。

    D1 的根因就在这里：``f.attrs["center_lon"]`` 是 ``array([480640.])``，直接
    ``_decode`` 得到 ``'[480640.]'``，再 ``float()`` 必崩。这里对字符串走正则、
    对数组走 ravel，两种形态都能取到第一个数值。
    """
    if v is None:
        return None
    if isinstance(v, (bytes, bytearray)):
        v = v.decode("utf-8", "ignore")
    if isinstance(v, str):
        m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", v)
        return float(m.group()) if m else None
    try:
        return float(np.ravel(v)[0])
    except Exception:
        return None


def _parse_pre_post(s: str) -> dict:
    """解析 ``"{'pre': 8, 'post': 9}"``，同时兼容双引号/JSON。"""
    if not s:
        return {}
    out: dict[str, int] = {}
    for k, v in re.findall(r"['\"]?(\w+)['\"]?\s*:\s*(\d+)", s):
        out[k] = int(v)
    return out


def _select_timestep(
    mask_all: np.ndarray,
    scl_all: np.ndarray | None,
    time: np.ndarray,
    ref: date,
    event_dt: date,
    cfg: dict,
) -> tuple[dict | None, bool, list[dict]]:
    """灾后窗口内最早可用；窗口内无解 → 全灾后最低云量并标记 fallback。"""
    cands: list[dict] = []
    for t in range(mask_all.shape[0]):
        d = ref + timedelta(days=int(time[t]))
        if d < event_dt:
            continue
        if scl_all is None:
            cloud, scl_ratio = 1.0, 0.0
        else:
            cloud = float(np.isin(scl_all[t], CLOUD_CODES).mean())
            scl_ratio = float(np.isin(scl_all[t], SCL_VALID).mean())
        cands.append(
            {
                "index": int(t),
                "date": d.isoformat(),
                "cloud": round(cloud, 4),
                "scl_valid_ratio": round(scl_ratio, 4),
                "days_after_event": int((d - event_dt).days),
            }
        )
    if not cands:
        return None, False, []

    window = int(cfg.get("post_event_window_days", 120))
    usable = [
        c
        for c in cands
        if c["days_after_event"] <= window
        and c["cloud"] <= cfg["max_cloud"]
        and c["scl_valid_ratio"] >= cfg["min_valid_ratio"]
    ]
    if usable:
        # 最早优先（同一天不可能重复），云量只作为并列时的次级键
        chosen = min(usable, key=lambda c: (c["days_after_event"], c["cloud"], c["date"]))
        fallback = False
    else:
        chosen = min(cands, key=lambda c: (c["cloud"], c["date"]))
        fallback = True
    return chosen, fallback, cands


def process_one(path: Path, cfg: dict, cache_dir: Path) -> tuple[dict | None, dict | None]:
    """处理单个 nc；返回 (清单记录, 隔离记录)。"""
    import h5py

    sid = path.stem
    ev = event_of(path)
    try:
        with h5py.File(path, "r") as f:
            attrs = {k: _decode(v) for k, v in f.attrs.items()}
            units = _decode(f["time"].attrs.get("units", ""))
            mask_all = f["MASK"][:]
            scl_all = f["SCL"][:] if "SCL" in f else None
            time = f["time"][:]
            band_names = [b for b in S2_10_BANDS if b in f]
            shape = tuple(f["B02"].shape[1:])
    except Exception as exc:  # 文件损坏/结构不符
        return None, {"sample_id": sid, "event": ev, "reasons": [f"READ {type(exc).__name__}: {exc}"]}

    annotated = attrs.get("annotated", "").strip().lower() in ("true", "1")
    if not annotated:
        return None, {
            "sample_id": sid, "event": ev,
            "reasons": ["unannotated: 无掩膜/无 event_date，属于未标注，不能当负样本"],
            "category": "unannotated",
        }

    ed = attrs.get("event_date", "").strip()
    if not ed:
        return None, {"sample_id": sid, "event": ev, "reasons": ["missing event_date"]}
    try:
        event_dt = date.fromisoformat(ed[:10])
        ref = date.fromisoformat(units.split("since")[-1].strip())
    except Exception:
        return None, {"sample_id": sid, "event": ev, "reasons": [f"unparsable dates ({ed!r}, {units!r})"]}

    if len(band_names) != 10:
        return None, {"sample_id": sid, "event": ev,
                      "reasons": [f"missing bands: {sorted(set(S2_10_BANDS) - set(band_names))}"]}
    if shape != (128, 128):
        return None, {"sample_id": sid, "event": ev, "reasons": [f"grid {shape} != (128,128)"]}

    chosen, fallback, cands = _select_timestep(mask_all, scl_all, time, ref, event_dt, cfg)
    if chosen is None:
        return None, {"sample_id": sid, "event": ev, "reasons": ["no post-event timestep"]}
    if chosen["cloud"] > cfg["max_cloud"]:
        return None, {"sample_id": sid, "event": ev,
                      "reasons": [f"best post-event cloud {chosen['cloud']:.0%} > {cfg['max_cloud']:.0%}"]}

    idx = chosen["index"]
    mask_raw = mask_all[idx]
    unknown = mask_raw == UNKNOWN_MASK_VALUE
    mask = np.where(unknown, 0, mask_raw).astype(np.uint8)
    scl = scl_all[idx] if scl_all is not None else None

    with h5py.File(path, "r") as f:
        x10 = np.stack([f[b][idx] for b in S2_10_BANDS]).astype(np.float32)

    # ---- valid10：SCL 允许清单 ∩ 波段有限/非 nodata ∩ 标签已知 ----
    if scl is None:
        scl_ok = np.ones(mask.shape, dtype=bool)
    else:
        scl_ok = np.isin(scl, SCL_VALID)
    finite_ok = np.isfinite(x10).all(axis=0)
    nodata = (x10 <= 0).any(axis=0)          # DN 不可能为负；0 是填充/无观测
    label_known = ~unknown
    valid = scl_ok & finite_ok & (~nodata) & label_known

    uniq = np.unique(mask)
    if not set(uniq.tolist()).issubset({0, 1}):
        return None, {"sample_id": sid, "event": ev, "reasons": [f"mask values {uniq[:5].tolist()}"]}

    valid_ratio = float(valid.mean())
    if valid_ratio < cfg["min_valid_ratio"]:
        return None, {"sample_id": sid, "event": ev,
                      "reasons": [f"valid ratio {valid_ratio:.1%} < {cfg['min_valid_ratio']:.0%}"]}

    mask_px = int((mask > 0).sum())
    labelled_under_invalid = float((mask[~valid] > 0).sum() / max(mask_px, 1))
    if labelled_under_invalid > cfg["max_label_invalid_ratio"]:
        return None, {"sample_id": sid, "event": ev,
                      "reasons": [f"{labelled_under_invalid:.0%} 的滑坡像素不可观测，监督不可核实"]}

    # 标签在瓦片边界被截断的比例（对象可能跨块）
    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    trunc = float((mask[border] > 0).sum() / max(mask_px, 1))

    cache_dir.mkdir(parents=True, exist_ok=True)
    npz = cache_dir / f"{sid}.npz"
    np.savez_compressed(
        npz,
        x10=x10.astype(np.int16),
        mask=mask,
        valid=valid.astype(bool),
    )

    cx = _num(attrs.get("center_lon"))
    cy = _num(attrs.get("center_lat"))

    pre_post = _parse_pre_post(attrs.get("pre_post_dates", ""))
    post_index = pre_post.get("post")
    if post_index is not None and not (0 <= post_index < len(time)):
        post_index = None
    days_at_post = None
    days_delta_vs_post = None
    if post_index is not None:
        post_date = ref + timedelta(days=int(time[post_index]))
        days_at_post = int((post_date - event_dt).days)
        days_delta_vs_post = int(chosen["days_after_event"] - days_at_post)

    rec = {
        "sample_id": sid,
        "event_id": ev,
        "date": chosen["date"],
        "event_date": ed,
        "time_index": idx,
        "days_after_event": chosen["days_after_event"],
        "observation_selection": {
            "n_candidates": len(cands),
            "window_days": int(cfg.get("post_event_window_days", 120)),
            "fallback": fallback,
            "chosen": chosen,
            "candidates": sorted(cands, key=lambda c: c["days_after_event"])[:8],
            "rule": (
                "earliest post-event timestep within window with cloud<=max_cloud and "
                "scl_valid>=min_valid_ratio; else global min-cloud post-event (fallback)"
            ),
        },
        "pre_post_dates": attrs.get("pre_post_dates", ""),
        "post_index": post_index,
        "days_at_post": days_at_post,
        "days_delta_vs_post": days_delta_vs_post,
        "temporal_fallback": bool(fallback),
        "crs": attrs.get("crs"),
        "bbox": [cx - 640, cy - 640, cx + 640, cy + 640] if cx is not None else None,
        "center_xy": [cx, cy] if cx is not None else None,
        "bands": list(S2_10_BANDS),
        "cloud_fraction": chosen["cloud"],
        "scl_valid_ratio_at_chosen": chosen["scl_valid_ratio"],
        "scl255_ratio": round(float((scl == 255).mean()), 4) if scl is not None else 0.0,
        "valid_ratio": round(valid_ratio, 4),
        "nodata_ratio": round(float(nodata.mean()), 4),
        "unknown_px": int(unknown.sum()),
        "mask_px": mask_px,
        "mask_coverage": round(float((mask > 0).mean()), 6),
        "label_status": "positive" if mask_px > 0 else "empty_annotated",
        "labels_under_invalid_ratio": round(labelled_under_invalid, 4),
        "label_border_truncation_ratio": round(trunc, 4),
        "temporal_risk": chosen["days_after_event"] > cfg["warn_days_after_event"],
        "cache": str(npz).replace("\\", "/"),
        "cache_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
        "source_uri": str(path).replace("\\", "/"),
        "split": None,
    }
    return rec, None


def compute_stats(records: list[dict], out_path: Path) -> dict:
    """训练集有效像素的逐波段 mean/std/分位数（DN 尺度）。

    分位数用 5 DN 一格的直方图累计，避免把上千万像素一次性读进内存。
    """
    n = np.zeros(10, dtype=np.float64)
    s = np.zeros(10, dtype=np.float64)
    s2 = np.zeros(10, dtype=np.float64)
    bin_dn = 5.0
    n_bins = 2048  # 0..10240 DN，覆盖 int16 实际量级
    hist = np.zeros((10, n_bins), dtype=np.int64)

    for r in records:
        d = np.load(r["cache"])
        x = d["x10"].astype(np.float64)
        v = d["valid"]
        for b in range(10):
            px = x[b][v]
            if not px.size:
                continue
            n[b] += px.size
            s[b] += px.sum()
            s2[b] += (px * px).sum()
            idx = np.clip((px / bin_dn).astype(np.int64), 0, n_bins - 1)
            hist[b] += np.bincount(idx, minlength=n_bins)

    mean = s / np.maximum(n, 1)
    var = np.maximum(s2 / np.maximum(n, 1) - mean**2, 0.0)
    std = np.sqrt(var)

    quantiles: dict[str, list[float]] = {}
    for name, q in (("p1", 0.01), ("p50", 0.50), ("p99", 0.99)):
        vals = []
        for b in range(10):
            if n[b] == 0:
                vals.append(0.0)
                continue
            cum = np.cumsum(hist[b])
            target = q * n[b]
            i = int(np.searchsorted(cum, target, side="left"))
            i = min(max(i, 0), n_bins - 1)
            vals.append(round((i + 0.5) * bin_dn, 1))
        quantiles[name] = vals

    stats = {
        "data_version": "s2_10band_v1",
        "bands": list(S2_10_BANDS),
        "unit": "DN (Sen12 原始存储值，未施加 BOA offset)",
        "fit_split": "train",
        "n_samples": len(records),
        "n_pixels_per_band": n.astype(int).tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "quantiles": quantiles,
        "note": (
            "逐波段 z-score 由均值吸收常数偏移；+1000 DN 的辐射口径争议不改变模型输入。"
            "val/test 只应用这份统计，不重新拟合。"
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, indent=1, ensure_ascii=False), encoding="utf-8")
    return stats


def _dist(vals: list[float]) -> dict:
    if not vals:
        return {}
    a = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(a.size),
        "min": round(float(a.min()), 2),
        "p10": round(float(np.quantile(a, 0.10)), 2),
        "p50": round(float(np.quantile(a, 0.50)), 2),
        "p90": round(float(np.quantile(a, 0.90)), 2),
        "max": round(float(a.max()), 2),
        "mean": round(float(a.mean()), 2),
    }


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "segformer" / "base.yaml"))
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_segformer_config(args.config, args.set)["segformer"]
    dcfg = cfg["data"]
    root = (ROOT / dcfg["source_root"]).resolve()
    cache_dir = (ROOT / dcfg["cache_dir"]).resolve()
    manifest = (ROOT / dcfg["manifest"]).resolve()
    quar_path = (ROOT / dcfg["quarantine"]).resolve()
    stats_path = (ROOT / dcfg["stats"]).resolve()

    files = sorted(root.glob("*_s2_*.nc"))
    if args.limit:
        files = files[: args.limit]
    print(f"[scan] {len(files)} 个 nc 文件")

    certified: list[dict] = []
    quarantined: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_one, p, dcfg, cache_dir): p for p in files}
        for i, fut in enumerate(as_completed(futs), 1):
            p = futs[fut]
            try:
                rec, bad = fut.result()
            except Exception as exc:
                quarantined.append({"sample_id": p.stem, "event": event_of(p),
                                    "reasons": [f"EXC {type(exc).__name__}: {exc}"]})
                continue
            if rec:
                certified.append(rec)
            if bad:
                quarantined.append(bad)
            if i % 250 == 0:
                print(f"  ... {i}/{len(files)}  通过 {len(certified)} / 隔离 {len(quarantined)}")

    # ---- 事件级互斥划分 ----
    test_ev = dcfg["test_event"]
    val_ev = dcfg["val_event"]
    for r in certified:
        if r["event_id"] == test_ev:
            r["split"] = "test"
        elif r["event_id"] == val_ev:
            r["split"] = "val"
        else:
            r["split"] = "train"

    certified.sort(key=lambda r: (r["split"], r["event_id"], r["sample_id"]))
    quarantined.sort(key=lambda r: (r.get("category", ""), r["event"], r["sample_id"]))

    # ---- 划分泄漏检查（同 CRS 下 bbox 是否相交） ----
    def overlap(a, b) -> bool:
        if not a or not b:
            return False
        return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])

    leaks = []
    by_split = defaultdict(list)
    for r in certified:
        by_split[r["split"]].append(r)
    for s1 in ("train", "val"):
        for s2n in (("val", "test") if s1 == "train" else ("test",)):
            for a in by_split[s1]:
                for b in by_split[s2n]:
                    if a["crs"] == b["crs"] and overlap(a["bbox"], b["bbox"]):
                        leaks.append({"split_a": s1, "a": a["sample_id"],
                                      "split_b": s2n, "b": b["sample_id"]})

    stats = compute_stats(by_split["train"], stats_path)

    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in certified),
                        encoding="utf-8")
    quar_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in quarantined),
                         encoding="utf-8")

    # ---- 汇总 ----
    summary = {}
    for split in ("train", "val", "test"):
        rows = by_split[split]
        pos = [r for r in rows if r["mask_px"] > 0]
        by_date = defaultdict(Counter)
        for r in rows:
            by_date[r["event_id"]][r["event_date"]] += 1
        summary[split] = {
            "n_samples": len(rows),
            "n_events": len({r["event_id"] for r in rows}),
            "events": dict(Counter(r["event_id"] for r in rows)),
            "n_event_dates": len({(r["event_id"], r["event_date"]) for r in rows}),
            "events_by_date": {k: dict(v) for k, v in sorted(by_date.items())},
            "n_positive_tiles": len(pos),
            "n_negative_tiles": len(rows) - len(pos),
            "positive_px": int(sum(r["mask_px"] for r in rows)),
            "small_objects_px_lt_50": int(sum(1 for r in rows if 0 < r["mask_px"] < 50)),
            "valid_px": int(sum(round(r["valid_ratio"] * 128 * 128) for r in rows)),
        }

    report = {
        "data_version": dcfg["version"],
        "n_certified": len(certified),
        "n_quarantined": len(quarantined),
        "splits": summary,
        "split_bbox_leaks": len(leaks),
        "stats": {
            "mean": stats["mean"],
            "std": stats["std"],
            "quantiles": stats["quantiles"],
            "n_pixels_per_band": stats["n_pixels_per_band"],
            "fit_split": "train",
        },
        "temporal": {
            "fallback_n": int(sum(1 for r in certified if r["temporal_fallback"])),
            "fallback_by_event": dict(Counter(r["event_id"] for r in certified if r["temporal_fallback"])),
            "days_after_event": _dist([r["days_after_event"] for r in certified]),
            "days_delta_vs_post": _dist([r["days_delta_vs_post"] for r in certified
                                         if r["days_delta_vs_post"] is not None]),
            "risk_n": int(sum(1 for r in certified if r["temporal_risk"])),
        },
        "quality": {
            "scl255_tiles": int(sum(1 for r in certified if r["scl255_ratio"] > 0)),
            "unknown_px_total": int(sum(r["unknown_px"] for r in certified)),
            "valid_ratio": _dist([r["valid_ratio"] for r in certified]),
            "cloud_fraction": _dist([r["cloud_fraction"] for r in certified]),
        },
        "rules": {
            "scl_valid": list(SCL_VALID),
            "cloud_codes": list(CLOUD_CODES),
            "post_event_window_days": int(dcfg.get("post_event_window_days", 120)),
            "max_cloud": dcfg["max_cloud"],
            "min_valid_ratio": dcfg["min_valid_ratio"],
            "max_label_invalid_ratio": dcfg["max_label_invalid_ratio"],
            "min_positive_px_filter": None,
            "valid10": "SCL in {4,5,6,7} & finite & all bands>0 & label known",
            "timestep": "earliest usable in post-event window; else global min-cloud (fallback)",
            "split": {"test_event": test_ev, "val_event": val_ev, "others": "train"},
        },
    }
    rules_sha = _sha256_bytes(json.dumps(report["rules"], sort_keys=True, ensure_ascii=False).encode("utf-8"))
    report["identity"] = {
        "script": "scripts/prepare_segformer_data.py",
        "script_sha256": _sha256_bytes((ROOT / "scripts" / "prepare_segformer_data.py").read_bytes()),
        "rules_sha256": rules_sha,
        "manifest_sha256": _sha256_bytes(manifest.read_bytes()),
        "stats_sha256": _sha256_bytes(stats_path.read_bytes()),
        "quarantine_sha256": _sha256_bytes(quar_path.read_bytes()),
    }
    (manifest.parent / f"{dcfg['version']}_summary.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n[certified] {len(certified)}   [quarantine] {len(quarantined)}")
    for k, v in summary.items():
        print(f"  {k:5s} {v['n_samples']:5d} 样本 | 事件 {v['events']} | 正块 {v['n_positive_tiles']}"
              f" | 负块 {v['n_negative_tiles']} | 正像素 {v['positive_px']}"
              f" | <50px 块 {v['small_objects_px_lt_50']}")
    print(f"[check] 跨 split bbox 重叠: {len(leaks)}")
    print(f"[temporal] fallback={report['temporal']['fallback_n']} risk={report['temporal']['risk_n']}"
          f" days p50={report['temporal']['days_after_event'].get('p50')}"
          f" p90={report['temporal']['days_after_event'].get('p90')}")
    print(f"[stats] mean={np.round(stats['mean'], 1).tolist()}")
    print(f"[stats] std ={np.round(stats['std'], 1).tolist()}")
    print(f"[stats] p50 ={stats['quantiles']['p50']}")
    print(f"[saved] {manifest}\n[saved] {quar_path}\n[saved] {stats_path}")


if __name__ == "__main__":
    main()
