"""为原固定事件留出 `event_ood_v4r1` 构建地形侧表（DEM＋坡度）。

为什么用侧表而不是重建缓存：`manifests/v4r1/protocol.json` 冻结了每个成员的
``cache`` 与 ``cache_sha256``，就地重写或换路径会让基线无法复核。侧表按
``pool_100.jsonl`` 的行序存放 DEM/坡度，原 npz 与清单完全不动。

产物：
- ``data/v4r1_geo/dem.npy``   int16，米
- ``data/v4r1_geo/slope.npy`` int16，0.01°（由 DEM 按 10 m 像元求梯度）
- ``data/v4r1_geo/nodata.npy`` bool，源 DEM 缺失/非有限
- ``data/v4r1_geo/meta.json`` 行序、单位、哈希与缺失统计
- ``manifests/v4r1/stats_geoboth.json`` / ``stats_dem.json``
  十波段沿用 ``stats_core25.json`` 原值（不重拟合），地形通道只由 TRAIN 拟合
- ``manifests/v4r1/protocol_geoboth.json`` / ``protocol_dem.json``
  在原协议基础上登记新统计，版本号 ``event_ood_v4r1+geoboth``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

from landslide_seg.config import file_sha256  # noqa: E402
from landslide_seg.training.protocol import canonical_hash  # noqa: E402

POOL = ROOT / "manifests" / "v4r1" / "pool_100.jsonl"
PROTOCOL = ROOT / "manifests" / "v4r1" / "protocol.json"
CORE_STATS = ROOT / "manifests" / "v4r1" / "stats_core25.json"
OUT = ROOT / "data" / "v4r1_geo"
PIXEL_M = 10.0


def read_rows(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8-sig").splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--suffix", default="", help="输出文件后缀，用于区分修正版本")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = read_rows(POOL)
    n = len(rows)
    print(f"样本 {n}（TRAIN {sum(r['protocol_role'] == 'TRAIN' for r in rows)} / "
          f"DEV {sum(r['protocol_role'] == 'DEV' for r in rows)}）")

    dem_path, slope_path, nodata_path = OUT / "dem.npy", OUT / "slope.npy", OUT / "nodata.npy"
    if dem_path.is_file() and not args.force:
        dem = np.load(dem_path, mmap_mode="r")
        slope = np.load(slope_path, mmap_mode="r")
        nodata = np.load(nodata_path, mmap_mode="r")
        print("[reuse] 侧表已存在")
    else:
        dem = np.lib.format.open_memmap(dem_path, mode="w+", dtype=np.int16, shape=(n, 128, 128))
        slope = np.lib.format.open_memmap(slope_path, mode="w+", dtype=np.int16, shape=(n, 128, 128))
        nodata = np.lib.format.open_memmap(nodata_path, mode="w+", dtype=bool, shape=(n, 128, 128))
        missing = []
        for i, r in enumerate(rows):
            sid = r["sample_id"]
            nc = ROOT / "data" / "sen12" / f"{sid}.nc"
            if not nc.is_file():
                missing.append({"sample_id": sid, "reason": "nc missing"})
                dem[i] = 0
                slope[i] = 0
                nodata[i] = True
                continue
            with xr.open_dataset(nc, engine="h5netcdf") as ds:
                if "DEM" not in ds:
                    missing.append({"sample_id": sid, "reason": "no DEM var"})
                    dem[i] = 0
                    slope[i] = 0
                    nodata[i] = True
                    continue
                d = ds["DEM"].values
                d = d[0] if d.ndim == 3 else d
                d = d.astype(np.float32)
            bad = ~np.isfinite(d)
            if bad.any():
                d = np.where(bad, np.nanmedian(d), d)
            gy, gx = np.gradient(d, PIXEL_M, PIXEL_M)
            sl = np.degrees(np.arctan(np.hypot(gx, gy)))
            dem[i] = np.rint(d).astype(np.int16)
            slope[i] = np.rint(sl * 100.0).astype(np.int16)
            nodata[i] = bad
            if (i + 1) % 250 == 0:
                print(f"  [geo] {i + 1}/{n}", flush=True)
        dem.flush(), slope.flush(), nodata.flush()
        meta = {"n": n, "rows": [r["sample_id"] for r in rows], "shape": [n, 128, 128],
                "dem_unit": "m", "slope_unit": "0.01deg", "pixel_m": PIXEL_M,
                "nodata_policy": "源 DEM 非有限记为 nodata 并按中值填充，仅用于输入；valid/标签不变",
                "missing": missing, "pool_sha256": file_sha256(POOL),
                "source": "data/sen12/<sid>.nc 的静态 DEM（15 时相相同，取 time=0）"}
        (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[missing] {len(missing)}")

    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    pos = {sid: i for i, sid in enumerate(meta["rows"])}
    train = [r for r in rows if r["protocol_role"] == "TRAIN"]
    dem_parts, slp_parts = [], []
    for r in train:
        i = pos[r["sample_id"]]
        with np.load(r["cache"]) as z:
            v = np.asarray(z["valid"], dtype=bool) & ~np.asarray(nodata[i])
        dem_parts.append(np.asarray(dem[i])[v].astype(np.int64))
        slp_parts.append(np.asarray(slope[i])[v].astype(np.int64))
    dem_flat, slp_flat = np.concatenate(dem_parts), np.concatenate(slp_parts)

    core = json.loads(CORE_STATS.read_text(encoding="utf-8"))
    train_ids = sorted(r["sample_id"] for r in train)
    train_events = sorted({r["event_id"] for r in train})

    def make(name: str, extra: list[tuple[str, np.ndarray, float, str]]) -> Path:
        st = dict(core)
        st["mean"] = list(core["mean"]) + [float(a.mean() / s) for _, a, s, _ in extra]
        st["std"] = list(core["std"]) + [float(max(a.std() / s, 1e-6)) for _, a, s, _ in extra]
        st["channels"] = list(core.get("bands", [])) + [n_.upper() for n_, _, _, _ in extra]
        st["extra_channel"] = name
        st["extra_unit"] = ",".join(u for _, _, _, u in extra)
        st["fit_split"] = "EVENT_OOD_V4R1/TRAIN"
        st["fit_sample_ids"] = train_ids
        st["fit_events"] = train_events
        st["n_samples"] = len(train_ids)
        st["note"] = ("十波段沿用 stats_core25.json（559 块共同 TRAIN 核心，未重拟合）；"
                      f"{name} 由本 TRAIN {len(train_ids)} 块有效像素拟合"
                      f"（n={int(dem_flat.size):,} 像素）")
        p = ROOT / "manifests" / "v4r1" / f"stats_{name}.json"
        p.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
        return p

    stats_dem = make("dem" + args.suffix, [("dem", dem_flat, 1.0, "m")])
    # 坡度侧表按 0.01° 存 int16；统计必须用同一原始单位，否则归一化会放大 100 倍。
    stats_geo = make("geoboth" + args.suffix,
                     [("dem", dem_flat, 1.0, "m"), ("slope", slp_flat, 1.0, "0.01deg")])

    base = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    outs = {}
    for variant, stats_path, desc in (("dem" + args.suffix, stats_dem, "十波段 + DEM"),
                                      ("geoboth" + args.suffix, stats_geo,
                                       "十波段 + DEM + 坡度（原始 0.01° 单位）")):
        payload = {k: v for k, v in base.items() if k != "sha256"}
        payload["files"] = dict(payload["files"])
        payload["files"][str(stats_path).replace("\\", "/")] = file_sha256(stats_path)
        payload["version"] = f"event_ood_v4r1+{variant}"
        payload["design"] = dict(payload.get("design", {}), extra_channel=variant)
        payload["terrain"] = {"stats": desc,
                              "fit": f"仅本 TRAIN {len(train_ids)} 块有效像素",
                              "source": "data/sen12 静态 DEM；坡度按 10 m 像元求梯度"}
        payload["sha256"] = canonical_hash(payload)
        p = ROOT / "manifests" / "v4r1" / f"protocol_{variant}.json"
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        outs[variant] = p

    report = {
        "n_rows": n, "n_train": len(train), "n_dev": n - len(train),
        "n_pixels": int(dem_flat.size),
        "dem": {"mean": float(dem_flat.mean()), "std": float(dem_flat.std()),
                "min": int(dem_flat.min()), "max": int(dem_flat.max())},
        "slope_deg": {"mean": float(slp_flat.mean() / 100), "std": float(slp_flat.std() / 100),
                      "p99": float(np.percentile(slp_flat, 99) / 100)},
        "missing": meta.get("missing", []),
        "stats": {v: file_sha256(p) for v, p in (("dem", stats_dem), ("geoboth", stats_geo))},
    }
    (OUT / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("n_train", "n_pixels", "dem", "slope_deg")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
