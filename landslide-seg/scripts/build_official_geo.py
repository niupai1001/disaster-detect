"""为官方单期协议构建带地形通道的新缓存（DEM 与坡度），不改动原缓存。

为什么另建：原 `data/s2_official_single/` 的 npz 哈希登记在基线的 protocol 里，
就地改写会让基线无法复核。这里生成独立目录与独立 manifest/protocol，
标签与 valid 直接沿用原缓存，保证与基线逐块可比。

产物：
- ``data/s2_official_single_geo/<sid>.npz``：x10 / mask / valid / **dem** / **slope**
- ``manifests/official_single_geo/{trainval,test}.jsonl``（与基线同一批样本）
- ``manifests/official_single_geo/stats_dem.json``、``stats_slope.json``（各自 11 通道）
- ``manifests/official_single_geo/protocol_{dem,slope}.json``
- ``manifests/official_single_geo/build_report.json``

通道定义：
- ``dem``：int16，整数米，来自源 nc 的静态 DEM（已验证 15 时相完全相同，取 time=0）
- ``slope``：int16，0.01° 为单位，由 DEM 用 10 m 像元间距求梯度后取 arctan
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

from landslide_seg.config import file_sha256, resolve_path  # noqa: E402
from landslide_seg.data.segformer_dataset import S2_10_BANDS  # noqa: E402
from landslide_seg.training.protocol import canonical_hash  # noqa: E402

SRC_MANIFEST = "manifests/official_single/trainval.jsonl"
SRC_TEST = "manifests/official_single/test.jsonl"
SRC_STATS = "manifests/official_single/stats_train.json"
OUT = ROOT / "manifests" / "official_single_geo"
CACHE = ROOT / "data" / "s2_official_single_geo"
PIXEL_M = 10.0


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    train_val = read_jsonl(resolve_path(SRC_MANIFEST))
    test = read_jsonl(resolve_path(SRC_TEST))
    records = train_val + test
    print(f"样本 {len(records)}（train+val {len(train_val)} / test {len(test)}）")

    missing_dem = []
    new_recs = []
    for i, rec in enumerate(records, 1):
        sid = rec["sample_id"]
        src_npz = resolve_path(rec["cache"])
        dst_npz = CACHE / f"{sid}.npz"
        nc = ROOT / "data" / "sen12" / f"{sid}.nc"
        if not nc.is_file():
            missing_dem.append({"sample_id": sid, "reason": "nc missing"})
            continue
        with xr.open_dataset(nc, engine="h5netcdf") as ds:
            if "DEM" not in ds:
                missing_dem.append({"sample_id": sid, "reason": "no DEM var"})
                continue
            dem = ds["DEM"].values
            dem = dem[0] if dem.ndim == 3 else dem
            dem = dem.astype(np.float32)
        gy, gx = np.gradient(dem, PIXEL_M, PIXEL_M)
        slope = np.degrees(np.arctan(np.hypot(gx, gy)))

        d = np.load(src_npz)
        payload = {
            "x10": np.asarray(d["x10"], dtype=np.int16),
            "mask": np.asarray(d["mask"], dtype=np.uint8),
            "valid": np.asarray(d["valid"], dtype=bool),
            "dem": np.rint(dem).astype(np.int16),
            "slope": np.rint(slope * 100.0).astype(np.int16),
        }
        buf = CACHE / f"{sid}.tmp.npz"
        np.savez_compressed(buf, **payload)
        sha = sha_bytes(buf.read_bytes())
        buf.replace(dst_npz)
        new_recs.append(dict(rec, cache=str(dst_npz).replace("\\", "/"), cache_sha256=sha))
        if i % 500 == 0:
            print(f"  {i}/{len(records)}", flush=True)

    def write_jsonl(rows, path: Path):
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    n_tv = len(train_val)
    write_jsonl(new_recs[:n_tv], OUT / "trainval.jsonl")
    write_jsonl(new_recs[n_tv:], OUT / "test.jsonl")

    # ---- 统计：十波段复用原 stats，附加通道按 TRAIN 有效像素拟合 ----
    base = json.loads(resolve_path(SRC_STATS).read_text(encoding="utf-8"))
    train_rows = [r for r in new_recs[:n_tv] if r["split"] == "train"]
    dem_parts, slp_parts = [], []
    for r in train_rows:
        d = np.load(r["cache"])
        v = np.asarray(d["valid"], dtype=bool)
        dem_parts.append(np.asarray(d["dem"])[v])
        slp_parts.append(np.asarray(d["slope"])[v])
    dem_flat = np.concatenate(dem_parts)
    slp_flat = np.concatenate(slp_parts)

    stats_paths = {}
    # 坡度缓存按 0.01° 存 int16，统计必须使用同一原始单位；除以 100 会造成 100 倍尺度错配。
    for name, arr, scale in (("dem", dem_flat, 1.0), ("slope", slp_flat, 1.0)):
        st = dict(base)
        st["mean"] = list(base["mean"]) + [float(arr.mean() / scale)]
        st["std"] = list(base["std"]) + [float(max(arr.std() / scale, 1e-6))]
        st["channels"] = list(S2_10_BANDS) + [name.upper()]
        st["extra_channel"] = name
        st["extra_unit"] = "m" if name == "dem" else "0.01 deg (centi-degree, same as cached array)"
        st["note"] = (f"十波段沿用官方 TRAIN 统计；{name} 由 TRAIN 有效像素拟合"
                      f"（n={arr.size:,} 像素），单位与缓存数组一致")
        p = OUT / f"stats_{name}.json"
        p.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
        stats_paths[name] = p

    # ---- protocol（两种变体各一份，只差 stats）----
    ident = {r["sample_id"]: {k: r[k] for k in ("event_id", "cache", "cache_sha256", "date", "time_index")}
             for r in new_recs}
    members = {sid: dict(v, role=("test" if any(t["sample_id"] == sid for t in new_recs[n_tv:]) else
                                  ("train" if any(t["sample_id"] == sid and t["split"] == "train"
                                                  for t in new_recs[:n_tv]) else "val")))
               for sid, v in ident.items()}
    train_ids = sorted(r["sample_id"] for r in new_recs[:n_tv] if r["split"] == "train")
    val_ids = sorted(r["sample_id"] for r in new_recs[:n_tv] if r["split"] == "val")
    test_ids = sorted(r["sample_id"] for r in new_recs[n_tv:])

    proto_paths = {}
    for name, sp in stats_paths.items():
        payload = {
            "version": f"BENCH_OFFICIAL_S12LS-LD_s2_single_date+{name}",
            "frozen": True,
            "design": {"extra_channel": name,
                       "train": sorted({r["event_id"] for r in new_recs[:n_tv] if r["split"] == "train"})},
            "members": members, "dev_ids": val_ids, "test_ids": test_ids,
            "pools": {"100": train_ids},
            "files": {
                str(OUT / "trainval.jsonl").replace("\\", "/"): file_sha256(OUT / "trainval.jsonl"),
                str(OUT / "test.jsonl").replace("\\", "/"): file_sha256(OUT / "test.jsonl"),
                str(sp).replace("\\", "/"): file_sha256(sp),
            },
            "notes": {
                "base_protocol": "manifests/official_single/protocol.json（同一批样本、同一标签与 valid）",
                "label_and_valid": "直接沿用基线缓存，未做任何改动",
                "extra_channel": {"dem": "静态高程，整数米", "slope": "由 DEM 求梯度，0.01°"}[name],
                "leak_note": "坡度由 DEM 推出，DEM 为静态底图，不含事件信息",
            },
        }
        payload["sha256"] = canonical_hash(payload)
        p = OUT / f"protocol_{name}.json"
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        proto_paths[name] = p

    rep = {
        "n_records": len(new_recs), "n_train": len(train_ids), "n_val": len(val_ids), "n_test": len(test_ids),
        "missing": missing_dem,
        "dem_stats": {"mean": float(dem_flat.mean()), "std": float(dem_flat.std()),
                      "min": int(dem_flat.min()), "max": int(dem_flat.max())},
        "slope_stats_deg": {"mean": float(slp_flat.mean() / 100), "std": float(slp_flat.std() / 100),
                            "p99": float(np.percentile(slp_flat, 99) / 100)},
        "files": {p.name: {"sha256": file_sha256(p)} for p in
                  [OUT / "trainval.jsonl", OUT / "test.jsonl", *stats_paths.values(), *proto_paths.values()]},
        "protocol_sha256": {k: json.loads(v.read_text(encoding="utf-8"))["sha256"] for k, v in proto_paths.items()},
    }
    (OUT / "build_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("n_records", "n_train", "n_val", "n_test", "dem_stats", "slope_stats_deg")},
                     indent=1, ensure_ascii=False))
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
