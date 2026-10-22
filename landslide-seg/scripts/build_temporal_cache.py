"""为 U-TAE 时序配对构建 15 时相缓存（memmap，非每 epoch 重开 nc）。

按执行计划 §3 C 与 §4：
- 输入 ``[T=15, C=10, H, W]`` + 静态 DEM + 单一目标 mask + 单一 valid；
- 两组（单期 / 15 期）共用同一批 4,715 块、同一标签、**同一套 valid 评价范围**；
- valid 直接沿用单期缓存（不按 15 期重算），保证评价范围一致；
- 用 memmap 分片，避免把浮点时序常驻内存。

产物 ``data/s2_temporal_v5/``：
- ``X15.npy``：``(N,15,10,128,128)`` int16，memmap
- ``dem.npy``：``(N,128,128)`` int16
- ``index.json``：每行的 sample_id、行号、time_index、真实日期列表
- ``meta.json``：形状、dtype、样本数、来源、校验
"""

from __future__ import annotations

import argparse
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

from landslide_seg.config import resolve_path  # noqa: E402
from landslide_seg.data.segformer_dataset import S2_10_BANDS  # noqa: E402

OUT = ROOT / "data" / "s2_temporal_v5"
SRC = ("manifests/official_single_geo/trainval.jsonl", "manifests/official_single_geo/test.jsonl")
T = 15


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for rel in SRC:
        rows += [json.loads(l) for l in resolve_path(rel).read_text(encoding="utf-8").splitlines() if l.strip()]
    n = len(rows)
    print(f"样本 {n}，目标形状 ({n},{T},{len(S2_10_BANDS)},128,128) int16 ≈ {n*T*10*128*128*2/2**30:.1f} GiB")

    x_path, dem_path = OUT / "X15.npy", OUT / "dem.npy"
    X = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.int16, shape=(n, T, len(S2_10_BANDS), 128, 128))
    DEM = np.lib.format.open_memmap(dem_path, mode="w+", dtype=np.int16, shape=(n, 128, 128))

    index, bad = [], []
    for i, r in enumerate(rows, 1):
        sid = r["sample_id"]
        nc = ROOT / "data" / "sen12" / f"{sid}.nc"
        try:
            with xr.open_dataset(nc, engine="h5netcdf") as ds:
                ds = ds.sortby("time")
                stack = np.stack([ds[b].values for b in S2_10_BANDS], axis=1)  # (T,C,H,W)
                dem = ds["DEM"].values
                dem = dem[0] if dem.ndim == 3 else dem
                times = ds["time"].values.astype("datetime64[D]").astype(str).tolist()
            if stack.shape != (T, len(S2_10_BANDS), 128, 128):
                raise ValueError(f"形状 {stack.shape} 不符")
            X[i - 1] = stack.astype(np.int16)
            DEM[i - 1] = np.rint(dem).astype(np.int16)
            index.append({"row": i - 1, "sample_id": sid, "split": r["split"],
                          "time_index_single": int(r["time_index"]), "dates": times,
                          "cache_single": r["cache"], "cache_sha256": r["cache_sha256"]})
        except Exception as exc:  # noqa: BLE001
            bad.append({"sample_id": sid, "error": f"{type(exc).__name__}: {exc}"[:160]})
        if i % 400 == 0:
            print(f"  {i}/{n}", flush=True)

    X.flush(); DEM.flush()
    (OUT / "index.json").write_text(json.dumps({"n": n, "rows": index, "bad": bad}, ensure_ascii=False), encoding="utf-8")
    meta = {
        "shape_X": [n, T, len(S2_10_BANDS), 128, 128], "dtype": "int16",
        "shape_dem": [n, 128, 128], "bands": list(S2_10_BANDS), "T": T,
        "source": [str(s) for s in SRC],
        "valid_policy": "沿用单期缓存的 valid（两组共用同一评价范围）",
        "label_policy": "沿用单期缓存 mask（MASK 在 15 期上恒定，已审计）",
        "n_ok": len(index), "n_bad": len(bad),
        "caveat": "受官方任务筛选影响的 4,715 块可用子集",
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[done] ok={len(index)} bad={len(bad)}")
    if bad:
        print("  失败样例:", bad[:3])


if __name__ == "__main__":
    main()
