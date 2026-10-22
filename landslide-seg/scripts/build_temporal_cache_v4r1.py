"""为本项目的数据划分（九事件 2223 训练 + 669 验证）构建 15 时相缓存。

``data/s2_temporal_v5`` 是按官方可用子集建的，缺少本项目划分中的 502 块。
本脚本新建 ``data/s2_temporal_v4r1``：已有的 2390 块直接从 v5 缓存复制，
缺的 502 块从源 NetCDF 读取；标签与 valid 仍由单期 npz 提供（U-TAE 数据集
只从这个缓存取影像与日期）。

与单期基线保持一致：不做 SCL/nodata 侧表处理，评价范围沿用单期 valid。
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

from landslide_seg.config import file_sha256  # noqa: E402
from landslide_seg.data.segformer_dataset import S2_10_BANDS  # noqa: E402

OUT = ROOT / "data" / "s2_temporal_v4r1"
SRC = ROOT / "data" / "s2_temporal_v5"
POOL = ROOT / "manifests" / "v4r1" / "pool_100.jsonl"
T = 15


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "index.json").is_file() and not args.force:
        print("[reuse] index.json 已存在")
        return

    rows = [json.loads(l) for l in POOL.read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    n = len(rows)
    print(f"样本 {n}，目标形状 ({n},{T},10,128,128) int16 ≈ {n * T * 10 * 128 * 128 * 2 / 2**30:.1f} GiB")

    idx5 = json.loads((SRC / "index.json").read_text(encoding="utf-8"))
    pos5 = {r["sample_id"]: r["row"] for r in idx5["rows"]}
    dates5 = {r["sample_id"]: r["dates"] for r in idx5["rows"]}
    X5 = np.load(SRC / "X15.npy", mmap_mode="r")
    D5 = np.load(SRC / "dem.npy", mmap_mode="r")

    X = np.lib.format.open_memmap(OUT / "X15.npy", mode="w+", dtype=np.int16,
                                  shape=(n, T, len(S2_10_BANDS), 128, 128))
    DEM = np.lib.format.open_memmap(OUT / "dem.npy", mode="w+", dtype=np.int16, shape=(n, 128, 128))

    index, bad, copied, built = [], [], 0, 0
    for i, r in enumerate(rows):
        sid = r["sample_id"]
        try:
            if sid in pos5:
                j = pos5[sid]
                X[i] = X5[j]
                DEM[i] = D5[j]
                dates = dates5[sid]
                copied += 1
            else:
                nc = ROOT / "data" / "sen12" / f"{sid}.nc"
                with xr.open_dataset(nc, engine="h5netcdf") as ds:
                    ds = ds.sortby("time")
                    stack = np.stack([ds[b].values for b in S2_10_BANDS], axis=1)
                    dem = ds["DEM"].values
                    dem = dem[0] if dem.ndim == 3 else dem
                    dates = ds["time"].values.astype("datetime64[D]").astype(str).tolist()
                if stack.shape != (T, len(S2_10_BANDS), 128, 128):
                    raise ValueError(f"形状 {stack.shape} 不符")
                X[i] = stack.astype(np.int16)
                DEM[i] = np.rint(dem).astype(np.int16)
                built += 1
            index.append({"row": i, "sample_id": sid, "split": r["split"],
                          "time_index_single": int(r["time_index"]), "dates": dates,
                          "cache_single": r["cache"], "cache_sha256": r["cache_sha256"]})
        except Exception as exc:  # noqa: BLE001
            bad.append({"sample_id": sid, "error": f"{type(exc).__name__}: {exc}"[:160]})
        if (i + 1) % 400 == 0:
            print(f"  {i + 1}/{n} (copied {copied} / built {built})", flush=True)

    X.flush(); DEM.flush()
    (OUT / "index.json").write_text(
        json.dumps({"n": n, "rows": index, "bad": bad}, ensure_ascii=False), encoding="utf-8")
    meta = {
        "shape_X": [n, T, len(S2_10_BANDS), 128, 128], "dtype": "int16",
        "shape_dem": [n, 128, 128], "bands": list(S2_10_BANDS), "T": T,
        "source": ["manifests/v4r1/pool_100.jsonl"],
        "copied_from_v5": copied, "built_from_nc": built, "n_ok": len(index), "n_bad": len(bad),
        "index_sha256": file_sha256(OUT / "index.json"),
        "valid_policy": "沿用单期缓存 valid；与单期基线一致，不做额外 nodata 侧表",
        "label_policy": "沿用单期缓存 mask",
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[done] ok={len(index)} bad={len(bad)} copied={copied} built={built}")


if __name__ == "__main__":
    main()
