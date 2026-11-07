"""Inventory of every local dataset that could feed Encoder V3.

For each directory: file count, size, one sample's structure. V3 needs real
reflectance + real SRF + native grid; HLS is explicitly not an independent
sensor in that spec, so it is listed separately.
"""

import glob
import json
import os
from pathlib import Path

ROOT = Path(r"N:\Users\yzh\Disater\earth-representation")
S2 = Path(r"N:\Users\yzh\Disater\landslide-seg\data\sen12")


def size_gb(files):
    return sum(os.path.getsize(f) for f in files) / 1e9


def nc_vars(path, limit=14):
    from scipy.io import netcdf_file
    try:
        with netcdf_file(str(path), "r", mmap=False) as nc:
            return list(nc.variables)[:limit], len(nc.variables)
    except Exception:
        return None, None


def s2_probe(path):
    import rasterio
    with rasterio.open(str(path)) as ds:
        subs = {s.rsplit(":", 1)[-1]: s for s in ds.subdatasets}
        with rasterio.open(subs["B02"]) as d:
            return sorted(subs), (d.width, d.height), d.count, str(d.crs)


def row(name, files, note=""):
    if not files:
        print(f"{name:26s} {'0':>7s} files")
        return
    print(f"{name:26s} {len(files):7d} files {size_gb(files):7.2f} GB  {note}")


print("== 原始 S2 patch（15 时相时间栈）==")
s2_files = sorted(S2.glob("*.nc"))
row("sen12 (S2)", s2_files)
if s2_files:
    bands, shape, t, crs = s2_probe(s2_files[0])
    print(f"   sample: bands={len(bands)} {bands[:6]}... size={shape} times={t} crs={crs}")

print("\n== 我下载的三源配对 ==")
for name in ("sen12_landsat", "sen12_hls"):
    files = sorted((ROOT / "data" / name).glob("*.nc"))
    row(name, files)
    if files:
        v, n = nc_vars(files[0])
        print(f"   sample vars ({n}): {v}")

print("\n== 试点 / 其他会话产出 ==")
for name in ("tight/sen12_landsat", "tight/sen12_hls",
             "companions_v2_selected_time/sen12_landsat",
             "companions_v2_selected_time/sen12_hls",
             "companions_v2", "companions_v2_evaluation"):
    files = sorted((ROOT / "data" / name).rglob("*")) if (ROOT / "data" / name).exists() else []
    files = [f for f in files if f.is_file()]
    row(name, files)

print("\n== EnMAP ==")
enmap = sorted((ROOT / "data" / "enmap").glob("*.TIF"))
enmap_pairs = sorted((ROOT / "data" / "enmap_pairs").glob("*.nc"))
row("enmap (整景)", enmap)
row("enmap_pairs (patch)", enmap_pairs)
if enmap:
    import rasterio
    with rasterio.open(enmap[0]) as ds:
        print(f"   sample: {ds.count} bands, {ds.width}x{ds.height}, {ds.crs}, {ds.dtypes[0]}")

print("\n== COG 管线产物 ==")
npz = sorted((ROOT / "data" / "interim" / "patches").glob("*.npz"))
row("interim/patches", npz)
print("\n== SRF ==")
for f in sorted((ROOT / "data" / "physics").rglob("*")):
    if f.is_file():
        print(f"  {f.name:44s} {f.stat().st_size/1e6:7.2f} MB")
print("\n== observation_model 自带数据 ==")
om = sorted((ROOT / "observation_model").glob("data*"))
for p in om:
    files = [f for f in p.rglob("*") if f.is_file()]
    row(p.name, files)
