"""Is the stored companion patch the same as a direct read at that bbox?

Compares the NetCDF written by download_companion with a fresh window read of
the same scene, bbox and grid. A mismatch points at cut_patch/写盘, not at S2.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from scipy.io import netcdf_file

ROOT = Path(r"N:\Users\yzh\Disater\earth-representation")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data import triples  # noqa: E402
from src.data.adapters import _base, landsat_c2  # noqa: E402
from download_companion import load_samples  # noqa: E402

TIGHT = ROOT / "data" / "tight"
samples = load_samples()
sid = "chimanimani_s2_1001"
meta = samples[sid]

with netcdf_file(str(TIGHT / "sen12_landsat" / f"{sid}.nc"), "r", mmap=False) as nc:
    stored = np.array(nc.variables["SR_B4"][:], dtype=np.float32)
    stored_nir = np.array(nc.variables["SR_B5"][:], dtype=np.float32)

# find the same scene the downloader used (nearest to the patch date)
import datetime as dt
scenes = landsat_c2.scenes(
    __import__("rasterio").warp.transform_bounds(meta["crs"], "EPSG:4326",
                                                 *meta["bbox"], densify_pts=21),
    (dt.date.fromisoformat(meta["date"]) - dt.timedelta(days=30)).isoformat(),
    (dt.date.fromisoformat(meta["date"]) + dt.timedelta(days=30)).isoformat(), limit=10)
print("scenes near date:", len(scenes))
scene = scenes[0]
grid = from_bounds(*meta["bbox"], 128, 128)
crs = CRS.from_epsg(int(meta["crs"].split(":")[1]))
fresh = _base.read_grid(scene.assets["red"], grid, crs, (128, 128)).astype(np.float32)
fresh_nir = _base.read_grid(scene.assets["nir08"], grid, crs, (128, 128)).astype(np.float32)
print("stored", stored.shape, stored.dtype, "fresh", fresh.shape, fresh.dtype)
print("stored mean", float(stored.mean()), "fresh mean", float(fresh.mean()))
print("identical fraction:", float(np.mean(stored == fresh)))
print("corr:", float(np.corrcoef(stored.ravel(), fresh.ravel())[0, 1]))
