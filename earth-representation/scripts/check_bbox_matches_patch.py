"""Does the S2 patch actually sit at its manifest bbox?

Reads the same bbox from the original S2 scene over STAC and correlates its NDVI
with the local S2 patch. If this is low too, the bbox in the manifest does not
describe the patch footprint and no companion download can be aligned with it.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.io import netcdf_file

ROOT = Path(r"N:\Users\yzh\Disater\earth-representation")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import _common as C  # noqa: E402
from src.data import triples  # noqa: E402
from src.data.adapters import sentinel2  # noqa: E402
from src.data.adapters import _base  # noqa: E402
from rasterio.crs import CRS  # noqa: E402
from rasterio.transform import from_bounds  # noqa: E402

S2_DIR = Path(r"N:\Users\yzh\Disater\landslide-seg\data\sen12")


def ndvi_s2(img):
    r, n = img[2], img[6]
    return (n - r) / (n + r + 1e-6)


def corr(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


from download_companion import load_samples  # noqa: E402

samples = load_samples()
items = triples.load_triples(limit=6)
checked = 0
for it in items:
    sid = it["sample_id"]
    meta = samples.get(sid)
    if not meta:
        continue
    from rasterio.warp import transform_bounds
    d0 = meta["date"]
    bbox_w = transform_bounds(meta["crs"], "EPSG:4326", *meta["bbox"], densify_pts=21)
    scenes = sentinel2.scenes(bbox_w, d0, d0, limit=5)
    if not scenes:
        print(sid, "no scene on", d0)
        continue
    scene = scenes[0]
    grid = from_bounds(*meta["bbox"], 128, 128)
    crs = CRS.from_epsg(int(meta["crs"].split(":")[1]))
    arr = []
    for key in ("B04", "B08"):
        a = _base.read_grid(scene.assets[key], grid, crs, (128, 128),
                            header=scene.header)
        arr.append(a.astype(np.float32) / 10000.0)
    stac_nd = (arr[1] - arr[0]) / (arr[1] + arr[0] + 1e-6)
    local_nd = ndvi_s2(it["s2"]["image"])   # already the post-event slice
    print(f"{sid}: STAC-vs-patch NDVI corr {corr(stac_nd, local_nd):+.3f}")
    checked += 1
    if checked >= 4:
        break
