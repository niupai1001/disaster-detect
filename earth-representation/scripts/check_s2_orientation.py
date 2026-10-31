"""Is the S2 patch stored transposed relative to a north-up raster read?

Compares the patch's NDVI with the same-date STAC window under identity,
transpose and flips. A clear winner identifies the storage convention.
"""

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds

ROOT = Path(r"N:\Users\yzh\Disater\earth-representation")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data import triples  # noqa: E402
from src.data.adapters import _base, sentinel2  # noqa: E402
from download_companion import load_samples  # noqa: E402

S2_DIR = Path(r"N:\Users\yzh\Disater\landslide-seg\data\sen12")
samples = load_samples()


def corr(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


def nd(red, nir):
    return (nir - red) / (nir + red + 1e-6)


tests = {
    "identity": lambda x: x,
    "transpose": lambda x: x.T,
    "flip_ud": np.flipud,
    "flip_lr": np.fliplr,
    "rot180": lambda x: np.rot90(x, 2),
    "transpose_flip": lambda x: np.flipud(x.T),
}

scores = {k: [] for k in tests}
for it in triples.load_triples(limit=12):
    sid = it["sample_id"]
    meta = samples.get(sid)
    if not meta:
        continue
    d0 = meta["date"]
    bbox_w = transform_bounds(meta["crs"], "EPSG:4326", *meta["bbox"], densify_pts=21)
    scenes = sentinel2.scenes(bbox_w, d0, d0, limit=5)
    if not scenes:
        continue
    scene = scenes[0]
    grid = from_bounds(*meta["bbox"], 128, 128)
    crs = CRS.from_epsg(int(meta["crs"].split(":")[1]))
    red = _base.read_grid(scene.assets["B04"], grid, crs, (128, 128), header=scene.header)
    nir = _base.read_grid(scene.assets["B08"], grid, crs, (128, 128), header=scene.header)
    stac_nd = nd(red.astype(np.float32) / 1e4, nir.astype(np.float32) / 1e4)
    patch_nd = nd(it["s2"]["image"][2], it["s2"]["image"][6])
    for name, fn in tests.items():
        t = fn(patch_nd)
        if t.shape == stac_nd.shape:
            scores[name].append(corr(stac_nd, t))

print("samples:", len(scores["identity"]))
for name, vals in scores.items():
    if vals:
        print(f"{name:16s} mean corr {np.mean(vals):+.3f}")
