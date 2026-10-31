"""Correlate the companion Landsat NDVI against every S2 time slice.

If one slice matches clearly, the geography is right and only the time index is
wrong. If none does, the stored companion array itself is misaligned.
"""

import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy.io import netcdf_file

ROOT = Path(r"N:\Users\yzh\Disater\earth-representation")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data import triples  # noqa: E402

S2_DIR = Path(r"N:\Users\yzh\Disater\landslide-seg\data\sen12")
TIGHT = ROOT / "data" / "tight"


def corr(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


def s2_slice(path, t):
    with rasterio.open(str(path)) as ds:
        subs = {s.rsplit(":", 1)[-1]: s for s in ds.subdatasets}
        red = rasterio.open(subs["B04"]).read(t) / 10000.0
        nir = rasterio.open(subs["B08"]).read(t) / 10000.0
    return (nir - red) / (nir + red + 1e-6)


for sid in ("chimanimani_s2_1001", "chimanimani_s2_1029"):
    p = TIGHT / "sen12_landsat" / f"{sid}.nc"
    if not p.exists():
        continue
    with netcdf_file(str(p), "r", mmap=False) as nc:
        ls = np.stack([np.array(nc.variables[v][:], dtype=np.float32)
                       for v in triples.LANDSAT_VARS]) / 10000.0
    ls_nd = (ls[4] - ls[3]) / (ls[4] + ls[3] + 1e-6)
    scores = []
    for t in range(1, 16):
        try:
            nd = s2_slice(S2_DIR / f"{sid}.nc", t)
        except Exception:
            break
        scores.append((t, corr(nd, ls_nd)))
    best = max(scores, key=lambda kv: kv[1])
    print(sid, "landsat NDVI mean %.2f std %.2f" % (ls_nd.mean(), ls_nd.std()))
    print("  per-slice corr:", [(t, round(c, 2)) for t, c in scores])
    print(f"  best slice t={best[0]} corr={best[1]:+.3f}")
