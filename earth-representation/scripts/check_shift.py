"""Search for a spatial shift between the S2 patch and its companion patch.

A clear peak at a non-zero shift means the two windows are offset; no peak at
all means they cover different ground.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.io import netcdf_file

ROOT = Path(r"N:\Users\yzh\Disater\earth-representation")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from src.data import triples  # noqa: E402

TIGHT = ROOT / "data" / "tight"


def scipy_read(path, variables):
    with netcdf_file(str(path), "r", mmap=False) as nc:
        return np.stack([np.array(nc.variables[v][:], dtype=np.float32) for v in variables])


def ndvi(img, red, nir):
    r, n = img[red], img[nir]
    return (n - r) / (n + r + 1e-6)


def corr(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


items = triples.load_triples(limit=8)
best = []
for it in items:
    p = TIGHT / "sen12_landsat" / f"{it['sample_id']}.nc"
    if not p.exists():
        continue
    ls = scipy_read(p, triples.LANDSAT_VARS) / 10000.0
    s2_nd = ndvi(it["s2"]["image"], 2, 6)
    ls_nd = ndvi(ls, 3, 4)
    scores = {}
    for dy in range(-24, 25, 4):
        for dx in range(-24, 25, 4):
            a = s2_nd[max(0, dy):128 + min(0, dy), max(0, dx):128 + min(0, dx)]
            b = ls_nd[max(0, -dy):128 + min(0, -dy), max(0, -dx):128 + min(0, -dx)]
            if a.shape == b.shape and a.size > 0:
                scores[(dy, dx)] = corr(a, b)
    top = sorted(scores.items(), key=lambda kv: -kv[1])[:2]
    best.append((it["sample_id"], top))
    print(it["sample_id"], "best shifts:", [(k, round(v, 3)) for k, v in top])

if best:
    peaks = [abs(t[0][0]) + abs(t[0][1]) for _, t in best if t]
    print("mean |shift| of the best match:", round(float(np.mean(peaks)), 1), "px")
