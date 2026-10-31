"""Is the companion patch a shifted/flipped version of the S2 patch?

Tries the obvious geometric transforms (transpose, flips, 180 rotation) and
reports which one, if any, correlates. A single transform winning clearly means
an orientation bug rather than a content mismatch.
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


def read_nc(path, variables):
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


transforms = {
    "identity": lambda x: x,
    "flip_ud": np.flipud,
    "flip_lr": np.fliplr,
    "transpose": lambda x: x.T,
    "rot180": lambda x: np.rot90(x, 2),
}

items = triples.load_triples(limit=20)
rows = {k: [] for k in transforms}
n = 0
for it in items:
    p = TIGHT / "sen12_landsat" / f"{it['sample_id']}.nc"
    if not p.exists():
        continue
    ls = read_nc(p, triples.LANDSAT_VARS) / 10000.0
    s2_nd = ndvi(it["s2"]["image"], 2, 6)          # S2 B04, B08
    ls_nd = ndvi(ls, 3, 4)                        # Landsat SR_B4, SR_B5
    for name, fn in transforms.items():
        t = fn(ls_nd)
        if t.shape == s2_nd.shape:
            rows[name].append(corr(s2_nd, t))
    n += 1

print("samples:", n)
for name, vals in rows.items():
    if vals:
        print(f"{name:10s} mean corr {np.mean(vals):+.3f}")
