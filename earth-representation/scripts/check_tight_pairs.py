"""NDVI correlation of the tight (same-week) pairs vs random pairs."""

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


def ndvi(img, names):
    idx = {n: i for i, n in enumerate(names)}
    red = idx.get("S2_B04") or idx.get("LANDSAT_SR_B4")
    nir = idx.get("S2_B08") or idx.get("LANDSAT_SR_B5")
    r, n = img[red], img[nir]
    return (n - r) / (n + r + 1e-6)


def corr(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


items = triples.load_triples(limit=40)
same, rand = [], []
rng = np.random.default_rng(0)
valid = []
for it in items:
    p = TIGHT / "sen12_landsat" / f"{it['sample_id']}.nc"
    if not p.exists():
        continue
    ls = read_nc(p, triples.LANDSAT_VARS) / 10000.0
    valid.append((it, ls))
print("usable tight pairs:", len(valid))
for i, (it, ls) in enumerate(valid):
    s2_nd = ndvi(it["s2"]["image"], it["s2"]["bands"])
    ls_nd = ndvi(ls, triples.LANDSAT_NAMES)
    same.append(corr(s2_nd, ls_nd))
    j = int(rng.integers(0, len(valid)))
    if j != i:
        other = ndvi(valid[j][1], triples.LANDSAT_NAMES)
        rand.append(corr(s2_nd, other))
print(f"NDVI corr same place (<=30d): mean {np.mean(same):.3f} median {np.median(same):.3f}")
print(f"NDVI corr random pair       : mean {np.mean(rand):.3f} median {np.median(rand):.3f}")
