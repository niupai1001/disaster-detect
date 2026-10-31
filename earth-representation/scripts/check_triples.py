"""Sanity check on the triples: are the paired patches really the same ground?

Compares NDVI maps of the same sample across sensors against random pairs. If
same-location correlation is not clearly higher, the pairing (not the encoder)
is the problem and no cross-sensor conclusion can be drawn.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"N:\Users\yzh\Disater\earth-representation")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data import triples  # noqa: E402


def ndvi(img, names):
    """NIR is S2/HLS B08 and Landsat SR_B5 in this band layout."""
    idx = {}
    for i, n in enumerate(names):
        idx[n] = i
    red = idx.get("S2_B04") or idx.get("HLS_B04") or idx.get("LANDSAT_SR_B4")
    nir = idx.get("S2_B08") or idx.get("HLS_B08") or idx.get("LANDSAT_SR_B5")
    r, n = img[red], img[nir]
    return (n - r) / (n + r + 1e-6)


def corr(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    d = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / d) if d > 0 else 0.0


items = triples.load_triples(limit=60)
print("triples", len(items))
same_nd, rand_nd = [], []
same_raw, rand_raw = [], []
rng = np.random.default_rng(0)
for i, it in enumerate(items):
    s2 = ndvi(it["s2"]["image"], it["s2"]["bands"])
    ls = ndvi(it["landsat"]["image"], it["landsat"]["bands"])
    hl = ndvi(it["hls"]["image"], it["hls"]["bands"])
    same_nd.append((corr(s2, ls), corr(s2, hl)))
    j = int(rng.integers(0, len(items)))
    if j != i:
        ls_j = ndvi(items[j]["landsat"]["image"], items[j]["landsat"]["bands"])
        rand_nd.append(corr(s2, ls_j))
    same_raw.append(corr(it["s2"]["image"][2], it["landsat"]["image"][2]))  # red band

print("NDVI corr s2 vs landsat (same place): mean %.3f" % np.mean([a for a, _ in same_nd]))
print("NDVI corr s2 vs hls     (same place): mean %.3f" % np.mean([b for _, b in same_nd]))
print("NDVI corr s2 vs landsat (random)    : mean %.3f" % np.mean(rand_nd))
print("red-band corr s2 vs landsat (same)  : mean %.3f" % np.mean(same_raw))
