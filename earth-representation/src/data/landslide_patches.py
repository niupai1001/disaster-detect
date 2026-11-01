"""Load the existing landslide-seg patches as SSL samples.

Those patches are 128x128 netCDF stacks (10 Sentinel-2 bands + SCL/MASK/DEM)
with no georeferencing. DINO pretraining does not need coordinates — it needs
many varied images — so this is the main training pool; the COG pipeline is only
needed when a patch must be tied to a place.
"""

from pathlib import Path

import numpy as np
import rasterio

import os
DEFAULT_ROOT = Path(os.environ.get("EARTH_S2_ROOT", Path(__file__).resolve().parents[3] / "landslide-seg" / "data" / "sen12"))
REFLECTANCE_BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A",
                     "B11", "B12")
CENTRE_NM = {"B02": 492.4, "B03": 559.8, "B04": 664.6, "B05": 704.1,
             "B06": 740.5, "B07": 782.8, "B08": 832.8, "B8A": 864.7,
             "B11": 1613.7, "B12": 2202.4}
GSD = {"B02": 10, "B03": 10, "B04": 10, "B08": 10, "B05": 20, "B06": 20,
       "B07": 20, "B8A": 20, "B11": 20, "B12": 20}
BAD_SCL = (0, 1, 3, 8, 9, 10, 11, 255)


def _read_patch(path, time_index=None):
    from .manifest_index import post_index
    index = (post_index(path.stem) if time_index is None else int(time_index)) + 1
    with rasterio.open(path) as ds:
        subs = {s.rsplit(":", 1)[-1]: s for s in ds.subdatasets}
        def read(name):
            with rasterio.open(subs[name]) as band:
                if not 1 <= index <= band.count:
                    raise IndexError((path, index, band.count))
                return band.read(index).T
        bands = np.stack([read(b) for b in REFLECTANCE_BANDS])
        valid = ~np.isin(read("SCL"), BAD_SCL)
    image = bands.astype(np.float32) / 10000.0
    valid &= np.isfinite(image).all(axis=0)
    return np.where(valid[None], image, 0), valid


def load(n=2048, root=DEFAULT_ROOT, min_valid=0.5, seed=0, allowed_ids=None):
    """Read up to n usable patches; cloudy patches are skipped, not masked."""
    files = sorted(p for p in Path(root).glob("*.nc") if allowed_ids is None or p.stem in allowed_ids)
    if not files:
        raise FileNotFoundError(f"no .nc patches under {root}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(files))
    out = []
    for idx in order:
        if len(out) >= n:
            break
        path = files[idx]
        try:
            image, valid = _read_patch(path)
        except Exception as exc:
            raise RuntimeError(f"cannot read {path}: {exc}") from exc
        if image is None or valid.mean() < min_valid:
            continue
        out.append({
            "image": image,
            "valid": float(valid.mean()), "valid_mask": valid,
            "bands": [f"S2_{b}" for b in REFLECTANCE_BANDS],
            "centre": np.array([CENTRE_NM[b] for b in REFLECTANCE_BANDS],
                               dtype=np.float32),
            "gsd": np.array([GSD[b] for b in REFLECTANCE_BANDS], dtype=np.float32),
            "path": str(path),
        })
    return out
