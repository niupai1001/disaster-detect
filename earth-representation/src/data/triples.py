"""Load (S2, Landsat, HLS) triples aligned on the same 128x128 patch grid.

The companion files were cut from the same ground footprint as the S2 patch, so
sample_id is the join key and the three arrays describe the same place at
roughly the same time.
"""

from pathlib import Path

import numpy as np
from scipy.io import netcdf_file

ROOT = Path(r"N:\Users\yzh\Disater\earth-representation\data")
# the original S2 patches live in the landslide-seg project, not here
S2_DIR = Path(r"N:\Users\yzh\Disater\landslide-seg\data\sen12")

S2_VARS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
LANDSAT_VARS = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
HLS_VARS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

# band names must match src/data/band_catalog.py
S2_NAMES = [f"S2_{v}" for v in S2_VARS]
LANDSAT_NAMES = [f"LANDSAT_SR_B{i}" for i in range(1, 8)]
HLS_NAMES = [f"HLS_{v}" for v in HLS_VARS]

CENTRE = {
    "S2_B02": 492.4, "S2_B03": 559.8, "S2_B04": 664.6, "S2_B05": 704.1,
    "S2_B06": 740.5, "S2_B07": 782.8, "S2_B08": 832.8, "S2_B8A": 864.7,
    "S2_B11": 1613.7, "S2_B12": 2202.4,
    "LANDSAT_SR_B1": 442.96, "LANDSAT_SR_B2": 481.59, "LANDSAT_SR_B3": 561.25,
    "LANDSAT_SR_B4": 654.59, "LANDSAT_SR_B5": 864.67, "LANDSAT_SR_B6": 1608.86,
    "LANDSAT_SR_B7": 2200.73,
    "HLS_B02": 492.4, "HLS_B03": 559.8, "HLS_B04": 664.6, "HLS_B05": 704.1,
    "HLS_B06": 740.5, "HLS_B07": 782.8, "HLS_B08": 832.8, "HLS_B8A": 864.7,
    "HLS_B11": 1613.7, "HLS_B12": 2202.4,
}
GSD = {k: (10 if k.startswith("S2_B0") or k in ("S2_B08",) else 20 if k.startswith("S2_")
           else 30) for k in CENTRE}
GSD.update({"S2_B02": 10, "S2_B03": 10, "S2_B04": 10, "S2_B08": 10,
            "S2_B05": 20, "S2_B06": 20, "S2_B07": 20, "S2_B8A": 20,
            "S2_B11": 20, "S2_B12": 20})
for k in list(GSD):
    if k.startswith("LANDSAT") or k.startswith("HLS"):
        GSD[k] = 30

SCALE = {"s2": 10000.0, "landsat": 10000.0, "hls": 10000.0}


def _read(path, variables):
    with netcdf_file(str(path), "r", mmap=False) as nc:
        return np.stack([np.array(nc.variables[v][:], dtype=np.float32)
                         for v in variables])


def _read_s2(path, variables, time_index=1):
    """S2 patches are NetCDF4 with a 15-step time axis; pick the matching date.

    Reading band 1 always gave the first acquisition (2018-09-06), not the date
    the manifest records, which silently mis-paired every cross-sensor test.
    """
    import rasterio
    with rasterio.open(str(path)) as ds:
        subs = {s.rsplit(":", 1)[-1]: s for s in ds.subdatasets}
        out = []
        for v in variables:
            with rasterio.open(subs[v]) as d:
                idx = min(max(1, time_index), d.count)
                # stored transposed: without .T the patch is a mirrored map
                out.append(d.read(idx).astype(np.float32).T)
        return np.stack(out)


def _compose(path, variables, names, scale, s2=False, time_index=1):
    a = (_read_s2(path, variables, time_index) if s2 else _read(path, variables)) / scale
    centre = np.array([CENTRE[n] for n in names], dtype=np.float32)
    gsd = np.array([GSD[n] for n in names], dtype=np.float32)
    return {"image": a, "centre": centre, "gsd": gsd, "bands": names}


def load_triples(limit=0, min_valid=0.0):
    from . import manifest_index
    """Yield dicts with s2 / landsat / hls entries for the same sample_id."""
    s2_dir = S2_DIR
    ls_dir = ROOT / "sen12_landsat"
    hls_dir = ROOT / "sen12_hls"
    ids = sorted({p.stem for p in ls_dir.glob("*.nc")} &
                 {p.stem for p in hls_dir.glob("*.nc")} &
                 {p.stem for p in s2_dir.glob("*.nc")})
    if limit:
        ids = ids[:limit]
    out = []
    for sid in ids:
        try:
            out.append({
                "sample_id": sid,
                "region": sid.split("_s2_")[0],
                "s2": _compose(s2_dir / f"{sid}.nc", S2_VARS, S2_NAMES, SCALE["s2"],
                                s2=True, time_index=manifest_index.post_index(sid) + 1),
                "landsat": _compose(ls_dir / f"{sid}.nc", LANDSAT_VARS,
                                    LANDSAT_NAMES, SCALE["landsat"]),
                "hls": _compose(hls_dir / f"{sid}.nc", HLS_VARS, HLS_NAMES, SCALE["hls"]),
            })
        except Exception:
            continue
    return out


def load_mixed(n_per_product=700, seed=0):
    """Training items mixing the three sensors, one entry per (sample, sensor)."""
    import numpy as np
    items = load_triples(limit=0)
    if not items:
        return []
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(items))
    out = []
    for i in idx:
        for key in ("s2", "landsat", "hls"):
            out.append({"image": items[i][key]["image"],
                        "centre": items[i][key]["centre"],
                        "gsd": items[i][key]["gsd"],
                        "bands": items[i][key]["bands"],
                        "path": f"{items[i]['sample_id']}_{key}",
                        "sensor": key})
            if len(out) >= 3 * n_per_product:
                return out
    return out
