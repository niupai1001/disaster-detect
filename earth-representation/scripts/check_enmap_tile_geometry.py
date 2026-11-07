"""Cross-check scheme C against a direct rasterio cut of the same scene.

Uses the one full scene kept in data/enmap: a patch whose bbox falls inside it is
cut both ways and compared. A near-1 correlation means the tile indexing and the
origin arithmetic are right.
"""

import glob
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform_bounds

ROOT = Path(r"N:\Users\yzh\Disater\earth-representation")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from download_companion import load_samples  # noqa: E402

scene = glob.glob(str(ROOT / "data" / "enmap" / "*SPECTRAL*.TIF"))[0]
samples = load_samples()

# the scene covers roughly 11.0-11.6 E, 44.0-44.6 N (Italy)
cands = []
for sid in sorted(samples):
    m = samples[sid]
    w = transform_bounds(m["crs"], "EPSG:4326", *m["bbox"], densify_pts=21)
    lon, lat = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
    if 11.0 < lon < 11.6 and 44.0 < lat < 44.6:
        cands.append(sid)
print("candidates in the local scene:", len(cands), cands[:3])

with rasterio.open(scene) as ds:
    print("scene", ds.count, ds.width, ds.height, ds.crs, ds.transform.a)

if cands:
    sid = cands[0]
    m = samples[sid]
    with rasterio.open(scene) as ds:
        win = rasterio.windows.from_bounds(*m["bbox"], ds.transform).round_offsets().round_lengths()
        win = rasterio.windows.Window(win.col_off, win.row_off, max(1, win.width), max(1, win.height))
        data = ds.read(window=win)
        grid = rasterio.windows.transform(win, ds.transform)
        print("window", data.shape)

    # same recipe as the tile path
    dst = np.zeros((data.shape[0], 128, 128), dtype=np.int16)
    dst_grid = from_bounds(*m["bbox"], 128, 128)
    patch_crs = CRS.from_epsg(int(m["crs"].split(":")[1]))
    for i in range(data.shape[0]):
        reproject(source=data[i], destination=dst[i], src_transform=grid,
                  src_crs=ds.crs, dst_transform=dst_grid, dst_crs=patch_crs,
                  resampling=Resampling.bilinear)

    from scipy.io import netcdf_file
    f = ROOT / "data" / "enmap_pairs" / f"{sid}.nc"
    if f.exists():
        nc = netcdf_file(str(f), "r", mmap=False)
        other = np.array(nc.variables["spectral"][:])
        nc.close()
        a, b = dst.astype(float).ravel(), other.astype(float).ravel()
        mask = (a > -30000) & (b > -30000)
        print("correlation with stored tile patch:",
              round(float(np.corrcoef(a[mask], b[mask])[0, 1]), 4), "| n:", int(mask.sum()))
    else:
        print("stored patch for this sample not present yet:", f.name)
