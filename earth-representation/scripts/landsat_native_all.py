"""Native 30 m Landsat cut for every paired patch, grouped to avoid per-patch search.

One STAC query per (region, date) group instead of one per patch; each patch then
only reads its own native window from the already-chosen scene. Existing outputs
are skipped, so the run can be resumed.
"""

import argparse
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS as PjCRS
from rasterio.warp import transform_bounds
from scipy.io import netcdf_file

ROOT = Path(r"N:\Users\yzh\Disater\earth-representation")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from download_companion import load_samples  # noqa: E402
from src.data.adapters import landsat_c2  # noqa: E402

OUT = ROOT / "data" / "landsat_native"
BANDS = ("coastal", "blue", "green", "red", "nir08", "swir16", "swir22")


def native_window(ds, bbox, crs):
    b = bbox
    if str(crs) != str(ds.crs):
        b = transform_bounds(crs, ds.crs, *bbox, densify_pts=21)
    px = ds.transform.a
    col0 = int(np.floor((b[0] - ds.transform.c) / px))
    col1 = int(np.ceil((b[2] - ds.transform.c) / px))
    row0 = int(np.floor((ds.transform.f - b[3]) / px))
    row1 = int(np.ceil((ds.transform.f - b[1]) / px))
    return rasterio.windows.Window(col0, row0, max(1, col1 - col0), max(1, row1 - row0))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    args = ap.parse_args(argv)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    samples = load_samples()
    ids = sorted(samples)
    if args.limit:
        ids = ids[args.offset:args.offset + args.limit]

    groups = defaultdict(list)
    for sid in ids:
        if (outdir / f"{sid}.nc").exists():
            continue
        m = samples[sid]
        groups[(sid.split("_s2_")[0], m["date"])].append(sid)
    print(f"pending patches: {sum(len(v) for v in groups.values())} in {len(groups)} groups", flush=True)

    manifest, done, failed = [], 0, 0
    for (region, date), sids in sorted(groups.items()):
        bboxes = [samples[s]["bbox"] for s in sids]
        crs = samples[sids[0]]["crs"]
        union = [min(b[0] for b in bboxes), min(b[1] for b in bboxes),
                 max(b[2] for b in bboxes), max(b[3] for b in bboxes)]
        w = transform_bounds(crs, "EPSG:4326", *union, densify_pts=21)
        d0 = dt.date.fromisoformat(date)
        try:
            scenes = landsat_c2.scenes(w, (d0 - dt.timedelta(days=40)).isoformat(),
                                       (d0 + dt.timedelta(days=40)).isoformat(), limit=40)
        except Exception as exc:
            print(f"  {region} {date}: search failed {type(exc).__name__}", flush=True)
            failed += len(sids)
            continue
        if not scenes:
            print(f"  {region} {date}: no scene", flush=True)
            failed += len(sids)
            continue

        def gap(sc):
            d = sc.datetime
            d = d.date() if hasattr(d, "date") else dt.date.fromisoformat(str(d)[:10])
            return abs((d - d0).days)

        scene = min(scenes, key=gap)
        try:
            with rasterio.Env(GDAL_HTTP_HEADER_FILE=scene.header):
                handles = {k: rasterio.open(scene.assets[k]) for k in scene.assets if k in BANDS}
                for sid in sids:
                    m = samples[sid]
                    cube, shape = [], None
                    for k in BANDS:
                        ds = handles.get(k)
                        if ds is None:
                            continue
                        a = ds.read(1, window=native_window(ds, m["bbox"], m["crs"]))
                        cube.append(a)
                        shape = a.shape
                    if not cube:
                        continue
                    arr = np.stack(cube).astype(np.int16)
                    with netcdf_file(str(outdir / f"{sid}.nc"), "w", version=2) as nc:
                        nc.createDimension("band", arr.shape[0])
                        nc.createDimension("y", arr.shape[1])
                        nc.createDimension("x", arr.shape[2])
                        v = nc.createVariable("sr", arr.dtype.char, ("band", "y", "x"))
                        v[:] = arr
                        setattr(nc, "native_gsd_m", np.int32(30))
                        setattr(nc, "resampled", b"False")
                        setattr(nc, "bands", ",".join(BANDS).encode())
                        setattr(nc, "scene_id", str(getattr(scene, "scene_id", "")).encode())
                    manifest.append({"sample_id": sid, "shape": list(arr.shape),
                                     "scene": str(getattr(scene, "scene_id", "")),
                                     "date_gap_days": int(gap(scene))})
                    done += 1
                for ds in handles.values():
                    ds.close()
        except Exception as exc:
            print(f"  {region} {date}: read failed {type(exc).__name__}: {str(exc)[:70]}", flush=True)
            failed += len(sids)
        print(f"  {region} {date}: scene gap {gap(scene)}d, {len(sids)} patches "
              f"(total {done})", flush=True)

    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"done {done}, failed {failed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
