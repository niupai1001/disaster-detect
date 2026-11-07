"""Native 30 m Landsat cut, one union read per group.

Reading each patch separately cost seven remote reads per patch (~9 s). Per
(group, band) the union bbox is read once - a few hundred pixels - and every
patch in the group is then sliced locally, which is where the time went.
Outputs are skipped if they exist, so the run resumes.
"""

import argparse
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform_bounds
from scipy.io import netcdf_file

ROOT = Path(r"N:\Users\yzh\Disater\earth-representation")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from download_companion import load_samples  # noqa: E402
from src.data.adapters import landsat_c2  # noqa: E402

OUT = ROOT / "data" / "landsat_native"
BANDS = ("coastal", "blue", "green", "red", "nir08", "swir16", "swir22")
PAD = 60      # metres of slack around the group bbox


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    samples = load_samples()
    ids = sorted(samples)
    if args.limit:
        ids = ids[: args.limit]

    groups = defaultdict(list)
    for sid in ids:
        if (outdir / f"{sid}.nc").exists():
            continue
        m = samples[sid]
        groups[(sid.split("_s2_")[0], m["date"])].append(sid)
    print(f"pending {sum(len(v) for v in groups.values())} patches in {len(groups)} groups",
          flush=True)

    manifest, done, failed = [], 0, 0
    for (region, date), sids in sorted(groups.items()):
        boxes = [samples[s]["bbox"] for s in sids]
        crs = samples[sids[0]]["crs"]
        union = [min(b[0] for b in boxes) - PAD, min(b[1] for b in boxes) - PAD,
                 max(b[2] for b in boxes) + PAD, max(b[3] for b in boxes) + PAD]
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
        # native grid of the scene, union window, one read per band
        try:
            with rasterio.Env(GDAL_HTTP_HEADER_FILE=scene.header):
                stacks = []
                grid = None
                for k in BANDS:
                    asset = scene.assets.get(k)
                    if not asset:
                        continue
                    with rasterio.open(asset) as ds:
                        b = union if str(crs) == str(ds.crs) else transform_bounds(
                            crs, ds.crs, *union, densify_pts=21)
                        px = ds.transform.a
                        c0 = int(np.floor((b[0] - ds.transform.c) / px))
                        c1 = int(np.ceil((b[2] - ds.transform.c) / px))
                        r0 = int(np.floor((ds.transform.f - b[3]) / px))
                        r1 = int(np.ceil((ds.transform.f - b[1]) / px))
                        win = rasterio.windows.Window(c0, r0, max(2, c1 - c0), max(2, r1 - r0))
                        a = ds.read(1, window=win)
                        stacks.append(a)
                        if grid is None:
                            grid = rasterio.windows.transform(win, ds.transform)
                            scene_crs = ds.crs
        except Exception as exc:
            print(f"  {region} {date}: union read failed {type(exc).__name__}: {str(exc)[:60]}",
                  flush=True)
            failed += len(sids)
            continue

        if not stacks:
            failed += len(sids)
            continue
        arr = np.stack(stacks)
        for sid in sids:
            m = samples[sid]
            patch_crs = rasterio.crs.CRS.from_epsg(int(m["crs"].split(":")[1]))
            h = max(1, int(round((m["bbox"][3] - m["bbox"][1]) / 30.0)))
            wpx = max(1, int(round((m["bbox"][2] - m["bbox"][0]) / 30.0)))
            if str(scene_crs) == str(patch_crs):
                # same CRS: slice whole scene-grid pixels, no interpolation at all
                px = grid.a
                c0 = int(round((m["bbox"][0] - grid.c) / px))
                r0 = int(round((grid.f - m["bbox"][3]) / px))
                c0 = max(0, min(c0, arr.shape[2] - wpx))
                r0 = max(0, min(r0, arr.shape[1] - h))
                dst = arr[:, r0:r0 + h, c0:c0 + wpx].astype(np.int16)
            else:
                dst = np.zeros((arr.shape[0], h, wpx), dtype=np.int16)
                reproject(source=arr, destination=dst, src_transform=grid,
                          src_crs=scene_crs, dst_transform=from_bounds(*m["bbox"], wpx, h),
                          dst_crs=patch_crs, resampling=Resampling.nearest)
            with netcdf_file(str(outdir / f"{sid}.nc"), "w", version=2) as nc:
                nc.createDimension("band", dst.shape[0])
                nc.createDimension("y", dst.shape[1])
                nc.createDimension("x", dst.shape[2])
                v = nc.createVariable("sr", dst.dtype.char, ("band", "y", "x"))
                v[:] = dst
                setattr(nc, "native_gsd_m", np.int32(30))
                setattr(nc, "resampled", b"False")
                setattr(nc, "bands", ",".join(BANDS).encode())
                setattr(nc, "scene_id", str(getattr(scene, "scene_id", "")).encode())
            manifest.append({"sample_id": sid, "shape": list(dst.shape),
                             "scene": str(getattr(scene, "scene_id", "")),
                             "date_gap_days": int(gap(scene))})
            done += 1
        print(f"  {region} {date}: gap {gap(scene)}d, {len(sids)} patches "
              f"(total {done})", flush=True)

    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"done {done}, failed {failed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
