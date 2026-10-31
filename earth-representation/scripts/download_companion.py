"""Download Landsat and HLS windows aligned to the existing S2 patches.

Method: per (event, date) group, pick one scene per product, read its bands once
over the group's union bbox, then cut every patch locally. Reading per patch
would cost ~10 remote opens each; this costs ~10 per group.

Output mirrors the existing sen12 layout: <sample_id>.nc, 128x128, int16.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform_bounds
from scipy.io import netcdf_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data.adapters import _base, hls_v2, landsat_c2  # noqa: E402

SX = Path(r"N:\Users\yzh\Disater\landslide-seg\manifests")
S2_BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
LANDSAT_VARS = {"coastal": "SR_B1", "blue": "SR_B2", "green": "SR_B3",
                "red": "SR_B4", "nir08": "SR_B5", "swir16": "SR_B6",
                "swir22": "SR_B7", "qa_pixel": "QA_PIXEL", "qa_radsat": "QA_RADSAT"}
HLS_VARS = {b: b for b in S2_BANDS}
HLS_VARS["Fmask"] = "Fmask"
SIZE = 128
GSD = 10


def load_samples():
    out = {}
    for p in SX.glob("*.jsonl"):
        try:
            for line in open(p, encoding="utf-8"):
                d = json.loads(line)
                if "bbox" in d and "sample_id" in d and "crs" in d:
                    out[d["sample_id"]] = d
        except Exception:
            continue
    return out


def to_wgs84(bbox, crs):
    return transform_bounds(crs, "EPSG:4326", *bbox, densify_pts=21)


def write_nc(path, variables, attrs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with netcdf_file(str(path), "w", version=2) as nc:
        nc.createDimension("y", SIZE)
        nc.createDimension("x", SIZE)
        for name, arr in variables.items():
            var = nc.createVariable(name, arr.dtype.char, ("y", "x"))
            var[:] = arr
        for k, v in attrs.items():
            setattr(nc, k, v)


def covering_scene(scenes, centre_wgs84, target_date=None):
    """Coverage test on the STAC item bbox.

    Opening every candidate scene to read its raster bounds costs one remote
    request each and dominated the run time; the catalogue already knows the
    footprint in WGS84.
    """
    lon, lat = centre_wgs84
    hits = [s for s in scenes
            if (b := s.extra.get("bbox_wgs84")) and b[0] <= lon <= b[2] and b[1] <= lat <= b[3]]
    if not hits:
        return None
    if target_date is None:
        return hits[0]
    # nearest acquisition, not the first one that happens to cover the point:
    # a 4-month gap changes land cover and makes cross-sensor pairs meaningless
    def gap(s):
        try:
            d = dt.date.fromisoformat(str(s.datetime)[:10])
            return abs((d - target_date).days)
        except Exception:
            return 10**6
    return min(hits, key=gap)


def read_union(scene, keys, union_wgs84, workers=6):
    """Read every band once over the union bbox, at native resolution."""
    west, south, east, north = union_wgs84
    crs = CRS.from_epsg(scene.epsg)
    a = transform_bounds("EPSG:4326", crs, west, south, east, north, densify_pts=21)
    width = max(16, int(round((a[2] - a[0]) / 30.0)))
    height = max(16, int(round((a[3] - a[1]) / 30.0)))
    grid = from_bounds(a[0], a[1], a[2], a[3], width, height)

    def one(key):
        arr = _base.read_grid(scene.assets[key], grid, crs, (height, width),
                              header=scene.header,
                              resampling=_base.Resampling.nearest
                              if key in ("qa_pixel", "qa_radsat", "Fmask")
                              else _base.Resampling.bilinear)
        return key, arr

    out = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for key, arr in pool.map(one, keys):
            out[key] = arr
    return out, grid, crs


def cut_patch(arr, src_grid, src_crs, bbox, crs):
    """Resample one band from the local union array onto the patch grid."""
    dst_grid = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], SIZE, SIZE)
    dst = np.zeros((SIZE, SIZE), dtype=arr.dtype)
    reproject(source=arr, destination=dst, src_transform=src_grid, src_crs=src_crs,
              dst_transform=dst_grid, dst_crs=crs, resampling=_base.Resampling.bilinear)
    return dst


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-event", nargs="*", default=None)
    ap.add_argument("--limit-groups", type=int, default=0)
    ap.add_argument("--products", nargs="*", default=["landsat", "hls"])
    ap.add_argument("--window-days", type=int, default=120)
    ap.add_argument("--max-gap-days", type=int, default=0,
                    help="drop a pair whose acquisition is further than this")
    ap.add_argument("--out", default=str(ROOT / "data"))
    args = ap.parse_args(argv)

    samples = load_samples()
    if args.only_event:
        samples = {k: v for k, v in samples.items()
                   if v.get("event_id") in args.only_event}
    groups = defaultdict(list)
    for sid, meta in samples.items():
        groups[(meta.get("event_id"), meta.get("date"))].append((sid, meta))
    keys_sorted = sorted(groups)
    if args.limit_groups:
        keys_sorted = keys_sorted[:args.limit_groups]
    print(f"samples {len(samples)} | groups {len(keys_sorted)}", flush=True)

    in_dirs = {
        "landsat": Path(args.out) / "sen12_landsat",
        "hls": Path(args.out) / "sen12_hls",
    }
    stat = defaultdict(int)
    t0 = dt.datetime.now()

    for gi, (event, date) in enumerate(keys_sorted, 1):
        # keep a patch if *any* product is still missing; the per-product check
        # below skips the ones already on disk
        batch = [(sid, m) for sid, m in groups[(event, date)]
                 if any(not (in_dirs[p] / f"{sid}.nc").exists() for p in args.products)]
        if not batch:
            continue
        d0 = dt.date.fromisoformat(date)
        start = (d0 - dt.timedelta(days=args.window_days)).isoformat()
        end = (d0 + dt.timedelta(days=args.window_days)).isoformat()
        centres = {sid: _centre(m) for sid, m in batch}
        all_w = [to_wgs84(m["bbox"], m["crs"]) for _, m in batch]
        union = [min(w[0] for w in all_w), min(w[1] for w in all_w),
                 max(w[2] for w in all_w), max(w[3] for w in all_w)]

        for product in args.products:
            try:
                if product == "landsat":
                    scenes = landsat_c2.scenes(union, start, end, limit=100, cloud_max=90)
                else:
                    scenes = hls_v2.scenes(union, start, end, limit=100,
                                           sensor="s30", cloud_max=90)
            except Exception as exc:
                print(f"  {event}/{date} {product} search: {exc}", flush=True)
                stat[f"{product}_search_error"] += 1
                continue

            # group the patches by the scene that covers them
            per_scene = defaultdict(list)
            for sid, meta in batch:
                if (in_dirs[product] / f"{sid}.nc").exists():
                    continue
                sc = covering_scene(scenes, centres[sid],
                                    dt.date.fromisoformat(meta["date"]))
                if sc is None:
                    stat[f"{product}_no_scene"] += 1
                else:
                    per_scene[sc.scene_id].append((sid, meta, sc))
            for scene_id, items in per_scene.items():
                scene = items[0][2]
                w = [to_wgs84(m["bbox"], m["crs"]) for _, m, _ in items]
                sc_union = [min(x[0] for x in w), min(x[1] for x in w),
                            max(x[2] for x in w), max(x[3] for x in w)]
                keys = list(LANDSAT_VARS) if product == "landsat" else list(HLS_VARS)
                try:
                    arrs, grid, crs = read_union(scene, keys, sc_union)
                except Exception as exc:
                    print(f"  read {scene_id} {product}: {type(exc).__name__}: {exc}",
                          flush=True)
                    stat[f"{product}_read_error"] += len(items)
                    continue
                for sid, meta, _ in items:
                    crs_patch = CRS.from_epsg(int(meta["crs"].split(":")[1]))
                    variables = {}
                    ok = True
                    for key, var in (LANDSAT_VARS if product == "landsat" else HLS_VARS).items():
                        arr = arrs.get(key)
                        if arr is None:
                            ok = False
                            break
                        variables[var] = cut_patch(arr, grid, crs, meta["bbox"],
                                                   crs_patch).astype(np.int16)
                    if not ok:
                        stat[f"{product}_missing_band"] += 1
                        continue
                    gap_days = abs((dt.date.fromisoformat(
                        str(scene.datetime)[:10]) - dt.date.fromisoformat(meta["date"])).days)
                    if args.max_gap_days and gap_days > args.max_gap_days:
                        stat[f"{product}_too_far"] += 1
                        continue
                    attrs = {
                        "source": "LANDSAT_C2_L2" if product == "landsat" else "HLS_S30_V2",
                        "scene_id": scene.scene_id, "sensing_date": scene.datetime,
                        "s2_sample_id": sid, "crs": str(meta["crs"]),
                        "native_gsd_m": 30, "resampled_to_m": 10,
                        "bands": ",".join(variables),
                        "date_gap_days": gap_days,
                    }
                    write_nc(in_dirs[product] / f"{sid}.nc", variables, attrs)
                    stat[product] += 1
        if gi % 5 == 0:
            print(f"[{gi}/{len(keys_sorted)}] {event}/{date} {dict(stat)} "
                  f"{round((dt.datetime.now()-t0).total_seconds())}s", flush=True)

    print("done", dict(stat),
          f"{round((dt.datetime.now()-t0).total_seconds())}s")
    return 0


def _centre(meta):
    b = meta["bbox"]
    w = to_wgs84(b, meta["crs"])
    return ((w[0] + w[2]) / 2, (w[1] + w[3]) / 2)


if __name__ == "__main__":
    raise SystemExit(main())

