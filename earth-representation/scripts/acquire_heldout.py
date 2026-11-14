"""Acquire held-out TEST and VAL top-up records from geography outside the frozen pools.

Candidates come from land rectangles that do not intersect any existing record
bbox. Every candidate must pass the same QA screen before it is written:
S2 requires SCL-clear support >=95% on the native 10/20 m grids, Landsat requires
QA_PIXEL bits 0-5 clear and QA_RADSAT==0 >=95%. Nothing is resampled into the
observation; only the QA mask is reprojected.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs/observation_model/encoder_v2/data_expansion_v2_heldout"
DEFAULT_RECEIPT = ROOT / "outputs/encoder_v3/data_download_plan/heldout_completion"
RELEASE = ROOT / "outputs/encoder_v3/data_download_plan/local_completion_release_v1/members.jsonl"
PC = "https://planetarycomputer.microsoft.com/api/stac/v1"
SEED = 20261114
S2_BANDS = [("B02", "B02", 10), ("B03", "B03", 10), ("B04", "B04", 10), ("B08", "B08", 10),
            ("B05", "B05", 20), ("B06", "B06", 20), ("B07", "B07", 20), ("B8A", "B8A", 20),
            ("B11", "B11", 20), ("B12", "B12", 20)]
LS_BANDS = [("coastal", "SR_B1"), ("blue", "SR_B2"), ("green", "SR_B3"), ("red", "SR_B4"),
            ("nir08", "SR_B5"), ("swir16", "SR_B6"), ("swir22", "SR_B7")]
# Remote, usually cloud-free land: high chance of >=95% clear support and no
# overlap with the frozen local pools.
LANDS = [(2, 12, 22, 30), (12, 25, 22, 30), (25, 34, 22, 30), (-12, 0, 18, 27),
         (-8, 3, 30, 36), (18, 30, 15, 25), (33, 46, 15, 25), (44, 56, 18, 25),
         (70, 78, 25, 30), (78, 88, 28, 34), (88, 98, 36, 42), (98, 108, 39, 44),
         (52, 64, 34, 40), (60, 72, 40, 46), (120, 132, -26, -20),
         (132, 144, -28, -20), (115, 125, -33, -27), (-70, -66, -26, -20),
         (-70, -66, -33, -27), (17, 23, -27, -22), (23, 29, -25, -20),
         (44, 52, 28, 33), (56, 66, 28, 33), (66, 72, 24, 27)]


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--cap", type=int, default=10_000_000_000)
    parser.add_argument("--s2-test", type=int, default=100)
    parser.add_argument("--l8-test", type=int, default=100)
    parser.add_argument("--l9-test", type=int, default=100)
    parser.add_argument("--l8-val", type=int, default=23)
    parser.add_argument("--l9-val", type=int, default=77)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--use-sites", type=int, default=1)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.receipt.mkdir(parents=True, exist_ok=True)
    os.environ["DSH_EXPANSION_OUT"] = str(args.out)
    os.environ["DSH_EXPANSION_CAP"] = str(args.cap)
    sys.path.insert(0, str(ROOT))

    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import Affine, array_bounds
    from rasterio.windows import from_bounds
    from rasterio.warp import transform_bounds, reproject
    from shapely.geometry import box
    from shapely.strtree import STRtree

    from observation_model import expansion_data as ed
    from observation_model.native_data import BAD

    existing = [json.loads(line) for line in RELEASE.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    existing_boxes = [box(*row["bbox"]) for row in existing if row.get("bbox")]
    tree = STRtree(existing_boxes)

    def overlaps(bbox, pad=0.01):
        probe = box(bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
        return any(probe.intersection(existing_boxes[int(i)]).area > 0
                   for i in tree.query(probe))

    net = ed.Network()
    rng = random.Random(SEED)
    cells = []
    site_dir = ROOT / "data/raw/multisensor/four_view_pilot_v1/sites"
    if args.use_sites and site_dir.exists():
        for site in sorted(site_dir.iterdir()):
            path = site / "site.json"
            if not path.exists():
                continue
            payload = load_json(path)
            try:
                coords = payload["footprint_wgs84"]["coordinates"][0]
            except Exception:
                continue
            lons = [point[0] for point in coords]
            lats = [point[1] for point in coords]
            cells.append([min(lons), min(lats), max(lons), max(lats)])
        cells = [cell for cell in cells if not overlaps(cell)]
        rng.shuffle(cells)
        print("SITES", len(cells), flush=True)
    if not cells:
        attempts = 0
        while len(cells) < 800 and attempts < 6000:
            attempts += 1
            lon_min, lon_max, lat_min, lat_max = rng.choice(LANDS)
            lon = rng.uniform(lon_min, lon_max)
            lat = rng.uniform(lat_min, lat_max)
            bbox = [lon - 0.02, lat - 0.02, lon + 0.02, lat + 0.02]
            if overlaps(bbox):
                continue
            cells.append(bbox)

    def stac_search(collection, bbox, extra=None):
        body = {"collections": [collection], "bbox": bbox,
                "datetime": "2015-01-01T00:00:00Z/2026-12-31T00:00:00Z", "limit": 20,
                "query": {"eo:cloud_cover": {"lt": 10}}}
        if extra:
            body["query"].update(extra)
        return net.json("POST", PC + "/search", json=body)

    def read_band(href, crs, transform10, shape10, path):
        net.crop(href, crs, tuple(Affine(*transform10))[:6], shape10, path, qa=False)
        with rasterio.open(path) as dataset:
            data = dataset.read(1)
            mask = dataset.dataset_mask() > 0
            return data, mask, dataset.transform

    def window_at(bbox, crs, transform, size=128):
        lon = (bbox[0] + bbox[2]) / 2
        lat = (bbox[1] + bbox[3]) / 2
        x, y = rasterio.warp.transform("EPSG:4326", crs, [lon], [lat])
        col = int(round((x[0] - transform.c) / transform.a))
        row = int(round((transform.f - y[0]) / abs(transform.e)))
        return rasterio.windows.Window(col - size // 2, row - size // 2, size, size)

    def valid_on_grid(mask_source, source_transform, crs, target_transform, target_shape):
        keep = np.zeros(target_shape, "uint8")
        reproject(mask_source.astype("uint8"), keep, src_transform=source_transform, src_crs=crs,
                  dst_transform=target_transform, dst_crs=crs, resampling=Resampling.min)
        return keep > 0

    s2_written = 0
    ls_counts = {"L8": 0, "L9": 0}
    val_counts = {"L8": 0, "L9": 0}
    for side in (args.out / "cache").glob("*.json"):
        row = load_json(side)
        if row.get("provenance") != "new_geography_heldout":
            continue
        sensor = row.get("sensor")
        if sensor == "S2" and row.get("role") == "TEST":
            s2_written += 1
        elif sensor in {"L8", "L9"} and row.get("role") == "TEST":
            ls_counts[sensor] += 1
        elif sensor in {"L8", "L9"} and row.get("role") == "VAL":
            val_counts[sensor] += 1
    events = []
    for bbox in cells:
        if s2_written >= args.s2_test and all(ls_counts[s] >= args.l8_test if s == "L8" else True for s in ls_counts):
            pass
        if s2_written >= args.s2_test:
            break
        try:
            payload = stac_search("sentinel-2-l2a", bbox)
            features = [f for f in payload.get("features", [])
                        if f["properties"].get("eo:cloud_cover", 100) < cloud_limit
                        and f["id"] not in used_s2]
            if not features:
                continue
            feature = min(features, key=lambda f: f["properties"].get("eo:cloud_cover", 100))
            assets = feature["assets"]
            crs = f"EPSG:{feature['properties']['proj:epsg']}"
            t10 = feature["assets"]["B02"]["proj:transform"]
            transform10 = Affine(*t10)
            height, width = feature["assets"]["B02"]["proj:shape"]
            window = window_at(bbox, crs, transform10, 128)
            if window.col_off < 0 or window.row_off < 0 or window.col_off + 128 > width or window.row_off + 128 > height:
                continue
            transform10 = rasterio.windows.transform(window, transform10)
            bounds10 = array_bounds(128, 128, transform10)
            actual = transform_bounds(crs, "EPSG:4326", *bounds10)
            if overlaps(actual):
                continue
            folder = args.receipt / "raw" / feature["id"]
            arrays, bands, fine = {}, [], True
            scl_path = folder / "SCL.tif"
            net.crop(assets["SCL"]["href"], crs, tuple(transform10)[:6], (128, 128), scl_path, qa=True)
            with rasterio.open(scl_path) as dataset:
                scl = dataset.read(1)
                clear10 = ~np.isin(scl, BAD)
            if float(clear10.mean()) < 0.95:
                continue
            for asset_key, band_key, gsd in S2_BANDS:
                path = folder / (band_key + ".tif")
                data, mask, native = read_band(assets[asset_key]["href"], crs, tuple(transform10)[:6],
                                               (128, 128), path)
                quality = mask & valid_on_grid(clear10, transform10, crs, native, data.shape)
                quality &= data != 0
                if float(quality.mean()) < 0.95:
                    fine = False
                    break
                x = data.astype("float32") * 0.0001 - 0.1
                arrays[band_key] = (x, quality & np.isfinite(x))
                bands.append(dict(band_key=band_key, crs=crs, transform=list(native)[:6],
                                  shape=list(data.shape), native_pixel_support_m=[gsd, gsd],
                                  source_asset=assets[asset_key]["href"], scale=0.0001, offset=-0.1,
                                  wavelength_nm=0.0, completion="new_geography_heldout"))
            if not fine or len(arrays) != 10:
                continue
            centre = ((actual[0] + actual[2]) / 2, (actual[1] + actual[3]) / 2)
            identifier = "s2_test_" + str(s2_written).zfill(3) + "_" + feature["id"][:16]
            row = ed.write_observation(identifier, arrays, dict(
                sensor="S2", instrument=feature["id"].split("_")[0], product=feature["id"],
                date=feature["properties"]["datetime"], role="TEST", bbox=list(actual),
                isolation_group=f"cell_{centre[0]:.2f}_{centre[1]:.2f}", bands=bands,
                processing_domain="original_L2_SR", provenance="new_geography_heldout",
                qa_provenance="SCL classes 0/1/3/8/9/10/11/255 excluded",
                acquisition_key=feature["id"], contract="S2_L2A_CORE10"))
            existing_boxes.append(box(*actual))
            tree = STRtree(existing_boxes)
            s2_written += 1
            used_s2.add(feature["id"])
            events.append({"id": identifier, "role": "TEST", "sensor": "S2"})
            if s2_written % 10 == 0:
                print("S2", s2_written, flush=True)
        except Exception as exc:
            events.append({"status": "excluded", "error": str(exc)[:160]})
            continue

    for sensor, test_target, val_target in (("L8", args.l8_test, args.l8_val),
                                            ("L9", args.l9_test, args.l9_val)):
        platform = {"L8": "landsat-8", "L9": "landsat-9"}[sensor]
        needed = test_target + val_target
        written = ls_counts[sensor] + val_counts[sensor]
        used_ls = set()
        for pass_index, bbox in enumerate(cells * 3):
            cloud_limit = (10, 30, 60)[pass_index % 3]
            if written >= needed:
                break
            try:
                payload = net.json("POST", PC + "/search", json={
                    "collections": ["landsat-c2-l2"], "bbox": bbox,
                    "datetime": "2015-01-01T00:00:00Z/2026-12-31T00:00:00Z", "limit": 20,
                    "query": {"platform": {"in": [platform]}, "eo:cloud_cover": {"lt": 10}}})
                features = [f for f in payload.get("features", [])
                            if f["properties"].get("platform") == platform
                            and f["properties"].get("eo:cloud_cover", 100) < cloud_limit
                            and f["id"] not in used_ls]
                if not features:
                    continue
                feature = min(features, key=lambda f: f["properties"].get("eo:cloud_cover", 100))
                assets = feature["assets"]
                crs = f"EPSG:{feature['properties']['proj:epsg']}"
                t30 = assets["blue"]["proj:transform"]
                transform30 = Affine(*t30)
                height, width = assets["blue"]["proj:shape"]
                window = window_at(bbox, crs, transform30, 128)
                if window.col_off < 0 or window.row_off < 0 or window.col_off + 128 > width or window.row_off + 128 > height:
                    continue
                transform30 = rasterio.windows.transform(window, transform30)
                bounds30 = array_bounds(128, 128, transform30)
                actual = transform_bounds(crs, "EPSG:4326", *bounds30)
                if overlaps(actual):
                    continue
                folder = args.receipt / "raw" / feature["id"]
                qa_paths = {}
                for key in ("qa_pixel", "qa_radsat"):
                    path = folder / (key + ".tif")
                    net.crop(assets[key]["href"], crs, tuple(transform30)[:6], (128, 128), path, qa=True)
                    qa_paths[key] = path
                with rasterio.open(qa_paths["qa_pixel"]) as dataset:
                    qa_pixel = dataset.read(1)
                with rasterio.open(qa_paths["qa_radsat"]) as dataset:
                    qa_radsat = dataset.read(1)
                clear = ((qa_pixel.astype("uint16") & 63) == 0) & (qa_radsat == 0)
                if float(clear.mean()) < 0.95:
                    continue
                arrays, bands, fine = {}, [], True
                for asset_key, band_key in LS_BANDS:
                    path = folder / (band_key + ".tif")
                    data, mask, native = read_band(assets[asset_key]["href"], crs,
                                                   tuple(transform30)[:6], (128, 128), path)
                    quality = mask & clear & (data != 0)
                    if float(quality.mean()) < 0.95:
                        fine = False
                        break
                    x = data.astype("float32") * 0.0000275 - 0.2
                    arrays[band_key] = (x, quality & np.isfinite(x))
                    bands.append(dict(band_key=band_key, crs=crs, transform=list(native)[:6],
                                      shape=list(data.shape), native_pixel_support_m=[30, 30],
                                      source_asset=assets[asset_key]["href"], scale=0.0000275,
                                      offset=-0.2, wavelength_nm=0.0,
                                      completion="new_geography_heldout"))
                if not fine or len(arrays) != 7:
                    continue
                role = "TEST" if written < test_target else "VAL"
                centre = ((actual[0] + actual[2]) / 2, (actual[1] + actual[3]) / 2)
                identifier = f"{sensor.lower()}_{role.lower()}_" + str(written).zfill(3) + "_" + feature["id"][:16]
                ed.write_observation(identifier, arrays, dict(
                    sensor=sensor, instrument="LC08" if sensor == "L8" else "LC09",
                    product=feature["id"], date=feature["properties"]["datetime"], role=role,
                    bbox=list(actual), isolation_group=f"cell_{centre[0]:.2f}_{centre[1]:.2f}",
                    bands=bands, processing_domain="original_L2_SR",
                    provenance="new_geography_heldout",
                    qa_provenance="QA_PIXEL bits0-5 clear + QA_RADSAT==0",
                    acquisition_key=feature["id"], contract=f"{sensor}_C2L2_SR7"))
                existing_boxes.append(box(*actual))
                tree = STRtree(existing_boxes)
                written += 1
                used_ls.add(feature["id"])
                events.append({"id": identifier, "role": role, "sensor": sensor})
                if written % 20 == 0:
                    print(sensor, written, flush=True)
            except Exception as exc:
                events.append({"status": "excluded", "error": str(exc)[:160]})
                continue

    summary = {"s2_test": s2_written, "landsat": ls_counts, "events": len(events),
               "network": load_json(args.out / "network.json")}
    write_json(args.receipt / "summary.json", summary)
    with (args.receipt / "events.jsonl").open("w", encoding="utf-8") as handle:
        for row in events:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("HELDOUT", json.dumps({"s2_test": s2_written, "network": summary["network"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
