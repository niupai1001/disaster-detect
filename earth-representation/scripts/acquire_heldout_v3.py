"""Phase 2 held-out acquisition, scene-parallel.

Scenes are independent work units; each scene chooses one role (VAL or TEST) so a
single acquisition never spans roles, and its windows become separate geographic
groups. Window reads run concurrently, which is what made the sequential version
too slow.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs/observation_model/encoder_v2/data_expansion_v2_heldout"
DEFAULT_RECEIPT = ROOT / "outputs/encoder_v3/data_download_plan/heldout_completion_v3"
FULL_RELEASE = ROOT / "outputs/encoder_v3/data_download_plan/local_completion_release_v1/members.jsonl"
PC = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_BANDS = [("B02", "B02", 10), ("B03", "B03", 10), ("B04", "B04", 10), ("B08", "B08", 10),
            ("B05", "B05", 20), ("B06", "B06", 20), ("B07", "B07", 20), ("B8A", "B8A", 20),
            ("B11", "B11", 20), ("B12", "B12", 20)]
LS_BANDS = [("coastal", "SR_B1"), ("blue", "SR_B2"), ("green", "SR_B3"), ("red", "SR_B4"),
            ("nir08", "SR_B5"), ("swir16", "SR_B6"), ("swir22", "SR_B7")]
REGIONS = [(1, 12, 21, 30), (12, 25, 21, 30), (25, 34, 21, 30), (-12, 0, 17, 27),
           (-8, 3, 29, 36), (18, 30, 14, 25), (33, 46, 14, 25), (44, 56, 17, 25),
           (70, 78, 24, 30), (78, 88, 27, 34), (88, 98, 35, 42), (98, 108, 38, 44),
           (52, 64, 33, 40), (60, 72, 39, 46), (120, 132, -27, -20),
           (132, 144, -29, -20), (115, 125, -34, -27), (-70, -66, -27, -20),
           (-70, -66, -34, -27), (17, 23, -28, -22), (23, 29, -26, -20),
           (44, 52, 27, 33), (56, 66, 27, 33), (66, 72, 23, 27)]
FRACTIONS = [(0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8), (0.5, 0.5), (0.35, 0.65), (0.65, 0.35)]


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
    parser.add_argument("--cap", type=int, default=8_000_000_000)
    parser.add_argument("--s2-test", type=int, default=84)
    parser.add_argument("--l8-val", type=int, default=70)
    parser.add_argument("--l8-test", type=int, default=99)
    parser.add_argument("--l9-val", type=int, default=76)
    parser.add_argument("--l9-test", type=int, default=99)
    parser.add_argument("--scenes", type=int, default=60, help="candidate scenes per sensor")
    parser.add_argument("--workers", type=int, default=12)
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
    from rasterio.windows import Window, transform as window_transform
    from rasterio.warp import transform_bounds, reproject
    from shapely.geometry import box
    from shapely.strtree import STRtree

    from observation_model import expansion_data as ed
    from observation_model.native_data import BAD

    released = [json.loads(line) for line in FULL_RELEASE.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    base_boxes = [box(*row["bbox"]) for row in released if row.get("bbox")]
    base_tree = STRtree(base_boxes)
    groups = {row["isolation_group"]: row["role"] for row in released}
    lock = threading.Lock()
    # Held-out records from earlier runs are already taken: never reuse their cells.
    initial_heldout = []
    for side in (args.out / "cache").glob("*.json"):
        row = load_json(side)
        if row.get("provenance", "").startswith("new_geography_heldout"):
            initial_heldout.append(row)
            groups.setdefault(row["isolation_group"], row.get("role"))
    new_boxes = [box(*row["bbox"]) for row in initial_heldout if row.get("bbox")]
    net = ed.Network()
    rng = random.Random(20261114)
    remaining = {"S2/TEST": args.s2_test, "L8/VAL": args.l8_val, "L8/TEST": args.l8_test,
                 "L9/VAL": args.l9_val, "L9/TEST": args.l9_test}
    written = {key: 0 for key in remaining}
    events = []
    event_lock = threading.Lock()

    def log(event):
        with event_lock:
            events.append(event)
            if len(events) % 20 == 0:
                print("EVENTS", len(events), json.dumps(written), flush=True)

    def overlaps(bbox, pad=0.01):
        probe = box(bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
        if any(probe.intersection(base_boxes[int(i)]).area > 0 for i in base_tree.query(probe)):
            return True
        return any(probe.intersection(other).area > 0 for other in new_boxes)

    def choose_role(sensor):
        options = [f"{sensor}/{role}" for role in ("VAL", "TEST") if remaining.get(f"{sensor}/{role}", 0) > 0]
        return max(options, key=lambda key: remaining[key]) if options else None

    def window_at(transform, height, width, fraction, size=128):
        col = max(0, min(int(width * fraction[0]) - size // 2, width - size))
        row = max(0, min(int(height * fraction[1]) - size // 2, height - size))
        return Window(col, row, size, size)

    def valid_on_grid(mask_source, source_transform, crs, target_transform, target_shape):
        keep = np.zeros(target_shape, "uint8")
        reproject(mask_source.astype("uint8"), keep, src_transform=source_transform, src_crs=crs,
                  dst_transform=target_transform, dst_crs=crs, resampling=Resampling.min)
        return keep > 0

    def read_band(href, crs, transform, shape, path):
        net.crop(href, crs, tuple(transform)[:6], shape, path, qa=False)
        with rasterio.open(path) as dataset:
            return dataset.read(1), dataset.dataset_mask() > 0, dataset.transform

    def process_s2(scene):
        role = choose_role("S2")
        if role is None:
            return
        feature = scene["feature"]
        try:
            assets = feature["assets"]
            crs = f"EPSG:{feature['properties']['proj:epsg']}"
            transform = Affine(*assets["B02"]["proj:transform"])
            height, width = assets["B02"]["proj:shape"]
            for fraction in FRACTIONS:
                with lock:
                    if remaining[role] <= 0:
                        return
                window = window_at(transform, height, width, fraction, 128)
                target = window_transform(window, transform)
                actual = transform_bounds(crs, "EPSG:4326", *array_bounds(128, 128, target))
                group = f"cell_{((actual[0] + actual[2]) / 2):.2f}_{((actual[1] + actual[3]) / 2):.2f}"
                with lock:
                    if overlaps(actual) or group in groups:
                        continue
                folder = args.receipt / "raw" / feature["id"] / group.replace(".", "_")
                arrays, bands, fine = {}, [], True
                scl_path = folder / "SCL.tif"
                net.crop(assets["SCL"]["href"], crs, tuple(target)[:6], (128, 128), scl_path, qa=True)
                with rasterio.open(scl_path) as dataset:
                    scl = dataset.read(1)
                clear = ~np.isin(scl, BAD)
                if float(clear.mean()) < 0.95:
                    continue
                for asset_key, band_key, gsd in S2_BANDS:
                    data, mask, native = read_band(assets[asset_key]["href"], crs,
                                                   tuple(target)[:6], (128, 128), folder / (band_key + ".tif"))
                    quality = mask & valid_on_grid(clear, target, crs, native, data.shape) & (data != 0)
                    if float(quality.mean()) < 0.95:
                        fine = False
                        break
                    x = data.astype("float32") * 0.0001 - 0.1
                    arrays[band_key] = (x, quality & np.isfinite(x))
                    bands.append(dict(band_key=band_key, crs=crs, transform=list(native)[:6],
                                      shape=list(data.shape), native_pixel_support_m=[gsd, gsd],
                                      source_asset=assets[asset_key]["href"], scale=0.0001,
                                      offset=-0.1, wavelength_nm=0.0,
                                      completion="new_geography_heldout_v3"))
                if not fine or len(arrays) != 10:
                    continue
                identifier = "s2_test_v3_" + hashlib.sha256((group + feature["id"]).encode()).hexdigest()[:12]
                with lock:
                    if remaining[role] <= 0 or group in groups or overlaps(actual):
                        continue
                    ed.write_observation(identifier, arrays, dict(
                        sensor="S2", instrument=feature["id"].split("_")[0], product=feature["id"],
                        date=feature["properties"]["datetime"], role="TEST", bbox=list(actual),
                        isolation_group=group, bands=bands, processing_domain="original_L2_SR",
                        provenance="new_geography_heldout_v3",
                        qa_provenance="SCL classes 0/1/3/8/9/10/11/255 excluded",
                        acquisition_key=feature["id"], contract="S2_L2A_CORE10"))
                    new_boxes.append(box(*actual))
                    groups[group] = "TEST"
                    remaining[role] -= 1
                    written[role] += 1
                log({"stage": "s2", "id": identifier, "role": "TEST", "group": group})
        except Exception as exc:
            log({"stage": "s2_feature", "id": feature.get("id"), "error": str(exc)[:160]})

    def process_landsat(sensor, scene):
        role = choose_role(sensor)
        if role is None:
            return
        feature = scene["feature"]
        try:
            assets = feature["assets"]
            crs = f"EPSG:{feature['properties']['proj:epsg']}"
            transform = Affine(*feature["properties"]["proj:transform"])
            height, width = feature["properties"]["proj:shape"]
            for fraction in FRACTIONS:
                with lock:
                    if remaining[role] <= 0:
                        return
                window = window_at(transform, height, width, fraction, 128)
                target = window_transform(window, transform)
                actual = transform_bounds(crs, "EPSG:4326", *array_bounds(128, 128, target))
                group = f"cell_{((actual[0] + actual[2]) / 2):.2f}_{((actual[1] + actual[3]) / 2):.2f}"
                with lock:
                    if overlaps(actual) or group in groups:
                        continue
                folder = args.receipt / "raw" / feature["id"] / group.replace(".", "_")
                qa_paths = {}
                for key in ("qa_pixel", "qa_radsat"):
                    path = folder / (key + ".tif")
                    net.crop(assets[key]["href"], crs, tuple(target)[:6], (128, 128), path, qa=True)
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
                    data, mask, native = read_band(assets[asset_key]["href"], crs,
                                                   tuple(target)[:6], (128, 128), folder / (band_key + ".tif"))
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
                                      completion="new_geography_heldout_v3"))
                if not fine or len(arrays) != 7:
                    continue
                selected = role.split("/")[1]
                identifier = f"{sensor.lower()}_{selected.lower()}_v3_" + hashlib.sha256((group + feature["id"]).encode()).hexdigest()[:12]
                with lock:
                    if remaining[role] <= 0 or group in groups or overlaps(actual):
                        continue
                    ed.write_observation(identifier, arrays, dict(
                        sensor=sensor, instrument="LC08" if sensor == "L8" else "LC09",
                        product=feature["id"], date=feature["properties"]["datetime"],
                        role=selected, bbox=list(actual), isolation_group=group, bands=bands,
                        processing_domain="original_L2_SR", provenance="new_geography_heldout_v3",
                        qa_provenance="QA_PIXEL bits0-5 clear + QA_RADSAT==0",
                        acquisition_key=feature["id"], contract=f"{sensor}_C2L2_SR7"))
                    new_boxes.append(box(*actual))
                    groups[group] = selected
                    remaining[role] -= 1
                    written[role] += 1
                log({"stage": sensor.lower(), "id": identifier, "role": selected, "group": group})
        except Exception as exc:
            log({"stage": f"{sensor.lower()}_feature", "id": feature.get("id"), "error": str(exc)[:160]})

    def gather(collection, query, limit=None):
        scenes = []
        for region in REGIONS:
            bbox = [region[0], region[2], region[1], region[3]]
            try:
                payload = net.json("POST", PC + "/search", json={
                    "collections": [collection], "bbox": bbox,
                    "datetime": "2015-01-01T00:00:00Z/2026-12-31T00:00:00Z", "limit": 40,
                    "query": query})
            except Exception as exc:
                log({"stage": "search", "collection": collection, "region": region,
                     "error": str(exc)[:160]})
                continue
            features = sorted(payload.get("features", []),
                              key=lambda f: f["properties"].get("eo:cloud_cover", 100))
            for feature in features[:5]:
                scenes.append({"feature": feature})
            if limit and len(scenes) >= limit:
                break
        rng.shuffle(scenes)
        return scenes

    for sensor, process in (("S2", process_s2), ("L8", process_landsat), ("L9", process_landsat)):
        if sensor == "S2":
            scenes = gather("sentinel-2-l2a", {"eo:cloud_cover": {"lt": 15}}, args.scenes)
        else:
            platform = {"L8": "landsat-8", "L9": "landsat-9"}[sensor]
            scenes = gather("landsat-c2-l2",
                            {"platform": {"in": [platform]}, "eo:cloud_cover": {"lt": 30}}, args.scenes)
        print("SCENES", sensor, len(scenes), json.dumps(remaining), flush=True)
        if not any(key.startswith(sensor + "/") and value > 0 for key, value in remaining.items()):
            continue
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            if sensor == "S2":
                list(pool.map(process_s2, scenes))
            else:
                list(pool.map(lambda scene: process_landsat(sensor, scene), scenes))

    summary = {"written": written, "remaining": remaining, "events": len(events),
               "network": load_json(args.out / "network.json") if (args.out / "network.json").exists()
               else {"received_bytes": net.budget.received, "cap_bytes": args.cap}}
    write_json(args.receipt / "summary.json", summary)
    with (args.receipt / "events.jsonl").open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    print("HELDOUT_V3", json.dumps({"written": written, "remaining": remaining,
                                    "network": summary["network"]["received_bytes"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
