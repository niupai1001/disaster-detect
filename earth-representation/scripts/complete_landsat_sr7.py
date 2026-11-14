"""Batch 2b: complete six-band Landsat 8 cache records to the seven-band SR contract.

One band (`SR_B1`) is read from the same product as the six frozen bands; the
existing valid mask is reused, so no new QA source is introduced. Frozen release
stays read-only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "observation_model/outputs/encoder_v2/data_expansion"
DEFAULT_ROOT = ROOT / "outputs/observation_model/encoder_v2/data_expansion_v2"
DEFAULT_RECEIPT = ROOT / "outputs/encoder_v3/data_download_plan/batch_b_landsat_completion"
OLD_KEYS = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
PC = "https://planetarycomputer.microsoft.com/api/stac/v1"


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
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--cap", type=int, default=0, help="remaining budget; 0 keeps the current ledger")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.receipt.mkdir(parents=True, exist_ok=True)
    lock = args.receipt / ".lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(descriptor, str(os.getpid()).encode())
        os.close(descriptor)
    except FileExistsError:
        print("ALREADY_RUNNING", lock)
        return 0
    os.environ["DSH_EXPANSION_OUT"] = str(args.root)
    if args.cap:
        os.environ["DSH_EXPANSION_CAP"] = str(args.cap)
    sys.path.insert(0, str(ROOT))

    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    from rasterio.warp import reproject

    from observation_model import expansion_data as ed

    published = [json.loads(line) for line in
                 (FROZEN / "members.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    side_dir = FROZEN / "cache"
    canonical = {row["id"]: row for row in
                 (json.loads(line) for line in (FROZEN / "canonical_input.jsonl")
                  .read_text(encoding="utf-8-sig").splitlines() if line.strip())}
    shared = {}
    for line in (FROZEN / "shared_input.jsonl").read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            row = json.loads(line)
            shared[row.get("source_id") or row.get("id")] = row

    candidates = {row["id"]: row for row in published
                  if row.get("sensor") in {"L8","L9"} and row.get("band_count") == 6}
    for side in (args.root / "cache").glob("*.json"):
        row = load_json(side)
        if row.get("sensor") in {"L8","L9"} and row.get("band_count") == 6:
            candidates[row["id"]] = row
    candidates = sorted(candidates.values(), key=lambda row: row["id"])
    net = ed.Network()

    def complete_one(sample_id):
        target_side = args.root / "cache" / (sample_id + ".json")
        if target_side.exists():
            existing = load_json(target_side)
            if existing.get("band_count") == 7:
                return "reused", {"id": sample_id}
        try:
            source_side = side_dir / (sample_id + ".json")
            if not source_side.exists():
                source_side = args.root / "cache" / (sample_id + ".json")
            member = load_json(source_side)
            cache_path = Path(member["cache"])
            if ed.digest(cache_path) != member["cache_sha256"]:
                return "excluded", {"id": sample_id, "error": "frozen member cache SHA differs"}
            with np.load(cache_path, allow_pickle=False) as stored:
                arrays, bands = {}, []
                for key in OLD_KEYS:
                    arrays[key] = (stored[key].astype("float32"), stored[key + "__valid"].astype(bool))
                    bands.append(dict(next(item for item in member["bands"] if item["band_key"] == key)))
                reference_valid = arrays[OLD_KEYS[0]][1]
            # The catalogue resolves the coastal asset for both cache and anchor records;
            # local crops do not carry the absolute original href.
            product = member.get("product")
            feature = net.json("GET", PC + "/collections/landsat-c2-l2/items/" + product)
            href = feature["assets"]["coastal"]["href"]
            anchor = member["bands"][0]
            scale, offset = float(anchor["scale"]), float(anchor["offset"])
            base = member["bands"][0]
            crs = base["crs"]
            transform = Affine(*base["transform"])
            shape = tuple(base["shape"])
            path = args.receipt / "raw" / sample_id / "SR_B1.tif"
            net.crop(href, crs, tuple(transform)[:6], shape, path, qa=False)
            with rasterio.open(path) as dataset:
                dn = dataset.read(1)
                mask = dataset.dataset_mask() > 0
                native = dataset.transform
                if abs(native.a - 30) > 1e-6 or abs(native.e + 30) > 1e-6 or native.b or native.d:
                    return "excluded", {"id": sample_id, "error": "SR_B1 is not a 30 m north-up raster"}
                keep = np.zeros(dn.shape, "uint8")
                reproject(reference_valid.astype("uint8"), keep, src_transform=transform, src_crs=crs,
                          dst_transform=native, dst_crs=crs, resampling=Resampling.min)
            quality = mask & (keep > 0) & (dn != 0)
            x = dn.astype("float32") * scale + offset
            quality &= np.isfinite(x)
            if float(quality.mean()) < 0.5:
                return "excluded", {"id": sample_id, "error": "SR_B1 valid coverage below50percent"}
            arrays["SR_B1"] = (x, quality)
            bands.append(dict(band_key="SR_B1", crs=crs, transform=list(native)[:6], shape=list(x.shape),
                              native_pixel_support_m=[30, 30], source_asset=href,
                              scale=scale, offset=offset,
                              wavelength_nm=442.96, completion="same_acquisition_native_30m"))
            metadata = {key: value for key, value in member.items()
                        if key not in {"id", "cache", "cache_sha256", "band_count", "valid_fraction",
                                       "training_authorized", "srf_scientific_ingestion", "units", "created_utc"}}
            metadata.update(bands=bands, provenance="l8_sr7_completed",
                            completed_from=str(source_side),
                            completed_from_sha256=ed.digest(source_side),
                            qa_provenance="frozen_QA_PIXEL_RADSAT_mask_reused", contract=str(member.get("sensor","L8"))+"_C2L2_SR7")
            if target_side.exists():
                existing = load_json(target_side)
                if existing.get("band_count") != 7:
                    # write_observation() reuses an existing side file; a six-band
                    # record must be replaced, not silently returned.
                    stale = Path(existing["cache"])
                    target_side.unlink(missing_ok=True)
                    stale.unlink(missing_ok=True)
            row = ed.write_observation(sample_id, arrays, metadata)
            return "accepted", {"id": sample_id, "bands": row["band_count"]}
        except RuntimeError as exc:
            if "DOWNLOAD_BUDGET_EXHAUSTED" in str(exc):
                return "budget_pending", {"id": sample_id, "error": "DOWNLOAD_BUDGET_EXHAUSTED"}
            return "excluded", {"id": sample_id, "error": str(exc)[:240]}
        except Exception as exc:
            message = str(exc).split("?")[0][:240]
            state = "retryable_network_error" if any(
                code in message.lower() for code in ["429", "timed out", "timeout", "403", "502", "503", "connection"]
            ) else "excluded"
            return state, {"id": sample_id, "error": message}

    ids = [row["id"] for row in candidates]
    if args.limit:
        ids = ids[: args.limit]
    start = net.budget.received
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for status, detail in pool.map(complete_one, ids):
            results.append((status, detail))
            with (args.receipt / "events_sr7.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"status": status, **detail}, ensure_ascii=False) + "\n")
    counts = {}
    for status, _ in results:
        counts[status] = counts.get(status, 0) + 1
    write_json(args.receipt / "sr7_summary.json", {
        "candidates": len(ids), "status_counts": counts,
        "bytes": net.budget.received - start, "received_total": net.budget.received,
        "network": load_json(args.root / "network.json") if (args.root / "network.json").exists() else None,
    })
    print("SR7", json.dumps({"candidates": len(ids), "status_counts": counts,
                             "bytes": net.budget.received - start}))
    lock.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
