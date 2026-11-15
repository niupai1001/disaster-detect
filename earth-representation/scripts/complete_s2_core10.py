"""Batch 1: complete four-band S2 lattice records to the native ten-band contract.

Read-only on the frozen expansion release. Writes completed observations into a
new expansion root (`DSH_EXPANSION_OUT`) and receipts into the batch directory.
The 20 m bands come from the same Sentinel-2 acquisition at their native 20 m
grid; nothing is interpolated, padded or taken from another date.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs/observation_model/encoder_v2/data_expansion_v2"
DEFAULT_RECEIPT = ROOT / "outputs/encoder_v3/data_download_plan/batch_a_s2_completion"
FROZEN = ROOT / "observation_model/outputs/encoder_v2/data_expansion"
SEED = 20261114
NEW_BANDS = [("rededge1", "B05", 704.1), ("rededge2", "B06", 740.5), ("rededge3", "B07", 782.8),
             ("nir08", "B8A", 864.7), ("swir16", "B11", 1613.7), ("swir22", "B12", 2202.4)]
OLD_KEYS = ["B02", "B03", "B04", "B08"]
BUDGET_ERROR = "DOWNLOAD_BUDGET_EXHAUSTED"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def append_row(path, row):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--cap", type=int, default=2_000_000_000)
    parser.add_argument("--mode", choices=["pilot", "complete"], default="pilot")
    parser.add_argument("--scope", choices=["val", "subset", "all"], default="val")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--shard", type=str, default="0/1", help="K/N shard selector")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    args.receipt.mkdir(parents=True, exist_ok=True)
    lock = args.receipt / (".lock_" + args.shard.replace("/", "_"))
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(descriptor, str(os.getpid()).encode())
        os.close(descriptor)
    except FileExistsError:
        print("ALREADY_RUNNING", lock)
        return 0
    os.environ["DSH_EXPANSION_OUT"] = str(args.out_root)
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
    canonical = {}
    for line in (FROZEN / "canonical_input.jsonl").read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            record = json.loads(line)
            canonical[record["id"]] = record
    shared = {}
    for line in (FROZEN / "shared_input.jsonl").read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            record = json.loads(line)
            shared[record.get("source_id") or record.get("id")] = record

    candidates = sorted(
        (row for row in published if row.get("sensor") == "S2" and row.get("band_count") == 4),
        key=lambda row: row["id"],
    )
    val_ids = [row["id"] for row in candidates if row["role"] == "VAL"]
    train_ids = [row["id"] for row in candidates if row["role"] == "TRAIN"]

    net = ed.Network()

    shard_index, shard_count = (int(value) for value in args.shard.split("/"))
    if not 0 <= shard_index < shard_count:
        raise SystemExit("invalid shard")

    def complete_one(sample_id):
        target = args.out_root / "cache" / (sample_id + ".json")
        if target.exists():
            return "reused", {"id": sample_id}
        try:
            member = load_json(side_dir / (sample_id + ".json"))
            cache_path = Path(member["cache"])
            if ed.digest(cache_path) != member["cache_sha256"]:
                return "excluded", {"id": sample_id, "error": "frozen member cache SHA differs"}
            references = shared[canonical[sample_id]["source_id"]]
            if sorted(band["band_key"] for band in member["bands"]) != sorted(OLD_KEYS):
                return "excluded", {"id": sample_id,
                                    "error": f"unexpected four-band set {sorted(b['band_key'] for b in member['bands'])}"}
            with np.load(cache_path, allow_pickle=False) as stored:
                arrays, bands = {}, []
                for key in OLD_KEYS:
                    value = stored[key].astype("float32")
                    quality = stored[key + "__valid"].astype(bool)
                    arrays[key] = (value, quality)
                    bands.append(dict(next(item for item in member["bands"] if item["band_key"] == key)))
                reference_valid = arrays[OLD_KEYS[0]][1]
            index = {name: position for position, name in enumerate(references["band_ids"])}
            base = member["bands"][0]
            crs = base["crs"]
            transform10 = Affine(*base["transform"])
            shape10 = tuple(base["shape"])
            raw_dir = args.receipt / "raw" / sample_id
            for source_name, band_key, wavelength in NEW_BANDS:
                if source_name in index:
                    asset = references["physical_transforms"][index[source_name]]
                else:
                    # Some lattice caches recorded only the four 10 m assets; the
                    # 20 m bands live next to them in the same product directory.
                    base_asset = member["bands"][0]
                    directory = base_asset["source_asset"].rsplit("/", 1)[0]
                    asset = dict(href=directory + "/" + band_key + ".tif",
                                 scale=base_asset["scale"], offset=base_asset["offset"])
                path = raw_dir / (band_key + ".tif")
                net.crop(asset["href"], crs, tuple(transform10)[:6], shape10, path, qa=False)
                with rasterio.open(path) as dataset:
                    dn = dataset.read(1)
                    mask = dataset.dataset_mask() > 0
                    transform20 = dataset.transform
                    if abs(transform20.a - 20) > 1e-6 or abs(transform20.e + 20) > 1e-6 or transform20.b or transform20.d:
                        return "excluded", {"id": sample_id, "error": f"{band_key} is not a 20 m north-up raster"}
                    keep = np.zeros(dn.shape, "uint8")
                    reproject(
                        reference_valid.astype("uint8"), keep,
                        src_transform=transform10, src_crs=crs,
                        dst_transform=transform20, dst_crs=crs,
                        resampling=Resampling.min,
                    )
                quality = mask & (keep > 0) & (dn != 0)
                x = dn.astype("float32") * float(asset["scale"]) + float(asset["offset"])
                quality &= np.isfinite(x)
                if float(quality.mean()) < 0.5:
                    return "excluded", {"id": sample_id, "error": f"{band_key} valid coverage below50percent"}
                arrays[band_key] = (x, quality)
                bands.append(dict(
                    band_key=band_key, crs=crs, transform=list(transform20)[:6],
                    shape=list(x.shape), native_pixel_support_m=[abs(transform20.a), abs(transform20.e)],
                    source_asset=asset["href"], scale=float(asset["scale"]),
                    offset=float(asset["offset"]),
                    wavelength_nm=wavelength,
                    completion="same_acquisition_native_20m",
                ))
            metadata = {key: value for key, value in member.items()
                        if key not in {"id", "cache", "cache_sha256", "band_count", "valid_fraction",
                                       "training_authorized", "srf_scientific_ingestion", "units", "created_utc"}}
            metadata.update(
                bands=bands,
                provenance="s2_core10_completed",
                completed_from=str(side_dir / (sample_id + ".json")),
                completed_from_sha256=ed.digest(side_dir / (sample_id + ".json")),
                qa_provenance="frozen_source_SCL_mask_downsampled_to_20m_support",
                contract="S2_L2A_CORE10",
            )
            row = ed.write_observation(sample_id, arrays, metadata)
            return "accepted", {"id": sample_id, "bands": row["band_count"],
                                "valid_fraction": row["valid_fraction"]}
        except RuntimeError as exc:
            if BUDGET_ERROR in str(exc):
                return "budget_pending", {"id": sample_id, "error": BUDGET_ERROR}
            return "excluded", {"id": sample_id, "error": str(exc)[:240]}
        except Exception as exc:  # network/geometry failures stay visible, never silently retried
            message = str(exc).split("?")[0][:240]
            state = "retryable_network_error" if any(
                code in message.lower() for code in ["429", "timed out", "timeout", "403", "502", "503", "connection"]
            ) else "excluded"
            return state, {"id": sample_id, "error": message}

    def run_batch(ids, event_file):
        results = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(complete_one, ids):
                status, detail = result
                results.append(result)
                append_row(event_file, {"status": status, **detail})
        return results

    def sha_records(results):
        return {
            status: sum(1 for name, _ in results if name == status)
            for status in {name for name, _ in results}
        }

    if args.mode == "pilot":
        rng = random.Random(SEED)
        ids = rng.sample(sorted(val_ids + train_ids), min(20, len(val_ids + train_ids)))
        start = net.budget.received
        results = run_batch(ids, args.receipt / "pilot_events.jsonl")
        spent = net.budget.received - start
        mean = spent / max(1, len(ids))
        decision = "all" if mean <= 60_000 else "subset" if mean <= 300_000 else "val_only"
        report = {
            "mode": "pilot", "seed": SEED, "records": ids,
            "status_counts": sha_records(results),
            "bytes": spent, "bytes_per_record": mean,
            "decision": decision, "cap": net.cap,
        }
        write_json(args.receipt / "pilot.json", report)
        net.budget.save()
        print("PILOT", json.dumps({k: report[k] for k in
                                   ("status_counts", "bytes", "bytes_per_record", "decision")}))
        return 0

    if args.scope == "val":
        ids = list(val_ids)
    elif args.scope == "all":
        ids = list(val_ids) + list(train_ids)
    else:
        pilot_file = args.receipt / "pilot.json"
        mean = float(load_json(pilot_file)["bytes_per_record"]) if pilot_file.exists() else 1_500_000
        remaining = max(0, net.cap - net.budget.received)
        val_outstanding = [sample_id for sample_id in val_ids
                           if not (args.out_root / "cache" / (sample_id + ".json")).exists()]
        fresh_train = [sample_id for sample_id in train_ids
                       if not (args.out_root / "cache" / (sample_id + ".json")).exists()]
        count = max(0, int(remaining // max(1.0, mean)) - len(val_outstanding))
        subset = random.Random(SEED).sample(fresh_train, min(count, len(fresh_train)))
        ids = val_outstanding + sorted(subset)
    if shard_count > 1:
        ids = [sample_id for sample_id in ids
               if int(hashlib.sha256(sample_id.encode()).hexdigest(), 16) % shard_count == shard_index]
    if args.limit:
        ids = ids[: args.limit]
    start = net.budget.received
    results = run_batch(ids, args.receipt / "events.jsonl")
    spent = net.budget.received - start
    counts = sha_records(results)
    write_json(args.receipt / "summary.json", {
        "mode": "complete", "scope": args.scope, "requested": len(ids),
        "status_counts": counts, "bytes": spent, "cap": net.cap,
        "received_total": net.budget.received,
        "root": str(args.out_root), "receipt": str(args.receipt),
    })
    network_file = args.out_root / "network.json"
    if network_file.exists():
        write_json(args.receipt / "network_batch_a.json", load_json(network_file))
    print("BATCH1", json.dumps({"requested": len(ids), "status_counts": counts,
                                "bytes": spent, "cap": net.cap}))
    lock.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
