"""Resolve the ten S2 calibration-pending records from frozen release facts.

Two shapes exist in the frozen release:
* native lattice caches (4 x 10 m channels recorded, 20 m bands still fetchable
  from the same product) -> resolve baseline and complete the ten-band contract;
* counterpart caches (12 channels already resampled to a 30 m grid) -> the
  calibration is known but the product is DERIVED, so it is recorded as such and
  never admitted as CORE_NATIVE.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "observation_model/outputs/encoder_v2/data_expansion"
OUT = ROOT / "outputs/observation_model/encoder_v2/data_expansion_v2_calibration"
RECEIPT = ROOT / "outputs/encoder_v3/data_download_plan/calibration_resolution"
BAND_NAMES = {"blue": "B02", "green": "B03", "red": "B04", "rededge1": "B05",
              "rededge2": "B06", "rededge3": "B07", "nir": "B08", "nir08": "B8A",
              "swir16": "B11", "swir22": "B12"}
NEW_BANDS = [("rededge1", "B05", 704.1), ("rededge2", "B06", 740.5), ("rededge3", "B07", 782.8),
             ("nir08", "B8A", 864.7), ("swir16", "B11", 1613.7), ("swir22", "B12", 2202.4)]


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
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--receipt", type=Path, default=RECEIPT)
    parser.add_argument("--cap", type=int, default=500_000_000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.receipt.mkdir(parents=True, exist_ok=True)
    os.environ["DSH_EXPANSION_OUT"] = str(args.out)
    os.environ["DSH_EXPANSION_CAP"] = str(args.cap)
    sys.path.insert(0, str(ROOT))

    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    from observation_model import expansion_data as ed

    net = ed.Network()
    pending = load_json(FROZEN / "calibration_pending.json")
    canonical = {json.loads(line)["id"]: json.loads(line)
                 for line in (FROZEN / "canonical_input.jsonl").read_text(encoding="utf-8-sig").splitlines()
                 if line.strip()}
    shared = {}
    for line in (FROZEN / "shared_input.jsonl").read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            row = json.loads(line)
            shared[row.get("source_id") or row.get("id")] = row

    resolved_native, derived, still = [], [], []
    for item in pending:
        record = canonical[item["id"]]
        original = shared[record["source_id"]]
        baseline = str((record.get("properties") or {}).get("s2:processing_baseline", ""))
        cache = Path(record["cache_uri"])
        with np.load(cache, allow_pickle=False) as stored:
            x = stored["x"].copy()
            valid = stored["pixel_valid"].copy()
            spectral = stored["metadata"].copy()
        if "physical_transforms" not in original:
            # Counterpart product: 12 channels resampled onto a 30 m grid.
            calibration = original.get("calibration") or {}
            errors = {}
            for band_id, entry in calibration.items():
                index = original["band_ids"].index(band_id)
                scale, offset = float(entry["scale"]), float(entry["offset"])
                quality = valid[index]
                dn = np.rint((x[index][quality] - offset) / scale) if quality.any() else np.array([0.0])
                errors[band_id] = float(np.max(np.abs(dn * scale + offset - x[index][quality]))) if quality.any() else None
            derived.append({
                "id": item["id"], "baseline": baseline, "product": record["product"],
                "bands": len(original["band_ids"]), "grid_m": abs(original["transform"][0]),
                "class": "DERIVED_ONLY", "reason": "12-channel product already resampled to a 30 m grid",
                "roundtrip_max_abs": errors,
            })
            continue
        # Native lattice route: four 10 m channels in the cache, 20 m bands from the product.
        index = {name: position for position, name in enumerate(original["band_ids"])}
        base_band = record["window"]
        crs = base_band["crs"]
        from rasterio.transform import Affine
        transform10 = Affine(*base_band["transform"])
        shape10 = tuple(base_band["shape"])
        arrays, bands = {}, []
        for band_id in ("blue", "green", "red", "nir"):
            position = original["band_ids"].index(band_id)
            band_key = BAND_NAMES[band_id]
            scale, offset = float(original["physical_transforms"][position]["scale"]), float(original["physical_transforms"][position]["offset"])
            quality = valid[position].copy()
            values = x[position]
            dn = np.rint((values[quality] - offset) / scale)
            error = float(np.max(np.abs(dn * scale + offset - values[quality]))) if quality.any() else float("inf")
            if error > max(abs(scale) * 0.51, 1e-6):
                still.append({"id": item["id"], "reason": f"{band_key} round-trip error {error}"})
                arrays = {}
                break
            arrays[band_key] = (values.astype("float32"), quality)
            bands.append(dict(band_key=band_key, crs=crs, transform=list(transform10)[:6],
                              shape=list(values.shape), native_pixel_support_m=[10, 10],
                              source_asset=original["physical_transforms"][position]["href"],
                              scale=scale, offset=offset, cached_already_scaled=True,
                              arithmetic_roundtrip_max_abs=error,
                              wavelength_nm=float(spectral[position][0]) * 1000))
        if not arrays:
            continue
        reference_valid = arrays["B02"][1]
        for source_name, band_key, wavelength in NEW_BANDS:
            asset = original["physical_transforms"][index[source_name]]
            path = args.receipt / "raw" / item["id"] / (band_key + ".tif")
            net.crop(asset["href"], crs, tuple(transform10)[:6], shape10, path, qa=False)
            with rasterio.open(path) as dataset:
                dn = dataset.read(1)
                mask = dataset.dataset_mask() > 0
                native = dataset.transform
                keep = np.zeros(dn.shape, "uint8")
                reproject(reference_valid.astype("uint8"), keep, src_transform=transform10, src_crs=crs,
                          dst_transform=native, dst_crs=crs, resampling=Resampling.min)
            quality = mask & (keep > 0) & (dn != 0)
            values = dn.astype("float32") * float(asset["scale"]) + float(asset["offset"])
            quality &= np.isfinite(values)
            if float(quality.mean()) < 0.5:
                still.append({"id": item["id"], "reason": f"{band_key} coverage below 50%"})
                arrays = {}
                break
            arrays[band_key] = (values, quality)
            bands.append(dict(band_key=band_key, crs=crs, transform=list(native)[:6],
                              shape=list(values.shape),
                              native_pixel_support_m=[abs(native.a), abs(native.e)],
                              source_asset=asset["href"], scale=float(asset["scale"]),
                              offset=float(asset["offset"]), wavelength_nm=wavelength,
                              completion="same_acquisition_native_20m"))
        if not arrays or len(arrays) != 10:
            continue
        ed.write_observation(item["id"], arrays, dict(
            sensor="S2", instrument=record.get("instrument") or record["product"].split("_")[0],
            product=record["product"], date=record["date"], role=record.get("effective_partition", "TRAIN"),
            bbox=record["bbox"], isolation_group=record["isolation_group"], bands=bands,
            processing_domain="original_L2_SR", provenance="s2_calibration_resolved",
            qa_provenance="frozen_source_SCL_mask_downsampled_to_20m_support",
            calibration_baseline=baseline, calibration_reference="canonical STAC properties",
            source_cache=str(cache), source_cache_sha256=ed.digest(cache),
            acquisition_key=record.get("acquisition_key"), contract="S2_L2A_CORE10"))
        resolved_native.append({"id": item["id"], "baseline": baseline, "product": record["product"]})

    summary = {"pending": len(pending),
               "resolved_core10": len(resolved_native), "resolved_derived_only": len(derived),
               "still_pending": len(still), "records": {
                   "core10": resolved_native, "derived": derived, "pending": still},
               "network": load_json(args.out / "network.json") if (args.out / "network.json").exists() else None,
               "root": str(args.out)}
    write_json(args.receipt / "resolution.json", summary)
    print("CALIBRATION", json.dumps({"pending": len(pending), "core10": len(resolved_native),
                                     "derived": len(derived), "still": len(still)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
