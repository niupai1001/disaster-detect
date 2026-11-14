"""Batch 1 analysis: calibration pending, coverage exclusions, TRAIN statistics, figures."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "observation_model/outputs/encoder_v2/data_expansion"
DEFAULT_ROOT = ROOT / "outputs/observation_model/encoder_v2/data_expansion_v2"
DEFAULT_RECEIPT = ROOT / "outputs/encoder_v3/data_download_plan/batch_a_s2_completion"
EARTH = "https://earth-search.aws.element84.com/v1/search"
S2_BANDS = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12"]


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--skip-calibration", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))

    import numpy as np

    receipt = args.receipt
    receipt.mkdir(parents=True, exist_ok=True)

    # --- calibration pending: ask Earth Search for the exact acquisition metadata
    calibration = []
    if not args.skip_calibration:
        from observation_model.metered_http import DownloadBudget
        budget = DownloadBudget(receipt / "analysis_network.json", 50_000_000)
        pending = load_json(FROZEN / "calibration_pending.json")
        canonical = {row["id"]: row for row in lines(FROZEN / "canonical_input.jsonl")}
        for row in pending:
            record = canonical.get(row["id"])
            if not record:
                calibration.append({**row, "resolution": "no canonical record"})
                continue
            product = record["product"]
            date = record["date"][:10]
            body = {"collections": ["sentinel-2-c1-l2a"], "bbox": record["bbox"],
                    "datetime": f"{date}T00:00:00Z/{date}T23:59:59Z", "limit": 20}
            try:
                payload = json.loads(budget.bytes_request("POST", EARTH, json=body))
                features = payload.get("features", [])
            except Exception as exc:
                calibration.append({**row, "resolution": "query failed", "error": str(exc)[:160]})
                continue
            baselines = sorted({feature["properties"].get("s2:processing_baseline") for feature in features
                                if feature["properties"].get("s2:processing_baseline")})
            exact = [feature for feature in features
                     if feature["id"].split("_")[0] == product.split("_")[0]
                     and feature["properties"].get("datetime", "")[:10] == date]
            calibration.append({
                "id": row["id"], "product": product, "date": date,
                "candidates": len(features), "exact_matches": len(exact),
                "baselines": baselines,
                "resolution": "unique_baseline" if len(exact) == 1 and len(baselines) == 1 else "still_pending",
                "evidence": "Earth Search sentinel-2-c1-l2a exact-day search; no substitute acquisition used",
            })
        budget.save()
    write_json(receipt / "calibration_review.json", {
        "pending": len(calibration),
        "resolved": sum(1 for row in calibration if row.get("resolution") == "unique_baseline"),
        "still_pending": sum(1 for row in calibration if row.get("resolution") == "still_pending"),
        "rows": calibration,
    })

    # --- coverage exclusions: recompute per-band valid coverage from the frozen cache
    excluded = [row for row in lines(FROZEN / "local_events.jsonl") if row.get("status") == "excluded"]
    canonical = {row["id"]: row for row in lines(FROZEN / "canonical_input.jsonl")}
    shared = {row.get("source_id") or row.get("id"): row for row in lines(FROZEN / "shared_input.jsonl")}
    review = []
    for row in excluded:
        record = canonical.get(row["id"])
        if not record:
            review.append({**row, "review": "no canonical record"})
            continue
        original = shared.get(record["source_id"])
        try:
            cache = Path(original["cache_uri"])
            with np.load(cache, allow_pickle=False) as stored:
                x = stored["x"]
                valid = stored["pixel_valid"]
            aliases = {"blue", "green", "red", "nir"}
            selected = ([index for index, name in enumerate(original["band_ids"]) if name in aliases]
                        if record["sensor"] == "S2" else list(range(len(x))))
            fractions = {original["band_ids"][index]: float(valid[index].mean()) for index in selected}
            review.append({
                "id": row["id"], "sensor": record["sensor"], "product": record["product"],
                "fractions": fractions, "max_fraction": max(fractions.values()),
                "review": "reason_confirmed" if max(fractions.values()) < 0.5 else "needs_recheck",
                "note": "recomputed from the frozen cache; no imagery re-read",
            })
        except Exception as exc:
            review.append({**row, "review": "cache unavailable", "error": str(exc)[:160]})
    write_json(receipt / "exclusion_review.json", {
        "excluded": len(review),
        "confirmed": sum(1 for row in review if row.get("review") == "reason_confirmed"),
        "needs_recheck": sum(1 for row in review if row.get("review") == "needs_recheck"),
        "rows": review,
    })

    # --- TRAIN-only statistics over the completed members
    completed = []
    for side in sorted((args.root / "cache").glob("*.json")):
        member = load_json(side)
        if member.get("provenance") == "s2_core10_completed":
            completed.append(member)
    accum = {band: {"sum": 0.0, "sumsq": 0.0, "pixels": 0} for band in S2_BANDS}
    train = [member for member in completed if member["role"] == "TRAIN"]
    for member in train:
        with np.load(member["cache"], allow_pickle=False) as stored:
            for band in S2_BANDS:
                if band not in stored.files:
                    continue
                value = stored[band]
                valid = stored[band + "__valid"]
                values = value[valid].astype("float64")
                accum[band]["sum"] += float(values.sum())
                accum[band]["sumsq"] += float((values ** 2).sum())
                accum[band]["pixels"] += int(values.size)
    statistics = {}
    for band, data in accum.items():
        if not data["pixels"]:
            statistics[band] = {"pixels": 0, "mean": None, "std": None}
            continue
        mean = data["sum"] / data["pixels"]
        variance = max(0.0, data["sumsq"] / data["pixels"] - mean ** 2)
        statistics[band] = {"pixels": data["pixels"], "mean": mean, "std": variance ** 0.5}
    write_json(receipt / "train_statistics.json", {
        "source": "completed S2_L2A_CORE10 members, TRAIN only",
        "members": len(train), "bands": statistics,
    })

    # --- figures: value distribution and one false-colour example
    if train:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sample = train[0]
        with np.load(sample["cache"], allow_pickle=False) as stored:
            red, green, blue = stored["B04"], stored["B03"], stored["B02"]
            nir = stored["B08"]
            swir = stored["B05"]
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        values = []
        for member in train[:100]:
            with np.load(member["cache"], allow_pickle=False) as stored:
                values.append(stored["B05"][stored["B05__valid"]])
        axes[0].hist(np.concatenate(values), bins=60, color="#3b6ea5")
        axes[0].set_title("B05 (20 m, completed, TRAIN sample)")
        axes[0].set_xlabel("surface reflectance")
        rgb = np.dstack([red, green, blue])
        lo, hi = np.percentile(rgb, [2, 98])
        axes[1].imshow(np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1))
        axes[1].set_title(f"completed record {sample['id'][:18]}…")
        axes[1].axis("off")
        figure.tight_layout()
        figure.savefig(receipt / "batch_a_distribution_and_example.png", dpi=140)
        plt.close(figure)

    write_json(receipt / "analysis_done.json", {
        "completed_members": len(completed), "train_members": len(train),
        "calibration_rows": len(calibration), "exclusion_rows": len(review),
    })
    print("ANALYSIS", json.dumps({"completed": len(completed), "train": len(train),
                                  "calibration": len(calibration), "exclusions": len(review)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
