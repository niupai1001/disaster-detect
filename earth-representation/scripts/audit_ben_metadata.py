"""BigEarthNet metadata evidence: patch counts, splits and quality lists from the official parquet."""
import json
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
P0 = ROOT / "outputs/encoder_v3/data_download_plan/external_p0"
OUT = ROOT / "outputs/encoder_v3/data_download_plan/external_evidence/ben_s2_metadata.json"


def main() -> int:
    meta = pq.read_table(P0 / "bigearthnet_metadata.parquet").to_pandas()
    snow = pq.read_table(P0 / "bigearthnet_snow_cloud_shadow.parquet").to_pandas()
    evidence = {
        "patches": int(len(meta)),
        "columns": list(meta.columns),
        "splits": {str(key): int(value) for key, value in meta["split"].value_counts().items()}
        if "split" in meta else None,
        "example_rows": meta.head(3).astype(str).to_dict(orient="records"),
        "label_count": int(meta["labels"].str.split(";").explode().nunique())
        if "labels" in meta else None,
        "snow_cloud_shadow_patches": int(len(snow)),
        "snow_cloud_shadow_columns": list(snow.columns),
        "source": "Zenodo 10891137 metadata.parquet + snow/cloud/shadow parquet, read locally",
        "note": "patch-level lists only; there is no per-pixel QA layer in this release",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("BEN_META", json.dumps({k: evidence[k] for k in ("patches", "splits", "snow_cloud_shadow_patches")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

