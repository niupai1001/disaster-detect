"""Batch 4 (P0): download the external metadata, splits and licence documents.

Only public objects are fetched; imagery tarballs are deliberately excluded.
Every download is size-verified, hashed and recorded with its fixed version.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/encoder_v3/data_download_plan/external_p0"
TIMEOUT = 120
CHUNK = 1 << 20
ZENODO = {
    "bigearthnet_v2": "10891137",
    "worldstrat_v1_1": "15382551",
    "sen2venus_v2": "14603764",
}
OBJECTS = [
    # (source, url, local name, expected minimum bytes)
    ("bigearthnet_v2", "https://zenodo.org/records/10891137/files/metadata.parquet?download=1", "bigearthnet_metadata.parquet", 3_616_349),
    ("bigearthnet_v2", "https://zenodo.org/records/10891137/files/metadata_for_patches_with_snow_cloud_or_shadow.parquet?download=1", "bigearthnet_snow_cloud_shadow.parquet", 710_162),
    ("bigearthnet_v2", "https://zenodo.org/records/10891137/files/Reference_Maps.tar.zst?download=1", "bigearthnet_reference_maps.tar.zst", 282_391_301),
    ("bigearthnet_l", "https://huggingface.co/datasets/isaaccorley/bigearthnet-l/resolve/main/bigearthnet-train.csv", "bigearthnet_l_train.csv", 21_156_152),
    ("bigearthnet_l", "https://huggingface.co/datasets/isaaccorley/bigearthnet-l/resolve/main/bigearthnet-val.csv", "bigearthnet_l_val.csv", 9_704_306),
    ("bigearthnet_l", "https://huggingface.co/datasets/isaaccorley/bigearthnet-l/resolve/main/bigearthnet-test.csv", "bigearthnet_l_test.csv", 9_872_180),
    ("bigearthnet_l", "https://huggingface.co/datasets/isaaccorley/bigearthnet-l/resolve/main/README.md", "bigearthnet_l_README.md", 1),
    ("fmow_full", "https://spacenet-dataset.s3.amazonaws.com/Hosted-Datasets/fmow/fmow-full/manifest.json.bz2", "fmow_manifest.json.bz2", 4_643_019),
    ("fmow_full", "https://spacenet-dataset.s3.amazonaws.com/Hosted-Datasets/fmow/fmow-full/LICENSE", "fmow_LICENSE", 15_096),
    ("fmow_full", "https://spacenet-dataset.s3.amazonaws.com/Hosted-Datasets/fmow/fmow-full/README.md", "fmow_README.md", 2_351),
    ("flair_hub", "https://huggingface.co/datasets/IGNF/FLAIR-HUB/resolve/main/README.md", "flair_hub_README.md", 13_621),
    ("sen2venus_v2", "https://zenodo.org/records/14603764/files/FGMANAUS.zip?download=1", "sen2venus_FGMANAUS.zip", 121_202_328),
]


def fetch(url: str, target: Path) -> dict:
    import requests

    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + f".{os.getpid()}.tmp")
    received = 0
    with requests.get(url, stream=True, timeout=(20, TIMEOUT)) as response:
        response.raise_for_status()
        with temp.open("wb") as handle:
            for block in response.iter_content(CHUNK):
                if block:
                    handle.write(block)
                    received += len(block)
    sha = hashlib.sha256(temp.read_bytes()).hexdigest()
    temp.replace(target)
    return {"bytes": received, "sha256": sha, "path": str(target)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    failures = []
    total = 0
    for source, url, name, minimum in OBJECTS:
        target = args.out / name
        if target.exists() and target.stat().st_size >= minimum:
            sha = hashlib.sha256(target.read_bytes()).hexdigest()
            records.append({"source": source, "url": url, "file": name,
                            "bytes": target.stat().st_size, "sha256": sha, "status": "reused"})
            total += target.stat().st_size
            continue
        try:
            info = fetch(url, target)
            ok = info["bytes"] >= minimum
            status = "ok" if ok else "size_below_expected"
            if not ok:
                failures.append({"file": name, "expected_min": minimum, "got": info["bytes"]})
            records.append({"source": source, "url": url, "file": name, **info, "status": status})
            total += info["bytes"]
        except Exception as exc:
            failures.append({"file": name, "error": str(exc)[:200]})
            records.append({"source": source, "url": url, "file": name, "status": "failed",
                            "error": str(exc)[:200]})
    catalogue = {
        "generated_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(timespec="seconds"),
        "versions": ZENODO,
        "licences": {
            "bigearthnet_v2": "CDLA-Permissive-1.0",
            "worldstrat_v1_1": "HR archives CC BY-NC 4.0, rest CC BY 4.0",
            "sen2venus_v2": "other-nc (non-commercial)",
            "bigearthnet_l": "see dataset README",
            "fmow_full": "see dataset LICENSE",
            "flair_hub": "see dataset README",
        },
        "objects": records, "failures": failures, "total_bytes": total,
    }
    (args.out / "source_catalog.json").write_text(
        json.dumps(catalogue, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.out / "downloads.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("P0", json.dumps({"objects": len(records), "failures": len(failures),
                            "total_bytes": total}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
