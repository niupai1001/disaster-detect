"""Patch manifests: one JSONL row per patch, plus the region split rule."""

import json
from pathlib import Path

# Geographic split agreed before Phase 1: whole regions only, never patches of
# one region spread across splits (a scene must not leak between train and test).
SPLIT_BY_REGION = {
    "val": ("hiroshima", "italy"),
    "test": ("newzealand", "usa_puertorico"),
}
DEFAULT_SPLIT = "train"


def split_of(region):
    for split, regions in SPLIT_BY_REGION.items():
        if region in regions:
            return split
    return DEFAULT_SPLIT


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_split_manifests(root, rows, provenance):
    """Write train/val/test jsonl by region plus provenance, return counts."""
    root = Path(root)
    buckets = {"train": [], "val": [], "test": []}
    for row in rows:
        buckets[split_of(row["region"])].append(row)
    for split, items in buckets.items():
        write_jsonl(root / f"{split}.jsonl", items)
    write_jsonl(root / "provenance.jsonl", provenance)
    return {k: len(v) for k, v in buckets.items()}
