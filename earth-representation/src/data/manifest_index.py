"""sample_id -> time index, read from the existing patch manifests.

The 128x128 patches hold 15 acquisitions; the manifest names the one that
matches the event post-date. Reading band 1 instead silently pairs every sample
with its earliest acquisition.
"""

import json
from pathlib import Path

import os
MANIFEST_DIR = Path(os.environ.get("EARTH_MANIFEST_ROOT", Path(__file__).resolve().parents[3] / "landslide-seg" / "manifests"))
_CACHE = None


def _load():
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    index = {}
    # Use the frozen selected-time manifest, not arbitrary overlapping histories.
    import json
    for line in (MANIFEST_DIR / "v4r1" / "pool_100.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        if "post_index" in d:
            index[d["sample_id"]] = int(d["post_index"])
    _CACHE = index
    return index


def post_index(sid):
    """0-based time index of the post-event acquisition."""
    return _load()[sid]
