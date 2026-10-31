"""sample_id -> time index, read from the existing patch manifests.

The 128x128 patches hold 15 acquisitions; the manifest names the one that
matches the event post-date. Reading band 1 instead silently pairs every sample
with its earliest acquisition.
"""

import json
from pathlib import Path

MANIFEST_DIR = Path(r"N:\Users\yzh\Disater\landslide-seg\manifests")
_CACHE = None


def _load():
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    index = {}
    for p in MANIFEST_DIR.glob("*.jsonl"):
        try:
            for line in open(p, encoding="utf-8"):
                d = json.loads(line)
                sid = d.get("sample_id")
                if sid and "post_index" in d and sid not in index:
                    index[sid] = int(d["post_index"])
        except Exception:
            continue
    _CACHE = index
    return index


def post_index(sid, default=0):
    """0-based time index of the post-event acquisition."""
    return _load().get(sid, default)
