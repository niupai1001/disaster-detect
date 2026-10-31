"""SAS signing with expiry, shared by the two Planetary Computer adapters.

The token lives about an hour; a long build outlived a cached one and every
later read failed with HTTP 403. Cache with a timestamp and refresh in time.
"""

import time
import urllib.parse

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import _common as C  # noqa: E402

TTL_SECONDS = 40 * 60
_CACHE = {}


def sas_for(href):
    u = urllib.parse.urlparse(href)
    key = (u.netloc.split(".")[0], u.path.strip("/").split("/")[0])
    now = time.time()
    hit = _CACHE.get(key)
    if hit and now - hit[0] < TTL_SECONDS:
        return hit[1]
    doc = C.get_json("https://planetarycomputer.microsoft.com/api/sas/v1/token/"
                     f"{key[0]}/{key[1]}", retries=3)
    _CACHE[key] = (now, doc["token"])
    return doc["token"]


def signed(href):
    return f"{href}?{sas_for(href)}"
