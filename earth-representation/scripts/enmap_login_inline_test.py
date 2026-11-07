"""Inline test: the login block copied verbatim from the script that reaches 206."""

import re
import sys
import urllib.parse
from pathlib import Path

import requests

ROOT = Path(r"N:\Users\yzh\Disater\earth-representation")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from scripts.enmap_download import forms  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0"}
STAC = "https://geoservice.dlr.de/eoc/ogc/stac/v1"

env = {}
for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()

feat = requests.post(f"{STAC}/search", json={
    "collections": ["ENMAP_HSI_L2A"], "bbox": [11.0, 44.0, 11.6, 44.6],
    "datetime": "2023-04-01T00:00:00Z/2023-07-01T23:59:59Z", "limit": 1},
    timeout=120, headers=UA).json()["features"][0]
href = feat["assets"]["image"]["href"]

s = requests.Session()
lp = s.get(href, headers=UA, timeout=120)
print("landed:", lp.url[:70])

eoc = next((f for f in forms(lp.text) if "password" in f["fields"]
            and f["fields"].get("client_name") is None), None)
data = dict(eoc["fields"])
data.update({"username": env["ENMAP_USER"], "password": env["ENMAP_PASS"]})
data.setdefault("_eventId", "submit")
r = s.post(urllib.parse.urljoin(lp.url, eoc["action"] or "login"), data=data,
           headers={**UA, "Referer": lp.url}, timeout=120)
print("login ->", r.status_code, r.url[:80])

probe = s.get(href, headers={**UA, "Range": "bytes=0-1023"}, timeout=120)
print("probe:", probe.status_code, probe.headers.get("content-type"),
      probe.headers.get("content-range"), probe.content[:4].hex())
