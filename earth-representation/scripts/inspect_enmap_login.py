"""Inspect the CAS login page: the EO-Lab (CODE-DE) entry point."""

import re

import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0"}

BASE = "https://geoservice.dlr.de/eoc/ogc/stac/v1"
feat = requests.post(f"{BASE}/search", json={
    "collections": ["ENMAP_HSI_L2A"], "bbox": [11.0, 44.0, 11.6, 44.6],
    "datetime": "2023-04-01T00:00:00Z/2023-07-01T23:59:59Z", "limit": 1},
    timeout=120, headers=UA).json()["features"][0]
href = feat["assets"]["vnir"]["href"]

s = requests.Session()
lp = s.get(href, headers=UA, timeout=120)
print("landed:", lp.status_code, lp.url[:80])
html = lp.text

for m in re.finditer(r"<form[^>]*>", html, re.I):
    print("FORM:", m.group(0)[:220])

# the EO-Lab form fields and any links pointing at cloudferro / EO-Lab
block = re.search(r'<form[^>]*id="formcodede3_enmap-dl"[^>]*>(.*?)</form>', html, re.S | re.I)
if block:
    body = block.group(1)
    for i in re.finditer(r"<input[^>]*>", body, re.I):
        print("  INPUT:", i.group(0)[:160])
    print("  BUTTONS:", re.findall(r"<button[^>]*>(.*?)</button>", body, re.S)[:3])

for kw in ("cloudferro", "EO-Lab", "eolab", "CODE-DE", "cloud"): 
    hits = [m.start() for m in re.finditer(kw, html)]
    if hits:
        print(f"mention {kw}: {len(hits)}")
        i = hits[0]
        print("   ...", re.sub(r"\s+", " ", html[max(0, i - 120):i + 160])[:260])
