"""Current EnMAP access status: login attempt + one asset probe."""

import re
import sys
import urllib.parse
from pathlib import Path

import requests

ROOT = Path(r"N:\Users\yzh\Disater\earth-representation")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0"}

env = {}
for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()

BASE = "https://geoservice.dlr.de/eoc/ogc/stac/v1"
feat = requests.post(f"{BASE}/search", json={
    "collections": ["ENMAP_HSI_L2A"], "bbox": [11.0, 44.0, 11.6, 44.6],
    "datetime": "2023-04-01T00:00:00Z/2023-07-01T23:59:59Z", "limit": 1},
    timeout=120, headers=UA).json()["features"][0]
href = feat["assets"]["vnir"]["href"]
print("scene:", feat["id"])

s = requests.Session()
lp = s.get(href, headers=UA, timeout=120)
print("asset request ->", lp.status_code, "| landed:", lp.url[:70])
if "login" in lp.url:
    block = re.search(r"<form[^>]*>(?:(?!</form>).)*name=\"password\".*?</form>", lp.text, re.S)
    if not block:
        print("no login form found")
        sys.exit(0)
    block = block.group(0)
    fields = dict(re.findall(r'<input[^>]*name="([^"]+)"[^>]*value="([^"]*)"', block))
    for n in re.findall(r'<input[^>]*name="([^"]+)"', block):
        fields.setdefault(n, "")
    action = (re.search(r'<form[^>]*action="([^"]+)"', block) or [None, "login"])[1]
    fields.update({"username": env.get("ENMAP_USER", ""),
                   "password": env.get("ENMAP_PASS", "")})
    r = s.post(urllib.parse.urljoin(lp.url, action), data=fields,
               headers={**UA, "Referer": lp.url}, timeout=120)
    print("login post ->", r.status_code, "| url:", r.url[:70])
    m = re.search(r"(Invalid[^<]{0,60}|Authentication failed[^<]{0,40}|error[^<]{0,40})", r.text)
    print("message:", m.group(0) if m else "-")
    if r.status_code == 200 and "login" not in r.url:
        h = s.head(href, headers=UA, timeout=120, allow_redirects=True)
        print("asset after login:", h.status_code, h.headers.get("content-type", ""),
              h.headers.get("content-length"))
