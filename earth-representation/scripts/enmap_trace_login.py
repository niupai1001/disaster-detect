"""Trace the redirect chain after the OTP step.

Login succeeds, yet the asset still returns HTML: the CAS ticket has to be
redeemed at the service before the session cookie exists. Print every hop.
"""

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
OTP = "311584"
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
href = feat["assets"]["vnir"]["href"]

s = requests.Session()
lp = s.get(href, headers=UA, timeout=120)
print("1 login page:", lp.status_code, lp.url[:70])
delegated = next(f for f in forms(lp.text) if f["fields"].get("client_name") == "codede3_enmap-dl")
r1 = s.post(urllib.parse.urljoin(lp.url, delegated["action"]), data=delegated["fields"],
            headers={**UA, "Referer": lp.url}, timeout=120)
print("2 delegated:", r1.status_code, r1.url[:70])
kc = next(f for f in forms(r1.text) if "password" in f["fields"])
data = dict(kc["fields"]); data.update({"username": env["CODEDE_USER"],
                                        "password": env["CODEDE_PASS"],
                                        "login": "Sign In"})
r2 = s.post(urllib.parse.urljoin(r1.url, kc["action"]), data=data,
            headers={**UA, "Referer": r1.url}, timeout=120)
print("3 password:", r2.status_code, r2.url[:70])
otp_form = next(f for f in forms(r2.text) if f["id"] == "kc-otp-login-form")
data = dict(otp_form["fields"]); data.update({"otp": OTP, "login": "Sign In"})
r3 = s.post(urllib.parse.urljoin(r2.url, otp_form["action"]), data=data,
            headers={**UA, "Referer": r2.url}, timeout=120)
print("4 otp:", r3.status_code, r3.url[:90])
print("   cookies:", [c.name for c in s.cookies])

# redeem the ticket at the service, then retry the asset
root = s.get("https://download.geoservice.dlr.de/ENMAP/files/", headers=UA, timeout=120)
print("5 service root:", root.status_code, root.url[:80])
r = s.get(href, headers={**UA, "Range": "bytes=0-1023"}, timeout=120, stream=True)
print("6 asset:", r.status_code, r.headers.get("content-type"), r.headers.get("content-range"))
