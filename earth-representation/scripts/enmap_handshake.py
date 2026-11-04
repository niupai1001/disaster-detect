"""Complete the CAS ticket handshake after OTP, then fetch one asset.

Keycloak returns to CAS, CAS issues a service ticket, and the download service
only sets its session cookie once that ticket is redeemed at the service URL.
Skipping that step leaves an authenticated CAS session with no download rights.
"""

import argparse
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
SERVICE = "https://download.geoservice.dlr.de/ENMAP/files/"


def env():
    out = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--otp", required=True)
    ap.add_argument("--asset", default="vnir")
    args = ap.parse_args(argv)

    e = env()
    feat = requests.post(f"{STAC}/search", json={
        "collections": ["ENMAP_HSI_L2A"], "bbox": [11.0, 44.0, 11.6, 44.6],
        "datetime": "2023-04-01T00:00:00Z/2023-07-01T23:59:59Z", "limit": 1},
        timeout=120, headers=UA).json()["features"][0]
    href = feat["assets"][args.asset]["href"]

    s = requests.Session()
    lp = s.get(href, headers=UA, timeout=120)
    print("1 login page:", lp.status_code, lp.url[:60])
    delegated = next(f for f in forms(lp.text)
                     if f["fields"].get("client_name") == "codede3_enmap-dl")
    r1 = s.post(urllib.parse.urljoin(lp.url, delegated["action"]),
                data=delegated["fields"], headers={**UA, "Referer": lp.url}, timeout=120)
    kc = next(f for f in forms(r1.text) if "password" in f["fields"])
    d = dict(kc["fields"])
    d.update({"username": e["CODEDE_USER"], "password": e["CODEDE_PASS"],
              "login": "Sign In"})
    r2 = s.post(urllib.parse.urljoin(r1.url, kc["action"]), data=d,
                headers={**UA, "Referer": r1.url}, timeout=120)
    otp = next(f for f in forms(r2.text) if f["id"] == "kc-otp-login-form")
    d = dict(otp["fields"])
    d.update({"otp": args.otp, "login": "Sign In"})
    r3 = s.post(urllib.parse.urljoin(r2.url, otp["action"]), data=d,
                headers={**UA, "Referer": r2.url}, timeout=120)
    print("4 otp:", r3.status_code, r3.url[:80])
    err = re.search(r'id="input-error"[^>]*>(.*?)</span>', r3.text, re.S)
    if err:
        print("   message:", re.sub(r"<[^>]+>", " ", err.group(1)).strip()[:80])

    # redeem the ticket at the service and, if CAS hands back a redirect, follow it
    root = s.get(SERVICE, headers=UA, timeout=120)
    print("5 service root:", root.status_code, root.url[:80])
    if "login" in root.url:
        print("   still not authorised at the service")
    elif root.url != SERVICE:
        again = s.get(root.url, headers=UA, timeout=120)
        print("6 follow:", again.status_code, again.url[:80])

    r = s.get(href, headers={**UA, "Range": "bytes=0-2047"}, timeout=120)
    print("7 asset:", r.status_code, r.headers.get("content-type"),
          len(r.content), "bytes", r.content[:4].hex())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
