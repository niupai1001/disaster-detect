"""One-shot EnMAP login+download with full diagnostics.

Prints the exact CAS error parameters when the OTP step fails, because
"expired code" and "account not entitled to the EnMAP client" both surface as a
redirect back to sso.eoc.dlr.de but mean different things.
"""

import argparse
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
    ap.add_argument("--assets", nargs="*", default=["vnir"])
    ap.add_argument("--out", default=str(ROOT / "data" / "enmap"))
    args = ap.parse_args(argv)

    e = env()
    feat = requests.post(f"{STAC}/search", json={
        "collections": ["ENMAP_HSI_L2A"], "bbox": [11.0, 44.0, 11.6, 44.6],
        "datetime": "2023-04-01T00:00:00Z/2023-07-01T23:59:59Z", "limit": 1},
        timeout=120, headers=UA).json()["features"][0]
    href = feat["assets"][args.assets[0]]["href"]

    s = requests.Session()
    lp = s.get(href, headers=UA, timeout=120)
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
    otp = next((f for f in forms(r2.text) if f["id"] == "kc-otp-login-form"), None)
    if otp is None:
        print("password step did not reach the OTP form; url:", r2.url[:100])
        return 1
    d = dict(otp["fields"])
    d.update({"otp": args.otp, "login": "Sign In"})
    r3 = s.post(urllib.parse.urljoin(r2.url, otp["action"]), data=d,
                headers={**UA, "Referer": r2.url}, timeout=120)
    print("otp ->", r3.status_code, r3.url[:120])
    q = urllib.parse.parse_qs(urllib.parse.urlparse(r3.url).query)
    for key in ("error", "error_description", "code"):
        if key in q:
            print(f"  {key} = {q[key][0][:120]}")

    if r3.status_code >= 400:
        # CAS rejected the assertion: the page text says why (expired ticket,
        # access_denied, invalid client, ...)
        text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", r3.text, flags=re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        print("  CAS page:", text[:400])
    if "login" in r3.url and "code=" not in r3.url:
        print("login not completed; not attempting the asset")
        return 1

    root = s.get(SERVICE, headers=UA, timeout=120)
    print("service:", root.status_code, root.url[:80])

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    ok = 0
    for key in args.assets:
        url = feat["assets"][key]["href"]
        name = url.rsplit("/", 1)[-1]
        dst = outdir / name
        with s.get(url, headers=UA, stream=True, timeout=300) as r:
            ctype = r.headers.get("content-type", "")
            print(f"{key}: {r.status_code} {ctype}")
            if r.status_code != 200 or "text/html" in ctype:
                continue
            tmp = dst.with_suffix(".part")
            n = 0
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(4 << 20):
                    fh.write(chunk)
                    n += len(chunk)
            tmp.replace(dst)
            print(f"  saved {name} {n/1e6:.1f} MB")
            ok += 1
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
