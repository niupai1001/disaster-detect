"""Download EnMAP L2A products through the CODE-DE (EO-Lab) login.

The account has OTP enabled, so the login is CAS -> Keycloak password -> OTP.
Both hops are real form posts; no viewer/HTTP client inside the training code.
"""

import argparse
import base64
import hashlib
import hmac
import json
import re
import struct
import time
import urllib.parse
from pathlib import Path

import requests

ROOT = Path(r"N:\Users\yzh\Disater\earth-representation")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0"}
STAC = "https://geoservice.dlr.de/eoc/ogc/stac/v1"


def totp(secret, period=30, digits=6):
    """Standard TOTP (HMAC-SHA1); the secret may be written with dashes."""
    clean = re.sub(r"[\s-]", "", secret).upper()
    clean += "=" * (-len(clean) % 8)
    key = base64.b32decode(clean)
    counter = int(time.time() // period)
    digest = hmac.new(key, struct.pack(">Q", counter), hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    code = (struct.unpack(">I", digest[offset:offset + 4])[0] & 0x7FFFFFFF)
    return f"{code % 10 ** digits:0{digits}d}"


def forms(html):
    out = []
    for m in re.finditer(r"<form[^>]*id=\"([^\"]*)\"[^>]*action=\"([^\"]*)\"(.*?)</form>",
                         html, re.S | re.I):
        fields = dict(re.findall(r'<input[^>]*name="([^"]+)"[^>]*value="([^"]*)"', m.group(3)))
        for n in re.findall(r'<input[^>]*name="([^"]+)"', m.group(3)):
            fields.setdefault(n, "")
        out.append({"id": m.group(1), "action": m.group(2).replace("&amp;", "&"),
                    "fields": fields})
    return out


def login(session, href, user, password, otp=None, secret=None):
    lp = session.get(href, headers=UA, timeout=120)
    if "login" not in lp.url and "sso" not in lp.url:
        return "already"
    delegated = next((f for f in forms(lp.text)
                      if f["fields"].get("client_name") == "codede3_enmap-dl"), None)
    if delegated is None:
        raise RuntimeError("CODE-DE entry form not found on the login page")
    r = session.post(urllib.parse.urljoin(lp.url, delegated["action"]),
                     data=delegated["fields"], headers={**UA, "Referer": lp.url},
                     timeout=120)

    kc = next((f for f in forms(r.text) if "password" in f["fields"]), None)
    if kc is None:
        raise RuntimeError(f"Keycloak password form not found (at {r.url[:60]})")
    data = dict(kc["fields"])
    data.update({"username": user, "password": password, "login": "Sign In"})
    r2 = session.post(urllib.parse.urljoin(r.url, kc["action"]), data=data,
                      headers={**UA, "Referer": r.url}, timeout=120)

    otp_form = next((f for f in forms(r2.text) if f["id"] == "kc-otp-login-form"), None)
    if otp_form is None:
        err = re.search(r'id="input-error"[^>]*>(.*?)</span>', r2.text, re.S)
        msg = re.sub(r"<[^>]+>", " ", err.group(1)).strip() if err else "unknown"
        raise RuntimeError(f"login failed: {msg}")

    code = otp or (totp(secret) if secret else None)
    if not code:
        raise RuntimeError("OTP required but neither code nor secret was given")
    data = dict(otp_form["fields"])
    data.update({"otp": code, "login": "Sign In"})
    r3 = session.post(urllib.parse.urljoin(r2.url, otp_form["action"]), data=data,
                      headers={**UA, "Referer": r2.url}, timeout=120)
    if "login-actions" in r3.url or "kc-otp" in r3.text:
        err = re.search(r'id="input-error"[^>]*>(.*?)</span>', r3.text, re.S)
        msg = re.sub(r"<[^>]+>", " ", err.group(1)).strip() if err else "still on OTP page"
        raise RuntimeError(f"OTP rejected: {msg}")
    return "ok"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="2260415@bjfu.edu.cn")
    ap.add_argument("--password", default=None)
    ap.add_argument("--secret", default=None, help="TOTP secret (base32)")
    ap.add_argument("--otp", default=None, help="one explicit code instead of the secret")
    ap.add_argument("--bbox", nargs=4, type=float,
                    default=[11.0, 44.0, 11.6, 44.6])
    ap.add_argument("--start", default="2023-04-01")
    ap.add_argument("--end", default="2023-07-01")
    ap.add_argument("--assets", nargs="*", default=["vnir", "swir"])
    ap.add_argument("--out", default=str(ROOT / "data" / "enmap"))
    ap.add_argument("--limit", type=int, default=1)
    args = ap.parse_args(argv)

    env = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    password = args.password or env.get("CODEDE_PASS")
    secret = args.secret or env.get("CODEDE_TOTP")

    doc = requests.post(f"{STAC}/search", json={
        "collections": ["ENMAP_HSI_L2A"], "bbox": args.bbox,
        "datetime": f"{args.start}T00:00:00Z/{args.end}T23:59:59Z",
        "limit": args.limit}, timeout=120, headers=UA).json()
    feats = doc.get("features", [])
    print(f"scenes: {len(feats)}")
    if not feats:
        return 1

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    status = login(session, feats[0]["assets"]["vnir"]["href"], args.user,
                   password, otp=args.otp, secret=secret)
    print("login:", status)

    manifest = []
    for feat in feats:
        for key in args.assets:
            asset = feat["assets"].get(key)
            if not asset:
                continue
            url = asset["href"]
            name = url.rsplit("/", 1)[-1]
            dst = outdir / name
            if dst.exists() and dst.stat().st_size > 0:
                print("  skip (exists)", name)
                continue
            with session.get(url, headers=UA, stream=True, timeout=300) as r:
                ctype = r.headers.get("content-type", "")
                if r.status_code != 200 or "text/html" in ctype:
                    print("  FAIL", name, r.status_code, ctype)
                    continue
                tmp = dst.with_suffix(".part")
                n = 0
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(4 << 20):
                        fh.write(chunk)
                        n += len(chunk)
                tmp.replace(dst)
            print(f"  ok {name} {n/1e6:.1f} MB")
            manifest.append({"scene": feat["id"], "asset": key, "file": str(dst),
                             "bytes": n})
    (outdir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
