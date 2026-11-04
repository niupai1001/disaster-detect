"""Which WildDog credential works at the DLR EOC CAS, and what does it say?

Prints the login page's own error text for each combination, so a wrong password
is distinguishable from an account/OTP problem.
"""

import re
import urllib.parse

import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0"}
STAC = "https://geoservice.dlr.de/eoc/ogc/stac/v1"

COMBOS = [
    ("WildDog", "mL6!p@z#c6S6XF5"),
    ("WildDog", "Lemon123!Lemon23"),
]

BASE = "https://geoservice.dlr.de/eoc/ogc/stac/v1"
feat = requests.post(f"{BASE}/search", json={
    "collections": ["ENMAP_HSI_L2A"], "bbox": [11.0, 44.0, 11.6, 44.6],
    "datetime": "2023-04-01T00:00:00Z/2023-07-01T23:59:59Z", "limit": 1},
    timeout=120, headers=UA).json()["features"][0]
href = feat["assets"]["vnir"]["href"]


def attempt(user, password):
    s = requests.Session()
    lp = s.get(href, headers=UA, timeout=120)
    block = re.search(r'<form[^>]*id="fm1"[^>]*>(.*?)</form>', lp.text, re.S)
    if not block:
        return "no fm1 form", ""
    fields = dict(re.findall(r'<input[^>]*name="([^"]+)"[^>]*value="([^"]*)"',
                             block.group(1)))
    for n in re.findall(r'<input[^>]*name="([^"]+)"', block.group(1)):
        fields.setdefault(n, "")
    fields.update({"username": user, "password": password})
    action = re.search(r'<form[^>]*id="fm1"[^>]*action="([^"]+)"', lp.text)
    url = urllib.parse.urljoin(lp.url, action.group(1) if action else "login")
    r = s.post(url, data=fields, headers={**UA, "Referer": lp.url}, timeout=120)

    ok = "login" not in r.url
    text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", r.text, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    m = re.search(r"(Invalid[^.]{0,80}|Authentication[^.]{0,80}|account[^.]{0,60}"
                  r"|locked[^.]{0,60}|disabled[^.]{0,60})", text, re.I)
    return ("ok" if ok else "401"), (m.group(0)[:100] if m else text[:120])


for user, pw in COMBOS:
    status, msg = attempt(user, pw)
    print(f"{user:26s} {'*' * len(pw):18s} -> {status} | {msg}")
