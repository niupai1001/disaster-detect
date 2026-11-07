"""Matrix: which account authenticates on the EOC CAS and on CODE-DE?

Two different identity systems sit behind the same login page, and they report
failures differently:
  EOC:      "Authentication attempt has failed"  = wrong credentials
            "missing privileges"                 = authenticated, not entitled
  CODE-DE:  Keycloak "Invalid username or password" / OTP form / CAS 401
"""

import re
import urllib.parse

import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0"}
STAC = "https://geoservice.dlr.de/eoc/ogc/stac/v1"

ACCOUNTS = [
    ("lemon123", "mL6!p@z#c6S6XF5"),
]

feat = requests.post(f"{STAC}/search", json={
    "collections": ["ENMAP_HSI_L2A"], "bbox": [11.0, 44.0, 11.6, 44.6],
    "datetime": "2023-04-01T00:00:00Z/2023-07-01T23:59:59Z", "limit": 1},
    timeout=120, headers=UA).json()["features"][0]
href = feat["assets"]["vnir"]["href"]


def text_of(html):
    t = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.S)
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def eoc(user, pw):
    s = requests.Session()
    lp = s.get(href, headers=UA, timeout=120)
    blk = re.search(r'<form[^>]*id="fm1"[^>]*>(.*?)</form>', lp.text, re.S)
    if not blk:
        return "no-form"
    fields = dict(re.findall(r'<input[^>]*name="([^"]+)"[^>]*value="([^"]*)"', blk.group(1)))
    for n in re.findall(r'<input[^>]*name="([^"]+)"', blk.group(1)):
        fields.setdefault(n, "")
    fields.update({"username": user, "password": pw})
    r = s.post(urllib.parse.urljoin(lp.url, "login"), data=fields,
               headers={**UA, "Referer": lp.url}, timeout=120)
    if "login" not in r.url:
        return "LOGIN OK"
    t = text_of(r.text)
    if "missing privileges" in t.lower():
        return "auth ok, NO PERMISSION"
    if "authentication attempt has failed" in t.lower():
        return "wrong credentials"
    m = re.search(r"(Authentication Failure[^.]{0,80})", t, re.I)
    return (m.group(1)[:60] if m else "unknown 401")


def codede(user, pw):
    s = requests.Session()
    lp = s.get(href, headers=UA, timeout=120)
    forms = []
    for m in re.finditer(r'<form[^>]*id="([^"]*)"[^>]*action="([^"]*)"(.*?)</form>',
                         lp.text, re.S | re.I):
        f = dict(re.findall(r'<input[^>]*name="([^"]+)"[^>]*value="([^"]*)"', m.group(3)))
        forms.append({"id": m.group(1), "action": m.group(2), "fields": f})
    delg = next((f for f in forms if f["fields"].get("client_name") == "codede3_enmap-dl"), None)
    if not delg:
        return "no CODE-DE entry"
    r1 = s.post(urllib.parse.urljoin(lp.url, delg["action"]), data=delg["fields"],
                headers={**UA, "Referer": lp.url}, timeout=120)
    kc = None
    for m in re.finditer(r'<form[^>]*id="kc-form-login"[^>]*action="([^"]*)"(.*?)</form>',
                         r1.text, re.S):
        kc = {"action": m.group(1).replace("&amp;", "&"),
              "fields": dict(re.findall(r'<input[^>]*name="([^"]+)"[^>]*value="([^"]*)"', m.group(2)))}
    if kc is None:
        return "no Keycloak form"
    d = dict(kc["fields"])
    d.update({"username": user, "password": pw, "login": "Sign In"})
    r2 = s.post(urllib.parse.urljoin(r1.url, kc["action"]), data=d,
                headers={**UA, "Referer": r1.url}, timeout=120)
    if "kc-otp-login-form" in r2.text:
        return "password ok, OTP required"
    t = text_of(r2.text)
    if "invalid username or password" in t.lower():
        return "wrong credentials"
    return "unknown state"


print(f"{'account':26s} {'EOC':28s} {'CODE-DE'}")
for user, pw in ACCOUNTS:
    print(f"{user:26s} {eoc(user, pw):28s} {codede(user, pw)}")
