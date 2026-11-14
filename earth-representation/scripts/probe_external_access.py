"""Access probes for GF-2, WorldView/GeoEye (SpaceNet), PlanetScope and SPOT.

Only listing and small examples are fetched. The goal is to record what can and
cannot be reached, with the licence and product facts needed for admission.
"""
from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/encoder_v3/data_download_plan/external_access"
S3 = "https://spacenet-dataset.s3.amazonaws.com/"
NS = "http://s3.amazonaws.com/doc/2006-03-01/"


def s3_list(prefix, max_keys=25):
    response = requests.get(S3, params={"list-type": "2", "prefix": prefix,
                                        "max-keys": max_keys}, timeout=60)
    response.raise_for_status()
    root = ET.fromstring(response.text)
    return [{"key": node.findtext("s:Key", namespaces={"s": NS}),
             "size": int(node.findtext("s:Size", namespaces={"s": NS}))}
            for node in root.findall("s:Contents", namespaces={"s": NS})]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    evidence = {}

    # --- GF-2: public Google Drive folders from the Five-Billion-Pixels page
    try:
        folder = "1924VnO08Gqo3Nv7Y4KirgJ9kqqCup7f0"
        page = requests.get(f"https://drive.google.com/embeddedfolderview?id={folder}#list",
                            timeout=60)
        names = re.findall(r'flip-entry-title">([^<]+)<', page.text)
        ids = re.findall(r'flip-entry-id="([^"]+)"', page.text)
        evidence["gf2_drive"] = {
            "status": "reachable" if page.ok else f"http_{page.status_code}",
            "folder": f"https://drive.google.com/drive/folders/{folder}",
            "entries_listed": len(names), "example_entries": list(zip(names, ids))[:10],
            "note": "public folder listing via embeddedfolderview; file download needs the per-file confirm token",
        }
    except Exception as exc:
        evidence["gf2_drive"] = {"status": "error", "error": str(exc)[:200]}

    # --- WorldView/GeoEye: SpaceNet public bucket
    try:
        listing = s3_list("spacenet/SN3_roads/")
        evidence["spacenet_sn3"] = {"status": "reachable", "prefix": "spacenet/SN3_roads/",
                                    "entries": listing[:15]}
    except Exception as exc:
        evidence["spacenet_sn3"] = {"status": "error", "error": str(exc)[:200]}
    for prefix in ("spacenet/SN2_buildings/", "spacenet/SN5_roads/"):
        try:
            listing = s3_list(prefix, 10)
            evidence[prefix.strip("/").replace("/", "_")] = {
                "status": "reachable", "entries": listing}
        except Exception as exc:
            evidence[prefix.strip("/").replace("/", "_")] = {
                "status": "error", "error": str(exc)[:200]}

    # --- PlanetScope: check the public STAC catalogue
    try:
        collections = requests.get("https://planetarycomputer.microsoft.com/api/stac/v1/collections",
                                   timeout=60).json()
        ids = [item["id"] for item in collections.get("collections", [])]
        planet = [item for item in ids if "planet" in item.lower() or "scope" in item.lower()]
        evidence["planetscope_catalogue"] = {
            "status": "checked", "planet_like_collections": planet,
            "note": "no public PlanetScope collection in the catalogue; PlanetScope requires an account/licence",
        }
    except Exception as exc:
        evidence["planetscope_catalogue"] = {"status": "error", "error": str(exc)[:200]}

    # --- SPOT: local evidence already gathered
    evidence["spot"] = {
        "worldstrat_raw": "local pilot sample is uint16 DN (405-7028) without calibration; CC BY-NC",
        "flair_hub": "SPOT RGBI published on a 1.6 m resampled grid; derived only",
        "verdict": "no CORE_NATIVE SPOT source without calibration coefficients",
    }

    (args.out / "access_evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("ACCESS", json.dumps({key: value.get("status") for key, value in evidence.items()},
                               ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
