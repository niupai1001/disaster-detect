"""Batch 6 (P2): download the selected external releases (no imagery re-collection).

The 2026-11-14 decision relaxed the P2 envelope to 40e9 bytes and dropped the
equal 5 GB split. Only sources that survived the P1 screen are fetched:
WorldStrat raw SPOT archive and the smaller SEN2VENuS stations. BEN/BEN-L are
deliberately skipped because the local S2/L8 pools already cover those
instruments; fMoW is rejected (no CRS/affine) and GF-2 is blocked.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/encoder_v3/data_download_plan/external_p2"
WORLDSTRAT = "https://zenodo.org/records/15382551/files/hr_dataset_raw.zip?download=1"
VENUS = {
    "SO2": "https://zenodo.org/records/14603764/files/SO2.zip?download=1",
    "ESTUAMAR": "https://zenodo.org/records/14603764/files/ESTUAMAR.zip?download=1",
    "SUDOUE-4": "https://zenodo.org/records/14603764/files/SUDOUE-4.zip?download=1",
    "K34-AMAZ": "https://zenodo.org/records/14603764/files/K34-AMAZ.zip?download=1",
}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def download(url, target, cap, ledger):
    import requests

    target.parent.mkdir(parents=True, exist_ok=True)
    have = target.stat().st_size if target.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, headers=headers, stream=True, timeout=(20, 120)) as response:
        if have and response.status_code != 206:
            have = 0
            target.unlink(missing_ok=True)
        response.raise_for_status()
        mode = "ab" if have else "wb"
        with target.open(mode) as handle:
            for chunk in response.iter_content(4 << 20):
                if not chunk:
                    continue
                ledger["received"] += len(chunk)
                if ledger["received"] > cap:
                    raise RuntimeError("P2_CAP_EXCEEDED")
                handle.write(chunk)
                if ledger["received"] % (256 << 20) < len(chunk):
                    write_json(args.out / "network.json",
                               {"received_bytes": ledger["received"], "cap_bytes": args.cap,
                                "unit": "HTTP response body bytes"})
    return target.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--cap", type=int, default=40_000_000_000)
    parser.add_argument("--skip-worldstrat", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ledger = load_json(args.out / "network.json") if (args.out / "network.json").exists() else {}
    ledger.setdefault("received", ledger.get("received_bytes", 0))
    ledger.pop("received_bytes", None)
    ledger["cap_bytes"] = args.cap
    records = []

    def note(source, name, url, status, size=None, error=None):
        records.append({"source": source, "file": name, "url": url, "status": status,
                        "bytes": size, "error": error,
                        "at": time.strftime("%Y-%m-%dT%H:%M:%S")})

    if not args.skip_worldstrat:
        target = args.out / "worldstrat" / "hr_dataset_raw.zip"
        try:
            size = download(WORLDSTRAT, target, args.cap, ledger)
            note("worldstrat_v1_1", target.name, WORLDSTRAT, "ok", size)
        except Exception as exc:
            note("worldstrat_v1_1", target.name, WORLDSTRAT, "failed", error=str(exc)[:200])
    for name, url in VENUS.items():
        target = args.out / "venus" / (name + ".zip")
        try:
            size = download(url, target, args.cap, ledger)
            note("sen2venus_v2", target.name, url, "ok", size)
        except Exception as exc:
            note("sen2venus_v2", target.name, url, "failed", error=str(exc)[:200])
    ledger = {"received_bytes": ledger["received"], "cap_bytes": args.cap,
              "unit": "HTTP response body bytes"}
    write_json(args.out / "network.json", ledger)
    write_json(args.out / "summary.json", {"records": records, "network": ledger,
                                           "skipped": {"BEN_S2": "local S2 already complete",
                                                       "BEN_L": "local L8 already complete",
                                                       "fMoW": "no CRS/affine (P1 Q05)",
                                                       "GF2": "blocked"}})
    with (args.out / "downloads.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("P2", json.dumps({"records": records, "network": ledger}, ensure_ascii=False)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
