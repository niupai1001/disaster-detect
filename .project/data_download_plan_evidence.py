"""Collect the verification evidence for the Encoder V3 data download plan.

Read-only: queries public APIs (Zenodo, Hugging Face, S3) and local frozen
artefacts, then writes one JSON file. Standard library only. It never downloads
imagery, never writes into data/, and never touches a frozen release.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "earth-representation"
PARENT = Path.home() / "Downloads" / "DATA_DOWNLOAD_AND_ADMISSION_PLAN.md"
OUT = ROOT / "outputs" / "encoder_v3" / "data_download_plan" / "plan_evidence_20261114.json"
TIMEOUT = 90
GB = 10**9
UA = {"User-Agent": "dsh-plan-evidence/1.0 (read-only)"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, method: str = "GET", params: dict | None = None):
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers=UA, method=method)
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return response.read(), dict(response.headers)


def fetch_json(url: str, params: dict | None = None):
    body, _ = fetch(url, params=params)
    return json.loads(body.decode("utf-8"))


def guarded(label: str, fn):
    try:
        return fn()
    except Exception as exc:  # recorded, never hidden
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}", "checked_at": now()}


def zenodo(record: str) -> dict:
    def collect():
        data = fetch_json(f"https://zenodo.org/api/records/{record}")
        files = [
            {"key": item["key"], "size_bytes": item["size"]}
            for item in sorted(data.get("files", []), key=lambda i: i["key"])
        ]
        meta = data.get("metadata", {})
        return {
            "status": "ok",
            "method": f"GET https://zenodo.org/api/records/{record}",
            "record_id": record,
            "title": meta.get("title"),
            "version": meta.get("version"),
            "license": (meta.get("license") or {}).get("id"),
            "file_count": len(files),
            "files": files,
            "checked_at": now(),
        }

    return guarded(f"zenodo:{record}", collect)


def hf(dataset: str) -> dict:
    def collect():
        meta = fetch_json(f"https://huggingface.co/api/datasets/{dataset}?full=true")
        tree = fetch_json(f"https://huggingface.co/api/datasets/{dataset}/tree/main?recursive=true")
        entries = [e for e in tree if e.get("type") == "file"]
        sizes = {e["path"]: e.get("size") for e in entries}
        interesting = sorted(
            (p for p in sizes if p.endswith((".zip", ".tar.gz", ".csv", ".md"))),
            key=lambda p: sizes[p] or 0,
        )
        departments = sorted({p.split("/")[-1].split("_")[0] for p in sizes if "/" in p})
        return {
            "status": "ok",
            "method": f"GET https://huggingface.co/api/datasets/{dataset} and /tree/main?recursive=true",
            "dataset": dataset,
            "gated": meta.get("gated"),
            "private": meta.get("private"),
            "last_modified": meta.get("lastModified"),
            "file_count": len(entries),
            "department_prefixes": len(departments),
            "files": [{"path": p, "size_bytes": sizes[p]} for p in interesting],
            "smallest": [
                {"path": p, "size_bytes": sizes[p]}
                for p in sorted(sizes, key=lambda p: sizes[p] or 0)[:10]
            ],
            "checked_at": now(),
        }

    return guarded(f"hf:{dataset}", collect)


def fmow() -> dict:
    def collect():
        body, _ = fetch(
            "https://spacenet-dataset.s3.amazonaws.com/",
            params={
                "list-type": "2",
                "prefix": "Hosted-Datasets/fmow/fmow-full/",
                "max-keys": "25",
            },
        )
        root = ET.fromstring(body)
        namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        entries = [
            {
                "key": node.findtext("s3:Key", namespaces=namespace),
                "size_bytes": int(node.findtext("s3:Size", namespaces=namespace)),
            }
            for node in root.findall("s3:Contents", namespaces=namespace)
        ]
        manifest_body, manifest_headers = fetch(
            "https://spacenet-dataset.s3.amazonaws.com/Hosted-Datasets/fmow/fmow-full/manifest.json.bz2",
            method="HEAD",
        )
        return {
            "status": "ok",
            "method": "anonymous S3 ListObjectsV2 + HEAD on spacenet-dataset",
            "root": "https://spacenet-dataset.s3.amazonaws.com/Hosted-Datasets/fmow/fmow-full/",
            "manifest_bytes": int(manifest_headers.get("Content-Length", 0)),
            "prefix_entries": entries,
            "checked_at": now(),
        }

    return guarded("s3:fmow-full", collect)


def five_billion_pixels() -> dict:
    def collect():
        body, _ = fetch("https://x-ytong.github.io/project/Five-Billion-Pixels.html")
        text = body.decode("utf-8", "replace")
        import re

        links = sorted(
            {
                m
                for m in re.findall(r'href="([^"]+)"', text)
                if "drive.google.com" in m or "pan.baidu.com" in m
            }
        )
        return {
            "status": "ok",
            "method": "GET project page and collect download links",
            "download_links": links,
            "note": "Only Google Drive folders and Baidu netdisk page; no per-object HTTP endpoint.",
            "checked_at": now(),
        }

    return guarded("web:five-billion-pixels", collect)


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def worldstrat_local() -> dict:
    def collect():
        paths = [
            ROOT / "data" / "raw" / "multisensor" / "four_view_pilot_v1",
            ROOT.parent / "landslide-seg" / "data" / "use_multisensor" / "four_view_pilot_v1",
        ]
        result: dict = {"copies": []}
        for path in paths:
            entry = {"path": str(path), "exists": path.exists()}
            if path.exists():
                manifest = path / "manifest.json"
                sites = path / "sites"
                entry["manifest_sha256"] = sha256(manifest)
                entry["manifest_bytes"] = manifest.stat().st_size
                entry["site_dirs"] = len([p for p in sites.iterdir() if p.is_dir()])
            result["copies"].append(entry)

        release = (
            ROOT.parent / "landslide-seg" / "outputs" / "use_data_pairing_plan" / "four_view_10gb_pilot"
        )
        for name in ("acceptance.json", "status.json", "network.json", "failures.json"):
            path = release / name
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if name == "acceptance.json":
                result["acceptance"] = {
                    "ok": payload.get("ok"),
                    "accepted_sites": payload.get("accepted_sites"),
                    "verified_raster_files": payload.get("verified_raster_files"),
                    "verified_raster_bytes": payload.get("verified_raster_bytes"),
                    "error_count": len(payload.get("errors") or []),
                    "site_list_sha256": hashlib.sha256(
                        "\n".join(payload.get("sites") or []).encode("utf-8")
                    ).hexdigest(),
                }
            elif name == "failures.json":
                result["failures"] = {"count": len(payload) if hasattr(payload, "__len__") else None}
            else:
                result[name.split(".")[0]] = payload

        sources = ROOT / "data" / "raw" / "multisensor" / "four_view_pilot_v1" / "sources"
        if (sources / "metadata.csv").exists():
            split_rows = read_csv_rows(sources / "stratified_train_val_test_split.csv")
            metadata_rows = read_csv_rows(sources / "metadata.csv")
            sites = [
                p.name
                for p in (ROOT / "data" / "raw" / "multisensor" / "four_view_pilot_v1" / "sites").iterdir()
                if p.is_dir()
            ]
            normalised = {s.replace("_", " ").strip(): s for s in sites}
            tiles = {row["tile"].strip() for row in split_rows}
            hit = [row for row in split_rows if row["tile"].strip() in normalised]
            pilot_rows = [row for row in metadata_rows if row["tile"] in normalised]
            clouds = sorted(float(row["cloud_cover"]) for row in pilot_rows)
            result["sources"] = {
                "metadata_bytes": (sources / "metadata.csv").stat().st_size,
                "metadata_rows": len(metadata_rows),
                "metadata_tiles": len({row["tile"] for row in metadata_rows}),
                "split_bytes": (sources / "stratified_train_val_test_split.csv").stat().st_size,
                "split_rows": len(split_rows),
                "split_tiles": len(tiles),
                "split_counts": dict(Counter(row["split"] for row in split_rows)),
                "split_by_source": {
                    f"{source}/{role}": count
                    for (source, role), count in Counter(
                        (row["source"], row["split"]) for row in split_rows
                    ).items()
                },
                "license_bytes": (sources / "LICENSE.txt").stat().st_size,
                "license_lines": [
                    line.strip()
                    for line in (sources / "LICENSE.txt").read_text(encoding="utf-8").splitlines()[:3]
                    if line.strip()
                ],
                "pilot_metadata_rows": len(pilot_rows),
                "pilot_cloud_cover": {
                    "count": len(clouds),
                    "median": round(clouds[len(clouds) // 2], 4) if clouds else None,
                    "p25": round(clouds[len(clouds) // 4], 4) if clouds else None,
                    "p75": round(clouds[3 * len(clouds) // 4], 4) if clouds else None,
                    "max": round(clouds[-1], 4) if clouds else None,
                },
            }
            result["pilot_split_coverage"] = {
                "mapped_tiles": len({row["tile"].strip() for row in hit}),
                "by_split": dict(Counter(row["split"] for row in hit)),
                "unmatched_tiles": sorted(set(normalised) - tiles),
            }
        return result

    return guarded("local:worldstrat", collect)


def expansion_local() -> dict:
    def collect():
        base = ROOT / "outputs" / "observation_model" / "encoder_v2" / "data_expansion"
        release = json.loads((base / "release.json").read_text(encoding="utf-8"))
        status = json.loads((base / "status.json").read_text(encoding="utf-8"))
        queue = json.loads((base / "queue.json").read_text(encoding="utf-8"))
        pending = json.loads((base / "pending_tasks.json").read_text(encoding="utf-8"))
        tasks = [
            json.loads(line)
            for line in (base / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        calibration = release.get("S2_calibration", {})
        return {
            "path": str(base),
            "members_jsonl_sha256": sha256(base / "members.jsonl"),
            "release": {
                "counts": release.get("counts"),
                "manifest_sha256": release.get("manifest_sha256"),
                "statistics_sha256": release.get("statistics_sha256"),
                "band_counts": release.get("band_counts"),
                "provenance_counts": release.get("provenance_counts"),
                "network": release.get("network"),
                "pending_tasks": release.get("pending_tasks"),
                "stage_status": release.get("stage_status"),
                "local_exclusion_reasons": release.get("local_exclusion_reasons"),
                "remote_status_counts": release.get("remote_status_counts"),
                "semantics": release.get("semantics"),
                "old_sources_unchanged": calibration.get("old_sources_unchanged"),
            },
            "calibration": {
                "status": calibration.get("status"),
                "counts": calibration.get("counts"),
                "source_rows": calibration.get("source_rows"),
                "network": calibration.get("network"),
            },
            "status": {
                "status": status.get("status"),
                "member_count": status.get("member_count"),
                "counts": status.get("counts"),
                "reason": status.get("reason"),
                "actual_process_exit_code": status.get("actual_process_exit_code"),
                "model_training_started": status.get("model_training_started"),
            },
            "queue": {"tasks": queue.get("tasks"), "kinds": queue.get("kinds")},
            "task_kinds": dict(Counter(task.get("kind") for task in tasks)),
            "pending_tasks": {
                "count": len(pending),
                "kinds": dict(Counter(task.get("kind") for task in pending)),
            },
            "cache_members": len(list((base / "cache").glob("*.json"))),
            "s2_calibrated_cache": len(list((base / "s2_calibrated_v2" / "cache").glob("*.json"))),
        }

    return guarded("local:expansion", collect)


def main() -> int:
    evidence = {
        "schema_version": 1,
        "generated_at": now(),
        "generator": ".project/data_download_plan_evidence.py",
        "purpose": "Evidence for outputs/documentation/encoder_v3/specifications/DATA_DOWNLOAD_EXECUTION_PLAN.md; read-only checks only.",
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "project_root": str(ROOT),
            "disk_free_bytes_N": shutil.disk_usage("N:\\").free,
            "tool_probe": {
                name: shutil.which(name)
                for name in ("zstd", "7z", "aws", "gdown", "rclone", "curl", "tar")
            },
        },
        "parent_plan": {
            "path": str(PARENT),
            "exists": PARENT.exists(),
            "bytes": PARENT.stat().st_size if PARENT.exists() else None,
            "sha256": sha256(PARENT) if PARENT.exists() else None,
            "last_write_utc": datetime.fromtimestamp(
                PARENT.stat().st_mtime, tz=timezone.utc
            ).isoformat(timespec="seconds")
            if PARENT.exists()
            else None,
        },
        "budget_definition": {
            "p2_total_bytes": 20 * GB,
            "bytes_unit": "1 GB = 1e9 bytes (decimal, same display convention as Zenodo and Hugging Face)",
            "per_class_default_bytes": 5 * GB,
            "per_class_fallback_bytes": round(20 * GB / 3),
            "local_completion_caps": {
                "batch_1_s2_bytes": 1_500_000_000,
                "batch_2_landsat_bytes": 500_000_000,
            },
            "external_metadata_cap_bytes": 400_000_000,
            "external_sample_cap_bytes": 4_000_000_000,
        },
        "sources": {},
        "local": {},
        "notes": [
            "No imagery downloaded by this script; sizes come from provider APIs or local files.",
            "Local completion (batches 1-3) is accounted separately from the 20 GB P2 cap.",
        ],
    }

    evidence["sources"]["zenodo_10891137_bigearthnet_v2"] = zenodo("10891137")
    evidence["sources"]["zenodo_15382551_worldstrat"] = zenodo("15382551")
    evidence["sources"]["zenodo_14603764_sen2venus"] = zenodo("14603764")
    evidence["sources"]["hf_bigearthnet_l"] = hf("isaaccorley/bigearthnet-l")
    evidence["sources"]["hf_flair_hub"] = hf("IGNF/FLAIR-HUB")
    evidence["sources"]["s3_fmow_full"] = fmow()
    evidence["sources"]["web_five_billion_pixels"] = five_billion_pixels()
    evidence["local"]["worldstrat"] = worldstrat_local()
    evidence["local"]["expansion_release"] = expansion_local()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print("WROTE", OUT)
    print("PARENT_SHA256", evidence["parent_plan"]["sha256"])
    for key, value in evidence["sources"].items():
        print("SOURCE", key, value.get("status"))
    for key, value in evidence["local"].items():
        print("LOCAL", key, value.get("status"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
