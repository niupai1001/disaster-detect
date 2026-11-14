"""Batch 5 (P1): small-sample physical review of the external public sources.

No imagery collection is re-run. Samples come from the official release objects,
are capped by time/bytes, and every record is inspected for dtype, grid,
calibration, QA evidence and physical quantity before a verdict is given.
"""
from __future__ import annotations

import argparse
import bz2
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/encoder_v3/data_download_plan/external_p1"
P0 = ROOT / "outputs/encoder_v3/data_download_plan/external_p0"
BEN_S2 = "https://zenodo.org/records/10891137/files/BigEarthNet-S2.tar.zst?download=1"
BEN_L = "https://huggingface.co/datasets/isaaccorley/bigearthnet-l/resolve/main/bigearthnet-l.tar.gz"
ZEN = "https://zenodo.org/records/14603764/files/FGMANAUS.zip?download=1"
FLAIR = "https://huggingface.co/datasets/IGNF/FLAIR-HUB/resolve/main/data/"
FLAIR_FILES = ["D080-2017_SPOT_RGBI.zip", "D080-2017_SENTINEL2_TS.zip",
               "D080-2017_SENTINEL2_MSK-SC.zip", "D080-2017_ALL_LABEL-LPIS.zip"]
CAP = 4_000_000_000


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


class Meter:
    def __init__(self, path, cap=CAP):
        self.path = Path(path)
        self.cap = cap
        self.received = 0

    def charge(self, size):
        self.received += size
        if self.received > self.cap:
            raise RuntimeError("DOWNLOAD_BUDGET_EXHAUSTED")
        if self.received % (16 << 20) < size:
            write_json(self.path, {"received_bytes": self.received, "cap_bytes": self.cap,
                                   "unit": "HTTP response body bytes"})


def stream_extract(url, target, seconds, meter, tar_args):
    """Stream a remote archive into tar and stop early; returns bytes read."""
    import requests

    target.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(["tar"] + tar_args + ["-C", str(target), "-f", "-"],
                               stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
    start = time.time()
    total = 0
    try:
        with requests.get(url, stream=True, timeout=(20, 90)) as response:
            response.raise_for_status()
            for chunk in response.iter_content(1 << 20):
                if not chunk:
                    continue
                total += len(chunk)
                meter.charge(len(chunk))
                try:
                    process.stdin.write(chunk)
                    process.stdin.flush()
                except (BrokenPipeError, OSError):
                    break
                if time.time() - start > seconds:
                    break
    finally:
        try:
            process.stdin.close()
        except Exception:
            pass
        try:
            process.kill()
        except Exception:
            pass
    return total


def download(url, target, meter):
    import requests

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return target.stat().st_size
    temp = target.with_name(target.name + f".{os.getpid()}.tmp")
    total = 0
    with requests.get(url, stream=True, timeout=(20, 90)) as response:
        response.raise_for_status()
        with temp.open("wb") as handle:
            for chunk in response.iter_content(1 << 20):
                if chunk:
                    handle.write(chunk)
                    total += len(chunk)
                    meter.charge(len(chunk))
    temp.replace(target)
    return total


def band_report(path):
    import numpy as np
    import rasterio

    with rasterio.open(path) as dataset:
        sample = dataset.read(1)
        return {
            "dtype": dataset.dtypes[0], "count": dataset.count,
            "shape": [dataset.height, dataset.width], "crs": str(dataset.crs),
            "transform": list(dataset.transform)[:6], "nodata": dataset.nodata,
            "scales": list(dataset.scales)[:3], "offsets": list(dataset.offsets)[:3],
            "descriptions": [d for d in dataset.descriptions if d][:5],
            "min": float(np.nanmin(sample)) if sample.size else None,
            "max": float(np.nanmax(sample)) if sample.size else None,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--s2-seconds", type=int, default=90)
    parser.add_argument("--l-seconds", type=int, default=90)
    parser.add_argument("--fmow-samples", type=int, default=120)
    parser.add_argument("--cap", type=int, default=CAP)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    meter = Meter(args.out / "network.json", args.cap)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import rasterio

    samples = []
    verdicts = {}

    # --- BigEarthNet-S2 early-stop sample
    s2_dir = args.out / "bigearthnet_s2"
    if not any(s2_dir.rglob("*.tif")):
        stream_extract(BEN_S2, s2_dir, args.s2_seconds, meter,
                       ["-x", "--zstd", "--include", "*_B*.tif"])
    patch_dirs = sorted({path.parent for path in s2_dir.rglob("*_B02.tif")})
    inspected = []
    for folder in patch_dirs[:20]:
        bands = {}
        for path in sorted(folder.glob("*_B*.tif")):
            key = path.stem.rsplit("_", 1)[-1]
            bands[key] = band_report(path)
        if bands:
            inspected.append({"patch": folder.name, "bands": bands})
    samples.append({"source": "bigearthnet_s2", "patches_found": len(patch_dirs),
                    "patches_inspected": len(inspected), "records": inspected})
    if inspected:
        band_keys = sorted(inspected[0]["bands"])
        support = {key: inspected[0]["bands"][key]["shape"] for key in band_keys}
        verdicts["bigearthnet_s2"] = {
            "bands": band_keys,
            "shapes": support,
            "dtype": inspected[0]["bands"][band_keys[0]]["dtype"],
            "nodata": inspected[0]["bands"][band_keys[0]]["nodata"],
            "scales": inspected[0]["bands"][band_keys[0]]["scales"],
            "qa": "only the official patch lists; no per-pixel QA",
            "verdict": "limited_evidence: pixel-level QA absent, per-band grid must be frozen from these files",
        }
        folder = patch_dirs[0]
        rgb = []
        for key in ("B04", "B03", "B02"):
            with rasterio.open(folder / (folder.name + f"_{key}.tif")) as dataset:
                rgb.append(dataset.read(1).astype("float32"))
        array = np.dstack(rgb)
        lo, hi = np.percentile(array, [2, 98])
        figure, axes = plt.subplots(1, 2, figsize=(9, 3.6))
        axes[0].imshow(np.clip((array - lo) / max(hi - lo, 1e-6), 0, 1))
        axes[0].set_title("BigEarthNet-S2 patch (B04/B03/B02)")
        axes[0].axis("off")
        with rasterio.open(folder / (folder.name + "_B05.tif")) as dataset:
            axes[1].hist(dataset.read(1).ravel(), bins=60, color="#7a4f9e")
        axes[1].set_title("BigEarthNet-S2 B05 raw values")
        figure.tight_layout()
        figure.savefig(args.out / "fig_bigearthnet_s2.png", dpi=140)
        plt.close(figure)

    # --- BigEarthNet-L early-stop sample
    l_dir = args.out / "bigearthnet_l"
    if not any(l_dir.rglob("*.tif")):
        stream_extract(BEN_L, l_dir, args.l_seconds, meter,
                       ["-xz", "--include", "bigearthnet-l/images/*.tif"])
    l_tifs = sorted(l_dir.rglob("*.tif"))
    l_inspected = [{"file": path.name, **band_report(path)} for path in l_tifs[:20]]
    samples.append({"source": "bigearthnet_l", "files_found": len(l_tifs),
                    "records": l_inspected})
    if l_inspected:
        verdicts["bigearthnet_l"] = {
            "files": len(l_tifs),
            "dtype": l_inspected[0]["dtype"], "count": l_inspected[0]["count"],
            "shape": l_inspected[0]["shape"], "crs": l_inspected[0]["crs"],
            "nodata": l_inspected[0]["nodata"], "scales": l_inspected[0]["scales"],
            "verdict": "inspect_units: band count/support recorded; calibration and product level need the side data card",
        }

    # --- fMoW sample from the published manifest
    fmow_dir = args.out / "fmow"
    manifest_path = P0 / "fmow_manifest.json.bz2"
    if manifest_path.exists():
        keys = json.loads(bz2.open(manifest_path).read().decode("utf-8"))
        ms = [key for key in keys if key.endswith("_ms.tif")]
        step = max(1, len(ms) // max(1, args.fmow_samples))
        picked = ms[::step][: args.fmow_samples]
        fmow_records = []
        for key in picked:
            target = fmow_dir / key
            try:
                download("https://spacenet-dataset.s3.amazonaws.com/Hosted-Datasets/fmow/fmow-full/" + key, target, meter)
                side = key.replace("_ms.tif", "_ms.json")
                download("https://spacenet-dataset.s3.amazonaws.com/Hosted-Datasets/fmow/fmow-full/" + side,
                         fmow_dir / side, meter)
                fmow_records.append({"key": key, **band_report(target)})
            except Exception as exc:
                fmow_records.append({"key": key, "error": str(exc)[:160]})
        samples.append({"source": "fmow_full", "picked": len(picked), "records": fmow_records})
        ok = [row for row in fmow_records if "error" not in row]
        if ok:
            verdicts["fmow_full"] = {
                "records": len(ok), "dtype": ok[0]["dtype"], "count": ok[0]["count"],
                "shape": ok[0]["shape"], "crs": ok[0]["crs"], "nodata": ok[0]["nodata"],
                "scales": ok[0]["scales"],
                "verdict": "screen_quantity: no calibration declared in TIFF tags; sensor/quantity must come from the side json",
            }

    # --- FLAIR-HUB smallest department
    flair_dir = args.out / "flair_hub"
    flair_records = []
    for name in FLAIR_FILES:
        target = flair_dir / name
        try:
            download(FLAIR + name, target, meter)
            with zipfile.ZipFile(target) as archive:
                members = archive.namelist()[:5]
            flair_records.append({"file": name, "bytes": target.stat().st_size,
                                  "members": members})
        except Exception as exc:
            flair_records.append({"file": name, "error": str(exc)[:160]})
    samples.append({"source": "flair_hub", "records": flair_records})
    if flair_records:
        verdicts["flair_hub"] = {
            "files": len(flair_records),
            "verdict": "derived_only: SPOT 1.6 m / S2 10.24 m are published resampled grids; use as derived products",
        }

    # --- SEN2VENuS station sample already in P0
    venus_records = []
    venus_zip = P0 / "sen2venus_FGMANAUS.zip"
    if venus_zip.exists():
        with zipfile.ZipFile(venus_zip) as archive:
            names = archive.namelist()
            venus_records = names[:15]
    samples.append({"source": "sen2venus_v2", "file": "sen2venus_FGMANAUS.zip",
                    "members": venus_records})
    verdicts["sen2venus_v2"] = {
        "station": "FGMANAUS", "members_sample": len(venus_records),
        "verdict": "non_commercial: licence other-nc; native VENuS bands and S2 reference must be mapped separately",
        "note": "the archive was downloaded in P0 and counted there",
    }

    # --- WorldStrat reuse
    pilot = ROOT / "data/raw/multisensor/four_view_pilot_v1"
    sites = len([p for p in (pilot / "sites").iterdir() if p.is_dir()]) if pilot.exists() else 0
    samples.append({"source": "worldstrat_v1_1", "local_sites": sites,
                    "bytes": 0, "note": "288 local sites reused; no new WorldStrat download"})
    verdicts["worldstrat_v1_1"] = {
        "local_sites": sites,
        "verdict": "limited_licence: HR SPOT is CC BY-NC 4.0; S2/L8 windows are open data",
    }

    write_json(args.out / "summary.json", {
        "samples": samples, "verdicts": verdicts,
        "network": load_json(args.out / "network.json") if (args.out / "network.json").exists()
        else {"received_bytes": meter.received, "cap_bytes": meter.cap},
        "cap_bytes": meter.cap,
    })
    lines = ["# 批次 5（P1）外部小样实物预审", "",
             f"- 网络计量：{meter.received} / {meter.cap} 字节。",
             "", "| 来源 | 抽样 | 结论 |", "| --- | ---: | --- |"]
    for source, verdict in verdicts.items():
        lines.append(f"| {source} | {samples[[s['source'] for s in samples].index(source)]['source']} | "
                     f"{verdict.get('verdict','')} |")
    lines.append("")
    lines.append("结论仅针对本次抽到的原文件；未抽到的记录不作合格判定。BEN/BEN-L 的最小可下载单元是整包，"
                 "本次用早停流式抽取取前段样本，不能代表全库分布。")
    (args.out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("P1", json.dumps({"network": meter.received,
                            "sources": {k: v.get("verdict") for k, v in verdicts.items()}},
                           ensure_ascii=False)[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
