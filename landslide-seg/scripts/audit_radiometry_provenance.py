"""核对下载来源、分卷抽样字节身份与整套时间缓存数值范围；不修改数据。"""
from __future__ import annotations
import csv
import hashlib
import json
import sys
import tarfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from landslide_seg.config import file_sha256
from landslide_seg.data.segformer_dataset import S2_10_BANDS


def main():
    download = ROOT / "data/sen12/download_summary.json"
    meta = json.loads(download.read_text(encoding="utf-8"))
    if meta.get("repo") != "paulhoehn/Sen12Landslides" or meta.get("prefix") != "data_harmonized/s2":
        raise ValueError("下载记录不支持当前harmonized来源声明")
    with (ROOT / "docs/assets/research/temporal_observations.csv").open(encoding="utf-8-sig", newline="") as f:
        hashes = {r["sample_id"]: r["source_sha256"] for r in csv.DictReader(f)}
    report = {"download_summary_sha256": file_sha256(download),
              "download_repo": meta["repo"], "download_prefix": meta["prefix"],
              "release_description": "https://huggingface.co/datasets/paulhoehn/Sen12Landslides/blob/main/README.md",
              "download_script_sha256": file_sha256(ROOT / "scripts/download_sen12_parts.py"),
              "upstream_helper_sha256": file_sha256(ROOT / "vendor/official/Sen12Landslides/src/utils/helpers.py"),
              "archive_samples": [], "errors": [],
              "limits": ["每卷只抽样一个当前成员，不声称全部nc均与归档逐字节核验",
                         "没有远端固定revision与每卷期望哈希，不声称远端完整校验闭环",
                         "上游helper按首日期决定offset且仅识别数字后缀波段，不覆盖B8A；不能由helper证明发布文件逐帧偏移正确",
                         "发布说明支持harmonized DN/10000解释；尚未独立逐像素核验BOA，不再重复做日期偏移校正"]}
    for archive in sorted((ROOT / "data/sen12").glob("s2_part*.tar.gz")):
        sample = {"archive": archive.name, "sha256": file_sha256(archive)}
        with tarfile.open(archive, "r|gz") as tf:
            for member in tf:
                sid = Path(member.name).stem
                if not member.isfile() or sid not in hashes:
                    continue
                with tf.extractfile(member) as src:
                    digest = hashlib.file_digest(src, "sha256").hexdigest()
                sample.update(sample_id=sid, nc_sha256=digest, matches_source=digest == hashes[sid])
                if digest != hashes[sid]:
                    report["errors"].append(f"{sid} differs from {archive.name}")
                break
        report["archive_samples"].append(sample)
        print(archive.name, sample.get("sample_id", "no current pool member"), flush=True)
    x = np.load(ROOT / "data/s2_temporal_v4r1/X15.npy", mmap_mode="r")
    if x.shape != (2892, 15, 10, 128, 128):
        raise ValueError(f"固定事件时间缓存形状不符: {x.shape}")
    lo, hi = np.full(10, np.inf), np.full(10, -np.inf)
    for start in range(0, len(x), 16):
        batch = np.array(x[start:start+16])
        lo = np.minimum(lo, batch.min(axis=(0, 1, 3, 4)))
        hi = np.maximum(hi, batch.max(axis=(0, 1, 3, 4)))
    report["cache_range"] = {b: {"min": int(a), "max": int(c)} for b, a, c in zip(S2_10_BANDS, lo, hi)}
    report["cache_shape"] = list(x.shape)
    report["status"] = "SOURCE_SUPPORTED_NOT_INDEPENDENT_CALIBRATION" if not report["errors"] else "FAILED"
    out = ROOT / "docs/assets/research/radiometry_provenance.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(report["status"], report["cache_range"])
    raise SystemExit(bool(report["errors"]))


if __name__ == "__main__":
    main()
