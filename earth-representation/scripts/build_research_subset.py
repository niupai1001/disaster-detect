"""D3 research subset: 400/100/100 per profile with a hard <=10 records per group.

The subset is drawn from the screened release only (`pass_core`). Missing TEST
records stay missing and are reported, never manufactured from TRAIN geography.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "outputs/encoder_v3/data_download_plan/local_completion_release_v1/members.jsonl"
AUDIT = ROOT / "outputs/encoder_v3/data_download_plan/admission_audit_v1/pass_core.jsonl"
OUT = ROOT / "outputs/encoder_v3/data_download_plan/research_subset_v1"
SEED = 20261114
TARGETS = {"TRAIN": 400, "VAL": 100, "TEST": 100}
GROUP_CAP = 10
PROFILE_BANDS = {
    "S2_L2A_CORE10": (["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
                      {"10": ["B02", "B03", "B04", "B08"],
                       "20": ["B05", "B06", "B07", "B8A", "B11", "B12"]}),
    "L8_C2L2_SR7": (["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"], {"30": None}),
    "L9_C2L2_SR7": (["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"], {"30": None}),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def key(seed, *parts):
    return hashlib.sha256((str(seed) + ":" + ":".join(map(str, parts))).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=RELEASE)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import numpy as np

    passed = {json.loads(line)["id"] for line in args.audit.read_text(encoding="utf-8").splitlines()
              if line.strip()}
    members = [json.loads(line) for line in args.release.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    eligible = [row for row in members if row["id"] in passed]
    print("eligible", len(eligible), "of", len(members), flush=True)

    by_profile = defaultdict(lambda: defaultdict(list))
    for row in eligible:
        contract = row.get("contract") or ""
        if contract in PROFILE_BANDS:
            by_profile[contract][row["role"]].append(row)

    chosen = {}
    gaps = {}
    group_counts = Counter()
    for profile, roles in sorted(by_profile.items()):
        chosen[profile] = {"TRAIN": [], "VAL": [], "TEST": []}
        for role, target in TARGETS.items():
            pool = roles.get(role, [])
            groups = defaultdict(list)
            for row in pool:
                groups[row["isolation_group"]].append(row)
            order = sorted(groups, key=lambda group: key(SEED, profile, role, group))
            picked = []
            for group in order:
                if len(picked) >= target:
                    break
                rows = sorted(groups[group], key=lambda row: key(SEED, profile, role, row["id"]))
                take = min(GROUP_CAP, target - len(picked), len(rows))
                picked.extend(rows[:take])
                group_counts[(profile, role, group)] += take
            chosen[profile][role] = picked
            gaps[f"{profile}/{role}"] = {"target": target, "available": len(pool),
                                         "selected": len(picked),
                                         "gap": max(0, target - len(picked))}

    subset_path = args.out / "members.jsonl"
    rows_out = []
    for profile, roles in chosen.items():
        for role, rows in roles.items():
            for row in rows:
                rows_out.append(dict(row, subset_profile=profile, subset_role=role))
    rows_out.sort(key=lambda row: (row["subset_profile"], row["subset_role"], row["id"]))
    with subset_path.open("w", encoding="utf-8") as handle:
        for row in rows_out:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    for split, name in (("TRAIN", "train.jsonl"), ("VAL", "val.jsonl"), ("TEST", "test.jsonl")):
        with (args.out / name).open("w", encoding="utf-8") as handle:
            for row in rows_out:
                if row["subset_role"] == split:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    # TRAIN-only statistics over the selected records
    accum = {}
    group_pixels = Counter()
    for row in rows_out:
        if row["subset_role"] != "TRAIN":
            continue
        with np.load(row["cache"], allow_pickle=False) as stored:
            for band in [item["band_key"] for item in row["bands"]]:
                if band not in stored.files:
                    continue
                values = stored[band][stored[band + "__valid"]].astype("float64")
                if not values.size:
                    continue
                key_name = f"{row['instrument']}:{band}"
                entry = accum.setdefault(key_name, [0, 0.0, 0.0])
                entry[0] += int(values.size)
                entry[1] += float(values.sum())
                entry[2] += float((values ** 2).sum())
                group_pixels[row["isolation_group"]] += int(values.size)
    statistics = {}
    for name, (count, total, sumsq) in sorted(accum.items()):
        mean = total / count
        statistics[name] = {"valid_pixels": count, "mean": mean,
                            "std": max(0.0, sumsq / count - mean ** 2) ** 0.5}
    write_json(args.out / "train_statistics.json", {
        "scope": "TRAIN-only physical reflectance by instrument and band over the research subset; no VAL/TEST fit",
        "bands": statistics,
    })

    quality = []
    for row in rows_out:
        coverage = row.get("valid_fraction") or {}
        quality.append({
            "sample_id": row["id"], "profile": row["subset_profile"], "role": row["subset_role"],
            "admission": "pass_core", "qa_provenance": row.get("qa_provenance"),
            "min_band_valid_fraction": min(coverage.values()) if coverage else None,
            "group": row["isolation_group"], "product": row["product"], "date": row.get("date"),
        })
    with (args.out / "quality_records.jsonl").open("w", encoding="utf-8") as handle:
        for row in quality:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    split_counts = Counter((row["subset_profile"], row["subset_role"]) for row in rows_out)
    max_per_group = max(group_counts.values()) if group_counts else 0
    split_audit = {
        "records": len(rows_out),
        "counts": {f"{profile}/{role}": split_counts.get((profile, role), 0)
                   for profile in sorted(chosen) for role in TARGETS},
        "group_cap": GROUP_CAP, "max_records_per_group": max_per_group,
        "groups_used": len(group_counts),
        "test_rule": "TEST records only from pools already assigned TEST; no TRAIN geography promoted",
        "val_rule": "VAL records keep the frozen role; no re-split",
    }
    write_json(args.out / "split_audit.json", split_audit)
    write_json(args.out / "gaps.json", gaps)

    contracts = {}
    for profile, (bands, support) in PROFILE_BANDS.items():
        contracts[profile] = {"bands": bands, "native_support_m": support,
                              "quantity": "surface_reflectance",
                              "qa": "frozen SCL / QA_PIXEL+QA_RADSAT masks",
                              "subset_role_counts": {role: split_counts.get((profile, role), 0)
                                                     for role in TARGETS}}
    write_json(args.out / "profile_contracts.json", contracts)
    write_json(args.out / "source_catalog.json", {
        "sources": {
            "S2": {"instrument": "S2A/S2B/S2C", "product": "Sentinel-2 L2A (Earth Search / Planetary Computer)",
                   "licence": "Copernicus open data"},
            "L8": {"instrument": "LC08", "product": "USGS Landsat Collection 2 Level-2",
                   "licence": "USGS open data"},
            "L9": {"instrument": "LC09", "product": "USGS Landsat Collection 2 Level-2",
                   "licence": "USGS open data"},
        },
        "external": {"BEN/BEN-L": "P0 reviewed; not used in this subset",
                      "fMoW": "rejected (no CRS/affine)", "WorldStrat": "DN-only in pilot sample",
                      "VENuS": "non-commercial", "GF2": "blocked"},
    })

    release = {
        "status": "research_subset_candidate",
        "records": len(rows_out),
        "manifest_sha256": sha256(subset_path),
        "statistics_sha256": sha256(args.out / "train_statistics.json"),
        "split_audit_sha256": sha256(args.out / "split_audit.json"),
        "quality_sha256": sha256(args.out / "quality_records.jsonl"),
        "training_authorized": False,
        "model_training_started": False,
        "gaps": gaps,
    }
    write_json(args.out / "release.json", release)

    report = ["# 研究子集 v1（400/100/100，单组≤10）", "",
              f"- 记录总数：{len(rows_out)}；按 profile/role：{json.dumps(split_audit['counts'], ensure_ascii=False)}。",
              f"- 单组上限 {GROUP_CAP}，实际单组最多 {max_per_group} 条；使用组 {len(group_counts)} 个。",
              "", "| 缺口 | 目标 | 可用 | 已选 | 缺口 |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, row in gaps.items():
        report.append(f"| {name} | {row['target']} | {row['available']} | {row['selected']} | {row['gap']} |")
    report += ["", "只有 `members.jsonl` 中列出的记录可进入实验；VAL/TEST 未补足部分按缺口报告，"
               "没有把 TRAIN 地理改角色凑数。", "", "统计只由 TRAIN 拟合；VAL/TEST 未参与。"]
    (args.out / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("SUBSET", json.dumps({"records": len(rows_out), "counts": split_audit["counts"],
                                "max_per_group": max_per_group, "gaps": gaps}, ensure_ascii=False)[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
