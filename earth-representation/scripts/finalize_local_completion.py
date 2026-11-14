"""Batch 3: freeze the local completion candidate without touching the frozen release."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "observation_model/outputs/encoder_v2/data_expansion"
DEFAULT_A = ROOT / "outputs/observation_model/encoder_v2/data_expansion_v2"
DEFAULT_B = ROOT / "outputs/observation_model/encoder_v2/data_expansion_v2_landsat"
DEFAULT_C = ROOT / "outputs/observation_model/encoder_v2/data_expansion_v2_heldout"
DEFAULT_OUT = ROOT / "outputs/encoder_v3/data_download_plan/local_completion_release_v1"
EXPECTED_FROZEN_MANIFEST = "b3c16c5b4a23e9454b20c32579203a3abc854430733bf2b11fccb7f533e0943c"
QA_POLICY = {
    "s2_bad_scl_classes": [0, 1, 3, 8, 9, 10, 11, 255],
    "landsat_qa_pixel_clear_bits": 63,
    "landsat_qa_radsat_required": 0,
    "min_valid_fraction_per_band": 0.5,
    "s2_20m_mask": "frozen S2 SCL mask downsampled to 20 m with min resampling",
    "landsat_sr7_mask": "frozen QA_PIXEL/QA_RADSAT mask reused at 30 m",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-a", type=Path, default=DEFAULT_A)
    parser.add_argument("--root-b", type=Path, default=DEFAULT_B)
    parser.add_argument("--root-c", type=Path, default=DEFAULT_C)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import numpy as np

    frozen_members = {row["id"]: row for row in lines(FROZEN / "members.jsonl")}
    frozen_manifest_sha = sha256(FROZEN / "members.jsonl")
    old_statistics = load_json(FROZEN / "statistics.json")

    completed = {}
    for side in sorted((args.root_a / "cache").glob("*.json")):
        member = load_json(side)
        if member.get("provenance") in {"s2_core10_completed"}:
            completed[member["id"]] = member
    upgraded = sorted(completed)
    missing = [sample_id for sample_id in upgraded if sample_id not in frozen_members]
    if missing:
        raise SystemExit(f"upgraded ids not in frozen release: {missing[:3]}")

    new_landsat = {}
    upgraded_l8 = {}
    for side in sorted((args.root_b / "cache").glob("*.json")):
        member = load_json(side)
        if member["id"] in frozen_members:
            if member.get("provenance") == "l8_sr7_completed":
                upgraded_l8[member["id"]] = member
        else:
            new_landsat[member["id"]] = member

    heldout = {}
    if args.root_c.exists():
        for side in sorted((args.root_c / "cache").glob("*.json")):
            row = load_json(side)
            if row["id"] not in frozen_members:
                heldout[row["id"]] = row

    # combined member list: frozen minus upgraded, plus upgraded, plus new Landsat, plus held-out
    combined = {sample_id: row for sample_id, row in frozen_members.items()
                if sample_id not in completed and sample_id not in upgraded_l8}
    combined.update(completed)
    combined.update(upgraded_l8)
    combined.update(new_landsat)
    combined.update(heldout)

    # Declare the QA source that the pipeline actually applied when the frozen side
    # record did not carry the label. Data and masks are untouched.
    def declare_qa(row):
        if row.get("qa_provenance") or row.get("QA_provenance"):
            return row
        provenance = row.get("provenance")
        sensor = str(row.get("sensor", "")).upper()
        if provenance == "new_native_Landsat_existing_local_anchor":
            label = "QA_PIXEL bits0-5 clear + QA_RADSAT==0 + aerosol_valid"
        elif sensor in {"L8", "L9"}:
            label = "QA_PIXEL bits0-5 clear + QA_RADSAT==0"
        else:
            label = "frozen_source_SCL_mask"
        row = dict(row)
        row["qa_provenance"] = label
        row["qa_provenance_declared_by"] = "pipeline_policy"
        return row

    ordered = [declare_qa(combined[sample_id]) for sample_id in sorted(combined)]
    members_path = args.out / "members.jsonl"
    with members_path.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def accum_from_stats(statistics):
        accum = {}
        for key, entry in statistics["bands"].items():
            n = int(entry["valid_pixels"])
            mean, std = float(entry["mean"]), float(entry["std"])
            accum[key] = [n, n * mean, n * (std ** 2 + mean ** 2)]
        return accum

    def add(accum, key, values):
        if values.size == 0:
            return
        mean = float(values.mean())
        accum.setdefault(key, [0, 0.0, 0.0])
        accum[key][0] += int(values.size)
        accum[key][1] += float(values.sum())
        accum[key][2] += float((values.astype("float64") ** 2).sum())

    def subtract(accum, key, values):
        if values.size == 0:
            return
        accum.setdefault(key, [0, 0.0, 0.0])
        accum[key][0] -= int(values.size)
        accum[key][1] -= float(values.sum())
        accum[key][2] -= float((values.astype("float64") ** 2).sum())

    def band_values(member, band):
        with np.load(member["cache"], allow_pickle=False) as stored:
            if band not in stored.files:
                return None
            return stored[band][stored[band + "__valid"]].astype("float64")

    accum = accum_from_stats(old_statistics)
    for sample_id in upgraded:
        old = frozen_members[sample_id]
        for band in [item["band_key"] for item in old["bands"]]:
            values = band_values(old, band)
            if values is not None:
                subtract(accum, f"{old['instrument']}:{band}", values)
    for member in completed.values():
        if member["role"] != "TRAIN":
            continue
        for band in [item["band_key"] for item in member["bands"]]:
            values = band_values(member, band)
            if values is not None:
                add(accum, f"{member['instrument']}:{band}", values)
    for sample_id, member in upgraded_l8.items():
        old = frozen_members[sample_id]
        for band in [item["band_key"] for item in old["bands"]]:
            values = band_values(old, band)
            if values is not None:
                subtract(accum, f"{old['instrument']}:{band}", values)
        if member["role"] != "TRAIN":
            continue
        for band in [item["band_key"] for item in member["bands"]]:
            values = band_values(member, band)
            if values is not None:
                add(accum, f"{member['instrument']}:{band}", values)
    for member in new_landsat.values():
        if member["role"] != "TRAIN":
            continue
        for band in [item["band_key"] for item in member["bands"]]:
            values = band_values(member, band)
            if values is not None:
                add(accum, f"{member['instrument']}:{band}", values)
    statistics = {}
    for key, (count, total, sumsq) in sorted(accum.items()):
        if count <= 0:
            continue
        mean = total / count
        variance = max(0.0, sumsq / count - mean ** 2)
        statistics[key] = {"valid_pixels": count, "mean": mean, "std": variance ** 0.5}
    write_json(args.out / "train_statistics.json", {
        "scope": "TRAIN-only physical reflectance by instrument and band; frozen statistics merged with the completion delta",
        "bands": statistics,
    })

    counts = Counter()
    for row in ordered:
        counts[f"{row['role']}/{row['sensor']}"] += 1
    band_counts = Counter(str(row["band_count"]) for row in ordered)
    provenance = Counter(row.get("provenance") for row in ordered)
    groups = defaultdict(set)
    group_roles = defaultdict(set)
    for row in ordered:
        groups[row["isolation_group"]].add(row["id"])
        group_roles[row["isolation_group"]].add(row["role"])
    max_group = max((len(values) for values in groups.values()), default=0)
    val_by_sensor = Counter(row["sensor"] for row in ordered if row["role"] == "VAL")
    test_by_sensor = Counter(row["sensor"] for row in ordered if row["role"] == "TEST")
    split_audit = {
        "records": len(ordered),
        "counts": dict(counts),
        "band_counts": dict(band_counts),
        "provenance": dict(provenance),
        "groups": len(groups),
        "max_records_per_group": max_group,
        "groups_spanning_roles": sum(1 for values in group_roles.values() if len(values) > 1),
        "val_by_sensor": dict(val_by_sensor),
        "test_by_sensor": dict(test_by_sensor),
        "test_records": sum(test_by_sensor.values()),
        "rule": "roles keep the frozen assignment; no TEST record was manufactured from TRAIN geography",
    }
    write_json(args.out / "split_audit.json", split_audit)

    remaining_four_band = sum(1 for row in ordered if row["sensor"] == "S2" and row["band_count"] == 4)
    remaining_six_band_l8 = sum(1 for row in ordered if row["sensor"] == "L8" and row["band_count"] == 6)
    gaps = {
        "s2_four_band_remaining": remaining_four_band,
        "l8_six_band_remaining": remaining_six_band_l8,
        "test_records": sum(test_by_sensor.values()),
        "val_shortfall": {"L8": max(0, 100 - val_by_sensor.get("L8", 0)),
                          "L9": max(0, 100 - val_by_sensor.get("L9", 0))},
        "enmap_in_release": sum(1 for row in ordered if "ENMAP" in row["sensor"].upper()),
        "external_sources": "BEN/BEN-L/WorldStrat/fMoW/VENµS/GF-2 remain plan-only; see DATA_DOWNLOAD_EXECUTION_PLAN.md",
    }
    write_json(args.out / "gaps.json", gaps)

    qa_digest = hashlib.sha256(json.dumps(QA_POLICY, sort_keys=True).encode()).hexdigest()
    release = {
        "status": "local_completion_candidate",
        "records": len(ordered),
        "upgraded_s2_core10": len(completed),
        "new_landsat_records": len(new_landsat), "new_heldout_records": len(heldout),
        "frozen_release_manifest_sha256": frozen_manifest_sha,
        "frozen_release_unchanged": frozen_manifest_sha == EXPECTED_FROZEN_MANIFEST,
        "manifest_sha256": sha256(members_path),
        "statistics_sha256": sha256(args.out / "train_statistics.json"),
        "split_audit_sha256": sha256(args.out / "split_audit.json"),
        "gaps_sha256": sha256(args.out / "gaps.json"),
        "qa_policy_sha256": qa_digest,
        "training_authorized": False,
        "model_training_started": False,
        "old_assets_deleted": False,
        "generated_from": {"root_a": str(args.root_a), "root_b": str(args.root_b)},
    }
    write_json(args.out / "release.json", release)

    report = f"""# 本机补全候选发布 v1

- 记录总数：{len(ordered)}；S2 完成十波段：{len(completed)}；新增 Landsat：{len(new_landsat)}；新增留出（TEST/VAL）：{len(heldout)}。
- 冻结发布 `members.jsonl` SHA256：`{frozen_manifest_sha}`，与预期一致：{release['frozen_release_unchanged']}；旧发布未被改写。
- 角色/传感器计数：{json.dumps(dict(counts), ensure_ascii=False)}。
- 波段数分布：{json.dumps(dict(band_counts), ensure_ascii=False)}。
- 组数与单组上限：{split_audit['groups']} 组，单组最多 {max_group} 条；跨角色组 {split_audit['groups_spanning_roles']} 个。
- TEST 记录：{split_audit['test_records']}；VAL 缺口：{json.dumps(gaps['val_shortfall'])}。
- S2 仍为四波段：{remaining_four_band} 条；L8 仍为六波段：{remaining_six_band_l8} 条。
- QA 政策哈希：`{qa_digest}`；统计与审计哈希见 `release.json`。
- 本发布仅为本机候选，未授权训练；外部来源仍在执行计划中。
"""
    (args.out / "REPORT.md").write_text(report, encoding="utf-8")
    print("RELEASE", json.dumps({"records": len(ordered), "upgraded": len(completed),
                                 "new_landsat": len(new_landsat),
                                 "frozen_unchanged": release["frozen_release_unchanged"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
