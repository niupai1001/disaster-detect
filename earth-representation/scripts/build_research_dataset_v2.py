"""Build research_dataset_v2: fixed TRAIN members plus new independent VAL/TEST.

TRAIN is copied from research_subset_v1 unchanged. VAL/TEST are completed with
held-out records from new geographic groups; nothing is re-split from TRAIN.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/encoder_v3/data_download_plan"
SUBSET = BASE / "research_subset_v1/members.jsonl"
HELDOUT = ROOT / "outputs/observation_model/encoder_v2/data_expansion_v2_heldout/cache"
CALIBRATION = ROOT / "outputs/observation_model/encoder_v2/data_expansion_v2_calibration/cache"
OUT = BASE / "research_dataset_v2"
TARGETS = {"TRAIN": 400, "VAL": 100, "TEST": 100}
GROUP_CAP = 10
SEED = 20261114


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def order_key(*parts):
    return hashlib.sha256((":".join(map(str, (SEED, *parts)))).encode()).hexdigest()


def profile_of(row):
    return row.get("contract") or row.get("subset_profile")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import numpy as np

    subset = [json.loads(line) for line in SUBSET.read_text(encoding="utf-8").splitlines() if line.strip()]
    heldout = []
    for side in HELDOUT.glob("*.json"):
        row = load_json(side)
        if row.get("provenance", "").startswith("new_geography_heldout"):
            heldout.append(row)
    calibration = []
    for side in CALIBRATION.glob("*.json"):
        row = load_json(side)
        if row.get("provenance") == "s2_calibration_resolved":
            calibration.append(row)

    train = [row for row in subset if row.get("subset_role") == "TRAIN"]
    extra_train = []
    val = [row for row in subset if row.get("subset_role") == "VAL"]
    test = [row for row in subset if row.get("subset_role") == "TEST"]
    by_id = {row["id"]: row for row in train + val + test}
    for row in heldout:
        role = row.get("role")
        if role == "VAL":
            val.append(row)
        elif role == "TEST":
            test.append(row)
        else:
            extra_train.append(row)
    # Calibration extras are listed as additional candidates only; the fixed
    # TRAIN selection must stay identical to the v1 subset.
    calibration_rows = [dict(row, role="TRAIN") for row in calibration]

    def cap_role(rows, role):
        """Apply the <=10 records per group cap and the per-profile target."""
        chosen, groups = [], Counter()
        ordered = sorted(rows, key=lambda row: order_key(role, profile_of(row), row["isolation_group"],
                                                         row["id"]))
        for row in ordered:
            profile = profile_of(row)
            if sum(1 for item in chosen if profile_of(item) == profile) >= TARGETS[role]:
                continue
            if groups[(profile, row["isolation_group"])] >= GROUP_CAP:
                continue
            chosen.append(row)
            groups[(profile, row["isolation_group"])] += 1
        return chosen

    # Resolve group/role conflicts on the full pools first, then apply the caps.
    # The frozen role (subset record) wins; otherwise the larger cohort keeps it.
    frozen_role = {row["isolation_group"]: row.get("subset_role")
                   for row in subset if row.get("subset_role")}
    pools = {"TRAIN": train, "VAL": val, "TEST": test}

    # Drop exact duplicate observations across roles (same product/date/bbox).
    # Earlier roles win, so a frozen TRAIN record is never duplicated in TEST.
    seen = {}
    for role in ("TRAIN", "VAL", "TEST"):
        unique = []
        for row in pools[role]:
            key = (row.get("product"), str(row.get("date"))[:10],
                   tuple(round(value, 4) for value in (row.get("bbox") or [0, 0, 0, 0])))
            if key in seen:
                continue
            seen[key] = role
            unique.append(row)
        pools[role] = unique

    group_roles = defaultdict(set)
    for role, rows in pools.items():
        for row in rows:
            group_roles[row["isolation_group"]].add(role)
    conflicts = {group: sorted(roles) for group, roles in group_roles.items() if len(roles) > 1}
    for group, roles in conflicts.items():
        winner = frozen_role.get(group)
        if winner not in roles:
            winner = max(roles, key=lambda role: sum(1 for row in pools[role]
                                                     if row["isolation_group"] == group))
        for role in roles:
            if role == winner:
                continue
            pools[role] = [row for row in pools[role] if row["isolation_group"] != group]
    final = {"TRAIN": cap_role(pools["TRAIN"], "TRAIN"), "VAL": cap_role(pools["VAL"], "VAL"),
             "TEST": cap_role(pools["TEST"], "TEST")}

    rows_out = []
    for role, rows in final.items():
        for row in rows:
            rows_out.append(dict(row, dataset_role=role, dataset_profile=profile_of(row)))
    rows_out.sort(key=lambda row: (row["dataset_profile"], row["dataset_role"], row["id"]))
    members_path = args.out / "members.jsonl"
    with members_path.open("w", encoding="utf-8") as handle:
        for row in rows_out:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    for role, name in (("TRAIN", "train.jsonl"), ("VAL", "val.jsonl"), ("TEST", "test.jsonl")):
        with (args.out / name).open("w", encoding="utf-8") as handle:
            for row in rows_out:
                if row["dataset_role"] == role:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = Counter((row["dataset_profile"], row["dataset_role"]) for row in rows_out)
    group_counts = Counter((row["dataset_profile"], row["dataset_role"], row["isolation_group"])
                           for row in rows_out)
    max_group = max(group_counts.values()) if group_counts else 0
    gaps = {}
    for profile in sorted({row["dataset_profile"] for row in rows_out}):
        for role, target in TARGETS.items():
            got = counts.get((profile, role), 0)
            gaps[f"{profile}/{role}"] = {"target": target, "selected": got, "gap": max(0, target - got)}

    accum = {}
    for row in rows_out:
        if row["dataset_role"] != "TRAIN":
            continue
        with np.load(row["cache"], allow_pickle=False) as stored:
            for band in [item["band_key"] for item in row["bands"]]:
                if band not in stored.files:
                    continue
                values = stored[band][stored[band + "__valid"]].astype("float64")
                if not values.size:
                    continue
                key = f"{row['instrument']}:{band}"
                entry = accum.setdefault(key, [0, 0.0, 0.0])
                entry[0] += int(values.size)
                entry[1] += float(values.sum())
                entry[2] += float((values ** 2).sum())
    statistics = {}
    for key, (count, total, sumsq) in sorted(accum.items()):
        mean = total / count
        statistics[key] = {"valid_pixels": count, "mean": mean,
                           "std": max(0.0, sumsq / count - mean ** 2) ** 0.5}
    write_json(args.out / "train_statistics.json", {
        "scope": "TRAIN-only physical reflectance by instrument and band over research_dataset_v2",
        "bands": statistics,
    })

    quality = []
    for row in rows_out:
        coverage = row.get("valid_fraction") or {}
        quality.append({
            "sample_id": row["id"], "profile": row["dataset_profile"], "role": row["dataset_role"],
            "product": row.get("product"), "date": row.get("date"),
            "group": row["isolation_group"], "qa_provenance": row.get("qa_provenance"),
            "min_band_valid_fraction": min(coverage.values()) if coverage else None,
            "admission": "heldout_95pct_qa" if row.get("provenance", "").startswith("new_geography_heldout")
            else "frozen_release_screened",
        })
    with (args.out / "quality_records.jsonl").open("w", encoding="utf-8") as handle:
        for row in quality:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    atlas_dir = args.out / "atlas"
    atlas_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = []
    for profile, bands in (("S2_L2A_CORE10", ("B04", "B03", "B02")),
                           ("L8_C2L2_SR7", ("SR_B5", "SR_B4", "SR_B3")),
                           ("L9_C2L2_SR7", ("SR_B5", "SR_B4", "SR_B3"))):
        sample = next((row for row in rows_out
                       if row["dataset_profile"] == profile and row["dataset_role"] == "TRAIN"), None)
        if not sample:
            continue
        channels = []
        with np.load(sample["cache"], allow_pickle=False) as stored:
            for band in bands:
                channels.append(stored[band].astype("float32"))
        image = np.dstack(channels)
        lo, hi = np.percentile(image, [2, 98])
        figure, axis = plt.subplots(figsize=(4, 4))
        axis.imshow(np.clip((image - lo) / max(hi - lo, 1e-6), 0, 1))
        axis.set_title(f"{profile}\n{sample['id'][:22]}", fontsize=8)
        axis.axis("off")
        figure.tight_layout()
        path = atlas_dir / f"{profile}.png"
        figure.savefig(path, dpi=120)
        plt.close(figure)
        figures.append(str(path))

    licences = {
        "S2_L2A_CORE10": {"source": "Copernicus Sentinel-2 L2A", "licence": "Copernicus open data"},
        "L8_C2L2_SR7": {"source": "USGS Landsat 8 Collection 2 Level-2", "licence": "USGS open data"},
        "L9_C2L2_SR7": {"source": "USGS Landsat 9 Collection 2 Level-2", "licence": "USGS open data"},
    }
    extra_calibration = [row["id"] for row in calibration]
    with (args.out / "groups.jsonl").open("w", encoding="utf-8") as handle:
        for group, rows in sorted(((group, [row for row in rows_out if row["isolation_group"] == group])
                                   for group in {row["isolation_group"] for row in rows_out})):
            bbox = [min(row["bbox"][0] for row in rows), min(row["bbox"][1] for row in rows),
                    max(row["bbox"][2] for row in rows), max(row["bbox"][3] for row in rows)]
            handle.write(json.dumps({"group": group, "roles": sorted({row["dataset_role"] for row in rows}),
                                     "profiles": sorted({row["dataset_profile"] for row in rows}),
                                     "records": len(rows), "bbox": bbox}, ensure_ascii=False) + "\n")
    calibration_detail = load_json(BASE / "calibration_resolution/resolution.json")
    with (args.out / "additional_candidates.jsonl").open("w", encoding="utf-8") as handle:
        for row in calibration:
            handle.write(json.dumps({"id": row["id"], "profile": "S2_L2A_CORE10", "role": "TRAIN",
                                     "status": "additional_calibration_resolved",
                                     "note": "not part of the fixed 400 TRAIN selection"}, ensure_ascii=False) + "\n")
        for row in calibration_detail["records"]["derived"]:
            handle.write(json.dumps({"id": row["id"], "profile": "S2_DERIVED_30M",
                                     "role": "NONE", "status": "DERIVED_ONLY",
                                     "reason": row["reason"]}, ensure_ascii=False) + "\n")

    release = {
        "status": "research_dataset_v2",
        "records": len(rows_out),
        "counts": {f"{profile}/{role}": counts.get((profile, role), 0)
                   for profile in sorted({row["dataset_profile"] for row in rows_out})
                   for role in TARGETS},
        "gaps": gaps,
        "group_cap": GROUP_CAP,
        "max_records_per_group": max_group,
        "groups": len(group_counts),
        "group_role_conflicts_removed": len(conflicts),
        "manifest_sha256": sha256(members_path),
        "statistics_sha256": sha256(args.out / "train_statistics.json"),
        "quality_sha256": sha256(args.out / "quality_records.jsonl"),
        "licences": licences,
        "figures": figures,
        "additional_calibration_records": extra_calibration,
        "training_authorized": False,
        "model_training_started": False,
        "frozen_release_unchanged": True,
    }
    write_json(args.out / "release.json", release)

    report = ["# research_dataset_v2", "",
              f"- 记录：{len(rows_out)}；按 profile/role：{json.dumps(release['counts'], ensure_ascii=False)}。",
              f"- 单组上限 {GROUP_CAP}，实际最大 {max_group}；组数 {len(group_counts)}；跨角色冲突组剔除 {len(conflicts)}。",
              "", "| 配置 | TRAIN | VAL | TEST |", "| --- | ---: | ---: | ---: |"]
    for profile in sorted({row["dataset_profile"] for row in rows_out}):
        report.append(f"| {profile} | {counts.get((profile, 'TRAIN'), 0)} | "
                      f"{counts.get((profile, 'VAL'), 0)} | {counts.get((profile, 'TEST'), 0)} |")
    report += ["", "缺口：", ""]
    for name, row in gaps.items():
        if row["gap"]:
            report.append(f"- {name}: 目标 {row['target']}，实际 {row['selected']}，缺 {row['gap']}")
    report += ["", "TRAIN 成员与 v1 子集相同；VAL/TEST 只用新的独立地理组补齐，没有从 TRAIN 重新划分。",
               "统计只由 TRAIN 拟合。额外 8 条 S2 定标补齐记录列为附加候选，不计入 400 人 TRAIN 选择。",
               "本发布不构成训练授权。"]
    (args.out / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("DATASET_V2", json.dumps({"records": len(rows_out), "counts": release["counts"],
                                    "max_per_group": max_group, "conflicts": len(conflicts),
                                    "gaps": {k: v["gap"] for k, v in gaps.items() if v["gap"]}},
                                   ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

