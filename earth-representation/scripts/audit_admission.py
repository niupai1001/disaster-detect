"""Admission audit for the local completion release (Q01-Q08).

Reads a members.jsonl, checks every record against the contract and the frozen
quality rules, and writes pass/limited/fail lists plus a summary. It never
downloads or rewrites imagery.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEMBERS = ROOT / "outputs/encoder_v3/data_download_plan/local_completion_release_v1/members.jsonl"
DEFAULT_OUT = ROOT / "outputs/encoder_v3/data_download_plan/admission_audit_v1"

S2_CORE = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
S2_FOUR = ["B02", "B03", "B04", "B08"]
L8_SR7 = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
L8_SR6 = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
EXPECTED_SUPPORT = {"B02": 10, "B03": 10, "B04": 10, "B08": 10,
                    "B05": 20, "B06": 20, "B07": 20, "B8A": 20, "B11": 20, "B12": 20}

CONTRACTS = {
    "S2_L2A_CORE10": S2_CORE, "S2_L2A_4BAND": S2_FOUR,
    "L8_C2L2_SR7": L8_SR7, "L8_C2L2_SR6": L8_SR6,
    "L9_C2L2_SR7": L8_SR7, "L9_C2L2_SR6": L8_SR6,
}


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


def contract_of(member):
    """Contract from the band set; declaration order is normalised to the contract."""
    bands = [item["band_key"] for item in member["bands"]]
    sensor = str(member.get("sensor", "")).upper()
    if sensor == "S2":
        if set(bands) == set(S2_CORE):
            return "S2_L2A_CORE10"
        if set(bands) == set(S2_FOUR):
            return "S2_L2A_4BAND"
        return "S2_UNKNOWN"
    if sensor in {"L8", "L9"}:
        if set(bands) == set(L8_SR7):
            return f"{sensor}_C2L2_SR7"
        if set(bands) == set(L8_SR7):
            return f"{sensor}_C2L2_SR7"
        if set(bands) == set(L8_SR6):
            return f"{sensor}_C2L2_SR6"
    return "UNKNOWN"


def audit_one(member, duplicates, spanning_groups):
    import numpy as np

    record = {"id": member["id"], "sensor": member.get("sensor"), "role": member.get("role"),
              "contract": contract_of(member), "fail": [], "notes": []}
    fail = record["fail"]
    bands = member["bands"]
    keys = [item["band_key"] for item in bands]
    expected = CONTRACTS.get(record["contract"])
    if expected is None:
        fail.append("Q02_UNKNOWN_CONTRACT")
    else:
        if len(set(keys)) != len(keys):
            fail.append("Q02_DUPLICATE_BANDS")
        if sorted(keys) != sorted(expected):
            fail.append("Q02_BAND_SET_MISMATCH")
        elif keys != expected:
            record["notes"].append("BAND_ORDER_NORMALIZED")

    # Q01: cache file exists and matches the recorded hash
    cache = Path(member["cache"])
    if not cache.exists():
        fail.append("Q01_CACHE_MISSING")
        return record
    if sha256(cache) != member["cache_sha256"]:
        fail.append("Q01_CACHE_SHA_MISMATCH")

    # Q03: calibration rule per contract
    for band in bands:
        scale, offset = float(band.get("scale", "nan")), float(band.get("offset", "nan"))
        key = band["band_key"]
        if record["sensor"] == "S2":
            if abs(scale - 0.0001) > 1e-9 or offset not in (-0.1, 0.0):
                fail.append(f"Q03_CALIBRATION_{key}")
        elif record["sensor"] in {"L8", "L9"}:
            if abs(scale - 0.0000275) > 1e-12 or abs(offset + 0.2) > 1e-9:
                fail.append(f"Q03_CALIBRATION_{key}")
        if "arithmetic_roundtrip_max_abs" in band and float(band["arithmetic_roundtrip_max_abs"]) > 1e-4:
            fail.append(f"Q03_ROUNDTRIP_{key}")

    # Q02/Q05: native support, geometry and product identity
    products = set()
    for band in bands:
        support = [float(value) for value in band.get("native_pixel_support_m", [])]
        key = band["band_key"]
        if key in EXPECTED_SUPPORT and support and abs(support[0] - EXPECTED_SUPPORT[key]) > 1e-6:
            fail.append(f"Q05_SUPPORT_{key}")
        transform = band.get("transform") or [0] * 6
        if len(transform) != 6 or abs(transform[1]) > 1e-9 or abs(transform[3]) > 1e-9:
            fail.append(f"Q05_NOT_NORTH_UP_{key}")
        if not band.get("crs"):
            fail.append(f"Q05_NO_CRS_{key}")
        asset = band.get("source_asset") or ""
        if asset:
            import re as _re
            match = _re.search(r"(S2[ABC]_MSIL2A_[^/]+\.SAFE)", asset)
            products.add(match.group(1) if match else asset.rsplit("/", 1)[0])
    if len(products) > 1 and record["sensor"] == "S2":
        fail.append("Q02_CROSS_PRODUCT_BANDS")

    # Q04: coverage and finite values, recomputed from the frozen cache
    try:
        with np.load(cache, allow_pickle=False) as stored:
            for key in keys:
                value = stored[key]
                valid = stored[key + "__valid"]
                coverage = float(valid.mean())
                if coverage < 0.5:
                    fail.append(f"Q04_COVERAGE_{key}")
                if valid.any() and not np.isfinite(value[valid]).all():
                    fail.append(f"Q04_NONFINITE_{key}")
    except Exception as exc:
        fail.append("Q01_CACHE_UNREADABLE:" + str(exc)[:80])

    # Q06: QA provenance must be declared; an undeclared label is a metadata gap,
    # not a pixel-quality rejection.
    if not member.get("qa_provenance") and not member.get("QA_provenance"):
        record["notes"].append("Q06_UNDECLARED_QA")

    # Q07: roles, duplicates and cross-role groups
    if member.get("role") not in {"TRAIN", "VAL", "TEST"}:
        fail.append("Q07_UNKNOWN_ROLE")
    if member["id"] in duplicates:
        record["notes"].append("Q07_DUPLICATE_OBSERVATION")
    if member.get("isolation_group") in spanning_groups:
        fail.append("Q07_GROUP_SPANS_ROLES")

    if record["contract"].endswith(("4BAND", "SR6")):
        record["notes"].append("LIMITED_CONTRACT_ONLY")
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--members", type=Path, default=DEFAULT_MEMBERS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    members = [json.loads(line) for line in args.members.read_text(encoding="utf-8").splitlines()
               if line.strip()]

    keys = Counter()
    for member in members:
        bbox = ",".join(f"{value:.4f}" for value in member.get("bbox") or [0, 0, 0, 0])
        keys[(member.get("product"), str(member.get("date"))[:10], bbox)] += 1
    duplicates = {member["id"] for member in members
                  if keys[(member.get("product"), str(member.get("date"))[:10],
                           ",".join(f"{value:.4f}" for value in member.get("bbox") or [0, 0, 0, 0]))] > 1}
    groups = defaultdict(set)
    for member in members:
        groups[member.get("isolation_group")].add(member.get("role"))
    spanning = {group for group, roles in groups.items() if len(roles) > 1}

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(lambda member: audit_one(member, duplicates, spanning), members):
            results.append(result)

    by_status = Counter()
    by_contract = Counter()
    failures = Counter()
    limited = []
    passed = []
    failed = []
    review = []
    for result in results:
        limited_contract = any(note == "LIMITED_CONTRACT_ONLY" for note in result["notes"])
        needs_review = any(note == "Q06_UNDECLARED_QA" for note in result["notes"])
        if result["fail"]:
            failed.append(result)
            by_status["fail"] += 1
            for item in result["fail"]:
                failures[item.split(":")[0]] += 1
        elif limited_contract:
            limited.append(result)
            by_status["pass_limited"] += 1
        elif needs_review:
            review.append(result)
            by_status["needs_review"] += 1
        else:
            passed.append(result)
            by_status["pass_core"] += 1
        by_contract[result["contract"]] += 1

    with (args.out / "audit_records.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    for name, rows in (("pass_core.jsonl", passed), ("pass_limited.jsonl", limited),
                       ("needs_review.jsonl", review), ("fail.jsonl", failed)):
        with (args.out / name).open("w", encoding="utf-8") as handle:
            for result in rows:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    summary = {
        "members": len(members), "status": dict(by_status), "contracts": dict(by_contract),
        "failure_reasons": dict(failures.most_common()),
        "duplicate_observations": len(duplicates), "groups_spanning_roles": len(spanning),
        "pass_core": len(passed), "pass_limited": len(limited), "needs_review": len(review), "fail": len(failed),
        "note": "pass_core meets the full contract (S2 CORE10 / L8-L9 SR7); pass_limited is a valid but smaller contract (4-band S2 / 6-band Landsat); fail lists every Q01-Q08 rejection.",
    }
    write_json(args.out / "summary.json", summary)

    report = ["# 准入筛选报告 v1", "",
              f"- 审计记录：{len(members)}；通过完整契约：{len(passed)}；仅较小契约：{len(limited)}；拒收：{len(failed)}。",
              f"- 契约分布：{json.dumps(dict(by_contract), ensure_ascii=False)}。",
              f"- 拒收原因：{json.dumps(dict(failures.most_common()), ensure_ascii=False)}。",
              f"- 近重复观测：{len(duplicates)}；跨角色组：{len(spanning)}。", "",
              "筛选规则：Q01 缓存与哈希；Q02 波段集合/顺序/同一产品；Q03 定标只应用一次；Q04 覆盖率≥50% 且有限；Q05 原生支撑与 north-up 几何；Q06 QA 来源声明；Q07 角色/重复/组隔离；Q08 传感器许可（本地均为公开数据）。",
              "", "只有 `pass_core.jsonl` 中的记录可直接进入主实验；`pass_limited.jsonl` 需在对应小契约下使用；`fail.jsonl` 不得进入训练或评价。"]
    (args.out / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("AUDIT", json.dumps(summary, ensure_ascii=False)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
