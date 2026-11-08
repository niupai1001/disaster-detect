"""Read-only data/SRF/statistics audit; no raster pixel reads."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
AUDIT = ROOT / "encoder_v3" / "reports" / "formal_readiness_audit"
RELEASE = ROOT / "observation_model" / "outputs" / "encoder_v2" / "data_expansion"
CONTRACTS = ROOT / "observation_model" / "outputs" / "encoder_v2" / "G1"
REPORTS = ROOT / "encoder_v3" / "reports"


def sha256(path):
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    release = json.loads((RELEASE / "release.json").read_text(encoding="utf-8-sig"))
    tests = json.loads((CONTRACTS / "data_contract_tests.json").read_text(encoding="utf-8-sig"))
    stats = json.loads((RELEASE / "statistics.json").read_text(encoding="utf-8-sig"))
    manifest = json.loads((CONTRACTS / "srf_manifest.json").read_text(encoding="utf-8-sig"))

    members = []
    with (RELEASE / "members.jsonl").open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                members.append(json.loads(line))
    contracts = {}
    with (CONTRACTS / "observation_contracts.jsonl").open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                contracts[row["id"]] = row

    roles = Counter(); instruments = Counter(); band_counts = Counter(); gsd = Counter(); products = Counter()
    ids = [row["id"] for row in members]
    for row in members:
        roles[row["role"]] += 1
        instruments[row["instrument"]] += 1
        band_counts[len(row["bands"])] += 1
        products[row.get("product")] += 1
        for band in row["bands"]:
            gsd[(band["transform"][0], -band["transform"][4])] += 1

    curves = {curve["id"]: curve for curve in manifest.get("curves", [])}
    curve_checks = []
    for identifier, curve in sorted(curves.items()):
        path = Path(curve["path"])
        record = dict(id=identifier, instrument=curve.get("instrument"), band=curve.get("band_key"),
                      source_type=curve.get("source_type"), wavelength_unit=curve.get("wavelength_unit"),
                      path_exists=path.exists(), declared_sha256=curve.get("sha256"))
        if path.exists():
            record["file_sha256"] = sha256(path)
            record["sha256_matches"] = record["file_sha256"] == curve.get("sha256")
            with np.load(path, allow_pickle=False) as arrays:
                wave = arrays["spectral_wavelength_grid_nm"]
                response = arrays["normalized_response"]
            record.update(points=int(len(wave)), ascending=bool(np.all(np.diff(wave) > 0)),
                          nonnegative=bool(np.all(response >= 0)),
                          finite=bool(np.isfinite(wave).all() and np.isfinite(response).all()),
                          integral=float(np.trapezoid(response, wave)))
        curve_checks.append(record)

    contract_curve_refs = Counter(); missing_curve = []; instrument_mismatch = []
    contract_bands = 0
    members_by_id = {row["id"]: row for row in members}
    for identifier, contract in contracts.items():
        row = members_by_id.get(identifier)
        if row is None:
            missing_curve.append(dict(id=identifier, reason="MISSING_MEMBER"))
            continue
        row_bands = {band["band_key"]: band for band in row["bands"]}
        for band in contract.get("bands", []):
            contract_bands += 1
            key = band["band_key"]
            if key not in row_bands:
                missing_curve.append(dict(id=identifier, band=key, reason="MISSING_MEMBER_BAND"))
                continue
            reference = band.get("spectral_response_ref")
            contract_curve_refs[reference] += 1
            if reference not in curves:
                missing_curve.append(dict(id=identifier, band=key, reference=reference, reason="MISSING_CURVE"))
                continue
            expected_prefix = row["instrument"]
            if not str(reference).startswith(expected_prefix):
                instrument_mismatch.append(dict(id=identifier, band=key, instrument=expected_prefix, reference=reference))

    need_stats = set()
    for row in members:
        for band in row["bands"]:
            need_stats.add(row["instrument"] + ":" + band["band_key"])
    stats_keys = set(stats.get("bands", {}))
    bad_std = [key for key, value in stats.get("bands", {}).items() if not isinstance(value.get("std"), (int, float)) or not np.isfinite(value["std"]) or value["std"] <= 0]

    native_cache = json.loads((REPORTS / "native_cache_audit.json").read_text(encoding="utf-8-sig"))
    lineage = json.loads((REPORTS / "native_lineage_audit.json").read_text(encoding="utf-8-sig"))
    train_stats = json.loads((REPORTS / "train_statistics_recomputed.json").read_text(encoding="utf-8-sig"))
    cross_role = json.loads((REPORTS / "cross_role_split_audit.json").read_text(encoding="utf-8-sig"))
    discovery = json.loads((REPORTS / "discovery_coverage_current.json").read_text(encoding="utf-8-sig"))
    b2 = json.loads((REPORTS / "pair_admission_evidence.json").read_text(encoding="utf-8-sig")) if (REPORTS / "pair_admission_evidence.json").exists() else {}

    output = dict(
        schema="encoder_v3_formal_readiness_data_audit_v1",
        generated_at=datetime.now(timezone.utc).isoformat(),
        release=dict(
            path=str(RELEASE), release_sha256=sha256(RELEASE / "release.json"),
            members_sha256=sha256(RELEASE / "members.jsonl"), statistics_sha256=sha256(RELEASE / "statistics.json"),
            declared_counts=release.get("counts"), actual_counts=dict(roles), members=len(members),
            data_release_accepted=release.get("data_release_accepted"),
            model_training_authorized=release.get("model_training_authorized"),
            normalization_statistics_fitted=release.get("normalization_statistics_fitted"),
            srf_physical_ingestion=release.get("srf_physical_ingestion"),
        ),
        contracts=dict(
            path=str(CONTRACTS), contract_tests_sha256=sha256(CONTRACTS / "data_contract_tests.json"),
            status=tests.get("status"), accepted=tests.get("accepted"), failed=tests.get("failed"),
            members_checked=tests.get("all_materialized_members_checked"),
            contract_bands=contract_bands, distinct_curve_refs=len(contract_curve_refs),
            missing_curve=missing_curve, instrument_mismatch=instrument_mismatch,
        ),
        population=dict(
            roles=dict(roles), instruments=dict(instruments), band_counts=dict(band_counts),
            gsd_counts={"x".join(map(str, key)): value for key, value in gsd.items()},
            products=dict(products), duplicate_ids=sorted(key for key, count in Counter(ids).items() if count > 1),
        ),
        statistics=dict(
            scope=stats.get("scope"), keys=len(stats_keys), needed=len(need_stats),
            missing_keys=sorted(need_stats - stats_keys), extra_keys=sorted(stats_keys - need_stats),
            bad_std=bad_std,
            all_std_finite_positive=not bad_std,
            recomputed_status=train_stats.get("status"), recomputed_members_checked=train_stats.get("members_checked"),
            recomputed_all_counts_match=train_stats.get("all_counts_match"),
            max_abs_mean_difference=train_stats.get("max_absolute_mean_difference"),
            max_abs_std_difference=train_stats.get("max_absolute_std_difference"),
        ),
        srf=dict(
            manifest_sha256=sha256(CONTRACTS / "srf_manifest.json"), curves=len(curves),
            source_types=dict(Counter(curve.get("source_type") for curve in curves.values())),
            instruments=dict(Counter(curve.get("instrument") for curve in curves.values())),
            checks=curve_checks, missing_curve=missing_curve,
            gaussian_approximations=len(manifest.get("gaussian_approximations") or []),
        ),
        existing_evidence=dict(
            native_cache_audit=dict(status=native_cache.get("status"), read_members=native_cache.get("read_members"), roles=native_cache.get("roles"), source_identities=native_cache.get("source_identities")),
            lineage=dict(status=lineage.get("status"), members_total=lineage.get("members_total")),
            cross_role=dict(status=cross_role.get("status"), counts=cross_role.get("counts")),
            discovery=dict(status=discovery.get("status"), scheduled_pending_chunks=discovery.get("scheduled_pending_chunks"), admitted_members=discovery.get("admitted_members"), access_error_chunks=len(discovery.get("access_error_chunks", []))),
            b2=dict(admitted_pairs=b2.get("admitted_pairs") if isinstance(b2, dict) else None),
        ),
        candidate_scope=dict(
            recommendation="Use the bounded 7,778-member native release as the candidate formal scope only after user approval; it is hash-bound, reread, lineage-reproduced and has matching TRAIN-only statistics.",
            not_approved=True,
            reasons=["TRAIN/VAL roles are present; TEST/GUARD are not in this release", "QA policy for the full release is not formally approved", "EnMAP is not admitted and its SRF is derived_gaussian", "B2 admitted pairs are zero"],
            notes=["File discovery coverage is not data admission", "No new member was promoted or repartitioned by this audit"],
        ),
    )
    out = AUDIT / "03_data_audit.json"
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(roles=dict(roles), members=len(members), curves=len(curves),
                          missing_curve=len(missing_curve), instrument_mismatch=len(instrument_mismatch),
                          stats_missing=len(need_stats - stats_keys), bad_std=len(bad_std),
                          output=str(out)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
