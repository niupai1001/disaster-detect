"""Read-only baseline snapshot for the EncoderV3 formal-readiness audit.

Never initializes CUDA, never opens raster pixels, never writes outside the
audit evidence directory. Run from the earth-representation repository root.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]  # .../earth-representation
AUDIT = ROOT / "encoder_v3" / "reports" / "formal_readiness_audit"
RELEASE = ROOT / "observation_model" / "outputs" / "encoder_v2" / "data_expansion"
CONTRACTS = ROOT / "observation_model" / "outputs" / "encoder_v2" / "G1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(*args: str) -> dict:
    try:
        cp = subprocess.run(args, cwd=str(ROOT), capture_output=True, text=True, timeout=60)
        return dict(args=list(args), returncode=cp.returncode, stdout=cp.stdout[-20000:], stderr=cp.stderr[-5000:])
    except Exception as exc:  # noqa: BLE001
        return dict(args=list(args), error=type(exc).__name__ + ": " + str(exc))


def package_version(name: str):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / "evidence").mkdir(exist_ok=True)
    status = run("git", "status", "--porcelain=v1")
    branch = run("git", "branch", "--show-current")
    head = run("git", "rev-parse", "HEAD")
    remotes = run("git", "remote", "-v")

    source_files = sorted((ROOT / "encoder_v3").glob("*.py"))
    source_hashes = {p.name: sha256(p) for p in source_files}
    test_files = sorted((ROOT / "encoder_v3").glob("test_*.py"))
    test_methods = {}
    total_tests = 0
    for path in test_files:
        text = path.read_text(encoding="utf-8")
        count = len(re.findall(r"^\s*def test_", text, re.MULTILINE))
        test_methods[path.name] = count
        total_tests += count

    data_files = {}
    for label, path in {
        "release": RELEASE / "release.json",
        "members": RELEASE / "members.jsonl",
        "statistics": RELEASE / "statistics.json",
        "data_contract_tests": CONTRACTS / "data_contract_tests.json",
        "observation_contracts": CONTRACTS / "observation_contracts.jsonl",
        "srf_manifest": CONTRACTS / "srf_manifest.json",
    }.items():
        if path.exists():
            data_files[label] = dict(path=str(path), bytes=path.stat().st_size, sha256=sha256(path))
        else:
            data_files[label] = dict(path=str(path), status="MISSING")

    key_reports = {}
    for name in (
        "admission_review_2026-11-07.json",
        "discovery_coverage_current.json",
        "engineering_tests.json",
        "train_statistics_recomputed.json",
        "native_cache_audit.json",
        "native_lineage_audit.json",
        "B1_expanded_evidence.json",
        "validation_metric_confirmation.json",
        "merge_rate_confirmation.json",
        "merge_B_confirmation.json",
        "merge_levels_confirmation.json",
        "fp32_equivalence_tolerance_review.json",
        "pilot_stop_decision.json",
        "pilot_runs/budget.json",
        "pilot_runs/pilot_stop_decision.json",
    ):
        path = ROOT / "encoder_v3" / "reports" / name
        if path.exists():
            key_reports[name] = dict(bytes=path.stat().st_size, sha256=sha256(path), mtime_ns=path.stat().st_mtime_ns)
        else:
            key_reports[name] = dict(status="EVIDENCE_MISSING")

    checkpoint_evidence = {}
    for name in ("update_check_a624c7d0c860", "peak_update_check", "fitting"):
        path = ROOT / "encoder_v3" / "reports" / "pilot_runs" / name
        if path.exists():
            files = []
            for child in sorted(path.rglob("*")):
                if child.is_file():
                    files.append(dict(name=str(child.relative_to(path)), bytes=child.stat().st_size))
            checkpoint_evidence[name] = dict(path=str(path), files=files)
        else:
            checkpoint_evidence[name] = dict(path=str(path), status="EVIDENCE_MISSING")

    existing = None
    report_path = ROOT / "encoder_v3" / "reports" / "engineering_tests.json"
    if report_path.exists():
        doc = json.loads(report_path.read_text(encoding="utf-8"))
        current_gates = source_hashes.get("gates.py")
        existing = dict(
            status=doc.get("status"),
            tests=doc.get("tests"),
            seconds=doc.get("seconds"),
            reported_gates_py_sha256=(doc.get("source_hashes") or {}).get("gates.py"),
            current_gates_py_sha256=current_gates,
            source_hash_mismatch=bool((doc.get("source_hashes") or {}).get("gates.py") not in (None, current_gates)),
            source_hash_count=len(doc.get("source_hashes") or {}),
            current_top_level_py_count=len(source_files),
            limitations=doc.get("limitations"),
        )

    config = ROOT / "encoder_v3" / "config.json"
    adaptive = ROOT / "encoder_v3" / "adaptive_config.json"
    gpu = run(
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu",
        "--format=csv",
    )

    baseline = dict(
        schema="encoder_v3_formal_readiness_baseline_v1",
        generated_at=datetime.now(timezone.utc).isoformat(),
        cwd=os.getcwd(),
        repo=dict(
            root=str(ROOT),
            branch=(branch.get("stdout") or "").strip(),
            head=(head.get("stdout") or "").strip(),
            requested_baseline="269a1e04c8a30510176da01231ce43c3d125bce7",
            status_returncode=status.get("returncode"),
            status_porcelain=(status.get("stdout") or "").splitlines(),
            remotes=(remotes.get("stdout") or "").strip(),
        ),
        python=dict(
            executable=sys.executable,
            version=sys.version,
            platform=platform.platform(),
            packages={name: package_version(name) for name in ("numpy", "torch", "rasterio", "shapely", "scipy", "pandas", "pyproj")},
            torch_cuda_build=None,
        ),
        gpu=gpu,
        config=dict(path=str(config), sha256=sha256(config), bytes=config.stat().st_size),
        adaptive_config=dict(path=str(adaptive), sha256=sha256(adaptive), bytes=adaptive.stat().st_size),
        source_hashes=source_hashes,
        source_hash_count=len(source_hashes),
        test_files=test_methods,
        test_method_count=total_tests,
        data=data_files,
        key_reports=key_reports,
        pilot_checkpoint_evidence=checkpoint_evidence,
        existing_engineering_report=existing,
        notes=[
            "Baseline only. No CUDA context, no raster pixel read, no optimizer step.",
            "The dirty working tree is recorded, not cleaned or committed.",
            "A test-report source hash mismatch means that report does not describe this HEAD.",
        ],
    )
    try:
        import torch  # noqa: PLC0415  (version inspection only; no CUDA context)

        baseline["python"]["torch_cuda_build"] = torch.version.cuda
        baseline["python"]["cuda_available_without_context"] = torch.cuda.is_available()
    except Exception as exc:  # noqa: BLE001
        baseline["python"]["torch_import_error"] = type(exc).__name__ + ": " + str(exc)

    out = AUDIT / "00_baseline.json"
    out.write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(branch=baseline["repo"]["branch"], head=baseline["repo"]["head"],
                          dirty=baseline["repo"]["status_porcelain"], source_files=len(source_hashes),
                          test_methods=total_tests, config_sha256=baseline["config"]["sha256"],
                          output=str(out)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
