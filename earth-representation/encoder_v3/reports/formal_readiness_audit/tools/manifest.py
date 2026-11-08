"""Write current source hashes and the full audit evidence SHA256 manifest."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ENC = ROOT / "encoder_v3"
AUDIT = ENC / "reports" / "formal_readiness_audit"
REPORTS = ENC / "reports"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    source = {path.name: sha256(path) for path in sorted(ENC.glob("*.py"))}
    current = dict(
        schema="encoder_v3_current_source_hashes_v1",
        generated_at=datetime.now(timezone.utc).isoformat(),
        source_hashes=source,
        config_sha256=sha256(ENC / "config.json"),
        adaptive_config_sha256=sha256(ENC / "adaptive_config.json"),
        data=dict(
            release=sha256(ROOT / "observation_model" / "outputs" / "encoder_v2" / "data_expansion" / "release.json"),
            members=sha256(ROOT / "observation_model" / "outputs" / "encoder_v2" / "data_expansion" / "members.jsonl"),
            statistics=sha256(ROOT / "observation_model" / "outputs" / "encoder_v2" / "data_expansion" / "statistics.json"),
            contracts=sha256(ROOT / "observation_model" / "outputs" / "encoder_v2" / "G1" / "observation_contracts.jsonl"),
            srf_manifest=sha256(ROOT / "observation_model" / "outputs" / "encoder_v2" / "G1" / "srf_manifest.json"),
            contract_tests=sha256(ROOT / "observation_model" / "outputs" / "encoder_v2" / "G1" / "data_contract_tests.json"),
        ),
        budget_sha256=sha256(REPORTS / "pilot_runs" / "budget.json"),
        stop_decision_sha256=sha256(REPORTS / "pilot_runs" / "pilot_stop_decision.json"),
    )
    (AUDIT / "evidence" / "current_source_hashes.json").write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")

    files = []
    for path in sorted(AUDIT.rglob("*")):
        if not path.is_file() or path.name == "sha256_manifest.json" or "__pycache__" in path.parts:
            continue
        files.append(dict(path=str(path.relative_to(AUDIT)).replace("\\", "/"), bytes=path.stat().st_size, sha256=sha256(path)))
    manifest = dict(
        schema="encoder_v3_formal_readiness_sha256_manifest_v1",
        generated_at=datetime.now(timezone.utc).isoformat(),
        baseline_head="269a1e04c8a30510176da01231ce43c3d125bce7",
        branch=__import__("subprocess").run(["git", "-C", str(ROOT), "branch", "--show-current"], capture_output=True, text=True).stdout.strip(),
        files=files,
        external=current,
    )
    (AUDIT / "evidence" / "sha256_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(files=len(files), manifest=str(AUDIT / "evidence" / "sha256_manifest.json"),
                          source_files=len(source), gates_sha256=source.get("gates.py")), indent=2))


if __name__ == "__main__":
    main()
