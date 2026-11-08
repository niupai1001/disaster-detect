"""Run the encoder_v3 unit suite and write a machine-readable evidence file.

    python reports/optical_selfsup_closure/tools/run_tests.py

Records the exact command, counts, durations, per-test outcomes and the SHA256 of
every changed source/test file, so 07_tests.json can cite verifiable evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CLOSURE = ROOT / "encoder_v3" / "reports" / "optical_selfsup_closure"
EVIDENCE = CLOSURE / "evidence"
CHANGED = (
    "gates.py", "validation_metric.py", "formal_schedule.py", "scheduled_data.py", "validation_panels.py",
    "formal_panels.py", "route_step.py", "route_control.py", "formal_runner.py", "selfsup_preflight.py",
    "selfsup_train.py", "selfsup_benchmark.py", "sampling_coverage.py", "test_optical_selfsup.py",
    "test_formal_runner.py", "test_route_step.py", "test_discovery.py", "config_optical_selfsup_v1.json",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the encoder_v3 unit suite with evidence output")
    parser.add_argument("--output", type=Path, default=EVIDENCE / "tests.json")
    parser.add_argument("--log", type=Path, default=EVIDENCE / "tests.log")
    args = parser.parse_args(argv)
    command = [sys.executable, "-m", "unittest", "discover", "-s", "encoder_v3", "-t", ".", "-p", "test_*.py", "-v"]
    started = datetime.now(timezone.utc)
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    seconds = (datetime.now(timezone.utc) - started).total_seconds()
    output = completed.stdout + completed.stderr
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(output, encoding="utf-8")
    matches = re.findall(r"^(test_\w+) \(([\w\.]+)\)(?: \.\.\.)? (ok|FAIL|ERROR|skipped.*)$", output, re.MULTILINE)
    tests = [dict(test=name, module=module, outcome=outcome.split()[0]) for name, module, outcome in matches]
    ran = re.search(r"^Ran (\d+) tests in ([0-9.]+)s", output, re.MULTILINE)
    failures = re.findall(r"^(?:FAIL|ERROR): (\w+) \(([\w\.]+)\)", output, re.MULTILINE)
    report = dict(
        schema="encoder_v3_unit_suite_evidence_v1",
        generated_at=datetime.now(timezone.utc).isoformat(),
        command=" ".join(Path(part).name if index == 0 else part for index, part in enumerate(command)),
        cwd=str(ROOT),
        started_at=started.isoformat(),
        seconds=round(seconds, 1),
        tests=int(ran.group(1)) if ran else len(tests),
        unittest_seconds=float(ran.group(2)) if ran else None,
        status="PASS" if completed.returncode == 0 else "FAIL",
        exit_code=completed.returncode,
        outcomes={name: sum(1 for test in tests if test["outcome"] == name) for name in ("ok", "FAIL", "ERROR", "skipped")},
        failures=[dict(test=name, module=module) for name, module in failures],
        skipped=[dict(test=test["test"], module=test["module"]) for test in tests if test["outcome"] == "skipped"],
        new_or_changed_tests=dict(
            optical_selfsup="test_optical_selfsup.py (17 cases: pool/sampler/recipe, S_intra, E0 gates, panel collect end-to-end)",
            runner="test_formal_runner.py (+4: lookahead single prepare, committed early-stop restore, wall pre-check, mock rejection)",
            route_step="test_route_step.py (+2: runner_state/optimizer_steps in checkpoint, injected gate)",
            discovery="test_discovery.py: explicit skip when ripgrep is absent in the environment",
        ),
        source_sha256={name: sha256(ROOT / "encoder_v3" / name) for name in CHANGED},
        log=dict(path=str(args.log.relative_to(ROOT)), sha256=sha256(args.log)),
        optimizer_updates=0,
    )
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "tests", "unittest_seconds", "outcomes", "failures")}, ensure_ascii=False, indent=2))
    return 0 if completed.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
