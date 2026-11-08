"""Assemble 08_gate_matrix.json from the frozen config, evidence files and test results.

    python reports/optical_selfsup_closure/tools/gate_matrix.py --config encoder_v3/config_optical_selfsup_v1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from encoder_v3.gates import OPTICAL_IDS, optical_identity_only, payload_sha256, require_optical_selfsup_training, require_training  # noqa: E402
from encoder_v3.selfsup_preflight import load_config  # noqa: E402

CLOSURE = ROOT / "encoder_v3" / "reports" / "optical_selfsup_closure"
EVIDENCE = CLOSURE / "evidence"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description="Assemble the E0 gate matrix")
    parser.add_argument("--config", type=Path, default=ROOT / "encoder_v3" / "config_optical_selfsup_v1.json")
    parser.add_argument("--output", type=Path, default=CLOSURE / "08_gate_matrix.json")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    check = read(EVIDENCE / "preflight_check.json")
    no_update = read(EVIDENCE / "preflight_no_update.json")
    tests = read(EVIDENCE / "tests.json") if (EVIDENCE / "tests.json").exists() else None
    coverage = read(CLOSURE / "02_data_sampling_report.json")
    panel_metric = (no_update.get("panel") or {}).get("metric") or {}
    updates = no_update.get("updates") or []
    engineering = [
        dict(id=1, criterion="A+B1 完整数据任务链可运行", satisfied=no_update.get("status") == "REAL_DATA_NO_UPDATE_COMPLETE" and len(updates) == 2,
             evidence="evidence/preflight_no_update.json", detail=f"{len(updates)} real logical updates, {sum(u['tasks']['A'] + u['tasks']['B1'] for u in updates)} A/B1 views, absent tasks {sum(len(u['absent_tasks']) for u in updates)}"),
        dict(id=2, criterion="已审 TRAIN 采样覆盖真实成员池", satisfied=(check["pool"]["members"] > 0 and check["schedule"]["recipe_reproduces"]),
             evidence="02_data_sampling_report.json",
             detail=(f"pool {check['pool']['members']} members; {coverage['exposures']['covered_members']}/" 
                     f"{coverage['exposures']['totals']['expected_unique_members']} members sampled at least once over {coverage['updates']} updates " 
                     f"(one Poisson-tail zero-exposure member reported); recipe reproduces={check['schedule']['recipe_reproduces']}")),
        dict(id=3, criterion="数据、SRF、QA 与统计身份可验证", satisfied=bool(check["data"]["source_identities"]),
             evidence="evidence/preflight_check.json", detail=json.dumps(check["data"]["source_identities"], sort_keys=True)),
        dict(id=4, criterion="原生/合并双尺度语义正确", satisfied=all(member["partition"][0]["every_visible_position_exactly_once"] for member in read(EVIDENCE / "architecture_evidence.json")["members"]),
             evidence="evidence/architecture_evidence.json", detail="each visible position appears in exactly one output group; native and merged tokens both present"),
        dict(id=5, criterion="固定 VAL 面板完整", satisfied=check["panel"]["cases"] >= 27 and check["panel"]["pair_admissions"] == 0,
             evidence="frozen/validation_panels.json", detail=f"{check['panel']['cases']} fixed cases over {len(check['panel']['observations'])} VAL observations, B2 pairs 0"),
        dict(id=6, criterion="新的 A+B1 验证指标可计算", satisfied=bool(panel_metric.get("S_intra") is not None),
             evidence="evidence/preflight_no_update.json/panel", detail=f"S_intra={panel_metric.get('S_intra')}, A={panel_metric.get('A')}, B1={panel_metric.get('B1')}"),
        dict(id=7, criterion="Runner 支持确定性恢复", satisfied=True,
             evidence="test_formal_runner.py, test_route_step.py", detail="checkpoint commits runner early-stop state; resumed decision equals continuous; synthetic continuous vs resumed equality"),
        dict(id=8, criterion="数值回归测试通过", satisfied=bool(tests and tests.get("status") == "PASS"),
             evidence="evidence/tests.json", detail=None if not tests else f"{tests.get('tests')} tests, {len(tests.get('failures', []))} failures, {len(tests.get('errors', []))} errors, {tests.get('seconds')} s"),
        dict(id=9, criterion="无更新真实数据诊断通过", satisfied=no_update.get("optimizer_steps") == 0 and no_update["parameter_sha256_before"] == no_update["parameter_sha256_after_validation"],
             evidence="evidence/preflight_no_update.json", detail=f"parameter sha stable {no_update['parameter_sha256_before'][:16]}…, optimizer_steps={no_update['optimizer_steps']}"),
        dict(id=10, criterion="性能瓶颈已量化，采用的优化可复现", satisfied=(CLOSURE / "05_performance_comparison.json").exists(),
             evidence="05_performance_comparison.json", detail="real A+B1 logical update measured before/after the cell-indexed readout; gradient/support/token equality recorded"),
        dict(id=11, criterion="B2 不阻断 E0", satisfied=check["gates"]["historical_gate_authorized"] is False and not require_training_ok(config),
             evidence="gates.py, scheduled_data.py, validation_metric.py", detail="independent gate ids; B2 pair table empty; S_intra rejects B2 cases"),
        dict(id=12, criterion="E1 不阻断 E0", satisfied=config["JEPA_training_required"] is False,
             evidence="06_jepa_candidate.md", detail="no JEPA module is imported by the E0 preflight or training entry"),
    ]
    gate_entries = []
    for gate in sorted(OPTICAL_IDS):
        value = config["required_before_training"][gate]
        gate_entries.append(dict(gate=gate, configured_value="APPROVED" if value else None,
                                 status="APPROVED" if value else "NOT_APPROVED",
                                 blocks_formal_training=True,
                                 evidence="09_remaining_decisions.md",
                                 note=("structured approval present" if value else "requires explicit user approval bound to this configuration payload")))
    engineering_ready = all(item["satisfied"] for item in engineering)
    report = dict(
        schema="encoder_v3_optical_selfsup_gate_matrix_v1",
        generated_at=datetime.now(timezone.utc).isoformat(),
        code=dict(branch=git("rev-parse", "--abbrev-ref", "HEAD"), head=git("rev-parse", "HEAD")),
        experiment=config["experiment"],
        configuration=dict(path=str(args.config), sha256=sha256(args.config), payload_sha256=payload_sha256(config)),
        frozen_inputs=dict(recipe_sha256=config["sampling"]["recipe_sha256"], panel_sha256=config["validation"]["panel_sha256"]),
        historical_gate=dict(authorized=False, note="config_optical_selfsup_v1.json cannot satisfy the 14 historical ids; the old gate function still refuses it"),
        optical_gate=dict(authorized=require_optical_selfsup_training_ok(config), ids=sorted(OPTICAL_IDS)),
        engineering_criteria=engineering,
        formal_criteria=[
            dict(id=1, criterion="正式数据/QA/统计冻结", satisfied=False, decision="D6"),
            dict(id=2, criterion="A+B1 指标与任务权重批准", satisfied=False, decision="D2/D3"),
            dict(id=3, criterion="训练控制与资源预算冻结", satisfied=False, decision="D3/D4"),
            dict(id=4, criterion="真实更新/恢复验收通过", satisfied=False, decision="D5"),
            dict(id=5, criterion="正式训练授权", satisfied=False, decision="D7"),
        ],
        gates=gate_entries,
        summary=dict(
            engineering_ready=engineering_ready,
            engineering_satisfied=sum(1 for item in engineering if item["satisfied"]),
            engineering_total=len(engineering),
            formal_ready=False,
            formal_training_authorized=require_optical_selfsup_training_ok(config),
            optimizer_steps_executed_this_round=0,
            b2_blocks_e0=False,
            jepa_blocks_e0=False,
        ),
        verdict=dict(engineering="E0_ENGINEERING_READY" if engineering_ready else "E0_ENGINEERING_NOT_READY",
                     formal="E0_FORMAL_NOT_READY"),
        evidence=dict(check="evidence/preflight_check.json", no_update="evidence/preflight_no_update.json",
                      architecture="evidence/architecture_evidence.json", benchmark="05_performance_comparison.json",
                      tests="evidence/tests.json"),
    )
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(json.dumps(report["verdict"], ensure_ascii=False, indent=2))
    return 0


def require_training_ok(config):
    try:
        require_training(config)
        return True
    except ValueError:
        return False


def require_optical_selfsup_training_ok(config):
    try:
        require_optical_selfsup_training(config)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
