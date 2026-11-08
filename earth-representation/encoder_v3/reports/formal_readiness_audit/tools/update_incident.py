import json
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "evidence" / "controlled_update" / "incident.json"
doc = json.loads(path.read_text(encoding="utf-8"))
doc["status"] = "CORRECTED_RUN_WITHIN_PREFLIGHT_TOLERANCE_NOT_FORMAL"
doc["observed"].update(
    attempts=3,
    optimizer_steps=9,
    unique_logical_updates=6,
    resume_repeats=3,
    overrun_unique_updates=4,
    overrun_resume_checks=2,
    corrected_status="CONTROLLED_UPDATE_AND_RESUME_WITHIN_PREFLIGHT_TOLERANCE_NOT_FORMAL",
    corrected_seconds=244.23065179999685,
    corrected_peak_allocated_GiB=7.039299964904785,
    corrected_parameter_max_abs=7.450580596923828e-09,
    corrected_optimizer_state_relative_l2=1.4694954629799982e-09,
    corrected_loss_abs_difference=2.4658084463169416e-08,
    corrected_parameters_equal=False,
    corrected_loss_equal=False,
    current_report_sha256="d05ba6f11f6321766fdb7b155f5ecac9eaf3e7fba028a1a1187f57d16874c246",
    current_checkpoint_sha256="ccca1f053d66777b2d5da79af11753f4e4a7d6f2f6f63f79f3461dbff25b4ea3",
)
doc["evidence"].update(
    current_report="encoder_v3/reports/formal_readiness_audit/evidence/controlled_update/report.json",
    current_report_sha256=doc["observed"]["current_report_sha256"],
    current_checkpoint_sha256=doc["observed"]["current_checkpoint_sha256"],
)
doc["next_action"] = (
    "User accepted the overrun under the addendum. The corrected run is within the old FP32 preflight tolerance "
    "(parameter max 7.45e-9, optimizer relative L2 1.47e-9, loss 2.47e-8) but is not a formal new-architecture tolerance. "
    "execution_equivalence remains BLOCKED for formal training until formal tolerance/approval and data/B2/panel/auth gates are resolved."
)
path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"status": doc["status"], "observed": doc["observed"], "next_action": doc["next_action"]}, ensure_ascii=False, indent=2))
