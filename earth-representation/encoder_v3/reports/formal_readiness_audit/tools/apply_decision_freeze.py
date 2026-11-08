"""Record the user's D4-D6 formal freeze as structured gate evidence.

It freezes only the architecture/Merge/mask values the user approved. It does
not approve data, validation panel, execution equivalence or formal training.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from encoder_v3.gates import payload_sha256

ROOT = Path(__file__).resolve().parents[4]
ENC = ROOT / "encoder_v3"
AUDIT = ENC / "reports" / "formal_readiness_audit"
CONFIG = ENC / "config.json"
USER = AUDIT / "approvals" / "user_decisions_2026-11-08.json"
CONFIRM = AUDIT / "approvals" / "D4_D6_formal_freeze.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["confirmed_formal"] = {
        "scope": "user-approved D4-D6 formal freeze; 2026-11-08",
        "architecture": {
            "srf_encoder": "Linear(1,64)-GELU-Linear(64,32); trapezoid response-area normalization",
            "hypernetwork": "34->64->320 GELU; 32x3x3 kernels + 32 bias",
            "local_network": "32->128 projection, two-layer Transformer, CLS readout",
            "global_transformer": "128 width, 4 layers, 4 heads, 512 FFN; full attention",
            "decoder": "162->128->1 shared MLP",
            "precision": "fp32",
            "execution": "local_batch 64, observation microbatch 1, indexed cell readout, packed off, retained geometry off",
        },
        "merge": {
            "family": "area_equivalent_hill",
            "amplitude": 0.8092248954506467,
            "scale_m": 4.010953026221307,
            "power": 1.219268281571556,
            "effective_gsd": "sqrt(gx*gy)",
            "span_rule": "1/(1-r)",
            "single_round": True,
            "levels": ["native", "merged"],
            "P_role": "task mask/physical grid context only; does not change merge membership",
        },
        "mask": {
            "count": 1,
            "target_patch_fraction": 0.5,
            "aspect_ratio": "log-uniform [0.5,2.0]",
            "position": "uniform feasible patch-aligned rectangle",
            "size_rounding": "nearest integer patch count, clipped to observation patch layout",
        },
    }
    config_sha = payload_sha256(config)
    source_sha = sha(AUDIT / "evidence" / "current_source_hashes.json")
    confirmations = {
        "schema": "encoder_v3_user_approval_v1",
        "user_answer": "批准推荐冻结包",
        "user_record": str(USER.relative_to(ROOT)),
        "user_record_sha256": sha(USER),
        "approved_at": "2026-11-08T00:05:44-05:00",
        "config_payload_sha256": config_sha,
        "source_manifest_sha256": source_sha,
        "scope": ["merge_configuration", "hypernetwork_hidden_structure", "mask_rectangle_distribution", "optimizer_choice"],
        "not_approved": ["frozen_data_manifest", "pair_admission", "validation_panel", "improvement_threshold",
                         "stability_window", "early_stop", "precision_acceptance", "resource_limit",
                         "execution_equivalence", "formal_training_authorization"],
        "panel_principle": "B1 single + multi count 2..C-1; B2 empty; formal S unavailable until B2 admission",
    }
    CONFIRM.write_text(json.dumps(confirmations, ensure_ascii=False, indent=2), encoding="utf-8")
    approval_ref = str(CONFIRM.relative_to(ROOT))
    tests = ["encoder_v3/reports/formal_readiness_audit/evidence/full_suite_gpu.json"]
    for gate in ("merge_configuration", "hypernetwork_hidden_structure", "mask_rectangle_distribution", "optimizer_choice"):
        config["required_before_training"][gate] = dict(
            gate_id=gate, status="APPROVED", scope="formal_training",
            approval_reference=approval_ref, source_sha256=source_sha,
            data_sha256=None, config_sha256=config_sha, tests=tests,
            reviewer="user", approved_at="2026-11-08T00:05:44-05:00",
        )
    CONFIG.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(config_sha=config_sha, approval=approval_ref,
                          approved=[k for k, v in config["required_before_training"].items() if isinstance(v, dict)],
                          remaining=[k for k, v in config["required_before_training"].items() if v is None]), indent=2))


if __name__ == "__main__":
    main()
