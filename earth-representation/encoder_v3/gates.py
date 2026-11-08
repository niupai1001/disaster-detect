"""Fail-closed formal-training gate with structured, hash-bound evidence.

A non-empty string, the literal True, a TODO, or a bare path is not an
approval. Each applicable gate needs an explicit user approval record that
binds the substantive config payload and the source/data identities used for
the review. The checked-in config deliberately remains unauthorized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

GATE_IDS = {
    "merge_configuration",
    "hypernetwork_hidden_structure",
    "frozen_data_manifest",
    "pair_admission",
    "mask_rectangle_distribution",
    "validation_panel",
    "improvement_threshold",
    "stability_window",
    "early_stop",
    "precision_acceptance",
    "resource_limit",
    "execution_equivalence",
    "optimizer_choice",
    "formal_training_authorization",
}
JEPA_IDS = {
    "latent_loss_and_normalization",
    "admitted_cross_source_pairs",
    "AB_stability_evidence",
    "joint_effective_batch_integration",
    "query_projection_review",
}
ADAPTIVE_IDS = {
    "spatial_hierarchy_review",
    "physical_patch_role_and_mask_review",
    "adaptive_gpu_efficiency_acceptance",
    "participation_curve_and_anisotropic_rule",
}
# Independent E0 (A+B1 intra-observation) admission. It shares the evidence
# format but none of the ids above: an E0 config cannot satisfy the historical
# A/B1/B2 gates by renaming them, and the historical gates stay closed.
OPTICAL_IDS = {
    "optical_data_freeze",
    "optical_sampling_recipe",
    "optical_task_protocol",
    "optical_validation_protocol",
    "optical_optimizer_schedule",
    "optical_resource_budget",
    "optical_update_recovery_acceptance",
    "optical_formal_training_authorization",
}
OPTICAL_SCOPE = "optical_selfsup_formal_training"
APPROVAL_FIELDS = (
    "gate_id",
    "status",
    "scope",
    "approval_reference",
    "source_sha256",
    "data_sha256",
    "config_sha256",
    "tests",
    "reviewer",
    "approved_at",
)
_HEX = set("0123456789abcdef")


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in _HEX for c in value):
        raise ValueError("Explicit lowercase SHA256 required")
    return value


def payload_sha256(config):
    """Hash the substantive configuration, excluding approval/pending blocks."""
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    payload = {key: value for key, value in config.items() if key not in ("required_before_training", "required_before_jepa")}
    try:
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Configuration is not canonically serializable; no approval can be verified") from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_approval(gate_id, value, *, scope, config_sha, source_sha256=None, data_sha256=None):
    if not isinstance(value, dict):
        raise ValueError(f"{gate_id}: structured approval record required, not {type(value).__name__}")
    missing = [field for field in APPROVAL_FIELDS if field not in value]
    if missing:
        raise ValueError(f"{gate_id}: incomplete approval record: {', '.join(missing)}")
    if value["gate_id"] != gate_id:
        raise ValueError(f"{gate_id}: approval gate_id differs")
    if value["status"] != "APPROVED":
        raise ValueError(f"{gate_id}: approval status is not APPROVED")
    if value["scope"] != scope:
        raise ValueError(f"{gate_id}: approval scope must be {scope}")
    if value["reviewer"] != "user":
        raise ValueError(f"{gate_id}: only an explicit user approval is accepted")
    for field in ("approval_reference", "approved_at"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise ValueError(f"{gate_id}: explicit non-empty {field} required")
    if not isinstance(value["tests"], list) or not value["tests"]:
        raise ValueError(f"{gate_id}: at least one test/review reference required")
    for item in value["tests"]:
        if isinstance(item, str):
            if not item.strip():
                raise ValueError(f"{gate_id}: empty test reference")
        elif isinstance(item, dict):
            if not isinstance(item.get("path"), str) or not item["path"].strip() or "sha256" not in item:
                raise ValueError(f"{gate_id}: test reference needs path and sha256")
            _sha(item["sha256"])
        else:
            raise ValueError(f"{gate_id}: invalid test reference")
    declared_source = _sha(value["source_sha256"])
    declared_config = _sha(value["config_sha256"])
    if declared_config != config_sha:
        raise ValueError(f"{gate_id}: approval is bound to a different configuration payload")
    if source_sha256 is not None and declared_source != _sha(source_sha256):
        raise ValueError(f"{gate_id}: approval is bound to a different source manifest")
    declared_data = value["data_sha256"]
    if declared_data is not None:
        _sha(declared_data)
        if data_sha256 is not None and declared_data != _sha(data_sha256):
            raise ValueError(f"{gate_id}: approval is bound to a different data manifest")
    return value


def require_training(config, *, source_manifest_sha256=None, data_manifest_sha256=None):
    """Validate every required formal-training gate; raise before any work."""
    required = config.get("required_before_training") if isinstance(config, dict) else None
    if not isinstance(required, dict) or not required:
        raise ValueError("Missing required_before_training")
    missing = sorted(GATE_IDS - set(required))
    unresolved = sorted(key for key in GATE_IDS & set(required) if required[key] is None or required[key] is False or required[key] == "")
    if missing or unresolved:
        raise ValueError("Training blocked: " + ", ".join(missing + unresolved))
    config_sha = payload_sha256(config)
    for gate_id in sorted(GATE_IDS):
        _validate_approval(
            gate_id,
            required[gate_id],
            scope="formal_training",
            config_sha=config_sha,
            source_sha256=source_manifest_sha256,
            data_sha256=data_manifest_sha256,
        )
    return True


def require_jepa(config, *, source_manifest_sha256=None, data_manifest_sha256=None):
    checks = config.get("required_before_jepa") if isinstance(config, dict) else None
    if not isinstance(checks, dict):
        raise ValueError("Missing required_before_jepa")
    missing = sorted(JEPA_IDS - set(checks))
    unresolved = sorted(key for key in JEPA_IDS & set(checks) if checks[key] is None or checks[key] is False or checks[key] == "")
    if missing or unresolved:
        raise ValueError("JEPA blocked: " + ", ".join(missing + unresolved))
    config_sha = payload_sha256(config)
    for gate_id in sorted(JEPA_IDS):
        _validate_approval(
            gate_id,
            checks[gate_id],
            scope="formal_jepa",
            config_sha=config_sha,
            source_sha256=source_manifest_sha256,
            data_sha256=data_manifest_sha256,
        )
    return True


def require_adaptive(adaptive_config, *, source_manifest_sha256=None, data_manifest_sha256=None):
    required = adaptive_config.get("required_before_training") if isinstance(adaptive_config, dict) else None
    if not isinstance(required, dict) or not required:
        raise ValueError("Missing adaptive required_before_training")
    missing = sorted(ADAPTIVE_IDS - set(required))
    unresolved = sorted(key for key in ADAPTIVE_IDS & set(required) if required[key] is None or required[key] is False or required[key] == "")
    if missing or unresolved:
        raise ValueError("Training blocked by adaptive gates: " + ", ".join(missing + unresolved))
    config_sha = payload_sha256(adaptive_config)
    for gate_id in sorted(ADAPTIVE_IDS):
        _validate_approval(
            gate_id,
            required[gate_id],
            scope="formal_training",
            config_sha=config_sha,
            source_sha256=source_manifest_sha256,
            data_sha256=data_manifest_sha256,
        )
    return True


def optical_identity_only(config):
    """Load-time identity check only (preflight); authorization lives in selfsup_train."""
    return True


def require_optical_selfsup_training(config, *, source_manifest_sha256=None, data_manifest_sha256=None):
    """E0 admission only: A+B1 intra-observation; B2, JEPA and task heads are out of scope."""
    required = config.get("required_before_training") if isinstance(config, dict) else None
    if not isinstance(required, dict) or not required:
        raise ValueError("Missing optical required_before_training")
    missing = sorted(OPTICAL_IDS - set(required))
    unresolved = sorted(key for key in OPTICAL_IDS & set(required) if required[key] is None or required[key] is False or required[key] == "")
    if missing or unresolved:
        raise ValueError("Optical self-supervised training blocked: " + ", ".join(missing + unresolved))
    config_sha = payload_sha256(config)
    for gate_id in sorted(OPTICAL_IDS):
        _validate_approval(
            gate_id,
            required[gate_id],
            scope=OPTICAL_SCOPE,
            config_sha=config_sha,
            source_sha256=source_manifest_sha256,
            data_sha256=data_manifest_sha256,
        )
    if config.get("B2_training_required") is not False:
        raise ValueError("E0 must declare B2_training_required=false")
    if config.get("JEPA_training_required") is not False:
        raise ValueError("E0 must declare JEPA_training_required=false")
    if config.get("m9_task_head_training_authorized") is not False:
        raise ValueError("Task-head training must remain unauthorized for E0")
    return True


def require_all(config, *, adaptive_config=None, jepa=False, source_manifest_sha256=None, data_manifest_sha256=None):
    require_training(config, source_manifest_sha256=source_manifest_sha256, data_manifest_sha256=data_manifest_sha256)
    if adaptive_config is not None:
        require_adaptive(adaptive_config, source_manifest_sha256=source_manifest_sha256, data_manifest_sha256=data_manifest_sha256)
    if jepa:
        require_jepa(config, source_manifest_sha256=source_manifest_sha256, data_manifest_sha256=data_manifest_sha256)
    if config.get("m9_task_head_training_authorized") is not False:
        raise ValueError("Task-head training must remain explicitly unauthorized for this audit")
    return True


def _main(argv=None):
    parser = argparse.ArgumentParser(description="Fail-closed EncoderV3 training gate")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--adaptive-config", type=Path, default=None)
    parser.add_argument("--jepa", action="store_true")
    parser.add_argument("--optical", action="store_true", help="Check the independent E0 A+B1 admission only")
    parser.add_argument("--source-manifest-sha256", default=None)
    parser.add_argument("--data-manifest-sha256", default=None)
    args = parser.parse_args(argv)
    try:
        adaptive = json.loads(args.adaptive_config.read_text(encoding="utf-8-sig")) if args.adaptive_config else None
        config = json.loads(args.config.read_text(encoding="utf-8-sig"))
        if args.optical:
            require_optical_selfsup_training(config, source_manifest_sha256=args.source_manifest_sha256, data_manifest_sha256=args.data_manifest_sha256)
        else:
            require_all(
                config,
                adaptive_config=adaptive,
                jepa=args.jepa,
                source_manifest_sha256=args.source_manifest_sha256,
                data_manifest_sha256=args.data_manifest_sha256,
            )
    except ValueError as exc:
        message = str(exc)
        print(message if message.startswith(("Training blocked", "JEPA blocked", "Optical self-supervised training blocked")) else "Training blocked: " + message, file=sys.stderr)
        return 2
    print("OPTICAL_TRAINING_AUTHORIZED" if args.optical else "TRAINING_AUTHORIZED")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
