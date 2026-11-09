"""Structured synthetic approval records for gate unit tests only.

These records make the gate code testable without weakening the checked-in
configuration: every record is bound to the substantive config payload, and the
real config remains unauthorized until a user writes matching records.
"""
from .gates import ADAPTIVE_IDS, GATE_IDS, JEPA_IDS, payload_sha256


def approval(config, gate_id, *, scope="formal_training", data_sha256=None, **overrides):
    record = dict(
        gate_id=gate_id,
        status="APPROVED",
        scope=scope,
        approval_reference="synthetic://formal-readiness-audit/{}".format(gate_id),
        source_sha256="a" * 64,
        data_sha256=data_sha256,
        config_sha256=payload_sha256(config),
        tests=["synthetic://formal-readiness-audit/{}".format(gate_id)],
        reviewer="user",
        approved_at="2026-11-07T00:00:00Z",
    )
    record.update(overrides)
    return record


def populate(config, *, jepa=False):
    config.setdefault("required_before_training", {})
    for gate_id in sorted(GATE_IDS):
        config["required_before_training"][gate_id] = approval(config, gate_id)
    if jepa:
        config.setdefault("required_before_jepa", {})
        for gate_id in sorted(JEPA_IDS):
            config["required_before_jepa"][gate_id] = approval(config, gate_id, scope="formal_jepa")
    return config


def populate_adaptive(adaptive_config):
    adaptive_config.setdefault("required_before_training", {})
    for gate_id in sorted(ADAPTIVE_IDS):
        adaptive_config["required_before_training"][gate_id] = approval(adaptive_config, gate_id)
    return adaptive_config
