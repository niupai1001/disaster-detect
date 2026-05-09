from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - exercised only when PyYAML is absent
    yaml = None


REQUIRED_KEYS = {
    "target_outcome",
    "contribution_type",
    "disaster_scope",
    "primary_backbone",
    "extension_backbone",
    "per_disaster_reporting",
    "category_unknown_required",
    "sealed_test_policy",
    "baseline_ladder",
    "stages",
    "acceptance",
}

REQUIRED_BASELINES = {
    "classical_remote_sensing",
    "deep_segmentation",
    "foundation_prithvi",
    "specialist_upper_bound",
    "category_unknown_lower_bound",
    "disaster_aware_prithvi",
}


def load_research_route(path: Path | str) -> dict[str, Any]:
    route_path = Path(path)
    text = route_path.read_text(encoding="utf-8")
    if route_path.suffix.lower() in {".yaml", ".yml"} and yaml is not None:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Research route {route_path} must contain a mapping")
    validate_research_route(data)
    return data


def validate_research_route(route: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_KEYS - set(route))
    if missing:
        raise ValueError(f"Research route missing required keys: {', '.join(missing)}")

    if route["target_outcome"] != "paper_experiment":
        raise ValueError("Research route target_outcome must be paper_experiment")
    if route["contribution_type"] != "method_plus_experiment":
        raise ValueError("Research route contribution_type must be method_plus_experiment")
    if list(route["disaster_scope"]) != ["C2", "C5"]:
        raise ValueError("Research route must preserve C2 and C5 in that order")
    if str(route["primary_backbone"]).lower() != "prithvi":
        raise ValueError("Research route primary_backbone must be Prithvi")
    if str(route["extension_backbone"]).lower() != "terramind":
        raise ValueError("Research route extension_backbone must be TerraMind")
    if route["per_disaster_reporting"] is not True:
        raise ValueError("Research route must require per-disaster reporting")
    if route["category_unknown_required"] is not True:
        raise ValueError("Research route must require category-unknown evaluation")
    if route["sealed_test_policy"] != "sealed_until_final":
        raise ValueError("Research route sealed test policy must remain sealed_until_final")

    baseline_ladder = set(route["baseline_ladder"])
    missing_baselines = sorted(REQUIRED_BASELINES - baseline_ladder)
    if missing_baselines:
        raise ValueError(f"Research route missing required baselines: {', '.join(missing_baselines)}")

    stage_ids = [stage.get("id") for stage in route["stages"] if isinstance(stage, dict)]
    expected_stage_ids = [f"E{index}" for index in range(8)]
    if stage_ids != expected_stage_ids:
        raise ValueError(f"Research route stages must be {', '.join(expected_stage_ids)}")

    acceptance = route["acceptance"]
    if not isinstance(acceptance, dict):
        raise ValueError("Research route acceptance must contain a mapping")
    if acceptance.get("c2_improvement_required") is not True:
        raise ValueError("Research route acceptance must require C2 improvement")
    if acceptance.get("fallback_if_no_c2_gain") != "analyze_data_label_modality_bottleneck":
        raise ValueError("Research route acceptance must define the C2 failure fallback")


def summarize_research_route(route: dict[str, Any]) -> dict[str, Any]:
    validate_research_route(route)
    return {
        "target_outcome": route["target_outcome"],
        "disaster_scope": route["disaster_scope"],
        "primary_backbone": route["primary_backbone"],
        "extension_backbone": route["extension_backbone"],
        "stages": [stage["id"] for stage in route["stages"]],
        "baseline_count": len(route["baseline_ladder"]),
        "sealed_test_policy": route["sealed_test_policy"],
    }
