from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - exercised only when PyYAML is absent
    yaml = None


REQUIRED_CONFIG_KEYS = {
    "task_id",
    "experiment_id",
    "class_scope",
    "input_channels",
    "split_policy",
    "model",
    "metrics",
    "cloud",
}


def load_config(path: Path | str) -> dict[str, Any]:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config {config_path} must contain a mapping")
    validate_config(data)
    return data


def validate_config(config: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_CONFIG_KEYS - set(config))
    if missing:
        raise ValueError(f"Config missing required keys: {', '.join(missing)}")
    if config["task_id"] != "task-afc7f2c25f8f":
        raise ValueError("Config task_id must remain task-afc7f2c25f8f")
    if config["class_scope"] not in {"binary_c2", "binary_c5", "multiclass_c2_c5"}:
        raise ValueError(f"Unsupported class_scope {config['class_scope']!r}")
    if list(config["input_channels"]) != ["F16", "F17"]:
        raise ValueError("v0.1 configs must use input_channels [F16, F17]")
    split_policy = config["split_policy"]
    if split_policy.get("allow_test_split_for_selection") is True:
        raise ValueError("Test split may not be used for model selection")
    if "validation" not in split_policy.get("selection_splits", []):
        raise ValueError("Validation split must be used for route selection")
    cloud = config["cloud"]
    if cloud.get("local_full_training_allowed") is True:
        raise ValueError("Local full training must remain disabled")

