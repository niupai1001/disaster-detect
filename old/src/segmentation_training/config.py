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
    if not str(config.get("task_id", "")).strip():
        raise ValueError("Config task_id is required")
    if config["class_scope"] not in {"binary_c2", "binary_c5", "multiclass_c2_c5"}:
        raise ValueError(f"Unsupported class_scope {config['class_scope']!r}")
    input_channels = list(config["input_channels"])
    if not input_channels or not all(str(channel).strip() for channel in input_channels):
        raise ValueError("input_channels must contain at least one channel")
    model_input_channels = config.get("model", {}).get("input_channels")
    if model_input_channels is not None and int(model_input_channels) != len(input_channels):
        raise ValueError("model.input_channels must match len(input_channels)")
    split_policy = config["split_policy"]
    if split_policy.get("test") != "sealed":
        raise ValueError("Test split must remain sealed")
    if split_policy.get("allow_test_split_for_selection") is True:
        raise ValueError("Test split may not be used for model selection")
    if "test" in split_policy.get("selection_splits", []):
        raise ValueError("Test split may not be used for model selection")
    if "validation" not in split_policy.get("selection_splits", []):
        raise ValueError("Validation split must be used for route selection")
    training = config.get("training", {})
    loss = training.get("loss")
    if loss is not None and loss not in {
        "cross_entropy",
        "weighted_cross_entropy",
        "cross_entropy_dice",
        "weighted_cross_entropy_dice",
    }:
        raise ValueError(f"Unsupported training.loss {loss!r}")
    sampler = training.get("sampler", {})
    train_policy = sampler.get("train_policy")
    if train_policy is not None and train_policy not in {"random", "foreground_biased"}:
        raise ValueError(f"Unsupported sampler train_policy {train_policy!r}")
    validation_policy = sampler.get("validation_policy")
    if validation_policy is not None and validation_policy not in {"center", "foreground_center"}:
        raise ValueError(f"Unsupported sampler validation_policy {validation_policy!r}")
    probability = sampler.get("foreground_probability")
    if probability is not None and not 0.0 <= float(probability) <= 1.0:
        raise ValueError("sampler foreground_probability must be between 0 and 1")
    min_foreground_pixels = sampler.get("min_foreground_pixels")
    if min_foreground_pixels is not None and int(min_foreground_pixels) < 0:
        raise ValueError("sampler min_foreground_pixels must be non-negative")
    min_foreground_fraction = sampler.get("min_foreground_fraction")
    if min_foreground_fraction is not None and not 0.0 <= float(min_foreground_fraction) <= 1.0:
        raise ValueError("sampler min_foreground_fraction must be between 0 and 1")
    max_attempts = sampler.get("max_attempts")
    if max_attempts is not None and int(max_attempts) < 1:
        raise ValueError("sampler max_attempts must be at least 1")
    diagnostics = config.get("diagnostics")
    if diagnostics is not None:
        if not diagnostics.get("parent_experiment_id"):
            raise ValueError("diagnostics.parent_experiment_id is required")
        if diagnostics.get("changed_factor") not in {"sampler", "loss_weighting", "loss_family", "observability"}:
            raise ValueError("diagnostics.changed_factor is unsupported")
    cloud = config["cloud"]
    if cloud.get("local_full_training_allowed") is True:
        raise ValueError("Local full training must remain disabled")
