"""Candidate formal AB schedule builder; it never approves or writes a release."""
from __future__ import annotations

import math

from .prepare_pilot import bounds
from .tasks import hidden_band_indices, task_rng


def _rectangle(row, patch_m, *, seed, update, rule):
    left, bottom, right, top = bounds(row)
    nx = math.ceil((right - left) / patch_m)
    ny = math.ceil((top - bottom) / patch_m)
    rng = task_rng(seed, update, row["id"], "A_rectangle_formal_candidate")
    low, high = rule["aspect_ratio"]
    ratio = math.exp(rng.uniform(math.log(low), math.log(high)))
    width = max(1, min(nx, round(math.sqrt(rule["target_patch_fraction"] * nx * ny * ratio))))
    height = max(1, min(ny, round(math.sqrt(rule["target_patch_fraction"] * nx * ny / ratio))))
    x = rng.randint(0, nx - width)
    y = rng.randint(0, ny - height)
    return [left + x * patch_m, bottom + y * patch_m, left + (x + width) * patch_m, bottom + (y + height) * patch_m]


def build_formal_candidate_schedule(dataset, members, *, updates, seed, mask_rule, p_values=(60.0, 120.0)):
    """Build a loader-compatible candidate document from explicit frozen inputs."""
    if type(updates) is not int or updates < 1 or updates > 10000:
        raise ValueError("Positive update count within the route cap required")
    if type(seed) is not int:
        raise ValueError("Integer seed required")
    if not isinstance(mask_rule, dict) or set(mask_rule) != {"count", "target_patch_fraction", "aspect_ratio", "position", "size_rounding"}:
        raise ValueError("Explicit mask rule required")
    aspect = mask_rule["aspect_ratio"]
    if aspect == "log-uniform [0.5,2.0]":
        aspect = (0.5, 2.0)
    if (mask_rule["count"] != 1 or not isinstance(mask_rule["target_patch_fraction"], (int, float)) or not 0 < mask_rule["target_patch_fraction"] <= 1
            or not isinstance(aspect, (list, tuple)) or len(aspect) != 2
            or not all(isinstance(value, (int, float)) and value > 0 and math.isfinite(value) for value in aspect)
            or mask_rule["position"] != "uniform feasible patch-aligned rectangle"
            or mask_rule["size_rounding"] != "nearest integer patch count, clipped to observation patch layout"):
        raise ValueError("Mask rule values differ from the reviewed preflight rule")
    normalized = dict(mask_rule, aspect_ratio=aspect)
    if not p_values or any(value not in (60.0, 120.0) for value in p_values):
        raise ValueError("Explicit P sequence from the frozen 60/120 panel required")
    members = list(members)
    if len(members) != 8 or len({row["id"] for row in members}) != 8:
        raise ValueError("Exactly eight distinct TRAIN members required")
    if any(row.get("role") != "TRAIN" for row in members):
        raise ValueError("Only TRAIN members belong in the AB schedule")
    eligible = {row["id"]: row["cache_sha256"] for row in members}
    entries = []
    for update in range(updates):
        patch = p_values[update % len(p_values)]
        slots = []
        for row in members:
            hidden = sorted(hidden_band_indices(len(row["bands"]), seed=seed, update=update, observation_id=row["id"]))
            left, bottom, _, _ = bounds(row)
            slot = dict(
                source_id=row["id"], origin_xy=[left, bottom], A_rectangle=_rectangle(row, patch, seed=seed, update=update, rule=normalized),
                B=dict(kind="B1", hidden_band_ids=[row["bands"][i]["band_key"] for i in hidden]) if len(row["bands"]) > 1 else dict(kind="B1_ABSENT"),
            )
            slots.append(slot)
        entries.append(dict(logical_update=update, P_m=patch, slots=slots))
    return dict(
        kind="frozen_AB_schedule_v1", status="CANDIDATE_NOT_APPROVED",
        source_identities=dataset.identities, eligible_members=eligible,
        pair_admissions={}, updates=entries,
        builder=dict(seed=seed, updates=updates, p_values=list(p_values), mask_rule=mask_rule),
    )
