"""Candidate formal AB schedule builder; it never approves or writes a release."""
from __future__ import annotations

import hashlib
import json
import math

from .prepare_pilot import bounds
from .tasks import hidden_band_indices, task_rng

SOURCE_FAMILY = {"S2": "S2", "L8": "Landsat", "L9": "Landsat", "Landsat": "Landsat"}
FAMILIES = ("S2", "Landsat")


def _mask_rule(mask_rule):
    """The single reviewed rectangle rule; both schedule builders share it."""
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
    return dict(mask_rule, aspect_ratio=aspect)


def _rectangle_in(left, bottom, right, top, patch_m, *, seed, update, identity, stream, rule):
    nx = math.ceil((right - left) / patch_m)
    ny = math.ceil((top - bottom) / patch_m)
    rng = task_rng(seed, update, identity, stream)
    low, high = rule["aspect_ratio"]
    ratio = math.exp(rng.uniform(math.log(low), math.log(high)))
    width = max(1, min(nx, round(math.sqrt(rule["target_patch_fraction"] * nx * ny * ratio))))
    height = max(1, min(ny, round(math.sqrt(rule["target_patch_fraction"] * nx * ny / ratio))))
    x = rng.randint(0, nx - width)
    y = rng.randint(0, ny - height)
    return [left + x * patch_m, bottom + y * patch_m, left + (x + width) * patch_m, bottom + (y + height) * patch_m]


def _rectangle(row, patch_m, *, seed, update, rule):
    left, bottom, right, top = bounds(row)
    return _rectangle_in(left, bottom, right, top, patch_m, seed=seed, update=update, identity=row["id"], stream="A_rectangle_formal_candidate", rule=rule)


def observation_bounds(observation):
    """Physical envelope of the loaded native bands; no row metadata needed."""
    boxes = []
    for band in observation.bands:
        a, _, left, _, e, top = band.affine
        h, w = band.reflectance.shape
        boxes.append((left, top + e * h, left + a * w, top))
    return min(value[0] for value in boxes), min(value[1] for value in boxes), max(value[2] for value in boxes), max(value[3] for value in boxes)


def optical_train_pool(dataset, *, min_bands=2, max_native_pixels=16384):
    """Frozen TRAIN member pool with an explicit native-scene compute cap.

    The cap is a data-property rule, not a silent filter: every excluded member
    is counted by sensor/band-count and the counts are frozen into the recipe.
    A scene above the cap is not trainable under the frozen full-attention
    Transformer inside the declared memory budget, so it needs an explicit
    research decision (larger budget, crop protocol or an efficient attention
    path) instead of being dropped quietly.
    """
    if getattr(dataset, "role", None) != "TRAIN":
        raise ValueError("Only a TRAIN release can define the E0 training pool")
    if type(min_bands) is not int or min_bands < 2:
        raise ValueError("E0 needs at least one visible and one hidden band")
    if type(max_native_pixels) is not int or max_native_pixels < 1:
        raise ValueError("Positive native-pixel cap required")
    eligible, excluded = [], {"below_min_bands": 0, "unknown_sensor": 0, "above_native_pixel_cap": 0}
    excluded_by_sensor = {}
    for row in dataset.rows:
        if row.get("role") != "TRAIN":
            raise ValueError("Pool construction refuses non-TRAIN members")
        family = SOURCE_FAMILY.get(row.get("sensor"))
        if family is None:
            excluded["unknown_sensor"] += 1
            continue
        pixels = max(band["shape"][0] * band["shape"][1] for band in row["bands"])
        if len(row["bands"]) < min_bands:
            excluded["below_min_bands"] += 1
            continue
        if pixels > max_native_pixels:
            excluded["above_native_pixel_cap"] += 1
            key = f"{row['sensor']}:{len(row['bands'])}bands"
            excluded_by_sensor[key] = excluded_by_sensor.get(key, 0) + 1
            continue
        eligible.append([row["id"], family, len(row["bands"]), row["cache_sha256"]])
    eligible.sort()
    families = {family: [row[0] for row in eligible if row[1] == family] for family in FAMILIES}
    payload = json.dumps(eligible, separators=(",", ":"), allow_nan=False).encode()
    return dict(members=eligible, families=families, sha256=hashlib.sha256(payload).hexdigest(), excluded=excluded,
                excluded_by_sensor=dict(sorted(excluded_by_sensor.items())), max_native_pixels=max_native_pixels,
                min_bands=min_bands)


def ranked_members(seed, update, family, ids, count):
    """Deterministic k-of-n without replacement: hash rank of (seed, update, member)."""
    if type(count) is not int or count < 1 or count > len(ids):
        raise ValueError("Requested slot count exceeds the frozen source pool")
    prefix = hashlib.sha256(f"{seed}:{update}:{family}".encode()).digest()
    ranked = sorted(ids, key=lambda identifier: hashlib.sha256(prefix + identifier.encode()).digest())
    return ranked[:count]


def build_optical_selfsup_schedule(dataset, *, updates, seed, slots_per_source, mask_rule, p_values=(60.0, 120.0), pool=None, max_native_pixels=16384):
    """Compact deterministic recipe for E0; per-update tensors/masks are never stored."""
    if type(updates) is not int or not 1 <= updates <= 10000:
        raise ValueError("Positive update count within the route cap required")
    if type(seed) is not int or type(slots_per_source) is not int or slots_per_source < 1 or 2 * slots_per_source != 8:
        raise ValueError("Integer seed and four slots per source (eight distinct members) required")
    normalized = _mask_rule(mask_rule)
    if not p_values or any(value not in (60.0, 120.0) for value in p_values):
        raise ValueError("Explicit P sequence from the frozen 60/120 panel required")
    pool = optical_train_pool(dataset, max_native_pixels=max_native_pixels) if pool is None else pool
    for family in FAMILIES:
        if len(pool["families"][family]) < slots_per_source:
            raise ValueError(f"Insufficient {family} TRAIN members for the frozen slot count")
    return dict(
        kind="optical_selfsup_sampling_recipe_v1",
        status="ENGINEERING_FROZEN_RESEARCH_APPROVAL_PENDING",
        source_identities=dataset.identities,
        pool=dict(rule="all TRAIN optical members with >=2 native bands and native scene pixels <= max_native_pixels",
                  min_bands=int(pool["min_bands"]), max_native_pixels=int(pool["max_native_pixels"]),
                  sha256=pool["sha256"], members={family: len(pool["families"][family]) for family in FAMILIES},
                  excluded=pool["excluded"], excluded_by_sensor=pool["excluded_by_sensor"],
                  release_members=len(dataset.rows)),
        sampling=dict(seed=seed, slots_per_source=slots_per_source, members_per_update=8, families=list(FAMILIES),
                      method="without-replacement hash ranking of sha256(seed:update:family) over the frozen pool"),
        updates=updates,
        P_values=list(p_values),
        mask_rule=normalized,
        tasks=dict(A="rectangle spatial hide; targets are cells fully inside the rectangle", B1="native visible-band holdout; every target band leaves the Encoder"),
        B2_used=False,
        builder=dict(writes_per_update_tensors=False),
    )


def build_formal_candidate_schedule(dataset, members, *, updates, seed, mask_rule, p_values=(60.0, 120.0)):
    """Build a loader-compatible candidate document from explicit frozen inputs."""
    if type(updates) is not int or updates < 1 or updates > 10000:
        raise ValueError("Positive update count within the route cap required")
    if type(seed) is not int:
        raise ValueError("Integer seed required")
    normalized = _mask_rule(mask_rule)
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
