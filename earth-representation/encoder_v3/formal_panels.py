"""Validation-panel document builder; caller supplies every case, builder only validates."""
from __future__ import annotations

import math


def build_validation_panel_document(dataset, cases, *, pair_admissions=None):
    """Build a loader-compatible panel from explicit cases; no case choice here."""
    if dataset.role != "VAL":
        raise ValueError("Only an explicit VAL release can build a validation panel")
    rows = {row["id"]: row for row in dataset.rows}
    if len(rows) != len(dataset.rows):
        raise ValueError("Duplicate VAL member identities")
    eligible = {identifier: row["cache_sha256"] for identifier, row in rows.items()}
    pair_admissions = {} if pair_admissions is None else dict(pair_admissions)
    panels = []
    identifiers = set()
    for case in cases:
        panel_id = case.get("panel_id")
        if not isinstance(panel_id, str) or not panel_id or panel_id in identifiers:
            raise ValueError("Unique explicit panel_id required")
        identifiers.add(panel_id)
        source = case.get("source_id")
        kind = case.get("kind")
        P_m = case.get("P_m")
        origin = case.get("origin_xy")
        if source not in rows or kind not in ("A", "B1", "B2") or P_m not in (60.0, 120.0):
            raise ValueError("Panel source/kind/P invalid")
        if not isinstance(origin, (list, tuple)) or len(origin) != 2 or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in origin):
            raise ValueError("Finite native origin required")
        normalized = dict(panel_id=panel_id, source_id=source, kind=kind, P_m=float(P_m), origin_xy=[float(value) for value in origin])
        if kind == "A":
            rectangle = case.get("rectangle")
            if not isinstance(rectangle, (list, tuple)) or len(rectangle) != 4 or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in rectangle) or rectangle[0] >= rectangle[2] or rectangle[1] >= rectangle[3]:
                raise ValueError("Explicit finite A rectangle required")
            normalized["rectangle"] = [float(value) for value in rectangle]
        elif kind == "B1":
            bands = [band["band_key"] for band in rows[source]["bands"]]
            hidden = case.get("hidden_band_ids")
            if not isinstance(hidden, list) or not hidden or len(set(hidden)) != len(hidden) or not set(hidden) < set(bands):
                raise ValueError("Explicit B1 hidden bands must leave visible support")
            normalized["hidden_band_ids"] = list(hidden)
        else:
            pair_id = case.get("pair_id")
            if not isinstance(pair_id, str) or pair_id not in pair_admissions:
                raise ValueError("B2 panel case needs an explicit admitted pair")
            normalized["pair_id"] = pair_id
        panels.append(normalized)
    if not panels:
        raise ValueError("At least one explicit validation case required")
    return dict(kind="frozen_validation_panels_v1", source_identities=dataset.identities,
                eligible_members=eligible, panels=panels, pair_admissions=pair_admissions,
                builder=dict(status="CANDIDATE_NOT_APPROVED", cases=len(panels)))
