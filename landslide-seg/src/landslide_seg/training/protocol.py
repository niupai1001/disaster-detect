"""Frozen experiment inputs; no data inference and no mutation at startup."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..config import file_sha256, resolve_path


def canonical_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def validate_protocol(cfg: dict, records: list[dict], stats: dict) -> dict | None:
    s = cfg["segformer"]
    path = s["data"].get("protocol")
    if not path:
        return None
    p = resolve_path(path)
    protocol = json.loads(p.read_text(encoding="utf-8"))
    payload = {k: v for k, v in protocol.items() if k != "sha256"}
    if not protocol.get("frozen") or canonical_hash(payload) != protocol.get("sha256"):
        raise ValueError("Protocol is not frozen or its hash has changed")
    for key in ("manifest", "stats"):
        actual = resolve_path(s["data"][key])
        expected = protocol["files"].get(str(actual).replace("\\", "/"))
        if expected is None or file_sha256(actual) != expected:
            raise ValueError(f"Unregistered or modified {key}: {actual}")
    ids = [r["sample_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample_id")
    for r in records:
        member = protocol["members"].get(r["sample_id"])
        if member is None:
            raise ValueError(f"Unknown sample: {r['sample_id']}")
        for field in ("event_id", "cache", "cache_sha256", "date", "time_index"):
            if r.get(field) != member.get(field):
                raise ValueError(f"Changed sample identity: {r['sample_id']} {field}")
        if r["split"] not in ("train", "val") or member["role"] != r["split"]:
            raise ValueError(f"Role mismatch: {r['sample_id']}")
    train_ids = {r["sample_id"] for r in records if r["split"] == "train"}
    val_ids = {r["sample_id"] for r in records if r["split"] == "val"}
    if val_ids != set(protocol["dev_ids"]):
        raise ValueError("DEV membership is incomplete")
    fit_ids = set(stats.get("fit_sample_ids", []))
    if not fit_ids or not fit_ids <= train_ids or fit_ids & val_ids:
        raise ValueError("Normalization fit members must be a nonempty subset of this train pool")
    if set(stats["fit_events"]) != {protocol["members"][sid]["event_id"] for sid in fit_ids}:
        raise ValueError("Normalization fit event identity mismatch")
    fraction = int(s["data"]["pool_fraction"])
    if train_ids != set(protocol["pools"][str(fraction)]):
        raise ValueError("Training membership does not match frozen pool")
    return {"path": str(p), "sha256": protocol["sha256"], "n_fit": len(fit_ids), "n_train": len(train_ids), "n_dev": len(val_ids)}
