from __future__ import annotations

import csv
import json
from pathlib import Path

from .manifest import (
    ModelInputRecord,
    format_band_paths,
    load_model_input_manifest,
    rebase_path,
    with_research_fields,
    write_manifest,
)


def build_c5_multiscene_manifest(
    *,
    manifest_path: Path,
    output_path: Path,
    required_channels: tuple[str, ...],
    top_k: int = 3,
    max_days_after_event: int = 30,
    cloud_data_root: Path,
) -> dict:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    records = load_model_input_manifest(manifest_path, required_channels=required_channels)
    _guard_parent_split_leakage(records)

    expanded: list[ModelInputRecord] = []
    blocked_rows = []
    for record in records:
        selected = _select_scene_candidates(
            record,
            required_channels=required_channels,
            top_k=top_k,
            max_days_after_event=max_days_after_event,
        )
        if not selected:
            blocked_rows.append(record.sample_id)
            continue
        for rank, scene in enumerate(selected, start=1):
            expanded.append(
                _expanded_record(
                    record,
                    scene=scene,
                    rank=rank,
                    required_channels=required_channels,
                    cloud_data_root=cloud_data_root,
                )
            )

    write_manifest(output_path, expanded)
    summary = {
        "source_manifest": str(manifest_path),
        "output_manifest": str(output_path),
        "source_records": len(records),
        "expanded_records": len(expanded),
        "blocked_records": len(blocked_rows),
        "top_k": top_k,
        "max_days_after_event": max_days_after_event,
        "required_channels": list(required_channels),
    }
    (output_path.parent / "multiscene_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _guard_parent_split_leakage(records: list[ModelInputRecord]) -> None:
    splits_by_parent: dict[str, set[str]] = {}
    for record in records:
        parent = record.row.get("parent_sample_id", "").strip() or record.sample_id
        splits_by_parent.setdefault(parent, set()).add(record.split)
    leaked = {parent: splits for parent, splits in splits_by_parent.items() if len(splits) > 1}
    if leaked:
        example = next(iter(sorted(leaked)))
        raise ValueError(f"parent sample split leakage: {example} appears in {sorted(leaked[example])}")


def _select_scene_candidates(
    record: ModelInputRecord,
    *,
    required_channels: tuple[str, ...],
    top_k: int,
    max_days_after_event: int,
) -> list[dict]:
    scenes = _candidate_scenes(record)
    candidates = []
    for scene in scenes:
        days = _parse_int(scene.get("days_after_event"))
        if days is None or days < 0 or days > max_days_after_event:
            continue
        if not _has_required_channels(scene, required_channels):
            continue
        candidates.append(scene)
    selected_scene_id = record.row.get("selected_scene_id", "")
    candidates.sort(
        key=lambda scene: (
            0 if scene.get("scene_id") == selected_scene_id else 1,
            _quality_value(scene.get("quality_score", record.row.get("quality_score", "1"))),
            _parse_int(scene.get("days_after_event")) or 10**9,
            str(scene.get("image_date", "")),
        )
    )
    return candidates[:top_k]


def _candidate_scenes(record: ModelInputRecord) -> list[dict]:
    value = record.row.get("candidate_scenes", "")
    if not value:
        return []
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return []
    return [scene for scene in payload if isinstance(scene, dict)] if isinstance(payload, list) else []


def _has_required_channels(scene: dict, required_channels: tuple[str, ...]) -> bool:
    band_paths = scene.get("band_paths")
    return isinstance(band_paths, dict) and all(channel in band_paths and str(band_paths[channel]) for channel in required_channels)


def _expanded_record(
    record: ModelInputRecord,
    *,
    scene: dict,
    rank: int,
    required_channels: tuple[str, ...],
    cloud_data_root: Path,
) -> ModelInputRecord:
    scene_id = str(scene.get("scene_id", scene.get("image_date", f"rank{rank}")))
    band_paths = {
        channel: rebase_path(str(scene["band_paths"][channel]), local_root=Path.cwd(), target_root=cloud_data_root)
        for channel in required_channels
    }
    row = dict(record.row)
    row.update(
        {
            "sample_id": f"{record.sample_id}__{scene_id}",
            "parent_sample_id": record.sample_id,
            "multiscene_rank": str(rank),
            "selected_scene_id": scene_id,
            "selected_image_date": str(scene.get("image_date", "")),
            "days_after_event": str(scene.get("days_after_event", "")),
            "temporal_role": "same_day" if str(scene.get("days_after_event", "")) == "0" else "post",
            "input_channels": ";".join(required_channels),
            "input_band_paths": format_band_paths(band_paths),
            "raw_channel_paths": format_band_paths(band_paths),
            "selection_strategy": f"top{rank}_multiscene_post_quality_safe",
            "quality_score": str(scene.get("quality_score", record.row.get("quality_score", ""))),
            "quality_score_status": str(scene.get("quality_score_status", record.row.get("quality_score_status", ""))),
            "quality_score_detail": str(scene.get("quality_score_detail", record.row.get("quality_score_detail", ""))),
            "temporal_qa_flags": "post_disaster_same_date_channels;multiscene_expanded;parent_split_inherited",
        }
    )
    expanded = ModelInputRecord(
        sample_id=row["sample_id"],
        event_id=record.event_id,
        split=record.split,
        class_id=record.class_id,
        class_name=record.class_name,
        mask_path=record.mask_path,
        input_channels=required_channels,
        input_band_paths=band_paths,
        row=row,
    )
    return with_research_fields(expanded)


def _parse_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _quality_value(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1.0
