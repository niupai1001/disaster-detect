from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def select_post_disaster_scene(
    row: dict[str, str],
    *,
    required_channels: Iterable[str],
    max_days_after_event: int | None = None,
    selection_splits: tuple[str, ...] = ("train", "validation"),
    selection_strategy: str = "earliest",
    scene_quality_scores: dict[str, dict[str, str]] | None = None,
) -> dict[str, str]:
    required = tuple(required_channels)
    if selection_strategy not in {"earliest", "quality_then_earliest"}:
        raise ValueError(f"Unsupported post-disaster scene selection strategy {selection_strategy!r}")
    if row.get("split") not in selection_splits:
        return _blocked(row, required, "blocked_sealed_test_split" if row.get("split") == "test" else "blocked_split")
    scenes = _candidate_scenes(row)
    if not scenes:
        return _blocked(row, required, "blocked_missing_candidate_scenes")

    complete_post_scenes = []
    saw_post = False
    for scene in scenes:
        days = _parse_int(str(scene.get("days_after_event", "")))
        image_date = str(scene.get("image_date", ""))
        event_date = str(row.get("event_date", scene.get("event_date", "")))
        if days is None and event_date and image_date:
            days = _days_between(event_date, image_date)
        if days is None:
            continue
        if days < 0:
            continue
        saw_post = True
        if max_days_after_event is not None and days > max_days_after_event:
            continue
        band_paths = scene.get("band_paths", {})
        if not isinstance(band_paths, dict):
            continue
        missing = [channel for channel in required if channel not in band_paths or not str(band_paths[channel])]
        if missing:
            continue
        quality = _quality_for_scene(scene, scene_quality_scores or {})
        complete_post_scenes.append((days, image_date, scene, band_paths, quality))

    if not complete_post_scenes:
        blocker = "blocked_missing_same_date_required_channels" if saw_post else "blocked_no_post_disaster_scene"
        return _blocked(row, required, blocker)

    if selection_strategy == "quality_then_earliest":
        days, image_date, scene, band_paths, quality = sorted(
            complete_post_scenes,
            key=lambda item: (_quality_value(item[4].get("quality_score", "")), item[0], item[1]),
        )[0]
    else:
        days, image_date, scene, band_paths, quality = sorted(complete_post_scenes, key=lambda item: (item[0], item[1]))[0]
    selected_paths = {channel: str(band_paths[channel]) for channel in required}
    temporal_role = "same_day" if days == 0 else "post"
    temporal_qa_flags = "post_disaster_same_date_channels"
    if selection_strategy == "quality_then_earliest":
        temporal_qa_flags += ";quality_ranked_cloud_proxy"
    return {
        **_base(row, required),
        "selection_status": "selected",
        "selected_image_date": image_date,
        "days_after_event": str(days),
        "temporal_role": temporal_role,
        "selected_scene_id": str(scene.get("scene_id", f"IM{image_date}" if image_date else "")),
        "channel_group_id": _channel_group_id(row, image_date, required),
        "raw_channel_paths": _format_band_paths(selected_paths),
        "input_channels": ";".join(required),
        "input_band_paths": _format_band_paths(selected_paths),
        "selection_strategy": selection_strategy,
        "quality_score": str(quality.get("quality_score", "")),
        "quality_score_status": str(quality.get("quality_score_status", "")),
        "quality_score_detail": str(quality.get("quality_score_detail", "")),
        "temporal_qa_flags": temporal_qa_flags,
        "blocker": "",
    }


def _candidate_scenes(row: dict[str, str]) -> list[dict]:
    value = row.get("candidate_scenes", "")
    if not value:
        return []
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


def _blocked(row: dict[str, str], required_channels: tuple[str, ...], blocker: str) -> dict[str, str]:
    return {
        **_base(row, required_channels),
        "selection_status": blocker,
        "selected_image_date": "",
        "days_after_event": "",
        "temporal_role": "",
        "selected_scene_id": "",
        "channel_group_id": "",
        "raw_channel_paths": "",
        "input_channels": ";".join(required_channels),
        "input_band_paths": "",
        "selection_strategy": "",
        "quality_score": "",
        "quality_score_status": "",
        "quality_score_detail": "",
        "temporal_qa_flags": blocker,
        "blocker": blocker,
    }


def _base(row: dict[str, str], required_channels: tuple[str, ...]) -> dict[str, str]:
    return {
        "sample_id": row.get("sample_id", ""),
        "event_id": row.get("event_id", ""),
        "event_date": row.get("event_date", ""),
        "class_id": row.get("class_id", ""),
        "class_name": row.get("class_name", ""),
        "split": row.get("split", ""),
        "mask_path": row.get("mask_path", ""),
        "required_channels": ";".join(required_channels),
    }


def _format_band_paths(band_paths: dict[str, str]) -> str:
    return ";".join(f"{channel}:{band_paths[channel]}" for channel in band_paths)


def _channel_group_id(row: dict[str, str], image_date: str, required_channels: tuple[str, ...]) -> str:
    sample_id = row.get("sample_id", "sample")
    channels = "-".join(required_channels)
    return f"{sample_id}_IM{image_date}_{channels}"


def _quality_for_scene(scene: dict, scene_quality_scores: dict[str, dict[str, str]]) -> dict[str, str]:
    keys = [str(scene.get("scene_id", "")), str(scene.get("image_date", ""))]
    for key in keys:
        if key and key in scene_quality_scores:
            return scene_quality_scores[key]
    return {
        "quality_score": "1",
        "quality_score_status": "unscored",
        "quality_score_detail": "no_quality_score_available",
    }


def _quality_value(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1.0


def _parse_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _days_between(event_date: str, image_date: str) -> int | None:
    from datetime import datetime

    try:
        event = datetime.strptime(event_date, "%Y%m%d")
        image = datetime.strptime(image_date, "%Y%m%d")
    except ValueError:
        return None
    return int((image - event).days)
