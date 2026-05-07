from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Iterable


IGNORED_DIRS = {".git", ".idea", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
IGNORED_FILES = {".DS_Store"}

CATEGORY_BY_ROOT = {
    "src": "source",
    "configs": "configuration",
    "tests": "tests",
    "docs": "documentation",
    "AGENTS.md": "project_instructions",
    ".gitignore": "project_instructions",
    "scripts": "operations",
    ".agent-team": "runtime_ledger",
    "workspace": "generated_workspace",
    "database": "local_data",
    "datasets": "local_data",
    "data": "local_data",
    "runs": "generated_runs",
    "cloud_runs": "generated_runs",
    "checkpoints": "model_artifacts",
}


def collect_inventory(root: Path | str, *, max_depth: int | None = None) -> dict:
    root_path = Path(root).resolve()
    top_level: dict[str, dict] = {}
    extensions: Counter[str] = Counter()
    total_files = 0
    total_bytes = 0

    for path in _walk_files(root_path, max_depth=max_depth):
        relative = path.relative_to(root_path)
        parts = relative.parts
        if not parts:
            continue
        root_name = parts[0]
        size = _file_size(path)
        extension = path.suffix or "[no_ext]"
        extensions[extension] += 1
        total_files += 1
        total_bytes += size

        item = top_level.setdefault(
            root_name,
            {
                "path": root_name,
                "category": CATEGORY_BY_ROOT.get(root_name, "other"),
                "files": 0,
                "bytes": 0,
            },
        )
        item["files"] += 1
        item["bytes"] += size

    roots = sorted(top_level.values(), key=lambda item: item["path"])
    largest_roots = sorted(roots, key=lambda item: item["bytes"], reverse=True)[:10]
    return {
        "root": str(root_path),
        "total_files": total_files,
        "total_bytes": total_bytes,
        "top_level": roots,
        "largest_roots": largest_roots,
        "extensions": dict(sorted(extensions.items(), key=lambda item: (-item[1], item[0]))),
    }


def format_markdown(report: dict) -> str:
    lines = [
        "# Project Inventory",
        "",
        f"- Root: `{report['root']}`",
        f"- Total files scanned: {report['total_files']}",
        f"- Total size scanned: {_format_bytes(report['total_bytes'])}",
        "",
        "## Top-Level Roots",
        "",
        "| Path | Category | Files | Size |",
        "| --- | --- | ---: | ---: |",
    ]
    for item in report["top_level"]:
        lines.append(
            f"| `{item['path']}` | {item['category']} | {item['files']} | {_format_bytes(item['bytes'])} |"
        )

    lines.extend(
        [
            "",
            "## Largest Roots",
            "",
            "| Path | Category | Files | Size |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for item in report["largest_roots"]:
        lines.append(
            f"| `{item['path']}` | {item['category']} | {item['files']} | {_format_bytes(item['bytes'])} |"
        )

    lines.extend(["", "## File Extensions", "", "| Extension | Files |", "| --- | ---: |"])
    for extension, count in report["extensions"].items():
        lines.append(f"| `{extension}` | {count} |")
    lines.append("")
    return "\n".join(lines)


def _walk_files(root: Path, *, max_depth: int | None) -> Iterable[Path]:
    for current, dirnames, filenames in os.walk(root):
        current_path = Path(current)
        relative_parts = current_path.relative_to(root).parts
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if dirname not in IGNORED_DIRS
            and (max_depth is None or len(relative_parts) + 1 < max_depth)
        ]
        for filename in filenames:
            if filename in IGNORED_FILES:
                continue
            path = current_path / filename
            if _is_ignored(path, root):
                continue
            if max_depth is not None and len(path.relative_to(root).parts) > max_depth:
                continue
            yield path


def _is_ignored(path: Path, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return True
    return any(part in IGNORED_DIRS for part in parts)


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"
