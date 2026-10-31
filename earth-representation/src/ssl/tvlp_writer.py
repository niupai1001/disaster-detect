"""Minimal TVLP 1.0 writer.

The official `TrainingRunWriter` starts worker threads that crash this Windows
build as soon as a CUDA context exists (0xC0000005, reproducible with a plain
Linear model, CPU-only is fine). The protocol document allows writing the files
directly instead, so this module does exactly that: no threads, no background
state, plain JSON lines.

Files written under <root>/<run_id>/: run.json, metrics.jsonl, events.jsonl.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

TVLP = "1.0"


def _now():
    return datetime.now(timezone.utc).isoformat()


class TvlpRun:
    def __init__(self, root, *, project, name, metric_definitions,
                 config=None, dataset=None, model=None, seed=None,
                 experiment=None, run_id=None):
        self.run_id = run_id or f"{name}-{datetime.now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.dir = Path(root) / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        for key, value in metric_definitions.items():
            if value.get("direction") not in ("min", "max", "none"):
                raise ValueError(f"{key}: direction must be min/max/none")
        meta = {
            "schema_version": TVLP,
            "run_id": self.run_id,
            "project": project,
            "name": name,
            "created_at": _now(),
            "config": config or {},
            "metric_definitions": metric_definitions,
        }
        if experiment:
            meta["experiment"] = experiment
        if model:
            meta["model"] = {"name": model} if isinstance(model, str) else model
        if dataset:
            meta["dataset"] = {"name": dataset} if isinstance(dataset, str) else dataset
        if seed is not None:
            meta["source"] = {"seed": int(seed), "framework": "pytorch"}
        (self.dir / "run.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        self._metrics = open(self.dir / "metrics.jsonl", "a", encoding="utf-8")
        self._events = open(self.dir / "events.jsonl", "a", encoding="utf-8")
        self.lifecycle("started")

    def lifecycle(self, status, message=None):
        """TVLP lifecycle event: only these fields are allowed by the schema."""
        row = {"type": "lifecycle", "timestamp": _now(), "status": status}
        if message:
            row["message"] = message
        self._events.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._events.flush()
        os.fsync(self._events.fileno())

    def log(self, phase, metrics, global_step=None, wall_time_sec=None):
        row = {"timestamp": _now(), "phase": phase,
               "metrics": {k: float(v) for k, v in metrics.items()}}
        if global_step is not None:
            row["global_step"] = int(global_step)
        if wall_time_sec is not None:
            row["wall_time_sec"] = float(wall_time_sec)
        self._metrics.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._metrics.flush()

    def finish(self, status="completed", message=None):
        self.lifecycle(status, message)
        self._metrics.close()
        self._events.close()
