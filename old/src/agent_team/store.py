from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .models import DEFAULT_AGENTS


class TeamStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.base = self.root / ".agent-team"

    @property
    def agents_path(self) -> Path:
        return self.base / "config" / "agents.json"

    @property
    def events_path(self) -> Path:
        return self.base / "logs" / "events.jsonl"

    def init(self) -> None:
        for path in [
            self.base / "config",
            self.base / "tasks",
            self.base / "messages" / "inbox",
            self.base / "messages" / "outbox",
            self.base / "artifacts",
            self.base / "logs",
        ]:
            path.mkdir(parents=True, exist_ok=True)
        if not self.agents_path.exists():
            self.save_agents(DEFAULT_AGENTS)
        self.events_path.touch(exist_ok=True)

    def load_agents(self) -> list[dict]:
        self._require_init()
        agents = self._read_json(self.agents_path)
        normalized, changed = self._normalize_agents(agents)
        if changed:
            self.save_agents(normalized)
        return normalized

    def save_agents(self, agents: list[dict]) -> None:
        self._write_json(self.agents_path, agents)

    def get_agent(self, agent_id: str) -> dict:
        for agent in self.load_agents():
            if agent["agent_id"] == agent_id:
                return agent
        raise KeyError(f"Unknown agent: {agent_id}")

    def set_agent_state(self, agent_id: str, state: str) -> None:
        self.set_agent_assignment(agent_id, state)

    def set_agent_assignment(
        self,
        agent_id: str,
        state: str,
        task_id: str | None = None,
        assignment_id: str | None = None,
    ) -> None:
        agents = self.load_agents()
        found = False
        for agent in agents:
            if agent["agent_id"] == agent_id:
                agent["state"] = state
                if state == "idle":
                    agent["active_task_id"] = None
                    agent["active_assignment_id"] = None
                elif task_id and assignment_id:
                    agent["active_task_id"] = task_id
                    agent["active_assignment_id"] = assignment_id
                found = True
                break
        if not found:
            raise KeyError(f"Unknown agent: {agent_id}")
        self.save_agents(agents)

    def create_task_record(
        self,
        brief: str,
        stage: str,
        assignee: str | None,
        assignment_id: str | None,
        assignments: list[dict] | None = None,
    ) -> dict:
        task_id = f"task-{uuid4().hex[:12]}"
        now = self._now()
        if assignments is None and assignee and assignment_id:
            assignments = [
                {
                    "agent_id": assignee,
                    "assignment_id": assignment_id,
                    "stage": stage,
                    "status": "assigned",
                }
            ]
        task = {
            "task_id": task_id,
            "brief": brief,
            "status": "assigned",
            "current_stage": stage,
            "assignee": assignee,
            "assignment_id": assignment_id,
            "assignments": assignments or [],
            "artifacts": [],
            "created_at": now,
            "updated_at": now,
            "blocking_reason": "",
        }
        self.save_task(task)
        return task

    def load_task(self, task_id: str) -> dict:
        return self._read_json(self.base / "tasks" / f"{task_id}.json")

    def list_tasks(self) -> list[dict]:
        tasks_dir = self.base / "tasks"
        if not tasks_dir.exists():
            return []
        tasks = [self._read_json(path) for path in tasks_dir.glob("*.json")]
        return sorted(tasks, key=lambda task: (task["created_at"], task["task_id"]))

    def save_task(self, task: dict) -> None:
        task["updated_at"] = self._now()
        self._write_json(self.base / "tasks" / f"{task['task_id']}.json", task)

    def write_message(self, agent_id: str, box: str, message: dict, *, actor_id: str) -> dict:
        self._ensure_message_access(agent_id, box, actor_id=actor_id, write=True)
        message = dict(message)
        message.setdefault("message_id", f"msg-{uuid4().hex[:12]}")
        message.setdefault("timestamp", self._now())
        path = self.base / "messages" / box / f"{agent_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(message, ensure_ascii=False, sort_keys=True) + "\n")
        return message

    def read_messages(self, agent_id: str, box: str, *, actor_id: str) -> list[dict]:
        self._ensure_message_access(agent_id, box, actor_id=actor_id, write=False)
        path = self.base / "messages" / box / f"{agent_id}.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def write_artifact(
        self,
        task_id: str,
        stage: str,
        filename: str,
        content: str,
        *,
        zh_content: str | None = None,
    ) -> dict:
        artifact_dir = self.base / "artifacts" / task_id / stage
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / filename
        path.write_text(content, encoding="utf-8")
        artifact = {
            "artifact_id": f"artifact-{uuid4().hex[:12]}",
            "task_id": task_id,
            "stage": stage,
            "path": str(path),
            "created_at": self._now(),
        }
        if zh_content:
            zh_path = self._zh_archive_path(path)
            zh_path.write_text(zh_content, encoding="utf-8")
            artifact["zh_path"] = str(zh_path)
        return artifact

    def append_event(self, event_type: str, **fields) -> dict:
        event = {
            "event_id": f"event-{uuid4().hex[:12]}",
            "timestamp": self._now(),
            "event_type": event_type,
            **fields,
        }
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        return event

    def tail_events(self, limit: int = 20) -> list[dict]:
        if not self.events_path.exists():
            return []
        lines = [line for line in self.events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [json.loads(line) for line in lines[-limit:]]

    def _require_init(self) -> None:
        if not self.agents_path.exists():
            raise FileNotFoundError("Team store is not initialized. Run `team init` first.")

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _normalize_agents(agents: list[dict]) -> tuple[list[dict], bool]:
        changed = False
        existing_by_id = {agent["agent_id"]: agent for agent in agents}
        normalized = []
        for default in DEFAULT_AGENTS:
            agent = existing_by_id.pop(default["agent_id"], None)
            record = dict(default)
            if agent:
                record.update(agent)
                for key in ["capabilities", "allowed_actions"]:
                    merged = list(dict.fromkeys(default.get(key, []) + agent.get(key, [])))
                    if merged != agent.get(key, []):
                        changed = True
                    record[key] = merged
            for key in ["active_task_id", "active_assignment_id"]:
                if key not in record:
                    record[key] = None
                    changed = True
            normalized.append(record)
        for agent in existing_by_id.values():
            record = dict(agent)
            for key in ["active_task_id", "active_assignment_id"]:
                if key not in record:
                    record[key] = None
                    changed = True
            normalized.append(record)
        if existing_by_id:
            changed = True
        return normalized, changed

    @staticmethod
    def _ensure_message_access(agent_id: str, box: str, *, actor_id: str, write: bool) -> None:
        if box not in {"inbox", "outbox"}:
            raise PermissionError(f"Unknown message box: {box}")
        if actor_id == "orchestrator":
            return
        if write and not (actor_id == agent_id and box == "outbox"):
            raise PermissionError(f"{actor_id} cannot write {agent_id} {box}")
        if not write and actor_id != agent_id:
            raise PermissionError(f"{actor_id} cannot read {agent_id} {box}")

    @staticmethod
    def _read_json(path: Path):
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_json(path: Path, payload) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @staticmethod
    def _zh_archive_path(path: Path) -> Path:
        if path.suffix:
            return path.with_name(f"{path.stem}-zh{path.suffix}")
        return path.with_name(f"{path.name}-zh.md")
