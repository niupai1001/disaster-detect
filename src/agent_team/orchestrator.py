from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from uuid import uuid4

from .models import BRAINSTORM_NOTE_ARTIFACTS, STAGE_PARTICIPANTS, STAGE_PREREQUISITES
from .providers import MockProvider
from .store import TeamStore


class PolicyError(RuntimeError):
    pass


class Orchestrator:
    def __init__(self, store: TeamStore, provider: Any | None = None):
        self.store = store
        self.provider = provider or MockProvider()

    def create_task(self, brief: str, stage: str = "brainstorm") -> dict:
        self._ensure_stage(stage)
        self._ensure_prerequisites_for_new_task(stage)
        participants = STAGE_PARTICIPANTS[stage]
        self._ensure_agents_available(participants)
        assignments = self._new_assignments(stage, participants)
        assignee = assignments[0]["agent_id"] if len(assignments) == 1 else None
        assignment_id = assignments[0]["assignment_id"] if len(assignments) == 1 else None
        task = self.store.create_task_record(brief, stage, assignee, assignment_id, assignments)
        for assignment in assignments:
            self._assign(task, assignment)
        return task

    def assign_stage(self, task_id: str, stage: str) -> dict:
        self._ensure_stage(stage)
        task = self.store.load_task(task_id)
        self._ensure_prerequisites(task, stage)
        participants = STAGE_PARTICIPANTS[stage]
        self._ensure_agents_available(participants)
        assignments = self._new_assignments(stage, participants)
        task["current_stage"] = stage
        task["status"] = "assigned"
        task["assignee"] = assignments[0]["agent_id"] if len(assignments) == 1 else None
        task["assignment_id"] = assignments[0]["assignment_id"] if len(assignments) == 1 else None
        task["assignments"] = assignments
        task["blocking_reason"] = ""
        self.store.save_task(task)
        for assignment in assignments:
            self._assign(task, assignment)
        return task

    def run_task(self, task_id: str) -> dict:
        task = self.store.load_task(task_id)
        stage = task["current_stage"]
        assignments = [assignment for assignment in task.get("assignments", []) if assignment["stage"] == stage]
        if len(assignments) > 1:
            return self._run_multi_participant_stage(task, assignments)
        assignment = assignments[0] if assignments else {
            "agent_id": task["assignee"],
            "assignment_id": task["assignment_id"],
            "stage": stage,
        }
        assignee = assignment["agent_id"]
        assignment_id = assignment["assignment_id"]
        self._recover_retryable_assignment_if_needed(task, assignee, assignment_id)
        self._ensure_assignment_is_valid(task, assignee, assignment_id)
        self.store.set_agent_assignment(assignee, "running", task_id, assignment_id)
        self.store.append_event(
            "agent.running",
            task_id=task_id,
            assignment_id=assignment_id,
            agent_id=assignee,
            stage=stage,
        )

        try:
            generated = self.provider.generate(
                stage=stage,
                brief=task["brief"],
                task_id=task_id,
                prior_artifacts=list(task["artifacts"]),
                agent_id=assignee,
                artifact_type=None,
            )
            artifact = self._write_generated_artifact(task_id, stage, generated)
        except Exception as exc:
            task["status"] = "failed"
            task["blocking_reason"] = str(exc)
            self.store.save_task(task)
            self.store.write_message(
                assignee,
                "outbox",
                {
                    "message_type": "error",
                    "task_id": task_id,
                    "assignment_id": assignment_id,
                    "stage": stage,
                    "error": str(exc),
                },
                actor_id=assignee,
            )
            self.store.append_event(
                "agent.failed",
                task_id=task_id,
                assignment_id=assignment_id,
                agent_id=assignee,
                stage=stage,
                error=str(exc),
            )
            self.store.set_agent_assignment(assignee, "assigned", task_id, assignment_id)
            raise
        artifact["artifact_type"] = generated["artifact_type"]
        artifact["summary"] = generated["summary"]
        task["artifacts"].append(artifact)
        task["status"] = "stage_complete"
        task["blocking_reason"] = ""
        for item in task.get("assignments", []):
            if item["stage"] == stage and item["assignment_id"] == assignment_id:
                item["status"] = "done"
        self.store.save_task(task)
        self.store.write_message(
            assignee,
            "outbox",
            {
                "message_type": "done",
                "task_id": task_id,
                "assignment_id": assignment_id,
                "stage": stage,
                "artifact_ref": artifact["path"],
            },
            actor_id=assignee,
        )
        self.store.append_event(
            "artifact.created",
            task_id=task_id,
            assignment_id=assignment_id,
            agent_id=assignee,
            stage=stage,
            artifact_ref=artifact["path"],
        )
        self.store.set_agent_assignment(assignee, "idle")
        self.store.append_event(
            "agent.released",
            task_id=task_id,
            assignment_id=assignment_id,
            agent_id=assignee,
            stage=stage,
        )
        return artifact

    def _run_multi_participant_stage(self, task: dict, assignments: list[dict]) -> dict:
        task_id = task["task_id"]
        stage = task["current_stage"]
        prior_artifacts = list(task["artifacts"])
        generated_by_assignment: dict[str, dict] = {}
        for assignment in assignments:
            assignee = assignment["agent_id"]
            assignment_id = assignment["assignment_id"]
            self._recover_retryable_assignment_if_needed(task, assignee, assignment_id)
            self._ensure_assignment_is_valid(task, assignee, assignment_id)
            self.store.set_agent_assignment(assignee, "running", task_id, assignment_id)
            self.store.append_event(
                "agent.running",
                task_id=task_id,
                assignment_id=assignment_id,
                agent_id=assignee,
                stage=stage,
            )

        try:
            with ThreadPoolExecutor(max_workers=len(assignments)) as executor:
                futures = {
                    executor.submit(
                        self.provider.generate,
                        stage=stage,
                        brief=task["brief"],
                        task_id=task_id,
                        prior_artifacts=prior_artifacts,
                        agent_id=assignment["agent_id"],
                        artifact_type=BRAINSTORM_NOTE_ARTIFACTS[assignment["agent_id"]],
                    ): assignment
                    for assignment in assignments
                }
                for future in as_completed(futures):
                    assignment = futures[future]
                    generated_by_assignment[assignment["assignment_id"]] = future.result()
        except Exception as exc:
            for assignment in assignments:
                self._mark_assignment_failed(
                    task,
                    assignment["agent_id"],
                    assignment["assignment_id"],
                    stage,
                    exc,
                )
            raise

        note_artifacts: list[dict] = []
        for assignment in assignments:
            assignee = assignment["agent_id"]
            assignment_id = assignment["assignment_id"]
            generated = generated_by_assignment[assignment_id]
            artifact = self._write_generated_artifact(task_id, stage, generated)
            artifact["artifact_type"] = generated["artifact_type"]
            artifact["summary"] = generated["summary"]
            artifact["agent_id"] = assignee
            artifact["assignment_id"] = assignment_id
            note_artifacts.append(artifact)
            self.store.write_message(
                assignee,
                "outbox",
                {
                    "message_type": "done",
                    "task_id": task_id,
                    "assignment_id": assignment_id,
                    "stage": stage,
                    "artifact_ref": artifact["path"],
                },
                actor_id=assignee,
            )
            self.store.append_event(
                "artifact.created",
                task_id=task_id,
                assignment_id=assignment_id,
                agent_id=assignee,
                stage=stage,
                artifact_ref=artifact["path"],
            )
            self.store.set_agent_assignment(assignee, "idle")
            self.store.append_event(
                "agent.released",
                task_id=task_id,
                assignment_id=assignment_id,
                agent_id=assignee,
                stage=stage,
            )

        synthesis = self.provider.synthesize_brainstorm(
            brief=task["brief"],
            task_id=task_id,
            note_artifacts=note_artifacts,
        )
        synthesis_artifact = self._write_generated_artifact(task_id, stage, synthesis)
        synthesis_artifact["artifact_type"] = synthesis["artifact_type"]
        synthesis_artifact["summary"] = synthesis["summary"]
        synthesis_artifact["source_artifacts"] = [artifact["path"] for artifact in note_artifacts]
        task["artifacts"].extend(note_artifacts)
        task["artifacts"].append(synthesis_artifact)
        task["status"] = "stage_complete"
        task["blocking_reason"] = ""
        for assignment in task["assignments"]:
            if assignment["stage"] == stage:
                assignment["status"] = "done"
        self.store.save_task(task)
        self.store.append_event(
            "synthesis.created",
            task_id=task_id,
            stage=stage,
            artifact_ref=synthesis_artifact["path"],
            source_artifacts=synthesis_artifact["source_artifacts"],
        )
        return synthesis_artifact

    def _assign(self, task: dict, assignment: dict) -> None:
        stage = assignment["stage"]
        assignee = assignment["agent_id"]
        assignment_id = assignment["assignment_id"]
        self.store.set_agent_assignment(assignee, "assigned", task["task_id"], assignment_id)
        message = self.store.write_message(
            assignee,
            "inbox",
            {
                "message_type": "assignment",
                "assignment_id": assignment_id,
                "task_id": task["task_id"],
                "assignee": assignee,
                "stage": stage,
                "objective": task["brief"],
                "scope": f"Produce the {stage} stage artifact only.",
                "inputs": [artifact["path"] for artifact in task["artifacts"]],
                "expected_outputs": [stage],
                "forbidden_actions": ["act_without_assignment", "read_other_agent_private_state"],
            },
            actor_id="orchestrator",
        )
        self.store.append_event(
            "assignment.created",
            task_id=task["task_id"],
            assignment_id=assignment_id,
            agent_id=assignee,
            stage=stage,
            output_message_ids=[message["message_id"]],
        )

    def _write_generated_artifact(self, task_id: str, stage: str, generated: dict) -> dict:
        return self.store.write_artifact(
            task_id,
            stage,
            generated["filename"],
            generated["content"],
            zh_content=generated.get("content_zh"),
        )

    def _ensure_assignment_is_valid(self, task: dict, assignee: str, assignment_id: str) -> None:
        agent = self.store.get_agent(assignee)
        if agent["state"] != "assigned":
            raise PolicyError(f"{assignee} is {agent['state']}; expected assigned")
        if agent["active_task_id"] != task["task_id"] or agent["active_assignment_id"] != assignment_id:
            raise PolicyError(f"{assignee} is not assigned to {task['task_id']}")
        messages = self.store.read_messages(assignee, "inbox", actor_id=assignee)
        has_assignment = any(
            message.get("message_type") == "assignment"
            and message.get("task_id") == task["task_id"]
            and message.get("assignment_id") == assignment_id
            and message.get("assignee") == assignee
            for message in messages
        )
        if not has_assignment:
            raise PolicyError(f"{assignee} has no explicit assignment for {task['task_id']}")

    def _ensure_prerequisites_for_new_task(self, stage: str) -> None:
        if stage not in {"brainstorm", "research"}:
            raise PolicyError(f"{stage} requires completed brainstorm; create a brainstorm task first")

    def _ensure_prerequisites(self, task: dict, stage: str) -> None:
        completed = {artifact["stage"] for artifact in task["artifacts"]}
        missing = [required for required in STAGE_PREREQUISITES.get(stage, []) if required not in completed]
        if missing:
            raise PolicyError(f"{stage} requires completed {' and '.join(missing)}")

    def _ensure_agents_available(self, agent_ids: list[str]) -> None:
        for agent_id in agent_ids:
            self._ensure_agent_available(agent_id)

    def _ensure_agent_available(self, agent_id: str) -> None:
        agent = self.store.get_agent(agent_id)
        if agent["state"] != "idle":
            raise PolicyError(
                f"{agent_id} is busy with {agent['active_task_id']} / {agent['active_assignment_id']}"
            )

    @staticmethod
    def _ensure_stage(stage: str) -> None:
        if stage not in STAGE_PARTICIPANTS:
            raise PolicyError(f"Unknown stage: {stage}")

    def _recover_retryable_assignment_if_needed(self, task: dict, assignee: str, assignment_id: str) -> None:
        agent = self.store.get_agent(assignee)
        if (
            agent["state"] == "running"
            and agent["active_task_id"] == task["task_id"]
            and agent["active_assignment_id"] == assignment_id
            and task.get("status") == "failed"
        ):
            self.store.set_agent_assignment(assignee, "assigned", task["task_id"], assignment_id)
            self.store.append_event(
                "agent.recovered",
                task_id=task["task_id"],
                assignment_id=assignment_id,
                agent_id=assignee,
                stage=task["current_stage"],
            )

    @staticmethod
    def _new_assignments(stage: str, participants: list[str]) -> list[dict]:
        return [
            {
                "agent_id": agent_id,
                "assignment_id": Orchestrator._new_assignment_id(),
                "stage": stage,
                "status": "assigned",
            }
            for agent_id in participants
        ]

    def _mark_assignment_failed(self, task: dict, assignee: str, assignment_id: str, stage: str, exc: Exception) -> None:
        task["status"] = "failed"
        task["blocking_reason"] = str(exc)
        self.store.save_task(task)
        self.store.write_message(
            assignee,
            "outbox",
            {
                "message_type": "error",
                "task_id": task["task_id"],
                "assignment_id": assignment_id,
                "stage": stage,
                "error": str(exc),
            },
            actor_id=assignee,
        )
        self.store.append_event(
            "agent.failed",
            task_id=task["task_id"],
            assignment_id=assignment_id,
            agent_id=assignee,
            stage=stage,
            error=str(exc),
        )
        self.store.set_agent_assignment(assignee, "assigned", task["task_id"], assignment_id)

    @staticmethod
    def _new_assignment_id() -> str:
        return f"assignment-{uuid4().hex[:12]}"
