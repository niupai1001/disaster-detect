from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .models import STAGE_ARTIFACTS


class ProviderError(RuntimeError):
    pass


def build_provider(
    provider: str | None = None,
    *,
    root: str | Path | None = None,
    client: Any | None = None,
    model: str | None = None,
):
    config = _load_provider_config(root)
    provider_name = (
        provider
        or os.environ.get("AGENT_TEAM_PROVIDER")
        or config.get("provider")
        or config.get("default_provider")
        or "mock"
    ).lower()
    if provider_name == "mock":
        return MockProvider()
    if provider_name == "llm":
        provider_model = (
            model
            or os.environ.get("AGENT_TEAM_LLM_MODEL")
            or config.get("model")
            or config.get("default_model")
        )
        return LLMProvider(
            root=root,
            client=client,
            model=provider_model,
            agent_models=config.get("agents", {}),
            api_key=config.get("api_key"),
            api_key_env=config.get("api_key_env"),
            base_url=config.get("base_url"),
        )
    raise ProviderError(f"Unknown provider: {provider_name}")


def _load_provider_config(root: str | Path | None) -> dict:
    if not root:
        return {}
    path = Path(root) / ".agent-team" / "config" / "provider.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProviderError(f"Invalid provider config: {path}") from exc
    if not isinstance(payload, dict):
        raise ProviderError(f"Provider config must be a JSON object: {path}")
    return payload


class LLMProvider:
    def __init__(
        self,
        *,
        root: str | Path | None = None,
        client: Any | None = None,
        model: str | None = None,
        agent_models: dict | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.2,
        archive_chinese: bool = True,
    ):
        self.root = Path(root) if root else None
        self.api_key_env = api_key_env
        self.api_key = self._resolve_api_key(api_key, api_key_env)
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.client = client or self._default_client()
        self.model = model or os.environ.get("AGENT_TEAM_LLM_MODEL") or "gpt-4o-mini"
        self.agent_models = agent_models or {}
        self.temperature = temperature
        self.archive_chinese = archive_chinese

    def model_for(self, agent_id: str | None) -> str:
        config = self.agent_models.get(agent_id or "")
        if isinstance(config, str):
            return config
        if isinstance(config, dict):
            return config.get("model") or self.model
        return self.model

    def generate(
        self,
        *,
        stage: str,
        brief: str,
        task_id: str,
        prior_artifacts: list[dict],
        agent_id: str | None = None,
        artifact_type: str | None = None,
    ) -> dict:
        artifact_type = artifact_type or STAGE_ARTIFACTS[stage]
        content = self._complete(
            self._system_prompt(agent_id, stage, artifact_type),
            self._artifact_prompt(
                stage=stage,
                brief=brief,
                task_id=task_id,
                prior_artifacts=prior_artifacts,
                agent_id=agent_id,
                artifact_type=artifact_type,
            ),
            agent_id=agent_id,
        )
        result = {
            "artifact_type": artifact_type,
            "filename": self._filename(stage, artifact_type, agent_id),
            "content": content,
            "summary": f"{artifact_type} created for {task_id}.",
        }
        if self.archive_chinese:
            result["content_zh"] = self._chinese_archive(content, artifact_type)
        return result

    def synthesize_brainstorm(self, *, brief: str, task_id: str, note_artifacts: list[dict]) -> dict:
        refs = "\n\n".join(self._artifact_excerpt(artifact) for artifact in note_artifacts)
        prompt = "\n".join(
            [
                f"Task id: {task_id}",
                "Stage: brainstorm synthesis",
                "",
                "Assignment brief:",
                brief,
                "",
                "Independent brainstorm notes:",
                refs,
                "",
                "Write a complete English Markdown BrainstormMemo.",
                "Preserve consensus, challenges, rebuttals, tradeoffs, rejected options, decision log, and next verification needs.",
                "Do not invent completed implementation work.",
            ]
        )
        content = self._complete(
            self._system_prompt("research.lead", "brainstorm", "BrainstormMemo"),
            prompt,
            agent_id="research.lead",
        )
        result = {
            "artifact_type": "BrainstormMemo",
            "filename": "brainstorm-BrainstormMemo.md",
            "content": content,
            "summary": f"BrainstormMemo synthesized for {task_id}.",
        }
        if self.archive_chinese:
            result["content_zh"] = self._chinese_archive(content, "BrainstormMemo")
        return result

    def _artifact_prompt(
        self,
        *,
        stage: str,
        brief: str,
        task_id: str,
        prior_artifacts: list[dict],
        agent_id: str | None,
        artifact_type: str,
    ) -> str:
        prior = "\n\n".join(self._artifact_excerpt(artifact) for artifact in prior_artifacts) or "No prior artifacts."
        return "\n".join(
            [
                f"Task id: {task_id}",
                f"Stage: {stage}",
                f"Agent id: {agent_id or 'orchestrator'}",
                f"Required artifact type: {artifact_type}",
                "",
                "Assignment brief:",
                brief,
                "",
                "Prior artifacts:",
                prior,
                "",
                "Return only a complete English Markdown artifact.",
                "Honor explicit assignment boundaries and do not act for unassigned departments.",
            ]
        )

    @staticmethod
    def _system_prompt(agent_id: str | None, stage: str, artifact_type: str) -> str:
        role_guidance = {
            "research.lead": "You are research.lead. Frame goals, domain assumptions, research routes, and challenge weak problem definitions. When synthesizing, preserve every department's independent reasoning.",
            "technical.lead": "You are technical.lead. Challenge feasibility, architecture boundaries, implementation complexity, and scale risks.",
            "qa.tester": "You are qa.tester. Challenge testability, acceptance criteria, observability, and likely failure modes.",
            "orchestrator": "You are the orchestrator. Synthesize department outputs without replacing their independent reasoning.",
        }
        sections = ""
        if stage == "brainstorm" and artifact_type != "BrainstormMemo":
            sections = "Include: Independent Position, Challenge Round, Rebuttal Round, Negotiation Position."
        elif artifact_type == "BrainstormMemo":
            sections = "Include: Independent Notes, Challenge Round, Rebuttal Round, Consensus, Tradeoff Matrix, Rejected Options, Decision Log, Next Verification."
        return "\n".join(
            [
                role_guidance.get(agent_id or "", "You are an assigned project agent."),
                f"Produce {artifact_type} for the {stage} stage.",
                sections,
                "Be specific to the provided project evidence. Keep the output Markdown and avoid placeholders.",
            ]
        )

    def _complete(self, system_prompt: str, user_prompt: str, *, agent_id: str | None = None) -> str:
        response = self.client.chat.completions.create(
            model=self.model_for(agent_id),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
        )
        return self._extract_content(response)

    def _chinese_archive(self, english_markdown: str, artifact_type: str) -> str:
        prompt = "\n".join(
            [
                f"Chinese archive for {artifact_type}.",
                "Translate the following English Markdown into Chinese with equivalent meaning.",
                "Keep Markdown structure, file references, ids, code spans, and tables intact.",
                "",
                english_markdown,
            ]
        )
        return self._complete(
            "You create faithful Chinese Markdown archives. Do not add new facts.",
            prompt,
            agent_id="orchestrator",
        )

    def _artifact_excerpt(self, artifact: dict) -> str:
        path = artifact.get("path", "")
        content = self._read_artifact_text(path)
        if len(content) > 6000:
            content = content[:6000] + "\n\n[truncated]"
        return f"## {artifact.get('artifact_type', 'Artifact')}: `{path}`\n\n{content}"

    def _read_artifact_text(self, path: str) -> str:
        if not path:
            return ""
        artifact_path = Path(path)
        if not artifact_path.is_absolute() and self.root:
            artifact_path = self.root / artifact_path
        if not artifact_path.exists():
            return "[artifact content unavailable]"
        return artifact_path.read_text(encoding="utf-8")

    @staticmethod
    def _extract_content(response: Any) -> str:
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError) as exc:
            raise ProviderError("LLM response did not include chat completion content") from exc
        if not content or not content.strip():
            raise ProviderError("LLM response was empty")
        return content.strip()

    @staticmethod
    def _resolve_api_key(api_key: str | None, api_key_env: str | None) -> str | None:
        if api_key:
            return api_key
        if api_key_env:
            return os.environ.get(api_key_env)
        return os.environ.get("OPENAI_API_KEY")

    def _default_client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ProviderError("Install the `openai` package to use LLMProvider") from exc
        kwargs = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return OpenAI(**kwargs)

    @staticmethod
    def _filename(stage: str, artifact_type: str, agent_id: str | None) -> str:
        if agent_id:
            safe_agent = agent_id.replace(".", "-")
            return f"{stage}-{safe_agent}-{artifact_type}.md"
        return f"{stage}-{artifact_type}.md"


class MockProvider:
    def generate(
        self,
        *,
        stage: str,
        brief: str,
        task_id: str,
        prior_artifacts: list[dict],
        agent_id: str | None = None,
        artifact_type: str | None = None,
    ) -> dict:
        artifact_type = artifact_type or STAGE_ARTIFACTS[stage]
        prior = ", ".join(artifact["stage"] for artifact in prior_artifacts) or "none"
        output = self._brainstorm_note(agent_id) if stage == "brainstorm" and agent_id else self._stage_output(stage)
        content = "\n".join(
            [
                f"# {artifact_type}",
                "",
                f"- task_id: `{task_id}`",
                f"- stage: `{stage}`",
                f"- agent_id: `{agent_id or 'orchestrator'}`",
                f"- prior_artifacts: `{prior}`",
                "",
                "## Assignment",
                brief,
                "",
                "## Output",
                output,
            ]
        )
        return {
            "artifact_type": artifact_type,
            "filename": self._filename(stage, artifact_type, agent_id),
            "content": content,
            "summary": f"{artifact_type} created for {task_id}.",
        }

    def synthesize_brainstorm(self, *, brief: str, task_id: str, note_artifacts: list[dict]) -> dict:
        refs = "\n".join(f"- {artifact['artifact_type']}: `{artifact['path']}`" for artifact in note_artifacts)
        content = "\n".join(
            [
                "# BrainstormMemo",
                "",
                f"- task_id: `{task_id}`",
                "- stage: `brainstorm`",
                "- synthesis_actor: `research.lead`",
                "",
                "## Assignment",
                brief,
                "",
                "## Independent Notes",
                refs,
                "",
                "## Challenge Round",
                "- Technical challenges research assumptions for implementation complexity.",
                "- QA challenges both departments on observability, testability, and acceptance criteria.",
                "- Research challenges technical shortcuts that weaken the original goal.",
                "",
                "## Rebuttal Round",
                "- Each department must either defend, narrow, or revise its position after challenge.",
                "",
                "## Tradeoff Matrix",
                "| Option | Benefit | Risk | Owner |",
                "| --- | --- | --- | --- |",
                "| Narrow MVP | Fast validation | May miss future scale needs | research.lead |",
                "| Full platform | Rich capability | Scope creep | technical.lead |",
                "| Test-first workflow | Verifiable progress | More upfront ceremony | qa.tester |",
                "",
                "## Decision Log",
                "- Keep departments idle until explicit assignment.",
                "- Use adversarial brainstorm before technical planning.",
                "- Preserve rejected options and unresolved disputes for later stages.",
                "",
                "## Next Verification",
                "- Technical stage must read this BrainstormMemo before producing a TechnicalPlan.",
            ]
        )
        return {
            "artifact_type": "BrainstormMemo",
            "filename": "brainstorm-BrainstormMemo.md",
            "content": content,
            "summary": f"BrainstormMemo synthesized for {task_id}.",
        }

    @staticmethod
    def _stage_output(stage: str) -> str:
        outputs = {
            "brainstorm": "Brainstorm participant produced a structured adversarial note.",
            "research": "Research department produced a bounded goal memo and handoff for technical planning.",
            "technical": "Technical department interpreted the goal into an implementation-ready plan.",
            "implementation": "Implementation department prepared a scoped development handoff.",
            "qa": "QA department produced an independent verification report.",
            "review": "Review department produced a readiness note.",
        }
        return outputs[stage]

    @staticmethod
    def _brainstorm_note(agent_id: str | None) -> str:
        perspectives = {
            "research.lead": "Goal framing, domain assumptions, research route, and problem definition.",
            "technical.lead": "Feasibility challenge, architecture boundary, implementation risk, and complexity tradeoffs.",
            "qa.tester": "Testability challenge, acceptance risks, failure modes, and verification strategy.",
        }
        return "\n".join(
            [
                "## Independent Position",
                perspectives.get(agent_id or "", "Participant perspective."),
                "",
                "## Challenge Round",
                "Challenge at least one other department's assumption and name the risk.",
                "",
                "## Rebuttal Round",
                "Defend, narrow, or revise the participant position after challenge.",
                "",
                "## Negotiation Position",
                "State what this department can accept, reject, and defer.",
            ]
        )

    @staticmethod
    def _filename(stage: str, artifact_type: str, agent_id: str | None) -> str:
        if agent_id:
            safe_agent = agent_id.replace(".", "-")
            return f"{stage}-{safe_agent}-{artifact_type}.md"
        return f"{stage}-{artifact_type}.md"
