import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_team.orchestrator import Orchestrator, PolicyError  # noqa: E402
from agent_team.providers import LLMProvider, MockProvider, build_provider  # noqa: E402
from agent_team.store import TeamStore  # noqa: E402


class FlakyProvider:
    def __init__(self):
        self.calls = 0
        self.mock = MockProvider()

    def generate(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("provider boom")
        return self.mock.generate(**kwargs)


class BilingualProvider:
    def generate(self, **_kwargs):
        return {
            "artifact_type": "ResearchMemo",
            "filename": "research-ResearchMemo.md",
            "content": "# ResearchMemo\n\nEnglish memo.",
            "content_zh": "# ResearchMemo\n\n中文备份。",
            "summary": "ResearchMemo created.",
        }


class SlowRecordingProvider:
    def __init__(self, delay=0.2):
        self.delay = delay
        self.calls = []
        self.lock = threading.Lock()
        self.started = []

    def generate(self, **kwargs):
        with self.lock:
            self.started.append((kwargs.get("agent_id"), time.monotonic()))
        time.sleep(self.delay)
        with self.lock:
            self.calls.append(kwargs)
        artifact_type = kwargs["artifact_type"]
        agent_id = kwargs["agent_id"]
        return {
            "artifact_type": artifact_type,
            "filename": f"brainstorm-{agent_id.replace('.', '-')}-{artifact_type}.md",
            "content": f"# {artifact_type}\n\nagent: `{agent_id}`",
            "summary": f"{artifact_type} created.",
        }

    def synthesize_brainstorm(self, *, brief, task_id, note_artifacts):
        return {
            "artifact_type": "BrainstormMemo",
            "filename": "brainstorm-BrainstormMemo.md",
            "content": "# BrainstormMemo\n\n## Challenge Round\n\n## Rebuttal Round\n\n## Tradeoff Matrix\n\n## Decision Log\n",
            "summary": "BrainstormMemo synthesized.",
        }


class FakeChatCompletionChoice:
    def __init__(self, content):
        self.message = type("Message", (), {"content": content})()


class FakeChatCompletionResponse:
    def __init__(self, content):
        self.choices = [FakeChatCompletionChoice(content)]


class FakeChatCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if "Chinese archive" in kwargs["messages"][-1]["content"]:
            return FakeChatCompletionResponse("# ResearchMemo\n\n中文归档")
        return FakeChatCompletionResponse("# ResearchMemo\n\n## Output\nLLM memo")


class FakeOpenAIClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": FakeChatCompletions()})()


class AgentTeamTests(unittest.TestCase):
    def make_store(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        store = TeamStore(root)
        store.init()
        return root, store

    def test_init_registers_all_agents_as_idle_without_artifacts(self):
        root, store = self.make_store()

        agents = store.load_agents()
        states = {agent["agent_id"]: agent["state"] for agent in agents}

        self.assertEqual(states["research.lead"], "idle")
        self.assertEqual(states["technical.lead"], "idle")
        self.assertEqual(states["implementation.dev"], "idle")
        self.assertEqual(states["qa.tester"], "idle")
        self.assertEqual(states["review.lead"], "idle")
        self.assertEqual(list((root / ".agent-team" / "artifacts").glob("*")), [])
        research = store.get_agent("research.lead")
        technical = store.get_agent("technical.lead")
        qa = store.get_agent("qa.tester")
        self.assertIn("write_brainstorm_note", research["allowed_actions"])
        self.assertIn("challenge_feasibility", technical["allowed_actions"])
        self.assertIn("challenge_testability", qa["allowed_actions"])

    def test_creating_research_task_assigns_only_research_lead(self):
        _root, store = self.make_store()
        orchestrator = Orchestrator(store)

        task = orchestrator.create_task("研究一个多灾种多光谱语义分割目标", stage="research")
        states = {agent["agent_id"]: agent["state"] for agent in store.load_agents()}
        inbox = store.read_messages("research.lead", "inbox", actor_id="research.lead")

        self.assertEqual(task["current_stage"], "research")
        self.assertEqual(states["research.lead"], "assigned")
        self.assertEqual(states["technical.lead"], "idle")
        self.assertEqual(states["qa.tester"], "idle")
        self.assertEqual(inbox[-1]["message_type"], "assignment")
        self.assertEqual(inbox[-1]["assignment_id"], task["assignment_id"])
        self.assertEqual(states["research.lead"], "assigned")
        agent = store.get_agent("research.lead")
        self.assertEqual(agent["active_task_id"], task["task_id"])
        self.assertEqual(agent["active_assignment_id"], task["assignment_id"])

    def test_creating_brainstorm_task_assigns_research_technical_and_qa(self):
        _root, store = self.make_store()
        orchestrator = Orchestrator(store)

        task = orchestrator.create_task("对抗式头脑风暴：多 Agent 开发框架", stage="brainstorm")
        states = {agent["agent_id"]: agent["state"] for agent in store.load_agents()}
        assigned_agents = [assignment["agent_id"] for assignment in task["assignments"]]

        self.assertEqual(task["current_stage"], "brainstorm")
        self.assertEqual(
            assigned_agents,
            ["research.lead", "technical.lead", "qa.tester"],
        )
        self.assertEqual(states["research.lead"], "assigned")
        self.assertEqual(states["technical.lead"], "assigned")
        self.assertEqual(states["qa.tester"], "assigned")
        self.assertEqual(states["implementation.dev"], "idle")
        self.assertEqual(states["review.lead"], "idle")
        for assignment in task["assignments"]:
            inbox = store.read_messages(assignment["agent_id"], "inbox", actor_id=assignment["agent_id"])
            self.assertEqual(inbox[-1]["message_type"], "assignment")
            self.assertEqual(inbox[-1]["stage"], "brainstorm")
            self.assertEqual(inbox[-1]["assignment_id"], assignment["assignment_id"])

    def test_run_brainstorm_writes_adversarial_notes_and_synthesis(self):
        _root, store = self.make_store()
        orchestrator = Orchestrator(store)
        task = orchestrator.create_task("规划跨项目可复用的多 Agent 团队", stage="brainstorm")

        result = orchestrator.run_task(task["task_id"])
        updated = store.load_task(task["task_id"])
        artifact_types = [artifact["artifact_type"] for artifact in updated["artifacts"]]
        states = {agent["agent_id"]: agent["state"] for agent in store.load_agents()}
        memo = Path(result["path"]).read_text(encoding="utf-8")

        self.assertEqual(result["artifact_type"], "BrainstormMemo")
        self.assertEqual(
            artifact_types,
            [
                "ResearchBrainstormNote",
                "TechnicalBrainstormNote",
                "QABrainstormNote",
                "BrainstormMemo",
            ],
        )
        self.assertIn("## Challenge Round", memo)
        self.assertIn("## Rebuttal Round", memo)
        self.assertIn("## Tradeoff Matrix", memo)
        self.assertIn("## Decision Log", memo)
        self.assertIn("synthesis_actor: `research.lead`", memo)
        self.assertEqual(states["research.lead"], "idle")
        self.assertEqual(states["technical.lead"], "idle")
        self.assertEqual(states["qa.tester"], "idle")

    def test_run_research_writes_handoff_and_releases_agent_to_idle(self):
        _root, store = self.make_store()
        orchestrator = Orchestrator(store)
        task = orchestrator.create_task("研究多 Agent 团队框架", stage="research")

        result = orchestrator.run_task(task["task_id"])
        updated = store.load_task(task["task_id"])
        states = {agent["agent_id"]: agent["state"] for agent in store.load_agents()}

        self.assertEqual(result["artifact_type"], "ResearchMemo")
        self.assertEqual(updated["status"], "stage_complete")
        self.assertEqual(updated["assignments"][0]["status"], "done")
        self.assertEqual(updated["artifacts"][0]["stage"], "research")
        self.assertTrue(Path(updated["artifacts"][0]["path"]).exists())
        self.assertEqual(states["research.lead"], "idle")
        agent = store.get_agent("research.lead")
        self.assertIsNone(agent["active_task_id"])
        self.assertIsNone(agent["active_assignment_id"])

    def test_provider_can_archive_chinese_markdown_next_to_english_artifact(self):
        _root, store = self.make_store()
        orchestrator = Orchestrator(store, provider=BilingualProvider())
        task = orchestrator.create_task("Write a research memo", stage="research")

        result = orchestrator.run_task(task["task_id"])
        updated = store.load_task(task["task_id"])

        zh_path = Path(result["zh_path"])
        self.assertEqual(Path(result["path"]).name, "research-ResearchMemo.md")
        self.assertEqual(zh_path.name, "research-ResearchMemo-zh.md")
        self.assertTrue(zh_path.exists())
        self.assertIn("中文备份", zh_path.read_text(encoding="utf-8"))
        self.assertEqual(updated["artifacts"][0]["zh_path"], result["zh_path"])

    def test_llm_provider_uses_chat_client_and_generates_chinese_archive(self):
        client = FakeOpenAIClient()
        provider = LLMProvider(client=client, model="fake-model")

        result = provider.generate(
            stage="research",
            brief="Build disaster segmentation.",
            task_id="task-123",
            prior_artifacts=[],
            agent_id="research.lead",
            artifact_type="ResearchMemo",
        )

        self.assertEqual(result["artifact_type"], "ResearchMemo")
        self.assertEqual(result["filename"], "research-research-lead-ResearchMemo.md")
        self.assertIn("LLM memo", result["content"])
        self.assertIn("中文归档", result["content_zh"])
        self.assertEqual(client.chat.completions.calls[0]["model"], "fake-model")
        self.assertIn("research.lead", client.chat.completions.calls[0]["messages"][0]["content"])

    def test_build_provider_selects_mock_or_llm(self):
        self.assertIsInstance(build_provider("mock"), MockProvider)
        self.assertIsInstance(build_provider("llm", client=FakeOpenAIClient(), model="fake-model"), LLMProvider)

    def test_build_provider_reads_provider_json_defaults(self):
        root, store = self.make_store()
        provider_config = store.base / "config" / "provider.json"
        provider_config.write_text(
            json.dumps(
                {
                    "provider": "llm",
                    "model": "deepseek-v4-flash",
                    "agents": {
                        "technical.lead": {"model": "deepseek-v4-pro"},
                    },
                }
            ),
            encoding="utf-8",
        )

        provider = build_provider(None, root=root, client=FakeOpenAIClient())

        self.assertIsInstance(provider, LLMProvider)
        self.assertEqual(provider.model, "deepseek-v4-flash")
        self.assertEqual(provider.model_for("technical.lead"), "deepseek-v4-pro")

    def test_build_provider_reads_api_key_and_base_url_from_provider_json(self):
        root, store = self.make_store()
        provider_config = store.base / "config" / "provider.json"
        provider_config.write_text(
            json.dumps(
                {
                    "provider": "llm",
                    "model": "deepseek-v4-pro",
                    "api_key": "json-key",
                    "base_url": "https://llm.example/v1",
                }
            ),
            encoding="utf-8",
        )

        provider = build_provider(None, root=root, client=FakeOpenAIClient())

        self.assertEqual(provider.api_key, "json-key")
        self.assertEqual(provider.base_url, "https://llm.example/v1")

    def test_build_provider_can_resolve_api_key_env_from_provider_json(self):
        root, store = self.make_store()
        provider_config = store.base / "config" / "provider.json"
        provider_config.write_text(
            json.dumps(
                {
                    "provider": "llm",
                    "model": "deepseek-v4-pro",
                    "api_key_env": "AGENT_TEAM_TEST_API_KEY",
                }
            ),
            encoding="utf-8",
        )
        old_value = os.environ.get("AGENT_TEAM_TEST_API_KEY")

        def restore_env():
            if old_value is None:
                os.environ.pop("AGENT_TEAM_TEST_API_KEY", None)
            else:
                os.environ["AGENT_TEAM_TEST_API_KEY"] = old_value

        self.addCleanup(restore_env)
        os.environ["AGENT_TEAM_TEST_API_KEY"] = "env-key"

        provider = build_provider(None, root=root, client=FakeOpenAIClient())

        self.assertEqual(provider.api_key, "env-key")

    def test_brainstorm_synthesis_uses_research_lead_model(self):
        client = FakeOpenAIClient()
        provider = LLMProvider(
            client=client,
            model="default-model",
            agent_models={"research.lead": {"model": "research-model"}},
        )

        provider.synthesize_brainstorm(brief="Synthesize notes.", task_id="task-123", note_artifacts=[])

        self.assertEqual(client.chat.completions.calls[0]["model"], "research-model")
        self.assertIn("research.lead", client.chat.completions.calls[0]["messages"][0]["content"])

    def test_brainstorm_agents_generate_in_parallel_and_do_not_see_peer_notes(self):
        _root, store = self.make_store()
        provider = SlowRecordingProvider(delay=0.2)
        orchestrator = Orchestrator(store, provider=provider)
        task = orchestrator.create_task("并行独立头脑风暴", stage="brainstorm")

        start = time.monotonic()
        orchestrator.run_task(task["task_id"])
        elapsed = time.monotonic() - start

        self.assertLess(elapsed, 0.45)
        self.assertEqual(
            sorted(call["agent_id"] for call in provider.calls),
            ["qa.tester", "research.lead", "technical.lead"],
        )
        for call in provider.calls:
            self.assertEqual(call["prior_artifacts"], [])
        starts = [timestamp for _agent_id, timestamp in provider.started]
        self.assertLess(max(starts) - min(starts), 0.1)

    def test_technical_stage_is_blocked_without_research_handoff(self):
        _root, store = self.make_store()
        orchestrator = Orchestrator(store)
        task = orchestrator.create_task("直接让技术部门开始", stage="brainstorm")

        with self.assertRaisesRegex(PolicyError, "requires completed brainstorm"):
            orchestrator.assign_stage(task["task_id"], "technical")

        agent = store.get_agent("technical.lead")
        self.assertEqual(agent["state"], "assigned")
        self.assertEqual(agent["active_task_id"], task["task_id"])
        assignment = next(item for item in task["assignments"] if item["agent_id"] == "technical.lead")
        self.assertEqual(agent["active_assignment_id"], assignment["assignment_id"])

    def test_technical_stage_runs_after_brainstorm_handoff(self):
        _root, store = self.make_store()
        orchestrator = Orchestrator(store)
        task = orchestrator.create_task("先头脑风暴再技术", stage="brainstorm")
        orchestrator.run_task(task["task_id"])

        orchestrator.assign_stage(task["task_id"], "technical")
        result = orchestrator.run_task(task["task_id"])
        updated = store.load_task(task["task_id"])

        self.assertEqual(result["artifact_type"], "TechnicalPlan")
        self.assertEqual(updated["assignments"][0]["status"], "done")
        self.assertEqual(
            [artifact["artifact_type"] for artifact in updated["artifacts"]],
            [
                "ResearchBrainstormNote",
                "TechnicalBrainstormNote",
                "QABrainstormNote",
                "BrainstormMemo",
                "TechnicalPlan",
            ],
        )

    def test_audit_log_records_assignment_and_completion_events(self):
        _root, store = self.make_store()
        orchestrator = Orchestrator(store)
        task = orchestrator.create_task("审计测试", stage="research")
        orchestrator.run_task(task["task_id"])

        events = store.tail_events(limit=20)
        event_types = [event["event_type"] for event in events]

        self.assertIn("assignment.created", event_types)
        self.assertIn("agent.running", event_types)
        self.assertIn("artifact.created", event_types)
        self.assertIn("agent.released", event_types)
        for event in events:
            self.assertIn("event_id", event)
            self.assertIn("timestamp", event)

    def test_failed_run_keeps_assignment_retryable_across_conversations(self):
        _root, store = self.make_store()
        provider = FlakyProvider()
        orchestrator = Orchestrator(store, provider=provider)
        task = orchestrator.create_task("失败恢复测试", stage="research")

        with self.assertRaisesRegex(RuntimeError, "provider boom"):
            orchestrator.run_task(task["task_id"])

        failed_task = store.load_task(task["task_id"])
        agent = store.get_agent("research.lead")
        self.assertEqual(failed_task["status"], "failed")
        self.assertIn("provider boom", failed_task["blocking_reason"])
        self.assertEqual(agent["state"], "assigned")
        self.assertEqual(agent["active_task_id"], task["task_id"])
        self.assertEqual(agent["active_assignment_id"], task["assignment_id"])

        resumed = Orchestrator(store, provider=provider)
        result = resumed.run_task(task["task_id"])
        resumed_task = store.load_task(task["task_id"])

        self.assertEqual(result["artifact_type"], "ResearchMemo")
        self.assertEqual(resumed_task["blocking_reason"], "")
        self.assertEqual(store.get_agent("research.lead")["state"], "idle")

    def test_failed_task_with_stale_running_agent_is_recovered_before_retry(self):
        _root, store = self.make_store()
        provider = FlakyProvider()
        orchestrator = Orchestrator(store, provider=provider)
        task = orchestrator.create_task("stale running recovery", stage="research")

        with self.assertRaisesRegex(RuntimeError, "provider boom"):
            orchestrator.run_task(task["task_id"])
        store.set_agent_assignment("research.lead", "running", task["task_id"], task["assignment_id"])

        resumed = Orchestrator(store, provider=provider)
        result = resumed.run_task(task["task_id"])
        resumed_task = store.load_task(task["task_id"])

        self.assertEqual(result["artifact_type"], "ResearchMemo")
        self.assertEqual(resumed_task["blocking_reason"], "")
        self.assertEqual(store.get_agent("research.lead")["state"], "idle")

    def test_busy_agent_rejects_second_assignment_until_released(self):
        _root, store = self.make_store()
        orchestrator = Orchestrator(store)
        first = orchestrator.create_task("第一个任务", stage="research")

        with self.assertRaisesRegex(PolicyError, "research.lead is busy"):
            orchestrator.create_task("第二个任务", stage="research")

        orchestrator.run_task(first["task_id"])
        second = orchestrator.create_task("第二个任务", stage="research")

        self.assertNotEqual(first["task_id"], second["task_id"])
        self.assertEqual(store.get_agent("research.lead")["active_task_id"], second["task_id"])

    def test_message_store_rejects_cross_agent_private_reads(self):
        _root, store = self.make_store()
        orchestrator = Orchestrator(store)
        orchestrator.create_task("消息隔离测试", stage="research")

        with self.assertRaisesRegex(PermissionError, "cannot read research.lead inbox"):
            store.read_messages("research.lead", "inbox", actor_id="technical.lead")

        messages = store.read_messages("research.lead", "inbox", actor_id="research.lead")
        self.assertEqual(messages[-1]["message_type"], "assignment")

    def test_store_lists_tasks_for_cross_conversation_resume(self):
        _root, store = self.make_store()
        orchestrator = Orchestrator(store)
        first = orchestrator.create_task("第一个任务", stage="research")
        orchestrator.run_task(first["task_id"])
        second = orchestrator.create_task("第二个任务", stage="research")

        task_ids = [task["task_id"] for task in store.list_tasks()]

        self.assertEqual(task_ids, [first["task_id"], second["task_id"]])

    def test_cli_runs_persistent_research_then_technical_handoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            brief = root / "brief.md"
            brief.write_text("建立一个跨对话多 Agent 团队", encoding="utf-8")
            env = {"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}

            init = subprocess.run(
                [sys.executable, "-m", "agent_team.cli", "--root", str(root), "init"],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            self.assertIn("initialized", init.stdout)

            created = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "agent_team.cli",
                    "--root",
                    str(root),
                    "task",
                    "create",
                    str(brief),
                ],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            task_id = json.loads(created.stdout)["task_id"]

            listed = subprocess.run(
                [sys.executable, "-m", "agent_team.cli", "--root", str(root), "task", "list"],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            self.assertEqual(json.loads(listed.stdout)[0]["task_id"], task_id)

            subprocess.run(
                [sys.executable, "-m", "agent_team.cli", "--root", str(root), "run", task_id],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "agent_team.cli",
                    "--root",
                    str(root),
                    "task",
                    "assign",
                    task_id,
                    "technical",
                ],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            subprocess.run(
                [sys.executable, "-m", "agent_team.cli", "--root", str(root), "run", task_id],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

            status = subprocess.run(
                [sys.executable, "-m", "agent_team.cli", "--root", str(root), "task", "status", task_id],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            payload = json.loads(status.stdout)
            self.assertEqual(
                [artifact["artifact_type"] for artifact in payload["artifacts"]],
                [
                    "ResearchBrainstormNote",
                    "TechnicalBrainstormNote",
                    "QABrainstormNote",
                    "BrainstormMemo",
                    "TechnicalPlan",
                ],
            )


if __name__ == "__main__":
    unittest.main()
