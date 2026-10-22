from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .orchestrator import Orchestrator, PolicyError
from .providers import ProviderError, build_provider
from .store import TeamStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="team")
    parser.add_argument("--root", default=".", help="Project root containing .agent-team")
    parser.add_argument(
        "--provider",
        default=os.environ.get("AGENT_TEAM_PROVIDER"),
        choices=["mock", "llm"],
        help="Artifact provider to use for run commands.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("AGENT_TEAM_LLM_MODEL"),
        help="Model name for --provider llm.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init")
    subparsers.add_parser("run").add_argument("task_id")

    agents_parser = subparsers.add_parser("agents")
    agents_sub = agents_parser.add_subparsers(dest="agents_command", required=True)
    agents_sub.add_parser("list")

    task_parser = subparsers.add_parser("task")
    task_sub = task_parser.add_subparsers(dest="task_command", required=True)
    task_sub.add_parser("list")
    create_parser = task_sub.add_parser("create")
    create_parser.add_argument("brief_file")
    create_parser.add_argument("--stage", default="brainstorm")
    status_parser = task_sub.add_parser("status")
    status_parser.add_argument("task_id")
    assign_parser = task_sub.add_parser("assign")
    assign_parser.add_argument("task_id")
    assign_parser.add_argument("stage")

    messages_parser = subparsers.add_parser("messages")
    messages_sub = messages_parser.add_subparsers(dest="messages_command", required=True)
    tail_parser = messages_sub.add_parser("tail")
    tail_parser.add_argument("--limit", type=int, default=20)

    args = parser.parse_args(argv)
    store = TeamStore(Path(args.root))

    try:
        if args.command == "init":
            store.init()
            print(f"initialized {store.base}")
            return 0

        provider = build_provider(args.provider, root=Path(args.root), model=args.model) if args.command == "run" else None
        orchestrator = Orchestrator(store, provider=provider)

        if args.command == "agents" and args.agents_command == "list":
            print(json.dumps(store.load_agents(), ensure_ascii=False, indent=2, sort_keys=True))
            return 0

        if args.command == "task" and args.task_command == "create":
            brief = Path(args.brief_file).read_text(encoding="utf-8")
            task = orchestrator.create_task(brief, stage=args.stage)
            print(json.dumps(task, ensure_ascii=False, sort_keys=True))
            return 0

        if args.command == "task" and args.task_command == "list":
            print(json.dumps(store.list_tasks(), ensure_ascii=False, indent=2, sort_keys=True))
            return 0

        if args.command == "task" and args.task_command == "status":
            print(json.dumps(store.load_task(args.task_id), ensure_ascii=False, indent=2, sort_keys=True))
            return 0

        if args.command == "task" and args.task_command == "assign":
            task = orchestrator.assign_stage(args.task_id, args.stage)
            print(json.dumps(task, ensure_ascii=False, sort_keys=True))
            return 0

        if args.command == "run":
            artifact = orchestrator.run_task(args.task_id)
            print(json.dumps(artifact, ensure_ascii=False, sort_keys=True))
            return 0

        if args.command == "messages" and args.messages_command == "tail":
            print(json.dumps(store.tail_events(args.limit), ensure_ascii=False, indent=2, sort_keys=True))
            return 0
    except (FileNotFoundError, PolicyError, ProviderError, KeyError) as exc:
        parser.exit(2, f"team: error: {exc}\n")

    parser.error("unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
