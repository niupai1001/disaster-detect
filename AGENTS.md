# Project Multi-Agent Team Instructions

This project includes a persistent local multi-agent team framework in `src/agent_team`.

## Core Rule

Departments and agents do nothing until the user or coordinator gives an explicit assignment. Visible context is not an assignment.

## Provider Rule

- Do not use external provider services for department artifacts in this project.
- Codex should dispatch and coordinate its own subagents for real research, technical, implementation, QA, and review work.
- `.agent-team/config/provider.json` should remain on the no-network `mock` provider as a ledger/dry-run fallback only; mock artifacts are not acceptable as final department outputs.

## CLI

Run commands from the project root:

```bash
PYTHONPATH=src python -m agent_team.cli --root . init
PYTHONPATH=src python -m agent_team.cli --root . agents list
PYTHONPATH=src python -m agent_team.cli --root . task list
PYTHONPATH=src python -m agent_team.cli --root . task status <task-id>
PYTHONPATH=src python -m agent_team.cli --root . messages tail
```

## Department Phrases

- If the user says "研发部门研究目标", "头脑风暴", or asks for research brainstorming, create a brief file for the goal only when needed, prefer `.agent-team/briefs/` over `docs/`, then run `task create <brief-file> --stage brainstorm` and `run <task-id>`. After the department artifact or living note absorbs it, delete the temporary brief.
- The `brainstorm` stage is an adversarial workshop. It assigns `research.lead`, `technical.lead`, and `qa.tester`; each produces an independent note with challenge, rebuttal, and negotiation sections. The orchestrator synthesizes a `BrainstormMemo`.
- For project-start brainstorms, do not converge only around the current local dataset or a single familiar algorithm. First create or reference a research/literature evidence layer, then require each participating agent to propose independent route options grounded in that evidence.
- A valid project-start `BrainstormMemo` must preserve multiple candidate method families when appropriate, such as classical remote-sensing indices, OBIA/classical ML, supervised segmentation, change detection, weak supervision, foundation-model-assisted workflows, and detection fallbacks.
- If the user says "技术部门开始干活", list tasks, choose the relevant task with a completed `BrainstormMemo`, then run `task assign <task-id> technical` and `run <task-id>`.
- If the user says "开发部门实现", assign and run `implementation`.
- If the user says "测试部门测试", assign and run `qa`.
- If the user says "评审部门评审", assign and run `review`.

## Handoff Discipline

- Do not skip stages unless the user explicitly says to bypass the workflow.
- Do not run technical work before a `BrainstormMemo` exists.
- Do not treat one department's artifact as another department's private state.
- Use `task status` and `messages tail` before resuming work in a new conversation.
- When task choice is ambiguous, ask the user which task to continue.

## Minimal Markdown Policy

- Keep the project directory lean. Do not create date-stamped Markdown files by default.
- Maintain only a small set of living project notes:
  - `docs/README.md`
  - `docs/ops/project-structure.md`
  - `docs/ops/review-index.md`
  - `docs/ops/work-log.md`
  - `docs/ops/next-actions.md`
  - `docs/research/method-landscape.md`
  - `docs/research/segmentation-model-data-hypotheses.md`
  - `docs/implementation/current-runbook.md`
- Department artifacts may still be written under `.agent-team/artifacts/` when a workflow stage requires them, but prefer final synthesized artifacts over many intermediate notes.
- Temporary brainstorm briefs belong under `.agent-team/briefs/`, not `docs/`, and should be deleted after their evidence is absorbed into a living note or department artifact.
- Do not colocate Chinese `-zh.md` copies in the project unless the user explicitly asks for a bilingual project artifact.
- Mirror substantial handoffs, decisions, and next actions into the Obsidian vault instead:
  `/Users/Lemon/Documents/Obsidian Vault/遥感灾害检测/Codex项目日志/`
- For Obsidian, Chinese summaries are welcome when useful. Keep project Markdown canonical and concise.
