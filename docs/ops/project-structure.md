---
title: Project Structure
date: 2026-05-07
tags:
  - ops
  - project-structure
  - review
status: active
---

# Project Structure

This project separates reviewable source files from generated data, runtime artifacts, and project memory. The documentation surface is intentionally small: prefer updating living notes over creating date-stamped Markdown files.

## Reviewable Source Surface

| Path | Purpose |
|---|---|
| `src/` | Maintained Python packages and command implementations. |
| `tests/` | Unit and contract tests for source behavior. |
| `configs/` | Reproducible experiment configuration. These are source files, not run outputs. |
| `docs/` | Small set of living human-facing notes. Do not use as a conversation archive. |
| `AGENTS.md` | Project agent/team operating rules. |

## Runtime And Generated Surface

| Path | Status | Policy |
|---|---|---|
| `.agent-team/tasks/` | Runtime ledger | Local task state. Keep as operational state, not review source. |
| `.agent-team/messages/` | Runtime ledger | Local message state. |
| `.agent-team/logs/` | Runtime ledger | Local event stream. |
| `.agent-team/briefs/` | Temporary workflow input | Use for short brainstorm/task briefs. Delete after absorption. |
| `.agent-team/artifacts/<task>/<stage>/*.md` | Department artifacts | Workflow outputs only. Prefer final synthesized artifacts over many intermediate notes. |
| `.agent-team/artifacts/<task>/cloud_*` | Legacy run outputs | Kept for compatibility. New long-lived outputs should move toward `workspace/runs/`. |
| `.agent-team/artifacts/<task>/implementation/training_bundle_*` | Legacy bundles | Kept for active cloud commands. Future bundles should move toward `workspace/bundles/`. |
| `database/` | Local data | Ignored. Do not review by walking the repo. |
| `workspace/` | Generated root | Ignored. Use for new local/cloud outputs. |

## Workspace Layout

Use this layout for new generated artifacts:

```text
workspace/
  data/
  bundles/
  runs/
  archives/
  scratch/
```

Training run families should expose a review-first layer beside raw run directories:

```text
<run-family-root>/
  summary.md
  review/
    review.md
    compact_metrics.csv
    raw_evidence.md
    figures/
  <individual-run>/
    metrics.json
    per_class_metrics.csv
    diagnostics/
    predictions/
    checkpoints/
    logs/
```

Human review should start at `review/review.md`; raw JSON, checkpoints, and logs are evidence, not the first review surface.

## Compatibility Rule

Do not move or delete existing `.agent-team/artifacts/...` paths while an active runbook still references them. Add a new path first, update runbooks, verify cloud commands, and only then archive legacy outputs.

## Minimal Markdown Rule

Maintain these project notes only unless the user explicitly asks for more:

- `docs/README.md`
- `docs/ops/project-structure.md`
- `docs/ops/review-index.md`
- `docs/ops/work-log.md`
- `docs/ops/next-actions.md`
- `docs/research/method-landscape.md`
- `docs/research/segmentation-model-data-hypotheses.md`
- `docs/implementation/current-runbook.md`

Temporary task briefs should live under `.agent-team/briefs/`, not `docs/`.

Substantial handoffs and Chinese summaries should be mirrored into the Obsidian vault:

```text
/Users/Lemon/Documents/Obsidian Vault/遥感灾害检测/Codex项目日志/
```

## Known Cleanup Candidates

| Path | Reason | Proposed Future Action |
|---|---|---|
| `tif查看.py` | Root-level exploratory utility with hard-coded local data paths. | Move into `scripts/` or replace with an argparse-based inspection command after current training runs are stable. |
| `.agent-team/artifacts/*/implementation/training_bundle_*` | Large generated bundles mixed with department notes. | Move future bundles to `workspace/bundles/`; archive old bundles after runbooks stop referencing them. |
| `.agent-team/artifacts/*/cloud_*` | Raw run outputs mixed with task artifacts. | Move future runs to `workspace/runs/`; keep Markdown summaries in `.agent-team/artifacts`. |
| date-stamped docs and colocated `-zh.md` copies | Hard to review and mostly duplicate living notes. | Do not recreate; update the living notes above and use Obsidian for Chinese handoffs. |

## Inventory Command

```bash
PYTHONPATH=src python scripts/project_inventory.py --root . --max-depth 4
```
