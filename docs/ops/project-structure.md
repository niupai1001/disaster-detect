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

This project separates reviewable source files from generated data and runtime artifacts. The current cloud training chain remains compatible with legacy `.agent-team/artifacts/...` paths until the active benchmark and ablation runs are returned.

## Reviewable Source Surface

| Path | Purpose |
|---|---|
| `src/` | Maintained Python packages and command implementations. |
| `tests/` | Unit and contract tests for source behavior. |
| `configs/` | Reproducible experiment configuration. These are source files, not run outputs. |
| `docs/` | Human-facing plans, runbooks, research notes, and review indexes. |
| `AGENTS.md` | Project agent/team operating rules. |

Review should start from these paths and the current Git diff.

## Runtime And Generated Surface

| Path | Status | Policy |
|---|---|---|
| `.agent-team/tasks/` | Runtime ledger | Local task state. Keep as operational state, not review source. |
| `.agent-team/messages/` | Runtime ledger | Local message state. |
| `.agent-team/logs/` | Runtime ledger | Local event stream. |
| `.agent-team/artifacts/<task>/<stage>/*.md` | Department artifacts | Useful for project memory; prefer Markdown summaries over raw outputs. |
| `.agent-team/artifacts/<task>/cloud_*` | Legacy run outputs | Kept for compatibility. New long-lived outputs should move toward `workspace/runs/`. |
| `.agent-team/artifacts/<task>/implementation/training_bundle_*` | Legacy bundles | Kept for active cloud commands. Future bundles should move toward `workspace/bundles/`. |
| `database/` | Local data | Ignored. Do not review by walking the repo. |
| `workspace/` | Future generated root | Ignored. Use for new local/cloud outputs after migration. |

## Future Workspace Layout

Use this layout for new generated artifacts once a runbook has been updated to opt in:

```text
workspace/
  data/
  bundles/
  runs/
  archives/
  scratch/
```

The path is ignored by Git. It is intended for local and cloud runtime state, not source review.

## Phase 2 Run Output Layout

Training run families should now expose a review-first layer beside the raw run directories:

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

`summary.md` remains for legacy scripts and quick checks. Human review should start at `review/review.md`; raw JSON, checkpoints, and logs are evidence, not the first review surface.

## Compatibility Rule

Do not move or delete existing `.agent-team/artifacts/...` paths while an active runbook still references them. Add a new path first, update runbooks, verify cloud commands, and only then archive legacy outputs.

## Known Cleanup Candidates

| Path | Reason | Proposed Future Action |
|---|---|---|
| `tif查看.py` | Root-level exploratory utility with hard-coded local data paths. | Move into `scripts/` or replace with an argparse-based inspection command after current training runs are stable. |
| `.agent-team/artifacts/*/implementation/training_bundle_*` | Large generated bundles mixed with department notes. | Move future bundles to `workspace/bundles/`; archive old bundles after runbooks stop referencing them. |
| `.agent-team/artifacts/*/cloud_*` | Raw run outputs mixed with task artifacts. | Move future runs to `workspace/runs/`; keep Markdown summaries in `.agent-team/artifacts`. |

## Inventory Command

Use this command to summarize the workspace without manually browsing generated outputs:

```bash
PYTHONPATH=src python scripts/project_inventory.py --root . --max-depth 4
```

The command reports top-level categories, largest roots, and file extensions. It ignores `.git`.

## Review Rule

A pull request or local review should not require opening raw `metrics.json`, `run_manifest.json`, checkpoints, or raster directories first. Each experiment family should provide a Markdown `summary.md`, runbook, or review note that links to raw evidence only when needed.
