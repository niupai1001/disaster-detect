# Project Structure Governance Design

Date: 2026-05-07

## Goal

Reduce review friction by separating maintainable project files from local data, bundles, run outputs, checkpoints, and agent runtime artifacts without breaking current cloud training commands.

## Problem

The repository itself is small, but the shared workspace is noisy:

- `database/` contains large local raster data.
- `.agent-team/artifacts/` mixes department Markdown, training bundles, cloud run outputs, checkpoints, diagnostics, and downloaded smoke evidence.
- Reviewers who inspect the workspace see many JSON, CSV, PNG, PT, and TIF files before they reach the code and decision records.

The current training runbooks and task ledger reference `.agent-team/artifacts/...`, so an immediate move would be risky.

## Design

Add a governance layer before moving data:

- Keep `src/`, `configs/`, `tests/`, and `docs/` as the reviewable source surface.
- Introduce `workspace/` as the future ignored root for local data, bundles, runs, archives, and scratch outputs.
- Keep `.agent-team/` as a runtime ledger. Going forward, use it primarily for task records, messages, and department Markdown artifacts.
- Preserve existing `.agent-team/artifacts/...` paths until current cloud benchmark and ablation runs are returned.
- Add an inventory tool that classifies files by purpose and produces Markdown for review.
- Add an ops review index so reviewers start from a small set of canonical files instead of walking run directories.

## Non-Goals

- Do not move existing `.agent-team/artifacts` content in this phase.
- Do not change active cloud run commands in current runbooks.
- Do not rewrite `agent_team` storage semantics in this phase.
- Do not delete local datasets or training outputs.

## Acceptance Criteria

- A reviewer can run one command to see the project file categories and the largest local state roots.
- `workspace/` is ignored by Git.
- Documentation states which directories are source, which are generated, and which are legacy-compatible runtime paths.
- Existing tests still pass.
- No current cloud training path is removed.

