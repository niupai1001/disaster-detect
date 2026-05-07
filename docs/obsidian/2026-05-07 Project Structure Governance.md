---
title: Project Structure Governance
date: 2026-05-07
tags:
  - ops
  - review
  - structure
  - obsidian
status: active
---

# Project Structure Governance

The repository now has a first-stage governance layer. It narrows the review surface while preserving current cloud run paths.

## Source Surface

- `src/`
- `configs/`
- `tests/`
- `docs/`
- `AGENTS.md`

## Runtime Surface

- `.agent-team/` remains a runtime ledger.
- Existing `.agent-team/artifacts/...` paths remain valid for current cloud runs.
- Future generated state should move toward ignored `workspace/`.

## Review Entrypoints

- [[docs/ops/project-structure|Project Structure]]
- [[docs/ops/review-index|Review Index]]

## Inventory Command

```bash
PYTHONPATH=src python scripts/project_inventory.py --root . --max-depth 3
```

