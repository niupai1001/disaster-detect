---
title: 项目结构治理
date: 2026-05-07
tags:
  - ops
  - review
  - structure
  - obsidian
status: active
---

# 项目结构治理

仓库现在已有第一阶段治理层。它收窄审查面，同时保留当前云端 run paths。

## Source Surface

- `src/`
- `configs/`
- `tests/`
- `docs/`
- `AGENTS.md`

## Runtime Surface

- `.agent-team/` 继续作为 runtime ledger。
- 当前 `.agent-team/artifacts/...` 路径继续对 cloud runs 有效。
- Future generated state 应逐步迁移到 ignored `workspace/`。

## 审查入口

- [[docs/ops/project-structure-zh|项目结构]]
- [[docs/ops/review-index-zh|审查入口]]

## Inventory 命令

```bash
PYTHONPATH=src python scripts/project_inventory.py --root . --max-depth 3
```

