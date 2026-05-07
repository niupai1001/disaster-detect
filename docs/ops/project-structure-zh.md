---
title: 项目结构
date: 2026-05-07
tags:
  - ops
  - project-structure
  - review
status: active
---

# 项目结构

本项目把可审查 source files 与 generated data/runtime artifacts 分离。当前云端训练链路在 active benchmark 和 ablation runs 返回前，继续兼容 legacy `.agent-team/artifacts/...` 路径。

## 可审查 Source Surface

| Path | Purpose |
|---|---|
| `src/` | 维护中的 Python packages 和命令实现。 |
| `tests/` | source behavior 的单元测试和契约测试。 |
| `configs/` | 可复现实验配置。这些是 source files，不是 run outputs。 |
| `docs/` | 面向人的 plans、runbooks、research notes 和 review indexes。 |
| `AGENTS.md` | 项目 agent/team 操作规则。 |

审查应从这些路径和当前 Git diff 开始。

## Runtime And Generated Surface

| Path | Status | Policy |
|---|---|---|
| `.agent-team/tasks/` | Runtime ledger | 本地任务状态。作为 operational state，不作为 review source。 |
| `.agent-team/messages/` | Runtime ledger | 本地消息状态。 |
| `.agent-team/logs/` | Runtime ledger | 本地事件流。 |
| `.agent-team/artifacts/<task>/<stage>/*.md` | Department artifacts | 对项目记忆有用；优先用 Markdown summaries，不直接审 raw outputs。 |
| `.agent-team/artifacts/<task>/cloud_*` | Legacy run outputs | 为兼容保留。新的长期 outputs 应逐步迁移到 `workspace/runs/`。 |
| `.agent-team/artifacts/<task>/implementation/training_bundle_*` | Legacy bundles | 为当前云端命令保留。未来 bundles 应逐步迁移到 `workspace/bundles/`。 |
| `database/` | Local data | 被忽略。不要通过遍历 repo 来审查。 |
| `workspace/` | Future generated root | 被忽略。迁移后用于新的 local/cloud outputs。 |

## Future Workspace Layout

当某个 runbook 明确 opt in 新路径后，新的 generated artifacts 使用：

```text
workspace/
  data/
  bundles/
  runs/
  archives/
  scratch/
```

该路径被 Git 忽略，只用于 local/cloud runtime state，不用于 source review。

## 二阶段 Run 输出结构

后续 training run family 应在 raw run directories 旁边提供 review-first 层：

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

`summary.md` 继续兼容旧脚本和快速检查。人工审查应该从 `review/review.md` 开始；raw JSON、checkpoints 和 logs 是证据，不是第一审查入口。

## 兼容规则

只要 active runbook 仍引用 `.agent-team/artifacts/...`，就不要移动或删除这些路径。先增加新路径，更新 runbooks，验证云端命令，再归档 legacy outputs。

## 已知清理候选项

| Path | Reason | Proposed Future Action |
|---|---|---|
| `tif查看.py` | 根目录 exploratory utility，带硬编码本地数据路径。 | 当前训练稳定后，移动到 `scripts/`，或替换为 argparse-based inspection command。 |
| `.agent-team/artifacts/*/implementation/training_bundle_*` | 大型 generated bundles 与部门 notes 混在一起。 | 后续 bundles 迁移到 `workspace/bundles/`；旧 runbooks 不再引用后再归档旧 bundles。 |
| `.agent-team/artifacts/*/cloud_*` | Raw run outputs 与 task artifacts 混在一起。 | 后续 runs 迁移到 `workspace/runs/`；`.agent-team/artifacts` 只保留 Markdown summaries。 |

## Inventory 命令

用下面命令汇总工作区，不需要手动浏览 generated outputs：

```bash
PYTHONPATH=src python scripts/project_inventory.py --root . --max-depth 4
```

该命令报告 top-level categories、largest roots 和 file extensions，并忽略 `.git`。

## 审查规则

Pull request 或本地审查不应该先打开 raw `metrics.json`、`run_manifest.json`、checkpoints 或 raster directories。每个实验 family 都应该提供 Markdown `summary.md`、runbook 或 review note，并只在需要时链接 raw evidence。
