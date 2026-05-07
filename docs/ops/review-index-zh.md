---
title: 审查入口
date: 2026-05-07
tags:
  - ops
  - review
  - segmentation
status: active
---

# 审查入口

遍历工作区前先看这里。仓库包含大型 ignored local data 和 generated run artifacts，因此审查应从 source files、runbooks 和 summaries 开始。

## 当前 Source Entry Points

| Area | Entry |
|---|---|
| Project instructions | `AGENTS.md` |
| Structure policy | `docs/ops/project-structure.md` |
| Architecture benchmark configs | `configs/segmentation_training/architectures/` |
| Architecture smoke configs | `configs/segmentation_training/architecture_smoke/` |
| U-Net channel ablation configs | `configs/segmentation_training/unet_channel_ablation/` |
| Model registry | `src/segmentation_training/models/registry.py` |
| Cloud training path | `src/segmentation_training/cloud.py` |
| Training CLI | `src/segmentation_training/cli.py` |
| Config validation tests | `tests/test_segmentation_training_config.py` |
| Model registry tests | `tests/test_segmentation_training_models.py` |
| Cloud training tests | `tests/test_segmentation_training_cloud.py` |

## 当前 Human Notes

| Topic | Entry |
|---|---|
| Architecture benchmark runway | `docs/implementation/2026-05-07-segmentation-v4-cloud-smoke-runbook.md` |
| Architecture benchmark review | `docs/implementation/2026-05-07-cloud-benchmark-review-zh.md` |
| U-Net channel ablation | `docs/implementation/2026-05-07-unet-channel-ablation-runbook.md` |
| V6 11-channel U-Net augmentation diagnostics | `docs/implementation/2026-05-07-v6-11ch-unet-augmentation-diagnostics-runbook-zh.md` |
| Model/data hypotheses | `docs/research/memos/2026-05-07-segmentation-model-data-hypotheses.md` |
| Post-disaster training summary | `docs/implementation/2026-05-07-post-disaster-multispectral-training-summary.md` |

## Runtime Evidence Entry Points

Runtime evidence 被 Git 忽略，而且可能很大。存在 summary 时优先看：

| Evidence | Preferred Entry |
|---|---|
| Architecture cloud smoke | `.agent-team/artifacts/task-6bd957f6e1ed/cloud_smoke/review/review.md` |
| Architecture cloud benchmark | `.agent-team/artifacts/task-6bd957f6e1ed/cloud_benchmark/review/review.md` |
| U-Net channel ablation smoke | `.agent-team/artifacts/task-6bd957f6e1ed/cloud_unet_channel_ablation_smoke/review/review.md` |
| U-Net channel ablation full run | `.agent-team/artifacts/task-6bd957f6e1ed/cloud_unet_channel_ablation/review/review.md` |
| V6 11-channel U-Net diagnostics | `workspace/runs/segmentation/v6_11ch_unet_aug_diagnostics/full/review/review.md` |

如果 summary 缺失，用下面命令生成：

```bash
PYTHONPATH=src python -m segmentation_training summarize-runs \
  --runs-dir <run-root> \
  --output <run-root>/summary.md
```

该命令也会写出 `<run-root>/review/review.md`、`<run-root>/review/compact_metrics.csv`、`<run-root>/review/raw_evidence.md`，并把关键图片复制到 `<run-root>/review/figures/`。

## Inventory 命令

```bash
PYTHONPATH=src python scripts/project_inventory.py --root . --max-depth 4
```

用输出判断当前看到的是 source、documentation、local data、runtime ledger 还是 generated workspace state。
