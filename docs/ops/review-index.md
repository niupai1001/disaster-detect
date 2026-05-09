---
title: Review Index
date: 2026-05-07
tags:
  - ops
  - review
  - segmentation
status: active
---

# Review Index

Start here before walking the workspace. The repository has large ignored local data and generated run artifacts, so review should use source files, living notes, runbooks, and summaries as entrypoints.

## Current Source Entry Points

| Area | Entry |
|---|---|
| Project instructions | `AGENTS.md` |
| Project notes index | `docs/README.md` |
| Structure policy | `docs/ops/project-structure.md` |
| Work log | `docs/ops/work-log.md` |
| Next actions | `docs/ops/next-actions.md` |
| Architecture benchmark configs | `configs/segmentation_training/architectures/` |
| Architecture smoke configs | `configs/segmentation_training/architecture_smoke/` |
| U-Net channel ablation configs | `configs/segmentation_training/unet_channel_ablation/` |
| V6 U-Net diagnostics configs | `configs/segmentation_training/v6_11ch_unet_aug_diagnostics/` |
| Model registry | `src/segmentation_training/models/registry.py` |
| Cloud training path | `src/segmentation_training/cloud.py` |
| Training CLI | `src/segmentation_training/cli.py` |
| Config validation tests | `tests/test_segmentation_training_config.py` |
| Model registry tests | `tests/test_segmentation_training_models.py` |
| Cloud training tests | `tests/test_segmentation_training_cloud.py` |

## Current Human Notes

| Topic | Entry |
|---|---|
| Current runbook | `docs/implementation/current-runbook.md` |
| Model/data hypotheses | `docs/research/segmentation-model-data-hypotheses.md` |
| Method landscape | `docs/research/method-landscape.md` |

## Runtime Evidence Entry Points

Runtime evidence is ignored by Git and may be large. Prefer these summary files when present:

| Evidence | Preferred Entry |
|---|---|
| Architecture cloud smoke | `.agent-team/artifacts/task-6bd957f6e1ed/cloud_smoke/review/review.md` |
| Architecture cloud benchmark | `.agent-team/artifacts/task-6bd957f6e1ed/cloud_benchmark/review/review.md` |
| U-Net channel ablation smoke | `.agent-team/artifacts/task-6bd957f6e1ed/cloud_unet_channel_ablation_smoke/review/review.md` |
| U-Net channel ablation full run | `.agent-team/artifacts/task-6bd957f6e1ed/cloud_unet_channel_ablation/review/review.md` |
| V6 11-channel U-Net diagnostics | `workspace/runs/segmentation/v6_11ch_unet_aug_diagnostics/full/review/review.md` |

If a summary is missing, generate it with:

```bash
PYTHONPATH=src python -m segmentation_training summarize-runs \
  --runs-dir <run-root> \
  --output <run-root>/summary.md
```

Use the inventory command before manually browsing generated outputs:

```bash
PYTHONPATH=src python scripts/project_inventory.py --root . --max-depth 4
```
