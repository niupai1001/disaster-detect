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

Start here before walking the workspace. The repository has large ignored local data and generated run artifacts, so review should use source files, runbooks, and summaries as entrypoints.

## Current Source Entry Points

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

## Current Human Notes

| Topic | Entry |
|---|---|
| Architecture benchmark runway | `docs/implementation/2026-05-07-segmentation-v4-cloud-smoke-runbook.md` |
| Architecture benchmark review | `docs/implementation/2026-05-07-cloud-benchmark-review.md` |
| U-Net channel ablation | `docs/implementation/2026-05-07-unet-channel-ablation-runbook.md` |
| V6 11-channel U-Net augmentation diagnostics | `docs/implementation/2026-05-07-v6-11ch-unet-augmentation-diagnostics-runbook.md` |
| Model/data hypotheses | `docs/research/memos/2026-05-07-segmentation-model-data-hypotheses.md` |
| Post-disaster training summary | `docs/implementation/2026-05-07-post-disaster-multispectral-training-summary.md` |

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

This command also writes `<run-root>/review/review.md`, `<run-root>/review/compact_metrics.csv`, `<run-root>/review/raw_evidence.md`, and selected figures under `<run-root>/review/figures/`.

## Inventory Command

```bash
PYTHONPATH=src python scripts/project_inventory.py --root . --max-depth 4
```

Use the output to decide whether you are looking at source, documentation, local data, runtime ledger, or generated workspace state.
