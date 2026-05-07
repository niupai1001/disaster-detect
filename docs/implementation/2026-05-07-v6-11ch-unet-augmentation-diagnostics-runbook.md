---
title: V6 11-Channel U-Net Augmentation Diagnostics Runbook
date: 2026-05-07
tags:
  - segmentation
  - unet
  - augmentation
  - diagnostics
status: ready-for-cloud-smoke
task: task-6bd957f6e1ed
---

# V6 11-Channel U-Net Augmentation Diagnostics Runbook

This runbook keeps the 11-channel post-disaster input contract fixed and tests whether safer training protocol changes improve U-Net behavior.

It does not treat single-date post-disaster imagery as a primary failure cause. The working hypothesis is now:

- U-Net may be too shallow for the current validation windows;
- channel selection alone did not solve the bottleneck;
- training augmentation, threshold calibration, full validation previews, and worst false-positive review should be added before another broad architecture sweep.

> [!warning]
> Do not use the sealed test split. These runs remain train/validation-only.

## New Workspace Layout

Use `workspace/` for new cloud outputs:

```text
workspace/
  runs/
    segmentation/
      v6_11ch_unet_aug_diagnostics/
        smoke/
        full/
        channel_contribution/
        review/
  bundles/
  archives/
```

The legacy `.agent-team/artifacts/...` paths remain readable for old evidence, but new run outputs should go under `workspace/runs/`.

## What Changed

- Input remains the same 11 channels: `F01,F02,F03,F04,F07,F11,F12,NDVI,NBR,NDMI,BRIGHTNESS`.
- U-Net remains the model family for this diagnostic run.
- Training adds safe spatial augmentation: horizontal flip, vertical flip, and 90-degree rotation.
- Full validation exports all validation previews instead of only the first 12.
- Worst false positives are exported separately for label/mask alignment review.
- A post-training `channel-contribution` command can evaluate zero-occlusion impact for each channel and configured channel groups.

## Smoke

On the cloud machine:

```bash
cd /root/autodl-tmp/New_project
git pull --ff-only origin codex/rtx4080-training-performance

export PYTHONPATH=src
export BUNDLE_DIR=.agent-team/artifacts/task-2ba5416843b1/implementation/training_bundle_v0_3_indices
export RUN_ROOT=workspace/runs/segmentation/v6_11ch_unet_aug_diagnostics/smoke

python -m segmentation_training train \
  --config configs/segmentation_training/v6_11ch_unet_aug_diagnostics/c5_11ch_unet_aug_smoke.yaml \
  --bundle-dir "$BUNDLE_DIR" \
  --run-dir "$RUN_ROOT/c5_11ch_unet_aug_smoke" \
  --cloud-confirm

python -m segmentation_training summarize-runs \
  --runs-dir "$RUN_ROOT" \
  --output "$RUN_ROOT/summary.md"
```

Smoke acceptance:

- `run_manifest.json` reports `test_split_read: false`;
- `metrics.json`, `per_class_metrics.csv`, `training_curves.*`, and checkpoint metadata exist;
- `predictions/per_sample/` contains all smoke validation previews;
- `predictions/worst_false_positives/` and `diagnostics/worst_false_positives.csv` exist.

## Full Run

After smoke passes:

```bash
export PYTHONPATH=src
export BUNDLE_DIR=.agent-team/artifacts/task-2ba5416843b1/implementation/training_bundle_v0_3_indices
export RUN_ROOT=workspace/runs/segmentation/v6_11ch_unet_aug_diagnostics/full

python -m segmentation_training train \
  --config configs/segmentation_training/v6_11ch_unet_aug_diagnostics/c5_11ch_unet_aug_rtx4080.yaml \
  --bundle-dir "$BUNDLE_DIR" \
  --run-dir "$RUN_ROOT/c5_11ch_unet_aug_rtx4080" \
  --cloud-confirm

python -m segmentation_training summarize-runs \
  --runs-dir "$RUN_ROOT" \
  --output "$RUN_ROOT/summary.md"
```

Review starts at:

```text
workspace/runs/segmentation/v6_11ch_unet_aug_diagnostics/full/review/review.md
```

## Channel Contribution

After the full run, evaluate 11-channel contribution using the best checkpoint:

```bash
export RUN_DIR=workspace/runs/segmentation/v6_11ch_unet_aug_diagnostics/full/c5_11ch_unet_aug_rtx4080
export CONTRIBUTION_DIR=workspace/runs/segmentation/v6_11ch_unet_aug_diagnostics/channel_contribution

python -m segmentation_training channel-contribution \
  --config configs/segmentation_training/v6_11ch_unet_aug_diagnostics/c5_11ch_unet_aug_rtx4080.yaml \
  --bundle-dir "$BUNDLE_DIR" \
  --checkpoint "$RUN_DIR/checkpoints/best_mean_iou.pt" \
  --output-dir "$CONTRIBUTION_DIR" \
  --cloud-confirm
```

This writes:

```text
channel_contribution.csv
channel_contribution_manifest.json
```

Interpretation: a larger `mean_iou_drop` after zeroing a channel/group means the trained 11-channel model depended more on that signal.

## Evidence To Return

Return or download:

- `full/review/review.md`;
- `full/review/compact_metrics.csv`;
- `full/c5_11ch_unet_aug_rtx4080/diagnostics/worst_false_positives.csv`;
- `full/c5_11ch_unet_aug_rtx4080/predictions/worst_false_positives/`;
- `full/c5_11ch_unet_aug_rtx4080/predictions/per_sample/`;
- `channel_contribution/channel_contribution.csv`;
- training curves and best checkpoint metadata.
