---
title: U-Net Channel Ablation Runbook
date: 2026-05-07
tags:
  - segmentation
  - ablation
  - unet
  - multispectral
status: ready-for-cloud-smoke
task: task-6bd957f6e1ed
---

# U-Net Channel Ablation Runbook

This runbook tests whether the weak U-Net signal is caused by the data contract, channel stack, or derived-index mix rather than architecture alone.

It keeps the model family fixed to U-Net and changes only `input_channels`.

> [!warning]
> Do not use the sealed test split. These runs are train/validation-only diagnostics.

## Ablation Matrix

| Config family | Channels | Purpose |
|---|---|---|
| RGB optical | `F04,F03,F02` | Check whether a simple RGB-like post-disaster signal is stronger than the current high-dimensional stack. |
| False color | `F07,F04,F03` | Check whether NIR + red/green separates vegetation, burn, or debris-related change better. |
| Base optical/SWIR | `F01,F02,F03,F04,F07,F11,F12` | Check whether raw aligned reflectance bands outperform derived-index mixing. |
| Indices only | `NDVI,NBR,NDMI,BRIGHTNESS` | Check whether hand-crafted index channels carry most of the usable signal. |
| Current 11-channel stack | `F01,F02,F03,F04,F07,F11,F12,NDVI,NBR,NDMI,BRIGHTNESS` | Control arm matching the current U-Net data contract. |

## Smoke First

On the cloud machine:

```bash
cd /root/autodl-tmp/New_project
git pull --ff-only origin codex/rtx4080-training-performance

export PYTHONPATH=src
export BUNDLE_DIR=.agent-team/artifacts/task-2ba5416843b1/implementation/training_bundle_v0_3_indices
export RUN_ROOT=.agent-team/artifacts/task-6bd957f6e1ed/cloud_unet_channel_ablation_smoke

for config in configs/segmentation_training/unet_channel_ablation/*_smoke.yaml; do
  name="$(basename "$config" .yaml)"
  python -m segmentation_training train \
    --config "$config" \
    --bundle-dir "$BUNDLE_DIR" \
    --run-dir "$RUN_ROOT/$name" \
    --cloud-confirm
done

python -m segmentation_training summarize-runs \
  --runs-dir "$RUN_ROOT" \
  --output "$RUN_ROOT/summary.md"
```

Smoke acceptance requires:

- every run reports `completed_cloud_training`;
- every `run_manifest.json` reports `test_split_read: false`;
- metrics, diagnostics, curves, contact sheets, and checkpoints exist;
- no channel set needs a separate sampler, loss, or learning-rate change.

## Full Validation-Only Ablation

After smoke passes:

```bash
export PYTHONPATH=src
export BUNDLE_DIR=.agent-team/artifacts/task-2ba5416843b1/implementation/training_bundle_v0_3_indices
export RUN_ROOT=.agent-team/artifacts/task-6bd957f6e1ed/cloud_unet_channel_ablation

configs=(
  configs/segmentation_training/unet_channel_ablation/c5_unet_rgb_optical_rtx4080.yaml
  configs/segmentation_training/unet_channel_ablation/c5_unet_false_color_rtx4080.yaml
  configs/segmentation_training/unet_channel_ablation/c5_unet_post_optical_7ch_rtx4080.yaml
  configs/segmentation_training/unet_channel_ablation/c5_unet_indices_only_rtx4080.yaml
  configs/segmentation_training/unet_channel_ablation/c5_unet_11ch_indices_rtx4080.yaml
)

for config in "${configs[@]}"; do
  name="$(basename "$config" .yaml)"
  python -m segmentation_training train \
    --config "$config" \
    --bundle-dir "$BUNDLE_DIR" \
    --run-dir "$RUN_ROOT/$name" \
    --cloud-confirm
done

python -m segmentation_training summarize-runs \
  --runs-dir "$RUN_ROOT" \
  --output "$RUN_ROOT/summary.md"
```

## Interpretation Rules

- If RGB or false-color beats 11-channel, suspect channel noise, normalization behavior, or derived-index mismatch.
- If 7-channel beats 11-channel, suspect the derived indices or their per-window normalization.
- If indices-only beats raw bands, suspect that U-Net is struggling to learn the physical ratios from sparse labels.
- If all channel sets fail similarly, prioritize label alignment, mask sparsity, class definition, sampler coverage, and loss/threshold behavior.
- Do not choose a final model from this ablation alone; it explains data contract behavior for U-Net.

## Evidence To Return

Archive and return:

- `summary.md`;
- each `run_manifest.json`;
- each `metrics.json`;
- each `per_class_metrics.csv`;
- `diagnostics/threshold_sweep.csv`;
- `diagnostics/foreground_window_coverage.csv`;
- `predictions/validation_contact_sheet.png`;
- `training_curves.png`;
- best checkpoint metadata.

