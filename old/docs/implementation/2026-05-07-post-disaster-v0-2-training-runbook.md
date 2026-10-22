# Post-Disaster v0.2 Training Runbook

Date: 2026-05-07

This runbook builds a post-disaster optical model-input contract, validates it, trains U-Net baselines, and writes visual feedback artifacts. It does not use the sealed test split for selection, normalization, separability, visual QA, threshold tuning, or model choice.

## Channel Preset

The first candidate stack is:

```text
F01 F02 F03 F04 F07 F11 F12
```

The index-augmented stack is:

```text
F01 F02 F03 F04 F07 F11 F12 NDVI NBR NDMI BRIGHTNESS
```

Treat this as a candidate optical/SWIR stack until `Fxx` physical band mapping is confirmed. The audit and validation commands below block missing, pre-event, or mixed-date rows. Selected channels are resampled to the mask grid before training.

Scene selection uses `quality_then_earliest`: rank complete post-disaster same-date scenes by an optical cloud-proxy score computed from RGB-like channels, then use days-after-event as a tie-breaker. This is not an official Sentinel cloud mask; inspect `quality_score_detail` and the model-input preview sheet before trusting a run.

## Build v0.2 Contract

```bash
PYTHONPATH=src python -m segmentation_contract \
  --project-root . \
  --output-dir .agent-team/artifacts/task-2ba5416843b1/implementation/segmentation_contract_v0_2 \
  --phase-b-build \
  --visual-qa \
  --model-input-previews \
  --post-disaster-audit \
  --post-selection-strategy quality_then_earliest \
  --post-window-days 90 \
  --required-channel F01 \
  --required-channel F02 \
  --required-channel F03 \
  --required-channel F04 \
  --required-channel F07 \
  --required-channel F11 \
  --required-channel F12 \
  --derived-index NDVI \
  --derived-index NBR \
  --derived-index NDMI \
  --derived-index BRIGHTNESS
```

Expected contract artifacts:

- `model_input_manifest.csv`
- `model_inputs/*.tif`
- `reports/post_disaster_channel_audit.csv`
- `reports/split_leakage_report.csv`
- `reports/post_disaster_model_input_summary.md`
- `reports/wkt_mask_alignment_report.csv`
- `reports/visual_qa_summary.md`
- `previews/contact_sheet.png`
- `previews/model_input_contact_sheet.png`
- `reports/model_input_visual_qa_summary.md`

## Validate Manifest

```bash
PYTHONPATH=src python -m segmentation_training validate-manifest \
  --contract-dir .agent-team/artifacts/task-2ba5416843b1/implementation/segmentation_contract_v0_2 \
  --required-channel F01 \
  --required-channel F02 \
  --required-channel F03 \
  --required-channel F04 \
  --required-channel F07 \
  --required-channel F11 \
  --required-channel F12
```

The current real build validates 668 train/validation rows for both the 7-channel raw stack and the 11-channel index-augmented stack. Do not train by silently dropping channels or falling back to `F16/F17`.

## Build Cloud Bundle

```bash
PYTHONPATH=src python -m segmentation_training bundle \
  --contract-dir .agent-team/artifacts/task-2ba5416843b1/implementation/segmentation_contract_v0_2 \
  --bundle-dir .agent-team/artifacts/task-2ba5416843b1/implementation/training_bundle_v0_2 \
  --bundle-version 0.2 \
  --class-scope binary_c5 \
  --cloud-data-root /data/training_bundle_v0_2 \
  --required-channel F01 \
  --required-channel F02 \
  --required-channel F03 \
  --required-channel F04 \
  --required-channel F07 \
  --required-channel F11 \
  --required-channel F12
```

Build a C2 bundle by changing `--class-scope binary_c2`.

Build the V3 index-augmented C5 bundle:

```bash
PYTHONPATH=src python -m segmentation_training bundle \
  --contract-dir .agent-team/artifacts/task-2ba5416843b1/implementation/segmentation_contract_v0_2 \
  --bundle-dir .agent-team/artifacts/task-2ba5416843b1/implementation/training_bundle_v0_3_indices \
  --bundle-version 0.3 \
  --class-scope binary_c5 \
  --cloud-data-root /data/training_bundle_v0_3_indices \
  --required-channel F01 \
  --required-channel F02 \
  --required-channel F03 \
  --required-channel F04 \
  --required-channel F07 \
  --required-channel F11 \
  --required-channel F12 \
  --required-channel NDVI \
  --required-channel NBR \
  --required-channel NDMI \
  --required-channel BRIGHTNESS
```

## Dry Run

```bash
PYTHONPATH=src python -m segmentation_training dry-run \
  --config configs/segmentation_training/v2_binary_c5_unet_post_optical.yaml \
  --bundle-dir .agent-team/artifacts/task-2ba5416843b1/implementation/training_bundle_v0_2 \
  --max-samples 8
```

V3 index dry run:

```bash
PYTHONPATH=src python -m segmentation_training dry-run \
  --config configs/segmentation_training/v3_binary_c5_unet_post_optical_indices_smoke.yaml \
  --bundle-dir .agent-team/artifacts/task-2ba5416843b1/implementation/training_bundle_v0_3_indices \
  --max-samples 8
```

## Train

Full training is intended for a Python environment with `torch`, `rasterio`, `shapely`, `numpy`, `Pillow`, `matplotlib`, and `PyYAML` installed. The bundle contains masks, aligned `model_inputs`, manifests, reports, and configs.

The current verified local training environment is:

```bash
/Users/Lemon/miniconda3/envs/d2l-zh/bin/python
```

If rebuilding the environment, install the geospatial runtime pieces before running contract or training verification:

```bash
/Users/Lemon/miniconda3/envs/d2l-zh/bin/python -m pip install rasterio==1.3.11 shapely==2.0.7
```

Smoke run:

```bash
PYTHONPATH=src python -m segmentation_training train \
  --config /data/training_bundle_v0_2/configs/v2_binary_c5_unet_post_optical_smoke.yaml \
  --bundle-dir /data/training_bundle_v0_2 \
  --run-dir /runs/post_disaster_v0_2/V2_binary_c5_unet_post_optical_smoke \
  --cloud-confirm
```

V3 index smoke run:

```bash
PYTHONPATH=src python -m segmentation_training train \
  --config /data/training_bundle_v0_3_indices/configs/v3_binary_c5_unet_post_optical_indices_smoke.yaml \
  --bundle-dir /data/training_bundle_v0_3_indices \
  --run-dir /runs/post_disaster_v0_3/V3_binary_c5_unet_post_optical_indices_smoke \
  --cloud-confirm
```

Verified local V3 index smoke run:

```bash
PYTHONPATH=src /Users/Lemon/miniconda3/envs/d2l-zh/bin/python -m segmentation_training train \
  --config .agent-team/artifacts/task-2ba5416843b1/implementation/training_bundle_v0_3_indices/configs/v3_binary_c5_unet_post_optical_indices_smoke.yaml \
  --bundle-dir .agent-team/artifacts/task-2ba5416843b1/implementation/training_bundle_v0_3_indices \
  --run-dir .agent-team/artifacts/task-2ba5416843b1/implementation/runs/v3_binary_c5_indices_smoke \
  --cloud-confirm
```

Verified output on 2026-05-07:

- `status`: `completed_cloud_training`
- `train_records`: 301
- `validation_records`: 78
- `input_channels`: `F01;F02;F03;F04;F07;F11;F12;NDVI;NBR;NDMI;BRIGHTNESS`
- `mean_iou`: `0.19590715476263967`
- `foreground_recall`: `1.0`
- `train_loss_last`: `1.2567567825317383`
- `diagnostics/artifact_inventory.json`: no missing expected artifacts

Full run:

```bash
PYTHONPATH=src python -m segmentation_training train \
  --config /data/training_bundle_v0_2/configs/v2_binary_c5_unet_post_optical.yaml \
  --bundle-dir /data/training_bundle_v0_2 \
  --run-dir /runs/post_disaster_v0_2/V2_binary_c5_unet_post_optical \
  --cloud-confirm
```

V3 index full run:

```bash
PYTHONPATH=src python -m segmentation_training train \
  --config /data/training_bundle_v0_3_indices/configs/v3_binary_c5_unet_post_optical_indices.yaml \
  --bundle-dir /data/training_bundle_v0_3_indices \
  --run-dir /runs/post_disaster_v0_3/V3_binary_c5_unet_post_optical_indices \
  --cloud-confirm
```

Run C2 by replacing the config and run directory with `v2_binary_c2_unet_post_optical`.

## Visual Feedback Outputs

Each completed run writes:

- `metrics.json`
- `per_class_metrics.csv`
- `confusion_matrix.csv`
- `diagnostics/validation_prediction_summary.csv`
- `diagnostics/threshold_sweep.csv`
- `diagnostics/foreground_probability_summary.csv`
- `diagnostics/foreground_window_coverage.csv`
- `predictions/validation_contact_sheet.png`
- `predictions/failures_contact_sheet.png`
- `predictions/per_sample/*.png`
- `training_curves.png`
- `training_curves.csv`

If a previous run lacks curves, generate them with:

```bash
PYTHONPATH=src python -m segmentation_training plot-run \
  --run-dir /runs/post_disaster_v0_2/V2_binary_c5_unet_post_optical
```
