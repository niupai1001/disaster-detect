# SegmentationTrainingBaseline v0.1 Cloud Runbook

Date: 2026-05-05

This runbook keeps `SegmentationTrainingBaseline v0.1` cloud-first. The local machine may validate manifests, build bundles, run unit tests, run synthetic tests, and perform tiny smoke checks. Full E1-E3 model training must run on the cloud server.

## Inputs

Canonical local contract:

```bash
.agent-team/artifacts/task-ef3a9c94ddb7/implementation/segmentation_contract_v0_1
```

Local bundle target:

```bash
.agent-team/artifacts/task-afc7f2c25f8f/implementation/training_bundle_v0_1
```

The v0.1 channel contract is `F16;F17`. The test split is sealed and must not be used for model selection.

## Local Preflight

Run from the project root:

```bash
PYTHONPATH=src python -m unittest discover -s tests

PYTHONPATH=src python -m segmentation_training validate-manifest \
  --contract-dir .agent-team/artifacts/task-ef3a9c94ddb7/implementation/segmentation_contract_v0_1

PYTHONPATH=src python -m segmentation_training bundle \
  --contract-dir .agent-team/artifacts/task-ef3a9c94ddb7/implementation/segmentation_contract_v0_1 \
  --bundle-dir .agent-team/artifacts/task-afc7f2c25f8f/implementation/training_bundle_v0_1 \
  --required-channel F16 \
  --required-channel F17 \
  --mode manifest-only

PYTHONPATH=src python -m segmentation_training dry-run \
  --config configs/segmentation_training/e1_binary_c5_unet.yaml \
  --contract-dir .agent-team/artifacts/task-ef3a9c94ddb7/implementation/segmentation_contract_v0_1 \
  --max-samples 4
```

Do not run `train --cloud-confirm` locally.

## Upload

Upload the repository source and the generated bundle to the cloud server. The bundle expects the cloud data root to be:

```bash
/data/training_bundle_v0_1
```

The bundle contains masks, metadata, source reports, configs, and `manifests/cloud_model_input_manifest.csv`. Large raster files are not copied into the bundle; mount or copy the project raster tree so paths under `/data/training_bundle_v0_1/database/...` resolve.

## Cloud Environment

Use Python 3.11. Install packages compatible with the cloud image and CUDA driver:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pillow pyyaml rasterio tqdm
# Install PyTorch from the official wheel selector for the server CUDA version.
```

Do not pin CUDA wheels in the repository until the actual image is known.

## Cloud Validation

On the cloud server, after mounting data:

```bash
export PYTHONPATH=src

python -m segmentation_training validate-manifest \
  --contract-dir /data/training_bundle_v0_1

python -m segmentation_training dry-run \
  --config /data/training_bundle_v0_1/configs/e1_binary_c5_unet.yaml \
  --bundle-dir /data/training_bundle_v0_1 \
  --max-samples 4
```

If this fails, fix mount paths or bundle generation before training.

## E1-E3 Cloud Runs

Create one run directory per experiment:

```bash
python -m segmentation_training train \
  --config /data/training_bundle_v0_1/configs/e1_binary_c5_unet.yaml \
  --bundle-dir /data/training_bundle_v0_1 \
  --run-dir /runs/segmentation_training/e1_binary_c5_unet \
  --cloud-confirm

python -m segmentation_training train \
  --config /data/training_bundle_v0_1/configs/e2_binary_c2_unet.yaml \
  --bundle-dir /data/training_bundle_v0_1 \
  --run-dir /runs/segmentation_training/e2_binary_c2_unet \
  --cloud-confirm

python -m segmentation_training train \
  --config /data/training_bundle_v0_1/configs/e3_multiclass_c2_c5_unet.yaml \
  --bundle-dir /data/training_bundle_v0_1 \
  --run-dir /runs/segmentation_training/e3_multiclass_c2_c5_unet \
  --cloud-confirm
```

Each returned run directory should contain `resolved_config.yaml`, `run_manifest.json`, metrics files, checkpoints, logs, and prediction previews.

## Return And Summarize

Download cloud run directories to:

```bash
.agent-team/artifacts/task-afc7f2c25f8f/cloud_runs
```

Then summarize locally:

```bash
PYTHONPATH=src python -m segmentation_training summarize-runs \
  --runs-dir .agent-team/artifacts/task-afc7f2c25f8f/cloud_runs \
  --output .agent-team/artifacts/task-afc7f2c25f8f/implementation/run_comparison_summary.md
```

Route selection must use validation evidence only. Do not use the sealed test split to choose between U-Net, ResUNet, DeepLabV3+, SegFormer, or foundation-model routes.

