#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_ROOT:-/root/autodl-tmp/New_project}"
export PYTHONPATH=src

MODE="${1:-smoke}"
C2_BUNDLE_DIR="${C2_BUNDLE_DIR:-workspace/bundles/training_bundle_c2_landslide_v0_1}"
C5_BUNDLE_DIR="${C5_BUNDLE_DIR:-workspace/bundles/p12_s2_channel_mapping_19ch}"
MERGED_BUNDLE_DIR="${MERGED_BUNDLE_DIR:-workspace/bundles/p15_c2_c5_common4}"
RUN_ROOT="${RUN_ROOT:-workspace/runs/segmentation/p15_c2_c5_common4_probe/${MODE}}"

if [[ ! -f "$C2_BUNDLE_DIR/manifests/cloud_model_input_manifest.csv" ]]; then
  echo "C2 bundle manifest not found: $C2_BUNDLE_DIR/manifests/cloud_model_input_manifest.csv" >&2
  echo "Set C2_BUNDLE_DIR=/path/to/c2 bundle and rerun." >&2
  exit 2
fi

if [[ ! -f "$C5_BUNDLE_DIR/manifests/cloud_model_input_manifest.csv" ]]; then
  echo "C5 bundle manifest not found: $C5_BUNDLE_DIR/manifests/cloud_model_input_manifest.csv" >&2
  echo "Set C5_BUNDLE_DIR=/path/to/c5 bundle and rerun." >&2
  exit 2
fi

python -m segmentation_training common-channel-bundle \
  --c2-bundle-dir "$C2_BUNDLE_DIR" \
  --c5-bundle-dir "$C5_BUNDLE_DIR" \
  --bundle-dir "$MERGED_BUNDLE_DIR" \
  --bundle-version p15_c2_c5_common4

mkdir -p "$RUN_ROOT"

if [[ "$MODE" == "smoke" ]]; then
  CONFIGS=(
    configs/segmentation_training/p15_c2_c5_common4/c2_c5_common4_unet_smoke.yaml
    configs/segmentation_training/p15_c2_c5_common4/c2_c5_common4_deeplabv3plus_smoke.yaml
  )
elif [[ "$MODE" == "full" ]]; then
  CONFIGS=(
    configs/segmentation_training/p15_c2_c5_common4/c2_c5_common4_unet_80ep.yaml
    configs/segmentation_training/p15_c2_c5_common4/c2_c5_common4_deeplabv3plus_80ep.yaml
  )
else
  echo "Usage: $0 [smoke|full]" >&2
  exit 2
fi

if [[ "$MODE" == "smoke" ]]; then
  python -m segmentation_training probe-preflight \
    --bundle-dir "$MERGED_BUNDLE_DIR" \
    --output-dir "$RUN_ROOT/preflight" \
    --run-root "$RUN_ROOT" \
    --config "${CONFIGS[0]}" \
    --config "${CONFIGS[1]}"
fi

for config in "${CONFIGS[@]}"; do
  run_name="$(basename "$config" .yaml)"
  echo "Starting P15 run: $run_name"
  python -m segmentation_training train \
    --config "$config" \
    --bundle-dir "$MERGED_BUNDLE_DIR" \
    --run-dir "$RUN_ROOT/$run_name" \
    --cloud-confirm
done

python -m segmentation_training summarize-runs \
  --runs-dir "$RUN_ROOT" \
  --output "$RUN_ROOT/summary.md"

echo "P15 C2+C5 common4 probe complete: $RUN_ROOT"
