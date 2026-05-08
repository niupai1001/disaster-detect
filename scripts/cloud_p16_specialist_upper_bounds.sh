#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_ROOT:-/root/autodl-tmp/New_project}"
export PYTHONPATH=src

MODE="${1:-smoke}"
C2_BUNDLE_DIR="${C2_BUNDLE_DIR:-workspace/bundles/training_bundle_c2_landslide_v0_1}"
C5_BUNDLE_DIR="${C5_BUNDLE_DIR:-workspace/bundles/p12_s2_channel_mapping_19ch}"
RUN_ROOT="${RUN_ROOT:-workspace/runs/segmentation/p16_specialist_upper_bounds/${MODE}}"

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

mkdir -p "$RUN_ROOT"

if [[ "$MODE" == "smoke" ]]; then
  C2_CONFIGS=(
    configs/segmentation_training/p16_specialist_upper_bounds/c2_landslide_9ch_unet_smoke.yaml
    configs/segmentation_training/p16_specialist_upper_bounds/c2_landslide_9ch_deeplabv3plus_smoke.yaml
  )
  C5_CONFIGS=(
    configs/segmentation_training/p16_specialist_upper_bounds/c5_water_burn_15ch_deeplabv3plus_smoke.yaml
  )
elif [[ "$MODE" == "full" ]]; then
  C2_CONFIGS=(
    configs/segmentation_training/p16_specialist_upper_bounds/c2_landslide_9ch_unet_80ep.yaml
    configs/segmentation_training/p16_specialist_upper_bounds/c2_landslide_9ch_deeplabv3plus_80ep.yaml
  )
  C5_CONFIGS=(
    configs/segmentation_training/p16_specialist_upper_bounds/c5_water_burn_15ch_deeplabv3plus_80ep.yaml
  )
else
  echo "Usage: $0 [smoke|full]" >&2
  exit 2
fi

if [[ "$MODE" == "smoke" ]]; then
  python -m segmentation_training probe-preflight \
    --bundle-dir "$C2_BUNDLE_DIR" \
    --output-dir "$RUN_ROOT/preflight_c2" \
    --run-root "$RUN_ROOT" \
    --config "${C2_CONFIGS[0]}" \
    --config "${C2_CONFIGS[1]}"

  python -m segmentation_training probe-preflight \
    --bundle-dir "$C5_BUNDLE_DIR" \
    --output-dir "$RUN_ROOT/preflight_c5" \
    --run-root "$RUN_ROOT" \
    --config "${C5_CONFIGS[0]}"
fi

for config in "${C2_CONFIGS[@]}"; do
  run_name="$(basename "$config" .yaml)"
  echo "Starting P16 C2 specialist run: $run_name"
  python -m segmentation_training train \
    --config "$config" \
    --bundle-dir "$C2_BUNDLE_DIR" \
    --run-dir "$RUN_ROOT/$run_name" \
    --cloud-confirm
done

for config in "${C5_CONFIGS[@]}"; do
  run_name="$(basename "$config" .yaml)"
  echo "Starting P16 C5 specialist run: $run_name"
  python -m segmentation_training train \
    --config "$config" \
    --bundle-dir "$C5_BUNDLE_DIR" \
    --run-dir "$RUN_ROOT/$run_name" \
    --cloud-confirm
done

python -m segmentation_training summarize-runs \
  --runs-dir "$RUN_ROOT" \
  --output "$RUN_ROOT/summary.md"

echo "P16 specialist upper-bound probe complete: $RUN_ROOT"
