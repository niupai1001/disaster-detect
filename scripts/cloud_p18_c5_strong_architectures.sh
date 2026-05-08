#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_ROOT:-/root/autodl-tmp/New_project}"
export PYTHONPATH=src

MODE="${1:-smoke}"
C5_BUNDLE_DIR="${C5_BUNDLE_DIR:-workspace/bundles/p12_s2_channel_mapping_19ch}"
RUN_ROOT="${RUN_ROOT:-workspace/runs/segmentation/p18_c5_strong_architectures/${MODE}}"

if [[ ! -f "$C5_BUNDLE_DIR/manifests/cloud_model_input_manifest.csv" ]]; then
  echo "C5 bundle manifest not found: $C5_BUNDLE_DIR/manifests/cloud_model_input_manifest.csv" >&2
  echo "Set C5_BUNDLE_DIR=/path/to/p12_s2_channel_mapping_19ch bundle and rerun." >&2
  exit 2
fi

if [[ "$MODE" == "smoke" ]]; then
  CONFIGS=(
    configs/segmentation_training/p18_c5_strong_architectures/c5_water_burn_15ch_deeplabv3plus_base64_smoke.yaml
    configs/segmentation_training/p18_c5_strong_architectures/c5_water_burn_15ch_attention_unet_base64_smoke.yaml
    configs/segmentation_training/p18_c5_strong_architectures/c5_water_burn_15ch_transformer_unet_base64_smoke.yaml
  )
elif [[ "$MODE" == "full" ]]; then
  CONFIGS=(
    configs/segmentation_training/p18_c5_strong_architectures/c5_water_burn_15ch_deeplabv3plus_base64_100ep.yaml
    configs/segmentation_training/p18_c5_strong_architectures/c5_water_burn_15ch_attention_unet_base64_100ep.yaml
    configs/segmentation_training/p18_c5_strong_architectures/c5_water_burn_15ch_transformer_unet_base64_100ep.yaml
  )
else
  echo "Usage: $0 [smoke|full]" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT"

if [[ "$MODE" == "smoke" ]]; then
  python -m segmentation_training model-preflight \
    --output-dir "$RUN_ROOT/model_preflight" \
    --batch-size 1 \
    --height 256 \
    --width 256 \
    --config "${CONFIGS[0]}" \
    --config "${CONFIGS[1]}" \
    --config "${CONFIGS[2]}"

  python -m segmentation_training probe-preflight \
    --bundle-dir "$C5_BUNDLE_DIR" \
    --output-dir "$RUN_ROOT/probe_preflight" \
    --run-root "$RUN_ROOT" \
    --config "${CONFIGS[0]}" \
    --config "${CONFIGS[1]}" \
    --config "${CONFIGS[2]}"
fi

for config in "${CONFIGS[@]}"; do
  run_name="$(basename "$config" .yaml)"
  echo "Starting P18 C5 strong architecture run: $run_name"
  python -m segmentation_training train \
    --config "$config" \
    --bundle-dir "$C5_BUNDLE_DIR" \
    --run-dir "$RUN_ROOT/$run_name" \
    --cloud-confirm
done

python -m segmentation_training summarize-runs \
  --runs-dir "$RUN_ROOT" \
  --output "$RUN_ROOT/summary.md" \
  --review-dir "$RUN_ROOT/review"

echo "P18 C5 strong architecture probe complete: $RUN_ROOT"
