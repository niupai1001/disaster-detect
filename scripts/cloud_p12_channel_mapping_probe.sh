#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_ROOT:-/root/autodl-tmp/New_project}"
export PYTHONPATH=src

MODE="${1:-smoke}"
CONTRACT_DIR="${CONTRACT_DIR:-workspace/contracts/p12_s2_channel_mapping_post_only}"
BUNDLE_DIR="${BUNDLE_DIR:-workspace/bundles/p12_s2_channel_mapping_19ch}"
RUN_ROOT="${RUN_ROOT:-workspace/runs/segmentation/p12_s2_channel_mapping_probe/${MODE}}"
CLOUD_DATA_ROOT="${CLOUD_DATA_ROOT:-$(pwd)}"

RAW_CHANNELS=(F01 F02 F03 F04 F05 F06 F07 F08 F11 F12)
DERIVED_CHANNELS=(NDVI NBR NDMI NDWI MNDWI NBR2 MIRBI BAIS2 BRIGHTNESS)
ALL_CHANNELS=("${RAW_CHANNELS[@]}" "${DERIVED_CHANNELS[@]}")

CONTRACT_ARGS=()
for channel in "${RAW_CHANNELS[@]}"; do
  CONTRACT_ARGS+=(--required-channel "$channel")
done
for index in "${DERIVED_CHANNELS[@]}"; do
  CONTRACT_ARGS+=(--derived-index "$index")
done

BUNDLE_ARGS=()
for channel in "${ALL_CHANNELS[@]}"; do
  BUNDLE_ARGS+=(--required-channel "$channel")
done

python -m segmentation_contract \
  --project-root . \
  --output-dir "$CONTRACT_DIR" \
  --phase-b-build \
  --visual-qa \
  --visual-qa-max-items 24 \
  --post-disaster-audit \
  --post-selection-strategy quality_then_earliest \
  --model-input-previews \
  --model-input-preview-max-items 24 \
  "${CONTRACT_ARGS[@]}"

python -m segmentation_training bundle \
  --contract-dir "$CONTRACT_DIR" \
  --bundle-dir "$BUNDLE_DIR" \
  --bundle-version p12_s2_channel_mapping_19ch \
  --mode manifest-only \
  --class-scope binary_c5 \
  --cloud-data-root "$CLOUD_DATA_ROOT" \
  "${BUNDLE_ARGS[@]}"

mkdir -p "$RUN_ROOT"

if [[ "$MODE" == "smoke" ]]; then
  CONFIGS=(
    configs/segmentation_training/channel_mapping_probe/c5_deeplabv3plus_s2_corrected_11ch_smoke.yaml
    configs/segmentation_training/channel_mapping_probe/c5_deeplabv3plus_s2_water_burn_15ch_smoke.yaml
    configs/segmentation_training/channel_mapping_probe/c5_deeplabv3plus_s2_rededge_bais2_19ch_smoke.yaml
  )
elif [[ "$MODE" == "full" ]]; then
  CONFIGS=(
    configs/segmentation_training/channel_mapping_probe/c5_deeplabv3plus_s2_corrected_11ch_rtx4080.yaml
    configs/segmentation_training/channel_mapping_probe/c5_deeplabv3plus_s2_water_burn_15ch_rtx4080.yaml
    configs/segmentation_training/channel_mapping_probe/c5_deeplabv3plus_s2_rededge_bais2_19ch_rtx4080.yaml
  )
else
  echo "Usage: $0 [smoke|full]" >&2
  exit 2
fi

for config in "${CONFIGS[@]}"; do
  run_name="$(basename "$config" .yaml)"
  python -m segmentation_training train \
    --config "$config" \
    --bundle-dir "$BUNDLE_DIR" \
    --run-dir "$RUN_ROOT/$run_name" \
    --cloud-confirm
done

python -m segmentation_training summarize-runs \
  --runs-dir "$RUN_ROOT" \
  --output "$RUN_ROOT/summary.md"

echo "P12 channel mapping probe complete: $RUN_ROOT"
