#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_ROOT:-/root/autodl-tmp/New_project}"
export PYTHONPATH=src

MODE="${1:-smoke}"
OLD_CONTRACT_DIR="${OLD_CONTRACT_DIR:-workspace/contracts/p16b_old_c2c5_f16f17}"
OLD_BUNDLE_DIR="${OLD_BUNDLE_DIR:-workspace/bundles/p16b_old_c2c5_f16f17}"
RUN_ROOT="${RUN_ROOT:-workspace/runs/segmentation/p16b_old_c2_recovery/${MODE}}"
REBUILD_OLD_BUNDLE="${REBUILD_OLD_BUNDLE:-auto}"

link_database_into_bundle() {
  local link_path="$OLD_BUNDLE_DIR/database"
  if [[ -e "$link_path" && ! -L "$link_path" ]]; then
    echo "$link_path exists and is not a symlink. Move it aside or set OLD_BUNDLE_DIR to a clean path." >&2
    exit 2
  fi
  ln -sfn "$(pwd)/database" "$link_path"
}

if [[ "$MODE" == "smoke" ]]; then
  CONFIGS=(
    configs/segmentation_training/p16b_old_c2_recovery/old_c2_f16f17_unet_smoke.yaml
    configs/segmentation_training/p16b_old_c2_recovery/old_c2_f16f17_deeplabv3plus_smoke.yaml
  )
elif [[ "$MODE" == "full" ]]; then
  CONFIGS=(
    configs/segmentation_training/p16b_old_c2_recovery/old_c2_f16f17_unet_80ep.yaml
    configs/segmentation_training/p16b_old_c2_recovery/old_c2_f16f17_deeplabv3plus_80ep.yaml
  )
else
  echo "Usage: $0 [smoke|full]" >&2
  exit 2
fi

if [[ "$REBUILD_OLD_BUNDLE" == "1" || "$REBUILD_OLD_BUNDLE" == "true" || ! -f "$OLD_BUNDLE_DIR/manifests/cloud_model_input_manifest.csv" ]]; then
  echo "Building old F16/F17 C2/C5 contract and bundle: $OLD_BUNDLE_DIR"
  python -m segmentation_contract \
    --project-root . \
    --output-dir "$OLD_CONTRACT_DIR" \
    --phase-b-build \
    --visual-qa \
    --visual-qa-max-items 24 \
    --channel-audit \
    --required-channel F16 \
    --required-channel F17

  mkdir -p "$OLD_BUNDLE_DIR"
  link_database_into_bundle

  python -m segmentation_training bundle \
    --contract-dir "$OLD_CONTRACT_DIR" \
    --bundle-dir "$OLD_BUNDLE_DIR" \
    --bundle-version p16b_old_c2c5_f16f17 \
    --mode manifest-only \
    --class-scope multiclass_c2_c5 \
    --cloud-data-root "$(pwd)/$OLD_BUNDLE_DIR" \
    --required-channel F16 \
    --required-channel F17
else
  mkdir -p "$OLD_BUNDLE_DIR"
  link_database_into_bundle
fi

mkdir -p "$RUN_ROOT"

python -m segmentation_training validate-manifest \
  --contract-dir "$OLD_BUNDLE_DIR" \
  --required-channel F16 \
  --required-channel F17

if [[ "$MODE" == "smoke" ]]; then
  python -m segmentation_training probe-preflight \
    --bundle-dir "$OLD_BUNDLE_DIR" \
    --output-dir "$RUN_ROOT/preflight" \
    --run-root "$RUN_ROOT" \
    --config "${CONFIGS[0]}" \
    --config "${CONFIGS[1]}"
fi

for config in "${CONFIGS[@]}"; do
  run_name="$(basename "$config" .yaml)"
  echo "Starting P16b old C2 recovery run: $run_name"
  python -m segmentation_training train \
    --config "$config" \
    --bundle-dir "$OLD_BUNDLE_DIR" \
    --run-dir "$RUN_ROOT/$run_name" \
    --cloud-confirm
done

python -m segmentation_training summarize-runs \
  --runs-dir "$RUN_ROOT" \
  --output "$RUN_ROOT/summary.md" \
  --review-dir "$RUN_ROOT/review"

echo "P16b old C2 recovery complete: $RUN_ROOT"
