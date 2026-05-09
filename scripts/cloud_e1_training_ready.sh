#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_ROOT:-/root/autodl-tmp/New_project}"
export PYTHONPATH=src

MODE="${1:-smoke}"
RUN_ROOT="${RUN_ROOT:-workspace/runs/segmentation/e1_training_ready/${MODE}}"

C2_BUNDLE_DIR="${C2_BUNDLE_DIR:-.agent-team/artifacts/task-2ba5416843b1/implementation/training_bundle_c2_landslide_v0_1}"
C5_BASE_BUNDLE_DIR="${C5_BASE_BUNDLE_DIR:-.agent-team/artifacts/task-2ba5416843b1/implementation/training_bundle_v0_3_indices}"
C5_MULTI_BUNDLE_DIR="${C5_MULTI_BUNDLE_DIR:-.agent-team/artifacts/task-2ba5416843b1/implementation/training_bundle_c5_multiscene_top3_v0_1}"
C5_CLOUD_DATA_ROOT="${C5_CLOUD_DATA_ROOT:-/data/training_bundle_v0_3_indices}"

C2_CHANNELS="B,G,R,NIR,ELEVATION,SLOPE,ASPECT,CURVATURE,TPI"
C5_CHANNELS="F01,F02,F03,F04,F07,F11,F12"

if [[ "$MODE" == "smoke" ]]; then
  C2_CONFIG="configs/segmentation_training/p16_specialist_upper_bounds/c2_landslide_9ch_deeplabv3plus_smoke.yaml"
  C5_CONFIG="configs/segmentation_training/e1_c5_multiscene_top3/c5_multiscene_top3_7ch_deeplabv3plus_smoke.yaml"
elif [[ "$MODE" == "full" ]]; then
  C2_CONFIG="configs/segmentation_training/p16_specialist_upper_bounds/c2_landslide_9ch_deeplabv3plus_80ep.yaml"
  C5_CONFIG="configs/segmentation_training/e1_c5_multiscene_top3/c5_multiscene_top3_7ch_deeplabv3plus_80ep.yaml"
else
  echo "Usage: $0 [smoke|full]" >&2
  exit 2
fi

require_manifest() {
  local bundle_dir="$1"
  local label="$2"
  if [[ ! -f "$bundle_dir/manifests/cloud_model_input_manifest.csv" ]]; then
    echo "$label manifest not found: $bundle_dir/manifests/cloud_model_input_manifest.csv" >&2
    exit 2
  fi
}

build_c5_multiscene_if_needed() {
  if [[ -f "$C5_MULTI_BUNDLE_DIR/manifests/cloud_model_input_manifest.csv" ]]; then
    return
  fi
  require_manifest "$C5_BASE_BUNDLE_DIR" "C5 base bundle"
  mkdir -p "$C5_MULTI_BUNDLE_DIR/manifests" "$C5_MULTI_BUNDLE_DIR/metadata"
  cp "$C5_BASE_BUNDLE_DIR/metadata/class_map.json" "$C5_MULTI_BUNDLE_DIR/metadata/class_map.json" 2>/dev/null || true
  cp "$C5_BASE_BUNDLE_DIR/metadata/channels.json" "$C5_MULTI_BUNDLE_DIR/metadata/channels.json" 2>/dev/null || true
  python -m segmentation_training c5-multiscene-manifest \
    --manifest "$C5_BASE_BUNDLE_DIR/manifests/cloud_model_input_manifest.csv" \
    --output "$C5_MULTI_BUNDLE_DIR/manifests/cloud_model_input_manifest.csv" \
    --required-channel F01 \
    --required-channel F02 \
    --required-channel F03 \
    --required-channel F04 \
    --required-channel F07 \
    --required-channel F11 \
    --required-channel F12 \
    --top-k 3 \
    --max-days-after-event 30 \
    --cloud-data-root "$C5_CLOUD_DATA_ROOT"
}

mkdir -p "$RUN_ROOT"

python -m segmentation_training validate-research-route \
  --route configs/research_routes/c2_c5_disaster_aware_foundation.yaml

build_c5_multiscene_if_needed
require_manifest "$C2_BUNDLE_DIR" "C2 bundle"
require_manifest "$C5_MULTI_BUNDLE_DIR" "C5 multiscene bundle"

python -m segmentation_training research-contract-audit \
  --bundle-source "c2|$C2_BUNDLE_DIR|$C2_CHANNELS" \
  --bundle-source "c5_multiscene_top3|$C5_MULTI_BUNDLE_DIR|$C5_CHANNELS" \
  --output-dir "$RUN_ROOT/contract_audit"

if [[ "$MODE" == "smoke" ]]; then
  python -m segmentation_training model-preflight \
    --output-dir "$RUN_ROOT/model_preflight" \
    --batch-size 1 \
    --height 256 \
    --width 256 \
    --config "$C2_CONFIG" \
    --config "$C5_CONFIG"

  python -m segmentation_training probe-preflight \
    --bundle-dir "$C2_BUNDLE_DIR" \
    --output-dir "$RUN_ROOT/c2_probe_preflight" \
    --run-root "$RUN_ROOT" \
    --config "$C2_CONFIG"

  python -m segmentation_training probe-preflight \
    --bundle-dir "$C5_MULTI_BUNDLE_DIR" \
    --output-dir "$RUN_ROOT/c5_multiscene_probe_preflight" \
    --run-root "$RUN_ROOT" \
    --config "$C5_CONFIG"
fi

C2_RUN_NAME="$(basename "$C2_CONFIG" .yaml)"
C5_RUN_NAME="$(basename "$C5_CONFIG" .yaml)"

echo "Starting E1 C2 run: $C2_RUN_NAME"
python -m segmentation_training train \
  --config "$C2_CONFIG" \
  --bundle-dir "$C2_BUNDLE_DIR" \
  --run-dir "$RUN_ROOT/$C2_RUN_NAME" \
  --cloud-confirm

echo "Starting E1 C5 multiscene run: $C5_RUN_NAME"
python -m segmentation_training train \
  --config "$C5_CONFIG" \
  --bundle-dir "$C5_MULTI_BUNDLE_DIR" \
  --run-dir "$RUN_ROOT/$C5_RUN_NAME" \
  --cloud-confirm

if [[ "$MODE" == "full" && -f "$RUN_ROOT/$C2_RUN_NAME/checkpoints/best_mean_iou.pt" ]]; then
  python -m segmentation_training channel-contribution \
    --config "$C2_CONFIG" \
    --bundle-dir "$C2_BUNDLE_DIR" \
    --checkpoint "$RUN_ROOT/$C2_RUN_NAME/checkpoints/best_mean_iou.pt" \
    --output-dir "$RUN_ROOT/$C2_RUN_NAME/diagnostics/channel_contribution" \
    --cloud-confirm
fi

python -m segmentation_training summarize-runs \
  --runs-dir "$RUN_ROOT" \
  --output "$RUN_ROOT/summary.md" \
  --review-dir "$RUN_ROOT/review"

echo "E1 training-ready suite complete: $RUN_ROOT"
