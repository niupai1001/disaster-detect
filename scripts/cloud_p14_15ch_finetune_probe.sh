#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_ROOT:-/root/autodl-tmp/New_project}"
export PYTHONPATH=src

MODE="${1:-smoke}"
BUNDLE_DIR="${BUNDLE_DIR:-workspace/bundles/p12_s2_channel_mapping_19ch}"
RUN_ROOT="${RUN_ROOT:-workspace/runs/segmentation/p14_15ch_finetune_probe/${MODE}}"
export P13_BEST_CHECKPOINT="${P13_BEST_CHECKPOINT:-workspace/runs/segmentation/p13_15ch_capacity_probe/full/c5_deeplabv3plus_15ch_base32_120ep/checkpoints/best_mean_iou.pt}"

if [[ ! -f "$P13_BEST_CHECKPOINT" ]]; then
  echo "P13 best checkpoint not found: $P13_BEST_CHECKPOINT" >&2
  echo "Set P13_BEST_CHECKPOINT=/path/to/best_mean_iou.pt and rerun." >&2
  exit 2
fi

if [[ ! -f "$BUNDLE_DIR/manifests/cloud_model_input_manifest.csv" ]]; then
  echo "Bundle manifest not found: $BUNDLE_DIR/manifests/cloud_model_input_manifest.csv" >&2
  echo "Set BUNDLE_DIR=/path/to/the P12 15/19ch bundle and rerun." >&2
  exit 2
fi

mkdir -p "$RUN_ROOT"

if [[ "$MODE" == "smoke" ]]; then
  CONFIGS=(
    configs/segmentation_training/p14_15ch_finetune/c5_deeplabv3plus_15ch_base32_ft_lr1e4_smoke.yaml
    configs/segmentation_training/p14_15ch_finetune/c5_deeplabv3plus_15ch_base32_ft_lr5e5_smoke.yaml
  )
elif [[ "$MODE" == "full" ]]; then
  CONFIGS=(
    configs/segmentation_training/p14_15ch_finetune/c5_deeplabv3plus_15ch_base32_ft_lr1e4_80ep.yaml
    configs/segmentation_training/p14_15ch_finetune/c5_deeplabv3plus_15ch_base32_ft_lr5e5_80ep.yaml
  )
else
  echo "Usage: $0 [smoke|full]" >&2
  exit 2
fi

if [[ "$MODE" == "smoke" ]]; then
  python -m segmentation_training probe-preflight \
    --bundle-dir "$BUNDLE_DIR" \
    --output-dir "$RUN_ROOT/preflight" \
    --run-root "$RUN_ROOT" \
    --config "${CONFIGS[0]}" \
    --config "${CONFIGS[1]}"
fi

for config in "${CONFIGS[@]}"; do
  run_name="$(basename "$config" .yaml)"
  echo "Starting P14 run: $run_name"
  python -m segmentation_training train \
    --config "$config" \
    --bundle-dir "$BUNDLE_DIR" \
    --run-dir "$RUN_ROOT/$run_name" \
    --cloud-confirm
done

python -m segmentation_training summarize-runs \
  --runs-dir "$RUN_ROOT" \
  --output "$RUN_ROOT/summary.md"

echo "P14 15ch fine-tune probe complete: $RUN_ROOT"
