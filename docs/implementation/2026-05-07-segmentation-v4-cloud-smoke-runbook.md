# SegmentationModelV4 Cloud Smoke Runbook

Date: 2026-05-07

This runbook prepares controlled cloud smoke/preflight runs for the 11-channel architecture benchmark runway. It is not a full benchmark and must not be used to select a winning model.

## Boundary

Allowed:

- Use train and validation splits only.
- Run one short smoke pass per architecture.
- Verify model initialization, first train/validation behavior, memory envelope, and required artifact generation.
- Collect validation-only metrics and contact sheets for QA/review.

Blocked:

- Do not use the sealed test split.
- Do not run full 80-epoch architecture benchmark training from this smoke step.
- Do not claim model quality or choose a winner from smoke results.
- Do not compare smoke results as a pure architecture benchmark if any architecture needs a different batch size, loss, sampler, learning rate, or channel set.

## Smoke Configs

Use the configs under:

```text
configs/segmentation_training/architecture_smoke/
```

These configs intentionally cap compute:

```text
max_epochs: 1
batch_size: 2
cloud.max_train_samples: 8
cloud.max_validation_samples: 4
training.performance.cache_records: false
```

They preserve the 11-channel input stack, class scope, validation-only selection, threshold sweep outputs, best checkpoint outputs, AMP, channels-last, and sealed-test discipline.

## Preflight Dry Run

Run this once per smoke config against the cloud bundle before training:

```bash
for config in configs/segmentation_training/architecture_smoke/*.yaml; do
  PYTHONPATH=src python -m segmentation_training dry-run \
    --config "$config" \
    --bundle-dir .agent-team/artifacts/task-2ba5416843b1/implementation/training_bundle_v0_3_indices \
    --max-samples 4
done
```

The dry-run result must report:

```text
test_split_read: false
status: dry_run_passed
```

## Cloud Smoke Commands

On the cloud machine with the 11-channel bundle mounted, run:

```bash
export PYTHONPATH=src
export BUNDLE_DIR=.agent-team/artifacts/task-2ba5416843b1/implementation/training_bundle_v0_3_indices
export RUN_ROOT=.agent-team/artifacts/task-6bd957f6e1ed/cloud_smoke

for config in configs/segmentation_training/architecture_smoke/*.yaml; do
  name="$(basename "$config" .yaml)"
  python -m segmentation_training train \
    --config "$config" \
    --bundle-dir "$BUNDLE_DIR" \
    --run-dir "$RUN_ROOT/$name" \
    --cloud-confirm
done
```

## Required Evidence Per Run

Each run directory must contain:

- `run_manifest.json`
- `metrics.json`
- `per_class_metrics.csv`
- `diagnostics/threshold_sweep.csv`
- `diagnostics/validation_prediction_summary.csv`
- `diagnostics/foreground_window_coverage.csv`
- `predictions/validation_contact_sheet.png`
- `training_curves.csv`
- `training_curves.png`
- `checkpoints/best_mean_iou.pt`
- `checkpoints/best_mean_iou.json`

Inspect `run_manifest.json` and confirm:

```text
test_split_read: false
cloud.max_train_samples: 8
cloud.max_validation_samples: 4
```

## Smoke Summary

After runs finish, summarize:

```bash
PYTHONPATH=src python -m segmentation_training summarize-runs \
  --runs-dir .agent-team/artifacts/task-6bd957f6e1ed/cloud_smoke \
  --output .agent-team/artifacts/task-6bd957f6e1ed/cloud_smoke/summary.md
```

This also creates the phase-2 review surface:

```text
.agent-team/artifacts/task-6bd957f6e1ed/cloud_smoke/review/
  review.md
  compact_metrics.csv
  raw_evidence.md
  figures/
```

## Next Gate

After cloud smoke, return to QA/review with:

- smoke summary;
- `review/review.md` and `review/compact_metrics.csv`;
- memory/runtime notes;
- one validation contact sheet per architecture;
- any failures or required composite-treatment deviations.

Full benchmark training remains blocked until QA/review accepts this smoke evidence.
