---
title: Context Compression And Next Actions
date: 2026-05-07
tags:
  - handoff
  - next-actions
  - segmentation
  - cloud
  - obsidian
status: active
task: task-6bd957f6e1ed
aliases:
  - Current Handoff
---

# Context Compression And Next Actions

## Current Branch

```text
codex/rtx4080-training-performance
```

Latest pushed commit:

```text
3512bf3 Add project structure governance layer
```

## Non-Negotiable Project Rules

- Do not use external providers for department artifacts.
- Codex dispatches and coordinates its own subagents for department work.
- Keep sealed test split sealed; architecture and channel selection use validation only.
- Do not treat smoke metrics as model-quality evidence.
- Do not move legacy `.agent-team/artifacts/...` paths until current cloud runbooks stop depending on them.

## Completed Work

- Architecture benchmark runway added for ResUNet, Attention U-Net, UNet++, DeepLabV3+, and Transformer-UNet.
- Architecture cloud smoke returned and passed runtime preflight.
- U-Net-only channel ablation configs added for RGB, false-color, 7-channel raw optical/SWIR, indices-only, and current 11-channel control.
- Obsidian-compatible model/data hypothesis note created.
- Project structure governance layer added with `docs/ops/`, `scripts/project_inventory.py`, and `workspace/` ignore policy.
- Full local tests passed after governance changes: `Ran 111 tests ... OK`.

## Immediate Cloud Actions

On the cloud machine:

```bash
cd /root/autodl-tmp/New_project
git pull --ff-only origin codex/rtx4080-training-performance
```

### Option A: Architecture Validation Benchmark

Run the managed validation-only architecture benchmark under:

```text
.agent-team/artifacts/task-6bd957f6e1ed/cloud_benchmark
```

Use configs listed in [[docs/implementation/2026-05-07-segmentation-v4-cloud-smoke-runbook|Architecture Smoke Runbook]] and [[docs/ops/review-index|Review Index]].

### Option B: U-Net Channel Ablation Smoke

Run smoke first:

```bash
export PYTHONPATH=src
export BUNDLE_DIR=.agent-team/artifacts/task-2ba5416843b1/implementation/training_bundle_v0_3_indices
export RUN_ROOT=.agent-team/artifacts/task-6bd957f6e1ed/cloud_unet_channel_ablation_smoke

for config in configs/segmentation_training/unet_channel_ablation/*_smoke.yaml; do
  name="$(basename "$config" .yaml)"
  python -m segmentation_training train \
    --config "$config" \
    --bundle-dir "$BUNDLE_DIR" \
    --run-dir "$RUN_ROOT/$name" \
    --cloud-confirm
done

python -m segmentation_training summarize-runs \
  --runs-dir "$RUN_ROOT" \
  --output "$RUN_ROOT/summary.md"
```

## Evidence To Return

For each cloud run family, return or paste:

- `summary.md`;
- each run's `run_manifest.json`;
- `metrics.json`;
- `per_class_metrics.csv`;
- validation contact sheets;
- training curves;
- any OOM or failure logs.

## After Results Return

1. QA/review accepts or rejects returned evidence.
2. If architecture benchmark succeeds, compare validation-only evidence without using sealed test.
3. If channel ablation shows simpler stacks beat 11-channel, investigate channel noise, normalization, and label/sample quality before escalating architecture complexity.
4. If all U-Net channel groups fail similarly, prioritize labels, alignment, foreground sparsity, sampler coverage, loss weighting, and thresholding.
5. Start phase 2 structure cleanup: migrate future run outputs toward `workspace/runs/`, future bundles toward `workspace/bundles/`, and leave only Markdown summaries in `.agent-team/artifacts`.

## Useful Local Commands

Inventory:

```bash
PYTHONPATH=src python scripts/project_inventory.py --root . --max-depth 3
```

Full tests:

```bash
PYTHONPATH=src /Users/Lemon/miniconda3/envs/d2l-zh/bin/python -m unittest discover -s tests
```

