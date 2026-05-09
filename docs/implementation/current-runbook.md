---
title: Current Runbook
date: 2026-05-08
tags:
  - segmentation
  - runbook
  - foundation-model
  - c2-c5
status: active
---

# Current Runbook

This runbook is the current source of truth for the next project work. The active route is **C2+C5 disaster-aware foundation segmentation**, with Prithvi as the main backbone and TerraMind as a later extension.

> [!warning]
> Do not use the sealed test split. All route selection remains validation-only until the final paper route is locked.

## Route Preflight

Validate the route contract before starting new training:

```bash
PYTHONPATH=src python -m segmentation_training validate-research-route \
  --route configs/research_routes/c2_c5_disaster_aware_foundation.yaml
```

Expected summary:

```text
disaster_scope: C2, C5
primary_backbone: Prithvi
extension_backbone: TerraMind
stages: E0..E7
sealed_test_policy: sealed_until_final
```

## E0 Data And Label Contract

Every C2/C5 route manifest used for the paper path must carry these research fields:

```text
disaster_id,label_confidence,ignore_mask_path
```

Rules:

- `disaster_id` must match class id: `1 -> C2`, `2 -> C5`.
- `label_confidence` must be `high`, `medium`, `low`, `unknown`, or a numeric score in `[0, 1]`.
- `ignore_mask_path` may be empty, but if present it must resolve inside the contract/bundle.
- The same `sample_id` may not appear in more than one split.
- C2 and C5 must be reported separately; macro metrics are secondary.

The Python validation hook is `segmentation_training.manifest.validate_research_contract`. Use it in dataset builders or preflight scripts when the upgraded manifest is produced.

The project-level audit command is:

```bash
PYTHONPATH=src python -m segmentation_training research-contract-audit \
  --bundle-source 'c2|.agent-team/artifacts/task-2ba5416843b1/implementation/training_bundle_c2_landslide_v0_1|B,G,R,NIR,ELEVATION,SLOPE,ASPECT,CURVATURE,TPI' \
  --bundle-source 'c5|.agent-team/artifacts/task-2ba5416843b1/implementation/training_bundle_v0_3_indices|F01,F02,F03,F04,F07,F11,F12,NDVI,NBR,NDMI,BRIGHTNESS' \
  --output-dir .agent-team/artifacts/task-2ba5416843b1/implementation/e0_c2_c5_research_contract_audit
```

Current E0 result after enriching the existing C2/C5 bundle manifests:

```text
status: ready_for_e1
records: 743
train: C2 291 / C5 301
validation: C2 73 / C5 78
missing disaster_id: 0
missing label_confidence: 0
missing ignore_mask_path column: 0
contract errors: 0
label_confidence: all unknown
ignore masks declared: 0
```

Audit package:

```text
.agent-team/artifacts/task-2ba5416843b1/implementation/e0_c2_c5_research_contract_audit/
```

E0 now permits E1 baseline design. Treat `label_confidence=unknown` as a conservative default, not as manual label review. E4 will still need a real relabel/ignore-mask ablation.

## E1 Baselines

Build only evidence-bearing baselines:

| Baseline | Purpose | Required reporting |
|---|---|---|
| Classical indices / OBIA | Sanity check disaster physics and label separability. | C2/C5/macro metrics, hard cases. |
| DeepLabV3+ / Transformer-UNet | Strong non-foundation reference. | Per-disaster IoU/Dice/F1, precision/recall, area ratio. |
| Specialist upper bounds | Test whether each disaster is learnable alone. | C2-only and C5-only validation metrics. |
| Category-unknown lower bound | Test realistic inference without manual disaster class. | Category accuracy plus routed mask score. |

Existing P16/P18 evidence may be reused as side evidence, but it is not the whole route.

### E1 Training-Ready Entry

C5 should no longer be limited to a single selected scene when multiple good post-event scenes exist. The current safe expansion is **top-3 same-day/post-event C5 scenes within 30 days**, using the same parent split for every expanded scene. This adds data without split leakage because different scenes from the same parent sample never cross train/validation/test.

Current C5 multiscene manifest:

```text
.agent-team/artifacts/task-2ba5416843b1/implementation/training_bundle_c5_multiscene_top3_v0_1/manifests/cloud_model_input_manifest.csv
```

Current E1 training-ready audit:

```text
.agent-team/artifacts/task-2ba5416843b1/implementation/e1_training_ready_audit/
status: ready_for_e1
records: 1163
train: C2 291 / C5 629
validation: C2 73 / C5 170
contract errors: 0
parent split leakage: 0
```

Cloud smoke command:

```bash
cd /root/autodl-tmp/New_project
git pull --ff-only
PROJECT_ROOT=/root/autodl-tmp/New_project \
RUN_ROOT=workspace/runs/segmentation/e1_training_ready/smoke \
scripts/cloud_e1_training_ready.sh smoke
```

Cloud full command after smoke passes:

```bash
cd /root/autodl-tmp/New_project
git pull --ff-only
PROJECT_ROOT=/root/autodl-tmp/New_project \
RUN_ROOT=workspace/runs/segmentation/e1_training_ready/full \
scripts/cloud_e1_training_ready.sh full
```

The script validates the route, rebuilds the lightweight C5 multiscene manifest if missing, audits C2+C5 manifests, runs model/probe preflights for smoke mode, then launches C2 terrain specialist and C5 multiscene DeepLabV3+ training. The full mode also writes a C2 terrain-vs-optical channel contribution diagnostic when the checkpoint is available.

## E2-E3 Prithvi Main Method

1. Audit local channel compatibility against Prithvi-compatible optical/HLS inputs.
2. Run a vanilla Prithvi optical-only fine-tune if weights, interface, and compute are available.
3. Add disaster-aware components:
   - category prediction branch for C2/C5;
   - disaster embedding or conditional segmentation head;
   - end-to-end category-unknown evaluation.
4. Promote only if C2 improves against the strongest fair baseline.

If Prithvi is blocked, document the specific incompatibility and keep strong deep models as baselines, not as the final foundation claim.

## E4-E5 Label And Terrain Ablations

C2 is the priority failure route. The next ablations must isolate:

- label-confidence weighting or filtering;
- ignore-mask policy for uncertain boundaries/cloud/shadow/bare-soil confusion;
- terrain adapter with `ELEVATION,SLOPE,ASPECT,CURVATURE,TPI`;
- C2 false positives on bare soil, shadows, steep terrain, and high-contrast mountain windows.

## E6-E7 Final Comparisons

E6 must report category-known versus category-unknown inference. E7 only starts after the Prithvi route is coherent; TerraMind is promoted only if weights/interface/compute are feasible and it gives a concrete C2 or multimodal benefit.

## Legacy Evidence To Keep

- P16 C2 specialist runs show C2 can learn foreground but remains weak.
- P18 C5 strong architecture runs are useful C5 reference evidence.
- V6 11-channel C5 diagnostics showed overprediction and calibration issues.
- P15 common4 weakened both classes and should not drive the main route.

Do not start another cloud training run unless it maps to E0-E7 and names the exact evidence it will produce.
