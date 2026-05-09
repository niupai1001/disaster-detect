---
title: Next Actions
date: 2026-05-08
tags:
  - ops
  - handoff
  - next-actions
status: active
---

# Next Actions

## Current Project State

The user rejected the previous plan because it still felt like a small target inside a small target. The active project route is now:

```text
C2+C5 disaster-aware foundation segmentation
paper/experiment priority
Prithvi main line + TerraMind extension
```

The route is encoded in:

```text
configs/research_routes/c2_c5_disaster_aware_foundation.yaml
```

and the current implementation runbook is:

```text
docs/implementation/current-runbook.md
```

## Immediate Next Steps

1. Validate the route contract:

   ```bash
   PYTHONPATH=src python -m segmentation_training validate-research-route \
     --route configs/research_routes/c2_c5_disaster_aware_foundation.yaml
   ```

2. Review the first E0 audit package:

   ```text
   .agent-team/artifacts/task-2ba5416843b1/implementation/e0_c2_c5_research_contract_audit/
   ```

   Current result after manifest enrichment:

   ```text
   status: ready_for_e1
   total_records: 743
   train: C2 291 / C5 301
   validation: C2 73 / C5 78
   missing disaster_id: 0
   missing label_confidence: 0
   missing ignore_mask_path column: 0
   contract_error_count: 0
   label_confidence: all unknown
   ```

3. Start the E1 training-ready suite on the cloud machine:

   ```bash
   cd /root/autodl-tmp/New_project
   git pull --ff-only
   PROJECT_ROOT=/root/autodl-tmp/New_project \
   RUN_ROOT=workspace/runs/segmentation/e1_training_ready/smoke \
   scripts/cloud_e1_training_ready.sh smoke
   ```

   If smoke passes, launch full C2/C5 baseline training:

   ```bash
   cd /root/autodl-tmp/New_project
   git pull --ff-only
   PROJECT_ROOT=/root/autodl-tmp/New_project \
   RUN_ROOT=workspace/runs/segmentation/e1_training_ready/full \
   scripts/cloud_e1_training_ready.sh full
   ```

   The suite trains the C2 terrain-aware specialist baseline and the C5 top-3 multiscene baseline under the research contract.

4. Keep the research-contract validation hook in all future manifest-build scripts:

   ```python
   from segmentation_training.manifest import load_model_input_manifest, validate_research_contract
   ```

5. Rerun the contract audit before any new cloud training or bundle regeneration:
   - C2/C5 row counts by split;
   - missing or mismatched `disaster_id`;
   - missing/bad `label_confidence`;
   - missing `ignore_mask_path` files where declared;
   - duplicate `sample_id` across splits;
   - sealed-test guard evidence.

6. Do not interpret `label_confidence=unknown` as manual review. E4 remains responsible for relabel/ignore-mask ablation.

## Current Data Expansion Decision

C5 now uses a leakage-safe top-3 multiscene manifest:

```text
.agent-team/artifacts/task-2ba5416843b1/implementation/training_bundle_c5_multiscene_top3_v0_1/
```

Expansion result:

```text
source C5 parent records: 379
expanded C5 rows: 799
rank1: 379
rank2: 256
rank3: 164
parent split leakage: 0
```

Combined C2 + C5 multiscene audit:

```text
status: ready_for_e1
total_records: 1163
train: C2 291 / C5 629
validation: C2 73 / C5 170
contract_error_count: 0
```

## Do Not Do Yet

- Do not use the sealed test split.
- Do not resume P-number experiments unless they map to E0-E7 in the research route.
- Do not let C5 results hide C2 failure.
- Do not start Prithvi/TerraMind training before channel/interface/compute compatibility is audited.
- Do not treat strong DeepLabV3+ or Transformer-UNet as the final method; they are baselines unless Prithvi is blocked and the route is revised.
- Do not create new date-stamped project Markdown; update living notes and mirror substantial handoffs into Obsidian.
