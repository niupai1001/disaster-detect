---
title: Project Notes
date: 2026-05-07
tags:
  - project
  - index
  - segmentation
status: active
---

# Project Notes

This directory is intentionally small. It keeps only living notes that are useful for review, handoff, and the next experiment decision.

## Current Project Summary

As of 2026-05-13, the project route is:

```text
C2+C5 disaster-aware foundation segmentation
paper/experiment priority
Prithvi main line + TerraMind extension
```

The project is currently in the transition from **E0 data contract validation** to **E1 baseline training**. The immediate objective is to run a training-ready baseline suite that produces reliable C2/C5 evidence before starting Prithvi fine-tuning.

### Scope

- C2: landslide / debris-flow segmentation, currently the main failure and improvement target.
- C5: fire / burned-area segmentation, useful reference route but not allowed to hide C2 weakness.
- Reporting must always separate C2, C5, and macro metrics.
- Sealed test remains locked; all route selection is validation-only.

### Implemented Work

- Locked the research route in `configs/research_routes/c2_c5_disaster_aware_foundation.yaml`.
- Added research manifest contract fields: `disaster_id`, `label_confidence`, `ignore_mask_path`.
- Added route validation and research-contract audit CLI paths.
- Enriched existing C2/C5 manifests to the research contract.
- Added leakage-safe C5 top-3 multiscene expansion.
- Added E1 training-ready cloud entry script: `scripts/cloud_e1_training_ready.sh`.
- Cleaned project Markdown down to living notes and active evidence.
- Tightened log/push rules so project-external work is not logged or pushed here by default.

### Stage Results

E0 C2/C5 contract audit:

```text
status: ready_for_e1
records: 743
train: C2 291 / C5 301
validation: C2 73 / C5 78
contract errors: 0
```

C5 multiscene expansion:

```text
source C5 parent records: 379
expanded C5 rows: 799
rank1: 379
rank2: 256
rank3: 164
parent split leakage: 0
```

E1 C2 + C5 multiscene audit:

```text
status: ready_for_e1
records: 1163
train: C2 291 / C5 629
validation: C2 73 / C5 170
contract errors: 0
```

### Model Plan

Current E1 baseline:

- C2: 9-channel DeepLabV3+ specialist baseline using `B,G,R,NIR,ELEVATION,SLOPE,ASPECT,CURVATURE,TPI`.
- C5: 7-channel DeepLabV3+ multiscene baseline using `F01,F02,F03,F04,F07,F11,F12`.

Main method after E1:

```text
Prithvi optical foundation backbone
+ terrain adapter
+ disaster category branch
+ conditional segmentation decoder
```

Prithvi handles optical/multispectral foundation representation. The terrain adapter handles DEM-derived geomorphological priors for C2, especially elevation, slope, aspect, curvature, and TPI.

### Current Next Action

Run cloud smoke first:

```bash
cd /root/autodl-tmp/New_project
git pull --ff-only
PROJECT_ROOT=/root/autodl-tmp/New_project \
RUN_ROOT=workspace/runs/segmentation/e1_training_ready/smoke \
scripts/cloud_e1_training_ready.sh smoke
```

If smoke passes, run full E1 baseline training:

```bash
cd /root/autodl-tmp/New_project
git pull --ff-only
PROJECT_ROOT=/root/autodl-tmp/New_project \
RUN_ROOT=workspace/runs/segmentation/e1_training_ready/full \
scripts/cloud_e1_training_ready.sh full
```

## Canonical Notes

| Note | Purpose |
|---|---|
| `docs/ops/project-structure.md` | Source/runtime/generated boundary and cleanup policy. |
| `docs/ops/review-index.md` | First place to look before reviewing code or experiment evidence. |
| `docs/ops/work-log.md` | Concise public log of meaningful project steps and evidence. |
| `docs/ops/next-actions.md` | Current handoff and next decisions. |
| `docs/research/method-landscape.md` | Stable research map for disaster delineation route families. |
| `docs/research/segmentation-model-data-hypotheses.md` | Living model, data, channel, and normalization hypotheses. |
| `docs/implementation/current-runbook.md` | Current train/validation-only run instructions. |

## Documentation Rule

Do not add a new Markdown file for every conversation, department thought, language copy, or experiment variant. Update the living note that owns the topic. When a temporary brief is needed for `agent_team`, put it under `.agent-team/briefs/`, keep it short, and delete it after the final department artifact or living note carries the durable information.

Important Chinese handoffs and work logs belong in the Obsidian vault, not as duplicated project files.
