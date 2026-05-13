---
title: Work Log
date: 2026-05-08
tags:
  - ops
  - work-log
status: active
---

# Work Log

This is the concise public project log. Keep only current decisions, evidence, blockers, and next actions.

Project-external work is not logged here unless the user explicitly asks to archive that outside work in this project.

## 2026-05-08

### Project Route Locked

- The project route is now **C2+C5 disaster-aware foundation segmentation**.
- Outcome priority: paper/experiment first.
- Main backbone: Prithvi.
- Extension backbone: TerraMind.
- C2 landslide/debris-flow must be reported separately and cannot be hidden by C5 fire performance.
- Route config:
  `configs/research_routes/c2_c5_disaster_aware_foundation.yaml`.
- Validation command:

```bash
PYTHONPATH=src python -m segmentation_training validate-research-route \
  --route configs/research_routes/c2_c5_disaster_aware_foundation.yaml
```

### Data Artifacts To Preserve

- C2 paired-chip contract:
  `.agent-team/artifacts/task-2ba5416843b1/implementation/landslide_paired_chip_contract_v0_1`
- C2 paired-chip bundle:
  `.agent-team/artifacts/task-2ba5416843b1/implementation/training_bundle_c2_landslide_v0_1`
- C5 active bundle:
  `.agent-team/artifacts/task-2ba5416843b1/implementation/training_bundle_v0_3_indices`

### E0 Audit Implemented And Unblocked

- Added `research-contract-audit` CLI and report generation.
- Generated first C2/C5 E0 package:
  `.agent-team/artifacts/task-2ba5416843b1/implementation/e0_c2_c5_research_contract_audit/`
- Added `enrich-research-manifest` CLI and updated manifest generation so new C2/C5 bundles carry research fields.
- Enriched the existing C2 and C5 bundle manifests in place.
- Result after enrichment: `ready_for_e1`.
- Records audited: `743`.
- Split counts: train C2 `291`, train C5 `301`, validation C2 `73`, validation C5 `78`.
- Missing fields after enrichment: `disaster_id=0`, `label_confidence=0`, `ignore_mask_path=0`.
- Contract errors after enrichment: `0`.
- Caveat: all `label_confidence` values are currently `unknown`, and no ignore masks are declared.
- Next action is E1 baseline design, while E4 remains responsible for real relabel/ignore-mask ablation.

### E1 Training-Ready State

- Implemented leakage-safe C5 multiscene expansion with `c5-multiscene-manifest`.
- C5 expansion policy: top-3 same-day/post-event scenes, 0-30 days after event, required channels `F01,F02,F03,F04,F07,F11,F12`, split inherited from parent sample.
- Generated C5 multiscene bundle:
  `.agent-team/artifacts/task-2ba5416843b1/implementation/training_bundle_c5_multiscene_top3_v0_1/`
- Expansion result: source C5 parents `379`, expanded rows `799`, rank1 `379`, rank2 `256`, rank3 `164`, parent split leakage `0`.
- Generated E1 combined audit:
  `.agent-team/artifacts/task-2ba5416843b1/implementation/e1_training_ready_audit/`
- E1 audit result: `ready_for_e1`, total records `1163`, train C2 `291`, train C5 `629`, validation C2 `73`, validation C5 `170`, contract errors `0`.
- Added cloud entry script:
  `scripts/cloud_e1_training_ready.sh`.
- Smoke command:

```bash
cd /root/autodl-tmp/New_project
git pull --ff-only
PROJECT_ROOT=/root/autodl-tmp/New_project \
RUN_ROOT=workspace/runs/segmentation/e1_training_ready/smoke \
scripts/cloud_e1_training_ready.sh smoke
```

- Full command after smoke passes:

```bash
cd /root/autodl-tmp/New_project
git pull --ff-only
PROJECT_ROOT=/root/autodl-tmp/New_project \
RUN_ROOT=workspace/runs/segmentation/e1_training_ready/full \
scripts/cloud_e1_training_ready.sh full
```

### Log Boundary Tightened

- Added rule: do not write project-external task logs here unless the user explicitly asks.
- Cleaned the Obsidian project work log by removing non-project entries.
