# Project Direction Brainstorm Brief: Model Strategy After SegmentationDatasetContract v0.1

Date: 2026-05-05

## Objective

Run a new adversarial brainstorm to decide where the overall remote-sensing disaster segmentation project should go next after `SegmentationDatasetContract v0.1`.

This is not a QA-only brainstorm. The central question is:

> Should the project now choose a model direction, build a training baseline, expand the data/input contract, add missing data, or run a structured experiment matrix before committing to a model family?

The output should define the next project route, the first model or non-model milestone, and the decision gates that prevent premature architecture lock-in.

## Current Progress To Absorb

Primary progress note:

- `docs/implementation/2026-05-05-segmentation-contract-progress.md`

Current implementation task:

- `task-ef3a9c94ddb7`

Key artifact root:

- `.agent-team/artifacts/task-ef3a9c94ddb7/implementation/segmentation_contract_v0_1/`

Important generated artifacts:

- `model_input_manifest.csv`
- `training_manifest.csv`
- `class_map.json`
- `channels.json`
- `reports/channel_alignment_report.csv`
- `reports/wkt_mask_alignment_report.csv`
- `reports/geometry_validity_report.csv`
- `reports/visual_qa_summary.md`
- `previews/contact_sheet.png`
- `previews/source_qa/`
- `masks/`

Current usable model-input facts:

| Item | Value |
| --- | ---: |
| Model input rows | 1,252 |
| C2 debris flow rows | 500 |
| C5 fire rows | 752 |
| Train split rows | 906 |
| Validation split rows | 234 |
| Test split rows | 112 |
| Required aligned channels | `F16;F17` |
| Ignored taxonomy | C3 as `255=ignore` / `taxonomy_pending` |

Important constraints:

- Current trainable data is only C2 debris flow and C5 fire.
- The safe aligned channel contract is currently `F16;F17`.
- Richer multispectral stacks require explicit resampling/alignment work.
- 84 label rows are blocked by missing reference rasters.
- C3 cannot be treated as foreground until taxonomy is resolved.
- The existing `datasets/c5_fire_yolo/` remains historical/fallback context, not the project target.

## Existing Research Evidence To Preserve

Use these as evidence, not as binding decisions:

- `docs/research/memos/2026-05-04-study-and-conversation-synthesis.md`
- `docs/research/matrices/2026-05-04-method-source-matrix-round-1.md`
- `docs/literature/2026-05-04-pre-project-method-landscape.md`
- `.agent-team/artifacts/task-b2d1d7cb8a5e/research/research-ResearchMemo.md`
- prior brainstorm: `.agent-team/artifacts/task-ef3a9c94ddb7/brainstorm/brainstorm-BrainstormMemo.md`
- prior technical plan: `.agent-team/artifacts/task-ef3a9c94ddb7/technical/technical-TechnicalPlan.md`

## Brainstorm Questions

1. Is it time to choose a model family, or should the next milestone be a training harness plus experiment matrix?
2. If choosing a first model, should the baseline be U-Net/ResUNet, DeepLabV3+, SegFormer/UPerNet, Mask2Former, a geospatial foundation model transfer route, or something else?
3. Can a meaningful baseline be trained with only `F16;F17`, or must the project first implement resampling/alignment for richer multispectral channels?
4. Should C2 and C5 be trained in one multiclass model immediately, or should the project first run controlled class-specific baselines?
5. Is debris-flow performance likely to be limited without terrain features such as DEM, slope, aspect, curvature, or flow accumulation?
6. Should the project prioritize missing raster recovery, C3 taxonomy resolution, richer channels, or model training?
7. What metrics and experiment gates should decide whether to continue with a simple baseline or escalate to transformer/foundation-model methods?
8. Where should spectral indices, SAM/SAM2, YOLO/DETR, change detection, and geospatial foundation models sit in the roadmap?

## Method Families That Must Be Considered

The brainstorm must explicitly discuss and may only reject/defer with reasons:

- simple supervised semantic segmentation baseline: U-Net / ResUNet;
- stronger CNN/decoder baseline: DeepLabV3+ or similar;
- transformer segmentation: SegFormer / UPerNet / Mask2Former;
- geospatial foundation model transfer: Prithvi/SatMAE-like feature extractor or fine-tuning route;
- weak-supervision route with ignore boundaries, label confidence, and partial labels;
- richer multispectral channel stack after resampling/alignment;
- spectral-index diagnostics or auxiliary channels;
- terrain-enhanced debris-flow route using DEM/slope-derived features;
- change detection if pre/post pairs can be validated;
- SAM/SAM2 as mask proposal/refinement, not truth;
- YOLO/DETR as detection fallback or prompt generator, not the canonical goal.

## Department Assignments

### research.lead

Frame the overall project route after the dataset contract. Challenge premature model selection, but also challenge endless data engineering. Recommend a research-valid next route that can produce learning about the project goal.

### technical.lead

Challenge feasibility. Compare model families against the current `F16;F17` manifest, data size, class balance, dependency cost, training complexity, and future extensibility. Define the minimum technical milestone needed before or during model choice.

### qa.tester

Challenge testability. Define the experiment protocol, metrics, leakage controls, model-selection gates, visual failure review, and stop/go criteria. Block routes that cannot produce interpretable evidence.

## Required Outputs

- `ResearchBrainstormNote`
- `TechnicalBrainstormNote`
- `QABrainstormNote`
- `BrainstormMemo`

The final `BrainstormMemo` must include:

- independent positions;
- challenge and rebuttal rounds;
- a route recommendation;
- candidate model family ranking;
- what should be implemented next;
- what must be measured before escalating;
- rejected or deferred routes;
- decision log;
- next valid department and suggested assignment.

## Non-Goals

- Do not train a model in this stage.
- Do not assign technical, implementation, QA, or review departments beyond this brainstorm.
- Do not treat QA approval alone as the project direction.
- Do not choose a final production architecture.
- Do not claim multi-disaster performance from C5-only or box-only evidence.
- Do not treat `F16;F17` as the final spectral design without experiment evidence.
