# Course-Corrected Brainstorm Brief: First Implementation Slice for Multispectral Disaster Segmentation

Date: 2026-05-04

## Objective

Run a new adversarial brainstorm to redefine the first implementation slice for the remote-sensing disaster project after the course correction.

The corrected project goal is:

> Given multispectral satellite imagery from a sensor such as Sentinel-2, detect whether disaster-affected regions exist, identify the disaster type, and localize the affected area.

The preferred output is a multi-class semantic segmentation mask or geospatial polygon, with disaster class labels and traceability to the source bands. This brainstorm must explicitly prevent the project from drifting back into a fire-only NBR/index pipeline or a YOLO bounding-box dataset pipeline.

## Required Context

Primary course-correction inputs:

- `docs/2026-05-04-research-memo-course-correction-method-.md`
- `docs/briefs/2026-05-04-course-corrected-multispectral-disaster-segmentation.md`
- Chinese archive: `docs/briefs/2026-05-04-course-corrected-multispectral-disaster-segmentation-zh.md`

Prior project context to treat as historical evidence, not binding direction:

- `task-7c670f442094`
- `.agent-team/artifacts/task-7c670f442094/brainstorm/brainstorm-BrainstormMemo.md`
- `.agent-team/artifacts/task-7c670f442094/technical/technical-TechnicalPlan.md`
- `.agent-team/artifacts/task-7c670f442094/review/review-review-lead-ReviewNote.md`

Research evidence layer to preserve:

- `.agent-team/artifacts/task-b2d1d7cb8a5e/research/research-ResearchMemo.md`
- `docs/research/matrices/2026-05-04-method-source-matrix-round-1.md`
- `docs/literature/2026-05-04-pre-project-method-landscape.md`
- `docs/research/learning-logs/`

## Local Data Facts From the Course Correction

Known categories:

| Category | Polygon Count | Event Count | Disaster Type |
| --- | ---: | ---: | --- |
| C2 | 500 | 271 | Debris flow |
| C3 | 50 | 28 | To confirm |
| C5 | 837 | 294 | Fire |

Known data assets:

- multispectral `.tif` imagery with local bands `F01` to `F17`;
- WKT polygon labels in local CSV files;
- existing `datasets/c5_fire_yolo/`, which is a fire-only experimental asset and must not define the project target.

## Brainstorm Question

What should the first implementation slice be, if the project target is multi-band satellite disaster localization and identification?

The answer must specify:

- model input representation;
- label and mask construction strategy;
- class scope for the first slice;
- expected output artifact;
- QA gates before training;
- what is explicitly deferred.

## Method Families That Must Be Considered

The workshop must discuss and may only reject or defer with reasons:

- multi-class semantic segmentation from WKT-derived masks;
- fire-first semantic segmentation as a narrow bootstrap, if it remains aligned with the project goal;
- weak supervision and ignore-region design for coarse polygons;
- multispectral channel stacks and spectral indices as input features;
- change detection if pre/post image pairs pass validation;
- classical spectral-index baselines as diagnostics;
- instance segmentation if instance separation matters;
- foundation-model-assisted or SAM-assisted mask proposal/refinement as tools, not truth;
- YOLO/DETR object detection only as fallback or prompt generation.

## Department Assignments

### research.lead

Frame the corrected problem definition and challenge any route that does not output disaster type plus spatial mask/polygon. Preserve evidence-grounded alternatives, but recommend a research-valid first slice.

### technical.lead

Challenge feasibility and define the minimum implementable slice. Specify data contracts, input channels, mask generation, split policy dependencies, architecture boundary, and what can be built without premature model lock-in.

### qa.tester

Challenge testability. Define acceptance criteria, leakage risks, geometry/mask alignment checks, label-noise policy, baseline metrics, visual inspection gates, and conditions that block training or success claims.

## Required Outputs

- `ResearchBrainstormNote`
- `TechnicalBrainstormNote`
- `QABrainstormNote`
- `BrainstormMemo`

The final `BrainstormMemo` must include independent positions, challenge/rebuttal/negotiation content, consensus, tradeoffs, decision log, rejected or deferred options, and the next valid department.

## Non-Goals

- Do not train a model.
- Do not run dataset mutation or data pipelines.
- Do not treat old fire-only YOLO data as the canonical target.
- Do not treat NBR or any spectral index as the final product.
- Do not select a final model architecture beyond defining the first implementation slice.
