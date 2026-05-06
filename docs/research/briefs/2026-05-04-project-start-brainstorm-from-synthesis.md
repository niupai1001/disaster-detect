# Project-Start Brainstorm Brief: Remote Sensing Disaster Delineation From Study Synthesis

## Objective

Start the formal project-start adversarial brainstorm using the study-and-conversation synthesis as the shared context. The workshop must convert the research learning, local data constraints, and conversation requirements into a route portfolio, a staged project entry point, and verification needs for later technical planning.

This is a brainstorm assignment only. It must not implement data processing, train a model, select a final architecture, or assign later departments.

## Primary Context To Absorb

- Current synthesis memo: `docs/research/memos/2026-05-04-study-and-conversation-synthesis.md`
- Research task: `task-b2d1d7cb8a5e`
- Research memo: `.agent-team/artifacts/task-b2d1d7cb8a5e/research/research-ResearchMemo.md`
- Chinese research archive: `.agent-team/artifacts/task-b2d1d7cb8a5e/research/research-ResearchMemo-zh.md`
- Source matrix: `docs/research/matrices/2026-05-04-method-source-matrix-round-1.md`
- Method landscape: `docs/literature/2026-05-04-pre-project-method-landscape.md`
- Prior evidence-grounded brainstorm, for historical reference only: `task-7a02ad0777e6`

Learning logs to absorb:

- `docs/research/learning-logs/2026-05-04-land8fire-wildfire-segmentation.md`
- `docs/research/learning-logs/2026-05-04-landslide4sense-benchmark.md`
- `docs/research/learning-logs/2026-05-04-sparse-annotations-festa.md`
- `docs/research/learning-logs/2026-05-04-satmae-pretraining.md`
- `docs/research/learning-logs/2026-05-04-xBD-disaster-change-detection.md` if present; otherwise use `docs/research/learning-logs/2026-05-04-xbd-disaster-change-detection.md`
- `docs/research/learning-logs/2026-05-04-sam-segment-anything.md`
- `docs/research/learning-logs/2026-05-04-safe-burned-area-extraction.md`
- `docs/research/learning-logs/2026-05-04-prithvi-geospatial-foundation-model.md`

## Local Project Context

- The project target is disaster-area localization and delineation from remote-sensing imagery, not image-level classification.
- Raw data lives under `database/`.
- Existing preprocessed data lives under `datasets/`, especially `datasets/c5_fire_yolo/`.
- `datasets/c5_fire_yolo/data.yaml` currently exposes one YOLO class: `0: fire`.
- `datasets/c5_fire_yolo/manifest.csv` links images, YOLO labels, bounding boxes, F07/F11/F12 source band paths, and `polygon_wkt`.
- Disaster CSV files under `database/csv文本/` include WKT polygons and category fields.
- Raw `.tif` filenames appear to encode event dates, image dates, tile/grid ids, and spectral band ids.

## Required Brainstorm Discipline

- Follow an adversarial workshop structure: independent position, challenge round, rebuttal round, and negotiation position.
- Preserve multiple candidate method families until local evidence supports narrowing.
- Ground route proposals in the research/literature evidence layer, not only in the current fire-only YOLO dataset.
- Treat Superpowers-style brainstorming as a quality bar: present competing approaches, name tradeoffs, recommend a staged direction, and keep implementation behind explicit approval gates.
- Keep all later departments idle until the user or coordinator explicitly assigns them.

## Method Families That Must Be Discussed

The brainstorm must explicitly discuss these route families and may reject or defer them only with evidence:

- classical remote-sensing indices and physical baselines;
- OBIA / classical ML with terrain and object features;
- supervised semantic segmentation;
- weak supervision from coarse WKT polygons, ignore regions, and label confidence;
- change detection using pre-event and post-event imagery if local dates allow pairing;
- SAM-assisted candidate mask generation or refinement;
- SAFE-like fire or burned-area candidate generation and refinement;
- geospatial foundation model transfer using SatMAE/Prithvi-like models;
- detection fallback using YOLO/DETR-style boxes for coarse localization or prompts.

## Questions For The Workshop

- What should count as the first project milestone: a data audit and readiness plan, fire-first multispectral segmentation, multi-class WKT mask construction, or a detector fallback plus later segmentation?
- Which supervision source should drive the first usable dataset: CSV WKT polygons, generated masks, existing YOLO labels, manual/visual QA, or weak-label composites?
- How should F07/F11/F12 and possible additional bands be represented: physical indices, false-color previews, stacked multi-band tensors, foundation-model-compatible schemas, or hybrid features?
- Does local metadata support pre/post change detection, and what audit is needed before relying on it?
- What must be true before debris-flow/landslide work can be considered viable, especially around DEM, slope, aspect, curvature, and flow accumulation?
- How should SAM, SAFE, SatMAE, Prithvi, and YOLO be positioned as tools without becoming unverified ground truth?
- What QA gates should block technical planning, model training, and claims of success?

## Department Assignments For Brainstorm

### research.lead

Frame the project goal from the synthesis memo and research evidence. Challenge premature convergence on YOLO, plain U-Net, SAM, or any single familiar path. Preserve route families and name evidence needed to narrow them.

### technical.lead

Challenge implementation feasibility. For each route, identify required local data, preprocessing, infrastructure, computational cost, integration risk, and the minimum audit artifact needed before planning implementation.

### qa.tester

Challenge testability. Define acceptance criteria, leakage checks, mask/CRS alignment checks, label-noise audits, split policy, visual inspection gates, and failure modes for each route family.

## Required Outputs

- `ResearchBrainstormNote`
- `TechnicalBrainstormNote`
- `QABrainstormNote`
- `BrainstormMemo`

The final `BrainstormMemo` must include independent notes, challenge round, rebuttal round, consensus, tradeoffs, rejected or deferred routes, decision log, and next verification needs.

## Non-Goals

- Do not start implementation.
- Do not train a model.
- Do not modify datasets.
- Do not select a final model architecture.
- Do not treat WKT polygons, generated masks, SAM outputs, SAFE outputs, or YOLO boxes as audited ground truth.
- Do not treat the current fire-only YOLO dataset as the project definition.
