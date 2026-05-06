# Evidence-Grounded Brainstorm Brief: Remote Sensing Disaster Delineation

## Objective

Run the formal adversarial brainstorm after the research-and-learning sprint. Each participating department must absorb the research department's learning artifacts before proposing routes. The goal is to preserve multiple viable method families and define what must be verified before technical planning begins.

## Research Artifacts To Absorb

- Research task: `task-b2d1d7cb8a5e`
- Research memo: `.agent-team/artifacts/task-b2d1d7cb8a5e/research/research-ResearchMemo.md`
- Chinese archive: `.agent-team/artifacts/task-b2d1d7cb8a5e/research/research-ResearchMemo-zh.md`
- Source matrix: `docs/research/matrices/2026-05-04-method-source-matrix-round-1.md`
- Method landscape: `docs/literature/2026-05-04-pre-project-method-landscape.md`

Learning logs to absorb:

- `docs/research/learning-logs/2026-05-04-land8fire-wildfire-segmentation.md`
- `docs/research/learning-logs/2026-05-04-landslide4sense-benchmark.md`
- `docs/research/learning-logs/2026-05-04-sparse-annotations-festa.md`
- `docs/research/learning-logs/2026-05-04-satmae-pretraining.md`
- `docs/research/learning-logs/2026-05-04-xbd-disaster-change-detection.md`
- `docs/research/learning-logs/2026-05-04-sam-segment-anything.md`
- `docs/research/learning-logs/2026-05-04-safe-burned-area-extraction.md`
- `docs/research/learning-logs/2026-05-04-prithvi-geospatial-foundation-model.md`

## Local Project Context

- Raw data lives under `database/`.
- Existing preprocessed dataset lives under `datasets/`, especially `datasets/c5_fire_yolo/`.
- `datasets/c5_fire_yolo/data.yaml` currently exposes one YOLO class: `0: fire`.
- `datasets/c5_fire_yolo/manifest.csv` links images, YOLO labels, bounding boxes, F07/F11/F12 source band paths, and `polygon_wkt`.
- Disaster CSV files under `database/csv文本/` include WKT polygons and category fields.
- Raw `.tif` filenames appear to encode event dates, image dates, tile/grid ids, and spectral band ids.

## Required Route Families To Preserve

The brainstorm must explicitly discuss these families and may reject or defer them only with evidence:

- classical remote-sensing indices and physical baselines;
- OBIA / classical ML with terrain and object features;
- supervised semantic segmentation;
- weak supervision from coarse WKT polygons and ignore regions;
- change detection using pre/post imagery;
- SAM-assisted candidate mask generation or refinement;
- geospatial foundation model transfer using SatMAE/Prithvi-like models;
- detection fallback using YOLO/DETR-style boxes.

## Department Assignments For Brainstorm

### research.lead

Frame the problem after the learning sprint. Challenge premature convergence, especially defaulting to YOLO, plain U-Net, or SAM without data compatibility evidence. Preserve research hypotheses and route families.

### technical.lead

Challenge implementation feasibility. For each route, identify required data, preprocessing, model infrastructure, computational cost, and integration risk. Define local-data audits required before any technical plan.

### qa.tester

Challenge testability and acceptance criteria. Define leakage checks, mask/CRS alignment checks, label-noise audits, split policy, metrics, and visual inspection gates.

## Required Outputs

- `ResearchBrainstormNote`
- `TechnicalBrainstormNote`
- `QABrainstormNote`
- `BrainstormMemo`

The final `BrainstormMemo` must include consensus, challenges, rebuttals, tradeoffs, rejected or deferred routes, decision log, and next verification needs.

## Non-Goals

- Do not start implementation.
- Do not train a model.
- Do not choose a final architecture.
- Do not treat any foundation model output as ground truth.
- Do not treat the current fire-only YOLO dataset as the project definition.
