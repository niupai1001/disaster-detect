# ResearchMemo: Remote Sensing Disaster Delineation Methods

Date: 2026-05-04

Task: `task-b2d1d7cb8a5e`

## Research Scope

The research sprint studied methods for remote-sensing disaster localization and delineation before any technical route is selected. The target project should prefer segmentation masks or geospatial polygons, keep boxes only as a fallback, and avoid treating the existing fire-only YOLO dataset as the default path.

## Evidence Layer

Learning logs now use a paper-first structure: problem, data, method, innovation, conclusions/limitations, project relevance, brainstorm constraints, and next checks.

| Source | Learning Log | Method Family |
| --- | --- | --- |
| Land8Fire | `docs/research/learning-logs/2026-05-04-land8fire-wildfire-segmentation.md` | multispectral wildfire segmentation benchmark |
| Landslide4Sense | `docs/research/learning-logs/2026-05-04-landslide4sense-benchmark.md` | landslide segmentation with optical + terrain data |
| Sparse annotations / FESTA | `docs/research/learning-logs/2026-05-04-sparse-annotations-festa.md` | weakly supervised remote-sensing segmentation |
| SatMAE | `docs/research/learning-logs/2026-05-04-satmae-pretraining.md` | multispectral/temporal self-supervised pretraining |
| xBD | `docs/research/learning-logs/2026-05-04-xbd-disaster-change-detection.md` | pre/post disaster damage assessment |
| Segment Anything | `docs/research/learning-logs/2026-05-04-sam-segment-anything.md` | promptable mask proposal |
| SAFE | `docs/research/learning-logs/2026-05-04-safe-burned-area-extraction.md` | burned-area candidate generation + SAM refinement |
| Prithvi | `docs/research/learning-logs/2026-05-04-prithvi-geospatial-foundation-model.md` | geospatial foundation model transfer |

The source matrix is `docs/research/matrices/2026-05-04-method-source-matrix-round-1.md`.

## What The Research Department Learned

Fire and burned-area work should be treated as multispectral segmentation, not as simple object detection. Land8Fire shows that the benchmark itself matters: band combinations, loss functions, class imbalance, and fire morphology can change conclusions. This project must keep NIR/SWIR-like bands and fire indices available instead of compressing everything into RGB.

Debris-flow work should not be folded into the fire route. Landslide4Sense shows that geomorphic disaster mapping benefits from DEM and slope. If debris flow is a serious target class, the project needs a terrain-data decision early: acquire DEM-derived features, explicitly run without them as a limitation, or separate debris-flow detection into a weaker exploratory track.

The local WKT labels should be treated as coarse supervision until proven otherwise. Sparse-label segmentation work suggests a better label model: reliable foreground, ignore boundary, reliable background, and partial-label training. A direct polygon-to-hard-mask conversion is risky.

Some disaster tasks are better framed as change detection. xBD is building-focused, but its pre/post event structure is directly relevant because local filenames appear to encode event dates and image dates. The project needs an event/date audit before it commits to a single-image model.

Foundation models must be split into distinct routes. SAM is useful for promptable mask proposal and annotation refinement. SatMAE and Prithvi are better candidates for multispectral or temporal remote-sensing transfer. None of these should be treated as ground truth or as automatic replacements for dataset QA.

## Candidate Route Portfolio

| Route | Keep Alive Because | First Evidence Needed |
| --- | --- | --- |
| Spectral-index baseline | physical prior for fire and vegetation change | band mapping, NBR/NDVI/SWIR overlays |
| OBIA/classical ML | useful for terrain/object-driven debris-flow candidates | DEM/slope availability, object features |
| Supervised segmentation | direct mask/polygon output | verified mask alignment and split policy |
| Weak supervision | matches coarse WKT reality | label-noise audit and ignore-zone design |
| Change detection | disasters often appear as temporal change | event/image date pair audit |
| SAM-assisted proposal | can refine boxes/polygons into candidate masks | prompt quality and spectral QA checks |
| Geospatial FM transfer | useful for small labeled multispectral data | local band/schema compatibility |
| Detection fallback | quick coarse localization baseline | mask-to-box quality and class balance |

## Constraints For The Formal Brainstorm

- Each department must read the learning logs and source matrix before proposing a route.
- The brainstorm must preserve competing method families; it must not converge on YOLO, U-Net, SAM, or any single familiar algorithm too early.
- Technical proposals must include local-data compatibility checks: band inventory, CRS/mask alignment, event/date pairing, DEM/slope availability, and class balance.
- QA proposals must include leakage checks by event, date, region, and neighboring patch.
- Any foundation-model proposal must state whether it is a proposal/refinement tool, a backbone/feature extractor, or a pseudo-label generator.
- Any segmentation proposal must specify how coarse WKT boundaries and ignore regions will be handled.

## Handoff

The research department recommends starting a new evidence-grounded adversarial brainstorm. The new brainstorm should explicitly absorb these research outputs and produce independent notes from research, technical, and QA before the orchestrator writes a revised `BrainstormMemo`.
