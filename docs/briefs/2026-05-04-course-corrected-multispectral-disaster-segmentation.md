# Course-Corrected Brief: Multispectral Disaster Localization and Identification

Date: 2026-05-04

Source memo: `docs/2026-05-04-research-memo-course-correction-method-.md`

## Corrected Project Goal

Given multispectral satellite imagery from a sensor such as Sentinel-2, determine whether disaster-affected regions exist in the image, identify the disaster type, and localize the affected area.

The expected system output is:

- disaster class or classes present in the scene;
- spatial localization of the disaster area, preferably as a semantic segmentation mask or geospatial polygon;
- optional confidence and traceability back to the input bands and source raster.

This replaces the earlier drift toward a fire-only spectral-index or YOLO bounding-box pipeline.

## Problem Framing

The project should be framed as a multi-class remote-sensing semantic segmentation problem.

Primary reasons:

- The local supervision includes WKT polygons, which naturally map to pixel masks.
- Disaster regions such as burned areas and debris flows are often irregular, elongated, or spatially diffuse.
- The target is disaster type plus location, not only an image-level label or rectangular box.
- Multiple disaster categories are present locally, so a single-class fire detector is not the project definition.

## Local Data Context

Known local categories:

| Category | Polygon Count | Event Count | Disaster Type |
| --- | ---: | ---: | --- |
| C2 | 500 | 271 | Debris flow |
| C3 | 50 | 28 | To confirm |
| C5 | 837 | 294 | Fire |

Known assets:

- multispectral `.tif` imagery with local band IDs such as `F01` to `F17`;
- WKT polygon labels in local CSV files;
- an existing `datasets/c5_fire_yolo/` dataset that should be treated as a fire-only experimental asset, not the target architecture.

## Main Method Direction

The first implementation direction should be multi-disaster semantic segmentation.

Candidate model families can include:

- U-Net-style segmentation baselines;
- DeepLabV3+, SegFormer, Mask2Former, or similar semantic segmentation models;
- weak-supervision variants if WKT polygons are coarse or noisy;
- multispectral input channels and selected spectral indices as auxiliary features.

Spectral indices such as NBR, dNBR, NDVI, BAI, and MIRBI should be treated as input features or diagnostic baselines, not as the final product.

## Secondary and Fallback Routes

These routes remain useful, but they should not redefine the project:

- Instance segmentation if separate disaster instances must be distinguished.
- Change detection if reliable pre-event and post-event pairs can be validated.
- Classical remote-sensing index thresholds as physics-grounded baselines.
- SAM-assisted or foundation-model-assisted mask proposal/refinement, with proposals never treated as ground truth.
- YOLO or DETR-style object detection as a coarse localization fallback or prompt generator only.

## Required Corrections to Previous Direction

The previous fire-only slice should be considered off target unless it is explicitly reframed as a narrow baseline for the C5 fire class.

The corrected first slice should answer:

> What model input representation lets us take multispectral imagery and output disaster type plus aligned mask?

It should not start from:

> How do we compute NBR or convert fire polygons into a YOLO dataset?

## Decision Constraints

Before training or selecting a final architecture, the project still needs to verify:

- band meanings and sensor mapping for `F01` to `F17`;
- raster, WKT polygon, generated mask, and preview alignment;
- category taxonomy, especially the meaning of C3;
- event-level split policy to avoid leakage;
- label-noise and ignore-region policy for polygon-derived masks;
- whether pre/post pairs are reliable enough for change detection.

## Recommended Next Department Action

The next valid action is a new course-corrected brainstorm or a revised technical plan, explicitly grounded in this brief and the source course-correction memo.

Recommended instruction:

`头脑风暴：基于 docs/2026-05-04-research-memo-course-correction-method-.md 和本 brief，重新定义多波段卫星灾害定位识别项目的第一实现切片。`

Do not let later departments treat the old fire-only YOLO dataset or spectral-index pipeline as the canonical project target.
