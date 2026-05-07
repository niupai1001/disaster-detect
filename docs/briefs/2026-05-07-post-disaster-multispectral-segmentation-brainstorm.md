# Post-Disaster Multispectral Segmentation Brainstorm Brief

Date: 2026-05-07

## Objective

Run an adversarial department brainstorm for the corrected project direction:

> The project goal is post-disaster recognition and segmentation. Do not treat pre-disaster prediction as a current objective.

The immediate question is how to move away from the current underused `F16/F17` two-channel training contract toward a post-disaster, multispectral segmentation route that can actually expose fire and debris-flow foreground signal.

## User Direction To Treat As Binding

- The target is post-disaster identification and pixel-level segmentation.
- Do not optimize for predicting disasters before they happen.
- The repeated zero foreground segmentation results should be interpreted in light of weak foreground/background separability and underuse of the generated data pipeline.
- The team should reassess whether richer multispectral data should be introduced instead of continuing to tune a single weak input view.

## Current Evidence To Absorb

- Current training configs still use `input_channels: F16;F17`.
- The training bundle contains 1,252 model-input rows: 500 C2 debris-flow rows, 752 C5 fire rows.
- Train/validation/test counts are 906 / 234 / 112; the test split remains sealed.
- Recent E1 C5 U-Net training reached low validation loss around `0.083` but kept `mean_iou=0.0000` and `foreground_recall=0.0000`.
- C5 foreground masks are sparse; median validation foreground fraction is roughly 0.3%.
- The contract preserves many raw source bands in `source_band_paths`, including Sentinel-2-like optical bands, SWIR-like bands, and Sentinel-1-like `F16/F17`, but `model_input_manifest.csv` currently exposes only `F16/F17`.
- Local paper notes emphasize that fire and landslide segmentation usually benefit from richer Sentinel-2 multispectral bands, SWIR/red-edge/NIR features, spectral indices such as NBR, dNBR, NDVI, MIRBI, and, when useful, SAR as an auxiliary modality.

## Research Artifacts To Absorb

- `/Users/Lemon/Desktop/Hermes/research/disaster-detection/README.md`
- `/Users/Lemon/Desktop/Hermes/research/disaster-detection/2023-Ghali-RS-wildfire-vit.md`
- `/Users/Lemon/Desktop/Hermes/research/disaster-detection/2023-BelenguerPlomer-RSE-burnseverity.md`
- `/Users/Lemon/Desktop/Hermes/research/disaster-detection/2022-Seydi-JAG-fire-attention.md`
- `/Users/Lemon/Desktop/Hermes/research/disaster-detection/2021-Prakash-Landslides-Swiss.md`
- `/Users/Lemon/Desktop/Hermes/research/disaster-detection/2023-HuBan-IJAG-sarmultimodal.md`
- `docs/research/memos/2026-05-04-study-and-conversation-synthesis.md`
- `docs/literature/2026-05-04-pre-project-method-landscape.md`

## Department Assignments

### research.lead

Frame the post-disaster segmentation route using the literature. Challenge any route that keeps treating `F16/F17` as enough evidence. Recommend which post-disaster multispectral channel families and model families should be prioritized for C5 fire and C2 debris-flow segmentation.

### technical.lead

Challenge feasibility. Define the minimum implementation plan for a `PostDisasterMultispectralContract v0.2`: channel/date auditing, post-disaster image selection, multispectral resampling/alignment, derived index channels, config changes, model adapters, and cloud bundle changes. Keep the sealed test split untouched.

### qa.tester

Challenge testability. Define acceptance gates that prove the new contract uses post-disaster data correctly, avoids leakage, preserves grid alignment, improves foreground separability, and produces interpretable validation evidence before any model-family claims.

## Questions For The Brainstorm

1. What is the most plausible root cause of repeated zero foreground segmentation under the current pipeline?
2. Which source bands should form the first post-disaster multispectral training stack for C5 fire?
3. Which source bands or auxiliary features should be prioritized for C2 debris-flow segmentation?
4. Should SAR `F16/F17` remain as auxiliary channels, be used only for diagnostics, or be excluded from the first post-disaster segmentation stack?
5. Which derived indices should be implemented first, and which require validated pre/post pairs versus post-only imagery?
6. What is the smallest data-contract change that would make the next training run scientifically meaningful?
7. Should the first model after the contract repair remain U-Net/ResUNet for measurement, or should SegFormer/ViT enter immediately once multi-band input exists?
8. What evidence must be produced before QA accepts that foreground/background separability has improved?

## Required Outputs

- `ResearchBrainstormNote`
- `TechnicalBrainstormNote`
- `QABrainstormNote`
- `BrainstormMemo`

The final `BrainstormMemo` must include independent department positions, challenge/rebuttal/negotiation sections, recommended route, implementation sequence, QA gates, deferred options, decision log, and next valid department.

## Non-Goals

- Do not train a new model in brainstorm.
- Do not use the sealed test split.
- Do not make pre-disaster prediction a current objective.
- Do not claim that transformer models solve missing input information.
- Do not treat `F16/F17` as the final spectral design.
- Do not treat multi-band expansion as useful unless channel date, grid alignment, and post-disaster suitability are auditable.
