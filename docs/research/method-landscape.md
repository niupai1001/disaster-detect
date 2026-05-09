---
title: Method Landscape
date: 2026-05-08
tags:
  - research
  - remote-sensing
  - disaster-delineation
  - foundation-model
status: active
---

# Method Landscape

The project goal is now a paper-oriented, reproducible method-and-experiment route for disaster localization and delineation. The active scope is **C2 landslide/debris-flow + C5 fire/burned area**. C5 remains useful reference evidence, but project progress must be judged with C2 and C5 reported separately plus a macro summary.

The locked route is defined in:

```text
configs/research_routes/c2_c5_disaster_aware_foundation.yaml
```

Validate it with:

```bash
PYTHONPATH=src python -m segmentation_training validate-research-route \
  --route configs/research_routes/c2_c5_disaster_aware_foundation.yaml
```

## Active Technical Route

| Layer | Decision | Purpose |
|---|---|---|
| Project outcome | Paper/experiment first | Stop treating isolated P-number runs as project progress. |
| Contribution | Method + experiments | Produce a publishable technical claim, not only engineering cleanup. |
| Disaster scope | C2 + C5 | Keep C2 landslide failure visible and use C5 as reference evidence. |
| Main backbone | Prithvi | First foundation backbone to audit and fine-tune. |
| Extension | TerraMind | Later multimodal/terrain-aware extension, not a blocker for stage one. |
| Method | Disaster-aware segmentation | Add category prediction and category-conditioned segmentation. |
| C2 support | Terrain adapter | Fuse DEM/slope/aspect/curvature/TPI with optical features. |

## Route Families To Preserve

| Route | Role in the new plan | First evidence needed |
|---|---|---|
| Classical remote-sensing indices | E1 sanity baseline, especially C5 burned area. | NBR, NDVI, NDMI, BAIS2/MIRBI separability and threshold behavior. |
| OBIA / classical ML | C2 sanity check and fallback evidence. | Object features, terrain features, texture, shape, and simple ML results. |
| Deep segmentation | Strong non-foundation baseline. | DeepLabV3+/Transformer-UNet fair validation metrics per disaster. |
| Specialist upper bound | Diagnose whether each disaster is learnable alone. | C2-only and C5-only validation results under sealed-test guard. |
| Category-unknown lower bound | Tests final inference realism. | Category prediction plus mask quality without manual disaster label. |
| Foundation-assisted | Main method track. | Prithvi input compatibility, fine-tuning path, and feature fusion evidence. |
| TerraMind extension | Second-stage research extension. | Weights/interface/compute feasibility and C2 terrain benefit. |
| Detection fallback | Only if C2 segmentation remains unstable. | Object-level or box quality compared with mask failure modes. |

## Experiment Sequence

| Stage | Question | Required output |
|---|---|---|
| E0 | Are data fields, label confidence, ignore masks, and splits defensible? | Research-contract audit with no split leakage. |
| E1 | What do classical and strong deep baselines achieve? | C2/C5/macro metrics and hard-case review. |
| E2 | Is Prithvi compatible and useful as a vanilla backbone? | Prithvi optical fine-tune result or documented incompatibility. |
| E3 | Does disaster awareness improve the route? | Category branch + conditional head result, especially C2 gain. |
| E4 | Do relabel/ignore/confidence policies change outcomes? | Label-policy ablation. |
| E5 | Does terrain fusion help C2? | Terrain adapter ablation with false-positive review. |
| E6 | How much does unknown category inference cost? | Category-known upper bound vs category-unknown lower bound. |
| E7 | Does TerraMind add enough value to promote? | Feasibility and extension result, or frozen future-work note. |

## Stop And Promote Rules

- Do not use the sealed test split until a final route is selected.
- Do not promote a method on C5 alone; C2 must improve or the route must explain why it cannot.
- If Prithvi is incompatible with the available channels, keep it as a documented audit result and use the strongest deep model only as a baseline, not as the main research claim.
- If C2 still fails after E3/E5, pivot the claim to data/label/modality bottleneck analysis before adding more model complexity.
