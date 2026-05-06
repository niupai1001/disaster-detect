# Pre-Project Method Landscape for Remote Sensing Disaster Delineation

Date: 2026-05-04

This note exists because the first brainstorm was too narrow. Before the project chooses YOLO, U-Net, SAM, or any other implementation path, it needs a broader research map of how similar remote-sensing disaster localization problems are handled.

## Scope

The project goal is not image classification. It is disaster localization and delineation from satellite imagery:

- preferred output: segmentation mask or geospatial polygon;
- fallback output: polygon box or rectangular box;
- target hazards: fire/burned area, debris flow/landslide-like geomorphic damage, and any additional labeled disaster categories in the local CSVs;
- local constraints: multispectral bands, coarse WKT polygons/masks, possible temporal image variants, and limited annotated samples.

## Method Families To Study

### 1. Classical Remote Sensing Indices And Rules

Fire and burned-area mapping has a long non-deep-learning tradition. The key lesson is not "use thresholds forever"; it is that spectral indices encode physical signals that deep models should not ignore.

Methods to study:

- `NBR`, `dNBR`, `RdNBR`
- `BAI`, `BAIS2`, `MIRBI`
- `NDVI`, `NDMI`, `NDWI`
- cloud, water, shadow, and dark-object masks
- thresholding, separability analysis, Random Forest, XGBoost

Relevant sources:

- Forest burned area / burn severity review: https://www.mdpi.com/2072-4292/14/19/4714
- BAIS2 Sentinel-2 burned area index: https://www.mdpi.com/2504-3900/2/7/364
- Sentinel-2 burned area segmentation with U-Net: https://www.mdpi.com/2072-4292/12/15/2422

Project implication: traditional indices should become baseline features, pseudo-label generators, QA overlays, or model input channels.

### 2. Supervised Semantic Segmentation

This is the direct route if masks are sufficiently trustworthy. Common model families include U-Net, ResU-Net, FCN, PSPNet, DeepLabV3+, UPerNet, SegFormer, Mask2Former, and other transformer-based decoders.

Relevant sources:

- Land8Fire wildfire segmentation benchmark: https://www.mdpi.com/2072-4292/17/16/2776
- Active fire detection dataset and deep learning study: https://www.sciencedirect.com/science/article/pii/S092427162100160X
- Remote sensing segmentation with small data survey: https://www.mdpi.com/2072-4292/15/20/4987

Project implication: segmentation is still the central target, but architecture selection should be delayed until label quality, band inventory, and split policy are known.

### 3. Landslide/Debris-Flow Mapping With Terrain And Objects

Debris-flow literature often overlaps with landslide, shallow landslide, and mass-movement mapping. The important distinction is that geomorphic disaster localization often needs terrain, texture, shape, and multi-temporal change, not only spectral appearance.

Methods to study:

- object-based image analysis (OBIA)
- Sentinel-2 pre/post change and vegetation loss
- DEM, slope, aspect, curvature, flow accumulation
- Random Forest / XGBoost object classification
- U-Net / ResU-Net semantic segmentation

Relevant sources:

- Landslide4Sense benchmark: https://arxiv.org/abs/2206.00515
- U-Net / ResU-Net transferability for landslide detection: https://www.nature.com/articles/s41598-021-94190-9
- Semi-automatic shallow landslide mapping with Sentinel-2 and GEE: https://nhess.copernicus.org/articles/23/2625/2023/
- Landslide mapping with object-based image analysis and open-source tools: https://www.sciencedirect.com/science/article/pii/S0013795221000119

Project implication: if debris-flow classes are important, the project should seek DEM/slope features early. A purely RGB or false-color image model is likely underpowered.

### 4. Change Detection And Disaster Damage Assessment

Some disaster tasks are better framed as change detection than single-image segmentation. Fire scars, floods, building damage, landslides, and debris flows can all benefit from pre-event and post-event imagery when available.

Methods to study:

- Siamese encoders for pre/post pairs
- binary and semantic change detection
- building/damage localization and severity classification
- event-level temporal grouping

Relevant sources:

- xBD building damage dataset: https://arxiv.org/abs/1911.09296
- Space-based Earth observations for disaster risk management: https://link.springer.com/article/10.1007/s10712-020-09586-5

Project implication: the file names already encode event and image dates. The project should audit whether each event has usable pre/post temporal context before assuming a single-image setup.

### 5. Weak Supervision, Coarse Labels, And Pseudo-Masks

The local WKT polygons and masks may be coarse. Treating them as perfect pixel labels can create brittle models. The project should study label-efficient segmentation before training.

Methods to study:

- sparse labels: points, scribbles, polygons
- reliable foreground / reliable background / ignore boundary
- CAM-based pseudo-labels
- affinity refinement
- pseudo-label bootstrapping with human review
- active learning for uncertain samples

Relevant sources:

- Sparse annotations for remote-sensing segmentation: https://github.com/Hua-YS/Semantic-Segmentation-with-Sparse-Labels
- Weakly supervised deep learning for remote-sensing segmentation: https://www.mdpi.com/2072-4292/12/2/207
- Remote sensing segmentation with small data survey: https://www.mdpi.com/2072-4292/15/20/4987

Project implication: the first dataset builder should preserve polygons, masks, bboxes, and ignore regions instead of collapsing everything into a single hard mask too early.

### 6. Foundation Models And Promptable Segmentation

Foundation models are not magic, but they widen the design space.

Methods to study:

- SAM / SAM-like promptable segmentation for mask proposal and refinement
- SAFE-like burned-area extraction with Sentinel-2 and SAM
- SatMAE / SatMAE++ for multispectral/temporal pretraining
- Prithvi / HLS geospatial foundation models for flood and burn-scar mapping
- remote-sensing vision-language or geospatial foundation models for transfer learning

Relevant sources:

- Segment Anything: https://arxiv.org/abs/2304.02643
- SAFE burned-area extraction with Sentinel-2 and SAM: https://www.mdpi.com/2072-4292/17/1/54
- SatMAE: https://arxiv.org/abs/2207.08051
- NASA/IBM Prithvi geospatial foundation model: https://www.earthdata.nasa.gov/news/blog/ibm-research-nasa-impact-release-open-source-geospatial-foundation-model-hugging-face
- Prithvi-EO-2.0 summary: https://research.ibm.com/publications/prithvi-eo-an-open-access-geospatial-foundation-model-advancing-earth-science-through-global-collaboration

Project implication: SAM can be used as a mask proposal/refinement tool, not as an unquestioned final segmenter. Geospatial foundation models may be more suitable than natural-image foundation models when multispectral bands matter.

## Candidate Route Portfolio

The project should keep several routes alive until a research sprint narrows them:

| Route | Core idea | Why it matters | First evidence needed |
| --- | --- | --- | --- |
| Spectral-index baseline | Use NBR/BAIS2/NDVI/DEM-style features | Strong physical prior, interpretable | Band mapping and index overlays |
| Classical ML / OBIA | Segment objects, classify with RF/XGBoost | Strong for landslide/debris-flow candidates | Object features, DEM/slope availability |
| Supervised segmentation | U-Net/DeepLab/SegFormer/Mask2Former | Direct mask output | Verified mask alignment and split policy |
| Change detection | Pre/post event models | Many disasters are change phenomena | Event/image date audit |
| Weak supervision | Coarse polygons -> reliable labels + ignore zones | Matches local rough annotation reality | Label-noise audit |
| Foundation-assisted | SAM/Prithvi/SatMAE for proposals or transfer | Modern transfer path | Compatibility with local bands and labels |
| Detection fallback | YOLO/DETR bboxes from polygons | Quick localization fallback | Mask-to-box quality and class balance |

## Research Sprint Before Project Start

Before formal implementation, run a short research sprint with these outputs:

1. Source matrix: 20-30 papers/datasets, grouped by fire, landslide/debris flow, weak supervision, change detection, foundation models.
2. Method matrix: inputs, labels, outputs, model family, metrics, data split, open code/data, relevance to local data.
3. Local-data compatibility sheet: required bands, DEM/SAR/pre-post needs, label precision assumptions, expected failure modes.
4. Three independent department proposals:
   - research.lead proposes broad research hypotheses and project framing;
   - technical.lead proposes implementation architectures and experiment infrastructure;
   - qa.tester proposes evaluation, leakage checks, and acceptance gates.
5. Revised BrainstormMemo that keeps competing routes alive instead of prematurely converging.
