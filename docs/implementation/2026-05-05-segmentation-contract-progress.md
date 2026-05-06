# SegmentationDatasetContract v0.1 Progress Report

Date: 2026-05-05

Task: `task-ef3a9c94ddb7`

## Summary

The project has completed the dataset-contract implementation path from Phase A metadata construction through Phase B mask generation, visual QA, and a conservative model-input contract. The current artifact set is sufficient for a later training data loader to consume verified samples, but the project workflow should still assign the QA department before treating the dataset as formally approved training data.

The implementation remains inside the corrected project scope: multispectral disaster semantic segmentation for C2 debris flow and C5 fire. C3 remains `255=ignore` / `taxonomy_pending` and is excluded from foreground training manifests.

## Implemented Development Slices

### Phase A: Schema and Manifest Contract

Implemented package:

- `src/segmentation_contract/`

Main generated artifacts:

- `samples_manifest.csv`
- `class_map.json`
- `channels.json`
- `split_policy.md`
- `skipped_label_rows.csv`
- `qa_summary.md`
- `dataset_contract.md`

Phase A established stable sample IDs, event-level split policy, class IDs, source CSV row traceability, source raster linkage, and audit logs for malformed or skipped rows.

### Phase B: Geospatial Mask Generation

Phase B now uses `shapely` and `rasterio` to parse WKT polygons, inspect reference rasters, rasterize semantic masks, and write geospatial reports.

Main generated artifacts:

- `masks/*.tif`
- `reports/geometry_validity_report.csv`
- `reports/wkt_mask_alignment_report.csv`
- `reports/phase_b_dependency_report.md`

Current Phase B result:

| Item | Count |
| --- | ---: |
| Manifest rows | 1,386 |
| Geometry valid rows | 1,382 |
| Invalid WKT rows | 4 |
| Masks written | 1,302 |
| Missing reference raster rows | 84 |

The 84 blocked rows are not usable for training until matching reference rasters are found or regenerated.

### Visual QA

Generated visual inspection artifacts:

- `previews/contact_sheet.png`
- `previews/source_qa/泥石流火灾_C2_debris_flow_gray_contact_sheet.png`
- `previews/source_qa/泥石流火灾_C5_fire_gray_contact_sheet.png`
- `previews/source_qa/泥石流火灾_training_samples_index.csv`
- `reports/visual_qa_summary.md`

The user cross-checked the preview overlays and confirmed that the apparent issues were mainly band-display limitations rather than mask alignment errors.

### Training and Model Input Manifests

Generated manifests:

- `training_manifest.csv`
- `model_input_manifest.csv`
- `reports/channel_alignment_report.csv`
- `reports/model_input_summary.md`

`training_manifest.csv` includes only rows where:

- `alignment_status == mask_written`
- `class_id != 255`

`model_input_manifest.csv` further confirms that required model-input channels are grid-aligned. The current conservative model-input contract uses `F16;F17`, because all trainable rows have those two bands aligned in CRS, transform, height, and width.

Current model-input result:

| Item | Count |
| --- | ---: |
| Model input rows | 1,252 |
| C2 debris flow rows | 500 |
| C5 fire rows | 752 |
| Train split rows | 906 |
| Validation split rows | 234 |
| Test split rows | 112 |
| Required channels | `F16;F17` |
| Channel alignment status | `aligned` for all 1,252 |

## Data Source Mapping

The currently usable trainable data comes from:

- `database/csv文本/泥石流火灾.csv`

Current usable rows by class:

| Source CSV | Class | Rows |
| --- | --- | ---: |
| `泥石流火灾.csv` | `C2_debris_flow` | 500 |
| `泥石流火灾.csv` | `C5_fire` | 752 |

Rows from these CSV files are currently blocked because no linked reference rasters were found:

- `database/csv文本/damage_poly_r3云南.csv`
- `database/csv文本/damage_poly_r3四川.csv`

## Important Technical Findings

1. `F16` and `F17` are the safest current model-input channels because they align across all 1,252 trainable rows.
2. Some RGB-like channel combinations, such as `F04/F03/F02`, have resolution mismatches in parts of the dataset and must not be stacked directly without resampling.
3. Richer multispectral stacks are still possible, but they need an explicit resampling/alignment implementation before training.
4. C3 is intentionally excluded from foreground training because its taxonomy remains pending.
5. The generated masks are suitable for training-loader development, but formal QA review should happen before model training claims.

## Current Artifact Root

All generated dataset-contract artifacts live under:

```text
.agent-team/artifacts/task-ef3a9c94ddb7/implementation/segmentation_contract_v0_1/
```

Most important files for the next development stage:

```text
model_input_manifest.csv
training_manifest.csv
class_map.json
reports/channel_alignment_report.csv
reports/wkt_mask_alignment_report.csv
reports/geometry_validity_report.csv
previews/contact_sheet.png
previews/source_qa/
masks/
```

## Verification

The latest development verification command passed:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

Result:

```text
Ran 33 tests
OK
```

## Remaining Risks and Next Steps

Recommended next department:

```text
测试部门测试
```

Recommended QA scope:

- verify Phase B masks and reports;
- inspect visual QA contact sheets;
- confirm `model_input_manifest.csv` excludes `255=ignore`;
- confirm split counts and leakage policy;
- confirm `F16/F17` alignment assumptions;
- record whether the artifact set can be approved for training-loader development.

If development continues before formal QA, the next safe implementation slice is a data-loader smoke path that reads `model_input_manifest.csv` and returns:

```text
X = [F16, F17, H, W]
y = [H, W]
```

This next slice should still avoid model training and architecture selection until QA has approved the dataset artifacts.
