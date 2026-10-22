# SegmentationTrainingBaseline v0.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a cloud-first segmentation training baseline harness for C2/C5 disaster masks without running full training locally.

**Architecture:** Local code validates manifests, builds a cloud-ready bundle, runs CPU-only tests and tiny smoke checks, and summarizes returned cloud runs. Cloud runs execute U-Net/ResUNet baseline experiments from the same configs and return metrics, checkpoints, logs, and prediction previews.

**Tech Stack:** Python 3.11, standard library config/CSV modules, rasterio for GeoTIFF IO, numpy/Pillow for metrics and previews, PyTorch only inside cloud/training modules, optional PyYAML for configs.

---

## Scope Rules

- Do not run full model training on this local machine.
- Local commands may run unit tests, synthetic tests, manifest validation, bundle validation, and tiny smoke checks.
- Cloud training commands must be documented and implemented but not executed locally.
- Keep `task-afc7f2c25f8f` and `task-ef3a9c94ddb7` references in generated run manifests.
- Preserve the test split; no model-selection command should read test rows by default.

## File Structure

Create:

```text
src/segmentation_training/__init__.py
src/segmentation_training/__main__.py
src/segmentation_training/cli.py
src/segmentation_training/manifest.py
src/segmentation_training/bundle.py
src/segmentation_training/config.py
src/segmentation_training/metrics.py
src/segmentation_training/preview.py
src/segmentation_training/cloud.py
src/segmentation_training/data.py
src/segmentation_training/sampler.py
src/segmentation_training/losses.py
src/segmentation_training/models/__init__.py
src/segmentation_training/models/unet.py
src/segmentation_training/models/resunet.py
configs/segmentation_training/e0_trivial.yaml
configs/segmentation_training/e1_binary_c5_unet.yaml
configs/segmentation_training/e2_binary_c2_unet.yaml
configs/segmentation_training/e3_multiclass_c2_c5_unet.yaml
docs/implementation/2026-05-05-segmentation-training-cloud-runbook.md
tests/test_segmentation_training_manifest.py
tests/test_segmentation_training_bundle.py
tests/test_segmentation_training_config.py
tests/test_segmentation_training_metrics.py
tests/test_segmentation_training_preview.py
```

Modify:

```text
src/segmentation_training/__init__.py
```

No existing `src/segmentation_contract/` files should be changed unless a missing contract field blocks the implementation.

### Task 1: Manifest Reader And Class Filtering

**Files:**
- Create: `src/segmentation_training/manifest.py`
- Test: `tests/test_segmentation_training_manifest.py`

- [ ] **Step 1: Write tests for manifest parsing**

Create tests that write a three-row CSV with `sample_id`, `split`, `class_id`, `class_name`, `mask_path`, `input_channels`, and `input_band_paths`. Assert that `load_model_input_manifest()` returns typed records, preserves split strings, parses `F16:/path/a.tif;F17:/path/b.tif`, and rejects missing required channels.

- [ ] **Step 2: Implement manifest dataclass and parser**

Implement `ModelInputRecord`, `parse_band_paths()`, `load_model_input_manifest()`, and `filter_records()`. Class scopes must support `binary_c2`, `binary_c5`, and `multiclass_c2_c5`.

- [ ] **Step 3: Verify**

Run:

```bash
PYTHONPATH=src python -m unittest tests.test_segmentation_training_manifest
```

Expected: tests pass without importing torch.

### Task 2: Cloud Bundle Builder

**Files:**
- Create: `src/segmentation_training/bundle.py`
- Test: `tests/test_segmentation_training_bundle.py`

- [ ] **Step 1: Write tests for manifest-only bundle**

Use temporary masks and fake raster paths. Assert that `build_training_bundle(..., mode="manifest-only")` creates:

```text
metadata/class_map.json
metadata/channels.json
metadata/bundle_manifest.json
manifests/cloud_model_input_manifest.csv
README.md
```

- [ ] **Step 2: Implement bundle creation**

Implement copying metadata, rebasing manifest paths to `cloud_data_root`, and writing SHA-256 checksums for files included in the bundle.

- [ ] **Step 3: Verify**

Run:

```bash
PYTHONPATH=src python -m unittest tests.test_segmentation_training_bundle
```

Expected: tests pass and no large raster copy is required.

### Task 3: Config Validation

**Files:**
- Create: `src/segmentation_training/config.py`
- Create: `configs/segmentation_training/e0_trivial.yaml`
- Create: `configs/segmentation_training/e1_binary_c5_unet.yaml`
- Create: `configs/segmentation_training/e2_binary_c2_unet.yaml`
- Create: `configs/segmentation_training/e3_multiclass_c2_c5_unet.yaml`
- Test: `tests/test_segmentation_training_config.py`

- [ ] **Step 1: Write config tests**

Assert that all four configs load, contain `task_id`, `experiment_id`, `class_scope`, `input_channels`, `split_policy`, `model`, `metrics`, and `cloud`, and reject configs where `allow_test_split_for_selection` is true.

- [ ] **Step 2: Implement config loader**

Use `yaml.safe_load` when PyYAML is installed; otherwise allow JSON-compatible YAML subset through a small fallback or skip config-file parsing with a clear error. Validation must be deterministic and explicit.

- [ ] **Step 3: Verify**

Run:

```bash
PYTHONPATH=src python -m unittest tests.test_segmentation_training_config
```

Expected: all configs validate and test-split misuse is rejected.

### Task 4: Metrics Module

**Files:**
- Create: `src/segmentation_training/metrics.py`
- Test: `tests/test_segmentation_training_metrics.py`

- [ ] **Step 1: Write metric tests**

Create small `numpy` arrays for labels and predictions with class IDs `0`, `1`, `2`, and `255`. Assert that ignored pixels do not affect counts and that per-class IoU, Dice/F1, precision, recall, and confusion matrix values are correct.

- [ ] **Step 2: Implement metrics**

Implement `compute_confusion_matrix()`, `per_class_metrics()`, and `summarize_area()`.

- [ ] **Step 3: Verify**

Run:

```bash
PYTHONPATH=src python -m unittest tests.test_segmentation_training_metrics
```

Expected: metrics pass without torch.

### Task 5: Preview Writer

**Files:**
- Create: `src/segmentation_training/preview.py`
- Test: `tests/test_segmentation_training_preview.py`

- [ ] **Step 1: Write preview tests**

Use small `numpy` arrays for two channels, ground truth, and predictions. Assert that `write_prediction_preview()` creates a PNG and that `write_contact_sheet()` combines multiple PNGs.

- [ ] **Step 2: Implement preview rendering**

Use Pillow. Render channel panel, label mask, prediction mask, and FP/FN overlay with class colors:

```text
background: black
C2: cyan
C5: red
ignore: gray
false positive: yellow
false negative: magenta
```

- [ ] **Step 3: Verify**

Run:

```bash
PYTHONPATH=src python -m unittest tests.test_segmentation_training_preview
```

Expected: PNG files are created in a temporary directory.

### Task 6: CLI Skeleton

**Files:**
- Create: `src/segmentation_training/cli.py`
- Create: `src/segmentation_training/__main__.py`
- Modify: `src/segmentation_training/__init__.py`

- [ ] **Step 1: Add CLI tests or smoke commands**

Add subcommands:

```text
validate-manifest
bundle
dry-run
train
summarize-runs
```

`train` must refuse to run locally unless `--cloud-confirm` is passed.

- [ ] **Step 2: Implement local-safe commands**

Wire `validate-manifest`, `bundle`, and `dry-run` to non-training code paths. `dry-run` can inspect the first few rows and verify masks/bands exist under the configured root.

- [ ] **Step 3: Verify**

Run:

```bash
PYTHONPATH=src python -m segmentation_training --help
PYTHONPATH=src python -m segmentation_training validate-manifest --contract-dir .agent-team/artifacts/task-ef3a9c94ddb7/implementation/segmentation_contract_v0_1
```

Expected: help prints and manifest validation reports counts.

### Task 7: Cloud Training Stubs And Runbook

**Files:**
- Create: `src/segmentation_training/cloud.py`
- Create: `src/segmentation_training/data.py`
- Create: `src/segmentation_training/sampler.py`
- Create: `src/segmentation_training/losses.py`
- Create: `src/segmentation_training/models/unet.py`
- Create: `src/segmentation_training/models/resunet.py`
- Create: `docs/implementation/2026-05-05-segmentation-training-cloud-runbook.md`

- [ ] **Step 1: Implement guarded torch imports**

All torch imports must happen inside training-specific modules or functions. Local tests that do not train must pass even if torch is unavailable.

- [ ] **Step 2: Implement cloud command contract**

`train` should validate config and data roots, write `run_manifest.json`, and only proceed when `--cloud-confirm` is present.

- [ ] **Step 3: Write runbook**

Document upload, environment creation, bundle validation, E1-E3 training commands, result download, and local summarization.

- [ ] **Step 4: Verify**

Run locally:

```bash
PYTHONPATH=src python -m segmentation_training train --config configs/segmentation_training/e1_binary_c5_unet.yaml
```

Expected: command refuses to train locally and tells the user to run on cloud with `--cloud-confirm`.

### Task 8: Full Local Verification

**Files:**
- All files above.

- [ ] **Step 1: Run all tests**

Run:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

Expected: all tests pass without GPU.

- [ ] **Step 2: Build local manifest-only bundle**

Run:

```bash
PYTHONPATH=src python -m segmentation_training bundle \
  --contract-dir .agent-team/artifacts/task-ef3a9c94ddb7/implementation/segmentation_contract_v0_1 \
  --bundle-dir .agent-team/artifacts/task-afc7f2c25f8f/implementation/training_bundle_v0_1 \
  --required-channel F16 \
  --required-channel F17 \
  --mode manifest-only
```

Expected: bundle metadata and rebased manifest are created; no full local training starts.

- [ ] **Step 3: Update implementation handoff**

When implementation is complete, write `.agent-team/artifacts/task-afc7f2c25f8f/implementation/implementation-implementation-dev-ImplementationHandoff.md` and a Chinese `-zh.md` archive.

## Self-Review

Spec coverage:

- Cloud-first training: covered by Scope Rules, Task 7, and runbook.
- U-Net/ResUNet baseline harness: covered by file structure and Tasks 6-7.
- E0-E4 matrix: covered by configs; E4 remains optional probe.
- Metrics and previews: covered by Tasks 4-5.
- No local full training: enforced by CLI guard and verification steps.

Placeholder scan:

- No `TBD` or open-ended "add tests" placeholders remain.

Type consistency:

- `ModelInputRecord`, manifest parser, bundle builder, configs, metrics, preview writer, and CLI names are consistent across tasks.
