# Multispectral Architecture Benchmarks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add comparable 11-channel segmentation architecture experiments for ResUNet, Attention U-Net, UNet++, DeepLabV3+, and Transformer-UNet while improving config organization.

**Architecture:** Keep the existing train CLI and data bundle contract stable. Introduce a model registry so `cloud.py` no longer hard-codes every model family, add focused model files under `src/segmentation_training/models/`, and place new benchmark configs under `configs/segmentation_training/architectures/`.

**Tech Stack:** Python, PyTorch, unittest, existing segmentation training CLI.

---

### Task 1: Model Registry

**Files:**
- Create: `src/segmentation_training/models/registry.py`
- Modify: `src/segmentation_training/models/__init__.py`
- Modify: `src/segmentation_training/cloud.py`
- Test: `tests/test_segmentation_training_models.py`

- [ ] Add a `build_model(config, input_channels, output_classes)` registry that supports current `unet` and `resunet`.
- [ ] Replace direct `build_unet` / `build_resunet` branching in `cloud.py` with the registry.
- [ ] Add tests that current model families produce `B,2,H,W` logits from `B,11,H,W`.

### Task 2: Additional Architectures

**Files:**
- Create: `src/segmentation_training/models/attention_unet.py`
- Create: `src/segmentation_training/models/unet_plus_plus.py`
- Create: `src/segmentation_training/models/deeplabv3_plus.py`
- Create: `src/segmentation_training/models/transformer_unet.py`
- Modify: `src/segmentation_training/models/registry.py`
- Test: `tests/test_segmentation_training_models.py`

- [ ] Implement each model with first-class 11-channel support.
- [ ] Keep logits at the same spatial resolution as labels.
- [ ] Add shape tests for all new families.

### Task 3: Architecture Config Directory

**Files:**
- Create: `configs/segmentation_training/architectures/c5_11ch_resunet_rtx4080.yaml`
- Create: `configs/segmentation_training/architectures/c5_11ch_attention_unet_rtx4080.yaml`
- Create: `configs/segmentation_training/architectures/c5_11ch_unetpp_rtx4080.yaml`
- Create: `configs/segmentation_training/architectures/c5_11ch_deeplabv3plus_rtx4080.yaml`
- Create: `configs/segmentation_training/architectures/c5_11ch_transformer_unet_rtx4080.yaml`
- Modify: `src/segmentation_training/config.py`
- Test: `tests/test_segmentation_training_config.py`

- [ ] Add configs using the same 11-channel input, split policy, stable loss/sampler/performance settings, high-threshold sweep, and best checkpoint outputs.
- [ ] Allow model families in config validation.
- [ ] Add config validation tests for every architecture file.

### Task 4: Verification And Publish

**Files:**
- Modify only files touched by Tasks 1-3.

- [ ] Run `PYTHONPATH=src /Users/Lemon/miniconda3/envs/d2l-zh/bin/python -m unittest discover -s tests`.
- [ ] Commit the changes.
- [ ] Push `codex/rtx4080-training-performance`.
