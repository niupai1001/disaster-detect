---
title: Segmentation Model Data Hypotheses
date: 2026-05-07
tags:
  - segmentation
  - multispectral
  - ablation
  - obsidian
status: active
task: task-6bd957f6e1ed
---

# Segmentation Model Data Hypotheses

Canonical note:

![[docs/research/memos/2026-05-07-segmentation-model-data-hypotheses]]

## Short Version

Poor U-Net performance should not be attributed to architecture alone. The current evidence also points toward data contract, channel composition, sparse foreground labels, per-window normalization, and sampler/threshold behavior.

## Active Diagnostic Split

- Architecture benchmark: compare model families with the same 11-channel contract.
- U-Net channel ablation: keep U-Net fixed and compare channel stacks.

