---
title: 分割模型与数据假设
date: 2026-05-07
tags:
  - segmentation
  - multispectral
  - ablation
  - obsidian
status: active
task: task-6bd957f6e1ed
---

# 分割模型与数据假设

Canonical note：

![[docs/research/memos/2026-05-07-segmentation-model-data-hypotheses-zh]]

## 简版

U-Net 表现弱不能只归因于 architecture。当前证据也指向 data contract、channel composition、sparse foreground labels、per-window normalization、sampler/threshold behavior。

## 当前诊断拆分

- Architecture benchmark：在同一个 11-channel contract 下比较 model families。
- U-Net channel ablation：固定 U-Net，只比较 channel stacks。

