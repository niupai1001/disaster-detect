---
title: Segmentation Model and Data Hypotheses
date: 2026-05-07
tags:
  - segmentation
  - remote-sensing
  - multispectral
  - ablation
  - obsidian
status: active
task: task-6bd957f6e1ed
aliases:
  - U-Net data hypothesis log
---

# Segmentation Model and Data Hypotheses

This note is the updateable project log for architecture, data-contract, channel-stack, and normalization hypotheses. It follows Obsidian-style Markdown so it can be moved into an Obsidian vault later.

Related notes:

- [[2026-05-07-unet-channel-ablation-runbook]]
- [[2026-05-07-segmentation-v4-cloud-smoke-runbook]]
- [[2026-05-07-post-disaster-multispectral-training-summary]]

## Current Position

The weak U-Net result should not be treated as proof that U-Net is inherently unsuitable. In published remote-sensing segmentation work, U-Net-like models often perform much better when labels, input distributions, preprocessing, and target definitions are stable.

For this project, the likely causes are mixed:

- architecture capacity and inductive bias;
- sparse foreground labels;
- label alignment or class-definition uncertainty;
- channel choice and derived-index usefulness;
- per-window normalization changing the physical meaning of spectral values;
- sampler and threshold behavior under rare foreground pixels.

> [!important]
> The next work should run architecture comparison and U-Net channel ablation as separate diagnostic axes. Do not let a better architecture hide a data-contract problem.

## How Multispectral Inputs Enter The Model

The current training path reads each selected raster channel for the same spatial window, normalizes it, then stacks the arrays as channel-first tensors:

```text
single sample: (channels, height, width)
batch:         (batch, channels, height, width)
```

For the current 11-channel route:

```text
F01,F02,F03,F04,F07,F11,F12,NDVI,NBR,NDMI,BRIGHTNESS
```

The U-Net first convolution is built with `input_channels: 11`, so the model receives multispectral data the same way a normal RGB network receives 3-channel imagery, but with 11 channels instead of 3.

## Important Caveat: Normalization

The cloud training path currently applies 2nd-98th percentile normalization per band per read window. This makes each channel numerically stable, but it can weaken absolute physical meaning. For indices such as NDVI, NBR, and NDMI, a model may see relative local contrast instead of a globally consistent index scale.

This is a strong candidate explanation if:

- 7 raw optical/SWIR channels beat 11 channels;
- indices-only behaves strangely;
- validation contact sheets show foreground predictions driven by texture instead of spectral separation.

## Architecture Track

The architecture runway now includes:

- U-Net;
- ResUNet;
- Attention U-Net;
- UNet++;
- DeepLabV3+;
- Transformer-UNet.

Cloud smoke passed for all six architectures on commit `247b1c6`, with `test_split_read: false`. Smoke confirms the runtime path and artifact generation only. It is not model-quality evidence.

## Data And Channel Track

The U-Net-only ablation isolates data/channel behavior by holding the model fixed and changing only the input channel stack.

Planned channel groups:

| Group | Channels | Hypothesis |
|---|---|---|
| RGB optical | `F04,F03,F02` | Simple visible-band signal may outperform a noisy high-dimensional stack. |
| False color | `F07,F04,F03` | NIR may improve separation of vegetation, burn, debris, or water-related context. |
| Base optical/SWIR | `F01,F02,F03,F04,F07,F11,F12` | Raw bands may be better than mixing derived indices under per-window normalization. |
| Indices only | `NDVI,NBR,NDMI,BRIGHTNESS` | Hand-crafted physical indices may carry most usable class signal. |
| Current 11-channel stack | `F01,F02,F03,F04,F07,F11,F12,NDVI,NBR,NDMI,BRIGHTNESS` | Control arm for the current data contract. |

## Decision Log

- 2026-05-07: Architecture smoke accepted; full model benchmark allowed only as validation-only cloud runs.
- 2026-05-07: Added U-Net-only channel ablation configs to test whether poor U-Net behavior is data/channel related.
- 2026-05-07: Deferred normalization-mode changes until channel ablation returns evidence, because changing normalization modifies the loader behavior and creates a second treatment axis.

## Next Evidence To Collect

- U-Net channel ablation smoke summary;
- U-Net channel ablation validation-only full summary;
- per-run `test_split_read: false`;
- validation contact sheets;
- threshold sweep behavior;
- foreground window coverage;
- predicted/label foreground area ratio;
- channel or index groups that fail differently from the current 11-channel stack.

## Interpretation Rules

If a simpler channel group beats the current 11-channel stack, investigate channel noise, index usefulness, normalization, and sample quality before escalating architecture complexity.

If all channel groups fail similarly, investigate labels, alignment, foreground sparsity, sampler coverage, loss weighting, and thresholding before assuming a model-family limitation.

