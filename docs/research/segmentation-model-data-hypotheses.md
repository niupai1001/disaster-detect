---
title: Segmentation Model Data Hypotheses
date: 2026-05-07
tags:
  - segmentation
  - remote-sensing
  - multispectral
  - hypotheses
status: active
task: task-6bd957f6e1ed
---

# Segmentation Model Data Hypotheses

## Current Position

Weak U-Net results are not proof that U-Net is inherently unsuitable. Likely causes remain mixed:

- architecture capacity and inductive bias;
- sparse foreground labels;
- label alignment or class-definition uncertainty;
- channel choice and derived-index usefulness;
- per-window normalization changing physical index meaning;
- sampler, loss, and threshold behavior under rare foreground pixels.

The next diagnostics should avoid changing too many axes at once.

## Current 11-Channel Stack

```text
F01,F02,F03,F04,F07,F11,F12,NDVI,NBR,NDMI,BRIGHTNESS
```

Training reads selected raster channels for the same spatial window, normalizes them, and stacks arrays as channel-first tensors:

```text
single sample: (channels, height, width)
batch:         (batch, channels, height, width)
```

## Normalization Caveat

The cloud training path applies 2nd-98th percentile normalization per band per read window. This stabilizes values but may weaken absolute physical meaning, especially for NDVI, NBR, NDMI, and brightness.

Watch for this pattern:

- simpler raw bands beat the 11-channel stack;
- indices-only behaves strangely;
- predictions look texture-driven rather than spectrally meaningful.

## Diagnostic Tracks

| Track | Purpose | Status |
|---|---|---|
| Architecture benchmark | Compare U-Net, ResUNet, Attention U-Net, UNet++, DeepLabV3+, Transformer-UNet on the same 11-channel contract. | Validation-only benchmark returned; DeepLabV3+ best final run, but not decisive. |
| U-Net channel ablation | Hold U-Net fixed and compare RGB, false color, 7-channel optical/SWIR, indices-only, and 11-channel control. | Useful if resumed, but current focus moved to V6 11-channel augmentation diagnostics. |
| V6 11-channel augmentation diagnostics | Keep 11 channels fixed; add safe spatial augmentation, full validation previews, worst false positives, and channel contribution. | Full validation package returned; failure shifted to foreground overprediction. |

## Latest V6 Evidence

Returned package: `/Users/Lemon/Downloads/full`.

Headline:

- best checkpoint around epoch 8 reached mean IoU about `0.2521` with foreground recall about `0.8684`;
- final epoch dropped to mean IoU about `0.1114`;
- final foreground precision was about `0.1144`, with recall about `0.8086`;
- predicted foreground area was about `7.07x` the labeled foreground area;
- threshold `0.65` improved IoU to about `0.1426` and reduced overprediction to about `3.79x`, but did not make the mask usable.

This changes the dominant diagnosis from "foreground absence" to "foreground expansion." The next experiment should explain why broad high-recall masks are being rewarded or selected.

## Combined Evidence From Recent Runs

Architecture benchmark:

- DeepLabV3+ was the strongest final validation run, with mean IoU about `0.3956`, precision about `0.5132`, recall about `0.6333`, and pred/label area about `1.23x`.
- Transformer-UNet was more conservative, with final mean IoU about `0.3077` and pred/label area about `0.90x`.
- Earlier U-Net reached a useful best checkpoint around mean IoU `0.3722`, then collapsed to final mean IoU about `0.0561` and pred/label area about `17.20x`.

U-Net channel ablation:

- RGB optical was the most stable final U-Net arm, final mean IoU about `0.2874` and pred/label area about `1.94x`.
- 11-channel indices had a competitive best checkpoint about `0.3472`, but lower final mean IoU around `0.2055`.
- 7-channel raw, false color, and indices-only arms showed strong overprediction, with pred/label area ratios around `9.07x`, `16.41x`, and `22.36x`.

Interpretation:

- Architecture matters, but architecture alone is not a full explanation.
- Channel choice and normalization matter, but the current failure also involves calibration, sampling, label/window geometry, and training dynamics.

## Interpretation Rules

- If augmentation improves U-Net stability, focus on training protocol, calibration, and validation evidence before another architecture sweep.
- If worst false positives reveal mask or alignment errors, prioritize data QA over model changes.
- If channel contribution shows weak or harmful index dependence, revisit channel construction and normalization.
- If validation thresholding improves metrics but leaves area ratio high, treat calibration as necessary but insufficient.
- If worst false positives cluster on mountain, bare-soil, shadow, burn-like, or high-contrast terrain, add hard-negative mining and label-window geometry review before model-family changes.
- If all controlled diagnostics fail similarly, investigate labels, foreground sparsity, sampler coverage, loss weighting, and thresholding before assuming a model-family limitation.

## Decision Log

- 2026-05-07: Architecture smoke accepted; full model benchmark allowed only as validation-only cloud runs.
- 2026-05-07: Architecture benchmark did not produce a decisive breakthrough; DeepLabV3+ had the best final validation result.
- 2026-05-07: U-Net channel ablation was defined to separate channel effects from model-family effects.
- 2026-05-07: V6 diagnostic became the active runbook: fixed 11-channel input plus safer training protocol, complete validation previews, worst false-positive review, and channel-contribution analysis.
- 2026-05-07: V6 full validation evidence showed foreground overprediction rather than all-background collapse; next direction should prioritize threshold/calibration, label/window geometry, hard negatives, sampler/loss behavior, and channel contribution.
- 2026-05-07: Project Markdown was consolidated into living notes; future Chinese handoffs go to Obsidian instead of duplicated project files.
