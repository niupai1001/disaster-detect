---
title: Cloud Benchmark Review
date: 2026-05-07
tags:
  - segmentation
  - cloud-benchmark
  - review
status: reviewed
task: task-6bd957f6e1ed
---

# Cloud Benchmark Review

This review covers the validation-only architecture benchmark returned in `/Users/Lemon/Downloads/cloud_benchmark`.

> [!warning]
> The sealed test split remains unused. These results are for validation ranking and diagnosis only.

## Verdict

The architecture benchmark did not produce a decisive model-quality breakthrough. DeepLabV3+ is the best final run, but the best-checkpoint results across several families are close enough that the next bottleneck is likely not architecture alone.

The strongest signal is diagnostic: plain U-Net reached a best checkpoint around `0.372` mean IoU and then collapsed to final mean IoU around `0.056`, with severe foreground overprediction. That pattern points toward training stability, threshold/calibration, sampler behavior, label quality, and the 11-channel data contract.

## Ranked Results

| Rank | Run | Family | Best epoch | Best mean IoU | Final mean IoU | Precision | Recall | Dice | Pred/label area |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `c5_11ch_deeplabv3plus_rtx4080` | deeplabv3_plus | 80 | 0.395640 | 0.395640 | 0.513237 | 0.633260 | 0.566966 | 1.233860 |
| 2 | `c5_11ch_transformer_unet_rtx4080` | transformer_unet | 47 | 0.361243 | 0.307692 | 0.497745 | 0.446241 | 0.470588 | 0.896525 |
| 3 | `c5_11ch_unetpp_rtx4080` | unet_plus_plus | 41 | 0.362645 | 0.301809 | 0.337972 | 0.738264 | 0.463676 | 2.184400 |
| 4 | `c5_11ch_attention_unet_rtx4080` | attention_unet | 69 | 0.353547 | 0.279561 | 0.290216 | 0.883918 | 0.436964 | 3.045720 |
| 5 | `c5_11ch_resunet_rtx4080` | resunet | 1 | 0.338532 | 0.122337 | 0.331231 | 0.162466 | 0.218003 | 0.490491 |
| 6 | `v3_binary_c5_unet_post_optical_indices_rtx4080` | unet | 19 | 0.372230 | 0.056114 | 0.056221 | 0.967037 | 0.106264 | 17.200600 |

## Findings

- DeepLabV3+ is the best final architecture and has the most usable final balance among the tested families.
- Transformer-UNet is more conservative and has the closest predicted/label foreground area ratio, but its final IoU still does not break through.
- UNet++ and Attention U-Net preserve high recall but overpredict foreground, especially Attention U-Net.
- ResUNet peaks immediately and then degrades, which suggests instability or mismatch with the current training treatment.
- U-Net is not simply incapable: it found a useful checkpoint, then collapsed into massive foreground overprediction by the final epoch.

## Decision

Move the next validation step to U-Net channel ablation and data-contract diagnosis before adding more architecture complexity. The ablation should compare RGB, false color, 7-channel raw optical/SWIR, indices-only, and the current 11-channel control under the same U-Net treatment.

## Directory Governance

The raw benchmark folder is too noisy for review. It contains many JSON, CSV, PNG, checkpoint, and log files mixed across run directories. Phase 2 changes the default review surface:

- keep raw evidence in run directories;
- generate `review/review.md` as the first human entrypoint;
- generate `review/compact_metrics.csv` for comparison;
- copy selected validation contact sheets and curves into `review/figures/`;
- index raw evidence from `review/raw_evidence.md`.

The regenerated local review entry is `/Users/Lemon/Downloads/cloud_benchmark/review/review.md`.
