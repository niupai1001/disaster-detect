---
title: 云端 Benchmark 审查
date: 2026-05-07
tags:
  - segmentation
  - cloud-benchmark
  - review
status: reviewed
task: task-6bd957f6e1ed
---

# 云端 Benchmark 审查

本审查对应 `/Users/Lemon/Downloads/cloud_benchmark` 中返回的 validation-only 架构 benchmark。

> [!warning]
> sealed test split 继续封存。本结果只用于 validation 排序和诊断。

## 结论

这次架构 benchmark 没有带来决定性的模型质量突破。DeepLabV3+ 是 final 表现最好的 run，但多个 family 的 best-checkpoint mean IoU 相互接近，说明下一处瓶颈大概率不只是 architecture。

最重要的信号是诊断性的：普通 U-Net 的 best checkpoint mean IoU 曾达到约 `0.372`，但 final mean IoU 掉到约 `0.056`，同时出现严重 foreground overprediction。这更像训练稳定性、threshold/calibration、sampler、label 质量和 11-channel data contract 的综合问题。

## 排名结果

| Rank | Run | Family | Best epoch | Best mean IoU | Final mean IoU | Precision | Recall | Dice | Pred/label area |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `c5_11ch_deeplabv3plus_rtx4080` | deeplabv3_plus | 80 | 0.395640 | 0.395640 | 0.513237 | 0.633260 | 0.566966 | 1.233860 |
| 2 | `c5_11ch_transformer_unet_rtx4080` | transformer_unet | 47 | 0.361243 | 0.307692 | 0.497745 | 0.446241 | 0.470588 | 0.896525 |
| 3 | `c5_11ch_unetpp_rtx4080` | unet_plus_plus | 41 | 0.362645 | 0.301809 | 0.337972 | 0.738264 | 0.463676 | 2.184400 |
| 4 | `c5_11ch_attention_unet_rtx4080` | attention_unet | 69 | 0.353547 | 0.279561 | 0.290216 | 0.883918 | 0.436964 | 3.045720 |
| 5 | `c5_11ch_resunet_rtx4080` | resunet | 1 | 0.338532 | 0.122337 | 0.331231 | 0.162466 | 0.218003 | 0.490491 |
| 6 | `v3_binary_c5_unet_post_optical_indices_rtx4080` | unet | 19 | 0.372230 | 0.056114 | 0.056221 | 0.967037 | 0.106264 | 17.200600 |

## 发现

- DeepLabV3+ 是当前 final 表现最好的架构，final precision/recall/area ratio 最可用。
- Transformer-UNet 更保守，pred/label foreground area ratio 最接近 1，但 final IoU 仍未突破。
- UNet++ 和 Attention U-Net 保留较高 recall，但都有 foreground 过预测，Attention U-Net 更明显。
- ResUNet 第一轮附近达到峰值后快速退化，提示当前训练 treatment 与该结构不匹配或不稳定。
- U-Net 并不是完全学不到：它找到过可用 checkpoint，但最终塌陷为大面积前景过预测。

## 决策

下一步先转向 U-Net channel ablation 和 data-contract 诊断，不继续优先堆架构复杂度。消融应在同一个 U-Net treatment 下比较 RGB、false color、7-channel raw optical/SWIR、indices-only 和当前 11-channel 对照。

## 目录治理

原始 benchmark 目录不适合人工审查：JSON、CSV、PNG、checkpoint 和 log 分散在多个 run 目录中。二阶段开始把默认审查入口改为：

- raw evidence 保留在 run 目录；
- `review/review.md` 作为人工第一入口；
- `review/compact_metrics.csv` 作为对比表；
- `review/figures/` 收集关键 validation contact sheets 和 curves；
- `review/raw_evidence.md` 只索引原始证据。

本地已重新生成的审查入口是 `/Users/Lemon/Downloads/cloud_benchmark/review/review.md`。
