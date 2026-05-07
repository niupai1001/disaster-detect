---
title: U-Net 通道消融运行手册
date: 2026-05-07
tags:
  - segmentation
  - ablation
  - unet
  - multispectral
status: ready-for-cloud-smoke
task: task-6bd957f6e1ed
---

# U-Net 通道消融运行手册

本手册用于验证 U-Net 表现弱是否来自数据契约、通道组合或派生指数混合，而不是只来自模型架构。

本实验固定模型 family 为 U-Net，只改变 `input_channels`。

> [!warning]
> 不允许使用 sealed test split。这些运行只用于 train/validation 诊断。

## 消融矩阵

| Config family | Channels | 目的 |
|---|---|---|
| RGB optical | `F04,F03,F02` | 检查简单 RGB-like 灾后信号是否强于当前高维通道栈。 |
| False color | `F07,F04,F03` | 检查 NIR + red/green 是否更能分离植被、烧毁或泥石流相关变化。 |
| Base optical/SWIR | `F01,F02,F03,F04,F07,F11,F12` | 检查原始对齐反射率波段是否优于派生指数混合。 |
| Indices only | `NDVI,NBR,NDMI,BRIGHTNESS` | 检查手工指数通道是否承载主要可用信号。 |
| Current 11-channel stack | `F01,F02,F03,F04,F07,F11,F12,NDVI,NBR,NDMI,BRIGHTNESS` | 对照组，匹配当前 U-Net 数据契约。 |

## 先跑 Smoke

在云端执行：

```bash
cd /root/autodl-tmp/New_project
git pull --ff-only origin codex/rtx4080-training-performance

export PYTHONPATH=src
export BUNDLE_DIR=.agent-team/artifacts/task-2ba5416843b1/implementation/training_bundle_v0_3_indices
export RUN_ROOT=.agent-team/artifacts/task-6bd957f6e1ed/cloud_unet_channel_ablation_smoke

for config in configs/segmentation_training/unet_channel_ablation/*_smoke.yaml; do
  name="$(basename "$config" .yaml)"
  python -m segmentation_training train \
    --config "$config" \
    --bundle-dir "$BUNDLE_DIR" \
    --run-dir "$RUN_ROOT/$name" \
    --cloud-confirm
done

python -m segmentation_training summarize-runs \
  --runs-dir "$RUN_ROOT" \
  --output "$RUN_ROOT/summary.md"
```

该汇总命令也会写出 `$RUN_ROOT/review/review.md`、`$RUN_ROOT/review/compact_metrics.csv`、`$RUN_ROOT/review/raw_evidence.md`，并把关键图片放到 `$RUN_ROOT/review/figures/`。

Smoke 验收要求：

- 每个 run 都报告 `completed_cloud_training`；
- 每个 `run_manifest.json` 都报告 `test_split_read: false`；
- metrics、diagnostics、curves、contact sheets 和 checkpoints 都存在；
- 没有任何通道组需要单独改变 sampler、loss 或 learning rate。

## 完整 Validation-Only 消融

Smoke 通过后执行：

```bash
export PYTHONPATH=src
export BUNDLE_DIR=.agent-team/artifacts/task-2ba5416843b1/implementation/training_bundle_v0_3_indices
export RUN_ROOT=.agent-team/artifacts/task-6bd957f6e1ed/cloud_unet_channel_ablation

configs=(
  configs/segmentation_training/unet_channel_ablation/c5_unet_rgb_optical_rtx4080.yaml
  configs/segmentation_training/unet_channel_ablation/c5_unet_false_color_rtx4080.yaml
  configs/segmentation_training/unet_channel_ablation/c5_unet_post_optical_7ch_rtx4080.yaml
  configs/segmentation_training/unet_channel_ablation/c5_unet_indices_only_rtx4080.yaml
  configs/segmentation_training/unet_channel_ablation/c5_unet_11ch_indices_rtx4080.yaml
)

for config in "${configs[@]}"; do
  name="$(basename "$config" .yaml)"
  python -m segmentation_training train \
    --config "$config" \
    --bundle-dir "$BUNDLE_DIR" \
    --run-dir "$RUN_ROOT/$name" \
    --cloud-confirm
done

python -m segmentation_training summarize-runs \
  --runs-dir "$RUN_ROOT" \
  --output "$RUN_ROOT/summary.md"
```

人工审查从 `$RUN_ROOT/review/review.md` 开始。只有追踪具体指标或运行时问题时再打开 raw JSON。

## 解释规则

- 如果 RGB 或 false-color 优于 11-channel，优先怀疑通道噪声、归一化行为或派生指数不匹配。
- 如果 7-channel 优于 11-channel，优先怀疑派生指数或派生指数的 per-window normalization。
- 如果 indices-only 优于 raw bands，说明 U-Net 可能很难从稀疏标签中自己学习物理比值关系。
- 如果所有通道组都差不多失败，优先检查标签对齐、mask 稀疏性、类别定义、sampler 覆盖和 loss/threshold 行为。
- 不要只从这个消融中选择最终模型；它只解释 U-Net 的数据契约行为。

## 需要返回的证据

归档并返回：

- `summary.md`；
- `review/review.md`；
- `review/compact_metrics.csv`；
- 每个 `run_manifest.json`；
- 每个 `metrics.json`；
- 每个 `per_class_metrics.csv`；
- `diagnostics/threshold_sweep.csv`；
- `diagnostics/foreground_window_coverage.csv`；
- `predictions/validation_contact_sheet.png`；
- `training_curves.png`；
- best checkpoint metadata。
