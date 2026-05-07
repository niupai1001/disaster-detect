---
title: V6 11 通道 U-Net 增强诊断运行手册
date: 2026-05-07
tags:
  - segmentation
  - unet
  - augmentation
  - diagnostics
status: ready-for-cloud-smoke
task: task-6bd957f6e1ed
---

# V6 11 通道 U-Net 增强诊断运行手册

本手册固定 11 通道灾后输入契约，测试更稳的训练协议是否能改善 U-Net 行为。

这里不把单期灾后影像当作主要失败原因。当前工作假设调整为：

- 当前 U-Net 对 256x256 validation windows 可能偏浅；
- 单纯通道选择没有解决瓶颈；
- 在继续大规模换架构前，应先加入训练增强、threshold calibration、全量 validation previews 和 worst false-positive 审查。

> [!warning]
> 不使用 sealed test split。本轮仍然只用于 train/validation 诊断。

## 新 Workspace 结构

新的云端输出使用 `workspace/`：

```text
workspace/
  runs/
    segmentation/
      v6_11ch_unet_aug_diagnostics/
        smoke/
        full/
        channel_contribution/
        review/
  bundles/
  archives/
```

旧的 `.agent-team/artifacts/...` 继续作为历史证据读取，但新的 run outputs 放到 `workspace/runs/`。

## 本轮变化

- 输入继续使用 11 通道：`F01,F02,F03,F04,F07,F11,F12,NDVI,NBR,NDMI,BRIGHTNESS`。
- 模型 family 继续固定为 U-Net。
- 训练加入安全空间增强：水平翻转、垂直翻转、90 度旋转。
- full validation 导出全部 validation previews，不再只导出前 12 张。
- 单独导出 worst false positives，用于 label/mask alignment 审查。
- 训练后可运行 `channel-contribution`，用 zero-occlusion 评估单通道和通道组贡献。

## Smoke

在云端执行：

```bash
cd /root/autodl-tmp/New_project
git pull --ff-only origin codex/rtx4080-training-performance

export PYTHONPATH=src
export BUNDLE_DIR=.agent-team/artifacts/task-2ba5416843b1/implementation/training_bundle_v0_3_indices
export RUN_ROOT=workspace/runs/segmentation/v6_11ch_unet_aug_diagnostics/smoke

python -m segmentation_training train \
  --config configs/segmentation_training/v6_11ch_unet_aug_diagnostics/c5_11ch_unet_aug_smoke.yaml \
  --bundle-dir "$BUNDLE_DIR" \
  --run-dir "$RUN_ROOT/c5_11ch_unet_aug_smoke" \
  --cloud-confirm

python -m segmentation_training summarize-runs \
  --runs-dir "$RUN_ROOT" \
  --output "$RUN_ROOT/summary.md"
```

Smoke 验收：

- `run_manifest.json` 报告 `test_split_read: false`；
- `metrics.json`、`per_class_metrics.csv`、`training_curves.*` 和 checkpoint metadata 存在；
- `predictions/per_sample/` 包含全部 smoke validation previews；
- `predictions/worst_false_positives/` 和 `diagnostics/worst_false_positives.csv` 存在。

## Full Run

Smoke 通过后执行：

```bash
export PYTHONPATH=src
export BUNDLE_DIR=.agent-team/artifacts/task-2ba5416843b1/implementation/training_bundle_v0_3_indices
export RUN_ROOT=workspace/runs/segmentation/v6_11ch_unet_aug_diagnostics/full

python -m segmentation_training train \
  --config configs/segmentation_training/v6_11ch_unet_aug_diagnostics/c5_11ch_unet_aug_rtx4080.yaml \
  --bundle-dir "$BUNDLE_DIR" \
  --run-dir "$RUN_ROOT/c5_11ch_unet_aug_rtx4080" \
  --cloud-confirm

python -m segmentation_training summarize-runs \
  --runs-dir "$RUN_ROOT" \
  --output "$RUN_ROOT/summary.md"
```

人工审查从这里开始：

```text
workspace/runs/segmentation/v6_11ch_unet_aug_diagnostics/full/review/review.md
```

## 通道贡献评估

Full run 完成后，用 best checkpoint 评估 11 通道贡献：

```bash
export RUN_DIR=workspace/runs/segmentation/v6_11ch_unet_aug_diagnostics/full/c5_11ch_unet_aug_rtx4080
export CONTRIBUTION_DIR=workspace/runs/segmentation/v6_11ch_unet_aug_diagnostics/channel_contribution

python -m segmentation_training channel-contribution \
  --config configs/segmentation_training/v6_11ch_unet_aug_diagnostics/c5_11ch_unet_aug_rtx4080.yaml \
  --bundle-dir "$BUNDLE_DIR" \
  --checkpoint "$RUN_DIR/checkpoints/best_mean_iou.pt" \
  --output-dir "$CONTRIBUTION_DIR" \
  --cloud-confirm
```

会写出：

```text
channel_contribution.csv
channel_contribution_manifest.json
```

解释方式：zero 掉某个通道或通道组后，`mean_iou_drop` 越大，说明训练好的 11 通道模型越依赖该信号。

## 需要返回的证据

返回或下载：

- `full/review/review.md`；
- `full/review/compact_metrics.csv`；
- `full/c5_11ch_unet_aug_rtx4080/diagnostics/worst_false_positives.csv`；
- `full/c5_11ch_unet_aug_rtx4080/predictions/worst_false_positives/`；
- `full/c5_11ch_unet_aug_rtx4080/predictions/per_sample/`；
- `channel_contribution/channel_contribution.csv`；
- training curves 和 best checkpoint metadata。
