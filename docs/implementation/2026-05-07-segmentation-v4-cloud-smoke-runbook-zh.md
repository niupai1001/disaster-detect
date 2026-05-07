# SegmentationModelV4 云端 Smoke 运行手册

日期：2026-05-07

本手册用于 11 通道架构 benchmark 跑道的受控 cloud smoke/preflight。它不是完整 benchmark，也不能用于选择最终模型。

## 边界

允许：

- 只使用 train 和 validation split。
- 每个架构只运行一次短 smoke。
- 验证模型初始化、首轮 train/validation 行为、显存范围和必需产物生成。
- 收集 validation-only 指标和 contact sheet，交给 QA/review。

禁止：

- 不使用 sealed test split。
- 不从 smoke 直接运行完整 80 epoch 架构 benchmark。
- 不从 smoke 结果宣称模型质量或选择 winner。
- 如果某个架构需要不同 batch size、loss、sampler、学习率或通道集，则不能把该结果当作纯架构比较。

## Smoke 配置

配置目录：

```text
configs/segmentation_training/architecture_smoke/
```

这些配置有意限制计算量：

```text
max_epochs: 1
batch_size: 2
cloud.max_train_samples: 8
cloud.max_validation_samples: 4
training.performance.cache_records: false
```

它们保留 11 通道输入、类别范围、validation-only 选择、阈值扫描输出、best checkpoint 输出、AMP、channels-last 和 sealed-test 纪律。

## Dry Run 预检

训练前对每个 smoke config 运行：

```bash
for config in configs/segmentation_training/architecture_smoke/*.yaml; do
  PYTHONPATH=src python -m segmentation_training dry-run \
    --config "$config" \
    --bundle-dir .agent-team/artifacts/task-2ba5416843b1/implementation/training_bundle_v0_3_indices \
    --max-samples 4
done
```

结果必须包含：

```text
test_split_read: false
status: dry_run_passed
```

## 云端 Smoke 命令

在挂载 11 通道 bundle 的云端机器上运行：

```bash
export PYTHONPATH=src
export BUNDLE_DIR=.agent-team/artifacts/task-2ba5416843b1/implementation/training_bundle_v0_3_indices
export RUN_ROOT=.agent-team/artifacts/task-6bd957f6e1ed/cloud_smoke

for config in configs/segmentation_training/architecture_smoke/*.yaml; do
  name="$(basename "$config" .yaml)"
  python -m segmentation_training train \
    --config "$config" \
    --bundle-dir "$BUNDLE_DIR" \
    --run-dir "$RUN_ROOT/$name" \
    --cloud-confirm
done
```

## 每个 Run 必需证据

每个 run 目录必须包含：

- `run_manifest.json`
- `metrics.json`
- `per_class_metrics.csv`
- `diagnostics/threshold_sweep.csv`
- `diagnostics/validation_prediction_summary.csv`
- `diagnostics/foreground_window_coverage.csv`
- `predictions/validation_contact_sheet.png`
- `training_curves.csv`
- `training_curves.png`
- `checkpoints/best_mean_iou.pt`
- `checkpoints/best_mean_iou.json`

检查 `run_manifest.json` 并确认：

```text
test_split_read: false
cloud.max_train_samples: 8
cloud.max_validation_samples: 4
```

## Smoke 汇总

运行完成后汇总：

```bash
PYTHONPATH=src python -m segmentation_training summarize-runs \
  --runs-dir .agent-team/artifacts/task-6bd957f6e1ed/cloud_smoke \
  --output .agent-team/artifacts/task-6bd957f6e1ed/cloud_smoke/summary.md
```

## 下一道 Gate

cloud smoke 后，把以下证据交回 QA/review：

- smoke summary；
- 显存和运行时记录；
- 每个架构一张 validation contact sheet；
- 所有失败或需要 composite treatment 的偏差。

QA/review 接受 smoke 证据前，完整 benchmark training 继续阻止。
