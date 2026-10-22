# 灾后 v0.2 训练运行手册

日期：2026-05-07

本手册用于构建灾后 optical 模型输入合同、校验、训练 U-Net baseline，并输出可视化反馈。sealed test split 不能用于选择、归一化、可分性分析、视觉 QA、阈值调优或模型选择。

## 通道预设

第一版候选 stack：

```text
F01 F02 F03 F04 F07 F11 F12
```

指数增强 stack：

```text
F01 F02 F03 F04 F07 F11 F12 NDVI NBR NDMI BRIGHTNESS
```

在 `Fxx` 物理波段映射确认前，它只是候选 optical/SWIR stack。下面的 audit 和 validation 会阻断缺通道、灾前或混日期样本；入选通道会先重采样到 mask 网格再进入训练。

场景选择使用 `quality_then_earliest`：先用 RGB-like 通道计算透明的光学云雾代理分数，对灾后同日期且通道齐全的 scene 排序，再用灾后天数打破平局。这不是 Sentinel 官方云掩膜；训练前要检查 `quality_score_detail` 和 model-input preview sheet。

## 构建 v0.2 合同

```bash
PYTHONPATH=src python -m segmentation_contract \
  --project-root . \
  --output-dir .agent-team/artifacts/task-2ba5416843b1/implementation/segmentation_contract_v0_2 \
  --phase-b-build \
  --visual-qa \
  --model-input-previews \
  --post-disaster-audit \
  --post-selection-strategy quality_then_earliest \
  --post-window-days 90 \
  --required-channel F01 \
  --required-channel F02 \
  --required-channel F03 \
  --required-channel F04 \
  --required-channel F07 \
  --required-channel F11 \
  --required-channel F12 \
  --derived-index NDVI \
  --derived-index NBR \
  --derived-index NDMI \
  --derived-index BRIGHTNESS
```

预期产物：

- `model_input_manifest.csv`
- `model_inputs/*.tif`
- `reports/post_disaster_channel_audit.csv`
- `reports/split_leakage_report.csv`
- `reports/post_disaster_model_input_summary.md`
- `reports/wkt_mask_alignment_report.csv`
- `reports/visual_qa_summary.md`
- `previews/contact_sheet.png`
- `previews/model_input_contact_sheet.png`
- `reports/model_input_visual_qa_summary.md`

## 校验 manifest

```bash
PYTHONPATH=src python -m segmentation_training validate-manifest \
  --contract-dir .agent-team/artifacts/task-2ba5416843b1/implementation/segmentation_contract_v0_2 \
  --required-channel F01 \
  --required-channel F02 \
  --required-channel F03 \
  --required-channel F04 \
  --required-channel F07 \
  --required-channel F11 \
  --required-channel F12
```

当前真实构建已通过 7 通道原始 stack 和 11 通道指数增强 stack 的 668 条 train/validation 样本校验。不能静默删通道，也不能退回 `F16/F17`。

## 构建云端 bundle

```bash
PYTHONPATH=src python -m segmentation_training bundle \
  --contract-dir .agent-team/artifacts/task-2ba5416843b1/implementation/segmentation_contract_v0_2 \
  --bundle-dir .agent-team/artifacts/task-2ba5416843b1/implementation/training_bundle_v0_2 \
  --bundle-version 0.2 \
  --class-scope binary_c5 \
  --cloud-data-root /data/training_bundle_v0_2 \
  --required-channel F01 \
  --required-channel F02 \
  --required-channel F03 \
  --required-channel F04 \
  --required-channel F07 \
  --required-channel F11 \
  --required-channel F12
```

构建 C2 bundle 时将 `--class-scope` 改为 `binary_c2`。

构建 V3 指数增强 C5 bundle：

```bash
PYTHONPATH=src python -m segmentation_training bundle \
  --contract-dir .agent-team/artifacts/task-2ba5416843b1/implementation/segmentation_contract_v0_2 \
  --bundle-dir .agent-team/artifacts/task-2ba5416843b1/implementation/training_bundle_v0_3_indices \
  --bundle-version 0.3 \
  --class-scope binary_c5 \
  --cloud-data-root /data/training_bundle_v0_3_indices \
  --required-channel F01 \
  --required-channel F02 \
  --required-channel F03 \
  --required-channel F04 \
  --required-channel F07 \
  --required-channel F11 \
  --required-channel F12 \
  --required-channel NDVI \
  --required-channel NBR \
  --required-channel NDMI \
  --required-channel BRIGHTNESS
```

## Dry Run

```bash
PYTHONPATH=src python -m segmentation_training dry-run \
  --config configs/segmentation_training/v2_binary_c5_unet_post_optical.yaml \
  --bundle-dir .agent-team/artifacts/task-2ba5416843b1/implementation/training_bundle_v0_2 \
  --max-samples 8
```

V3 指数 dry run：

```bash
PYTHONPATH=src python -m segmentation_training dry-run \
  --config configs/segmentation_training/v3_binary_c5_unet_post_optical_indices_smoke.yaml \
  --bundle-dir .agent-team/artifacts/task-2ba5416843b1/implementation/training_bundle_v0_3_indices \
  --max-samples 8
```

## 训练

完整训练应在安装了 `torch`、`rasterio`、`shapely`、`numpy`、`Pillow`、`matplotlib` 和 `PyYAML` 的 Python 环境运行。bundle 已包含 masks、对齐后的 `model_inputs`、manifests、reports 和 configs。

当前已验证的本机训练环境是：

```bash
/Users/Lemon/miniconda3/envs/d2l-zh/bin/python
```

如果重建环境，先安装 geospatial runtime 依赖，再跑合同或训练验证：

```bash
/Users/Lemon/miniconda3/envs/d2l-zh/bin/python -m pip install rasterio==1.3.11 shapely==2.0.7
```

Smoke run：

```bash
PYTHONPATH=src python -m segmentation_training train \
  --config /data/training_bundle_v0_2/configs/v2_binary_c5_unet_post_optical_smoke.yaml \
  --bundle-dir /data/training_bundle_v0_2 \
  --run-dir /runs/post_disaster_v0_2/V2_binary_c5_unet_post_optical_smoke \
  --cloud-confirm
```

V3 指数 smoke run：

```bash
PYTHONPATH=src python -m segmentation_training train \
  --config /data/training_bundle_v0_3_indices/configs/v3_binary_c5_unet_post_optical_indices_smoke.yaml \
  --bundle-dir /data/training_bundle_v0_3_indices \
  --run-dir /runs/post_disaster_v0_3/V3_binary_c5_unet_post_optical_indices_smoke \
  --cloud-confirm
```

已验证的本机 V3 指数 smoke run：

```bash
PYTHONPATH=src /Users/Lemon/miniconda3/envs/d2l-zh/bin/python -m segmentation_training train \
  --config .agent-team/artifacts/task-2ba5416843b1/implementation/training_bundle_v0_3_indices/configs/v3_binary_c5_unet_post_optical_indices_smoke.yaml \
  --bundle-dir .agent-team/artifacts/task-2ba5416843b1/implementation/training_bundle_v0_3_indices \
  --run-dir .agent-team/artifacts/task-2ba5416843b1/implementation/runs/v3_binary_c5_indices_smoke \
  --cloud-confirm
```

2026-05-07 验证输出：

- `status`：`completed_cloud_training`
- `train_records`：301
- `validation_records`：78
- `input_channels`：`F01;F02;F03;F04;F07;F11;F12;NDVI;NBR;NDMI;BRIGHTNESS`
- `mean_iou`：`0.19590715476263967`
- `foreground_recall`：`1.0`
- `train_loss_last`：`1.2567567825317383`
- `diagnostics/artifact_inventory.json`：预期产物无缺失

完整 run：

```bash
PYTHONPATH=src python -m segmentation_training train \
  --config /data/training_bundle_v0_2/configs/v2_binary_c5_unet_post_optical.yaml \
  --bundle-dir /data/training_bundle_v0_2 \
  --run-dir /runs/post_disaster_v0_2/V2_binary_c5_unet_post_optical \
  --cloud-confirm
```

V3 指数完整 run：

```bash
PYTHONPATH=src python -m segmentation_training train \
  --config /data/training_bundle_v0_3_indices/configs/v3_binary_c5_unet_post_optical_indices.yaml \
  --bundle-dir /data/training_bundle_v0_3_indices \
  --run-dir /runs/post_disaster_v0_3/V3_binary_c5_unet_post_optical_indices \
  --cloud-confirm
```

C2 训练替换为 `v2_binary_c2_unet_post_optical` 配置和 run 目录。

## 可视化反馈产物

每个完成的 run 会写出：

- `metrics.json`
- `per_class_metrics.csv`
- `confusion_matrix.csv`
- `diagnostics/validation_prediction_summary.csv`
- `diagnostics/threshold_sweep.csv`
- `diagnostics/foreground_probability_summary.csv`
- `diagnostics/foreground_window_coverage.csv`
- `predictions/validation_contact_sheet.png`
- `predictions/failures_contact_sheet.png`
- `predictions/per_sample/*.png`
- `training_curves.png`
- `training_curves.csv`

旧 run 如果缺曲线，可单独生成：

```bash
PYTHONPATH=src python -m segmentation_training plot-run \
  --run-dir /runs/post_disaster_v0_2/V2_binary_c5_unet_post_optical
```
