# SegmentationTrainingBaseline v0.1 云端运行手册

日期：2026-05-05

本文档保持 `SegmentationTrainingBaseline v0.1` 的云端优先约束。本机只允许做 manifest 校验、bundle 构建、单元测试、synthetic tests 和 tiny smoke。完整的 E1-E3 模型训练必须在云服务器运行。

## 输入

本机标准数据契约：

```bash
.agent-team/artifacts/task-ef3a9c94ddb7/implementation/segmentation_contract_v0_1
```

本机 bundle 输出位置：

```bash
.agent-team/artifacts/task-afc7f2c25f8f/implementation/training_bundle_v0_1
```

v0.1 的通道契约是 `F16;F17`。test split 是封存测试集，不能用于模型路线选择。

## 本机预检

在项目根目录运行：

```bash
PYTHONPATH=src python -m unittest discover -s tests

PYTHONPATH=src python -m segmentation_training validate-manifest \
  --contract-dir .agent-team/artifacts/task-ef3a9c94ddb7/implementation/segmentation_contract_v0_1

PYTHONPATH=src python -m segmentation_training bundle \
  --contract-dir .agent-team/artifacts/task-ef3a9c94ddb7/implementation/segmentation_contract_v0_1 \
  --bundle-dir .agent-team/artifacts/task-afc7f2c25f8f/implementation/training_bundle_v0_1 \
  --required-channel F16 \
  --required-channel F17 \
  --mode manifest-only

PYTHONPATH=src python -m segmentation_training dry-run \
  --config configs/segmentation_training/e1_binary_c5_unet.yaml \
  --contract-dir .agent-team/artifacts/task-ef3a9c94ddb7/implementation/segmentation_contract_v0_1 \
  --max-samples 4
```

不要在本机运行 `train --cloud-confirm`。

## 上传

把仓库源码和生成的 bundle 上传到云服务器。bundle 期望云端数据根目录为：

```bash
/data/training_bundle_v0_1
```

bundle 包含 masks、metadata、source reports、configs 和 `manifests/cloud_model_input_manifest.csv`。大型 raster 文件不会被复制进 bundle；需要挂载或复制项目 raster 目录，使 `/data/training_bundle_v0_1/database/...` 下的路径可以解析。

## 云端环境

使用 Python 3.11。根据云镜像和 CUDA driver 安装兼容依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pillow pyyaml rasterio tqdm
# PyTorch 根据服务器 CUDA 版本从官方 wheel selector 安装。
```

在确认云镜像之前，不要在仓库里固定 CUDA wheel。

## 云端校验

云端挂载数据后运行：

```bash
export PYTHONPATH=src

python -m segmentation_training validate-manifest \
  --contract-dir /data/training_bundle_v0_1

python -m segmentation_training dry-run \
  --config /data/training_bundle_v0_1/configs/e1_binary_c5_unet.yaml \
  --bundle-dir /data/training_bundle_v0_1 \
  --max-samples 4
```

如果失败，先修正挂载路径或重新生成 bundle，再开始训练。

## E1-E3 云端运行

每个实验使用独立 run 目录：

```bash
python -m segmentation_training train \
  --config /data/training_bundle_v0_1/configs/e1_binary_c5_unet.yaml \
  --bundle-dir /data/training_bundle_v0_1 \
  --run-dir /runs/segmentation_training/e1_binary_c5_unet \
  --cloud-confirm

python -m segmentation_training train \
  --config /data/training_bundle_v0_1/configs/e2_binary_c2_unet.yaml \
  --bundle-dir /data/training_bundle_v0_1 \
  --run-dir /runs/segmentation_training/e2_binary_c2_unet \
  --cloud-confirm

python -m segmentation_training train \
  --config /data/training_bundle_v0_1/configs/e3_multiclass_c2_c5_unet.yaml \
  --bundle-dir /data/training_bundle_v0_1 \
  --run-dir /runs/segmentation_training/e3_multiclass_c2_c5_unet \
  --cloud-confirm
```

每个返回的 run 目录应包含 `resolved_config.yaml`、`run_manifest.json`、指标文件、checkpoint、日志和预测预览。

## 返回与汇总

把云端 run 目录下载到：

```bash
.agent-team/artifacts/task-afc7f2c25f8f/cloud_runs
```

然后在本机汇总：

```bash
PYTHONPATH=src python -m segmentation_training summarize-runs \
  --runs-dir .agent-team/artifacts/task-afc7f2c25f8f/cloud_runs \
  --output .agent-team/artifacts/task-afc7f2c25f8f/implementation/run_comparison_summary.md
```

路线选择只能使用 validation 证据。不能用封存的 test split 在 U-Net、ResUNet、DeepLabV3+、SegFormer 或 foundation-model 路线之间做选择。

