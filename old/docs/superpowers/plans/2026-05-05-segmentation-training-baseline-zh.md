# SegmentationTrainingBaseline v0.1 实现计划

> **给 agentic workers：** REQUIRED SUB-SKILL：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 按任务执行本计划。步骤使用 checkbox (`- [ ]`) 语法跟踪。

**目标：** 构建云端优先的 C2/C5 灾害语义分割 baseline 训练 harness，且不在本机执行完整训练。

**架构：** 本机代码负责 manifest 校验、云端 bundle 构建、CPU-only tests、tiny smoke checks 和云端结果汇总。云端使用同一套 configs 执行 U-Net/ResUNet baseline，并返回 metrics、checkpoints、logs 和 prediction previews。

**技术栈：** Python 3.11、标准库 config/CSV、rasterio 读取 GeoTIFF、numpy/Pillow 处理指标与预览；PyTorch 只在云端训练模块中导入；PyYAML 可选用于 configs。

---

## 范围规则

- 不在本机运行完整模型训练。
- 本机只允许运行单元测试、synthetic tests、manifest validation、bundle validation 和 tiny smoke checks。
- 云端训练命令需要实现和文档化，但不在本机执行。
- 生成的 run manifest 必须保留 `task-afc7f2c25f8f` 与 `task-ef3a9c94ddb7` 引用。
- 保持 test split 封存；模型选择命令默认不得读取 test rows。

## 文件结构

创建：

```text
src/segmentation_training/__init__.py
src/segmentation_training/__main__.py
src/segmentation_training/cli.py
src/segmentation_training/manifest.py
src/segmentation_training/bundle.py
src/segmentation_training/config.py
src/segmentation_training/metrics.py
src/segmentation_training/preview.py
src/segmentation_training/cloud.py
src/segmentation_training/data.py
src/segmentation_training/sampler.py
src/segmentation_training/losses.py
src/segmentation_training/models/__init__.py
src/segmentation_training/models/unet.py
src/segmentation_training/models/resunet.py
configs/segmentation_training/e0_trivial.yaml
configs/segmentation_training/e1_binary_c5_unet.yaml
configs/segmentation_training/e2_binary_c2_unet.yaml
configs/segmentation_training/e3_multiclass_c2_c5_unet.yaml
docs/implementation/2026-05-05-segmentation-training-cloud-runbook.md
tests/test_segmentation_training_manifest.py
tests/test_segmentation_training_bundle.py
tests/test_segmentation_training_config.py
tests/test_segmentation_training_metrics.py
tests/test_segmentation_training_preview.py
```

修改：

```text
src/segmentation_training/__init__.py
```

除非缺少数据契约字段阻塞实现，否则不要改动已有 `src/segmentation_contract/` 文件。

### Task 1: Manifest Reader And Class Filtering

**Files:**
- Create: `src/segmentation_training/manifest.py`
- Test: `tests/test_segmentation_training_manifest.py`

- [ ] **Step 1: 编写 manifest parsing 测试**

创建三行 CSV，字段包含 `sample_id`、`split`、`class_id`、`class_name`、`mask_path`、`input_channels` 和 `input_band_paths`。断言 `load_model_input_manifest()` 返回 typed records、保留 split、解析 `F16:/path/a.tif;F17:/path/b.tif`，并拒绝缺失必需通道。

- [ ] **Step 2: 实现 manifest dataclass 和 parser**

实现 `ModelInputRecord`、`parse_band_paths()`、`load_model_input_manifest()` 和 `filter_records()`。Class scopes 必须支持 `binary_c2`、`binary_c5` 和 `multiclass_c2_c5`。

- [ ] **Step 3: 验证**

运行：

```bash
PYTHONPATH=src python -m unittest tests.test_segmentation_training_manifest
```

预期：测试通过且不导入 torch。

### Task 2: Cloud Bundle Builder

**Files:**
- Create: `src/segmentation_training/bundle.py`
- Test: `tests/test_segmentation_training_bundle.py`

- [ ] **Step 1: 编写 manifest-only bundle 测试**

使用临时 masks 和 fake raster paths。断言 `build_training_bundle(..., mode="manifest-only")` 创建：

```text
metadata/class_map.json
metadata/channels.json
metadata/bundle_manifest.json
manifests/cloud_model_input_manifest.csv
README.md
```

- [ ] **Step 2: 实现 bundle 创建**

实现 metadata 复制、路径重写到 `cloud_data_root`，并为 bundle 内文件写入 SHA-256。

- [ ] **Step 3: 验证**

运行：

```bash
PYTHONPATH=src python -m unittest tests.test_segmentation_training_bundle
```

预期：测试通过且不需要复制大型 raster。

### Task 3: Config Validation

**Files:**
- Create: `src/segmentation_training/config.py`
- Create: `configs/segmentation_training/e0_trivial.yaml`
- Create: `configs/segmentation_training/e1_binary_c5_unet.yaml`
- Create: `configs/segmentation_training/e2_binary_c2_unet.yaml`
- Create: `configs/segmentation_training/e3_multiclass_c2_c5_unet.yaml`
- Test: `tests/test_segmentation_training_config.py`

- [ ] **Step 1: 编写 config 测试**

断言四个 configs 都能加载，且包含 `task_id`、`experiment_id`、`class_scope`、`input_channels`、`split_policy`、`model`、`metrics` 和 `cloud`；若 `allow_test_split_for_selection` 为 true，则拒绝。

- [ ] **Step 2: 实现 config loader**

安装 PyYAML 时使用 `yaml.safe_load`；否则通过清晰错误提示要求安装 PyYAML。Validation 必须确定、显式。

- [ ] **Step 3: 验证**

运行：

```bash
PYTHONPATH=src python -m unittest tests.test_segmentation_training_config
```

预期：所有 configs 通过校验，test split 误用被拒绝。

### Task 4: Metrics Module

**Files:**
- Create: `src/segmentation_training/metrics.py`
- Test: `tests/test_segmentation_training_metrics.py`

- [ ] **Step 1: 编写 metric 测试**

创建小型 `numpy` arrays，包含 labels 与 predictions，类别为 `0`、`1`、`2` 和 `255`。断言 ignore pixels 不影响统计，并且 per-class IoU、Dice/F1、precision、recall 和 confusion matrix 正确。

- [ ] **Step 2: 实现 metrics**

实现 `compute_confusion_matrix()`、`per_class_metrics()` 和 `summarize_area()`。

- [ ] **Step 3: 验证**

运行：

```bash
PYTHONPATH=src python -m unittest tests.test_segmentation_training_metrics
```

预期：metrics 测试不需要 torch。

### Task 5: Preview Writer

**Files:**
- Create: `src/segmentation_training/preview.py`
- Test: `tests/test_segmentation_training_preview.py`

- [ ] **Step 1: 编写 preview 测试**

使用小型 `numpy` arrays 表示双通道、ground truth 和 predictions。断言 `write_prediction_preview()` 创建 PNG，`write_contact_sheet()` 能合并多个 PNG。

- [ ] **Step 2: 实现 preview rendering**

使用 Pillow。渲染 channel panel、label mask、prediction mask 和 FP/FN overlay，颜色：

```text
background: black
C2: cyan
C5: red
ignore: gray
false positive: yellow
false negative: magenta
```

- [ ] **Step 3: 验证**

运行：

```bash
PYTHONPATH=src python -m unittest tests.test_segmentation_training_preview
```

预期：临时目录中生成 PNG 文件。

### Task 6: CLI Skeleton

**Files:**
- Create: `src/segmentation_training/cli.py`
- Create: `src/segmentation_training/__main__.py`
- Modify: `src/segmentation_training/__init__.py`

- [ ] **Step 1: 添加 CLI tests 或 smoke commands**

添加子命令：

```text
validate-manifest
bundle
dry-run
train
summarize-runs
```

`train` 在没有 `--cloud-confirm` 时必须拒绝运行。

- [ ] **Step 2: 实现本机安全命令**

将 `validate-manifest`、`bundle` 和 `dry-run` 接到非训练路径。`dry-run` 可检查前几行，并验证 masks/bands 在配置 root 下存在。

- [ ] **Step 3: 验证**

运行：

```bash
PYTHONPATH=src python -m segmentation_training --help
PYTHONPATH=src python -m segmentation_training validate-manifest --contract-dir .agent-team/artifacts/task-ef3a9c94ddb7/implementation/segmentation_contract_v0_1
```

预期：help 正常输出，manifest validation 报告计数。

### Task 7: Cloud Training Stubs And Runbook

**Files:**
- Create: `src/segmentation_training/cloud.py`
- Create: `src/segmentation_training/data.py`
- Create: `src/segmentation_training/sampler.py`
- Create: `src/segmentation_training/losses.py`
- Create: `src/segmentation_training/models/unet.py`
- Create: `src/segmentation_training/models/resunet.py`
- Create: `docs/implementation/2026-05-05-segmentation-training-cloud-runbook.md`

- [ ] **Step 1: 实现 guarded torch imports**

所有 torch imports 必须发生在训练专用模块或函数内部。不训练的本机测试即使没有 torch 也必须通过。

- [ ] **Step 2: 实现 cloud command contract**

`train` 应验证 config 和 data roots，写入 `run_manifest.json`，并且只有传入 `--cloud-confirm` 才继续。

- [ ] **Step 3: 编写 runbook**

文档化 upload、environment creation、bundle validation、E1-E3 training commands、result download 和 local summarization。

- [ ] **Step 4: 验证**

本机运行：

```bash
PYTHONPATH=src python -m segmentation_training train --config configs/segmentation_training/e1_binary_c5_unet.yaml
```

预期：命令拒绝本机训练，并提示在云端使用 `--cloud-confirm`。

### Task 8: Full Local Verification

**Files:**
- All files above.

- [ ] **Step 1: 运行全部测试**

运行：

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

预期：全部测试无需 GPU 即可通过。

- [ ] **Step 2: 构建本机 manifest-only bundle**

运行：

```bash
PYTHONPATH=src python -m segmentation_training bundle \
  --contract-dir .agent-team/artifacts/task-ef3a9c94ddb7/implementation/segmentation_contract_v0_1 \
  --bundle-dir .agent-team/artifacts/task-afc7f2c25f8f/implementation/training_bundle_v0_1 \
  --required-channel F16 \
  --required-channel F17 \
  --mode manifest-only
```

预期：bundle metadata 和 rebased manifest 创建；不会启动本机完整训练。

- [ ] **Step 3: 更新 implementation handoff**

实现完成后，写入 `.agent-team/artifacts/task-afc7f2c25f8f/implementation/implementation-implementation-dev-ImplementationHandoff.md` 以及中文 `-zh.md` 归档。

## 自检

Spec coverage：

- 云端优先训练：由 Scope Rules、Task 7 和 runbook 覆盖。
- U-Net/ResUNet baseline harness：由文件结构和 Tasks 6-7 覆盖。
- E0-E4 矩阵：由 configs 覆盖；E4 保留为 optional probe。
- Metrics 与 previews：由 Tasks 4-5 覆盖。
- 不在本机完整训练：由 CLI guard 和验证步骤强制。

Placeholder scan：

- 没有 `TBD` 或开放式“add tests”占位。

Type consistency：

- `ModelInputRecord`、manifest parser、bundle builder、configs、metrics、preview writer 和 CLI 名称在任务间一致。
