# SegmentationDatasetContract v0.1 阶段性进展报告

日期：2026-05-05

任务：`task-ef3a9c94ddb7`

## 概要

本阶段已经完成从 Phase A 元数据合同，到 Phase B mask 生成、可视化 QA、保守模型输入合同的开发路径。当前 artifact 已经可以支撑后续训练数据读取器开发，但按项目流程，仍建议先交给测试部门验证后，再把它视为正式批准的训练数据集。

当前实现仍然保持在修正后的项目目标内：面向 C2 泥石流与 C5 火灾的多光谱灾害语义分割。C3 继续保持为 `255=ignore` / `taxonomy_pending`，并已从前景训练 manifest 中排除。

## 已完成的开发切片

### Phase A：Schema 与 Manifest 合同

已实现包：

- `src/segmentation_contract/`

主要生成 artifacts：

- `samples_manifest.csv`
- `class_map.json`
- `channels.json`
- `split_policy.md`
- `skipped_label_rows.csv`
- `qa_summary.md`
- `dataset_contract.md`

Phase A 建立了稳定的 sample ID、事件级 split 策略、类别 ID、源 CSV 行号追溯、源 raster 关联，以及 malformed/skipped rows 的审计记录。

### Phase B：地理 Mask 生成

Phase B 现在使用 `shapely` 与 `rasterio` 解析 WKT polygon、检查 reference raster、栅格化语义 mask，并写出地理验证报告。

主要生成 artifacts：

- `masks/*.tif`
- `reports/geometry_validity_report.csv`
- `reports/wkt_mask_alignment_report.csv`
- `reports/phase_b_dependency_report.md`

当前 Phase B 结果：

| 项目 | 数量 |
| --- | ---: |
| Manifest 总行数 | 1,386 |
| 几何有效行 | 1,382 |
| Invalid WKT 行 | 4 |
| 已生成 mask | 1,302 |
| 缺 reference raster 阻塞行 | 84 |

这 84 条 blocked rows 暂时不能用于训练，除非后续找到或重新生成对应 reference raster。

### Visual QA

已生成可视化检查 artifacts：

- `previews/contact_sheet.png`
- `previews/source_qa/泥石流火灾_C2_debris_flow_gray_contact_sheet.png`
- `previews/source_qa/泥石流火灾_C5_fire_gray_contact_sheet.png`
- `previews/source_qa/泥石流火灾_training_samples_index.csv`
- `reports/visual_qa_summary.md`

用户已经交叉检查 preview overlay，并确认目前看到的疑似问题主要来自波段显示方式，而不是 mask 对齐错误。

### 训练 Manifest 与模型输入 Manifest

已生成：

- `training_manifest.csv`
- `model_input_manifest.csv`
- `reports/channel_alignment_report.csv`
- `reports/model_input_summary.md`

`training_manifest.csv` 只包含满足以下条件的样本：

- `alignment_status == mask_written`
- `class_id != 255`

`model_input_manifest.csv` 进一步确认必需输入通道已经网格对齐。当前保守模型输入合同使用 `F16;F17`，因为所有可训练行的这两个波段在 CRS、transform、height、width 上全部对齐。

当前模型输入结果：

| 项目 | 数量 |
| --- | ---: |
| 模型输入行 | 1,252 |
| C2 泥石流行 | 500 |
| C5 火灾行 | 752 |
| Train split 行 | 906 |
| Validation split 行 | 234 |
| Test split 行 | 112 |
| 必需通道 | `F16;F17` |
| 通道对齐状态 | 1,252 行全部 `aligned` |

## 数据来源映射

当前可训练数据来自：

- `database/csv文本/泥石流火灾.csv`

当前可用行按类别统计：

| Source CSV | 类别 | 行数 |
| --- | --- | ---: |
| `泥石流火灾.csv` | `C2_debris_flow` | 500 |
| `泥石流火灾.csv` | `C5_fire` | 752 |

以下 CSV 的行当前因为没有找到 linked reference raster 而被阻塞：

- `database/csv文本/damage_poly_r3云南.csv`
- `database/csv文本/damage_poly_r3四川.csv`

## 重要技术发现

1. `F16` 和 `F17` 是当前最安全的模型输入通道，因为它们在全部 1,252 条可训练行上对齐。
2. 一些类似 RGB 的通道组合，例如 `F04/F03/F02`，在部分数据上存在分辨率不一致，不能在未重采样的情况下直接 stack。
3. 更丰富的多光谱 stack 仍然可以做，但需要先实现明确的重采样/对齐流程。
4. C3 因 taxonomy 未确认，已被有意排除在前景训练之外。
5. 当前生成的 masks 已适合进入训练读取器开发，但正式训练结论前应先做 QA 部门评审。

## 当前 Artifact 根目录

所有 dataset-contract artifacts 位于：

```text
.agent-team/artifacts/task-ef3a9c94ddb7/implementation/segmentation_contract_v0_1/
```

后续开发最关键文件：

```text
model_input_manifest.csv
training_manifest.csv
class_map.json
reports/channel_alignment_report.csv
reports/wkt_mask_alignment_report.csv
reports/geometry_validity_report.csv
previews/contact_sheet.png
previews/source_qa/
masks/
```

## 验证情况

最新开发验证命令已通过：

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

结果：

```text
Ran 33 tests
OK
```

## 剩余风险与下一步

推荐下一部门：

```text
测试部门测试
```

推荐 QA 范围：

- 验证 Phase B masks 与 reports；
- 检查 visual QA contact sheets；
- 确认 `model_input_manifest.csv` 已排除 `255=ignore`；
- 确认 split 统计与 leakage policy；
- 确认 `F16/F17` 的通道对齐假设；
- 记录当前 artifact set 是否可以批准进入训练读取器开发。

如果在正式 QA 前继续开发，下一片安全实现应是 data-loader smoke path：读取 `model_input_manifest.csv` 并返回：

```text
X = [F16, F17, H, W]
y = [H, W]
```

这个下一切片仍应避免模型训练和架构选择，直到 QA 批准当前数据 artifacts。
