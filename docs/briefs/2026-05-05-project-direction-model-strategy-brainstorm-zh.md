# 项目方向头脑风暴简报：SegmentationDatasetContract v0.1 之后的模型策略

日期：2026-05-05

## 目标

发起一轮新的对抗式头脑风暴，用来决定 `SegmentationDatasetContract v0.1` 之后，整个遥感灾害分割项目应该往哪里走。

这不是只围绕 QA 的头脑风暴。核心问题是：

> 项目现在应该选择模型方向、建立训练 baseline、扩展数据/输入契约、补齐缺失数据，还是先做一个结构化实验矩阵再承诺某个模型家族？

本轮输出应明确下一条项目路线、第一项模型或非模型里程碑，以及防止过早锁定架构的决策门槛。

## 当前进展上下文

主要进展说明：

- `docs/implementation/2026-05-05-segmentation-contract-progress.md`

当前实现任务：

- `task-ef3a9c94ddb7`

关键 artifact 根目录：

- `.agent-team/artifacts/task-ef3a9c94ddb7/implementation/segmentation_contract_v0_1/`

重要已生成 artifact：

- `model_input_manifest.csv`
- `training_manifest.csv`
- `class_map.json`
- `channels.json`
- `reports/channel_alignment_report.csv`
- `reports/wkt_mask_alignment_report.csv`
- `reports/geometry_validity_report.csv`
- `reports/visual_qa_summary.md`
- `previews/contact_sheet.png`
- `previews/source_qa/`
- `masks/`

当前可用模型输入事实：

| 项目 | 数值 |
| --- | ---: |
| 模型输入样本数 | 1,252 |
| C2 泥石流样本数 | 500 |
| C5 火灾样本数 | 752 |
| 训练集样本数 | 906 |
| 验证集样本数 | 234 |
| 测试集样本数 | 112 |
| 当前安全对齐通道 | `F16;F17` |
| 暂不训练类别 | C3 为 `255=ignore` / `taxonomy_pending` |

重要约束：

- 当前可训练数据只有 C2 泥石流和 C5 火灾。
- 当前安全对齐通道契约是 `F16;F17`。
- 更丰富的多光谱 stack 需要显式重采样/对齐实现。
- 84 行标签因为缺少参考栅格仍被阻塞。
- C3 在 taxonomy 未确认前不能作为前景类别。
- 现有 `datasets/c5_fire_yolo/` 仍只是历史/后备上下文，不是项目目标。

## 需要保留的研究证据

以下内容应作为证据使用，而不是强制决策：

- `docs/research/memos/2026-05-04-study-and-conversation-synthesis.md`
- `docs/research/matrices/2026-05-04-method-source-matrix-round-1.md`
- `docs/literature/2026-05-04-pre-project-method-landscape.md`
- `.agent-team/artifacts/task-b2d1d7cb8a5e/research/research-ResearchMemo.md`
- 上一轮头脑风暴：`.agent-team/artifacts/task-ef3a9c94ddb7/brainstorm/brainstorm-BrainstormMemo.md`
- 上一轮技术计划：`.agent-team/artifacts/task-ef3a9c94ddb7/technical/technical-TechnicalPlan.md`

## 头脑风暴问题

1. 现在是否到了选择模型家族的时候，还是下一步应先做训练框架和实验矩阵？
2. 如果选择第一个模型，baseline 应该是 U-Net/ResUNet、DeepLabV3+、SegFormer/UPerNet、Mask2Former、遥感基础模型迁移，还是其他路线？
3. 只用 `F16;F17` 是否能训练出有意义的 baseline，还是必须先实现更丰富多光谱通道的重采样/对齐？
4. C2 和 C5 是否应立即进入同一个多类别模型，还是先做受控的单类别 baseline？
5. 如果没有 DEM、slope、aspect、curvature 或 flow accumulation，泥石流性能是否很可能受限？
6. 下一步优先级应是找回缺失栅格、确认 C3 taxonomy、扩展通道，还是开始模型训练？
7. 应用什么指标和实验门槛来判断是继续简单 baseline，还是升级到 transformer/基础模型？
8. 光谱指数、SAM/SAM2、YOLO/DETR、变化检测和遥感基础模型应放在路线图中的什么位置？

## 必须讨论的方法族

本轮必须显式讨论以下路线，只有给出理由后才能拒绝或延后：

- 简单监督语义分割 baseline：U-Net / ResUNet；
- 更强 CNN/decoder baseline：DeepLabV3+ 或类似路线；
- Transformer 分割：SegFormer / UPerNet / Mask2Former；
- 遥感基础模型迁移：Prithvi/SatMAE 类特征提取或微调路线；
- 带 ignore boundary、label confidence、partial labels 的弱监督路线；
- 重采样/对齐后的更丰富多光谱通道 stack；
- 光谱指数诊断或辅助通道；
- 基于 DEM/slope 派生特征的地形增强泥石流路线；
- 若灾前/灾后配对可验证，则保留变化检测路线；
- SAM/SAM2 作为 mask proposal/refinement，而不是真值；
- YOLO/DETR 作为检测后备或 prompt generator，而非规范目标。

## 部门分工

### research.lead

在数据契约之后重新框定总体项目路线。既要挑战过早模型选择，也要挑战无止境数据工程。推荐一条研究上有效、能真正产生项目学习的下一步路线。

### technical.lead

挑战可行性。结合当前 `F16;F17` manifest、数据规模、类别平衡、依赖成本、训练复杂度和未来扩展性比较模型家族。定义模型选择前或模型选择过程中最低限度的技术里程碑。

### qa.tester

挑战可测试性。定义实验协议、指标、防泄漏控制、模型选择门槛、视觉失败复盘和 stop/go 标准。阻止无法产生可解释证据的路线。

## 必需输出

- `ResearchBrainstormNote`
- `TechnicalBrainstormNote`
- `QABrainstormNote`
- `BrainstormMemo`

最终 `BrainstormMemo` 必须包含：

- 独立立场；
- challenge 与 rebuttal；
- 路线推荐；
- 候选模型家族排序；
- 下一步应实现什么；
- 升级路线前必须测量什么；
- 被拒绝或延后的路线；
- 决策记录；
- 下一合法部门和建议分配。

## 非目标

- 本阶段不训练模型。
- 本阶段不分配技术、实现、QA 或评审部门。
- 不把 QA 批准本身当成项目方向。
- 不选择最终生产架构。
- 不用 C5-only 或 box-only 证据声称多灾种性能。
- 不在没有实验证据前把 `F16;F17` 当成最终光谱设计。
