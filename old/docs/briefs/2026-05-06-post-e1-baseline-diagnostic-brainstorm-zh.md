# E1 后基线诊断头脑风暴简报

日期：2026-05-06

## 目标

在第一次 `SegmentationTrainingBaseline v0.1` 云端训练信号返回后，开展一次聚焦的对抗式头脑风暴。

核心问题是：

> E1 C5 火灾 U-Net 的 validation loss 较低，但 `mean_iou=0.0000` 且 `foreground_recall=0.0000`。下一步应该先诊断并修复 U-Net 基线，引入 ViT/Transformer 分割路线，还是在受控门槛下两者并行推进？

项目所有者当前判断必须作为本次讨论的明确输入：

> 后续大概率需要引入 ViT 风格模型进行训练，但仅凭当前 E1 结果，不可能直接得出 U-Net 完全没效果的结论。

本次讨论不能滑向两个极端：

- 不能从一次失败的 E1 尝试就宣布 U-Net 无效；
- 如果修正后的基线仍无法学习前景，也不能忽略 Transformer/ViT 路线规划。

## 需要吸收的当前状态

主要状态备忘录：

- `docs/implementation/2026-05-06-training-baseline-status-and-next-steps.md`
- 中文归档：`docs/implementation/2026-05-06-training-baseline-status-and-next-steps-zh.md`

父级项目方向任务：

- `task-afc7f2c25f8f`

父任务相关产物：

- `.agent-team/artifacts/task-afc7f2c25f8f/brainstorm/brainstorm-BrainstormMemo.md`
- `.agent-team/artifacts/task-afc7f2c25f8f/technical/technical-technical-lead-TechnicalPlan.md`
- `.agent-team/artifacts/task-afc7f2c25f8f/implementation/implementation-implementation-dev-ImplementationHandoff.md`
- `.agent-team/artifacts/task-afc7f2c25f8f/qa/qa-qa-tester-TestReport.md`
- `.agent-team/artifacts/task-afc7f2c25f8f/review/review-review-lead-ReviewNote.md`

状态备忘录中的当前训练事实：

| 项目 | 数值 |
| --- | ---: |
| 模型输入行数 | 1,252 |
| C2 泥石流行数 | 500 |
| C5 火灾行数 | 752 |
| 训练集行数 | 906 |
| 验证集行数 | 234 |
| 测试集行数 | 112 |
| 当前安全输入 | `F16/F17` |
| E1 模型 | 二分类 C5 U-Net |
| E1 validation loss | 约 `0.083`，约十轮后稳定 |
| E1 validation `mean_iou` | `0.0000` |
| E1 validation foreground recall | `0.0000` |

封存测试集必须继续封存，不得用于路线选择。

## 头脑风暴问题

1. 低 validation loss 但前景 IoU/recall 为零，最可能的解释是什么？
2. 在把 E1 解释为模型家族证据之前，必须先运行哪些诊断？
3. 什么证据能证明 U-Net 其实学到了一些弱信号，只是受阈值、采样、loss 权重或解码影响而失败？
4. 什么证据能证明当前 `F16/F17` 输入契约不足以支撑 C5 火灾分割？
5. foreground-aware sampler 和 loss 调整应该先测试哪些，并如何做到一次只改变一个因素？
6. ViT/Transformer 模型应在路线图中何时进入：立即并行探针、带门槛的 E1c/E4 实验，还是在更丰富通道完成后？
7. 本项目第一种 Transformer 候选应是什么：原生 ViT 分割、SegFormer、UPerNet、Mask2Former、SatMAE/Prithvi 类迁移，还是其他路线？
8. 哪些指标、预测摘要和可视化预览门槛决定继续修复 U-Net、加入 ResUNet/DeepLabV3+，或升级到 Transformer/基础模型方法？

## 部门立场要求

### research.lead

从科学解释角度框定结果。挑战“U-Net 已无效”的过早结论，同时保留 ViT/Transformer 分割的研究理由。说明遥感分割实践如何看待小数据、类别不平衡、前景稀疏、多光谱输入和 Transformer 升级。

### technical.lead

挑战实现可行性。定义最小诊断和 E1b/E1c 实验方案，隔离 sampler、loss、指标解码和模型家族变量。比较加入 ViT/SegFormer 风格适配器与修复当前 U-Net 训练循环和数据管线的实际成本。

### qa.tester

挑战可测试性。定义失败标签、验收门槛、预测摘要报告、前景面积检查、threshold/argmax 审计、视觉 QA 要求和 stop/go 标准。阻止任何同时改变多个因素、导致证据不可解释的路线。

## 必需输出

- `ResearchBrainstormNote`
- `TechnicalBrainstormNote`
- `QABrainstormNote`
- `BrainstormMemo`

最终 `BrainstormMemo` 必须包括：

- 各部门独立立场；
- challenge、rebuttal 和协商共识；
- 诊断优先路线；
- ViT/Transformer 引入门槛；
- 一次只改变一个因素的实验矩阵；
- 被拒绝或延后的选项；
- 决策记录；
- 下一有效部门和建议指令。

## 非目标

- 本阶段不训练新模型。
- 不使用封存测试集。
- 不选择最终生产架构。
- 不把 E1 结果当作 C5 不可学习的证明。
- 不把 ViT 当作能自动修复采样、loss、指标、标签或输入通道缺陷的魔法方案。
