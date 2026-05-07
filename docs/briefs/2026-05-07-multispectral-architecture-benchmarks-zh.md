# 多光谱架构基准头脑风暴 Brief

日期：2026-05-07

## 目标

在最新训练信号显示当前网络家族可能难以满足灾后多光谱分割目标之后，启动部门工作流。

当前任务是为 11 通道分割输入评估并实现一个受控的架构基准跑道，依据包括：

- 实施计划：`docs/superpowers/plans/2026-05-07-multispectral-architecture-benchmarks.md`；
- 当前灾后多光谱路线：`task-2ba5416843b1`；
- E1 后诊断路线：`task-6c4353fb293d`；
- `configs/segmentation_training/` 下的现有训练配置；
- `src/segmentation_training/models/` 下的现有模型代码。

## 需要吸收的当前证据

- 早期 U-Net 风格训练在稀疏 mask 下给出的前景证据较弱或不稳定。
- 项目已经从利用不足的 `F16/F17` 合同转向更丰富的灾后 optical、SWIR/NIR、派生指数和辅助通道 stack。
- 当前 11 通道配置使用灾后 optical 与指数通道。
- 单个 U-Net/ResUNet 结果已经不足以判断瓶颈到底来自架构家族还是输入合同。
- 封存 test split 必须继续封存；架构选择只能使用 train/validation 证据。

## 必须保留的架构家族

各部门必须保留一个可比较的基准集合：

- 现有 U-Net；
- 现有 ResUNet；
- Attention U-Net；
- UNet++；
- DeepLabV3+；
- Transformer-UNet。

这些家族在技术上合理的范围内，应共享相同的数据合同、输入通道、划分策略、loss/sampler/performance 设置、threshold sweep、prediction summary 输出和 best-checkpoint 输出。

## 部门任务

### research.lead

界定当前训练信号能证明什么、不能证明什么。挑战“当前网络已经不可能”的过早判断，同时支持在同一 11 通道证据下进行更宽的架构基准。解释所选家族与遥感火灾、泥石流分割任务的关系。

### technical.lead

挑战可行性。定义 model registry、新模型 adapter、架构专用配置和验证测试的最小实施计划。保持训练 CLI 与 bundle contract 稳定。

### qa.tester

挑战可测试性。定义 shape compatibility、config validation、封存 test split 纪律、可比较架构设置和 cloud-run readiness 的验收门槛。阻止在没有可追溯性的情况下同时改变数据、loss、sampler 和架构的 benchmark。

## 必需输出

- `ResearchBrainstormNote`
- `TechnicalBrainstormNote`
- `QABrainstormNote`
- `BrainstormMemo`

最终 `BrainstormMemo` 必须包含独立部门立场、挑战、回应、协商共识、推荐实施顺序、QA 门槛、暂缓选项、决策日志和下一个有效部门。

## 非目标

- 不在 brainstorm 阶段训练模型。
- 不使用封存 test split。
- 不从 shape tests 或 config validation 宣称最终胜出架构。
- 添加架构 adapter 时不改变 cloud data bundle contract。
- 不把 Transformer-UNet 当成输入合同或标签质量诊断的替代品。
