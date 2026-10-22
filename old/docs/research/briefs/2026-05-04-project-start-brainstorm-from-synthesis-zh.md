# 项目启动头脑风暴 Brief：基于学习与对话综合文档的遥感灾害范围勾画

## 目标

使用“调研学习与对话关键信息整理”作为共享上下文，启动正式的项目启动型对抗式头脑风暴。本轮工作坊需要把已有研究学习、本地数据约束和对话中形成的流程要求，转化为候选技术路线组合、分阶段项目入口，以及后续技术规划前必须验证的事项。

本任务只属于头脑风暴阶段。不得实现数据处理，不得训练模型，不得选择最终架构，也不得分派后续部门。

## 必须吸收的主要上下文

- 当前综合 memo：`docs/research/memos/2026-05-04-study-and-conversation-synthesis.md`
- 研发调研任务：`task-b2d1d7cb8a5e`
- 研发 memo：`.agent-team/artifacts/task-b2d1d7cb8a5e/research/research-ResearchMemo.md`
- 中文研发归档：`.agent-team/artifacts/task-b2d1d7cb8a5e/research/research-ResearchMemo-zh.md`
- 来源矩阵：`docs/research/matrices/2026-05-04-method-source-matrix-round-1.md`
- 方法图谱：`docs/literature/2026-05-04-pre-project-method-landscape.md`
- 既有 evidence-grounded brainstorm 仅作历史参考：`task-7a02ad0777e6`

需要吸收的学习日志：

- `docs/research/learning-logs/2026-05-04-land8fire-wildfire-segmentation.md`
- `docs/research/learning-logs/2026-05-04-landslide4sense-benchmark.md`
- `docs/research/learning-logs/2026-05-04-sparse-annotations-festa.md`
- `docs/research/learning-logs/2026-05-04-satmae-pretraining.md`
- 若存在则读取 `docs/research/learning-logs/2026-05-04-xBD-disaster-change-detection.md`；否则读取 `docs/research/learning-logs/2026-05-04-xbd-disaster-change-detection.md`
- `docs/research/learning-logs/2026-05-04-sam-segment-anything.md`
- `docs/research/learning-logs/2026-05-04-safe-burned-area-extraction.md`
- `docs/research/learning-logs/2026-05-04-prithvi-geospatial-foundation-model.md`

## 本地项目上下文

- 项目目标是从遥感影像中定位并勾画灾害范围，不是图像级分类。
- 原始数据位于 `database/`。
- 预处理数据位于 `datasets/`，尤其是 `datasets/c5_fire_yolo/`。
- `datasets/c5_fire_yolo/data.yaml` 当前只暴露一个 YOLO 类：`0: fire`。
- `datasets/c5_fire_yolo/manifest.csv` 关联图像、YOLO label、bbox、F07/F11/F12 源波段路径和 `polygon_wkt`。
- `database/csv文本/` 下的灾害 CSV 包含 WKT polygon 和类别字段。
- 原始 `.tif` 文件名似乎编码了 event date、image date、tile/grid id 和 spectral band id。

## 头脑风暴纪律

- 按对抗式工作坊结构执行：独立立场、挑战回合、反驳回合、协商立场。
- 在本地证据足以收敛前，必须保留多个候选方法族。
- 路线建议要基于研究/文献证据层，而不能只围绕当前 fire-only YOLO 数据集。
- 参考 Superpowers 的头脑风暴质量标准：提出竞争方案，说明取舍，推荐分阶段方向，并把实现留在明确批准之后。
- 后续部门必须保持 idle，直到用户或 coordinator 明确分派。

## 必须讨论的方法族

头脑风暴必须显式讨论以下路线。只有在有证据时，才能拒绝或延期：

- 经典遥感指数与物理 baseline；
- 带地形和对象特征的 OBIA / classical ML；
- 全监督语义分割；
- 基于粗 WKT polygon、ignore 区域和 label confidence 的弱监督；
- 如果本地日期支持配对，则讨论灾前/灾后变化检测；
- SAM 辅助候选 mask 生成或细化；
- SAFE 式火灾/烧毁区候选生成与细化；
- 使用 SatMAE/Prithvi 类模型的遥感基础模型迁移；
- 使用 YOLO/DETR 式 box 做粗定位或 prompt 的检测退路。

## 工作坊问题

- 第一个项目里程碑应该是什么：数据审计与 readiness plan、火灾优先的多光谱分割、多类别 WKT mask 构建，还是先检测退路再转分割？
- 第一版可用数据集应以什么监督来源为主：CSV WKT polygon、生成 mask、已有 YOLO label、人工/可视化 QA，还是弱标签组合？
- F07/F11/F12 和潜在其他波段应该如何表示：物理指数、false-color preview、stacked multi-band tensor、基础模型兼容 schema，还是混合特征？
- 本地元数据是否支持灾前/灾后变化检测？在依赖它之前需要什么审计？
- 泥石流/滑坡路线可行前必须满足什么条件，尤其是 DEM、slope、aspect、curvature 和 flow accumulation？
- SAM、SAFE、SatMAE、Prithvi 和 YOLO 应该如何作为工具进入流程，同时避免被当成未经验证的 ground truth？
- 哪些 QA gate 应阻止技术规划、模型训练和成功声明？

## 头脑风暴部门分派

### research.lead

基于综合 memo 和研究证据重新界定项目目标。挑战过早收敛到 YOLO、普通 U-Net、SAM 或任何单一路线。保留方法族，并说明后续收敛需要什么证据。

### technical.lead

挑战实现可行性。针对每条路线识别所需本地数据、预处理、基础设施、计算成本、集成风险，以及进入实现规划前最低限度需要产出的审计 artifact。

### qa.tester

挑战可测试性。为每个方法族定义验收标准、泄漏检查、mask/CRS 对齐检查、标签噪声审计、划分策略、可视化检查 gate 和失败模式。

## 必需输出

- `ResearchBrainstormNote`
- `TechnicalBrainstormNote`
- `QABrainstormNote`
- `BrainstormMemo`

最终 `BrainstormMemo` 必须包含独立意见、挑战回合、反驳回合、共识、取舍、拒绝或延期路线、决策记录，以及下一步验证需求。

## 非目标

- 不开始实现。
- 不训练模型。
- 不修改数据集。
- 不选择最终模型架构。
- 不把 WKT polygon、生成 mask、SAM 输出、SAFE 输出或 YOLO box 当成已审计 ground truth。
- 不把当前 fire-only YOLO 数据集当成项目定义。
