# 证据驱动头脑风暴 brief：遥感灾害范围识别

## 目标

在研发学习与调研之后，启动正式 adversarial brainstorm。每个参与部门必须先吸收研发部门的学习产物，再提出路线。目标是保留多个可行方法族，并定义技术规划前必须验证的内容。

## 必须吸收的研发产物

- 研发任务：`task-b2d1d7cb8a5e`
- 研究备忘录：`.agent-team/artifacts/task-b2d1d7cb8a5e/research/research-ResearchMemo.md`
- 中文归档：`.agent-team/artifacts/task-b2d1d7cb8a5e/research/research-ResearchMemo-zh.md`
- 来源矩阵：`docs/research/matrices/2026-05-04-method-source-matrix-round-1.md`
- 方法图谱：`docs/literature/2026-05-04-pre-project-method-landscape.md`

必须吸收的学习日志：

- `docs/research/learning-logs/2026-05-04-land8fire-wildfire-segmentation.md`
- `docs/research/learning-logs/2026-05-04-landslide4sense-benchmark.md`
- `docs/research/learning-logs/2026-05-04-sparse-annotations-festa.md`
- `docs/research/learning-logs/2026-05-04-satmae-pretraining.md`
- `docs/research/learning-logs/2026-05-04-xbd-disaster-change-detection.md`
- `docs/research/learning-logs/2026-05-04-sam-segment-anything.md`
- `docs/research/learning-logs/2026-05-04-safe-burned-area-extraction.md`
- `docs/research/learning-logs/2026-05-04-prithvi-geospatial-foundation-model.md`

## 本地项目上下文

- 原始数据位于 `database/`。
- 预处理数据位于 `datasets/`，尤其是 `datasets/c5_fire_yolo/`。
- `datasets/c5_fire_yolo/data.yaml` 当前只有一个 YOLO 类：`0: fire`。
- `datasets/c5_fire_yolo/manifest.csv` 关联图像、YOLO 标签、bbox、F07/F11/F12 源波段路径和 `polygon_wkt`。
- `database/csv文本/` 下的灾害 CSV 包含 WKT polygon 和类别字段。
- 原始 `.tif` 文件名似乎编码 event date、image date、tile/grid id 和 spectral band id。

## 必须保留讨论的方法族

头脑风暴必须显式讨论这些方法族；只有在有证据时才能拒绝或推迟：

- 传统遥感指数和物理 baseline；
- 带地形和对象特征的 OBIA / classical ML；
- 全监督语义分割；
- 粗 WKT polygon 与 ignore region 弱监督；
- 灾前/灾后变化检测；
- SAM 辅助候选 mask 生成或细化；
- SatMAE/Prithvi 类遥感基础模型迁移；
- YOLO/DETR 式检测 fallback。

## 各部门头脑风暴任务

### research.lead

在学习 sprint 后重新定义问题。挑战过早收敛，尤其是没有数据兼容证据就默认 YOLO、普通 U-Net 或 SAM。保留研究假设和方法族。

### technical.lead

挑战实现可行性。对每条路线识别所需数据、预处理、模型基础设施、计算成本和集成风险。定义技术规划前必须完成的本地数据审计。

### qa.tester

挑战可测试性和验收标准。定义泄漏检查、mask/CRS 对齐检查、标签噪声审计、划分策略、指标和可视化检查门。

## 必需输出

- `ResearchBrainstormNote`
- `TechnicalBrainstormNote`
- `QABrainstormNote`
- `BrainstormMemo`

最终 `BrainstormMemo` 必须包含共识、挑战、回应、权衡、拒绝或推迟路线、决策记录和下一步验证需求。

## 非目标

- 不开始实现。
- 不训练模型。
- 不选择最终架构。
- 不把任何基础模型输出当作 ground truth。
- 不把当前 fire-only YOLO 数据集当作项目定义。
