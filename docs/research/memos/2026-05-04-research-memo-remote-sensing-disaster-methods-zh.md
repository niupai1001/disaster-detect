# 研究备忘录：遥感灾害范围识别方法

日期：2026-05-04

任务：`task-b2d1d7cb8a5e`

## 调研范围

本轮研发调研在技术路线选择前，学习遥感灾害定位与范围勾画方法。项目优先输出 segmentation mask 或地理 polygon，bbox 只作为退路，不能把当前 fire-only YOLO 数据集当作默认路线。

## 证据层

学习日志现在采用“先读论文本身，再谈项目启发”的结构：问题、数据、方法、创新点、实验结论与局限、项目启发、头脑风暴约束、下一步检查。

| 来源 | 学习日志 | 方法族 |
| --- | --- | --- |
| Land8Fire | `docs/research/learning-logs/2026-05-04-land8fire-wildfire-segmentation.md` | 多光谱 wildfire segmentation benchmark |
| Landslide4Sense | `docs/research/learning-logs/2026-05-04-landslide4sense-benchmark.md` | 光学 + 地形数据滑坡分割 |
| Sparse annotations / FESTA | `docs/research/learning-logs/2026-05-04-sparse-annotations-festa.md` | 遥感弱监督分割 |
| SatMAE | `docs/research/learning-logs/2026-05-04-satmae-pretraining.md` | 多光谱/多时相自监督预训练 |
| xBD | `docs/research/learning-logs/2026-05-04-xbd-disaster-change-detection.md` | 灾前/灾后损毁评估 |
| Segment Anything | `docs/research/learning-logs/2026-05-04-sam-segment-anything.md` | promptable mask proposal |
| SAFE | `docs/research/learning-logs/2026-05-04-safe-burned-area-extraction.md` | burned-area 候选生成 + SAM 细化 |
| Prithvi | `docs/research/learning-logs/2026-05-04-prithvi-geospatial-foundation-model.md` | 遥感基础模型迁移 |

来源矩阵：`docs/research/matrices/2026-05-04-method-source-matrix-round-1.md`。

## 研发部门学到的内容

火灾和烧毁区应被视为多光谱分割问题，而不是简单检测问题。Land8Fire 说明 benchmark 设计本身很重要：波段组合、loss、类别不平衡和火区形态都会改变结论。本项目必须保留 NIR/SWIR 类波段和火灾指数，不应过早压成 RGB。

泥石流不能并入火灾路线。Landslide4Sense 说明地貌灾害范围识别受 DEM 和 slope 影响明显。如果泥石流是重要类别，需要尽早决定：补充 DEM 衍生特征、明确无 DEM 的限制，或把泥石流作为较弱的探索路线。

本地 WKT 标签在证明精确前应按粗标注处理。稀疏标注分割研究提示更合理的标签组织方式：可靠前景、忽略边界、可靠背景和 partial-label 训练。直接把 polygon rasterize 成 hard mask 风险较高。

部分灾害更适合变化检测。xBD 虽然聚焦建筑损毁，但其灾前/灾后事件结构很有借鉴意义，因为本地文件名似乎包含 event date 和 image date。项目需要先做事件/日期审计，再决定是否采用单图模型。

基础模型必须拆开讨论。SAM 适合 promptable mask proposal 和标注细化；SatMAE、Prithvi 更适合多光谱或多时相遥感迁移。它们都不能替代标签审计、地理配准和评价设计。

## 候选路线组合

| 路线 | 为什么保留 | 第一证据 |
| --- | --- | --- |
| 光谱指数 baseline | 火灾和植被变化有物理先验 | 波段映射，NBR/NDVI/SWIR 叠加图 |
| OBIA/classical ML | 适合地形/对象驱动的泥石流候选 | DEM/slope 可用性，对象特征 |
| 全监督分割 | 直接输出 mask/polygon | mask 对齐和划分策略 |
| 弱监督 | 匹配本地粗 WKT 现实 | 标签噪声审计和 ignore-zone 设计 |
| 变化检测 | 很多灾害是时间变化现象 | event/image date 配对审计 |
| SAM 辅助 proposal | 可从框/polygon 细化候选 mask | prompt 质量和光谱 QA |
| 遥感基础模型迁移 | 适合小标注多光谱数据 | 本地波段/schema 兼容性 |
| 检测退路 | 快速粗定位 baseline | mask-to-box 质量和类别平衡 |

## 正式头脑风暴约束

- 各部门必须先阅读学习日志和来源矩阵，再提出路线。
- 头脑风暴必须保留竞争方法族，不能过早收敛到 YOLO、U-Net、SAM 或任何单一熟悉算法。
- 技术部门方案必须包含本地数据兼容性检查：波段清单、CRS/mask 对齐、事件/日期配对、DEM/slope 可用性、类别平衡。
- QA 方案必须包含按事件、日期、区域和相邻 patch 的泄漏检查。
- 任一基础模型方案必须说明它是 proposal/refinement 工具、backbone/feature extractor，还是伪标签生成器。
- 任一分割方案必须说明如何处理粗 WKT 边界和 ignore region。

## 交接

研发部门建议启动新的 evidence-grounded adversarial brainstorm。新头脑风暴应明确吸收本轮研发产物，由 research、technical、QA 分别写独立 note，再由 orchestrator 综合成修订版 `BrainstormMemo`。
