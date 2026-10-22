# 遥感灾害范围识别项目前置方法图谱

日期：2026-05-04

这份记录是因为第一版头脑风暴太窄。项目在选择 YOLO、U-Net、SAM 或其他实现路线前，需要先了解类似遥感灾害定位问题现在通常是怎么做的。

## 范围

项目目标不是图像级分类，而是从卫星影像中定位并勾画灾害范围：

- 优先输出：分割 mask 或地理多边形；
- 退路输出：多边形框或矩形框；
- 目标灾害：火灾/烧毁区、泥石流/滑坡类地貌灾害，以及本地 CSV 中可能存在的其他灾害类别；
- 本地约束：多光谱波段、粗 WKT polygon/mask、可能存在多时相影像、标注样本有限。

## 需要学习的方法族

### 1. 传统遥感指数与规则

火灾和烧毁区制图有很长的非深度学习传统。这里的关键不是永远使用阈值，而是光谱指数包含了深度模型不应忽略的物理信号。

需要研究：

- `NBR`, `dNBR`, `RdNBR`
- `BAI`, `BAIS2`, `MIRBI`
- `NDVI`, `NDMI`, `NDWI`
- 云、水体、阴影、暗目标 mask
- 阈值、可分性分析、Random Forest、XGBoost

相关来源：

- 森林烧毁区/火烧强度综述：https://www.mdpi.com/2072-4292/14/19/4714
- Sentinel-2 烧毁区指数 BAIS2：https://www.mdpi.com/2504-3900/2/7/364
- Sentinel-2 U-Net 烧毁区分割：https://www.mdpi.com/2072-4292/12/15/2422

对项目的意义：传统指数应成为 baseline 特征、伪标签生成器、QA 叠加图，或模型输入通道。

### 2. 全监督语义分割

如果 mask 足够可信，这是最直接的路线。常见模型包括 U-Net、ResU-Net、FCN、PSPNet、DeepLabV3+、UPerNet、SegFormer、Mask2Former 以及其他 Transformer decoder。

相关来源：

- Land8Fire wildfire segmentation benchmark：https://www.mdpi.com/2072-4292/17/16/2776
- Active fire detection 数据集与深度学习研究：https://www.sciencedirect.com/science/article/pii/S092427162100160X
- 遥感小数据语义分割综述：https://www.mdpi.com/2072-4292/15/20/4987

对项目的意义：分割仍是中心目标，但模型架构选择要等标签质量、波段清单和划分策略明确后再决定。

### 3. 带地形和对象特征的滑坡/泥石流制图

泥石流文献经常和 landslide、shallow landslide、mass movement 混在一起。关键区别是：地貌灾害范围识别通常需要地形、纹理、形状和多时相变化，不只是光谱外观。

需要研究：

- Object-Based Image Analysis，OBIA
- Sentinel-2 灾前/灾后变化与植被损失
- DEM、坡度、坡向、曲率、汇流面积
- Random Forest / XGBoost 对象分类
- U-Net / ResU-Net 语义分割

相关来源：

- Landslide4Sense benchmark：https://arxiv.org/abs/2206.00515
- U-Net / ResU-Net 滑坡检测迁移性评估：https://www.nature.com/articles/s41598-021-94190-9
- Sentinel-2 + GEE 浅层滑坡半自动制图：https://nhess.copernicus.org/articles/23/2625/2023/
- 基于 OBIA 和开源工具的滑坡制图：https://www.sciencedirect.com/science/article/pii/S0013795221000119

对项目的意义：如果泥石流类别重要，应尽早寻找 DEM/slope 特征。纯 RGB 或假彩色图像模型很可能不够。

### 4. 变化检测与灾害损失评估

有些灾害任务更适合建模为变化检测，而不是单张图分割。火烧痕迹、洪水、建筑损毁、滑坡和泥石流在有灾前/灾后影像时都可能受益。

需要研究：

- 灾前/灾后 Siamese encoder
- 二值和语义变化检测
- 建筑/损害定位与强度分类
- 事件级时间分组

相关来源：

- xBD 建筑损毁数据集：https://arxiv.org/abs/1911.09296
- 面向灾害风险管理的星基地球观测：https://link.springer.com/article/10.1007/s10712-020-09586-5

对项目的意义：本地文件名已经编码 event date 和 image date。项目应该先审计每个事件是否有可用的灾前/灾后时间上下文，而不是直接假设单图像设定。

### 5. 弱监督、粗标注与伪 mask

本地 WKT polygon 和 mask 可能比较粗。把它们当作完美像素标签，可能训练出脆弱模型。正式训练前应先研究 label-efficient segmentation。

需要研究：

- 稀疏标签：点、scribble、polygon
- reliable foreground / reliable background / ignore boundary
- CAM 伪标签
- affinity refinement
- 人工复核闭环下的伪标签迭代
- 对不确定样本做主动学习

相关来源：

- 遥感稀疏标注分割：https://github.com/Hua-YS/Semantic-Segmentation-with-Sparse-Labels
- 遥感弱监督深度分割：https://www.mdpi.com/2072-4292/12/2/207
- 遥感小数据语义分割综述：https://www.mdpi.com/2072-4292/15/20/4987

对项目的意义：第一版 dataset builder 应保留 polygon、mask、bbox 和 ignore region，而不是过早压扁成单一 hard mask。

### 6. 基础模型与 promptable segmentation

基础模型不是万能答案，但能拓宽设计空间。

需要研究：

- SAM / SAM-like promptable segmentation，用于 mask proposal 和 refinement；
- SAFE 这类 Sentinel-2 + SAM 的烧毁区提取；
- SatMAE / SatMAE++ 多光谱/多时相预训练；
- Prithvi / HLS geospatial foundation model，用于洪水和火烧痕迹制图；
- 遥感 vision-language 或 geospatial foundation model 的迁移学习。

相关来源：

- Segment Anything：https://arxiv.org/abs/2304.02643
- SAFE Sentinel-2 + SAM 烧毁区提取：https://www.mdpi.com/2072-4292/17/1/54
- SatMAE：https://arxiv.org/abs/2207.08051
- NASA/IBM Prithvi geospatial foundation model：https://www.earthdata.nasa.gov/news/blog/ibm-research-nasa-impact-release-open-source-geospatial-foundation-model-hugging-face
- Prithvi-EO-2.0 概述：https://research.ibm.com/publications/prithvi-eo-an-open-access-geospatial-foundation-model-advancing-earth-science-through-global-collaboration

对项目的意义：SAM 可以作为 mask proposal/refinement 工具，但不能被无条件当作最终分割器。当地物多光谱信息很重要时，地理空间基础模型可能比自然图像基础模型更合适。

## 候选路线组合

项目在 research sprint 前应同时保留多条路线：

| 路线 | 核心想法 | 为什么重要 | 第一证据 |
| --- | --- | --- | --- |
| 光谱指数 baseline | 使用 NBR/BAIS2/NDVI/DEM 类特征 | 物理先验强、可解释 | 波段映射和指数叠加图 |
| 传统 ML / OBIA | 对象分割后用 RF/XGBoost 分类 | 适合滑坡/泥石流候选区域 | 对象特征、DEM/slope 可用性 |
| 全监督分割 | U-Net/DeepLab/SegFormer/Mask2Former | 直接输出 mask | mask 对齐和划分策略验证 |
| 变化检测 | 灾前/灾后事件模型 | 很多灾害本质是变化现象 | event/image date 审计 |
| 弱监督 | 粗 polygon -> 可靠标签 + ignore zone | 匹配本地粗标注现实 | 标签噪声审计 |
| 基础模型辅助 | SAM/Prithvi/SatMAE 做 proposal 或迁移 | 现代迁移路线 | 与本地波段/标签兼容性 |
| 检测退路 | 从 polygon 生成 YOLO/DETR bbox | 快速定位 fallback | mask-to-box 质量和类别平衡 |

## 项目正式开始前的 Research Sprint

正式实现前，先跑一个短 research sprint，产出：

1. 来源矩阵：20-30 篇论文/数据集，按火灾、滑坡/泥石流、弱监督、变化检测、基础模型分组。
2. 方法矩阵：输入、标签、输出、模型家族、指标、数据划分、代码/数据开放性、与本地数据相关性。
3. 本地数据兼容表：需要哪些波段、DEM/SAR/灾前灾后条件、标签精度假设、预期失败模式。
4. 三个独立部门提案：
   - research.lead 提出研究假设和项目框架；
   - technical.lead 提出实现架构和实验基础设施；
   - qa.tester 提出评估、泄漏检查和验收门；
5. 修订版 BrainstormMemo，保留竞争路线，而不是过早收敛。
