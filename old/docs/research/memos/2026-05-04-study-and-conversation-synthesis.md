# 调研学习与对话关键信息整理

日期：2026-05-04

状态：头脑风暴前整理文档

## 本文档用途

本文档汇总本轮遥感灾害范围识别项目中已经学习到的论文/资料内容，以及本次对话中形成的关键流程要求、项目约束和后续检查项。它不是正式 `BrainstormMemo`，也不启动任何新部门任务。后续若继续多智能体流程，应先以本文档和已有学习日志作为共享上下文。

## 本次对话形成的关键要求

1. 先清理项目中的无用 Markdown 和中间产物，再进行研发学习与调研。
2. 研发学习要与用户同步，学习到的每篇论文或主要来源都要写 Markdown 学习日志。
3. 学习日志不能只写“对项目有什么用”，还必须先讲清楚论文本身：
   - 这篇论文旨在解决什么问题；
   - 使用了什么数据；
   - 使用了什么方法；
   - 创新点是什么；
   - 实验结论与局限；
   - 对本项目的启发；
   - 对后续头脑风暴的约束；
   - 下一步要查什么。
4. 在完成调研与学习后，最新要求是：先不急着头脑风暴，而是把学习内容和本次对话关键信息整理成一个 Markdown。
5. 多智能体部门规则仍然有效：部门和 agent 只有在用户或 coordinator 明确分配后才行动；可见上下文不是 assignment。

## 已完成的清理

已删除的无用/中间文件类型：

- macOS 生成的 `.DS_Store`；
- `docs/archive/2026-05-04-superseded-*` 下明确过期的 Markdown 中间稿。

保留的内容：

- `.agent-team/` 台账及其 artifact，因为它们是多智能体流程历史记录；
- `docs/research/learning-logs/` 下的学习日志；
- `docs/research/matrices/` 下的方法/来源矩阵；
- `docs/literature/` 下的方法图谱；
- 研发 memo 与本整理文档。

## 技能与工具使用记录

- 使用了项目的 `multi-agent-team` 工作流规则，确保不把可见上下文误当成部门任务。
- 使用了 `find-skills` 查询文献综述相关技能，发现 `davila7/claude-code-templates@literature-review` 等 literature review 技能可供参考；本项目未全局安装新技能，而是吸收其结构化学习记录思路。
- 本地 `OPENAI_API_KEY` 未设置，因此不能依赖 LLM provider 自动生成高质量 artifact；已手工把占位 artifact 替换为基于学习日志的实稿。

## 本地项目上下文

项目目标不是图像级分类，而是从遥感影像中定位并勾画灾害范围。

已知本地数据线索：

- 原始数据位于 `database/`。
- 预处理数据位于 `datasets/`，尤其是 `datasets/c5_fire_yolo/`。
- `datasets/c5_fire_yolo/data.yaml` 当前只有一个 YOLO 类：`0: fire`。
- `datasets/c5_fire_yolo/manifest.csv` 关联图像、YOLO label、bbox、F07/F11/F12 源波段路径和 `polygon_wkt`。
- `database/csv文本/` 下的灾害 CSV 包含 WKT polygon 和类别字段。
- 原始 `.tif` 文件名似乎包含 event date、image date、tile/grid id 和 spectral band id。

本地数据的关键风险：

- 当前 YOLO 数据集只暴露 fire 类，不能代表整个项目目标；
- WKT polygon 未经审计前不能默认是精确像素级真值；
- F07/F11/F12 的物理波段含义需要映射；
- raster、mask、bbox、polygon、preview 之间的 CRS 和像素对齐需要验证；
- event date 和 image date 可能支持变化检测，但需要先做配对审计；
- 泥石流/滑坡路线可能需要 DEM、slope、aspect、curvature、flow accumulation 等地形特征。

## 学习材料总览

| 来源 | 学习日志 | 方法族 |
| --- | --- | --- |
| Land8Fire | `docs/research/learning-logs/2026-05-04-land8fire-wildfire-segmentation.md` | 多光谱火灾分割 benchmark |
| Landslide4Sense | `docs/research/learning-logs/2026-05-04-landslide4sense-benchmark.md` | 光学 + 地形数据滑坡分割 |
| Sparse Annotations / FESTA | `docs/research/learning-logs/2026-05-04-sparse-annotations-festa.md` | 遥感弱监督分割 |
| SatMAE | `docs/research/learning-logs/2026-05-04-satmae-pretraining.md` | 多光谱/多时相自监督预训练 |
| xBD | `docs/research/learning-logs/2026-05-04-xbd-disaster-change-detection.md` | 灾前/灾后损毁评估 |
| Segment Anything | `docs/research/learning-logs/2026-05-04-sam-segment-anything.md` | promptable mask proposal |
| SAFE | `docs/research/learning-logs/2026-05-04-safe-burned-area-extraction.md` | burned-area 候选生成 + SAM/SAM2 细化 |
| Prithvi | `docs/research/learning-logs/2026-05-04-prithvi-geospatial-foundation-model.md` | 遥感基础模型迁移 |

## 论文/来源学习内容

### Land8Fire

Land8Fire 解决的是 wildfire semantic segmentation 缺少可靠 benchmark 的问题。它认为已有火灾遥感数据集常有真值粗糙、光谱覆盖不足、火像素与背景严重不平衡等问题。它的数据来自 Landsat 8 多光谱 patch，包含超过 20,000 个多光谱图像 patch，并提供人工校验的 fire mask。

方法上，它用 CVAT 等工具对基于 Schroeder 条件生成的火点 mask 做人工验证，然后裁剪成 256 x 256 patch。模型层面比较 UNet、DeepLabV3+、SegFormer、Mask2Former 等 CNN/Transformer 分割模型；实验变量包括 cross-entropy、focal loss 和不同 Landsat 8 波段组合，特别关注 NIR、SWIR1、SWIR2。

创新点在于：发布人工校验的多光谱火灾分割数据集；提供模型、loss、波段组合的系统 benchmark；强调 NIR/SWIR 对火灾检测的重要性；单独分析 fire pixel imbalance 和 clustered fire 等任务难点。

对本项目的关键启发是：火灾方向不能只沿用 `datasets/c5_fire_yolo` 的 bbox 思路，应转向多光谱 mask/segmentation benchmark；F07/F11/F12 和可能的 NIR/SWIR/指数通道必须保留；focal loss 不能凭经验直接定，需要实验验证。

### Landslide4Sense

Landslide4Sense 解决的是遥感滑坡检测缺少统一公开 benchmark 的问题。滑坡边界在光学影像中不稳定，受土壤、植被、阴影、灾后恢复阶段和传感器条件影响。它的数据包含 3,799 个 patch，融合 Sentinel-2 optical layers、ALOS PALSAR 派生 DEM 和 slope，标签是 landslide/non-landslide mask，覆盖 Iburi、Kodagu、Gorkha、Taiwan 等地区和事件。

方法上，它把光学波段、DEM 和 slope 作为多源输入，比较 11 个分割模型，包括 U-Net、ResU-Net、PSPNet、DeepLab-v2、DeepLab-v3+ 等，并强调空间独立测试。

创新点在于：把 Sentinel-2 光学数据与 DEM/slope 结合；面向滑坡边界检测而不是粗定位；跨多个区域和事件建立 benchmark；明确地形信息对滑坡边界识别的价值。

对本项目的关键启发是：泥石流不能简单当成“另一类火灾”。它更接近 landslide/mass movement，需要地形、纹理、沟谷和汇流等特征。如果本项目没有 DEM/slope，泥石流路线应被标注为受限探索路线。

### Sparse Annotations / FESTA

Sparse Annotations / FESTA 解决的是遥感语义分割过度依赖高质量像素级标注的问题。非常高分辨率遥感影像精细标注昂贵且需要专业解译人员，因此论文研究用 scribble 等 sparse annotation 训练分割模型。

方法上，它只在有 scribble 标签的像素上计算 supervised segmentation loss，并提出 FESTA，即 FEature and Spatial relaTional regulArization，用空间邻域关系和特征相似关系为未标注区域提供正则，而不是把未标注区域直接当背景。

创新点在于：证明稀疏标签可以训练遥感分割模型；用空间和特征关系补足监督；专门适配 remote sensing imagery；提示训练管线必须支持 partial label、ignore index 和不确定区域。

对本项目的关键启发是：WKT polygon 不能直接 rasterize 成完美 hard mask。更稳妥的做法是把 polygon 转成 reliable foreground、ignore boundary、reliable background，并在训练中支持 ignore index 和 label confidence。

### SatMAE

SatMAE 解决的是自然图像 MAE 不能充分适配卫星遥感多时相、多光谱结构的问题。它面向 temporal 或 multispectral satellite imagery，利用大量无标注遥感数据做 masked autoencoder 预训练，再迁移到 land cover classification、semantic segmentation 等下游任务。

方法上，它以 MAE 为基础，随机遮盖 image patches 后重建；加入 temporal embedding 处理时间维；用 spectral positional encoding 处理不同波段组；预训练后再 fine-tune 或 transfer。

创新点在于：不是把卫星影像硬压成 RGB，而是在模型中显式表达时间和光谱结构；使用 spectral positional encoding 让模型理解波段差异；把遥感无标注数据转化为下游任务的预训练优势。

对本项目的关键启发是：如果本地有大量未标注 TIFF，多光谱/多时相自监督预训练或特征迁移值得保留；不要过早把本地 TIFF 压成三通道 JPG，否则会丢掉 SatMAE 类方法的主要价值。

### xBD

xBD 解决的是灾后建筑损毁评估缺少大规模标准化训练数据的问题。它的数据包含灾前/灾后 satellite imagery、building polygons、ordinal damage level、卫星元数据，以及 fire、water、smoke 等环境因素 bounding boxes。数据集包含 850,736 个建筑标注，覆盖 45,362 km2。

方法上，xBD 主要贡献是数据集与任务框架：为同一地区组织 pre-event/post-event 影像对，把灾害识别拆为 localization 与 damage classification/change assessment，而不是只做单图分类。

创新点在于：数据规模大；把 pre/post change detection、object localization 和 ordinal damage label 放进同一框架；强调灾害响应实际价值；保留 satellite metadata，便于分析影像条件。

对本项目的关键启发是：很多灾害任务不是单图像识别，而是变化理解。火灾烧毁区、泥石流、滑坡都可能更适合 change detection。本项目应把 event key 和 image date 作为一级字段，先审计是否存在可用灾前/灾后影像。

### Segment Anything

Segment Anything 解决的是通用图像分割模型难以跨任务、跨数据集泛化的问题。它提出 promptable segmentation task、promptable segmentation model 和大规模 SA-1B 数据集。SA-1B 包含 11M images 和超过 1B masks。

方法上，SAM 由 image encoder、prompt encoder 和 mask decoder 组成，输入 prompt 可以是 point、box、mask 等，输出是与 prompt 对应的 object mask proposal。它不是语义类别分割器，而是 promptable mask proposal 模型。

创新点在于：统一 promptable segmentation；构建极大规模 mask 数据集；用 prompt encoder 整合点、框、mask 等提示；面向 zero-shot transfer。

对本项目的关键启发是：SAM 可以帮助从 WKT polygon 或 bbox 生成候选 mask，也可用于粗标注细化和人工交互，但不能把 SAM 输出当 ground truth。SAM 不天然理解 fire、debris flow 或多光谱遥感数据。

### SAFE

SAFE 解决的是快速、高精度 wildfire burned-area extraction 的问题。传统火灾监测常在 MODIS/VIIRS 高时效但低空间分辨率，与 Sentinel-2 高空间分辨率但处理成本较高之间取舍。SAFE 结合 Sentinel-2 optical imagery、MODIS MCD64A1 burned area product、MODIS/VIIRS hotspots 和 Dynamic World land-cover data。

方法上，它先用 MODIS/VIIRS 火点和 Sentinel-2 生成候选区域，再用 SAM/SAM2 对候选区域进行细化分割，并通过 land-cover、spectral indices 和局部规则进行后处理。实验覆盖 grassland 和 forest wildfire 场景。

创新点在于：把热点产品时效性和 Sentinel-2 空间细节结合；把 SAM 用作候选 burned area 精细化工具；先缩小候选区域再运行分割以降低计算成本；SAFE 输出还可用于训练区域轻量模型。

对本项目的关键启发是：火灾路线可以考虑“候选生成 -> 分割细化 -> 光谱/地理 QA -> 轻量模型训练”的闭环。SAM 的合理位置是候选区细化，而不是替代遥感物理特征。

### Prithvi

Prithvi 解决的是通用 AI foundation model 与遥感地球观测数据结构不匹配的问题。普通视觉基础模型多基于自然图像，而遥感影像具有多光谱、多时相、大范围地理覆盖和传感器物理约束。Prithvi 使用 Harmonized Landsat and Sentinel-2 imagery 进行遥感基础模型预训练，并开源到 Hugging Face。

方法上，Prithvi 是 temporal Vision Transformer / geospatial foundation model，预训练方式类似 masked modeling，从遮盖信息中学习地表时空光谱表示，再通过 fine-tuning 迁移到 flood mapping、fire burn scar detection、crop classification 等任务。

创新点在于：使用 HLS 多光谱地球观测数据，而不是自然图像数据；NASA/IBM 开源模型和 workflow；在 flood 和 burn scar 等任务中显示出较好的数据效率；一个 base model 可支持多个地球科学任务。

对本项目的关键启发是：对多光谱、小标注数据，Prithvi/SatMAE 类遥感基础模型可能比 SAM 更适合作为 backbone 或 feature extractor。但前提是本地 TIFF 能映射到 HLS/Landsat/Sentinel 类输入 schema。

## 横向方法认识

### 火灾/烧毁区

火灾路线不能被简化成 YOLO 检测。应至少保留：

- 多光谱分割 benchmark；
- NIR/SWIR 和 NBR、dNBR、BAIS2、NDVI、NDMI 等指数或物理 baseline；
- SAFE 式候选生成与细化；
- bbox fallback 或 SAM prompt，而不是 bbox 主线。

### 泥石流/滑坡

泥石流/滑坡路线不能挂在火灾路线后面。它需要单独讨论：

- DEM、slope、aspect、curvature、flow accumulation；
- OBIA/classical ML；
- terrain-aware segmentation；
- 跨区域、跨事件泛化；
- 如果没有地形数据，需要明确限制。

### 粗标注和弱监督

本项目的 WKT polygon 需要被视为待审计、可能粗糙的标注。更合理的标签结构是：

- reliable foreground；
- ignore boundary；
- reliable background；
- label confidence；
- partial label / ignore index。

### 变化检测

event date 和 image date 是重要资产，不应只藏在文件名里。若可构建灾前/灾后配对，应保留 change detection 路线；若不能，也要明确审计结论。

### 基础模型

基础模型不能混成一类：

- SAM：自然图像 promptable segmentation，更适合 proposal/refinement；
- SAFE：把 SAM/SAM2 放进遥感火灾候选区提取流程；
- SatMAE：遥感多光谱/多时相自监督预训练；
- Prithvi：HLS/Landsat/Sentinel 风格 geospatial foundation model。

基础模型输出不能当 ground truth，必须通过光谱、几何、人工或 QA 流程复核。

## 当前应保留的候选路线

| 路线 | 为什么保留 | 第一证据 |
| --- | --- | --- |
| 光谱指数 baseline | 火灾和植被变化有物理先验 | 波段映射、NBR/NDVI/SWIR 叠加图 |
| OBIA/classical ML | 适合地形/对象驱动的泥石流候选 | DEM/slope 可用性、对象特征 |
| 全监督分割 | 直接输出 mask/polygon | mask 对齐、标签质量、划分策略 |
| 弱监督 | 匹配粗 WKT 现实 | label-noise 审计、ignore-zone 设计 |
| 变化检测 | 很多灾害是时间变化现象 | event/image date 配对审计 |
| SAM 辅助 proposal | 可从框或 polygon 细化候选 mask | prompt 质量、光谱 QA、人工复核负担 |
| 遥感基础模型迁移 | 适合小标注多光谱数据 | 本地波段到 HLS/Landsat/Sentinel schema 的映射 |
| 检测退路 | 快速粗定位和 prompt 生成 | bbox 质量、class balance、mask-to-box 质量 |

## 后续头脑风暴前必须准备的审计

1. 波段审计：确认 F07/F11/F12 和其他 TIFF 波段的物理意义、尺度、分辨率。
2. 标注审计：比较 `polygon_wkt`、mask、bbox、YOLO label、preview overlay。
3. CRS/对齐审计：确认 raster、polygon、mask、bbox 在地理坐标和像素坐标上没有错位。
4. 时间审计：解析 event date 与 image date，判断是否存在有效灾前/灾后配对。
5. 地形审计：确认 DEM/slope/aspect/curvature/flow accumulation 是否能获取并对齐。
6. 类别审计：枚举 CSV 中灾害类别，确认 fire、泥石流、其他类别的样本量和不平衡。
7. 划分审计：制定按事件、日期、区域、tile 和相邻 patch 防泄漏的 split 规则。
8. 可视化审计：为每条 baseline 或候选路线先生成 contact sheet 和 overlay，再讨论数值指标。

## 流程状态说明

当前 `.agent-team` 中已有以下任务记录：

- `task-b2d1d7cb8a5e`：研发学习/调研任务，已完成，产出 `ResearchMemo`。
- `task-7a02ad0777e6`：此前按上一轮要求创建过 evidence-grounded brainstorm 任务，并生成了相关 artifact。

根据最新指令，接下来不继续推进任何新头脑风暴或部门任务。`task-7a02ad0777e6` 的 artifact 可作为历史参考，但后续正式流程应以用户再次明确下达“头脑风暴”或对应部门指令为准。

## 当前结论

本项目现在最重要的不是立刻选模型，而是把遥感灾害范围识别拆成可验证的问题：

- 火灾是否能做多光谱分割；
- 泥石流/滑坡是否具备地形数据条件；
- WKT polygon 是否足以训练，还是只能作为弱监督；
- 本地文件名和元数据是否支持变化检测；
- SAM、SatMAE、Prithvi 应作为哪一类工具进入流程；
- YOLO 是否只作为 fallback，而不是主路线；
- 所有路线是否都有可视化和数据审计证据。

后续如果继续多智能体流程，建议先让技术部门基于本文档创建“数据审计与实验 readiness plan”，而不是直接实现或训练模型。
