# 方法/来源矩阵：第 1 轮

日期：2026-05-04

## 学习日志标准

本轮之后，每篇论文或主要来源的学习日志必须至少包含：

- 论文旨在解决什么问题；
- 使用的数据；
- 使用的方法；
- 创新点；
- 实验结论与局限；
- 对本项目的启发；
- 对头脑风暴的约束；
- 下一步要查。

## 来源矩阵

| 来源 | 学习日志 | 任务 | 输入数据 | 标签/监督 | 输出 | 方法族 | 开放性 | 对本项目意义 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Land8Fire | `docs/research/learning-logs/2026-05-04-land8fire-wildfire-segmentation.md` | wildfire segmentation benchmark | Landsat 8 multispectral patches | 人工校验 fire mask | fire mask | UNet / DeepLabV3+ / SegFormer / Mask2Former benchmark | Paper + GitHub | 火灾应做多光谱分割 benchmark，不应只做 YOLO bbox |
| Landslide4Sense | `docs/research/learning-logs/2026-05-04-landslide4sense-benchmark.md` | landslide detection | Sentinel-2 optical layers + DEM + slope | landslide/non-landslide mask | landslide mask | U-Net / ResU-Net / PSPNet / DeepLab 等 | arXiv + Zenodo | 泥石流/滑坡需要 DEM/slope，不应只用 RGB/假彩色 |
| Sparse Annotations / FESTA | `docs/research/learning-logs/2026-05-04-sparse-annotations-festa.md` | 弱监督遥感分割 | VHR aerial/remote-sensing images | sparse/scribble labels | semantic segmentation mask | 稀疏标注 + FESTA 空间/特征关系正则 | arXiv + GitHub | 本地粗 WKT polygon 应按弱监督处理 |
| SatMAE | `docs/research/learning-logs/2026-05-04-satmae-pretraining.md` | 遥感自监督预训练 | temporal / multispectral satellite imagery | 自监督 masked reconstruction | transfer features / downstream segmentation | MAE / Transformer pretraining | arXiv + project | 小数据、多波段任务可考虑遥感预训练迁移 |
| xBD | `docs/research/learning-logs/2026-05-04-xbd-disaster-change-detection.md` | 灾后建筑损毁评估 | pre/post satellite imagery | building polygons + damage labels | localization + damage class | change detection / damage assessment dataset | arXiv | 本地 event/image date 可能支持灾前/灾后变化检测 |
| Segment Anything | `docs/research/learning-logs/2026-05-04-sam-segment-anything.md` | promptable segmentation | natural images | 1B+ masks from SA-1B | object mask proposal | promptable segmentation foundation model | arXiv + project | 可做候选 mask、粗标注细化或交互工具，不可直接当真值 |
| SAFE | `docs/research/learning-logs/2026-05-04-safe-burned-area-extraction.md` | burned-area extraction | Sentinel-2 + MODIS/VIIRS hotspots + land cover | remote-sensing product / candidate constraints | burned-area mask | candidate generation + SAM/SAM2 refinement + rules | MDPI | 火灾可走“热点/事件候选 -> SAM 细化 -> 光谱 QA”路线 |
| Prithvi | `docs/research/learning-logs/2026-05-04-prithvi-geospatial-foundation-model.md` | geospatial foundation model transfer | HLS / Landsat / Sentinel-like EO imagery | self-supervised + downstream labels | flood/burn scar/crop 等 downstream outputs | temporal ViT / geospatial foundation model | NASA/IBM open release | 遥感基础模型路线需要与本地波段兼容性审计 |

## 第 1 轮初步结论

- 研究视野必须至少覆盖火灾、滑坡/泥石流、弱监督、变化检测、自然图像基础模型、遥感基础模型。
- YOLO bbox 只能作为 fallback，不是主线。
- 本地下一步要审计：波段映射、DEM/slope 可用性、event/image date 是否可配对、WKT 粗细与 CRS 对齐。
- “基础模型”不能混为一谈：SAM 是 promptable mask proposal，SatMAE/Prithvi 是遥感预训练/迁移路线。
- 后续所有部门必须先吸收这些日志，再进入真正的 adversarial brainstorm。
