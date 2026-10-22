# 学习日志：Land8Fire wildfire segmentation benchmark

日期：2026-05-04

来源：

- Paper: https://www.mdpi.com/2072-4292/17/16/2776
- GitHub: https://github.com/UARK-AICV/Land8Fire

## 论文旨在解决什么问题

这篇论文认为卫星火灾分割研究受三个问题限制：已有数据集的真值粗糙、光谱覆盖不足、火像素与背景严重不平衡。它的目标不是做图像级火灾分类，而是建立一个可复现的 wildfire semantic segmentation benchmark，让研究者能在同一数据、同一划分和同一评价指标下比较模型、损失函数和波段组合。

## 使用的数据

- 数据来自 Landsat 8，多光谱影像 patch。
- 数据集包含超过 20,000 个多光谱图像 patch，并提供人工校验的 fire mask。
- 数据构建基于 ActiveFire 的 Landsat 8 影像，再通过人工标注/校验提高 ground truth 可靠性。
- Landsat 8 原有 11 个波段，论文排除分辨率不同的全色 Band 8，重点使用 10 个 30 m 波段。
- 论文强调火像素分布覆盖 small、scattered、clustered fire 等不同形态，而不是只追求地理或季节多样性。

## 使用的方法

- 数据层面：用 CVAT 等标注工具对 Schroeder 条件生成的火点 mask 进行人工验证，再通过连通域/边界框扩展裁剪成 256 x 256 patch。
- 模型层面：比较 CNN 与 Transformer 分割模型，包括 UNet、DeepLabV3+、SegFormer、Mask2Former 等。
- 实验变量：比较 cross-entropy 与 focal loss，也比较不同 Landsat 8 波段组合，特别关注 NIR、SWIR1、SWIR2。
- 评价层面：使用 IoU、F1、precision、recall、mAccuracy 等分割指标，并讨论只看 accuracy 会掩盖漏检的问题。

## 创新点

1. 数据创新：提供人工校验的 Landsat 8 多光谱火灾分割数据集，补足已有火灾数据集 ground truth 粗糙的问题。
2. Benchmark 创新：不只发布数据，还给出模型、loss、波段组合的系统比较。
3. 光谱分析创新：明确指出 SWIR1、SWIR2 和 NIR 对火灾检测尤其关键，不能只依赖 RGB 或假彩色图。
4. 任务诊断创新：论文把 fire pixel imbalance 和 clustered fire 情况单独拿出来分析，说明 focal loss 对小目标有帮助但在 clustered fires 中可能降低 recall。

## 实验结论与局限

- 多光谱信息显著重要，NIR/SWIR 通道在烟、云或复杂背景下更有价值。
- focal loss 不是自动更优；在火区聚集时可能比 cross-entropy 更容易漏掉部分火像素。
- 论文数据更偏 active wildfire/fire pixels，不完全等同于 post-fire burned-area mapping。
- 数据分辨率和任务定义受 Landsat 8 约束，迁移到本项目数据时必须重新检查波段、空间分辨率和标注语义。

## 对本项目的启发

1. 火灾方向不能只沿用 `datasets/c5_fire_yolo` 的 bbox 思路，应转向 mask/segmentation benchmark。
2. 本项目要保留源波段，并设计波段组合实验：假彩色、F07/F11/F12、NIR/SWIR、指数通道都应可比较。
3. 第一批实验不要只看一个模型，要设计小型 baseline set。
4. 损失函数必须实验验证，不能凭“类别不平衡所以 focal loss”直接定案。

## 对头脑风暴的约束

- YOLO 只能是 fallback，不应作为火灾主路线。
- 火灾主路线应包含“多光谱分割 benchmark”。
- 需要把波段组合当作核心研究变量。

## 下一步要查

- Land8Fire 的精确 split 策略是否按地区/事件避免泄漏。
- 它的 GitHub 数据与代码是否完整可复现。
- 本地 `F07/F11/F12` 与 Landsat/Sentinel 常见 NIR/SWIR 波段的对应关系。
