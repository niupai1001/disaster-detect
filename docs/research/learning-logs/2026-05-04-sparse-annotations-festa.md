# 学习日志：Semantic Segmentation of Remote Sensing Images with Sparse Annotations

日期：2026-05-04

来源：

- Paper: https://arxiv.org/abs/2101.03492
- Code: https://github.com/Hua-YS/Semantic-Segmentation-with-Sparse-Labels

## 论文旨在解决什么问题

这篇论文解决遥感语义分割对高质量像素级标注依赖过强的问题。非常高分辨率遥感影像的精细标注耗时、昂贵，并且常常需要专业解译人员。论文希望用更容易绘制的稀疏标注，例如 scribble，让 CNN 仍然能学习到可用的分割模型。

## 使用的数据

- 论文面向 very high resolution aerial/remote-sensing images。
- 监督形式不是完整像素级 mask，而是 incomplete/sparse annotations，主要是少量 scribble 标注。
- 任务仍是 semantic segmentation，未标注区域不直接当作背景。
- 代码仓库提供了稀疏标注训练流程，便于复现实验和迁移方法。

## 使用的方法

- 基础训练：只在有 scribble 标签的像素上计算 supervised segmentation loss。
- FESTA 方法：FEature and Spatial relaTional regulArization，用空间邻域关系和特征相似关系给未标注区域补充无监督约束。
- 核心思想：让空间相近且特征相似的像素在表示或预测上更一致，而不是强迫所有未标注像素变成背景。
- 训练输出：仍输出完整语义分割 mask，但训练信号来自稀疏标注加关系正则。

## 创新点

1. 标注形式创新：证明遥感分割不一定必须依赖密集像素级标签，scribble 等稀疏标签可以成为可训练监督。
2. 损失设计创新：FESTA 把空间结构和特征结构作为正则信号，补足稀疏标签的监督缺口。
3. 任务适配创新：专门讨论 remote sensing imagery，而不是直接从自然图像弱监督迁移。
4. 工程启发：训练管线需要支持 partial label、ignore index 和不确定区域，而不是只有 hard mask。

## 实验结论与局限

- 稀疏标注可以显著降低标注成本，并仍然训练出有竞争力的分割模型。
- 方法主要围绕 scribble；本项目的 WKT polygon 需要转换成可靠前景、边界忽略区和可靠背景，不能机械等同。
- 如果 polygon 本身偏移或类别噪声大，关系正则也不能自动修复错误真值。

## 对本项目的启发

1. 不要把 WKT polygon 直接 rasterize 后当成完美 ground truth。
2. Dataset builder 应支持三类区域：可靠前景、可靠背景、ignore boundary。
3. 粗 polygon 可以腐蚀出 reliable foreground，膨胀边缘作为 ignore zone。
4. 模型训练时应支持 partial label / ignore index。

## 对头脑风暴的约束

- 必须保留“弱监督/粗标注分割”路线。
- QA 要单独审计 label noise。
- 技术方案不能只输出普通 mask，还要记录 label confidence 或 ignore 区域。

## 下一步要查

- FESTA 是否适合卫星多光谱数据，还是主要针对高分辨率航空 RGB。
- 是否有 polygon-level weak supervision 的更直接实现。
- 如何把本地 WKT 自动转换为 core/boundary/background。
