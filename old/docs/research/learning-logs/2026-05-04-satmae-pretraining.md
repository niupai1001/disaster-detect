# 学习日志：SatMAE

日期：2026-05-04

来源：

- Paper: https://arxiv.org/abs/2207.08051
- Project: https://sustainlab-group.github.io/SatMAE/

## 论文旨在解决什么问题

SatMAE 要解决的是卫星遥感领域缺少适配多时相、多光谱结构的自监督预训练方法。自然图像 MAE 已经证明无监督预训练能提升下游任务，但卫星影像不仅是 RGB 图，还有时间维和光谱维。论文希望让 Transformer 通过自监督预训练学习遥感影像的时空光谱表示，再迁移到 land cover classification、semantic segmentation 等下游任务。

## 使用的数据

- 数据类型是 temporal 或 multispectral satellite imagery。
- 论文利用遥感领域大量无标注数据做 masked autoencoder 预训练。
- 下游任务包括 land cover classification、semantic segmentation 等遥感任务。
- 项目页面提供代码和数据入口，方便复现或迁移。

## 使用的方法

- 基础框架：Masked Autoencoder，随机遮盖 image patches 后重建。
- 时间建模：加入 temporal embedding，并在时间维上独立 mask image patches。
- 光谱建模：把多光谱波段编码为不同 band groups，并加入 spectral positional encoding。
- 迁移方式：预训练后在有标签下游任务上 fine-tune 或 transfer。

## 创新点

1. 架构适配创新：不是把卫星影像硬压成 RGB，而是在 MAE 中显式加入时间和光谱编码。
2. 光谱位置创新：使用 spectral positional encoding 处理不同波段组，使模型理解波段差异。
3. 迁移学习创新：把遥感无标注数据转化为下游分割/分类任务的预训练优势。
4. 实验贡献：论文报告相对既有方法在监督 benchmark 和 transfer learning 下游任务上有明显提升。

## 实验结论与局限

- 自监督预训练对标注有限的遥感任务有价值。
- SatMAE 是通用遥感预训练方法，不是火灾或泥石流专用模型。
- 适配本项目前需要检查本地波段、时间结构、影像数量和 patch 构建方式是否满足预训练或 fine-tuning 要求。
- 如果只把本地 TIFF 转成 3 通道 JPG，会直接放弃 SatMAE 这类方法的主要价值。

## 对本项目的启发

1. 要审计无标注影像是否足够多，是否能作为预训练或特征提取资源。
2. 模型路线中应加入“遥感预训练/迁移学习”，而不是只有 U-Net from scratch。
3. 多光谱输入应该保留，不要过早压成 3 通道 JPG。
4. 如果后续做小样本分割，SatMAE/Prithvi 类模型值得评估。

## 对头脑风暴的约束

- 基础模型路线应分成两类：自然图像基础模型 SAM，遥感基础模型 SatMAE/Prithvi。
- 技术部门要判断本地波段是否能适配这些模型。
- QA 要检查迁移实验是否真的比普通 baseline 强，而不是只因模型名字新。

## 下一步要查

- SatMAE 支持哪些数据集和波段。
- 是否有可直接用于 segmentation fine-tuning 的开源权重和代码。
- 和 Prithvi、DOFA、RemoteCLIP 等遥感基础模型如何比较。
