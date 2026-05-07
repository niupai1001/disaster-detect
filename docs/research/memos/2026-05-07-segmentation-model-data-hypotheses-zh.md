---
title: 分割模型与数据假设记录
date: 2026-05-07
tags:
  - segmentation
  - remote-sensing
  - multispectral
  - ablation
  - obsidian
status: active
task: task-6bd957f6e1ed
aliases:
  - U-Net 数据假设记录
---

# 分割模型与数据假设记录

这是一份可持续更新的项目记录，用于维护架构、数据契约、通道栈和归一化假设。它使用 Obsidian 风格 Markdown，之后可以迁移到 Obsidian vault。

相关笔记：

- [[2026-05-07-unet-channel-ablation-runbook]]
- [[2026-05-07-segmentation-v4-cloud-smoke-runbook]]
- [[2026-05-07-post-disaster-multispectral-training-summary]]

## 当前判断

当前 U-Net 结果弱，不能直接证明 U-Net 本身不适合。遥感分割论文中的 U-Net-like 模型通常在标签、输入分布、预处理和目标定义稳定时表现更好。

对本项目来说，可能原因是混合的：

- 架构容量与归纳偏置；
- 前景标签稀疏；
- 标签对齐或类别定义不确定；
- 通道选择与派生指数是否有效；
- per-window normalization 改变光谱值物理含义；
- 稀疏前景像素下的 sampler 与 threshold 行为。

> [!important]
> 下一步应该把架构比较和 U-Net 通道消融作为两条独立诊断轴。不要让更强架构掩盖数据契约问题。

## 多光谱输入如何进入模型

当前训练路径会为同一个空间 window 读取每个选定 raster channel，归一化后按 channel-first 方式堆叠：

```text
single sample: (channels, height, width)
batch:         (batch, channels, height, width)
```

当前 11-channel 路线是：

```text
F01,F02,F03,F04,F07,F11,F12,NDVI,NBR,NDMI,BRIGHTNESS
```

U-Net 第一层卷积使用 `input_channels: 11` 构建，所以模型接收多光谱数据的方式和普通 RGB 网络接收 3 通道图像类似，只是这里有 11 个通道。

## 重要注意：归一化

云端训练路径目前对每个 band 的每个读取 window 做 2%-98% percentile normalization。这会让数值稳定，但可能削弱绝对物理含义。对于 NDVI、NBR、NDMI 这类指数，模型看到的可能是局部相对对比，而不是全局一致的指数尺度。

如果出现以下情况，这会成为强候选原因：

- 7 个 raw optical/SWIR channels 优于 11 channels；
- indices-only 表现异常；
- validation contact sheets 显示前景预测更像被纹理驱动，而不是光谱分离驱动。

## 架构路线

当前 architecture runway 包括：

- U-Net；
- ResUNet；
- Attention U-Net；
- UNet++；
- DeepLabV3+；
- Transformer-UNet。

六个架构已经在 commit `247b1c6` 上通过 cloud smoke，并保持 `test_split_read: false`。Smoke 只证明运行链路和产物生成，不是模型质量证据。

## 数据与通道路线

U-Net-only ablation 固定模型，只改变输入通道栈，用来隔离数据与通道行为。

计划通道组：

| Group | Channels | 假设 |
|---|---|---|
| RGB optical | `F04,F03,F02` | 简单可见光信号可能优于噪声较高的高维通道栈。 |
| False color | `F07,F04,F03` | NIR 可能更好地区分植被、烧毁、泥石流或水体相关上下文。 |
| Base optical/SWIR | `F01,F02,F03,F04,F07,F11,F12` | 在 per-window normalization 下，raw bands 可能优于派生指数混合。 |
| Indices only | `NDVI,NBR,NDMI,BRIGHTNESS` | 手工物理指数可能承载主要可用类别信号。 |
| Current 11-channel stack | `F01,F02,F03,F04,F07,F11,F12,NDVI,NBR,NDMI,BRIGHTNESS` | 当前数据契约的对照组。 |

## 决策记录

- 2026-05-07：Architecture smoke 已接受；完整模型 benchmark 只允许作为 validation-only 云端运行。
- 2026-05-07：新增 U-Net-only channel ablation configs，用于验证 U-Net 弱表现是否和数据/通道有关。
- 2026-05-07：暂缓 normalization-mode 代码变更，等通道消融返回证据后再做，因为改归一化会改变 loader 行为并引入第二个处理轴。

## 下一步需要收集的证据

- U-Net channel ablation smoke summary；
- U-Net channel ablation validation-only full summary；
- 每个 run 的 `test_split_read: false`；
- validation contact sheets；
- threshold sweep 行为；
- foreground window coverage；
- predicted/label foreground area ratio；
- 与当前 11-channel stack 失败模式不同的通道组。

## 解释规则

如果更简单的通道组优于当前 11-channel stack，先调查通道噪声、指数有效性、归一化和样本质量，再继续升级架构复杂度。

如果所有通道组都类似失败，先调查标签、对齐、前景稀疏性、sampler 覆盖、loss weighting 和 thresholding，再假设是模型 family 限制。

