# 灾后多光谱分割训练阶段总结与下一步目标

日期：2026-05-07

## 当前项目方向

项目目标已经收敛为灾后识别与像素级分割，不再把灾前预测作为当前阶段目标。

本阶段的核心判断是：反复出现零前景召回并不能简单归因于 U-Net 无效，更重要的问题是原训练合同对数据管线利用不足。原有训练主要依赖 `F16/F17`，而论文和本地数据都说明灾后火灾、泥石流等目标更需要 optical、NIR、SWIR 和指数特征。

## 本阶段完成内容

训练输入已经从弱输入路线切换为灾后多波段路线。

当前 V3 输入栈为 11 通道：

```text
F01 F02 F03 F04 F07 F11 F12 NDVI NBR NDMI BRIGHTNESS
```

已完成的工程能力包括：

- 灾后同日期 scene 选择。
- 云雾质量代理排序。
- 多波段重采样到 mask 网格。
- 派生指数通道生成。
- C5 fire 训练 bundle 构建。
- RTX 4080 训练配置。
- AMP、`channels_last`、`cudnn_benchmark`、`cache_records`。
- 11 通道可视化横条 bug 修复。
- `best_mean_iou.pt` 和 `best_mean_iou.json` 保存。
- 阈值扫描扩展到 `0.9`。

相关训练分支：

```text
codex/rtx4080-training-performance
```

## 训练结果观察

U-Net baseline 已经不再是零前景召回，说明多波段输入确实让模型获得了可学习信号。

当前 stable run 的主要结果：

```text
last mean_iou          ≈ 0.2501
best mean_iou          ≈ 0.3041
foreground_recall      ≈ 0.83
foreground_precision   ≈ 0.26
predicted/label area   ≈ 3.17x
```

对比前一版 run：

```text
previous last mean_iou ≈ 0.2176
previous best mean_iou ≈ 0.3536
stable last mean_iou   ≈ 0.2501
stable best mean_iou   ≈ 0.3041
```

stable 配置让训练观测更清楚，最终指标略有改善，但没有形成决定性突破。

## 关键结论

本阶段最大的提升是观测能力，而不是模型能力。

修复后的可视化显示，模型并非完全学不到火灾前景。它能覆盖一部分火灾主体，但会把大量相似地物、裸地、山地纹理、亮色道路、云雾边缘等预测为前景。

当前主要问题是：

```text
召回不低，但精度很低；
预测前景面积约为真实标签 3 倍；
训练曲线仍然震荡；
普通 U-Net 的表达能力和上下文建模能力不足。
```

因此，继续在当前 U-Net 上小幅调 batch、loss 或学习率，预期收益有限。

## 下一步目标

下一阶段目标应调整为更强分割架构对照实验。

目标描述：

```text
在固定 V3 11 通道输入、同一 split、同一验证集和同一可视化/阈值扫描协议的前提下，引入更强分割架构，验证是否突破当前 U-Net 上限。
```

推荐实验名：

```text
SegmentationModelV4: stronger architecture comparison
```

## 下一步优先路线

第一优先级是低成本架构升级：

- `ResUNet`
- `UNet++`
- `DeepLabV3+`

其中 `DeepLabV3+` 或 `UNet++` 更值得优先考虑，因为当前错误主要集中在边界、多尺度上下文和背景误报上。

第二优先级是预训练 encoder 或 SegFormer：

- SegFormer
- 预训练 encoder + 11 通道 adapter
- 遥感 foundation model encoder

这一路线潜在收益更大，但工程复杂度也更高，需要处理 11 通道输入适配和预训练权重迁移。

## 下一阶段验收标准

下一阶段不应只看单个 `mean_iou`，而应输出架构对照表：

```text
model_family
best_mean_iou
foreground_precision
foreground_recall
foreground_dice
predicted/label area ratio
best checkpoint epoch
validation_contact_sheet
```

最低可接受目标：

```text
best_mean_iou > 0.35
foreground_precision 明显高于 0.26
predicted/label area ratio 从约 3x 降到接近 1-2x
可视化中明显减少大面积背景误报
```

如果更强 CNN 架构仍无法突破当前上限，则应进入 SegFormer 或预训练 encoder 路线，而不是继续微调普通 U-Net。

