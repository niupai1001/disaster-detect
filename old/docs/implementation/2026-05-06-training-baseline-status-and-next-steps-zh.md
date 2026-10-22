# SegmentationTrainingBaseline 当前状态与下一步

日期：2026-05-06

## 目的

本文记录第一次 `SegmentationTrainingBaseline v0.1` 云端训练尝试后的项目状态，并汇总近期实现、QA、评审和训练讨论中的核心结论。

本文刻意不记录云端环境配置和运行错误排障细节。

## 当前项目状态

项目现在已经有一套云端优先的分割 baseline 源码流程：

- `segmentation_contract` 用于构建和校验数据契约。
- `segmentation_training` 用于校验模型 manifest、构建训练 bundle、运行 E0-E3 实验、写出指标并生成预测预览。
- 仓库已由 Git 管理并推送到 GitHub，但不包含本地数据、生成 artifact、raster、checkpoint 或训练输出。
- 本地 Git 已忽略 `database/`、`.agent-team/`、raster 文件、checkpoint 和训练输出。
- 当前训练 baseline 使用安全输入契约：`F16` 和 `F17`。

当前数据契约包含：

- `1,252` 条 model-input 记录。
- C2 泥石流记录：`500`。
- C5 火灾记录：`752`。
- train 记录：`906`。
- validation 记录：`234`。
- test 记录：`112`。
- test split 仍然封存，不能用于路线选择。

当前 `F16/F17` 的 train/validation raster 体量约为：

- E1 C5 train + validation：`2.61 GB`。
- E2 C2 train + validation：`636.61 MB`。
- E3 C2/C5 train + validation：`3.23 GB`。

## 已实现的 Baseline

当前实验矩阵：

| 实验 | 模型 | 类别范围 | 输入 | 作用 |
| --- | --- | --- | --- | --- |
| E0 | trivial baseline | C2/C5 | 仅 masks | 指标地板线和 sanity check |
| E1 | U-Net | 二分类 C5 火灾 | `F16/F17` | 首个火灾信号探针 |
| E2 | U-Net | 二分类 C2 泥石流 | `F16/F17` | 首个泥石流信号探针 |
| E3 | U-Net | C2/C5 多分类 | `F16/F17` | 单模型可行性与类别混淆检查 |

代码中已经有 ResUNet adapter，但当前 E1-E3 配置使用的是 `model.family: unet`。

云端训练 runner 当前包含：

- windowed training，并对小 raster 做 padding；
- epoch 和 batch 级训练进度日志；
- epoch 后 validation 指标；
- 非有限 raster 归一化防护；
- 全 ignore training window 跳过；
- 非有限 loss 防护；
- 指标与预测预览写出。

## E1 观察结果

当前 C5 火灾二分类 U-Net baseline 没有在 validation 上产生有效的前景检测。

观察到的模式：

- 约十个 epoch 后，validation loss 稳定在 `0.083` 左右；
- `mean_iou` 一直是 `0.0000`；
- `foreground_recall` 一直是 `0.0000`。

这是一个有意义的结果：模型看起来找到了一个低 loss 的背景主导解，而不是学到了有用的 C5 前景信号。在类别不平衡条件下，低 validation loss 本身不能说明模型成功。

## 解释

当前 E1 结果说明：在只使用 `F16/F17` 和当前 loss/sampling 设置时，baseline 还没有展示出可用的 C5 前景学习信号。

下一步最需要调查的可能原因：

1. **前景稀疏和类别不平衡**
   模型可能通过预测全背景来最小化 loss。

2. **窗口采样不够偏向前景**
   随机窗口可能包含太少正样本 mask 面积，尤其是在 padding 和小窗口处理后。

3. **loss 对前景不够敏感**
   普通 cross entropy 在背景占优时可能奖励背景主导预测，除非谨慎使用 class weights 或前景感知 loss。

4. **指标编码或预测解码需要审计**
   `mean_iou=0` 和 `foreground_recall=0` 非常严重，下一步应确认预测是否真的全背景，还是指标/label 编码压制了前景。

5. **`F16/F17` 本身可能不足**
   这仍然可能成立，但在检查 sampling、loss weighting 和 prediction previews 之前，不应直接下结论。

## 暂时不要下的结论

现在不要直接认定：

- U-Net 不适合本项目；
- C5 不可学习；
- 需要立刻升级到 transformer 或 foundation model；
- `F16/F17` 一定无用；
- 数据契约无效。

第一次 E1 信号失败应触发诊断工作，而不是立即升级模型家族。

## 推荐下一步

下一步应先对 E1 做聚焦诊断，再决定是否将 E2/E3 作为路线选择证据。

推荐诊断任务：

1. 写出 E1 预测诊断报告：
   - 预测前景像素数；
   - 标签前景像素数；
   - 预测/标签面积比；
   - per-sample foreground recall；
   - 全背景预测样本数。

2. 检查 validation prediction previews：
   - 确认预测是否全背景；
   - 对比失败样本的 label mask 与 F16/F17 面板；
   - 检查输入通道中前景是否有可见信号。

3. 增加前景感知训练控制：
   - foreground-biased window sampler；
   - 对部分训练窗口设置最小正样本像素阈值；
   - 从 train masks 计算 class weights；
   - 可控地加入 Dice 或 CE+Dice，不作为隐藏默认行为。

4. 以 `E1b` 形式重新运行，每次只改变一个因素。

5. 只有在 E1b 后，才应解释 E2/E3 对路线选择的意义。

## 建议的下一部门指令

下一有效部门：technical 或 implementation，取决于团队是想先出计划还是直接打补丁。

建议技术部门指令：

```text
技术部门开始干活：基于 E1 的 validation_loss 低但 mean_iou/foreground_recall 为 0 的现象，制定 E1 diagnostic and foreground-aware sampling/loss plan；不要直接升级模型家族。
```

建议开发部门指令：

```text
开发部门实现：为 SegmentationTrainingBaseline 增加 E1 预测诊断报告、foreground-biased window sampler、class-weight computation 和 E1b 配置；保持 test split 封存。
```

