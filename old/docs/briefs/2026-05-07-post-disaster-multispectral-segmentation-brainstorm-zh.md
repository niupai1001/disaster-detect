# 灾后多光谱分割头脑风暴 Brief

日期：2026-05-07

## 目标

为修正后的项目方向运行一次对抗式部门头脑风暴：

> 项目目标是灾后识别与分割。当前不把灾前预测作为目标。

当前核心问题是：如何从利用率过低的 `F16/F17` 双通道训练合同，转向真正能暴露火灾和泥石流前景信号的灾后多光谱分割路线。

## 用户方向

- 目标是灾后识别和像素级分割。
- 不优化“灾害发生前预测”。
- 反复出现零前景分割，应结合前景/背景可分性弱、现有数据管线利用率低来解释。
- 团队应重新评估是否引入更丰富的多光谱数据，而不是继续只调一个弱输入视角。

## 当前证据

- 当前训练配置仍使用 `input_channels: F16;F17`。
- training bundle 包含 1,252 条 model-input 记录：500 条 C2 泥石流，752 条 C5 火灾。
- train/validation/test 数量为 906 / 234 / 112；test split 继续封存。
- 最近 E1 C5 U-Net 训练 validation loss 约为 `0.083`，但 `mean_iou=0.0000`、`foreground_recall=0.0000`。
- C5 前景 mask 稀疏；validation 前景比例中位数约 0.3%。
- 数据合同在 `source_band_paths` 中保留了很多原始波段，包括类似 Sentinel-2 光学、SWIR，以及类似 Sentinel-1 的 `F16/F17`，但当前 `model_input_manifest.csv` 只暴露 `F16/F17`。
- 本地论文笔记强调，火灾和滑坡/泥石流分割通常受益于更丰富的 Sentinel-2 多光谱波段、SWIR/红边/NIR、NBR、dNBR、NDVI、MIRBI 等光谱指数，以及在合适场景下作为辅助模态的 SAR。

## 需要吸收的研究资料

- `/Users/Lemon/Desktop/Hermes/research/disaster-detection/README.md`
- `/Users/Lemon/Desktop/Hermes/research/disaster-detection/2023-Ghali-RS-wildfire-vit.md`
- `/Users/Lemon/Desktop/Hermes/research/disaster-detection/2023-BelenguerPlomer-RSE-burnseverity.md`
- `/Users/Lemon/Desktop/Hermes/research/disaster-detection/2022-Seydi-JAG-fire-attention.md`
- `/Users/Lemon/Desktop/Hermes/research/disaster-detection/2021-Prakash-Landslides-Swiss.md`
- `/Users/Lemon/Desktop/Hermes/research/disaster-detection/2023-HuBan-IJAG-sarmultimodal.md`
- `docs/research/memos/2026-05-04-study-and-conversation-synthesis.md`
- `docs/literature/2026-05-04-pre-project-method-landscape.md`

## 部门任务

### research.lead

基于文献重新定义灾后分割路线。挑战继续把 `F16/F17` 当作充分证据的方案。推荐 C5 火灾与 C2 泥石流分割应优先尝试的灾后多光谱通道家族和模型家族。

### technical.lead

挑战可行性。定义 `PostDisasterMultispectralContract v0.2` 的最小实现计划：通道/日期审计、灾后影像选择、多光谱重采样与对齐、派生指数通道、配置变更、模型 adapter、cloud bundle 变更。继续封存 test split。

### qa.tester

挑战可测试性。定义验收门槛，证明新合同正确使用灾后数据、避免泄漏、保持网格对齐、提升前景可分性，并在任何模型家族结论前产出可解释的 validation 证据。

## 头脑风暴问题

1. 当前 pipeline 反复零前景分割最可能的根因是什么？
2. C5 火灾第一版灾后多光谱训练栈应使用哪些源波段？
3. C2 泥石流应优先使用哪些源波段或辅助特征？
4. SAR `F16/F17` 应保留为辅助通道、只做诊断，还是排除出第一版灾后分割栈？
5. 应优先实现哪些派生指数，哪些必须依赖验证过的灾前/灾后配对，哪些可用灾后单时相？
6. 让下一轮训练具备科学意义的最小数据合同变更是什么？
7. 合同修复后第一个模型仍应使用 U-Net/ResUNet 作为测量工具，还是多波段输入一旦具备就立即引入 SegFormer/ViT？
8. QA 接受“前景/背景可分性改善”前必须看到什么证据？

## 必需输出

- `ResearchBrainstormNote`
- `TechnicalBrainstormNote`
- `QABrainstormNote`
- `BrainstormMemo`

最终 `BrainstormMemo` 必须包含独立部门立场、挑战/反驳/协商、推荐路线、实现顺序、QA 门槛、暂缓选项、决策记录和下一有效部门。

## 非目标

- 不在 brainstorm 中训练新模型。
- 不使用封存测试集。
- 不把灾前预测作为当前目标。
- 不声称 transformer 能解决输入信息缺失。
- 不把 `F16/F17` 当成最终光谱设计。
- 不把多波段扩展本身视为有效，除非通道日期、网格对齐和灾后适用性可审计。
