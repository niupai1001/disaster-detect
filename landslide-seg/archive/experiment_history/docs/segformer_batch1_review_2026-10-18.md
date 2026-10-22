# SegFormer 第一批训练复核、失败案例与下一阶段计划

本轮结论：先修正时间标签契约和评测口径，再扩充有追溯依据的数据、做少量受控训练策略实验。保留 MiT-B0、A0/A1/B0/B1；不直接延长到 300 epoch，不立即换 B2。

本次只新增审阅脚本、诊断结果和本文。未修改生产模型、训练配置、数据缓存、标签或 checkpoint。32 个 best/last 文件（包括先导运行）在分析前后 SHA-256 一致。正式训练证据来自四组 × 三种子；重新推理及波段敏感性诊断使用 seed=1234，不能把单种子诊断写成三种子结论。

## 1. 当前到底训练了什么

已读取 `dataset_and_modules.md`，但它描述的是旧的 72 样本、12 波段双骨干路线，不能作为本批 SegFormer 的训练记录。本批以 `segformer_report_2026-10-18.md`、实际 resolved_config、metrics.jsonl、checkpoint 和当前代码为准。

| 组件 | A0 | A1 | B0 | B1 |
|---|---|---|---|---|
| MiT-B0 分层编码器 | 全量微调 | 全量微调 | 全量微调 | 全量微调 |
| 官方 SegFormer 解码器 | 微调 | 微调 | 微调 | 微调 |
| 新二值分类头 | 训练 | 训练 | 训练 | 训练 |
| 扩展后的 10 通道首层 | 训练 | 训练 | — | — |
| 可学习 10→3 投影（含 bias） | — | — | 训练 | 训练 |
| 10 通道 Spectral Stem | — | 训练 | — | 训练 |
| 残差融合头 ΔL | — | 训练 | — | 训练 |
| TerraMind / AnySat / TiM / 文本语义 / 原粗细两阶段头 | 未参与 | 未参与 | 未参与 | 未参与 |

参数总数/可训练数分别为 A0 3,725,377、A1 3,753,602、B0 3,714,434、B1 3,742,659；每组总数与可训练数相等。不是只训练分割头，也不是冻结编码器。

证据：12 个 best.pt 的加载记录均无 missing/unexpected/mismatched keys；与原预训练文件可比较的编码器、解码器浮点张量均有变化，优化器状态也包含实际更新步数。A 的新增七通道由零变为非零；B 的七个非 RGB 列也由零变为非零；Stem 融合末层已离开零初始化。完整记录见 `checkpoint_audit.json`。

使用的是 `nvidia/segformer-b0-finetuned-ade-512-512`，包含 ADE20K 微调后的编码器和解码器；150 类 classifier 被替换。成熟组件来自 Transformers 的 `SegformerForSemanticSegmentation`。标准解码器输出再插值到输入尺寸，符合[官方接口](https://huggingface.co/docs/transformers/model_doc/segformer)。

## 2. 原结果支持什么，不支持什么

原报告三种子均值：

| 组别 | Hiroshima IoU | Italy IoU | 最佳 epoch（日志从 0 起算） |
|---|---:|---:|---|
| A0 | 0.257 | 0.015 | 34 / 50 / 34 |
| A1 | 0.294 | 0.014 | 24 / 26 / 21 |
| B0 | 0.274 | 0.018 | 37 / 27 / 32 |
| B1 | 0.313 | 0.023 | 16 / 15 / 26 |

本次 seed=1234、best.pt、阈值 0.5、无增强重推理：

| 组别 | train IoU | Hiroshima IoU | Italy IoU |
|---|---:|---:|---:|
| A0 | 0.39217 | 0.25675 | 0.01649 |
| A1 | 0.40574 | 0.28254 | 0.01127 |
| B0 | 0.41344 | 0.26834 | 0.02445 |
| B1 | 0.39444 | 0.31577 | 0.02175 |

B1 是三种子平均表现最好的当前候选，但不是每个种子、每个区域都最好：本次 Italy 的 B0 高于 B1。B1 的训练集 IoU 也不最高。

B1 的最佳点出现在 epoch 15–26；训练 loss 最终继续下降到约 0.297–0.298，而验证 IoU 从约 0.307–0.317 回落到 0.233–0.252。这支持“继续同样训练不能自动解决问题”，不支持“模型已经学会所有训练样本”。训练集中依然存在大量漏检和地区差异：seed1234 的 B1 在 Hokkaido/Chimanimani/Indonesia/Dominicamaria/China 的 IoU 分别约 0.577/0.430/0.291/0.177/0.146。

![训练曲线，三种子均值及标准差](../outputs/segformer/s2_10band_v1/review_batch1/curves.png)

### 对“RGB 敏感、其他波段利用不足”的判断

这是合理的待验证假设，当前证据还不能直接确立因果：

1. B1 的投影输出是三个**学习到的混合通道**，不再是纯 RGB；RGB 选择器只是初始化。
2. B1 的 Stem 直接读取全部 10 通道，绕过 10→3 瓶颈。因此不能把 B1 等同于 RGB SegFormer。
3. A 的七个新增波段卷积权重属于 encoder，实际基础 LR 是 **6e-5**；B 的投影属于 new，LR 是 **6e-4**。文档“新增层均 6e-4”与 A 的实际情况不符。两方案遵循相同按模块分组规则，但“输入适配形式”和“适配权重更新速率”存在混杂。
4. RGB 也经过本训练集逐波段 z-score，并非原 ADE/ImageNet 预处理；不能仅凭权重来自 RGB 就解释全部差异。

### 固定 B1 权重的波段与分支诊断

在标准化后的输入上，把指定路径的七个非 RGB 通道置 0，即替换为训练均值；模型始终 eval，无参数修改。以下是 seed1234 全 split 的 IoU：

| 推理条件 | train | Hiroshima | Italy |
|---|---:|---:|---:|
| 原始 B1 | 0.39444 | 0.31577 | 0.02175 |
| 两条路径的非 RGB 均替换为训练均值 | 0.31385 | 0.31317 | 0.03103 |
| 仅投影路径的非 RGB 替换为训练均值 | 0.37434 | 0.28714 | 0.01997 |
| 仅 Stem 路径的非 RGB 替换为训练均值 | 0.33895 | 0.32320 | 0.03189 |
| 只输出联合训练模型的 base_logits，去掉 ΔL | 0.29089 | 0.21913 | 0.01192 |

解释：投影路径在 Hiroshima 确实依赖非 RGB 信息；Stem 的非 RGB 依赖在训练区域有益，在这两个留出区域未显示同方向收益。应优先研究光谱依赖是否过拟合、各路径是否相互补偿。均值替换会制造分布外输入，路径作用也非线性，**不能**由此证明“RGB 训练更好”、删除某些波段，或把 base_logits 当作单独训练的 B0。

## 3. 优先解决的问题

### P0：多日期属性被不一致地压成单一日期

`scripts/prepare_segformer_data.py` 中：

- `date.fromisoformat(ed[:10])` 只取 `event_date` 的第一个日期；
- `_parse_pre_post()` 遍历全部 pre/post 对并覆盖字典，最后保留**最后一对**。

实测 215/1,277 个训练样本的 event_date 含多个日期（China 26、Indonesia 189）。120 个所选影像早于日期列表中的最晚日期（China 8、Indonesia 112）。还有 114 个缓存 post 参考日期早于所用的第一个事件日期；这是不同事件的日期/索引被错配后的风险，不能把所有 `days_delta_vs_post` 直接解释为同一事件的选图延迟。

例如 `china_s2_1004`：事件日期列表包含 2017、2018、2019 年，选图为 **2018-05-15**，早于两个 2019 年日期；第一对 post=3，缓存却记录最后一对 post=6。`indonesia_s2_157` 的多个对象/日期与当前全幅事件编目掩膜混在一起，本次 B1 有效 GT 1,579 px 全部漏检。

已证实的是**解析和时间契约不一致**，不是已经逐对象确认 120 张全部标签错误。部分日期对应的对象是否落入当前像素掩膜，还须查原始 inventory 的 ann_id、几何与事件日期。不能简单把日期改成 max(event_date) 就认定解决：较早滑坡可能已恢复，且日期字段与 ann_id 数量并非总是一一对应。

下一版应显式保存事件列表和每组 pre/post；支持逐对象时相一致的监督，或将无法核实的多事件样本隔离为待审清单。主任务仍是单期 S2 分割，多期图像只用于标签核查。

### P0：对象/尺度指标与有效区口径不一致

`training/segformer_metrics.py::evaluate_tile()` 的像素 TP/FP/FN 正确使用 valid，但连通域直接对原始 gt 和 pred 建立，预测无效区会改变连通性、预测面积及匹配 IoU；排除截断 GT 并不能自动消除这种影响。

另外：

- 各尺度 `pixel_recall` 只计入通过对象 IoU 匹配的交集，未匹配但预测正确的前景像素被丢弃。它不是通常定义的尺度像素召回。
- tiny 的 `matched_eval` 从未递增，聚合也没有输出完整的 tiny 可评价对象召回。原报告 tiny 全对象匹配数不能除以可评价对象数来推算新指标。
- 旧案例图没有遮蔽 valid=False；旧脚本“大目标”按整张瓦片前景总面积筛选，不能代表真正的大对象。

本次单独定义了审阅指标：保留原始 GT 对象面积/身份；主对象指标只使用未截断 GT；先遮蔽预测无效区，再建立预测连通域并与可评价 GT 一对一贪心匹配；尺度像素召回独立按正确预测像素统计；tiny 的分子分母同时输出。这不是静默覆盖旧指标，需要在下一版协议中固定。

本次用三个人工数组回归用例分别复现了 tiny 漏记、尺度像素召回遗漏、无效预测改变匹配的问题。这些人工数据只验证指标代码，不作为模型效果数据。

这些问题**没有推翻现有像素 IoU 排名，也不能解释 Italy 的全部低分**：8 组 seed1234 的 val/test 重推理中，7 组像素 IoU 与原评测完全一致，A0-val 相差约 0.0000011；主对象召回的修正幅度也较小，例如 B1-val 从 0.5361 到 0.5375、B1-test 从 0.0427 到 0.0437。修正重点是保证后续机制解释与小对象指标可信。

另一个日志命名问题：`SegFormerSeg.projection_change()` 把三个 RGB 选择器系数之外的所有系数称为 non_rgb，其中也包含 RGB 通道交叉混合。本次权重审计严格只读取第 3–9 列，确认七个额外波段已参与学习；今后应修正原日志命名。

seed1234 的 B1 复核：

| 数据 | 像素 P | 像素 R | Dice | 主对象召回 | tiny 匹配/可评价对象 | 完全无 TP 的正瓦片 |
|---|---:|---:|---:|---:|---:|---:|
| train | 0.623 | 0.518 | 0.566 | 0.307 | 3/213 | 199/1277 |
| Hiroshima | 0.470 | 0.490 | 0.480 | 0.537 | 13/77 | 87/446 |
| Italy | 0.034 | 0.059 | 0.043 | 0.044 | 0/32 | 313/382 |

以上 tiny 是未截断子集，不与原报告包含截断对象、再累加三个种子的数字混用。三次种子重复观测不等于三倍独立样本。Italy 的 tiny 像素仅命中 1/125，且没有形成合格对象匹配。

### P1：负例、数据质量与域覆盖不足

训练集 1,277 张全部含正标签，虽然损失已经见过大量背景像素，但缺少经核实的完整无滑坡场景。两张 Hiroshima 空 mask 瓦片共享 ann_id=1964；空 mask 与 annotated=True 仅证明缓存情况，尚不足以称为独立人工核实的负样本。

案例显示亮地表混淆，优先补充经审阅的农田、建筑/硬化地表、道路、河床、裸岩等困难背景。未标注样本不能自动视为无滑坡。

扩量目标也应修正：39,556 是多传感器文件总量，不是可直接加入训练的 S2 样本数。[官方数据卡](https://huggingface.co/datasets/paulhoehn/Sen12Landslides)列出 S2 共 13,628、其中 annotated 6,737。实际可用量还要扣除留出区域、质量、时相歧义和空间重复，不能承诺扩到这些数量。

辐射数据也要保持版本身份。[官方仓库](https://github.com/PaulH97/Sen12Landslides)说明 raw 与 harmonized 的基线校正不同。现有报告“全数据加同一常数会被 z-score 抵消”只对全样本一致平移成立，不能排除跨年份/产品基线的差异。当前本轮未独立完成产品级辐射复核，不据此给影像加减 1000，也不把它列为已证实的失败主因。

### 原报告需要收紧的表述

- “四组和旧模型都低，因此排除实现错误”：只能说明失败具有共性，不能排除共用标签、时相、归一化或评测问题。
- “尺度加权外推得到 0.34–0.43 的上限”：这是粗尺度分层下的标准化假设值，不是性能上界；同档内仍有尺寸、形状和标签质量差异，剩余差距不能全部因果归为域差异。
- “1/4 网格物理上无法检出 tiny”：表述过强，B1 已有全分辨率路径；1–4 px 的信息和标注误差确实限制辨识，但不是据此判定无解。
- “5 个训练事件”：这里是 5 个地区组，China/Indonesia 包含多个日期/事件，必须区分地区、事件和对象。

## 4. 系统案例审阅

固定 seed1234 的四组 best.pt，在 train 1277 / Hiroshima 448 / Italy 382 张上推理。按预定失败类别确定性筛选 30 张，而不是抽样估计总体发生率；总体漏检比例使用全 split 计数。覆盖训练五个地区组，附成功对照，避免只看最差样本。

图示：绿=TP，红=FP，蓝=FN，灰=无效区域。RGB/NIR-R-G 采用展示用拉伸，不是模型输入。所有错误计数基于 valid。对象放大图使用最近邻显示原像素，不做超分辨率。

| 类型 | train | Hiroshima | Italy |
|---|---|---|---|
| 完全漏检 | indonesia_157：1,579 GT px，TP=0 | hiroshima_944：813 GT px，TP=0 | italy_2275：318 GT px，TP=0 |
| 部分检出 | hokkaido_545：R=0.640 | hiroshima_1074：R=0.596 | italy_2276：R=0.217 |
| 明显标签相对误检 | hokkaido_508：FP=2,436 | hiroshima_1135：FP=1,025，亮色人工地表 | italy_1449：FP=1,464，亮色地块 |
| 边界偏移候选 | hokkaido_940 | hiroshima_941 | italy_217 |
| 时相核查候选 | china_642，多日期元数据歧义 | hiroshima_1029，原 post 图重云 | italy_1301，早期图云遮挡 |
| tiny 1–4 px | dominicamaria_560 | hiroshima_580 | italy_2265 |
| 小对象 | dominicamaria_845 | hiroshima_1075 | italy_2350 |
| 中对象 | indonesia_708 | hiroshima_1319 | italy_1162 |
| 大对象 | indonesia_455 | hiroshima_1373 | **该事件无大对象，N/A** |

表中简写对应 `<region>_s2_<id>`。small<50、medium=50–199、large≥200 px；tiny 是 small 子集。放大图的普通 small 另选 5–49 px，以免与 tiny 重复。

观察与解释：

1. Hiroshima_944 中细长亮带与标签有可见对应，四组共同漏检；不是所有失败都能归为不可观测标签。Hiroshima_1074 的主干检出而部分分叉、尾部遗漏，体现结构连续性和边界问题。
2. Hiroshima_1135 的大块 FP 覆盖亮色规则人工地表；Italy_1449 的 FP 沿亮色地块扩张。支持困难背景不足/亮度纹理混淆的假设，但具体地物类别和是否漏标仍需地理参考核查。
3. Hokkaido_545/508 的预测会把相邻前景连接、填满标签间隙；“有命中但形状不准”和“完全漏检”需要分开处理。不可一味提高正类权重。
4. Indonesia_157/708/455 的标签覆盖某些视觉证据弱的区域，且多事件元数据存在风险。先查事件—对象—日期对应，不能把它们一概当作模型能力不足。
5. 在共同有效区平移预测 ±3 px 的诊断中，三个边界候选的最优相对 IoU 增量约为 0.107、0.097、0.073；位移分别 (-1,0)、(-1,0)、(-1,+1)。只是筛选候选，不证明全局配准偏移；不应用到正式指标，不自动平移标签。
6. China_642 的缓存 post 参考为 2017-09-07，chosen 为 2021-05-09；鉴于 first event/last post 混配，不能据这张图直接宣称“同一事件晚了 1340 天”。Hiroshima_1029、Italy_1301 的旧 post 图被云遮挡，换到更清晰日期有合理性；仅凭日期变化不足以判定标签错时。三张均保留“未确认”。
7. 对象放大图显示 tiny 漏检很普遍；Italy_1162 的中等细长对象也被共同漏检。因此后续不仅补 1–4 px，还要兼顾细长形状、背景差异和可观察性。

图册入口（每组 3 页）：

- train：[第 1 页](../outputs/segformer/s2_10band_v1/review_batch1/train_cases_1.png)、[第 2 页](../outputs/segformer/s2_10band_v1/review_batch1/train_cases_2.png)、[第 3 页](../outputs/segformer/s2_10band_v1/review_batch1/train_cases_3.png)。
- Hiroshima：[第 1 页](../outputs/segformer/s2_10band_v1/review_batch1/val_cases_1.png)、[第 2 页](../outputs/segformer/s2_10band_v1/review_batch1/val_cases_2.png)、[第 3 页](../outputs/segformer/s2_10band_v1/review_batch1/val_cases_3.png)。
- Italy：[第 1 页](../outputs/segformer/s2_10band_v1/review_batch1/test_cases_1.png)、[第 2 页](../outputs/segformer/s2_10band_v1/review_batch1/test_cases_2.png)、[第 3 页](../outputs/segformer/s2_10band_v1/review_batch1/test_cases_3.png)。
- 对象放大：[train](../outputs/segformer/s2_10band_v1/review_batch1/object_crops_train.png)、[Hiroshima](../outputs/segformer/s2_10band_v1/review_batch1/object_crops_val.png)、[Italy](../outputs/segformer/s2_10band_v1/review_batch1/object_crops_test.png)。
- 时相候选：[train](../outputs/segformer/s2_10band_v1/review_batch1/temporal_train.png)、[Hiroshima](../outputs/segformer/s2_10band_v1/review_batch1/temporal_val.png)、[Italy](../outputs/segformer/s2_10band_v1/review_batch1/temporal_test.png)。

## 5. 下一阶段执行计划

### 阶段一：建立可信评测与清洁数据版本

**先完成，之后才开始新训练。**

1. 修正评测：统一 valid、对象匹配、尺度像素召回、tiny 分母；增加全空、部分无效、跨无效区连接、截断、多个 GT 合并预测等用例。为全部 12 个现有 best 重新评测，旧结果保留，新增 metric_version，不重新选 checkpoint。
2. 时间契约：把 215 个多日期训练样本列为专项审阅集；优先核实 120 个早于最晚事件的样本。查 inventory 几何与 ann_id，保存对象—日期映射、观测可用性和 date_confidence。无法可靠映射时隔离，不能凭模型输出删标签。另审 87 个 temporal fallback 与超长灾后间隔样本。
3. 在原清单上生成 `s2_10band_v2_clean`，保留旧版；每条排除/重选原因可追踪。统计只用新 train 拟合；审核缓存波段、有效值、SCL、raw/harmonized 来源及处理基线。不给全数据统一添加未经核实的 offset。
4. 手工核查候选边界样本的 CRS、仿射网格、像素中心及标签栅格化规则；区分真实配准错误、标注概化和模型平滑。不通过全局平移“修高指标”。
5. 先声明验证协议：现有 Hiroshima 用于历史对照，Italy 是已多次查看的历史留出集；后续策略选择使用训练地区内部按事件/空间组划分的多个验证折或新增验证事件。若要报告最终泛化，另设未用于本轮诊断的新事件留出集。

验收：每个保留样本的时间解释明确；所有指标单元用例通过；12 checkpoint 复评可复现；每个数据版本、划分和统计有哈希。

### 阶段二：定向扩量，保持架构与损失不变

优先增加有可靠标签的独立事件、细长/小型浅层滑坡、农田—丘陵场景，以及经核实困难负例。先审核本地剩余样本及官方其余 S2 分卷，按新增有效对象、事件和地理覆盖衡量价值，不按下载文件数衡量。

扩量采用可追溯阶梯：clean 子集 → 加入一批新事件/样本 → 加入经核实困难负例。每批发布数量、正负比例、尺度、时相、质量、空间重复检查。未标注 2,783 张只可进入待审候选池，不自动补零标签。相邻地块/同一滑坡的时相或切片保持在同一 split；若从更大原图生成重叠裁块，也必须按地理组隔离。

先用 B1 和 A1 做成对小预算筛查，合格数据版本再跑完整 A0/A1/B0/B1 × 3 seeds。数据增益对比使用相同更新步数预算，同时记录图像曝光次数和覆盖率；另列固定曝光次数的补充比较，避免把更多数据与更多更新混为一谈。保留 BCE+Dice、阈值 0.5 作为共同基线。

验收：多个开发验证区域有一致或可解释的改善；小对象召回提升没有被困难负例误报大增抵消。不存在足够可靠负例时，瓦片误报率仍为 N/A，不承诺业务可用性。

### 阶段三：一次只改一个训练因素

| 顺序 | 对照 | 目的与约束 |
|---|---|---|
| T1 | A 的首层适配 LR：6e-5 vs 6e-4；其余编码器 LR 不变 | 检验 A/B 的学习率混杂。若整层升 LR，RGB 继承权重也会升 LR，必须记录；严格分离可用 RGB 卷积 + 新增七通道残差参数化，但需前向等价测试，不重写 SegFormer。 |
| T2 | 自然采样 vs 有上限的事件/尺度均衡采样 | 提高小对象和弱势地区曝光；保留随机上下文与困难背景，防止全是正中心裁块。先用已核实样本，避免反复强化噪声。 |
| T3 | 统一全量微调 vs 新层短预热后全量微调 | 若出现预训练迁移不稳定再试；固定总步数，预热计入预算，其他参数相同。 |
| T4 | 原增强 vs 经物理约束的轻度光谱/亮度增强 | 多波段相关性与波段顺序不变；先验证增强不会破坏标签可观察性，不直接套自然图像 RGB 色相扰动。 |

T1/T2 优先，T3/T4 按证据推进，不做所有组合的盲目网格搜索。先 1 个固定种子筛选，再 3 个配对种子确认；改变公共协议后四组重跑。训练预算按 step 预注册，不因 Italy 结果临时加时。早停规则和 patience 在开发验证集上预先固定，保留 best/last 和完整曲线。

工程先补：`max_steps` 目前只在 epoch 后检查，下一轮按 step 预算时应在 batch 内准确停止；resume 要恢复累计 global_step、历史和随机流，并避免 last.pt 中 best 元数据落后一个 epoch。这些未影响当前 max_steps=null 的完整运行，但会影响下一轮可复现性。

### 阶段四：解释波段贡献，决定是否调整 Stem

在同一清洁数据、同一协议下，增加机制对照；不替代四个主实验：

1. 标准 RGB 三通道 SegFormer，无 Stem，对比 B0：检验学习到的非 RGB 投影增益。
2. RGB 主干 + RGB-only Stem，对比完整 B1；Stem 的非 RGB 输入可固定训练均值并保持同构参数，或明确记录参数差异。
3. B1 的投影只见 RGB / Stem 只见 RGB 两种**重新训练**的路径对照，区分各路径的光谱贡献；推理均值替换仅作补充诊断。
4. 在验证集预注册红边、NIR、SWIR 分组消融，使用三种子并记录不确定性。不能单看投影权重大小判定波段价值。

若清洁数据和采样调整后仍有明显合并、轮廓扩张，再单独测试一个轻量边界监督或融合调整，与现有 Stem 基线成对比较。暂不添加新完整编码器、超分辨率、语义、多模态或两阶段模块。

## 6. 交付与验证状态

**本轮已实际完成**：

- 检查当前代码、旧/新报告、公共协议和数据契约；审计 12 个正式最佳权重及优化器状态。
- 四方案 seed1234 对全部 2,107 样本重推理，共 8,428 次样本推理；重算有效区指标。
- 固定 B1 进行四种额外推理诊断，各覆盖全部 2,107 样本。
- 30 个样本、9 张对照图板、3 张对象放大图板、3 张时相图板；全部图板已查看。
- 核实多日期解析行为并统计风险清单；抽查原始 NetCDF 属性。
- 3 个指标回归用例实际通过；32 个 checkpoint 哈希前后一致。

**尚未完成，不能写成结论**：另外两个种子的完整新口径复评、专家标签纠错、对象事件日期映射、产品级辐射复核、新数据扩充、受控 RGB 重训练、下一轮训练收益及新事件泛化。

主要阻碍是标签时间解释、可靠负例与事件覆盖，不是当前算力不足。现有资源已支持本轮全量推理；新数据最终合格量未知。

新增文件：

- `scripts/review_segformer_batch1.py`：检查点审计、推理、指标、分支诊断、案例图。
- `scripts/review_segformer_batch1_supplement.py`：多日期审计、原始属性检查、指标回归、对象放大、旧新指标对照。
- 本文；全部新结果在 `outputs/segformer/s2_10band_v1/review_batch1/`。

复现（项目根目录）：

```powershell
$reviewPython = 'C:\Users\yzh\.workbuddy\binaries\python\envs\tm\Scripts\python.exe'
& $reviewPython scripts/review_segformer_batch1.py --device cuda:1
& $reviewPython scripts/review_segformer_batch1_supplement.py
```

关键机器可读证据：`checkpoint_audit.json`、`checkpoint_integrity.json`、`metrics_valid_region.json`、`sensitivity_B1_seed1234.json`、`multidate_audit.json`、`source_metadata_examples.json`、`case_selection.json`、`object_crops.json`、`metric_regression_checks.json`、`legacy_comparison.json`。概率保留 float32，以免 float16 在 0.5 附近改变二值决策。
