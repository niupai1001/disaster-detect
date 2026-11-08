# 01\_EncoderV3\_LatentResponse\_Architecture (1)

# EncoderV3-LR v1.2：共享传感器条件记忆驱动的统一遥感编码器

> **状态**：2026-11-08 文档修订版 v1.2；本次交付仅修改规范，未修改仓库代码、未运行模型或授权真实参数更新。原 v1.1 声明的实现状态须由执行 Agent 按实际工作区重新核查。本版覆盖此前文档中“每个通道一个 P32”的提案；以本版定义为准。方案中用户已明确选择的原则与为 Agent 消除歧义的首版工程默认值分别标注。工程默认值不是经实验验证的最优参数，Agent 不得自行改写。
>
> **核查的现有代码基线**：仓库 [niupai1001/disaster-detect](https://github.com/niupai1001/disaster-detect/tree/7d1db632c40a8dd7c6264cb95d97ec03235b2b61/earth-representation/encoder_v3)；分支 `dev/optical-intra-selfsup-v1`；审阅 SHA `7d1db632c40a8dd7c6264cb95d97ec03235b2b61`。Agent 执行前重新核查实际 HEAD 并记录差异。新架构必须有独立身份、代码类、配置和训练目录，不覆盖旧 Participation 结果。
>
> **目标**：每个已注册卫星载荷**仅有一个持久化、可学习的 32 维描述器 **`P_s`。`P_s` 通过一个**所有载荷共享**的 KV 投影器成为可查询的条件记忆；负责生成权重的模块分别提供自身 Query，按任务与通道位置/原生 GSD 从同一个 KV 中读取所需条件。全部载荷共用同一 Encoder、动态卷积核基、通道融合器、空间 Transformer 和 Decoder。不要求 SRF/PSF/MTF 作为推理必需输入。

## 0. 本轮确认与适用范围

本版基于原 v1.1 两份文档，以及本轮对话中用户明确确认的两项修改：

- **取消首版独立高维 skip**：不默认建立逐像元或逐通道高维旁路，以跨通道融合后的 summary 为主要表示。`F_c`、`U[j,c]`、`h[j,c]` 是前端中间态，不作为独立 Decoder 输入长期保留。独立 skip 可在未来另定并冻结消融协议；本轮不实现、不运行该消融。
- **A 的完全隐藏区域由 Decoder 查询上下文**：Encoder 只编码可见观测，不为完全隐藏区域添加可学习空白内容 Token；Decoder 从可见 summary 与全局上下文读出，并显式接收目标像元相对位置、支持尺寸及目标通道条件。

“原生细节”指原生观测的空间差异及真实支持，不等同于卷积后的 32 维中间特征。本版既不要求无损保存每个原生像元的高维特征，也不宣称压缩后已完整保留全部细节。

**没有随本轮一起改变的部分**：载荷 P32、KV 数、动态卷积、Sub-Patch 尺度、融合后的 `R(C)` 局部 latent、空间主干预算和 A/B1 权重保持原规范。融合后的局部 latent 不是逐通道 skip，不能因取消 skip 而擅自删除；本轮也没有授权新增原始标量到 Decoder 的独立旁路。

**仍未确认的部分**见 §14：标准化/输出/损失的精确合同、LocalEmbed 和位置表达等具体运算、空间读出和流式执行细节、目录与“整景”的具体数据范围。Agent 先提交接口提案及依据；只对依赖这些选择的实现暂停，不阻塞已明确的独立模块。不能把本轮对 A 读出方式的“认可”扩大为对这些候选方案或真实训练的批准。

原文参考工作、旧代码审阅 SHA 和历史数字作为来源内容保留，本次文档修订不构成新一轮外部核查或实验。文档版本为 `spec_revision=1.2`；架构族标识沿用 `latent_response_sensor_memory_v1`，实际运行身份还须绑定两份规格的 hash 和已确认接口，不能仅凭架构族名称判断检查点兼容。

## 1. 思路演变及本次修订理由

| 阶段 | 提案及动机 | 识别的问题 | 当前结论 |
| --- | --- | --- | --- |
| 初始物理假设 | SRF 在光谱维积分，PSF 在空间维模糊；希望学习隐藏的地表空间—光谱表征，而非精确恢复真实反射率 | 成像算子不完全可知、反演不唯一 | 保留“传感器影响观测”的物理先验；不宣称精确恢复 PSF/SRF |
| 旧 EncoderV3 | 完整 SRF + GSD → Hypernetwork → 每波段 3×3 卷积 → Participation Merge → 局部/全局 Transformer | Token 随波段数和原生像元数量增长，`Σ N_view²` 成本过高，S2 全十波段场景因旧上限被大量排除 | 替换 Token 组织与全局 Attention，复用可靠的数据/几何/任务基础设施 |
| 第一轮重构 | 已知 PSF 引导动态卷积；已知 SRF 引导跨波段 Patch 融合 | 对 S2/Landsat/EnMAP/GF/S1 不存在共同且完整的必需 SRF+PSF 输入合同；SAR 不遵循光学 SRF 定义 | 真实物理响应不作为首版 `forward` 必需输入；未来只作教师/校准证据 |
| 普通共享 Conv | 取消全部物理描述 | 过度丢失通道观测特征，无法形成差异化动态权重 | 不采用无条件共享 Conv 作为最终方案 |
| **已否决：逐通道 P32** | 每个载荷的每个波段注册一个独立可学习 P32 | 变成随全部波段累计增长的身份词表，稀疏曝光；不能直接迁移到未知波段 | **彻底撤销 **`ChannelKey → nn.Parameter(P32)`** 的方案** |
| **已否决：支持集推断器/共享响应码本作为前置** | 多场景影像 → 内容特征编码 → Q → 共享词表 → P32 | 为条件权重生成额外大型表征与训练需求，易混淆地表场景和仪器属性，投入与目标不成比例 | **不实现支持影像采样器、内容 Q 生成网络、64×32 响应原型词表或场景一致性损失** |
| **当前版本** | 每个卫星/载荷一个 P32；P32 是 K/V 信息源；动态卷积、通道融合和目标解码各自用 Q 读取 | 单个 P32 直接做单 Key/Value 注意力会使 Softmax 恒为 1；同一 P32 下不同波段需要可区分的 Q | **共享 KV 投影为多个槽位；模块级/通道级 Query；所有生成与主干权重共享** |

本版创新假设是：**一个低维、持久化、载荷级的潜在测量条件，在不依赖完整物理响应参数的情况下，能否由不同模块和不同通道主动查询，以统一生成空间特征提取、通道融合和目标重建所需的条件。** 这仍只是研究假设；不会自动保证词元是可解释 PSF/SRF，也不会自动实现未知卫星零样本泛化。

## 2. 参考工作的借鉴边界

1. **DOFA**：*Neural Plasticity-Inspired Multimodal Foundation Model for Earth Observation*（2024），[论文](https://arxiv.org/abs/2403.15356)、[代码](https://github.com/zhu-xlab/DOFA)。借鉴“条件输入 → 生成动态 Patch/投影权重 → 共享主干”；DOFA 重点使用波长条件，本版改为**载荷级潜在描述由模块查询**，不是逐波段波长查表。
2. **UniverSat**：*UniverSat: Resolution- and Modality-Agnostic Transformers for Earth Observation*（2026），[论文](https://arxiv.org/abs/2606.23503)、[项目入口](https://gastruc.github.io/universat)。其 LFF 只是标量升维，后面仍有沿像元、通道、时间、子 Patch 的逐轴 Cross-Attention；借鉴统一参数与分轴融合。本版并不复制其全部模态元数据条件和时间处理。
3. **Panopticon**：*Panopticon: Advancing Any-Sensor Foundation Models for Earth Observation*（CVPRW 2025），[论文](https://openaccess.thecvf.com/content/CVPR2025W/EarthVision/html/Waldmann_Panopticon_Advancing_Any-Sensor_Foundation_Models_for_Earth_Observation_CVPRW_2025_paper.html)。共享单通道特征提取及跨通道查询已有先例；跨通道 Cross-Attention 本身不构成唯一创新。
4. **CondConv**：*Conditionally Parameterized Convolutions for Efficient Inference*（NeurIPS 2019），[论文](https://papers.neurips.cc/paper_files/paper/2019/hash/f2201f5191c4e92cc5af043eebfd0946-Abstract.html)。借鉴少量共享卷积核基及动态组合系数。
5. **Set Transformer**：ICML 2019，[论文](https://proceedings.mlr.press/v97/lee19d.html)。借鉴集合聚合与可学习查询的区分；**本版不引入支持影像集合推断器**。
6. **Swin Transformer**：ICCV 2021，[论文](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.html)。借鉴局部注意力、层次空间聚合；避免大场景完全全局二次复杂度。

**实际差异**：本版 `P_s` 不是 DOFA 的中心波长、不是 UniverSat 直接使用的实测传感器物理描述、也不是为每个通道维护的 Embedding。它是每个载荷一个、由所有模块通过共享 KV 读取机制利用的可学习条件记忆。

## 3. 已确定的不可擅改原则

1. **载荷级 P32**：一个稳定的 `SensorKey=(platform,instrument)` 对应一个、且只有一个 `P_s ∈ R^32`。同一载荷的所有波段、所有场景、不同输入通道子集复用该参数。`P_s` 只在首次登记该载荷时初始化一次，随后持续优化；**绝不按影像、按输入通道数或按波段新建 P**。S2A/MSI、S2B/MSI、L8/OLI 等按实际载荷身份区分。多载荷观测分别引用自己的 `P_s`，但执行同一网络结构。
2. **P32 提供 K/V，生成权重的模块提供 Q**：`P_s` 不按维度指定“空间/光谱区”，也不先拆成固定子向量。空间卷积、通道融合、目标 Decoder 各有共享的 Query 生成规则，从同一个 KV 中读取条件。Query 可以因通道而异，但其生成器与注意力投影参数在所有卫星间共享。
3. **单 P32 必须扩展为多个 KV 槽位**：只有一个 K/V 时 attention 恒返回 V，与 Q 无关。共享 `MemoryProjector` 从完整 P32 生成多个可学习投影槽位；这**不是**增加多份载荷描述器，也不是通道词表。
4. **通道身份由确定性的元数据表达，不是独立的可学习 P**：至少要有原生载荷通道目录中的稳定顺位 `ordinal/total_channels`，以及原生 `GSDx/GSDy`。子集输入和 B1 隐藏**不重新编号**；同一通道在同一载荷中的 Query 元数据始终相同。不能拿当前张量下标当身份，也不能把光学波长强加给 SAR 极化通道。
5. **一个 Encoder、一条主干**：无卫星/模态专用 Conv、Hypernetwork、通道融合、空间 Transformer 和 Decoder。影像预处理可依据物理量执行已知的定标/归一化，这是数据合同，不是双 Encoder。
6. **SRF/PSF/MTF 可作为未来物理教师，但不参与本轮 **`forward`；不得依数据是否含这些文件切换执行路径。`P_s` 是潜在测量条件，不能未经证据称为反演出的 PSF/SRF。
7. **原生网格、先通道融合后远距离空间通信**：不同 GSD 原始数组不强制重采样；Sub-Patch 按真实地理支持融合，长距离空间主干只输入每个 Sub-Patch 一个 summary，融合后的局部潜在通道 Token 保留供局部交互/Decoder。首版无独立逐像元、逐通道高维 skip，也无 micro-cell fine skip。
8. **不固定裁剪、不靠旧像元上限排除大场景**；允许整景等价的分块、Halo、激活检查点与流式全局汇总。
9. **先 E0 A+B1**，不以 B2、JEPA、GF/EnMAP/S1 真实数据、物理教师或下游评测作为首版光学训练准入前置。
10. 新模型/配置/检查点和授权身份与旧 `participation_selected_windows_v1` 分离；历史证据不冒充新架构结果。
11. **A 空白区域在 Decoder 侧预测**：不在 Encoder 添加可学习空白内容 Token；可见 summary、全局上下文、目标相对位置/支持尺寸和目标通道条件共同支持读出。几何占位不作为真实内容 K/V；目标有效性不得与可见输入支持混为一谈。

## 4. 输入、注册与“一个描述器”的准确含义

### 4.1 持久化传感器注册表，不是逐通道词表

```text
SensorKey = (platform, instrument)
SensorRegistry:
  S2A/MSI  → P_S2A_MSI[32]
  S2B/MSI  → P_S2B_MSI[32]
  L8/OLI   → P_L8_OLI[32]
  L9/OLI2  → P_L9_OLI2[32]
  ...      → 每新增一个真实载荷只增加一组 P32

ChannelCatalog[sensor]:
  channel_id → canonical_ordinal, total_channels, native_grid metadata
  （目录没有逐波段可学习参数；顺位对缺失波段和子集输入稳定）
```

`P` 参数规模为 `32 × 注册载荷数`，**不是** `32 × 注册总波段数`，与 EnMAP 的数百波段不发生描述器参数线性增长。一张 S2 十波段影像只拥有一组 `P_S2`；B1 隐藏四个波段后剩下六个可见波段，仍读取同一 P。批内不同卫星各读取自己的 P，但用相同的 MemoryProjector 和 Encoder。

采用单独 `nn.Parameter`/`ParameterDict` 保存每个载荷的 P；`optimizer.zero_grad(set_to_none=True)` 后，未参与当前前向的 P 必须保持 `grad is None`，避免被密集 AdamW 的历史动量或 weight decay 意外更新。Checkpoint 保存 SensorKey 排序、P 参数、Optimizer 状态和注册/目录哈希。**新增注册不得在训练运行中静默改变优化器参数组或恢复身份**；新载荷应在新实验/显式适配任务开始前登记。

### 4.2 通道目录的作用与限制

每个通道的输入为 `(values[H,W], valid, visible, affine, CRS, SensorKey, channel_id)`。通道目录根据原始产品定义静态存储 `channel_id→(ordinal,total_channels)`；`GSDx/GSDy` 从真实仿射网格读取。目录属于**非可学习的输入结构元数据**，不能转换成 per-band Embedding 词表。

Query 的基础元数据定义为：

\\\[
m\_{s,c}=\\big\[\\mathrm{Fourier}((\\mathrm{ordinal}*{s,c}+\\tfrac12)/C\_**s),\\;\\log g*{x,c},\\;\\log g\_{y,c}\\big\].
\\\]

所有新旧载荷都调用同一个共享投影网络 `f_meta(m)`。这是通道身份和采样几何条件，**不是 SRF/PSF 的真实数值**。相同顺位在不同载荷上不保证波长/极化等价，因此不能宣称其自动完成物理光谱对齐。任意未知载荷若不能提供稳定的通道目录与网格，须先建立输入合同，不能随机打乱通道序号伪造身份。

数据侧须有可信 QA/NoData、原生几何及正确量纲。初版真实训练只使用现有 S2/Landsat 光学 TRAIN/VAL；SAR 未来进入前先在数据层完成散射量定标和相容的稳健数值标准化，而不是用未经校准 DN 假装表面反射率。现有 `NativeOpticalDataset` 依赖 SRF 文件做**历史数据审计**，可以继续保留这一数据读取要求；**新模型的前向条件不读取 SRF**，两者不矛盾。

## 5. Stage 0：载荷 P32 → 共享 K/V 条件记忆

从唯一的 P32 生成 `M_mem` 个 Key 与 Value：

\\\[
P\_s\\in\\mathbb R^{32},\\quad
(K\_s,V\_s)=\\mathrm{reshape}(\\mathrm{MemoryProjector}*\\theta(P\_**s)),
**\\quad K\_**s,V\_**s\\in\\mathbb R^{M*{mem}\\times32}.
\\\]

**首版固定** `M_mem=8`、`d_k=d_v=32`，`MemoryProjector` 是所有载荷共享的两层小型 MLP（`32→64→512`，GELU；512=2×8×32），学习得到的槽位顺序全局一致。P 的全部 32 维都作为输入，不切分 PSF/SRF 维，不维护每波段独立参数。注意：投影成八个 KV 槽位**不增加 P 所携带的独立信息容量**；它只是让不同的 Q 可以检索不同组合。

模块 `t ∈ {conv,channel,decoder}` 为当前通道生成 Query：

\\\[
Q\_{t,s,c}=q\_t^0 + f\_{t,\\theta}(m\_{s,c})\\in\\mathbb R^{32},
\\\]

其中 `q_t^0` 是**每模块一组、全载荷共享**的 learned seed，`f_t` 是全载荷共享的轻量投影。无每卫星 Query 参数、无 per-band Query Embedding。读取：

\\\[
z\_{t,s,c}=\\mathrm{softmax}\\left(\\frac{Q\_{t,s,c}K\_s^\\top}{\\sqrt{32}}\\right)V\_s\\in\\mathbb R^{32}.
\\\]

- `Q_conv` 读取空间权重生成条件 `z_conv`。
- `Q_channel` 读取通道融合的条件 `z_channel`。
- `Q_decoder` 读取目标通道重建条件 `z_decoder`。**这是 B1 的必要条件**：所有目标波段共享同一个 P32，Decoder 若只用 P32 就无法区分 B04/B08；Decoder 必须同时接收目标通道目录/几何所决定的 Query，且不能读取目标像元值。

`Q` 是由目标模块定义的需求，再依据**该通道静态目录与 GSD**调整；**Q 不从目标通道当前影像值提取**，没有支持集编码网络，也不引入场景内容误充仪器响应的环节。注意力权重 `a[t,s,c,slot]` 可记录供审计，不宣称每个槽位对应真实 PSF/SRF。

## 6. Stage 1：由 KV 读取结果生成原生动态卷积

`z_conv` 驱动少量共享卷积基的组合：

\\\[
\\alpha\_{s,c}=\\operatorname{softmax}(W\_\\alpha z\_{conv,s,c}+b\_\\alpha)\\in\\mathbb R^{8},
\\quad W^{conv}*{s,c}=W\_**0+\\sum*{m=1}^{8}\\alpha\_{s,c,m}B\_m.
\\\]

\\\[
F\_{s,c}=\\operatorname{GELU}\\big(\\operatorname{MaskedConv}*{3\\times3}
**(X*{s,c}, W^{conv}\_{s,c})\\big)
\\in\\mathbb R^{H\_c\\times W\_c\\times32}.
\\\]

`W0` 与八组 `B_m` 全局共享，形状均为 `[32,1,3,3]`，带共享 `[32]` bias；**不维护每卫星/波段的静态 Conv 核**。`P_s` 的参数量仅 32；通道数变多只增加动态读取、卷积执行工作量，不增加新网络参数。先把不可见/无效值置零，再使用可见邻域计数归一化和固定 halo=1 的等价分块；任何隐藏值不能参与计算。共享卷积基与 `MemoryProjector`、Query 生成器都必须收到梯度。

本版不再对卷积基执行第二层独立 QKV 检索：先用模块 Q 从 P/KV 得到 `z_conv`，再以 `Linear→Softmax` 形成卷积基系数，即符合你的低成本权重生成目标。不能把 P 当作一个乘在特征上的普通卫星偏置来替代动态权重。

## 7. Stage 2：原生网格的地理 Sub-Patch 与 micro-cell

保留原先重构已确定的原生几何思路。每个 observation 由**完整原生通道目录的几何元数据**定义 `g_ref=min_c(min(gx_c,gy_c))`，与 B1 被遮盖的通道是否可见无关。以固定地理原点构建 `L_sub=8*g_ref` 的 Sub-Patch；每个 Sub-Patch 划成 `4×4` 个地理 micro-cell（**工程默认值，尚无实验最优性结论**）。

每个通道在自身原生网格先产生 `F_c[H_c,W_c,32]`，之后使用实际像元与 micro-cell 的地理交叠面积，形成 `U[j,c,16,32]` 与显式 `support_fraction`、`visible_count`。按面积归一聚合，仅计算 `valid & visible` 的原生值；无支持位置是 mask，占位不是假观测。

这里进行的是**潜在特征的地理支持对齐**，不是把原始粗波段插值为细波段。较粗原生像元对多个子区域可能有空间重叠，必须报告其实际支持，不得解释为多个独立细节观测。共享局部汇聚器得到 `h[j,c,128]`；其具体位置敏感运算由 §14 的接口提案确定。`F_c`、`U[j,c]`、`h[j,c]` 在参与通道融合后不形成独立 Decoder skip。初版沿用当前工程的 north-up、同 CRS、投影米制准入；未知变换明确报错，不允许偷偷重采样。整个场景可流式分块与重计算，但不能固定裁剪训练数据。

**内存与梯度说明**：取消旁路只取消长期可查询的表示接口，不代表自动消除了反向激活开销。中间态可按等价执行方案分块、重计算或在完成其计算依赖后释放；不得用无依据的 `detach` 切断训练梯度。原始标量仍作为数据和监督真值存在，但不是新增的 Decoder 内容输入。具体生命周期与峰值内存须实测。

**几何口径待核对**：本节沿用 v1.1 的“完整原生通道目录的几何”表述；原执行计划把 `full_geometry` 称为“完整输入场景的非数值网格元数据”。两者涉及的实际通道范围仍须按 §14 核实，不能默选一种后声称已得到确认。不变约束是 B1 隐藏不能改变参考尺度和地理锚点。

## 8. Stage 3：通道条件融合——与 Memory Read 的 Q 严格区分

先由 `Q_channel` 从**同一载荷 P32 所形成的 KV**读取 `z_channel,s,c`，再与 `h[j,c,128]`、有效支持一同生成通道融合的 Key、Value：

\\\[
k\_{j,c}=f\_K(\[h\_{j,c};z\_{channel,s,c};\\mathrm{support}*{j,c}**\]),\\quad
**v*{j,c}=f\_V(\[h\_{j,c};z\_{channel,s,c};\\mathrm{support}\_{j,c}\]).
\\\]

之后是另一个**影像内容聚合注意力**，其 Query 由共享生成器提供，不是从 P32 直接取出的 Q：

\\\[
\[z\_j^{sum}; z\_{j,1:R(C\_j)}^{latent}\]
=\\operatorname{CrossAttn}(\[Q\_{sum};Q\_{latent,1:R(C\_j)}\],K\_j,V\_j).
\\\]

- `Q_channel`：回答“我在融合这个输入通道时，应从传感器 KV 读取什么条件？”；它使用静态通道目录/网格。
- `Q_sum/Q_latent`：回答“我要从当前 Sub-Patch 的**可见通道特征集合**聚合出什么空间摘要/潜在通道信息？”；数量不等同于输入通道数。

**实施默认动态预算**：`R(C)=min(16, ceil(sqrt(C)))`，`C` 为该 Sub-Patch 内有有效可见支持的通道数。`C=1/2/4/10/224` 时 `R=1/2/2/4/15`；`C=0` 时没有内容输出、只保留受掩码保护的几何占位；不创建可学习空白内容 Token，不把该占位当作可见 K/V。其区域内合法的 A 预测目标保留，由 §10.1 的 Decoder 路径处理。`Q_latent` 按连续槽位坐标生成，不能按波段数或卫星型号分别设参数。通道任意排列时，只要值/几何/ChannelKey 同步重排，集合融合结果应等价。B1 隐藏通道**不进入融合 K/V**，即使 P\_s 本身被其他可见通道共同使用。

每 Sub-Patch 只将**一个** `z_sum[j,128]` 交给长距离空间主干；其余 `R(C_j)` 个潜在 Token 留在局部用于细节、光谱关系与 Decoder，不得将 `N_sub×R` 重新送入全局完全 Self-Attention。

## 9. Stage 4：分层空间 Transformer

```text
Spatial summary [N_sub,128]
  → 8×8 局部窗口 Attention（2 层，4 头）
  → 4×4 空间支持聚合 → 粗尺度 summary
  → 32 个共享全局 latent Query Cross-Attention 粗尺度
  → 全局 latent Self-Attention（2 层，4 头）
  → 全局上下文回注细尺度 summary

Local spectral latent [Σ_j R(C_j),128]
  → 只与所属 Sub-Patch、近邻细尺度 summary 作 1 层局部交互
无独立 micro-cell / 逐通道 / 逐像元高维 skip
A 完全隐藏位置 → Decoder 目标几何查询可见 summary + 全局 context
```

空间尺度定义为 **细 **`L_sub`**、粗 **`4×L_sub`，不是固定 10m/60m。全局 32 latent 是跨空间远距离通信容量，和 `R(C)`（局部通道容量）是不同维度。大型整景必须采用能够保留全景信息的 streaming、Halo、激活重计算/两遍汇总等方式；分块不等于裁剪，Transformer 不得在全景 summary 上偷偷退化为完整 `O(N_sub²)` 注意力。

## 10. 自监督 E0：A、B1 与目标 Query 必须自洽

- **A：空间矩形隐藏**。在任何动态卷积/地理支持汇聚之前，按原生 `valid & visible` 隐藏目标值。masked 卷积邻域及所有局部/全局路径不得读取隐藏反射率。Decoder 对原生目标像元以目标通道的 `Q_decoder(s,c) → (K_s,V_s) → z_decoder` 和目标几何支持查询预测，不能只使用相同的 P\_s。
- **B1：完整通道隐藏**。从 Encoder 输入删除被隐藏通道的所有反射率与可学习特征；**允许**提供它的非数值目录信息（`SensorKey`、`channel_id`、固定顺位、GSD/仿射）给目标 Decoder 和固定几何规划。Decoder 的 `Q_decoder(s,c_target)` 在目标反射率不可见的条件下读取同一 P\_s，生成与其他目标通道不同的测量查询。绝不能将目标反射率、目标值统计或从目标特征生成的 Q 输入 Encoder/Decoder。
- **统一监督**：首版光学 S2/Landsat 使用同源已审 TRAIN-only 统计量，将原生标量标准化；未来 SAR 使用正确标定的散射量，但不作为首版准入。Loss 固定 `0.5·L_A + 0.5·L_B1`；任务内部按目标通道→观测→来源等权，保留旧 E0 合理统计口径。不使用旧 2000 步 `curriculum` 自动改变 A/B1 权重。输入变换、Decoder 输出域、监督目标域及旧 NMSE 的衔接仍属 §14 待审接口；不能重复除以原始标准差平方，也不能把讨论中的候选公式当作已获批准。
- **Decoder**：一个共享标量 Decoder，以融合后的 fine summary 为主要表示，结合可用的融合后局部 latent、全局 context 与 `z_decoder`；不读取独立高维 skip。目标像元相对位置与真实支持尺寸必须实际参与预测计算，不能只用于检索索引。A 完全隐藏区域按 §10.1 读出，不要求目标区域存在内容 Token 或局部 latent。精确支持掩码与分块监督可复用旧方法，但新 Representation 和 A 查询支持规则需新适配。
- **关键结构检查**：对于同一传感器同一目标坐标，B04 和 B08 使用同一个 P32，**但 **`Q_decoder`**、读取系数和目标条件应可区分**；否则 B1 是不可识别的重建任务。不同目标的预测不必在随机初始化时一定不同，但结构必须有产生不同预测的能力和梯度。

### 10.1 已确认：A 完全隐藏区域的读出合同

**信息路径**：

```text
可见通道/空间位置
  → 原生动态卷积 → micro-cell / LocalEmbed → 通道融合
  → 可见空间 summary → 分层空间主干 → 可见 summary 与全局 context

目标通道静态目录 + 原生 GSD → Q_decoder → 同载荷 KV → z_decoder
目标像元几何 → 相对位置、支持尺寸 → Decoder 的空间读出/预测

Decoder(可见 summary, 全局 context, 目标几何, z_decoder) → 目标标量
```

这是数据依赖合同，不预先指定 Decoder 必须用 Cross-Attention、固定邻域或某个 MLP。具体检索/聚合方式、相对坐标参考系及编码归一化由 §14 的提案列明后确认。

**两种查询不能混淆**：`Q_decoder(s,c)` 仍是由目标通道静态目录和 GSD 生成的传感器记忆 Query，不读取目标像元值，也不因逐像元空间位置而改成另一种载荷查询。逐像元相对位置和支持尺寸进入 Decoder 的空间读出/预测路径；不新增逐像元可学习参数。现有共享全局 latent Query 仍保留，不属于被取消的空白位置内容 Token。

**三类掩码/支持语义分开**：

| 语义 | 含义 | A 完全隐藏的合法目标 |
| --- | --- | --- |
| 可见输入支持 | 是否存在可进入 Encoder 的真实可见观测 | 可以为假 |
| 目标有效性 | 是否满足原生 QA、任务目标选择与几何合同 | 应按原任务独立判定，不能由输入可见性替代 |
| 上下文可用性 | 可见 summary / 全局 context 是否能为读出提供信息 | 不要求与目标区域直接重叠 |

因此，不得因为目标区域没有可见 Token，或与可见区域无面积重叠，就把该区域全部标为“不受支持”并移出 A 损失；必须通过已确认的上下文读出处理。整观测完全没有可见上下文属于另一边界情形，应单列报告，不能自动替换样本、伪造观测或用零损失隐藏失败。

**应补充的直接测试**：A 隐藏矩形覆盖至少一个完整 Sub-Patch，目标区没有 Encoder 内容 Token，但合法目标仍有有限预测和监督；同一目标通道、同一 Sub-Patch 内的不同目标几何确实进入网络，结构能够产生不同预测；无隐藏值输入 Decoder；几何分块不改变对应目标的预测和梯度。不能强制随机初始化时任意两个像元的数值一定不同，只检验可区分能力和实际计算依赖。

**隐藏值投毒测试**：保持目标真值、valid/visible、任务选择、几何和冻结 TRAIN-only 统计量不变，仅对不可见输入副本注入 NaN/大值。避免输入与监督真值共享存储导致标签也被改写；比较输出，以及在同一监督/损失下的参数梯度。不允许用当前目标观测重估统计量后把梯度差异归因为泄漏。

## 11. 已知/未知载荷的训练边界

**已知载荷 v0**：一次实验启动前明确 SensorRegistry 键集合及通道目录。每个载荷 P32 在初始化时随机生成一次，联合训练共享网络。S2 与 Landsat 交替输入时只查询各自已有 P32；同一载荷的四/十波段子集使用**同一个** P32，且任一可见通道都能给该 P32 传递梯度。没有逐波段独立参数或支持集推断成本。

**新载荷 v1（接口预留，非本轮训练任务）**：为一个新 `SensorKey` 注册**一组**新的 32 维 P，并建立通道目录；冻结共享主干，以少量该载荷的无标签 A/B1 样本仅优化 P，保存新实验/适配身份。**随机新 P 不等于零样本已适配**；没有训练曝光或适配不能保证测量响应正确。未知传感器自动一次前向生成 P 的网络**不在本版范围**，也不是训练前必需功能。

**物理教师（未来）**：真实 SRF/MTF/PSF 对可获取的载荷或通道提供监督/对齐信号。`P_s` 是载荷级而非逐波段级，所以未来物理监督不能简单把每条 SRF 当作 P\_s 的独立目标；应结合通道 Query 与由 P\_s 生成的条件共同监督。仪器级与处理产品级的 PSF 不得混同。

## 12. 与已有代码/数据的关系与已知风险

旧证据 `encoder_v3/reports/optical_selfsup_closure/10_final_report.md` 记录八观测十六视图共约 492,595 / 642,123 Token，`ΣN_view²≈4.01×10^10/6.37×10^10`；旧 `max_native_pixels=16384` 使 586 个大型 S2 十波段训练成员被排除。该数据只用于解释重构动机，**不是新架构实测加速**。新模型应分别统计全景 `N_sub`、`ΣR(C_j)`、粗尺度数、全局 32 latent、实际 wall time 与显存，报告全部有效原生像元的覆盖。

旧 `formal_runner.py` 调用 `execute_route_update` 时传递 `runner_state=dict(state)`，而验证回调在更新内部发生，导致保存的状态可能滞后；新版 runner 必须修复验证完成后的状态事务并实测验证边界恢复。旧 A-hidden/批处理一致性证据与报告表述也有需重新核实的地方；本版不能继承旧“已经验证正确”的笼统结论。

另有三个当前研究限制必须如实陈述：

1. 一个载荷级 P32 对全通道共享，无法凭空包含任意高光谱数百个响应函数的全部细节；不同通道的差异依赖稳定 Channel Query 与共享模型，从而是有信息瓶颈的近似。
2. 标准化通道顺位加 GSD **不是** SRF，也不代表跨载荷同序号通道物理等价；缺少外部标定时响应的可解释性受到限制。
3. 若某载荷训练样本很少，P32 仍会学习不足；但这与为每个波段维护独立参数相比，稀疏参数问题已从“按波段”降为“按载荷”。

## 13. 完成定义及排除事项

**当前实现完成的条件**：每载荷一个持久化 P32；同一 P32 被所有通道复用、形成多个 KV 槽；卷积/通道融合/Decoder 的模块与通道 Query 具有结构可区分性和有效计算/梯度连接（不等于已证明学习收益）；原生卷积输出 32 维；地理 Sub-Patch；动态局部 `R(C)`；一个空间 summary 进入可扩展分层 Transformer；无独立高维 skip、A 完全隐藏区域的 Decoder 读出及目标几何依赖通过测试；A/B1 隔离成立；待审接口已确认；真实数据前后向无更新通过；得到新版本独立的工程准入证据。不要求完成大规模训练。

**当前不实施**：逐波段 P32、输入图像内容推断 P 的前置网络、多场景支持集/固定 64×32 响应码本、物理教师/真实 PSF 反演、未知载荷零样本推断、SAR 真实 E0、B2、JEPA、下游任务、独立高维 skip 及其消融、论文级消融或长周期训练。

## 14. 待审接口：Agent 先提交提案，不代用户作结构决定

以下不是新增研究任务，而是原规范尚未唯一确定、实现时必须选择的运算或数据合同。Agent 可一次性给出最小推荐方案、理由、形状、掩码和测试方法；未经确认，不将候选写成正式配置。已明确且不依赖这些选择的代码和测试可以继续。

| 编号 | 待明确接口 | 需提交的最小内容 |
| --- | --- | --- |
| I1 | 标准化—输出—损失 | 输入变换、冻结统计的键/来源、输出域、监督域、逐通道误差与层次归约、原单位指标还原；说明是否复用旧 NMSE、如何避免重复标准化 |
| I2 | LocalEmbed / 未完整定义的小投影 | 4×4 micro-cell 的位置和有效支持怎样进入 `h[j,c,128]`；位置敏感运算、必要的 Fourier 编码/小投影/latent Query 生成细节；不添加独立高维 skip |
| I3 | 空间主干 / 整景等价执行 | 已给定层数和预算内的位置表达、粗聚合、空窗口处理、全局回注、分块依赖和注意力归一化；明确张量驻留、重计算及前后向等价对照 |
| I4 | Decoder 空间读出 | 遵守 §10.1，列明可见 summary 检索与聚合、全局 context 接入、目标相对坐标参考系/尺度归一化、支持尺寸与 `z_decoder` 的接入；不加入 Encoder 空白内容 Token |
| I5 | 目录 / full\_geometry / 场景边界 | 根据实际元数据区分完整合法通道目录与当前观测网格；明确 `g_ref` 取值集合；区分完整入库 observation/cache 与原始卫星产品整幅影像，报告实际覆盖和缺失证据，不能默认为同一范围 |

I1 的一个**讨论候选，尚未批准**是：用固定 TRAIN-only 均值/标准差生成 `z=(x−μ)/σ`，Decoder 预测 `z`，逐通道只计算一次标准化域 MSE，再按既定层次汇总；需要物理量单位指标时逆变换。用户对 §10.1 的认可不构成对该公式的单独确认。

确认记录须列出用户确认内容或经审核的规格版本，不能由 Agent 自填批准。若固定默认值无法运行，仍按执行计划的变更控制报告，不静默降低 R(C)、层数、通道数或覆盖范围。

> **配套文件**：[执行计划](02_EncoderV3_LatentResponse_Agent_Execution_Plan.md) 与 [Agent 交付说明](03_Agent_Handoff.md)。两份核心规范必须使用同一 v1.2 版本；不得混用 v1.1 的 skip 要求或旧逐通道 P32 段落。
