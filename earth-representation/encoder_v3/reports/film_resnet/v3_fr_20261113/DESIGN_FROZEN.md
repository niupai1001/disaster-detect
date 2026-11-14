# V3\_FR\_Architecture\_Design

# V3-FR：FiLM 条件化 ResNet 与多尺度 Transformer 架构设计

**版本：1.0｜日期：2026-11-13｜状态：架构主线已确定，工程候选待实现与验收**
**配套文件：** [V3_FR_Agent_Implementation_Plan.md](V3_FR_Agent_Implementation_Plan.md)
**对象：** `niupai1001/disaster-detect / earth-representation`
**简称：** V3-FR = Encoder V3 的 FiLM–ResNet 候选。不是 RetNet，不替换成 RMT，也不是分类版 ResNet-50。

> 本文确定可实现的信息流、接口与验证义务。所有新增数值超参数是首个工程候选值，不代表用户已确认正式训练配置，更不代表质量已经验证。本文及配套计划不启动任何训练或修改远端仓库。

---

## 0. 文档效力、事实与假设

### 0.1 阅读规则

- **硬约束 H：** 实现不得自行改变；改变需新设计版本和明确的变更记录。
- **工程默认 E：** 本文提供可直接实现的首版配置。Agent 应先完成容量普查和预检，再冻结实际值；修改必须在训练前登记，不能训练后解释。
- **几何推导 G：** 从原生网格、显式物理尺度和配置计算，不按传感器名称硬编码。
- **实验候选 X：** 只有指定对照可以启用，不能在默认路径中偷偷叠加。

示意图不是实现规范。尤其图中 `4×4`、三个尺度的大小和双向箭头，不得代替本文的计算图与接口契约。

### 0.2 当前可追溯起点

本次核对时，开发分支仍指向 `d330714874274c69eb07bf2466f1d8b74f708661`；诊断结果在独立分支 `results/v3-ab-spatial-diagnostics-20261113`，提交 `521ace47907a458f309133320e6dcc66de88850c`。\[P1–P4\]

实际历史运行：`83f833a1ef5e4515825c1457b6e4278d`。诊断目录：

```text
earth-representation/encoder_v3/reports/ab_spatial_diagnostics/v3_ab_20261113/
```

该目录的冻结配置、源码差异补丁、计算链、任务契约和结果文件共同构成比较依据。运行源码并不等于仅检出开发分支就一定完整可用：必须依照冻结清单、补丁及本地发布核对，不得根据文字猜写旧模型后称为原基线。\[P1–P4\]

已有结果支持：短程 A-only 对部分 S2 的 A 像元误差有收益；B1 的纹理对既有局部对比度支路有明显依赖；本次 B1 预热课程未通过筛选。它们**没有证明**动态卷积、融合或 Decoder 是唯一根因。旧 P2 缺乏有效的拟合闭环及末态产物，本版要补强，而非直接沿用其失败作容量结论。\[P3\]

### 0.3 本版研究判断

工作假设是：当前主路径在入口编码、块内布局传递、隐藏位置区分和空间展开上存在可改进的技术约束。这是待验证假设，不是已经确认的全部故障。

本版不讨论或承诺隐式学会 PSF/SRF，不增加反卷积、物理响应重建、物理先验损失或对应辨识实验。已有描述器字段可以保留以保证接口兼容，但不得据此声称已获得这些能力。

---

## 1. 目标与硬约束

### 1.1 目标

1. 在进入大范围空间建模前，使可见数值、可见性和块内位置得到明确、可追踪的表示。
2. 用描述器调节共享特征处理，不再让描述器生成整个前端空间卷积核。
3. 用浅层 ResNet 提取跨打包块的局部空间特征；用现有多尺度 Transformer 处理更大范围上下文。
4. 使 A 的不同隐藏位置具有不同的位置条件，而非只能从同一零向量读取全局摘要。
5. 让 Decoder 的块内输出具有可学习的位置自由度，并按原生足迹评价。
6. 在不新增独立原生高维旁路的前提下，分离前端、FiLM、位置编码和 Decoder 的贡献。

### 1.2 硬约束登记表

| ID | 约束 |
| --- | --- |
| H01 | 使用 **ResNet 残差卷积前端**，不使用 RetNet/Retention/RMT 替代。 |
| H02 | 不覆盖历史源码、发布、权重、数据和运行身份；新实现独立目录，旧实现可只读调用。 |
| H03 | 隐藏数值在所有内容计算前剔除；QA 无效值、隐藏值不得进入核、统计、局部细节、缓存或可见性选择。 |
| H04 | 不新增逐原生像元、逐通道长期驻留的高维 Skip/KNN 库；不新增原始影像直达新 Decoder 的路径。 |
| H05 | 本轮不新增 PSF/SRF 建模或验证，不引入 JEPA、外部预训练、额外预训练任务。 |
| H06 | 打包只重排；后续升维、残差块、融合和压缩不冒称无损。 |
| H07 | 像元原生支持、打包所有权、卷积感受野、可见支持、公共特征单元，必须分别记录。 |
| H08 | 不按传感器名设置模型分支；按打包倍率/合法几何共享的小型适配器允许存在，须记录参数。 |
| H09 | 第一工程版沿用当前三阶段压缩全局注意力主干，保持主干宽度/深度/头数与基线；位置条件作为显式开关。 |
| H10 | 不把历史 V3-H 的 `both / neighbor / parallel` 自动移植为本版新三主干。历史实验保留，本版先控制主干变量。 |
| H11 | 融合后的空间槽位顺序必须显式保留；禁止以“多个 Query”冒充多个位置，再不登记地压成单个均值。 |
| H12 | A/B1 是输入可见性与监督协议，不是仅在输出端贴两个标签；共享 Encoder，任务状态与评价分开。 |
| H13 | 既有 B1 对比度支路是历史事实，兼容对照中显式保留；关闭只作为单独干预。它不属于新前端的新增能力。 |
| H14 | 图像显示和评价分开；误差用未截断浮点预测计算，不用拼接真值图或自动拉伸后的图片。 |
| H15 | 训练、归一化、掩码和成员划分沿用冻结协议；不按新候选验证表现挑样本。 |
| H16 | 不承诺准确恢复缺乏观测依据的任意纹理，不把重建改善直接等同于下游表征改善。 |

---

## 2. 总计算图与范围

```text
原生通道数值 + QA + 网格/足迹
            │
      TaskViewBuilder(A/B1)
            │ 仅合法可见数值；目标另存
       NativeOrderedPack
            │ 数值 / 可见掩码 / 结构支持 / 子位置同步重排
       SharedContentLift
            │
       FiLMResNetFrontend
            │ 打包格上的浅层残差提取；相邻格交互
       PhaseAwareFusion
            │ 原生相位读出 → 空间槽位汇聚 → 同槽跨波段融合
            │ [N_ref, J, d_slot] → 无压缩重排 [1, D_f, H_ref, W_ref]
       SpatialPositionAdapter
            │ 空间/可见性状态明确，查询位置可区分
       ExistingMultiscaleBackbone
            │ F1 / F2 / F3；本版为已有逐级主干
       PyramidDecoder
            │ 原多尺度融合 + 可独立替换的末端空间展开
       TargetConditionedReadout
            │ 原生几何投影、目标选择；无目标数值输入
       原生预测 → TaskLoss / Evaluation

已有元数据 → SharedDescriptor(d_c)
                         ├→ FiLM heads（局部 ResNet）
                         ├→ Fusion condition heads
                         └→ Target readout head

兼容 B1 wrapper：主路径预测 + 已有、显式启用的合法可见对比度贡献
```

**主干尺度方向：** Encoder 沿用 Fine→Middle→Coarse 的既有逐级生成；Decoder 自顶向下读多尺度。本文没有新增三套并行 Encoder，也没有新增 Encoder 双向 cross-attention。若以后验证此类设计，应单独版本化，而非解释为图中箭头的默认含义。

**输入范围：** 第一版以当前冻结光学 A/B1 数据为执行范围。接口保留扩展可能，但不声称已经验证 SAR、点云、旋转网格或跨 CRS 联合输入。

---

## 3. 输入、可见性与类型接口

### 3.1 必须区分的掩码

| 名称 | 含义 | 用途 |
| --- | --- | --- |
| `structural_support` | 原始网格内真实像元/足迹；人工 padding 不在其中 | 几何、padding、合法 Query/K/V 域 |
| `qa_valid` | 质量允许参与任务的原生像元 | 目标与输入资格 |
| `visible` | 本任务本次视图允许输入的像元 | 内容编码，且须是 QA-valid 子集 |
| `target` | 本任务监督位置 | 预测选择与 loss；不送入内容编码作真值特征 |
| `observed_support` | 由可见原生支持计算的覆盖/存在性 | 告知模型何处有观测 |
| `inferred_feature_valid` | 经上下文处理后可输出特征的位置 | 不能冒充直接观测支持 |

`visible ∩ target = ∅` 适用于正常 A/B1；全可见自编码诊断是明确命名的独立协议，允许重建可见输入，但不得计入 A/B1成绩。

### 3.2 值与条件权限

```python
@dataclass(frozen=True)
class NativeBand:
    values: Tensor       # 原生、冻结 TRAIN 统计标准化后的浮点值
    qa_valid: Tensor     # bool [Hc, Wc]
    grid: NativeGrid     # affine / CRS / shape / footprint
    metadata: ChannelMetadata

@dataclass(frozen=True)
class TaskView:
    visible: tuple[Tensor, ...]
    targets: tuple[TargetSpec, ...]  # 几何、通道条件、输出选择；不含真值
    supervision: Supervision      # 只允许 wrapper / loss 访问
    context_geometry: SceneGeometry
```

模型不应通过 `sample` 对象随意访问全量真值。编码入口接收已经清理的值或只读 `VisibleBand`；Decoder 不接收 `NativeBand.values`。既有 B1 支路是 wrapper 中单独审计的兼容模块。

使用 `where(visible, value, 0)` 清理，先清理再执行依赖数值的统计/变换，避免 `NaN * 0` 残留。`0` 是标准化空间填充值，不是已观测的物理反射率零；可见掩码必须始终显式存在。\[P4\]

完全隐藏波段不产生可见前端特征，不参与可见描述器均值或波段集合条件。目标描述器只进入已有契约允许的目标读出。场景参考几何在视图外预先冻结，不应随请求预测哪个波段而变化。

---

## 4. 原生有序打包 NativeOrderedPack

### 4.1 工程倍率规则 E/G

使用标量倍率 `r_c ∈ {1, 2, 4}`，取：

\\\[
r\_c=\\max\\{r\\in\\{1,2,4\\}:r\\max(g\_{x,c},g\_{y,c})\\le L\_{pack}\\},
\\\]

若集合为空，取1。第一工程默认 `L_pack = 40 m`，仅用于与当前40米基线保持可解释联系；它是配置，不是所有数据的自然尺度。

示例：10米→r=4；20米→r=2；30米→r=1。若 GSD 大于40米，仍为r=1，不能把后续更密集特征格解释为新增观测分辨率。各倍率由几何得到，不按 S2/L8 名称设置。

GSD必须是按现有合法几何定义得到的米制间距，不得直接把经纬度affine数值当米。非方形像元仍按实际 x/y 尺寸记录支持。首版只在已支持的轴对齐、同一投影几何上运行；不支持的输入显式拒绝，不以包围盒代替旋转像元精确支持。

### 4.2 元素顺序

对右/下边界作最少零padding，使尺寸能被r整除；padding的所有有效/可见位为False。固定相位顺序：

\\\[
a=a\_y r+a\_x,\\quad 0\\le a\_y,a\_x\<r.
\\\]

输出：

- `u[c, a, h_p, w_p]`：有序可见数值，隐藏值为0；
- `v[c, a, h_p, w_p]`：原生相位可见掩码；
- `s[c, a, h_p, w_p]`：结构支持；
- `phase_geometry`：每个相位的原生索引、中心与足迹；
- `pack_geometry`：打包父格所有权及尺寸；
- `crop_spec`：逆重排后裁回原形状所需信息。

PixelUnshuffle/PixelShuffle 的重排关系是路线B的参考，不要求只能使用这两个API。\[R3–R4\]

### 4.3 兼容不同r的统一入口

使用最多16个有序数值槽位；r²个有效槽位按上面的顺序填入，其余是结构无效槽位。原生像元槽位数16只是 `r_max=4` 的工程结果，不能写成传感器常量。

打包倍率和几何必须随数据携带。相同通道索引在不同r下不自动代表相同物理位置；后续相位读出依照r与实际几何解释，不跨传感器按槽位编号直接硬对齐。

### 4.4 保证的边界

**保证：** 清理后的合法数值和相应掩码，可通过逆重排逐元素恢复。

**不保证：** 卷积、FiLM、升维残差、融合或主干之后仍可逆。

不可用非线性层之后的恢复误差证明“重排本身有损”；也不能只通过重排单测，就宣称整个前端保真。

---

## 5. 共享内容升维 SharedContentLift

### 5.1 首版接口

输入是打包数值、掩码和结构支持；输出 `H0[c, D_front, H_p, W_p]`。工程默认 `D_front=32`。

建议的可测试初始化：

\\\[
H\_0=P\_r u+E\_m(v,s)+\\eta\\,E\_{res}(u,v,s),
\\\]

其中：

- `P_r` 是固定满列秩嵌入，将实际数值槽位写入前16维的对应位置；
- `E_m` 的初始输出只占后16维，编码可见/结构状态；
- `E_res` 是共享轻量MLP，最后一层初始化为零；工程默认 `η=0.1`；
- 没有任何可见相位的父格，在可见前端中输出零并标记 `observed=False`。

在初始化、且给定掩码时，可直接从数值承载维读回u；这项初始可读性应有测试。训练后若学习残差改变这些维度，不再宣称精确可逆。

不得直接对单个原始标量执行 LayerNorm 后替代原值。后续归一化位于已升维的学习分支，输入内容仍走残差路径。

### 5.2 参数共享

优先共享 `E_m/E_res`，借助槽位padding兼容r。相位读出的r适配器按倍率共享，不按卫星名、波段名建立独立网络。已有描述器保留基线字段，不新增可学习传感器ID表作为捷径。

---

## 6. 描述器与 FiLMResNetFrontend

### 6.1 描述器

沿用现有有效字段解析和缺失/适用位约定，输出 `d_c ∈ R^32`。不把文件路径、观测ID、地区ID或数据角色作为学习输入。输入GSD等已知几何属性可保留；SRF曲线、PSF核和额外物理训练目标不在本版新增范围。\[P1\]

三个独立输出头：

1. `FiLMHead_l(d_c)`：对应第l个残差块的γ/β；
2. `FusionCondition(d_c)`：同槽跨波段融合条件；
3. `TargetReadout(d_t)`：目标通道读出条件。

共享描述器不意味着三个头共享所有参数。完全隐藏的目标描述器不进入可见前端条件汇总。

### 6.2 FiLM定义

\\\[
\\widetilde H=(1+\\Delta\\gamma\_l(d\_c))\\odot H+\\beta\_l(d\_c).
\\\]

每个通道、每层生成 `2 D_front` 个标量，在该通道的所有空间位置共享，不逐像元生成。FiLM的逐特征仿射条件化来自\[R1\]；恒等初始化、放在残差分支和具体插入位置是本项目设计选择，不是论文对本项目效果的保证。

初始 `Δγ=0, β=0`。记录γ/β分布，不在训练后静默clamp。可对照关闭FiLM，但仍保留描述器在融合和目标读出的作用。

### 6.3 浅层ResNet块 E

工程默认2个残差块；每块2个3×3、stride=1共享卷积，宽度32。不是分类ResNet stem，不带7×7 stride2、max-pooling、分类头或额外降采样。

```text
输入H ────────────────────────────────────┐
  └→ 每位置通道LN → FiLM → GELU             │
      → masked 3×3 conv                     │
      → 每位置通道LN → GELU                 │
      → masked 3×3 conv → residual_scale ── + → 输出
```

`residual_scale` 初始0.1，可学习；第二卷积采用小幅非零初始化，使分支初始接近恒等而非同时把多个梯度入口关死。不得同时将残差门、最后卷积和调制入口全部置为阻断梯度的零值。

ResNet用于学习相对输入的残差是参考\[R2\]的基本思想；本项目的浅层结构、掩码策略及FiLM组合均需独立验证。

### 6.4 掩码卷积契约

首版可用下面明确的掩码卷积作为工程实现，而不是把零填充当普通观测：

\\\[
MC(H,O)=\\frac{K\*(H\\odot O)}{\\max(n(O)/k^2,1/k^2)}+b,
\\\]

仅在中心有可见父格支持 `O0=True` 时保留输出；`n=0` 的输出置零。分母仅由掩码计算，单独记录。每块仍以原始 `O0` 为输出可见支持，不把邻域推断出的内容升级成直接观测。

这是模型算子定义，不保证任意学习卷积的常数场保持；常数场守恒只在定义为归一化面积平均的几何算子上断言。

前端归一化采用**每位置、沿特征维的LayerNorm**，不使用跨影像空间统计的BatchNorm/GroupNorm来制造切块依赖。原始mask状态另外保留，不依赖网络自行从零值推断。

### 6.5 跨打包块交互与执行

卷积必须跨相邻父格。两个双3×3块的最大依赖半径为4个打包格；切块的halo按真实算子组合计算，不能固定猜一个像元。结果只写回中心区域，halo不重复计loss。

允许按通道/行块流式执行和重计算；不允许每个打包块独立前向后拼接。分块必须保持相同邻域和相同参数梯度，不能把减少上下文当作等价优化。

---

## 7. 空间槽位保留的几何跨波段融合

### 7.1 为什么必须定义这个接口

当前历史融合是四个光谱Query读取后，将128维压成每公共格32维；四个Query没有显式子格位置。\[P1\]

新前端不能仅换一个编码器后，再不加审计地把原生相位信息用父格平均抹去。本文定义一个明确的**空间槽位融合**候选，并承认它仍然是受控有损表示。

### 7.2 工程空间布局 E/G

公共父格物理边长 `L_ref` 工程默认40米。场景范围由冻结的完整参考几何计算：

\\\[
H\_{ref}=\\lceil(y\_{max}-o\_y)/L\_{ref}\\rceil,
\\quad W\_{ref}=\\lceil(x\_{max}-o\_x)/L\_{ref}\\rceil.
\\\]

原点沿用可复核基线几何规则，且不随执行chunk、目标请求或遮挡掩码改变。

每个父格分为 `q×q` 个**空间槽位**。首版 `q=2, J=4, d_slot=8`，因此 `J*d_slot=32=D_fine`。槽位有固定、不同的物理支持与相对中心，顺序固定为行优先。

这里的2×2是特征容量配置，不是声称所有观测都具备20米细节。它与原生r\_c相互独立；原生GSD、实际可见足迹必须继续携带。q的修改是新配置；不能为多放槽位而静默压低每槽宽度。

### 7.3 从父格特征按相位读出，而不是重复广播父格向量

对原生打包父格j中的相位a，由 `H[c,j]` 生成低维局部描述：

\\\[
z\_{c,j,a}=U\_{r,a}H\_{c,j},\\quad z\\in\\mathbb R^8.
\\\]

`U_r`可一次输出所有r²相位，但按需分块物化；初始至少一个输出分量与对应数值承载维有关。不得把同一个H原样复制到全部相位，再称每个相位保留独立内容。

此处**不能再读取原始values**补充细节。数值来源必须是主路径H；原生几何和掩码仍可使用。因而这不是新增原始影像旁路。

每个相位对应原生像元支持；卷积感受野只作上下文来源，不替代该所有权足迹。

### 7.4 同一槽位内的空间汇聚

对每个波段c、公共槽位s，精确求可见原生相位与槽位的正面积交叠 `a_{c,i,s}`，建立可缓存几何关联。仅 `visible=True` 的相位贡献内容。

为了不使左/右等布局退化为同一个普通平均，空间值嵌入至少允许内容与位置相乘，例如：

\\\[
\\psi\_{c,i,s}=V\\bigl(\[z\_i,\\delta\_x z\_i,\\delta\_y z\_i,\\delta\_x\\delta\_y z\_i\]\\bigr),
\\\]

其中相对中心偏移δ按槽位边长归一化；V是共享投影。之后：

\\\[
u\_{c,s}=\\frac{\\sum\_i a\_{c,i,s}\\psi\_{c,i,s}}{\\sum\_i a\_{c,i,s}},\\qquad
\\rho\_{c,s}=\\frac{\\sum\_i a\_{c,i,s}}{|s|}.
\\\]

空支持返回零和invalid标记。原生像元互不重叠时ρ位于\[0,1\]，误差容差之外越界应报错。高阶位置矩只提供可区分性，不是对任意块内图案的无损编码保证。

粗像元与多个槽位相交时，可按交叠分配其内容，但这些贡献仍来自同一粗观测，不能计为新增独立观测。

### 7.5 同槽跨波段融合

在固定空间槽位s中跨可见通道做条件attention：

\\\[
q\_s=Q(\\text{slot position},\\text{visible descriptor summary}),
\\quad k\_{c,s}=K(u\_{c,s},d\_c,\\rho\_{c,s}),
\\quad v\_{c,s}=V\_f(u\_{c,s},d\_c),
\\\]

\\\[
Z\_s=\\sum\_{c:\\rho\_{c,s}\>0}\\mathrm{softmax}*c\\left**(
**q\_**s^Tk*{c,s}/\\sqrt{d\_{slot}}+\\log\\rho\_{c,s}\\right)v\_{c,s}.
\\\]

先在每个通道内进行面积归约，再在通道间融合，避免细分辨率通道仅因像元数量多而获得重复计数优势。所有通道仍需逐一使用，不允许为加速静默截断高光谱波段。

空槽输出可学习mask内容或零内容，并携带明确的observed=False；位置条件由下一节提供。该操作融合的是**同一空间槽位内的通道轴**，不是把全部空间槽位一起聚成一个无位置摘要。

### 7.6 向现有主干的无压缩适配

输出 `Z[N_ref, 4, 8]`，只执行固定顺序的reshape：

\\\[
\[N\_{ref},4,8\]\\leftrightarrow\[1,32,H\_{ref},W\_{ref}\].
\\\]

**本接口不增加128→32等新的学习压缩层。** 四个空间槽位被编码为四组通道，而不是变成4倍空间Token；主干仍在父格网格上运行。解码时也不能把它们错当四个独立Transformer位置。

Transformer可进一步混合这些通道；不要求槽位语义在每一层始终不变。初始化和接口处的槽位顺序明确，后续可读性由探针验证。

这一首版选择保持主干输入宽度与Token数接近基线，但每槽8维仍可能是瓶颈。若被证据否定，再建立“保留槽位为空间Token/增宽”的独立版本，不能在本轮自动扩容。

### 7.7 流式与缓存

几何关联由完整原生affine、CRS、尺寸、r、L\_ref、q、原点和实现版本决定；与像元值、任务、QA及随机掩码无关。实际可见性每次视图过滤。

按槽位tile及通道块处理，可用在线softmax累计通道状态；不能缓存带梯度的上轮特征，也不能缓存过时可见性。缓存的通道覆盖总和不是跨通道联合覆盖面积。

---

## 8. 隐藏位置区分与多尺度Transformer

### 8.1 继承的主干 E

首版保持历史工程配置：\[P2\]

| 阶段 | 特征宽度 | heads | blocks | K/V压缩倍率 | 父格物理采样 |
| --- | ---: | ---: | ---: | ---: | --- |
| Fine | 32 | 2 | 2 | 8 | L\_ref |
| Middle | 64 | 4 | 2 | 4 | 2L\_ref |
| Coarse | 96 | 4 | 2 | 2 | 4L\_ref |

阶段之间沿用3×3 stride2变换。各级真实形状按输入几何和padding计算，不写死空间大小。保留现有压缩全局K/V注意力及FFN，不替换成另一套Transformer。

首版默认不增加三分支并行生成，不新增Encoder双向交互；多尺度输出与Decoder融合保留。未来是否恢复历史V3-H式并行/双向设计另立试验。

### 8.2 位置适配器的必要职责

不同隐藏位置可能拥有相同内容，但不能因此被强制赋予同一个位置条件。位置编码不能从目标真值或目标纹理计算。

采用相对完整场景原点的连续坐标，兼顾归一化场景位置和物理尺度：

- `x_rel / max(scene_width, L_ref)`、`y_rel / max(scene_height, L_ref)`；
- 该级物理步长、场景宽高相对L\_ref的对数；
- 沿x/y、周期为 `{4,16,64}*L_ref` 的sin/cos。

不输入绝对经纬度/地区ID用于记忆场景。相对坐标与几何参数在执行切块时仍以完整场景为准。

### 8.3 首版位置注入 E/X

`position_mode = qk_physical`：分别为各级生成与其宽度相同的位置特征，添加到内容归一化后的Query输入及压缩K的位置输入；V维持内容路径。压缩K的坐标按真实池化格中心/支持中心计算，不简单截取错误位置。

`position_mode = none` 必须退化为原主干数值路径，用于等价性门禁。位置投影采用小幅非零初始化，不能全部置零后以“已经加入位置”作为完成证明。

空槽/隐藏位置在结构支持内可作为Query；padding不是合法Query/K/V。`observed=False`与`structural=False`不是同一个mask。已有上下文生成的特征可以参与后续层，但必须与原始观测覆盖统计区分。

测试重点是不同隐藏位置的Query条件是否可区分，以及构造非均匀K/V时是否产生位置相关读取；不要求所有自然图中的隐藏位置必须输出不同值。

---

## 9. Decoder与原生读出

### 9.1 模块边界

```text
EncodedPyramid(F1,F2,F3, supports, geometry)
       → ExistingTopDownFusion(width=32)
       → TerminalSpatialExpander(strategy)
       → SharedFineField
       → TargetDescriptorReadout
       → NativeFootprintProjection
       → target selection
```

Encoder不接收监督真值；Decoder只接收编码结果、目标描述器、位置/足迹和目标选择。改变要预测的目标集合，不应改变相同目标位置的主路径预测。

### 9.2 两种明确的末端策略

**D0：**`legacy_continuous`**。** 原多尺度融合、原8维共享场插值与细化、原线性目标读出；只读兼容基线。

**D1：**`learned_subpixel_residual`**，本版优先候选。** 保持多尺度融合与目标读出接口不变，只替换最后空间展开模块：

1. `1×1 Conv: 32 → (r_dec² × 8)`；
2. PixelShuffle或等价重排到更密的共享特征场；
3. 局部残差细化 `3×3: 8→16 → GELU → 3×3:16→8`，stride均为1；
4. 仍用原目标描述器线性读出，以隔离空间展开与读出非线性；
5. 精确按目标原生足迹归约，之后才应用目标mask。

工程默认 `r_dec=4`，对应L\_ref/4的**输出采样间隔**；它不代表传感器真实分辨率。不同相位有独立的输出权重，邻近父格通过上游融合及高分辨率局部卷积交换信息。\[R4\]

D1的参数量与计算量通常不同，须报告；不称为等容量实验。以平滑基函数或邻域等价构造初始化可另测，但不得把不完全等价的初始化称为精确复用。

### 9.3 不得复用一个历史总开关制造混杂

新增 `terminal_expander` 独立开关，**不要直接切换旧 **`decoder_mode`。历史总开关还会改变顶层上采样路径及B1支路调用，直接切换不构成末端单因素对照。\[P1\]

`target_readout = legacy_linear` 首版固定。位置条件非线性读出属于后续X候选，不与D1同时默认引入。共享场8维仍可能限制光谱表达，本版不声称已解决这一问题。

### 9.4 B1历史支路的隔离

建立独立 `LegacyB1Detail`，与Decoder模式无关。兼容门禁复现历史的可见同网格5×5局部对比度、目标条件增益和适用网格条件。\[P4\]

本轮配对训练中默认保留它；`main_path_only=True` 用于全可见诊断及只读干预。两种输出必须同时可获得：`main_prediction`、`legacy_detail_contribution`、`total_prediction`。

保持当前40米参考配置时，适用的最大GSD为20米。若以后改变L\_ref，必须登记支路阈值是否保持物理20米还是依旧按历史比例计算，不能让几何参数顺便改变此支路。

---

## 10. 关键张量账本

以下是单观测、逐网格分组的示意。C不同通道可能有不同H/W，不能先强制堆成同尺寸原始影像。

| 接口 | 形状 | 保留或改变的轴 |
| --- | --- | --- |
| 原生波段 | `C × variable[Hc,Wc]` | 原生像元与原生几何 |
| 清理数值 | 同原生 | 隐藏/无效值置零，掩码保留 |
| 打包 | `[C_group,r²,Hp,Wp]` | 空间相位→有序通道，无求平均 |
| padded槽位 | `[C_group,16,Hp,Wp]` | 补无效槽位，不添加观测 |
| 内容升维 | `[C_group,32,Hp,Wp]` | 学习表达，初始数值可读 |
| FiLM参数/块 | `[C_group,2,32]` | 广播空间维，不逐像元生成 |
| ResNet输出 | `[C_group,32,Hp,Wp]` | 保持空间尺寸，局部跨格交互 |
| 按需相位描述 | `[tile_native,8]` | 从H读取；不长期驻留全图逐像元高维库 |
| 每通道空间槽 | `[channel_chunk,N_ref,4,8]` | 几何和位置条件汇聚，可流式 |
| 融合槽位 | `[N_ref,4,8]` | 消去观测通道轴，保留4个明确空间槽 |
| 主干输入 | `[1,32,H_ref,W_ref]` | 仅reshape，不额外学习压缩 |
| 多尺度输出 | `[1,32,H,W] / [1,64,H2,W2] / [1,96,H3,W3]` | 继承层次下采样 |
| Decoder父格场 | `[1,32,H,W]` | 多尺度融合 |
| D1末端重排前 | `[1,8*r_dec²,H,W]` | 为子位置生成不同权重输出 |
| 细场 | `[1,8,H*r_dec,W*r_dec]` | 共享表示，不按目标通道复制长期存储 |
| 原生输出 | 每目标波段自己的原生网格/packed target | 面积归约后选择目标 |

例：10米原生128×128、L\_pack=L\_ref=40且参考原点对齐，r=4，父格32×32；空间槽位为每父格2×2；主干仍32×32位置；D1可输出128×128的采样场。错位、边缘和其他GSD必须按几何计算，不能套这个整除例子。

---

## 11. A/B1监督与优化契约

完整沿用冻结任务契约，除非新实验明确登记。\[P4\]

- A：128米地理遮挡区域、名义比例0.55；原生像元足迹与被选区域相交则按原协议成为目标。实际像元比例不强制等于0.55。
- B1：沿用已冻结整波段选择逻辑；完全隐藏波段无可见内容特征。允许目标描述器进入目标读出。
- 正常比较目标：逐目标波段像元Huber δ=1，再平均波段；B1保留0.2倍原生相邻目标梯度监督。
- 任务权重：符合资格的A/B1按原0.5/0.5归一化；无资格任务不得执行前向或影响状态。
- 数据：冻结TRAIN归一化、QA、成员和分组，VAL固定视图；不得因新几何困难悄悄删样本。
- 不新增A梯度损失，不改变遮挡难度，不自动恢复B1预热课程。
- 工程拟合诊断可用MSE直接拟合，但必须独立命名，不把其loss和正式Huber混画。

首版比较沿用AdamW和原6990步学习率前缀，不把233步工程预算压成一个新cosine周期。原配置为lr=1e-4、min\_lr=1e-5、weight\_decay=0.04、warmup=12、全局batch32、FP32；实际以P0冻结记录为准，禁止据此自动升级PyTorch或CUDA。\[P2\]

---

## 12. 实现组织与API建议

新增目录，不要求旧仓库已经存在以下路径：

```text
src/encoder_v3/film_resnet/
  config.py             # 严格schema、配置版本与约束
  types.py              # visible inputs / pack / slots / pyramid / targets
  packing.py            # 有序打包、padding、inverse、倍率规则
  lift.py               # 内容升维与初始可读性
  film.py               # 条件头与恒等开关
  resnet.py             # masked stride1残差块与halo
  geometry.py           # 支持现有几何的槽位映射、缓存
  fusion.py             # 相位读出、位置值嵌入、跨波段流式融合
  position.py           # Q/K几何位置条件
  backbone_adapter.py   # 继承现有主干，不复制私自修改历史代码
  decoder.py            # top-down / terminal / readout独立开关
  legacy_detail.py      # B1既有贡献的隔离兼容
  model.py              # Encoder与Decoder组合
  tasks.py              # task wrapper；复用已验收协议
  diagnostics.py        # 只读审计与原始张量导出
  train.py              # 独立工程入口、恢复与停止

tests/encoder_v3/film_resnet/
configs/film_resnet/
scripts/run_v3_fr.py
outputs/encoder_v3/film_resnet/<experiment_id>/
```

关键方法建议：

```python
pack(visible_band, pack_spec) -> PackedBand
unpack_values(packed_values, crop_spec) -> Tensor
frontend(packed, descriptor) -> FrontendBand
fusion(frontend_bands, scene_geometry) -> SpatialSlots
encoder(visible_bands, scene_geometry) -> EncodedPyramid
decoder(encoded, target_specs) -> NativePredictions
wrapper(task_view, enabled_tasks) -> PredictionAndLoss
```

`Encoder`输出不得要求保留原始图像才能被下游使用。训练后可丢弃Decoder；主路径空间信息必须存在于Encoder输出，而不是只存在于任务旁路。

---

## 13. 首个完整工程配置 E

下列配置是待实现schema的示例，不是当前已有CLI可直接运行的声明。Agent应按其语义实现并输出解析后配置；不得用默认值悄悄覆盖未知字段。

```json
{
  "architecture": "v3_film_resnet",
  "schema_version": "1.0",
  "descriptor": {"reuse_baseline_schema": true, "width": 32},
  "packing": {
    "method": "ordered_native",
    "target_metres": 40.0,
    "allowed_factors": [1, 2, 4],
    "selection": "largest_factor_not_exceeding_target",
    "padding": "right_bottom_zero_invalid",
    "phase_order": "row_major",
    "max_value_slots": 16
  },
  "frontend": {
    "width": 32,
    "lift": "fixed_value_embedding_plus_learned_residual",
    "blocks": 2,
    "convolutions_per_block": 2,
    "kernel_size": 3,
    "stride": 1,
    "normalization": "per_location_channel_layernorm",
    "residual_scale_init": 0.1,
    "film": {"enabled": true, "identity_init": true},
    "visible_output_support": "original_pack_visible"
  },
  "fusion": {
    "reference_metres": 40.0,
    "slot_grid": [2, 2],
    "slot_width": 8,
    "phase_feature_width": 8,
    "spatial_value_embedding": "content_relative_position_products",
    "channel_fusion": "descriptor_conditioned_attention",
    "adapter": "reshape_only",
    "no_support": "mask_content_with_observed_flag"
  },
  "backbone": {
    "implementation": "existing_mit_global_reduced",
    "widths": [32, 64, 96],
    "heads": [2, 4, 4],
    "depths": [2, 2, 2],
    "kv_reductions": [8, 4, 2],
    "cross_scale": "existing_bottom_up",
    "position_mode": "qk_physical"
  },
  "decoder": {
    "pyramid_fusion": "existing_top_down",
    "width": 32,
    "field_width": 8,
    "expansion_factor": 4,
    "terminal_expander": "learned_subpixel_residual",
    "refine_hidden_width": 16,
    "target_readout": "legacy_linear",
    "legacy_b1_detail": true
  },
  "objectives": {
    "reuse_frozen_task_contract": true,
    "task_weights": {"A": 0.5, "B1": 0.5},
    "huber_delta": 1.0,
    "b1_gradient_weight": 0.2,
    "new_physics_objective": false
  },
  "execution": {
    "precision": "float32",
    "reference_executor": "eager",
    "cuda_graphs": false,
    "long_training_authorized": false
  }
}
```

需要schema强制校验：`slot_count*slot_width == widths[0]`；`max_value_slots >= max(r)^2`；主干宽度可整除heads；pack、融合、Decoder相位顺序可互转；子格面积和父格面积一致；不能把target数值放入任何条件参数；无法支持的几何显式报错。

配置中的40米、q=2和r\_dec=4只冻结为本次工程参照，不是永久架构定理。改变这些参数必须报告实际Token、显存、配对数、原生读出误差和比较身份。

---

## 14. 性能与容量约束

设各通道打包后位置数为N\_c，宽度D，B个残差块。两次标准3×3卷积的MAC量约为：

\\\[
MAC\_{front}\\approx\\sum\_c 18 B N\_c D^2,
\\\]

不含归一化、FiLM、融合、反向与重计算。FiLM参数生成按每通道每层一次计；广播的激活运算仍按空间位置计，不能当零成本。

相位读出和精确几何汇聚可能仍与原生有效像元数量相关；打包不使这些成本自动消失。多波段注意力必须统计实际参与波段、槽位和支持关联，不能只报Transformer FLOPs。

必须分别报告：冷/热缓存、前向/反向/验证/保存、数据与几何时间、每设备峰值、整场景与chunk、失败重试成本。不得根据参数量宣称更快。

高维中间特征按tile与通道块物化，完成聚合后释放。默认不存全场景所有波段的相位K/V。为避免OOM而减少halo、丢波段、丢目标或缩小完整场景属于模型/数据变化，不是执行优化。

新模型首先用eager验证；历史CUDA Graph不能直接套用新计算图。重新捕获或多卡执行必须通过前向、梯度和完整下一步优化器状态对照。

---

## 15. 验收指标与可视化

### 15.1 内容利用与退化诊断

每个固定样本/波段/任务至少保存：真值与预测均值、标准差、分位数、有限值比例、相关系数、方差比、有效目标像元数、真实/预测梯度能量、signed梯度误差、相对标准化零预测基线的改善。

低方差真值的归一化比值单独标记，不让近零分母制造巨大指标。只计算合法目标；占位白块/NaN不能参与统计。

### 15.2 空间指标

- 原历史MSE与高纹理梯度指标，保留归约定义；
- 边界/内部残差与有效像元对数，至少原相位与半格相位；
- A按到合法可见像元的物理距离分层；
- B1按GSD、可见波段数与主路径/历史支路贡献分层；
- 在完整合法支持的诊断图中做分尺度高通残差；带孔洞掩码不得直接零填充FFT后宣称频谱证据；
- 简单斜坡、阶跃、细线、脉冲及方向相反图案的相位响应。

边界比接近1仅说明指定统计下没有额外边界残差增幅，不等于没有块效应。

### 15.3 必存可视化产物

每组保存可重新绘制的 `.npz` 或等价浮点张量及元数据：`truth, visible_values, qa, visible_mask, target_mask, main_prediction, detail_contribution, total_prediction, geometry, display_scale`。

图至少有真值、实际可见输入、**未与真值拼接的预测**、目标mask、绝对误差和梯度误差。B1除目标波段的空白输入图外，应展示预先指定的实际可见参考波段。

同样本候选共用真值冻结的色阶，并另存原始数值范围/超界比例。主图不对每个模型分别自动拉伸。图名写清Run、更新步、任务、波段、成员、选例规则；`best`必须注明是相对MSE变化而非最佳重建质量。

---

## 16. 已知风险与本版不作的承诺

| 风险 | 处理方式 |
| --- | --- |
| FiLM缩放造成幅值退化 | 恒等初始化、γ/β统计、关闭对照、原始输出分布 |
| 打包相位敏感或边界伪影 | 全图/halo等价、相位偏移诊断、跨父格ResNet交互 |
| 槽位每槽8维不够 | 明确作为压缩候选，读出探针；不把四槽当无限容量 |
| 编码相位被融合再次抹平 | 分别检查相位读出、空间归约、跨波段融合和reshape |
| 新前端仍丢内容 | 可逆重排正控、初始读回、小样本AE、前端探针 |
| A隐藏位置仍近似相同 | Q/K位置适配器、逐层方差与查询差异；不只增加共享mask token |
| D1出现棋盘/相位伪影 | 常数/斜坡训练及输出相位统计，D0/D1独立对照 |
| B1旁路掩盖主路径弱 | 保存主路径与额外贡献，AE关闭旁路，评价不混淆 |
| 新模型更慢 | 完整成本和最大场景前后向；不默认更快 |
| 单种子短程结论脆弱 | 短程仅筛选，晋级须独立预算的多种子与下游评价 |

不得因某个候选失败就直接恢复逐像元高维库，也不得用“未收敛”绕过基本可读性与实现门禁。

---

## 17. 设计—验证追踪矩阵

| 设计项 | 实现模块 | 必须的验证 | 计划阶段 |
| --- | --- | --- | --- |
| 原生打包无损 | packing.py | inverse、padding、mask、梯度、r=1/2/4 | S2 |
| 初始内容可读 | lift.py | 固定左逆、隐藏值干预、零支持 | S2 |
| FiLM调制 | film.py | 恒等开关、描述器响应、共享参数 | S2 |
| 浅层ResNet | resnet.py | 跨格响应、halo、参数梯度等价 | S2 |
| 相位与空间槽 | geometry.py / fusion.py | 原生相位置换、方向图案、交叠解析参考 | S3 |
| 跨波段而非跨空间混叠 | fusion.py | 通道顺序不变性、分块softmax、布局读出 | S3 |
| 隐藏位置条件 | position.py | none等价、位置Query差异、非均匀K/V试验 | S3 |
| 主干继承 | backbone\_adapter.py | 配置、shape、参数映射与旧路径等价 | S1/S3 |
| 独立末端展开 | decoder.py | D0复算、D1相位顺序、原生映射 | S4 |
| 既有B1贡献 | legacy\_detail.py | 开关独立、固定权重干预复核 | S1/S4 |
| 信息/优化诊断 | diagnostics.py | 小样本有效拟合、全中间产物 | S5 |
| 质量与成本 | train.py / eval | 配对掩码、初始化、曲线与资源账本 | S6–S7 |

---

## 18. 参考与证据登记

外部材料只支持基本机制，不构成本候选在遥感数据上有效的证据。文中组合、公式中的项目适配、容量与初始化属于本设计。

- **\[****R1\] FiLM.** Perez et al., *FiLM: Visual Reasoning with a General Conditioning Layer*, AAAI 2018。支持逐特征仿射条件化。
  - [https://arxiv.org/abs/1709.07871](https://arxiv.org/abs/1709.07871)
- **\[****R2\] ResNet.** He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016。支持残差学习形式，不代表采用完整分类网络。
  - [https://www.microsoft.com/en-us/research/publication/deep-residual-learning-for-image-recognition/](https://www.microsoft.com/en-us/research/publication/deep-residual-learning-for-image-recognition/)
- **\[****R3\] PyTorch PixelUnshuffle.** 支持空间到通道的元素重排定义。
  - [https://docs.pytorch.org/docs/stable/generated/torch.nn.PixelUnshuffle.html](https://docs.pytorch.org/docs/stable/generated/torch.nn.PixelUnshuffle.html)
- **\[****R4\] PyTorch PixelShuffle.** 支持通道到空间的重排定义；不保证学习到的展开无伪影。
  - [https://docs.pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html](https://docs.pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html)

项目证据统一固定到提交 `521ace47907a458f309133320e6dcc66de88850c`：

- **\[****P1\] 计算链与源码补丁：**
  - [https://github.com/niupai1001/disaster-detect/blob/521ace47907a458f309133320e6dcc66de88850c/earth-representation/encoder_v3/reports/ab_spatial_diagnostics/v3_ab_20261113/representation_trace.csv](https://github.com/niupai1001/disaster-detect/blob/521ace47907a458f309133320e6dcc66de88850c/earth-representation/encoder_v3/reports/ab_spatial_diagnostics/v3_ab_20261113/representation_trace.csv)
  - [https://github.com/niupai1001/disaster-detect/blob/521ace47907a458f309133320e6dcc66de88850c/earth-representation/encoder_v3/reports/ab_spatial_diagnostics/v3_ab_20261113/p0_baseline/worktree_vs_HEAD.patch](https://github.com/niupai1001/disaster-detect/blob/521ace47907a458f309133320e6dcc66de88850c/earth-representation/encoder_v3/reports/ab_spatial_diagnostics/v3_ab_20261113/p0_baseline/worktree_vs_HEAD.patch)
- **\[****P2\] 冻结配置：**
  - [https://github.com/niupai1001/disaster-detect/blob/521ace47907a458f309133320e6dcc66de88850c/earth-representation/encoder_v3/reports/ab_spatial_diagnostics/v3_ab_20261113/p0_baseline/config.json](https://github.com/niupai1001/disaster-detect/blob/521ace47907a458f309133320e6dcc66de88850c/earth-representation/encoder_v3/reports/ab_spatial_diagnostics/v3_ab_20261113/p0_baseline/config.json)
- **\[****P3\] 诊断报告与决策：**
  - [https://github.com/niupai1001/disaster-detect/blob/521ace47907a458f309133320e6dcc66de88850c/earth-representation/encoder_v3/reports/ab_spatial_diagnostics/v3_ab_20261113/summary/REPORT.md](https://github.com/niupai1001/disaster-detect/blob/521ace47907a458f309133320e6dcc66de88850c/earth-representation/encoder_v3/reports/ab_spatial_diagnostics/v3_ab_20261113/summary/REPORT.md)
  - [https://github.com/niupai1001/disaster-detect/blob/521ace47907a458f309133320e6dcc66de88850c/earth-representation/encoder_v3/reports/ab_spatial_diagnostics/v3_ab_20261113/summary/DECISION.md](https://github.com/niupai1001/disaster-detect/blob/521ace47907a458f309133320e6dcc66de88850c/earth-representation/encoder_v3/reports/ab_spatial_diagnostics/v3_ab_20261113/summary/DECISION.md)
- **\[****P4\] 任务契约：**
  - [https://github.com/niupai1001/disaster-detect/blob/521ace47907a458f309133320e6dcc66de88850c/earth-representation/encoder_v3/reports/ab_spatial_diagnostics/v3_ab_20261113/task_contract.json](https://github.com/niupai1001/disaster-detect/blob/521ace47907a458f309133320e6dcc66de88850c/earth-representation/encoder_v3/reports/ab_spatial_diagnostics/v3_ab_20261113/task_contract.json)

执行时版本若变化，先登记差异，不用新HEAD覆盖本版引用。API能力应以实际安装依赖验证，不因网页显示新版本而升级运行环境。
