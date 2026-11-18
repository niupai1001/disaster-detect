# GeoToken–DualViT R3：每个地理 Patch 一个联合 Token

日期：2026-11-17。状态：实施规格＋局部张量参考；未训练、未修改仓库、未验证真实速度。

## 0. 本轮权威与范围

本方案按最近确认的设计收敛：无 Stem、描述器关闭，两个分支各一层标准 Transformer；分支内部先交互、后展平投影，各形成一个分支 Token；两个分支沿特征维拼接并投影为一个地理 Token；只用一层跨地理 Patch Transformer；单尺度输出。数据为核心 Sentinel-2，TRAIN:开发 VAL=7:3，全局父观测 batch=16。

本文件替代 R1、R2 中以下冲突条款：残差 Stem、完整细粒度 Token 全局融合、Sub-patch 级完整 Decoder 序列、两个跨 Patch Block、1000 TRAIN+100 VAL，以及把 B1 描述成删除整个地理 Token 的说法。旧文件只读归档。

标准的是 Transformer Block 与 A 的非对称 MAE 机制；双路径和固定布局压缩是本项目的组合设计。不能宣称已由 ViT/MAE 论文验证。此次不评价未读取的最新仓库运行，也不将旧结果当成本方案的成绩。

## 1. 固定计算图

```text
原始原生波段值＋QA＋固定布局（无可学习物理条件）
  → 固定地理 Patch → A/B1 独立遮盖 → 仅收集可见内容
    ├─ 光谱：每波段整个 Patch Linear → 波段 Self-Attention+FFN
    │         → 按原 band 槽位整理 → Flatten+Linear → s_p[D]
    └─ 空间：每波段 2×2 Sub-patch Linear → 每波段局部 Self-Attention+FFN
              → 按 band/row/col 槽位整理 → Flatten+Linear → v_p[D]
  → [v_p;s_p] 特征拼接，Linear(2D,D)＋可见波段标记
  → 每个可见地理 Patch 一个 Token
  → 固定地理位置编码 → 一层跨 Patch ViT → Encoder 最终层
  → MAE 投影/序列恢复 → 两层轻量 Decoder → 原生 Patch 向量预测
```

禁止把 [N,d] 的 Flatten 简单重命名为 [1,Nd] 后送入后端。必须投影到固定 D。禁止将分支全部细粒度 Token 再展平为全窗口长序列。

## 2. 几何、波段与固定布局

P 为窗口地理 Patch 数；N 为冻结布局的波段数；n_b=h_b*w_b 为波段 b 的每地理 Patch 原生像元数；M_b=n_b/4 为其 2×2 Sub-patch 数；R=sum_b n_b。

首版工程起点：地理 Patch=120m，步长120m；窗口4×4个 Patch=480m；2×2原生像元为 Sub-patch。只在核验共同米制北向原生栅格与边界对齐后使用。不插值原始数据，不把实际已重采样产品声称为原生。几何不满足则拒绝并报告，不能静默取整。

波段顺序、各原生 shape、输出 offsets、Sub-patch row-major 顺序和父观测锚点全部写入 Layout，并哈希冻结。N 不凭预算例子写死；从绑定数据清单确定。首版要求原始观测具有同一已声明波段布局；真正自然缺波段与任意新传感器泛化不在本轮验收范围。B1 是从完整合法布局模拟删除值，目标 band 及几何可以已知。

例：四个10m波段各12×12，六个20m波段各6×6，则 N=10、sum M_b=198、R=792。这只是成本例子，不是声称实际数据必定采用这些波段。60m波段若在真实绑定布局中，可对应2×2像元，不自动添加或删除。

先按父/地理组分割数据，再切窗口。大图每次读取登记窗口；不同曝光轮换窗口并记覆盖。小图可使用无效 padding：它没有观测值、不计监督、不作为真实可见 Patch。A至少需要两个有合法内容的地理 Patch。padding不是训练 mask。

## 3. 输入与线性 Embedding

数值为原生反射率，QA/finite/padding 与训练遮盖分开保存。不使用逐影像真值均值/方差归一化。无效输入用 where 安全置零，不能 NaN*0。

只将可见 (p,b) 输入两个分支；A隐藏区域、B1隐藏波段的原始值不进入 Embedding、Attention、统计或缓存。QA元数据可以使用，隐藏内容不可以。

光谱 embedding 按 h_b,w_b 共享（不是每颗卫星一个网络）：
E[p,b] = Linear_value(flatten(x[p,b])) + Linear_QA(flatten(valid[p,b])) + fixed_band_code[b]。
空间 embedding 对每个原生2×2块：
E_sp[p,b,s] = Linear_value(4 values) + Linear_QA(4 valid bits) + PE_local(s) + fixed_band_code[b]。

QA投影是输入有效性编码，不是物理描述器。没有重叠卷积、残差 Stem、预先邻域网络。可用 kernel=stride 的无重叠卷积实现等价线性投影，但不能夹入额外非线性特征提取。

## 4. 两个局部分支

统一使用 Pre-LN Block：x += MSA(LN(x)); x += MLP(LN(x))。宽度 d=128、4头、MLP倍率4、GELU、dropout=drop_path=0、LayerNorm eps=1e-6，无CLS。光谱与空间各一套参数，空间参数在各波段/地理 Patch间共享。

光谱：固定 p，在可见波段上执行一层 Block，输出 [N_visible,d]。隐藏 band 真正从该序列中删除；不能让 band 重新编号。QA无有效内容的 band 不作Key，可见性单独处理。
空间：固定 (p,b)，在 M_b 个局部位置执行一层 Block，输出 [M_b,d]。只计算可见波段；QA全无效的 Sub-patch屏蔽，空输入必须数值安全且不作为观测证据。

Self-Attention的职责是压缩前的输入相关交互；后续 Flatten只是重排，Linear才是压缩。两者不等价，但前者是否值得成本仍需实验。当前不增加多层分支或学习 Query。

## 5. 固定槽位压缩：让 B1 在不读隐藏值的情况下闭合

定义零初始化的固定缓存：S_fixed[p,N,d]，V_fixed[p,sum M_b,d]。仅把实际算出的可见输出 scatter 回原 band/sub-patch 索引；缺波段的缓存保持零。不是先算隐藏表征再清零，也不是用可学习 mask token在分支Attention中占位。

s_p = W_s Flatten(S_fixed[p]) + b_s，W_s: Nd→D。
v_p = W_v Flatten(V_fixed[p]) + b_v，W_v: (sum M_b)d→D。
z_p = W_f [v_p;s_p] + b_f + W_present a，W_f:2D→D。
a[N]为当前可见波段的显式标记。D=128。该标记用于区分缺失和数值为零，不是物理调制。

这个固定布局限制本轮模型：它不是任意 N、任意 GSD 的即插即用通用模型；新 Layout改变投影输入尺寸，必须另注册版本。

## 6. 一层跨地理 Patch Encoder

给可见 z_p 添加固定二维地理位置编码，按父观测分别批量执行一层标准 Block，D=128、4头、MLP4。输出最终 LayerNorm 后的 [B,P_visible,128]。

这是唯一跨地理 Patch的 Encoder Attention。分支融合投影本身不是额外 Transformer。没有层次下采样、金字塔、细粒度全局序列或旧槽位机制。

公共输出 GeoEncoding：values、ids_keep、valid、patch_grid、layout_hash、band_visible。禁止原始像元旁路或两个分支内部缓存直通 Decoder。

## 7. A：真正的地理 Patch 级 MAE

选择75%合法地理 Patch作为隐藏区域，所有波段同步隐藏。先生成mask，再进行可见内容读取。取整规则为 kept=max(1,floor(P_valid*0.25))，同时至少留一个隐藏有效Patch。对固定16块就是4可见、12隐藏。

A两分支只处理可见Patch；后端Encoder长度P_visible。廉价存储中的 padding不是隐藏Token计算的替代。记录每模块实际长度与执行pair数量。

Decoder先投影可见GeoToken到64维，按ids_keep scatter回P个地理位置，其他训练隐藏位置填同一个可学习mask_token[1,1,64]。随后对完整合法集合添加固定地理位置编码。QA全空/padding可用独立无效槽位，但不充当可评分mask。

官方MAE使用同类“只编码可见、解码时补齐”机制[S2,S3]；本模型的双路径前端和多原生布局是自定义扩展。

## 8. B1：波段缺失重建，不伪称整个地理 Token 被删除

每个视图隐藏一个合法波段，其全部地理位置数值均不可见。其他波段仍产生地理Token，所以 Encoder 与 Decoder 的地理长度均为P。

B1无须为整个地理位置补mask_token。缺失信息通过两分支固定零槽位及 a[N]明确标记。Decoder在同一个共享头上输出所有波段，loss只落在隐藏波段。不存在“B1也删除75%地理Token”的捷径。

每个父观测隐藏band按确定性均衡排程轮换，身份不重编号。不能把B1的计算节省假设为与A相同。

## 9. 共享轻量 Decoder

Encoder→Decoder Linear(128,64)。两层Pre-LN Block，64维、4头、MLP4；最终LN；共享Linear(64,R)，R为整个地理Patch全部波段的原生像元总数。

在A与B1均可把 band_visible 的确定性可见性投影加到所有有效地理Token，避免目标缺失布局歧义。A全为可用band，B1明确缺失band。不是每band一个mask_token，也不是每band新增一套Decoder。

输出为[B,P,R]。按冻结offsets拆分为各band的[h_b,w_b]，再按地理Patch行列回填原生目标网格。每个原生目标只计一次。

本轮不使用PixelShuffle-2头作为整个Decoder输出，因为一个Decoder Token现在覆盖整个地理Patch，不是一个Sub-patch。也不使用坐标MLP、面积求积、目标Query读取或全Sub-patch序列恢复。

Decoder仅服务预训练/重建验证，下游可移除。Encoder推理无训练mask时输出[B,P,128]。

## 10. 必须承认的容量约束

展平投影是有损压缩候选，不保证细节保留。固定Layout空间压缩矩阵可能参数较多：例中198*128*128=3,244,032个权重；不把少Token误称为参数必然少。

Linear(64,R)意味着每个预测Patch向量落在不超过64维的同一仿射输出子空间（考虑最终LN，实际可用集合还可能更受限）。这不能精确表示所有任意R维纹理，但可能足以学习任务相关低维结构；需验证，不能声称无损。若固定样本/Decoder上限诊断显示这个瓶颈，单独登记后才调整D或decoder_width；不偷偷恢复数千输出Token。

压缩前Attention有额外成本，也可能在当前数据上无收益。主线先验证一个版本；只有需回答这一机制时，再保留FFN及同投影、仅关闭Attention残差作对照，不把删除整个Block的收益都归给Attention。

## 11. 成本边界

Block近似 C(L,d)=12Ld²+2L²d MAC，MLP倍率4。实际空间分支仍为P*sum_b C(M_b,d)，谱分支为P*C(N_v,d)。压缩后跨Patch只为C(P_visible,D)，Decoder为2C(P,64)。固定压缩矩阵在B1仍计算全部固定槽位。

本包04按P=16、N=10、4个12×12与6个6×6波段核算，包含独立QA投影、各压缩、Encoder→Decoder、输出Linear；不包括norm/激活/softmax/反向/IO/通信。数值不代表实际GPU速度，也不证明同质量优于经典ViT。

新序列长度短，速度瓶颈主要可能在大量局部空间Block及数据/小算子开销。必须真批量化，不能先把模型做成逐Patch循环再用多进程补救。

## 12. 决策与来源

用户已明确：无Stem、描述器关闭、两分支先交互再各自合并、每地理Patch一个Token、后续一层ViT、S2 7:3、batch16。
工程候选（为本轮冻结的试验值）：120m、2×2、4×4窗口、128/64宽度、Decoder2层、75%A、每B1一个band；真实band清单及父观测身份必须本地绑定。

报告依据：用户附件《PD-HERE_V1_架构审阅与性能诊断报告》§4要求同成员/QA/目标/归约和最小因素对照；§6区分工程、基本拟合、重建和表征验收。它不提供本R3结构的实验证据。
官方来源：见03_SOURCES.md。参考代码通过的只是05所列局部张量检查；尚未执行生产loader、几何、训练、DDP、GPU速度或泛化验收。
