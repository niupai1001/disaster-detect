# interface_revision.md — 审查后的接口纠正与待确认项

日期：2026-11-08。关联：`REVIEW.md` / `AGENT_FIX_TASK.md`（针对提交 `d806e84`）。
本文件不重新讨论已确认主路线；只把审查指出的**笔误/遗漏**与**仍需补充确认的选择**
分开，并给出具体张量/计算。**待确认项不写入正式模型配置。**

## 0. 已直接纠正（无需新确认）

| 编号 | 位置 | 原问题 | 纠正后 |
| --- | --- | --- | --- |
| IR2 | I2 LocalEmbed | “每轴 B=4”却写成 16×8=128，输入 656 | 每微格 16 维（每轴 4 频 × sin/cos），16 微格=256；总宽 **512+16+256=784** |
| IR3 | I2 LocalEmbed | support=0 的 U 未硬屏蔽 | 投影前 `U_masked = where(support>0, U, 0)`，并由测试注入无支持槽大值验证 `h` 不变 |
| IR4 | I4 Decoder | 拼接宽度未写，`q_geo` 维数未写 | `q_geo=16+2+1=19`；拼接 `[a_local 128, a_global 128, a_latent 128, q_geo 19, z_decoder 32, has_local 1, has_latent 1]` = **437** → `Linear(437,128)→GELU→Linear(128,1)` |
| IR7a | I3 资源 | 49×49 的 4×4 粗格写成 576 | `ceil(49/4)²=13²=`**169**；大场景 347×283→87×71=**6,177** |
| IR7b | I4 资源 | kNN=8 被当作搜索成本 | k=8 只是选中数；必须给空间索引（Sub-Patch 桶+环形扩展或每景 KDTree），使用精确 KDTree 或等价精确 kNN，记录实际距离计算数、候选数和耗时；不采用半径截断近似 |
| F1 | native_patch | 按中心唯一归属 patch，漏算跨 patch 面积 | 已改为枚举全部正面积相交 micro-cell/Sub-Patch；新增面积守恒、梯度、跨 4 patch、非法矩形测试 |
| F2 | types.NativeGrid | CRS 只检查字符串非空 | 已解析 CRS，要求 projected + metre/meter；经纬度、英尺、非法 CRS 拒绝，不重投影 |
| F3 | I5/Q7、confirmation | 385² 被写成最大缓存 | 已更正：完整 cache 无 385² 上限；manifest 最大 S2B 2769×2264，见 `reports/latent_response/data_pool_reconciliation.json` |

命名校正：`B8A` 在 S2 原生顺序中位于 `B08` 与 `B11` 之间，不能按字符串排序；
`release_catalogs` 用显式顺序。

## 1. 已由用户确认的补充选择（本次回复：确认，按上述推荐继续）

| 编号 | 条目 | 具体候选（张量/计算） | 影响 | 状态 |
| --- | --- | --- | --- | --- |
| IR1 | **R(C) 局部 latent 如何进入 Decoder**：现 I4 公式没有使用点，照抄会让局部谱分支对 loss 无作用 | 目标 `q`（128）对局部 K/V 做 1 层 4 头 masked Cross-Attention；K/V = 命中 Sub-Patch 的 fine summary（加相对位置 Fourier）**与融合后 R(C) latent**（加类型嵌入）。每个命中子块 ≤R(C) 个，k≤8 → 最多 128 个 latent key；输出 `a_latent[128]`。无可用 latent 时 `a_latent=0, has_latent=0`。不恢复逐通道/micro-cell skip，不删除 R(C) | Decoder 是否真正训练局部谱分支；A/B1 loss 路径 | **CONFIRMED**（联合局部读出309维，见下） |
| IR5 | **ModuleQuery 自身频率与小投影**：确认记录只固定了 LocalEmbed/latent/target 的频带，没固定 `F_meta` | 目录顺位分数 `u=(ordinal+0.5)/C`；`F_meta=3`（1,2,4 周期）→ `[sin,cos]` 6 维 + `log gx,log gy` = **8** 维；`f_t: Linear(8,64)→GELU→Linear(64,32)`；`q_t^0` 为每模块共享 learned `[32]`。与 `B_local=4`、`B_lat=3`、`B_pos=4` 是不同参数 | 三个模块 Query 的输入宽度/参数；目录哈希不变 | **CONFIRMED**（共享投影） |
| IR6 | **位置编码公共周期**：1…2048m 正弦有公共周期 2048m，`PE(x+2048)=PE(x)`；`L_sub=80m` 时沿轴相隔 128 个 Sub-Patch（10240m）重复 | 以不可变 `full_geometry` 的场景原点 `(x0,y0)` 与全景 `(W,H)` 为公共参考：`d=(x−x0,y−y0)`；正弦用 `sin/cos(2π d/2^k), k=0..11`（场景相对，避免 UTM 大数相位），再拼非周期 `(d_x/W, d_y/H)` 两维；`Linear(50,128)`。所有块共用同一参考，分块等价 | 长程可区分性；位置项参数；分块一致性 | **CONFIRMED**（场景相对参考） |

> IR1 与 IR6 修改的是已确认 I4/I3 的计算细节；在用户确认前，相关实现停在依赖边界，
> 不写入配置、不宣称接口闭合。

## 2. 直接验证方法（确认后执行）

1. **IR1**：固定输入，把 `a_latent` 置 0 与正常路径比较，loss 必须出现差异；R(C) latent
   的梯度非零有限；实际检索有效 latent 集合为空时 `has_latent=0`（目标自身隐藏并不排除邻域 latent） 且仍从 summary/global
   得到有限预测。
2. **IR2/IR3**：构造 support=0 槽并注入大 U，`h` 必须逐位不变；Fourier 维数为
   16 维/微格，输入宽度 784；置换微格后 `h` 改变。
3. **IR4**：拼接张量宽度 437；首层 `Linear(437,128)`；改变 `q_geo` 任一分量，
   预测或梯度必须变化。
4. **IR6**：同一场景内 `PE(x)` 与 `PE(x+10240)` 不再相等（线性全景项区分）；
   `full` 与窗口行分块输出在 `atol=1e-5,rtol=1e-4` 内一致；不同场景各自参考系一致。
5. **IR7**：385² 粗格 169；2769×2264 粗格 6,177；记录每目标实际扫描的子块/距离数，
   而不是只报 k=8。

## 3. 不在本文件内决定的事项

- 目录 10/7/6、5 个载荷键、`full_geometry` 取遮盖前观测全部已加载通道：已确认，保持。
- 训练损失 0.5A+0.5B1、R(C) 公式、P32/KV/卷积/Sub-Patch 预算：不变。
- E2/E3 授权、旧检查点与历史预算：不变，仍不可执行。

> 执行纠错：q_geo = 16维目标位置 Fourier + 2维 log(gx/L_sub), log(gy/L_sub) + 1维 log(area/L_sub²)，共19维。三路读出的437维及 Linear(437,128)仅为IR1待确认候选算术，不是正式配置；最终读出路径见 plan_execution/interface_decisions.md。

## 当前实施合同

用户已确认 plan_execution/interface_decisions.md 的推荐组合。IR1 使用联合 summary+R(C) latent 局部注意力输出 a_local[128]，加全局 a_global[128]；最终 [128+128+19+32+1+1]=309，Linear(309,128)。437只保留为未采用的三路备选算术。has_latent 按检索邻域实际有效latent集合判定。IR5共享8→64→32投影加三个独立seed；IR6按全部已加载通道确定统一场景原点/宽高，不编码跨场景绝对位置。上文含“待确认/确认后”的历史说明由本段取代。
