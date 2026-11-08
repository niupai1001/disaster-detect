# interface_proposal.md — EncoderV3-LR v1.2 接口提案（含 2026-11-08 纠错）

状态：`spec_revision=1.2`，`architecture_id=latent_response_sensor_memory_v1`。
本文件最初是待审提案；2026-11-08 用户已按推荐确认 Q1–Q7，确认记录见
`interface_confirmation.json`。同日独立审查 F1–F7 的确定性纠错已并入本文件与代码；
新增或仍需补充选择的接口条目集中列在 `interface_revision.md`（IR1–IR7），
未确认前不写入正式配置，也不得据此宣称 E1 通过。

标注约定：**【事实】**为已核查的规范原文、仓库代码或数据文件；**【推断】**为由
事实推出但未直接验证的判断；**【建议】**为实现选择，待确认后才成为配置。

核查基线：分支 `dev/encoder-v3-sensor-memory-v1`，base HEAD
`7d1db632c40a8dd7c6264cb95d97ec03235b2b61`，工作区保留原有未提交修改
`earth-representation/scripts/enmap_pairs_valid.py`（与本架构无关，未改动）。
规格文件已冻结到 `encoder_v3/docs/`（SHA256：architecture `001ea601…`、
agent_plan `782c868b…`、handoff `f0cca2f9…`，完整值见
`reports/latent_response/phase1_evidence.json`）。

## 0. 核查结论摘要

| 项目 | 结论 | 依据 |
| --- | --- | --- |
| 新架构代码 | 本轮之前**不存在**（`git log --all -- encoder_v3/latent_response` 为空） | 【事实】git |
| 旧 Participation 路径 | 保留、可导入；本轮抽查 21 项旧 CPU 测试通过 | 【事实】测试命令见阶段报告 |
| 旧报告 PASS | 不作为新架构证据；旧 A-hidden 结论按 01 §12 仍需重新核实 | 【事实】01 §12 |
| 本轮已实现 | Registry、SensorMemory、ModuleQuery、DynamicKernelBank/masked conv、Sub-Patch/micro-cell 几何 | 【事实】代码与 37 项测试 |
| 尚未实现 | ChannelFusion、空间主干、Decoder、data/task adapter、preflight、train、E1 | 【事实】无对应文件 |
| 真实 `optimizer.step` | 0 次 | 【事实】本轮只跑 CPU 合成测试 |
| I1–I5 | 全部 `PENDING` | 【事实】本文件 |

三份文档的矛盾与缺口集中在 §7，不自行选择解释。

---

## I1 标准化与监督

### 当前规范依据
- 01 §10：首版光学 S2/Landsat 使用**同源已审 TRAIN-only 统计量**标准化原生标量；
  Loss 固定 `0.5·L_A + 0.5·L_B1`，任务内部按目标通道→观测→来源等权；不得使用旧
  2000 步 `curriculum`；禁止重复除以原始标准差平方。同时 §14 把“输入变换、输出域、
  监督域、旧 NMSE 衔接”列为待审。
- 02 D6：固定损失、TRAIN-only 统计、旧 `curriculum` 禁用；讨论中的“标准化域 MSE”
  未经确认不得冻结。
- 旧代码事实：`losses.py:band_loss` 为 `MSE(p,y)/σ²`；`streaming.py:128/135` 为
  `Σ(p−y)²/(N·σ²)`；`losses.hierarchical_loss` 为 band→observation→source 等权；
  `validation_metric.aggregate_intra_validation` 为 A/B1 各 0.5、B1 单隐藏/多隐藏
  各 0.5 且多隐藏按隐藏数分层。旧指标还报告 reflectance 单位的 mae/rmse/bias。

### 已核查事实
- 【事实】`data_expansion/statistics.json` 的 `scope` 为
  `TRAIN-only physical reflectance by instrument and band; no held-out fit`，
  43 个 `instrument:band` 键，字段仅 `mean/std/valid_pixels`；数据单位为
  `surface_reflectance` 且 `values_already_scaled=true`。
- 【事实】旧路径只用了 `std`，没有减均值；因此旧 NMSE 的分母是 `σ²`（即方差），
  而不是当前观测/批次的方差。
- 【事实】A/B1 的 target 来自 `tasks.py`，真值与输入共享底层缓存数组，隐藏必须
  在进入任何可学习运算前完成（`spatial_hide` / `_spectral_view`）。

### 尚未确定
1. Decoder 输出域：标准化域 `z`、原始 reflectance，还是两者之一。
2. 输入标准化是否减均值（`z=(x−μ)/σ`）还是只除以 `σ`。
3. 逐通道损失的分母、零支持通道的处置、训练期层次归约的确切形式。
4. 旧 NMSE 数值如何与“标准化域 MSE”衔接，以及 reflectance 指标如何还原。
5. B1 的目标通道统计量使用 TRAIN-only `μ/σ` 是否作为正式合同（不是从目标值现算）。

### 最小推荐方案【建议】
- **输入变换**：每个通道一次 `z=(x−μ_c)/σ_c`，`μ_c,σ_c` 来自冻结的 TRAIN-only
  `statistics.json`，按 `SensorKey + channel_id` 键取；缺失键 fail-closed，禁止用
  当前观测重估，禁止 VAL 拟合。变换只作用于 `valid` 元素；`visible=false` 的
  隐藏值随后续 mask 置零，不参与计算。
- **Decoder 输出域**：直接输出标准化域标量 `ẑ`（无 squash），unsupported 输出 NaN。
- **监督目标域**：`ŷ=(y−μ_c)/σ_c`，同一份冻结统计量。逐通道损失
  `L_c = mean((ẑ−ŷ)²)`，**不再除以 `σ²`**。
- **旧 NMSE 衔接（代数恒等）**：若 `ẑ=ẑ_y` 由同一 `μ,σ` 反变换，则
  `L_c ≡ MSE_y/σ²`，即旧 `band_loss`/`streaming` 的 NMSE 数值定义。两种实现
  （输出 z 做一次 MSE，或输出 y 除以 `σ²`）在数学上等价；推荐前者只为数值条件更好。
  **避免重复除以标准差的办法**：损失函数签名只接收已经标准化的预测与标签，不接收
  `std`；`std/μ` 只出现在数据适配器与指标还原函数中，单元测试断言两条路径等价。
- **层次归约**：训练沿用旧语义——先按目标通道算逐通道标量，再在视图内对有效目标
  通道等权平均，再按 observation→source 等权；A/B1 固定 0.5/0.5。验证沿用
  `validation_metric` 的 A/B1 两族与 B1 单/多隐藏分层。零支持目标不计入均值但
  必须记录 `NO_VALID_SUPPORTED_TARGET`，不得静默补零。
- **指标还原**：`y=y_z·σ+μ`；`rmse_phys=rmse_z·σ`、`mae_phys=mae_z·σ`、
  `bias_phys=bias_z·σ`、`NMSE=mean((ẑ−ŷ)²)`。若预测恒为 TRAIN 均值 `μ`，则
  `NMSE=1+((mean_target−μ)/σ)²`；对 VAL 或任意子集**不保证等于 1**，必须实际计算，
  不能把 1 当作通用验收阈值。新旧数据/面板不同，不能直接宣称可比。

### 必要备选与取舍
- 备选 A：Decoder 输出 reflectance，损失按旧 `MSE/σ²`。与推荐方案等价但输出动态
  范围受 `μ` 影响，且实现上更容易“又除一次 σ²”；只在需要直接输出物理量时采用。
- 备选 B：只除 `σ` 不减 `μ`（`z'=y/σ`）。损失仍等于旧 NMSE，但目标不以 0 为中心，
  对偏置项与初始化更敏感；不推荐。
- 备选 C：每通道在 batch/observation 内重标准化。**违反 TRAIN-only 冻结统计与
  B1 泄漏边界，排除**。

### 张量与 mask
- `ChannelObservation.values [H,W]`：已按本接口变换的标量；`valid/visible bool [H,W]`。
- 目标 `truth_z [Q]`、预测 `pred_z [Q]`、`supported bool [Q]`；loss 只在
  `target_valid & supported` 上求和，按该通道的 `N` 归一，再进入层次归约。
- 统计量 `μ,σ` 形状 `[C_catalog]` 的查表结果，不进入 autograd 图。

### 资源影响
统计量 43×2 个 float，可忽略；标准化与反变换为逐元素运算；不改变任何网络宽度、
Token 数或注意力规模。真正的成本变化来自 I2–I4，不来自 I1。

### 直接验证方法
1. 合成测试：固定 `μ,σ`，比较 `MSE((p−μ)/σ,(y−μ)/σ)` 与 `MSE(p,y)/σ²`，
   `atol=1e-6`；两者必须一致且只出现一次 `σ`。
2. 还原测试：`z→y` 往返误差 < 1e-6；`rmse_phys` 等于 `rmse_z·σ`。
3. 泄漏测试：只对隐藏输入副本投毒，保持真值/统计量/mask 不变，loss 与梯度一致
   （该测试属 D6；接口确认后随 E1 执行）。
4. 旧口径对齐：在 CPU 合成视图上分别走“旧 `band_loss`”与“新 z 域损失”，逐通道
   数值一致；该测试不读真实数据。

### 待确认（→ §6 Q1）
是否批准：标准化域输出与单次 MSE、`μ/σ` 均来自 `statistics.json`、指标按上述还原。

---

## I2 LocalEmbed 与未定小投影

### 当前规范依据
- 01 §7：原生特征按真实交叠面积形成 `U[j,c,16,32]` 与显式 `support_fraction`、
  `visible_count`；共享局部汇聚器得到 `h[j,c,128]`；“其具体位置敏感运算由 §14 的
  接口提案确定”；`F_c/U/h` 只作前端中间态，**不形成独立 Decoder skip**。
- 01 §8：`Q_sum/Q_latent` 从当前 Sub-Patch 的可见通道特征集合聚合出 1 个 summary 与
  `R(C)=min(16,ceil(sqrt(C)))` 个局部潜在 Token；`Q_latent` 按连续槽位坐标生成，
  不按波段数或卫星型号设参数；每 Sub-Patch 只有 1 个 summary 进入长距离主干。
- 01 §14 I2；02 D4/A2：4×4 micro-cell 为工程默认值，LocalEmbed 具体算子待确认。

### 已核查事实
- 【事实】旧代码没有 micro-cell 特征，也没有局部潜在 Token；`participation_*` 的
  Merge 组不能当作新 Token 支持（02 §B）。
- 【事实】本轮已实现 `native_patch.aggregate_microcells`：按真实交叠面积输出
  `U [P,16,32]`、`support [P,16]`（可见交叠面积/微格面积）、`count [P,16]`；
  无支持微格为 `U=0, support=0, count=0`，隐藏像元不贡献（测试已覆盖）。
- 【事实】`reference_gsd` 取全部原生网格 `min(gx,gy)`；S2 10m/20m→10m、
  Landsat 30m，对应 `L_sub=80m/240m`、微格 20m/60m（见 I5）。

### 尚未确定
1. `U`、`support_fraction`、`visible_count`、微格位置如何进入 `h[j,c,128]`。
2. 位置编码是否需要 Fourier、频带数、归一化坐标；小投影的宽度与层数。
3. `h` 是否保留 4×4 空间排列（禁止用无位置信息平均替代）。
4. `Q_sum` 与连续 `Q_latent,r` 的具体生成器（与 I5 的 `C` 归一化耦合）。
5. `R(C)` 的坐标归一化用 `(r+0.5)/R` 还是别的连续坐标；槽位基数是否随 `R` 变。

### 最小推荐方案【建议】
- **保留 4×4 排列的 LocalEmbed（v0）**：
  `h[j,c] = Linear( [vec16(U_masked) (512), vec16(support) (16), vec16(Fourier2D(微格中心相对 Sub-Patch 中心 / L_sub)) (16×16)] ) → 128`。
  Fourier2D 按**每轴 4 个频率**、每轴 sin+cos：每微格 `(x: 4 sin + 4 cos, y: 4 sin + 4 cos)=16` 维，
  16 微格共 256 维；输入总宽 **512+16+256=784**（修正原 656 的算术错误），
  单层线性 + bias，无激活、无注意力，约 100k 参数，全载荷共享。
  `U_masked = where(support>0, U, 0)` 在投影前**硬屏蔽**，不依赖网络自行忽略无支持值。
  微格坐标固定（`(col+0.5)/4,(row+0.5)/4`），因此在单层 flatten+Linear 中位置项等价于
  固定偏置，位置敏感性来自各槽位的不同权重；不宣称额外位置机制已被证明有效。
  16 个微格按固定顺序展平，**不是** permutation-invariant 平均；`support` 显式告诉
  网络哪些槽位是真实观测。`visible_count` 只进审计/representation，不重复进入输入。
- **`Q_sum`**：一个全载荷共享的 learned `[128]` 向量（不依赖 C、传感器、波段）。
- **`Q_latent,r`**：`Linear(Fourier((r+0.5)/R(C))), B_lat=3 → 128`，共享参数，
  `r=0..R−1`；`C=0` 时不生成内容 Token（只保留几何占位，不建可学习空白 Token）。
- **`R(C)`**：严格 `min(16,ceil(sqrt(C)))`，`C` 为该 Sub-Patch 内有有效可见支持的
  输入通道数（不是目录通道数，也不是隐藏目标数）。
- **`ModuleQuery` 的频带是独立参数**：`F_meta`（目录顺位 Fourier）与上述
  `B_local/B_lat/B_pos` 不是同一个数；`F_meta` 与小投影宽度尚未写进确认记录，见
  `interface_revision.md` IR5。

### 必要备选与取舍
- 备选 A：16 个微格 Token → 1 层 4 头 self-attention → 可学习 query 池化。表达力更强，
  但每通道多约 3×128² 参数与 16² 注意力，且与 I3 的细尺度注意力职责重叠；v0 不取。
- 备选 B：`U` 的 16 槽直接平均（无位置）。**违反“不能默认用无位置信息的平均替代
  局部空间编码”，排除**。
- 备选 C：可学习 16 位置 embedding 代替 Fourier。参数少，但位置语义绑定 `4×4`
  与固定 `L_sub` 比例；若未来改微格数需要换权重。Fourier 更稳，推荐 Fourier。

### 张量与 mask
- 输入 `U [C_j,16,32]`、`support [C_j,16]`、`count [C_j,16]`、微格中心 `[16,2]`。
- 输出 `h [C_j,128]`，`C_j>0` 才存在；全隐藏通道不产生 `h`、不进入融合 K/V。
- 微格无支持时 `U=0, support=0`，由 `support` 通道区分“零值观测”与“无观测”；
  不使用 NaN 占位进入可学习运算。

### 资源影响【推断】
S2 385² 十波段（样例，非最大）：`N_sub≈49²=2401`，每 Sub-Patch 每通道 784×128≈100k MAC；
10 通道约 2.4 GMAC 的前端投影（不含卷积与融合），与原生卷积 3×3×32 同量级。
256/512 通道合成场景该投影按通道线性增长，不新增每波段参数。
真实最大缓存（S2B 2769×2264）约 41 倍于 385²，必须按实际 `N_sub` 与分块重算，不能把
385² 的数当成上界。

### 直接验证方法
1. 排列敏感性：固定 `U/support`，置换 16 个微格后 `h` 必须改变；置换回去恢复。
2. 掩码正确性：某微格 `support=0` 时其 `U` 不进入 `h`（把该槽 `U` 改成大值，
   `h` 不变）；隐藏像元的 NaN 不影响 `h`。
3. `R(C)` 表：`C=0/1/2/4/10/224 → 0/1/2/2/4/15`；`Q_latent` 对不同 `r` 可区分，
   且同一 `r,R` 在不同通道/传感器上完全相同。
4. 形状/梯度：`h` 对投影参数与 `U` 的梯度有限；`F_c/U/h` 不进入 Decoder 输入
   （结构测试：Decoder 接口不接收这些张量）。

### 待确认（→ §6 Q2）
Fourier 频带数（LocalEmbed `B=4`、latent `B_lat=3`）、LocalEmbed 采用“展平线性投影”
而非 micro-cell 注意力、`Q_sum` 为单一 learned 向量。

---

## I3 空间主干与整景等价执行

### 当前规范依据
- 01 §9：`[N_sub,128]` → 8×8 局部窗口 Attention（2 层、4 头）→ 4×4 空间支持聚合 →
  32 个全局 latent Query Cross-Attention → 全局 latent Self-Attention（2 层、4 头）→
  全局上下文回注细尺度；局部谱 Token 只与所属/近邻 Sub-Patch 作 1 层交互；
  不得在整景 summary 上退化为完整 `O(N_sub²)`。
- 02 A2/D5：预算固定；位置表达、粗聚合、空窗口、回注、跨块归一化待 I3 确认；
  不得用逐块独立 Softmax 平均冒充完整注意力。

### 已核查事实
- 【事实】新路径没有主干实现；旧 `model.Encoder.global_net` 是全长全局 Transformer，
  不满足新预算，不能直接复用。
- 【事实】旧路径的 `layout.native_layout` 提供确定性中心归属与地理锚点，可作为
  Sub-Patch 索引的参考实现；但它没有 micro-cell、没有窗口/粗尺度层次。

### 尚未确定
1. 位置编码：绝对米制、相对 Sub-Patch、还是可学习；频率/归一化。
2. 8×8 窗口边界如何定义（全局索引空间还是逐块）；边缘窗口是否补 Token。
3. 4×4 粗聚合算子（均值/面积加权/学习）；空粗格处置。
4. 32 个全局 latent 的初始化、Cross-Attention 的 mask、回注算子（Cross-Attn/相加/门控）。
5. 分块执行的最小可交换单元、Halo、两遍汇总与激活重计算；前后向等价容差。
6. 局部谱 Token 的 1 层交互范围（同 Sub-Patch + 8 邻域？）。

### 最小推荐方案【建议】
- **位置表达（修订提案，IR6，待确认）**：以**不可变 `full_geometry`** 的场景原点
  `o=(x0,y0)` 与全景尺度 `(W,H)` 为所有块共用的公共参考：`d=(x−x0,y−y0)`；正弦分量
  用 `sin/cos(2π·d/2^k), k=0..11`（场景相对坐标，避免 UTM 大数相位），再拼接
  **非周期**的全景线性坐标 `(d_x/W, d_y/H)`，`Linear(50,128)` 投影后加到 summary。
  原“纯 1…2048m 正弦、绝对 UTM”的编码有公共周期 2048m（`PE(x+2048)=PE(x)`），
  在 `L_sub=80m` 栅格上每 128 个 Sub-Patch 重复，故不能称为全景唯一绝对位置。
  新增全景线性坐标使全景内位置可区分；参考系由所有块共享，不破坏分块等价。
  这是对已确认 I3 公式的修订，确认前不冻结、不写入正式配置。
- **局部窗口**：在全局 Sub-Patch 索引空间以 `floor(index/8)` 划分 8×8 窗口，
  原点固定；不足 8×8 的边缘窗口用 key-padding mask 补齐（不产生虚假内容），
  窗口间不通信；2 层 pre-LN、4 头、残差。
- **粗聚合**：非重叠 4×4 均值，只对 `valid` 细 Token 求均值并记 count；粗 Token
  位置取有效细 Token 位置的均值；全空粗格 masked，不生成零 Token。
- **全局**：32 个 learned latent 对全部粗 Token 做 masked Cross-Attention（1 层），
  再做 2 层 latent self-attention；粗 Token 数约 `N_sub/16`，注意力规模
  `32×N_coarse`，不是 `N_sub²`。
- **回注**：细 Token 作为 Query、32 个全局 latent 作为 K/V 做 1 层 Cross-Attention，
  残差 + LayerNorm 加回细尺度。
- **空窗口**：全空窗口整体跳过（不进入注意力、不产生输出）；整景无任何有效 Token 时
  返回空 representation 并单列报告，不伪造观测、不记零损失。
- **分块**：chunk = 整数个 8 行窗口带；局部层带 1 个窗口行 Halo；全局层需要全部粗
  Token，采用两遍：第一遍算细 summary+粗 Token（可 checkpoint），第二遍全局+回注
  （必要时重算细 summary）。所有 softmax 在完整窗口/完整粗集合上做，禁止“逐块
  softmax 后平均”。
- **局部谱交互**：每个 Sub-Patch 的 `R(C)` 个 latent Token 作为 Query，对
  “本 Sub-Patch + 8 邻域”的细 summary 做 1 层 masked Cross-Attention；
  不展成全景注意力。

### 必要备选与取舍
- 备选 A：可学习相对位置偏置（Swin 式）。表达力强，但引入窗口/尺度相关参数，
  且分块等价需要额外对齐；v0 不取。
- 备选 B：粗尺度用 stride-4 卷积/池化学习聚合。参数更多，且“空间支持聚合”语义
  不如显式均值+count 可审计；v0 不取。
- 备选 C：单遍流式 + 近似全局（如 EMA/线性注意力）。会改变模型计算，需另立规格；
  v0 不用。

### 张量与 mask
- 细 `Z [N_sub,128]`、`valid [N_sub]`、`pos [N_sub,2]`（米制）。
- 窗口 padding mask `[n_win, 8,8]`；粗 `Zc [N_c,128]`、`valid_c [N_c]`、
  `count_c [N_c]`；全局 `G [32,128]`。
- 分块接口额外带 `window_row_range` 与 Halo 行，输出拼回全局索引。

### 资源影响【推断】
S2 385²（样例）：`N_sub=49×49=2401`，窗口 `7×7=49`（最后一个不满），局部注意力约
`49×64²×2 层≈40 万` pair/层；**粗格 `ceil(49/4)²=13²=169`**（修正原 576 的错误），
全局 `32×169`；对照全长全局 `N_sub²≈5.8M` pair。
真实最大 S2B 2769×2264@10m：`N_sub≈347×283=98,201`，窗口 `44×36=1,584`，
粗格 `87×71=6,177`，全局 `32×6,177`；全长 `N_sub²≈9.6×10^9` pair（不可行），
必须流式/两遍执行。Landsat 128²@30m：`L_sub=240m` → `16×16=256`，窗口 `2×2`，
粗格 `4×4=16`。以上为公式估算，实测在 E1 记录。

### 直接验证方法
1. 分块等价：同一输入，`full` vs `chunk=2/4 窗口行` 的细/粗/全局输出在
   `atol=1e-5,rtol=1e-4` 内一致；记录最大差与相对范数。
2. 窗口边界：分块边界处的 Token 与整景对应 Token 完全一致；边缘 padding 不改变结果。
3. 空窗口：人为挖空一个窗口，输出形状不变、该窗口无 Token、全局 latent 有限。
4. 远距离传播：只改变远处一个 Sub-Patch 的输入，全局 latent 与另一端的细 Token
   必须变化（证明全局路径非局部）。
5. 无全景矩阵：记录每视图最大注意力 pair 数，断言不含 `N_sub×N_sub`。

### 待确认（→ §6 Q3）
绝对米制位置编码（`K=12`）、4×4 均值粗聚合、32 latent + Cross-Attn 回注、
按窗口行分块 + 两遍全局的执行方案。

---

## I4 Decoder 空间读出

### 当前规范依据
- 01 §10/§10.1：A 完全隐藏区域由 Decoder 从可见 summary 与全局 context 读出；
  目标相对位置、支持尺寸、目标通道条件必须**实际进入预测计算**；`Q_decoder(s,c)`
  只由静态目录/GSD 生成、不读目标像元值；逐像元几何进入空间读出路径，两者不得混淆；
  没有可见 Token 或没有直接面积重叠都不能把合法目标移出 A 损失。
- 01 §14 I4；02 D6：Decoder 以融合 summary 为主要表示，结合融合后局部 latent、
  全局 context 与 `z_decoder`；无独立高维 skip；不得新增原始标量旁路。

### 已核查事实
- 【事实】旧 `ReflectanceDecoder` 有两种读出：B1 用原生 cell 交叠面积加权；
  A 用“掩码区域边界邻接 + `1/(1+|rel|²)`”权重；条件为 SRF32；`supported` 由
  `weights.sum>0` 决定，无支持输出 NaN。
- 【事实】旧 A 规则用 **Token 中心**是否在隐藏矩形外（`legacy_center_outside`）作为
  邻接筛选，并要求 Token 代表区域与掩码边界接触；它与 B1 的原生 cell 面积重叠规则
  是两套不同语义，不能混用。该筛选会把完全隐藏 Sub-Patch 的目标判为无支持；新合同
  要求这些目标仍可预测（01 §10.1），因此旧 A 读出不能直接复用。
- 【事实】旧 decoder 已有逐像素 chunked 反传（`streaming.stream_task_roots`），
  可作为新 Decoder 的执行骨架；其 `supported=False` 的 NaN 约定可沿用。

### 尚未确定
1. 检索方式（kNN / 固定邻域 / 全 summary 注意力）与 k。
2. 聚合算子（注意力 / 距离加权 / MLP）。
3. 目标相对坐标参考系与尺度归一化（Sub-Patch 中心？目标自身？全局？）。
4. 支持尺寸的表达（原生像元面积、目标 cell 面积、log 与否）。
5. 目标通道条件的接入（`z_decoder` 与几何如何拼接、是否门控）。
6. 同一通道同一 Sub-Patch 内不同目标像元如何获得可区分输入。
7. 无局部支持时的全局回退与“整观测无可见上下文”的边界报告。

### 最小推荐方案【建议】
- **参考系**：以目标所属 Sub-Patch 中心为原点，所有位置除以 `L_sub`：
  目标 `t=(x_t−x_j)/L_sub`（∈[−0.5,0.5]），候选 summary 位置
  `p_k=(x_k−x_j)/L_sub`（邻域可到 ±2）。参考系只依赖几何与 Sub-Patch 索引，
  分块/batch/通道顺序不变。
- **检索**：在可见细 summary 集合上取距目标中心最近的 `k=8` 个（欧氏距离，
  以 `L_sub` 为单位；同 Sub-Patch 若有 summary 必取）。附 `mask`；kNN 只做索引，
  不把坐标当内容。
- **Query**：`q_geo = [Fourier2D(t, B_pos=4) (16), log(gx/L_sub), log(gy/L_sub),
  log(area/L_sub²)]` 共 **11** 维；`q = MLP([z_decoder(32), q_geo(11)]) → 128`。
  `z_decoder` 由静态 `Q_decoder(s,c)` 读同一 KV 得到，只吃 `ChannelMeta + log(GSD)`，不读像元。
- **聚合（含 R(C) latent 的明确读取，IR1，待确认）**：1 层 4 头 masked Cross-Attention，
  Q=`q`，K/V = 局部 summary（加相对位置 Fourier `p_k−t`）与**融合后 `R(C)` latent**
  （加类型嵌入）；输出 `a_local` 128。再以同一 `q` 对 32 个全局 latent 做 1 层
  masked Cross-Attention 得 `a_global` 128。`R(C)` 读取方式：取目标 kNN 命中的
  Sub-Patch 的融合后 latent Token（每子块 ≤`R(C)` 个，k≤8 → ≤128 个 key），
  与 summary 一起进入同一局部注意力；没有可用 latent 时置 `a_latent=0, has_latent=0`。
  这样局部谱分支实际进入 loss，不是独立高维 skip，也不删除该分支。最终拼接
  `[a_local(128), a_global(128), a_latent(128), q_geo(11), z_decoder(32),
  has_local(1), has_latent(1)]` 共 **429** 维 → `Linear(429,128) → GELU →
  Linear(128,1)` 输出 `ẑ`。`has_local=0` 时 `a_local=0`（不是 NaN），全局路径仍可用。
- **可区分性**：`t` 与支持尺寸既进入 `q`（改变注意力权重），也进入最终 MLP；
  同一通道、同一 Sub-Patch 的两个目标像元因此有不同的 query、不同的检索权重和
  不同的 MLP 输入。测试要求 `∂ẑ/∂t` 非零（除初始化对称点外），不要求随机初始
  化下任意两像元数值必不同。
- **支持尺寸**：目标原生 cell 面积 / `L_sub²`、原生 `gx,gy` 的对数；目标 cell 由
  原生仿射与目标索引精确给出，不从预测值反推。
- **NaN 规则**：仅当局部与全局都没有可用上下文（整观测无可见 summary）时输出 NaN，
  并单列报告；有全局 context 就必须给出有限预测并计入 A 损失。
- **无独立 skip**：Decoder 输入只有融合后 summary、`R(C)` 局部 latent、全局 latent、
  `z_decoder`、目标几何；不接收 `F_c/U/h`、micro-cell 特征或原始标量。

### 必要备选与取舍
- 备选 A：固定 3×3 邻域面积加权 + MLP（旧 B1 风格）。便宜，但在目标周围没有可见
  支持时退化；仍需全局回退。v0 不用作唯一路径，可作为 kNN 的退化对照。
- 备选 B：所有目标像元只对全局 latent 做 Cross-Attention。实现最简单、天然覆盖 A
  隐藏区，但丢失局部细节与“近邻支持”语义；与“以 fine summary 为主要表示”冲突，
  不推荐。
- 备选 C：逐像元位置编码进入 key（把位置当特征）。会让同一 summary 对不同目标
  呈现不同 K/V，等价于把几何混入内容检索；与“逐像元几何进入空间读出、不改变
  载荷 Query”不冲突，但增加 K/V 计算。v0 用 query 侧几何 + 相对位置偏置即可。

### 张量与 mask
- 目标：`cells [Q,4]`、`target_meta`、`target_gx/gy`、`subpatch_index [Q,2]`、
  `z_decoder [Q,32]`（或 `[32]` 广播）、`target_valid [Q]`。
- 上下文：`summary [N,128]`、`summary_pos [N,2]`、`summary_valid [N]`、
  `global [32,128]`、`global_valid [32]`（可选）。
- 输出：`ẑ [Q]`、`supported [Q]`；`supported` 只表示“有可用上下文”，与目标有效性
  `target_valid` 分开记录（01 §10.1 三类语义）。

### 资源影响【推断】
`k=8` 只是**选中邻居数，不是搜索成本**。实现必须给出空间索引（按 Sub-Patch 索引的
桶 + 环形扩展，或每景一次 `cKDTree`），并记录实际扫描距离次数；正常密度约扫描
1–4 个子块，A 完全隐藏区附近可能更远，需设最大环半径并在超界时走全局回退，
避免退化为 O(N) 全量扫描。
注意力：局部 K/V ≤ `8 summary + 8×R(C)` 个 latent（真实 10 通道 `R≤4` → ≤40 key；
224 通道合成 `R=16` → ≤136 key）；全局 32 key；最终 MLP 429→128→1 约 55k MAC/像元。
S2 385² 单波段约 148k 目标像元，按上述约 1×10^10 MAC 量级，必须沿用逐像素 chunk
反传（`chunk_pixels`）与索引；真实最大 2769×2264 单波段约 6.3M 目标像元，需在 E1
实测分块耗时/显存，不能把 385² 当上界。

### 直接验证方法
1. 完全隐藏 Sub-Patch：A 掩码覆盖整块，目标区无 Encoder 内容 Token，Decoder 仍输出
   有限 `ẑ`、`supported=True` 并计入损失（不能因无重叠变 NaN）。
2. 几何敏感性：同一通道、同一 Sub-Patch 内两个不同目标中心 → `q` 不同；
   `∂ẑ/∂t` 非零；固定其它输入时改变目标支持尺寸 → 输出改变。
3. 通道身份：B04/B08 在同一 P32、同一位置下 `Q_decoder` 与 `z_decoder` 不同，
   读出条件不同（B1 可识别性）。
4. 无泄漏：只投毒隐藏输入副本（固定真值/统计/mask），输出与同损失梯度按
   `atol=1e-5,rtol=1e-4` 一致；Decoder 不接收任何 `F/U/h` 或原始标量旁路。
5. 分块等价：分块目标读出与整幅读出逐目标一致；`streamed vs reference` 记录最大差。

### 待确认（→ §6 Q4）
kNN `k=8` + 局部/全局两路 Cross-Attention + 融合 MLP 的读出方案；
目标相对坐标以 Sub-Patch 中心归一化；`has_local` 标志与 NaN 规则。

---

## I5 数据身份与范围

### 当前规范依据
- 01 §4.1：`SensorKey=(platform,instrument)`，示例 `S2A/MSI`、`S2B/MSI`、
  `L8/OLI`、`L9/OLI2`；每载荷一个 P32；ChannelCatalog 存 `ordinal/total_channels`。
- 01 §7：`g_ref=min_c(min(gx_c,gy_c))` 用“完整原生通道目录的几何”，与 B1 可见性无关。
- 01 §14 I5；02 §C/§D1/§D4：`full_geometry` 是“完整输入场景的非数值网格元数据”，
  场景范围、“完整 observation/cache”与“原始卫星产品整幅”必须核实，不能默认为同一范围。

### 已核查事实
以 `data_expansion/release.json`、`members.jsonl`（7778 行）、`statistics.json`、
`G1/observation_contracts.jsonl`、成员 `.npz` 实测：

| 事实 | 证据 |
| --- | --- |
| 载荷身份字段为 `sensor`∈{S2,L8,L9}、`instrument`∈{S2A,S2B,S2C,LC08,LC09}，共 5 个组合 | `members.jsonl` |
| 数据中没有 `MSI/OLI/OLI2` 字段值 | 同上 |
| S2 通道并集 = B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12（10 个）；无 B01/B09/B10 | 4/10 波段成员与 statistics 43 键 |
| LC08 并集 = SR_B1…SR_B7（7 个）；LC09 = SR_B2…SR_B7（6 个） | 同上 |
| 4 波段 S2 缓存只有 4 个通道数组；10 波段只有 10 个；L8 7 波段只有 7 个 | `.npz` keys 实测 |
| `observation_contracts.jsonl` 只列出已加载通道的 grid，没有未加载通道几何 | 首条 contract 实测 |
| 统计量 43 键，字段 `mean/std/valid_pixels`，scope 明确 TRAIN-only | `statistics.json` |
| 形状跨度：Landsat 128²@30m；S2 4 波段 128²@10m；S2 10 波段从 193²/385² 到 **S2B 2769×2264@10m（+1385×1133@20m）**、S2A 2016×2011、S2C 412×412；最大单波段 `H×W=6,269,016` | 逐成员重读 `members.jsonl` |
| 20m 网格可与 10m 原点相差 10m（如 y 起点 6184550 vs 6184560） | 10 波段成员 |
| TRAIN 10 波段 S2 594 个，按旧谓词 `max_band(H×W)>16384` 排除 586 个；VAL 10 波段 98 个（97 个满足）；全 release 683 个成员满足（TRAIN 586） | `data_pool_reconciliation.json` |
| 对全部成员，`max_side>128` 与 `max_band(H×W)>16384` 计数相同（683/586）；旧配方谓词仍是面积条件，报告不得把边长当通用等价式 | 同上 |
| bbox 从约 1.28km（128²@10m）到约 27.7km×22.6km（2769×2264@10m）；原始 L2A 产品为 10980² 像元量级 | 成员 bbox/`product`/`source_asset` |

【事实】因此：release 的 “scene” 是**缓存窗口**，不是原始产品整幅；缓存与合同都
不保存未加载通道的几何。

【纠正】2026-11-08 审查 F3：先前版本把 385² 写成“最大缓存”，与本表同批证据
（S2A 2016×2011、S2B 2769×2264、S2C 412×412）冲突。本版已更正为实际最大值，
**不新增任何上限、不裁剪、不删除合格成员**；“整景=完整 observation/cache”的
已确认语义不变。逐成员对账与旧谓词计数见
`reports/latent_response/data_pool_reconciliation.json`。

### 尚未确定 / 冲突
1. **SensorKey 命名**：规范示例 `(S2A,MSI)` 与数据字段 `(S2,S2A)` 不一致。若直接把
   数据字段拼成 key，会得到 `("S2","S2A")`、`("L8","LC08")` 等 5 个键，语义上是
   “卫星族+卫星/仪器代号”，与规范示例的“卫星+仪器类型”不同。
2. **ChannelCatalog 范围**：release 并集 10 个 S2 通道，但 L2A 产品原生目录通常含
   B01/B09/B10（13 个）；数据层没有产品级完整目录证据。`ordinal/total_channels`
   取 10 还是 13、ordinal 从 0 还是 1 开始，都会改变 `Q_conv/Q_channel/Q_decoder`
   的输入，从而影响 checkpoint 身份。
3. **`full_geometry` 定义**：01 §7 用“完整原生通道目录的几何”，02 §C 用“完整输入
   场景的非数值通道网格元数据”；数据只提供观测记录中**已加载**通道的网格。二者
   不能默认为同一范围。
4. **`g_ref` 取值集合**：按观测记录几何为 {10m(S2), 30m(Landsat)} → `L_sub`∈{80,240}m，
   微格∈{20,60}m；若某 S2 视图只保留 20m 波段（数据中不存在，但 B1 理论上可隐藏
   全部 10m 波段），按“可见输入”算会变成 20m，锚点漂移——必须用与可见性无关的
   完整记录几何。
5. **场景范围**：E1 的“超过旧 16384 像元上限的完整十波段 scene”在本 release 对应
   完整**缓存**，样例 193²/385²，实际最大 2769×2264（单波段 6,269,016 像元）；
   原始产品整幅（约 10980²）没有本地加载路径。
6. **VAL 覆盖**：VAL 有 S2（335：4/10 波段）与 L8（48：7 波段），没有 L9；TRAIN 有
   L9 524 个 6 波段成员。E1 的“Landsat 六/七波段各≥1”可由 L8 满足，但 L9 无 VAL 面板。

### 最小推荐方案【建议】
- **SensorKey 用数据字段原值**：`SensorKey(platform=sensor, instrument=instrument)`，
  得到 5 个键 `("S2","S2A")/("S2","S2B")/("S2","S2C")/("L8","LC08")/("L9","LC09")`；
  不新增别名表、不猜测 `MSI/OLI`。若确认要恢复规范示例命名，必须另立“数据合同
  字段→SensorKey”映射并绑定到目录哈希。
- **ChannelCatalog（v0 建议）**：每个载荷取 release 中该载荷全部出现过的通道并集，
  按原生产品顺序排列（S2：B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12；LC08：
  SR_B1…B7；LC09：SR_B2…B7），0-based `canonical_ordinal`，`total_channels` 为
  并集大小（10/7/6）。明确记录：这是**release 目录**，不是 L2A 13 通道产品目录。
- **`full_geometry`（v0 建议）**：观测记录中**全部通道**（含 B1 目标通道、含 A 隐藏
  区域）的非数值网格元数据（CRS/affine/shape），由 `tasks.py` 的 targets 保留；
  它不随 `visible` 变化。`g_ref`、`L_sub`、地理锚点只由它决定；B1 删通道不得改变。
  它**不等于**原始产品整幅，也不包含未加载通道。
- **场景范围（v0 建议）**：本阶段与 E1 的“整景”= 完整 observation/cache，
  **没有 385² 上限**；已盘点最大为 S2B 2769×2264@10m（单波段 6,269,016 像元）。
  原始产品整幅训练/评测不在本轮，需要新的数据加载与授权。
  报告同时给出 cache 覆盖与产品 bbox 的差异，不能互相冒充。
- **注册集合**：v0 只注册真实出现的 5 个键；S2C 虽然规范示例未列，但数据存在，
  不能并入 S2A/S2B。

### 必要备选与取舍
- 备选 A（目录）：用 L2A 产品 13 通道目录（含 B01/B09/B10），ordinal 以产品顺序
  计。更接近“完整原生通道目录”，但当前数据没有 60m 通道几何与统计，需新增数据
  合同；且 `total_channels=13` 会改变所有 Query 输入。
- 备选 B（命名）：建立 `S2A→MSI`、`LC08→OLI` 别名表。可满足规范示例字面，但别名
  表本身是新的数据合同，必须版本化并进目录哈希。
- 备选 C（`full_geometry`）：只用当前 `visible` 通道几何。实现简单，但 B1 隐藏
  10m 波段时 `g_ref` 会漂移，**违反“B1 隐藏不得改变参考尺度与锚点”，排除**。
- 备选 D（场景）：把原始产品整幅作为场景。需要新下载/窗口外读取，超出本轮授权与
  数据合同，不采用。

### 张量与 mask
- `FullGeometry`：`channel_id → NativeGrid(affine, shape, crs)`；只读、非数值。
- 目录哈希 `channel_catalog_sha256` 覆盖：SensorKey 排序、每键 total/ordinal、
  `full_geometry` 的通道集合（不含像元值）。
- `g_ref`、`L_sub` 为标量，进入执行计划与 checkpoint 身份，不进入 autograd。

### 资源影响【推断】
`g_ref` 决定 `N_sub`：385²@10m、`L_sub=80` → 49×49≈2401；128²@30m、`L_sub=240`
→ 16×16=256。若误用 20m 作 `g_ref`（只保留 20m 波段的假设视图），`N_sub` 会缩小
4 倍，空间覆盖与 Token 数全部改变；因此该选择是身份级别的，不是性能调优。

### 直接验证方法
1. 目录/身份：同一 S2 载荷 4/10 波段成员得到同一 `SensorKey` 与同一 P；L8/L9 各
   一个 P；`#P==#SensorKey==5`（fixture 测试已覆盖结构，真实目录待确认）。
2. `g_ref` 不变性：对同一观测构造 4→10 波段、以及 B1 隐藏 10m 通道的视图，
   `g_ref/L_sub/锚点` 完全相同；用 `tasks.py` 目标通道保留的网格验证。
3. 目录范围敏感性：10 通道 vs 13 通道目录产生不同 `channel_catalog_sha256` 与不同
   `Q` 输入（测试已用不同 total 证明身份会变）。
4. 覆盖报告：从 release 生成 TRAIN/VAL 成员数、波段数、>16384 成员数、每载荷通道
   并集、`g_ref` 集合；与 `release.json` 的 586 排除数对账（实测 586/594 TRAIN）。
5. 范围报告：同时输出 cache 形状/bbox 与产品 id/source_asset，明确二者差异，
   不用 cache 覆盖证明产品整幅能力。

### 待确认（→ §6 Q5–Q7）
SensorKey 用数据原值；目录取 release 并集 10/7/6 且 0-based；`full_geometry` 取
观测记录全部通道；E1 的“整景”接受完整 cache 范围。

---

## 6. 集中待确认问题（最少集）

每条都给出推荐项；Q1–Q7 已于 2026-11-08 按推荐确认（记录见
`interface_confirmation.json`）。审查后对 Q2 的 Fourier 维数算术、Q3 的位置周期、
Q4 的 R(C) 读出/拼接宽度做了纠正或补充；其中位置编码公式与 R(C) 的具体使用方式
属于待重新确认的修订项（IR1、IR6），见 `interface_revision.md`。

| 编号 | 问题 | 推荐 | 影响 |
| --- | --- | --- | --- |
| Q1 | I1：Decoder 输出标准化域 `z`、逐通道只做一次 MSE、`μ/σ` 取 `statistics.json`？ | 是 | 损失/指标/checkpoint 兼容；避免重复除 σ² |
| Q2 | I2：LocalEmbed 用“保留 4×4 排列的展平线性投影”（Fourier `B=4`），`Q_latent` 用 `B_lat=3` 连续坐标，`Q_sum` 为单一 learned 向量？ | 是 | `h[j,c,128]` 的参数与计算；不改 R(C) |
| Q3 | I3：绝对米制正弦位置（`K=12`）、4×4 均值粗聚合、32 latent+Cross-Attn 回注、按窗口行分块+两遍全局？ | 是 | 主干预算内的算子与分块等价方案 |
| Q4 | I4：kNN `k=8` + 局部/全局两路 Cross-Attention + 融合 MLP；以所属 Sub-Patch 中心归一化相对坐标？ | 是 | A 隐藏区读出、同通道不同像元可区分性 |
| Q5 | I5：SensorKey 直接使用数据字段 `(sensor,instrument)`（5 个键），不造 MSI/OLI 别名？ | 是 | 注册表/checkpoint 身份 |
| Q6 | I5：ChannelCatalog 取 release 并集 10/7/6，0-based ordinal，`total=并集大小`；不采用 13 通道产品目录？ | 是（v0） | Query 输入、目录哈希 |
| Q7 | I5：`full_geometry` = 观测记录全部已加载通道（含 B1 目标通道）的网格；E1“整景”接受完整 observation/cache（无 385² 上限；已盘点最大 2769×2264），不要求产品整幅？ | 是（v0） | g_ref/锚点稳定性、E1 覆盖范围声明 |

若 Q6 选 13 通道目录、Q5 选别名、Q7 选产品整幅，均需新增数据合同与目录版本，
不能在 Agent 侧静默切换。

---

## 7. 文档矛盾与缺口（不自行选择解释）

| 编号 | 条目 | 影响 | 处理 |
| --- | --- | --- | --- |
| C1 | 01 §4.1 / 02 A2 的 SensorKey 示例 `S2A/MSI`、`L8/OLI` 与数据字段 `sensor=S2, instrument=S2A`、`sensor=L8, instrument=LC08` 不一致 | 注册键、checkpoint 身份、报告中的载荷命名 | 列入 Q5；本轮代码用数据原值，未写死示例名 |
| C2 | 01 §7 的 `full_geometry` 为“完整原生通道目录的几何”；02 §C 为“完整输入场景的非数值通道网格元数据”；数据只有已加载通道网格 | `g_ref`、锚点、B1 不变性、E1 覆盖声明 | 列入 Q7；本轮实现为“观测记录全部通道”，未实现未加载目录几何 |
| C3 | 01 §10 已把“TRAIN-only 统计量标准化原生标量”写成监督协议，01 §14 I1 又把“输入变换”列为待审 | 适配器是否可以先标准化 | 不冲突但需在 Q1 一次确认输入+输出+损失域 |
| C4 | 01 §14 的“标准化域 MSE”是候选，03 明确“不能视为已批准” | 损失实现 | 本文件按候选处理，Q1 确认前不冻结 |
| C5 | 01 §7 保留融合后 `R(C)` 局部 latent，03/02 又强调“无独立高维 skip” | 易把 R(C) 误删或误当 skip | 本文件按“R(C) 保留、F/U/h 不进 Decoder”执行；测试已断言 Decoder 不接收前端张量（待 Decoder 实现后） |
| C6 | 02 §D4 说“所有符合数据合同的完整场景都可成为训练候选”，但 release 的 scene 是缓存窗口（样例 385²，实际最大 2769×2264@10m），不是产品整幅 | E1 大场景覆盖的可宣称范围 | 列入 Q7；报告区分 cache 与产品 bbox |
| C7 | 01 §12 指出旧 `formal_runner` 验证回写状态可能滞后；旧 README 记载已有修复/测试 | 新 runner 不能继承旧“通过”结论 | 新 runner 实现时单测“验证完成→checkpoint→resume”；本轮未改旧 runner |

## 8. 本提案不改变的内容

P32 粒度与 32 维、`M_mem=8`、`d_k=d_v=32`、动态卷积公式、`L_sub=8·g_ref`、
4×4 micro-cell、`R(C)=min(16,ceil(sqrt(C)))`、`D=128`、主干层数/头数、
A/B1 `0.5/0.5`、数据资格规则（取消旧 `max_native_pixels=16384` 过滤）均保持
01/02 原文；本文件只提出未定接口的最小实现，不改变上述冻结项。
