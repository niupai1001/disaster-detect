# v4r4 执行报告：SegFormer 原网格细节的早期双向交换

日期：2026-10-20
状态：**两组正式训练已完成，接口门槛全部通过；按计划 §7 的停止条件，建议放弃该结构。**
本文区分三类结论：**已运行**（有产物可复现）、**未验证**（没跑，不能推断）、**存在阻碍**（想做但被什么挡住了）。

---

## 0. 一句话结论

交换块**确实工作了**（残差系数从 0 学起来，两方向都有梯度、都发生了更新），但它没有带来更好的模型：
按阈值无关的 AP，Cross 0.4430 与 LateDetail 0.4425 **打平**；按 0.5 阈值，Cross 只是把工作点推向更保守——精确率 +0.036、召回 −0.037，
对象级可评估命中从 901 降到 810（少 91 个），未匹配预测对象从 5.11 降到 4.07/瓦片。
Cross 相对原模型在 best 与 last 两个检查点上都是小幅负值（−0.0048 / −0.0036，方向一致、区间都跨 0），
因此"没有改进"这一条是稳的。

按计划预定的规则——不超过原模型、且对象级没有改善——**判定为不支持替换主模型**。

> **勘误（2026-10-20 追加）**：本报告初版曾写"容量对照 Self 明显更差（AP −0.026，区间不跨 0），说明在编码阶段加深一层本身就是有害的"。
> 这个结论**过强，已撤回**。补做的检查（§1.5.1）显示：同一检验在 last 检查点上给出方向相反、同样"不跨 0"的结果。
> 真实情况是 Self 的**早期峰值被压低、后期水平更高**，属于轨迹形状改变，不是容量天花板降低。
> 可复述的版本见 §1.5.1；"加容量有害"不属于本报告能支持的结论。

---

## 1. 已运行

### 1.1 架构（除 exchange 外与历史 LateDetail 逐项相同）

| 项 | 取值 |
|---|---|
| 主干 | 官方 MiT-B0（`nvidia/segformer-b0-finetuned-ade-512-512`），`direct10` 首层 10 通道适配 |
| 原网格分支 | `SpectralStem`，32 通道，stride 全 1，输出 `[B,32,128,128]` |
| 交换位置 | 编码阶段 2 输出之后、阶段 3 之前，**仅一次** |
| 阶段 2 特征 | `[B,64,16,16]`（运行时断言，见 §2.1） |
| 交换子模块 | `P=Conv2d(64,32,1)`、`H_E=Conv(d→d,3)+GELU+Conv(d→64,1)`、`H_D=Conv(d→d,3)+GELU+Conv(d→32,1)`、`α_E[64]`、`α_D[32]`（零初始化） |
| 末端 | 保持原样：`logits = 原 decoder 上采样 logits + 原细节残差头 ΔL`，不新增分割头、不加粗定位监督 |
| 新增参数 | **+23,840**（占总参数 0.635%） |

复用而**未重写**的官方组件：`SegformerOverlapPatchEmbeddings`、`SegformerLayer`、各阶段 `nn.LayerNorm`、
`SegformerDecodeHead`（含 `SegformerMLP`/`linear_fuse`/`batch_norm`）、`decode_head.classifier`（替换为单 logit）。
`E_new` 既替换交给解码器的阶段 2 特征，也作为阶段 3 patch embedding 的输入；阶段 1/3/4 输出原样传入。

### 1.2 接口门槛（计划 §5，七道全过）

| 门槛 | 关键读数 | 结果 |
|---|---|---|
| G1 形状 | input `[2,10,128,128]` → E `[2,64,16,16]` → D `[2,32,128,128]` → stage3 `[2,160,8,8]` → stage4 `[2,256,4,4]` | PASS |
| G1 传导 | 调 α_E 后 stage3 最大变化 **0.2736**（E_new 真的进了阶段 3） | PASS |
| G2 初始等价 | 与 LateDetail 最大像素差 **0.0**（Self / Cross 都是精确 0，不是"≤1e-6"） | PASS |
| G2 参数身份 | 公共参数哈希 `c9deeddce71f0766…`（三方一致）；新增参数哈希 `016ff85f27da7ebd…`（Self=Cross） | PASS |
| G3 分组 | 无重复、无遗漏；交换参数全部落在 `new` 组（6e-4），未被并入 encoder（6e-5） | PASS |
| G3 参数量 | Self = Cross = **3,777,442**（精确相等） | PASS |
| G4 梯度 | 20 步窗口内 `p/h_e/h_d/α_e/α_d` 均出现非零梯度（首步 `h_e` 为 0 属预期，被 α=0 乘掉） | PASS |
| G5 干预 | α_D=0 时 D 变化 **0.0**；α_D≠0 时 D 变化 ~0.9；缩放 D 使 Cross 的 stage3 变化 0.6178、Self 为 **0.0** | PASS |
| G6 拟合 | 固定 8 样本 100 步：loss 0.8006 → 0.3368(Self)/0.3389(Cross)；两方向 α 均离开零 | PASS |
| G7 工程 | checkpoint 往返输出差 **0.0**；batch 64 前反向无 OOM（峰值 5.5 GiB 分配）；227 个梯度张量全有限；全参数 FP32 | PASS |

初始 α 为 0 使两架构在起点与 LateDetail **逐像素相同**，因此后续差异只能来自训练过程，不来自初始化。

### 1.3 正式训练（严格同协议，单种子 1234）

| 项 | 取值 |
|---|---|
| 数据 | `manifests/v4r1/pool_100.jsonl`，TRAIN 2,223 / DEV 669（hiroshima 448 + thrissur 221） |
| 归一化 | `manifests/v4r1/stats_core25.json`（TRAIN-only 559 拟合） |
| 预算 | batch 64，3,600 次成功更新，**230,400 曝光**，两组完全相同 |
| 优化 | encoder 6e-5 / 其余 6e-4，warmup 5% + poly，wd 0.01，clip 1.0 |
| 损失/增强 | 0.5 BCE + 0.5 Dice；natural 采样 + 原 flip/rot90；无新辅助 loss |
| 精度 | FP32，`amp: false` |
| 选模 | 每 100 步全量诊断，按两 DEV 事件平均 IoU |
| 耗时 | Self 906.2 s / Cross 905.8 s（LateDetail 738.4 s） |
| 训练峰值显存 | Self 5,222 MiB / Cross 5,350 MiB（LateDetail 4,694 MiB） |

两组均 `complete: true`，`skipped_batches: 0`，曝光 103–104 次/样本（2,223 个全被覆盖）。

### 1.4 三架构对照

**DEV 两种口径**（`best` 在 36 个 checkpoint 里挑最大，会偏高；固定步数才可横比）：

| 架构 | best DEV | best@步 | @500 | @1200 | @2400 | @3600 | TRAIN IoU@3600 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LateDetail | **0.3081** | 500 | 0.3081 | 0.2796 | 0.2588 | 0.2430 | 0.4895 |
| Self（容量对照） | 0.2969 | 500 | 0.2969 | 0.2873 | 0.2689 | **0.2554** | 0.4891 |
| Cross（真交换） | 0.3033 | 500 | 0.3033 | 0.2828 | 0.2582 | 0.2393 | 0.4881 |

三条曲线形状一致：500 步见顶，之后 TRAIN 继续升、DEV 持续降。**架构差异没有改变过拟合结构**。

但固定步数表里有一个必须单独指出的现象：**Self 与其他两条的差异是随步数换向的**。

| 步数 | LateDetail | Self | Cross | Self − LateDetail | Cross − LateDetail |
|---:|---:|---:|---:|---:|---:|
| 500（三条的 best） | 0.3081 | 0.2969 | 0.3033 | **−0.0112** | −0.0048 |
| 1200 | 0.2796 | 0.2873 | 0.2828 | +0.0077 | +0.0032 |
| 2400 | 0.2588 | 0.2689 | 0.2582 | +0.0101 | −0.0006 |
| 3600 | 0.2430 | 0.2554 | 0.2393 | **+0.0124** | −0.0037 |
| 平台均值（≥600 步） | 0.2523 (sd 0.0152) | **0.2620 (sd 0.0107)** | 0.2476 (sd 0.0149) | +0.0097 | −0.0047 |

即：**Self 的早期峰值更低（−0.011），但下降更慢，600 步之后每个读数都更高（最多 +0.012），变异也更小。**
选模规则取 best，所以 Self 输；若按固定步数或平台水平，Self 反而更稳。这不是"容量有害"，是"峰值被削平了"。

Cross 则是另一回事：它相对 LateDetail 在 500 和 3600 上分别是 −0.0048 和 −0.0037，**方向一致、幅度都小**，属于稳定的"略差一点"。

**best 检查点明细**：

| 架构 | 事件均值 | hiroshima | thrissur | 合并 IoU | Dice | Precision | Recall | **AP** | 边界 F1 | 对象召回 | 未匹配预测对象/瓦片 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LateDetail | **0.3081** | 0.3114 | **0.3049** | **0.3095** | **0.4727** | 0.4680 | **0.4775** | 0.4425 | **0.4806** | 0.521 | 5.11 |
| Self | 0.2969 | 0.3035 | 0.2902 | 0.2998 | 0.4613 | 0.4560 | 0.4667 | 0.4166 | 0.4715 | **0.537** | 5.14 |
| Cross | 0.3034 | **0.3134** | 0.2933 | 0.3073 | 0.4702 | **0.5044** | 0.4403 | **0.4430** | 0.4781 | 0.482 | **4.07** |

两点必须一起看：

1. **AP 是阈值无关的**：Cross 0.4430 vs LateDetail 0.4425，差 +0.0005，实质打平；Self 0.4166 明显更低。这是三架构里唯一不受工作点影响的比较。
2. **0.5 阈值下 Cross 的"下降"是工作点右移**：精确率 +0.036、召回 −0.037，几乎等量反向；未匹配预测对象从 5.11 降到 4.07/瓦片。交换块把模型推得更保守，不是变得更准。

### 1.5 差异显著性（逐瓦片配对、按事件分层 bootstrap 5000 次）

| 比较 | Δ(DEV 均值) | 95% 区间 | 逐事件同向 | 结论 |
|---|---:|---|---:|---|
| Cross − Self（主比较） | **+0.0065** | [−0.0003, +0.0136] | 2/2 | 方向一致但**区间跨 0**，不能说显著 |
| Cross − LateDetail | −0.0048 | [−0.0099, +0.0004] | 1/2 | 跨 0，**无改进**（thrissur −0.0116） |
| Self − LateDetail | −0.0113 | [−0.0156, −0.0071] | 0/2 | 不跨 0，但**仅在 best 口径**（last 上反向，见 §1.5.1） |

仅这一行区间不跨 0 —— 但**它只在"best 检查点"这一个口径下成立**，见 §1.5.1。

#### 1.5.1 同一检验在 last 检查点上换向

`last.pt` 是第 3,600 步，也有完整逐瓦片结果，可以对同一批瓦片重跑完全相同的检验：

| 比较 | best@500 | 判定 | last@3600 | 判定 |
|---|---:|---|---:|---|
| Cross − Self | +0.0065 [−0.0003, +0.0136] | 跨 0 | **−0.0161 [−0.0269, −0.0046]** | 不跨 0 |
| Cross − LateDetail | −0.0048 [−0.0099, +0.0004] | 跨 0 | −0.0036 [−0.0133, +0.0057] | 跨 0 |
| Self − LateDetail | **−0.0113 [−0.0155, −0.0069]** | 不跨 0 | **+0.0125 [+0.0026, +0.0219]** | 不跨 0 |

两条结论：

1. **"Self 显著更差"不成立。** 换一个检查点，同一个 Self 就变成"显著更好"，而且同样不跨 0。
   这说明在"2 个事件 + 图块级重采样"下，这个检验**没有能力分辨架构**，它分辨的是"你选哪个检查点"。
2. **"Cross 优于 Self"也不成立。** best 上 +0.0065、last 上 −0.0161，方向直接相反。
   而 Cross − LateDetail 在两个检查点上都是小幅负值（−0.0048 / −0.0036），**只有这一项是方向稳定的**。

因此 §4 的判定不能引用"Self 显著更差"，只能引用"Cross 在两个检查点都没超过原模型"。

### 1.6 对象级配对（同一 GT 对象身份）

| 尺寸档 | 对象数 | LateDetail | Self | Cross |
|---|---:|---:|---:|---:|
| 1–4 px | 77 | 0.104 (8) | 0.104 (8) | **0.130 (10)** |
| 5–9 px | 75 | 0.307 (23) | **0.347 (26)** | 0.267 (20) |
| 10–19 px | 101 | 0.347 (35) | **0.366 (37)** | 0.238 (24) |
| 20–49 px | 754 | 0.496 (374) | **0.516 (389)** | 0.450 (339) |
| 50–199 px | 573 | 0.639 (366) | **0.653 (374)** | 0.611 (350) |
| ≥200 px | 99 | **0.667 (66)** | 0.646 (64) | 0.626 (62) |

- **Cross − Self**：新增命中 **40**，失去 **133**（净 −93）
- **Cross − LateDetail**：新增 **36**，失去 **103**（净 −67）
- **Self − LateDetail**：新增 **72**，失去 **46**（净 +26）

Cross 唯一变好的是 1–4 px 档（8→10 个，绝对值太小不足以支撑结论），其余各档全部下降。
Self 反而在 5–199 px 三档都最多——它的问题在精确率（多预测、多误检），不在对象覆盖。

**三架构共同漏检 715 个，其中 ≤19 px **171** 个、≤4 px **63** 个**——与计划引用的 v4r3 数字（172 / 63）几乎完全一致。
说明这批难例对本轮结构改动**完全不敏感**：换交互位置没有触及它们的失败原因。

### 1.7 交换块确实学到了东西

| 检查点 | Self \|α_E\|mean | Self \|α_D\|mean | Cross \|α_E\|mean | Cross \|α_D\|mean |
|---|---:|---:|---:|---:|
| best@500 | 0.0136 | 0.0825 | 0.0236 | 0.0376 |
| last@3600 | 0.0266 | 0.1167 | 0.0485 | 0.0315 |

全部离开零，`P / H_E` 权重范数也显著变化（P 约 11.6→11.7）。因此 §1.4/§1.6 的负面结果**不是模块没接上造成的空转**，
而是一个真在起作用、但方向不对的结构。

---

## 2. 未验证（不能从本轮推断）

1. **没有做阈值扫描。** 因为 Cross 的精确率/召回几乎是等量互换，无法排除"Cross 的差异等价于换了一个工作点"。
   要证伪这一点，需在 DEV 上扫 0.3–0.7 并比较各自最优阈值的 IoU；本轮只用固定 0.5，**因此"Cross 更保守"是描述，不是结论**。
2. **只有单种子（1234）。** 0.003–0.008 级的差值在本协议下没有统计力。Cross − Self 的区间在 best 上跨 0，
   但在 last 上不跨 0 且方向相反（§1.5.1）——**这恰恰说明跨不跨 0 不是架构判据**。需要 42/777 两个种子。
3. **FINAL（newzealand，338 块）未评测**，按规矩保持冻结。因此本报告没有任何独立泛化证据。
4. **只测了一个交换位置（阶段 2）与一种通道宽度。** 计划明确不在本轮扩展，故"阶段 3 之后""双向各一次""宽度加倍"全部未验证。
5. **没有检验"交换是否该带注意力/门控"。** 本轮刻意只用逐通道 α，避免把两件事混在一起。
6. **TRAIN 侧没有独立结论价值。** 三架构 TRAIN 合并 IoU 都在 0.368–0.369，事件均值 0.269–0.273，差异在噪声内。

---

## 3. 存在阻碍

1. **`data_raw` 仍未下载**，因此"harmonized vs raw 输入"这一维度无法进入任何对照——与本轮无关，但持续阻塞相邻问题。
2. **DEV 的小目标几乎只来自 hiroshima。** 253 个 ≤19 px 可评估对象里 251 个在 hiroshima，而 hiroshima 是 `SEEN_DEV`。
   本轮所有与 1–19 px 相关的读数，实际上都是在**单个非盲事件**上得到的。
3. **thrissur 只覆盖 ≥20 px 目标**（1–9 px 为 0 个，10–19 px 仅 2 个）。它能检验"大目标是否退化"，不能检验"小目标是否改善"。
4. **3,600 步预算与最优步数严重错配**：三条 run 的 best 都在 500 步，之后 3,100 步没有产生泛化收益。这是既有协议问题，非本轮引入。

---

## 4. 决策（按计划 §7 逐条）

| 判据 | 结果 | 说明 |
|---|---|---|
| 1. 工程是否正确 | **通过** | 七道门槛全过；精确更新/曝光可追溯；初始身份哈希可核对；α 确认已学习 |
| 2. 是否超过容量对照 | **不成立** | best 上 +0.0065（区间跨 0），last 上 −0.0161（区间不跨 0）——**方向相反，无法判定**；且 best 口径对象级净 −93 |
| 3. 是否带来实际改进 | **不成立（这一条最稳）** | Cross − LateDetail 在 best 与 last 上都是小幅负值（−0.0048 / −0.0036，区间均跨 0），方向一致；AP 打平（+0.0005）；对象命中净 −67 |
| 4. 是否继续该结构 | **停止** | 不尝试更多交换层、更宽通道、更大骨干、不同 LR 或注意力组合 |

**建议：放弃"编码阶段早期双向交换"这条路线，主模型仍为 LateDetail（v4r1 `detailres_n100`）。**

支撑这个建议的只有判据 3 那一行：**Cross 在两个检查点上都没超过原模型，且对象级召回更差**。
判据 2（超过容量对照）实际上**没有结论**——它在两个检查点上符号相反（§1.5.1），不应被当作"已排除"。

这一轮并非白跑，它留下三份可复用资产，但第 (b) 项要按修正后的措辞取用：
(a) 一个被验证过的、可安全挂载的编码期干预点（零初始化恒等、可插拔、不破坏历史 checkpoint）；
(b) 一个**关于峰值而非天花板**的观察：同路径残差会压低第 500 步的峰值（−0.011）但让后期更高、更稳（+0.012、sd 0.0107 vs 0.0152）。
    它说明在本协议下"选 best 检查点"这一做法本身会系统性偏好"峰值高但衰减快"的模型——这是对评估方法的贡献，不是对架构的判决；
(c) 把 715 个三架构共同漏检对象固化成了带 GT 身份的 CSV，可直接给下一轮的对象/尺度监督用。

按计划要求，下一轮若继续，应**单独**检验对象/尺度监督（例如按对象面积的加权、或小目标过采样），
不再与结构改动混在同一次实验里。

---

## 5. 局限（引用本节结论时必须一并说明）

1. 单种子；所有区间是"图块抽样"区间，图块在事件内高度相关，事件级有效样本量只有 2，实际不确定性比区间更宽。
2. DEV 同时是选择集：`best` 口径系统性偏高。三条 run 的 best 都在第 500 步、共 36 个候选点，选择偏差量级与观察到的架构差异相当。
3. hiroshima 为 `SEEN_DEV`，其绝对数值不可与文献基线横比。
4. 只用 0.5 阈值；阈值无关的 AP 是更公平的口径，但 AP 也未做事件级 bootstrap。
5. 对象匹配用 IoU≥0.25 一对一贪心，匹配数对预测对象粒度敏感（Cross 预测对象更少更整块，天然吃亏）。
6. **图块级 bootstrap 在本数据上不具备架构判别力（本轮新增的直接证据）。** 同一批瓦片、同一个检验，
   在 best@500 上判 Self"不跨 0 更差"、在 last@3600 上判 Self"不跨 0 更好"（§1.5.1）。
   区间不跨 0 只能说明"在该检查点的图块抽样下方向稳定"，**不能**当作"架构差异已确立"。
   要真正判定架构，需要多种子（42/777）而不是更多瓦片。

---

## 6. 产物索引

| 文件 | 内容 |
|---|---|
| `reports/v4r4_interface_validation.md` / `.json` | 七道门槛逐项 PASS/FAIL、初始身份哈希、参数分组、100 步拟合曲线、显存 |
| `reports/v4r4_results/arms_comparison.md` | 三架构对照（两种口径、逐事件、对象、成本） |
| `reports/v4r4_results/arms_summary.json` | 上表的机器可读版 |
| `reports/v4r4_results/dev_significance.md` / `.json` | 配对 bootstrap（best 口径）、逐事件差值、固定步数表 |
| `reports/v4r4_results/checkpoint_flip.md` / `.json` | **best vs last 同一检验的方向翻转证据**（§1.5.1，可重跑） |
| `reports/v4r4_results/objects_paired_best_val.csv` | **2,808 个 GT 对象 × 三架构命中状态**（逐对象，含面积/IoU） |
| `reports/v4r4_results/objects_paired_best_val.md` | 按尺寸档召回、新增/失去命中、共同漏检清单 |
| `reports/v4r4_results/cases_best_val.png` / `.json` | 8 个固定案例（GT / 三架构误差图），跨面积档 |
| `configs/segformer/v4r4/…self_residual….yaml` | Self 配置（容量对照） |
| `configs/segformer/v4r4/…cross_exchange….yaml` | Cross 配置（真交换；除 `exchange` 外与 Self 逐项相同，已程序化核对） |
| `src/landslide_seg/models/exchange.py` | 交换块实现 |
| `src/landslide_seg/models/segformer.py` | `_forward_staged`：逐阶段手动编排，插入交换 |
| `src/landslide_seg/models/factory.py` | `model.exchange` 字段与校验 |
| `tests/test_exchange.py` | 11 条语义锁定（零初始化恒等、参数量相等、跨路隔离、形状校验） |
| `scripts/validate_v4r4_exchange.py` | 门槛验证（可重跑） |
| `scripts/compare_v4r4_dev.py` | 显著性 |
| `scripts/pair_v4r4_objects.py` | 对象级配对 |
| `scripts/summarize_v4r4.py` | 三架构汇总 |
| `scripts/make_v4r4_cases.py` | 固定案例图 |
| `outputs/segformer/s2_10band_v4r4/<experiment>/1234/` | best/last、metrics.jsonl、train_summary.json、sample_exposures.json、run_meta.json（含源码 sha256 快照）、eval/ |
| `reports/v4r4_logs/` | 训练与评测原始日志 |

### 复现命令

```bash
PY=C:/Users/yzh/.workbuddy/binaries/python/envs/tm/Scripts/python.exe

# 1) 门槛（必须先全部 PASS）
$PY scripts/validate_v4r4_exchange.py --device cuda:0

# 2) 正式训练（按计划顺序：先 Self，后 Cross）
$PY scripts/train_segformer.py --config configs/segformer/v4r4/segformer_b0_detail_early_self_residual_n100_u3600_s1234.yaml --device cuda:1
$PY scripts/train_segformer.py --config configs/segformer/v4r4/segformer_b0_detail_early_cross_exchange_n100_u3600_s1234.yaml --device cuda:1

# 3) 评测（best/last × train/val；val 保存概率图供对象配对）
RUN=outputs/segformer/s2_10band_v4r4/<experiment>/1234
$PY scripts/evaluate_segformer.py --checkpoint $RUN/best.pt --split val --tag best_val --device cuda:0 --save-predictions
$PY scripts/evaluate_segformer.py --checkpoint $RUN/last.pt --split val --tag last_val --device cuda:0
$PY scripts/evaluate_segformer.py --checkpoint $RUN/best.pt --split train --tag best_train --device cuda:0
$PY scripts/evaluate_segformer.py --checkpoint $RUN/last.pt --split train --tag last_train --device cuda:0

# 4) 汇总
$PY scripts/summarize_v4r4.py
$PY scripts/compare_v4r4_dev.py
$PY scripts/compare_v4r4_checkpoints.py   # best vs last 的方向翻转核对
$PY scripts/pair_v4r4_objects.py \
  --late  outputs/segformer/s2_10band_v4r1_p20/segformer_b0_direct10_detailres_n100_u3600_s1234/1234/eval/predictions_best_val.npz \
  --self  outputs/segformer/s2_10band_v4r4/segformer_b0_detail_early_self_residual_n100_u3600_s1234/1234/eval/predictions_best_val.npz \
  --cross outputs/segformer/s2_10band_v4r4/segformer_b0_detail_early_cross_exchange_n100_u3600_s1234/1234/eval/predictions_best_val.npz
$PY scripts/make_v4r4_cases.py
```
