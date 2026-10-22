# Sen12Landslides 单期滑坡分割 · v4 完整报告

> **2026-10-19 执行修订说明**：本报告记录修正前的初始准备状态。后续审阅发现旧统计包含新 DEV/FINAL、`max_steps` 未实现精确预算、双 DEV 未参与 checkpoint 选择等问题；“准备全部完成/可立即启动”和“从未查看的 FINAL”表述已不再适用。新实验使用独立 `v4r1` 协议、共同559个训练图块的统计、精确更新预算和双 DEV 选择，原有数据与权重保留。请使用 [v4r1复现说明](N:/Users/yzh/Disater/landslide-seg/docs/v4r1_reproduction.md)，勿直接运行本报告 §9 的旧命令。审阅证据见 [修订审阅报告](N:/Users/yzh/Disater/landslide-seg/reports/v4_review_and_next_actions_2026-10-19.md)。

**日期**：2026-10-19
**项目**：`N:/Users/yzh/Disater/landslide-seg`
**范围**：数据补齐 → 数据契约审计 → 实验协议冻结 → 指标口径测试 → 官方基准对齐 → 端到端验证 → 下一步计划
**状态**：准备工作全部完成，M2（SegFormer 数据量学习曲线）可立即启动；**尚无任何模型性能结论**

---

## 0. 摘要

1. **数据补齐了**。官方 `data_harmonized/s2` 共 28 个分卷，此前只下载 20 个。补齐后本地 `.nc` 达官方全量 **13,628**，通过两段式准入得到 **6,048** 个可训练/可评测图块（v3 是 4,772）。
2. **训练事件从 7 个扩到 9 个，事件总数从 9 个扩到 15 个**。协议内训练集 1,480 → **2,223**（+50%）。
3. **必须两段式准入**。只跑一次 `prepare_segformer_data.py` 会让多事件图块用到早于最晚事件日期的影像——相当于静默回滚了 v3 已修的问题。这是本轮最容易踩、也最容易看不见的坑。
4. **实验协议已冻结**（`split_v4.json`，sha256 `aac30121…`）：TRAIN 2,223 / DEV 669 / FINAL 338 / DIAG 2,795，事件级互斥，且 FINAL 从未参与任何选择。
5. **口径已上锁**：数据契约 17 项抽样全通过、指标 14 例单元测试（全套 78 passed）、56 个 run（54 个历史 + 2 次冒烟）的预算账全部对上（0 例外）。端到端冒烟跑通训练 + 三路评测。

**最需要你注意的三件事**：
- FINAL（newzealand）只剩 **338 块**，20.4% 用了时相 fallback —— 单事件留出的方差会很大；
- **hiroshima 不是盲测**（v1–v3 一直用它选 checkpoint），只能算 `SEEN_DEV`；
- 官方 >0.83 的 F1 是 **macro-average**，不是前景 IoU；官方是 **多时相 + DEM**，本项目是**单期 10 波段**，两边的数不能直接比。

---

## 1. 这一轮要解决什么问题

原始计划（《执行计划2》）的核心判断是：当前 SegFormer 在 v3 上 val 前景 IoU 0.2996±0.0067、test 0.0651，**先要判断"是数据不够"还是"表征/优化/任务定义有瓶颈"**，而不是继续往网络上叠模块。执行顺序被规定为：

```
P0 复核（协议/数据/指标可信） → P1 数据规模与质量 → P2 多编码器 → P3 拓展
```

本轮严格按这个顺序做完了 **P0 与 P1 的数据准备部分**。四条执行原则被贯彻：

| 原则 | 落地方式 |
|---|---|
| 不覆盖历史 | v1 / v2_clean / v3 及其 checkpoint、manifest、哈希全部只读；新结果进 `s2_10band_v4*` |
| 先让有效数据进入模型 | 补齐 8 个分卷 + 两段式准入，train 从 1,480 → 2,223 |
| 口径先锁死 | 契约审计 + 指标单测 + 预算对账，三项都在训练前完成 |
| 不允许静默替换 | 多事件裁决、raw/harmonized 缺失、FINAL 缩小，全部写进本报告 |

对原计划的**两处主动调整**（已与你确认）：

1. 不做 fixed-update / fixed-exposure 双预算矩阵，只保留固定 3,600 步；
2. 不做未标注图块的 SSL（6,889 块本轮完全不使用）。

---

## 2. 数据

### 2.1 补齐过程

官方仓库 `data_harmonized/s2` 实际有 27 个完整卷 + 1 个残卷（part28，128 个文件）。此前本地只有 part01–20 = 10,000 个 `.nc`，缺 3,628 个。

补齐 part21–28（9.7 GiB，日志 `outputs/segformer/download_parts_21_28_console.log`）后达 13,628。过程中 part25 遇到 xethub CDN 超时，脚本的 Range 断点续传自动重试完成。

**新增数据来自旧事件补齐及新事件扩展**（例如 kyrgyzstan1 原本已有277个nc）：

| 事件 | 补齐前 nc | 补齐后 nc | 补齐后可用 |
|---|---:|---:|---:|
| kyrgyzstan1 | 277 | 1,566 | 642 |
| newzealand | 0 | 1,173 | 338 |
| thrissur | 0 | 427 | 221 |
| kyrgyzstan2 | 0 | 416 | 181 |
| usa_puertorico | 0 | 91 | 44 |
| usa_alaska | 0 | 147 | 15 |
| nepal | 0 | 14 | 8 |
| lanaodelnorte | 0 | 71 | 0（全部未标注） |

### 2.2 为什么必须两段式准入

`prepare_segformer_data.py` 里选时相用的是 `event_date[:10]`，即**第一个日期**。而 Sen12 中相当一部分图块挂了多个事件日期（形如 `2018-12-16,2017-07-24,None`）。如果只跑一次准入，这些图块可能选到"晚于第一个事件、但早于最晚事件"的影像——用它训练等于让模型在灾前影像上学灾后标签。

v3 当年是用 `v3_raw → v3_p20` 两步解决的。本轮如果直接对 `s2_10band_v4` 开训，就会**静默丢掉这个修正**。所以补了同样的第二段：

| 阶段 | 认证图块 | 隔离 | 说明 |
|---|---:|---:|---|
| `s2_10band_v4`（原始准入） | 6,303 | 7,325 | 未标注 6,889 + 标签不可观测 331 + 其它 105 |
| **`s2_10band_v4_p20`（最终）** | **6,048** | 255 | 多事件无法在窗口内找到合规时相 |

p20 的裁决分布：`single_date 5,148 / reselect 456 / chosen_after_latest 444 / quarantined 255`。

**代价必须记录**：隔离的 255 块中 newzealand 占 157 块（495 → 338）。也就是说，新数据里最有价值的事件，在时间契约上损失了 32%。

### 2.3 最终漏斗

```
13,628 个 .nc
  ├─ 未标注                     6,889  → 本轮不使用（不做 SSL、不当负样本）
  ├─ 标签不可观测（>60% 落在无效区）   331  → 剔除
  ├─ 质量/时相不合格                360  → 剔除（原始准入 105 + 多事件裁决 255）
  └─ 认证                       6,048
       ├─ TRAIN（9 事件）          2,223
       ├─ DEV（hiroshima 448 + thrissur 221）  669
       ├─ FINAL（newzealand）        338
       ├─ DIAG（italy）            2,795
       └─ EXCLUDE（nepal 8 + usa_alaska 15）    23
```

### 2.4 逐事件明细（p20）

| 事件 | 可用 | 角色 | 前景像素 | 时相 fallback | 未标注 | 标签不可观测 |
|---|---:|---|---:|---:|---:|---:|
| italy | 2,795 | DIAG | 812,453 | 0 | 2,462 | 64 |
| kyrgyzstan1 | 642 | TRAIN | 421,324 | 53（8.3%） | 780 | 59 |
| chimanimani | 406 | TRAIN | 164,169 | 0 | 710 | 17 |
| dominicamaria | 374 | TRAIN | 208,317 | 1 | 745 | 29 |
| newzealand | 338 | FINAL | 266,532 | 69（20.4%） | 589 | 30 |
| indonesia | 235 | TRAIN | 99,902 | 34（14.5%） | 273 | 27 |
| thrissur | 221 | DEV | 80,743 | 0 | 206 | 0 |
| kyrgyzstan2 | 181 | TRAIN | 37,561 | 10 | 201 | 20 |
| hokkaido | 150 | TRAIN | 337,491 | 0 | 140 | 0 |
| itogon | 119 | TRAIN | 17,524 | 0 | 110 | 6 |
| china | 72 | TRAIN | 13,284 | 30（41.7%） | 78 | 4 |
| usa_puertorico | 44 | TRAIN | 4,886 | 0 | 44 | 3 |
| usa_alaska | 15 | EXCLUDE | 74,686 | 1 | 69 | 61 |
| nepal | 8 | EXCLUDE | 1,889 | 0 | 6 | 0 |
| hiroshima | 448 | DEV | 129,078 | 0 | 405 | 11 |
| lanaodelnorte | 0 | — | 0 | 0 | 71 | 0 |
| **合计** | **6,048** | | **2,669,839** | **198** | **6,889** | **331** |

三点值得单独说：

- **china 41.7% 用 fallback**，虽然只有 72 块，但中国区样本的时间不可靠程度最高；
- **usa_alaska 61 个图块的标签不可观测**，占该事件 nc 的 41%，这个事件基本没有监督价值，EXCLUDE 是对的；
- **lanaodelnorte 71 个 nc 全部 `annotated=False`**，官方虽给了区域但没给标注，说明"有文件"≠"有标签"。

### 2.5 与 v3 的差异（这是后面所有比较的前提）

| 项 | v3（part01–20） | v4_p20（part01–28） |
|---|---:|---:|
| 本地 `.nc` | 10,000 | 13,628 |
| 认证图块 | 4,772 | 6,048 |
| 清单内 train | 1,480（7 事件） | 2,805（13 事件） |
| **协议内 TRAIN** | — | **2,223（9 事件）** |
| 事件总数 | 9 | 15 |
| 时相 fallback | 106 | 198 |

**注意**：v3 与 v4 的差异不只是"数据变多"，还包含"事件从 7 个变 9 个"和"时相规则重跑"。所以 v3 → v4 的性能变化**不能**单因子归因于数据量。

---

## 3. 实验协议

### 3.1 四类协议必须分开（沿用原计划定义）

| 协议 | 数据/任务 | 能回答什么 | 不能回答什么 |
|---|---|---|---|
| `BENCH_OFFICIAL` | 官方 `S12LS-LD` split + 官方多时相+DEM | 与论文可比性 | 不代表跨事件泛化 |
| `OOD_LEGACY` | v3 的 Train/Hiroshima/Italy | 与历史数值回归对比 | Italy 已被反复查看，不是盲测 |
| **`OOD_V4`** | 本轮冻结的事件互斥划分 | 跨事件泛化（主线） | 只有 newzealand 是干净留出 |
| `SPATIAL_ADAPT` | 同事件空间块互斥 | 同事件域适应 | 不是 strict event-OOD |

本轮只新建 `OOD_V4`，其余三个保留原样。

### 3.2 OOD_V4 冻结划分

`manifests/split_v4.json`，`frozen = true`，`sha256 = aac3012179683dc9…`

| 角色 | 事件 | 图块 | 选它的理由 |
|---|---|---:|---|
| **TRAIN** | chimanimani, dominicamaria, indonesia, kyrgyzstan1, kyrgyzstan2, hokkaido, itogon, china, usa_puertorico | **2,223** | 9 个事件跨 4 大洲，地貌/植被/气候差异大 |
| **DEV** | hiroshima（`SEEN_DEV`）、thrissur | 669 | hiroshima 保证与 v1–v3 可比；thrissur 提供第二个独立开发事件 |
| **FINAL** | newzealand | 338 | 唯一从未参与任何训练或选择的事件 |
| **DIAG** | italy | 2,795 | 历史上被反复查看，只能诊断 |
| **EXCLUDE** | nepal(8)、usa_alaska(15) | 23 | 样本 <20，指标方差过大 |

设计上的取舍，逐条说明：

- **hiroshima 不能进 FINAL**。它在 v1–v3 里是 val，用于选 best checkpoint。把它当盲测等于自欺。但同时它又必须留在 DEV，否则 v3 的历史数值没有可比对象。所以标 `SEEN_DEV`，读结果时只用于"是否与历史同向"，不用于"是否泛化"。
- **newzealand 是唯一真留出**，代价是只有 338 块、20.4% 时相 fallback。这意味着 FINAL 上的标准差会很大，**不允许**用它做任何模型选择或阈值选择。
- **italy 占全部标注的 46%**（2,795/6,048），但它被看过太多次。本轮明确：只画固定诊断点，不参与筛选。
- **EXCLUDE 与官方一致**：官方 README 写明 usa_alaska/nepal 属于 `S12LS-AD-unaligned`，现行任务切分不含它们。

> 顺带记录一个纠错：08:00 版本的划分曾基于未裁决的 v4（FINAL=495）冻结过一版 hash，在**任何训练启动前**已重新冻结。不存在两套协议并行，也没有任何实验用过旧 hash。

### 3.3 嵌套训练池

`manifests/allowed_train_pool.csv`，按 `sha256(sample_id)` 在每个事件内排序取前 `ceil(k·n)`，保证严格嵌套 25% ⊂ 50% ⊂ 75% ⊂ 100%，且**每档 9 个事件都在**（这一点很关键：如果小档丢掉小事件，"数据量"和"事件覆盖"就又混在一起了）。

| 档 | 图块 | 事件 | 前景像素 | 前景像素占比 |
|---|---:|---:|---:|---:|
| 25% | 559 | 9 | 313,984 | 3.43% |
| 50% | 1,113 | 9 | 665,407 | 3.65% |
| 75% | 1,671 | 9 | 997,527 | 3.64% |
| 100% | 2,223 | 9 | 1,304,458 | 3.58% |

前景占比在 3.43%–3.65% 之间（极差 0.22 个百分点），与官方"约 3% 滑坡像素"的量级一致。分层基本有效；25% 档略低 0.15 pp，说明小档的极端小目标略少，读曲线时若 25% 点偏低，要排除这个成分。

### 3.4 派生清单（训练器只认 `split` 字段，所以做了翻译）

| 文件 | train | val | test |
|---|---:|---:|---:|
| `manifests/v4/pool_25.jsonl` | 559 | hiroshima 448 | newzealand 338 |
| `manifests/v4/pool_50.jsonl` | 1,113 | 448 | 338 |
| `manifests/v4/pool_75.jsonl` | 1,671 | 448 | 338 |
| `manifests/v4/pool_100.jsonl` | 2,223 | 448 | 338 |
| `manifests/v4/dev_thrissur.jsonl` | — | — | thrissur 221 |
| `manifests/v4/diag_italy.jsonl` | — | — | italy 2,795 |

---

## 4. 训练预算：一次结案

原计划提到"v1 约 3,000 vs 汇总文档 3,600 optimizer steps 的冲突"。核对 56 个 run 的 `run_meta.json` + `train_summary.json` 后，结论明确：

```
total_steps = epochs × ceil(n_train / batch_size)
```

- v1：150 × ceil(1,277/64) = 150 × 20 = **3,000**
- v3：150 × ceil(1,480/64) = 150 × 24 = **3,600**

**56 个 run 全部满足，0 例外**。所以这不是两套协议，而是同一"固定 epoch"协议在不同 N 下的自然结果。

这个发现直接改变了 M2 的设计：如果沿用"固定 epoch"跑数据量曲线，**数据量增加会同时带来优化预算增加**，曲线就无法归因于数据。因此 v4 统一改为**固定步数** `max_steps = 3,600`，与 v3 的绝对预算严格对齐。

好消息：`max_steps` 在 `segformer_trainer.py`（472–475、597 行）**本来就有实现**，包含 scheduler 的 `total_steps` 换算，不需要新写模块，只改配置。

---

## 5. 验证证据

### 5.1 数据契约审计（17 项全通过）

`scripts/audit_v4_contract.py`，抽样 45 块覆盖 15 个事件，**读真实 nc 而不只看清单字段**：

| 检查 | 结果 | 为什么重要 |
|---|---|---|
| 10 波段集合与顺序 == `S2_10_BANDS` | 45/45 | 顺序错会把 B08 当成 B8A |
| dtype int16、DN ∈ [0, 30000] | 45/45（实测上限 10,000） | 与官方 harmonized 的 [0,10000] 截断一致，反证"本地确实是 harmonized" |
| `valid_ratio` 由 SCL 独立复算（±0.02） | 45/45 | 防止 valid 掩膜与清单脱钩 |
| `mask_px` 由 MASK 独立复算 | 45/45 | 防止标签统计被缓存污染 |
| `date == time_origin + time[idx]` | 45/45 | 防止时相索引错位 |
| 15 时相 MASK 恒定 | 45/45 | Sen12 是单事件掩膜，若随时间变说明读错轴 |
| `date ≥ 最晚 event_date`、`days_after_event` 一致 | 45/45 | 这条直接验证 p20 是否生效 |
| npz 缓存 sha256 | 8/8 | 缓存与清单一一对应 |
| 事件跨 split | 0 | 事件级互斥 |
| bbox 跨 split 重叠 | 0 对 | 空间泄漏 |

### 5.2 指标口径测试

`tests/test_metrics_pooled.py`，**14 个用例**，全套 **78 passed**。锁住的口径：

1. **前景并集为 0 时 IoU 未定义**（`None`/NaN），不得用 1.0 抬高 macro IoU；交集为0但并集非0时IoU为0；
2. 完全重叠 = 1.0、零交集 = 0、`prob == 0.5` 判为正类（`>=` 语义）；
3. **无效像素真正剔除**，不能变成 TN 稀释 FPR——同一组 4 个误报，有效区从 16 减到 8 时，FPR 从 4/16 变 4/8；
4. `pooled ≠ tile-macro`：构造用例里 pooled = 9/17 ≈ 0.529，macro = 0.75，差 > 0.2；
5. 同一批 TP/FP/FN 下 `F1 = 2·IoU/(1+IoU)`；
6. 负图块的 `is_negative_tile` / `pred_positive_on_negative` 标识正确。

这组测试是"历史数值可比"的保险丝：以后任何人改动口径都会红，而不是静默换掉。

### 5.3 端到端冒烟（不是结果）

| run | 配置 | 表现 |
|---|---|---|
| `SMOKE_POOL25` | 54 步 / 6 epoch | 12.1 s，val IoU 0.0（步数太少，全零预测）——证明能跑完、能存 ckpt |
| `SMOKE2_POOL25` | 360 步 / 40 epoch | 1.0 min，best val IoU **0.2573** @ epoch 34 |

用 `SMOKE2` 的 `best.pt` 验证三路评测都能出数：

| 集合 | 图块 | 前景 IoU | Recall |
|---|---:|---:|---:|
| DEV-hiroshima | 448 | 0.2573 | — |
| DEV-thrissur | 221 | 0.2682 | — |
| FINAL-newzealand | 338 | 0.1376 | 0.1872 |

**这些数字只证明链路非退化**：360 步、25% 数据、单 seed。既不能当 v4 基线，也不能与 v3 的 3,600 步数值比。

---

## 6. 官方基准对齐

来源：GitHub README（2026-10-19 抓取）+ HF 数据集目录 API + 论文 DOI。

### 6.1 数据规模（官方原表）

| 模态 | Positive | Negative | Total |
|---|---:|---:|---:|
| S2 | 6,737 | 6,891 | 13,628 |
| S2 任务子集 `S12LS-LD` | **4,988** | — | — |

本地 6,048 比 `S12LS-LD` 的 4,988 多约 1,060 块。差异原因（inventory 过滤、可见性阈值、事件内去重）**尚未逐项核实**——在拿到官方 `splits.json` 之前，**不能声称"我们复现了 S12LS-LD"**。

### 6.2 官方基线（binary 指标，S2 + DEM，15 时相，75 epoch，seeds 42/123/777）

| 模型 | AP | F1 | IoU | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| U-ConvLSTM | 65.13 | 61.95 | **44.88** | 60.59 | 63.92 |
| U-TAE | 67.75 | 61.80 | **44.74** | 53.19 | 74.90 |
| ConvGRU | 60.00 | 59.06 | 41.91 | 56.72 | 61.77 |
| U-Net3D | 62.08 | 58.82 | 41.66 | 55.75 | 62.56 |

### 6.3 三条不可越界的约束

1. 论文摘要里 >0.83 的 F1 是 **macro-average**，不是前景 IoU。README 自己写了这一点；
2. 官方是 **15 时相 + DEM**，本项目主线是**单期 10 波段**——不是同一输入条件；
3. 官方是**随机分层 split**，本项目是**事件级互斥 OOD**——不是同一评测协议。

所以本项目的数目前只能与自己的 v1–v3 比。要对官方比，必须先做 `B-SINGLE`（官方 split + 固定单期）与 `B-TEMP`（官方多时相+DEM），这是 M3 之后的事。

---

## 7. 与《执行计划2》的逐条对照

| 计划条目 | 状态 | 说明 |
|---|---|---|
| P0-A 官方事实对齐 | **部分完成** | 已抓 README/数据规模/基线表/harmonized 定义；未 clone 固定 commit、未拉官方 `splits.json` |
| P0-B 现有代码/数据审计 | **主要完成** | 预算、指标、时相、SCL、valid、泄漏、缓存均审计；缺 `region/inventory_id` 等官方未提供字段 |
| P0-C 标签可见性人工审核 | **未做** | 需要抽样出图 + 人工分级，本轮未启动 |
| P1-0 冻结模型与训练设置 | **完成** | `configs/segformer/v4/base.yaml`，固定 3,600 步 |
| P1-1 建立 v4 训练候选池 | **完成** | 6,048 → 协议内 2,223，含事件/像素/质量分层 |
| P1-2 嵌套学习曲线 | **未开始** | 清单与池已就绪，命令见 §9 |
| P1-3 数据量 vs 事件多样性 | **未做** | 触发式：仅在曲线平台时才做 |
| P1-4 数据质量消融 | **部分可行** | `Q_DATE`、`Q_NORM` 可做；`Q_SOURCE` 因缺 `data_raw` **不可做** |
| P1-5 可靠负样本 | **未做** | 需要负样本审核流程，未标注池禁止直接用 |
| P1-6 训练能力与误差机制诊断 | **未做** | 依赖 P1-2 产物 |
| P1-7 官方条件单期对照 | **未做** | 需要官方 `splits.json` |
| P2 多编码器竞争 | **未开始** | 候选清单已定（见 §10） |
| P3 自监督/跨模态拓展 | **冻结** | 按计划需前面证据支撑 |

---

## 8. 风险与限制（读结果前必须先读）

1. **FINAL 只有 338 块，且 20.4% 时相 fallback**。单事件留出的方差很大，报告必须给逐事件表 + 区间，不能只报一个池化 IoU。若 M2 显示 FINAL 与 DEV 结论不一致，优先信"两个 DEV 同向、FINAL 方向不矛盾"，而不是拿 FINAL 的单点结果下结论。
2. **hiroshima 是 `SEEN_DEV`**。任何"在 DEV 上提升"都要同时看 thrissur 与 FINAL 是否同向，否则可能是对 hiroshima 的过拟合。
3. **fallback 时相不是严格的灾后首景**。china 41.7%、newzealand 20.4%、indonesia 14.5% 属于此列。跨事件比较时必须单列，否则会把"时相问题"误读成"域泛化能力"。
4. **v3 → v4 不能单因子归因**。数据量、事件数、时相规则三者同时变了。要拆开，得等 M2 曲线 + 必要时的事件多样性对照（P1-3）。
5. **没有 raw 产品**。`Q_SOURCE`（raw vs harmonized 配对消融）本轮做不了，需要再下约 10 GB。
6. **标签可见性未人工核查**（P0-C）。目前的"标签不可观测 331 块"是自动规则算出来的，不是人看出来的。若 M2 出现"Train IoU 也很低"，必须先补这一步再谈模型能力。

---

## 9. 下一步：M2 可直接启动

### 9.1 命令

```bash
PY=C:/Users/yzh/.workbuddy/binaries/python/envs/tm/Scripts/python.exe

# 4 档 × seed 1234，固定 3,600 步
for k in 25 50 75 100; do
  $PY scripts/train_segformer.py --config configs/segformer/v4/base.yaml \
     --set segformer.experiment=P1_SIZE_${k}_B0STEM \
     --set segformer.data.manifest=manifests/v4/pool_${k}.jsonl
done

# 每档跑完，三路评测
$PY scripts/evaluate_segformer.py --checkpoint <run>/best.pt --split test \
   --manifest manifests/v4/pool_100.jsonl --tag final
$PY scripts/evaluate_segformer.py --checkpoint <run>/best.pt --split test \
   --manifest manifests/v4/dev_thrissur.jsonl --tag dev_thrissur
$PY scripts/evaluate_segformer.py --checkpoint <run>/best.pt --split test \
   --manifest manifests/v4/diag_italy.jsonl --tag diag_italy
```

### 9.2 成本

360 步含每轮 val 用了 1.0 min → 3,600 步约 **10 分钟/run**。4 档 × 3 seeds + 两端无 Stem 对照（共 14 run）≈ **2–2.5 小时**。算力不是瓶颈，瓶颈是读结果的纪律。

### 9.3 判据（先写死，避免看完数再解释）

| 观测 | 解释 | 下一步 |
|---|---|---|
| Train / DEV / FINAL 随 N 同向上升，100% 处仍在涨 | 数据是主限制，且**已触本地天花板** | 报告 `DATA_LIMITED`，转事件多样性或外部数据 |
| 50% 即平台，增加数据无提升 | 数据量不是主限制 | 转 M3 编码器 |
| Train 高、DEV 高、FINAL 极低 | 事件依赖/捷径学习 | 事件多样性 + 域泛化线 |
| Train 本身低 | 优化/标签/输入有问题 | 先补 P0-C 标签审核，再谈模型 |

### 9.4 M3 编码器清单（候选，未启动）

| 波次 | ID | 编码器 | 输入 | 现状 |
|---|---|---|---|---|
| 一 | M0 | SegFormer-B0 + Stem | 10 band | 已有基线 |
| 一 | M1 | SegFormer-B2 | 10 band | 需下权重 |
| 一 | M2 | DOFA（ViT-B，波长条件） | 10 band 原生匹配 | 需适配器 |
| 一 | M3 | Prithvi-EO-2.0-300M | **6 band/30m**，需波段子集+重采样 | 需适配器 + 6-band SegFormer 对照 |
| 二 | M4 | TerraMind v1 small + TiM | 12 band S2L2A | **仓库已有 adapter 与权重** |
| 二 | M5 | AnySat base dense | 10 band + 日期 | **仓库已有 adapter 与权重** |
| 二 | Cost | ResNet18 U-Net | 10 band | 已有 U0 |

Prithvi 必须配一个 6-band SegFormer 对照，否则"波段数不同"会被误读成"编码器更强"。

---

## 10. 产物索引

### 数据

| 文件 | 内容 |
|---|---|
| `data/sen12/` | 13,628 个官方 harmonized `.nc` |
| `data/s2_10band_v4_p20/` | 6,048 个 npz 缓存 + `stats.json`（train 拟合） |
| `manifests/s2_10band_v4_p20.jsonl` | 最终清单（6,048） |
| `manifests/s2_10band_v4_p20_quarantine.jsonl` | 隔离 255 |
| `manifests/source_inventory.csv` | 逐块资产表 |
| `manifests/split_v4.json` | 冻结划分（sha256 `aac30121…`） |
| `manifests/allowed_train_pool.csv` | 嵌套池成员标记 |
| `manifests/v4/pool_*.jsonl` | 四档训练清单 |
| `manifests/v4/{dev_thrissur,diag_italy}.jsonl` | 单事件评测清单 |

### 报告

| 文件 | 内容 |
|---|---|
| `reports/v4_full_report_2026-10-19.md` | **本报告（自包含）** |
| `reports/P0_data_contract_audit.md` | 17 项契约审计明细 |
| `reports/P0_inventory_summary.md` | 逐事件资产与剔除原因 |
| `reports/P0_run_reconciliation.md` / `.csv` | 56 个 run 的预算对账 |
| `reports/P0_official_benchmark_protocol.md` | 官方协议与口径差异 |
| `docs/plan_v4_execution_2026-10-19.md` | 执行计划（M1/M2/M3） |

### 代码

| 脚本 | 作用 |
|---|---|
| `scripts/reconcile_runs.py` | 扫描全部 run，核对预算算术 |
| `scripts/build_source_inventory.py` | 清单 → 资产表 + 事件汇总 |
| `scripts/build_v4_protocol.py` | 事件划分 + 嵌套池（`--freeze` 打哈希） |
| `scripts/make_pool_manifests.py` | 按档生成训练/评测清单 |
| `scripts/audit_v4_contract.py` | 读真实 nc 的契约审计 |
| `scripts/evaluate_segformer.py` | 新增 `--manifest/--tag`，支持分事件评测 |
| `tests/test_metrics_pooled.py` | 14 例指标口径测试 |

### 配置

`configs/segformer/v4/base.yaml`：`direct10 + Stem`、固定 3,600 步、batch 64、fp32、AdamW、`5% warmup + poly`、BCE+Dice 各 0.5、翻转+90° 旋转。

---

## 11. 一句话收尾

这一轮没有产出任何"更好的模型"，产出的是**一个可以支撑结论的实验底座**：数据从 10,000 补到 13,628 并做了两段式准入，训练集从 1,480 扩到 2,223 且事件从 7 个扩到 9 个，协议冻结并留出一个真正没被看过的 FINAL，口径用 17 项审计 + 14 个单测锁死。接下来该让数据说话了。
