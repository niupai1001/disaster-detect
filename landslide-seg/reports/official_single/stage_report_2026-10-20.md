# BENCH_OFFICIAL 单期主实验：结果与阶段判断

日期：2026-10-20
对应计划：`docs/next_execution_plan_official_single_2026-10-20.md`

## 0. 状态

| 步骤 | 状态 |
|---|---|
| 1 固定官方协议 | 完成 |
| 2 映射本地成员 | 完成 |
| 3 单期数据 + 最小验证 | 完成（11 项全过） |
| **4 单期主实验训练与评测** | **完成** |
| 5 官方多时相＋DEM 参考 | **未运行**（可执行性已核验，成本超本阶段预算，见 §5） |
| 6 阶段判断 | 本节 §6 |

> **重要前置声明**：按用户 2026-10-20 决策（选项 A），官方 TRAIN 已包含
> newzealand 的 292 块、hiroshima/thrissur 的 322 块、italy 的 1,135 块。
> **newzealand 的 FINAL 盲测身份终止**；此后任何报告不得再称其为未接触训练事件。
> 原事件 OOD 协议及其全部历史产物保持原样，但两组数值不可并表。

---

## 1. 主实验设置

| 项 | 值 |
|---|---|
| 协议 | 官方 `S12LS-LD` harmonized `s2`，commit `2eb025bd…`，split sha256 `8f0c024f…` |
| 样本（单期可用子集） | train 3,012 / val 758 / test 945（官方 3,192 / 798 / 998，排除 273 块并逐块记录原因） |
| 输入 | 十波段单期 128×128，无 DEM、无多时相 |
| 模型 | SegFormer-B0（ADE 预训练，direct10）+ 原网格 32 通道末端细节分支，3.75 M 参数 |
| 归一化 | 只由官方 TRAIN 拟合（`stats_train.json`），未复用任何旧统计 |
| 训练 | seed 1234、batch 64、FP32、3,600 成功更新、230,400 曝光、AdamW（encoder 6e-5 / 其余 6e-4）、0.5 BCE + 0.5 Dice |
| 选模 | 官方 VAL 前景 pooled IoU@0.5（每 100 更新评测） |
| 耗时 / 峰值显存 | 687 s / 4,694 MiB |

**曲线**（每 100 更新）：VAL pooled IoU 0.3053 @500 → 0.3424 @1200 → 0.3546 @2400 → 0.3604 @3600。
与事件 OOD 的形态不同（那边 500 步见顶后一路下滑），这里全程单调上升，best 落在第 2,700 步。

---

## 2. 表 A · 总体指标

| split | 检查点 | 合并 IoU | Dice | Precision | Recall | AP | 边界 F1 | 对象召回 | 事件均值 | FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VAL | best | **0.3623** | 0.5319 | 0.5685 | 0.4996 | 0.5176 | 0.5282 | 0.4535 | 0.2933 | 0.01444 |
| VAL | last | 0.3604 | 0.5299 | 0.5566 | 0.5055 | 0.5112 | 0.5279 | 0.4545 | 0.2915 | 0.01533 |
| TEST | best | **0.3275** | 0.4935 | 0.5404 | 0.4541 | 0.4830 | 0.5139 | 0.4602 | 0.2860 | 0.01436 |
| TEST | last | 0.3265 | 0.4923 | 0.5279 | 0.4613 | 0.4786 | 0.5113 | 0.4648 | 0.2842 | 0.01533 |

主结果为 VAL 选出的 **best**（第 2,700 步）。TEST 只评测，未参与任何选择。

---

## 3. 表 B · 逐事件（best，TEST）

| 事件 | IoU | AP | | 事件 | IoU | AP |
|---|---:|---:|---|---|---:|---:|
| hiroshima | 0.4956 | 0.6948 | | kyrgyzstan2 | 0.0923 | 0.1258 |
| thrissur | 0.4698 | 0.6996 | | dominicamaria | 0.1478 | 0.2456 |
| hokkaido | 0.4596 | 0.6940 | | china | 0.1233 | 0.0954 |
| chimanimani | 0.3945 | 0.5596 | | usa_puertorico | 0.1047 | 0.1756 |
| italy | 0.3768 | 0.5481 | | itogon | 0.2420 | 0.3147 |
| newzealand | 0.2970 | 0.4603 | | kyrgyzstan1 | 0.2496 | 0.3693 |
| indonesia | 0.2650 | 0.3595 | | | | |

事件间差异很大：头部四个事件 0.46–0.50，尾部三个事件 0.09–0.15。
注意 hiroshima / thrissur / newzealand / italy 均在官方 TRAIN 中有样本（随机划分的必然结果），
其数值不可理解为对未见事件的外推能力。

---

## 4. 表 C · 尺度召回（TEST，对象级，命中/分母）

| 检查点 | 1–4 px | 5–9 px | 10–19 px | 20–49 px | 50–199 px | ≥200 px |
|---|---:|---:|---:|---:|---:|---:|
| best | 0.210 (86/409) | 0.377 (841/2233) | 0.503 (1054/2094) | 0.505 (971/1922) | 0.546 (594/1088) | 0.590 (85/144) |
| last | 0.218 (89/409) | 0.387 (864/2233) | 0.503 (1054/2094) | 0.509 (979/1922) | 0.549 (597/1088) | 0.583 (84/144) |

---

## 5. 表 D · 与事件 OOD 协议的对照（两套协议各用各的测试集）

| 项 | 事件 OOD（原协议，`v4r1 detailres_n100`） | 官方单期（本次） |
|---|---:|---:|
| 训练样本 | 2,223（9 事件） | 3,012（官方随机 split） |
| 归一化统计 | TRAIN-only 559 | TRAIN-only 3,012 |
| 划分性质 | **事件级互斥**（跨事件外推） | 随机划分（同事件跨 split） |
| 测试集 | DEV 669（hiroshima+thrissur） | 官方 VAL 758 / TEST 945 |
| 合并 IoU（主结果） | 0.3095（DEV best@500） | 0.3623 VAL / 0.3275 TEST |
| AP | 0.4425 | 0.5176 VAL / 0.4830 TEST |
| 对象召回 | 0.5211 | 0.4535 VAL / 0.4602 TEST |
| 1–4 px 对象召回 | 0.104 (8/77) | 0.210 (86/409) |
| 曲线形态 | 500 步见顶后下滑 | 全程上升，best@2700 |

**这两列不是同一个测试集，不能相减当作"协议难度"的估计**：样本量、统计、划分性质和测试事件全都变了。
只能作为"两套条件各自的表现"并列呈现。

---

## 6. 阶段判断（按计划 §7）

| 计划判据 | 本次情况 | 是否落入 |
|---|---|---|
| 单期官方条件明显优于当前事件 OOD | VAL 侧明显更高（0.3623 vs 0.3095），TEST 侧幅度小（0.3275） | **部分落入** |
| 单期仍低而本地官方参考高 | 官方参考未运行，无法判断 | 否 |
| 两者都低 | 不适用 | 否 |
| 单期可用但小对象/边界仍弱 | 1–4 px 对象召回 0.210，仍是六档中最低；边界 F1 0.51 | **是** |

对应计划给出的处理：

1. **协议/分布难度是重要因素，但不能仅归因 split。** 官方条件更好，但训练样本从 2,223 增到 3,012、
   统计文件也换成了 3,012 块的拟合结果；而且随机划分让测试事件在训练中出现过。
   按计划要求，保留跨事件研究价值，**不宣称主模型已解决未知事件问题**。
2. **小对象与边界仍弱。** 在确认错误可观测性后，才决定后续表征与空间信息路线；不立即回到损失微调。
3. **官方参考未运行**，因此"我们的单期 vs 官方多时相+DEM"这一比较**尚未完成**，官方表值只能作来源背景。

---

## 7. 官方多时相＋DEM 参考：核验结果与未运行原因

按计划 §6，选择了官方 U-TAE 作为第一候选并做了可执行性核验（`src/models/utae.py`，1.07 M 参数）：

| 项 | 实测 |
|---|---|
| 依赖 | xarray / lightning / torchmetrics / rioxarray / geopandas 已具备；补装 h5netcdf 后官方 dataset 可直接读本地 nc |
| 数据格式 | 与官方一致（15 时相 × 11 通道 128×128，`MASK` ∈ {0,1}），无需改文件布局 |
| 单步成本 | batch 4 前反向 0.104 s，峰值 3,454 MiB |
| 官方预算 | 75 epochs × 3,192 样本 = 59,850 步，仅算力约 6,225 s |
| **实际瓶颈** | 每次 `__getitem__` 都用 xarray 重开 nc 并读 15×11×128×128，未做缓存时 I/O 需数小时 |
| 预估单 seed 墙钟 | **5–8 小时**（含 I/O，不含调参） |

**结论：未运行。** 理由：需要先建多时相缓存（约 30–40 分钟）才能把 75 epoch 压到可接受时长，
加上训练本身已超出本阶段核验预算；按计划"不能执行时明确阻碍，不用 README 数值冒充本地结果"处理。
详细记录见 `reports/official_single/official_reference_feasibility.json`。

若要继续，建议顺序：① 建 15 时相 + DEM 的 npz 缓存；② 用官方 hydra 配置单 seed 复训 U-TAE；
③ 在同一可用子集上评测，才可与本次单期结果配对比较。

---

## 8. 产物

| 内容 | 路径 |
|---|---|
| 训练产物（best/last、曲线、曝光、run_meta 含源码哈希） | `outputs/segformer/official_single/SegFormer-LateDetail-OfficialSplit-SingleDate/1234/` |
| 结果表 | `reports/official_single/results_summary.md`、`results.json` |
| 协议与映射 | `manifests/official_single/{protocol_lock.json,mapping.csv,overlap.md,trainval.jsonl,test.jsonl,stats_train.json}` |
| 官方定义摘录 | `manifests/official_single/official_source_facts.json` |
| 门槛验证 | `reports/official_single/interface_validation.{md,json}` |
| 固定案例 | `reports/official_single/cases_test.png` / `.json` |
| 官方参考核验 | `reports/official_single/official_reference_feasibility.json` |
| 准备阶段报告 | `reports/official_single/prep_report_2026-10-20.md` |
| 决策记录 | `docs/DECISION_LOG.md`、`manifests/official_single/protocol_lock.json`（`decision` 字段） |
| 配置 | `configs/segformer/official_single/*.yaml` |
| 脚本 | `build_official_single_protocol.py`、`validate_official_single.py`、`summarize_official_single.py`、`make_official_cases.py` |
