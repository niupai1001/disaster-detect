# BENCH_OFFICIAL 单期协议：步骤 1–3 完成报告与待决策事项

日期：2026-10-20
对应计划：`docs/next_execution_plan_official_single_2026-10-20.md`

---

## 0. 状态

| 步骤 | 状态 |
|---|---|
| 1 固定官方协议 | **完成** |
| 2 映射本地成员 | **完成** |
| 3 构建单期数据 + 最小验证 | **完成**（11 项门槛全过） |
| 4 单期主实验训练 | **暂停，等待决策**（见 §5） |
| 5 官方多时相＋DEM 参考 | 未开始 |
| 6 阶段判断 | 未开始 |

---

## 1. 步骤 1：官方协议已锁定

| 项 | 值 |
|---|---|
| 仓库 | `https://github.com/PaulH97/Sen12Landslides` |
| commit | `2eb025bd76bef449d2f7cf4aaea861bc4cfeed51`（2026-10-06） |
| 任务 | `S12LS-LD`，harmonized，`s2` |
| `splits.json` | sha256 `8f0c024f…` |
| `norm.json` | sha256 `a97e01cb…` |
| 官方成员数 | train 3,192 / val 798 / test 998，合计 **4,988**（与 README 一致） |

**划分规则**（取自 `configs/splits/s12ls_ld.yaml`，sha256 已记录）：
只含有标注图块（`min_annotated_pixel: 50`、`annotated_only: true`），排除 `usa_alaska` 与 `nepal`，
`val_size/test_size = 0.2/0.2`，**seed 42 的随机划分**。

> 关键性质：**官方 split 不是事件级互斥**。同一事件会同时出现在 train/val/test。
> 因此官方条件与本项目事件 OOD 的分数差异，首先是"同分布随机划分 vs 跨事件外推"的协议差异。

**训练与指标定义**（`configs/lit_module/binary.yaml`、`src/lit_module.py`，sha256 已记录）：

| 项 | 官方 | 本项目 |
|---|---|---|
| 输入 | S2 十波段 + DEM，15 时相 | 十波段单期，无 DEM |
| 损失 | BCEDice，`pos_weight=5`，`dice_w=0.5` | BCE+Dice 等权，`pos_weight=1` |
| 优化 | AdamW `lr=1e-3` 单组，wd 0.01 | encoder 6e-5 / 其余 6e-4 |
| 预算 | 75 epochs，3 seeds（42/123/777） | 3,600 更新，seed 1234 |
| 指标 | torchmetrics Binary{F1,P,R,IoU}@0.5、AUROC、AP | pooled 前景 IoU@0.5 + 对象/边界诊断 |
| **像素范围** | **全 128×128 像素，未排除无效像素** | **仅 valid 像素**（SCL 允许 ∩ 非 nodata ∩ 标签已知） |

像素范围这一条**不是形式差异**：用一个可手算的 2×2 例子验证过——同一预测，把那个 FP 像素标为无效后，
pooled IoU 从 1/3 变为 1/2。所以若要和官方数值并表，必须另跑一次"全像素"口径并单独命名，
不能直接把官方表值放进我们的表。

---

## 2. 步骤 2：成员映射

| split | 官方 | 本地 nc 存在 | 单期规则后可用 | 被排除 |
|---|---:|---:|---:|---:|
| train | 3,192 | 3,192 | **3,012** | 180 |
| val | 798 | 798 | **758** | 40 |
| test | 998 | 998 | **945** | 53 |
| 合计 | 4,988 | 4,988 | **4,715** | 273 |

**本地 nc 一个不缺**（4,988/4,988），磁盘上 13,631 个 nc 覆盖了官方全部成员。

**273 块被排除的原因**（沿用项目固定的单期灾后契约，不偷偷补样）：
云量超阈值（灾后最优时相仍 >50%）、valid 比例 <50%、滑坡像素不可观测（监督不可核实）、
个别 `event_date` 字段不可解析（如 `'None,2016-08-22'`）。

**时相 fallback**：train 240 / val 58 / test 65 块在 120 天窗口内无合规时相，按规则取了全灾后最低云并标记。

已知局限：这是"**官方 split 的可用子集**"，不是完整官方成员。按计划要求，官方参考若要做配对比较，
必须在同一子集上额外评测。

---

## 3. 步骤 3：最小验证（11 项全过）

| 检查 | 结果 |
|---|---|
| `validate_protocol` 在官方协议上通过 | PASS（n_train 3,012 / n_val 758） |
| 三个 split 读取正常 | PASS（`[B,10,128,128]`，掩膜只含 {0,1}） |
| pooled IoU 与手算一致 | PASS（1/3） |
| 全像素 vs valid 口径差异可复现 | PASS（1/3 → 1/2），记为 WARN 提示 |
| 预训练权重已加载（非随机初始化） | PASS（missing 0 / unexpected 0） |
| batch 64 前反向无 OOM、梯度有限 | PASS（峰值 4,649 MiB） |
| 全参数 FP32 | PASS |
| checkpoint 重载输出一致 | PASS（差 0.0） |

统计文件 `stats_train.json` **只由官方 TRAIN 的 3,012 块拟合**，未复用任何旧统计。
选模接口已按计划改成可配置：本协议用 `dev_pooled_iou`（官方 VAL 前景 pooled IoU@0.5），
事件 OOD 主线仍用 `dev_event_mean_iou`，互不影响。

---

## 4. 与现有留出角色的交集（决策依据）

官方 TRAIN 的 3,012 个可用图块，按现有角色分布：

| 现有角色 | 落在官方 TRAIN 的图块数 |
|---|---:|
| 事件 OOD 的 TRAIN 事件 | 1,263 |
| **DIAGNOSTIC（Italy）** | **1,135** |
| **DEV（hiroshima + thrissur）** | **322** |
| **FINAL（newzealand，唯一盲测）** | **292** |
| exclude_small | 0 |

逐事件（可用块）：

| 事件 | 现有角色 | train | val | test |
|---|---|---:|---:|---:|
| italy | diagnostic | 1,135 | 286 | 355 |
| kyrgyzstan1 | train | 387 | 105 | 129 |
| newzealand | **final** | **292** | 70 | 91 |
| chimanimani | train | 216 | 53 | 68 |
| dominicamaria | train | 213 | 51 | 66 |
| hiroshima | **dev** | **203** | 52 | 63 |
| indonesia | train | 149 | 37 | 44 |
| thrissur | **dev** | **119** | 30 | 37 |
| kyrgyzstan2 | train | 98 | 25 | 29 |
| hokkaido | train | 92 | 22 | 30 |
| itogon | train | 50 | 12 | 16 |
| china | train | 40 | 10 | 12 |
| usa_puertorico | train | 18 | 5 | 5 |

---

## 5. 需要你决定（计划 §4 明确要求，不得静默处理）

跑精确官方 split，就必然要在 **newzealand 的 292 块**（当前唯一盲测 FINAL）、
**hiroshima/thrissur 的 322 块**（当前 DEV）、以及 **Italy 的 1,135 块**上训练。

### 选项 A：启用独立官方协议

- 在 `BENCH_OFFICIAL` 命名空间下按官方 split 训练与评测，事件 OOD 的历史产物一律不动。
- 代价：**newzealand 此后不再是"项目未接触过的训练事件"**，现有 FINAL 盲测身份作废；
  hiroshima/thrissur 也不再是干净的 DEV。Italy 本来就是诊断角色，影响最小。
- 收益：可以直接回答"官方任务下这套单期系统表现如何"，并可与官方参考在同一子集上配对比较。

### 选项 B：保持现有角色，停止精确官方 split 训练

- 不训练官方 split；已完成的协议锁定、映射表、单期数据与统计保留，随时可重启。
- 代价：无法给出官方条件下的单期数值；只能报告"官方协议已锁定但未训练"。
- 收益：FINAL（newzealand）盲测身份完整保留，后续若要做真正的跨事件外推验证仍可用。

### 我的建议（供参考，不代替你的决定）

如果**近期还会用 newzealand 做盲测**，选 B；如果**"官方条件下能打多少分"是当前最关键的未知**，选 A，
但要在之后所有报告里明确标注 newzealand 的盲测身份已终止。

另有一个折中不写入选项：只训练官方 TRAIN 减去这三类重叠块，但那就不再是官方 split，
计划已明确禁止"静默移除重叠成员后仍称官方 split 复现"，所以只能作为另一条独立实验另行申报。

---

## 6. 产物

| 内容 | 路径 |
|---|---|
| 协议锁定（commit、哈希、规则、逐 split 计数） | `manifests/official_single/protocol_lock.json` |
| 官方 split/训练/指标定义摘录 | `manifests/official_single/official_source_facts.json` |
| 成员映射（4,988 行） | `manifests/official_single/mapping.csv` |
| 交集表 | `manifests/official_single/overlap.{json,md}` |
| 单期 manifest | `manifests/official_single/trainval.jsonl`、`test.jsonl` |
| TRAIN-only 统计 | `manifests/official_single/stats_train.json` |
| 协议（含 members/dev_ids/test_ids/pools） | `manifests/official_single/protocol.json` |
| 排除清单 | `manifests/official_single/exclusions.jsonl` |
| 最小验证 | `reports/official_single/interface_validation.{md,json}` |
| 配置（已就绪，未训练） | `configs/segformer/official_single/*.yaml` |
| 官方仓库（锁定 commit） | `vendor/official/Sen12Landslides/` |
