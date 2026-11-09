# v4 执行计划（修订版 · 对齐《执行计划2》）

> 修订时间：2026-10-19 08:15。本版按你四条意见重写：①数据可补齐 ②不做那么多包和曲线 ③编码器不只 DOFA ④暂不做 SSL。
> 上一版（08:00）关于"标注池已耗尽"的判断已在 §1 更正。

---

## 1. 数据补齐结果（已执行）

官方 `data_harmonized/s2` 共 **28 卷**，此前本地只有 part01–20。已补齐 21–28（9.7 GiB），本地 `.nc` 从 10,000 → **13,628**。
准入走**两段式**（与 v3 相同，不能省）：`s2_10band_v4`（原始准入）→ **`s2_10band_v4_p20`**（多事件时相裁决）。

| 项 | v3（part01–20） | v4 原始准入 | **v4_p20（最终）** |
|---|---:|---:|---:|
| 本地 `.nc` | 10,000 | 13,628 | 13,628 |
| 认证图块 | 4,772 | 6,303 | **6,048** |
| 清单内 train | 1,480 | 3,060 | 2,805 |
| **OOD_V4 train** | — | — | **2,223（9 事件）** |
| train 事件数 | 7 | 13 | 9（协议内） |
| val / test | 448 / 2,795 | 448 / 2,795 | 448 / 2,795 |
| 未标注（不使用） | 5,055 | 6,889 | 6,889 |

**p20 裁决结果**：`single_date 5,148 / reselect 456 / chosen_after_latest 444 / quarantined 255`。
255 块因"最晚事件日期后 120 天内无可用时相"被隔离，其中 newzealand 157 块（该事件 495 → 338）。

新事件：`newzealand 338`、`thrissur 221`、`kyrgyzstan2 181`、`usa_puertorico 44`、`usa_alaska 15`、`nepal 8`；
`kyrgyzstan1` 从 133 涨到 **642**。资产明细见 `manifests/source_inventory.csv`、`reports/P0_inventory_summary.md`。

**两个必须随结果一起报的质量点**：
- 时相 `fallback` **198** 块：newzealand 20.4%、china 41.7%、indonesia 14.5%、kyrgyzstan1 8.3%；
- 标签不可观测剔除 **331** 块（usa_alaska 61、kyrgyzstan1 59、italy 64）。

未标注图块（6,889）本轮**不使用**：不进监督、不做 SSL、不当负样本。

---

## 2. 三个里程碑（合并掉原来的 G0–G4 五个包）

### M1 · 复核与数据补齐（无 GPU，约 2 天）

| 任务 | 产出 | 状态 |
|---|---|---|
| 1. 补齐 part21–28 → 两段式准入 → `s2_10band_v4_p20` | `manifests/s2_10band_v4_p20*.jsonl`、`reports/P0_inventory_summary.md` | **DONE** |
| 2. 历史 run 预算溯源 | `reports/P0_run_reconciliation.{md,csv}`（54 run，0 例外） | **DONE** |
| 3. 事件划分 + 嵌套训练池（已冻结） | `manifests/split_v4.json`（`frozen=true`, sha `aac30121…`）、`allowed_train_pool.csv` | **DONE** |
| 4. 数据契约复核：波段/DN/SCL/valid/时相/泄漏 | `reports/P0_data_contract_audit.md`（17/17 PASS） | **DONE** |
| 5. 指标单测补齐 | `tests/test_metrics_pooled.py`（14 例，全套 78 passed） | **DONE** |
| 6. 官方协议对照 | `reports/P0_official_benchmark_protocol.md` | **DONE** |
| 7. 派生清单（4 档池 + DEV/FINAL/DIAG） | `manifests/v4/pool_{25,50,75,100}.jsonl`、`dev_thrissur.jsonl`、`diag_italy.jsonl` | **DONE** |
| 8. 端到端冒烟 | `outputs/segformer/s2_10band_v4_p20/SMOKE2_POOL25/` | **DONE** |

**冻结的划分**（`split_v4.json`，已 `--freeze`）：

| 角色 | 事件 | 图块 |
|---|---|---:|
| TRAIN | chimanimani, dominicamaria, indonesia, kyrgyzstan1, kyrgyzstan2, hokkaido, itogon, china, usa_puertorico | **2,223** |
| DEV | hiroshima（`SEEN_DEV`，与 v1–v3 可比）、thrissur | 669 |
| FINAL | newzealand（从未参与任何选择，p20 后 338） | 338 |
| DIAGNOSTIC | italy | 2,795 |
| EXCLUDE | nepal(8)、usa_alaska(15)：样本过少只报告 | 23 |

**预算器只保留一个**：`fixed_update`，统一 **3,600 steps**（与 v3 完全对齐；`max_steps` 在 `segformer_trainer.py` 已实现，无需新代码）。不做双预算矩阵；只有曲线在 100% 处仍明显上升时，才补一次同曝光对照。

### M2 · SegFormer 学习曲线（GPU 约 3 小时，主交付）

嵌套池已生成（`allowed_train_pool.csv`，按 `sha256(sample_id)` 在每个事件内排序，保证 25%⊂50%⊂75%⊂100% 且 9 个事件每档都在）：

| 档 | 图块 | 事件 | 前景像素 |
|---|---:|---:|---:|
| 25% | 559 | 9 | 313,984 |
| 50% | 1,113 | 9 | 665,407 |
| 75% | 1,671 | 9 | 997,527 |
| 100% | 2,223 | 9 | 1,304,458 |

- 主模型 `direct10 + Spectral Stem`，seed 1234 先跑，稳定后 3 seeds
- 两端各加一个 **无 Stem** 对照（`25%`、`100%`），只在图上画 2 个点，不铺满
- 一张三联图：Train pooled IoU / DEV 逐事件 / FINAL 一次性；italy 只作固定诊断点，不参与筛选
- 配套诊断（同一批 run 顺带产出，不额外开实验）：AP、面积档对象召回、`FP objects/tile`、概率直方图
- 产物：`reports/P1_learning_curve.csv`、`P1_summary_and_next_decision.md`、`P1_error_atlas/`

**只在曲线出现平台时才追加两项**（否则跳过并记录跳过原因）：
- 事件多样性（同 N≈1,200，2 / 4 / N 事件数）——回答"加图块"还是"加事件"
- 质量配对消融（`raw` vs `harmonized` 同图块、`Q_DATE` 两种时相规则）

### M3 · 编码器竞争（GPU 约 6 小时）

同一 `split_v4`、同一预算、同一指标。**两阶段**，不是笛卡尔积：

| 波次 | ID | 编码器 | 输入 | 现状 |
|---|---|---|---|---|
| 一 | M0 | SegFormer-B0 + Stem | 10 band | 已有基线 |
| 一 | M1 | SegFormer-B2 | 10 band | 需下权重，同族容量对照 |
| 一 | M2 | DOFA（ViT-B，波长条件） | 10 band **原生匹配** | 需适配器 |
| 一 | M3 | Prithvi-EO-2.0-300M | **6 band（30m HLS）**，需波段子集+重采样 | 需适配器 + 6-band SegFormer 子对照 |
| 二 | M4 | TerraMind v1 small + TiM | 12 band S2L2A | **仓库已有 adapter 与权重**，成本最低 |
| 二 | M5 | AnySat base dense | 10 band + 日期 | **仓库已有 adapter 与权重** |
| 二 | Cost | ResNet18 U-Net | 10 band | 已有 U0 |

关于你问的"为什么只有 DOFA"：原因不是排除 Prithvi，原计划把 M3 类（Prithvi / DINOv3-sat / TerraMind）列为备选，是为了避免一开始就铺成大网格。这个仓库已经跑通过 TerraMind + AnySat 的 dense 特征（V1 双骨干路线，`src/landslide_seg/encoders/`），所以 M4/M5 的适配成本几乎为零，应该排在第二波而不是砍掉。Prithvi 放在第一波，但要**同时**跑一个 6-band SegFormer 对照，否则"6 波段 vs 10 波段"的输入差异会被误读成编码器差异。

每波都是：冻结特征 probe（同容量 head）→ 完整微调 → 参数量/GFLOPs/显存/延迟单列。
产物：`reports/P2_encoder_registry.md`、`P2_probe_results.csv`、`P2_finetune_results.csv`、`P2_encoder_recommendation.md`。

---

## 3. 明确不做

- AFF、条件粗细头、语义文本、DEM/SAR、时相堆叠、对比学习损失
- 未标注图块的 SSL / 半监督（本轮冻结）
- fixed-update × fixed-exposure 双预算矩阵
- 25/50/75/100% × 全编码器 的笛卡尔积
- 在 italy 上调参；历史 v1/v2_clean/v3 及其 checkpoint、manifest、哈希一律只读

---

## 4. 顺序

```
M1 数据补齐+复核(2d) → M2 学习曲线(3h GPU) → M3 编码器(6h GPU)
```

M2 结束必须先回答三个问题再进 M3：数据补到多少、Train 是否学会、加数据后 DEV 与 FINAL 是否同向提升。

---

## 5. 仍需你确认的两点

1. **hiroshima 的身份**：它做了多轮模型选择的 val，严格说已不是盲测。建议新协议里降级为 `SEEN_DEV`，FINAL 用补齐后事件分布再定（等 M1 盘完新事件再给你具体名单），而不是现在锁定。
2. **M2 的 3 seeds 是否全档都跑**（12 组 × 7 min ≈ 1.5 h，成本很低），还是只在 `25%` 与 `100%` 两端跑 3 seeds、中间两档单 seed。
