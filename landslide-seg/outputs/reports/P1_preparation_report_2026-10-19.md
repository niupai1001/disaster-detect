# v4 准备工作报告

**日期**：2026-10-19
**范围**：数据补齐、数据契约审计、协议冻结、指标测试、官方口径对照、端到端冒烟。
**结论**：准备工作全部完成，M2（SegFormer 数据量学习曲线）可以按本文 §7 的命令直接启动。
**本报告不含任何模型性能结论** —— 唯一跑过的训练是两次冒烟，只用于证明链路非退化。

---

## 1. 数据：从 10,000 到 13,628

官方 `data_harmonized/s2` 实际有 **28 个分卷**，此前只下载了 part01–20。补齐 21–28（9.7 GiB）后本地 `.nc` 达到官方全量 13,628。

准入**不能一步到**：`prepare_segformer_data.py` 用 `event_date[:10]` 选时相，多事件图块会用到早于最晚事件日期的影像。必须再跑一次多事件裁决（与 v3 的 `v3_raw → v3_p20` 相同）：

| 阶段 | 认证图块 | 说明 |
|---|---:|---|
| `s2_10band_v4`（原始准入） | 6,303 | 隔离 7,325（其中未标注 6,889、标签不可观测 331、其它 105） |
| `s2_10band_v4_p20`（多事件裁决） | **6,048** | 再隔离 255（`multi_event_unresolvable`） |

裁决分布：`single_date 5,148 / reselect 456 / chosen_after_latest 444 / quarantined 255`。
代价最大的是 newzealand：495 → **338**（157 块在"最晚事件 + 120 天"内没有可用时相）。

### 1.1 与 v3 对比

| 项 | v3 | v4_p20 | 变化 |
|---|---:|---:|---|
| 本地 `.nc` | 10,000 | 13,628 | +36% |
| 认证图块 | 4,772 | 6,048 | +27% |
| 协议内 train（OOD_V4） | 1,480 | **2,223** | **+50%** |
| train 事件数 | 7 | **9** | +2 |
| 事件总数 | 9 | 15 | +6 |

新增事件：`newzealand 338`、`thrissur 221`、`kyrgyzstan2 181`、`usa_puertorico 44`、`usa_alaska 15`、`nepal 8`；`kyrgyzstan1` 从 133 → **642**。
`lanaodelnorte` 有 71 个 nc 但全部未标注，可用 0。

### 1.2 必须随结果一起报的质量事实

- **时相 fallback 198 块**：china 41.7%、newzealand 20.4%、indonesia 14.5%、kyrgyzstan1 8.3%。这些图块在灾后 120 天窗口内无合规时相，退化为"全灾后最低云量"。
- **标签不可观测剔除 331 块**：usa_alaska 61、kyrgyzstan1 59、Italy 64。
- **未标注 6,889 块**：本轮不用于监督、不做 SSL、不当负样本。
- 逐事件明细：`reports/P0_inventory_summary.md`；逐块明细：`manifests/source_inventory.csv`。

---

## 2. 协议：已冻结的 OOD_V4

`manifests/split_v4.json`，`frozen=true`，`sha256 = aac3012179683dc9…`。

| 角色 | 事件 | 图块 |
|---|---|---:|
| TRAIN | chimanimani, dominicamaria, indonesia, kyrgyzstan1, kyrgyzstan2, hokkaido, itogon, china, usa_puertorico | 2,223 |
| DEV | hiroshima（`SEEN_DEV`）、thrissur | 669 |
| FINAL | newzealand | 338 |
| DIAGNOSTIC | italy | 2,795 |
| EXCLUDE | nepal(8)、usa_alaska(15) | 23 |

设计说明：

- **hiroshima 只能算 `SEEN_DEV`**。它在 v1–v3 一直用于选 checkpoint，放进 FINAL 是自欺。留在 DEV 的价值是与历史数值直接可比。
- **newzealand 是唯一从未参与任何选择的留出集**，但只有 338 块，且 20.4% 用了时相 fallback，结论必须带这个前提。
- 官方任务本身也不含 usa_alaska / nepal（README 说明它们在 `S12LS-AD-unaligned` 里），所以 EXCLUDE 与官方一致，不是我们挑剔。
- 早先基于 v4（未裁决）冻结过一版 hash，**在任何训练启动前已重新冻结**，不存在两套协议并行。

### 2.1 嵌套训练池（`manifests/v4/pool_*.jsonl`）

按 `sha256(sample_id)` 在每个事件内排序取前 k·n，保证 25% ⊂ 50% ⊂ 75% ⊂ 100% 且每档都覆盖 9 个事件：

| 档 | 图块 | 事件 | 前景像素 |
|---|---:|---:|---:|
| 25% | 559 | 9 | 313,984 |
| 50% | 1,113 | 9 | 665,407 |
| 75% | 1,671 | 9 | 997,527 |
| 100% | 2,223 | 9 | 1,304,458 |

每份派生清单的结构：`train = 该档池`、`val = hiroshima`、`test = newzealand`。
另外两组只有单一事件的评测清单：`dev_thrissur.jsonl`（221）、`diag_italy.jsonl`（2,795）。

---

## 3. 预算口径：3,000 / 3,600 的疑案已结

`reports/P0_run_reconciliation.md`：扫描 **56 个 run**（54 个历史 + 2 次冒烟），全部满足

```
total_steps = epochs × ceil(n_train / batch_size)
```

**0 个例外**。v1 是 150 × 20 = 3,000（n=1,277），v3 是 150 × 24 = 3,600（n=1,480）——同一协议在不同 N 下的自然结果，不是两套协议。

后果：若照搬"固定 epoch"跑数据量曲线，数据量和优化预算会被绑定。**v4 统一改为固定步数** `max_steps = 3,600`，与 v3 的绝对预算对齐。
`max_steps` 在 `segformer_trainer.py`（472–475、597 行）本来就有实现，包含 scheduler 的 `total_steps`，**不需要新写模块**，只改配置。

---

## 4. 数据契约审计（17/17 PASS）

`reports/P0_data_contract_audit.md`，抽样 45 块覆盖 15 个事件，读真实 nc 而非清单字段：

| 类别 | 结果 |
|---|---|
| 10 波段集合与顺序 == `S2_10_BANDS` | 45/45 |
| dtype int16、DN ∈ [0, 30000]（实测上限 10,000，与 harmonized 截断一致） | 45/45 |
| `valid_ratio` 可由 SCL 独立复算（±0.02） | 45/45 |
| `mask_px` 可由 MASK 独立复算 | 45/45 |
| `date == time_origin + time[idx]` | 45/45 |
| 15 时相 MASK 恒定 | 45/45 |
| `date ≥ 最晚 event_date`、`days_after_event ≥ 0` 且与日期差一致 | 45/45 |
| npz 缓存 sha256 | 8/8 |
| 事件跨 split | 0 |
| bbox 跨 split 重叠 | 0 对 |

已知缺口：本地只有 `data_harmonized`，`data_raw` 未下载，因此执行计划的 `Q_SOURCE`（raw vs harmonized 配对消融）**本轮做不了**，需要再下约 10 GB。

---

## 5. 指标口径测试

`tests/test_metrics_pooled.py` 新增 **14 个用例**，全套 **78 passed**。锁住的口径：

1. 前景交集为 0 时 IoU **未定义**（`None`/NaN），不得用 1.0 抬高 macro IoU；
2. 完全重叠 = 1.0、零交集 = 0、`prob == 0.5` 判为正类；
3. 无效像素**真正剔除**，不能变成 TN 稀释 FPR（同一组误报，有效区减半时 FPR 从 4/16 变 4/8）；
4. `pooled ≠ tile-macro`（构造用例里 9/17 ≈ 0.529 vs macro 0.75，差 > 0.2）；
5. 同一批 TP/FP/FN 下 `F1 = 2·IoU/(1+IoU)`；
6. 负图块的 `is_negative_tile` / `pred_positive_on_negative` 标识正确。

这些测试是"历史数值可比"的保险丝：任何一条被改动就会失败，而不是静默换口径。

---

## 6. 端到端冒烟（不构成结果）

| run | 配置 | 结果 |
|---|---|---|
| `SMOKE_POOL25` | 54 步，6 epoch | 12.1 s，val IoU 0.0（步数太少，全零预测）——证明能跑完、能存 ckpt |
| `SMOKE2_POOL25` | 360 步，40 epoch | 1.0 min，**best val IoU 0.2573 @ epoch 34** |

用 `SMOKE2` 的 `best.pt` 验证三路评测都能出数：

| 集合 | 图块 | 前景 IoU | 备注 |
|---|---:|---:|---|
| DEV-hiroshima（val） | 448 | 0.2573 | 用于选 checkpoint |
| DEV-thrissur | 221 | 0.2682 | `--manifest manifests/v4/dev_thrissur.jsonl --tag dev_thrissur` |
| FINAL-newzealand | 338 | 0.1376（recall 0.187） | `--manifest manifests/v4/pool_25.jsonl --split test --tag final` |
| DIAG-italy | 2,795 | 未跑 | 与 v3 的 0.3068 直接对照时才需要 |

注意：这些数字来自 **360 步、25% 数据、单 seed** 的冒烟，**不能**当作"v4 基线"，也不能与 v3 的 3,600 步数值比较。

---

## 7. M2 启动清单（可直接执行）

```bash
PY=C:/Users/yzh/.workbuddy/binaries/python/envs/tm/Scripts/python.exe

# 4 档 × seed 1234，固定 3,600 步
for k in 25 50 75 100; do
  $PY scripts/train_segformer.py --config configs/segformer/v4/base.yaml \
     --set segformer.experiment=P1_SIZE_${k}_B0STEM \
     --set segformer.data.manifest=manifests/v4/pool_${k}.jsonl
done

# 评测：DEV(hiroshima 已在训练内做 val) + thrissur + FINAL + DIAG
$PY scripts/evaluate_segformer.py --checkpoint <run>/best.pt --split test \
   --manifest manifests/v4/pool_100.jsonl --tag final
$PY scripts/evaluate_segformer.py --checkpoint <run>/best.pt --split test \
   --manifest manifests/v4/dev_thrissur.jsonl --tag dev_thrissur
$PY scripts/evaluate_segformer.py --checkpoint <run>/best.pt --split test \
   --manifest manifests/v4/diag_italy.jsonl --tag diag_italy
```

成本估算（实测外推）：360 步含每轮 val 用了 1.0 min → 3,600 步约 **10 分钟/run**；4 档 × 3 seeds + 两端无 Stem 对照（共 14 run）≈ **2–2.5 小时**。

---

## 8. 官方口径（摘要，详见 `reports/P0_official_benchmark_protocol.md`）

官方 `S12LS-LD`（S2）基线是 **多时相 + DEM**、75 epoch、3 seeds（42/123/777）、binary 指标：

| 模型 | AP | F1 | IoU |
|---|---:|---:|---:|
| U-ConvLSTM | 65.13 | 61.95 | 44.88 |
| U-TAE | 67.75 | 61.80 | 44.74 |
| ConvGRU | 60.00 | 59.06 | 41.91 |
| U-Net3D | 62.08 | 58.82 | 41.66 |

三条不可越界的约束：

1. 论文摘要里 >0.83 的 F1 是 **macro-average**，不是前景 IoU；
2. 官方是 15 时相 + DEM，本项目主线是单期 10 波段，**不是同一输入条件**；
3. 官方是随机分层 split，本项目是事件级互斥 OOD，**不可直接排名**。

要对官方比，得先做 `B-SINGLE`（官方 split + 固定单期）与 `B-TEMP`（官方多时相+DEM），那是 M3 之后的事。

---

## 9. 风险与限制（读结果前先读这里）

1. **FINAL 只有 338 块**，且 20.4% 是时相 fallback。单事件留出的方差会很大，报告必须给 per-event 与置信区间，不能只报一个池化 IoU。
2. **hiroshima 不是盲测**，任何"在 DEV 上提升"都要同时看 FINAL 与 thrissur 是否同向。
3. **delta 类结论受 fallback 影响**：newzealand / china / indonesia 的部分时相不是严格灾后首景，跨事件比较时要单列。
4. **v3 → v4 的数值变化不能直接归因于"数据变多"**：数据集从 7 事件扩到 9 事件、时相规则也重跑了一遍。要分解，得等 M2 曲线 + 必要时的事件多样性对照。
5. **无 raw 产品**，光谱版本这一维本轮无法验证。

---

## 10. 产物索引

| 文件 | 内容 |
|---|---|
| `manifests/s2_10band_v4_p20{,_quarantine}.jsonl` | 最终数据清单（6,048 / 255） |
| `manifests/source_inventory.csv` | 逐块资产表（6,048 行） |
| `manifests/split_v4.json` | 冻结的 OOD_V4 划分（含 sha256） |
| `manifests/allowed_train_pool.csv` | 嵌套池成员标记 |
| `manifests/v4/pool_{25,50,75,100}.jsonl` | 四档训练清单 |
| `manifests/v4/{dev_thrissur,diag_italy}.jsonl` | 单事件评测清单 |
| `reports/P0_inventory_summary.md` | 逐事件资产与剔除原因 |
| `reports/P0_data_contract_audit.md` | 17 项契约审计 |
| `reports/P0_run_reconciliation.{md,csv}` | 56 个 run 的预算对账 |
| `reports/P0_official_benchmark_protocol.md` | 官方协议与口径差异 |
| `tests/test_metrics_pooled.py` | 指标口径测试（14 例） |
| `docs/plan_v4_execution_2026-10-19.md` | 计划（已同步为 v4_p20 数字） |

新增脚本：`scripts/reconcile_runs.py`、`scripts/build_source_inventory.py`、`scripts/build_v4_protocol.py`、`scripts/make_pool_manifests.py`、`scripts/audit_v4_contract.py`；`scripts/evaluate_segformer.py` 增加 `--manifest/--tag`（此前只能按 run 内清单评一个 split）。
