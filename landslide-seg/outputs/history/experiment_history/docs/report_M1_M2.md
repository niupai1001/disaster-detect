# M1 重验 + M2 训练报告

> 日期：2026-10-17 深夜  
> 范围：数据认证 → 训练缺陷修复 → 可复现接口 → M1 重验收 → M2 事件互斥训练  
> 原则：只报告实际跑通的结果；无法核实的样本隔离，不用未标注图块充当负样本。

---

## 0. 结论速览

| 项 | 状态 | 关键数字 |
|---|---|---|
| 数据认证 | ✅ 完成 | 候选 100 → **通过 72** / 隔离 28；4 个事件；覆盖 3 个 split |
| 训练缺陷修复 | ✅ 6 项全部修复并有断言 | 见 §2 |
| 可复现接口 | ✅ YAML 驱动 + 种子固定 + checkpoint 元数据 | 见 §3 |
| M1 重验收 | ✅ **all_passed = True** | 8 项检查全过，见 §4 |
| M2 稳定训练 | ⚠️ 训练收敛，但**跨事件泛化很差** | val IoU 0.270 / test IoU 0.011 |
| 消融 | ⏸ 未开始（按要求等模型稳定后再做，当前样本量不足以支撑结论） | — |

**最重要的一句**：管线已经能稳定学习（train loss 1.31 → 0.19），但在**没见过的 italy 事件**上 IoU 只有 0.01。
这是真实结果，不是 bug；原因是训练集只有 26 个图块、2 个事件。要谈"泛化提升"，必须先扩数据。

---

## 1. 数据认证

脚本：`scripts/certify_data.py`　清单：`manifests/certified.jsonl` / `quarantine.jsonl`

### 1.1 认证项（逐条可查）

| 编号 | 检查内容 | 判据 | 实测 |
|---|---|---|---|
| C1 | 源产品 | STAC **同一天**命中同一景 L2A，记录 item id | 全部命中，item 与日期写入清单 |
| C2 | 波段 | 10 波段 + 回补 B01/B09，无 NaN | 12 波段齐全 |
| C3 | 空间网格 | 128×128 @10 m，CRS 一致 | `EPSG:32736` 等 |
| C4 | 辐射 | DN ∈ [0,10000]（±5%） | 全部通过 |
| C5 | 配准 | STAC B08 与 Sen12 B08 相关 ≥ 0.90 | 通过样本中位 **0.980**，最低 0.909 |
| C6 | 标签 | {0,1}；正样本 ≥20 px；云下标签 ≤20% | 3 个因云下标签过多被隔离 |
| C7 | 有效像素 | SCL 派生 valid ≥ 50% | 中位 **96.9%** |

### 1.2 认证结果

```
候选 100 → 通过 72（正样本 72） / 隔离 28
事件分布：chimanimani 19 | dominicamaria 7 | hiroshima 21 | italy 25
掩膜像素：min 23 / p25 60 / 中位 180 / p75 716 / max 4044
云量：中位 0.0%，p90 0.3%（因为按事件日期重选了最清晰的灾后观测）
```

**隔离原因**：C5 配准 27 个、C6 云下标签 3 个。

还有一类必须单独说明的：**680 个未标注图块**（`annotated=False`）。
实测它们的 `MASK` 全为 0、且**没有 `event_date`**，即"没有做标注"而不是"确认无滑坡"。
按规范不能当负样本，已在隔离清单中记录（见 `docs/blockers.md` B6）。

### 1.3 一个关键修正：不能盲用 `pre_post['post']`

实测发现很多样本被指定的"灾后时相"整幅被云覆盖（italy 某样本 SCL=9 占 100%），
而同一景更晚的灾后时相是晴空。

因此改为：**在所有 `date ≥ event_date` 的时相中，选云量最低的一个**。
Sen12 的掩膜是事件滑坡编目、对所有时相相同，所以任何灾后观测都满足"灾后单期观测"的定义，
这不构成标签替换。改完之后可用样本从个位数升到每事件 20+。

---

## 2. 修复的训练缺陷（6 项）

| # | 问题 | 原来 | 现在 | 证据 |
|---|---|---|---|---|
| 1 | **AnySat 日期索引** | `tm_yday`（1–365） | `tm_yday - 1`（AnySat 官方 0–364） | `data/sen12.py::doy` |
| 2 | **冻结骨干被切到 train** | `model.train()` 让全部子模块进入训练态，DropPath/BN 统计漂移 | 每次训练步后强制 `terramind/anysat/text.eval()` | A3：frozen moved **0/10** |
| 3 | **负样本被丢弃** | `min_mask_px=1` 直接跳过空掩膜图块 | 保留逻辑已就绪（`keep_negatives`），但本数据集**没有已验证负样本**，见 §5 | `pipeline.batches_from_sen12` |
| 4 | **valid / 未知标签未覆盖全部监督** | presence 与区域-文本不受 valid 约束 | presence 目标用 valid 掩蔽；区域池化只用 valid 像素；`mask<0` 记为无效 | `losses.py` / `full_model.py` |
| 5 | **语义缺前景-背景对照** | 整图池化成一个向量，标签是"本图有无滑坡" | **同一图内**前景池化 + 背景池化，分别对类别原型做 1/0 匹配 | `region_embedding [B,2,D]`，A2 shapes 可见 |
| 6 | **指标把无效像素算成 TN** | `p &= v` 让无效像素变成 (0,0)，稀释 FPR/OA | 展平后**只保留 valid 像素** | A8：全无效样本计数全 0 |

补充修复：`--set` 多次传参互相覆盖（只生效最后一个）→ 改为 `action="append"` 后合并。

---

## 3. 可复现训练接口

- **YAML 是唯一事实来源**：`configs/default.yaml` 控制模型（通道/尺度/TiM 开关/丢弃率）、
  损失权重、优化器、数据清单、种子、阈值。命令行只做 `--set a.b=c` 覆盖。
- **随机状态固定**：`seed=1234`，同时锁 Python/NumPy/torch/CUDA，并关闭 cuDNN benchmark
  与 TF32（AnySat 注意力对 TF32 敏感）。实际生效值写入 checkpoint。
- **checkpoint 元数据**（`outputs/logs/run_meta.json` 同内容）：
  - `config`：合并后的完整配置（含本次覆盖值）
  - `weights`：三个官方权重的 **大小 + 首尾块 SHA-256**
    （terramind `3d045db2…` / anysat `178b71c6…` / remoteclip `d9d20407…`）
  - `manifest`：数据清单 SHA-256 + 样本数 + 前 200 个 sample_id
  - `splits`：各 split 的完整样本 ID 列表
  - `seed_info`、`params`、`torch` 版本
- checkpoint 只保存非冻结部分（41.3 MB），冻结骨干由官方权重重建，避免版本混淆。

---

## 4. M1 重新验收（`scripts/accept_m1.py`）

在**认证后的真实小批量**（train split，batch=2）上执行：

| 检查 | 结果 |
|---|---|
| A1 三个官方权重加载 | ✅ 504.2 / 503.8 / 605.2 MB，身份哈希已记录 |
| A2 前向形状 | `x12[2,12,128,128]` → `fine[2,1,128,128]`、`region[2,2,512]`、金字塔 128/64/32/16 |
| A3 反向与参数更新 | ✅ 可训练 **10/10** 更新；冻结 **0/10** 变化 |
| A4 保存→新实例加载→推理 | ✅ `max|Δ| = 0.0`（checkpoint 41.3 MB） |
| A5 无 GT 泄漏 | ✅ 翻转 `region_mask` 或置 `None`，logits 差异 **0.0** |
| A6 空图输出 | ✅ 强制门控时输出空；真实负样本 **N/A**（数据集无已验证负样本） |
| A7 可视化 | ✅ `outputs/visualizations/m1_accept_*.png` |
| A8 指标排除无效像素 | ✅ 全无效样本计数 `{0,0,0,0}` |

**`all_passed = True`**（`outputs/logs/accept_m1.json`）

---

## 5. M2 训练

### 5.1 划分（`scripts/make_splits.py`）

| split | 事件 | 样本 | 互斥性 |
|---|---|---|---|
| train | chimanimani + dominicamaria | 26 | — |
| val | **hiroshima** | 21 | 事件互斥 |
| test | **italy** | 25 | 事件互斥 |

校验：跨 split 的空间块 = **0**；每个事件只出现在一个 split。

> 注意：因为没有第 4 个事件可留，val 与 test 都是"整事件留出"，
> 而 train 内部没有再做空间块切分（`--val-block-frac 0`）。这是样本量所限的取舍，已记录。

### 5.2 训练配置与曲线

```
epochs=60, lr=3e-4→1e-3, batch=2, branch_dropout=0,
λ_coarse=0.5, λ_region_text=0.1, λ_presence=0（无负样本，关闭）
骨干与文本塔冻结；可训练 3.5M / 总 526.0M
```

- 780 步，单步 0.22 s
- train loss：前 10 步均值 **1.306** → 后 10 步均值 **0.189**
- 保存 `outputs/checkpoints/last.pt`

### 5.3 指标

| split | IoU | Dice | Precision | Recall | FPR |
|---|---|---|---|---|---|
| val（hiroshima，未见事件） | **0.270** | 0.425 | 0.483 | 0.379 | 0.021 |
| test（italy，未见事件） | **0.011** | 0.021 | 0.022 | 0.019 | 0.005 |

**怎么解读**

- 训练集 loss 明确下降、val 上能到 0.27，说明**整条链路（TiM + AnySat + 语义 + 粗细两阶段）确实在学习**，
  不是"接通但没用"。
- test 掉到 0.01 是**跨事件泛化的失败**：26 个训练图块、2 个事件，模型学到的是这两个事件的地貌外观。
  italy（2023 艾米利亚-罗马涅）与 chimanimani（2019 热带气旋）在植被、土壤、成像条件上都不同。
- 这个差距**不能**用"架构无效"解释，也**不能**用"架构有效"解释——样本量根本不足以区分。
- 也正因如此，现在做 −TiM / −AnySat 消融没有意义：在 26 个样本上任何模块差异都会被噪声吞掉。

---

## 6. 已知限制（诚实清单）

1. **没有已验证负样本**：`annotated=False` 的图块无掩膜、无事件日期，属未标注。
   影响：空图拒检无法训练/验证；`presence` 损失已关闭，`predict` 的门控默认关闭。
2. **样本量太小**：72 个认证图块，训练 26 个。任何精度结论都不具备统计意义。
3. **B01/B09 是 60→10 m 插值**，TiM 的高频输入仍是插值产物。
4. **C5 隔离了 27 个样本**：跨源 B08 相关在 0.33–0.89 之间，原因尚未完全定位
   （该时相与事件相隔较远、或景/轨道差异）。隔离是保守正确，但会损失数据。
5. **掩膜对所有时相相同**：只能做灾后单期评估，不能做变化检测标签。
6. **消融未做**：按要求等"完整模型稳定"后再做，而当前稳定性受样本量限制。

---

## 7. 复现命令

```bash
# 1. 认证（需网络：STAC + COG）
python scripts/certify_data.py --per-event 25 --workers 8

# 2. 划分
python scripts/make_splits.py --test-event italy --val-event hiroshima --val-block-frac 0

# 3. M1 验收
python scripts/accept_m1.py

# 4. M2 训练
python scripts/train_full.py --config configs/default.yaml \
  --set train.epochs=60 optim.lr=0.001 model.branch_dropout=0.0 train.log_every=13

# 5. 测试
python -m pytest tests/ -q     # 10 passed
```

产物：`manifests/certified.jsonl`、`outputs/checkpoints/last.pt`、
`outputs/logs/{accept_m1,metrics,train_history,run_meta}.json`、`outputs/visualizations/m1_accept_*.png`

---

## 8. 下一步（建议顺序）

1. **扩数据**：把 28 个分卷拉全（约 39 GB），认证到 5–10 个事件、每事件 ≥150 图块；
2. **补负样本**：找带"确认无滑坡"标注的数据源，或对未标注图块做人工抽查后转为负样本；
3. **重跑 M2**：≥5 个训练事件 + 事件互斥 test，再看 val/test 差距是否收敛；
4. **然后才做 M3 消融**（−TiM / −AnySat / −SemAlign / −SemLoc / −SemRef / −CoarseFine）。
