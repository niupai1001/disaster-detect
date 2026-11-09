# P1-3 结果：固定图块数、改变训练事件覆盖

日期：2026-10-20
性质：**客观结果汇总，不含解释性结论。**
同 N=1251、同归一化统计、同架构、同损失、同增强、同预算（batch 64 / 3,600 次成功更新）、同 seed=1234。

---

## 1. 两个训练集的组成

选择规则见 `manifests/p13/selection_rule.json`（训练前冻结，只依赖 TRAIN 的 GT 属性，未使用 DEV 或任何模型输出）。

| 项 | FewEvents | AllEvents | 共同核心（两组交集） |
|---|---:|---:|---:|
| 图块数 N | 1,251 | 1,251 | 703 |
| 事件数 | 3 | 9 | 3 |
| 前景像素占比 | 0.03834 | 0.03891 | 0.03696 |
| GT 对象数 | 9,535 | 8,783 | 5,072 |
| 平均对象面积 (px) | 69.15 | 76.90 | 70.31 |

### 1.1 对象面积档分布（对象数占比）

| 面积档 | FewEvents | AllEvents | 差值 |
|---|---:|---:|---:|
| 1–4 px | 0.1273 | 0.1055 | +0.0218 |
| 5–9 px | 0.1668 | 0.1193 | +0.0474 |
| 10–19 px | 0.1853 | 0.1531 | +0.0322 |
| 20–49 px | 0.2675 | 0.2914 | −0.0238 |
| 50–199 px | 0.1929 | 0.2585 | −0.0656 |
| ≥200 px | 0.0602 | 0.0722 | −0.0120 |

TVD(面积档) = **0.1014**；相对前景比差 = **0.0148**；合计 score = **0.1162**。

**未消除的差异（据实记录）**：FewEvents 小目标（≤19 px）占比偏高 +0.101，中大型（≥50 px）偏低 −0.078；平均对象面积小 7.75 px（约 10%）；对象总数多 752 个。

### 1.2 每事件配额

| 事件 | 可用 | FewEvents | AllEvents |
|---|---:|---:|---:|
| kyrgyzstan1 | 642 | 642 | 361 |
| chimanimani | 406 | 0 | 229 |
| dominicamaria | 374 | 374 | 210 |
| indonesia | 235 | 235 | 132 |
| kyrgyzstan2 | 181 | 0 | 102 |
| hokkaido | 150 | 0 | 84 |
| itogon | 119 | 0 | 67 |
| china | 72 | 0 | 41 |
| usa_puertorico | 44 | 0 | 25 |

### 1.3 为何是这 3 个事件（候选前 5）

枚举 k∈{2,3,4} 的全部组合，按 `TVD + 相对前景比差` 升序：

| k | 事件 | N | score | TVD | 相对前景 |
|---:|---|---:|---:|---:|---:|
| **3** | **dominicamaria, indonesia, kyrgyzstan1** | **1,251** | **0.1162** | 0.1014 | 0.0148 |
| 3 | chimanimani, dominicamaria, kyrgyzstan1 | 1,422 | 0.1230 | 0.0825 | 0.0405 |
| 4 | china, dominicamaria, indonesia, kyrgyzstan1 | 1,323 | 0.1423 | 0.0990 | 0.0433 |
| 4 | chimanimani, dominicamaria, kyrgyzstan1, usa_puertorico | 1,466 | 0.1425 | 0.0839 | 0.0585 |
| 2 | dominicamaria, kyrgyzstan1 | 1,016 | 0.1853 | 0.1482 | 0.0371 |

注：第 2 名组合的 TVD 更低但前景比差更大；本表按声明的 score 取第 1 名。

---

## 2. 训练与选模

| 组 | n_train | n_val | best DEV | best@步 | @500 | @1200 | @2400 | @3600 | TRAIN IoU@3600 | 秒 | 曝光 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P13-FewEvents | 1,251 | 669 | 0.2300 | 200 | 0.1890 | 0.1725 | 0.1535 | 0.1421 | 0.4877 | 1,071.1 | 230,400 |
| P13-AllEvents | 1,251 | 669 | 0.2908 | 400 | 0.2663 | 0.2691 | 0.2178 | 0.2040 | 0.5325 | 972.9 | 230,400 |
| REF-2223（背景参照） | 2,223 | 669 | 0.3081 | 500 | 0.3081 | 0.2796 | 0.2588 | 0.2430 | 0.4895 | 738.4 | 230,400 |

两组 `complete: true`、`skipped_batches: 0`、唯一曝光样本 1,251。

---

## 3. DEV（best 检查点）

| 组 | 事件均值 | 合并 IoU | Dice | Precision | Recall | AP | 边界 F1 | 对象召回（可评估） | 负瓦片 FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P13-FewEvents | 0.2300 | 0.1999 | 0.3331 | 0.2913 | 0.3890 | 0.2653 | 0.3305 | 0.3615 | 0.01857 |
| P13-AllEvents | 0.2908 | 0.2876 | 0.4467 | 0.4569 | 0.4369 | 0.4100 | 0.4562 | 0.5235 | 0.01019 |
| REF-2223 | 0.3081 | 0.3095 | 0.4727 | 0.4680 | 0.4775 | 0.4425 | 0.4806 | 0.5211 | 0.01065 |

### 3.1 逐事件

| 组 | 事件 | IoU | AP | Dice | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| P13-FewEvents | hiroshima | 0.1720 | 0.2072 | 0.2935 | 0.2275 | 0.4132 |
| P13-FewEvents | thrissur | 0.2881 | 0.4609 | 0.4473 | 0.6187 | 0.3502 |
| P13-AllEvents | hiroshima | 0.2826 | 0.3807 | 0.4407 | 0.4046 | 0.4838 |
| P13-AllEvents | thrissur | 0.2989 | 0.4970 | 0.4602 | 0.6316 | 0.3620 |
| REF-2223 | hiroshima | 0.3114 | 0.4384 | 0.4749 | 0.4150 | 0.5549 |
| REF-2223 | thrissur | 0.3049 | 0.5312 | 0.4673 | 0.6892 | 0.3535 |

### 3.2 尺度召回（对象级，括号为 命中/分母）

| 组 | 1–4 px | 5–9 px | 10–19 px | 20–49 px | 50–199 px | ≥200 px |
|---|---:|---:|---:|---:|---:|---:|
| P13-FewEvents | 0.078 (6/77) | 0.160 (12/75) | 0.149 (15/101) | 0.316 (238/754) | 0.480 (275/573) | 0.616 (61/99) |
| P13-AllEvents | 0.169 (13/77) | 0.333 (25/75) | 0.376 (38/101) | 0.515 (388/754) | 0.618 (354/573) | 0.616 (61/99) |
| REF-2223 | 0.104 (8/77) | 0.307 (23/75) | 0.347 (35/101) | 0.497 (375/754) | 0.642 (368/573) | 0.667 (66/99) |

### 3.3 尺度像素召回

| 组 | 1–4 px | 5–9 px | 10–19 px | 20–49 px | 50–199 px | ≥200 px |
|---|---:|---:|---:|---:|---:|---:|
| P13-FewEvents | 0.133 | 0.154 | 0.197 | 0.296 | 0.387 | 0.443 |
| P13-AllEvents | 0.185 | 0.273 | 0.288 | 0.427 | 0.469 | 0.417 |
| REF-2223 | 0.175 | 0.261 | 0.267 | 0.405 | 0.502 | 0.496 |

---

## 4. last 检查点（第 3,600 步）

| 组 | 事件均值 | 合并 IoU | AP | 对象召回（可评估） |
|---|---:|---:|---:|---:|
| P13-FewEvents | 0.1421 | 0.1136 | 0.1487 | 0.1429 |
| P13-AllEvents | 0.2040 | 0.2006 | 0.3026 | 0.3055 |
| REF-2223 | 0.2430 | 0.2456 | 0.3619 | 0.3401 |

---

## 5. TRAIN 侧（best 检查点）

| 组 | 事件均值 | 合并 IoU | AP | 对象召回（可评估） |
|---|---:|---:|---:|---:|
| P13-FewEvents | 0.2533 | 0.2573 | 0.3710 | 0.1364 |
| P13-AllEvents | 0.2749 | 0.3698 | 0.5326 | 0.2874 |
| REF-2223 | 0.2728 | 0.3681 | 0.5314 | 0.2902 |

---

## 6. 配对比较（AllEvents − FewEvents）

逐瓦片配对、按事件分层 bootstrap 5,000 次（seed 20261020），669 块：

| 检查点 | FewEvents | AllEvents | Δ | 95% 区间 | 跨 0 | P(Δ>0) |
|---|---:|---:|---:|---|---:|---:|
| best_val | 0.2300 | 0.2908 | **+0.0607** | [+0.0422, +0.0778] | 否 | 1.000 |
| last_val | 0.1421 | 0.2040 | **+0.0619** | [+0.0443, +0.0791] | 否 | 1.000 |

逐事件差值：

| 检查点 | hiroshima | thrissur |
|---|---:|---:|
| best_val | +0.1106 | +0.0108 |
| last_val | +0.1198 | +0.0040 |

两个检查点方向一致、区间均不跨 0；但事件级有效样本量只有 2，且 hiroshima 为 `SEEN_DEV`。

---

## 7. 成本

| 组 | 总参数 | encoder | decoder | new | 训练峰值显存 MiB | 训练秒 |
|---|---:|---:|---:|---:|---:|---:|
| P13-FewEvents | 3,753,602 | 3,330,368 | 395,009 | 28,225 | 4,693 | 1,071.1 |
| P13-AllEvents | 3,753,602 | 3,330,368 | 395,009 | 28,225 | 4,693 | 972.9 |

---

## 8. 固定错误案例

`reports/p13/cases_dev.png`：8 例，选例规则为「至少被两组之一漏掉一个可评估对象」，按漏检对象的最小面积档（1–4 → 5–19 → 20–49）分层，同一档内按漏检数降序，覆盖两个事件。列：RGB / GT / FewEvents 误差 / AllEvents 误差（绿=TP，红=FP，蓝=FN）。逐例清单见 `reports/p13/cases_dev.json`。

---

## 9. 事实性局限

1. 单 seed（1234）；单次训练；未做多种子重复。
2. DEV 同时是选择集，`best` 口径系统性偏高；两个检查点已并列给出。
3. 事件级有效样本量 = 2；图块级 bootstrap 的区间必然偏窄。
4. 归一化统计只由两组交集（703 块）拟合，两组共用，因此也不同于 REF-2223 使用的 `stats_core25.json`——**REF 与两组都不可直接横比，仅作背景**。
5. FewEvents 的 3 个事件不是随机抽取，而是按"与 AllEvents 分布最接近"挑选；因此本对照衡量的是**这一组预先选定的具体组合**，不能外推为"事件数"的普遍因果效应。
6. 未评测 FINAL（newzealand）与 Italy，留出身份未变。

---

## 10. 复现命令

```bash
PY=C:/Users/yzh/.workbuddy/binaries/python/envs/tm/Scripts/python.exe

# 1) 构建两组清单 + 冻结选择规则 + 共同核心统计
$PY scripts/build_p13_event_coverage.py --force

# 2) 训练
$PY scripts/train_segformer.py --config configs/segformer/p13/segformer_b0_latedetail_fewevents_n1251_s1234.yaml  --device cuda:1
$PY scripts/train_segformer.py --config configs/segformer/p13/segformer_b0_latedetail_allevents_n1251_s1234.yaml  --device cuda:0

# 3) 评测（best/last × train/val；val 保存概率图）
$PY scripts/evaluate_segformer.py --checkpoint <run>/best.pt --split val   --tag best_val   --save-predictions
$PY scripts/evaluate_segformer.py --checkpoint <run>/last.pt --split val   --tag last_val
$PY scripts/evaluate_segformer.py --checkpoint <run>/best.pt --split train --tag best_train
$PY scripts/evaluate_segformer.py --checkpoint <run>/last.pt --split train --tag last_train

# 4) 汇总
$PY scripts/summarize_p13.py
$PY scripts/compare_p13.py
$PY scripts/make_p13_cases.py --few <few>/eval/predictions_best_val.npz --all <all>/eval/predictions_best_val.npz
```
