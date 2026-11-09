# P0 官方协议对照：Sen12Landslides

> 抓取时间：2026-10-19。来源按"可核查"原则逐条列出来源与口径，不做数值排名。
> 本文只做协议对照；**没有**完成官方基线复现。

## 1. 来源

| 编号 | 来源 | 版本/时间 | 用途 |
|---|---|---|---|
| S1 | `https://github.com/PaulH97/Sen12Landslides` README（main 分支） | 2026-10-19 抓取 | 数据规模、任务切分、基线表、harmonized 定义 |
| S2 | Hugging Face `paulhoehn/Sen12Landslides`（`data_harmonized/s2` 目录 API） | 2026-10-19 抓取，28 个分卷 | 本地数据来源与卷数 |
| S3 | Höhn et al., *Scientific Data* 12:1772 (2025), doi:10.1038/s41597-025-06167-2 | 论文 | 任务定义与实验 3（六地理群组 leave-one-cluster-out） |
| S4 | `https://github.com/VMarsocci/pangaea-bench` | 未核 | 后续模型比较参考，本轮未使用 |
| S5 | `https://github.com/zhu-xlab/DOFA` | 未核 | P2/M2 编码器候选，本轮未使用 |

**未做**：官方仓库本地 clone 与 commit 锁定。正式复现前必须固定 commit 并把 `tasks/S12LS-LD/harmonized/s2/splits.json` 落到本地。

## 2. 数据规模（S1 原表）

| 模态 | Positive | Negative | Total |
|---|---:|---:|---:|
| S1-asc | 6,492 | 6,814 | 13,306 |
| S1-dsc | 6,347 | 6,275 | 12,622 |
| **S2** | **6,737** | **6,891** | **13,628** |
| Aligned | 6,026 | 5,693 | 11,719 |

任务切分：

| 模态 | S12LS-LD | S12LS-AD(pos.) | S12LS-AD(neg.) |
|---|---:|---:|---:|
| **S2** | **4,988** | 3,262 | 4,824 |

- `S12LS-LD` = 滑坡检测，**只含有标注的正图块**；
- `S12LS-AD` = 异常检测，LD 作为正类 + 无标注图块作为负类，且**只保留单事件 inventory**（每条都有 `event_date`）；
- `S12LS-AD-unaligned`（早期版本）"still including the usa_alaska/nepal inventories" → **现行任务切分不含 usa_alaska / nepal**。

**由此确认**：本地 `s2_10band_v4` 的 6,303 块是"全部可用的已标注 S2"，比 LD 的 4,988 多约 1,300 块。
差异原因尚未逐一核实（候选：LD 的 inventory 过滤规则、可见性阈值、事件内去重）。**在拿到官方 `splits.json` 之前，不得声称"我们复现了 S12LS-LD"。**

## 3. Harmonized 与 raw（S1）

| 项 | harmonized（本地使用） | raw |
|---|---|---|
| S2 辐射 | 对 ESA Baseline 04.00（2022-01-25 起）的 **+1000 DN 偏移**做校正 | 不校正 |
| S2 值域 | 截断到 **[0, 10000] DN** | 原始值 |
| DEM | 截断 **[0, 8800] m** | 不截断 |
| S1 | 线性功率 → dB，截断 [-50, 10] | 线性功率 |

本地 `data/sen12` 只有 `data_harmonized`；`data_raw` 未下载。因此：

- `Q_SOURCE`（raw vs harmonized 配对消融）**本轮不可做**，需再下约 10 GB；
- 审计中 DN 上限实测恰为 10,000（`reports/P0_data_contract_audit.md`），与 harmonized 的截断一致，可作为"本地确实是 harmonized"的间接证据。

## 4. 图块格式（S1）

128×128 px × **15 个时相**；S2 波段 `B02–B08, B8A, B11, B12`（10 个）+ `SCL` + `DEM` + `MASK`；
属性含 `ann_id, ann_bbox, event_date, date_confidence, pre_post_dates, annotated, satellite, center_lat, center_lon, crs`。

本地审计已逐项验证：10 波段集合与顺序一致、`MASK` 在 15 个时相上恒定、`date = time_origin + time[idx]`、`valid_ratio` 可由 SCL 独立复算（±0.02）。

## 5. 官方基线（S1，binary 指标，S2+DEM，多时相）

| 模型 | AP | F1 | IoU | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| U-ConvLSTM | 65.13 | 61.95 | **44.88** | 60.59 | 63.92 |
| U-Net3D | 62.08 | 58.82 | 41.66 | 55.75 | 62.56 |
| ConvGRU | 60.00 | 59.06 | 41.91 | 56.72 | 61.77 |
| U-TAE | 67.75 | 61.80 | **44.74** | 53.19 | 74.90 |

训练条件：harmonized `S12LS-LD` split，`lit_module=binary`，**75 epochs**，3 seeds（42/123/777），测试集跨 seed 平均。

**口径警告（S1 明确说明）**：
- 论文里的 macro-average 指标与这里的 landslide 二分类指标**不是同一个量**；
- 因此**不能**把论文摘要里 >0.83 的 F1 当作前景 IoU，也不能与本项目的 IoU 直接比；
- 官方是**多时相 + DEM**，本项目主线是**单期 10 波段**，两者不是同一输入条件下的比较。

## 6. 与本项目的逐项差异

| 维度 | 官方 S12LS-LD 基线 | 本项目 v4 主线 | 是否可直接比较 |
|---|---|---|---|
| 任务 | 多时相滑坡检测 | **单期**滑坡痕迹分割 | ✗ |
| 输入 | S2 10 波段 + DEM，15 时相 | S2 10 波段单期，无 DEM | ✗ |
| 数据规模 | 4,988 正图块 | 6,303（train 3,060） | 需先对齐 split |
| 划分 | 官方随机分层 split | **事件级互斥**（OOD_V4） | ✗ |
| 归一化 | 官方 `norm.json` | train 有效像素逐波段 z-score | 近似，需核对 |
| 指标实现 | binary foreground AP/F1/IoU（官方脚本） | pooled 前景 IoU@0.5 + 对象/边界诊断 | 口径需逐条核对 |
| 训练预算 | 75 epochs | 3,600 optimizer steps | 不可直接换算 |

**结论**：本项目的数值目前**只能与自己的 v1–v3 比**。要与官方比，必须先完成 `B-SINGLE`（官方 split + 固定单期规则）与 `B-TEMP`（官方多时相+DEM）。这两项属于 M3 之后的 P1-7，不进 M2。

## 7. 待办

1. 拉取 `tasks/S12LS-LD/harmonized/s2/splits.json` 与 `norm.json`，核对 4,988 个 patch id 与本地 6,303 的交集，解释差的 1,300 块；
2. 官方仓库固定 commit，写进 `reports/P0_official_benchmark_protocol.md` 的修订记录；
3. 若要做 `B-TEMP`，需要 `DEM` 通道与多时相 loader —— 本地 nc 已含 DEM，工程上可行，但属于新增输入，必须独立申报。
