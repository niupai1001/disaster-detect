# SegFormer 输入契约（`s2_10band_v1`）

> 生成时间：2026-10-18。对应脚本 `scripts/prepare_segformer_data.py`，summary
> `manifests/s2_10band_v1_summary.json`。本文件记录**实际数据版本的身份**，
> 任何规则改动都必须升版本并重跑四组。

## 1. 版本身份（可直接核对）

| 项 | 值 |
|---|---|
| 版本 | `s2_10band_v1` |
| 清单 | `manifests/s2_10band_v1.jsonl`（2,107 行） |
| 隔离清单 | `manifests/s2_10band_v1_quarantine.jsonl`（2,893 行） |
| 缓存 | `data/s2_10band_v1/*.npz`（2,107 个） |
| 统计 | `data/s2_10band_v1/stats.json`（train 有效像素拟合） |
| `script_sha256` | `03526cb59d03a49ccf194b3898bee89e3b1ba43f900377a7e408ef524c85d854` |
| `rules_sha256` | `3e5ab74b99c787916a7dcabd501bdb9a7114725cc844b58a68aefb09ed2b0e01` |
| `manifest_sha256` | `9142afbbee280fbe29873f166352f5db47aa0b193d565a0c64eafdf942f2ce9f` |
| `stats_sha256` | `2b74bf0c29d99523006878186ba7995b8f66a27fd7a6409d0844e3a90a010b9c` |
| `quarantine_sha256` | `f89f54a8c6c549c4d188840339bff3ef4f5441e4b4f1dc5ffbce1fd3ac3f7383` |

## 2. 波段

固定顺序（写死，不依赖字典/集合顺序）：

```
B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12
```

- 全部来自 Sen12Landslides 单文件的原生 10 波段，**不做 B01/B09 回补**；
- 真彩色取 `R=B04(idx 2), G=B03(idx 1), B=B02(idx 0)`；
- 20 m 原生波段（B05/B06/B07/B8A/B11/B12）在数据集内已重采样到 10 m 共同网格，
  "原分辨率"指该共同网格，不代表新增传感器细节。

## 3. 时相选择

每个瓦片有 15 个时相，掩膜逐期相同（事件级编目）。选择规则：

1. 只考虑 `date >= event_date` 的时相；
2. 在 **灾后 120 天窗口**内，取**最早**满足 `cloud ≤ 0.5` 且 `SCL 允许清单占比 ≥ 0.5` 的时相；
3. 窗口内无解 → 全灾后取最低云量（`temporal_fallback=True`），并照常做最终有效比检查。

实际分布（2,107 个认证样本）：

| 指标 | 值 |
|---|---|
| 灾后天数 p10 / p50 / p90 / max | 5 / 13 / 50 / 985 |
| fallback 数 | 87（china 33、indonesia 53、dominicamaria 1） |
| 灾后天数 > 365 的风险样本 | 16 |
| 与标注 `pre_post.post` 的天数差 p50 / p90 | +10 / +55 |

> 为什么不用"全灾后最低云量"作为唯一规则：实测它与标注给出的 post 时相一致率只有 7.8%，
> p90 滞后 193 天，会把灾后 1–3 年、痕迹已部分恢复的影像当成"灾后观测"。

### 3.1 多日期 `event_date` 的已知风险

China/Indonesia 的 `event_date` 可能是**日期列表**，而当前 P0 只取第一个日期作为"灾后"基准：

| 项 | 数量 |
|---|---:|
| 多日期 `event_date`（train） | 215（china 26、indonesia 189） |
| 所选影像早于最晚日期 | 120（china 8、indonesia 112） |
| `post` 索引早于首个日期 | 114（china 20、indonesia 94） |
| `temporal_fallback` | 87 |

→ 审阅清单 `manifests/s2_10band_v1_multidate_review.csv`（215 行，含全部标志）。
`days_after_event` 对多日期样本**没有统一的物理含义**，不能把所有 `days_delta_vs_post`
解释成"同一事件的选图延迟"。

是否真的标签错配尚未确认：需要回到原始 inventory 的 `ann_id`、几何与事件日期逐对象核实；
在核实前不改日期、不改标签、不删样本。

## 3.2 `s2_10band_v2_clean` 裁决结果（2026-10-18）

对 v1 的 215 个多日期训练样本按"时间证据"裁决（不看模型输出、不改标签）：

| 类别 | 样本数 | 规则 |
|---|---:|---|
| 单日期 | 1,892 | 原样保留 |
| 选图晚于最晚事件 | 95 | 保留；`days_after_event` 基准改为最晚事件日期 |
| 重选时相 | 80 | 最晚事件日期之后 120 天内的最早可用时相（cloud≤0.5、valid≥0.5、不可观测滑坡≤60%） |
| 隔离 | 40 | 窗口内无可用时相（china 5、indonesia 35），理由 `multi_event_unresolvable` |

- 重选样本的 `days_after_latest` 范围 0–120 天（p50 = 5.0）。
- 版本规模：**2,067**（train 1,237 / val 448 / test 382）；**val/test 样本 ID 与 v1 完全一致**，
  因此这次裁决只改变训练集，不影响留出集。
- 身份：`manifest_sha256=cee256690f407d5e…`、
  `stats_sha256=d5f83146be7efa1d…`、
  父版本 `parent_manifest=9142afbbee280fbe…`。
- 清单：`manifests/s2_10band_v2_clean.jsonl`、`_quarantine.jsonl`、`_summary.json`；
  缓存：`data/s2_10band_v2_clean/`；统计：同目录 `stats.json`（只用新 train 拟合）。
- 每条记录新增 `timing_rule` / `event_date_list` / `event_date_ref` / `days_after_event_ref`，
  可追溯到 v1 的 `parent_version` 与 `days_after_event_first`。

**仍未做**：对象—`ann_id` 级别的标签核对；被隔离的 40 个样本是否可通过人工核验恢复。
本版本不声称"标签已修正"，只保证"训练影像时间契约自洽"。


## 3.3 `s2_10band_v3_p20`（parts 1–20，10,000 个 nc）

| 项 | 值 |
|---|---|
| 来源 | Sen12 S2 harmonized `part01–part20`（每卷 500 个 nc，共 10,000） |
| 认证 | raw 4,772 → 清洁裁决后 **4,723** |
| train | **1,480**（chimanimani 406 / china 72 / dominicamaria 374 / hokkaido 150 / indonesia 235 / **itogon 119** / **kyrgyzstan1 124**） |
| val | 448（hiroshima，与 v1/v2 完全一致） |
| test | **2,795**（italy；38,677 个对象，small 92.0%、large 252） |
| 裁决 | 与 §3.2 同规则：single 4,473 / keep-after-latest 134 / reselect 116 / quarantine 49 |
| 身份 | `manifest_sha256=6834ed0a…`、`stats_sha256=4071ade3…`、父版本 `s2_10band_v3_p20_raw` |

注意：v3 的 test 已从 382 张扩到 2,795 张，**与 v1/v2 的 test 不是同一集合**；
跨版本比较只能用 val（hiroshima 448 张未变）。

## 4. `valid10`

```
valid10 = (SCL ∈ {4,5,6,7}) & 10 波段全部有限 & 10 波段全部 > 0 & 标签已知
```

- SCL 用**允许清单**：4 植被 / 5 裸地 / 6 水 / 7 未分类；0/1/2/3/8/9/10/11 与其它未定义码（含 255）一律无效；
- DN = 0 视为填充/无观测（Sen12 实测 DN ∈ [1,10000]）；
- 未知标签（掩膜 255）置 `valid=False`、掩膜写 0，不计损失也不当负样本；
  本版本实测未知像素为 0；
- 认证门槛：`valid_ratio ≥ 0.5`，且不可观测的滑坡像素 ≤ 60%。

实测：valid_ratio 均值 0.93（p10 0.77），所选时相云量均值 0.07（p90 0.23）。

## 5. 标准化

`stats.json` 由 **train split** 的有效像素拟合，val/test 只应用、不重新拟合。

| 波段 | B02 | B03 | B04 | B05 | B06 | B07 | B08 | B8A | B11 | B12 |
|---|---|---|---|---|---|---|---|---|---|---|
| mean (DN) | 518.3 | 740.5 | 652.5 | 1178.8 | 2369.3 | 2776.4 | 2824.1 | 3068.9 | 1931.5 | 1086.7 |
| std (DN) | 838.4 | 830.9 | 868.6 | 860.7 | 1006.9 | 1113.7 | 1153.1 | 1139.0 | 747.5 | 624.2 |
| p50 (DN) | 337.5 | 577.5 | 437.5 | 1032.5 | 2312.5 | 2727.5 | 2752.5 | 3022.5 | 1867.5 | 962.5 |

- 逐波段 z-score：`(x - mean) / std`；
- 处理基线 `+1000 DN` 的常数偏移会被均值吸收，因此本路线**不施加 `boa_offset_dn`**；
  若要算 NDVI 等物理指数，必须单独加回 +1000；
- 输入到预训练首层的是"标准化后的 DN"，与 ImageNet 自然图像分布不同，这是路线本身的前提，
  不声称二者等价。

## 6. 划分

事件级互斥（不做内部 val-block 切分），跨 split bbox 重叠实测 **0**。

| split | 样本 | 区域组 | 事件日期数 | 正瓦片 | 负瓦片 | 正像素 | 有效像素 |
|---|---:|---|---:|---:|---:|---:|---:|
| train | 1,277 | chimanimani 406 / china 77 / dominicamaria 374 / hokkaido 150 / indonesia 270 | 259 | 1,277 | 0 | 881,760 | 19,142,004 |
| val | 448 | hiroshima 448 | 1 | 446 | 2 | 129,078 | 7,036,149 |
| test | 382 | italy 382 | 1 | 382 | 0 | 18,459 | 6,007,478 |

注意：
- `china`（77）与 `indonesia`（270）是**多事件区域组**，报告层按 `(region, event_date)` 分解；
- 可靠负瓦片只有 **2** 个（都在 val），瓦片级误报率只能写 N/A；
- test 只有 italy 一个事件，属"历史固定留出集"，不宣称盲测或强泛化。

## 7. 标签的真实形态（train split 对象级审计）

来源 `scripts/audit_segformer_labels.py` → `outputs/segformer/_audit/labels_train.json`。

| 指标 | 值 |
|---|---|
| GT 正像素 / 有效像素 | 4.61% |
| GT 对象数 | 12,827（平均 10.0 个/正瓦片） |
| 对象面积 p50 / p90 / max | 29 px (0.29 ha) / 144 px (1.44 ha) / 4,272 px (42.7 ha) |
| 分档（全部对象） | small 8,531 / medium 3,493 / large 803 |
| **截断对象**（触边或邻无效区） | 6,711（**52.3%**） |
| 截断后可用于主指标的对象 | 6,116（small 4,495 / medium 1,456 / large 165） |
| 1–4 px 极小对象 | 735（5.7%） |

因此：
- 尺度档位（50 px / 200 px）由对象级分布支撑，冻结不再调整；
- 对象召回必须分层：主指标用**未截断对象**，同时报告全部对象与截断对象；
- 边界 F1 必须排除距瓦片边缘/无效区 ≤ 容差的像素，否则瓦片边缘会被当成滑坡边界。

## 8. 缓存格式

```python
np.load("data/s2_10band_v1/<sample_id>.npz")
# x10   : int16   [10,128,128]  原始 DN（未加 +1000，未标准化）
# mask  : uint8   [128,128]     {0,1}；未知标签写 0 且 valid=False
# valid : bool    [128,128]     valid10
```

## 9. 隔离与已知限制

| 项 | 数量 / 说明 |
|---|---|
| 未标注瓦片（`annotated=False`） | 2,783，隔离，不当负样本 |
| 其它隔离 | 110（valid_ratio < 50% 或滑坡像素不可观测 > 60%） |
| 已知限制 | 负瓦片仅 2 个；test 单事件；标签为事件级编目、非逐期勾绘；52.3% 对象被瓦片边界截断 |
