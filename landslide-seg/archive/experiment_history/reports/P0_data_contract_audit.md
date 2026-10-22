# P0 数据契约审计：`s2_10band_v4_p20`

- 抽样 **45** 块（每个事件约 3 块，seed=0），覆盖 15 个事件。
- 清单总计 6,048 块。

## 1. 逐项检查

| 检查 | 抽样数 | 通过 | 结论 |
|---|---:|---:|---|
| 10 波段集合与顺序 == S2_10_BANDS | 45 | 45 | PASS |
| date ≥ event_date | 45 | 45 | PASS |
| date == time_origin + time[idx] | 45 | 45 | PASS |
| days_after_event ≥ 0 | 45 | 45 | PASS |
| days_after_event 与日期差一致 | 45 | 45 | PASS |
| DN ∈ [0, 30000] | 45 | 45 | PASS |
| 波段 dtype = int16 | 45 | 45 | PASS |
| 含 DEM（本任务不使用） | 45 | 45 | PASS |
| 含 MASK | 45 | 45 | PASS |
| 含 SCL | 45 | 45 | PASS |
| MASK 取值 ⊆ {0,1,255} | 45 | 45 | PASS |
| mask_px 可由 MASK 复算 | 45 | 45 | PASS |
| 15 时相 MASK 一致 | 45 | 45 | PASS |
| 时相数 = 15 | 45 | 45 | PASS |
| SCL 取值 ⊆ 0–11 ∪ {255} | 45 | 45 | PASS |
| time_index 合法 | 45 | 45 | PASS |
| valid_ratio 可由 SCL 复算（±0.02） | 45 | 45 | PASS |

## 2. 缓存与泄漏

- npz 缓存哈希核对：8/8 一致。
- 事件跨 split：PASS（每个事件只属于一个 split）。
- bbox 跨 split 重叠对：0。
- 跨 split 相邻（bbox 间距 ≤ 0 m）：0 对
- 重复 sample_id：0。
- 多事件图块：抽样中 10 块；其中所选时相早于最晚事件日期：**0** 块。（>0 说明 p20 多事件裁决没有生效，必须重跑该步骤）

## 3. 抽样明细（前 10 块）

| sample_id | 事件 | DN 范围 | SCL | MASK | date | 复算 valid | 清单 valid |
|---|---|---|---|---|---|---:|---:|
| chimanimani_s2_1486 | chimanimani | 68–7536 | 4,5,6,7,8,9 | [0, 1] | 2019-03-20 | 0.9841 | 0.9841 |
| chimanimani_s2_838 | chimanimani | 25–4388 | 4,5,6,7,8 | [0, 1] | 2019-03-20 | 0.9995 | 0.9995 |
| chimanimani_s2_1536 | chimanimani | 31–7596 | 2,3,4,5,6,7 | [0, 1] | 2019-03-20 | 0.996 | 0.996 |
| china_s2_1045 | china | 116–6664 | 4,5,6,7 | [0, 1] | 2019-04-25 | 1.0 | 1.0 |
| china_s2_680 | china | 1–10000 | 2,3,4,5,6,7 | [0, 1] | 2018-11-01 | 0.8685 | 0.8685 |
| china_s2_953 | china | 590–7676 | 4,5,6,7 | [0, 1] | 2019-06-29 | 1.0 | 1.0 |
| dominicamaria_s2_755 | dominicamaria | 17–9032 | 2,3,4,5,6,7,8,9 | [0, 1] | 2017-10-29 | 0.7695 | 0.7695 |
| dominicamaria_s2_664 | dominicamaria | 104–9032 | 2,3,4,5,6,7,8,9… | [0, 1] | 2017-10-29 | 0.9778 | 0.9778 |
| dominicamaria_s2_550 | dominicamaria | 55–10000 | 2,3,4,5,6,7,8,9 | [0, 1] | 2017-10-29 | 0.9951 | 0.9951 |
| hiroshima_s2_1848 | hiroshima | 135–5468 | 4,5,6 | [0, 1] | 2018-07-16 | 1.0 | 1.0 |

## 4. 异常

- 无。

## 5. 已知限制

- 本地只有官方 `data_harmonized` 一套产品；`data_raw` 未下载，因此 `raw vs harmonized` 配对消融（执行计划 P1-4 `Q_SOURCE`）暂不可做，需要时再补约 10 GB。
- DEM 存在于 nc 中但本任务不使用；它属于官方多时相+DEM 基线（P1-7 `B-TEMP`），不进单期主实验。

## 6. 复现

```bash
python scripts/audit_v4_contract.py --version s2_10band_v4_p20 --sample 48
```
