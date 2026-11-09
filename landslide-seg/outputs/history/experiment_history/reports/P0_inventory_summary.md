# P0 资产汇总：`s2_10band_v4_p20`

- 认证图块 **6,048**；隔离 **7,580**；逐块明细见 `manifests/source_inventory.csv`。
- `可用`=通过准入且通过灾后时间契约的图块；`未标注`=官方 `annotated=False`，本轮不当负样本；
  `标签不可观测`=GT 像素大部分落在 SCL 无效区，监督不可核实。

| 事件 | 可用 | 其中 train/val/test | 前景像素 | 时相 fallback | 未标注 | 标签不可观测 | 其它剔除 |
|---|---:|---|---:|---:|---:|---:|---:|
| chimanimani | 406 | train 406 | 164,169 | 0 | 710 | 17 | 0 |
| china | 72 | train 72 | 13,284 | 30 | 78 | 4 | 5 |
| dominicamaria | 374 | train 374 | 208,317 | 1 | 745 | 29 | 0 |
| hiroshima | 448 | val 448 | 129,078 | 0 | 405 | 11 | 0 |
| hokkaido | 150 | train 150 | 337,491 | 0 | 140 | 0 | 0 |
| indonesia | 235 | train 235 | 99,902 | 34 | 273 | 27 | 38 |
| italy | 2,795 | test 2795 | 812,453 | 0 | 2,462 | 64 | 0 |
| itogon | 119 | train 119 | 17,524 | 0 | 110 | 6 | 0 |
| kyrgyzstan1 | 642 | train 642 | 421,324 | 53 | 780 | 59 | 85 |
| kyrgyzstan2 | 181 | train 181 | 37,561 | 10 | 201 | 20 | 14 |
| lanaodelnorte | 0 | - | 0 | 0 | 71 | 0 | 0 |
| nepal | 8 | train 8 | 1,889 | 0 | 6 | 0 | 0 |
| newzealand | 338 | train 338 | 266,532 | 69 | 589 | 30 | 216 |
| thrissur | 221 | train 221 | 80,743 | 0 | 206 | 0 | 0 |
| usa_alaska | 15 | train 15 | 74,686 | 1 | 69 | 61 | 2 |
| usa_puertorico | 44 | train 44 | 4,886 | 0 | 44 | 3 | 0 |
| **合计** | **6,048** | | **2,669,839** | **198** | **6,889** | **331** | **360** |

## 时相 fallback 事件排行

`fallback` = 灾后 120 天窗口内无可用时相，退化为全灾后最低云量。数量大的事件在跨事件比较时要单独声明。

| 事件 | fallback 图块 | 占该事件可用图块 |
|---|---:|---:|
| newzealand | 69 | 20.4% |
| kyrgyzstan1 | 53 | 8.3% |
| indonesia | 34 | 14.5% |
| china | 30 | 41.7% |
| kyrgyzstan2 | 10 | 5.5% |
| dominicamaria | 1 | 0.3% |
| usa_alaska | 1 | 6.7% |

## 复现

```bash
python scripts/build_source_inventory.py --version s2_10band_v4_p20
```
