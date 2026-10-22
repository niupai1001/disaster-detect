# 官方 split 与现有事件角色的交集

来源 `https://github.com/PaulH97/Sen12Landslides` @ `2eb025bd76bef449d2f7cf4aaea861bc4cfeed51`

## 1. 每个 split 的成员按现有角色分布

| split | dev | diagnostic | final | train |
|---|---:|---:|---:|---:|
| train | 322 | 1135 | 292 | 1263 |
| val | 82 | 286 | 70 | 320 |
| test | 100 | 355 | 91 | 399 |

## 2. 官方 TRAIN 落在现有角色上的图块数

- 当前 **dev**：322
- 当前 **diagnostic**：1135
- 当前 **final**：292
- 当前 **exclude_small**：0
- 当前 **train**：1263

## 3. 逐事件

| 事件 | 现有角色 | train | val | test | 合计 |
|---|---|---:|---:|---:|---:|
| italy | diagnostic | 1135 | 286 | 355 | 1776 |
| kyrgyzstan1 | train | 387 | 105 | 129 | 621 |
| newzealand | final | 292 | 70 | 91 | 453 |
| chimanimani | train | 216 | 53 | 68 | 337 |
| dominicamaria | train | 213 | 51 | 66 | 330 |
| hiroshima | dev | 203 | 52 | 63 | 318 |
| indonesia | train | 149 | 37 | 44 | 230 |
| thrissur | dev | 119 | 30 | 37 | 186 |
| kyrgyzstan2 | train | 98 | 25 | 29 | 152 |
| hokkaido | train | 92 | 22 | 30 | 144 |
| itogon | train | 50 | 12 | 16 | 78 |
| china | train | 40 | 10 | 12 | 62 |
| usa_puertorico | train | 18 | 5 | 5 | 28 |
