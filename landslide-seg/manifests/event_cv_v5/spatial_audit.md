# EVENT_CV_V5 空间关系审计

- 图块 4715，事件 13
- 邻接判据：轴对齐间隙 ≤ 10 m；图块跨度 1280 m
- 同 CRS 内为精确 bbox 判定；跨 CRS 只能给中心点测地距离下界（<2 km 需人工确认）

## 需要关注的事件对

| 事件 A | 事件 B | 同 CRS | 重叠对数 | 10m 邻接对数 | 同CRS最小间隙(m) | 最小中心距(km) |
|---|---|---|---:|---:|---:|---:|
| kyrgyzstan1 | kyrgyzstan2 | 是 | 0 | 1 | 0.0 | 1.28 |

## 全部事件对（按最小中心距升序，前 12）

| 事件 A | 事件 B | 重叠 | 邻接 | 最小中心距(km) |
|---|---|---:|---:|---:|
| kyrgyzstan1 | kyrgyzstan2 | 0 | 1 | 1.28 |
| dominicamaria | usa_puertorico | 0 | 0 | 616.96 |
| hiroshima | hokkaido | 0 | 0 | 1196.12 |
| indonesia | itogon | 0 | 0 | 1932.95 |
| hiroshima | itogon | 0 | 0 | 2281.24 |
| china | itogon | 0 | 0 | 2356.66 |
| china | hiroshima | 0 | 0 | 2714.68 |
| china | kyrgyzstan2 | 0 | 0 | 2833.66 |
| china | kyrgyzstan1 | 0 | 0 | 2838.62 |
| kyrgyzstan2 | thrissur | 0 | 0 | 3309.36 |
| kyrgyzstan1 | thrissur | 0 | 0 | 3337.01 |
| hokkaido | itogon | 0 | 0 | 3541.16 |
