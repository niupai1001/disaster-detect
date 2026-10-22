# P1-3 事件覆盖对照：两组训练集的组成

来源 `manifests/v4r1/pool_100.jsonl`（sha `b51422e2e8a2cd49…`）；选择规则只依赖 TRAIN 的 GT 属性，未使用 DEV 或任何模型输出。

N = **1251**；FewEvents = ['dominicamaria', 'indonesia', 'kyrgyzstan1']（3 个事件）；AllEvents = 九个事件按可用量比例分摊同一 N。

## 1. 总体组成

| 项 | FewEvents | AllEvents | 共同核心 |
|---|---:|---:|---:|
| 图块数 | 1251 | 1251 | 703 |
| 事件数 | 3 | 9 | 3 |
| 前景像素占比 | 0.03834 | 0.03891 | 0.03696 |
| GT 对象数 | 9535 | 8783 | 5072 |
| 平均对象面积(px) | 69.15 | 76.90 | 70.31 |

## 2. 对象面积档分布（对象数占比）

| 面积档 | FewEvents | AllEvents | 差值 |
|---|---:|---:|---:|
| 1_4 | 0.1273 | 0.1055 | +0.0218 |
| 5_9 | 0.1668 | 0.1193 | +0.0474 |
| 10_19 | 0.1853 | 0.1531 | +0.0322 |
| 20_49 | 0.2675 | 0.2914 | -0.0238 |
| 50_199 | 0.1929 | 0.2585 | -0.0656 |
| ge200 | 0.0602 | 0.0722 | -0.0120 |

TVD(面积档) = 0.1014；相对前景比差 = 0.0148；合计 score = 0.1162

## 3. 每事件配额

| 事件 | 可用 | FewEvents | AllEvents | 全量占比 |
|---|---:|---:|---:|---:|
| kyrgyzstan1 | 642 | 642 | 361 | 0.2888 |
| chimanimani | 406 | 0 | 229 | 0.1826 |
| dominicamaria | 374 | 374 | 210 | 0.1682 |
| indonesia | 235 | 235 | 132 | 0.1057 |
| kyrgyzstan2 | 181 | 0 | 102 | 0.0814 |
| hokkaido | 150 | 0 | 84 | 0.0675 |
| itogon | 119 | 0 | 67 | 0.0535 |
| china | 72 | 0 | 41 | 0.0324 |
| usa_puertorico | 44 | 0 | 25 | 0.0198 |

共同核心（两组都含）703 块 = FewEvents 的 56.2%、AllEvents 的 56.2%。

## 4. 全部候选（按 score 升序，前 20）

| k | 事件 | N | score | TVD | 相对前景 |
|---:|---|---:|---:|---:|---:|
| 3 | dominicamaria, indonesia, kyrgyzstan1 | 1251 | 0.1162 | 0.1014 | 0.0148 |
| 3 | chimanimani, dominicamaria, kyrgyzstan1 | 1422 | 0.1230 | 0.0825 | 0.0405 |
| 4 | china, dominicamaria, indonesia, kyrgyzstan1 | 1323 | 0.1423 | 0.0990 | 0.0433 |
| 4 | chimanimani, dominicamaria, kyrgyzstan1, usa_puertorico | 1466 | 0.1425 | 0.0839 | 0.0585 |
| 4 | dominicamaria, indonesia, kyrgyzstan1, usa_puertorico | 1295 | 0.1445 | 0.1054 | 0.0392 |
| 3 | china, dominicamaria, kyrgyzstan1 | 1088 | 0.1485 | 0.1454 | 0.0031 |
| 4 | chimanimani, china, dominicamaria, kyrgyzstan1 | 1494 | 0.1488 | 0.0779 | 0.0708 |
| 4 | chimanimani, dominicamaria, indonesia, kyrgyzstan1 | 1657 | 0.1548 | 0.0559 | 0.0989 |
| 3 | dominicamaria, kyrgyzstan1, usa_puertorico | 1060 | 0.1625 | 0.1527 | 0.0099 |
| 4 | dominicamaria, indonesia, itogon, kyrgyzstan1 | 1370 | 0.1675 | 0.0988 | 0.0687 |
| 4 | dominicamaria, indonesia, kyrgyzstan1, kyrgyzstan2 | 1432 | 0.1740 | 0.0865 | 0.0874 |
| 2 | dominicamaria, kyrgyzstan1 | 1016 | 0.1853 | 0.1482 | 0.0371 |
| 4 | china, dominicamaria, kyrgyzstan1, usa_puertorico | 1132 | 0.1858 | 0.1433 | 0.0425 |
| 3 | dominicamaria, itogon, kyrgyzstan1 | 1135 | 0.1878 | 0.1423 | 0.0455 |
| 4 | chimanimani, dominicamaria, itogon, kyrgyzstan1 | 1541 | 0.1888 | 0.0805 | 0.1083 |
| 3 | dominicamaria, kyrgyzstan1, kyrgyzstan2 | 1197 | 0.1970 | 0.1301 | 0.0670 |
| 4 | chimanimani, dominicamaria, kyrgyzstan1, kyrgyzstan2 | 1603 | 0.2033 | 0.0712 | 0.1321 |
| 4 | chimanimani, dominicamaria, hokkaido, kyrgyzstan2 | 1111 | 0.2115 | 0.0740 | 0.1375 |
| 3 | china, kyrgyzstan1, usa_puertorico | 758 | 0.2145 | 0.1978 | 0.0167 |
| 2 | itogon, kyrgyzstan1 | 761 | 0.2146 | 0.1847 | 0.0299 |
