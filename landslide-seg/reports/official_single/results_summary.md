# BENCH_OFFICIAL 单期主实验（SegFormer-LateDetail-OfficialSplit-SingleDate）

官方 S12LS-LD split 的**单期**变体（十波段、无 DEM、无多时相）。选模用官方 VAL 前景 pooled IoU@0.5；TEST 只评测、不参与任何选择。

## 1. 训练

| 项 | 值 |
|---|---|
| n_train / n_val | 3012 / 758 |
| 选模口径 | `dev_pooled_iou` |
| best 步 | 2700 |
| best VAL pooled IoU | 0.3623 |
| 训练耗时 | 687.0 s |
| 参数 | 3,753,602 |
| 训练峰值显存 | 4694 MiB |
| 曝光 | 230400 |

## 2. 曲线（每 100 更新）

| 步 | VAL 事件均值 | VAL pooled IoU | TRAIN pooled IoU |
|---:|---:|---:|---:|
| 500 | 0.2542 | 0.3053 | 0.3384 |
| 1200 | 0.2840 | 0.3424 | 0.4108 |
| 2400 | 0.2866 | 0.3546 | 0.4540 |
| 3600 | 0.2914 | 0.3604 | 0.4691 |

## 3.1 VAL：总体

| 检查点 | 事件均值 | 合并 IoU | Dice | Precision | Recall | AP | 边界 F1 | 对象召回 | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| best | 0.2933 | 0.3623 | 0.5319 | 0.5685 | 0.4996 | 0.5176 | 0.5282 | 0.4535 | 0.01444 |
| last | 0.2915 | 0.3604 | 0.5299 | 0.5566 | 0.5055 | 0.5112 | 0.5279 | 0.4545 | 0.01533 |

### 逐事件（val）

| 检查点 | 事件 | IoU | AP |
|---|---|---:|---:|
| best | chimanimani | 0.4456 | 0.6376 |
| best | china | 0.1087 | 0.1030 |
| best | dominicamaria | 0.1587 | 0.2131 |
| best | hiroshima | 0.4993 | 0.6840 |
| best | hokkaido | 0.5206 | 0.7423 |
| best | indonesia | 0.1866 | 0.2701 |
| best | italy | 0.3807 | 0.5499 |
| best | itogon | 0.2195 | 0.3219 |
| best | kyrgyzstan1 | 0.3334 | 0.4711 |
| best | kyrgyzstan2 | 0.0738 | 0.1021 |
| best | newzealand | 0.3658 | 0.5198 |
| best | thrissur | 0.5083 | 0.7202 |
| best | usa_puertorico | 0.0116 | 0.0171 |
| last | chimanimani | 0.4463 | 0.6390 |
| last | china | 0.1025 | 0.0895 |
| last | dominicamaria | 0.1625 | 0.2107 |
| last | hiroshima | 0.4987 | 0.6813 |
| last | hokkaido | 0.5255 | 0.7438 |
| last | indonesia | 0.1709 | 0.2437 |
| last | italy | 0.3816 | 0.5485 |
| last | itogon | 0.2113 | 0.3144 |
| last | kyrgyzstan1 | 0.3249 | 0.4558 |
| last | kyrgyzstan2 | 0.0779 | 0.1054 |
| last | newzealand | 0.3645 | 0.5096 |
| last | thrissur | 0.5086 | 0.7174 |
| last | usa_puertorico | 0.0134 | 0.0160 |

### 尺度召回（val，对象级）

| 检查点 | 1_4 px | 5_9 px | 10_19 px | 20_49 px | 50_199 px | ge200 px |
|---|---:|---:|---:|---:|---:|---:|
| best | 0.229 (68/297) | 0.358 (617/1723) | 0.486 (809/1663) | 0.490 (731/1491) | 0.573 (501/875) | 0.595 (72/121) |
| last | 0.236 (70/297) | 0.361 (622/1723) | 0.489 (813/1663) | 0.495 (738/1491) | 0.560 (490/875) | 0.587 (71/121) |

## 3.2 TEST：总体

| 检查点 | 事件均值 | 合并 IoU | Dice | Precision | Recall | AP | 边界 F1 | 对象召回 | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| best | 0.2860 | 0.3275 | 0.4935 | 0.5404 | 0.4541 | 0.4830 | 0.5139 | 0.4602 | 0.01436 |
| last | 0.2842 | 0.3265 | 0.4923 | 0.5279 | 0.4613 | 0.4786 | 0.5113 | 0.4648 | 0.01533 |

### 逐事件（test）

| 检查点 | 事件 | IoU | AP |
|---|---|---:|---:|
| best | chimanimani | 0.3945 | 0.5596 |
| best | china | 0.1233 | 0.0954 |
| best | dominicamaria | 0.1478 | 0.2456 |
| best | hiroshima | 0.4956 | 0.6948 |
| best | hokkaido | 0.4596 | 0.6940 |
| best | indonesia | 0.2650 | 0.3595 |
| best | italy | 0.3768 | 0.5481 |
| best | itogon | 0.2420 | 0.3147 |
| best | kyrgyzstan1 | 0.2496 | 0.3693 |
| best | kyrgyzstan2 | 0.0923 | 0.1258 |
| best | newzealand | 0.2970 | 0.4603 |
| best | thrissur | 0.4698 | 0.6996 |
| best | usa_puertorico | 0.1047 | 0.1756 |
| last | chimanimani | 0.3928 | 0.5534 |
| last | china | 0.1191 | 0.0938 |
| last | dominicamaria | 0.1567 | 0.2432 |
| last | hiroshima | 0.4937 | 0.6951 |
| last | hokkaido | 0.4713 | 0.6988 |
| last | indonesia | 0.2531 | 0.3479 |
| last | italy | 0.3771 | 0.5470 |
| last | itogon | 0.2300 | 0.3002 |
| last | kyrgyzstan1 | 0.2460 | 0.3624 |
| last | kyrgyzstan2 | 0.0977 | 0.1364 |
| last | newzealand | 0.2964 | 0.4516 |
| last | thrissur | 0.4709 | 0.7030 |
| last | usa_puertorico | 0.0895 | 0.1694 |

### 尺度召回（test，对象级）

| 检查点 | 1_4 px | 5_9 px | 10_19 px | 20_49 px | 50_199 px | ge200 px |
|---|---:|---:|---:|---:|---:|---:|
| best | 0.210 (86/409) | 0.377 (841/2233) | 0.503 (1054/2094) | 0.505 (971/1922) | 0.546 (594/1088) | 0.590 (85/144) |
| last | 0.218 (89/409) | 0.387 (864/2233) | 0.503 (1054/2094) | 0.509 (979/1922) | 0.549 (597/1088) | 0.583 (84/144) |

## 4. 口径说明

- 本表是**单期**变体，官方基线是多时相 + DEM，二者不是同一输入条件，差值不等于时间信息或 DEM 的贡献。
- 本项目主口径只统计 valid 像素；官方 torchmetrics 用全像素（见 `interface_validation.md` 的手算例）。与官方表值并排前需按同一口径复算。
- 训练预算是 3,600 成功更新，不等于官方的 75 epochs × 3 seeds。
- 官方 split 是随机划分，同事件跨 train/val/test，故此处的 VAL/TEST 不是跨事件外推。
