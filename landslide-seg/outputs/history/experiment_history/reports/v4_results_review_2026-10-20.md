# v4 系列实验审阅（自动化核对）

数据源：`outputs/segformer/s2_10band_v4*/**/train_summary.json` + 三批结果 CSV + 冻结清单。只读核对，不重算像素。

## 1. run 元数据

| run | n_train | n_val | steps | epochs | best_epoch | best_dev | 秒 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `outputs/segformer/s2_10band_v4_p20/SMOKE2_POOL25/1234` | 559 | 448 | 360 | 40 | 34 | 0.2573 | 58 |
| `outputs/segformer/s2_10band_v4_p20/SMOKE_POOL25/1234` | 559 | 448 | 54 | 6 | 0 | 0.0000 | 12 |
| `outputs/segformer/s2_10band_v4r1_p20/segformer_b0_direct10_detailres_n025_u3600_s1234/1234` | 559 | 669 | 3600 | None | None | 0.2870 | 472 |
| `outputs/segformer/s2_10band_v4r1_p20/segformer_b0_direct10_detailres_n050_u3600_s1234/1234` | 1113 | 669 | 3600 | None | None | 0.3112 | 588 |
| `outputs/segformer/s2_10band_v4r1_p20/segformer_b0_direct10_detailres_n075_u3600_s1234/1234` | 1671 | 669 | 3600 | None | None | 0.3143 | 671 |
| `outputs/segformer/s2_10band_v4r1_p20/segformer_b0_direct10_detailres_n100_u3600_s1234/1234` | 2223 | 669 | 3600 | None | None | 0.3081 | 738 |
| `outputs/segformer/s2_10band_v4r1_p20/segformer_b0_direct10_n025_u3600_s1234/1234` | 559 | 669 | 3600 | None | None | 0.2813 | 434 |
| `outputs/segformer/s2_10band_v4r1_p20/segformer_b0_direct10_n100_u3600_s1234/1234` | 2223 | 669 | 3600 | None | None | 0.2909 | 684 |
| `outputs/segformer/s2_10band_v4r1_p20/validation_direct10_detailres_u13/1234` | 559 | 669 | 13 | None | None | 0.0000 | 19 |
| `outputs/segformer/s2_10band_v4r2_probe/segformer_b0_direct10_frozen_best_decodeprobe_n100_u3600_s1234/1234` | 2223 | 669 | 3600 | None | None | 0.2731 | 490 |
| `outputs/segformer/s2_10band_v4r2_probe/segformer_b0_direct10_frozen_last_decodeprobe_n100_u3600_s1234/1234` | 2223 | 669 | 3600 | None | None | 0.2463 | 513 |
| `outputs/segformer/s2_10band_v4r2_probe/validation_frozen_best_decodeprobe_u13/1234` | 2223 | 669 | 13 | None | None | 0.0001 | 45 |
| `outputs/segformer/s2_10band_v4r3_dofa/dofa_base_upernet_pyramid_binary_n100_u3600_s1234/1234` | 2223 | 669 | 3600 | None | None | 0.1969 | 1113 |
| `outputs/segformer/s2_10band_v4r3_low_encoder_lr/segformer_b0_direct10_detailres_pretrained_encoder_low_lr_n100_u3600_s1234/1234` | 2223 | 669 | 3600 | None | None | 0.2823 | 696 |
| `outputs/segformer/s2_10band_v4r3_low_encoder_lr/validation_detailres_low_encoder_lr_u3/1234` | 2223 | 669 | 3 | None | None | 0.0008 | 35 |

## 2. 结果自洽性

- 参与核对的记录：**20** 条；`dev_mean = (hiroshima + thrissur)/2` 不成立：**0** 条。
- 已评测 FINAL/Italy 的记录：**0** 条。

## 3. 训练池与冻结池一致性

| 清单 | train 图块 | 不在冻结池 | 与冻结池交集 |
|---|---:|---:|---:|
| `pool_100.jsonl` | 2223 | 0 | 2223 |
| `pool_25.jsonl` | 559 | 0 | 559 |
| `pool_50.jsonl` | 1113 | 0 | 1113 |
| `pool_75.jsonl` | 1671 | 0 | 1671 |

## 4. 跨批次对照

| 批次 | 条件 | ckpt | 步 | TRAIN IoU | DEV 均值 | Hiroshima | Thrissur | AP | 对象召回 | 1–4px |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v4r1 | segformer_b0_direct10_detailres_n025_u3600_s1234 |  | 200 | 0.3074 | - | 0.2633 | 0.3108 | 0.3751 | 0.5051 | 14/77 |
| v4r1 | segformer_b0_direct10_detailres_n100_u3600_s1234 |  | 500 | 0.3681 | - | 0.3114 | 0.3049 | 0.4425 | 0.5211 | 8/77 |
| v4r1 | segformer_b0_direct10_detailres_n050_u3600_s1234 |  | 400 | 0.3763 | - | 0.3095 | 0.3128 | 0.4403 | 0.5515 | 10/77 |
| v4r1 | segformer_b0_direct10_detailres_n075_u3600_s1234 |  | 400 | 0.3604 | - | 0.3271 | 0.3014 | 0.4614 | 0.4818 | 8/77 |
| v4r1 | segformer_b0_direct10_n025_u3600_s1234 |  | 800 | 0.4521 | - | 0.2575 | 0.3051 | 0.3593 | 0.3812 | 2/77 |
| v4r1 | segformer_b0_direct10_n100_u3600_s1234 |  | 1200 | 0.4039 | - | 0.2621 | 0.3196 | 0.3958 | 0.3693 | 5/77 |
| v4r2 | joint_finetune | best | 1200 | 0.4039 | 0.2909 | 0.2621 | 0.3196 | 0.3958 | 0.3693 | 5/77 |
| v4r2 | frozen_best | best | 1100 | 0.4109 | 0.2731 | 0.2479 | 0.2983 | 0.3694 | 0.3597 | 6/77 |
| v4r2 | frozen_best | last | 3600 | 0.4176 | 0.2591 | 0.2384 | 0.2797 | 0.3539 | 0.3311 | 3/77 |
| v4r2 | frozen_last | best | 600 | 0.4584 | 0.2463 | 0.2318 | 0.2607 | 0.3450 | 0.3073 | 1/77 |
| v4r2 | joint_finetune | last | 3600 | 0.4609 | 0.2415 | 0.2262 | 0.2567 | 0.3419 | 0.2865 | 3/77 |
| v4r2 | frozen_last | last | 3600 | 0.4658 | 0.2337 | 0.2203 | 0.2471 | 0.3272 | 0.2871 | 2/77 |
| v4r3 | segformer_detail | best | 500 | 0.3681 | 0.3081 | 0.3114 | 0.3049 | 0.4425 | 0.5211 | 8/77 |
| v4r3 | segformer_standard | best | 1200 | 0.4039 | 0.2909 | 0.2621 | 0.3196 | 0.3958 | 0.3693 | 5/77 |
| v4r3 | segformer_detail_low_encoder_lr | best | 2300 | 0.3929 | 0.2823 | 0.2638 | 0.3009 | 0.3852 | 0.4616 | 9/77 |
| v4r3 | segformer_detail_low_encoder_lr | last | 3600 | 0.4049 | 0.2706 | 0.2565 | 0.2846 | 0.3762 | 0.4217 | 8/77 |
| v4r3 | segformer_detail | last | 3600 | 0.4895 | 0.2430 | 0.2513 | 0.2346 | 0.3619 | 0.3401 | 4/77 |
| v4r3 | segformer_standard | last | 3600 | 0.4609 | 0.2415 | 0.2262 | 0.2567 | 0.3419 | 0.2865 | 3/77 |
| v4r3 | dofa_upernet_pyramid | best | 500 | 0.3456 | 0.1969 | 0.1785 | 0.2154 | 0.2691 | 0.2287 | 1/77 |
| v4r3 | dofa_upernet_pyramid | last | 3600 | 0.4882 | 0.1100 | 0.0809 | 0.1390 | 0.1351 | 0.1042 | 1/77 |
