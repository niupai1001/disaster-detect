# 2,000 景无遮盖 Autoencoder 对照

三组使用同一 2,000 TRAIN、272 景共同可评分 VAL、batch32、seed17 和 630 次更新。0% 遮盖，目标为全部 QA 有效原生像元。预测保持各波段原生网格。

| 模型 | best 轮次 | best VAL MSE | 第 10 轮 VAL MSE | 原生梯度相关 | 共同模板能量 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 120 米光谱 Transformer＋空间卷积分组加权，Decoder 128 | 9 | 0.003557 | 0.003768 | 0.118 | 51.9% |
| 120 米光谱 Transformer＋空间卷积分组加权，Decoder 512 | 7 | 0.003139 | 0.003321 | 0.087 | 39.6% |
| 60M地理Patch-单空间ViT，Decoder 128 | 10 | 0.001292 | 0.001292 | 0.222 | 3.82% |

纹理指标统一使用物理 120 米块，均取第 10 轮。512 维卷积双分支的像元 MSE 降幅约 95% 来自均值偏差减小。60 米空间 ViT 同时改善像元 MSE、跨块边界误差和重复模板。Sentinel‑2 仍有块状误差。架构对照同时改变 Patch 尺寸与 Encoder。

- [三组训练曲线](spatial60_unmasked_2000_ep10_seed17/architecture_comparison_curves.png)
- [60 米空间 ViT · 报告](spatial60_unmasked_2000_ep10_seed17/findings.md) · [核心卫星图](spatial60_unmasked_2000_ep10_seed17/predictions/last_epoch_010/core_overview.png) · [全卫星总览](spatial60_unmasked_2000_ep10_seed17/predictions/last_epoch_010/overview.png)
- [120 米 Decoder 128 · 报告](spectral_conv256_unmasked_2000_ep10_seed17/findings.md) · [第 9 轮 best 图](spectral_conv256_unmasked_2000_ep10_seed17/predictions/best_epoch_009/overview.png)
- [120 米 Decoder 512 · 报告](spectral_conv256_unmasked_decoder512_2000_ep10_seed17/findings.md) · [第 7 轮 best 图](spectral_conv256_unmasked_decoder512_2000_ep10_seed17/predictions/best_epoch_007/overview.png)
- [冻结 TRAIN 清单](../../../pd_here_s60/sensor_gsd_validation_seed17/coverage_qc_revision/screened_train2000_candidate.json) · [冻结 VAL 清单](../../../pd_here_s60/sensor_gsd_validation_seed17/coverage_qc_revision/screened_val275_candidate.json) · [本地权重校验清单](checkpoint_inventory.json)

训练依赖项目本地 Core v4 和扩展 v6 的冻结发布及原生缓存，发布哈希、配置和逐轮状态见各运行目录。权重存放于本地，校验清单记录 best 与 last 各自的 SHA-256。

[训练入口](../../../../scripts/train_pd_here_s60_autoencoder.py) · [原生评价入口](../../../../scripts/evaluate_pd_here_s60_autoencoder.py)
