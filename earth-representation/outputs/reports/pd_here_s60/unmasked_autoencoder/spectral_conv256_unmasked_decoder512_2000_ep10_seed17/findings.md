# 无遮盖 Autoencoder · Decoder 128 与 512 维对照

两组使用相同的 2,000 TRAIN、272 个可评分留出父观测、seed17、batch32 和 630 次 AdamW 更新。融合仍为 256 维。Decoder 保持两层、每层四头，输入投影、位置编码、可用性投影、Transformer 和原生布局输出头的输入宽度一并扩到 512。两组均从头训练，同形状参数使用相同初值。损失重建全部 QA 有效原生像元，按像元、波段、父观测等权归约。成员哈希和固定预测样本逐一相同。

## 训练和像元误差

| 指标 | Decoder 128 | Decoder 512 |
| --- | ---: | ---: |
| 参数量 | 47,308,672 | 181,242,496 |
| 原生线性输出头参数 | 42,960,096 | 170,841,312 |
| 训练与逐轮验证 | 133.38 s | 156.01 s |
| 轮耗时中位数 | 12.82 s | 15.34 s |
| 装载到训练完成 | 182.30 s | 228.21 s |
| PyTorch 峰值分配 | 2.10 GB | 3.87 GB |
| best 轮次 | 9 | 7 |
| best TRAIN MSE | 0.00269143 | 0.00264546 |
| best VAL MSE | 0.00355696 | 0.00313862 |
| 第 10 轮 TRAIN MSE | 0.00254864 | 0.00247814 |
| 第 10 轮 VAL MSE | 0.00376752 | 0.00332139 |

512 维组的真实尾批合批用时 0.489 秒，驻留逐景用时 0.687 秒。两种执行下融合层梯度余弦为 0.999997。两组均用驻留合批。原生输出头增参占总增参 95.5%。

## 第 10 轮原生像元和纹理

| 留出指标 | Decoder 128 | Decoder 512 |
| --- | ---: | ---: |
| 像元 MSE，逐景复算 | 0.00376565 | 0.00332128 |
| Bias² | 0.00251196 | 0.00208826 |
| CenteredMSE | 0.00125369 | 0.00123301 |
| 块内去均值 MSE | 0.00106504 | 0.00108765 |
| 原生相邻梯度 MSE | 0.00042886 | 0.00042999 |
| 块内梯度 MSE | 0.00040172 | 0.00041476 |
| 跨块边界梯度 MSE | 0.00064778 | 0.00052830 |
| 原生梯度相关 | 0.11785 | 0.08672 |
| 共同模板能量比 | 51.91% | 39.61% |

逐块真值均值基线的像元和梯度 MSE 分别为 0.00098193 和 0.00040424。512 维组的像元 MSE 在 201/272 景下降，原生梯度 MSE 在 125/272 景下降，梯度相关在 107/272 景上升。全域像元 MSE 的降幅约 95% 来自 Bias² 下降。共同模板在 106/135 景减少，真值平均占比为 0.83%。跨块边界梯度误差下降 18.4%，块内梯度误差上升 3.2%。

按各自 best 比较，512 维组的原生梯度 MSE 为 0.00043127，对照组为 0.00041943。扩宽改善像元均值和重复模板占比，但当前同预算下仍没有恢复更多原生细节。固定 Landsat 8 样例的红波段 MSE 从 0.00137 降到 0.000653，在同一真值色标下仍接近纯黑，梯度相关从 0.039 降到 0.033。

| 固定留出分层 | 景数 | VAL MSE 128 → 512 | 原生梯度 MSE 128 → 512 | 梯度相关 128 → 512 |
| --- | ---: | ---: | ---: | ---: |
| Sentinel‑2 MSI | 177 | 0.004310 → 0.003930 | 0.000479 → 0.000496 | 0.088 → 0.047 |
| Landsat 8 OLI | 43 | 0.002506 → 0.001890 | 0.000232 → 0.000221 | 0.195 → 0.173 |
| Landsat 9 OLI‑2 | 10 | 0.003917 → 0.002651 | 0.000372 → 0.000358 | 0.197 → 0.179 |

三个未支持留出布局的传感器继续使用标注为 TRAIN 的固定样例绘图。所有传感器的逐波段指标、有效像元计数和 GSD 保存在原生评价记录。

[同预算训练曲线](width_comparison_curves.png) · [第 10 轮核心卫星图](predictions/last_epoch_010/core_overview.png) · [第 10 轮全卫星图](predictions/last_epoch_010/overview.png) · [第 7 轮 best 图](predictions/best_epoch_007/overview.png) · [逐景原生指标](predictions/last_epoch_010/native_metrics.json) · [128 维报告](../spectral_conv256_unmasked_2000_ep10_seed17/findings.md) · [加速预检](preflight.json)

## 复现

```powershell
../landslide-seg/.venv-use-independent/Scripts/python.exe scripts/train_pd_here_s60_autoencoder.py --run spectral_conv256_unmasked_decoder512_2000_ep10_repro --device cuda:1 --epochs 10 --batch 32 --decoder-width 512
../landslide-seg/.venv-use-independent/Scripts/python.exe scripts/evaluate_pd_here_s60_autoencoder.py --run spectral_conv256_unmasked_decoder512_2000_ep10_repro --device cuda:1 --checkpoint last
```

同一 Run 恢复时在训练命令后追加 `--resume`。
