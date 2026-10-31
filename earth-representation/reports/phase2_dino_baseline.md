# Phase 2 Report — 轻量 CNN 可变波段编码器与 DINO 基线

生成时间：2026-10-31
项目根目录：`N:\Users\yzh\Disater\earth-representation`
GPU：RTX 5090 32 GB，torch 2.11.0+cu128

## 1. 执行范围

已完成：
- 共享单波段 CNN Stem、波段描述符嵌入、masked mean 聚合（另有轻量注意力实现，未作为默认）。
- ResNet-18 规格空间主干，输出 stride 4/8/16/32 的多尺度特征。
- DINO 教师—学生：EMA（参数与 BN 缓冲一起更新）、中心化、温度、projection head、2 global + 2 local 视图。
- 训练脚本、配置、日志与 checkpoint。
- 29 个单元测试（含可变波段、波段重排不变性、mask、金字塔尺寸）。

未完成：
- **没有可用的 DINO 基线**：在现有数据与预算下 loss 不能稳定下降（见第 4 节），因此不报告任何“表征质量”结论。
- 没有线性 probe / 分割 probe（计划放在后续阶段，本轮未做）。
- `attention` 聚合只实现未训练比较。

## 2. 实现

| 文件 | 内容 |
|---|---|
| `src/models/band_stem.py` | 单波段 1→width、/4 下采样；波段描述符 = (中心波长, log GSD) |
| `src/models/band_aggregator.py` | masked mean（默认）；descriptor attention（可选） |
| `src/models/resnet_encoder.py` | BasicBlock + ResNet-18 宽度配置，输出 s4/s8/s16/s32 |
| `src/models/unified_encoder.py` | 任意 C/H/W 输入 → 特征金字塔 + 全局向量 |
| `src/models/dino_head.py` | 3 层 MLP + weight-norm 原型层，推理时移除 |
| `src/ssl/ema_teacher.py` | EMA，缓冲区同步，动量封顶 0.999 |
| `src/ssl/multi_crop.py` | 2 global + 2 local 视图，含翻转与 90° 旋转 |
| `src/ssl/dino_loss.py` | 交叉熵 + teacher 中心化 + 温度锐化 |
| `src/ssl/train_dino.py` | 训练循环、LR warmup+cosine、日志、checkpoint |

模型规模（`configs/model_cnn.yaml`，width=32）：

| 部分 | 参数量 |
|---|---|
| 编码器 | 11,156,720 |
| DINO head | 562,304 |
| 合计 | 11,719,024 |

接口（推理时只需要影像与自动读取的产品元数据）：

```python
out = model(image, centre_nm, gsd_m)   # image [B,C,H,W], 描述符 [C] 或 [B,C]
out["global"]  # [B, 512]
out["s4"|"s8"|"s16"|"s32"]  # 多尺度特征图
```

## 3. 训练设置

- 数据：Phase 1 落地的 npz 补丁（本轮开始时 82 个，含 S2 10 波段 / Landsat 7 波段 / HLS 6 波段）。
- 按产品分组组 batch（不同产品波段数不同，不能混批）。
- batch 8，lr 1e-3，warmup 10% + cosine，weight decay 0.04，out_dim 64，teacher_temp 0.07，student_temp 0.1，center momentum 0.99，EMA 0.99（封顶，不升到 1.0），前 20 步冻结原型层。
- 300 步，单卡，约 1.4–1.5 s/步。

运行命令：

```bash
python -m src.ssl.train_dino --config configs/model_cnn.yaml --steps 300 --tag tune9
```

## 4. 结果

300 步的 loss 轨迹（`outputs/dino/tune9.jsonl`）：

| step | loss |
|---|---|
| 1 | 4.053 |
| 5 | 3.825 |
| 10 | 3.785 |
| 100 | ~3.87 |
| 200 | ~3.96 |
| 300 | 3.995 |

对照：均匀分布的理论值 ln(64) = 4.159。

可见：
1. 前 ~10 步 loss 从 4.05 降到 3.78，之后缓慢回升到 3.99。
2. 全程低于 ln(64)，说明预测没有完全塌到均匀，但**没有持续下降**。
3. 早期下降更像是 student 与初始 teacher 对齐，随教师漂移而回升；不构成“表征在变好”的证据。

未发现数值异常：无 NaN/Inf，梯度裁剪生效，teacher 与 student 参数同步更新，checkpoint 可加载。

## 5. 失败与风险

1. **DINO 基线不成立（未达成验收）**。计划要求“loss 稳定下降、无明显 collapse”，本轮 300 步内没有满足。原因是数据量（82 个补丁、batch 8）与步数远低于 DINO 的常规规模，不是代码报错；但没有证据就不能宣称基线可用。
2. **早期版本有三个真实 bug，已修**：
   - 损失未按 batch 归一化，batch 从 2 变 4 时 loss 正好翻倍（9.76→19.5），会误导“loss 爆炸”的判断。
   - `update_center` 只对视图维求均值，把 batch 维留在了中心向量里，下一个不同 batch 大小就广播失败。
   - EMA 动量升到 1.0，教师被冻结而学生继续更新，loss 必然回升。
3. **teacher BN 统计**：教师以 eval 模式运行，若不同步 BN 缓冲，它会一直持有随机初始化时的统计量。已改为缓冲一起 EMA。
4. **超参数未做系统搜索**：当前值只是“能让 loss 有限且教师跟得上学生”，不应作为调参结论。
5. `attention` 聚合与 `masked mean` 的对照未运行（计划列为可选消融）。

## 6. 下一阶段建议

要得到可用的 DINO 基线，需要按量级补三件事，按优先级：

1. **数据**：从 82 个补丁提到 ≥2000 个（15 区域 × 3 产品 × 40+，或加入多时相），并让每个产品的样本量可比。
2. **步数**：≥10k 步（batch 64–128），当前 300 步只够验证管线。
3. **评估**：加一个冻结特征的线性 probe（哪怕先用“产品/区域分类”作代理），否则无法判断 loss 变化是否对应表征变化。

在这些满足之前，不建议进入 Phase 3 的“物理增强优于普通增强”结论比较——会用噪声当信号。

## 7. Round 2 追加：塌缩定位与三次修复（仍未收敛）

在初版基础上做了四项针对性修改，均以实测为准：

| 改动 | 动机 | 结果 |
|---|---|---|
| 原型层排除 weight decay | 衰减把 weight_norm 的 scale 压向 0，所有原型趋同 | proto_std 从 0 变为缓慢上升（0.007→0.015），损失不再固定在 ln(D) |
| head 输入加 LayerNorm | 编码器全局特征方差仅 5e-3，中心化后几乎无信号 | 无明显改善，仍塌缩 |
| 批次跨产品混合（补齐到 10 波段 + mask） | 同批同景导致批内差异近 0 | proto_std 稳定在 0.085，但 600 步后仍降到 0.012 |
| 训练加速（整批一次前向） | 逐样本前向 16 次/步 | 2.5 s/步 → 0.3 s/步，8 倍 |

新证据（outputs/dino/mix1500.jsonl、.tmp/diag_encoder.py）：

- 初始化时，8 个**不同区域**的 S2 patch 经编码器后全局特征余弦相似度 0.9929–0.9982，跨样本标准差 0.0052；而跨原型标准差 0.069。即：编码器对输入几乎不敏感，中心化一减就只剩噪声。
- 监督探针（同产品内分类 12 个区域，batch 16 混批）准确率 20 步 0.44 → 60 步 1.00（随机 0.08）。**编码器可训练、特征可分**，问题不在主干。
- 1500 步混合批次训练：proto_std 0.085 → 0.012，loss 回升到 ln(64)=4.14，仍属塌缩。

结论：在 300 patch、batch 8 的规模下，DINO 的教师—学生目标会稳定走向均匀解；这不是某一处实现 bug（三处 bug 已修，且监督探针能学到 1.0）。要得到可用基线，需要数量级更大的数据与步数，或换用对 batch 更友好的目标。

## 8. Round 3 追加：定位到 head 与教师输出，仍未解决

先纠正上一轮的一个错误结论：**“编码器对内容不敏感”是测量伪影**。当时用 .eval() 且把同一张图复制成一批，BN 批统计退化。用真实 8 张不同 patch、train 模式重测：全局特征跨样本 std 0.169，平均成对余弦 0.979——编码器是敏感的。此结论已从报告正文中撤回。

本轮四项尝试：

| 改动 | 结果 |
|---|---|
| 描述符从加法偏置改为 near-identity FiLM（原偏置幅度是内容的 17 倍） | 消除一个真实缺陷，但未改变塌缩 |
| 教师 BN 预热（前 20 步 train 模式，再 eval） | proto_std 不再归零，但 sample_std 仍从 0.05 掉到 0.002 |
| batch 32（跨产品混批后可行） | sample_std 仍掉到 5e-4 |
| head 单独降学习率（0.05×） | 无效，sample_std 仍掉到 0.0016 |

关键测量（加载 outputs/dino/batch32.pt）：

| 对象 | 模式 | 特征跨样本 std | logits 跨样本 std | logits 跨原型 std |
|---|---|---|---|---|
| student | eval | 0.181 | 0.000009 | 0.061 |
| student | train | 0.397 | 0.0059 | 0.058 |
| teacher | eval | 0.182 | 0.000005 | 0.030 |

即：**特征在变，head 的 logits 不变**。head 在训练中塌成近似秩 1，任何输入都映射到同一个归一化向量；教师目标因此与输入无关，DINO 退化成“拟合一个固定分布”，损失只能停在 ln(D) 附近。

## 9. 结论（与计划验收对照）

Phase 2 验收中的“同一模型可前向处理多种波段”“波段重排一致性”“推理只需影像+元数据”均满足并有测试；**“DINO loss 稳定下降、无 collapse”不满足**，且已排除编码器、BN、EMA、中心化、损失归一化、批次同质性、head 学习率等因素。

继续需要二者之一：更大的数据与算力（≥2000 patch、≥10k 步、batch 32–64，约数小时），或改用对小 batch 更稳健的目标。两者都超出当前授权，需用户决定。

## 10. Round 4：改用官方实现，塌缩消失（根因确认在自写 loss/head）

按“已有官方实现就不要自写”的原则，改为直接使用 facebookresearch/dino 的代码，放在 `vendor/dino/`（`main_dino.py`、`vision_transformer.py`、`utils.py`，从 GitHub 原始文件获取）。只保留本项目特有的部分：可变波段编码器、三源数据加载、物理增强。

| 组件 | 来源 |
|---|---|
| DINOHead | 官方 `vision_transformer.DINOHead` |
| DINOLoss（含 teacher 温度 warmup、center EMA） | 官方 `main_dino.DINOLoss` |
| MultiCropWrapper、cosine_scheduler、clip_gradients、cancel_gradients_last_layer | 官方 `utils.py` |
| 编码器 / 数据 / 增强 | 本项目 |

官方 loss 的 `update_center` 无条件调用 `dist.all_reduce`；本机 Windows 上初始化 gloo 进程组会直接段错误（0xC0000005），因此把这两个集合通信替换为单进程等价物，loss 数学本身不改。

结果对比（同为 512 patch、batch 32）：

| 实现 | 300 步内的 loss 形状 | 结论 |
|---|---|---|
| 自写 loss/head | 精确停在 ln(64)=4.1589 不动 | 均匀塌缩 |
| 官方 loss/head | 先升到 5.44（≈ln256），再回落到 4.5 | 正常 DINO 曲线 |

长跑（3000 步，2048 patch）到第 575 步时 loss 已降到 3.30–3.52，明显低于 ln(256)=5.545，且与自写实现的“钉死”行为完全不同。

结论：此前所有关于“数据不够 / 主干塌缩”的判断都被推翻；问题出在自写的 DINO loss/head 上，官方实现直接解决了。自写版本保留在 `src/ssl/` 中仅供对照，后续实验一律走 `scripts/train_dino_official.py`。
