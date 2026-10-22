# SegFormer 公共训练协议（已冻结）

> 依据：`docs/segformer_execution_plan_2026-10-18_v2.md` §7，加上本轮 P2/P3/P4 实测。
> 冻结时间：2026-10-18，正式四组开跑之前。**冻结后四组不得单独延长训练或改公共项**。

## 1. 冻结表

| 项 | 值 | 依据 |
|---|---|---|
| 数据版本 | `s2_10band_v1` | `docs/segformer_input_contract.md` |
| 划分 | train 1,277 / val 448 / test 382 | 事件级互斥，bbox 重叠 0 |
| 模型 | `SegFormerForSemanticSegmentation`（MiT-B0 + 官方解码器），替换 150 类分类器为单 logit | 官方权重加载无缺失/多余键 |
| 输入适配 | A0/A1 `direct10`；B0/B1 `project10to3` | 初始化等价已测试 |
| Stem | A1/B1 开启（32 通道、stride=1、残差末层零初始化） | ΔL 初值严格为 0 |
| batch | **64** | 实测峰值：A1/B1 4.7 GB、B0 1.7 GB（31.8 GB 卡） |
| epochs | **150**（20 步/epoch → 3,000 次更新） | 见 §2 的预算换算 |
| 精度 | fp32（AMP 关闭） | bs=64 时 bf16 只快约 2%，不换取数值风险 |
| 优化器 | AdamW，wd=0.01 | 计划 §7 |
| 学习率 | 编码器 6e-5；解码器+新增层 6e-4 | 同上；分组由 `SegFormerSeg.param_groups()` 给出 |
| 调度 | 5% warmup + polynomial(power=1.0) | 同上 |
| 梯度裁剪 | 1.0 | 同上 |
| 损失 | `0.5 × masked BCEWithLogits + 0.5 × masked SoftDice` | 复用 `training/losses.py` |
| 全无效 batch | 跳过并计数 | 本数据集实测未触发 |
| 增强 | 水平/垂直翻转 + 90° 旋转（k∈0..3），影像/mask/valid 同步 | 随机流由 (seed, epoch, index) 决定 |
| 数据加载 | `preload: true`（int16 张量常驻，约 0.6 GB） | 每 epoch 重复解压是主要开销 |
| 主阈值 | 0.5 | 扫描只在 val 上做，单列报告 |
| 模型选择 | 验证集前景 IoU 最大的 epoch（best.pt），另存 last.pt | 测试集不参与任何选择 |
| 种子 | 1234 / 2026 / 3407 | 计划 §7 |

**预算换算**：v1 的"100 epoch × batch 8"= 14,700 次更新；batch 64 时 100 epoch 只有 1,900 步。
本协议按**更新步数**定预算：150 epoch × ceil(1,277/64)=20 步 = **3,000 步**，与 v1 同量级偏保守。

## 2. 实测证据

### 2.1 吞吐与显存（真实数据，batch 64，fp32）

| 组 | 秒/epoch | 150 epoch 预计 | 训练峰值 allocated | 峰值 reserved |
|---|---:|---:|---:|---:|
| A0 direct10 | 1.56 | ≈4.0 min | — | — |
| A1 direct10 + stem | 2.52 | ≈6.3 min | 4,690 MiB | 7,350 MiB |
| B0 project10to3 | 1.83 | ≈4.6 min | 1,733 MiB | 2,182 MiB |
| B1 project10to3 + stem | 2.61 | ≈6.5 min | 4,702 MiB | 7,366 MiB |

12 次运行串行约 1 小时；两卡并行可减半（本轮正式结果全部在 `cuda:0` 上串行完成，保证配对可比）。

### 2.2 过拟合诊断（P2，A0，8 瓦片，500 步）

| 指标 | 值 |
|---|---|
| loss | 0.800 → 0.086（下降 89.3%） |
| 训练瓦片平均 IoU | 0.812（500 步） |
| 每个瓦片都有预测正像素 | 是 |
| 耗时 | 13.9 s |

产物：`outputs/segformer/_overfit/A0/report.json`、`overfit.png`。

### 2.3 先导训练（P3/P4）

| 组 | epochs | best val IoU | 说明 |
|---|---:|---:|---|
| A0 | 30 | 0.2390 @ 25 | 曲线在 0.23–0.24 平台，无发散 |
| A1 | 10 | 0.2326 | stem 梯度正常，显存 4.7 GB |
| B0 | 10 | 0.1737 | 投影非 RGB 系数可训练 |
| B1 | 10 | 0.2668 | 收敛最快的一组（短预算） |

> 先导结论只用于"链路可用 / 显存与步时符合预期"，不作为最终比较；
> 正式比较一律使用 150 epoch 的同预算结果。

## 3. 产物与复现

每次 run 写到 `outputs/segformer/<data_version>/<experiment>/<seed>/`：

```
resolved_config.json   # 实际生效的完整配置
run_meta.json          # 清单/统计/权重身份、seed、设备、样本 ID、步数预算
metrics.jsonl          # 逐 epoch：loss / val IoU、Dice、P、R / lr / 耗时
best.pt / last.pt      # 含 model/optimizer/scheduler/RNG/config/meta
train_summary.json     # best_epoch、best_val_iou、峰值显存、历史
eval/                  # 评测阶段生成
```

- `resume` 会恢复优化器、调度器、RNG 与 epoch；
- 全新进程用 `load_checkpoint()` 重建模型；保存→重载的 logits 差实测 **0.0**；
- 正式训练完成后不得再改 `base.yaml` 的公共项；若必须改，四组全部重跑并升数据/协议版本。

## 4. 明确不在本轮协议内

- 多卡 DDP、梯度累积（batch 64 已足够快）；
- 阈值扫描作为主结果（只在 val 上做补充分析）；
- 最小对象面积后处理（如启用必须四组一致并单列）；
- B2 四组完整训练（只做前向/反向/重载兼容性验证）。
