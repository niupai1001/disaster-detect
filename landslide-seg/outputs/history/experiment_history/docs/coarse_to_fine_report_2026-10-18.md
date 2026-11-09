# 粗定位引导精分执行报告（2026-10-19）

> 执行 [`coarse_to_fine_execution_plan_2026-10-18.md`](coarse_to_fine_execution_plan_2026-10-18.md)。
> 数据 `s2_10band_v3_p20`；val=hiroshima（448），test=Italy（2,795，历史诊断集，非盲测）；
> 全部新实验 **seed=1234**，单种子只支持开发判断，不宣称统计显著性。
> 预算：有效 batch 64、FP32、**3,600 次 optimizer 更新**（150 epoch × 24 步）。

## 1. 实现（P1）

| 组件 | 实现 |
|---|---|
| 粗输出 | 官方 MiT-B0 + 官方 SegFormer 解码器 → `base_logits`（`C`），双线性上采样到输入尺寸 |
| 精分头 | `concat(up(D), S, P) → 1×1 Conv(289,64) → GELU → 3×3 Conv(64,64) → GELU → 1×1 Conv(64,1)`，末层零初始化 |
| 条件 `P` | `sigmoid(up(C))`；CF0 用全零；训练/推理都用预测值，禁止 GT 替代 |
| 最终输出 | `F = C + Δ`（初始 Δ≡0，严格等于粗输出） |
| A1-AUX | 结构不变，只额外监督 `C` |
| U0 | SMP `Unet(resnet18, imagenet)`，10 通道（RGB→B04/B03/B02，其余 0 初始化且可训练） |
| 损失 | `L = Lseg(F) + 0.5·Lseg(up(C))`（A1-AUX/CF1/CF0）；A1-base/U0 只用 `Lseg` |
| 工厂 | `models/factory.py`：按 `model.type` 构造、参数分组、checkpoint 重建；旧配置无 `type` 时按 segformer |
| 配置 | `configs/segformer/coarse_to_fine/{base,a1_base,a1_aux,cf1,cf0,u0}.yaml` |
| 测试 | 新增 6 项（CF 初始化等价、条件敏感性、CF0 不敏感、U-Net 形状/分组、粗损失、旧 A1 checkpoint 重载） |
| 初始化控制 | `train_one` 在建模前调用 `set_seed(seed)`（random/numpy/torch/cuda + cuDNN deterministic）；实测 CF1/CF0 的 Stem 与精分头 body 状态哈希完全一致；DataLoader 用 (seed, epoch) 生成器 |

## 2. P2 运行门槛（全部通过）

| 组 | 参数 | 损失有限 | 首步有梯度参数 | 第 5 步 refiner 梯度 | 条件敏感性 max\|Δ\| | 重载 max\|Δ\| |
|---|---:|---|---:|---:|---:|---:|
| A1_AUX | 3,753,602 | 是 | 207/215 | — | — | 0.0 |
| CF1 | 3,809,155 | 是 | 207/217 | 0.0330 | **2.96e-05** | 0.0 |
| CF0 | 3,809,155 | 是 | 207/217 | 0.0303 | **0.0** | 0.0 |
| U0 | 14,350,161 | 是 | 92/92 | — | — | 0.0 |

- CF1 替换 `P` 后输出改变（>1e-6），CF0 对替换不敏感 → 条件路径接线正确；
- 零初始化精分头在首步上游梯度为 0、第 5 步已非零（0.033），不是断路。

**8 图过拟合（关闭增强，500 步）**

| 组 | loss | 前景 Dice | 判定 |
|---|---|---|---|
| CF1 | 1.201 → 0.056（−95.4%） | **0.979** | 通过（≥0.90） |
| U0 | 0.898 → 0.0013 | **1.000** | 通过 |


### 2.1 补充门槛（P2 其余项）

| 检查 | 结果 |
|---|---|
| 空标注 batch（GT 全 0、valid 正常） | loss 有限、梯度有限（Dice 不产生 NaN） |
| 全无效 batch + 正常 batch | 跳过 1 个、仅 1 次 optimizer step、loss 有限 |
| optimizer/scheduler 状态恢复 | LR 2.33e-6 → 2.33e-6，`last_epoch` 6 → 6 |
| 旧 A1 漂移（当前代码重跑 val） | **0.3068 = 历史 0.3068** |

产物：`outputs/segformer/coarse_to_fine/extra_gates.json`。

## 3. 训练结果（P3，单 seed，3,600 步）

### 3.1 val（hiroshima，模型选择集）

| 组 | 最终 IoU | 粗 IoU | Δ(细−粗) | 最终对象召回 | 粗对象召回 | 边界 F1 | best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| A1（历史基线） | 0.3068 | 0.2696 | +0.0372 | **0.5219** | 0.2885 | 0.5155 | 15 |
| A1_AUX | 0.2976 | 0.2595 | +0.0380 | 0.4689 | 0.3303 | 0.5050 | 14 |
| **CF1** | **0.3158** | 0.2612 | +0.0545 | 0.4922 | 0.3536 | **0.5277** | 24 |
| CF0 | 0.3135 | 0.2609 | +0.0526 | 0.4894 | 0.3586 | 0.5243 | 24 |
| CF1_B（边界加权） | 0.3143 | 0.2616 | +0.0527 | **0.5064** | 0.3663 | 0.5279 | 24 |
| U0 | 0.2583 | 0.2583 | 0 | 0.3593 | 0.3593 | 0.4625 | 25 |

### 3.2 test（Italy，历史诊断集，不是盲测）

| 组 | 最终 IoU | 粗 IoU | 最终对象召回 | 边界 F1 |
|---|---:|---:|---:|---:|
| **A1** | **0.0683** | 0.0293 | **0.0483** | **0.1466** |
| A1_AUX | 0.0639 | 0.0505 | 0.0420 | 0.1368 |
| CF1 | 0.0628 | 0.0518 | 0.0420 | 0.1360 |
| CF0 | 0.0632 | 0.0525 | 0.0427 | 0.1365 |
| CF1_B（边界加权） | 0.0664 | 0.0531 | 0.0467 | 0.1446 |
| U0 | 0.0567 | 0.0567 | 0.0307 | 0.1136 |

### 3.3 成本

| 组 | 参数（全部可训练） | B=1 延迟 p50/p95 | 吞吐（B=1） | 训练峰值 | 单 run 时长 |
|---|---:|---|---:|---:|---:|
| A1 | 3,753,602 | 4.98 / 5.69 ms | 201 样本/s | 4,689 MiB | 7.5 min |
| A1_AUX | 3,753,602 | 5.11 / 8.48 ms | 196 样本/s | 4,694 MiB | 6.2 min |
| CF1/CF0 | 3,809,155 | 5.08 / 6.79 ms | 197 样本/s | 5,218 MiB | 6.5 min |
| U0 | 14,350,161 | **2.26 / 2.64 ms** | **443 样本/s** | **1,852 MiB** | 4.6 min |

计时口径：cuda:0、fp32、batch=1、128×128、不含数据读取；预热 10 次、计时 50 次取分位。


### 3.4 粗/细损失分项与参数组 LR（best checkpoint，val，threshold-free）

| 组 | fine BCE | fine Dice | coarse BCE | coarse Dice | total | best epoch | LR 组 |
|---|---:|---:|---:|---:|---:|---:|---|
| A1 | 0.0727 | 0.6995 | 0.2058 | 0.9325 | 0.6706* | 15 | 5.64e-5 / 5.64e-4 |
| A1_AUX | 0.0711 | 0.7169 | 0.0604 | 0.8033 | 0.6099 | 14 | 5.68e-5 / 5.68e-4 |
| CF1 | 0.0684 | 0.6906 | 0.0617 | 0.7596 | 0.5848 | 24 | 5.26e-5 / 5.26e-4 |
| CF0 | 0.0690 | 0.6915 | 0.0620 | 0.7596 | 0.5857 | 24 | 5.26e-5 / 5.26e-4 |
| CF1_B | 0.0669 | 0.6964 | 0.0617 | 0.7584 | 0.5867 | 24 | 5.26e-5 / 5.26e-4 |
| U0 | 0.0787 | 0.7410 | — | — | 0.4099 | 25 | 5.22e-5 / 5.22e-4 |

\* A1 没有粗监督，它的 coarse BCE 0.2058 说明"粗输出未校准"；因此它的 total 不是训练目标，
**不能跨目标函数比较 total**。A1-AUX/CF 的 coarse BCE 降到 0.06 量级，但最终 IoU 并未因此提高。

## 4. 定位覆盖与阶段案例

**按粗概率排序的有效像素覆盖率**（val，5%/10%/20% 面积预算）：

| 组 | 5% 像素 / 对象 | 10% 像素 / 对象 | 20% 像素 / 对象 |
|---|---|---|---|
| A1 | 0.678 / 0.763 | 0.801 / 0.856 | 0.887 / 0.924 |
| CF1 | 0.651 / 0.769 | 0.793 / 0.868 | 0.884 / 0.923 |
| CF0 | 0.650 / 0.770 | 0.793 / 0.866 | 0.883 / 0.922 |
| U0 | 0.600 / 0.732 | 0.715 / 0.826 | 0.805 / 0.900 |

→ **定位不是瓶颈**：20% 预算即可覆盖 88% 的 GT 像素、92% 的对象（val）。test 上同样 20% 预算覆盖 61–68% 像素，差距主要来自域/尺度而不是"找不到"。

阶段案例（`CF1_val_stage_cases.png`，4 行 × 8 列；四行分别是找回、误检减少、精分变差、边界修正。列：RGB / 假彩色 B12-B8A-B04 / GT / 粗概率 / 粗 mask / 细 mask / 误差图（TP绿/FP红/FN蓝）/ |细−粗| 概率差）：

1. `hiroshima_s2_1171`：粗 IoU 0.000 → 细 **0.297**（找回）；
2. `hiroshima_s2_512`：粗 FP 896 → 细 FP 579（减少，但该图 GT 仅 19 px，仍 0 IoU）；
3. `hiroshima_s2_578`：粗 **0.640** → 细 **0.157**（细阶段把正确的细长目标改坏）。


### 4.1 细阶段对粗阶段的逐图影响（CF1）

| split | 图块 | 平均 ΔIoU(细−粗) | 细好 >0.05 | 细差 >0.05 | FP 增加 >50 px |
|---|---:|---:|---:|---:|---:|
| val | 448 | +0.0366 | 36.6% | 4.9% | 83（18.5%） |
| test | 2,795 | +0.0090 | 6.9% | 0.8% | 440（15.7%） |

细阶段整体是净收益，但在一小部分图上会把粗阶段改坏并抬高误检（val 4.9% 的图变差、
18.5% 的图 FP 增加超过 50 px）。这就是"细阶段改坏粗阶段"的定量边界；
三类案例见 `CF1_val_stage_cases.png`，训练/Italy 版本见 `CF1_train_stage_cases.png`、`CF1_test_stage_cases.png`。

训练曲线：`training_curves.png`（左：训练损失；右：val IoU）。
注意 A1-AUX/CF1/CF0 的损失含 0.5×粗项，**总损失大小不能跨目标函数比较**。


### 4.2 分档对象召回与负瓦片

val（hiroshima）：

| 组 | small 对象召回 | medium | large | tiny(1–4px) 对象召回 |
|---|---:|---:|---:|---:|
| A1 | **0.436** | **0.680** | **0.812** | 0.156 |
| A1_AUX | 0.390 | 0.611 | 0.771 | 0.156 |
| CF1 | 0.407 | 0.652 | 0.771 | 0.156 |
| CF0 | 0.404 | 0.649 | 0.771 | 0.143 |
| CF1_B | 0.428 | 0.652 | 0.771 | **0.169** |
| U0 | 0.275 | 0.509 | 0.688 | 0.052 |

负瓦片：val 仅 2 个，**六组都把 2/2 预测为正**（样本太少，不能据此给出误报率）；
test 无负瓦片（N/A）。

## 5. 决策（按执行计划 §7）

| 对照 | 结果 | 结论 |
|---|---|---|
| CF1 vs A1-AUX | +0.0182 val | 精分头结构有正收益 |
| **CF1 vs CF0** | **+0.0023 val**（对象召回 +0.003，边界 +0.003） | **不能证明"显式定位条件"有贡献**：收益来自新增精分头的容量 |
| CF1/CF0 vs A1 | +0.0090 / +0.0067 val；test −0.005/−0.005；对象召回 −0.03 | 只在 val 上小幅提升，未迁移到 test；不能宣称整体改善 |
| A1-AUX vs A1 | −0.0092 val | 单独增加粗定位辅助监督**有害** |
| U0 vs A1 | −0.0485 val / −0.0116 test | 成熟卷积对照明显低于 SegFormer；符合"预训练任务/容量不同"的预期 |

**按计划规则的原话**：CF1 优于 A1-AUX 但不优于 CF0 → 只能支持"新增精分结构"的价值，不能证明显式定位条件有贡献。本轮不据此修改主模型，也不追加参数网格。

**下一步（若继续）**：定位覆盖已经很高（20%→92% 对象），优先解决精分阶段的精度/边界与"细阶段改坏粗阶段"的案例（如 hiroshima_s2_578），而不是继续加强定位或换主干。


### 5.1 第二轮：边界加权损失（CF1-B，单因素）

按计划 §7"边界差则考虑边界细化"，只改一项：精分 BCE 按 GT 边界带加权
（`band = dilate(gt) − erode(gt)`，权重 1+0.5·band，width=2；Dice、粗损失、结构、预算不变）。

| 指标 | CF1 | CF1-B | Δ |
|---|---:|---:|---:|
| val IoU | 0.3158 | 0.3143 | −0.0015 |
| val 对象召回 | 0.4922 | **0.5064** | +0.0142 |
| val 边界 F1 | 0.5277 | 0.5279 | +0.0002 |
| test IoU | 0.0628 | 0.0664 | +0.0036 |
| test 对象召回 | 0.0420 | 0.0467 | +0.0047 |
| test 边界 F1 | 0.1360 | 0.1446 | +0.0086 |

**结论**：边界加权把指标重心从像素 IoU 移向对象召回/边界（val 召回 +1.4pt、test +0.5pt），
但 val IoU 反而 −0.0015，test 仍低于 A1（0.0683）。单 seed 下**不足以采用**；
它说明"细阶段的问题不只是边界带权重"，把粗阶段正确结果改坏的案例（如 hiroshima_s2_578）
需要的是局部判别/不确定性处理，而不是全局边界加权。按计划保留该失败结果，不进入主模型。

## 6. 限制

- 单 seed、单 val 事件；test 是已多次查看的 Italy 诊断集，不构成盲测；
- U0 的预训练任务（ImageNet 分类）与 SegFormer（ADE 分割）不同，比较是"方案效果与成本"，不是纯架构因果；
- 粗/细比较共享同一 checkpoint，但两者的超参/损失权重未单独调优（λc 固定 0.5）；
- 没有做多种子，因此不宣称任何显著性或跨种子稳定性。

## 7. 产物与复现

```powershell
$PY = "C:\Users\yzh\.workbuddy\binaries\python\envs\tm\Scripts\python.exe"

& $PY scripts/gate_coarse_to_fine.py --models a1_aux cf1 cf0 u0
& $PY scripts/overfit_segformer.py --config configs/segformer/coarse_to_fine/cf1.yaml --tiles 8 --max-steps 500
& $PY scripts/overfit_segformer.py --config configs/segformer/coarse_to_fine/u0.yaml  --tiles 8 --max-steps 500
& $PY scripts/train_segformer.py --config configs/segformer/coarse_to_fine/a1_aux.yaml --device cuda:0
& $PY scripts/train_segformer.py --config configs/segformer/coarse_to_fine/cf1.yaml    --device cuda:0
& $PY scripts/train_segformer.py --config configs/segformer/coarse_to_fine/cf0.yaml    --device cuda:0
& $PY scripts/train_segformer.py --config configs/segformer/coarse_to_fine/u0.yaml     --device cuda:1
& $PY scripts/analyze_coarse_to_fine.py --experiments A1 A1_AUX CF1 CF0 U0 --splits val test
& $PY scripts/visualize_coarse_to_fine_cases.py --experiment CF1 --split val
```

| 产物 | 位置 |
|---|---|
| 门槛 | `outputs/segformer/coarse_to_fine/gates.json`、`extra_gates.json` |
| P0 基线快照 | `outputs/segformer/coarse_to_fine/baseline_snapshot.json`（A1/1234 配置、权重/清单/checkpoint 哈希、环境） |
| 损失分项 | `outputs/segformer/coarse_to_fine/loss_breakdown.{json,md}` |
| 过拟合 | `outputs/segformer/_overfit/{CF1,U0}/report.json` |
| 训练 | `outputs/segformer/s2_10band_v3_p20/{A1_AUX,CF1,CF0,U0}/1234/` |
| 阶段/覆盖 | `outputs/segformer/coarse_to_fine/coarse_to_fine_analysis.{json,md}` |
| 案例 | `outputs/segformer/coarse_to_fine/CF1_{val,train,test}_stage_cases.{png,json}` |
| CF1-B 配置/结果 | `configs/segformer/coarse_to_fine/cf1_boundary.yaml`、`outputs/segformer/s2_10band_v3_p20/CF1_B/1234/` |
| 训练曲线 | `outputs/segformer/coarse_to_fine/training_curves.png` |
