# 单头分割与语义—细节注意力融合执行报告（2026-10-19）

> 审阅补充：详见 [研究复盘](research_reassessment_2026-10-19.md)。第2节表格实际使用0.5阈值，0.4只被选择并保存在PR曲线中；第5节前景内部W方差大于边界；注意力实现还增加了局部深度3×3，故应标为AFF变体。历史训练结果不因此改变，但相关机制结论需要收紧。

> 执行 [`single_head_attention_execution_plan_2026-10-19.md`](single_head_attention_execution_plan_2026-10-19.md)。
> 数据 `s2_10band_v3_p20`；val=hiroshima（448），test=Italy（2,795，历史诊断集，非盲测）；
> 全部新实验 **seed=1234**、3,600 次更新、batch 64、FP32；单种子只支持开发判断。
> 一句话结论：**N2 与 N1 持平（val 0.2690 vs 0.2692），AFF 注意力没有可测增益；单头路径相对 N0 是“val 降、test 升”的权衡，不进入主线。**

## 1. 实现（P1/P2）

| 文件 | 内容 |
|---|---|
| `models/feature_fusion.py`（新增） | `AFF`（WACV 2021，单次、64 通道、r=4，末 BN 零初始化→W=0.5）、`SingleHeadFusion`（U/V 投影 + SUM/AFF + 单头）、`SingleHeadSeg` |
| `models/factory.py` | `model.type=single` → `SingleHeadSeg`；旧 segformer/unet 路径不变 |
| `config.py` | `model.fusion=sum|aff`、`fusion_width=64`；旧默认行为保留 |
| `training/segformer_trainer.py` | 单头模型拒绝 `coarse_weight>0`；pretrained 身份记录覆盖 single |
| `configs/segformer/single_head/` | `base.yaml`、`n0.yaml`（CF0_NOAUX）、`n1.yaml`（SINGLE_SUM）、`n2.yaml`（SINGLE_AFF） |
| `tests/test_segformer_single_head.py` | 5 项：共享初始化/输出等价、W=0.5 退化、形状/参数分组、重载、拒绝粗损失 |

架构（与计划 §3 一致）：

```
D = 官方 MiT-B0 + 官方解码器特征 (256×32×32)
U = up(GN8(Conv1×1(D,256→64)))
V = GN8(Conv1×1(S,32→64))
Z = U + V            (N1)
  = AFF(U,V)         (N2)  AFF: W=sigmoid(local+global), Z=2W⊙U+2(1-W)⊙V
F = Conv3×3(64,64) → GN8 → GELU → Conv1×1(64,1)
```

- 官方 `decode_head.classifier` 替换为 `Identity`，直接把解码特征 D 接出来，不保留未参与计算的旧二分类头；
- 最终 1×1 用标准小随机权重、bias=0（不做零初始化）；
- `set_seed(seed)` 在建模前调用，N1/N2 的 stem、proj、head 初始张量逐位一致（实测 bitwise equal）。

## 2. P0：粗定位独立评价（CF0/CF1）

冻结阈值 **0.4**（在 Hiroshima 上按粗像素 F1 选择，Italy 只诊断）。

| 组 | split | 输出 | IoU | Dice | P | R | 预测面积 |
|---|---|---|---:|---:|---:|---:|---:|
| CF0 | val | coarse C | 0.2609 | 0.4139 | 0.3953 | 0.4343 | 307 px |
| CF0 | val | final F | **0.3135** | 0.4774 | 0.4320 | 0.5334 | 345 px |
| CF1 | val | coarse C | 0.2612 | 0.4143 | 0.3988 | 0.4309 | 302 px |
| CF1 | val | final F | **0.3158** | 0.4800 | 0.4392 | 0.5291 | 337 px |
| CF0 | test | coarse C | 0.0525 | 0.0998 | 0.1142 | 0.0887 | 221 px |
| CF0 | test | final F | 0.0632 | 0.1190 | 0.1299 | 0.1097 | 241 px |
| CF1 | test | coarse C | 0.0518 | 0.0986 | 0.1137 | 0.0870 | 218 px |
| CF1 | test | final F | 0.0628 | 0.1182 | 0.1312 | 0.1076 | 234 px |

候选预算（按粗概率排序；CF0≈CF1，取 CF0）：

| split | 预算 | 区域 Precision | GT 覆盖 | 对象触及 | 对象内部覆盖 |
|---|---|---:|---:|---:|---:|
| val | 5% | 0.191 | 0.650 | 0.770 | 0.602 |
| val | 10% | 0.130 | 0.793 | 0.866 | 0.760 |
| val | 20% | 0.078 | 0.883 | 0.922 | 0.869 |
| test | 5% | 0.089 | 0.244 | 0.282 | 0.183 |
| test | 10% | 0.074 | 0.413 | 0.460 | 0.335 |
| test | 20% | 0.057 | 0.632 | 0.675 | 0.555 |

**读法**：val 上 20% 面积预算就能覆盖 88% 的 GT、触及 92% 的对象，粗定位不是瓶颈；
test 上同预算只覆盖 63%，差距主要来自域/尺度，而不是“粗阶段找不到”。
产物：`outputs/segformer/coarse_localization/coarse_localization.{json,md}`。

## 3. 工程门槛（P2）

| 检查 | 结果 |
|---|---|
| N1/N2 共享初始化 | proj_d/proj_s/head/stem 逐位一致 |
| N2 初始 W | 恒 0.5，输出与 N1 **bitwise equal** |
| 5 步更新 | 损失有限；235/235 参数有梯度；AFF 15/15 参数有梯度；W std 2.3e-5（开始偏离 0.5） |
| 空标注 / 全无效 | 空 GT 损失有限；全无效 batch 跳过且只计 1 次 optimizer step |
| batch1 eval | 正常（AFF 全局分支 BN 用 running stats） |
| 重载 / 恢复 | max\|Δ\|=0；LR 与 last_epoch 连续 |
| N0 | 粗项不在 loss；粗分类器仍有梯度（17.1）；18,561 个 fusion 参数不参与计算 |
| 8 图过拟合（500 步） | **N1 Dice 0.9944、N2 Dice 0.9975**（目标 ≥0.90） |

## 4. 正式结果（3,600 步，seed 1234）

| 组 | val IoU | val macro IoU | val 对象召回 | val 边界F1 | test IoU | test 对象召回 | test 边界F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CF0（粗辅助，参照） | 0.3135 | — | 0.489 | 0.524 | 0.0632 | 0.0427 | 0.137 |
| **N0 = CF0_NOAUX** | 0.2987 | 0.2145 | 0.4547 | 0.5012 | 0.0529 | 0.0294 | 0.1112 |
| **N1 = SINGLE_SUM** | 0.2692 | 0.1922 | 0.4448 | 0.4566 | **0.0755** | **0.0674** | **0.1668** |
| **N2 = SINGLE_AFF** | 0.2690 | 0.1907 | 0.4342 | 0.4578 | 0.0727 | 0.0647 | 0.1617 |
| A1（历史基线） | 0.3068 | — | 0.522 | 0.516 | 0.0683 | 0.0483 | 0.147 |

- **N0 vs CF0**：去掉粗辅助损失后 val −0.0148、test −0.0103 → 在 CF0 结构上，粗辅助监督是正贡献（与 A1-AUX 的结论不同，说明它依赖结构）。
- **N1/N2 vs N0**：val −0.0295/−0.0297，但 **test +0.0226/+0.0198**，对象召回从 0.029 提到 0.067/0.065，边界 F1 从 0.111 提到 0.167/0.162 → 明显的“val 降、test 升”权衡。
- **N2 vs N1**：val −0.0002、test −0.0028、对象召回 −0.0027、边界 +0.0012 → **没有稳定增益**。

成本（cuda:0，fp32，batch=1，128×128，预热 10/计时 50）：

| 组 | 参数（全部可训练） | B=1 延迟 p50/p95 |
|---|---:|---|
| N0 | 3,809,155（含 18,561 未用 fusion） | 5.09 / 8.28 ms |
| N1 | 3,790,529 | 5.14 / 6.68 ms |
| N2 | 3,795,121（AFF +4,592） | 5.74 / 10.16 ms |
| U0（历史对照） | 14,350,161 | 2.26 / 2.64 ms |

## 5. N2 的 W 分布与敏感性

| 区域 | 均值 | 标准差 | 饱和 <0.05 / >0.95 |
|---|---:|---:|---:|
| 全部有效像素 | 0.5017 | 0.0184 | ~4e-7 / ~6e-7 |
| 前景内部 | 0.4957 | 0.0488 | 0 / 0 |
| 边界带 | 0.4990 | **0.0403** | 0 / 0 |
| 背景 | 0.5018 | 0.0176 | 0 / 0 |

- W 几乎没有饱和，长期在 0.5 附近；**边界带方差最大**（0.040 vs 背景 0.018），说明注意力在边界处更活跃，但没有偏向某一分支。
- 固定 checkpoint 的敏感性（96 张 val）：

| 干预 | 平均 IoU | Δ vs 模型 |
|---|---:|---:|
| 模型自身 W | 0.2404 | — |
| W→0.5（退化为相加） | 0.2360 | **−0.0043** |
| W→1（仅语义 2U） | 0.1528 | −0.0876 |
| W→0（仅细节 2V） | 0.1512 | −0.0891 |

→ 两路都不可少；学到的 W 相对 0.5 只带来 +0.004 的局部收益，与 N2≈N1 的整体结果一致。
这是分布改变下的敏感性检查，不是重训消融。

## 6. 决策（按计划 §8）

1. **N2 没有稳定收益且没有显著增加误检**（test 对象召回略降、边界略升）→ 保留负结果，
   **优先采用 N1 作为简单参照**，不自动堆叠更多注意力模块。
2. **单头路径不是免费简化**：相对 N0，val −0.030、test +0.023。按 val 选模型的协议它不占优；
   但 test 的对象召回/边界明显更好，值得作为“域/尺度权衡”单独跟进，而不是替换主线。
3. **N0 的结论**：在 CF0 结构上，粗辅助监督 +0.015 val；删掉它会让最终输出变差。
   这与 A1-AUX（在 A1 上加粗监督 −0.009）并不矛盾——辅助项的效果依赖结构。

## 7. 案例

`SINGLE_{val,train,test}_cases.png`：4 行 × 8 列
（RGB / GT / N1 概率 / N2 概率 / N1 mask / N2 mask / N2 误差图 / |N2−N1|），
四行分别是 N2 最大改善、N1 误检减少、N2 新增误检、边界修正。

## 8. 限制

- 单 seed；val 单事件；Italy 是已多次查看的诊断集，不是盲测；
- AFF 采用零初始化末 BN 的本项目适配（原实现为常规初始化），初始 W=0.5 是工程选择；
- 单头模型 3,600 步、与 N0 同预算，但结构差异（去 C+Δ、去粗头、单头输出）是打包改变，不能单独归因；
- 不宣称统计显著性；N2≈N1 的差异（0.0002）远小于单 seed 波动。

## 9. 产物与复现

```powershell
$PY = "C:\Users\yzh\.workbuddy\binaries\python\envs\tm\Scripts\python.exe"

& $PY scripts/gate_single_head.py
& $PY scripts/overfit_segformer.py --config configs/segformer/single_head/n1.yaml --tiles 8 --max-steps 500
& $PY scripts/overfit_segformer.py --config configs/segformer/single_head/n2.yaml --tiles 8 --max-steps 500
& $PY scripts/train_segformer.py --config configs/segformer/single_head/n0.yaml --device cuda:1
& $PY scripts/train_segformer.py --config configs/segformer/single_head/n1.yaml --device cuda:0
& $PY scripts/train_segformer.py --config configs/segformer/single_head/n2.yaml --device cuda:0
& $PY scripts/evaluate_single_head.py --experiments CF0_NOAUX SINGLE_SUM SINGLE_AFF --splits val test
& $PY scripts/analyze_single_head_attention.py --split val
& $PY scripts/visualize_single_head_cases.py --split val
& $PY scripts/analyze_coarse_localization.py --experiments CF0 CF1 --splits val test
```

| 产物 | 位置 |
|---|---|
| 门槛 | `outputs/segformer/single_head/gates.json` |
| 训练 | `outputs/segformer/s2_10band_v3_p20/{CF0_NOAUX,SINGLE_SUM,SINGLE_AFF}/1234/` |
| 评测 | `outputs/segformer/single_head/eval_v2.{json,md}` |
| 注意力 | `outputs/segformer/single_head/attention_analysis.json`、`attn_W_val.png` |
| 案例 | `outputs/segformer/single_head/SINGLE_{val,train,test}_cases.{png,json}` |
| 粗定位 | `outputs/segformer/coarse_localization/coarse_localization.{json,md}` |
