# 一张影像进入 DOFA 与 AnySat 后的实际输出

最后维护：2026-10-30。本文只说明**当前 USE-Next v2 的 P1 原生读出**中，一块影像进入冻结 DOFA / AnySat 后到底得到什么张量、这些张量是什么含义，以及“能不能得到不同模态的影像”。它不是新的阶段总结，成绩仍以[项目数据与结果](DATA_EVENT_PROTOCOL.md)和[阶段台账](assets/research/stage_ledger.json)为准。

## 阅读约定与设置

**数据角色。** 用【我们的数据划分（留事件验证）】的开发验证块 `hiroshima_s2_1000`（广岛事件，2018-07-11，128×128 像素＝1280 米见方，十波段 Sentinel-2 地表反射率，附逐像元有效掩膜；本块有效像元占 57.7%）。选它是因为它已出现在 P1 的 669 块预测里，便于逐字节核对；本块不参与任何训练。

**模型与权重。** 两路由[原生读出](../src/landslide_seg/models/use_next_v2/readout.py)装配：
- DOFA：官方 `vit_base_patch16(img_size=224)` 密集编码器，加载发布权重 `weights/dofa/XShadow_DOFA_base/DOFA_ViT_base_e100.pth`（173 个张量，**只有编码器**，无解码器），见[anchors.py](../src/landslide_seg/models/use_next_v2/anchors.py)。
- AnySat：官方 `AnySat(model_size='base')`，`checkpoint['state_dict']` 严格加载（missing=0、unexpected=0），同样冻结。
- 两路都只训练 1×1 投影与末端分类头：DOFA 可训练 98,529 参数，AnySat 49,313 参数；编码器在全过程哈希不变（DOFA `1af3845a…`，AnySat `b1f574b7…`）。

**口径。** 下文的“输出”一律指网络返回的张量；“概率”指 `sigmoid(logits)` 后的单通道滑坡概率，只在有效区内计分；“级”指编码器输出的特征图层级。所有数值来自一次 CPU 前向（`formal_updates = 0`），不是重新训练或重新选模。

## 主表一：张量形状与含义（batch = 1）

| 阶段 | DOFA 路径 | AnySat 路径 |
|---|---|---|
| 输入 | 10×128×128（反射率＋有效掩膜） | 同左，另加观测日序 |
| 预处理 | 同范围双线性插值到 10×224×224（相当于 5.71 米/像素的重采样，不增加真实细节） | 按官方统计标准化，保持 10 米网格 |
| 编码粒度 | patch 16 像素 → 196 个 token，12 个 Transformer block，取第 4/6/8/12 个 block 输出 | patch 40 米 → 32×32＝1024 个 patch，每个 patch 含 4×4 个 10 米子 patch |
| 编码器输出 | 4 份 `(1, 768, 14, 14)`，每格约 91 米 | 1 份 `(1, 128, 128, 1536)`＝上下文 768 通道＋局部 768 通道 |
| 1×1 投影 | 4 个 `Conv2d(768,32,1)` → 4 份 `(1,32,128,128)` | 2 个 `Conv2d(768,32,1)` → 2 份 `(1,32,128,128)` |
| 融合 | 4 份等权平均 | 2 份等权平均 |
| 分类头 | GroupNorm(4,32)→GELU→`Conv2d(32,1,1)` | 同左（同一结构、同一随机种子初始化） |
| 最终输出 | logits `(1,1,128,128)` → 概率 `128×128` | 同左 |

两个要点：DOFA 的四级特征**空间步长相同**（都是 stride 16），只是深度不同，所以这里的“多级”是语义深度，不是金字塔分辨率；AnySat 的 1536 通道是**两段拼接**，前 768 通道在每个 40 米 patch 内恒定，后 768 通道是 10 米子 patch 向量。

## 主表二：本块实测

| 指标 | DOFA 路径 | AnySat 路径 |
|---|---|---|
| 编码器前向（单块 CPU） | 0.07–0.10 秒 | 2.4–2.5 秒 |
| 各级特征标准差 | 0.377 / 0.546 / 0.958 / **5.504** | 上下文 0.181，局部 0.750 |
| 特征极值 | 第 4 级最大 288.87 | 上下文 [-0.69, 0.57] |
| patch 内一致性 | — | 上下文块内 std 最大 **0.000**；局部块内 std 均值 0.150 |
| 概率（有效区内） | 均值 0.031，>0.5 的像元占 2.19% | 均值 0.027，>0.5 的像元占 1.76% |
| 概率（无效区内，不计分） | 均值 0.0075 | 均值 0.208 |
| 与本块真值 IoU@0.5 | 0.000 | 0.047 |
| 与已存预测 `p1_predictions/<路由>/best/0000.npz` 的最大差 | 6.5e-6 | 3.7e-6 |

差值是 CPU 与原 CUDA 的浮点求和顺序不同造成的，不是权重或输入不同。本块真值只有 58 个像素（占有效区 0.35%），所以 0.000 与 0.047 **不构成模型比较**；两路的整体水平见[基线比较](../outputs/use_next_v2/audit/P1_BASELINE_COMPARISON.json)。

## 图：同一块影像在两套冻结编码器里的样子

![DOFA与AnySat单块输出](../outputs/use_next_v2/explain_native/native_outputs.png)

每格内容与看点（灰色＝无效像元，不参与评分）：

- **上左一**：输入十波段真彩色（红/绿/蓝，2–98 百分位拉伸）。灰色为无效像元，占 42%。
- **上左二**：滑坡真值。只有 3 处小斑块（共 58 像素），任何单块 IoU 都会被小目标主导。
- **上左三**：DOFA 第 4 级原生特征的三主成分伪彩。能看到 14×14 的粗格被放大到 128×128，边界是平滑过渡——它没有 10 米级的空间支撑。
- **上左四**：DOFA 四份 32 通道投影取平均后的伪彩，是真正进入分类头的表示；比单级细，但仍是插值结果。
- **下左一**：AnySat 上下文段。应看**规则的 40 米方格**：同一 patch 内颜色完全一致（块内 std＝0），这是“每 patch 一个向量”的直接后果。
- **下左二**：AnySat 局部段。纹理明显更细，能看到 4×4 子 patch 网格，说明 10 米信息保留在这一段。
- **下左三／下左四**：两路的最终滑坡概率（只显示有效区）。两路在有效区内都很稀疏；亮度主要集中在图块边缘和少量散点上，并没有覆盖真值的小斑块——这与 0.000／0.047 的单块 IoU 一致。

## 展开结论：关于“得到不同模态的图像”

**结论一：按现在的接线，DOFA 和 AnySat 不会输出影像，更不会输出另一种模态的影像。** 机制上，代码只调用编码器前向并取密集特征（`NativeDOFA.forward` 返回四级特征，`NativeAnySat.forward` 返回 `output='dense'` 的特征），随后是 1×1 投影与分类头；没有任何解码器把特征还原成像素。证据：DOFA 发布权重只有 173 个张量，前缀为 `blocks / patch_embed / pos_embed / cls_token / mask_token / norm / projector`，**没有 `decoder_*`**；AnySat 的 `state_dict` 能对 `Any_module` 严格加载，说明也只有投影器与 Transformer 主干。边界：这证明“当前仓库拿不到影像”，不证明“这两个模型原理上不能重建”。

**结论二：这两篇方法里的“任意模态”，在代码里表现为“同一套冻结权重可以吃不同波段组合／不同传感器”，不是“生成另一种模态的图”。** DOFA 的入口是**波长条件**的 patch embedding：权重生成器把每个波段的中心波长（微米）编码后现场生成该波段的 16×16 卷积核，因此同一权重既能跑十波段（0.49–2.20 µm），也能跑三波段 RGB（0.665/0.560/0.490 µm，见 `dofa_rgb` 探针与历史四折配置 [F1_dofa_rgb.yaml](../configs/segformer/event_cv_v5_probe/F1_dofa_rgb.yaml)）。AnySat 的入口是**逐模态投影器**：官方同一 checkpoint 含 11 个 tokenizer（`s2 / s1 / s1-asc / l8 / l7 / alos / modis / aerial / aerial-flair / spot / naip`）。边界：我们的实现只放开了一部分——DOFA 必须是光学地表反射率且波长已知（`NativeDOFA.prepare` 对 SAR 或未知波长直接报错），AnySat 只接受 S2 十波段、10 米正交格网且边长能被 4 整除（`NativeAnySat.prepare`）。SAR 分支尚未接入，属于待实现，而不是已被否定。

**结论三：要真的产出“另一种模态的影像”，缺的是重建解码器，而且必须自己训练。** DOFA 官方仓库确实带这个结构：`pretraining/models_base_ofa_mae.py` 的 `decoder_pred = Dynamic_MLP_Decoder(...)`，按请求波长把编码特征解码回该波长的 patch。它不在发布权重里，我们也没有训练过；AnySat 侧同类重建头同样未加载。因此这一步目前只能写“未实现、未验证”，不能把本文的特征图当作重建结果。

**结论四：当前输出形态带来两个可观察后果。** 其一，AnySat 的上下文段在 40 米 patch 内恒定，经 1×1 投影与双线性放大后，在 10 米网格上必然留下方格台阶（本块可见，且与 `anchors.py` 中“块内 std＝0”的实测注释一致）。其二，DOFA 的有效空间支撑约 91 米，因此它的概率图平滑，对 58 像素这种小目标几乎不响应。另有一个不计分但值得记录的现象：编码器对无效区输入是被置零的，DOFA 在那里输出接近 0（0.0075），AnySat 却给出 0.208 的平均概率——无效区不进入评分，但说明 AnySat 对“全零输入”并不中性。以上是“机制→图样”的解释，**不是**两路整体优劣的因果结论：两路的可训练头容量、输入尺度和初始化都不同。

## 限制与下一步

- 单块、单种子；两路读出头均为 P1 的 3000 步冻结读出。本块真值极小，表内 IoU 只能作为“输出长什么样”的旁证。
- 前向耗时是单样本 CPU 测量，含 Python 开销，不是吞吐基准；GPU 上的正式成本见 P1 `summary.json`。
- 未验证：DOFA 在十波段以外的波长组合（我们只跑过 3 与 10 波段）、AnySat 的 s1/l8 等分支在本机数据上能否给出可用特征、以及任何重建解码器的重构误差。
- 若下一步要做“跨模态图像”，最小路径是：固定成对 S2/S1（或 S2 多波段子集）样本 → 只训练 DOFA 的 `Dynamic_MLP_Decoder`（或 AnySat 对应重建头）→ 重建误差与下游分割 IoU **分开**评估，不能用一个指标代替另一个。

## 附录：复现

```powershell
# 只读单块前向：不训练、不改权重、不写预测
.venv-use-independent\Scripts\python.exe -X utf8 scripts\use_next_v2\explain_native_outputs.py
```

产物：本页图 [`native_outputs.png`](../outputs/use_next_v2/explain_native/native_outputs.png)、逐项数值 [`outputs.json`](../outputs/use_next_v2/explain_native/outputs.json)、脚本 [`explain_native_outputs.py`](../scripts/use_next_v2/explain_native_outputs.py)。

核对用权重与输入：`runs/p1_dofa/best.pt`、`runs/p1_anysat/best.pt`（两路在 3000 步的 best 与 last 参数相同，只有优化器/RNG 状态不同），输入统计 [stats_core25.json](../manifests/v4r1/stats_core25.json)（`4cab777b…`），成员清单 [pool_100.jsonl](../manifests/v4r1/pool_100.jsonl)（`b51422e2…`）。
