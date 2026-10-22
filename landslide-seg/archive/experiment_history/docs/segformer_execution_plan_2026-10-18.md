# SegFormer 单期 Sentinel-2 滑坡分割执行计划

> **2026-10-18 更新**：本文的"当前起点"已过时（P0 已试跑一次并失败，模型封装已部分落地）。
> 现状核实、缺陷清单、执行顺序与预算一律以 [`segformer_execution_plan_2026-10-18_v2.md`](segformer_execution_plan_2026-10-18_v2.md) 为准；本文保留作为设计契约（§2–§4、§8）的原始出处。

日期：2026-10-18。状态：PLAN，尚未实施 SegFormer、下载其权重或运行四组实验。

本计划依据当前代码、`dataset_and_modules.md`、`train_run_2026-10-18.md`、`test_report_2026-10-18.md`、`report_pipeline.md`、`radiometry_verification.md`、认证清单和训练日志制定。以本轮用户需求为准：主要训练路线调整为 SegFormer；旧的 TerraMind / AnySat / TiM / 语义及粗细分割代码继续保留。

## 1. 当前起点与必须先澄清的问题

| 核查项    | 当前事实                                                                              | 对计划的影响                                     |
| ------ | --------------------------------------------------------------------------------- | ------------------------------------------ |
| 原始数据   | 实际列出 5,003 个 `*_s2_*.nc`                                                          | 优先检查本地可用样本，再决定是否下载更多分卷                     |
| 当前认证清单 | 156 条，train=80 / val=36 / test=40，7 个事件标识                                         | 可开展工程验证与先导实验，不能仅凭数量认定足够支撑泛化结论              |
| 事件划分   | train：chimanimani/china/dominicamaria/hokkaido/indonesia；val：hiroshima；test：italy | 保留事件归属，四组使用完全相同的清单；重新生成清单时验证事件与实际 bbox 重叠  |
| 旧模型训练  | `outputs_run2/logs/metrics.json`：val IoU 0.23444、test IoU 0.01771                 | 属于旧架构历史结果，本轮未重跑；不能归因于某个模块或数据量单一因素          |
| 报告时效性  | `dataset_and_modules.md` 仍写 72 样本/26 训练块                                          | 文档结论需与最新清单区分；特征可视化不能替代训练后能力验证              |
| 预处理代码  | `train_full.py` 的训练 `make_batches` 未传 boa_offset，val/test 传入配置的 +1000             | 存在训练/评估不一致；不能直接把旧的跨事件性能差异认定为纯域偏移           |
| 输入质量掩膜 | 当前缓存 valid 同时受 B01/B09 空洞影响                                                       | 新 10 波段路线必须从原始 10 波段、SCL、标签有效性重新生成 valid10 |
| 时相与标签  | 报告指出事件 mask 在 15 期重复；当前选择最清晰灾后时相                                                  | “灾后”不自动保证该时相仍与编目边界一致，需要记录事件间隔与可观察性复核       |
| 依赖与资源  | 专用环境 torch 2.11.0+cu128；transformers 未安装；两张 RTX 5090，各约 32 GB                     | 优先同一 GPU 跑 B0；先验证版本兼容与峰值占用，再确定批量           |
| 权重     | 项目 weights 下现有三套旧模型权重，未见 SegFormer 文件                                             | SegFormer 官方权重获取和加载是实施阶段的显式验收项             |

辐射报告中的“所有样本都应 +1000”暂不作为已核定事实。`verify_radiometry.py` 在元数据缺失时会默认 scale=1e-4、offset=0；这不能证明已经正确处理产品基线。应查明原始 Sen12 的 harmonization、对照产品的 processing baseline 与量化/offset 元数据，再决定转换。不能用验证集精度来选择“物理真值”。

## 2. 成熟组件与模型契约

优先使用 Hugging Face Transformers 的 `SegformerForSemanticSegmentation`，复用 MiT 分层编码器和 SegFormer 解码器。项目新增薄封装，处理输入适配、二分类输出、可选细节融合与统一输出尺寸，不复制 Transformer 或解码器算法。

- 第一轮骨干：MiT-B0；配置接口支持后续 MiT-B2。
- 首选权重：NVIDIA 发布的 `nvidia/segformer-b0-finetuned-ade-512-512`，复用已预训练的编码器和解码器，替换最终 ADE 类别分类器为单 logit 二分类器。
- B2 后续采用对应 NVIDIA B2 分割权重并单独记录预训练来源。四组 B0 实验不能混用 ADE 分割权重与 ImageNet-only MiT 权重。
- 锁定包版本、模型仓库 revision、完整文件 SHA256、许可证和加载键清单。若首选权重获取失败，明确阻塞；ImageNet-only `nvidia/mit-b0` 可作为显式备选，但解码器将是新初始化，必须四组统一并说明变更。
- 输入 `[B,10,H,W]`，输出 logits `[B,1,H,W]`，推理输出 sigmoid 概率及二值 mask。默认使用原始 128×128 共同 10m 网格；不为了 512 权重命名而把影像放大到 512。
- 固定波段顺序：`B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12`。其中 20m 原生波段已有重采样，Spectral Stem 的“原分辨率”指共同输入网格，不代表新增传感器细节。
- 二分类保持背景=0、滑坡=1；不启用 ADE 的 reduce_labels，不走 RGB 图像处理器默认 resize/rescale=1/255，也不使用其默认多类损失。

## 3. 两种输入适配的确定实现

### A：直接 10 通道

先正常加载三通道预训练模型，再显式替换第一层 patch embedding 卷积；保留输出通道、kernel、stride、padding、bias。

```text
W10[:,2] = Wrgb[:,0]  # R → B04
W10[:,1] = Wrgb[:,1]  # G → B03
W10[:,0] = Wrgb[:,2]  # B → B02
其他七个输入通道的卷积权重初始化为 0，全部允许训练
```

采用零初始化新增通道，是为了严格保留初始化时的 RGB 响应；这些通道有真实多光谱输入，训练后可以获得非零梯度。检查 RGB 拷贝误差=0、其余预训练层一致、新通道梯度和更新非零。不能用宽泛的 ignore_mismatched_sizes 导致整个第一层随机重建。

### B：可学习 10→3 投影

新增 `Conv2d(10,3,kernel_size=1)`，初始选择 B04/B03/B02，其他系数和 bias 为 0；其全部权重和 bias 均可训练，后接原有 RGB SegFormer。记录训练后非 RGB 系数更新，证明它不是固定选带。

A/B 使用同一标准化后的 10 通道输入。对同一输入、相同下游参数，在 eval 下验证 A 与 B 的初始化结果都等于“提取对应 RGB 通道送入原三通道模型”的结果。该测试验证适配正确性，不声称卫星反射率等同自然 RGB 图像分布。

## 4. 可选 Spectral Stem 与可比性

```text
共同标准化 X10 ── 输入适配 A/B ── 官方 MiT ── 官方 SegFormer 解码器
       │                                          │
       │                                   解码特征 D（通常 1/4 网格）
       │                                          │ 双线性上采样
       └─ 1×1 Conv(10,32) ─ GN/GELU ─ 3×3 Conv(32,32) ─ GN/GELU
                                                  │
                               concat[up(D), Stem(X10)] → 轻量残差预测
```

- Stem 全部 stride=1，3×3 padding=1，无池化、无下采样。
- Stem 关闭：直接执行原 SegFormer 二分类 logits 的上采样路径，不创建/调用额外融合头。
- Stem 开启：保留标准预测 `Lbase`，从上采样解码特征和 32 通道 Stem 特征得到 `ΔL`，最终 `L=Lbase+ΔL`。残差头建议 `1×1 Conv→GELU→1×1 Conv(…,1)`；末层零初始化，使初始输出与对应无 Stem 模型一致。
- 从官方分类器之前取得解码特征；优先通过包装/受控特征捕获复用官方 forward，避免手写 MLP 解码器。特征抽取位置随锁定版本做契约测试。
- 残差末层零初始化会让其上游在首次 backward 时暂时没有梯度；在后续步骤验证 Stem 及融合层都有更新，避免把这一预期行为误报为断路。
- A1/B1 的实验结论是“原网格细节分支及其必要融合头”的联合增益。仅四组实验不能进一步分离额外参数量与光谱/空间信息各自的贡献，不扩张本轮实验范围。

## 5. 数据与预处理阶段（P0）

1. 复用 `sen12.py` 的 nc 读取与波段顺序、`certified.py` 的清单/哈希机制、现有事件标识。建立独立 `s2_10band_v1` 清单与缓存，不覆盖旧的 12 波段数据产物。
2. 在本地 5,003 个文件上重新评估单期 10 波段可用性。无需通过 B01/B09 回补或其相关门槛；旧清单中因辅助波段回补失败而隔离的样本可重新检查，不能直接全部放行。
3. 选择有依据的灾后单期，记录采集日期、time_index、event_date、间隔天数、选择理由、SCL、可见性和 label_status。优先核实编目对应的灾后观测；改选更晚晴空景时，检查滑坡痕迹及边界是否仍适用，避免只按最小云量机械选择。
4. 重建 valid10：观测有效性/SCL、所有所需波段的有限值与源 nodata、标签已知区域的交集。保留未知标签，不把无事件时间或 annotated=False 的空 mask 当作负样本。
5. 取消旧认证器“正像素至少 20”的自动筛除规则，审查极小目标的标签可靠性；否则会在数据入口排除本轮重点研究的小滑坡。记录对象在云/图块边界上的截断情况。
6. 核定源数据物理尺度后，统一转换并标准化。初版采用训练集有效像素统计的逐波段均值/标准差，保存统计文件和样本身份；val/test 只应用训练统计。所有实验共用该规则，不用逐图拉伸或测试集统计。RGB 与新增波段的尺度差异在输入分布报告中明确呈现。
7. train/val/test 调用同一个预处理函数。测试相同样本在关闭增强时经三条路径输出完全一致，并覆盖 offset=0/非零配置是否实际生效。
8. 固定事件级划分；核对 bbox 实际重叠及 CRS，不只检查 5km 网格编号。当前 `make_splits.py` 的默认内部 val-block 划分可能使剩余事件跨 split，旧 split 字段也可能残留；需修正后用于本轮。

数据集规模策略：先用核验后的小清单做工程与过拟合测试，随后扩展本地所有可用标注样本。156 条旧清单是先导起点；正式训练规模以重新认证结果为准，报告每 split 的事件/图块/对象/小目标数量和有效正像素数。若仍只有少量独立事件，应完成四组先导实验并标注统计局限，不能宣称已充分验证泛化。

现有 italy 测试集已被多轮查看和用于诊断，应记为“历史固定留出集”，不再宣称从未接触的最终盲测。四组仍固定该划分以便配对比较；若需强泛化结论，之后新增从未用于开发的事件作为独立外部测试。不能因为分数低而把 italy 部分样本混入训练以改善该测试结果。

P0 产物：10 波段清单/隔离清单、缓存哈希、划分身份、统计文件、时相与辐射核验报告。未知辐射转换或标签不匹配样本阻塞正式训练，但不阻塞模型接口开发。

## 6. 实施与实验顺序

| 阶段         | 工作                                        | 验收与产物                                           |
| ---------- | ----------------------------------------- | ----------------------------------------------- |
| P0 数据与依赖   | 完成上节；安装并锁定兼容 Transformers；获取官方权重          | 数据清单、环境锁、权重加载报告；不能把 config 初始化当预训练              |
| P1 A0 工程   | 官方 SegFormer 封装、10 通道迁移、二分类输出、统一 BCE+Dice | 真实 batch forward/backward；尺寸、梯度、参数更新、全新实例保存重载测试 |
| P2 A0 学习验证 | 2–8 个可靠训练图块，关闭增强做过拟合；再用核定清单先导训练           | loss/IoU/Dice 曲线、正像素预测、图像对照；检查是否全背景、边界容量限制或标签错位 |
| P3 A1      | 增加原网格 Stem 与融合，其他设置沿用 A0                  | 开关一致性、Stem 梯度更新；过拟合与同协议训练                       |
| P4 B0      | 可训练投影＋原 RGB SegFormer                     | 初始化等价、投影非 RGB 系数梯度与更新；过拟合与同协议训练                 |
| P5 B1      | 在 B0 上启用同一 Stem                           | 相同检查与训练；生成四组配对结果                                |
| P6 正式对照    | 扩样后冻结新数据版本，必要时重跑所有组；四组相同预算                | 最佳模型、三种种子结果、案例与效率表、失败记录                         |
| P7 B2 兼容   | 在 B0 四组闭环后验证配置切换与权重迁移                     | 至少 B2 前向/反向/重载测试；完整 B2 四组属于后续扩展                 |

执行顺序固定 A0→A1→B0→B1。A0 可先调试合理训练参数，然后冻结四组公共设置；后续若因错误修复改变公共路径或数据版本，重跑受影响组。每组正式训练从同一来源预训练权重重新开始，A1/B1 不加载 A0/B0 已训练分割权重，以免增加训练预算。

过拟合验证先设 500 次更新诊断上限。要求 loss 与训练分割指标有明确改善、能定位正区域，输出不退化为常数；不能把一次 backward 当作学习成功。未达到预设诊断目标应检查数据、损失和优化，不自动扩大正式训练。极小目标的 1/4 解码网格限制单独记录，不硬编造一个必达训练 Dice。

## 7. 四组公共训练协议

| 实验 | backbone | 输入适配         | Stem  |
| -- | -------- | ------------ | ----- |
| A0 | MiT-B0   | direct10     | false |
| A1 | MiT-B0   | direct10     | true  |
| B0 | MiT-B0   | project10to3 | false |
| B1 | MiT-B0   | project10to3 | true  |

- 训练编码器、官方解码器、输入适配器与二分类器；无需长期冻结整个 SegFormer。所有组采用相同参数分组规则。
- 初始候选：AdamW，编码器 lr=6e-5，新层/解码器 lr=6e-4，weight_decay=0.01，梯度裁剪=1，5% warmup 后 polynomial decay。以上是待 A0 验证的启动参数，不是已验证最佳值。
- 初始 batch=8（128×128），以 A1/B1 最大显存实测确定共同有效 batch；必要时统一梯度累积。100 epochs 作为首轮正式预算候选；四组相同数据访问和更新预算，不依组单独延长训练。
- 统一损失 `0.5*masked_BCEWithLogits + 0.5*masked_SoftDice`，先复用并核验现有函数。全无效样本跳过并计数；不额外引入粗头、presence、文本或属性损失。
- 使用同一组翻转/90°旋转增强，影像、mask、valid 同步变换；不在四组中改变分辨率或使用不同采样规则。对小目标的增强不能使其标签消失而未记录。
- 每 epoch 验证，按验证集前景 IoU 保存 best，另存 last；测试集不参与调参、阈值选择与 checkpoint 选择。
- 先固定 threshold=0.5 报主结果；需要阈值扫描时仅在验证集按统一规则确定，作为另列结果。
- 先每组一个 seed 形成完整结果，再使用配对 seeds `[1234,2026,3407]` 完成四组×三种子。报告均值/标准差、逐事件和逐样本值。图块高度相关时不把像素当独立统计样本。
- 公共层用相同权重与初始化；增强和采样随机数流独立于模型初始化，避免新增 Stem 改变后续样本顺序。
- 性能允许时启用 bf16 autocast 并记录；先验证 fp32 正确性。相同精度与硬件用于四组测量。

## 8. 评价定义在正式训练前冻结

1. **像素指标**：有效区域内滑坡 IoU、Dice、Precision、Recall；给出聚合混淆计数、逐样本 CSV、逐事件汇总。复用已修正的 `metrics.py`，验证不将无效像素计入 TN。
2. **尺度分组**：GT 在有效、已知区域中的连通域，8 邻接；默认面积档候选为 small<0.5ha、medium=0.5–2ha、large≥2ha（10m 像素为 100m²）。在仅查看训练分布后核定档位，四组固定，报告各档对象数；1–4 像素目标另列计数和召回以暴露极小目标问题。
3. **小滑坡召回**：分别报告对象内像素召回与对象检测召回；对象召回采用预测/GT 连通域一对一匹配，IoU≥0.25 为预设主阈值、≥0.5 为严格补充，避免“一像素碰到即检出”。标签跨块或碰到无效区域的对象单独记录截断比例，不能无说明地重复计数。
4. **边界 F1**：主容差 1 像素（10m），可补充 2 像素；GT 与预测边界提取保持一致，排除距未知区域/图块外边缘不足容差的部分，防止有效掩膜边缘被当滑坡边界。空边界规则明确并测试。
5. **无滑坡图块误报率**：仅对经过核实的负图块计算“出现至少一个预测正像素的图块占比”；如使用最小对象面积后处理，应四组一致并另报。没有可靠负图块就写 N/A，保留背景像素 FPR 作为不同指标。
6. **效率**：总/可训练参数数；B=1、128×128 的推理峰值 allocated/reserved 显存、延迟 p50/p95 与吞吐。单 GPU、相同精度、预热 20 次/计时 100 次、CUDA 同步或 events；模型计算与读盘预处理分别计时。训练显存另报共同 batch 下峰值。
7. **案例分析**：固定可视化样本集合，包括小/中/大目标、低对比、背景裸地、云边界；保存 RGB/假彩色、GT、valid、概率图、预测、TP/FP/FN。按验证集规则选漏检/误检案例，测试案例仅在实验冻结后分析。

四组比较报告：输入适配效应 A0 vs B0、A1 vs B1；Stem 效应 A1−A0、B1−B0；可计算二者差值来观察交互。必须同时看 precision/recall，防止只靠扩大预测范围提高小目标召回。

## 9. 在现有工程中的修改范围

| 文件/目录（拟议）                                                                                                  | 处理                                                         |
| ---------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| `data/sen12.py`、`data/band_mapping.py`                                                                     | 复用读取/波段契约；补充单期标签可用性校验                                      |
| `data/segformer_dataset.py`                                                                                | 新增懒加载 Dataset、valid10、统一转换/标准化及同步增强                        |
| `scripts/prepare_segformer_data.py`                                                                        | 复用现有认证/划分工具能力，生成新 10 波段清单和缓存                               |
| `models/segformer.py`                                                                                      | 成熟模型薄封装、精确权重迁移、解码特征接口、尺寸恢复                                 |
| `models/spectral_stem.py`                                                                                  | 小型原网格分支与残差融合头                                              |
| `training/segformer_trainer.py`                                                                            | 单输出训练流程；复用公共工具，增加 best/last/resume/逐 epoch 验证              |
| `training/losses.py`、`metrics.py`                                                                          | 复用二值分割函数；将尺度与边界指标作为扩展，保持旧接口兼容                              |
| `config.py`                                                                                                | 复用 YAML 合并/seed/身份记录；增加独立 SegFormer schema，避免继承旧模型默认值污染新实验 |
| `configs/segformer/{base,a0,a1,b0,b1}.yaml`                                                                | 共用基础配置，组间只切输入适配/Stem；支持 backbone=b2                        |
| `scripts/{smoke_segformer,overfit_segformer,train_segformer,evaluate_segformer,run_segformer_ablation}.py` | 独立入口，避免旧 Trainer 强制粗细/语义接口；共用新训练器                          |
| `tests/test_segformer_*.py`                                                                                | 初始化等价、输入顺序、梯度、开关、mask 损失、尺度/边界指标与 checkpoint 测试            |
| `pyproject.toml` 与环境锁文件                                                                                    | 添加 SegFormer 可选依赖及真实数据依赖，锁定实际可用版本                          |
| `docs/`、`outputs/segformer/<data_version>/<experiment>/<seed>/`                                            | 保存计划、输入契约、实验记录、曲线、模型、指标、可视化和加载报告                           |

现有 `train_full.py` 等旧路线继续保留；其预处理不一致问题记录并修复时加回归验证，不利用旧结果替代新路线的验收。

## 10. 交付、阻塞与执行命令约定

以下为待实现入口，不表示当前已可运行：

```powershell
python scripts/prepare_segformer_data.py --config configs/segformer/base.yaml
python scripts/smoke_segformer.py --config configs/segformer/a0.yaml
python scripts/overfit_segformer.py --config configs/segformer/a0.yaml
python scripts/train_segformer.py --config configs/segformer/a0.yaml
# A0 通过后依次执行 a1.yaml、b0.yaml、b1.yaml
python scripts/run_segformer_ablation.py --order A0 A1 B0 B1 --seeds 1234 2026 3407
python scripts/evaluate_segformer.py --checkpoint <best.pt> --split test
```

每次运行保存 resolved_config、数据与划分 hash、统计文件、权重 revision/hash、加载缺失/新增/改形状键、软件版本、seed、优化器/调度器/RNG 状态、best/last、训练曲线和预测。全新进程重载 best 后评估结果应在声明数值容差内复现；resume 需恢复优化器、调度器、混合精度状态及随机状态。

完成标准：四组真实预训练加载与接口测试通过；真实小样本学习验证完成；相同固定数据协议下完成训练与评测；交付可复现配置、best/last、全部要求指标或明确 N/A、资源测量和错误案例。未达到精度预期仍如实交付真实指标，但必须区分“实验完成”与“模型有效”。

当前明确待办/风险：transformers 尚未安装，SegFormer 权重尚未获取；统一辐射口径与时相标签适用性需要核验；可靠负图块未建立；新的 10 波段有效样本规模尚未知。GPU 看起来支持开展 B0 小图实验，但 batch 与工期仅在 A0 profiler/先导训练后估算。本轮计划不承诺未经运行的时间、精度或提升。

## 11. 本计划核对的上游依据


- [Hugging Face SegFormer 文档](https://huggingface.co/docs/transformers/model_doc/segformer)：成熟 MiT 与轻量解码器接口、输入通道配置、标签处理。
- [NVIDIA B0 分割权重](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)：首选完整预训练权重来源。
- [NVIDIA MiT-B0](https://huggingface.co/nvidia/mit-b0)：ImageNet-only 编码器备选，不能与完整分割权重混称。
- [NVlabs 官方 SegFormer](https://github.com/NVlabs/SegFormer)：原始开源实现与模型来源。
- [Transformers 模型源码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/segformer/modeling_segformer.py)：输入卷积与解码器特征提取位置，实施时必须以锁定版本再次确认。
- [Copernicus Sentinel-2 L2A 数据说明](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S2L2A.html)：波段分辨率和数据单位/协调处理核对依据。
