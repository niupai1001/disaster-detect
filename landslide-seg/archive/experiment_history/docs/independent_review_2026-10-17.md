# Landslide-Seg V1 独立架构与工程审阅

审阅日期：2026-10-17（用户时区）。范围：阅读用户提供的 `C:/Users/yzh/Downloads/Landslide-Seg_v1.md`，核对当前代码、官方实现与文档，实际运行独立诊断。MD 用作研究验收依据；本轮用户请求是检查审阅，因此未修改生产模型、数据协议、原训练脚本及原有验收报告。

## 结论

完整架构的主要模块确实存在，并且真实权重的 GPU 接口链路可运行。当前仍有会改变训练目标、输入协议、训练/推理行为及指标含义的问题，**不建议接受 README / M1_status 中“全部 DONE”的结论，也不建议直接开始正式 M2 或模块贡献消融**。

独立判定：M0 = PARTIAL；M1 = PARTIAL，真实数据合规验收尚未通过；M2 = 未具备正式实验条件；M3 = 尚未开展。这里没有判定 Sen12Landslides 数据本身不可用，而是现有回补与验收证据不足以证明它满足全部要求。

## 实际检查与证据

- 专用 Python 环境实际可执行：Python 3.13.14、torch 2.11.0+cu128、terratorch 1.2.13、timm 1.0.30、rasterio 1.5.2、pyproj 3.8.0；检测到两张 RTX 5090。启动时有 Python 基路径定位警告，但本次没有阻止程序运行。
- `python -m pytest tests -q`：10 passed。现有测试不能覆盖下面列出的训练/数据/指标问题。
- 官方权重加载、合成输入完整 forward/loss/backward、下游参数更新、checkpoint 保存与同实例重载均实际运行。总参数 525,990,558，可训练参数 3,460,337；本次合成诊断 CUDA 峰值 allocated 约 3.15 GiB，仅代表 B=1 / 128×128 / 当前冻结配置。
- 独立包裹官方 `sampler.generate` 计数，三次完整前向累计调用三次，证明 TiM 生成真实执行；不是仅检查配置中的 `tim_ran=True`。
- AnySat 40m / 160m、TerraMind TiM、文本类别原型、融合、粗细头同时存在。输出 fine/coarse `[1,1,128,128]`，presence `[1,1]`。
- 固定 **Python random 和 torch 两套随机状态**，仅改变训练用 region_mask，fine logits 最大差值为 0。本次未发现该接口把 GT 泄漏到预测；仅固定 torch seed 不够，因为官方 TiM 使用 Python random 产生采样 seed。
- 读取六个真实 nc 文件核对 time、维度、x/y、pre_post。`chimanimani_s2_1001.nc` 的 post=9，对应真实日期 2019-03-20，正确零起点 DOY=78，当前加载器返回 79。数组维度确为 `(time,x,y)`，因此与常见栅格 `(row,col)` 的转置有元数据依据，不应简单删除转置。
- 默认沙箱读取 STAC 遇到 WinError 10013。只读公开数据访问在允许网络的执行上下文中成功：获取 `S2A_36KVC_20190320_0_L2A`，保存本次原始诊断输入；有效像素比例约 96.90%。B08 对照相关系数 0.974364、平均偏差 -25.10 DN、RMSE 132.28 DN。**高相关不等于同产品、同辐射处理或逐像素网格验收通过**。
- 原有日志 `outputs/logs/real_inference.json` 中 post_date=2018-09-06 是日志字段错误；实际 post 日期为 2019-03-20，不能据此误判模型输入用了灾前时相。

- 对上述真实输入实际完成一次完整训练步、保存、同实例重载与推理，状态为 `REAL_RUNTIME_PASSED_DATA_COMPLIANCE_UNVERIFIED`；fine 输出 `[1,1,128,128]`，固定随机状态的重载前后概率最大差值为 0。这证明当前真实输入路径可执行，尚未证明该输入满足同源产品/辐射/几何的正式验收，也没有替代全新模型实例重建验证。

原始独立证据：`outputs/review/checks.json`、`unit_checks.json`、`backfill_checks.json`、`real_runtime.json` 以及对应 console 日志。所有独立输出均在 `outputs/review/`；原检查点没有被覆盖。

## 必须优先解决的问题

### R1 / P1：把“同一天候选产品”自动认证为“同一观测”

位置：`src/landslide_seg/data/s2_backfill.py:59`、`:66`、`:150`；`data/sen12.py:255`。

检索仅用中心点与日期，多个候选按整景云量排序，未与原始 10 波段的 product/granule/datatake 身份匹配。随后 `assemble_full` 无条件写入 `fully_observed=True`。训练路径没有重叠波段一致性验收门槛；推理路径的 `align_check` 只打印 B08 相关值，不会拒绝不合格样本。

网格也不是从读取到的 x/y 与维度构建，而是写死 128×128、10m 和中心±640m。`read_asset_window` 假定源栅格 CRS 与目标 bounds 相同；未校验不同 MGRS/UTM 候选。源窗口可能包含小数尺寸，而读取后实际数组尺寸与 `src.window_transform(win)` 的尺度也需要检查。是否已经造成当前样本偏移，**尚未证明**。

本次样本的源产品 baseline 为 02.11，不能把后续 baseline offset 风险反推成该样本已经存在 offset 错误。实际问题是代码没有处理和校验通用产品的 radiometric scale/offset、原 10 波段处理基线、完整覆盖范围和逐波段误差阈值。

修复标准：保留候选产品完整身份及资产元数据；建立源观测关联，或采用同一个可追溯原始产品的全部 12 波段；从真实坐标构建网格；做多波段偏差/RMSE/像素位移与覆盖率验收，失败即 quarantine。合规认证与“通道都存在”必须是不同状态字段。

### R2 / P1：训练和推理都主动剔除所有空图块

位置：`data/pipeline.py:191`、`:207`；`scripts/real_inference.py:88`。

`min_mask_px=1` 会丢弃真实空 mask；`real_inference.py` 也跳过空 mask。结果是 presence 没有正常真实负样本，无法验证空图拒检和无滑坡图块误报率。这违反文档明确要求；5 步训练 loss 下降不能证明该问题无害。

修复标准：保留经核实的负图块；区分“未标注/未知”与“真实无滑坡”，尤其本地 `1000.nc` 的 annotated=False、事件字段为空，不能直接把它认证为可靠负样本。保存 label_status 与有效区域，采用可追溯的正负采样策略。

### R3 / P1：区域语义训练只有正例，空区域反而产生非零嵌入

位置：`training/trainer.py:74`、`:82`；`models/semantic.py:152`；`training/losses.py:123`。

注释宣称使用“滑坡区域 vs 同图背景”对照，但 trainer 每张图只池化 GT 前景区域，只给一个是否存在滑坡的 region_label。真实空图又被 R2 过滤，因此该真实数据训练路径实际都是语义匹配正例，投影可通过向单一原型靠拢降低损失，不能证明学会前景—背景语义区别。

即使补回空图，空 mask 池化后仍通过含偏置的 region_proj，独立实测空区域嵌入范数约 1，而非文档声称的零；它代表网络偏置，不是背景视觉特征。此外 region_mask 没有与 valid 相乘，云遮挡/未知像素可进入语义监督。

修复标准：显式构造有效前景与有效、可靠背景区域；按区域可观测性跳过空区域；训练时至少包含有意义的正负对照。属性原型目前只注册未参与计算、无属性损失是透明记录的状态；缺可靠属性标注时保持类别训练符合研究范围，不应造属性描述。

### R4 / P1：冻结权重没有保持冻结模块的 eval 状态

位置：`training/trainer.py:68`；`encoders/anysat_scale.py` 与 `terramind_tim.py` 的构造和 forward。

构造函数里调用一次 eval 后，trainer 的 `self.model.train()` 会递归将所有骨干设回训练模式。独立实测 TerraMind / AnySat / 文本塔的 training 均为 True，参数 requires_grad 均为 False；S2 实际路径中的 attention Dropout(p=0.1) 被启用。

`no_grad()` 不能阻止 Dropout，也不能防止存在 BatchNorm 时的 running statistics 更新。本次已证明 S2 Dropout 生效；没有声称所有骨干均实际发生了 BN 漂移。

修复标准：覆盖 train() 或在每次切换训练状态后显式固定冻结骨干 eval；用实际 S2 分支输出和运行状态测试。骨干若未来解冻，也必须同时修改无条件 `@torch.no_grad()`，不能只设置 freeze=False。

### R5 / P1：质量掩膜没有覆盖全部监督目标

位置：`training/losses.py:114`；`training/trainer.py:74`、`:82`。

当全图 valid=0 时，coarse/fine 分割损失为 0，但 presence 仍生成“无滑坡”标签并产生损失。独立实测 presence=0.693147，总损失=0.138629。这会把不可观测图块当负样本；滑坡全部被云遮挡时也有同样的标签含义问题。

语义区域标签按原始 mask 判断，而 presence 按 mask×valid 判断，两项可互相矛盾。需要由 label_status、有效覆盖及正区域可见性决定哪些任务可以监督，不能用一个像素掩膜自动推断整图真实阴性。

另有独立边界错误：valid=None 时 `_reduce` 直接返回未归约 BCE 图，CombinedLoss.total 不再是标量；默认 backward 会失败。当前 Trainer 总是传 valid，因此主路径暂时未触发，但损失函数公开接口不成立。

### R6 / P1：无效像素被算进 TN，FPR 偏低

位置：`training/metrics.py:11`、`:25`。

先执行 p&valid 和 g&valid，再算 `~p & ~g`，使所有无效像素变为真负例。独立实测全无效 2×2 输入返回 TN=4，应为 0。IoU 的 TP/FP/FN 在这一实现下不因这个 TN 问题直接变化，但 pixel FPR 的分母会虚增；不能笼统声称所有指标都错。

此外当前 `false_positive_rate` 是背景像素级 FP/(FP+TN)，不是文档要求的“无滑坡图块误报率”。需要分别实现并命名。

### R7 / P1：AnySat 日期索引有确定的 off-by-one

位置：`data/sen12.py:60`、`:69`。

官方 README 定义 01/01=0。当前 `tm_yday` 从 1 起，实测 2019-03-20 返回 79，应为 78。缺时间基准时还静默返回 180；pre_post 解析失败静默选择末时相，没有标为隔离。真实协议路径应拒绝未知日期/事件选择，不能用占位日期通过验收。

## 其他工程与架构问题

### R8 / P2：正式训练配置和数据划分尚未接通

`scripts/train_full.py` 不读取 `configs/model_full.yaml`，而是 `build_model()`、`CombinedLoss()` 默认值；更改 YAML 的尺度、hidden、冻结或损失配置不会改变这条训练路径。Sen12 分支固定生成 batch=1，`--batch-size` 对它不起作用。

脚本在 `batches[0]` 上评估，属于训练样本回看；没有事件/地理互斥清单、独立 val/test、best checkpoint、逐样本评估导出或正式 resume CLI。没有固定完整随机状态。它可作为短步 smoke 训练入口，当前不能当作可复现的正式 M2 训练流程。

### R9 / P2：有不可达的可训练层、尚未验证的多层融合

`models/coarse.py` 在自顶向下融合后计算各层 smooth，但损失输出只使用 smooth 后的最细层与最粗层，中间两个 smooth 的结果不参与最终预测。独立 coarse+presence backward 证实 `smooth.1`、`smooth.2` 的六个参数 grad=None。

`full_model.py:152` 仅用 TerraMind 选出层中的最后一层；其余三层被提取但未融合。AnySat 确有多物理尺度调用，因此完整骨干并未缺失；但“TerraMind 多层特征参与融合”的表述不成立。金字塔是对同一组来源插值/降采样得到，可作为起点，不能声称每一层都包含不同骨干层级的证据。

`FeatureMap.stride_m` 是记录字段，融合器没有用它检查覆盖范围、原点和网格；现有脉冲测试的 30 个种子均通过，但它仅证明合成同网格插值附近峰值，不证明真实双骨干地理配准。

### R10 / P2：检查点和性能日志的适用范围较窄

默认保存时无条件排除骨干与文本权重，即便将来它们可训练也会被丢弃；权重 revision/hash、预处理、模型配置与数据清单未作为必要身份信息绑定。现有独立 SHA256 已存入 checks.json，但未与官方发布文件校验值匹配，所以它仅是本地版本指纹。

AnySat vendor 是源码快照，未见 Git commit 锁定；pyproject 只有宽泛下界，缺 rasterio/pyproj 等实际真实数据依赖。TerraMind 在全局 HF 缓存取权重，不能仅凭仓库 weights 目录文件存在推断实际加载的是该文件。当前特殊 import shim 和镜像缓存 fallback 可运行，但 largest-blob 复制策略没有严格验证该 blob 就是目标文件，应改成精确文件与 hash 解析。

trainer 的 latency_s 只量前向，CUDA 无同步，未覆盖 backward/optimizer；不能拿它解释训练步耗时或 GPU 推理延迟。TiM 本身有随机采样，应固定并记录随机策略。

`real_inference.py` 重新 build 未训练下游头、不加载训练检查点，它的预测图不能当作刚训练模型的推理结果。日志 post_date 只写了时间基准而未加当前 time 偏移，需修正可追溯性。

## 建议的修复顺序与验收

1. 先完成 R1 的样本认证与隔离，修复日期协议；保存一份可追溯、网格和辐射误差有阈值的真实小清单。
2. 修复真实负样本采样、valid/unknown 的任务级监督、前景/背景语义配对与指标。
3. 固定骨干 eval，补充实际训练状态检查；接通 YAML、配置身份和检查点元数据，处理无梯度层。
4. 在认证后的同一真实输入上重新执行全部模块 forward/backward、全新实例 reload、无 GT 推理与可视化。验证负图可输出空结果，指标只统计有效像素。
5. 建立事件/空间互斥 train/val/test 后再开展 M2；完整模型验收通过后才做 M3，不需要删除任何必选模块。

## 可复现命令

以下命令工作目录为本项目，`python` 指本次使用的专用环境解释器：
`C:/Users/yzh/.workbuddy/binaries/python/envs/tm/Scripts/python.exe`。

```powershell
python -m pytest tests -q
python scripts/review_checks.py
python scripts/review_unit_checks.py
python scripts/review_backfill_checks.py
python scripts/review_real_runtime.py
```

`review_checks.py` 的真实网络部分可能在默认沙箱被拒绝，错误已保留；公开数据回补复测由单独的 `review_backfill_checks.py` 记录。`review_real_runtime.py` 依赖前一步产生的真实输入 npz。这些诊断不会替代真实数据认证，也不构成精度实验。

## 官方依据

- AnySat 官方输入、日期零起点、dense 输出要求：https://github.com/gastruc/AnySat
- TerraMind 官方与实际安装的 TerraTorch 实现：https://github.com/IBM/terramind
- TerraTorch 模态、波段与 TiM 文档：https://github.com/torchgeo/terratorch/blob/main/docs/guide/terramind.md
- Sen12Landslides 官方：https://github.com/PaulH97/Sen12Landslides
- RemoteCLIP 官方推荐 OpenCLIP 加载：https://github.com/ChenDelong1999/RemoteCLIP

本项目通过 clip-anytorch 加载成功，这证明当前路径可执行；尚未进行与官方 OpenCLIP 文本输出的数值等价对照，不能把“使用不同加载库”本身判作模型错误。
