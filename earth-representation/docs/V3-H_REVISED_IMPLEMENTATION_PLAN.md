# Encoder V3-H-R 实施、检查与实验矩阵

基线分支为 `dev/encoder-v3-sensor-memory-v1`。本阶段使用独立工程配置 [config_v3_h_revised_engineering_v1.json](../configs/config_v3_h_revised_engineering_v1.json)。设计契约、张量、共享关系和复杂度见 [修订设计](V3-H_REVISED_DESIGN.md)。

## 实现和验收

| 阶段 | 实现和信息流 | 验收证据 |
|---|---|---|
| 0 | 独立配置。按 Profile 绑定成员、TRAIN统计、物理元数据及旧掩码代码哈希。候选范围与预算见设计 | 工程配置可解析。记录工程 Profile 的 SHA256 |
| 1 | 共享 Descriptor 与卷积核生成，每可见通道一次，卷积独立 | 类型、缺失值、来源置换、卷积核差异与隐藏值隔离 |
| 2 | 原生索引、正面积跨通道边、共享稀疏注意力 | 稠密参考输出及梯度、224通道邻接上限、无邻居残差 |
| 3 | 每 `(Middle, channel)` 逐通道压缩；同通道空间与异通道交互；联合分支 | 双路径张量形状、多 Query、隐藏通道缺席及反向路径检查 |
| 4 | 全球固定分区，Regional→Global，Global 内部 Self-Attention | 全球 ID、负坐标、旋转、日期变更线、极区及配对数 |
| 5 | 目标条件化的局部和粗尺度两次读取 | A 与旧掩码逐像元对照，B1 光学限定，改变隐藏值不改变预测，预测对隐藏值梯度为零 |
| 6 | 合成过拟合、保存恢复、分块梯度与 CUDA Profile | 测试命令与结果见下节，真实 Profile 单列 |
| 7 | 首个逐像元A+B1正式Run已确认工程模型容量、等权目标和1 Epoch预算；A-only/B1-only等其他实验仍为候选 | 独立正式配置、冻结源码与真实恢复门禁；不据此宣称整体质量/效率问题解决 |

旧 A 的 `mask_patch_metres=128` 和 `mask_ratio=0.55` 独立于 Middle/Regional 步长。B1 以现有采样器从至少两个有有效像元的光学通道选取整波段。按目标有效像元平均每通道 MSE，然后对目标通道等权汇总。训练包装器提供单观测、单任务损失与逐通道结果。联合实验需按实际可执行任务对权重重新归一，并对观测等权。

## 复现

使用项目的 Python 3.12 环境。本次工程验证使用 PyTorch 2.11.0+cu128、Shapely 2.2.0、pyproj 3.8.0。

```powershell
python -B -I -c "import ssl,pytest;raise SystemExit(pytest.main(['tests/encoder_v3/hierarchical_revised','tests/encoder_v3/hierarchical','tests/encoder_v3/shared_mae','-o','pythonpath=src','-q','-p','no:cacheprovider']))"
python -B -I scripts/run_v3_h_revised.py synthetic --task A --output outputs/encoder_v3/hierarchical_revised/new_synthetic.json
python -B -I scripts/run_v3_h_revised.py real --device cuda:0 --sensor EnMAP --rank max --task B1 --geometry-cache outputs/encoder_v3/hierarchical_revised/cache --output outputs/encoder_v3/hierarchical_revised/new_enmap.json
```

`run_v3_h_revised.py`仍只支持合成和真实样本工程Profile。正式入口独立为`train_v3_h_revised.py`，要求明确授权配置、冻结源码发布和匹配的真实恢复门禁。每次迭代独立记录前向、反向和 AdamW 更新。大样本可用 `--offload` 将反向保存张量卸载到主存。参数与优化器状态仅属于本次工程进程，Profile 结束即丢弃。使用新输出路径避免覆盖证据。

Profile JSON 记录各阶段 Token、Latent、Q/K/V、配对及度分布。理论 FLOPs 按前向乘加计算，未包含 softmax、归一化、掩码等逐元素运算。CUDA phase event 覆盖设备流时间区间，也可能包含主机提交间隙。Gather/Scatter event 为前向边算子的索引区间，CPU 索引准备与复制提交另列；其余反向阶段计入整体事件与 wall-clock。峰值显存含该进程的 PyTorch allocated 和 reserved。

## 效率复审状态

旧容量测试未构成整体效率放行。当前102项测试通过；已修复Decoder共享上下文的重复展开、空交叠候选检查和通道条件投影重复计算。独立共享粗尺度候选完成三种子、等初始化及等更新次数的工程消融，但未证明质量无损，默认模型未切换，详见 [设计与效率复审](../outputs/encoder_v3/hierarchical_revised/efficiency_v2/AUDIT.md)。执行块与逻辑Attention配对数分开计量；新分组路径的Gather/Scatter独立计时为空值，不能解释为零成本。

## 已验证结果

首次工程验收89项测试通过，其中V3-H-R新增17项。效率复审后的最新测试、真实输出／梯度等价验证及速度结果见[复审报告](../outputs/encoder_v3/hierarchical_revised/efficiency_v2/AUDIT.md)。旧V3-H和共享MAE的72项回归也纳入复审。合成 A/B1 均通过前向、反向和工程更新；小型联合目标测试同时验证过拟合及下一步恢复。

S2、L8、L9、EnMAP 的中位和最大样本共16个 A/B1 CUDA 工程视图全部通过。最大 S2 的 A 和 B1 分别达到15.80和10.64 GiB峰值 allocated，含冷启动的前向和反向合计约375和321秒。该容量结果绑定工程种子17和单观测执行，未覆盖所有随机掩码或正式逻辑批量。

真实样本的逐任务结果、索引耗时、显存与配置绑定见 [工程 Profile](../outputs/encoder_v3/hierarchical_revised/engineering_v1/PROFILE_REPORT.md)。候选尺度的清单普查见 [预算数据](../outputs/encoder_v3/hierarchical_revised/engineering_v1/candidate_budget.json)。

## 独立共享粗尺度候选的工程消融

用户另行授权 `shared_coarse` 候选与质量消融，验证后再决定采用。运行入口仍只允许工程配置；新配置不替代原逐像元配置，不开启正式训练。

实现：固定Regional支持签名缓存、按目标通道分组、共享粗尺度读取、逐像元局部读取与原生标量输出。具体信息流与复杂度见设计文档。工程测试增加逐边参考梯度、隐藏值和输入梯度隔离、固定支持跨界身份、持久缓存、通道置换、强制卸载和下一步恢复。

配对消融使用三个初始化种子17/29/43，各版本16次完整模型更新；同一冻结TRAIN中S2/L8/L9/EnMAP的原生容量中位观测，任务A/B1轮流覆盖，目标掩码复用原规则。每个种子两版本初始参数哈希必须相同。VAL小面板只从既有S2/L8各选择容量四分位的三个完整观测；EnMAP/L9另标TRAIN异掩码诊断，不算泛化验证。未裁剪观测或改变数据发布。

原始逐通道损失、配对初始化、工程权重与源码快照存于 `outputs/encoder_v3/hierarchical_revised/shared_coarse_candidate_v1/paired_quality`。完整验收结论和容量数据见[复审报告](../outputs/encoder_v3/hierarchical_revised/efficiency_v2/AUDIT.md)。16次更新和六个VAL观测不能证明收敛后无损；不据此自动采用候选。

## 实验矩阵

| 候选 | A | B1 | 汇报 |
|---|---:|---:|---|
| A-only | 1 | 0 | A原生误差，另报B1泛化读数 |
| B1-only | 0 | 1 | B1光谱误差，另报A泛化读数 |
| A+B1 | 0.5 | 0.5 | 分别报告A/B1，权重为候选 |

三组独立随机初始化，固定数据成员、QA、TRAIN 统计及验证掩码种子。成员不适用某任务时报告原因及覆盖率，不改变冻结清单。报告分任务、分传感器、分通道指标及相同设备上的计算成本。当前 VAL 仅 S2 与 L8，EnMAP/L9 工程样本用于容量检查，不代表泛化评价。

## 首个正式A+B1 Run（已按用户要求暂停）

用户在获知效率/短消融质量限制后明确要求正式训练，并确认本次参数：随机初始化的逐像元基线；Fine/Middle/Coarse=32/64/64，其余模型工程参数不变；A/B1各0.5；AdamW学习率1e-4、weight decay 0.04；FP32、cuda:1、逻辑batch32、逐观测梯度累积；12步预热后余弦下降到1e-5；仅1 Epoch，随后完整VAL并停止。无额外梯度裁剪，不加载工程消融或历史权重。

独立配置为[config_v3_h_revised_formal_epoch1.json](../configs/config_v3_h_revised_formal_epoch1.json)，发布位于[formal_epoch1_v1](../outputs/encoder_v3/hierarchical_revised/formal_epoch1_v1)。167份源码/配置/必要元数据冻结到runtime；原数据仍以只读绑定读取。106项测试通过，新增正式目标权重、7442成员全覆盖/233批/尾批18、调度端点、恢复及完成预算拒绝重启测试。冻结运行时的S2/L8真实前后向与下一步恢复检查通过，最大参数差3.58e-7；门禁权重丢弃，正式权重重新初始化。

Run ID：`099bc7001f58477f94618b3a372c37de`。唯一当前状态见[status.json](../outputs/encoder_v3/hierarchical_revised/formal_epoch1_v1/formal_run/status.json)。每批保存完整状态，记录观测/任务资格与逐通道误差；结束后按原383 VAL成员逐项报告A/B1。没有任务资格的成员保留资格记录，整个批次无任务则显式失败，不能静默删成员；恢复时不擅自重放结果不明确的优化器步骤。

本次正式授权不将之前未完成的全池吞吐、收敛后质量或SAR容量检查改记为通过。完整Epoch耗时以本轮运行记录为准。共享粗尺度候选未采用。

## 条件续训安排：额外10 Epoch（已暂停）

用户随后明确确认：当前1 Epoch验收通过后，从其最终检查点再续训10 Epoch，累计11 Epoch。续训保留模型和AdamW状态、A/B1等权及既定数据/掩码契约，使用新Run ID；学习率按新的10轮预算重新执行12步预热、1e-4余弦降至1e-5。训练采样从全局epoch 1开始，避免新Run误用epoch 0的划分与隐藏视图；续训每轮均做完整383 VAL。

自动放行需同时满足：既有架构/泄漏/恢复检查通过；7442 TRAIN与383 VAL完整覆盖；训练与验证损失、梯度有限；最终检查点和源码/配置/数据绑定一致；A和B1完整VAL均值都严格低于相同成员、相同掩码的初始权重对照。逐通道差异另报。不通过则停止，不放宽标准。

初始完整VAL对照已保存在[initial_validation_v1](../outputs/encoder_v3/hierarchical_revised/formal_epoch1_v1/initial_validation_v1/report.json)：A=2.1874394534，B1=2.1810137814，均覆盖383观测；未进行优化器更新。该指标是随机初始化基线，不是训练后成绩。

验收脚本`verify_v3_h_revised_acceptance.py`只读检查并生成结论；`prepare_v3_h_revised_continuation.py`仅在已完成父Run和全部验收通过时生成10轮配置。新训练入口核验父检查点与验收报告哈希，禁止继承不同模型或数据来源，保留AdamW动量并重置本次独立预算计数。用户随后因训练时间过长明确暂停。父Run已STOPPED于42/233次更新，累计1344/7442训练观测，墙钟1507.48秒（25.12分钟）；尚未完成整轮或训练后383 VAL，不能执行最终验收，也不得自动启动10轮续训。检查点、初始对照、更新日志及167份冻结文件保留；恢复须有新的用户指令。运行冻结源码不修改。暂停时核验last.pt游标42、AdamW步数42及167份冻结文件哈希；推送前当前工作树111项回归测试全部通过（26.22秒）。完整Epoch耗时和训练后质量仍未验收，不能用测试通过替代。
