# USE v4：光谱Token注入MiT-B0｜两Agent双卡执行交接

> 2026-10-27 · 用户修订：取消5k探针、正式29000步 · 当前仅计划，未启动训练

## 1、本轮决定、预算与范围

本文件按用户要求提供给下游Agent执行。它是[阶段台账](assets/research/stage_ledger.json)中 `use-v4-mit` 的分发快照；阶段目标、预算、状态、结果和下一步只以台账为准，修改先写台账再同步快照。当前只完成计划归档，工程实现、运行配置和正式命令均待A交付，不把计划当成已经实现或训练。

**用户2026-10-27确认：继续取消5k检查点探针，只评价10k、20k。** 此决定替代附件中的“六项探针、正式31000步”。5k权重保留追溯，不做探针、不参与编码器选模。只分两个Agent：A工程，B独立验收、双卡训练和报告。

| 项目 | 作业与停止预算 | 成功更新数 |
|---|---|---:|
| 工程试训 | 一个v4候选，200步；每50步无标签验证，正式重新初始化 | 200，单列 |
| 无标签预训练 | 一个v4候选，连续20000步 | 20000 |
| 冻结恢复探针 | 10k、20k各Fine和Context两项，每项1000步 | 4000 |
| 冻结USE＋轻量任务头 | 一个分割消费者，5000步 | 5000 |
| **正式研究预算上限** | **6个正式作业：1预训练＋4探针＋1分割** | **29000** |

仅验证“继承v3光谱前端→空间适配器→冻结预训练MiT-B0”的一个完整候选。v2/v3编码器不重训；不新增U-Net或第二套MiT，不做前置消融、Query压缩、Patch尺寸修改、LoRA、JEPA、教师预测。LoRA只列后续候选，不写空接口和预留代码。

20000步是首轮停止预算，不宣称充分收敛，不自动续至30k。失败、中止、校准、可丢弃测试更新及工程200步分别登记；门禁失败允许少执行，不为凑预算继续。当前请求只制定和写入计划，不启动训练或替A/B认领。

## 2、已经核实的来源与比较边界

当前仓库HEAD与附件一致：`93aa16c8c9f8e0979d4bf582db28cff4e7db0944`；工作树已有他人未提交文档，不覆盖、不回退。执行前重新记录分支、HEAD、工作树和实际源码哈希；不切换到默认main，不把单个Git提交当作全部运行身份。

| 来源 | 已核实事实 | 运行要求 |
|---|---|---|
| v3入选权重 | `outputs/use_v3/runs/pretrain/best_unlabeled.pt`，对应20k | 固定读取，不按滑坡成绩换权重 |
| 本地MiT-B0 | `weights/segformer/segformer-b0-finetuned-ade-512-512`，config与safetensors均存在 | 本地加载，禁止暗中联网或换规格 |
| 运行库 | 当前环境torch 2.11.0+cu128，transformers 4.57.6 | A锁定实际环境；变化须重验 |
| 原生MiT阶段 | 宽度32/64/160/256、深度2/2/2/2、sr_ratio 8/4/2/1 | Stage3是160，不是旧v3的128 |

已核实SHA256：
- v3 checkpoint：`e20e32f9915974c570d2ad9a064af3bd46b75f0e0d78bafbf698637895533886`。
- MiT config.json：`209caa9091e4632f7c8883c11170cd08ad29af68b23c09590aa4a5befb1a2a7f`。
- MiT model.safetensors：`6ae39addd01de6b1b8bde2cf677d43a5cd733424b8d186de3f95d1c51fee23f9`。
- 原附件：`c4ad716fbcc593fd54f0aeecddab6ca6755b8f53633055a56e98c69f3f07433f`。

工程需重验上述哈希，并把加载所需配置、权重、源码和环境全部封入release。使用safetensors这一固定权重源；不要随加载器优先级静默切换bin文件。

v3正式62000更新已经完成：统一探针入选20k，总分0.9210166；冻结v3＋完整SegFormer最佳事件等权IoU约0.1258，U-Net约0.1305；直接十波段SegFormer细节参照约0.3081。来源为既有v3审计和固定文档，历史run不改写。

v4采用外部ImageNet/ADE20K预训练MiT，并只训练轻量下游头；历史v3包含完整下游空间编码器且训练自由度不同。因此本轮是系统方案比较，不是严格单因素或纯从零自监督比较。无标签分数、分割质量、计算成本分别判断；单seed不宣称显著性。此前v3成本门禁被用户跳过不表示v4自动豁免。

## 3、数据、监督与配置锁定

无标签：沿用11132个TRAIN观测、599个PRETRAIN_VAL验证观测和239个隔离位置；每次更新6份S2＋2份Landsat。核对成员、地理去重/隔离、文件哈希、传感器统计及质量掩膜。此阶段不接入正在讨论的通用数据扩量，不恢复下载，不更换日期、GSD或来源。

滑坡迁移采用“我们的数据划分（留事件验证）”：九事件2223块训练，广岛448＋特里苏尔221块开发验证；十波段沿用559块TRAIN核心拟合的stats_core25。Italy和NewZealand不用于本轮研发/选模，不做四折或官方划分，不添加地形。只用现有单期反射率、质量/波段掩膜、可选物理元数据及空间来源。

现有v3源码实际对三种无标签视图使用metadata_mode='known'。本轮正式训练继续known，以免同时改变训练分布；partial/unknown用于200步工程稳定性检查和固定评价分层。不能宣称缺失元数据能力已被专门训练。若未来改混合训练，须另修订配置和比较解释。

A生成新的 `configs/use_v4/execution.yaml`、工程配置和校准文件，显式写出：来源、参数组、掩码、五项权重、成员/统计、预算、日程、选模、任务矩阵、设备规则和输出。正式实现配置之前均标“待实现”，不拿v3旧配置冒充适配。

实际配置及所有命令行覆盖必须经过 `scripts/check_research_protocol.py`；同时核查真实数据和前后向。成员统计、无标签校准和监督十波段stats是不同来源，不能互相替代。滑坡标签和事件ID不得进入USE接口。

## 4、唯一候选的编码器与权重边界

```text
输入影像＋质量/波段掩膜＋可选物理元数据
  → 继承v3光谱前端：Patch2、逐波段32维、波段注意力、FFN、五Query读出
      ├─ Q1–Q4＋原像元残差 → Fine_early，32通道、原分辨率
      └─ Q5，32通道、1/2网格
           → 新Conv3×3 stride2 padding1
           → 原MiT Stage1 Patch LayerNorm（只执行一次）
           → Stage1 Blocks与末端Norm → Stage2 → Stage3 → Stage4
           → 32/64/160/256通道，1/4、1/8、1/16、1/32网格
输出：Fine_early＋四尺度Context＋valid/stride/spatial_metadata
```

继承v3的tokenizer、physical、spectral_norm、spectral、ffn、queries、readout、residual、residual_out，逐张量torch.equal和哈希核对。旧inject、merges、stages、projections、feedback_gate、local不进入v4执行；旧v3最终Fine含Context反馈，不能拿它当Fine_early。

MiT跳过第一层RGB的7×7、stride4卷积；适配器直接给Stage1的32通道1/4网格。按锁定版真实的encoder.patch_embeddings / block / layer_norm分阶段调用。Stage1 Patch Norm只执行一次；Stage2–4保留原Patch Embedding/Blocks/Norm。不调用整模型pixel_values入口，不重复降采样或运行第二套空间主干。原RGB卷积在复用清单标为主动跳过，不将未加载键一概忽略。

128输入参考输出为Fine[B,32,128,128]，四尺度[B,32,32,32]、[B,64,16,16]、[B,160,8,8]、[B,256,4,4]；127/129等奇数尺寸按真实ceil/padding和最终裁剪处理。输出固定为fine、context、valid、stride、spatial_metadata及metadata_used；诊断中间Token按需返回。

MiT所有原始参数requires_grad=False并固定eval，优化器排除；**不能用no_grad包住无标签训练中的MiT forward**，梯度必须穿过它回到光谱前端和适配器。光谱前端允许更新，适配器和重建头随机初始化。ADE20K的150类classifier不属于编码器。

质量隔离：无效原始数值先归零；适配和各阶段传播空间支持掩膜，不使用标签valid。原生MiT Attention无任意空间key mask接口，必须做最小显式适配：按实际Patch卷积和sr卷积传播有效支持，屏蔽完全无支持的K/V位置；无效Query/阶段输出置零，全无效K/V行用安全占位并清零输出，样本整体全无效则拒绝。保留原预训练Q/K/V、SR、Norm和FFN权重，不重新初始化它们；混合感受野的零填补语义如实记录。

验收包含两条路径：独立RGB原生forward与原生staged forward各阶段assert_close（全有效）；v4注入路径的掩膜、数值和梯度测试。建议FP32、TF32关闭，rtol1e-5、atol1e-6；失败先定位，不放宽容差掩盖实现差异。

## 5、无标签目标与200步工程门槛

复用v3三视图：spectral隐藏可识别波段，fine同步25%像素隐藏，context按16/32/64循环空间块。保留五项目标与原有效像元筛选，权重spectral/fine/context/gradient/consistency为1/1/1/0.1/0.1，按3.2归一化。

重建器是任务模块：四尺度分别通过32/64/160/256→32学习投影、自顶向下融合形成32通道融合特征，再与Fine_early进入既有形式的输出波段条件恢复头。它不属于最终USE编码器。旧main引用在新入口明确映射为重建器的融合特征，用于原32通道归一化Context一致性；这改变了该项作用位置，须写入比较限制，不暗称完全同结构。

summary只来自可见像元。梯度误差仅计算两端原始有效且至少一端被遮挡的邻接对。保留v3空间池化口径：含隐藏像元的粗格计算原始全有效区域池化误差；单独打印可见/隐藏贡献，本轮不静默修正已知口径。光谱不可识别目标跳过并统计原因，不伪造零误差。

新网络用固定TRAIN每传感器32份、seed1234重新校准内部五项损失分母。恢复/梯度取简单基线，一致性取初始网络；分母非有限或≤1e-8报错。原始误差、归一化项、加权项、有效目标数和各路径梯度分别保存；比较不使用两版各自内部归一化训练loss。

工程试训200个成功更新，每50步无标签验证；partial/unknown在固定工程样本诊断，不改变正式known配置。校准、恢复检查及测试的可丢弃更新另外登记。正式开始从锁定v3来源＋锁定MiT＋同seed新适配器/头重新初始化，不能续用工程200步状态。

A必须证明：
- C=1/3/6/10/16；H/W=127/128/129/256；联合波段置换、pad/删除等价、未知元数据稳定、全无效拒绝。
- 预训练复用逐张量一致，Stage1注入点和单次Norm正确；hook证明没有旧v3空间主干或第二套MiT。
- 各合法目标反向可达光谱前端、适配器和MiT输入；MiT无参数梯度、字节哈希不变。不能要求初始零残差头所有层第一步都有梯度。
- 无效NaN/任意值替换不影响合法输出；质量掩膜、人工遮挡、输出支持分开；无summary泄漏。
- 同seed、保存/恢复优化器及RNG、微批累积等价通过。
- 固定TRAIN批次只读屏蔽/替换Context观察恢复变化，记录Fine捷径诊断；不当成新增训练消融或效果因果结论。

A交付READY_FOR_AUDIT；B独立复核关键工程证据、实际前后向和成本，再发布绑定release哈希的READY_FOR_TRAINING。此轮计划不实施代码或运行上述训练。

## 6、20000步预训练、四探针选模和唯一分割消费者

预训练固定：FP32、TF32关闭、有效batch8（micro2×accum4）、seed1234、head_seed1235；AdamW weight_decay0.01、clip1.0。光谱复用参数组峰值LR1e-4，空间适配器/MAE头峰值LR6e-4，MiT不进优化器；前1000步线性预热，随后余弦下降至20000步。按成功更新停止，不运行中手调。

保存initial、u5000、u10000、u20000、last，包含完整优化器、RNG、数据/配置身份；每步原始/归一化损失、传感器成员、有效目标、路径梯度与成本。原重建头每1000步验证仅作曲线，不决定USE入选。

只做四个冻结探针：probe_10000_fine、probe_10000_context、probe_20000_fine、probe_20000_context。冻结完整USE（光谱、适配器、MiT），每项1000步，AdamW LR6e-4、WD0.01、clip1、预热50步后余弦、相同batch和seed1235，末态评价，不按探针验证提前停。

Fine探针读取32通道Fine_early；Context探针读取原生32/64/160/256四尺度，用每尺度Conv1×1到32、上采样平均，再接与既有v3同宽度的恢复头。不同候选的初始探针张量一致；v3/v4的Stage3维度差使该投影多1024权重（实现时核对），明确容量局限，不增“参数匹配”实验。

在固定599个PRETRAIN_VAL、相同mask种子和known元数据下，使用**既有v3公共评价分母和原始误差定义**：每传感器分别取Fine像素、Context光谱、Context空间三类等权，空间16/32/64等权，然后S2/Landsat等权。只在10k/20k间选分数最低者，同分选较早，发布best_unlabeled及SHA256。v4内部训练分母不替代公共评分分母；附未归一化/物理单位误差。partial/unknown分层只诊断，光谱无可识别目标写未测原因。

记录Fine/各Context方差、有效秩、空间相关、跨波段子集检索，Q1–Q5两两RMS/余弦和注入分布。非有限、全常量、冻结被破坏、探针参数没有实际更新等工程失败阻断新的分割训练；1000步是否充分收敛如实报告，不自动加步，不要求必须超过v3才报告结果。

入选后仅训练一个消费者：固定ADE20K初始化的SegFormer All-MLP Decode Head（四尺度输入）＋新单通道滑坡classifier＋既有ResidualFusionHead。Fine_early直接作为32通道细节输入，不跑SpectralStem或第二套encoder。复用256通道decoder特征，原分辨率logits＝上采样base logits＋ResidualFusionHead残差，残差末层零初始化；不复用150类输出。

分割5000步，batch8/micro2×accum4、AdamW LR6e-4/WD0.01/clip1、前250步预热后余弦，沿用原增强和0.5 BCE＋0.5 Dice。头中的BatchNorm运行统计沿用现有freeze_bn规则（eval，仿射参数可训练）。阈值0.5，每500步按广岛/特里苏尔等权IoU选任务头best，另外报告last。USE全程eval且参数/buffer冻结，优化器仅含任务头；作业首尾与每次检查点前后完整SHA256，逐步检查模式和优化器边界，不每次forward拷贝全权重。LoRA不存在。

分割入选只影响任务头，不反向更换USE。报告逐事件/事件等权与像素IoU、Dice、AP、Precision/Recall、边界F1、对象尺度召回；固定Recall取0.5和0.8；按有效像素概率排序，在Recall达到目标的阈值中取最小FPR，同时报告实际Recall和阈值；不可达记未测，不插值或据此调部署阈值。

## 7、两个Agent认领与交接

工作目录：`N:/Users/yzh/Disater/landslide-seg`。先读RESEARCH_INDEX最小接力、项目AGENTS和research_policy，再查manage_agent_work；任务名不同不代表可以重复旧成果。全部代码编写/审查使用[ponytail skill](C:/Users/yzh/.codex/skills/ponytail/SKILL.md)，优先复用组件，不改旧release绑定代码。

**A／use-v4-mit-engineering**：代码、配置、测试、200步试训和不可变release。精确认领：
- src/landslide_seg/models/use_v4_mit.py、src/landslide_seg/models/use_v4_reconstruction.py。
- scripts/verify_use_v4_mit.py、scripts/pretrain_use_v4.py、scripts/evaluate_use_v4.py、scripts/benchmark_use_v4.py、scripts/run_use_v4.py。
- tests/test_use_v4_mit.py、tests/test_use_v4_masks.py、tests/test_use_v4_loading.py。
- configs/use_v4、outputs/use_v4/engineering、outputs/use_v4/releases。

可把小型辅助逻辑留在上述已有新文件；确需其他文件先补精确认领，不占整个src/scripts/tests。A不认领正式runs或共享研究文档，不启动正式实验。

**B／use-v4-mit-training**：独立验收、双卡队列、结果与固定文档。精确认领：
- outputs/use_v4/runs、outputs/use_v4/runtime、outputs/use_v4/audit、outputs/use_v4/benchmarks。
- docs/assets/research/stage_ledger.json、docs/assets/research/use_v4、docs/MODEL_EVOLUTION.md、docs/DATA_EVENT_PROTOCOL.md、docs/RESEARCH_INDEX.md。
- docs/USE_V4_MIT_EXECUTION_HANDOFF.md仅用于必要的同源快照同步。

```text
python -X utf8 scripts/manage_agent_work.py list
python -X utf8 scripts/manage_agent_work.py claim <任务ID> --owner <真实聊天ID> --title <职责> --scope <精确相对路径>
```

模板必须替换占位符，多个范围重复--scope。不替其他Agent认领，不使用别人的owner，不关旧v2登记。已完成ID不可重复claim，修复用新ID；block写清原因/恢复证据。一个文件只有一个活动写入者，严守共享锁。

交接：A交付源码/依赖/配置/全部覆盖/成员/统计/初始化/预算/选模/真实命令与哈希，封装READY_FOR_AUDIT。B独立核查并生成绑定同一release_sha的READY_FOR_TRAINING，才开始正式队列。B遇代码问题交回A，不自行改训练代码；受保护文件改变使原门禁失效。因修复失败导致的额外更新单列，不擅自重跑已完成同配置任务。

## 8、双卡队列、成本和大图门禁

B统一管理两个物理GPU UUID，一卡一个计算进程，不使用DDP，不翻倍单实验batch。运行前核验可见设备映射、真实进程、显存和前后向，使用现有每卡锁；显示占用不拒绝，不终止别人的进程。仅空闲4096MiB旧阈值不够，必须覆盖实测峰值及余量。

| 顺序 | GPU 0 | GPU 1 |
|---|---|---|
| 工程 | 原生/staged参考和必要接口测试 | v4工程200步 |
| 成本验收 | 本轮计算暂停 | v3/v4/直入同卡串行测速 |
| 正式开始 | 等待10k权重；可做CPU日志核对 | v4预训练20000步 |
| 10k已完整保存 | 10k Fine探针→10k Context探针，各1000 | 继续v4预训练 |
| 20k预训练结束 | 20k Fine探针1000 | 20k Context探针1000 |
| 无标签选模通过 | 空闲 | 唯一分割头5000 |
| 最终成本复核 | 本轮计算暂停 | 同卡串行复核最终权重成本 |

只有原子保存完成且哈希核对的检查点可被探针读取。预训练期间不运行滑坡分割。依赖决定了部分时间空闲，不为了占满两卡增加实验或DDP。

成本矩阵：C=3/6/10/16、128/256/512，基础batch1 FP32、TF32关闭；batch8先静态预检。每项30次预热、100次同步测量、3轮独立重复，报告中位/p90、图像每秒、峰值allocated/reserved、参数量/激活和输出字节、Top10及MAC/FLOPs覆盖。裸编码器、冻结封装、完整消费者、IO、权重哈希/CPU拷贝分开；三视图训练步拆为前向、反向、优化器。测训练成本用可丢弃副本，不改正式权重。

对照分两类：v4裸编码器对v3裸编码器；v4＋轻量头对历史v3＋完整SegFormer及直接十波段SegFormer。真实十波段消费者只测C10，其他波段只做编码器，不伪造可用接口。历史权重固定，统一同卡重测所需成本，不重训旧模型。128正式micro2训练必须放得下；512 B1 C10纯编码器和完整消费者冷/热前向必须成功。256/512训练成本先静态预检，不安全则记未测/容量不足，不强迫分配。

将附件“显著更贵”具体化为工程警戒：同条件512、B1、C10，若裸编码器或完整消费者的中位耗时/峰值allocated比对应v3路径高20%以上，先阻断正式放行、检查重复编码/网格/哈希/转置，A仅做数学等价优化，B复测。20%是资源警戒值，不是统计显著性。三轮耗时明显不稳定则记不可信并查负载，不能挑最快一轮。仍不满足则提交原因和未执行预算，不悄悄换模型或扩大预算。

512通过后，1024先静态预算，安全时最多一次前向；2048/4096/10000只做静态及切片估算。切片为512、重叠64、步长448，维度L窗口数为ceil(max(L-512,0)/448)+1：4096为9×9＝81窗，10000为23×23＝529窗。报告padding和重叠计算放大、边缘有效上下文及拼接/IO开销；切片不能恢复跨窗注意力。MiT序列缩减注意力仍非严格线性，不宣称无限分辨率。

## 9、报告、回写和最小复现交付

B一次给出最多三张合并表：无标签/探针及分布诊断、冻结分割best/last与逐事件、分辨率成本。模型行名用“SegFormer B0｜光谱Token注入冻结MiT｜All-MLP＋Fine残差”等架构说明，代号只作追溯。每个主要结论配架构、曲线、固定案例或诊断图，解释机制/证据/边界，推测写未验证。

固定六例沿用现有案例ID和坐标，所有模型使用相同输入/有效掩膜；再按预登记规则列高误报案例，不能凭视觉挑好图。图展示原图、Fine/Context表征、真值、概率、0.5预测；各模型独立PCA颜色不作跨模型语义等价。报告外部监督来源、单seed、20k是否收敛、探针参数差异、Fine_early与旧Fine不同、冻结范围和消费者训练自由度。

产物含：加载复用/跳过/随机/冻结清单；源码/配置/覆盖/环境/成员/统计/预训练/检查点SHA256；测试与恢复记录；逐步损失和梯度、曝光账本；四探针评价、USE选模、任务头best/last；成本原始CSV/JSON、实际GPU小时及成功/中止/失败更新。

阶段目标、计划、观察、结果和下一步只更新stage_ledger的use-v4-mit；模型连接/真实加载到MODEL_EVOLUTION，指标和案例到DATA_EVENT_PROTOCOL，入口仅短接力。图在docs/assets/research/use_v4，运行在outputs/use_v4。不追加日期报告，不回写旧run，不手改生成区。
```text
python -X utf8 scripts/query_research.py --refresh-stages
```
新增run索引时才刷新inventory；核验固定四份文档链接、计划快照与台账一致及生成幂等。最后finish附真实证据，未通过记block/失败/未测，不能以进程退出代表验收完成。

A必须交付能直接运行的最小复现命令，覆盖prepare/verify/engineering/freeze/run/probe/select/segment/evaluate/benchmark，命令以最终实现为准；本计划不编造尚不存在的CLI标志。

若只恢复改善、分割未改善，结论为迁移价值未证实；若质量改善但成本过高，说明权衡；不自动追加LoRA、Query/Patch修改或新预算。

## 10、直接分发给两个Agent的提示词

### 给A：工程Agent

请读取 N:/Users/yzh/Disater/landslide-seg/docs/USE_V4_MIT_EXECUTION_HANDOFF.md 和项目最小接力，以用户最后确认的“取消5k探针、只评10k/20k、正式29000步、工程200步另列”为准，不照抄原附件31000预算。所有代码工作使用ponytail skill。先查认领，以真实聊天ID认领use-v4-mit-engineering的精确范围。

只实现一个候选：继承固定v3的光谱前端和五Query，保留早期Fine，第五Query经新stride2空间适配器接本地预训练MiT-B0。跳过RGB首卷积，只调用一次原Patch Norm；完整复用并冻结MiT原始参数，梯度仍可穿过它回到上游。旧v3空间主干和第二套MiT均不运行。无LoRA、JEPA、Query/Patch修改，不重训v2/v3。

完成三视图五项目标、四个统一探针、已有All-MLP解码头＋Fine残差消费者、掩膜隔离、权重/梯度/恢复测试和200步工程试训。正式known元数据；partial/unknown作工程和分层诊断。冻结实际源码、环境、配置、来源、成员、统计、初始化与预算，交付可直接运行的命令、证据和READY_FOR_AUDIT。不要启动正式队列。B发现代码问题仍由你修复，改release后重新验收。

### 给B：验收、双卡训练和报告Agent

请读取 N:/Users/yzh/Disater/landslide-seg/docs/USE_V4_MIT_EXECUTION_HANDOFF.md 和项目最小接力，先查认领，再以真实聊天ID认领use-v4-mit-training的运行与固定文档范围。你统一负责独立验收、双卡正式队列和结果回写，不修改A的工程代码。

核验固定v3的20k权重及本地MiT-B0来源、逐参数复用/冻结、staged数值对齐、掩膜无泄漏、梯度穿过MiT、200步工程证据、实际配置预检和512同卡资源门禁；旧v3曾跳过成本门禁不等于本轮免验。通过后发布绑定release的READY_FOR_TRAINING。

仅执行正式上限29000更新：一个v4无标签20000；10k/20k各Fine、Context探针1000，共4000；无标签选定后唯一All-MLP＋Fine残差任务头5000。禁止5k探针、旧编码器重训、U-Net、第二套空间编码器或LoRA。GPU1预训练，GPU0在10k原子检查点保存后串行做其两个探针；20k两个探针分卡，随后只跑一个分割任务。不用DDP，不为占满卡加任务；受阻交A修复。

最终核对USE与MiT冻结哈希、选模、预算、初始化、曝光及实际成本，交付三张合并表、曲线和固定案例。更新唯一台账、固定模型/数据文档和总入口，说明外部监督初始化、探针容量和消费者自由度差异；失败、未测和未完成如实记录。
