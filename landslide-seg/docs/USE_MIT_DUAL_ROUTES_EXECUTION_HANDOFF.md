# USE双路线：预训练MiT迁移｜两Agent双卡执行交接

> 2026-10-28 · 正式59000步上限 · 编码器固定20k末态 · 当前仅计划，未启动实验

## 1、本轮目标、主矩阵与固定预算

本文件是用户要求的可分发执行快照，唯一阶段事实和计划源是[阶段台账](assets/research/stage_ledger.json)中的 `use-mit-dual-routes`。当前只制定和写入计划，尚未实现双路线或启动训练；附件中的执行命令不等于本会话已获启动授权。两名执行者称“工程Agent”和“训练Agent”，不要与架构路线A/B混淆。

目标：直接比较保留视觉入口的光谱残差方案、光谱Token教师对齐方案和同接口视觉参照。停止继续追加零散BN/残差调参诊断；本轮不增加组件消融、LoRA、Query/Patch修改、新数据、旧模型重训或预算续尾。

| 架构描述（路径代号） | 编码器主Token | 无标签预训练 | 冻结恢复探针 | 同构分割任务头 |
|---|---|---:|---:|---:|
| 原生视觉入口＋固定早期Fine（C0） | 固定RGB Patch投影＋原Patch LN | 0 | 0，不用未训练头作排名 | 5000 |
| 原生视觉入口＋光谱零起点残差（A） | RGB投影＋USE Q5残差→原Patch LN | 20000 | Fine/Context各1000 | 5000 |
| 光谱Token直接生成＋RGB教师对齐（B） | USE Q5适配→原Patch LN；教师只训练期存在 | 20000 | Fine/Context各1000 | 5000 |
| **正式上限** | **9项：2预训练＋4探针＋3分割** | **40000** | **4000** | **15000** |

**正式合计59000次成功更新。** 工程试训每路线最多200步，共最多400，另列；校准/可丢弃检查、失败、中止、恢复重放和推理成本分别登记。C0没有预训练，不增加C0恢复探针。默认执行新C0分割，除非训练Agent先证明已有结果在结构、Fine、初始化、成员、样本顺序、预算上全部相同并出具复用审计；省下的预算不转给其他组。

主编码器固定使用两条路线的u20000末态，不根据探针、10k曲线或滑坡DEV反选编码器。保存5k/10k作追溯，不新增5k/10k探针。恢复分数弱本身不阻断分割；只因工程错误停机，不为弱组改学习率、对齐系数或续训。

## 2、现场现状、已有来源与边界

2026-10-28核查：仓库HEAD与附件相同，为 `578749d68a46be20bd1e60790a61c4c8798201a3`；已有未提交文档与诊断源码须保留。实施前再次记录实际分支、HEAD、工作树差异和逐文件哈希，不切换main，不回退他人内容。

v4已完成29000正式更新并验收：公共恢复评分20k为1.1491454；轻量头best/last两事件等权IoU约0.0600/0.0597。后续TRAIN-only BN后校准未修复，完整融合IoU降到约0.0292。历史v3＋完整SegFormer约0.1258，现有十波段直入细节参照约0.3081。它们仅作历史定位，训练自由度、消费者和初始化不同，不能替代新C0或作单因素归因。

已存在并可复用的组件：
- `src/landslide_seg/models/use_v4_mit.py` 的光谱前端、`propagate/staged/masked_block` 及本地MiT加载。
- `use_v4_reconstruction.py` 的32/64/160/256原生投影、恢复头和同构All-MLP＋Fine残差分割头。
- 既有数据、三视图五项目标、公共评价、设备锁、保存恢复与度量逻辑。应以新入口组合它们，不修改旧release绑定源码。

共同起点是v3的固定无标签入选权重，不是已适配20000步的v4，也不是任何滑坡监督最佳模型：
- v3：`outputs/use_v3/runs/pretrain/best_unlabeled.pt`；SHA256 `e20e32f9915974c570d2ad9a064af3bd46b75f0e0d78bafbf698637895533886`。
- 本地MiT：`weights/segformer/segformer-b0-finetuned-ade-512-512/model.safetensors`；SHA256 `6ae39addd01de6b1b8bde2cf677d43a5cd733424b8d186de3f95d1c51fee23f9`。
- processor配置SHA256 `8039d1d210abaa7117ad78e58cdfd6141a2ec72c03dae891b3cd76737e422c6c`。
- 无标签成员清单SHA256 `b23c43487927f57d49980754915e22f5ca3c23eef6d258e0bd935e7604d74578`。
- 公共v3评价分母SHA256 `eae806d0a2b111397a44fb56054c5912b51e36700cdedc4fef10fa04a42e8a95`。
- 本次附件SHA256 `2df3ea9726b2504e699dbb81c1805248d1f866790100dd73e8157d4f42861fc8`。

MiT具有ImageNet/ADE20K外部监督历史，不能称整个模型从零纯自监督。两条路线是不同整体方案：A依赖RGB、适配器容量不同，B增加教师目标；A/B的Fine可随光谱前端训练变化，C0 Fine固定于v3初始来源。A优于C0也不能无证据全部归因于Context残差。

## 3、数据与RGB准入：先复核、再冻结

无标签固定11132 TRAIN、599 PRETRAIN_VAL、239隔离位置；训练每次更新6份S2＋2份Landsat，微批2×累积4。监督采用“我们的数据划分（留事件验证）”：九事件2223块训练，广岛448＋特里苏尔221块开发验证，seed1234，stats_core25由559块TRAIN核心拟合。Italy/NewZealand不用，不做四折或官方划分，不更换日期、GSD、地形、标签或数据来源，下载保持暂停。

本会话只读清单并抽取每种命名/传感器/角色代表缓存，得到：

| 数据角色 | S2 | Landsat | 清单可映射RGB |
|---|---:|---:|---:|
| 无标签TRAIN | 9510 | 1622 | 11132/11132 |
| 无标签验证PRETRAIN_VAL | 521 | 78 | 599/599 |

既有缓存审计中的所有波段共同有效面积下限：TRAIN S2约0.80005、Landsat约0.80060，验证分别约0.82886/0.81256。这说明已有准入记录支持RGB可用，并不等于本次重新逐图解压验证。P0仍须复核实际文件哈希、RGB三波段质量、单位/元数据和教师支持率，保存逐成员审计，不能把清单名称当成全部像元合格。

明确RGB别名表：S2外部缓存`red/green/blue`、本地缓存`B04/B03/B02`；Landsat当前清单为`red/green/blue`，代表缓存中心波长约0.65/0.56/0.48μm。按名字和来源注册表映射并核对已知物理波长，不能按“前三个”猜测。不是临时按波长最近邻生成伪RGB，unknown metadata也不抹掉已核验的来源映射。

默认保持全部11132/599共同成员。任一来源映射无法核验、成员/质量与锁定证据矛盾，就记录受阻及共同合法子集候选，不由Agent自行删样本或让A/B用不同数据。即使B可对无RGB样本继续MAE，主实验也不擅自扩展B独有来源。P0通过后锁定实际成员、统计、RGB映射、支持门槛和预算。

## 4、RGB处理与学生/教师接口

USE输入是传感器TRAIN统计标准化后的反射率；RGB入口是物理反射率的独立归一化，两者不是同一张量。对每通道按实际对齐的mean/std解除z-score；若监督缓存以DN保存，则先按原管线转换为物理反射率，不能把DN和反射率统计混用。缓存已完成scale/offset，不重复加减产品offset。代表缓存有负反射率，保留已验收物理值，不用显示拉伸或重复缩放“修正外观”。

固定RGB顺序R/G/B，沿用本地processor均值[0.485,0.456,0.406]、标准差[0.229,0.224,0.225]；物理反射率直接归一化，`do_resize=False`、`do_rescale=False`。不放大到512，不乘255后再除255，不做百分位拉伸、gamma或新增裁剪。质量无效及学生人工遮挡的RGB值先填物理0，再做该归一化，支持掩膜单独传递。

新入口建议 `forward(observation, rgb_visible=None, rgb_valid=None)`：保留现有OpticalObservation白名单，标签不能进入编码计算。路线A/C0的RGB是调用端从同一学生观测构造的辅助输入；路线B部署入口只需observation，不依赖RGB或教师。图像裁剪、翻转/旋转与掩膜在所有学生、教师视图中完全同坐标，记录变换矩阵。

A无标签三个视图的RGB都从对应**已人工遮挡**的学生输入构造。整波段隐藏时不能从完整影像补RGB；像素/块隐藏同步到视觉入口。缺失RGB真实来源时A/C0明确拒绝或标为不适用，不能宣称任意波段支持；人工遮挡不等于来源缺失，允许以已知映射填0并传播支持。

B教师目标使用同一TRAIN样本、同一几何增强、未人工遮挡但真实质量有效的RGB。教师只有冻结原Patch Conv＋Patch LN，固定eval/no_grad，不运行第二套Stage1–4。教师作为预训练任务模块保存在编码器外，只生成stop-gradient目标，不进入学生前向或visible summary；公共验证和部署不依赖教师。允许缓存同一批原始教师目标供三个学生视图共用，不能跨样本/变换错配。

教师Token有效支持预登记：原7×7/s4感受野内RGB三波段共同真实有效面积≥80%；学生同位置可见支持面积≥50%，以实际Patch2＋适配器卷积感受野计算，边缘分母只计图内像元。二者交集才算对齐位置，并记录每传感器/每视图比例。无有效对齐位置的样本仅贡献MAE，对齐项记0及原因，不伪造目标；工程200步任一传感器累计无对齐、或超过10%更新完全无对齐，先阻断并核查映射/掩码，不改0.20系数掩盖问题。

## 5、三组架构与目标的准确实现

共同光谱前端从同一v3逐张量继承tokenizer、physical、spectral_norm、spectral、ffn、queries、readout、residual、residual_out。保留Patch2、五Query、波段注意力和元数据接口；Q1–Q4＋像元残差构成早期Fine[B,32,H,W]，Q5为[B,32,ceil(H/2),ceil(W/2)]。不执行旧v3空间主干、inject/feedback/local。MiT Stage1–4宽度32/64/160/256、步长4/8/16/32，每个学生只运行一次完整空间主干，原始参数全冻结eval，但训练期保留对上游的梯度，不能整体no_grad。

视觉入口残差路线A：
```text
Z_rgb = 原MiT RGB Conv7×7/s4(RGB_visible)
Z_spec = Conv1×1_zero(Conv3×3/s2/p1(Q5))
T_A = 原Patch LayerNorm(flatten(Z_rgb + Z_spec))
T_A → 原MiT Stage1 Blocks/末端Norm → Stage2–4
```
新3×3常规随机初始化，末端1×1权重和bias为0，不再同时增加零门控。初始Z_spec逐元素0；同样RGB和空间支持下Patch前后Token及四阶段Context应与原生视觉staged路径assert_close。全有效时与官方原生forward核对。第一步末端1×1应有梯度，前面的3×3/上游经该残差路径可为0，后续应接通，不能把零初始化预期当断路。

光谱Token对齐路线B：
```text
T_student = 原Patch LayerNorm(flatten(Conv3×3/s2/p1(Q5)))
T_student → 原MiT Stage1–4
T_teacher = stopgrad(原Patch LayerNorm(flatten(原RGB Conv(RGB_original))))
L_align = 有效位置mean[1-cosine(T_student,T_teacher)]
L_B = L_MAE + 0.20 L_align
```
取原Patch LN之后、Stage1 Block之前的32维Token，通道单位化计算余弦。三个学生视图等权，先对每样本有效Token平均，再对样本平均，无对齐样本贡献0且计覆盖；随微批累积归一，不因缺样本临时放大系数。教师权重来自原ADE预训练，不是任何滑坡监督模型。导出的B编码器与独立加载入口不得包含或依赖教师/视觉预处理，需对无RGB的合法波段子集实测加载和前向。

C0：使用固定原RGB Patch Conv＋Patch LN，接同一冻结MiT，Fine固定来自共同v3前端；无光谱残差、无无标签更新。它仍执行生成Fine所需光谱前端，成本不能冒充纯RGB直入模型。

三组复用既有支持传播和masked_block，原始无效数值先归零，禁止标签valid进入编码器。A第一阶段空间支持为RGB卷积支持与光谱适配支持的并集；C0仅视觉支持；B仅光谱支持。零残差等价测试在全有效或显式相同支持掩膜下进行，不因部分遮挡支持规则不同而宣称C0全条件等价。记录混合感受野的填0语义，保护原预训练投影，不新增注意力架构。

MAE沿用v4三视图及五项定义：spectral、fine、16/32/64轮换context，加0.1 gradient和0.1 consistency，总和除3.2。恢复头读取早期Fine和完整原生四尺度学习投影；保留现有空间粗格混入可见区域的已记录口径，不临时换目标。A/B统一使用既有v4 TRAIN-only五项校准分母（验收时锁定实际SHA），不按各路线重新调归一化；公共探针用独立v3公共分母。B只额外加0.20余弦项，不再为align拟合新分母。known元数据为正式训练，partial/unknown仅接口和评价分层。

## 6、正式训练、公共探针与同构任务头

A/B工程各最多200步、每50步记录MAE/对齐/梯度和验证，使用可丢弃副本，正式重新从共同起点初始化。A、B的新增适配器容量不同，报告参数数；共享3×3及公共光谱初始张量按键拷贝核对，不只比较seed。新任务头统一seed1235并保存共享初始state，避免不同编码器构造消耗随机数造成头不一致。

A/B正式预训练各20000成功更新：FP32/TF32关，有效batch8=micro2×accum4；seed1234，采样6 S2＋2 Landsat，双方相同样本与增强/三视图mask流。AdamW WD0.01、clip1；光谱前端峰值LR1e-4，新适配器/重建头6e-4；预热1000步后余弦至20000。MiT、A视觉卷积、B教师排除优化器并固定eval；光谱前端可训练。

保存initial、u5000、u10000、u20000、last和优化器/RNG/成员/配置身份，每1000步记录无标签验证曲线。**u20000直接作为固定评价编码器，不执行USE best-checkpoint选择**。A残差/RGB RMS比例和参数变化在初始/5k/10k/20k记录；在固定无标签验证样本只读将A残差置0、使用同一已训练重建头比较影响，不能借此挑权重或融合系数。B记录MAE与align原始/加权值、cosine、梯度和每传感器覆盖。有限曲线检查不能变成无目标长诊断。

四探针只对应最终20k：A Fine、A Context、B Fine、B Context，各1000步，末态评价。**探针参数只在无标签TRAIN更新，599 PRETRAIN_VAL仅评价**，修正附件“在验证集上训练探针”可能引起的歧义。冻结整个编码器与MiT，Fine只读32通道早期Fine，Context每尺度1×1到32后上采样平均，接现有同容量恢复头。两路线同类头初始张量、数据、掩码和预算相同；AdamW LR6e-4/WD0.01/clip1，预热50后余弦，batch8。C0不训练探针，描述性特征不与训练后恢复头排名。

公共评分沿用固定v3分母及原始目标口径：known元数据、相同599成员和mask；Fine像素、Context光谱、Context空间三类等权，空间16/32/64等权，S2/Landsat等权，保留物理/未归一化误差。恢复弱仍继续规定分割；只有非有限、冻结破坏、错误加载/数据泄漏、实际梯度断开、全常量等工程失败才阻断。1000步不代表充分收敛，不增步。

C0/A/B分割各5000步，完整编码器全冻结eval，仅训练同一个ADE初始化的All-MLP Decode Head＋新单通道classifier＋ResidualFusionHead(256,32)。输入MiT原生四尺度，Fine直接使用32通道早期读出，不再执行SpectralStem或第二套MiT。残差末层零初始化；三组Decoder/classifier/残差初始张量必须一致，BN原运行统计eval、仿射按原任务头规则训练，不做BN对照或后校准。

分割沿用TRAIN2223/DEV669、同样本步序和增强，0.5 BCE＋0.5 Dice，FP32/TF32关，micro2×accum4；AdamW LR6e-4/WD0.01/clip1，预热250后余弦至5000。每500步按两事件等权IoU选任务头best，last另外报告。阈值0.5，标签不更新USE、不选择A/B编码器。冻结哈希在作业首尾及保存点核验，逐步检查优化器/模式，不把全参数CPU哈希耗时混进神经网络前向。

## 7、只有两个Agent：认领、交接和失败处理

工作目录：`N:/Users/yzh/Disater/landslide-seg`。全部代码编写、修复和审查使用[ponytail skill](C:/Users/yzh/.codex/skills/ponytail/SKILL.md)。先读RESEARCH_INDEX最小接力、项目AGENTS、research_policy，查认领与真实进程，再精确认领；模型路线A/B并不对应两个Agent的工作划分。

**工程Agent，任务ID use-mit-dual-routes-engineering**：
- 认领 `src/landslide_seg/models/use_mit_dual_routes.py`。
- 认领 `scripts/run_use_mit_dual_routes.py`、`scripts/verify_use_mit_dual_routes.py`。
- 认领 `tests/test_use_mit_dual_routes.py`、`configs/use_mit_dual_routes`。
- 认领 `outputs/use_mit_dual_routes/engineering`、`outputs/use_mit_dual_routes/releases`。
负责RGB成员审计、两路线/C0实现、必要测试、A/B最多200+200工程试训和不可变release。尽量在这几个新文件组合旧组件；确需新增辅助文件先补认领，不改旧release源码，不抢整个src/scripts/tests，不启动正式队列。

**训练Agent，任务ID use-mit-dual-routes-training**：
- 认领 `outputs/use_mit_dual_routes/runs`、`runtime`、`audit`、`benchmarks`、`figures`（以上runtime等均在同一outputs/use_mit_dual_routes下）。
- 认领 `docs/assets/research/stage_ledger.json`、`docs/assets/research/use_mit_dual_routes`、`docs/MODEL_EVOLUTION.md`、`docs/DATA_EVENT_PROTOCOL.md`、`docs/RESEARCH_INDEX.md`。
- 需要同步分发快照时认领 `docs/USE_MIT_DUAL_ROUTES_EXECUTION_HANDOFF.md`。
负责独立核验、统一双卡队列、全部正式更新、评价与固定文档回写；不自行改模型代码，问题交工程Agent修复。

```text
python -X utf8 scripts/manage_agent_work.py list
python -X utf8 scripts/manage_agent_work.py claim <任务ID> --owner <真实聊天ID> --title <职责> --scope <精确相对路径>
```
多个scope重复参数，必须完整填写相对路径，不直接执行占位符。一个文件只有一个活动写入者；不替对方登记，不使用别人的owner，不关闭旧v2/旧队列。已完成任务修复用新ID，不重复claim。受阻block并给原因/恢复证据，完成先存证据再finish。

工程Agent交付READY_FOR_AUDIT：实际源码和依赖、RGB映射/预处理/支持门槛、成员统计、共同起点、参数组、预算、目标/公共评分、配置及所有覆盖、真实命令和200+200证据。训练Agent独立核对后发布与release SHA绑定的READY_FOR_TRAINING。修改受保护代码、配置或数据使原门禁失效，修复后重新核验。

最终实现CLI尚不存在，由工程Agent用真实--help和日志交付prepare/verify/engineering/freeze/run/probe/segment/evaluate/benchmark等命令，不编造旧脚本参数。登记文件只记职责和证据，计划事实不另抄一套。

## 8、双卡队列和有限成本验收

训练Agent统一管理两卡；按真实GPU UUID和可见设备映射记录设备，一卡一个计算进程，不用DDP、不翻倍单实验batch，不终止别人的任务。启动前检查进程、实测显存和前后向，使用现有每卡锁；显示占用不作为拒绝条件。设备故障迁移需登记并验证恢复，不增加实验。

| 阶段 | GPU 0 | GPU 1 |
|---|---|---|
| 工程 | 视觉入口残差路线最多200步 | 光谱Token对齐路线最多200步 |
| 独立成本核查 | 本轮计算暂停 | C0/A/B同卡串行测量 |
| 正式预训练 | 路线A 20000 | 路线B 20000 |
| 全部编码器冻结后 | A Fine→A Context，共2000 | B Fine→B Context，共2000 |
| 同构分割第一轮 | C0任务头5000 | A任务头5000 |
| 同构分割第二轮 | B任务头5000 | 空闲；日志/CPU汇总 |
| 最终成本核对 | 本轮计算暂停 | 固定正式权重同卡串行复测 |

C0依赖release和共同起点，不依赖预训练成绩；为便于盲于中途DEV，默认预训练结束后统一跑三组分割。空闲卡不填充新实验。C0/A/B头训练次序固定，设备差异不拿来解释速度倍数；成本统一同卡重测。

128、256、512，C10/B1/FP32/TF32关：C0/A/B裸编码器、冻结封装、完整分割消费者分别测30预热、100同步重复、3轮，报告中位/p90、吞吐、峰值allocated/reserved、参数/激活/输出字节及Top10；IO/预处理/CPU→GPU、完整哈希校验单独计时。A视觉额外卷积与RGB处理、B训练期教师开销均计入相应范围；B部署不得携带教师。旧v4/十波段只作注明结构范围的同卡历史成本参考。

正式必须通过128 micro2×accum4两个路线的真实三视图更新，参数边界和显存有余量；512 B1先静态预检，不安全则保留证据、标未测，不强行OOM。本轮成本报告不设相对历史模型的倍率停止阈值，不因模型较慢自动改预算；成本问题记录原因，真实128无法运行才阻断正式队列。1024仅资源安全时最多一次前向，否则静态；2048及更大只静态/切片估计，不新增训练尺度。

所有训练测速用可丢弃副本，优化器step单列工程消耗，不改正式权重。MiT序列缩减注意力并非严格线性，切片不能恢复跨窗上下文。

## 9、验收、主要比较与固定文档交付

工程最小验收：
- A零残差初始化精确，原视觉入口/单次Patch LN/四Stage对齐，第一步末端投影梯度及后续上游接通；B教师stop-gradient，学生梯度穿过冻结MiT。
- 三组加载来源和MiT全部原参数逐张量一致/哈希不变；C0初始Fine固定；A/B光谱继承相同，Fine训练变化有记录；三组任务头初始张量一致。
- C=1/3/6/10/16、127/128/129/256、联合置换/填充/删除/未知元数据：B覆盖完整合法形状，A/C0只在RGB可映射条件通过，无RGB条件明确不适用/拒绝，不伪造RGB。
- 无效NaN/任意值替换、全无效拒绝、奇数对齐/裁剪、质量/人工遮挡/输出掩膜分离；RGB人工隐藏同步，改变隐藏真值不能改变学生前向。改变教师目标只能改变B的对齐损失，不能直接改变学生特征。
- 教师不含滑坡监督权重，部署独立加载不加载教师；同sample确定性、微批归一化和完整优化器/RNG恢复等价。
- P0逐样本RGB合法性、A/B各≤200步、128真实训练、配置及所有覆盖协议预检、release身份和设备边界完整。恢复弱不当工程错误，错误不靠多训练掩盖。

正式结果主要比较：
1. 视觉入口残差方案对同接口视觉参照：是否有净收益；Fine也发生训练变化，不能只归因光谱残差。
2. 光谱对齐方案对同接口视觉参照：移除部署视觉入口后是否仍有可用迁移。
3. 两路线的质量、RGB依赖和计算权衡；对齐loss下降不等于分割收益。
4. 历史v4/v3/十波段仅单独标记历史行，不能代替新参照或单因素归因。

最终最多三张合并主表：预训练/四探针与对齐覆盖和残差贡献；C0/A/B分割best/last及两事件；同卡成本/适用条件。主分割指标为两事件等权IoU（0.5阈值），同时列像素池化IoU、Dice、Precision/Recall、AP、FPR、边界F1及对象召回（1–4像素、<50像素）。固定Recall 0.5/0.8时取达到该Recall的最小FPR并报告实际Recall/阈值，只作描述，不反调阈值。

完整曲线、原始评价JSON、逐样本概率/有效掩膜、固定六例和相同坐标误报案例必须保留；每个主要结论配图，解释机制＋证据＋限制，PCA颜色不作跨模型语义等价。单seed、两开发事件反复研发、外部监督、Fine变化及容量差异全部披露。

阶段计划/状态/结果/下一步只回写stage_ledger的use-mit-dual-routes；结构与实际加载到MODEL_EVOLUTION，成员/指标/结果到DATA_EVENT_PROTOCOL，短接力到RESEARCH_INDEX；图在docs/assets/research/use_mit_dual_routes。此文件是同源分发快照，不另立日期研究报告，不改旧run/权重。
```text
python -X utf8 scripts/query_research.py --refresh-stages
```
只在新增run索引时刷新inventory，核验四份固定文档链接和生成幂等，再结束登记。若A/B都不超过C0，诚实报告本数据/预算下未形成净收益；若只有单事件或阈值指标改善，说明边界。进一步解冻/LoRA/新系数/数据扩量另行授权，不自动执行。

## 10、可直接分发的两段提示词

### 发给工程Agent

请读取 N:/Users/yzh/Disater/landslide-seg/docs/USE_MIT_DUAL_ROUTES_EXECUTION_HANDOFF.md，以该文件和唯一阶段台账use-mit-dual-routes为本轮契约。所有代码工作使用ponytail skill。先查认领，以真实聊天ID认领use-mit-dual-routes-engineering的精确范围，不接管旧实验。

实现视觉入口残差路线、光谱Token教师对齐路线和同接口C0；它们是三种模型，不是三名Agent。优先组合既有v4前端、masked MiT、恢复和分割头，不修改旧release。先做全成员RGB映射/单位/质量审计；A的RGB来自同一遮挡学生输入，B教师仅同样本真实RGB的stop-gradient目标。A零起点残差须数值等价原视觉入口；B只增加固定0.20对齐项，教师不进入部署。MiT冻结eval但保留上游梯度，不运行第二套空间主干。

完成必要测试、A/B各最多200步工程试训和不可变release，交付真实可运行命令、来源/成员/统计/初始化/参数组/预算证据及READY_FOR_AUDIT。正式上限59000，编码器固定u20000，无5k/10k探针、不按DEV选USE。不要启动正式队列；训练Agent发现代码问题由你修复并重发release。

### 发给训练Agent

请读取 N:/Users/yzh/Disater/landslide-seg/docs/USE_MIT_DUAL_ROUTES_EXECUTION_HANDOFF.md，先查认领，以真实聊天ID认领use-mit-dual-routes-training的运行与固定文档范围。你统一负责独立验收、双卡训练、评价和回写，不自行修改工程代码。架构A/B不对应Agent角色。

先核验RGB成员/物理单位、A遮挡RGB无泄漏、零残差数值等价、B教师stop-gradient/部署独立、MiT冻结和上游梯度、200+200工程证据、128真实更新及实际配置预检。通过后发布绑定release的门禁。两卡分别预训练A/B各20000，再四项冻结探针各1000、C0/A/B同构分割头各5000，正式上限59000；工程/失败/恢复重放另列。编码器直接用u20000，探针只在TRAIN更新、599验证仅评价。恢复弱仍执行分割，工程错误才停。

任务头三组同一初始化和BN规则，完整USE冻结；不改0.20、学习率、Query/Patch、不加LoRA/新BN实验或旧模型重训。双卡一卡一个进程，不用DDP、不为占满卡加任务；故障交工程Agent。最终交三张合并表、曲线/固定例、原始评价、哈希和实际成本，更新唯一台账及固定文档；区别C0/A/B严格同构比较与历史不同自由度参照，失败和未测如实记录。
