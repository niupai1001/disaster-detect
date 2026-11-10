# 跨卫星表征的Agent职责与维护规则

## 负责什么

负责USE V1、USE V2、共享单波段CNN和Encoder V1–V3。Encoder V3包含原参与率聚合与latent-response设计演变，不再分成两个研究版本。滑坡只是本工程的一项下游验证。

## 谁维护哪里

| 职责 | 负责的内容 | 交付位置 |
|---|---|---|
| 方案负责人 | 目的、想法、继承与替换、参考来源、推动变化的关键发现 | DESIGN.md |
| 数据负责人 | 资源、来源、数量单位、输入配置、质量与使用边界 | DATA.md及实际清单 |
| 工程负责人 | 实现、配置、测试、性能与恢复 | src、configs、tests、独立运行产物 |
| 执行负责人 | 已授权实验的预检、推进、预算、异常和产物完整性 | outputs及任务登记 |
| 评价负责人 | 正式结果、可比条件、best/last及版本解释 | RESULTS.md及验收文件 |
| 文档协调者 | 命名、跨文档一致性、引用和目录路由 | 五份固定文档 |

角色可兼任，不能把启动进程或工程检查称为研究完成。共享正文按整文件认领，由一个持有者合并增量。

## 开工与交接

先读根AGENTS、README及本页，再只读相关专题。使用原任务登记查询范围与持有者，保留已完成成果；未知作者写待核实，旧active不证明进程存活，也不表示可自动接管。当前登记入口为Encoder V3outputs/encoder_v3/agent_work.json；USE仍沿用滑坡工程登记器，查询工具为根.project/inspect_workspace.py。

交接写清：任务ID、目标、负责人、授权、具体文件范围、依赖、实际配置与覆盖、成员与统计哈希、初始化、预算、选模、运行状态、证据、未验证项和唯一下一动作。纯文档任务不填训练参数，也不为补齐表格启动训练。

## 每份文档如何维护

- DESIGN说明为什么这样设计，关键发现只写推动变化的少数结果；预算和测试不单列版本。
- RESULTS保存每个正式方案的完整比较，解释机制与证据，不复制设计全文。
- DATA按资源组织，说明有什么、从哪里来、怎样读取；实验保留成员与处理配置，不复制一套阶段数据史。
- README说明安装、目录、命令与验证，不代替设计或分工。
- 本页维护职责、认领与交接，不追加每日进度。

新增内容替换对应小节，不在文末按日期无限追加。配置、曲线、原始评价和验收保存后再回写固定文档并finish；受阻写原因、证据和下一动作。旧文档、历史队列和已有checkpoint都不构成新实验授权。

## 执行约束

既有用户停止的训练和预检预算保持停止。Encoder V3当前方案按本次研究演变资料统一设计；接口的实际实现、数值冻结与验收须另核对，不能凭文档措辞启动真实更新。旧214次预检属于工程记录，不是正式结果。配对机制、偏移数值与配对准入分别核验。

数据、成员、质量规则和统计未变时复用已有审核，只核验实际绑定；发生变化才检查受影响部分。运行使用冻结发布，不能原地改动在跑或可恢复作业绑定的源码和配置。

## V3-H文档范围与后续任务

本次用户明确要求新增[V3-H设计](V3-H_DESIGN.md)与[实施计划](V3-H_IMPLEMENTATION_PLAN.md)，作为五份固定正文规则的特定例外。详细规范和唯一阶段计划保存在这两份文件，README提供入口，DESIGN提供演变导览；RESULTS/DATA本次不变。

本轮文档协调者为当前Codex聊天，范围仅README、DESIGN、AGENT_ROLES及两份V3-H Markdown。已查询原登记与工作树；本地主分支原有未提交DESIGN及登记增量保留，不覆盖、不自动纳入本次提交。阶段任务卡见[计划§18](V3-H_IMPLEMENTATION_PLAN.md#18-可执行任务依赖与门禁)，角色均待实际认领，不代表已派发或完成。

后续用户明确要求“继续完成试验进入训练”，当前任务`encoder-v3-h-implementation-training`由本聊天持有新hierarchical源码、对应测试、新配置/入口和独立输出目录。旧宽范围任务身份保留，本次不改旧shared_mae/latent_response或冻结作业；登记中的范围重叠由本轮具体授权说明，不能解释为接管全部Encoder V3。

执行顺序：契约冻结→动态几何/并行Token→三主干→双向交互→Decoder/MAE→JEPA→Hybrid闭环→profiling/审查→工程交接。各阶段以设计ID、代码符号、测试及实测证据逐项对应验收；不能以输出形状或loss下降代替设计一致性。本聊天负责实现，用户授权的`/root/v3_h_independent_review`负责只读独立审查，已在CPU回归及实际合成CLI恢复检查后放行当前MAE；不替代Hybrid容量或研究效果验收。

后续任务仍在原`outputs/encoder_v3/agent_work.json`核对owner和具体范围，不抢占整个src/encoder_v3。实现、试验和新训练遵循本轮用户授权，首段独立MAE配置与门禁见实施计划§20；Hybrid真实训练另用配置和身份，不静默开启。已发现的OOM及非确定性恢复失败保留为工程证据，未解决前不能放行。

## 当前训练与执行状态

### V3-H首段独立MAE

用户选择独立MAE10 Epoch后再做Hybrid。原`release_mae_v1`/Run `16b8d41b69ec4224bbb01cc252564f23`保留STOPPED165；加速`e659ccc`/`release_mae_accelerated_v1/execution_gpu1`/Run `cd7c5ee4fa22441c844bfb93586c1d2b`完成208更新后计算209批CUDA非法访存，保留FAILED与last200/best175。用户要求再次开始训练并优先双卡；从200迁移到独立`release_mae_dual_v1`执行身份，原Run不改。仍以10个完整Epoch/2330逻辑批次为数据覆盖，8次未checkpoint更新的重放在spent_updates另计，Hybrid未启动。

同一任务`encoder-v3-h-implementation-training`负责双卡门禁和恢复。`parallel.TwoDeviceBatch`两个worker各有独立TRAIN适配器及物理catalog，避免EnMAP LRU共享；全局观测均值梯度相加，单次clip/optimizer/scheduler/EMA后广播，任一worker失败本批不更新。50项CPU回归、201/209真实批次配对、两最大原生观测并发容量及双卡完整状态下一步恢复通过，短段中位加速1.94倍。运行使用与门禁相同的同步CUDA、确定性FP32，每批保存；此前异步非法访存仍未确定根因，不宣称已根治。审查只读，改动不触碰旧冻结模型或原始数据。

用户已明确暂停当前实验，phase为USER_PAUSED，STOPPED1312/5.632 Epoch、spent1320（含8重放），last1312、best932/第4轮末/完整383VAL0.4444412768，pause_record.json保留指令与检查点SHA。源码`8372d7a`仍冻结于`release_mae_dual_v1/runtime`，Viewer Run `3666b4153ea5457697c9b3188e8b5958`保持原身份，不移除STOP_REQUESTED，不自动续训或启动Hybrid。当前只读最佳权重可视化脚本为scripts/visualize_v3_h.py，图集reconstruction_best_v2；383VAL复算精确一致，18重建案例/4光谱，独立审查核对选样/掩膜/刻度/角色。协调者保留原任务与暂停预算，不接管其他GPU进程、旧LR或已完成共享MAE预算。

### V3-H A-only独立训练

用户另行授权“暂停B1，仅A；优化好加速后开始”，并明确选择“随机初始化，A-only独立训练10 Epoch”。新任务`encoder-v3-h-a-only-training`由本聊天持有hierarchical执行/几何、对应测试、config_v3_h_a_only_v1、独立a_only证据和release_a_only_v1；旧任务与STOPPED1312不恢复，旧检查点不能作为新初始化。新A-only指标与旧0.7A+0.3B1综合指标分开，Viewer方案标mae-a-only。

enabled_tasks覆盖训练、VAL、梯度诊断与检查；可视化只画启用任务，A-only光谱选择隐藏空间位置，不执行B1。几何门禁逐边对照旧归一化Chebyshev谓词（含nextafter半径/小数GSD/大图CSR），不扩大邻域。双卡VAL在worker内关闭梯度、使用独立数据适配器，事务恢复主/副模式。独立55项CPU审查、CUDA旧A对照、最大原生双卡容量、25批异步稳定与完整恢复通过后才新冻结启动。32次固定四源TRAIN工程学习只作可学习性/颜色诊断，不计正式预算；实际正式Run仍从0随机初始化。

新任务已进入A_ONLY_TRAINING：release_a_only_v1/runtime冻结172文件、源码提交8172965，Viewer Run `65525b250ad34747a6024a2cea636754`，mae-a-only方案独立显示。初始化383VAL2.1474302937，best0参数哈希与随机seed17一致；前28步已核对只有A日志、实际optimizer更新及checkpoint，spent=cursor无旧成本继承。10 Epoch为本段新上限，旧AB任务仍paused_by_user/STOPPED1312。正式运行与工程35次独立更新分开；当前进度见新发布status.json；前两完整Epoch实测17.05/18.95分钟，结果与限制归RESULTS。用户请求运行中可视化时，仅复制已验证检查点，在独立进程中推理并核验对应完整VAL记录；不修改在跑runtime、权重或optimizer。当前第525步图集a_only/reconstruction_node_525_v2，独立审查只读，训练继续，不据此外推收敛或完整预算完成。

### 无物理描述器短段对照

用户已授权暂时训练无描述器架构并考虑计算效率，新任务encoder-v3-h-descriptor-free-trial持有hierarchical/descriptor_free.py、对应model/train入口、测试、config_v3_h_descriptor_free_v1、.project/v3_h_descriptor_free_gate.py及独立descriptor_free/release_descriptor_free_v1产物。旧A-only安全停止704/3.0215Epoch，best641，原AB仍STOPPED1312；两者权重及日志不改写，不续入新初始化。

本段明确保留直接学习的波段身份，移除的是物理描述器/Memory/动态卷积。CPU真实入口必须验证max_updates233与2330步LR日程分离、TRIAL_COMPLETE状态；独立审查只读。双卡容量、梯度对照、完整恢复、性能及固定TRAIN学习门禁通过后才能冻结新身份并按本次用户授权启动；工程权重丢弃。新正式短段仅233步上限，后续预算另依据用户要求，不能因epochs10字段自动跑满十轮。

### 旧共享描述器MAE与历史暂停作业

共享描述器与地理Patch MAE的首段正式训练已正常完成10Epoch、2330次更新，状态COMPLETE。最佳点为第10轮末，best与last模型权重一致；没有因查看结果、可视化或文档维护自动续训。旧LR第255步继续保持原暂停状态，旧冻结源码和检查点不改写。

正式运行的冻结源码为outputs/encoder_v3/shared_mae/release_L128_v4/runtime，配置为该目录configs/config_shared_mae_v1.json；同发布目录保存工程/固定样本/Epoch入口/速度验收与readiness.json、authorization.json。formal_run为完整正式结果，reconstruction_best_v2为最佳权重的只读复核与图集。三来源各8成员的工程学习检查与正式10轮分开，不把预检更新数写作正式Epoch。

本次实际配置为L=128m、单GPU FP32、Batch32，TRAIN7442、VAL383、观测等权。每轮完整验证4次，加一次初始化共41次；TVLP位于项目training-runs，真实恢复检查点才延续Run并开启新Session，显示名不含种子。完整身份、曲线和权重以RESULTS中的原始产物链接为准；README只提供运行与查看入口，DESIGN负责当前机制、演变和未解决问题，DATA区分总资源与冻结训练子集。

冷启动、小批次、稳定批次与完整Epoch成本分开记录。当前已获得完整10轮实测，旧25–30分钟估算不再作为实测结论。双卡同步未实现，Windows环境无NCCL；本段没有改变其他GPU作业。

后续评价需保留原成员、QA、统计与固定评价遮盖，区分优化调度导致的平台、重建质量与Encoder迁移能力。没有独立EnMAP VAL，不宣称高光谱泛化。新增训练或实验按用户明确任务推进，本次维护不改变模型、数据、权重或已完成运行。
