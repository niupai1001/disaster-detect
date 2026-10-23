# 所有计划、执行、分析 Agent 的项目接力规则

本文件继承上级 `../AGENTS.md`。最新用户纠正优先：**事件留出不等于四折；后续回到原有固定事件留出基线，不默认四折，也不默认官方split。**

## 开工必读与共享协议

1. 先读 `docs/RESEARCH_INDEX.md` 最小接力与 `docs/assets/research/research_policy.json`；再按任务选一份固定专题文档。阶段目标/变化/结果查 `scripts/query_research.py --stage <阶段>`。
2. 默认协议为 **`event_ood_v4r1`、单seed1234**：九个TRAIN事件共2223块；DEV为Hiroshima448块＋Thrissur221块，二事件等权IoU选模。原配置：`configs/segformer/v4r1/segformer_b0_direct10_detailres_n100_u3600_s1234.yaml`。
3. 成员锁：`manifests/v4r1/protocol.json`；训练与DEV清单：`manifests/v4r1/pool_100.jsonl`，角色TRAIN/DEV（split为train/val）。十波段统计沿用 `manifests/v4r1/stats_core25.json`，由559块共同TRAIN核心拟合；**不因训练2223块而擅自改成全TRAIN重拟合**。新增波段/DEM/指数统计另做本TRAIN适配，登记冻结锁并说明变化。
4. `eval.selection_metric=dev_event_mean_iou`。DEV是已反复使用的开发集合，不称独立盲测TEST。Italy2795块是DIAG，NewZealand338块是FINAL；后者开发时不评测，历史接触限制保留。不得把DIAG/FINAL混进训练或用其反馈反复选模。
5. **EVENT_CV_V5四折只保留既有诊断与历史证据。** 不要求每个新方案跑四折，不以补齐剩余四折作为后续前置，不把2223/669基线与四折均值混排。历史四折复查须在当前任务中明确选择；批量脚本默认拒绝启动四折队列。
6. 官方新训练仅在本固定事件留出基线取得明确进步、冻结候选后安排。旧官方脚本/历史计划/已有checkpoint不是自动续跑授权。约0.02只是投入参考，单seed不宣称显著性。
7. 学习、统计拟合、自监督和同源弱监督只能用当前TRAIN（或协议明确的TRAIN子集）。允许通用预训练；禁止用含DEV/FINAL/DIAG事件监督的checkpoint冒充无泄漏起点。缓存名字与official_split字段不能代替实际成员核对。

## 计划交给执行 Agent 必须包含

先写：`协议=event_ood_v4r1原固定事件留出；TRAIN=2223；DEV=Hiroshima448+Thrissur221；seed=1234；不跑四折`。随后附实际配置/覆盖、清单/统计、输入时相/波段/地形、初始化、预算、选模、输出及回写目标。配置尚未适配就写待实现，不能拿四折或官方配置顶替。

执行前只读预检（项目根目录）：

```powershell
& 'C:/Users/yzh/.workbuddy/binaries/python/envs/tm/Scripts/python.exe' -X utf8 scripts/check_research_protocol.py --config configs/segformer/v4r1/segformer_b0_direct10_detailres_n100_u3600_s1234.yaml
```

有命令行`--set`也传给检查器；预检不要求重训已有基线。检查器不加载模型/图像、不重算统计，不等于训练验收。工程smoke单列且不进入正式研究结果。历史四折只在明确复查时使用 `--protocol EVENT_CV_V5`；相应四折启动器还需显式历史参数，不属于默认队列。

## 阶段与文档回写

- 报告先说明数据角色、输入、模型组件、监督目标来源及损失，再解释结果。正文使用“九事件训练、广岛与特里苏尔验证”等中文名称；`event_ood_v4r1`、DEV、INNER/OUTER等仅在代码映射处解释后出现。不要用“杠杆、塌陷、六臂、软评分”等未定义短语代替机制。模型名必须说明实际连接；“真实日期”须明确为源观测日期相对帧序号的编码修正。自监督要写清重建原高程，标签比例监督要写清来自同一套人工滑坡标注；α变化不等于有效贡献。

- 原固定留出基线上的地形、光谱入口对照已完成；最新计划按台账temporal-mechanism路由，不继续扩四折，不提前多路叠加。原四折运行结果照实保留，不改写成原基线成绩。本轮规则更正不自动中止其他Agent现有进程。
- 阶段目标、相邻变化、结果、下一步唯一维护 `docs/assets/research/stage_ledger.json`，查询工具生成模型阶段表；稳定协议只写共享规则。规划/审阅Agent更新规则与判断，执行Agent回写真正配置、命令/覆盖、哈希、完成摘要和评测。
- 只维护总入口、DATA_EVENT_PROTOCOL、DATA_OFFICIAL、MODEL_EVOLUTION四份固定文档；不再生成日期报告或平行阶段总结。原数据、配置、run、checkpoint不移动或覆盖。
- 多Agent写共享文件前重读相关小节，增量更新；不从别的聊天猜测协议，不因历史任务存在就启动遗留训练。

- 2026-10-22补充：入口“实验可能回答了什么”必须从最早工程覆盖至当前，按台账comparisons生成具体比较、评价集合、数值、证据路径及边界；不能只报告最新批次。比较必须明确两个模型及输入/结构差异，禁止省略0.4775等参照数值属于哪个模型。难懂训练概念要给出数据到目标的具体例子，不只换术语。

- 展示约定补充：数据栏统一“我们的数据划分（留事件验证）”或“官方数据划分（未按事件隔离）”；成员版本作补充。当前参照、最新已完成但未采用的对照、历史模型/诊断、待执行计划必须分开。结构变化与分割前怎样训练地形参数分列，不用“随机编码/重建预训练”单独代替模型变化；它们分别是无提前训练对照和遮挡补高程的具体自监督任务。

- 审查后计划路由（2026-10-22）：最新阶段为 `temporal-mechanism`，详见台账及入口生成计划。地形和六组光谱入口已完成；保留单期任务，优先规划同一SegFormer的时间汇聚四组，截止日期组条件执行。现有时间脚本仍硬编码官方人数和pooled选模，必须先适配原事件划分，不能只替换配置启动。四组尚未实施；计划不代表已授权自动训练。

- 2026-10-23执行状态更正：上述“四组尚未实施”已过期，四组已完成并核验；本轮用户明确授权实际整改和有限实验补全。正在执行的队列见`outputs/research_completion/status.json`，先核验实际进程与产物，不重复启动。新增光谱显式接口和`harmonized_dn_v1`均有独立开关；旧配置保持原数值，接口通过不代表正式光谱实验完成。下载记录支持官方harmonized来源，不能仅因nc缺少scale属性就把所有光谱工作记为受阻；独立定标与B8A偏移正确性边界仍须保留。阶段实际状态以台账、完成摘要和评价为准。
