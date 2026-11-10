# Cross Satellite Representation

跨卫星地表表征子项目：共享光谱输入、自监督表征、原生网格编码与目标条件预测。

## 环境

使用已有Python 3.12环境 `../landslide-seg/.venv-use-independent`。复用已安装的PyTorch、NumPy、SciPy、rasterio、h5py、matplotlib和pytest；具体实验环境以冻结发布记录为准。本地凭证放.env，不提交版本库。

## 目录结构

```text
earth-representation/
├─ src/                 模型与数据实现
│  ├─ models/           共享CNN与视觉模型
│  ├─ observation_model/  Encoder V1、V2
│  └─ encoder_v3/       Encoder V3，含latent_response
├─ configs/             配置与旧USE运行依赖索引
├─ scripts/             启动、数据制备、评价与查看器入口
├─ tests/               单元与工程回归测试
├─ data/                raw、optical、processed、physics、manifests
├─ weights/             模型权重，不入代码版本库
├─ outputs/             运行、评价、报告和历史来源
├─ vendor/              上游实现
├─ docs/                DESIGN、RESULTS、DATA、AGENT_ROLES及配图
└─ pyproject.toml       测试配置
```

`outputs`保存实验ID、配置、曲线、评价、验收及冻结发布；历史报告与文档来源分别集中在reports、documentation。数据目录中的原始观测和处理缓存分开，不按模型代数重复解释数据。

## 运行入口

在本目录、已激活的项目环境中运行：

```powershell
python scripts/run_module.py encoder_v3.gates --help
python -m pytest --collect-only -q
python -m pytest -q
```

训练必须使用对应方案的实际配置、冻结成员和用户授权；上面的命令不启动正式训练。查看已有训练页面可运行 `scripts/viewer/open_training_viewer.ps1`。

旧USE实现仍复用滑坡工程的运行包，具体依赖在 `configs/legacy_runtime_inventory.json`。旧模块和数据路径由兼容目录映射到当前实际位置，不复制第二份源码或数据；新开发使用本页展示的目录。工作树源码路径调整后须重新预检，不能把旧冻结发布验收直接用于修改后的源码。

## 文档入口

简洁Decoder已完成实现与质量/加速门禁：[机制与L9原因](docs/DESIGN.md#简洁原生残差decoder)、[工程结果](docs/RESULTS.md#简洁decoder与l9排查)、[实施顺序](docs/V3-H_IMPLEMENTATION_PLAN.md#26-简洁decoder实施与30轮训练)。新配置config_v3_h_simple_decoder_v1.json保留原物理Encoder，7316参数读出头，恢复EnMAP，随机初始化A-only双卡30 Epoch/6990更新、预热50步。正式状态以release_simple_decoder_v1/formal_run为准；工程权重丢弃，旧运行不恢复。

- [项目方案与设计](docs/DESIGN.md)：研究目的、代际演变和模型思路。
- [正式研究结果](docs/RESULTS.md)：完整比较及各方案解释。
- [数据资源与输入](docs/DATA.md)：来源、数量、配置和图示。
- [Agent职责](docs/AGENT_ROLES.md)：维护范围、交接和执行约束。
- [V3-H架构设计](docs/V3-H_DESIGN.md)：Encoder V3下一阶段的几何驱动三尺度、both / neighbor / parallel及信息权限。
- [V3-H实施计划](docs/V3-H_IMPLEMENTATION_PLAN.md)：P0–P8依赖、Agent任务、门禁及设计—代码追踪矩阵；独立hierarchical实现正在完成训练前门禁。

## V3-H独立实现与工程检查

新代码为`src/encoder_v3/hierarchical`，显式L和原生GSD驱动三尺度分区；配置为`configs/config_v3_h_mae_v1.json`。旧shared_mae源码、冻结发布与历史权重不变。

```powershell
python -I scripts/run_v3_h.py check --config configs/config_v3_h_mae_v1.json --mode samples --output outputs/encoder_v3/hierarchical/engineering/新的样本检查目录
```

`samples/capacity/batch`只做真实前后向、不更新参数；`stability`进行8次独立工程更新并核验保存恢复，不记作正式Epoch。入口预载标准库ssl，避免旧`src/ssl`包遮蔽标准库。测试使用已激活项目环境：

```powershell
python -I -c "import ssl,pytest;raise SystemExit(pytest.main(['tests/encoder_v3/hierarchical','tests/encoder_v3/shared_mae','-o','pythonpath=src','-q']))"
```

`train`必须提供绑定架构/代码/配置/冻结成员的authorization与readiness。单卡加速段在完成208次更新后、计算209批时CUDA非法访存退出；该Run `cd7c5ee4fa22441c844bfb93586c1d2b`及last200/best175保留。用户再次要求开始训练并优先可用双卡；新配置为`configs/config_v3_h_dual_v1.json`，双卡门禁通过后从200步在`outputs/encoder_v3/hierarchical/release_mae_dual_v1`新身份恢复，来源谱系与重放记录保留。总数据覆盖仍为10 Epoch/2330个逻辑批次，201—208的8次重放成本在durable spent_updates中另计，不能抹去已执行成本。新[实时状态](outputs/encoder_v3/hierarchical/release_mae_dual_v1/formal_run/status.json)及[更新记录](outputs/encoder_v3/hierarchical/release_mae_dual_v1/formal_run/updates.jsonl)为准，Hybrid未开启。

双卡采用单进程、两副本、独立数据读取缓存和几何缓存：一个原始Batch32按原生像元估算分工，每卡损失除实际全局观测数，梯度相加后由主卡统一clip/AdamW/scheduler/EMA一次并广播；尾批仍18。无需NCCL或未经同步的多进程。50项CPU测试、真实201/209批单/双卡损失与梯度、两大原生观测同时容量及双卡完整下一步恢复通过；短段中位加速约1.94倍，证据`performance/dual_gate_v1`。当前正式执行沿用`CUDA_LAUNCH_BLOCKING=1`及确定性FP32、每批保存；此前异步非法访存根因未确定，同步重放PASS不能证明异步缺陷已消失。

旧A/B1实验已按用户要求安全暂停：STOPPED1312，约5.632 Epoch，spent_updates1320含8次重放；last保存1312，best为932步/第4轮末，完整383VAL=0.4444412768。Viewer Run ID为`3666b4153ea5457697c9b3188e8b5958`，不自动续训或开启Hybrid。冻结源码提交`8372d7aabf9ec25af35181fda1bb90a2689cca95`，runtime内170个依赖文件不变。只读最佳权重图集见[重建可视化](outputs/encoder_v3/hierarchical/reconstruction_best_v2/index.html)及[结果说明](docs/RESULTS.md#encoder-v3-h-层次化mae暂停实验)；脚本`scripts/visualize_v3_h.py`独立导入冻结运行，不改模型或权重。

用户随后选择随机初始化、A-only独立10 Epoch，配置`configs/config_v3_h_a_only_v1.json`为A=1/B1=0、JEPA=0，旧A/B1保持暂停。新训练/验证/梯度诊断/检查与可视化均过滤零权重任务，B1不执行；无可用A的成员明确拒绝，当前7442TRAIN和383VAL全部有A。双卡全局Batch32、L128、mask0.55、FP32、原生目标/QA/统计不变，新发布为`outputs/encoder_v3/hierarchical/release_a_only_v1`，全新Run从0开始，不加载旧best或last。

本次加速用分块Cartesian精确邻接构造替代大图cKDTree邻居列表/全局排序，并行完整VAL，并在门禁通过后使用异步CUDA（CUDA_LAUNCH_BLOCKING=0）。55项CPU测试、原A路径损失/梯度/边/目标对照、双卡最大容量、完整恢复及25批异步短段通过。证据在`outputs/encoder_v3/hierarchical/a_only`：最大场景冷A前后向59.65→44.14秒（1.35倍），热态约31秒基本不变；普通样本热态约0.93—1.17倍有波动，不宣称普遍加速。单/双卡383VAL 23.87→16.65秒，结果相同；A-only普通Batch32约3—5秒。取消B1本身的减负另记，不混称等价代码优化；前两完整Epoch含验证/保存实测17.05/18.95分钟，不能据此声称效率优化充分或色差已解决。

原描述器A-only现已按新试验切换要求安全停止在704步/3.0215 Epoch，Viewer Run `65525b250ad34747a6024a2cea636754`，源码提交`8172965077e4b9f4d788dd4e7252da8cb4dd9a4c`冻结于release_a_only_v1/runtime（172文件，含可视化入口）。已验证随机初始化参数hash0958271a…、未加载checkpoint，初始383VAL A-MSE=2.1474302937。当前[状态](outputs/encoder_v3/hierarchical/release_a_only_v1/formal_run/status.json)和[更新](outputs/encoder_v3/hierarchical/release_a_only_v1/formal_run/updates.jsonl)优先；第525步/2.254 Epoch完整VAL=0.4123939128的[快照图集](outputs/encoder_v3/hierarchical/a_only/reconstruction_node_525_v2/index.html)在训练继续时独立推理生成，说明见[结果](docs/RESULTS.md#encoder-v3-h-a-only运行中诊断)。A-only新指标不与旧综合VAL混画。

加速证据统一在`outputs/encoder_v3/hierarchical/performance`：同卡paired数值/梯度对照及相同前183批无更新回放已证明实际加速。均批18.27→13.54秒（约1.35倍），含3次完整383VAL及检查点总耗时减少约26.88%；两张5090并行共享CPU/I/O，保留此测量条件。用户明确无需等待完整Epoch，两条长回放已停止；有效同批前缀可以验收速度，不冒称完整Epoch耗时。候选47项CPU测试、最大完整原生场景A/B1容量通过；失败轨迹及各版本证据不混用，详见[V3-H实施计划](docs/V3-H_IMPLEMENTATION_PLAN.md#22-加速候选的验收与继续顺序)。

## 当前共享描述器与全局MAE入口

当前实现位于`src/encoder_v3/shared_mae`，配置为`configs/config_shared_mae_v1.json`。Encoder全局128维4层，Decoder全局64维2层。用户已确认正式地理边长`patch_metres=128m`；公式仍为p=L/GSD，没有80m默认值。

```powershell
python -m pytest tests/encoder_v3/shared_mae -q
python scripts/run_module.py encoder_v3.shared_mae.scale_check --output outputs/encoder_v3/shared_mae/新的尺度检查目录
python scripts/run_module.py encoder_v3.shared_mae.precision_check --output outputs/encoder_v3/shared_mae/新的精度检查目录
python scripts/run_module.py encoder_v3.shared_mae.check --mode audit --output outputs/encoder_v3/shared_mae/新的元数据检查目录
```

以上命令不更新参数。`shared_mae.train`为实际训练入口，要求绑定身份的配置、授权和验收文件；简短工程检查不会自动签发正式训练验收。数据成员、原生QA和TRAIN统计复用已有准入。

首段正式训练采用随机初始化，现已正常完成10Epoch、2330次更新，状态为COMPLETE；固定源码为`outputs/encoder_v3/shared_mae/release_L128_v4/runtime`，配置在该目录`configs/config_shared_mae_v1.json`，验收与授权为上级目录`readiness.json`和`authorization.json`，实际运行目录为上级`formal_run`。入口使用L=128m、FP32、全局Encoder128×4/Decoder64×2、Batch32，首段10Epoch；旧检查权重不用于正式初始化。16项测试、完整原生最大场景前后向、24成员固定学习和20批真实Epoch路径均通过。诊断记录不作为正式成绩。

每Epoch全部7442个TRAIN一次，233批末批18；每轮第59/117/175/233批进行完整383成员VAL，另外在正式启动前建立一次基线。训练期间每10批及验证后保存best/last，连续两次验证超过最佳损失10%时安全停止。本段最佳点在第10轮末，best与last模型权重一致。已完成运行不因查看结果或维护文档自动续训。运行中可用`STOP_REQUESTED`请求安全保存；恢复须真正加载检查点并保持匹配的代码、数据及配置身份。

归一化原生MSE的默认曲线为训练批次平均损失和完整VAL损失，单位z²、越低越好、角色loss；辅助为学习率、裁剪前梯度范数、S2/Landsat验证损失。日志统一为项目`training-runs`，新Run与真实恢复的新Session分开。当前仅单GPU，无NCCL及已验证双卡同步路径，不启动未同步多进程。

性能门槛采用同一批32个原生观测前后向对比；该局部吞吐与完整10轮端到端实测分开报告，不以单个小样本代替整轮成本。相对网格面积映射缓存跨场景复用，同网格解码通道合批、批量读取QA判断，顺序读取NPZ减少零散访问；未裁图或减少目标像元。详细证据在同一发布目录。

## 已完成结果与原生重建图

完整383成员VAL从2.2752降至0.6510 z²；最终S2为0.6978、Landsat为0.3241。含验证及保存总计174.5分钟，平均17.45分钟/轮。完整成绩、逐轮表格及解释以[RESULTS](docs/RESULTS.md#encoder-v3-共享描述器与地理patch-mae)为准；当前属于验证平台，重建图仍有平滑和部分波段偏差，未证实充分收敛、独立EnMAP泛化或Encoder迁移有效。

[最佳权重](outputs/encoder_v3/shared_mae/release_L128_v4/formal_run/best.pt)、[完整末态](outputs/encoder_v3/shared_mae/release_L128_v4/formal_run/last.pt)、[运行状态](outputs/encoder_v3/shared_mae/release_L128_v4/formal_run/status.json)、[可视化图集](outputs/encoder_v3/shared_mae/reconstruction_best_v2/index.html)均指向原产物。第三列为纯隐藏区预测，第四列含可见原图；B输入图只显示被隐藏的目标波段，其余可见波段仍参与模型输入。

复现最佳权重可视化（输出目录须尚不存在；只推理，不更新权重）：

```powershell
python scripts/visualize_shared_mae.py --release outputs/encoder_v3/shared_mae/release_L128_v4 --output outputs/encoder_v3/shared_mae/reconstruction_new
```

## 历史latent-response入口与暂停状态

历史入口为`src/encoder_v3/latent_response/train.py`，包含载荷级描述器、联合局部/全局309维读出及精确近邻检索。旧运行`outputs/encoder_v3/latent_response/training_diagnosis/continuation/e3`停在255次更新，因完整验证相对200步退步28.4%暂停；历史权重、日志、冻结源码与先前更早的中止检查各自保留，不改写为本段MAE成绩。

旧LR的Epoch工程配置及证据位于`configs/config_lr_epochs_v1.json`、`outputs/encoder_v3/latent_response/epoch_training`。其约100分钟/轮来自抽样估算，未完整执行Epoch，不能与本段完整10轮实测混作同一口径。旧近邻/局部读出实现属于历史分支，当前设计见[DESIGN](docs/DESIGN.md)。

## Training Viewer（TVLP 2.0）

当前入口`src/encoder_v3/shared_mae/train.py`复用用户提供且逐字节核对一致的`src/encoder_v3/latent_response/tvlp_writer.py`；旧LR入口与元数据逻辑保留在`latent_response`。在Viewer中添加`N:\Users\yzh\Disater\earth-representation\training-runs`；每次新训练独立目录和Run ID。同实验/方案保持稳定ID，显示名称不包含种子。

当前MAE方案默认曲线为训练批次平均损失、完整验证集平均损失，均按观测等权，单位z²、越低越好、角色loss。辅助数字为学习率、裁剪前梯度范数、S2验证损失、Landsat验证损失。另记录整轮训练均值、批次观测数、累计观测数和耗时。旧运行保留旧元数据和归约口径。

日志只来自实际循环，step保留参数更新计数，epoch为累计处理观测数除以7,442。只有真正加载原检查点才恢复原Run并开启新Session；新的观测等权训练必须使用新方案身份。上下文管理器记录完成或异常，写入只限主进程；这不代表已实现分布式训练。Epoch入口的尾批、覆盖、恢复和TVLP日志均有自动化检查。

## 无物理描述器的A-only短段对照

用户要求考虑计算效率并暂时训练无描述器架构。配置为configs/config_v3_h_descriptor_free_v1.json：共享普通1→32的3×3卷积，同原生网格波段作为batch处理；融合和输出改为TRAIN预先登记的267个instrument/band身份Embedding，移除物理Memory、物理查询与动态卷积生成。面积加权波段融合及层次主干保留，因此这不是仅卷积单因素消融。L/GSD/原生几何与QA不变，数据适配器仍验证物理元数据，模型不读取物理描述器。

随机初始化、A=1/B1=0/JEPA=0、双卡Batch32；短段上限233次更新（一个完整Epoch），保留原2330更新的学习率日程和1轮预热，完成标记TRIAL_COMPLETE，不冒充10轮完成。旧AB STOPPED1312及原描述器A-only STOPPED704不自动恢复。58项CPU检查和只读独立审查通过；真实GPU容量、数值、恢复与速度证据在outputs/encoder_v3/hierarchical/descriptor_free，新Run已TRIAL_COMPLETE，233次更新/1Epoch/7442观测，Viewer `dde013a749f5431abb20aea2a587eb6d`，源码提交`0ff7831d07b3aebd45875a17a8d6f7356090750b`冻结174文件；随机初始参数SHA cf3da97b…已核实、无checkpoint加载，初始完整383VAL=2.1762696610，已核验前5步真实A-only更新及每批checkpoint。以[实时状态](outputs/encoder_v3/hierarchical/release_descriptor_free_v1/formal_run/status.json)与[启动核验](outputs/encoder_v3/hierarchical/release_descriptor_free_v1/training_started.json)为准，完整Epoch含验证/保存19.64分钟（更新17.73分钟、其余1.91分钟），末态完整VAL0.4689640854仍差于可见均值0.2079677751；不自动继续预算，详见[完成记录](outputs/encoder_v3/hierarchical/release_descriptor_free_v1/trial_completion.json)。

无物理描述器短段第233步最佳权重的[只读重建图集](outputs/encoder_v3/hierarchical/descriptor_free/reconstruction_best_v1/index.html)已生成，含S2/L8独立VAL分位样本、L9/EnMAP TRAIN诊断、光谱与完整短段曲线；推理不恢复训练，细节与色偏限制见RESULTS。

后续按用户要求固定原带物理描述器Encoder，仅做Decoder工程排查；[对照曲线](outputs/encoder_v3/hierarchical/structure_diagnosis/decoder_control_summary.png)、[汇总](outputs/encoder_v3/hierarchical/structure_diagnosis/summary.json)和[解释](docs/RESULTS.md#固定原物理encoder的decoder工程排查非正式成绩)已保存。工程复现入口为仓库.project/v3_h_frozen_encoder_decoder_probe.py，所有正式运行保持停止。

## 排除EnMAP的原架构短段对照

用户已选择“原含描述器架构随机重训，排除EnMAP”。配置configs/config_v3_h_no_enmap_v1.json保留物理Memory、条件卷积、原波段融合、原层次Encoder和原Decoder；A=1/B1=0/JEPA=0，随机seed17，不加载任何旧或工程权重。exclude_enmap只为本次TRAIN选择原CombinedDataset的光学base，7395观测（S2 5349/L8 1522/L9 524）、232批尾3；VAL383不变。主入口、检查入口和双卡独立读取器都绑定相同配置，update拒绝意外EnMAP训练行；原始数据、统计和所有旧冻结运行不变。

首段max_updates232仅一个完整Epoch，epochs10只保留当前2320步的学习率日程，预热232步，结束TRIAL_COMPLETE。相同随机种子作用于不同成员，洗牌顺序/批次与每轮日程步数随成员变化，因此单次结果不能直接证明EnMAP影响的唯一机制。工程门禁在outputs/encoder_v3/hierarchical/no_enmap；正式运行以release_no_enmap_v1/formal_run为准，不因实验完成自动重新启用EnMAP。

排除EnMAP的正式短段已RUNNING：Run e5a83915eb97484ebb2081f20df44e50，源码e843a0f59bdbb40adda3d24cd34262ed35ac7a8f、175文件冻结于release_no_enmap_v1/runtime。随机初始参数0958271a…与原含EnMAP实验一致，初始383VAL同为2.1474302937；已核验真实A-only更新、checkpoint与已执行前缀零EnMAP输入。实际进度以release_no_enmap_v1/formal_run/status.json及training_started.json为准，max_updates232后TRIAL_COMPLETE，不将启动视为质量改善或因果结论。

排除EnMAP短段现已TRIAL_COMPLETE232/1Epoch，7395观测全部唯一覆盖且实际输入EnMAP为0；完整VAL0.4961607213，含验证/保存17分16秒。[最佳权重图集](outputs/encoder_v3/hierarchical/no_enmap/reconstruction_best_v1/index.html)和[完成记录](outputs/encoder_v3/hierarchical/release_no_enmap_v1/trial_completion.json)已保存，训练保持停止。
