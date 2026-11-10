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

`train`必须提供绑定新架构/代码/配置/冻结成员的authorization与readiness。用户当前授权完成试验后进入新V3-H训练；具体运行状态以`outputs/encoder_v3/hierarchical`下证据为准，命令示例不恢复旧训练。

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
