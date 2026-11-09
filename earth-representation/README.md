# Cross Satellite Representation

跨卫星地表表征子项目：共享光谱输入、自监督表征、原生网格编码与目标条件预测。

## 环境

使用已有Python 3.12环境 `../landslide-seg/.venv-use-independent`。复用已安装的PyTorch、NumPy、rasterio、h5py、matplotlib和pytest；具体实验环境以冻结发布记录为准。本地凭证放.env，不提交版本库。

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

训练必须使用对应方案的实际配置、冻结成员和已授权预算；上面的命令不启动正式训练。查看已有训练页面可运行 `scripts/viewer/open_training_viewer.ps1`。

旧USE实现仍复用滑坡工程的运行包，具体依赖在 `configs/legacy_runtime_inventory.json`。旧模块和数据路径由兼容目录映射到当前实际位置，不复制第二份源码或数据；新开发使用本页展示的目录。工作树源码路径调整后须重新预检，不能把旧冻结发布验收直接用于修改后的源码。

## Encoder V3 latent-response 无更新检查

已确认联合局部读出309维、共享模块Query投影与统一场景参考。使用项目已有环境：

```powershell
python scripts/run_module.py encoder_v3.latent_response.preflight --mode check
python scripts/run_module.py encoder_v3.latent_response.preflight --mode no-update --device cuda:0
```

当前S2/Landsat E1无更新检查已通过；EnMAP因原始QA/XML及坏波段规则等条件未齐而未准入。完整范围与限制见`outputs/encoder_v3/latent_response/plan_execution/execution_report.md`。

新配置在`configs/config_lr_e0_v1.json`。无更新检查保留完整cache，不做优化器更新；真实训练入口默认拒绝，E2/E3需另行授权。结果保存在`outputs/encoder_v3/latent_response/plan_execution/e1/readiness.json`。

## 文档入口

- [项目方案与设计](docs/DESIGN.md)：研究目的、代际演变和模型思路。
- [正式研究结果](docs/RESULTS.md)：完整比较及各方案解释。
- [数据资源与输入](docs/DATA.md)：来源、数量、配置和图示。
- [Agent职责](docs/AGENT_ROLES.md)：维护范围、交接和执行约束。
