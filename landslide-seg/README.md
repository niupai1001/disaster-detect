# Landslide Segmentation

SegFormer、U-Net、UTAE滑坡分割子项目。跨卫星编码器及其迁移研究由 `../earth-representation` 维护。

## 环境安装

Python >=3.11，已有运行环境为 `.venv-use-independent`。新环境依赖以pyproject.toml为准：

```powershell
python -m pip install -e ".[segformer,dev]"
```

安装不获取数据或权重，不启动训练。复现历史实验使用其冻结环境和配置。

## 目录结构

```text
landslide-seg/
├─ src/                 数据、模型、训练和推理实现
├─ scripts/             制备、预检、训练、评价与查询入口
├─ configs/             实验配置
├─ tests/               单元与工程测试
├─ manifests/           成员清单、统计和协议锁
├─ data/
│  ├─ raw/              Sen12Landslides、Landslide4Sense
│  ├─ processed/        输入与地形处理缓存
│  └─ cache/            内容寻址缓存
├─ weights/             权重
├─ outputs/             正式运行、reports、history和documentation
├─ vendor/              上游实现
├─ docs/                四份固定专题及配图
└─ pyproject.toml       依赖与测试配置
```

原始数据、派生输入和实验产物分别存放。处理缓存的旧代号只用于配置追溯，不表示数据集本身按模型代数分版；实际输入含义见DATA。

## 配置与验证

```powershell
python scripts/check_research_protocol.py --config <实际配置>
python scripts/manage_agent_work.py list
python -m pytest --collect-only -q
python -m pytest -q
```

训练前预检实际配置及全部覆盖。运行入口按已选方案和预算使用，不能从历史队列自动恢复训练。为保持历史复现，旧资源路径通过兼容目录指向当前位置；新开发使用本页目录。

## 文档入口

- [项目方案与设计](docs/DESIGN.md)
- [正式研究结果](docs/RESULTS.md)
- [数据资源与输入](docs/DATA.md)
- [Agent职责](docs/AGENT_ROLES.md)
