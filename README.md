# disaster-detect

单期 Sentinel-2 多光谱影像的滑坡/灾害分割研究仓库。主体为 `landslide-seg/`：九事件训练、广岛与特里苏尔两事件验证的 SegFormer 分割研究，包含源代码、配置、清单、实验报告、归档历史与测试；`old/` 保存本仓库早期（2026-05）的工程与研究文档。

> 说明：本仓库只保存**代码、配置、清单与报告**。原始影像、掩膜、模型权重、逐 run 输出等大体积产物不入库（本地合计约 176 GB），原因与获取方式见下文「已入库 / 未入库」。

## 仓库结构

```
.
├── AGENTS.md                 # 工作区级协作规则（报告口径、固定文档路由）
├── README.md                 # 本文件
├── landslide-seg/            # 当前研究主体
│   ├── AGENTS.md             # 项目接力规则（协议、成员锁、阶段回写）
│   ├── README.md             # 项目定位与目录路由
│   ├── src/                  # 源码：数据、模型、训练、评价、融合与工具
│   ├── scripts/              # 训练/评价/查询/检查脚本
│   ├── configs/              # 实验配置（按版本分目录）
│   ├── manifests/            # 数据清单、成员锁、统计文件
│   ├── docs/                 # 固定研究文档（入口、数据协议、模型演变）
│   ├── reports/              # 实验报告与执行记录
│   ├── archive/              # 历史阶段文档与报告快照（含原路径与 SHA256）
│   ├── tests/                # 单元测试
│   └── pyproject.toml
├── L4S_sample/               # Landslide4Sense 小样本（h5）
└── old/                      # 仓库原有内容归档（见下）
```

## old/：原有内容归档

本仓库在重建为 `master` 之前，`main` 分支上的内容是 2026-05 的早期工程：`src/agent_team`（多 Agent 编排）、`src/segmentation_contract`（分割数据契约）、`src/segmentation_training`（U-Net/ResUNet 训练骨架），配套 `configs/segmentation_training/`、`docs/` 研究简报与 `tests/`。

这些文件**原样移动**到 `old/` 目录，仅作历史保留，不代表当前实现；`main` 分支仍保留其原始提交历史。当前开发请看 `landslide-seg/`。

## 已入库 / 未入库

| 内容 | 是否入库 | 说明 |
|---|---|---|
| `landslide-seg/` 源码、脚本、配置、测试 | ✅ | 约 3 MB |
| `landslide-seg/docs`、`reports`、`archive` | ✅ | 报告、固定文档与历史归档（约 108 MB） |
| `landslide-seg/manifests` | ✅ | 图块清单与协议锁（jsonl/csv，约 210 MB） |
| `L4S_sample/` | ✅ | 小样本，用于冒烟测试 |
| `landslide-seg/data`、`images`、`annotations` | ❌ | 原始/处理后影像与掩膜，约 130 GB |
| `landslide-seg/outputs*`、`weights` | ❌ | 逐 run 输出与 checkpoint，约 36 GB |
| `landslide-seg/vendor`、`exp`、`_diag` | ❌ | 第三方代码克隆与诊断产物 |

未入库内容不随仓库分发。复现实验时需按 `landslide-seg/docs/DATA_EVENT_PROTOCOL.md` 与 `manifests/` 中的清单自行准备数据；训练前先跑协议预检（只读，不训练）：

```powershell
python -X utf8 scripts/check_research_protocol.py --config configs/segformer/v4r1/segformer_b0_direct10_detailres_n100_u3600_s1234.yaml
```

## 当前研究入口

- [`landslide-seg/docs/RESEARCH_INDEX.md`](landslide-seg/docs/RESEARCH_INDEX.md)：数据划分、当前参照模型与结论（最小接力）
- [`landslide-seg/docs/DATA_EVENT_PROTOCOL.md`](landslide-seg/docs/DATA_EVENT_PROTOCOL.md)：九事件训练、广岛＋特里苏尔验证的成员与指标
- [`landslide-seg/docs/DATA_OFFICIAL.md`](landslide-seg/docs/DATA_OFFICIAL.md)：官方划分的历史对照
- [`landslide-seg/docs/MODEL_EVOLUTION.md`](landslide-seg/docs/MODEL_EVOLUTION.md)：模型结构、监督来源与训练方式
- [`landslide-seg/docs/RESEARCH_INDEX.md#实验可能回答了什么`](landslide-seg/docs/RESEARCH_INDEX.md)：按需展开全部实验比较与证据路径

## 分支

- `master`：当前主干（本仓库内容 + `old/` 归档）。
- `main`：重建前的原始主干，保留早期工程与提交历史。

协作维护、任务认领与贡献记录统一从[研究入口维护路由](landslide-seg/docs/RESEARCH_INDEX.md#文档与维护路由)进入。日常Agent只读最小接力，不默认全文读研究正文或old历史。
