# Landslide-Seg

> **当前研究入口（2026-10-21）**：[固定研究文档](docs/RESEARCH_INDEX.md)。数据分为[官方划分](docs/DATA_OFFICIAL.md)与[项目跨事件协议](docs/DATA_EVENT_PROTOCOL.md)，架构与新名称见[SegFormer模型演变](docs/MODEL_EVOLUTION.md)。后续默认跨事件开发，明显进步后再新增官方比较。
>
> **所有 Agent 开工必读**：[项目规则](AGENTS.md)与[共享协议](docs/assets/research/research_policy.json)。默认 **原固定事件留出event_ood_v4r1：TRAIN2223、DEV=Hiroshima448＋Thrissur221、seed1234**，沿用原SegFormer基线，不默认四折或官方split。阶段查询 `scripts/query_research.py`。
>
> **以下V1介绍是历史工程记录，不代表当前主模型或数据规模。** 当前以SegFormer及原网格细节路径为主；旧TerraMind/AnySat/语义代码保留，完成状态与后续任务以固定入口为准。

当前执行前先检查实际配置（只读CPU，不训练）：

```powershell
& 'C:/Users/yzh/.workbuddy/binaries/python/envs/tm/Scripts/python.exe' -X utf8 scripts/check_research_protocol.py --config configs/segformer/v4r1/segformer_b0_direct10_detailres_n100_u3600_s1234.yaml
```

正式实验检查原固定事件留出配置；有`--set`覆盖时同步传入。两套历史四折启动器现在默认拒绝启动，只有当前任务明确选择历史复查才用显式参数。普通训练入口必须主动预检；旧官方恢复器继续默认拒绝运行，不以历史脚本存在作为授权。

## 历史 V1 工程介绍

单期 Sentinel-2 L2A 滑坡识别与分割的研究模型。
第一版即实现**完整架构**：TerraMind V1 Small + TiM、AnySat 多尺度 dense 特征、
冻结 RemoteCLIP 语义三阶段、粗定位 + 精细分割两阶段。不做"先删模块再补"的简化版。

> **当前状态：M1 = DONE（已重验收，all_passed=True）；M2 = 已跑通，但跨事件泛化很差。**
>
> - 数据：**72 个认证样本**（4 个事件，C1–C7 七项校验），28 个隔离，680 个未标注图块单独记录；
> - M1：前向/反向/保存/新实例重载/无 GT 泄漏/指标排除无效像素，8 项检查全过；
> - M2：事件互斥划分 train 26 / val 21 / test 25，train loss 1.31→0.19，
>   **val IoU 0.270，test IoU 0.011** —— 训练集太小（26 个图块、2 个事件），
>   这个差距是数据量问题，不足以支撑任何架构结论；
> - 消融（M3）**未做**，按要求等模型稳定后再做。
>
> 详见 [`docs/report_M1_M2.md`](archive/experiment_history/docs/report_M1_M2.md)。
>
> 数据源说明：CAS 滑坡数据集（成都山地所）实测为 RGB 三通道、无坐标系，**不可用**；
> 已改用 Sen12Landslides（含 B8A）+ 同一 L2A 产品回补 B01/B09。
> 详见 [`docs/data_source_review.md`](archive/experiment_history/docs/data_source_review.md)。

---

## 数据源评审结论

| 数据集 | 波段 | 坐标系 | 日期 | 可用于本项目 |
|---|---|---|---|---|
| Landslide4Sense | 14（B01–B12 + 坡度 + DEM，**无 B8A**） | ✗ | ✗ | ✗（缺 B8A，无法回溯） |
| **CAS Landslide Dataset**（中科院成都山地所） | **3（RGB uint8）** | **✗** | 仅区间 | ✗ |
| **Sen12Landslides** | **10（B02–B12，含 B8A）** + SCL | ✓ EPSG | ✓ | **✓（TiM 需回补 B01/B09）** |

---

## 架构

```
完整 12 波段 S2L2A ──┬─→ TerraMind V1 Small + TiM(LULC) ─→ token 特征 (14×14×384, 冻结)
                     │
                     └─→ 10 波段 + 日期 ─→ AnySat Base dense (40m/160m, 1536ch, 冻结)
                                              │
类别/属性文本 → RemoteCLIP(冻结) → 语义原型 ──┤
                                              ↓
                            通道投影 + 物理 stride 对齐 + 门控融合 → P0..P3
                                              ↓
                        语义相似度 → CoarseLocator → 粗 mask + presence
                                              ↓
                        精细解码器(吃**预测**粗概率 + FiLM 语义调制) → 细 mask
```

## 目录

```
landslide-seg/
├── docs/                 # 审计、数据契约、阻塞、形状、实验日志、M1 状态
├── configs/              # model_full.yaml
├── src/landslide_seg/
│   ├── data/             # 波段契约、L4S 读取、批次构造、合成张量
│   ├── encoders/         # TerraMind(TiM)、AnySat 适配器
│   ├── models/           # 融合、语义、粗定位、精细化、完整模型
│   ├── training/         # 损失、指标、训练器
│   └── inference/        # 可视化
├── scripts/              # audit / smoke / train 入口
├── tests/                # 10 个单元测试
└── weights/  vendor/     # 官方权重与官方代码（不提交 Git）
```

## 环境

```bash
# 已建好的专用环境
C:/Users/yzh/.workbuddy/binaries/python/envs/tm/Scripts/python.exe
# torch 2.11.0+cu128 / terratorch 1.2.13 / timm 1.0.30 / clip-anytorch
```

权重（全部官方发布，放在 `weights/`）：

| 文件 | 来源 |
|---|---|
| `TerraMind_v1_small.pt` | `ibm-esa-geospatial/TerraMind-1.0-small` |
| `AnySat_base.pth` | `g-astruc/AnySat/resolve/main/models/AnySat.pth` |
| `RemoteCLIP-ViT-B-32.pt` | `chendelong/RemoteCLIP` |

HF 走镜像：`export HF_ENDPOINT=https://hf-mirror.com`（`ensure_hf_cache()` 会自动处理）。

## 运行

```bash
python scripts/audit_environment.py                                  # 环境审计
python scripts/audit_data_contract.py --probe 16                     # L4S 契约（缺 B8A，拒绝）
python scripts/audit_sen12.py --limit 3                              # Sen12 真实 10 波段 + AnySat
python scripts/real_inference.py --limit 6                           # 真实 12 波段全模型推理
python scripts/train_full.py --source sen12 --steps 5                # 真实数据训练
python scripts/smoke_terramind.py                                    # TiM 真实权重 + ON/OFF 对照
python scripts/smoke_anysat.py --scales 40 80 160                    # 多尺度 dense 特征
python scripts/smoke_full_model.py --steps 3                         # 完整闭环（合成输入）
python scripts/train_full.py --source real                           # L4S 路径，预期退出码 2
python -m pytest tests/ -q                                           # 10 passed
```

## 数据要求（硬约束）

| 用途 | 必需波段 |
|---|---|
| TerraMind S2L2A（TiM 必须完整） | `B01…B12` **含 B8A**，不含 B10 |
| AnySat s2 | `B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12` + 采集日期 |

现有 Landslide4Sense 为 `B01…B12(+SLOPE,DEM)`，**缺 B8A**、无地理/日期信息，
因此不能用于正式训练。`policy="strict"` 会直接抛错；`policy="synthetic_smoke"`
仅用于接口测试，并强制在 provenance 里标注 `fully_observed=False`。

## 尚未验证（不要当成结论）

- 真实数据上的任何精度（mIoU / Dice / Recall）；
- TiM、AnySat 多尺度、语义三阶段各自的贡献（属 M3 消融）；
- 跨区域泛化与鲁棒性。

## 引用

- TerraMind: https://github.com/IBM/terramind · https://arxiv.org/abs/2504.11171
- AnySat: https://github.com/gastruc/AnySat · https://arxiv.org/abs/2412.14123
- RemoteCLIP: https://github.com/ChenDelong1999/RemoteCLIP
- Landslide4Sense: https://github.com/iarai/Landslide4Sense-2022

---

## SegFormer 路线（2026-10-18 起）

上面描述的是 V1 双骨干路线（保留）。新增的 SegFormer 单期 10 波段路线独立于它：

| 项 | 值 |
|---|---|
| 数据版本 | `s2_10band_v1`（2,107 个认证瓦片：train 1,277 / val 448 / test 382） |
| 输入 | 10 波段 `B02…B12`，`[10,128,128]`，逐波段 z-score（train 统计） |
| 模型 | `nvidia/segformer-b0-finetuned-ade-512-512` 薄封装（MiT-B0 + 官方解码器，单 logit） |
| 对照 | A0 direct10 / A1 direct10+Stem / B0 project10to3 / B1 project10to3+Stem |
| 结果 | 见 [`docs/segformer_report_2026-10-18.md`](archive/experiment_history/docs/segformer_report_2026-10-18.md) |
| 契约 | [`docs/segformer_input_contract.md`](archive/experiment_history/docs/segformer_input_contract.md)、[`docs/segformer_training_protocol.md`](archive/experiment_history/docs/segformer_training_protocol.md) |

```bash
PY=C:/Users/yzh/.workbuddy/binaries/python/envs/tm/Scripts/python.exe

$PY scripts/prepare_segformer_data.py --config configs/segformer/base.yaml --workers 8
$PY -m pytest tests/test_segformer_config.py tests/test_segformer_data.py tests/test_segformer_metrics.py tests/test_segformer_adapter.py tests/test_segformer_model.py -q
$PY scripts/smoke_segformer.py --config configs/segformer/a0.yaml
$PY scripts/overfit_segformer.py --config configs/segformer/a0.yaml --tiles 8
$PY scripts/run_segformer_ablation.py --order A0 A1 B0 B1 --seeds 1234 2026 3407
$PY scripts/summarize_segformer_ablation.py
```

执行计划与实测状态：[`docs/segformer_execution_plan_2026-10-18_v2.md`](archive/experiment_history/docs/segformer_execution_plan_2026-10-18_v2.md)。
