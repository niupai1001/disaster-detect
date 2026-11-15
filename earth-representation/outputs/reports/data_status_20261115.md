# Encoder V3 数据现状报告（2026-11-15）

本报告记录第二阶段结束后的数据结论与 `research_dataset_v2` 的实测状态，作为后续实验与维护的入口。所有数字均来自本次磁盘核验，不使用估计值。

## 1. 结论摘要

1. **主数据集达到目标**：S2、L8、L9 三种配置各 **400/100/100**，共 1,800 条，VAL/TEST 缺口为 0。
2. **独立留出是真实新增**：VAL/TEST 全部来自新的独立地理组/采集组，未从 TRAIN 重新划分；TRAIN 成员与 v1 子集逐 id 相同。
3. **准入全部通过**：1,800/1,800 满足完整契约（Q01–Q08），拒收 0、受限契约 0、待审 0、跨角色重复 0。
4. **历史数据未动**：冻结发布 `members.jsonl` SHA256 `b3c16c5b…` 未变；未启动训练、未改模型代码、EnMAP 保持 `RESERVED_ONLY`。
5. **外部来源只保留受限候选**：没有任何外部来源进入主 TRAIN/VAL/TEST；不满足标准的来源以 `CANDIDATE_LIMITED_QA`/`DN_ONLY`/`PENDING_LICENCE` 等类别登记，缺口如实报告。

## 2. 主数据集

路径：`earth-representation/outputs/encoder_v3/data_download_plan/research_dataset_v2/`

| 项目 | 实测 |
| --- | --- |
| 记录总数 | 1,800 |
| TRAIN / VAL / TEST | 1,200 / 300 / 300 |
| S2_L2A_CORE10 | 400 / 100 / 100 |
| L8_C2L2_SR7 | 400 / 100 / 100 |
| L9_C2L2_SR7 | 400 / 100 / 100 |
| 缺口 | 0（`gaps` 全为 0） |
| 地理组 | 1,676 组；单组最多 10 条（上限 10） |
| 跨角色组 | 0 |
| 缓存完整性 | 1,800 条缓存全部存在，SHA-256 不匹配 0 |
| 准入审计 | `admission_audit_v2`：1,800/1,800 `pass_core`；重复观测 0 |
| 哈希一致性 | members / train_statistics / quality 与 `release.json` 一致 |
| 训练授权 | `training_authorized=false`，未启动训练 |
| EnMAP | 0 条 |

哈希（`release.json`）：

- `members.jsonl`：`7c841987…`
- `train_statistics.json`：`97510c27…`
- `quality_records.jsonl`：`5a161d0b…`
- 冻结发布 `members.jsonl`：`b3c16c5b…`（未变）

数据集目录内容：`members.jsonl`、`train/val/test.jsonl`、`groups.jsonl`、`quality_records.jsonl`、`train_statistics.json`（仅 TRAIN 拟合）、`additional_candidates.jsonl`（8 条定标补齐 CORE10 候选 + 2 条 `DERIVED_ONLY`）、`atlas/`（三配置示例图）、`release.json`、`REPORT.md`。

## 3. 数据构成

| 配置 | 波段与原生网格 | 物理量 | QA 来源 |
| --- | --- | --- | --- |
| S2_L2A_CORE10 | 10 波段：B02/03/04/08 @10 m + B05/06/07/B8A/B11/B12 @20 m | 地表反射率（sen2cor，DN/10000） | SCL 排除 0/1/3/8/9/10/11/255；20 m 掩膜由 SCL 下采样 |
| L8_C2L2_SR7 | 7 波段 SR_B1–SR_B7 @30 m | C2 L2 地表反射率 `DN×0.0000275−0.2` | `QA_PIXEL` bits0-5 clear 且 `QA_RADSAT==0` |
| L9_C2L2_SR7 | 7 波段 SR_B1–SR_B7 @30 m | 同上 | 同上 |

来源与抽样规则：

- TRAIN：沿用 v1 子集的 400 条/配置，对应冻结发布中的既有成员，成员集合未变。
- VAL/TEST：held-out 采集共 462 条候选（S2 TEST 102、L8 TEST 105 / VAL 77、L9 TEST 100 / VAL 78），按目标与单组≤10 抽样 600 条入集；窗口统一 128 像素，S2 的 SCL 清除率、Landsat 的 QA 清除率均要求 ≥95%；同一景只归一个角色。
- 定标补齐：10 条 S2 pending 全部结案，其中 8 条补为 CORE10 附加 TRAIN 候选（不计入固定 400），2 条 30 m 重采样记录登记为 `DERIVED_ONLY`，不进入数据集。

## 4. 质量控制

- **Q01–Q08 全量审计**：缓存存在与哈希、波段集合/顺序/同一产品、定标只应用一次、覆盖率≥50% 且有限、原生支撑与 north-up 几何、QA 来源声明、角色/重复/组隔离、传感器许可。
- **统计隔离**：`train_statistics.json` 只由 TRAIN 拟合，VAL/TEST 未参与。
- **去重**：跨角色完全重复观测（同产品/日期/包围盒）已剔除并补采，最终为 0。
- **组上限**：全量 1,676 组、单组最多 10 条；全量候选池中更大的冻结组不直接用于训练。

## 5. 底层池与磁盘占用

| 数据根 | 记录 | 体积 |
| --- | ---: | ---: |
| 冻结发布 `data_expansion` | 7,788 | 2,948.8 MB |
| `data_expansion_v2`（S2 十波段补全） | 4,992 | 891.4 MB |
| `data_expansion_v2_landsat`（队列 + SR7） | 2,449 | 550.0 MB |
| `data_expansion_v2_heldout`（新地理 VAL/TEST） | 462 | 112.5 MB |
| `data_expansion_v2_calibration`（定标补齐） | 8 | 2.2 MB |

另有全量候选发布 `local_completion_release_v1`（8,360 条：S2 CORE10 5,692、L8 SR7 2,121、L9 SR7 547）与研究子集 `research_subset_v1`（1,360 条）。数据集清单与图集约 19.2 MB；外部样本（GF-2 403 MB、SpaceNet 764 MB）另存 `external_access/`，不计入数据集。

## 6. 外部来源状态（候选，未入主数据集）

| 来源 | 类别 | 关键限制 |
| --- | --- | --- |
| BigEarthNet-S2 | `CANDIDATE_LIMITED_QA` | 原生 10/20/60 m 可行、sen2cor 反射率、官方划分；无逐像元 QA |
| BigEarthNet-L | `CANDIDATE_LIMITED_QA` | 7 波段 30 m、C2 L2 SR DN（单位经官方加载器确认）；无逐像元 QA |
| GF-2 | `DN_ONLY` | 150 张 16 位原图（B,G,R,NIR）、无 CRS、仅 `.rpb` RPC、值 0–1001、无定标；8 位副本不得替代 |
| WorldView/GeoEye | `DN_ONLY` | SpaceNet 8 波段 WV-3、EPSG:4326 度、DN 274–1025、无定标 |
| PlanetScope | `PENDING_LICENCE` | 公开目录仅 NICFI 月度镶嵌（派生）；原生需账号/许可 |
| SPOT | DN / DERIVED | WorldStrat 原始为 DN（CC BY-NC）；FLAIR-HUB 为 1.6 m 重采样 |
| VENµS | `RESTRICTED_NC` | `other-nc` 非商业；20 m 波段与 5 m 参考须分开登记 |
| EnMAP | `RESERVED_ONLY` | 不下载、不计数、不参与统计 |

逐项决定与理由见 `DECISIONS_20261114.md`，机器可读类别见 `source_admission_v1.json`。

## 7. 已知缺口与风险

1. BEN-S2 / BEN-L 只有“官方清单/代码 + 人工复核”级质量证据，不能按可靠像元 QA 层级使用；如需主实验级质量需另补逐像元 QA。
2. GF-2 与 WorldView/GeoEye 为 DN、且几何分别为 RPC / 度制；若进 `CORE_NATIVE`，必须取得原始定标系数并新增几何适配器。
3. PlanetScope 原生数据需许可渠道，本轮未申请。
4. 目前没有满足 `CORE_NATIVE` 的非 S2/Landsat 类别，按缺口报告处理，不为凑卫星数量降低标准。
5. `docs/AGENT_ROLES.md` 与 `encoder_v3/reports/agent_work.json` 中的本轮入口与任务记录仍未提交（两文件带有他人未完成改动，避免混提交）。

## 8. 复现与入口

| 内容 | 路径 |
| --- | --- |
| 数据集发布 | `outputs/encoder_v3/data_download_plan/research_dataset_v2/release.json` |
| 准入审计 | `outputs/encoder_v3/data_download_plan/admission_audit_v2/summary.json` |
| 第二阶段报告 | `outputs/encoder_v3/data_download_plan/PHASE2_EXECUTION_REPORT.md` |
| 准入决定 | `outputs/encoder_v3/data_download_plan/DECISIONS_20261114.md`、`source_admission_v1.json` |
| 全量证据 | `outputs/encoder_v3/data_download_plan/plan_evidence_20261114.json` |
| 定标结案 | `outputs/encoder_v3/data_download_plan/calibration_resolution/resolution.json` |
| 外部证据 | `outputs/encoder_v3/data_download_plan/external_evidence/`、`external_access/` |

常用校验命令（在 `earth-representation/` 下）：

```powershell
python -m pytest tests/observation_model -q
python scripts/audit_admission.py --members outputs/encoder_v3/data_download_plan/research_dataset_v2/members.jsonl --out outputs/encoder_v3/data_download_plan/admission_audit_v2
```

代码提交：`eccc3b0`（数据补全、留出采集、审计与发布脚本；未含他人改动）。数据集与证据位于被忽略的 `outputs/`，不在 git 历史中。

## 9. 使用约束

- 本数据集不构成训练授权；正式训练需单独授权并绑定冻结身份。
- 主实验使用 `research_dataset_v2` 的成员；不得把全量候选池（含单组 116 条的大组）直接当作训练集。
- 外部候选只能在声明的质量层级内用于独立表征研究，不得冒充 `CORE_NATIVE`。
- EnMAP 在本轮所有计数、划分与统计中均为 0。
