# 主线收敛记录

整理标识为 `2026-11-15-2a749bfc74`。`main` 是唯一长期主线。研究入口见[地表表征](../../earth-representation/README.md)和[滑坡分割](../../landslide-seg/README.md)。

## 分支处置

下表给出发布前审计的完整 SHA。每行的归档标签以 `archive/2026-11-15-2a749bfc74/` 开头，后接原分支名。旧 main 另有同名归档标签，指向更新前的 SHA。历史结果、冻结身份和原训练配置保留原始含义。

| 原分支 | 原 SHA | 去向与核验 |
| --- | --- | --- |
| audit/formal-readiness-269a1e0 | `5c99db6f448dc91db0627feec6aeb5602c6fecba` | FR 祖先。保留 encoder_v3/reports/formal_readiness_audit 下的正式就绪记录及证据。 |
| codex/rtx4080-training-performance | `fae9af6ff7a86c73d034674b63ca214cc2fa88c6` | RTX 4080 云训练、C5 通道消融与架构对照、预览、缓存和交接记录完整归档。历史测试通过。 |
| codex/v6-11ch-aug-diagnostics | `48b81d29fbb6d31c1c6d99262207604dc6cf58e3` | V6 波段映射、11 通道和多场景实验、P14–P18、C2/C5 路线、预检及锁定 TEST 约束完整归档。历史测试通过。 |
| dev/data-phase2-report | `5fab31c30b54020a7c5ea9b6b58df2d720940105` | 保留数据扩展、准入、准备脚本、observation_model 的两套兼容路径、测试和 data_status_20261115.md。 |
| dev/encoder-v3-sensor-memory-v1 | `d330714874274c69eb07bf2466f1d8b74f708661` | FR 祖先。V3 编码器及 A+B1 工程源码、配置、测试和报告保留。 |
| dev/optical-intra-selfsup-v1 | `7d1db632c40a8dd7c6264cb95d97ec03235b2b61` | FR 祖先。E0 光学自监督演进与 config_optical_selfsup_v1.json 保留。 |
| dev/pd-here-v1 | `b0d3852df7c86eee97575ba338a7a9f02e27dd17` | 保留 encoder_v4、config_pd_here_v1.json、run_pd_here.py、测试及 reports/pd_here_v1。既有正式训练入口保持原约定。 |
| earth-representation | `269a1e04c8a30510176da01231ce43c3d125bce7` | FR 祖先。V3 早期参与式聚合、配对几何及物理参考材料在历史和现行 V3 工程中可追溯。 |
| main | `d8cb567ce2b4ae7f48466544d0958c40c81fbf51` | 旧滑坡基线、栅格/掩膜预检和原云训练体系完整归档。保留提交历史，现行滑坡走独立的子项目入口。 |
| master | `c648170f8864eb7eb501d0813cbc386ced1d2967` | FR 祖先。USE 数据准入、P2/P3 和现行滑坡源码与测试保留在双工程树。 |
| results/v3-ab-spatial-diagnostics-20261113 | `521ace47907a458f309133320e6dcc66de88850c` | FR 祖先。保留 ab_spatial_diagnostics/v3_ab_20261113 下的冻结计划、身份、基线 patch 和报告。 |
| results/v3-fr-20261113 | `8d561c0725f8c8c7abebef9ef632b2a9a683893e` | 内容起点。保留 experiments/v3-fr 源码和 film_resnet/v3_fr_20261113 的运行身份、冻结基线和公开结果。 |

## 旧根目录工程与当前入口

| 旧路径 | 对应范围 | 内容处置 |
| --- | --- | --- |
| `src/segmentation_training/` 与旧根 CLI | `landslide-seg/src/landslide_seg/`、`landslide-seg/scripts/` | 现行 SegFormer、U-Net、UTAE 使用自身的数据契约、显式通道和验证实现。旧栅格 guard、NaN 处理、进度和预览按其旧 CLI 完整归档。SegFormer 验证已在 sigmoid 前把 logits 转为 float32，旧 AMP 概率统计补丁对应另一套统计接口。 |
| `configs/segmentation_training/` 与 `configs/research_routes/` | `landslide-seg/configs/` | RTX 4080 开关、缓存及通道实验绑定旧训练器。C2/C5、Prithvi/TerraMind 和旧锁定测试方案属于各自的历史运行契约，连同清单规则和完整配置归档。 |
| `scripts/`、`tests/`、旧影像检查工具 | `landslide-seg/scripts/`、`landslide-seg/tests/` | 当前数据准入、训练、指标和解释使用子项目入口。旧训练脚本、C2/C5 契约测试、通道诊断和预览测试随原入口归档。 |
| `docs/`、`AGENTS.md` 与 `.gitignore` | 两个子项目的 README、专题文档和根协作规则 | 旧结果、交接、治理和忽略规则归档。现行协作补充起始 SHA、串行整合及提交审查规则。 |

旧 main 和两个 Codex 分支与 FR 无共同祖先。三棵历史树分别做过路径与 blob 对照，旧根树合计 403 条路径记录（跨分支重复计数），现行 FR 树中无相同 blob 的对应迁移证据。旧 C2/C5 契约的灾种范围、标签角色、波段映射和锁定 TEST 规则均随原树保留，避免将它们混入现行滑坡协议。核对上述用途后，仅为保留三段历史分别进行保持树不变的合并，并校验合并前后树 SHA 一致。原始历史和归档标签均可独立复查。旧 main 是新 main 的祖先。

## 验证与复现范围

- 相同 Python 3.12 环境下，FR 基线的 V3 和 observation_model 有 56 项通过、2 项跳过。合入 PD-HERE 和数据准备后的同范围及新增测试有 71 项通过、2 项跳过。测试目录为 `tests/encoder_v3/hierarchical_revised`、`tests/encoder_v3/shared_mae`、`tests/observation_model` 和 `tests/encoder_v4`。
- 滑坡基线的 `test_segformer_config.py`、`test_segformer_metrics.py` 和 `test_band_mapping.py` 有 19 项通过、1 项跳过。现行滑坡树与 FR 相同。历史 RTX 与 V6 的配置、指标及适用的 C2/C5 研究路线测试分别有 19 和 32 项通过。
- PD-HERE、数据集构建、准入及 V3 gates 的 `--help` 通过。Python 语法、PD-HERE JSON、数据准备兼容文件一致性、根目录、冲突标记以及删除项均已检查。地表表征 README 的 22 个本地历史运行链接在 FR 基线也无法从纯 Git 克隆解析。
- 实验报告、source_revision、冻结源码和 patch 随原历史保留。真实数据、GPU、正式训练及完整历史复现仍取决于本机数据、权重、日志和运行环境。历史冻结记录保留既有路径语义。面向当前用户的入口与新增说明使用可移植路径。

## 恢复

发布前已在仓库外保存完整远端镜像、验证过的 bundle 和引用清单，并从 bundle 实际恢复核对。实验数据、本地修改、运行目录和工作区副本单独保存。

```sh
git fetch origin tag archive/2026-11-15-2a749bfc74/dev/pd-here-v1
git branch restore/pd-here archive/2026-11-15-2a749bfc74/dev/pd-here-v1
git clone --bare --branch main /path/to/remote.bundle disaster-detect-restored.git
git -C disaster-detect-restored.git fsck --full --no-dangling
```

在旧工作区切换前，保全本地独有提交、stash、未提交修改和未跟踪成果，再执行 `git fetch origin`。优先在独立工作树或新克隆验证 `origin/main`，尤其是旧分支曾使用根目录工程的情况。
