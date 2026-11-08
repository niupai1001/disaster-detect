# 第一阶段修正后 readiness 摘要

日期：2026-11-08。基线提交 `d806e84`（第一阶段）。本摘要描述该提交之后的修正状态；
具体修正提交以 `git log` 为准。本摘要只陈述当前实际状态；历史文件
（`phase1_report.md`、`phase1_evidence.json`）保留其当时内容，不因后来的确认/纠正而改写。

## 1. 状态总表

| 维度 | 当前状态 | 依据 |
| --- | --- | --- |
| I1–I5 选择 | **CONFIRMED**（2026-11-08，Q1–Q7 按推荐） | `interface_confirmation.json` |
| 接口补充修订 | **IR1 / IR5 / IR6 PENDING**；IR2–IR4、IR7 已按笔误/遗漏纠正 | `interface_revision.md` |
| F1 面积汇聚 | **FIXED**：每个 native cell 向所有正面积相交 micro-cell/Sub-Patch 贡献 | `native_patch.aggregate_microcells` + 新测试 |
| F2 CRS 准入 | **FIXED**：解析 CRS，要求 projected + metre/meter | `types.metric_projected_crs` |
| F3 数据最大值 | **CORRECTED**：完整 cache 无 385² 上限；最大 S2B 2769×2264 | `data_pool_reconciliation.json` |
| 新架构实现 | **PARTIAL**：Registry/SensorMemory/Query/memory-read/动态卷积/几何；融合、主干、Decoder、adapter 未实现 | 代码目录 |
| E1 真实无更新 | **NOT EXECUTED** | 本摘要 |
| E2/E3 | **NOT AUTHORIZED** | 不变 |
| 真实 `optimizer.step` | **0** | 本轮只跑 CPU 合成测试 |

## 2. 本轮修正内容

### 2.1 F1 面积汇聚（高优先级，已修）
- 旧实现按 cell 中心唯一归属一个 Sub-Patch，跨边界面积被漏算；审查反例
  `patch=8, cell=[7,0,9,2]` 真实面积 4，旧实现只累计 2。
- 新实现枚举每个 cell 与实际正面积相交的全部 micro-cell，再映射到
  `(patch_x, patch_y, slot)` 做稀疏 `index_add_`；非法/NaN/负面积矩形 fail-closed；
  单 cell 展开超过 256 条边时拒绝而不是静默裁剪。
- 测试：跨 2 个 patch 的反例、跨 16 个 patch 的大 cell（面积 324）、随机 50 组
  面积守恒、带 mask 的特征积分与输入梯度、跨四象限负索引、非法矩形。
- 未做：大景 GPU 稀疏边构建的峰值内存/耗时实测；本轮为 CPU 正确性。

### 2.2 F2 米制投影 CRS（高优先级，已修）
- `NativeGrid` 现在用 `rasterio.crs.CRS` 解析并检查 `is_projected` 与
  `linear_units ∈ {metre,meter}`；`EPSG:4326`、`EPSG:2263`、`not-a-crs`、空串均拒绝，
  不做静默重投影。
- 未做：`FullGeometry` 跨通道共同 CRS/网格框架检查（adapter 实现时补）。

### 2.3 F3 数据最大值与旧谓词对账（已纠正）
- 逐成员重读 `members.jsonl`（7,778 成员，元数据，不读影像值）：
  - 最大单波段 `H×W=6,269,016`（S2B 2769×2264@10m；同成员 20m 波段 1385×1133）；
  - S2A 最大 2016×2011（4,054,176）；S2C 最大 412×412（169,744）；
  - Landsat（LC08/LC09）最大 128×128（16,384）。
- 旧 `max_native_pixels=16384` 谓词按 `max_band(H×W)>16384` 重算：
  TRAIN 586（=S2 十波段 594 中的 586），全 release 683；`max_side>128` 在本数据上
  恰好同数（683/586），但报告仍按面积谓词，不把边长当通用等价式。
- **没有新增上限、没有裁剪、没有删除合格成员**；错误只在文本与确认记录中更正。
- 产物：`data_pool_reconciliation.json`、`data_pool_members.csv`（逐成员 ID/role/
  SensorKey/通道/形状/面积/GSD/缓存 SHA）、`channel_catalog_frozen.json`
  （5 键、10/7/6、0-based、B8A 在 B08 与 B11 之间、catalog SHA256）。

## 3. 接口修订（详见 `interface_revision.md`）

- **IR1（待确认）**：R(C) 融合后 latent 进入 Decoder 局部 K/V，输出 `a_latent[128]`；
  最终拼接 429 维；不恢复逐通道 skip。
- **IR5（待确认）**：`ModuleQuery` 自身 `F_meta=3`、`8→64→32`，与
  LocalEmbed/latent/target 的频带分开。
- **IR6（待确认）**：1…2048m 正弦有 2048m 公共周期；建议加入由不可变
  `full_geometry` 决定的全景线性坐标 `(d_x/W,d_y/H)`，所有块共用。
- IR2/IR3/IR4/IR7 属算术/遗漏纠正：LocalEmbed 784、support 硬屏蔽、Decoder 拼接 429、
  49×49 粗格 169、kNN 搜索成本与 k 分开。

## 4. 实际执行与测试

环境：项目 venv，Python 3.12.14，torch 2.11.0+cu128，本轮 **CPU only**。
审查环境为 Python 3.13 / torch 2.10.0+cpu；两边数值结论独立，未在审查环境重跑本仓库。

```powershell
# 新架构（含审查回归）
& $py -m unittest discover -s encoder_v3/latent_response/tests
# Ran 50 tests, OK（0 skip）

# 旧路径 CPU 抽查
& $py -m unittest encoder_v3.test_tasks encoder_v3.test_model `
        encoder_v3.test_geometry_cache encoder_v3.test_validation_metric encoder_v3.test_streaming
# Ran 21 tests, OK
```

测试文件：`test_registry_memory.py`、`test_module_queries.py`、`test_dynamic_conv.py`、
`test_native_patch.py`（新增面积守恒/梯度/跨 patch/CRS/非法矩形）、
`test_memory_read.py`（QK Softmax 读出与 P→KV→Query→read 组件链）。

失败证据保留：本轮首次修复尝试中 `test_large_cell` 触发了 64 边的资源 guard、
随机面积守恒测试的 float32 绝对容差过严、跨 patch 测试用了错误的 slot 0；均已在
测试/实现中修正，不把失败隐藏为通过。完整测试日志与文件哈希见
`phase1_correction_evidence.json`。

## 5. 优化器步数

| 项目 | 数量 |
| --- | --- |
| 真实数据 `optimizer.step()` | **0** |
| CPU 合成 AdamW 单步（本轮） | 6 次（每次全套测试执行 1 次，含证据生成运行） |
| 自第一阶段起累计 CPU 合成步 | 11 次（第一阶段 5 + 本轮 6），与真实更新分开 |
| E2/E3 | NOT_AUTHORIZED / 未执行 |

## 6. 仍未完成 / 依赖边界

- ChannelFusion、空间主干、Decoder、data/task adapter、preflight、train 未实现；
  IR1/IR6 确认前不写这些依赖部分。
- D7 checkpoint 身份校验未实现：裸 `state_dict` 不能作为恢复入口，必须拒绝目录/
  注册/配置/采样身份变化，并验证优化器、RNG、验证后 live state 与下一逻辑更新恢复。
- E1 未执行：无真实数据前后向、无 checkpoint 身份、无大场景流式资源实测。
- IR1/IR5/IR6 未确认前不生成正式 `config_lr_e0_v1.json`；本文中的数值不是批准配置。
