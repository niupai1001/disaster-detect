# EncoderV3 正式训练前最终审查报告

**审查基线**：`269a1e04c8a30510176da01231ce43c3d125bce7`（`earth-representation` 仓库；审计分支 `audit/formal-readiness-269a1e0`）
**审查日期**：2026-11-08
**最终判定**：**`NOT_READY`**

> 本报告不签发训练授权，不启动正式训练。仅在 D10 单独授权下执行了受控真实更新验证；先前超用经用户 addendum 接受，修正版 2 次唯一更新 + 1 次恢复通过旧 FP32 预检容差（非正式新架构容差）。未使用已停止的 186 次预检预算，不训练任务头。

---

## 1. 版本、环境与证据边界

| 项目 | 结果 |
|---|---|
| Git 分支 / HEAD | `audit/formal-readiness-269a1e0` / `269a1e04c8a30510176da01231ce43c3d125bce7` |
| 工作树 | 保留既有未提交改动：`scripts/enmap_cog_read.py`、`scripts/enmap_pairs_valid.py` 修改；`reports/runtime_fault_fragments/CPU_autograd_102964/` 两个文件删除。审计未清理、未覆盖这些改动。 |
| Python / PyTorch | 3.12.14 / 2.11.0+cu128（CUDA 12.8） |
| GPU | 2×RTX 5090，驱动 621.14；按进程使用 GPU1；当前隔离探测通过 |
| 数据发布 | `observation_model/outputs/encoder_v2/data_expansion`；契约 `.../G1` |
| 数据成员 | 7,778：7,395 TRAIN / 383 VAL；无 TEST/GUARD |
| 配置 | `config.json` SHA256 `ee3137…d5184`；`adaptive_config.json` SHA256 `dceeb5…b292` |
| 预检账本 | 214 reserved / 214 completed；剩余 186 已停止、未使用 |
| 当前测试 | 223 项；GPU 全量 223 通过、0 跳过（测试 78.1 s）；CPU 运行 223 项（215 通过、8 个 CUDA 跳过，8 项随后在 GPU 通过） |
| 正式授权 | `formal_training_authorization=null`；`m9_task_head_training_authorized=false` |

**证据边界**：所有新证据在 `encoder_v3/reports/formal_readiness_audit/`；历史报告未覆盖。完整 SHA256 清单见 `evidence/sha256_manifest.json`。

**环境说明**：测试在 `encoder_v3` 下创建临时目录时曾被 Windows 文件权限阻挡；已用 DSH 权限诊断修复当前用户对该目录树的完全控制权，文件内容与所有者未变，回滚记录在 `_diag/formal_audit_acl/`。这是测试环境修复，不改模型、数据或训练配置。文件名扫描测试依赖 `rg`（ripgrep）在 PATH 中；审计运行已显式记录所用 `rg` 路径，未增加生产回退扫描器。

---

## 2. 研究架构与实际实现对照

| 模块 | SHA256（完整值见基线/当前清单） | 设计预期 | 实际行为 | 覆盖与结论 |
|---|---|---|---|---|
| `model.py` | `e0ba10b8…7bbf7d`（基线） | SRF 连续编码、共享 Hypernetwork、原生动态卷积、共享 Decoder | 1→64→32 SRF；34→64→320；32×3×3+32 bias；单波段与 grouped 路径逐值一致；dense A 已支持 `represented_boundary_v1` | SRF 采样 1 nm vs 2 nm 最大差 8.94e-8；serial/grouped 最大差 0；A 隐藏 NaN 隔离通过 |
| `participation_layout.py` | `c8d11902…7718` | GSD→r→s，连续窗口，完整组选择 | 0.5/1.21/10/20/30 m 实际率 0.75/0.6567/0.1992/0.0996/0.0645；组大小 2–16；确定性可复现 | `PASS`（工程与数值）；正式曲线已由 D4-D6 冻结 |
| `participation_encoder.py` | 当前清单 | 共享局部网络、原生/合并双尺度、可变 Token | 使用既有 32→128、两层 Transformer、CLS 读出；P 只进上下文，不改变 Merge | `PASS`；无递归 Merge、无 10 m 分支 |
| `adaptive_streaming.py` | 当前清单 | A/B 监督计划与支持计数 | A 空邻接时曾错误声明全支持；已修复为 0；B2 查询框一次 ±Δ | `FIXED`；无支持案例显式 |
| `spatial_index.py` | 当前清单 | 精确原像元交叠读取 | STRtree + cell-indexed；A 使用 `represented_boundary_v1` | `PASS`（测试通过） |
| `streaming.py` | 当前清单 | 全目标、分块、精确分层损失 | 每波段像素 NMSE / TRAIN std；分块累计；一次 Encoder backward | `PASS`（工程）；真实更新未验收 |
| `joint.py` / `adaptive_joint.py` | 当前清单 | 8 观测有效批量、来源等权、一次裁剪 | 合成/真实无更新均验证；任务缺席时来源权重会重归一化 | `REVIEW_REQUIRED`（缺席协议未冻结） |
| `training.py` | 当前清单 | AdamW/调度/EMA/完整 checkpoint/RNG | 合成恢复、EMA、RNG、plateau 测试通过 | `PASS`（合成）；真实恢复未验收 |
| `route_step.py` / `route_control.py` / `route_branch.py` | 当前清单 | 单次更新事务、共同前缀、两路线 | 事务/恢复/分叉与单路线 Runner 合成测试通过；无自动两路线/JEPA 编排 | `PARTIAL`（Runner 已实现单路线控制流） |
| `validation_panels.py` / `validation_metric.py` | 当前清单 | 固定面板 + 已确认 S 指标 | 指标已实现并接入；正式面板/支持集未冻结；B2 缺失时 S 不可用 | `PARTIAL` |
| `jepa.py` | 当前清单 | 独立 EMA Teacher、独立 Predictor | 合成梯度/EMA/隔离通过；五项条件全 null，无正式 latent loss | `BLOCKED`（设计上保持关闭） |
| `pair_geometry.py` | 当前清单 | B2 仅改查询框、原生 Encoder 不变 | 机制实现，准入 0；`tuple*2` 是 `[dx,dy,dx,dy]`，不是双倍平移 | `BLOCKED_AS_DESIGNED`；误报已澄清 |
| `data.py` | 当前清单 | 哈希绑定只读原生读取 | 7,778 缓存重读通过；统计仍标 `engineering_train_stds` | `PARTIAL`（统计未正式冻结） |

**当前源码完整哈希**：见 `evidence/current_source_hashes.json`（由当前 `verify` 报告提取）；`model.py` 行为在 A 路径修复后已变化，基线哈希不再代表当前文件。

---

## 3. 缺陷、严重等级与状态

共登记 23 条（`defect_register.json`）：已修复 5 条，误报澄清 1 条，其余为失败/阻塞/待决策。关键条目：

| ID | 等级 | 问题 | 状态 |
|---|---|---|---|
| V3-AUDIT-001 | P1 | admission 汇总 1/6/7，逐项实际 1/5/8 | 已校正报告，历史未改 |
| V3-AUDIT-002 | P1 | `engineering_tests.json` 59 项、gates.py 哈希过期 | 已由 223 项当前测试取代 |
| V3-AUDIT-003 | P0 | 门禁可被任意非空字符串绕过 | **已修复**（结构化、哈希绑定、JEPA/adaptive/task-head） |
| V3-AUDIT-004 | P0 | 已确认分层验证指标未实现（`metric=None`） | **已修复**（纯函数 + 接入 + 手算测试） |
| V3-AUDIT-005 | P0 | 无完整 Runner；只有单步与测试调用 | 单路线 Runner 已实现；自动两路线/JEPA/正式日程面板仍缺 |
| V3-AUDIT-006 | P0 | 正式 10,000 日程/完整面板未冻结；pilot 面板多隐藏仅 count=2 | 候选日程/面板构建器已实现；正式日程/面板待决策 |
| V3-AUDIT-007 | P1 | TRAIN 统计量仍为 engineering，未正式冻结 | 证据支持冻结，待用户 |
| V3-AUDIT-008 | P0 | 无正式数据清单/QA；无 TEST/GUARD | 待用户 |
| V3-AUDIT-009 | P0 | B2 准入 0 | 设计上 fail-closed，待决策 |
| V3-AUDIT-010 | P2 | B1 502/2,576（19.5%）为选择性子集 | 已在报告中量化 |
| V3-AUDIT-011 | P0 | JEPA 五项条件全 null、无正式 latent loss | 保持关闭 |
| V3-AUDIT-012 | P1 | dense A 旧 centroid-outside 与 indexed 不一致 | **已修复** |
| V3-AUDIT-013 | P3 | A 空邻接时 supported_count 错误 | **已修复** |
| V3-AUDIT-014 | P1 | warmup/验证/上限/早停已显式参数化；阈值/稳定窗口/正式资源未冻结 | 部分支持，待决策 |
| V3-AUDIT-015 | P1 | SRF/Decoder/Transformer/执行默认值未正式冻结 | 已批准冻结（D4-D6） |
| V3-AUDIT-016 | P0 | 受控恢复仅在旧 FP32 预检容差内，不是正式新架构容差 | `PARTIAL_PREFLIGHT_TOLERANCE` |
| V3-AUDIT-017 | P3 | 疑似 B2 平移因子 2 | **误报**：`tuple*2` 是四元素重复，平移一次 |
| V3-AUDIT-018 | P2 | 矩形遮盖 vs Merge 支持单位不等价 | `REVIEW_REQUIRED` |
| V3-AUDIT-019 | P2 | 缺席任务来源权重静默重归一化 | `REVIEW_REQUIRED` |
| V3-AUDIT-020 | P2 | EnMAP 无实测 SRF、未在发布内 | `NOT_ADMITTED` |
| V3-AUDIT-021 | P2 | 历史 CUDA 原生访问冲突 | 当前运行探测通过，历史保留 |
| V3-AUDIT-022 | P3 | 文件名发现依赖 `rg`，初始 PATH 缺失 | 环境说明，已记录 |
| V3-AUDIT-023 | P0 | 先前超用/失败后，修正版 2 次唯一更新 + 1 次恢复通过旧 FP32 预检容差（参数 max 7.45e-9、优化器相对 L2 1.47e-9、loss 差 2.47e-8）；非正式容差 | 旧容差内通过；正式验收仍未批准 |

**已修复的回归证据**：新增 `test_gates_formal.py`、`test_validation_metric.py`、`test_formal_semantics.py`、`test_formal_runner.py`；完整 223 项在 GPU1 全量通过；CPU 运行 223 项（8 个 CUDA 用例跳过，随后在 GPU 全通过）。

---

## 4. 数据、SRF、统计量与配对

- **数据身份**：成员/契约/统计/SRF 哈希与 `01`/`03` 一致；7,778 成员缓存重读通过（`CACHE_REREAD_PASS_NOT_V3_ADMISSION`）；7,778 血缘复现通过（`LINEAGE_REPRODUCTION_PASS_NOT_V3_ADMISSION`）。这些是来源证据，不是 V3 正式准入。
- **SRF**：96 条曲线，78 条 `measured_band_averaged_srf`、18 条 `transformed_instrument_rsr`；S2A/B/C 各 26、LC08/LC09 各 9；所有文件 SHA256 匹配、波长递增、响应非负、积分 1；43/43 个 `instrument:band` 键均有曲线映射，无缺失/无仪器错配。EnMAP 用户选择“包含但需单独准入”，但 `03a_enmap_admission.json` 显示：不在 7,778 成员、无实测 SRF、无统计键、无原生契约，当前 `BLOCKED_NOT_ADMITTED`，未并入训练或统计。
- **统计量**：43/43 键齐全，std 有限且正；7,395 个 TRAIN 缓存独立重算与历史统计计数一致，最大均值差 2.5e-16、标准差差 4.7e-16。结论是“来源与数值可复现”，不是正式冻结；仍需 D3。
- **角色隔离**：TRAIN/VAL 跨角色精确 native footprint 重叠 0；但只有 TRAIN/VAL，没有 TEST/GUARD；protected 仅按声明 bounds 检查，未读像素。
- **B1**：502 个案例 / 2,576 个可能组合（19.5%）。4 波段各 14/14；7 波段各 56/126；10 波段各 125/1,022；无支持案例 0；参数哈希不变。它是选择性子集流程诊断，不是重建质量或训练恢复证据。
- **B2**：349 个元数据候选、315 个成员、0 准入；两个短间隔案例最佳共享偏移约 (0, +10) m / (+2.5, +10) m；23.71 h 案例反向留出 4/16 变差；阈值/校准隔离/处理域未冻结。B2 保持不可训练。
- **数据范围建议**：可把 bounded 7,778 发布作为正式候选范围，但必须由用户批准成员/QA/TEST-GUARD 边界；审计不自行缩小或扩大。

---

## 5. A / B1 / B2 / JEPA 正确性

- **A**：隐藏反射率替换为 NaN 后 Token 与 group_ids 完全一致；卷积/Merge/Decoder 不读取隐藏值。矩形遮盖与 Merge 支持单位不等价：一个诊断中 5 个组被部分遮盖、21 个组整体被遮盖；这是 `REVIEW_REQUIRED`，不是已批准语义。
- **B1**：隐藏波段不进入 Encoder；目标 SRF 只进入 Decoder 条件；单/多隐藏、2..C−1 隐藏数在 502 诊断中均有覆盖（但非全组合）；可见波段数变化不改变参数。正式面板仍缺完整多隐藏 count。
- **B2**：fail-closed 正确；查询框只做一次 ±Δ；Encoder/原生 Affine/Merge 原点不变；真实偏移未应用、准入 0。
- **共享 Decoder**：A/B1/B2 共用同一 `ReflectanceDecoder` 参数；B2 只加无参数查询适配；dense/indexed A 支持集合与预测一致（2e-6 测试内）。
- **分层损失**：合成测试验证来源/观测/波段权重、分块一致、单次裁剪；高光谱多波段不会仅凭像元数压倒多光谱来源。待决策：任务缺席时是否允许来源权重重归一化（当前代码会）。
- **JEPA**：独立 EMA Teacher、stop-grad、Predictor 只接 Z 与目标物理条件、目标潜在 detach、Decoder 无 JEPA 梯度；但五项条件全 null、无正式 latent loss/normalization，保持关闭。

---

## 6. GPU、数值与性能

- **运行时**：隔离子进程 `torch.ones(8, cuda)` 通过（`CUDA_RUNTIME_OK`）；全量 223 项测试在 GPU1 通过、0 跳过（测试 78.1 s）；历史 `nvcuda64.dll` 访问冲突保留为历史证据，本次未复现，未改驱动/全局环境。
- **真实无更新诊断**：8 个已确认 TRAIN 成员、16 个 A/B1 视图；真实前向/反向 72.5 s，峰值分配 7.02 GiB，加权损失 6.55，梯度范数 1602，非有限梯度 0；参数哈希前后一致、预算哈希前后一致、0 次 optimizer 更新、无 checkpoint。
- **D10 受控更新（用户已授权，超用经 addendum 接受）**：三次尝试累计 6 次唯一更新/9 个 optimizer step；前两次失败（首次精确恢复不等、第二次 optimizer 状态设备错误）。修正版严格 2 次唯一更新 + 1 次恢复，参数最大差 7.45e-9、优化器状态相对 L2 1.47e-9、loss 差 2.47e-8，落在旧 FP32 预检容差内，但不是正式新架构容差；`execution_equivalence` 仍未正式批准。pilot 预算仍 214/214、186 未用。详见 `evidence/controlled_update/report.json` 与 `incident.json`。
- **数值**：SRF 相同曲线 1 nm vs 2 nm 最大差 8.94e-8；Hypernetwork serial vs grouped 最大差 0；P60/P120 Merge 成员、中心、bounds 完全一致；dense/indexed A 在 2e-6 内一致。
- **容差**：只确认旧的 FP32 单 GPU 预检容差；BF16、DDP、所有微批量、JEPA EMA、正式训练容差均未批准，不能自动推广。
- **性能**：本轮只做真实 serial no-update；microbatch 2/4/8、packed、cell-indexed、retained geometry 未做新的正式等价/性能验收；默认保持已审参考路径。

---

## 7. Runner、恢复、两路线与可复现性

- **已完成**：`route_step.execute_route_update` 单次事务、`route_control` 共同前缀/两路线身份、`route_branch` 分叉、`training.save/restore_checkpoint` 的合成测试通过；`formal_runner.run_route` 已把 `schedule.prepare → move_views → execute_route_update → next signature` 串成单路线循环，并支持显式 `max_updates`/`max_wall_seconds`/早停 patience/min_delta、首输入签名校验和 `panel_validator`；`formal_schedule.build_formal_candidate_schedule` 与 `formal_panels.build_validation_panel_document` 可从显式输入生成 loader-compatible 候选日程/面板；新增 7 项 Runner/构建器测试。故障注入覆盖计算前/反向/优化器前后/检查点写入/日志失败/恢复身份。
- **未完成**：没有自动的 shared-prefix → 两条正式路线编排，也没有 JEPA runner 路径；没有正式 10,000 更新日程/面板冻结，pilot 只有 200 更新候选；真实数据 + 正式日程/面板 + 授权下的 Runner 尚未跑通。
- **恢复**：合成检查点保存 Encoder/Decoder/AdamW/Scheduler/Teacher/Predictor/RNG/游标/身份；D10 修正版已在旧 FP32 预检容差内完成受控真实 checkpoint/恢复；`restore_checkpoint` 的 optimizer 状态设备缺陷已修复；正式新架构容差仍需批准。
- **两路线**：合成分叉保持共同前缀、LR/RNG/预算/验证；真实共享前缀 checkpoint 不存在，A+B / A+B+JEPA 均未启动。

---

## 8. 14 项门禁与额外门禁

**计数校正**：旧 admission 报告写 1/6/7；逐项实际为 **1 SATISFIED / 5 PARTIAL_PREFLIGHT / 8 NOT_SATISFIED**。D4-D6 后，`optimizer_choice`、`merge_configuration`、`hypernetwork_hidden_structure`、`mask_rectangle_distribution` 已有结构化用户批准；其余 10 项未批准。

| 门禁 | 当前研究状态 | 结构化门禁 |
|---|---|---|
| optimizer_choice | SATISFIED（AdamW，用户已选） | 已批准（D4-D6） |
| merge_configuration | PARTIAL（单轮 B、曲线/P 关系已冻结；正式数据仍缺） | 已批准（D4-D6） |
| hypernetwork_hidden_structure | PARTIAL（仅预检） | 已批准（D4-D6） |
| frozen_data_manifest | NOT（无正式清单/QA/TEST-GUARD） | 未批准 |
| pair_admission | NOT（0 准入） | 未批准 |
| mask_rectangle_distribution | PARTIAL（仅预检；单位语义待定） | 已批准（D4-D6） |
| validation_panel | PARTIAL（指标已实现；正式面板未冻结、B2 缺失） | 未批准 |
| improvement_threshold | NOT | 未批准 |
| stability_window | NOT | 未批准 |
| early_stop | NOT | 未批准 |
| precision_acceptance | PARTIAL（旧 FP32 预检） | 未批准 |
| resource_limit | NOT | 未批准 |
| execution_equivalence | NOT/仅旧 FP32 预检容差（D10 修正版） | 未批准 |
| formal_training_authorization | NOT | 未批准 |

**额外门禁**：`adaptive_config.json` 四项（spatial hierarchy、physical patch/mask、GPU efficiency、participation curve/anisotropic）均未结构化批准；JEPA 五项全未批准；`m9_task_head_training_authorized=false` 保持；`formal_step` 在真实配置中不存在。

**负向验证**：`python -m encoder_v3.gates --config encoder_v3/config.json` 返回 2，仍拒绝 10 项未决门禁；带 `--adaptive-config/--jepa` 同样拒绝。真实配置只填入了 D4-D6 的四项结构化批准，`formal_training_authorization` 仍为空。

---

## 9. 最终判定

### `NOT_READY`

硬性未满足项：

1. **无正式数据清单/QA/TEST-GUARD**（V3-AUDIT-008）。
2. **B2 准入 0**；正式 B 分项和跨源能力不能训练/验证（V3-AUDIT-009）。
3. **正式日程/验证面板未冻结**；已有候选日程/面板构建器，但 B1 多隐藏 count 不完整、B2 缺席时正式 S 不可用（V3-AUDIT-006）。
4. **Runner 只覆盖单路线**；自动两路线/JEPA 编排与正式日程/面板仍缺（V3-AUDIT-005/006）。
5. **improvement_threshold / stability_window 与研究层面的 resource_limit 未冻结**；早停与更新/墙钟上限已是显式 Runner 参数（V3-AUDIT-014）。
6. **D10 修正版通过旧 FP32 预检容差，但正式新架构容差/批准缺失**；先前超用已记录并经用户 addendum 接受；`execution_equivalence` 仍不能正式通过（V3-AUDIT-016/023）。
7. **JEPA 五项条件未批准**，必须保持关闭（V3-AUDIT-011）。
8. **正式训练授权未签发**（`formal_training_authorization=null`）。

因此不能判为 `ENGINEERING_READY_PENDING_APPROVAL`：即使研究者批准所有数据/科学项，仍缺正式新架构容差/验收、正式数据/B2/面板和自动两路线/JEPA 编排；也不能生成 `FORMAL_TRAINING_AUTHORIZED`。

---

## 10. 最短阻塞解除路径

1. **D1–D3**：批准正式数据范围、全发布 QA 策略、TRAIN-only 统计冻结；建立成员/角色/QA/SRF/统计哈希清单。
2. **D4–D6**：把 SRF/Hypernetwork/Decoder/Transformer/执行默认值和 Merge 曲线升级为正式冻结；明确 P 与遮盖单位；构建正式 10,000 更新日程与完整 B1（count 2..C−1）/B2 面板。
3. **工程**：单路线 Runner、早停钩子与候选日程/面板构建器已实现；补齐 shared-prefix → A+B/A+B+JEPA 自动编排、正式日程/面板冻结，以及稳定性窗口/正式阈值并回归。
4. **D9**：用固定验证重复波动确定 improvement/stability 阈值，冻结 warmup/课程/patience/LR/资源上限。
5. **D10 修正版**：已完成 2 次唯一更新 + 1 次恢复并通过旧 FP32 预检容差；下一步是批准/冻结新架构正式容差并复核，不再重复真实更新。
6. **D7/D8**：B2 保持 0 直到校准/偏移/残差/变化阈值批准；JEPA 保持关闭直到五项证据齐备。
7. 重新跑门禁矩阵；只有全部适用门禁通过且用户签发 `formal_training_authorization`，才能进入正式训练授权审查。

---

## 11. 证据文件与哈希

| 文件 | 内容 |
|---|---|
| `00_baseline.json` | 版本、环境、源码/配置/数据/报告哈希、测试报告过期证据 |
| `01_evidence_reconciliation.json` | 1/5/8 校正、发现覆盖、214/214 账本、旧报告失效 |
| `02_architecture_audit.json` | SRF/Hyper/Merge/GSD/P/隔离/参数表 |
| `03_data_audit.json` | 7,778 成员、96 SRF、43 统计键、角色/发现边界 |
| `04_task_audit.json` | A/B1/B2/JEPA、B1 覆盖率、矩形 vs Merge 诊断 |
| `05_training_pipeline_audit.json` | Runner/调度/验证/checkpoint/路线缺口 |
| `06_numerical_audit.json` | 223 项回归、数值检查、容差范围、真实无更新 + 受控更新事件 |
| `07_runtime_audit.json` | CUDA 隔离探测、历史故障、当前状态 |
| `08_execution_audit.json` | D10 修正版旧容差通过、正式新架构容差未批、预算不变 |
| `09_gate_matrix.json` | 14 项 + adaptive + JEPA 逐项矩阵 |
| `10_decision_requests.md` | 研究者决策请求、B2/受控更新方案 |
| `defect_register.json` | 23 条缺陷/误报/边界 |
| `evidence/sha256_manifest.json` | 上述报告、日志、工具、证据的完整 SHA256 |
| `evidence/full_suite_gpu.json` | 223 项 GPU 全量通过、CUDA 可用 |
| `evidence/cpu_verify.json` | 223 项 CPU 运行（215 通过、8 个 CUDA 跳过） |
| `evidence/controlled_update/incident.json` | D10 超用/失败/修正版通过旧容差、restore 设备修复 |
| `evidence/controlled_update/report.json` | 首次受控更新报告（状态 FAILED） |
| `approvals/D4_D6_formal_freeze.json` | D4-D6 用户批准与正式冻结记录 |
| `03a_enmap_admission.json` | EnMAP 独立准入审查（未准入） |
| `evidence/real_ab1_no_update.json` | 真实 8 成员 A/B1 前向/反向，0 更新 |
| `evidence/cuda_probe.json` | GPU1 隔离探测通过 |
| `logs/` | 原始测试/真实诊断日志 |
| `tools/` | 可复现的审计/证据生成脚本 |

**结论重申**：本轮完成了当前版本的工程回归、结构/数据/任务核查、五个已修复/冻结项、真实无更新诊断和逐项准入矩阵；D4-D6 已按用户批准正式冻结，D10 修正版在旧 FP32 预检容差内完成受控更新/恢复，但正式新架构容差未批准，先前超用仍记录在案。正式训练仍被数据清单（含 EnMAP 独立准入）、B2=0、正式面板/日程、Runner 两路线/JEPA、控制阈值、正式执行等价和正式授权硬性阻塞。**不得以本报告作为训练授权。**

