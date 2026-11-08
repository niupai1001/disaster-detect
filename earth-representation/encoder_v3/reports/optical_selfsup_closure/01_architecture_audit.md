# 01 架构与任务审核（E0：A 空间遮盖 + B1 原生波段遮盖）

**审核对象**：`earth-representation/encoder_v3/` 下 `model.py`、`participation_layout.py`、`participation_encoder.py`、
`adaptive_window.py`、`tasks.py`、`joint.py`、`adaptive_joint.py`、`formal_schedule.py`、`scheduled_data.py`、
`formal_runner.py`、`formal_panels.py`、`validation_panels.py`、`validation_metric.py`、`training.py`、
`route_step.py`、`gates.py`、`spatial_index.py`、`participation_decoder.py` 共 18 个模块。

**审核方式**：先复用 `reports/formal_readiness_audit/` 已有的架构、任务、数值与执行审核证据，再逐项对照本轮的
E0 约束复核代码；只有旧证据未覆盖、且会直接决定 A/B1 训练正确性的项目才新增真实数据证据
（`evidence/architecture_evidence.json`、17 项新增单元测试）。本轮没有做全项目复扫，也没有修改已冻结的
Encoder 结构。

## 结论摘要

1. **已冻结的 Encoder 结构没有被改动**：连续真实 SRF 输入、`SRF + log(GSDx) + log(GSDy)` 条件、
   共享 Hypernetwork、每波段动态 3×3 卷积、确定性单轮 Participation Merge、原生 Token 与合并 Token 共同进入
   共享局部/全局 Transformer、完整注意力，全部保持原实现。参数总量 1,240,225 = 旧审核 Encoder 的 1,219,232 + 共享 Decoder 的 20,993；Encoder 本身逐参数未变（见 `05_performance_comparison.json` 的 `parameter_count` 与旧 `02_architecture_audit.json` 的参数表）。
2. **E0 的训练入口此前确实被旧 B2 门禁阻塞**，本轮通过了"独立门禁 + 独立指标"解决，而不是把旧门禁标记为通过。
   旧门禁函数、旧指标函数、旧配置与旧实验路径全部保留且未放宽。
3. **发现并修复了 8 个确定的工程缺陷**（下表），全部有新增回归测试；没有发现会改变已冻结科学语义的缺陷。
4. **发现一个能力边界**：发布中 594 个 Sentinel-2 十波段成员的原生场景远大于其他成员（最大 2769×2769 像素，
   估算约 3,250 万 Token），在完整注意力下无法在既定显存预算内训练。本轮没有丢弃 Token、没有改 Transformer，
   而是把"原生场景像素上限"写成**可计数、可审计、进入配方哈希**的数据规则，并把它列为待批准的科研决策。

## 逐项确认（任务书 §2 的 10 条）

### 1. SRF、GSD、动态卷积符合现有设计

- `SRFEncoder`：`Linear(1,64)-GELU-Linear(64,32)`，按响应面积做梯形归一化（`torch.trapezoid(response,wavelength)/area`），
  并强制波长严格递增、响应非负、面积为正。旧审核已测：同一曲线在不同采样点上最大差 8.9e-8，四舍五入误差内一致；
  所有 SRF 参数都收到梯度（`02_architecture_audit.json/srf`）。
- `Encoder.native/native_grouped`：条件输入为 32 维 SRF 特征 + `log([|dx|,|dy|])` 共 34 维，经 `34→64→320` GELU
  Hypernetwork 生成 32 个 3×3 核与 32 个偏置；卷积结果按**可见**邻居数 `9/count` 归一化后再加偏置，
  因此隐藏像元既不计入分母也不贡献分子。
- GSD 取原生仿射的 `|a|` 与 `|e|`，不做取整、不重采样；Merge 窗口由 `g=sqrt(gx*gy)` 经已确认的 Hill 曲线
  `r(g)=a/(1+(g/g0)^p)` 与 `span=1/(1-r)` 得到，窗口尺度不学习。旧审核 5 个 GSD 档位的实际参与率与目标值
  最大偏差 0.0006（0.5–30 m），确定性为真（`02_architecture_audit.json/merge`）。

### 2. 原生 Token 与合并 Token 共同进入全局 Transformer

`participation_layout` 对每个窗口组输出两种 Token：被选中的完整组输出一个合并 Token，其余可见位置各自输出
单像元 Token；`participation_encoder._prepare_with_layouts` 把它们放进同一个 `groups` 列表，`_local_tokens`
按实际组长度分桶做共享局部 Transformer，然后**全部** Token 进入 `global_net`（完整注意力）。代码路径与旧审核
一致，没有筛掉任一类 Token，也没有新增双分支或递归层。

### 3. 每个有效可见像元只归属一个输出组

新增真实数据证据（`evidence/architecture_evidence.json/members[*].partition`）：对真实成员按原生波段实际计算
分组，`组内位置总数 == 去重位置数 == 可见位置数` 且 `keep[indices]` 全为真；合并组与单像元组数量分别记录。
测试侧另有 `test_optical_selfsup.py::test_prepare_builds_eight_distinct_members_with_A_and_B1` 与既有
`test_participation_layout.py` 覆盖。模型不会把同一位置放进两个 Token，因此不存在重复计数导致的损失权重偏差。

### 4. A 的隐藏反射率不进入 Encoder

`spatial_hide` 只改变 `visible` 掩码；`Encoder.native` 用 `torch.where(keep, reflectance, 0)` 清零隐藏位置，
并在 3×3 均值中只按 `keep` 计数。旧审核用"隐藏像元反射率替换为 NaN 后 Token 与组标识完全不变"验证了一次；
本轮对两个真实成员复验（隐藏位置分别为该波段 128² 场景中矩形覆盖的有效像元），`tokens_equal=true`、
`group_ids_equal=true`、输出有限。也就是说隐藏像元的**数值**不参与任何 Token。

### 5. B1 的隐藏波段不进入 Encoder

`_spectral_view` 只用未隐藏波段构造 `task.visible`，隐藏波段只进入 `task.targets`；几何规划
（`prepare_geometry_task`）只遍历 `task.visible`，目标波段的 SRF 仅在 Decoder 条件里通过
`encoder.srf(target.wavelengths_nm, ...)` 使用（`stream_task_roots`）。真实证据：对两个成员，B1 的
`visible ∩ targets = ∅`，且 `visible ∪ targets` 等于该成员全部原生波段，`visible` 非空。

### 6. A 的部分遮盖 Merge 组不会发生数值泄漏

这里要区分两件事：

- **数值不泄漏**：被遮掩位置的反射率不进入任何 Token（第 4 条）。已复验。
- **分组会随可见支持变化**：Merge 的"参与者"定义基于可见位置（组内至少两个有效可见位置），所以矩形遮住
  某组的一部分时，该组的参与率、可参与组集合以及确定性选组前缀都可能变化。这不是泄漏，而是"聚合由可见支持
  决定"的既定语义；旧审核 `04_task_audit.json/A/mask_unit_diagnostic` 已记录矩形遮盖会切穿 Merge 组
  （5 组部分隐藏、21 组完全隐藏），并明确"矩形单位与 Merge 支持单位不等价"。本轮**没有**改变这一语义，
  也没有把它当成缺陷"修复"。

### 7. 空间支持、QA、监督像元和损失归一化一致

- 监督位置：`indices = where(target.mask & band.valid_mask)`，即只有"任务目标 ∧ QA 有效"的像元被监督。
- A 的目标是矩形内**完整包含**的原生像元（`spatial_hide` 的 `contained`），Encoder 看不到它们；
  相邻 Token 用 `represented_boundary_v1`（有效代表区域边/角邻接）检索。
- 损失：每个目标波段先算 `Σerr² × w / (active × count × σ²)`，其中 `count` 是几何规划给出的
  **受支持**像元数，`σ` 是该波段 TRAIN-only 标准差；随后按任务内波段平均、按来源/课程权重加权。
- 一致性由两处硬检查保证：规划时 `supported_count` 与实际 Decoder 支持数必须相等
  （`stream_task_roots` 中 `seen != count` 直接抛错）；本轮真实证据里 plan/decoder 支持数逐波段相同
  （`evidence/architecture_evidence.json/members[*].support.*.decoder_equals_plan`）。
- 真实更新实测：裁剪前总梯度范数 2,909 / 2,056（更新 0/1，阈值 3），逐模块为 Encoder 2,870/2,025、
  Decoder 475/359；逐波段支持数与 Decoder 实际支持数完全一致。该量级来自 `1/σ²` 的 NMSE 归一化与逐 Token
  误差累加，不是支持数或归一化错误，详见 `09_remaining_decisions.md` D3。
- 没有"用 0 代替缺失"的路径：无支持的真实任务返回 `NO_VALID_SUPPORTED_TARGET` 且不计入该任务的平均，
  但有效批量/来源权重仍按冻结规则计算，不做静默重归一化。

### 8. 不同观测在批处理时没有特征串扰

`ParticipationEncoder.forward_batch` 按观测独立准备几何，局部 Token 按长度分桶后作为**批维**送入共享
局部 Transformer，全局 Transformer 只对同长度观测做批堆叠（注意力在批维内独立），没有使用跨观测的 padding
或掩码。真实证据：同一观测成对送入批处理时，两份 Token 的最大绝对差为 0，与单独编码的差也在浮点重排量级；
测试 `test_participation_batch.py` 覆盖了不同波段数与不同长度的情形。E0 默认微批量为 1，
因此训练路径本来就是一观测一图，批处理只是可选执行方式。

### 9. 真实光学训练不需要跨卫星配对

E0 的任务只有 A 与 B1，二者都只读取单一观测。`joint.layout` 要求"8 个不同观测，每个观测恰好一个 A 和一个 B"，
不要求 B2；`adaptive_joint.accumulate_adaptive_joint` 只有在传入 JEPA 视图时才要求配对与准入。E0 的采样配方
`B2_used=false`、面板 `pair_admissions` 为空，preflight 在 16 个 A/B1 视图上完成真实前后向，未触及任何配对逻辑。

### 10. 不存在旧 B2 配对门禁阻塞 E0 的情况

旧阻塞有三处，本轮分别处理，且都没有放宽旧门禁：

| 阻塞点 | 旧行为 | 本轮处理 |
|---|---|---|
| 训练门禁 | `require_training` 要求 14 项旧门禁（含 `pair_admission`、`frozen_data_manifest` 等） | 新增独立 `require_optical_selfsup_training` 与 8 个 E0 门禁 id；旧函数、旧 id、旧配置原样保留 |
| 指标 | `aggregate_validation` 无 B2 时不可用（`S` 为 None） | 新增独立 `aggregate_intra_validation`（`S_intra=(A+B1)/2`）；旧函数仍是旧名旧语义 |
| 面板/日程加载器 | 构造时强制 `require_training(config)` | 增加显式 `gate=` 参数，默认仍是 `require_training`（旧路径不放宽）；E0 在预检用 `optical_identity_only`，在训练入口用 E0 门禁 |

`evidence/preflight_check.json` 同时报告 `historical_gate_authorized=false` 与
`optical_training_authorized=false`：E0 的检查可以在旧门禁全关的情况下完成，而正式训练仍然被关闭。

## 本轮修复的确定缺陷

| 编号 | 缺陷 | 影响 | 修复 | 回归测试 |
|---|---|---|---|---|
| D1 | `formal_runner.run_route` 在每个逻辑更新里把**下一个**更新准备两次（循环末尾一次、下一轮开头一次） | 每次更新多一次完整数据读取 + Merge 几何规划（CPU 瓶颈的直接来源） | 一步前瞻缓存：每轮只 `prepare` 一次，窗口耗尽即停止 | `test_next_schedule_entry_is_prepared_once` |
| D2 | 早停状态（best/stale）只存在内存，checkpoint 不保存 | 中断恢复后早停决策与连续执行不一致 | 早停状态随每个 checkpoint 写入 `sampler.runner_state`，恢复时读回；并在循环顶部按已提交状态先判定 | `test_runner_restores_the_committed_early_stop_decision`、`test_route_step` 新用例 |
| D3 | 墙钟预算只在完整参数更新之后检查 | 已超预算仍会再完成一次完整更新 | 循环顶部先检查，再决定是否准备/执行下一次更新 | `test_wall_limit_is_checked_before_the_next_update` |
| D4 | 旧指标/门禁把 E0 与 B2 绑在一起 | E0 无法独立验证或启动 | 独立 `S_intra`、独立门禁、显式 `gate=` 注入 | `test_optical_selfsup.py` 指标与门禁用例 |
| D5 | `formal_schedule` 候选生成器要求固定 8 个成员并在所有更新中复用 | 只训练固定 8 个观测 | 新增惰性采样配方与 `OpticalSelfsupSchedule`（见 `02_data_sampling_report.json`）；旧生成器保留 | 采样确定性/去重/来源均衡用例 |
| D6 | 生产入口没有显式拒绝测试替身 | Mock 对象可能进入 checkpoint/训练事务 | `_check_modules` 与 `run_route` 检查模块真实类型 | `test_mock_modules_cannot_enter_the_production_route` |
| D7 | 指标 JSON 没有区分"逻辑更新"与"真实优化器步" | 恢复与预算核算容易混淆 | 每次更新记录 `optimizer_steps=1` 与更新前游标 | `test_runner_state_and_optimizer_steps_are_committed` |
| D8 | 采样日程没有记录"池级排除" | 大场景被排除时无法审计 | 池规则写入配方（含上限与逐传感器排除计数），加载器重算并校验 | `test_pool_cap_cannot_be_silently_lifted_by_a_recipe` |

## 未修改、只记录为后续研究的项

- 矩形遮盖与 Merge 支持单位不等价（旧语义，保留）。若未来要把 A 的目标改成"整组遮盖"，属于科研变更，
  需要新的协议、面板和指标。
- 1 个可用波段的观测无法构造 B1（`hidden_band_indices` 直接报错）；2 个波段时只能单隐藏。
  当前发布最小 4 波段，E0 池规则要求 ≥2 波段，这一边界写进了配方与报告。
- 完整注意力对超长 Token 序列的二次成本（见第 4 节的边界与 `05_performance_comparison.json`）。
- 张量并行、BF16、JEPA 同观测表征预测：不在本轮改动范围。

## 数据与代码证据

- 真实数据证据：`evidence/architecture_evidence.json`
- 预检证据（配置/数据/日程/面板/环境 + 真实前后向）：`evidence/preflight_check.json`、`evidence/preflight_no_update.json`
- 单测：`test_optical_selfsup.py`（17 项）、`test_formal_runner.py`（8 项，含 4 项新增）、`test_route_step.py`（新增 2 项）、
  `test_discovery.py`（在缺少 ripgrep 的环境下显式跳过而非报错）
- 旧审核复用：`reports/formal_readiness_audit/02_architecture_audit.json`、`04_task_audit.json`、
  `06_numerical_audit.json`、`08_execution_audit.json`
