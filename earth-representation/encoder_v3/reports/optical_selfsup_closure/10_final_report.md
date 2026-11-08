# 10 最终报告（E0：A 空间遮盖 + B1 原生波段遮盖）

**实验**：`optical_intra_selfsup_v1`
**分支**：`dev/optical-intra-selfsup-v1`（自 `5c99db6` 创建，保留原有未提交修改）
**本轮优化器步数**：0
**E0 门禁**：8 项全部 `null`（未授权）

## 1. 现在处于什么状态

A+B1 的数据、任务、指标、runner、恢复、真实无更新诊断与性能量化都已完成并通过测试；
**正式训练入口存在、可执行、默认被门禁拒绝**（`TRAINING_ENTRY_BLOCKED ... exit=2`）。
剩下的是科研决定（数据/QA 冻结、池上限、指标权重、调度与资源、真实更新验收、最终授权），
不是继续写代码。

| 判定 | 结论 | 依据 |
|---|---|---|
| E0_ENGINEERING_READY | **是**（12/12 项见 `08_gate_matrix.json`） | 244 项测试通过、2 个真实逻辑更新、固定面板 `S_intra` 可算、真实无更新诊断通过 |
| E0_FORMAL_READY | **否** | 8 项 E0 门禁未批准；缺一次独立授权的真实更新/恢复验收 |

## 2. 本轮交付物

- **代码**：`config_optical_selfsup_v1.json`、`selfsup_preflight.py`、`selfsup_train.py`、
  `selfsup_benchmark.py`、`sampling_coverage.py`，以及对 `gates.py`、`validation_metric.py`、
  `formal_schedule.py`、`scheduled_data.py`、`validation_panels.py`、`formal_panels.py`、
  `route_step.py`、`route_control.py`、`formal_runner.py` 的最小修复（8 个确定缺陷）。
- **测试**：`test_optical_selfsup.py`（17 项）+ `test_formal_runner.py`（+4）+ `test_route_step.py`（+2）
  + `test_discovery.py`（环境缺 ripgrep 时显式跳过）。全套 244 项：243 通过、1 跳过、0 失败。
- **冻结输入**：`frozen/sampling_recipe.json`（10,000 更新的惰性采样配方，无逐更新张量）、
  `frozen/validation_panels.json`（54 个固定案例）。
- **证据**：`evidence/preflight_check.json`、`evidence/preflight_no_update.json`、
  `evidence/architecture_evidence.json`、`evidence/benchmark_real_updates.json`、
  `evidence/benchmark_sweep.json`、`evidence/readout_before_after.log`、`evidence/tests.json`。
- **报告**：`00`–`09` 与本文。

## 3. 关键实测数字（同一台机器、单卡 RTX 5090、FP32、无优化器步）

| 指标 | 值 |
|---|---|
| 训练池 | 6,809 成员（S2 4,763 / Landsat 2,046），计数排除 586 个超限 S2 十波段场景 |
| 采样（10,000 更新） | 80,000 个 A + 80,000 个 B1；来源严格 4+4；成员覆盖 6,808/6,809 |
| B1 隐藏数量 | 单隐藏 39,951 / 多隐藏 40,049；多隐藏覆盖 2..9 |
| 固定面板 | 54 例 / 3 个 VAL 观测 / B2 配对 0；`S_intra` 可计算（随机初始化，非质量结论） |
| 真实更新 Token | 492,595（P60）/ 642,123（P120） |
| 真实更新耗时 | 50.2 s / 65.4 s（前后向，采纳的读出+局部批 1024）；含 CPU 准备为约 62.5 / 78.4 s |
| 峰值显存 | 5.11 GiB / 5.74 GiB（分配）、6.41 / 6.08 GiB（保留） |
| 全局注意力规模 ΣN² | 4.01e10 / 6.37e10（每观测 Token² 求和） |
| 裁剪前梯度范数 | 2,909 / 2,056（阈值 3） |
| 读出优化 | B1 54.0 s → 3.1 s（17.5×，两次测量 17.5–21.7×），支持/Token 完全一致，NMSE 相对差 1.5e-9 |
| 局部批 1024 | 1.79×（3 次重复），梯度相对 L2 8.6e-8，显存不变 |
| 10,000 更新外推 | 约 7.2–9.1 天（含 CPU 准备，线性外推；不含每 250 步验证 92.8 s 与 I/O） |

## 4. 八个必答问题

### 4.1 A+B1 是否已经具备工程训练条件？

是。数据身份、SRF/GSD/动态卷积、原生与合并双尺度、A/B1 隐藏值隔离、支持与损失归一化、批处理独立性、
无配对依赖、独立验证指标、确定性恢复、真实数据前后向、性能瓶颈量化与采用的优化，都有代码或实测证据
（`01_architecture_audit.md`、`05_performance_comparison.json`、`08_gate_matrix.json`）。
"工程就绪"不等于"可以开训"：8 项门禁仍未批准。

### 4.2 与旧版相比，计算时间和峰值显存分别降低多少？

- **读出**：旧参考路径 54.0 s → 精确 cell-indexed 3.1 s（17.5×，同一视图、同一权重、支持数与 Token 数相同，
  NMSE 相对差 1.5e-9）。这是本轮的主导优化，因为旧路径对每个候选 Token 做一次 Python 循环。
- **局部 Transformer**：局部批 64 → 1024 在小场景有界对照上为 1.79×（3 次重复，显存 815 vs 798 MB，
  梯度相对 L2 8.6e-8）；在真实模态更新上为 1.90×/1.82×（50.2 s vs 95.4 s、65.4 s vs 118.9 s），已采纳。
  局部批 256/512 为 1.60×/1.62×。
- **未采纳**：打包输出 1.09×（噪声内）、Matrix A 读出 1.04×（本案例不足以证明收益）、
  观测微批量 2/4/8 只有 1.01–1.04× 且峰值显存升到 1.6/2.8/4.9 倍。
- **端到端**：优化后的真实更新为 50–65 s（前后向）、峰值 5.1–5.7 GiB；含 CPU 准备约 63–78 s。旧参考读出的完整更新未能在本轮预算内跑完
  （单视图 54 s × 16 视图），因此端到端"优化前"只报告逐视图测量，不编造整更新数字；
  这也是 `05_performance_comparison.json` 的 limitations 之一。

### 4.3 是否仍有训练正确性缺陷？

没有已知的正确性缺陷落在已审路径上。需要如实记录两点：

1. 真实数据预检过程中出现过**一次** `participation_layout` 的 `np.minimum.reduceat` 越界异常，
   之后用同一配置、同一数据、同一代码重复运行（含全部 54 个面板案例与两个更新）都没有复现；
   本轮按"环境/内存瞬时问题"记录，未改动该模块。若要正式训练，建议在预检后追加一次同配置重复运行作为环境稳定性检查。
2. 缺的不是正确性，而是**真实更新/恢复验收**（需要授权），以及 D1–D4 的科研数值选择。

### 4.4 S2/Landsat 是否均可以参与当前训练？

可以。两者都在冻结池内，每个逻辑更新固定 4+4 个 slot（来源等权），暴露分布可审计（`02_data_sampling_report.json`）；
固定面板同时包含两者。S2 的十波段成员因原生场景上限只保留 8 个（113–127 像素），几乎不会出现在训练中；
这是一个已记录的数值边界，不是"某个来源不能训练"。

### 4.5 EnMAP 以后接入需要补哪些数据条件，而不是重写哪些模块？

需要补：进入发布清单的成员、可追溯的真实 SRF（不能是派生高斯）、`instrument:band` 的 TRAIN-only 统计键、
原生波段合同与 ≥2 个光学波段、场景像素不超上限、以及（若要维持来源等权）把 EnMAP 作为独立来源分配 slot 并重算覆盖报告。
**不需要重写**：连续 SRF Encoder、GSD 条件、逐波段动态卷积、单轮 Participation Merge、双尺度 Token、
共享 Transformer 与 A/B1 任务逻辑；它们都不假设卫星身份或统一波段数。详见 `09_remaining_decisions.md` D7。

### 4.6 同观测 JEPA 是否值得进入下一实验？

值得作为**下一轮独立 E1 候选**，但本轮不实现、不启用。理由：几何与 Predictor 可复用，视图协议、目标区域、
潜在损失归一化与非坍塌诊断必须新写（约 100–150 行 + 独立配置/门禁 + 合成稳定性测试），
成本不是"最小改造"；且 A 任务已经在同一隐藏区域做反射率监督，E1 的增量价值必须靠消融证明。
E0 的 A+B1 结果是 E1 的比较基准。详见 `06_jepa_candidate.md`。

### 4.7 距离 E0 正式启动只差哪些具体批准或测试？

1. D6 数据/QA/统计量正式冻结（`optical_data_freeze`）；
2. D1 训练池原生像素上限（`optical_sampling_recipe`）；
3. D2 `S_intra` 权重 0.5/0.5 与固定面板（`optical_validation_protocol`）；
4. D3 优化器/裁剪阈值/调度/课程（`optical_optimizer_schedule`、`optical_task_protocol`）；
5. D4 更新上限与墙钟/显存预算（`optical_resource_budget`；按实测 10,000 步约 7–9 天）；
6. D5 一次独立授权的真实更新 + 中断恢复验收（`optical_update_recovery_acceptance`）；
7. 最终训练授权（`optical_formal_training_authorization`）。
每项都在 `09_remaining_decisions.md` 给出选项、建议与影响。

### 4.8 是否存在任何会造成额外真实优化器更新的风险？

没有发现自动更新路径：`selfsup_preflight`（两个模式）与 `selfsup_benchmark` 都不调用 `optimizer.step`，
并在结束时校验参数 SHA256 不变；`selfsup_train --mode train` 第一行就 `require_optical_selfsup_training`，
当前 8 项门禁为 `null` 时直接拒绝；`execute_route_update` 在参数上限、输入签名、面板身份或门禁任一不符时
在任何工作之前报错；checkpoint 与 attempt 文件禁止重放；墙钟检查已移到更新之前；没有自动重试。
唯一能产生真实更新的是"批准门禁后显式运行 `--mode train`"或直接调用带批准配置的 `execute_route_update`，
两者都需要人为动作。

## 5. 与旧基线的差异（可复核）

| 项 | 旧（`5c99db6`） | 新 |
|---|---|---|
| 训练门禁 | 14 项旧门禁；E0 被 B2/frozen_data 等阻塞 | 8 项独立 E0 门禁；旧门禁仍拒绝 E0 配置 |
| 验证指标 | `S=(A+B)/2`，要求 B2 | 独立 `S_intra=(A+B1)/2`，B2 出现即报错；旧函数未改 |
| 训练日程 | 固定 8 个成员重复 | 冻结池 + 确定性 4+4 采样，10,000 更新惰性生成 |
| Runner | 每步准备两次；早停状态不入 checkpoint；墙钟只在更新后查 | 一步前瞻；早停状态随 checkpoint 提交；墙钟前置；拒绝 Mock |
| 指标记录 | 只有逻辑更新 | 每次更新记录 `optimizer_steps=1` 与更新前游标 |
| 执行默认 | 参考读出、局部批 64 | 精确 cell-indexed 读出（17.5×）、局部批 1024（1.79×），数值回归通过 |
| 训练入口 | 无 E0 入口 | `selfsup_preflight`（check/no-update）+ `selfsup_train`（默认拒绝） |

## 6. 已知限制

- 远程仓库状态未能在本环境确认（`git fetch` SSH 管道错误）；本地 `origin/audit/formal-readiness-269a1e0`
  与基线一致。
- 真实证据只覆盖 2 个逻辑更新 + 54 个面板案例 + 有限性能案例，不等于质量验收。
- 一次布局异常未复现，已如实记录。
- 10,000 更新外推是线性估算，不含每 250 步的固定面板验证（实测 92.8 s/次）与数据 I/O。
- 没有 TEST/GUARD 集，评价只能报告 VAL。
- 性能数字只对本次硬件、FP32、单卡有效。
