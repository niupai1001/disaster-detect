# 04 训练 Runner 与恢复系统

**结论**：E0 的 runner、事务式 checkpoint 与恢复路径已经可以在"配置/数据/日程/面板身份全部冻结"的前提下
确定性地续跑；本轮的 5 个 runner 缺陷已修复并有回归测试。**正式训练仍然被独立门禁关闭**，本轮没有执行任何
真实优化器更新。

## 1. 复用了什么，没有重写什么

E0 没有另起一套训练框架，直接复用：

- `route_step.execute_route_update`：一次逻辑更新 = 一次前向/反向 + 一次 `optimizer.step()` + 一次原子 checkpoint 提交，
  带 attempt 日志、锁文件、失败不重放。
- `route_control`（`RouteProgress`、`save_route_checkpoint`、`restore_route_checkpoint`、`inspect_route_checkpoint`）：
  完整保存模型、优化器、调度器、随机状态、`frozen_ids` 与优化器参数顺序。
- `formal_runner.run_route`：单路线推进循环。
- `validation_panels.FrozenValidationPanels` + `validation_metric`：固定面板与指标。
- `plan_prefetch.move_views`：CPU→设备搬运。

新增的只有 E0 需要的"入口与接线"：`config_optical_selfsup_v1.json`、`require_optical_selfsup_training`、
`selfsup_preflight`、`selfsup_train`，以及 schedule/panel 加载器的显式 `gate=` 参数。

## 2. 本轮修复的 runner 缺陷

| 编号 | 现象 | 根因 | 修复 |
|---|---|---|---|
| R1 | 每个逻辑更新把下一个更新准备两次 | 循环末尾算 `next_packet`，下一轮开头又 `prepare(index)` 一次 | 一步前瞻缓存：每个索引只准备一次；预算耗尽/早停时不再准备 |
| R2 | 恢复后早停决策与不中断不一致 | `best_validation`、`stale_validations` 只存在内存 | 早停状态随 checkpoint 写入 `sampler.runner_state`（`execute_route_update(runner_state=...)`），恢复时读回 |
| R3 | 已超墙钟预算仍会完成一次完整更新 | 只有更新后的检查 | 循环顶部先检查墙钟与已提交早停状态，再决定是否准备和执行 |
| R4 | 中断后无法判断"是否已更新参数" | 只有 checkpoint 存在性 | 保留原有 attempt 事务：`PREPARED → FORWARD_BACKWARD_QUEUED → OPTIMIZER_ABOUT_TO_EXECUTE → UPDATE_AND_EMA_COMPLETED → CHECKPOINT_COMMITTED`，并且 `optimizer_may_have_executed` 一旦置真就不允许自动重放；`save_route_checkpoint` 对已存在文件 `FileExistsError` |
| R5 | 指标 JSON 不能区分逻辑更新与优化器步 | 之前只有 `result.optimizer_updates=0`（表示"本函数未步进"的历史字段） | 每次更新额外记录 `optimizer_steps=1`、`optimizer_steps_before=<更新前游标>` |
| R6 | 生产入口不拒绝测试替身 | 没有类型检查 | `_check_modules` 拒绝 `unittest`/`mock` 模块与非 `torch.nn.Module`；`run_route` 在入口再查一次 |

对应测试：`test_formal_runner.py::test_next_schedule_entry_is_prepared_once`、
`test_runner_restores_the_committed_early_stop_decision`、`test_wall_limit_is_checked_before_the_next_update`、
`test_mock_modules_cannot_enter_the_production_route`；`test_route_step.py::test_runner_state_and_optimizer_steps_are_committed_with_the_checkpoint`、
`test_default_gate_blocks_and_an_injected_gate_is_used`。

## 3. 恢复语义（为什么续跑等于不中断）

一次逻辑更新 `u` 的提交顺序固定为：

1. `progress.next()` 先占位（超过 10000 上限时在**任何工作之前**拒绝）。
2. 校验输入签名（成员/掩码/目标索引/原点/P/Merge 配方/数据版本）。
3. 写 attempt 记录（`PREPARED`）并加锁。
4. 前向/反向累加梯度（不步进）。
5. 记录 `OPTIMIZER_ABOUT_TO_EXECUTE` → `optimizer.step()`。
6. 如需验证：切 eval、`no_grad` 收集固定面板、恢复训练态、`scheduler.validate`。
7. 写 checkpoint（模型/优化器/调度器/RNG/`sampler`，其中 `sampler` 含 `next_update`、`next_input_signature`、
   `runner_state`），随后写 metrics，最后标记 `COMPLETE`。

恢复时 `selfsup_train --resume <checkpoint>` 使用 `restore_route_checkpoint` 恢复以上全部状态，并取
`sampler.next_input_signature` 作为下一次更新的期望签名、取 `sampler.runner_state` 作为早停状态。
因为输入签名只依赖冻结数据与配方，所以"中断后第一次准备的输入"与"不中断时的第 u+1 次输入"逐字节相同
（`update_signature` 覆盖任务签名、目标索引哈希、原点、物理尺度、来源统计量）。

## 4. 启动入口

```bash
# 1) 只检查配置、数据、日程、面板、支持与运行环境（不读任何波段）
python -m encoder_v3.selfsup_preflight --config encoder_v3/config_optical_selfsup_v1.json --mode check

# 2) 真实数据、无优化器更新的前后向诊断（默认 P60/P120 两个真实逻辑更新 + 固定面板）
python -m encoder_v3.selfsup_preflight --config encoder_v3/config_optical_selfsup_v1.json --mode no-update --updates 2

# 3) 正式训练入口（默认门禁锁定；本轮不调用 --mode train）
python -m encoder_v3.selfsup_train --config encoder_v3/config_optical_selfsup_v1.json --mode check
python -m encoder_v3.selfsup_train --config encoder_v3/config_optical_selfsup_v1.json --mode train --device cuda:1 --directory <run dir>

# 4) 门禁命令行
python -m encoder_v3.gates --config encoder_v3/config_optical_selfsup_v1.json --optical
```

所有入口都先做参数解析与配置校验（缺字段、错 schema、旧门禁 id 混入、B2/JEPA 开关未置 false、任务头未置 false
都会在读取任何数据前报错）。`--mode train` 第一件事就是 `require_optical_selfsup_training(config)`，
当前 8 项 E0 门禁全为 `null`，因此它会以
`TRAINING_ENTRY_BLOCKED: Optical self-supervised training blocked: ...` 退出（退出码 2），不会创建目录、
不会读波段、不会步进优化器。

真实执行记录（本轮）：

- `selfsup_preflight --mode check`：通过，见 `evidence/preflight_check.json`。
- `selfsup_preflight --mode no-update --updates 2`：真实 TRAIN 数据、16 个 A/B1 视图的前向+反向，
  `optimizer_steps=0`，参数 SHA256 前后一致，见 `evidence/preflight_no_update.json`。
- `selfsup_train --mode check`：通过，报告 E0 门禁全关。
- `selfsup_train --mode train`：被门禁拒绝（本轮**没有**执行真实更新）。

## 5. 仍然需要人工批准的恢复/更新验收

按任务书第 8 节 D 类，真实 `optimizer.step` 本轮默认禁止。E0 正式启动前还需要一次**独立、受预算约束**的
"真实更新 + 中断恢复"验收：2 次匹配更新（P60/P120）+ 1 次中断续跑，记录参数、AdamW 动量、调度器 LR、
Python/NumPy/Torch/CUDA 随机状态、下一次输入签名与预算账本。该验收需要新的明确授权；此前已停止的
186 次预检更新预算不得恢复使用。

合成更新/恢复已经在既有测试中覆盖（前向、反向、AdamW、梯度累积、checkpoint、中断恢复、学习率调度、早停、
输出/梯度回归）；合成更新不等于真实数据训练授权。

## 6. 边界与风险

- 目前只有单路线（`phase="shared"`）。旧的 `A+B` / `A+B+JEPA` 分支初始化、`fork` 与双路线记账保持原样，
  但没有被 E0 使用；E0 不需要 2000 更新前缀。
- 每个逻辑更新写一个完整 checkpoint（约百 MB 级）。10000 次更新的存储需求需要在资源门禁里显式批准；
  本轮没有实现滚动保留，因为那会改变"任意更新都可恢复"的语义。
- 墙钟检查只能把超支限制在"最多一次更新"的量级；它不会中止正在执行的更新（中止会留下不确定的参数状态，
  按设计必须人工对账）。
- 真实一次更新的耗时远高于早先 28 秒的记录（当时 8 个观测的 Token 总量约 54.8 万、且场景更小）。
  本轮在真实 128 像素场景上重新测量，耗时与显存见 `05_performance_comparison.json` 与
  `evidence/preflight_no_update.json`；这是资源配置决策，不是 runner 缺陷。
