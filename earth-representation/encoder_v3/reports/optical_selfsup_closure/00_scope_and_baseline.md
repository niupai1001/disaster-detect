# 00 本轮范围与基线

**实验名**：`optical_intra_selfsup_v1`（E0：A 空间遮盖重建 + B1 原生波段遮盖重建）

**本轮目标**：把第一阶段多源光学自监督预训练系统推进到"科研配置与数据正式冻结、获得训练授权后可直接启动"的
状态；不追求 B2 配对、跨卫星几何校准、JEPA 或 EnMAP 准入。

## 1. 仓库与分支

| 项 | 值 |
|---|---|
| 仓库 | `niupai1001/disaster-detect`（本地 `N:/Users/yzh/Disater`） |
| 目标目录 | `earth-representation/encoder_v3/` |
| 最新已审基线 | `audit/formal-readiness-269a1e0`，提交 `5c99db6f448dc91db0627feec6aeb5602c6fecba` |
| 本轮开发分支 | `dev/optical-intra-selfsup-v1`（从 `5c99db6` 创建，保留原有未提交修改） |
| 未提交修改 | `reports/runtime_fault_fragments/CPU_autograd_102964/` 下两个历史运行产物被删除（本轮未触碰、未提交） |
| 远程 HEAD 比较 | 本环境 `git fetch` 因 SSH 管道错误（`couldn't create signal pipe, Win32 error 5`）失败；本地 `origin/audit/formal-readiness-269a1e0` 指向 `5c99db6`，与基线一致，`origin/master` 为 `c648170`。**本轮无法确认远程是否又有新提交**，这是环境限制，不是仓库状态结论 |

`git status` 在开工时确认：除上述两条删除外工作区干净，没有覆盖或重写更新的工作。

## 2. 本轮做了什么

1. **P0 架构与任务审核**：复核 18 个模块，修复 8 个确定缺陷，新增真实数据证据与 21 项回归测试
   （见 `01_architecture_audit.md`、`07_tests.json`）。
2. **P1 数据与采样系统**：把"固定 8 个成员"的候选日程改成"冻结 TRAIN 成员池 + 确定性来源均衡采样 + 惰性任务生成"，
   并给出全量曝光/覆盖/任务分布统计（`02_data_sampling_report.json`）。
3. **P2 独立验证指标**：新增 `S_intra=(L_A+L_B1)/2`、固定 VAL 面板与分层诊断，旧 `S=(A+B)/2` 完全隔离并保留
   （`03_validation_protocol.md`）。
4. **P3 训练入口与恢复**：新增独立 E0 配置、独立门禁函数、`selfsup_preflight` 两个模式与 `selfsup_train`
   正式入口；修复 runner 的重复准备、早停状态、墙钟检查等问题（`04_runner_and_recovery.md`）。
5. **P4 计算优化**：完成真实数据无优化器更新的测量，定位并采用一个可验证的执行优化（精确 cell-indexed 读出，
   B1 读出 55.5 s → 2.6 s，支持数/Token/NMSE 一致），并测量局部批大小、打包、微批量与全局注意力边界
   （`05_performance_comparison.json`）。
6. **P5 同观测 JEPA**：完成可行性评估与候选设计，未实现、未启用（`06_jepa_candidate.md`）。
7. **P6 测试与门禁**：逻辑测试、合成训练测试、真实数据无更新测试、门禁矩阵与最终报告
   （`07_tests.json`、`08_gate_matrix.json`、`10_final_report.md`）。

## 3. 明确没有做的事

- 没有执行任何真实优化器更新（`optimizer_steps=0`；此前已停止的 186 次预检预算没有恢复使用）。
- 没有修改已冻结的 Encoder 结构（SRF、GSD 条件、Hypernetwork、动态卷积、单轮 Merge、双尺度 Token、
  局部/全局 Transformer 宽度与层数、完整注意力）。
- 没有删除或改写 B2 代码、历史配对审查结果；没有运行跨源配准搜索或重新处理 349 个候选配对；
  没有为凑 B2 修改原始反射率或 QA。
- 没有引入卫星 ID Embedding、PSF、递归 Merge、双分支 Token 冗余，也没有统一不同卫星的波段数或重采样原生波段。
- 没有把 JEPA 接入 E0 的训练入口或指标。

## 4. 运行环境（测量与证据的可复现条件）

| 项 | 值 |
|---|---|
| Python | 3.12.14（`landslide-seg/.venv-use-independent`） |
| PyTorch | 2.11.0+cu128 |
| GPU | 2 × NVIDIA GeForce RTX 5090（32 GB）；本轮所有 GPU 证据在物理 GPU 1（`CUDA_VISIBLE_DEVICES=1`，进程内 `cuda:0`）上产生 |
| 数据发布 | `observation_model/outputs/encoder_v2/data_expansion`（7,778 成员：TRAIN 7,395 / VAL 383） |
| 物理契约 | `observation_model/outputs/encoder_v2/G1` |
| 数据身份 | members `b3c16c5b…943c`，contracts `5a3eacd7…3a5a`，srf `e6c7ee16…ac4d`，statistics `d2849268…684f` |

说明：本机历史上出现过 CUDA 原生访问冲突（`pair_quality_runtime_diagnostics.json`），本轮所有 GPU 运行均正常结束；
但这不等于已排除该环境问题。

## 5. 判定口径

- **工程就绪（E0_ENGINEERING_READY）**：数据/任务/指标/runner/恢复/数值回归/真实无更新诊断/性能量化全部通过，
  且 B2、E1 不阻断。见 `08_gate_matrix.json`。
- **正式就绪（E0_FORMAL_READY）**：还需要科研批准（数据与 QA 冻结、S_intra 权重、资源与更新上限）与一次
  独立授权的真实更新/恢复验收。**本轮不把工程通过当作正式训练授权**；`required_before_training` 的 8 项
  E0 门禁全部保持 `null`。

## 6. 复现命令

```bash
cd earth-representation
PY=.../landslide-seg/.venv-use-independent/Scripts/python.exe

# 冻结输入（重建并校验哈希）
$PY encoder_v3/reports/optical_selfsup_closure/tools/build_frozen_inputs.py \
    --config encoder_v3/config_optical_selfsup_v1.json --verify

# 预检（配置/数据/日程/面板/环境；不读波段）
$PY -m encoder_v3.selfsup_preflight --config encoder_v3/config_optical_selfsup_v1.json --mode check

# 真实数据、无优化器更新的前后向诊断
$PY -m encoder_v3.selfsup_preflight --config encoder_v3/config_optical_selfsup_v1.json \
    --mode no-update --device cuda:0 --updates 2

# 采样覆盖与曝光统计
$PY -m encoder_v3.sampling_coverage --config encoder_v3/config_optical_selfsup_v1.json \
    --output encoder_v3/reports/optical_selfsup_closure/02_data_sampling_report.json

# 真实无更新性能对照
$PY -m encoder_v3.selfsup_benchmark --config encoder_v3/config_optical_selfsup_v1.json \
    --device cuda:0 --output encoder_v3/reports/optical_selfsup_closure/05_performance_comparison.json

# 门禁（应被拒绝）
$PY -m encoder_v3.gates --config encoder_v3/config_optical_selfsup_v1.json --optical

# 全部单元测试
$PY -m unittest discover -s encoder_v3 -t . -p "test_*.py"
```

## 7. 复用与限制

- 复用旧审核证据：`reports/formal_readiness_audit/`（架构、数据、任务、数值、运行时、执行、门禁矩阵）
  与 `reports/merge_rate_confirmation.json`、`reports/validation_metric_confirmation.json`、
  `reports/participation_current_execution_evidence.json` 等；没有重复做无意义的全项目扫描。
- 限制：
  - 远程仓库状态未能在本环境确认（SSH 失败）；
  - 本轮真实数据证据只覆盖有限成员与有限更新（预检 2 个逻辑更新 + 固定面板；性能对照有限案例），
    不等于质量验收；
  - 单次真实更新的耗时/显存是在本机单 GPU、FP32、完整注意力下的实测值，换硬件或换精度需要重测；
  - 所有性能数字都是"本轮测量"，不与旧报告中的单次加速相乘。
