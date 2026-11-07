# Encoder V3 训练准入审查（只读）

日期：2026-11-07
范围：`config.json` 的 `required_before_training` 14 项
依据：计划文档 §13「配置冻结清单」、`gates.py`（fail-closed）、仓库内既有批文
约束：**未修改任何原始配置或既有报告**，本文件与同名 `.json` 为新增审查产物

## 一、总表

| # | 准入项 | 判定 | 现有证据 | 缺口 |
|---|---|---|---|---|
| 1 | `optimizer_choice` | ✅ **满足** | `pilot_runs/pilot_stop_decision.json`：用户「不用继续预算，使用AdamW」 | — |
| 2 | `merge_configuration` | ⚠️ 仅预检 | `merge_B_confirmation`（B 机制）、`merge_levels_confirmation`（单轮B，scope=**formal**）、`merge_rate_confirmation`（曲线，预检） | 正式曲线组合与「60/120m × 任务遮盖」关系未审 |
| 3 | `hypernetwork_hidden_structure` | ⚠️ 仅预检 | `hypernetwork_preflight_review.json`：34→64→320、GELU、log(gsd) | 批文自注 `scope=preflight_only` |
| 4 | `frozen_data_manifest` | ❌ 未满足 | `physical_patch_role_review`（无新裁剪策略）、`frozen_pair_source_review`（real_pairs_admitted=0） | 合格清单与划分未冻结；名称发现 430 块待查、10 目标未全覆盖（M0） |
| 5 | `pair_admission` | ❌ 未满足 | `B2_QA_policy_confirmation`（仅 4 个观测）、`pair_spatial_holdout_disjoint`（admitted_pairs=0） | 偏移数值、校准隔离、残差/处理域准入均未定 |
| 6 | `mask_rectangle_distribution` | ⚠️ 仅预检 | `spatial_mask_preflight_review.json` | `scope=preflight_mask_rule_only` |
| 7 | `validation_panel` | ⚠️ 仅预检 | `pilot_members_panels_review`（成员+面板哈希）、`validation_metric_confirmation`（分层等权公式） | 批文用于预检；`real_image_reads=0` |
| 8 | `improvement_threshold` | ❌ 未满足 | 指标公式已确认 | **没有「多大改进算改进」的数值** |
| 9 | `stability_window` | ❌ 未满足 | 无 | 验证间隔、观察窗口、不确定性阈值均缺 |
| 10 | `early_stop` | ❌ 未满足 | 仅 `pilot_stop_decision`（预检被叫停） | 非正式早停规则 |
| 11 | `precision_acceptance` | ⚠️ 仅预检 | `fp32_equivalence_tolerance_review`：2e-6 / 1e-5 / 1e-6 / 2e-6 | `scope=preflight_only` |
| 12 | `resource_limit` | ❌ 未满足 | 仅 `gpu_update_execution_review`（有界检查） | 无计算资源上限、无总预算（`budget.json` 不存在） |
| 13 | `execution_equivalence` | ❌ 未满足 | `route_step_runtime_incident`（CPU autograd 与 CUDA 原生崩溃）、`cuda_fault_event_evidence` | 当前运行时未通过；历史 146 项不能替代当前验收 |
| 14 | `formal_training_authorization` | ❌ 未满足 | 无 | 必须为 `true`，只能由用户签发 |

**统计：满足 1 / 仅预检 6 / 未满足 7。** 当前 `gates.py` 会直接抛 `Training blocked`，`formal_training_started=false`，真实更新仍为 214。

## 二、结论（机制 + 证据 + 边界）

**机制层面已经定下来的部分，与「正式训练可否开始」不是同一件事。** 仓库里有一批写得很清楚的用户批文，但它们大多把范围写死在预检：

- `merge_levels_confirmation.json` 的 scope 是 `formal_merge_architecture`，这是少数按正式范围签的；
- 其余五项（超网络、遮盖规则、精度容差、成员面板、Merge 曲线）都写着 preflight。

也就是说：**不是没人签字，而是签的是「可用于预检」，不是「可进入正式训练」**。这五个字之差，是当前最主要的阻塞。

**另外有两条是硬缺口，不是签字问题：**

1. `frozen_data_manifest` 与 `pair_admission`——数据侧本身没冻结：配对真实准入数为 0，名称发现仍有 430 块未查。这个只能靠继续做数据工作，不能靠批文解决。
2. `execution_equivalence`——当前运行时存在原生访问冲突（`0xC0000409` 类），真实更新与恢复没在现环境下验收过。

**边界**：本审查只核对「仓库里是否有对应证据及其范围」，不重新验证数值本身；历史 146 项回归、214 次真实更新的记录我按原样引用，没有复跑。

## 三、需要你确认的五件事

1. **四项策略数值**：`improvement_threshold`、`stability_window`、`early_stop`、`resource_limit`。这些是设计决策，Agent 不能替你定。
2. **五项预检批文是否升级为正式冻结**：merge 曲线、超网络结构、遮盖规则、精度容差、验证面板。若升级，需明确写「以预检值为正式值」。
3. **`execution_equivalence` 如何处理**：先修运行时（需要重启 GPU 或排查崩溃），还是以预检容差放行。
4. **数据与配对准入是否缩范围**：是否把 `frozen_data_manifest` / `pair_admission` 限定到已复核的 4 个观测，而不是等全量发现完成。
5. **`formal_training_authorization` 是否签发**：这一项只能你签，且应在 1–4 有结论之后再签。

## 四、产物

- `reports/admission_review_2026-11-07.json`（机器可读，含逐项证据与缺口）
- 本文件（人读版）
- 未改动：`config.json`、所有既有报告、代码与结果
