# EncoderV3 正式训练前研究者决策请求

**基线**：`269a1e04c8a30510176da01231ce43c3d125bce7`（审计分支 `audit/formal-readiness-269a1e0`）
**状态**：2026-11-08 已收到用户选择；D4-D6 已正式冻结，D7-D8 已确认，D9 接受候选控制值，D10 已执行但发生授权超用并失败，正式训练授权仍保持未授权。本文件不是训练授权。

## 2026-11-08 用户审批结果

| 决策 | 用户选择 | 执行结果 |
|---|---|---|
| D1–D3 | 包含 EnMAP（需单独准入） | EnMAP 未在 7,778 成员、无实测 SRF、无统计键；`03a_enmap_admission.json` 判 `BLOCKED_NOT_ADMITTED`，不得并入训练/统计 |
| D4–D6 | 批准推荐冻结包 | 架构/Merge/矩形遮盖/执行默认值已写入 `confirmed_formal`；`optimizer_choice`、`merge_configuration`、`hypernetwork_hidden_structure`、`mask_rectangle_distribution` 已结构化批准；验证面板仅批原则，未冻结精确面板 |
| D7–D8 | B2 只读诊断、JEPA 关闭 | B2 准入仍 0；JEPA 五项条件仍 null，保持关闭 |
| D9 | 接受候选控制值，其余另定 | warmup 100/课程 2000/验证 250/patience 4/LR×0.5/min 1e-5/10,000 作为候选控制；improvement/stability/resource 仍未冻结 |
| D10 | 授权最小验证 | 先前超用经 addendum 接受；三次尝试累计 6 次唯一更新/9 步。修正版严格 2 次唯一更新 + 1 次恢复，参数 max 7.45e-9、优化器相对 L2 1.47e-9、loss 差 2.47e-8，落在旧 FP32 预检容差内，非正式新架构容差；见 `evidence/controlled_update/report.json` / `incident.json` |
| formal_training_authorization | 保持未授权 | 仍为 null |


## D1. 正式数据范围与成员清单

- **背景**：现有冻结原生发布有 7,778 个成员（7,395 TRAIN / 383 VAL），已做缓存重读、血缘复现、统计重算和跨角色元数据隔离；但没有正式数据清单批文，也没有 TEST/GUARD。
- **当前设计**：`frozen_data_manifest=null`；pilot 只用了 8 个 TRAIN / 8 个 VAL 观测。
- **选项**：
  - A（推荐）：把该 bounded 7,778 成员发布作为正式候选范围，冻结成员/角色/波段/缓存/SRF/QA/统计身份；另建 TEST/GUARD 或明确本轮无 TEST/GUARD。
  - B：只批准其中一个合格子集，并给出逐成员排除理由和空间隔离证据。
  - C：继续全机名称补扫后再定；代价是无法在近期冻结数据。
- **推荐**：A，理由是数据身份、缓存重读、血缘、统计和跨角色隔离证据完整；未审计的挂载目录/压缩包不进入训练范围，但列为覆盖边界。
- **影响**：决定正式训练成员、验证面板、统计量和资源规模；不批准则 `frozen_data_manifest` 阻塞。

## D2. 全发布 QA 策略

- **背景**：只对 B2 初始四个观测有历史 QA 确认；全量 7,778 成员的 QA/处理域策略没有正式批文，`lineage_qa_header_review.json` 仍有未解释的 categorical-warp 警告。
- **选项**：接受现有 native QA 规则（推荐，已有重读/血缘/统计证据）；或为正式范围增加 SCL/气溶胶/云/阴影规则并重算；或缩小到 QA 证据完整的子集。
- **影响**：QA 决定有效像元、监督支持、统计量和数据角色；不批准则不能冻结数据清单。

## D3. TRAIN-only 统计量正式冻结

- **背景**：`engineering_train_stds` 仍是 engineering 名称；`statistics.json` 声明 TRAIN-only；独立重算 7,395 个 TRAIN 缓存与历史统计计数一致，最大均值差 2.5e-16、标准差差 4.7e-16，43/43 个键齐全且 std 有限正。
- **选项**：A（推荐）把该统计量正式冻结并绑定当前 release/statistics 哈希，代码保留名称并加正式别名；B 重算并替换统计文件（会改变数据版本）；C 不接受历史统计。
- **影响**：决定归一化尺度和所有 NMSE；不批准则 `frozen_data_manifest` 与损失尺度未定。

## D4. 架构/执行默认值正式冻结

- **背景**：以下值目前是候选或代码默认值，不在 14 项门禁内：SRF 编码器结构/波长归一化、动态卷积边界规则、Decoder 162→128→1 与分块、Transformer 4/4 层、4 heads、512 FFN、local batch 64、microbatch 1、cell-indexed Decoder、packed/retained geometry 默认关闭。
- **选项**：A（推荐）把当前已测试值逐项列入正式冻结清单并给出源码/配置哈希；B 只冻结主架构，执行参数另行小规模预检；C 改变任何结构/默认值（属于新设计）。
- **推荐**：A，保持与当前 215 项回归和真实无更新诊断一致；执行加速仍默认关闭，直到有正式性能/等价证据。
- **影响**：决定可复现性和 checkpoint 解释；不批准则 `hypernetwork_hidden_structure` 等门禁不完整。

## D5. Participation 曲线与 P/遮盖关系

- **背景**：GSD→r 曲线当前只确认用于预检；单轮 B、原生/合并双尺度已确认；P=60/120 与矩形遮盖/ Merge 支持单位的关系未正式定稿。
- **选项**：A（推荐）把现有全精度 Hill 曲线升级为正式冻结，明确 P 只作任务遮盖/物理网格上下文、不改变 Merge；B 重新拟合曲线；C 改变遮盖单位（矩形 vs Merge 支持）。
- **证据**：0.5/1.21/10/20/30 m 的实际率分别为 0.75/0.6567/0.1992/0.0996/0.0645，30 m 窗口约 32.05 m；P60/P120 在同输入下 Merge 组成员完全一致。
- **影响**：决定 Token 数、参与率、A 遮盖语义；不批准则 `merge_configuration`、`mask_rectangle_distribution` 仍为预检。

## D6. 正式验证面板与 B1 多隐藏覆盖

- **背景**：已确认指标要求 B1 单隐藏/多隐藏各半，多隐藏内部按隐藏波段数 2..C−1 等权；现有 pilot 面板只含多隐藏 count=2，且 B2 为 0。
- **选项**：
  - A（推荐）：冻结正式面板，每个 7/10 波段观测覆盖 hidden count 2..C−1；B2 缺席时正式 S 不可用，另设明确标注的 A+B1 诊断，不替换正式标量。
  - B：修改已确认指标，仅在可用 hidden count 上等权（与既有确认冲突，需重新确认公式）。
  - C：暂不冻结面板，正式 S 保持不可用。
- **影响**：决定验证指标是否可计算、学习率调度是否有输入；不批准则 `validation_panel` 阻塞。

## D7. B2 配对决策包

- **背景**：349 个元数据候选、0 个准入；两个短间隔案例约 10 m 偏移，23.71 h 案例反向留出 4/16 变差；配对阈值、校准隔离、残差和处理域均未批准。
- **选项**：
  - A（推荐）：保持 0 准入；先批准校准数据隔离与固定偏移准则，再批准具体偏移和 TRAIN/VAL 配对；B2 正式训练继续关闭。
  - B：仅批准两个短间隔案例做只读诊断，不进入梯度。
  - C：现在批准一组偏移并训练（缺独立校准/残差/变化风险证据，不建议）。
- **影响**：B2 损失、验证 B 分项、JEPA 配对和跨源能力；不批准则 `pair_admission` 与 B2 仍为 0。

## D8. JEPA 五项条件与潜在损失

- **背景**：`required_before_jepa` 五项全为 null；无正式 latent loss/normalization；Predictor/EMA 只有工程候选。
- **选项**：A（推荐）保持 JEPA 关闭，先补 latent loss、跨源配对、A/B 稳定、有效批量集成、query projection 五项证据；B 只做独立只读诊断；C 现在启用（无依据，不建议）。
- **影响**：两路线之一和 JEPA 辅助目标；不批准则 JEPA 分支不可启动。

## D9. Warmup/课程/平台期/稳定性/早停/资源上限

- **背景**：候选值为 warmup 100、课程 2000、验证 250、patience 4、LR×0.5、min LR 1e-5、每条路线 10,000；`RouteSchedule` 已支持显式 warmup/验证间隔/更新上限，但这些数值仍是候选默认值，且 `improvement_threshold`、`stability_window`、`early_stop`、`resource_limit` 未批准。
- **选项**：A（推荐）用固定验证重复的波动确定 improvement/stability 阈值，再冻结 warmup/课程/patience/LR/资源上限；B 先接受上述候选作为工程诊断，不称为正式值；C 维持未定，正式训练不启动。
- **影响**：决定调度、早停和成本控制；不批准则 4 项门禁和 Runner 控制仍不完整。

## D10. 受控真实更新/恢复验证授权

- **背景**：真实更新闭环是正式训练就绪的必要证据；当前累计 214 次预检更新已完成，剩余 186 次已停止，不能使用；本轮只做了真实无更新诊断（16 视图，72.5 s，7.02 GiB，参数/预算未变）。
- **请求**：单独授权一次最小验证（不是正式训练、不是任务头）：
  - 输入：`pilot_study_candidate.json` 的 8 个 TRAIN 成员与 update-0 日程，执行时绑定缓存哈希；
  - 上限：1 张空闲 GPU，≤10 GiB 分配，≤20 分钟，2 次匹配 AdamW 更新（P60/P120）+ 1 次中断/恢复的下一步；不写正式训练目录；
  - 记录：参数、AdamW 动量、scheduler LR、Python/NumPy/Torch/CUDA RNG、下一输入签名、budget 账本；
  - 停止条件：非有限 loss/grad、原生 CUDA 故障、参数/预算/日程签名漂移；
  - 预算来源：新的独立授权；不得使用已停止的 186 次。
- **影响**：不授权则 `execution_equivalence` 保持 `BLOCKED`，最终只能是 `NOT_READY`。

## D11. 任务头与 EnMAP

- **任务头**：`m9_task_head_training_authorized=false` 必须保持；正式训练授权不自动包含任务头。
- **EnMAP**：本机仅一个场景、SRF 为 `derived_gaussian`、不在 7,778 成员发布内；推荐本轮不纳入正式训练，不把派生高斯当实测 SRF。若纳入，需单独给出真实数据、SRF 状态与准入证据。

## D12. TEST/GUARD 与精度

- **TEST/GUARD**：当前发布只有 TRAIN/VAL；若要最终测试和保护区，需另建冻结清单与空间隔离证据；否则正式评价只能报告 VAL，不能声称独立 TEST。
- **精度**：已确认 FP32 预检容差只覆盖旧范围；新 Participation 架构、BF16、DDP、所有微批量、JEPA EMA 均未获正式容差。推荐正式首轮只用 FP32 参考路径；不得自动放宽容差。

