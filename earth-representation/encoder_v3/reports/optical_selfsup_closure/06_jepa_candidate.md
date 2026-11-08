# 06 同观测 JEPA 候选评估（E1 候选，不阻塞 E0）

**结论**：现有 `jepa.py` 的几何与预测器可以复用，但"同观测 Teacher/Student"不是最小改动就能接上的：
至少需要新增视图协议、目标区域选择、潜在损失与归一化、非坍塌诊断四部分。**本轮不实现、不启用、不参与
E0 的训练入口或指标**；建议作为下一轮 E1 的独立实验，并以 E0 的 A+B1 结果为比较基准。

## 1. 现有实现能直接复用什么

| 组件 | 现状 | 同观测能否复用 |
|---|---|---|
| EMA Teacher | `make_teacher` / `update_teacher`（`training.py`）做参数指数滑动平均并冻结梯度 | 可以，1.24M 参数，复制成本可忽略 |
| Teacher 查询几何 | `teacher_support_queries`：计算 **Student 可见足迹** 与 Teacher Token 区域的交叠，并输出相对位置/支持/SRF/GSD 条件 | 只能复用"交叠与条件"的写法，**选择规则要反过来**（见第 2 节） |
| Predictor | `JEPAPredictor`：条件只用元数据（相对位置、log 支持、SRF、log GSD），对 Student Token 做单头注意力 | 可以直接复用 |
| 损失外壳 | `cross_source_latent_loss` 要求"两个不同观测 + 配对准入 + 已审阅 loss_fn" | **不能复用**：它拒绝同一观测（`source.identifier==target.identifier` 直接报错），且要求跨源配对 |
| 防泄漏 | Student 分支只编码可见波段/可见位置；目标只来自 Teacher | 可以沿用同一原则 |

因此"最小改造"的真实边界是：**几何与预测器复用，视图协议与损失必须新写**。

## 2. 同观测候选方案（本轮设计，未实现）

```text
同一次原生观测 O（同一 SRF、同一 GSD、同一 Merge 规则、同一原点）
        │
        ├── Student 视图：把矩形 R 从可见支持中移除（与 A 任务同一套遮盖规则）
        │        └── Encoder(student 可见位置) → Z_s
        │
        └── EMA Teacher 视图：完整可见观测（固定、stop-grad）
                 └── Encoder(完整观测) → Z_t
                          │
  查询集合 Q = Teacher Token 中"代表区域完全落在 R 内"的 Token
                          │
  Predictor(条件=相对位置/支持/SRF/GSD, 键值=Z_s) → 预测 Z_t[Q]
                          │
  loss = 归一化均方误差（stop-grad 目标）+ 非坍塌约束
```

要点：

- **不需要跨卫星配对**：Student 与 Teacher 是同一次观测的两个视图，SRF/GSD/Merge 天然一致。
- **目标区域明确**：Q 只取完全落在隐藏矩形内的 Teacher Token；Student 的可见足迹与 Q 不相交，
  且 Student 的输入里不存在目标位置的反射率（与 A 任务的隔离证明相同）。
- **停止梯度**：Teacher 参数 `requires_grad=False`、前向在 `no_grad` 下；目标张量 detach。
- **Predictor 与 Decoder 职责分离**：Predictor 只做"潜在空间的条件预测"，不读反射率、不共享解码器参数；
  Decoder 仍只用于 A/B1 的反射率重建。
- **归一化**：候选形式为逐 Token 的 L2 归一化均方误差（或按 Teacher 目标维度标准差的 NMSE），
  二选一必须在 E1 前冻结，不能训练中切换。
- **非坍塌检查**（每次验证都报告，不参与反传）：
  1. Teacher 目标的逐维标准差与均值；
  2. 目标协方差的特征值谱（有效秩）；
  3. "常数预测"基线损失与当前损失之比（若接近 1 则说明表征退化）；
  4. Student 与 Teacher 表征的余弦相似度分布。

## 3. 工程成本与风险

- 代码量估计：新增约 100–150 行（视图构造、Q 选择、损失、诊断）+ 独立配置与门禁 + 合成稳定性测试。
  这不是"最小改造"，而是一个独立实验入口。
- 单次更新成本：每个观测多一次 Teacher 前向（与 Student 前向同量级），本机实测 Encoder 前向约
  0.3–3 秒/观测，主要成本仍是反射率读出（A/B1），因此 E1 的增量大约在 +10%–20%；但若 E1 与 A/B1
  共享 Student 前向，可进一步降到 +5%–10%（需要把遮盖视图与 A 任务合并——这会改变 A 的任务定义，
  必须单独评审）。
- 风险：
  - 坍塌：潜在目标若不做方差/协方差约束，容易退化为常量（因此第 2 节的诊断是硬要求）；
  - 冗余：A 任务已经在反射率空间监督同一隐藏区域，E1 的增量价值需要靠消融证明，不能假设；
  - EMA 与优化器状态：Teacher 不进优化器，恢复时必须与 Student 同步重建（现有 `make_teacher` 已处理）；
  - 与 E0 的指标隔离：E1 不能改变 `S_intra`，只能作为独立报告项。

## 4. 建议

1. **本轮不启用**：`config_optical_selfsup_v1.json` 中 `JEPA_training_required=false`，
   `required_before_jepa` 五项仍全部为 `null`，`selfsup_train`/`selfsup_preflight` 不引用 JEPA 模块。
2. **下一轮作为 E1 独立实验**：先写同观测视图与损失，做合成稳定性测试（固定随机种子下的坍塌/梯度有限性/恢复一致性），
   再在固定 VAL 面板上与 E0 的 A+B1 结果比较；不要求跨源配对，也不改变 E0 的门禁。
3. **是否值得**：值得进入下一轮"候选消融"，但**现在不能声称会带来表征质量提升**。理由是 E1 的主要变量
   （潜在目标归一化、Q 的选择、非坍塌约束）都还没有实测；在没有这些证据前，E0 的 A+B1 是唯一可比较的基线。

## 5. 与任务书要求逐条对照

| 要求 | 本轮状态 |
|---|---|
| 不要求跨卫星配对 | 设计满足（同观测两视图） |
| Teacher 停止梯度并采用 EMA | 复用现有 `make_teacher`/`update_teacher`，设计满足 |
| Student 不能访问被预测的目标信息 | Student 视图移除隐藏区域；与 A 任务同一隔离证明（`evidence/architecture_evidence.json`） |
| Predictor 与 Decoder 职责分离 | 设计满足（Predictor 只吃元数据条件） |
| 检查表征坍塌 | 已列出四项诊断，**尚未实现** |
| E1 不影响 E0 的训练入口 | 满足：E0 配置与入口不含 JEPA；`JEPA_training_required=false` |
| 不得临时拼接无稳定性验证的 JEPA 损失 | 遵守：本轮没有实现任何 JEPA 损失 |
