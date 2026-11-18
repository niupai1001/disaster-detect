# 本轮交付：GeoToken–DualViT R3

唯一核心接口：每个地理 Patch 最终输出一个128维联合Token；Self-Attention不减数量，分支末端的Flatten+Linear明确承担压缩。A采用地理Patch级MAE，B1保留所有地理Token、只删除隐藏band内容。

先读00_MODEL_PLAN.md，再读01_TRAINING_AND_HANDOFF.md；02_CONFIG.yaml是新实现规格，不是旧runner可直接执行配置。真实数据身份字段必须本地绑定。R1/R2仅归档，不能与R3冲突条款混用。

- 00_MODEL_PLAN.md：模型、形状、原生布局、A/B1与压缩限制。
- 01_TRAINING_AND_HANDOFF.md：7:3、batch16、DDP、阶段预算与验收。
- 02_CONFIG.yaml：机器可读实施规格。
- 03_SOURCES.md：官方依据与用户报告适用范围。
- 04_COMPUTE_BUDGET.json：理论前向MAC，非GPU测速。
- 05_REFERENCE_TEST_RESULTS.json：实际执行的17项局部CPU张量检查。
- reference/model.py：小型张量参考实现，不含生产原生数据/CRS/加载器/DDP/训练入口。
- reference/test_reference.py：复现局部测试。
- tools/compute_budget.py：重算声明假设下的预算。

复现：
```bash
python reference/test_reference.py
python tools/compute_budget.py
```
测试环境为PyTorch 2.10.0+cpu。无新训练、无真实影像、无GPU速度或质量结论。数据、旧结果、仓库均未被修改。
