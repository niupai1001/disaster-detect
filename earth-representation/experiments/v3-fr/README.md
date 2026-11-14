# V3-FR 独立工程实现

新增包位于 `src/encoder_v3/film_resnet/`。运行入口以指定的原始发布 runtime 提供冻结依赖，加载本目录的候选包。候选使用浅层 FiLM ResNet、同槽跨波段融合、冻结 Transformer 和独立 D1 末端。历史支路有独立开关。

在 `earth-representation` 工作目录使用已验收的项目 Python。先创建新的 campaign 目录和 `acceptance_policy.json`，再用绝对路径调用以下命令。

```text
python -B <本目录>/scripts/run_v3_fr.py inspect --release RELEASE --prior PRIOR --output CAMPAIGN --design DESIGN --plan PLAN
python -B <本目录>/scripts/run_v3_fr.py replay --release RELEASE --prior PRIOR --output CAMPAIGN
python -B <本目录>/scripts/run_v3_fr.py executor --release RELEASE --prior PRIOR --output CAMPAIGN
```

设置环境变量 `V3_FR_RUNTIME=RELEASE/runtime`，运行 `python -B -m pytest -q -p no:cacheprovider <本目录>/tests --junitxml=CAMPAIGN/candidate_unit_tests.xml`。以下子命令使用相同的 `--release --prior --output` 参数，依次执行 `candidate`、`resume-gate`、`fit`、`ridge`、`compare`。已有阶段使用独立目录。中断处理及优化器尝试记录保存在每组目录，检查点包含完整恢复状态。正式预算需要单独授权。

`RELEASE` 是历史 30 轮冻结发布，`PRIOR` 是上轮 P0–P5 结果目录，`CAMPAIGN` 是本轮输出。实验注册表给出各组父项。复现需要原始成员和归一化统计。公开报告中的路径副本不能代替本机原始发布。候选代码位于本目录，绑定哈希由运行入口记录。
