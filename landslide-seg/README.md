# Landslide-Seg

当前研究 Sentinel-2 多光谱影像的滑坡分割。先读[研究入口的最小接力](docs/RESEARCH_INDEX.md#新-agent-最小接力)，Agent执行遵循[项目规则](AGENTS.md)。当前参照、数据用途和证据均从入口按需定位。

- `src/`、`scripts/`：实现及训练、评价、查询入口。
- `configs/`、`manifests/`：实际配置、冻结成员与统计；训练前按项目规则预检。
- `docs/`：四份固定研究正文；维护归属见[路由表](docs/RESEARCH_INDEX.md#文档与维护路由)。
- `docs/assets/research/`：阶段台账、协议、协作登记、机器索引及图表；不默认全文读取。
- `outputs/`、`reports/`：原运行产物与日志，保留原路径；先查任务登记和实际进程，避免重复启动。
- `archive/`：历史原文、报告快照及原路径/SHA256，见[归档入口](archive/README.md)。
- `tests/`：实现验证。Python专用环境见研究入口。

早期TerraMind/TiM、AnySat、语义与粗细分割系统属于历史工程，不是当前参照。旧README全文与旧命令已保存在[归档索引](archive/experiment_history/catalog.json)，用 `scripts/query_research.py --history README.md --limit 5` 查找；历史命令不是当前执行授权。

新实验从实际配置开始：`python -X utf8 scripts/check_research_protocol.py --config <实际配置> [--set key=value]`。预检只读，不训练；完整输入、初始化、预算和验收要求见项目规则。
