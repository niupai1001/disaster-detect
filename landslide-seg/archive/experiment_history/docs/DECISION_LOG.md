# 决策记录

## 2026-10-21 · 固定文档维护与官方比较门槛

- 用户要求维护少量固定报告。当前入口为 `RESEARCH_INDEX.md`、`DATA_OFFICIAL.md`、`DATA_EVENT_PROTOCOL.md`、`MODEL_EVOLUTION.md`；以后在对应文件内更新，历史阶段报告只作证据。
- 默认在项目跨事件协议下开发；模型明显进步、候选冻结后，才新增官方split比较。不自动执行旧官方协议续跑链。已有官方结果和原训练数据使用历史保留，不恢复历史盲测身份。
- 实验名称按架构/观测/机制展示，协议/折/seed/训练处理单列，机器别名映射保留原路径与配置哈希。仅修改文档和分析索引，没有训练、推理、移动或重写checkpoint。


## 2026-10-20 · 启用官方单期协议（选项 A）

- **决定**：按 `docs/next_execution_plan_official_single_2026-10-20.md` §4 的选项 A 执行，
  在 `BENCH_OFFICIAL` 命名空间下使用官方 S12LS-LD split 训练与评测；**原事件 OOD 协议及其全部产物保留不变**。
- **依据**：官方 TRAIN 可用块中落在 newzealand 292、hiroshima+thrissur 322、italy 1135；
  用户确认接受该代价以换取"官方条件下能打多少分"的答案。
- **后果（不可逆，必须写进后续所有报告）**：
  - `newzealand` 的 FINAL 盲测身份**终止**；
  - `hiroshima` / `thrissur` 不再是干净的 DEV；
  - 事件 OOD 的历史数值只在原协议内部有效，不得再与官方条件数值并表。
- **锁定**：官方 commit `2eb025bd76bef449d2f7cf4aaea861bc4cfeed51`；
  协议 sha256 `16bbd71d1b8dfb9d…`（见 `manifests/official_single/protocol_lock.json`）。
