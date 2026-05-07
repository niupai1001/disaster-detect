# 项目结构治理设计

日期：2026-05-07

## 目标

在不破坏当前云端训练命令的前提下，把可维护项目文件与本地数据、bundle、run outputs、checkpoints、agent runtime artifacts 分离，降低审查成本。

## 问题

仓库本身不大，但共享工作区很嘈杂：

- `database/` 包含大量本地 raster 数据。
- `.agent-team/artifacts/` 同时混合了部门 Markdown、training bundles、cloud run outputs、checkpoints、diagnostics 和下载回来的 smoke 证据。
- 审查者查看工作区时，会先看到大量 JSON、CSV、PNG、PT 和 TIF 文件，而不是代码和决策记录。

当前训练 runbooks 和 task ledger 都引用 `.agent-team/artifacts/...`，所以立刻移动这些路径风险较高。

## 设计

先增加治理层，再移动数据：

- 保持 `src/`、`configs/`、`tests/` 和 `docs/` 作为可审查 source surface。
- 引入 `workspace/` 作为未来被 Git 忽略的本地数据、bundle、runs、archives 和 scratch outputs 根目录。
- 保持 `.agent-team/` 作为 runtime ledger。后续主要用于 task records、messages 和部门 Markdown artifacts。
- 当前 cloud benchmark 和 ablation 结果返回前，继续保留已有 `.agent-team/artifacts/...` 路径。
- 增加 inventory 工具，对文件按用途分类并输出 Markdown 供审查。
- 增加 ops review index，让审查从少量 canonical files 开始，而不是遍历 run directories。

## 非目标

- 本阶段不移动现有 `.agent-team/artifacts` 内容。
- 本阶段不改变当前 runbooks 中的有效云端命令。
- 本阶段不重写 `agent_team` 存储语义。
- 本阶段不删除本地 datasets 或 training outputs。

## 验收标准

- 审查者可以运行一个命令查看项目文件类别和最大的本地状态目录。
- `workspace/` 被 Git 忽略。
- 文档说明哪些目录是 source、哪些是 generated、哪些是 legacy-compatible runtime paths。
- 现有测试继续通过。
- 当前云端训练路径不被移除。

