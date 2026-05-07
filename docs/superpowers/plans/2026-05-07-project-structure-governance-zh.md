# 项目结构治理实施计划

> **给 agentic workers：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 按任务执行。本计划使用 checkbox（`- [ ]`）追踪。

**目标：** 增加一层安全的项目治理能力，收窄审查范围，并为未来 artifact 迁移做准备，同时不破坏当前训练路径。

**架构：** 在 `src/project_ops/` 下实现一个小型 inventory library，通过 `scripts/project_inventory.py` 暴露 CLI，并在 `docs/ops/` 记录 source/generated/runtime 边界。本阶段只做加法，保持所有 legacy `.agent-team/artifacts/...` 引用有效。

**技术栈：** Python 标准库、`unittest`、Markdown 文档、现有 `.gitignore`。

---

### 任务 1：Inventory 工具

**文件：**
- 创建：`src/project_ops/__init__.py`
- 创建：`src/project_ops/inventory.py`
- 创建：`scripts/project_inventory.py`
- 测试：`tests/test_project_inventory.py`

- [ ] **步骤 1：写失败测试**

测试一个 synthetic workspace 是否能被分类到 source、docs、config、runtime、data 和 generated buckets。

- [ ] **步骤 2：运行测试确认失败**

运行：`PYTHONPATH=src /Users/Lemon/miniconda3/envs/d2l-zh/bin/python tests/test_project_inventory.py`

预期：失败，因为 `project_ops.inventory` 还不存在。

- [ ] **步骤 3：实现 inventory module**

增加 `collect_inventory(root)` 和 `format_markdown(report)`，只使用 Python 标准库。排除 `.git` 遍历，统计文件扩展名，分类 top-level paths，并报告最大 roots。

- [ ] **步骤 4：增加 CLI wrapper**

增加 `scripts/project_inventory.py`，维护者可以运行：

```bash
PYTHONPATH=src python scripts/project_inventory.py --root .
```

- [ ] **步骤 5：确认测试通过**

运行：`PYTHONPATH=src /Users/Lemon/miniconda3/envs/d2l-zh/bin/python tests/test_project_inventory.py`

### 任务 2：Ops 文档

**文件：**
- 创建：`docs/ops/project-structure.md`
- 创建：`docs/ops/review-index.md`
- 修改：`.gitignore`

- [ ] **步骤 1：记录结构策略**

写一份简洁的目录契约，覆盖 source、configs、docs、tests、ignored data、runtime ledger、future workspace outputs 和 legacy-compatible paths。

- [ ] **步骤 2：记录审查入口**

写 review index，列出当前 architecture benchmark、U-Net channel ablation、training runbooks 和 task status 的 canonical files。

- [ ] **步骤 3：忽略未来 workspace outputs**

在 `.gitignore` 中加入 `workspace/`。

### 任务 3：验证并提交

**文件：**
- 任务 1-2 中的全部文件。

- [ ] **步骤 1：运行 focused tests**

运行：`PYTHONPATH=src /Users/Lemon/miniconda3/envs/d2l-zh/bin/python tests/test_project_inventory.py`

- [ ] **步骤 2：运行完整测试**

运行：`PYTHONPATH=src /Users/Lemon/miniconda3/envs/d2l-zh/bin/python -m unittest discover -s tests`

- [ ] **步骤 3：运行 whitespace check**

运行：`git diff --check`

- [ ] **步骤 4：生成 inventory preview**

运行：`PYTHONPATH=src python scripts/project_inventory.py --root . --max-depth 3`

- [ ] **步骤 5：提交并推送**

提交信息：`Add project structure governance layer`

