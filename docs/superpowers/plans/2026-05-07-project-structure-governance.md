# Project Structure Governance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a safe governance layer that narrows review scope and prepares future artifact migration without breaking current training paths.

**Architecture:** Implement a small inventory library under `src/project_ops/`, expose it through `scripts/project_inventory.py`, and document the source/generated/runtime boundaries under `docs/ops/`. This phase is additive and keeps all legacy `.agent-team/artifacts/...` references valid.

**Tech Stack:** Python standard library, `unittest`, Markdown docs, existing `.gitignore`.

---

### Task 1: Inventory Tool

**Files:**
- Create: `src/project_ops/__init__.py`
- Create: `src/project_ops/inventory.py`
- Create: `scripts/project_inventory.py`
- Test: `tests/test_project_inventory.py`

- [ ] **Step 1: Write failing tests**

Test that a synthetic workspace is classified into source, docs, config, runtime, data, and generated buckets.

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src /Users/Lemon/miniconda3/envs/d2l-zh/bin/python tests/test_project_inventory.py`

Expected: failure because `project_ops.inventory` does not exist.

- [ ] **Step 3: Implement the inventory module**

Add `collect_inventory(root)` and `format_markdown(report)` using only the Python standard library. Exclude `.git` traversal, count file extensions, classify top-level paths, and report largest roots.

- [ ] **Step 4: Add CLI wrapper**

Add `scripts/project_inventory.py` so maintainers can run:

```bash
PYTHONPATH=src python scripts/project_inventory.py --root .
```

- [ ] **Step 5: Verify tests pass**

Run: `PYTHONPATH=src /Users/Lemon/miniconda3/envs/d2l-zh/bin/python tests/test_project_inventory.py`

### Task 2: Ops Documentation

**Files:**
- Create: `docs/ops/project-structure.md`
- Create: `docs/ops/review-index.md`
- Modify: `.gitignore`

- [ ] **Step 1: Document structure policy**

Write a concise directory contract for source, configs, docs, tests, ignored data, runtime ledger, future workspace outputs, and legacy-compatible paths.

- [ ] **Step 2: Document review entrypoints**

Write a review index listing the canonical files for current architecture benchmark, U-Net channel ablation, training runbooks, and task status.

- [ ] **Step 3: Ignore future workspace outputs**

Add `workspace/` and its intended children to `.gitignore`.

### Task 3: Verification And Commit

**Files:**
- All files from Tasks 1-2.

- [ ] **Step 1: Run focused tests**

Run: `PYTHONPATH=src /Users/Lemon/miniconda3/envs/d2l-zh/bin/python tests/test_project_inventory.py`

- [ ] **Step 2: Run full tests**

Run: `PYTHONPATH=src /Users/Lemon/miniconda3/envs/d2l-zh/bin/python -m unittest discover -s tests`

- [ ] **Step 3: Run whitespace check**

Run: `git diff --check`

- [ ] **Step 4: Generate inventory preview**

Run: `PYTHONPATH=src python scripts/project_inventory.py --root . --max-depth 3`

- [ ] **Step 5: Commit and push**

Commit message: `Add project structure governance layer`

