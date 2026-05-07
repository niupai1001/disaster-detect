---
title: 上下文压缩与下一步行动
date: 2026-05-07
tags:
  - handoff
  - next-actions
  - segmentation
  - cloud
  - obsidian
status: active
task: task-6bd957f6e1ed
aliases:
  - 当前交接
---

# 上下文压缩与下一步行动

## 当前分支

```text
codex/rtx4080-training-performance
```

最新已推送提交：

```text
3512bf3 Add project structure governance layer
```

## 不可破坏的项目规则

- 部门 artifacts 不使用外部 provider。
- Codex 自己派发并协调 subagents。
- sealed test split 继续封存；架构与通道选择只使用 validation。
- 不把 smoke metrics 当作模型质量证据。
- 当前 cloud runbooks 仍依赖 legacy `.agent-team/artifacts/...` 时，不移动这些路径。

## 已完成工作

- 已增加 ResUNet、Attention U-Net、UNet++、DeepLabV3+、Transformer-UNet 的架构 benchmark runway。
- 架构 cloud smoke 已返回，并通过 runtime preflight。
- 已增加 U-Net-only channel ablation configs：RGB、false-color、7-channel raw optical/SWIR、indices-only、当前 11-channel 对照组。
- 已创建 Obsidian-compatible 模型/数据假设记录。
- 已增加项目结构治理层：`docs/ops/`、`scripts/project_inventory.py`、`workspace/` ignore policy。
- 治理层改动后，本地完整测试通过：`Ran 111 tests ... OK`。

## 立刻要做的云端操作

在云端：

```bash
cd /root/autodl-tmp/New_project
git pull --ff-only origin codex/rtx4080-training-performance
```

### 选项 A：架构 Validation Benchmark

运行受控 validation-only architecture benchmark，输出到：

```text
.agent-team/artifacts/task-6bd957f6e1ed/cloud_benchmark
```

Configs 参考 [[docs/implementation/2026-05-07-segmentation-v4-cloud-smoke-runbook-zh|架构 Smoke Runbook]] 和 [[docs/ops/review-index-zh|审查入口]]。

### 选项 B：U-Net 通道消融 Smoke

先跑 smoke：

```bash
export PYTHONPATH=src
export BUNDLE_DIR=.agent-team/artifacts/task-2ba5416843b1/implementation/training_bundle_v0_3_indices
export RUN_ROOT=.agent-team/artifacts/task-6bd957f6e1ed/cloud_unet_channel_ablation_smoke

for config in configs/segmentation_training/unet_channel_ablation/*_smoke.yaml; do
  name="$(basename "$config" .yaml)"
  python -m segmentation_training train \
    --config "$config" \
    --bundle-dir "$BUNDLE_DIR" \
    --run-dir "$RUN_ROOT/$name" \
    --cloud-confirm
done

python -m segmentation_training summarize-runs \
  --runs-dir "$RUN_ROOT" \
  --output "$RUN_ROOT/summary.md"
```

## 需要返回的证据

每个 cloud run family 返回或粘贴：

- `summary.md`；
- 每个 run 的 `run_manifest.json`；
- `metrics.json`；
- `per_class_metrics.csv`；
- validation contact sheets；
- training curves；
- 任何 OOM 或失败日志。

## 结果返回后

1. QA/review 接受或拒绝返回证据。
2. 如果 architecture benchmark 成功，只比较 validation-only 证据，不使用 sealed test。
3. 如果 channel ablation 显示更简单的 stack 优于 11-channel，先查 channel noise、normalization 和 label/sample quality，再升级架构复杂度。
4. 如果所有 U-Net channel groups 都类似失败，优先查 labels、alignment、foreground sparsity、sampler coverage、loss weighting 和 thresholding。
5. 启动第二阶段结构清理：future run outputs 迁移到 `workspace/runs/`，future bundles 迁移到 `workspace/bundles/`，`.agent-team/artifacts` 只保留 Markdown summaries。

## 常用本地命令

Inventory：

```bash
PYTHONPATH=src python scripts/project_inventory.py --root . --max-depth 3
```

完整测试：

```bash
PYTHONPATH=src /Users/Lemon/miniconda3/envs/d2l-zh/bin/python -m unittest discover -s tests
```

