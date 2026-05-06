# SegmentationTrainingBaseline Status And Next Steps

Date: 2026-05-06

## Purpose

This memo captures the current project state after the first `SegmentationTrainingBaseline v0.1` cloud-training attempt and records the core conclusions from the recent implementation, QA, review, and training discussion.

It intentionally does not document cloud environment setup or operational error details.

## Current Project State

The project now has a source-controlled codebase for a cloud-first segmentation baseline workflow:

- `segmentation_contract` builds and validates the dataset contract.
- `segmentation_training` validates model manifests, builds training bundles, runs E0-E3 experiments, writes metrics, and produces prediction previews.
- The repository is managed in Git and pushed to GitHub without local data, generated artifacts, rasters, checkpoints, or run outputs.
- Local Git ignores `database/`, `.agent-team/`, raster files, checkpoints, and training outputs.
- The training baseline uses the current safe input contract: `F16` and `F17`.

The current dataset contract has:

- `1,252` model-input rows.
- C2 debris-flow rows: `500`.
- C5 fire rows: `752`.
- train rows: `906`.
- validation rows: `234`.
- test rows: `112`.
- test split remains sealed and must not be used for route selection.

The current train/validation raster footprint for `F16/F17` is approximately:

- E1 C5 train + validation: `2.61 GB`.
- E2 C2 train + validation: `636.61 MB`.
- E3 C2/C5 train + validation: `3.23 GB`.

## Implemented Baseline

The active experiment matrix is:

| Experiment | Model | Class Scope | Inputs | Role |
| --- | --- | --- | --- | --- |
| E0 | trivial baseline | C2/C5 | masks only | sanity floor and metric check |
| E1 | U-Net | binary C5 fire | `F16/F17` | first fire-signal probe |
| E2 | U-Net | binary C2 debris flow | `F16/F17` | first debris-flow signal probe |
| E3 | U-Net | multiclass C2/C5 | `F16/F17` | one-model viability and confusion check |

ResUNet adapter code exists, but current E1-E3 configs use `model.family: unet`.

The cloud training runner now includes:

- windowed training with padding for small rasters;
- progress logging by epoch and batch;
- validation metrics after epochs;
- non-finite raster normalization guard;
- all-ignore training-window skip;
- non-finite loss guard;
- metrics and prediction-preview writers.

## Observed E1 Result

The C5 fire binary U-Net baseline did not produce useful validation foreground detection in the current attempt.

Observed pattern:

- validation loss stabilized around `0.083` after roughly ten epochs;
- `mean_iou` stayed at `0.0000`;
- `foreground_recall` stayed at `0.0000`.

This is a meaningful result: the model appears to have found a low-loss background-dominant solution rather than learning a useful C5 foreground signal. A low validation loss alone is not evidence of success under class imbalance.

## Interpretation

The current E1 result suggests that the baseline is not yet demonstrating a learnable foreground signal for C5 using only `F16/F17` and the current loss/sampling setup.

Most likely explanations to investigate next:

1. **Foreground sparsity and imbalance**
   The model may minimize loss by predicting background everywhere.

2. **Window sampling is not foreground-biased enough**
   Random windows may contain too little positive mask area, especially after padding and small-window handling.

3. **Loss is not foreground-sensitive enough**
   Plain cross entropy can reward background dominance unless class weights or foreground-aware losses are used carefully.

4. **Metric encoding or prediction decoding needs audit**
   Because `mean_iou=0` and `foreground_recall=0` are severe, the next step should confirm whether predictions are truly all background or whether metrics/label encoding are suppressing foreground.

5. **`F16/F17` may be insufficient by themselves**
   This remains possible, but should not be concluded until sampling, loss weighting, and prediction previews are inspected.

## Do Not Conclude Yet

Do not conclude that:

- U-Net is unsuitable for the project;
- C5 is not learnable;
- transformer or foundation models are needed immediately;
- `F16/F17` are definitely useless;
- the dataset contract is invalid.

The first failed E1 signal should trigger diagnostic work before model-family escalation.

## Recommended Next Step

The next technical move should be a focused E1 diagnostic pass before running or interpreting E2/E3 as route-selection evidence.

Recommended diagnostic tasks:

1. Write a prediction-summary report for E1:
   - predicted foreground pixel count;
   - label foreground pixel count;
   - predicted/label area ratio;
   - per-sample foreground recall;
   - count of all-background predictions.

2. Inspect validation prediction previews:
   - confirm whether predictions are all background;
   - compare failure cases with label masks and F16/F17 panels;
   - check if foreground is visually plausible in the input channels.

3. Add foreground-aware training controls:
   - foreground-biased window sampler;
   - minimum positive-pixel threshold for some training windows;
   - class weighting derived from train masks;
   - optional Dice or CE+Dice as a controlled experiment, not as hidden default behavior.

4. Re-run E1 as `E1b` with only one changed factor at a time.

5. Only after E1b should E2 and E3 be interpreted for route decisions.

## Suggested Next Department Assignment

Next valid department: technical or implementation, depending on whether the team wants a plan or immediate patch.

Suggested technical assignment:

```text
技术部门开始干活：基于 E1 的 validation_loss 低但 mean_iou/foreground_recall 为 0 的现象，制定 E1 diagnostic and foreground-aware sampling/loss plan；不要直接升级模型家族。
```

Suggested implementation assignment:

```text
开发部门实现：为 SegmentationTrainingBaseline 增加 E1 预测诊断报告、foreground-biased window sampler、class-weight computation 和 E1b 配置；保持 test split 封存。
```

