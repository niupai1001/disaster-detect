# Post-E1 Baseline Diagnostic Brainstorm Brief

Date: 2026-05-06

## Objective

Run a focused adversarial brainstorm after the first `SegmentationTrainingBaseline v0.1` cloud-training signal.

The central question is:

> E1 C5 fire U-Net reached low validation loss but produced `mean_iou=0.0000` and `foreground_recall=0.0000`. Should the next route diagnose and repair the U-Net baseline first, introduce ViT/transformer segmentation, or do both under a controlled gate?

The project owner's current judgment must be treated as an explicit workshop input:

> A ViT-style model probably needs to be introduced for later training, but it is not plausible to conclude that U-Net has absolutely no effect from the current E1 result alone.

The workshop must not collapse into either extreme:

- do not declare U-Net useless from one failed E1 attempt;
- do not ignore the need to plan a transformer/ViT route if the corrected baseline still cannot learn foreground.

## Current Status To Absorb

Primary status memo:

- `docs/implementation/2026-05-06-training-baseline-status-and-next-steps.md`
- Chinese archive: `docs/implementation/2026-05-06-training-baseline-status-and-next-steps-zh.md`

Parent project-direction task:

- `task-afc7f2c25f8f`

Relevant artifacts from the parent task:

- `.agent-team/artifacts/task-afc7f2c25f8f/brainstorm/brainstorm-BrainstormMemo.md`
- `.agent-team/artifacts/task-afc7f2c25f8f/technical/technical-technical-lead-TechnicalPlan.md`
- `.agent-team/artifacts/task-afc7f2c25f8f/implementation/implementation-implementation-dev-ImplementationHandoff.md`
- `.agent-team/artifacts/task-afc7f2c25f8f/qa/qa-qa-tester-TestReport.md`
- `.agent-team/artifacts/task-afc7f2c25f8f/review/review-review-lead-ReviewNote.md`

Current training facts from the status memo:

| Item | Value |
| --- | ---: |
| Model input rows | 1,252 |
| C2 debris-flow rows | 500 |
| C5 fire rows | 752 |
| Train rows | 906 |
| Validation rows | 234 |
| Test rows | 112 |
| Safe current inputs | `F16/F17` |
| E1 model | binary C5 U-Net |
| E1 validation loss | about `0.083` after roughly ten epochs |
| E1 validation `mean_iou` | `0.0000` |
| E1 validation foreground recall | `0.0000` |

The sealed test split must remain sealed and must not be used for route selection.

## Brainstorm Questions

1. What are the most likely explanations for low validation loss with zero foreground IoU/recall?
2. Which diagnostics must be run before interpreting E1 as model-family evidence?
3. What would prove that U-Net is learning something weakly but failing thresholding, sampling, loss weighting, or decoding?
4. What would prove that the current `F16/F17` input contract is insufficient for C5 fire segmentation?
5. Which foreground-aware sampler and loss changes should be tested first, and how should they be isolated one factor at a time?
6. Where should a ViT/transformer model enter the roadmap: immediate parallel probe, gated E1c/E4 experiment, or later after richer channels?
7. Which transformer family is the best first candidate for this project: vanilla ViT segmentation, SegFormer, UPerNet, Mask2Former, SatMAE/Prithvi-style transfer, or another route?
8. What metric, prediction-summary, and visual-preview gates decide whether to keep repairing U-Net, add ResUNet/DeepLabV3+, or escalate to transformer/foundation-model methods?

## Required Department Positions

### research.lead

Frame the result scientifically. Challenge premature claims that U-Net is invalid, while also preserving the research case for ViT/transformer segmentation. Explain what published remote-sensing segmentation practice suggests about small data, class imbalance, foreground sparsity, multispectral inputs, and transformer escalation.

### technical.lead

Challenge implementation feasibility. Define the minimum diagnostic and E1b/E1c experiment plan that isolates sampler, loss, metric decoding, and model-family variables. Compare the practical cost of adding a ViT/SegFormer-style adapter against repairing the current U-Net training loop and data pipeline.

### qa.tester

Challenge testability. Define failure labels, acceptance gates, prediction-summary reports, foreground-area checks, threshold/argmax audits, visual QA requirements, and stop/go criteria. Block any route that changes multiple factors without preserving interpretability.

## Required Outputs

- `ResearchBrainstormNote`
- `TechnicalBrainstormNote`
- `QABrainstormNote`
- `BrainstormMemo`

The final `BrainstormMemo` must include:

- independent department positions;
- challenges, rebuttals, and negotiated consensus;
- a diagnosis-first route;
- a ViT/transformer introduction gate;
- a one-factor-at-a-time experiment matrix;
- rejected or deferred options;
- decision log;
- next valid department and suggested assignment.

## Non-Goals

- Do not train another model in this stage.
- Do not use the sealed test split.
- Do not choose a final production architecture.
- Do not treat the E1 result as proof that C5 is unlearnable.
- Do not treat ViT as a magic fix for sampling, loss, metric, label, or input-channel defects.
