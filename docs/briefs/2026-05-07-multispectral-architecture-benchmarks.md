# Multispectral Architecture Benchmark Brainstorm Brief

Date: 2026-05-07

## Objective

Run a department workflow after the latest training signal suggests the current network family may not be sufficient for the post-disaster multispectral segmentation goal.

The immediate task is to evaluate and implement a controlled architecture benchmark runway for 11-channel segmentation inputs, based on:

- implementation plan: `docs/superpowers/plans/2026-05-07-multispectral-architecture-benchmarks.md`;
- current post-disaster multispectral route: `task-2ba5416843b1`;
- post-E1 diagnostic route: `task-6c4353fb293d`;
- existing training configs under `configs/segmentation_training/`;
- existing model code under `src/segmentation_training/models/`.

## Current Evidence To Absorb

- Earlier U-Net-style training produced weak or unstable foreground evidence under sparse masks.
- The project has moved from the underused `F16/F17` contract toward richer post-disaster optical, SWIR/NIR, derived-index, and auxiliary channel stacks.
- The current 11-channel configs use post-disaster optical and index channels.
- A single U-Net/ResUNet result is no longer enough to decide whether the architecture family or the input contract is the bottleneck.
- The sealed test split must remain sealed; architecture selection must use train/validation evidence only.

## Required Architecture Families

The departments must preserve a comparable benchmark set:

- existing U-Net;
- existing ResUNet;
- Attention U-Net;
- UNet++;
- DeepLabV3+;
- Transformer-UNet.

These families should share the same data contract, input channels, split policy, loss/sampler/performance settings, threshold sweep, prediction summary outputs, and best-checkpoint outputs wherever technically reasonable.

## Department Assignments

### research.lead

Frame what the current training signal can and cannot prove. Challenge premature claims that the current network is impossible, while supporting a broader architecture benchmark when the same 11-channel evidence is used consistently. Explain how the selected families relate to remote-sensing fire and debris-flow segmentation.

### technical.lead

Challenge feasibility. Define the minimum implementation plan for a model registry, additional model adapters, architecture-specific configs, and validation tests. Keep the training CLI and bundle contract stable.

### qa.tester

Challenge testability. Define the acceptance gates for shape compatibility, config validation, sealed-test discipline, comparable architecture settings, and cloud-run readiness. Block any benchmark that changes data, loss, sampler, and architecture at the same time without traceability.

## Required Outputs

- `ResearchBrainstormNote`
- `TechnicalBrainstormNote`
- `QABrainstormNote`
- `BrainstormMemo`

The final `BrainstormMemo` must include independent department positions, challenges, rebuttals, negotiated consensus, recommended implementation sequence, QA gates, deferred options, decision log, and next valid department.

## Non-Goals

- Do not train models in brainstorm.
- Do not use the sealed test split.
- Do not claim a final winning architecture from shape tests or config validation.
- Do not change the cloud data bundle contract while adding architecture adapters.
- Do not let Transformer-UNet become a substitute for input-contract or label-quality diagnosis.
