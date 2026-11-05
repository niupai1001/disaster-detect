# Encoder V3 — isolated implementation

This directory is independent of the read-only synced sources and the historical N: project. Importing it never starts training.

## Run checks

```powershell
& 'N:/Users/yzh/Disater/landslide-seg/.venv-use-independent/Scripts/python.exe' -m encoder_v3.verify --output encoder_v3/reports/engineering_tests.json
& 'N:/Users/yzh/Disater/landslide-seg/.venv-use-independent/Scripts/python.exe' -m encoder_v3.gates
```

The gate command is expected to reject the current configuration. Its unresolved values must not be silently filled.

## Implemented

- Shared continuous SRF encoder, explicit-candidate Hypernetwork, native masked convolution.
- Center-assigned physical patches, serial reference and length-bucketed batched local Transformer, full global Transformer.
- Represented native cell unions distinct from convolution receptive-field semantics; exact B-area intersections.
- Shared scalar reflectance MLP; unsupported outputs marked NaN with a separate support mask.
- Spatial hidden-value isolation; JEPA predictor and EMA helper; C-AdamW candidate; checked plateau and atomic RNG-aware checkpoints.
- Halo-correct tiled convolution and same-shape grouped dynamic convolution, with output/gradient comparisons.
- Geometry-only CPU layout construction with batched GPU indexing; local length buckets never quantize GSD or truncate positions.
- Sparse STRtree Decoder lookup with exact native-cell intersection areas, plus dense reference readout.
- Hash-bound TRAIN/VAL native-cache reader; deterministic B1 plans, explicit A masks, fail-closed B2 pairing, hierarchical losses.
- Exact full-target streaming loss: Decoder graphs released per pixel chunk, accumulated Z/SRF gradients backpropagated through Encoder once. No target sampling or chunk-wise mean bias.
- Manifest-bound spawn-worker prefetch, custom observation pinning and commit-only replay cursor; prefetched batches never advance logical progress.
- Durable pilot-budget reservations: hard total/per-candidate caps, exclusive writes and conservative crash accounting. The real ledger records all completed/reserved updates, including fitting; reports/execution_status.json summarizes the latest snapshot.
- Effective-batch-8 A/B joint reference with exact task/source/observation reduction and one final gradient clipping; no optimizer step.
- Observation microbatches 2/4/8 group compatible independent bands and local sequences, then run full global attention with an observation batch dimension and padding mask. A/B are encoded independently; no token truncation.
- Optional native-cell STRtree readout vectorizes exact overlap contributions, replacing per-Token GPU loops. The reference token-index path remains the model default; reviewed single-GPU FP32 pilot execution uses the cell path on its confirmed scope.
- Per-Token GSD, source SRF condition, true support extents and audit-only band provenance. SRF conditions reuse the native convolution graph. Local convolution/Merge support (`receptive_cells`) is distinct from the union of all visible input regions shared by final full-attention tokens (`full_input_cells`); overlapping regions describe a union, not an area sum.
- Review-bound real GPU update/resume checker, full prediction/gradient/state comparisons, warmup, complete bias/LayerNorm decay exclusions, durable budget accounting and own-process deadline. Checkpoints save RNG only for model-used CUDA devices.

## Current evidence

- `reports/engineering_tests.json`: current covered engineering test count and source hashes, including cache tampering, RSR provenance, tiling, batching, geometry holes, hidden values, exact chunked gradients, prefetch replay, budget crash accounting, EMA and checkpoint RNG.
- `reports/native_cache_audit.json`: 7,778 existing native-cache members reread, with hashes, grids, finite valid reflectance and true SRF checks. This is not full-machine V3 admission. The initial over-restrictive RSR-label failure is retained in `native_cache_audit_before_rsr_fix.json`.
- `reports/native_lineage_audit.json`: all 7,778 existing TRAIN/VAL caches exactly reproduced from local parents/native TIFs, including valid masks; 1,721 original-QA recalculations and 6,057 parent-mask-only traces. Source files have a separate hashed identity manifest. This reproduces historical processing, not new QA-policy approval. `lineage_qa_header_review.json` checks 4,844 QA headers; the earlier categorical-warp warning remains unexplained, not erased.
- `reports/inventory_release_reconciliation.json`: 13,702 canonical records reconciled with the expanded 7,778-member release, including 810 additional Landsat anchors. No protected-to-TRAIN promotion or missing anchor is found in this metadata join; it is not all-machine admission or B2 certification.
- `reports/real_s2_probe.json`, `real_landsat_indexed_layout_probe.json`, `real_full_s2_indexed_probe.json`: real TRAIN-only forward/backward diagnostics, no optimizer updates.
- `reports/real_decoder_index_probe.json`: matched real S2 readout with 38,025 tokens / 512 diagnostic target pixels; approximately 6.94x Decoder-only speedup. Excludes Encoder, I/O, optimizer and saving; not end-to-end acceptance.
- `reports/real_precision_probe.json`: FP32/BF16 small-case comparison and full 38,025-token S2 BF16 backward. Tolerances and optimizer/recovery comparisons remain unapproved. A previous float32 cosine-accumulation artifact is retained in `real_precision_probe_before_metric_fix.json`; corrected comparisons accumulate in float64.
- `reports/real_full_target_streaming.json`: all 32,768 valid native pixels of two hidden bands in one real TRAIN observation used for chunked backward, no optimizer step. Historical source TRAIN statistics are hash-checked but are not adopted as V3 statistics.
- `reports/real_prefetch_probe.json`: identical real effective batches of 8 tested with 0/2 workers and pinned memory; startup separated from subsequent waits. Four CPU-only batches do not establish long-run GPU-overlap performance.
- `reports/dual_gpu_communication_restricted.json`: loopback network permission failure. `dual_gpu_communication_unrestricted.json`: CPU all-reduce succeeds after authorized diagnostic escalation, but GPU all-reduce terminates a worker with Windows exit code 3221225477. This is not DDP acceptance or proof of a hardware limitation.
- `reports/independent_gpu_batch8_unrestricted.json`: independent per-device workers both time out during CUDA initialization. No workload timings or speedup obtained; failure is retained and is not evidence that independent execution cannot work.
- `reports/real_joint_batch8.json`: eight real TRAIN observations, sixteen A/B1 views, all supported targets, shared hierarchical gradient accumulation and clipping, zero optimizer updates. B2 remains absent. This is the serial observation-microbatch-1 reference.
- `reports/inventory_scope_review.json`: additional known-root metadata review. EnMAP catalogue availability is separate from the empty local EnMAP manifest; no EnMAP image is admitted. Other mounted data roots and source associations remain unaudited.
- `reports/pilot_study_review.md`: readable proposal for pilot members, masks, validation and execution. Its underlying candidate file is not an authorization to update parameters.
- `reports/real_microbatch_p120.json`: matched full-target observation microbatches 1/2/4/8. One timed repeat after warmup gives 17.37/18.04/17.74/17.31 seconds; no convincing gain. Padding rises to 72.3% for microbatch 8, so larger batches are not selected by this evidence.
- `reports/real_task_python_profile.json`: one complete real Landsat B1 task; Decoder geometry accounts for about 7.13 seconds of the 7.88-second Python wall profile, including 18,760 per-Token overlap calls. Host timings include possible GPU waits and are not kernel durations. CUDA profiler interruption and CPU event-summary timeout remain in separate failure reports.
- `reports/real_cell_index_comparison.json`: P120 matched effective batch 8, all 377,001 supervision pixels and 11,042 tokens. Two timed repeats after warmup give median 19.08 seconds for token indexing versus 6.00 seconds for cell indexing, about 3.18x forward/backward speedup; peak allocated memory remains approximately 1.15 GB. Loss difference 2.49e-8 and relative gradient difference 1.09e-7 are measurements, not approved tolerances. Excludes loading/planning/transfer, optimizer and whole-experiment cost.
- `reports/real_cell_index_p60_comparison.json`: P60 matched full-target comparison, 388,761 supervision pixels / 41,063 tokens, median 45.40 versus 11.02 seconds (4.12x) and relative gradient difference 1.14e-7. Different P values use their own matched masks; these are performance diagnostics, not separate training groups. `reports/gpu_efficiency_review.md` collects the scope and remaining checks; device sampling covers only a partial old-path interval.
- `reports/pilot_runs/update_check_a624c7d0c860/report.json`: eight user-confirmed real GPU parameter updates (four per optimizer) used the first two reviewed warmup inputs at P60/P120, including matched old/new readout and interrupted/resumed next update. All eight reservations completed in about 203 seconds, peak allocation 2.35 GB. Predictions/parameters differ at small floating-point levels; RNG, LR and exposure signatures match. No peak-LR, EMA or whole-training equivalence claim. `gpu_update_checkpoint_review.json` independently inspects saved RNG/state structure.

## Not yet accepted for formal training

- The user confirmed Hypernetwork 34→64→320/GELU/log-GSD for preflight in `reports/hypernetwork_preflight_review.json`; this is not formal-training authorization. JEPA query projection remains an engineering candidate.
- The user confirmed the single-rectangle preflight mask rule in `reports/spatial_mask_preflight_review.json`; this confirmation is scoped to preflight.
- Members and fixed panels are now confirmed in `reports/pilot_members_panels_review.json`; the bounded eight-update check is confirmed in `reports/gpu_update_execution_review.json`. Formal-training settings remain unresolved. The FP32 numerical limits are confirmed in reports/fp32_equivalence_tolerance_review.json and passed on the first two measured warmup inputs. This does not accept peak-LR, BF16 or multi-GPU execution.
- North-up metric grids only. Rotated/geographic-grid admission is blocked rather than resampled.
- A neighbor lookup checks represented native-cell union contact with the rectangular masked-region boundary, including corners. Its performance and real-grid admission still require testing.
- Full-machine source association/admission, pair registration/change-risk review, complete bounded optimizer comparison, broader execution-equivalence acceptance, independent dual-GPU route profiling and research-ledger integration remain outstanding.
- Tiny CPU fixtures verify the exact next update after checkpoint restore for AdamW and C-AdamW, including moments, RNG, LR and EMA. This does not establish real-GPU recovery equivalence or consume real pilot budget.
- Float64 CPU fixtures compare microbatches 2/4/8 and both sparse-geometry paths against the serial reference through one AdamW update. FP32 near-zero gradients can produce micro-scale AdamW update differences; reviewed real FP32 update tolerances apply only within their recorded scope. Fixture coverage does not establish real-GPU update acceptance.
- Existing evidence is partial engineering coverage, not optimizer-selection evidence or formal GPU-performance acceptance. Rotated grids and unresolved configuration fields remain blocked.
- Eight real checks plus matched fitting updates consume the same original 400-update budget. The resumable fitting driver is running from existing checkpoints toward Warmup update 100 per optimizer. No formal pretraining or task-head training has started.

Budgets remain 400 optimizer-pilot updates in total and 10,000 logical updates per formal route. Read reports/pilot_runs/budget.json for current reservations/completions; do not infer remaining budget from a static document. The task-head budget is unset. Formal training remains fail-closed. The bounded pilot driver cannot launch formal routes.

## GPU restart and resumed pilot

The user restarted the GPU after native CUDA initialization crashes. The isolated tensor probe recovered; the runtime currently exposes one RTX 5090. Both optimizers completed the first two fitting updates and fixed validation, consuming 12 total pilot updates including the earlier eight diagnostics. Offline checkpoint comparisons against earlier measured steps passed the reviewed parameter/optimizer bounds without additional updates. See reports/cuda_runtime_recovery.json and reports/gpu_restart_checkpoint_comparison.json. The same checkpoints continue toward update 100, with budget retained for peak-LR/update/resume diagnostics. Whole-device samples are not process-specific utilization.

The cross-source catalog retains 349 metadata candidate pairs, with zero admitted. Candidate QA support measurement preserves native masks and cells and cannot establish registration, processing-version consistency, temporal acceptance or unchanged surfaces.
