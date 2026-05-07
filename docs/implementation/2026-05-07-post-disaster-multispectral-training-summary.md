# Post-Disaster Multispectral Segmentation Training Summary and Next Goals

Date: 2026-05-07

## Current Direction

The project direction is now post-disaster recognition and pixel-level segmentation. Pre-disaster prediction is not a current-stage objective.

The central lesson from this stage is that repeated zero foreground recall should not be interpreted only as U-Net failure. A larger issue was underuse of the available data pipeline. Earlier training relied mainly on `F16/F17`, while both the literature and local data suggest that post-disaster fire and debris-flow segmentation need optical, NIR, SWIR, and derived index features.

## Completed Work

The training input has moved from a weak-input route to a post-disaster multispectral route.

The current V3 input stack has 11 channels:

```text
F01 F02 F03 F04 F07 F11 F12 NDVI NBR NDMI BRIGHTNESS
```

Implemented capabilities include:

- Post-disaster same-date scene selection.
- Cloud-proxy quality ranking.
- Multiband resampling to the mask grid.
- Derived index channel generation.
- C5 fire training bundle construction.
- RTX 4080 training configuration.
- AMP, `channels_last`, `cudnn_benchmark`, and `cache_records`.
- 11-channel preview rendering fix.
- `best_mean_iou.pt` and `best_mean_iou.json` checkpoint saving.
- Threshold sweep extension to `0.9`.

Related training branch:

```text
codex/rtx4080-training-performance
```

## Training Observations

The U-Net baseline no longer has zero foreground recall, which means the multispectral input provides learnable signal.

The current stable run produced:

```text
last mean_iou          ≈ 0.2501
best mean_iou          ≈ 0.3041
foreground_recall      ≈ 0.83
foreground_precision   ≈ 0.26
predicted/label area   ≈ 3.17x
```

Compared with the previous run:

```text
previous last mean_iou ≈ 0.2176
previous best mean_iou ≈ 0.3536
stable last mean_iou   ≈ 0.2501
stable best mean_iou   ≈ 0.3041
```

The stable configuration improved observability and slightly improved the final metric, but it did not produce a decisive model-quality breakthrough.

## Key Conclusion

The largest improvement in this stage was observability, not model capability.

The corrected visualization shows that the model is not failing to learn fire foreground entirely. It can cover part of the fire body, but it also predicts many similar background regions, including bare land, mountain textures, bright roads, and cloud-edge artifacts.

The main failure pattern is:

```text
Recall is non-zero and often high, but precision is low.
Predicted foreground area is about 3x the label area.
Training remains unstable.
Plain U-Net has limited representation and context modeling capacity for this task.
```

Further small adjustments to batch size, loss, or learning rate on the same plain U-Net are unlikely to deliver large gains.

## Next Goal

The next stage should be a stronger segmentation architecture comparison.

Goal:

```text
Introduce stronger segmentation architectures while holding the V3 11-channel input, split policy, validation set, visualization protocol, and threshold sweep constant. Verify whether the current U-Net ceiling can be exceeded.
```

Recommended experiment name:

```text
SegmentationModelV4: stronger architecture comparison
```

## Recommended Routes

The first priority is a low-cost architecture upgrade:

- `ResUNet`
- `UNet++`
- `DeepLabV3+`

`DeepLabV3+` or `UNet++` should be prioritized because the current failures are mainly boundary quality, multiscale context, and background false positives.

The second priority is a pretrained encoder or SegFormer route:

- SegFormer
- Pretrained encoder with an 11-channel adapter
- Remote-sensing foundation-model encoder

This route has higher upside but also higher implementation complexity because the 11-channel input must be adapted to pretrained weights.

## Next Acceptance Criteria

The next stage should report an architecture comparison table, not a single metric:

```text
model_family
best_mean_iou
foreground_precision
foreground_recall
foreground_dice
predicted/label area ratio
best checkpoint epoch
validation_contact_sheet
```

Minimum target:

```text
best_mean_iou > 0.35
foreground_precision clearly above 0.26
predicted/label area ratio reduced from about 3x toward 1-2x
validation previews show visibly fewer large background false positives
```

If stronger CNN architectures still cannot exceed the current ceiling, the project should move to SegFormer or a pretrained encoder route instead of continuing to tune plain U-Net.

