from __future__ import annotations


def class_weights_from_counts(pixel_counts, *, max_weight: float = 20.0) -> list[float]:
    counts = [max(0, int(count)) for count in pixel_counts]
    if not counts:
        raise ValueError("pixel_counts must not be empty")
    background = max(1, counts[0])
    weights = [1.0]
    for count in counts[1:]:
        if count <= 0:
            weights.append(float(max_weight))
        else:
            weights.append(float(min(max_weight, max(1.0, background / count))))
    return weights


def build_cross_entropy_loss(*, ignore_index: int = 255, class_weights=None):
    import torch

    weight = None
    if class_weights is not None:
        weight = torch.tensor(class_weights, dtype=torch.float32)
    return torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=weight)


def build_segmentation_loss(
    *,
    loss_name: str,
    ignore_index: int = 255,
    class_weights=None,
    dice_weight: float = 1.0,
):
    if loss_name in {"cross_entropy", "weighted_cross_entropy"}:
        return build_cross_entropy_loss(ignore_index=ignore_index, class_weights=class_weights)
    if loss_name == "cross_entropy_dice":
        return _CrossEntropyDiceLoss(ignore_index=ignore_index, class_weights=class_weights, dice_weight=dice_weight)
    raise ValueError(f"Unsupported loss {loss_name!r}")


class _CrossEntropyDiceLoss:
    def __init__(self, *, ignore_index: int, class_weights=None, dice_weight: float = 1.0):
        import torch

        self.torch = torch
        self.cross_entropy = build_cross_entropy_loss(ignore_index=ignore_index, class_weights=class_weights)
        self.ignore_index = ignore_index
        self.dice_weight = float(dice_weight)

    def to(self, device):
        self.cross_entropy.to(device)
        return self

    def __call__(self, logits, labels):
        torch = self.torch
        ce_loss = self.cross_entropy(logits, labels)
        valid = labels != self.ignore_index
        if not bool(valid.any().item()):
            return ce_loss
        probabilities = torch.softmax(logits, dim=1)
        dice_losses = []
        for class_index in range(1, logits.shape[1]):
            target = torch.logical_and(valid, labels == class_index).float()
            prediction = probabilities[:, class_index, :, :] * valid.float()
            denom = prediction.sum() + target.sum()
            if float(denom.detach().cpu().item()) == 0.0:
                continue
            dice = (2.0 * (prediction * target).sum() + 1.0) / (denom + 1.0)
            dice_losses.append(1.0 - dice)
        if not dice_losses:
            return ce_loss
        return ce_loss + self.dice_weight * torch.stack(dice_losses).mean()
