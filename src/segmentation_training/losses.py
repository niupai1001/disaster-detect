from __future__ import annotations


def build_cross_entropy_loss(*, ignore_index: int = 255, class_weights=None):
    import torch

    weight = None
    if class_weights is not None:
        weight = torch.tensor(class_weights, dtype=torch.float32)
    return torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=weight)

