"""Compatibility with frozen A masks and value-isolated optical B1 views."""
from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace

import torch
from torch import nn

from encoder_v3.hierarchical.geometry import area_map, partitions
from encoder_v3.hierarchical.model import HierarchicalMAE, occupied_regions
from encoder_v3.tasks import hidden_band_indices

from .frontend import attribute
from .geometry import Grid
from .model import Decoder, Encoder, VisibleChannel
from .routes import NativeIndex


@lru_cache(maxsize=8)
def legacy_a_layout(grids, metres, device):
    # Only the legacy mask's Coarse map is required. No legacy model is instantiated.
    geo, parts, _ = partitions(grids, metres)
    unique = {g: area_map(g, geo, parts[2], device) for g in set(grids)}
    return geo, parts, [(None, None, unique[g]) for g in grids], None, None


def task_masks(sample, config, kind, seed, epoch):
    channels = sample.channels
    if kind == "A":
        layout = legacy_a_layout(tuple(c.grid for c in channels), config.mask_patch_metres,
                                 channels[0].valid.device)
        options = SimpleNamespace(mask_ratio=config.mask_ratio)
        return HierarchicalMAE.masks(options, sample, layout, "A", seed, epoch)[:2]
    if kind != "B1":
        raise ValueError("A or B1 required")
    eligible = [i for i, c in enumerate(channels)
                if attribute(c.physical, "modality") == "optical" and bool(c.valid.any())]
    if len(eligible) < 2:
        raise ValueError("B1 requires two valid optical channels")
    hidden = {eligible[i] for i in hidden_band_indices(len(eligible), seed=seed, update=epoch,
                                                     observation_id=sample.identifier)}
    keeps = [torch.zeros_like(c.valid) if i in hidden else c.valid for i, c in enumerate(channels)]
    targets = [c.valid if i in hidden else torch.zeros_like(c.valid) for i, c in enumerate(channels)]
    return keeps, targets


def available_tasks(sample, config):
    channels = sample.channels
    grids = tuple(c.grid for c in channels)
    tasks = []
    if int(occupied_regions(sample, legacy_a_layout(grids, config.mask_patch_metres,
                                                   channels[0].valid.device)).sum()) >= 2:
        tasks.append("A")
    if sum(attribute(c.physical, "modality") == "optical" and bool(c.valid.any()) for c in channels) >= 2:
        tasks.append("B1")
    return tasks


@dataclass
class TaskView:
    visible: tuple
    requests: NativeIndex
    truth: tuple

    @classmethod
    def build(cls, sample, keeps, targets):
        grids = tuple(Grid.from_native(c.grid) for c in sample.channels)
        visible = tuple(VisibleChannel(torch.where(keep, c.values, 0.), keep, grid, c.physical)
                        for c, keep, grid in zip(sample.channels, keeps, grids))
        requests = NativeIndex.build(grids, targets)
        truth = tuple(c.values.detach()[mask] for c, mask in zip(sample.channels, targets))
        return cls(visible, requests, truth)


@dataclass
class Result:
    loss: torch.Tensor
    channel_loss: dict
    predictions: torch.Tensor
    encoded: object
    targets: NativeIndex


class Pretraining(nn.Module):
    def __init__(self, config, *, decoder_class=Decoder):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = decoder_class(config)

    def forward(self, sample, *, kind, seed, epoch=0, metrics=None):
        view = TaskView.build(sample, *task_masks(sample, self.config, kind, seed, epoch))
        encoded = self.encoder(view.visible, metrics)
        specs = [c.physical for c in view.visible]
        hidden = [c for c, ids in enumerate(view.requests.ids)
                  if len(ids) and not len(encoded.native.ids[c])]
        descriptor = encoded.descriptor
        if hidden:
            target = self.encoder.descriptor([specs[c] for c in hidden])
            descriptor = descriptor.index_copy(0, torch.tensor(hidden, device=target.device), target)
        predicted = self.decoder(encoded, view.requests, specs, descriptor, metrics)
        losses = {i: (predicted[start:end].float()-truth.float()).square().mean()
                  for i, (start, end, truth) in enumerate(zip(view.requests.offsets,
                                                            view.requests.offsets[1:], view.truth))
                  if end > start}
        return Result(torch.stack(list(losses.values())).mean(), losses, predicted, encoded, view.requests)
