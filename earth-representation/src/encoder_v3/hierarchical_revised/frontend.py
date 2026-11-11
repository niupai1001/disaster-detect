"""Typed metadata and one shared conditional native convolution."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from encoder_v3.shared_mae.metadata import POLARIZATIONS, QUANTITIES
from encoder_v3.shared_mae.model import ConditionalConvolution


def attribute(spec, name):
    return getattr(spec, name, None)


def typed_fields(spec):
    modality = attribute(spec, "modality")
    optical, sar = modality == "optical", modality == "sar"
    fields = []
    for name, applies in (("gx", True), ("gy", True), ("center_um", optical),
                          ("fwhm_um", optical), ("frequency_ghz", sar), ("incidence_deg", sar)):
        value = attribute(spec, name)
        valid = (applies and value is not None and math.isfinite(value)
                 and (0 <= value <= 90 if name == "incidence_deg" else value > 0))
        fields.extend((value/90 if valid and name == "incidence_deg"
                       else math.log(value) if valid else 0., float(valid), float(applies)))
    for value, types in ((modality, ("optical", "sar")),
                         (attribute(spec, "quantity"), QUANTITIES),
                         (attribute(spec, "polarization") if sar else None, POLARIZATIONS[1:])):
        fields.extend(float(value == category) for category in (*types, None))
        if value not in types and value is not None:
            fields[-1] = 1.
    return fields


class Descriptor(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(len(typed_fields(None)), 2*width),
                                     nn.GELU(), nn.Linear(2*width, width))

    def forward(self, specs):
        weight = self.network[0].weight
        return self.network(weight.new_tensor([typed_fields(spec) for spec in specs]))


class NativeConvolution(nn.Module):
    def __init__(self, descriptor, width, bases):
        super().__init__()
        self.generator = ConditionalConvolution(width, bases)
        self.generator.coefficients = nn.Linear(descriptor, bases)

    def forward(self, values, keep, kernel, bias, tile_rows):
        clean = torch.where(keep, values, 0.)
        height = len(values)
        parts = []
        for start in range(0, height, tile_rows):
            end = min(start + tile_rows, height)
            lo, hi = max(0, start-1), min(height, end+1)
            feature = F.conv2d(clean[None, None, lo:hi], kernel, bias, padding=1)
            native = feature[0, :, start-lo:end-lo].permute(1, 2, 0)
            parts.append(F.gelu(native[keep[start:end]]))
        return torch.cat(parts)
