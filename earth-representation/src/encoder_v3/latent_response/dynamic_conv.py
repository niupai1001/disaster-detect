"""Shared dynamic 3x3 kernel bank and mask-aware native convolution.

Kernel composition is the frozen specification formula
``W0 + sum_m alpha_m B_m`` with ``alpha = softmax(Linear(z_conv))``.
Parameter initialization is an engineering default, not a frozen optimum.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class DynamicKernelBank(nn.Module):
    def __init__(self, *, d=32, bases=8, kernel_size=3):
        super().__init__()
        if min(d, bases, kernel_size) < 1:
            raise ValueError('Positive descriptor, basis and kernel sizes required')
        self.base = nn.Parameter(torch.empty(d, 1, kernel_size, kernel_size))
        self.basis = nn.Parameter(torch.empty(bases, d, 1, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(d))
        nn.init.kaiming_uniform_(self.base, a=math.sqrt(5))
        nn.init.normal_(self.basis, std=0.01)
        self.coefficients_layer = nn.Linear(d, bases)

    def coefficients(self, z_conv):
        if z_conv.ndim != 2 or z_conv.shape[-1] != self.base.shape[0]:
            raise ValueError('z_conv must be [N, descriptor width]')
        return torch.softmax(self.coefficients_layer(z_conv), -1)

    def kernels(self, z_conv):
        alpha = self.coefficients(z_conv)
        return self.base[None] + torch.einsum('nb,bdkhw->ndkhw', alpha, self.basis)


def masked_native_conv(values, keep, kernels, bias, *, tile_rows=None):
    """Masked 3x3 stride-1 convolution with visible-neighbour count compensation and GELU.

    values [N,H,W]; keep bool [N,H,W] = valid & visible; kernels [N,C,1,3,3]; bias [C].
    Hidden or invalid values are replaced by zero before the convolution, so no
    hidden scalar can enter the result or its gradient.
    """
    if values.ndim != 3 or not values.is_floating_point():
        raise ValueError('Floating [N,H,W] native values required')
    n, height, width = values.shape
    if keep.shape != values.shape or keep.dtype is not torch.bool:
        raise ValueError('Boolean keep mask matching values required')
    channels = bias.shape[0] if bias.ndim == 1 else -1
    if kernels.ndim != 5 or kernels.shape != (n, channels, 1, 3, 3) or kernels.dtype != values.dtype:
        raise ValueError('Per-sample [N,C,1,3,3] kernels matching the bias required')
    rows = height if tile_rows is None else tile_rows
    if type(rows) is not int or rows < 1:
        raise ValueError('Positive integer tile height required')
    x = torch.where(keep, values, values.new_zeros(()))
    weight = kernels.reshape(n * channels, 1, 3, 3)
    ones = values.new_ones(1, 1, 3, 3)
    outputs = []
    for begin in range(0, height, rows):
        end = min(begin + rows, height)
        low, high = max(0, begin - 1), min(height, end + 1)
        count = F.conv2d(keep[:, None, low:high].to(values.dtype), ones, padding=1)
        output = F.conv2d(x[:, low:high].reshape(1, n, high - low, width), weight, padding=1, groups=n)
        output = output.reshape(n, channels, high - low, width)
        output = output * (9.0 / count.clamp_min(1)) + bias[None, :, None, None]
        outputs.append(output[..., begin - low:end - low, :])
    return F.gelu(torch.cat(outputs, -2))
