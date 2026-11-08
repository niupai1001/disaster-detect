"""Shared sensor-memory read: z = softmax(Q K^T / sqrt(d_k)) V (frozen specification formula)."""
import math
import torch


def read_memory(query, keys, values, *, scale=None):
    """query [...,d]; keys [...,M,d]; values [...,M,d_v]; returns weights [...,M], read [...,d_v].

    One shared implementation for Q_conv/Q_channel/Q_decoder. Empty memory is rejected
    instead of producing a NaN softmax.
    """
    if query.ndim < 1 or keys.ndim < 2 or values.ndim < 2:
        raise ValueError('Query [...,d] and memory [...,M,d] tensors required')
    if keys.shape[-2] != values.shape[-2] or keys.shape[:-2] != values.shape[:-2]:
        raise ValueError('Key and value memories must share batch and slot dimensions')
    if keys.shape[-1] != query.shape[-1] or keys.shape[-2] < 1:
        raise ValueError('Nonempty memory with matching key width required')
    if not all(tensor.is_floating_point() for tensor in (query, keys, values)):
        raise ValueError('Floating query, key and value tensors required')
    if not all(bool(torch.isfinite(tensor).all()) for tensor in (query, keys, values)):
        raise ValueError('Finite query, key and value tensors required')
    if scale is None:
        scale = 1.0 / math.sqrt(keys.shape[-1])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Positive finite attention scale required')
    weights = torch.softmax((query @ keys.transpose(-1, -2)) * scale, -1)
    return weights, weights @ values
