"""Shared sensor-memory read: z = softmax(Q K^T / sqrt(d_k)) V (frozen specification formula)."""
import math
import torch


def read_memory(query, keys, values, *, scale=None):
    """Read shared or exactly batch-aligned memory without batch broadcasting.

    Shared keys[M,d], values[M,dv] accept query[...,d]. Batched memory
    keys[*batch,M,d], values[*batch,M,dv] accepts query[*batch,d] (one
    per sample) or query[*batch,Q,d] (multiple per sample). Batch shapes
    must match exactly. Returns query.shape[:-1] + (M,) / (dv,).
    """
    if query.ndim < 1 or keys.ndim < 2 or values.ndim < 2:
        raise ValueError('Query [...,d] and memory [...,M,d] tensors required')
    if keys.shape[-2] != values.shape[-2] or keys.shape[:-2] != values.shape[:-2]:
        raise ValueError('Key and value memories must share batch and slot dimensions')
    if keys.shape[-1] != query.shape[-1] or keys.shape[-2] < 1 or keys.shape[-1] < 1:
        raise ValueError('Nonempty memory with matching key width required')
    if not all(tensor.is_floating_point() for tensor in (query, keys, values)):
        raise ValueError('Floating query, key and value tensors required')
    if not all(bool(torch.isfinite(tensor).all()) for tensor in (query, keys, values)):
        raise ValueError('Finite query, key and value tensors required')
    if any(t.dtype != query.dtype or t.device != query.device for t in (keys, values)):
        raise ValueError('Query, keys and values must share dtype and device')
    batch = keys.shape[:-2]
    single = False
    if batch:
        single = query.shape[:-1] == batch
        if not single and query.shape[:-2] != batch:
            raise ValueError('Query and memory batch dimensions must match exactly')
    if scale is None:
        scale = 1.0 / math.sqrt(keys.shape[-1])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Positive finite attention scale required')
    aligned_query = query.unsqueeze(-2) if single else query
    weights = torch.softmax((aligned_query @ keys.transpose(-1, -2)) * scale, -1)
    read = weights @ values
    return (weights.squeeze(-2), read.squeeze(-2)) if single else (weights, read)
