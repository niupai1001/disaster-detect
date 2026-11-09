"""Physical-unit metrics and equal band/observation/source reduction."""
from collections import defaultdict
import math
import torch


def band_loss(prediction, truth, valid, supported, *, train_std):
    if prediction.shape!=truth.shape or valid.shape!=truth.shape or supported.shape!=truth.shape:
        raise ValueError('Target, prediction and masks must match')
    if valid.dtype!=torch.bool or supported.dtype!=torch.bool or not math.isfinite(train_std) or train_std<=0:
        raise ValueError('Boolean masks and positive TRAIN-only standard deviation required')
    keep=valid & supported
    if not bool(keep.any()): return None,dict(status='NO_VALID_SUPPORTED_TARGET',count=0,unsupported=int((valid & ~supported).sum()))
    p,y=prediction[keep],truth[keep]
    if not bool(torch.isfinite(p).all() & torch.isfinite(y).all()): raise FloatingPointError('Nonfinite supported prediction or label')
    error=p-y; mse=error.square().mean(); loss=mse/(train_std**2)
    return loss,dict(status='OK',count=int(keep.sum()),unsupported=int((valid & ~supported).sum()),nmse=float(loss.detach()),mae=float(error.detach().abs().mean()),rmse=float(mse.detach().sqrt()),bias=float(error.detach().mean()),prediction_variance=float(p.detach().var(unbiased=False)),target_variance=float(y.detach().var(unbiased=False)))


def hierarchical_loss(records):
    """Records carry source, observation, band and a scalar loss (or None)."""
    observations=defaultdict(list)
    identities=set()
    for record in records:
        identity=(record['source'],record['observation'],record['band'])
        if identity in identities: raise ValueError('Duplicate band loss: combine pixel chunks before reduction')
        identities.add(identity)
        if record['loss'] is not None: observations[identity[:2]].append(record['loss'])
    sources=defaultdict(list)
    for (source,_),losses in observations.items(): sources[source].append(torch.stack(losses).mean())
    if not sources: return None
    return torch.stack([torch.stack(losses).mean() for losses in sources.values()]).mean()
