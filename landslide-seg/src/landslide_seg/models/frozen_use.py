"""Task-independent USE boundary, including parameter and buffer integrity."""
import hashlib

import torch
from torch import nn


def state_sha256(module):
    digest = hashlib.sha256()
    # named_buffers also includes nonpersistent runtime buffers omitted by state_dict.
    tensors = [('parameter.' + name, tensor) for name, tensor in module.named_parameters()]
    tensors += [('buffer.' + name, tensor) for name, tensor in module.named_buffers()]
    for name, tensor in sorted(tensors):
        value = tensor.detach().cpu().contiguous()
        digest.update(str((name, str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class FrozenUSE(nn.Module):
    """Load the selected SSL state before wrapping; never reset the reference hash."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder.requires_grad_(False).eval()
        self.reference_sha256 = state_sha256(encoder)
        self.train(False)

    def train(self, mode=True):
        super().train(False)
        self.encoder.requires_grad_(False)
        return self

    def verify(self, optimizer=None):
        if any(m.training for m in self.encoder.modules()) or any(p.requires_grad or p.grad is not None
                                        for p in self.encoder.parameters()):
            raise RuntimeError('Frozen USE training/gradient boundary violated')
        if optimizer is not None:
            identities = {id(p) for p in self.encoder.parameters()}
            if any(id(p) in identities for group in optimizer.param_groups for p in group['params']):
                raise RuntimeError('Optimizer contains frozen USE parameters')
        if state_sha256(self.encoder) != self.reference_sha256:
            raise RuntimeError('Frozen USE parameter/buffer SHA256 changed')
        return self.reference_sha256

    def forward(self, observation):
        allowed = {'x', 'band_metadata', 'known_mask', 'band_valid', 'pixel_valid',
                   'band_ids', 'spatial_metadata'}
        if set(observation) - allowed:
            raise ValueError('Task labels must not enter USE')
        self.verify()
        with torch.no_grad():
            output = self.encoder(observation)
        self.verify()
        return {key: value.detach() if torch.is_tensor(value) else value
                for key, value in output.items()}
