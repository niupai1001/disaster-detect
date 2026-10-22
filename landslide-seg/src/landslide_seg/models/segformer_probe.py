"""Frozen encoder / common official decoder probe; no new segmentation components."""
from __future__ import annotations

import hashlib

import torch

from ..config import file_sha256, resolve_path
from .segformer import SegFormerSeg


def state_digest(state) -> str:
    """Hash named tensors, including buffers, independently of torch.save metadata."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class FrozenEncoderSegFormer(SegFormerSeg):
    def __init__(self, *, probe: dict, **kwargs):
        if kwargs.get("stem") or kwargs.get("refine", "none") != "none" or kwargs.get("adapter") != "direct10":
            raise ValueError("Decoder probe requires direct10, no stem and no refiner")
        source = resolve_path(probe["checkpoint"])
        identity = file_sha256(source)
        if identity != probe["checkpoint_sha256"]:
            raise ValueError("Probe source checkpoint hash differs")
        # Fresh official decoder + seed-controlled binary head, before loading encoder.
        super().__init__(**kwargs)
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        model_config = checkpoint["config"]["segformer"]["model"]
        if (model_config.get("type", "segformer") != "segformer"
                or model_config.get("stem", False)
                or model_config.get("adapter") != self.adapter_type
                or model_config.get("backbone") != self.backbone):
            raise ValueError("Probe source architecture differs")
        if checkpoint["global_step"] != int(probe["source_step"]):
            raise ValueError("Probe source update differs")
        if checkpoint["meta"]["stats_sha256"] != probe["stats_sha256"]:
            raise ValueError("Probe source normalization differs")
        if checkpoint["meta"]["manifest"]["sha256"] != probe["manifest_sha256"]:
            raise ValueError("Probe source manifest differs")
        prefix = "model.segformer."
        encoder = {name[len(prefix):]: value for name, value in checkpoint["model"].items()
                   if name.startswith(prefix)}
        self.model.segformer.load_state_dict(encoder, strict=True)
        self.model.segformer.requires_grad_(False)
        self.model.segformer.eval()
        self.probe_report = {
            "kind": "frozen_encoder_common_official_decoder_nonlinear_probe",
            "source_checkpoint": str(source), "source_checkpoint_sha256": identity,
            "source_step": checkpoint["global_step"],
            "source_experiment": checkpoint["config"]["segformer"]["experiment"],
            "source_stats_sha256": checkpoint["meta"]["stats_sha256"],
            "source_manifest_sha256": checkpoint["meta"]["manifest"]["sha256"],
            "encoder_initial_sha256": state_digest(self.model.segformer.state_dict()),
            "decoder_initial_sha256": state_digest(self.model.decode_head.state_dict()),
            "frozen_parameters": sum(p.numel() for p in self.model.segformer.parameters()),
            "decoder_parameters": sum(p.numel() for p in self.model.decode_head.parameters()),
            "source_decoder_loaded": False,
        }

    def train(self, mode: bool = True):
        super().train(mode)
        # The existing trainer calls model.train() on every training chunk.
        self.model.segformer.eval()
        return self

    def probe_integrity(self) -> dict:
        encoder_hash = state_digest(self.model.segformer.state_dict())
        decoder_hash = state_digest(self.model.decode_head.state_dict())
        return {
            "encoder_sha256": encoder_hash, "decoder_sha256": decoder_hash,
            "encoder_unchanged": encoder_hash == self.probe_report["encoder_initial_sha256"],
            "decoder_changed": decoder_hash != self.probe_report["decoder_initial_sha256"],
            "encoder_has_grad": any(p.grad is not None for p in self.model.segformer.parameters()),
            "encoder_training": self.model.segformer.training,
        }
