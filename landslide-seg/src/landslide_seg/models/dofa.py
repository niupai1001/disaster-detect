"""Thin assembly: official DOFA dense encoder + MMSeg neck + HF UPerNet head."""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from ..config import resolve_path, file_sha256
from ..data.segformer_dataset import S2_10_BANDS


def _module(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


class DOFASeg(nn.Module):
    def __init__(self, config):
        super().__init__()
        for path, expected in config["code_sha256"].items():
            if file_sha256(resolve_path(path)) != expected:
                raise ValueError(f"DOFA component source changed: {path}")
        checkpoint = resolve_path(config["checkpoint"])
        if file_sha256(checkpoint) != config["checkpoint_sha256"]:
            raise ValueError("DOFA pretrained checkpoint hash differs")
        source = resolve_path(config["code_dir"])
        runtime = resolve_path(config["runtime_dir"])
        # Isolated pure-Python optional dependencies; no environment installation.
        for path in (source, runtime):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        official = _module("landslide_dofa_official_dense", source / "downstream_tasks/geobench_segmentation/model.py")
        position = _module("landslide_dofa_official_position", source / "pretraining/util/pos_embed.py")
        pyramid = _module("landslide_mmseg_feature2pyramid", runtime / "mmseg/models/necks/featurepyramid.py")
        from transformers import UperNetConfig
        from transformers.models.upernet.modeling_upernet import UperNetHead
        self.image_size = int(config.get("image_size", 128))
        if self.image_size != 128 or list(config["band_order"]) != list(S2_10_BANDS):
            raise ValueError("DOFA control requires unchanged 128 grid and ten-band order")
        self.wavelengths = tuple(float(v) for v in config["wavelengths_micrometers"])
        if len(self.wavelengths) != 10 or any(v <= 0 or v > 3 for v in self.wavelengths):
            raise ValueError("Ten optical wavelengths in micrometers required")
        self.encoder = official.vit_base_patch16(img_size=128, drop_path_rate=0.)
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = weights.get("model", weights)
        old_pos = list(state["pos_embed"].shape)
        position.interpolate_pos_embed(self.encoder, state)
        loaded = self.encoder.load_state_dict(state, strict=False)
        allowed = {"mask_token", "norm.weight", "norm.bias", "projector.weight", "projector.bias"}
        if loaded.missing_keys or set(loaded.unexpected_keys) - allowed:
            raise ValueError(f"DOFA encoder weights incomplete: {loaded}")
        self.neck = pyramid.Feature2Pyramid(embed_dim=768, rescales=[4, 2, 1, .5])
        head_config = UperNetConfig(hidden_size=int(config.get("decoder_hidden_size", 512)),
                                   num_labels=1, use_auxiliary_head=False)
        self.decoder = UperNetHead(head_config, in_channels=[768]*4)
        self.backbone = "dofa_base_patch16"
        self.adapter_type = "wavelength_conditioned_10band"
        self.use_stem = False
        self.refine = "none"
        self.freeze_encoder = bool(config.get("freeze_encoder", False))
        if self.freeze_encoder:
            self.encoder.requires_grad_(False)
            self.encoder.eval()
        self.load_info = {"missing_keys": list(loaded.missing_keys), "unexpected_keys": list(loaded.unexpected_keys),
                          "unexpected_keys_reason": "pretraining-only mask/norm/projector omitted by official dense encoder",
                          "checkpoint": str(checkpoint), "checkpoint_sha256": config["checkpoint_sha256"],
                          "position_before": old_pos, "position_after": list(self.encoder.pos_embed.shape),
                          "position_method": "official interpolate_pos_embed; input image not resized",
                          "code_sha256": config["code_sha256"], "official_commit": config["official_commit"],
                          "decoder": "Transformers UperNetHead 4.57.6", "neck": "MMSegmentation Feature2Pyramid 1.2.2",
                          "encoder_frozen": self.freeze_encoder}
        self.adapter_report = {"type": self.adapter_type, "band_order": list(S2_10_BANDS),
                               "wavelengths_micrometers": list(self.wavelengths),
                               "wavelength_source": config["wavelength_source"], "input_size": 128,
                               "native_feature_stride": 16, "decoder_pyramid_strides": [4, 8, 16, 32],
                               "note": "learned feature rescaling is not additional observed spatial detail; single binary head"}

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def forward(self, x, *, return_features=False, **kwargs):
        if x.ndim != 4 or x.shape[1:] != (10, 128, 128):
            raise ValueError(f"DOFA expects Bx10x128x128, received {tuple(x.shape)}")
        features = self.encoder(x, self.wavelengths)
        pyramid = self.neck(features)
        logits = F.interpolate(self.decoder(pyramid), size=x.shape[-2:], mode="bilinear", align_corners=False)
        result = {"logits": logits, "base_logits": logits}
        if return_features:
            result.update(encoder_features=features, pyramid_features=pyramid)
        return result

    def param_groups(self):
        return {"encoder": [p for p in self.encoder.parameters() if p.requires_grad],
                "decoder": [p for p in self.decoder.parameters() if p.requires_grad],
                "new": [p for p in self.neck.parameters() if p.requires_grad]}

    def count_params(self):
        groups = self.param_groups()
        return {"total": sum(p.numel() for p in self.parameters()),
                "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
                **{name: sum(p.numel() for p in params) for name, params in groups.items()}}

    def projection_change(self):
        return None
