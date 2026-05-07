from __future__ import annotations

from .attention_unet import build_attention_unet
from .deeplabv3_plus import build_deeplabv3_plus
from .resunet import build_resunet
from .transformer_unet import build_transformer_unet
from .unet import build_unet
from .unet_plus_plus import build_unet_plus_plus


def _config_value(config, key, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _normalized_family(config) -> str:
    family = str(_config_value(config, "family", "unet")).lower()
    aliases = {
        "unet++": "unet_plus_plus",
        "unetpp": "unet_plus_plus",
        "deeplabv3plus": "deeplabv3_plus",
    }
    return aliases.get(family, family)


def build_model(config, *, input_channels: int, output_classes: int):
    family = _normalized_family(config)
    base_channels = int(_config_value(config, "base_channels", 32))

    if family == "unet":
        return build_unet(input_channels=input_channels, output_classes=output_classes, base_channels=base_channels)
    if family == "resunet":
        return build_resunet(input_channels=input_channels, output_classes=output_classes, base_channels=base_channels)
    if family == "attention_unet":
        return build_attention_unet(input_channels=input_channels, output_classes=output_classes, base_channels=base_channels)
    if family == "unet_plus_plus":
        return build_unet_plus_plus(input_channels=input_channels, output_classes=output_classes, base_channels=base_channels)
    if family == "deeplabv3_plus":
        return build_deeplabv3_plus(input_channels=input_channels, output_classes=output_classes, base_channels=base_channels)
    if family == "transformer_unet":
        return build_transformer_unet(
            input_channels=input_channels,
            output_classes=output_classes,
            base_channels=base_channels,
            transformer_layers=int(_config_value(config, "transformer_layers", 1)),
            num_heads=int(_config_value(config, "num_heads", 2)),
            token_grid=int(_config_value(config, "token_grid", _config_value(config, "patch_size", 16))),
        )

    raise ValueError(f"Unsupported model family: {family}")
