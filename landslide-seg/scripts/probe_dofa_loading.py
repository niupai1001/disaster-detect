"""Read-only actual official backbone load and optional decoder dependency probe."""
import importlib.util
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch
torch.set_num_threads(4)
VENDOR = ROOT / "vendor/dofa-official-0cfb7e1"
sys.path.insert(0, str(VENDOR))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


if __name__ == "__main__":
    official = module("landslide_official_dofa_seg", VENDOR / "downstream_tasks/geobench_segmentation/model.py")
    position = module("landslide_dofa_pos_embed", VENDOR / "pretraining/util/pos_embed.py")
    backbone = official.vit_base_patch16(img_size=128, drop_path_rate=0.)
    path = ROOT / "weights/dofa/XShadow_DOFA_base/DOFA_ViT_base_e100.pth"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model", checkpoint)
    position.interpolate_pos_embed(backbone, state)
    loading = backbone.load_state_dict(state, strict=False)
    report = {"missing_keys": loading.missing_keys, "unexpected_keys": loading.unexpected_keys,
              "params": sum(p.numel() for p in backbone.parameters()), "pos_shape": list(backbone.pos_embed.shape)}
    try:
        from terratorch.models.necks import LearnedInterpolateToPyramidal
        from terratorch.models.decoders.upernet_decoder import UperNetDecoder
        report["terratorch_decoder_import"] = "PASS"
    except Exception as exc:
        report["terratorch_decoder_import"] = repr(exc)
    from transformers import UperNetConfig
    from transformers.models.upernet.modeling_upernet import UperNetHead
    head = UperNetHead(UperNetConfig(hidden_size=512, num_labels=1, use_auxiliary_head=False), in_channels=[768]*4)
    report["transformers_upernet_params"] = sum(p.numel() for p in head.parameters())
    print(json.dumps(report, indent=1))
    destination = ROOT / "reports/v4r3_dofa_loading_probe.json"
    destination.write_text(json.dumps(report, indent=1), encoding="utf-8")
