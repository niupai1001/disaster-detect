import json
import sys
from pathlib import Path

import torch

ROOT = Path(r"N:\Users\yzh\Disater\landslide-seg")
sys.path.insert(0, str(ROOT / "src"))
from landslide_seg.config import load_segformer_config  # noqa: E402
from landslide_seg.models.factory import build_model  # noqa: E402

torch.manual_seed(0)
for name in ["s1_indices", "s1_control", "s3_a_linear", "s3_b_idattn", "s3_c_wavelength_attn",
             "s3_d_local_bias", "s3_e_global_bias", "s4_add", "s4_fixed_gate",
             "s4_learn_gate", "s4_capctrl"]:
    cfg = load_segformer_config(ROOT / "configs/segformer/v4r2_spectral" / f"{name}.yaml")
    model = build_model(cfg["segformer"])
    model.eval()
    counts = model.count_params()
    rep = model.spectral_report() if hasattr(model, "spectral_report") else {}
    x = torch.randn(2, 10, 128, 128)
    with torch.no_grad():
        out = model(x)
    n_new = counts.get("new")
    print(f"{name:22s} logits={tuple(out['logits'].shape)} new={n_new:7d} "
          f"total={counts['total']:9d} mech={json.dumps(rep, ensure_ascii=False)[:150]}")
    if name.startswith("s4"):
        hook = model.visual.model.decode_head.classifier.register_forward_pre_hook(
            model.visual._capture)
        with torch.no_grad():
            base = model.visual(x)["logits"]
        hook.remove()
        diff = (out["logits"] - base).abs().max().item()
        print("   gate initial max|diff| vs visual:", diff)
        info = out["spectral_info"]
        assert info["alpha"] == 0.0 and info["corr_abs_mean"] == 0.0, info
        assert diff < 1e-6, diff
    if name.startswith("s1"):
        w = model.visual.model.segformer.encoder.patch_embeddings[0].proj.weight
        assert w.shape[1] == 14
        print("   first conv extra col abs sum:", float(w[:, 10:].abs().sum()),
              "base col sum:", float(w[:, :10].abs().sum()))
