"""Fixed 128-grid DOFA binary system config and provenance; no split changes."""
import ast
import importlib.util
import json
import sys
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from landslide_seg.config import load_segformer_config, file_sha256
from landslide_seg.data.segformer_dataset import S2_10_BANDS


if __name__ == "__main__":
    cfg = load_segformer_config(ROOT / "configs/segformer/v4r1/segformer_b0_direct10_n100_u3600_s1234.yaml")
    s = cfg["segformer"]
    s["experiment"] = "dofa_base_upernet_pyramid_binary_n100_u3600_s1234"
    s["data"]["version"] = "s2_10band_v4r3_dofa"
    source = Path("C:/Users/yzh/.workbuddy/binaries/python/envs/tm/Lib/site-packages/terratorch/models/backbones/dofa_vit.py")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    waves = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "waves_list" for t in node.targets))
    keys = ["BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3", "NIR_BROAD", "NIR_NARROW", "SWIR_1", "SWIR_2"]
    wavelengths = [waves[key] for key in keys]
    mapping = {"source": "TerraTorch 1.2.13 dofa_vit.py waves_list; nominal optical wavelengths in micrometers",
               "source_sha256": file_sha256(source), "bands": list(S2_10_BANDS), "names": keys, "wavelengths": wavelengths}
    wave_path = ROOT / "vendor/dofa_wavelength_mapping_terratorch_1_2_13.json"
    content = json.dumps(mapping, indent=1)
    if wave_path.exists() and wave_path.read_text(encoding="utf-8") != content:
        raise FileExistsError("Wavelength mapping changed")
    wave_path.write_text(content, encoding="utf-8")
    identity = json.loads((ROOT / "weights/dofa/XShadow_DOFA_base/weight_identity.json").read_text(encoding="utf-8"))
    sources = ["vendor/dofa-official-0cfb7e1/downstream_tasks/geobench_segmentation/model.py",
               "vendor/dofa-official-0cfb7e1/wave_dynamic_layer.py", "vendor/dofa-official-0cfb7e1/pretraining/util/pos_embed.py",
               "vendor/dofa_runtime/mmseg/models/necks/featurepyramid.py", wave_path.relative_to(ROOT).as_posix()]
    s["model"] = {"type": "dofa", "backbone": "dofa_base_patch16", "stem": False,
                  "code_dir": "vendor/dofa-official-0cfb7e1", "runtime_dir": "vendor/dofa_runtime",
                  "official_commit": identity["official_code_commit"],
                  "checkpoint": "weights/dofa/XShadow_DOFA_base/DOFA_ViT_base_e100.pth",
                  "checkpoint_sha256": identity["sha256"], "pretrained_dir": "weights/dofa/XShadow_DOFA_base",
                  "weight_revision": identity["revision"], "band_order": list(S2_10_BANDS),
                  "wavelengths_micrometers": wavelengths, "wavelength_source": mapping["source"],
                  "image_size": 128, "decoder_hidden_size": 512, "freeze_encoder": False,
                  "code_sha256": {path: file_sha256(ROOT / path) for path in sources}}
    s["optim"].pop("first_conv_lr", None)
    s["train"].update(amp=True, amp_dtype="bf16")
    destination = ROOT / "configs/segformer/v4r3" / f"{s['experiment']}.yaml"
    destination.parent.mkdir(exist_ok=True)
    text = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
    if destination.exists() and destination.read_text(encoding="utf-8") != text:
        raise FileExistsError("DOFA configuration changed")
    destination.write_text(text, encoding="utf-8")
    print(destination)
